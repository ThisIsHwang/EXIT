#!/usr/bin/env python3
"""End-to-end EXIT compression and reader pipeline.

Retrieval is intentionally external: callers provide already retrieved
documents so the same pipeline can be used with paper retrieval artifacts or
with an application-specific retriever.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressors import EXITCompressor, SearchResult
from compressors.baselines.exit.core import build_qa_prompt


@dataclass(frozen=True)
class Document:
    """Retrieved document supplied to the EXIT pipeline."""

    title: str
    text: str
    score: float = 1.0


class ExitRAG:
    """EXIT compression followed by an instruction-tuned reader."""

    def __init__(
        self,
        compression_base_model: str = "google/gemma-2b-it",
        compression_model: str = "doubleyyh/exit-gemma-2b",
        reader_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        batch_size: int = 8,
        threshold: float = 0.5,
        cache_dir: str = "./cache",
        compression_base_revision: Optional[str] = None,
        compression_revision: Optional[str] = None,
        reader_revision: Optional[str] = None,
        load_compressor_in_4bit: bool = True,
        max_new_tokens: int = 100,
        retriever_model: Optional[str] = None,
    ) -> None:
        """Load the paper compressor and downstream reader.

        ``retriever_model`` is retained as a deprecated alias for
        ``compression_base_model``.  Earlier versions used that misleading name
        even though this class never performs retrieval.
        """

        if retriever_model is not None:
            warnings.warn(
                "retriever_model is deprecated; use compression_base_model. "
                "ExitRAG receives retrieved documents and does not load a retriever.",
                DeprecationWarning,
                stacklevel=2,
            )
            compression_base_model = retriever_model
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        print("Loading EXIT compressor and reader...")
        self.compressor = EXITCompressor(
            base_model=compression_base_model,
            checkpoint=compression_model,
            device=device,
            cache_dir=cache_dir,
            base_revision=compression_base_revision,
            checkpoint_revision=compression_revision,
            batch_size=batch_size,
            threshold=threshold,
            load_in_4bit=load_compressor_in_4bit,
        )

        reader_kwargs = {
            "cache_dir": cache_dir,
            "device_map": "auto" if device is None else {"": device},
            "revision": reader_revision,
        }
        if device is None or not str(device).startswith("cpu"):
            reader_kwargs["torch_dtype"] = torch.float16
        self.reader = AutoModelForCausalLM.from_pretrained(
            reader_model,
            **reader_kwargs,
        )
        self.reader.eval()
        self.reader_tokenizer = AutoTokenizer.from_pretrained(
            reader_model,
            cache_dir=cache_dir,
            revision=reader_revision,
        )
        if self.reader_tokenizer.pad_token_id is None:
            self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token

        self.reader_device = next(self.reader.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.last_compression_time = 0.0
        self.last_compression_result = None

    @staticmethod
    def _synchronize_cuda() -> None:
        """Make wall-clock latency include queued CUDA work."""

        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_index)

    @staticmethod
    def _as_search_results(documents: List[Document]) -> List[SearchResult]:
        return [
            SearchResult(
                evi_id=index,
                docid=index,
                title=document.title,
                text=document.text,
                score=document.score,
            )
            for index, document in enumerate(documents)
        ]

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: Optional[float] = None,
    ) -> Tuple[str, List[bool], List[float]]:
        """Split, batch-score, and reassemble retrieved documents."""

        self._synchronize_cuda()
        started_at = time.perf_counter()
        result = self.compressor.compress_with_details(
            query=query,
            documents=self._as_search_results(documents),
            threshold=threshold,
        )
        self._synchronize_cuda()
        self.last_compression_time = time.perf_counter() - started_at
        self.last_compression_result = result

        print(f"Compression time: {self.last_compression_time:.2f}s")
        print(
            f"Compressed {len(result.selected_sentences)}/{len(result.sentences)} sentences"
        )
        return result.text, list(result.selections), list(result.scores)

    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """Generate an answer with the paper's Table 7 reader prompt."""

        self._synchronize_cuda()
        started_at = time.perf_counter()
        prompt_content = build_qa_prompt(query=query, context=context)
        prompt = self.reader_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.reader_tokenizer(prompt, return_tensors="pt")

        limits = []
        model_limit = getattr(self.reader.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            limits.append(model_limit)
        tokenizer_limit = getattr(self.reader_tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            limits.append(tokenizer_limit)
        reader_limit = min(limits) if limits else None
        required_tokens = inputs.input_ids.size(1) + self.max_new_tokens
        if reader_limit is not None and required_tokens > reader_limit:
            raise ValueError(
                f"reader prompt plus max_new_tokens requires {required_tokens} "
                f"tokens, exceeding the reader limit of {reader_limit}"
            )
        inputs = inputs.to(self.reader_device)

        with torch.inference_mode():
            outputs = self.reader.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.reader_tokenizer.pad_token_id,
                eos_token_id=self.reader_tokenizer.eos_token_id,
                do_sample=False,
            )
        self._synchronize_cuda()

        answer = self.reader_tokenizer.decode(
            outputs[0][inputs.input_ids.size(1) :],
            skip_special_tokens=True,
        ).strip()
        reading_time = time.perf_counter() - started_at
        return answer, reading_time

    def run_rag(
        self,
        query: str,
        documents: List[Document],
        compression_threshold: Optional[float] = None,
    ) -> dict:
        """Run compression and reading while reporting paper latency fields."""

        compressed_text, selections, scores = self.compress_documents(
            query=query,
            documents=documents,
            threshold=compression_threshold,
        )
        answer, reading_time = self.generate_answer(query, compressed_text)

        original_context = "\n\n".join(
            f"{document.title}\n{document.text}" if document.title else document.text
            for document in documents
        )
        original_tokens = len(
            self.reader_tokenizer.encode(original_context, add_special_tokens=False)
        )
        compressed_tokens = len(
            self.reader_tokenizer.encode(compressed_text, add_special_tokens=False)
        )
        details = self.last_compression_result

        return {
            "query": query,
            "compressed_context": compressed_text,
            "answer": answer,
            "sentence_selections": selections,
            "relevance_scores": scores,
            "sentences": list(details.sentences),
            "sentence_document_indices": list(details.document_indices),
            "sentence_indices": list(details.sentence_indices),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_time": self.last_compression_time,
            "reading_time": reading_time,
            "generation_time": reading_time,
            "total_time": self.last_compression_time + reading_time,
        }


def main() -> None:
    """Run the small example used in the README."""

    rag = ExitRAG()
    query = "How do solid-state drives (SSDs) improve computer performance?"
    documents = [
        Document(
            title="Computer Storage Technologies",
            text=(
                "Solid-state drives use flash memory to store data without moving parts. "
                "Unlike traditional hard drives, SSDs have no mechanical components. "
                "The absence of physical movement allows for much faster data access speeds. "
                "I bought my computer last week. "
                "SSDs significantly reduce boot times and application loading speeds. "
                "They consume less power and are more reliable than mechanical drives. "
                "The price of SSDs has decreased significantly in recent years."
            ),
        )
    ]
    result = rag.run_rag(query, documents)
    print("\nQuery:", result["query"])
    print("\nCompressed Context:", result["compressed_context"])
    print("\nAnswer:", result["answer"])
    print(f"\nTotal Time: {result['total_time']:.2f}s")


if __name__ == "__main__":
    main()
