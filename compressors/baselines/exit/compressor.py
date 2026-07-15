"""EXIT implementation for context-aware extractive compression."""

from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import BaseCompressor, SearchResult


class EXITCompressor(BaseCompressor):
    """EXIT: Context-aware extractive compression."""

    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint: str = None,
        device: str = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5,
        max_length: Optional[int] = None,
    ):
        """Initialize EXIT compressor.

        Args:
            base_model: Base model path.
            checkpoint: Path to trained checkpoint.
            device: Device to use (None for auto).
            cache_dir: Cache directory for models.
            batch_size: Batch size for sentence classification.
            threshold: Confidence threshold for selection.
            max_length: Maximum classifier prompt length. Defaults to the
                model/tokenizer context limit.
        """
        self.batch_size = batch_size
        self.threshold = threshold

        # Initialize tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model.
        model_kwargs = {
            "device_map": "auto" if device is None else device,
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        }

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )

        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint,
            )
        else:
            self.model = self.base_model

        # Prepare model.
        self.model.eval()
        if hasattr(self.model, "half") and not getattr(self.model, "is_quantized", False):
            self.model.half()

        # Resolve the real context limit. Gemma-2B uses 8,192 positions, while
        # some tokenizers expose a very large sentinel instead of a finite limit.
        model_limit = getattr(self.base_model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        finite_limits = [
            int(limit)
            for limit in (model_limit, tokenizer_limit)
            if limit is not None and 0 < int(limit) < 1_000_000
        ]
        inferred_limit = min(finite_limits) if finite_limits else 8192
        if max_length is not None and max_length > inferred_limit:
            raise ValueError(
                f"max_length={max_length} exceeds the model context limit "
                f"({inferred_limit})."
            )
        self.max_length = int(max_length or inferred_limit)

        # Cache device and token IDs.
        self.device = next(self.model.parameters()).device
        yes_token_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_token_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_token_ids) != 1 or len(no_token_ids) != 1:
            raise ValueError('EXIT expects "Yes" and "No" to each map to one token.')
        self.yes_token_id = yes_token_ids[0]
        self.no_token_id = no_token_ids[0]

        # Clear GPU memory.
        torch.cuda.empty_cache()

    @lru_cache(maxsize=1024)
    def _generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> str:
        """Generate prompt for relevance classification."""
        return (
            f"<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f"<start_of_turn>model\n"
        )

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch of sentences.

        The prompt is never silently truncated. Truncating the right side would
        remove the candidate sentence, the Yes/No instruction, and the model
        turn marker, making every candidate in a long retrieval set receive the
        same score.
        """
        prompts = [
            self._generate_prompt(query, context, sentence)
            for query, context, sentence in zip(queries, contexts, sentences)
        ]

        unpadded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        prompt_lengths = [len(input_ids) for input_ids in unpadded["input_ids"]]
        longest_prompt = max(prompt_lengths, default=0)
        if longest_prompt > self.max_length:
            raise ValueError(
                "EXIT classifier prompt exceeds the model context window "
                f"({longest_prompt} > {self.max_length} tokens). Pass each "
                "candidate sentence with only its containing document as "
                "`Full context`, or split an unusually long document. The "
                "compressor intentionally refuses to truncate away the "
                "candidate sentence and Yes/No instruction."
            )

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[:, -1, :]
        relevant_logits = torch.stack(
            [
                next_token_logits[:, self.yes_token_id],
                next_token_logits[:, self.no_token_id],
            ],
            dim=1,
        )
        probs = torch.softmax(relevant_logits, dim=1)
        predictions = [
            "Yes" if label_index == 0 else "No"
            for label_index in probs.argmax(dim=1).cpu().tolist()
        ]

        return predictions, probs

    @staticmethod
    def _group_document_sentences(
        documents: List[SearchResult],
    ) -> List[List[SearchResult]]:
        """Group sentence-level search results by their source document."""
        grouped = OrderedDict()
        for document in documents:
            grouped.setdefault(document.evi_id, []).append(document)
        return list(grouped.values())

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        """Compress documents using context-aware sentence extraction.

        Each candidate sentence is scored against its containing document, as
        defined in the paper, rather than against a concatenation of every
        retrieved document.

        Args:
            query: Input question.
            documents: Sentence-level search results. Sentences that share the
                same ``evi_id`` are treated as one source document.

        Returns:
            A list containing one SearchResult with the compressed text.
        """
        selected_documents = []

        for document_group in self._group_document_sentences(documents):
            sentences = [
                document.text.strip()
                for document in document_group
                if document.text and document.text.strip()
            ]
            if not sentences:
                continue

            title = next(
                (document.title.strip() for document in document_group if document.title),
                "",
            )
            document_text = " ".join(sentences)
            context = f"{title}\n{document_text}" if title else document_text

            selected_sentences = []
            for start in range(0, len(sentences), self.batch_size):
                sentence_batch = sentences[start : start + self.batch_size]
                _, probs = self._predict_batch(
                    [query] * len(sentence_batch),
                    [context] * len(sentence_batch),
                    sentence_batch,
                )
                selected_sentences.extend(
                    sentence
                    for sentence, probability in zip(
                        sentence_batch, probs[:, 0].detach().cpu().tolist()
                    )
                    if probability >= self.threshold
                )

            if selected_sentences:
                selected_documents.append(" ".join(selected_sentences))

        compressed_text = "\n\n".join(selected_documents)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text=compressed_text,
                score=1.0,
            )
        ]
