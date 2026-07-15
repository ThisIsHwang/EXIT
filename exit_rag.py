#!/usr/bin/env python3
"""
EXIT RAG Pipeline Quickstart

This script demonstrates an end-to-end RAG pipeline using EXIT for context
compression.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import spacy
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Document:
    """Container for document content."""

    title: str
    text: str
    score: float = 1.0


class ExitRAG:
    """End-to-end Retrieval-Augmented Generation with EXIT compression."""

    def __init__(
        self,
        retriever_model: str = "google/gemma-2b-it",
        compression_model: str = "doubleyyh/exit-gemma-2b",
        reader_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        compression_max_length: int = 4096,
    ):
        # Initialize models.
        print("Loading models...")

        # Initialize EXIT compression model.
        base_model = AutoModelForCausalLM.from_pretrained(
            retriever_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.exit_model = PeftModel.from_pretrained(
            base_model,
            compression_model,
        )
        self.exit_model.eval()
        self.exit_tokenizer = AutoTokenizer.from_pretrained(
            retriever_model
        )
        if self.exit_tokenizer.pad_token is None:
            self.exit_tokenizer.pad_token = self.exit_tokenizer.eos_token

        finite_limits = [compression_max_length]
        for limit in (
            getattr(base_model.config, "max_position_embeddings", None),
            getattr(self.exit_tokenizer, "model_max_length", None),
        ):
            if isinstance(limit, int) and 0 < limit < 1_000_000:
                finite_limits.append(limit)
        self.compression_max_length = min(finite_limits)
        self.exit_device = next(self.exit_model.parameters()).device

        yes_ids = self.exit_tokenizer.encode(
            "Yes",
            add_special_tokens=False,
        )
        no_ids = self.exit_tokenizer.encode(
            "No",
            add_special_tokens=False,
        )
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                'EXIT expects "Yes" and "No" to each map to one tokenizer token.'
            )
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        # Initialize reader model.
        self.reader = AutoModelForCausalLM.from_pretrained(
            reader_model,
            device_map="auto",
        )
        self.reader_tokenizer = AutoTokenizer.from_pretrained(
            reader_model
        )

        # Initialize sentence splitter.
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        self.nlp.enable_pipe("senter")

        self.device = device

    @staticmethod
    def _truncate_context_tokens(
        context_ids: List[int],
        budget: int,
    ) -> List[int]:
        """Keep the beginning and end of an overlong document context."""
        if len(context_ids) <= budget:
            return context_ids
        if budget <= 0:
            return []

        head = (budget + 1) // 2
        tail = budget - head
        if tail == 0:
            return context_ids[:head]
        return context_ids[:head] + context_ids[-tail:]

    def _encode_exit_prompt(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> List[int]:
        """Encode the EXIT prompt while truncating only document context."""
        prefix = (
            f"<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n"
        )
        suffix = (
            f"\nSentence:\n{sentence}\n"
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f"<start_of_turn>model\n"
        )

        prefix_ids = self.exit_tokenizer.encode(
            prefix,
            add_special_tokens=False,
        )
        context_ids = self.exit_tokenizer.encode(
            context,
            add_special_tokens=False,
        )
        suffix_ids = self.exit_tokenizer.encode(
            suffix,
            add_special_tokens=False,
        )
        special_tokens = self.exit_tokenizer.num_special_tokens_to_add(
            pair=False
        )

        context_budget = (
            self.compression_max_length
            - special_tokens
            - len(prefix_ids)
            - len(suffix_ids)
        )
        if context_budget < 0:
            raise ValueError(
                "The EXIT query and candidate sentence exceed "
                f"compression_max_length={self.compression_max_length} "
                "before document context is added."
            )

        context_ids = self._truncate_context_tokens(
            context_ids,
            context_budget,
        )
        prompt_ids = prefix_ids + context_ids + suffix_ids
        return self.exit_tokenizer.build_inputs_with_special_tokens(
            prompt_ids
        )

    def get_sentence_relevance(
        self,
        query: str,
        context: str,
        sentence: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """Determine whether a sentence is relevant using EXIT."""
        input_ids = self._encode_exit_prompt(
            query,
            context,
            sentence,
        )
        inputs = {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.exit_device,
            ),
            "attention_mask": torch.ones(
                (1, len(input_ids)),
                dtype=torch.long,
                device=self.exit_device,
            ),
        }

        with torch.no_grad():
            outputs = self.exit_model(**inputs)

        logits = outputs.logits[
            0,
            -1,
            [self.yes_token_id, self.no_token_id],
        ]
        probability = torch.softmax(logits, dim=0)[0].item()
        return probability >= threshold, probability

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5,
    ) -> Tuple[str, List[bool], List[float]]:
        """Compress documents using their own passage as EXIT context.

        The model is trained to score each candidate sentence against the
        sentence's containing document. Retrieved documents are therefore
        processed independently rather than concatenated into one global
        ``Full context`` field.
        """
        start_time = time.time()

        selected_documents: List[str] = []
        relevance_scores: List[float] = []
        selections: List[bool] = []
        total_sentences = 0
        total_selected = 0

        for document in documents:
            context = (
                f"{document.title}\n{document.text}"
                if document.title
                else document.text
            )
            sentences = [
                sentence.text.strip()
                for sentence in self.nlp(document.text).sents
                if sentence.text.strip()
            ]
            total_sentences += len(sentences)

            selected_from_document: List[str] = []
            for sentence in sentences:
                is_relevant, score = self.get_sentence_relevance(
                    query,
                    context,
                    sentence,
                    threshold,
                )
                selections.append(is_relevant)
                relevance_scores.append(score)
                if is_relevant:
                    selected_from_document.append(sentence)
                    total_selected += 1

            if selected_from_document:
                selected_documents.append(
                    " ".join(selected_from_document)
                )

        compressed_text = "\n\n".join(selected_documents)

        compression_time = time.time() - start_time
        print(f"Compression time: {compression_time:.2f}s")
        print(
            f"Compressed {total_selected}/{total_sentences} sentences"
        )

        return compressed_text, selections, relevance_scores

    def generate_answer(
        self,
        query: str,
        context: str,
    ) -> Tuple[str, float]:
        """Generate an answer using compressed context."""
        start_time = time.time()

        chat = [
            {
                "role": "system",
                "content": (
                    "Context information is below.\n"
                    "---------------------\n"
                    f"{context}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query. Do not provide any explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\nAnswer: ",
            },
        ]

        prompt = self.reader_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.reader_tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.reader.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                pad_token_id=self.reader_tokenizer.eos_token_id,
                do_sample=False,
            )

        answer = self.reader_tokenizer.decode(
            outputs[0][inputs.input_ids.size(1):],
            skip_special_tokens=True,
        ).strip()

        generation_time = time.time() - start_time
        return answer, generation_time

    def run_rag(
        self,
        query: str,
        documents: List[Document],
        compression_threshold: float = 0.5,
    ) -> dict:
        """Run complete RAG pipeline with compression."""
        compressed_text, selections, scores = self.compress_documents(
            query,
            documents,
            compression_threshold,
        )
        answer, generation_time = self.generate_answer(
            query,
            compressed_text,
        )

        return {
            "query": query,
            "compressed_context": compressed_text,
            "answer": answer,
            "sentence_selections": selections,
            "relevance_scores": scores,
            "generation_time": generation_time,
        }


def main():
    """Demonstrate usage of the EXIT RAG pipeline."""
    rag = ExitRAG()

    query = "How do solid-state drives (SSDs) improve computer performance?"
    documents = [
        Document(
            title="Computer Storage Technologies",
            text="""
            Solid-state drives use flash memory to store data without moving parts.
            Unlike traditional hard drives, SSDs have no mechanical components.
            The absence of physical movement allows for much faster data access speeds.
            I bought my computer last week.
            SSDs significantly reduce boot times and application loading speeds.
            They consume less power and are more reliable than mechanical drives.
            The price of SSDs has decreased significantly in recent years.
            """,
        )
    ]

    result = rag.run_rag(query, documents)

    print("\nQuery:", result["query"])
    print("\nCompressed Context:", result["compressed_context"])
    print("\nAnswer:", result["answer"])
    print(f"\nGeneration Time: {result['generation_time']:.2f}s")


if __name__ == "__main__":
    main()
