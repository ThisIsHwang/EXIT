"""EXIT implementation for context-aware extractive compression."""

from itertools import groupby
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
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5,
    ):
        """Initialize the EXIT compressor.

        Args:
            base_model: Base causal language model used by the classifier.
            checkpoint: EXIT PEFT adapter path or Hugging Face model ID.
            device: Device or device map passed to Transformers. ``None`` uses auto.
            cache_dir: Directory for downloaded model files.
            batch_size: Number of sentence-classification prompts per forward pass.
            threshold: Minimum normalized ``Yes`` probability for sentence retention.
        """
        self.batch_size = batch_size
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            cache_dir=cache_dir,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")
        self.max_input_tokens = self._resolve_max_input_tokens()

        torch.cuda.empty_cache()

    def _single_token_id(self, label: str) -> int:
        """Return the token ID for a single-token classifier label."""
        token_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f'Expected classifier label "{label}" to map to one token, '
                f"but got token IDs {token_ids}."
            )
        return token_ids[0]

    def _resolve_max_input_tokens(self) -> Optional[int]:
        """Resolve a usable model/tokenizer context-window limit."""
        limits = []

        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            limits.append(model_limit)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        # Hugging Face uses very large sentinel values when no tokenizer limit is set.
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            limits.append(tokenizer_limit)

        return min(limits) if limits else None

    @staticmethod
    def _generate_prompt(query: str, context: str, sentence: str) -> str:
        """Generate the prompt used for sentence relevance classification."""
        return (
            "<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            'Is this sentence useful in answering the query? '
            'Answer only "Yes" or "No".<end_of_turn>\n'
            "<start_of_turn>model\n"
        )

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch without silently truncating prompts."""
        prompts = [
            self._generate_prompt(query, context, sentence)
            for query, context, sentence in zip(queries, contexts, sentences)
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )

        prompt_lengths = inputs["attention_mask"].sum(dim=1)
        if self.max_input_tokens is not None:
            too_long = prompt_lengths > self.max_input_tokens
            if torch.any(too_long):
                longest = int(prompt_lengths.max().item())
                raise ValueError(
                    f"EXIT compression prompt is {longest} tokens, exceeding the "
                    f"model limit of {self.max_input_tokens}. Each sentence must be "
                    "scored against only its containing document. For an unusually "
                    "long single document, split or chunk that document explicitly "
                    "instead of relying on tokenizer truncation."
                )

        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            label_logits = torch.stack(
                [
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id],
                ],
                dim=1,
            )
            probabilities = torch.softmax(label_logits, dim=1)

        predictions = [
            "Yes" if label_index == 0 else "No"
            for label_index in probabilities.argmax(dim=1).cpu().tolist()
        ]
        return predictions, probabilities

    @staticmethod
    def _document_groups(
        documents: List[SearchResult],
    ) -> List[List[SearchResult]]:
        """Group consecutive sentence records that belong to one document."""
        return [
            list(group)
            for _, group in groupby(documents, key=lambda document: document.evi_id)
        ]

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        """Compress sentence records with document-local classifier contexts.

        ``documents`` is expected to contain sentence-level ``SearchResult`` records,
        with consecutive records sharing an ``evi_id`` belonging to the same source
        document. Every candidate sentence is classified against the title and full
        text of that source document, never against a concatenation of all retrieved
        documents.
        """
        if not documents:
            return []

        groups = self._document_groups(documents)
        candidate_records = []

        for group_index, group in enumerate(groups):
            title = group[0].title.strip() if group[0].title else ""
            sentence_texts = [
                record.text.strip()
                for record in group
                if record.text and record.text.strip()
            ]
            document_text = " ".join(sentence_texts)
            document_context = (
                f"{title}\n{document_text}" if title else document_text
            )

            for record in group:
                sentence = record.text.strip() if record.text else ""
                if sentence:
                    candidate_records.append(
                        (group_index, document_context, sentence)
                    )

        selected_by_group: List[List[str]] = [[] for _ in groups]

        for start in range(0, len(candidate_records), self.batch_size):
            batch = candidate_records[start : start + self.batch_size]
            _, probabilities = self._predict_batch(
                [query] * len(batch),
                [item[1] for item in batch],
                [item[2] for item in batch],
            )

            for (group_index, _, sentence), probability in zip(
                batch,
                probabilities[:, 0].detach().cpu().tolist(),
            ):
                if probability >= self.threshold:
                    selected_by_group[group_index].append(sentence)

        compressed_documents = [
            " ".join(selected_sentences)
            for selected_sentences in selected_by_group
            if selected_sentences
        ]
        compressed_text = "\n\n".join(compressed_documents)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text=compressed_text,
                score=1.0,
            )
        ]
