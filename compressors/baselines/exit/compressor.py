"""EXIT implementation for context-aware extractive compression."""

from collections import OrderedDict
from functools import lru_cache
from typing import List, Tuple

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
        max_length: int = 4096,
    ):
        """Initialize EXIT compressor.

        Args:
            base_model: Base model path.
            checkpoint: Path to trained checkpoint.
            device: Device to use (None for auto).
            cache_dir: Directory for downloaded models.
            batch_size: Number of candidate sentences scored per forward pass.
            threshold: Minimum normalized P(Yes) required for selection.
            max_length: Maximum classifier prompt length in tokens.
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

        self.model.eval()

        # Respect both the requested limit and any finite model/tokenizer limits.
        finite_limits = [max_length]
        for limit in (
            getattr(self.base_model.config, "max_position_embeddings", None),
            getattr(self.tokenizer, "model_max_length", None),
        ):
            if isinstance(limit, int) and 0 < limit < 1_000_000:
                finite_limits.append(limit)
        self.max_length = min(finite_limits)

        # Cache device and label token IDs.
        self.device = next(self.model.parameters()).device
        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                'EXIT expects "Yes" and "No" to each map to one tokenizer token.'
            )
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @lru_cache(maxsize=1024)
    def _generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> str:
        """Generate an untruncated relevance-classification prompt."""
        return (
            f"<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f"<start_of_turn>model\n"
        )

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

        # Preserve the title/opening and the end of the containing document.
        head = (budget + 1) // 2
        tail = budget - head
        if tail == 0:
            return context_ids[:head]
        return context_ids[:head] + context_ids[-tail:]

    def _encode_prompt(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> List[int]:
        """Encode a prompt while truncating only the document context.

        Generic right-side truncation can remove the candidate sentence, the
        Yes/No instruction, and the assistant turn marker. Generic left-side
        truncation can remove the query. This method preserves both and applies
        the token budget only to ``Full context``.
        """
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

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)

        context_budget = (
            self.max_length
            - special_tokens
            - len(prefix_ids)
            - len(suffix_ids)
        )
        if context_budget < 0:
            raise ValueError(
                "The EXIT query and candidate sentence exceed the configured "
                f"max_length={self.max_length} before document context is added."
            )

        context_ids = self._truncate_context_tokens(
            context_ids,
            context_budget,
        )
        prompt_ids = prefix_ids + context_ids + suffix_ids
        return self.tokenizer.build_inputs_with_special_tokens(prompt_ids)

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch of candidate sentences."""
        if not (len(queries) == len(contexts) == len(sentences)):
            raise ValueError("queries, contexts, and sentences must have equal length")
        if not queries:
            return [], torch.empty((0, 2))

        features = []
        for query, context, sentence in zip(queries, contexts, sentences):
            input_ids = self._encode_prompt(query, context, sentence)
            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                }
            )

        inputs = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.no_grad():
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
    def _build_document_context(documents: List[SearchResult]) -> str:
        """Reconstruct the containing document for a group of candidates."""
        title = next(
            (
                doc.title.strip()
                for doc in documents
                if doc.title and doc.title.strip()
            ),
            "",
        )
        body = " ".join(
            doc.text.strip()
            for doc in documents
            if doc.text and doc.text.strip()
        )
        return f"{title}\n{body}" if title else body

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        """Compress candidates using each candidate's containing document.

        ``documents`` may contain sentence-level candidates. Candidates sharing
        an ``evi_id`` are treated as belonging to the same retrieved document.
        The EXIT paper conditions a candidate sentence on that document, not on
        a concatenation of every top-k retrieved document.
        """
        if not documents:
            return []

        document_groups = OrderedDict()
        for document in documents:
            document_groups.setdefault(document.evi_id, []).append(document)

        queries: List[str] = []
        contexts: List[str] = []
        sentences: List[str] = []
        ordered_candidates: List[SearchResult] = []

        for group in document_groups.values():
            document_context = self._build_document_context(group)
            for candidate in group:
                queries.append(query)
                contexts.append(document_context)
                sentences.append(candidate.text)
                ordered_candidates.append(candidate)

        yes_probabilities: List[float] = []
        for start in range(0, len(ordered_candidates), self.batch_size):
            end = start + self.batch_size
            _, probabilities = self._predict_batch(
                queries[start:end],
                contexts[start:end],
                sentences[start:end],
            )
            yes_probabilities.extend(
                probabilities[:, 0].detach().cpu().tolist()
            )

        selected_by_document = OrderedDict(
            (evi_id, [])
            for evi_id in document_groups
        )
        for candidate, probability in zip(
            ordered_candidates,
            yes_probabilities,
        ):
            if probability >= self.threshold:
                selected_by_document[candidate.evi_id].append(
                    candidate.text.strip()
                )

        selected_texts = [
            " ".join(texts)
            for texts in selected_by_document.values()
            if texts
        ]
        compressed_text = "\n\n".join(selected_texts)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text=compressed_text,
                score=1.0,
            )
        ]
