"""EXIT implementation for context-aware extractive compression."""

import warnings
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import BaseCompressor, SearchResult


class EXITCompressor(BaseCompressor):
    """EXIT: context-aware extractive compression.

    Each candidate sentence is classified with the query and the *containing
    document* as context. ``top_k`` controls how many documents are processed;
    it must not turn into one concatenated context shared by every sentence.
    """

    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5,
        max_input_length: Optional[int] = None,
        label_mass_warning_threshold: float = 0.5,
    ):
        """Initialize the EXIT compressor.

        Args:
            base_model: Base model path.
            checkpoint: PEFT adapter path.
            device: Device map passed to Transformers (``None`` uses ``auto``).
            cache_dir: Directory for downloaded models.
            batch_size: Number of sentence decisions per forward pass.
            threshold: Normalized ``P(Yes) / (P(Yes) + P(No))`` threshold.
            max_input_length: Maximum classifier prompt length. If omitted,
                use the model/tokenizer limit (8192 for the released model).
            label_mass_warning_threshold: Warn when the full-vocabulary
                probability mass assigned to ``Yes`` and ``No`` falls below
                this value. A low value usually means that the prompt template
                or context window is invalid for the classifier.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= label_mass_warning_threshold <= 1.0:
            raise ValueError(
                "label_mass_warning_threshold must be between 0 and 1"
            )

        self.batch_size = batch_size
        self.threshold = threshold
        self.label_mass_warning_threshold = label_mass_warning_threshold
        self._warned_about_context_truncation = False
        self._warned_about_label_mass = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
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
        self.model = (
            PeftModel.from_pretrained(self.base_model, checkpoint)
            if checkpoint
            else self.base_model
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.max_input_length = self._resolve_max_input_length(
            max_input_length
        )
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _single_token_id(self, label: str) -> int:
        token_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"EXIT expects {label!r} to be one token, but the tokenizer "
                f"produced {token_ids}. Use the tokenizer paired with the "
                "released base model."
            )
        return token_ids[0]

    def _resolve_max_input_length(self, requested: Optional[int]) -> int:
        if requested is not None:
            if requested < 1:
                raise ValueError("max_input_length must be positive")
            return requested

        candidates = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if (
            isinstance(tokenizer_limit, int)
            and 0 < tokenizer_limit < 1_000_000
        ):
            candidates.append(tokenizer_limit)

        config = getattr(self.model, "config", None)
        for name in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

        return min(candidates) if candidates else 8192

    @staticmethod
    def _generate_prompt(query: str, context: str, sentence: str) -> str:
        """Generate the same prompt template used during EXIT training."""
        return (
            "<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            'Is this sentence useful in answering the query? '
            'Answer only "Yes" or "No".<end_of_turn>\n'
            "<start_of_turn>model\n"
        )

    def _fit_prompt_to_context_window(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> str:
        """Trim only document context while preserving the classifier tail."""
        prompt = self._generate_prompt(query, context, sentence)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_ids) <= self.max_input_length:
            return prompt

        empty_context_prompt = self._generate_prompt(query, "", sentence)
        fixed_length = len(
            self.tokenizer.encode(
                empty_context_prompt,
                add_special_tokens=True,
            )
        )
        context_budget = self.max_input_length - fixed_length
        if context_budget <= 0:
            raise ValueError(
                "The query, candidate sentence, and EXIT instruction exceed "
                f"max_input_length={self.max_input_length}; the document "
                "context cannot be safely truncated."
            )

        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=False,
        )
        context_ids = context_ids[:context_budget]

        # Decoding and re-tokenizing can change a boundary by a few tokens.
        # Reduce the context until the complete prompt fits, never truncating
        # the candidate sentence or the Yes/No instruction.
        while True:
            truncated_context = self.tokenizer.decode(
                context_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            prompt = self._generate_prompt(
                query,
                truncated_context,
                sentence,
            )
            overflow = (
                len(self.tokenizer.encode(prompt, add_special_tokens=True))
                - self.max_input_length
            )
            if overflow <= 0:
                if not self._warned_about_context_truncation:
                    warnings.warn(
                        "An EXIT document exceeded the classifier context "
                        "window. Only the document context was truncated; "
                        "the query, candidate sentence, and Yes/No instruction "
                        "were preserved.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_about_context_truncation = True
                return prompt
            if not context_ids:
                break
            context_ids = context_ids[: max(0, len(context_ids) - overflow - 8)]

        raise ValueError(
            "Unable to fit the EXIT prompt while preserving the query, "
            "candidate sentence, and classifier instruction."
        )

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Predict sentence relevance and return label-space diagnostics."""
        prompts = [
            self._fit_prompt_to_context_window(query, context, sentence)
            for query, context, sentence in zip(
                queries,
                contexts,
                sentences,
            )
        ]
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

        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[:, -1, :].float()
        label_logits = next_token_logits[:, [
            self.yes_token_id,
            self.no_token_id,
        ]]
        scores = torch.softmax(label_logits, dim=1)
        label_mass = torch.exp(
            torch.logsumexp(label_logits, dim=1)
            - torch.logsumexp(next_token_logits, dim=1)
        )

        if (
            not self._warned_about_label_mass
            and torch.any(label_mass < self.label_mass_warning_threshold)
        ):
            minimum_mass = label_mass.min().item()
            warnings.warn(
                "EXIT assigned little next-token probability mass to the "
                f"Yes/No label space (minimum={minimum_mass:.4f}). This "
                "usually indicates a mismatched tokenizer/template/model or "
                "a malformed prompt. The normalized Yes/No score should not "
                "be interpreted as calibrated confidence in that case.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_about_label_mass = True

        predictions = [
            "Yes" if label_index == 0 else "No"
            for label_index in scores.argmax(dim=1).cpu().tolist()
        ]
        return predictions, scores, label_mass

    @staticmethod
    def _group_by_document(
        documents: List[SearchResult],
    ) -> List[List[SearchResult]]:
        """Group consecutive sentence records belonging to one document."""
        groups: List[List[SearchResult]] = []
        for item in documents:
            if not groups or groups[-1][0].evi_id != item.evi_id:
                groups.append([item])
            else:
                groups[-1].append(item)
        return groups

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        """Compress sentence records using their containing document context.

        ``documents`` is expected to contain sentence-level ``SearchResult``
        records. Consecutive records with the same ``evi_id`` are treated as
        sentences from one retrieved document. The full text of that document,
        not the concatenation of all retrieved documents, is used as context.
        """
        if not documents:
            return []

        groups = self._group_by_document(documents)
        examples = []
        for group_index, group in enumerate(groups):
            title = next((item.title for item in group if item.title), "")
            body = " ".join(item.text.strip() for item in group if item.text.strip())
            context = f"{title}\n{body}" if title else body
            for item in group:
                if item.text.strip():
                    examples.append((group_index, context, item.text.strip()))

        selected_by_group: List[List[str]] = [[] for _ in groups]
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            _, scores, _ = self._predict_batch(
                [query] * len(batch),
                [context for _, context, _ in batch],
                [sentence for _, _, sentence in batch],
            )
            for (group_index, _, sentence), score in zip(batch, scores[:, 0]):
                if score.item() >= self.threshold:
                    selected_by_group[group_index].append(sentence)

        selected_texts = [
            " ".join(sentences)
            for sentences in selected_by_group
            if sentences
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
