"""EXIT context-aware extractive compression."""

from collections import OrderedDict
from contextlib import nullcontext
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import warnings

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ...base import BaseCompressor, SearchResult
from .core import (
    CompressionResult,
    EXITDocument,
    assemble_compression,
    batched,
    build_compression_prompt,
    prepare_candidates,
    validate_threshold,
)


class EXITCompressor(BaseCompressor):
    """Paper-faithful EXIT sentence relevance classifier.

    Retrieved records sharing ``(evi_id, docid)`` are treated as fragments of
    one source document.  The fragments are rejoined before sentence splitting,
    which supports both document-level inputs and legacy sentence-level inputs.
    """

    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint: Optional[str] = "doubleyyh/exit-gemma-2b",
        device: Optional[str] = None,
        cache_dir: str = "./cache",
        base_revision: Optional[str] = None,
        checkpoint_revision: Optional[str] = None,
        batch_size: int = 8,
        threshold: float = 0.5,
        load_in_4bit: bool = True,
        max_input_tokens: Optional[int] = None,
        label_mass_warning_threshold: float = 0.5,
        sentence_splitter: Optional[Callable[[str], Iterable[str]]] = None,
    ) -> None:
        """Load the released EXIT classifier.

        Args:
            base_model: Gemma base model used by the released adapter.
            checkpoint: PEFT adapter path/model ID. ``None`` uses the base model.
            device: Device for the model; ``None`` delegates placement to Accelerate.
            cache_dir: Hugging Face cache directory.
            base_revision: Optional immutable base-model revision.
            checkpoint_revision: Optional immutable adapter revision.
            batch_size: Sentence prompts evaluated in one forward pass.
            threshold: Minimum normalized Yes probability used for retention.
            load_in_4bit: Use the paper's 4-bit inference configuration.
            max_input_tokens: Optional stricter prompt length limit.
            label_mass_warning_threshold: Warn if the unnormalized next-token
                probability assigned to the Yes/No label space is below this value.
            sentence_splitter: Optional deterministic splitter for integration/tests.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        validate_threshold(threshold)
        validate_threshold(label_mass_warning_threshold)
        if max_input_tokens is not None and max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        cpu_requested = device is not None and str(device).startswith("cpu")
        cpu_runtime = cpu_requested or (
            device is None and not torch.cuda.is_available()
        )
        if load_in_4bit and (cpu_requested or not torch.cuda.is_available()):
            raise RuntimeError(
                "4-bit EXIT inference requires a CUDA device. Set load_in_4bit=False "
                "for a CPU compatibility run (paper latency will not be comparable)."
            )

        self.batch_size = batch_size
        self.threshold = threshold
        self.label_mass_warning_threshold = label_mass_warning_threshold
        self.sentence_splitter = sentence_splitter or self._spacy_sentence_splitter()
        self._warned_about_label_mass = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            cache_dir=cache_dir,
            revision=base_revision,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_dtype = torch.float32 if cpu_runtime else torch.float16
        model_kwargs = {
            "cache_dir": cache_dir,
            "device_map": "auto" if device is None else {"": device},
            "torch_dtype": model_dtype,
            "revision": base_revision,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )
        self.model = (
            PeftModel.from_pretrained(
                self.base_model,
                checkpoint,
                revision=checkpoint_revision,
            )
            if checkpoint
            else self.base_model
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")

        model_limit = self._resolve_model_limit()
        if max_input_tokens is not None and model_limit is not None:
            self.max_input_tokens = min(max_input_tokens, model_limit)
        else:
            self.max_input_tokens = max_input_tokens or model_limit

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _spacy_sentence_splitter() -> Callable[[str], Iterable[str]]:
        """Create the rule-based spaCy splitter described in the paper."""

        try:
            from spacy.lang.en import English
        except ImportError as error:
            raise RuntimeError(
                "spaCy is required for EXIT sentence decomposition. "
                "Install the project requirements first."
            ) from error

        nlp = English()
        nlp.add_pipe("sentencizer")

        def split(text: str) -> Iterable[str]:
            return (sentence.text for sentence in nlp(text).sents)

        return split

    def _single_token_id(self, label: str) -> int:
        """Return the ID of one classifier label, rejecting tokenizer drift."""

        token_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"EXIT expects {label!r} to map to one token, but got {token_ids}. "
                "Use the tokenizer paired with the released Gemma base model."
            )
        return token_ids[0]

    def _resolve_model_limit(self) -> Optional[int]:
        """Resolve the smallest real tokenizer/model context limit."""

        limits = []
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            limits.append(model_limit)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            limits.append(tokenizer_limit)
        return min(limits) if limits else None

    @staticmethod
    def _generate_prompt(query: str, context: str, sentence: str) -> str:
        """Backward-compatible alias for the shared paper prompt."""

        return build_compression_prompt(query, context, sentence)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], torch.Tensor]:
        """Score one batch using normalized Yes-vs-No next-token logits."""

        if not (len(queries) == len(contexts) == len(sentences)):
            raise ValueError("queries, contexts, and sentences must have equal length")
        if not queries:
            return [], torch.empty((0, 2), dtype=torch.float32)

        prompts = [
            build_compression_prompt(query, context, sentence)
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
        if self.max_input_tokens is not None and torch.any(
            prompt_lengths > self.max_input_tokens
        ):
            longest = int(prompt_lengths.max().item())
            raise ValueError(
                f"EXIT compression prompt is {longest} tokens, exceeding the "
                f"model limit of {self.max_input_tokens}. EXIT must score each "
                "sentence against its complete containing document; split or "
                "chunk that source document explicitly instead of truncating it."
            )

        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }
        with torch.inference_mode(), self._autocast_context():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            label_logits = next_token_logits[
                :, [self.yes_token_id, self.no_token_id]
            ].float()
            probabilities = torch.softmax(label_logits, dim=1)
            label_mass = torch.exp(
                torch.logsumexp(label_logits, dim=1)
                - torch.logsumexp(next_token_logits.float(), dim=1)
            )

        if not self._warned_about_label_mass and torch.any(
            label_mass < self.label_mass_warning_threshold
        ):
            warnings.warn(
                "EXIT assigned little next-token probability mass to the Yes/No "
                f"labels (minimum={label_mass.min().item():.4f}). Check that the "
                "released adapter, Gemma tokenizer, and prompt template match; the "
                "normalized Yes/No score is not calibrated in this condition.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_about_label_mass = True

        predictions = [
            "Yes" if label_index == 0 else "No"
            for label_index in probabilities.argmax(dim=1).cpu().tolist()
        ]
        return predictions, probabilities

    def score_sentences(
        self,
        query: str,
        contexts: Sequence[str],
        sentences: Sequence[str],
    ) -> List[float]:
        """Score candidate sentences in parallel batches."""

        if len(contexts) != len(sentences):
            raise ValueError("contexts and sentences must have the same length")

        pairs = list(zip(contexts, sentences))
        scores = []
        for batch in batched(pairs, self.batch_size):
            _, probabilities = self._predict_batch(
                [query] * len(batch),
                [context for context, _ in batch],
                [sentence for _, sentence in batch],
            )
            scores.extend(probabilities[:, 0].detach().cpu().tolist())
        return scores

    @staticmethod
    def _source_documents(documents: Sequence[SearchResult]) -> List[EXITDocument]:
        """Reconstruct source documents without requiring contiguous records."""

        grouped = OrderedDict()
        for document in documents:
            key = (document.evi_id, document.docid)
            if key not in grouped:
                grouped[key] = {
                    "title": document.title or "",
                    "fragments": [],
                }
            elif not grouped[key]["title"] and document.title:
                grouped[key]["title"] = document.title

            if document.text and document.text.strip():
                grouped[key]["fragments"].append(document.text.strip())

        return [
            EXITDocument(
                document_id=key,
                title=values["title"],
                text=" ".join(values["fragments"]),
            )
            for key, values in grouped.items()
            if values["fragments"]
        ]

    def compress_with_details(
        self,
        query: str,
        documents: Sequence[SearchResult],
        threshold: Optional[float] = None,
    ) -> CompressionResult:
        """Run all three EXIT stages and return sentence-level details."""

        active_threshold = self.threshold if threshold is None else threshold
        validate_threshold(active_threshold)
        source_documents = self._source_documents(documents)
        candidates = prepare_candidates(
            source_documents,
            sentence_splitter=self.sentence_splitter,
        )
        scores = self.score_sentences(
            query=query,
            contexts=[candidate.context for candidate in candidates],
            sentences=[candidate.sentence for candidate in candidates],
        )
        return assemble_compression(candidates, scores, active_threshold)

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        """Compress retrieved documents into one reader context."""

        if not documents:
            return []

        result = self.compress_with_details(query, documents)
        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text=result.text,
                score=1.0,
            )
        ]
