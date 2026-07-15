#!/usr/bin/env python3
"""
EXIT RAG Pipeline Quickstart
This script demonstrates an end-to-end RAG pipeline using EXIT for context compression.
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
    ):
        # Initialize models
        print("Loading models...")

        # Initialize EXIT compression model
        base_model = AutoModelForCausalLM.from_pretrained(
            retriever_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.exit_model = PeftModel.from_pretrained(base_model, compression_model)
        self.exit_model.eval()
        self.exit_tokenizer = AutoTokenizer.from_pretrained(retriever_model)
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")

        # Initialize reader model
        self.reader = AutoModelForCausalLM.from_pretrained(
            reader_model,
            device_map="auto",
        )
        self.reader.eval()
        self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model)

        # Initialize sentence splitter
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )
        self.nlp.enable_pipe("senter")

        self.device = device

    def _single_token_id(self, label: str) -> int:
        """Return the token ID for a single-token classifier label."""

        token_ids = self.exit_tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f'Expected classifier label "{label}" to map to one token, '
                f"but got token IDs {token_ids}."
            )
        return token_ids[0]

    @staticmethod
    def _compression_prompt(query: str, context: str, sentence: str) -> str:
        """Build the prompt used by the EXIT sentence classifier."""

        return f"""<start_of_turn>user
Query:
{query}
Full context:
{context}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
<start_of_turn>model
"""

    def get_sentence_relevance(
        self,
        query: str,
        context: str,
        sentence: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """Determine whether a sentence is relevant using the EXIT model."""

        prompt = self._compression_prompt(query, context, sentence)
        inputs = self.exit_tokenizer(prompt, return_tensors="pt")

        max_length = getattr(self.exit_model.config, "max_position_embeddings", None)
        input_length = inputs["input_ids"].shape[-1]
        if max_length is not None and input_length > max_length:
            raise ValueError(
                f"EXIT compression prompt is {input_length} tokens, exceeding the "
                f"model limit of {max_length}. Score each sentence against only its "
                "containing document; do not concatenate all retrieved documents into "
                "one classifier context."
            )

        inputs = inputs.to(self.exit_model.device)

        with torch.no_grad():
            outputs = self.exit_model(**inputs)
            logits = outputs.logits[
                0,
                -1,
                [self.yes_token_id, self.no_token_id],
            ]
            relevance_probability = torch.softmax(logits, dim=0)[0].item()

        return relevance_probability >= threshold, relevance_probability

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5,
    ) -> Tuple[str, List[bool], List[float]]:
        """Compress documents using EXIT.

        Each candidate sentence is scored against its own containing document,
        matching the paper and the classifier's training data. Prompts may be
        batched for throughput, but contexts must remain document-local.
        """

        start_time = time.time()

        compressed_documents = []
        relevance_scores = []
        selections = []
        total_sentences = 0
        selected_sentence_count = 0

        for document in documents:
            document_text = document.text.strip()
            document_context = (
                f"{document.title}\n{document_text}" if document.title else document_text
            )
            sentences = [
                sent.text.strip()
                for sent in self.nlp(document_text).sents
                if sent.text.strip()
            ]
            total_sentences += len(sentences)

            selected_in_document = []
            for sentence in sentences:
                is_relevant, score = self.get_sentence_relevance(
                    query=query,
                    context=document_context,
                    sentence=sentence,
                    threshold=threshold,
                )
                selections.append(is_relevant)
                relevance_scores.append(score)
                if is_relevant:
                    selected_in_document.append(sentence)
                    selected_sentence_count += 1

            if selected_in_document:
                compressed_documents.append(" ".join(selected_in_document))

        compressed_text = "\n\n".join(compressed_documents)

        compression_time = time.time() - start_time
        print(f"Compression time: {compression_time:.2f}s")
        print(f"Compressed {selected_sentence_count}/{total_sentences} sentences")

        return compressed_text, selections, relevance_scores

    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """Generate an answer using the compressed context."""

        start_time = time.time()

        # Format prompt
        chat = [
            {
                "role": "system",
                "content": (
                    "Context information is below.\n"
                    "---------------------\n"
                    f"{context}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, answer "
                    "the query. Do not provide any explanation."
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

        # Generate answer
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
            outputs[0][inputs.input_ids.size(1) :],
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
        """Run the complete RAG pipeline with compression."""

        # 1. Compress documents
        compressed_text, selections, scores = self.compress_documents(
            query,
            documents,
            compression_threshold,
        )

        # 2. Generate answer
        answer, generation_time = self.generate_answer(query, compressed_text)

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

    # Initialize pipeline
    rag = ExitRAG()

    # Example query and documents
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

    # Run pipeline
    result = rag.run_rag(query, documents)

    # Print results
    print("\nQuery:", result["query"])
    print("\nCompressed Context:", result["compressed_context"])
    print("\nAnswer:", result["answer"])
    print(f"\nGeneration Time: {result['generation_time']:.2f}s")


if __name__ == "__main__":
    main()
