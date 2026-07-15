# EXIT: Context-Aware Extractive Compression for RAG

[![arXiv](https://img.shields.io/badge/arXiv-2412.12559-b31b1b.svg)](https://arxiv.org/abs/2412.12559)

Official implementation of **EXIT: Context-Aware Extractive Compression for
Enhancing Retrieval-Augmented Generation**.

EXIT splits each retrieved document into sentences, scores every sentence with
its complete containing document as context, and keeps sentences whose
normalized `Yes` probability passes a threshold. Selected sentences are
reassembled in their original document and sentence order.

```text
retrieved document -> sentence candidates -> batched Yes/No scoring -> ordered context
                                          P(Yes)
score = ----------------------------------------------------------------
        P(Yes) + P(No) over the two exact next-token classifier labels
```

## Installation

For normal development:

```bash
conda create -n exit python=3.10
conda activate exit
pip install -r requirements.txt
```

For a Linux/CUDA paper run, `requirements-paper.txt` provides an auditable
reference stack:

```bash
pip install -r requirements-paper.txt
```

The paper explicitly reports vLLM 0.5.5 but does not publish the other package
versions. The remaining pins in that file are a compatible repository profile,
not a claim about the authors' original environment.

The released Llama readers are gated Hugging Face models, so accept their
licenses and authenticate before running the example.

## Quickstart

`ExitRAG` accepts already retrieved documents. It does not load a retriever.

```python
from exit_rag import Document, ExitRAG

rag = ExitRAG(
    compression_base_model="google/gemma-2b-it",
    compression_model="doubleyyh/exit-gemma-2b",
    reader_model="meta-llama/Llama-3.1-8B-Instruct",
    batch_size=8,
)

query = "How do solid-state drives improve computer performance?"
documents = [
    Document(
        title="Computer Storage Technologies",
        text=(
            "Solid-state drives use flash memory without moving parts. "
            "They provide faster data access and reduce boot and loading times. "
            "I bought my computer last week."
        ),
    )
]

result = rag.run_rag(query, documents)
print(result["compressed_context"])
print(result["answer"])
print(result["compression_time"], result["reading_time"], result["total_time"])
```

The compressor batches sentence prompts, but each prompt contains only the
candidate's source document. It never concatenates all Top-k documents into
`Full context`. A document that exceeds the Gemma context window raises an
error rather than silently truncating the candidate sentence or classifier
instruction.

## Reproducing the paper setup

The immutable reference settings and all items not reported by the paper are in
[`configs/paper_v3.json`](configs/paper_v3.json). The primary evaluation setup
is:

| Component | Paper setting |
|---|---|
| Retriever | Contriever-MSMARCO |
| Corpus | December 2018 Wikipedia dump |
| Retrieval depths | Top-5 and Top-20 |
| Compressor | Gemma-2B-it + EXIT adapter, 4-bit/float16 |
| Threshold | 0.5 |
| Readers | Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct |
| Splits | NQ dev, TriviaQA test, HotpotQA dev, 2WikiMultiHopQA dev |
| Metrics | EM, F1, token count, compression/read/total latency |

The repository runner starts from pre-retrieved CompAct-style JSON or JSONL:

```json
{
  "question": "Who wrote the novel?",
  "answers": ["Example Author"],
  "ctxs": [
    {"title": "Document title", "text": "Document text", "score": 12.3}
  ]
}
```

Validate an artifact without loading models:

```bash
python reproduce.py \
  --input data/hotpotqa-dev.jsonl \
  --output results/hotpotqa-top5.jsonl \
  --dataset hotpotqa \
  --top-k 5 \
  --validate-only
```

Run compression and QA:

```bash
python reproduce.py \
  --input data/hotpotqa-dev.jsonl \
  --output results/hotpotqa-top5.jsonl \
  --dataset hotpotqa \
  --top-k 5 \
  --latency-repeats 5
```

Raw predictions and sentence decisions are written to the requested JSONL. A
neighboring `*.summary.json` records metrics, token retention, per-stage
latency, input SHA-256, Git SHA, requested/resolved model revisions, and the
runtime environment.

The runner currently uses Transformers. The paper's latency numbers use vLLM
0.5.5 and therefore must not be compared directly with these timings. Retrieval
latency is excluded because the paper's reported end-to-end table measures
compression plus reader generation.

### Important reproduction limits

The paper does not report exact dataset/index/model revisions, the retrieval
index construction parameters, random seed, spaCy pipeline/version, LoRA target
modules, quantization subtype, maximum sequence length, or reader stopping
configuration. Pin these values and retain the generated manifests for any
strict comparison. The v3 paper also contains a few result-table differences,
so report confidence intervals and raw artifacts rather than only one headline
number.

## Training the compressor

The training pipeline uses HotpotQA supporting-fact annotations:

- positive: supporting-fact sentence;
- hard negative: another sentence in the same supporting document;
- random negative: a sentence paired with a different query;
- target ratio: `positive : hard negative : random negative = 2 : 1 : 1`.

Training retains HotpotQA's supplied sentence arrays so supporting-fact indices
stay aligned; inference uses spaCy's rule-based sentencizer. This is recorded in
the dataset manifest because the paper does not publish the exact alignment
procedure used between those two representations.

See [`train/README.md`](train/README.md) for deterministic data preparation,
paper hyperparameters, checkpointing, and classifier evaluation commands.

## Tests

```bash
pytest -q
```

The regression suite fixes the paper-critical behavior: document-local context,
batched sentence scoring, exact Yes/No normalization, no silent overflow,
original-order reassembly, and the valid empty-compression case.

## Citation

```bibtex
@article{hwang2024exit,
  title={EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation},
  author={Hwang, Taeho and Cho, Sukmin and Jeong, Soyeong and Song, Hoyun and Han, SeungYoon and Park, Jong C.},
  journal={arXiv preprint arXiv:2412.12559},
  year={2024}
}
```
