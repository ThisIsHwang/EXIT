# Reproducing the EXIT classifier

These scripts reproduce the classifier data and fine-tuning setup described in
[EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented
Generation](https://arxiv.org/abs/2412.12559).

## 1. Build the HotpotQA dataset

Download `hotpot_train_v1.1.json`, then run:

```bash
python -m train.datasampling \
  --dataset_path data/hotpotqa/hotpot_train_v1.1.json \
  --save_dir data/exit-classifier \
  --validation_size 1000 \
  --seed 42
```

The split happens at query level before sentence sampling, so a query cannot
leak between train and validation. Query assignment uses a stable SHA-256 rank
of `(seed, query_id)` rather than input order or process-global randomness. The
paper does not publish its seed or exact query split; seed 42 and this split
algorithm are repository choices recorded in the manifest.

For each split, the script creates the paper's exact integer ratio:

| Type | Ratio | Construction |
| --- | ---: | --- |
| Positive | 2 | HotpotQA supporting-fact sentence |
| Hard negative | 1 | Non-supporting sentence in the same supporting passage |
| Random negative | 1 | Current query paired with a passage/sentence from a different query |

At most one positive is omitted when a split has an odd positive count. If hard
negatives are the limiting category, all categories are reduced together; rows
are never duplicated to manufacture the ratio. HotpotQA's provided sentence
boundaries are retained because supporting-fact annotations refer to those
sentence indices.

The output is:

```text
data/exit-classifier/
|-- train_dataset/
|-- validation_dataset/
`-- manifest.json
```

`manifest.json` records the input SHA-256, seed, split algorithm, query and row
digests, category counts, shared prompt hash, and package versions. It also
verifies that every random-negative source query differs from its paired query.
Training, inference, and evaluation use the same prompt implementation in
`compressors/baselines/exit/core.py`.

## 2. Train Gemma-2B-it

Install `requirements-paper.txt` on a CUDA system with bitsandbytes, then run:

```bash
python -m train.train \
  --model_id google/gemma-2b-it \
  --model_revision YOUR_PINNED_REVISION \
  --train_dataset data/exit-classifier/train_dataset \
  --validation_dataset data/exit-classifier/validation_dataset \
  --output_dir outputs/exit-gemma-2b \
  --seed 42
```

The paper-reported defaults are:

| Setting | Value |
| --- | ---: |
| Per-device batch size | 8 |
| Gradient accumulation | 8 |
| Learning rate | `1e-5` |
| Weight decay | `0.1` |
| Warmup ratio | `0.03` |
| Epochs | 1 |
| Optimizer | `paged_adamw_8bit` |
| Quantization | 4-bit, fp16 compute |
| LoRA | rank 64, alpha 32, dropout 0.05 |

Loss is masked to the single final `Yes`/`No` completion token. The base model
is prepared for k-bit training and wrapped with PEFT exactly once. Prompts that
exceed `max_seq_length` fail explicitly instead of silently removing document
context. The best checkpoint is selected by validation loss.

`training_manifest.json` records the run configuration, Git SHA, package
versions, dataset-manifest hash, requested/resolved model revision, and
implementation choices the paper did not report (NF4/double quantization, LoRA
target modules, loss scope, and overflow policy).

Weights & Biases is optional and is not imported or initialized by default:

```bash
python -m train.train \
  --train_dataset data/exit-classifier/train_dataset \
  --validation_dataset data/exit-classifier/validation_dataset \
  --output_dir outputs/exit-gemma-2b \
  --wandb_project exit \
  --experiment_name exit-gemma-2b-seed42
```

Resume with `--resume_from_checkpoint outputs/exit-gemma-2b/checkpoint-N`.

## 3. Evaluate the adapter

```bash
python -m train.evaluate \
  --base_model google/gemma-2b-it \
  --base_revision YOUR_PINNED_REVISION \
  --checkpoint outputs/exit-gemma-2b/final_model \
  --validation_dataset data/exit-classifier/validation_dataset \
  --output_dir outputs/exit-gemma-2b/evaluation \
  --threshold 0.5 \
  --batch_size 16 \
  --seed 42
```

Evaluation normalizes only the next-token `Yes` and `No` logits and applies the
paper's default threshold of 0.5. It writes JSON, a text classification report,
the confusion matrix, individual predictions, and metrics broken down by hard
versus random negatives. It also rejects overlong prompts instead of truncating
them. Evaluation uses the paper's 4-bit profile by default; use
`--no_4bit --device cpu` only for a non-comparable CPU compatibility run.

`train/evalutate.py` remains as a compatibility wrapper for the historical
misspelled entry point. New code should use `train/evaluate.py`.
