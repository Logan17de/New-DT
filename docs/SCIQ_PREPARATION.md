# SciQ download and word-space tokenization

The preparation command downloads `allenai/sciq`, builds the tokenizer from the
**training split only**, and keeps validation/test content out of the pretraining
corpus.

## Install

```bash
pip install -e ".[data]"
```

## Run

```bash
new-dt-prepare-sciq \
  --output data/sciq \
  --lowercase \
  --min-frequency 2
```

The root wrapper is equivalent:

```bash
python prepare_sciq.py --output data/sciq --lowercase --min-frequency 2
```

Use `--force` to replace files in an existing output directory. `--max-vocab N`
limits the vocabulary including `<pad>`, `<unk>`, `<bos>`, and `<eos>`.

## Complete Google Colab block

Run this in one Colab code cell:

```bash
!rm -rf /content/New-DT
!git clone https://github.com/Logan17de/New-DT.git /content/New-DT
%cd /content/New-DT
!git pull origin main
!pip install -q -e ".[data]"
!python prepare_sciq.py \
  --output /content/New-DT/data/sciq \
  --lowercase \
  --min-frequency 2 \
  --force
!ls -lh /content/New-DT/data/sciq
```

For later Colab sessions where the repository folder already exists:

```bash
%cd /content/New-DT
!git pull origin main
!pip install -q -e ".[data]"
!python prepare_sciq.py --output data/sciq --lowercase --min-frequency 2 --force
```

## Data separation

Tokenizer vocabulary training uses these fields from SciQ train only:

```text
support
question
correct_answer
distractor1
distractor2
distractor3
```

The LM pretraining text contains only unique non-empty `support` passages from the
train split. Questions and answers are intentionally excluded from that corpus so
the later QA test is not silently converted into memorization training.

## Output

```text
data/sciq/
├── tokenizer.json
├── vocab.txt
├── tokenizer_training.txt
├── pretrain_train.txt
├── pretrain_train_tokens.pt
├── qa_train.jsonl
├── qa_validation.jsonl
├── qa_test.jsonl
└── metadata.json
```

`pretrain_train_tokens.pt` contains an `int32` token stream and tokenizer metadata.
`metadata.json` records corpus hashes, byte/token counts, split sizes, and OOV rates
for all three QA splits.

## Use in the small comparison

For an immediate architecture-only language-model comparison, pass the generated
support corpus to both models:

```bash
new-dt-compare \
  --data data/sciq/pretrain_train.txt \
  --model both \
  --lowercase \
  --min-frequency 2 \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --steps 1000
```

That command rebuilds a support-only vocabulary, but it remains a fair GPT-versus-
sDT comparison because both models receive exactly the same tokenizer and batches.
It will not have identical IDs to `tokenizer.json`, whose vocabulary also covers
train questions and answers. Preserve `tokenizer.json` and
`pretrain_train_tokens.pt` as the canonical artifacts for the later pretraining-plus-
QA workflow.
