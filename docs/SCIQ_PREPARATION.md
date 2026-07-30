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

## Complete Google Colab preparation

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

The LM pretraining stream contains only unique non-empty `support` passages from
the train split. Questions and answers are excluded from LM pretraining so held-out
QA does not become silent memorization training.

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

`pretrain_train_tokens.pt` contains the exact `int32` token stream.
`metadata.json` records corpus hashes, byte/token counts, split sizes, and OOV rates.

## Train from the saved tokenizer and token IDs

```bash
new-dt-compare \
  --prepared-data data/sciq \
  --model both \
  --run-name sciq_static \
  --device cuda \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --ffn-dim 128 \
  --seq-len 64 \
  --steps 1000 \
  --batch-size 8 \
  --structure-interval 0
```

The comparison loader reads `tokenizer.json` and `pretrain_train_tokens.pt`
directly. It does not rebuild the tokenizer from `pretrain_train.txt`. The saved
vocabulary therefore remains exactly the canonical SciQ vocabulary used for later
QA evaluation.

The loader verifies format/version, vocabulary size, EOS ID, token count, dtype,
and ID range before training. Both GPT and sDT receive the same loaded tensor and
the same deterministic train/validation split.

A compact repository-safe archive named `pretrain_train_tokens.pt.gz.b64` is also
supported. It is decoded in memory and must contain the same PyTorch payload.
