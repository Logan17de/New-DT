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

The current comparison runner deterministically rebuilds the same tokenizer from
its `--data` text. To pretrain on the SciQ support corpus, pass the generated text:

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

Because `WordSpaceTokenizer` is deterministic, matching `--lowercase`,
`--min-frequency`, and `--max-vocab` recreates the same support-corpus vocabulary.
The saved tokenizer additionally includes question/answer vocabulary for later QA
evaluation, so retain `data/sciq/tokenizer.json` as the canonical tokenizer for the
full pretraining-plus-QA experiment.
