# Canonical SciQ prepared data

`metadata.json` records the validated package used for the first GPT-versus-sDT
experiment:

- vocabulary: 26,466 word-space tokens;
- pretraining stream: 805,311 token IDs;
- unique support passages: 10,473;
- tokenizer: lowercase, minimum frequency 2;
- special IDs: pad=0, unk=1, bos=2, eos=3.

The generated corpus files are intentionally produced by `new-dt-prepare-sciq`
rather than duplicated in source control. Place or generate these files beside this
README before training:

```text
tokenizer.json
pretrain_train_tokens.pt
qa_train.jsonl
qa_validation.jsonl
qa_test.jsonl
```

Generate them:

```bash
new-dt-prepare-sciq --output data/sciq --lowercase --min-frequency 2 --force
```

Or extract the prepared package so the final path is `data/sciq/tokenizer.json`.
Then train without rebuilding token IDs:

```bash
new-dt-compare --prepared-data data/sciq --model both ...
```
