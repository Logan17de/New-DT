# Upload the prepared SciQ package

Place the prepared archive at:

```text
data/sciq.zip
```

The archive should contain a top-level `sciq/` directory with at least:

```text
sciq/tokenizer.json
sciq/pretrain_train_tokens.pt
sciq/metadata.json
```

The training CLI reads these files directly from the ZIP; extraction is not required.

In the GitHub website, open the `data` directory, choose **Add file → Upload files**, and upload `sciq.zip`.
