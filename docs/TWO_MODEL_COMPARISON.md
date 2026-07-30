# Two-model comparison

The primary experiment contains exactly two models:

1. **GPT** — conventional shared attention and FFN matrices.
2. **DT** — scalar-routed attention and FFN with split/merge enabled.

`--model both` trains these two models once on the same tokenizer, token stream, split, batch plan, optimizer schedule, and untied LM-head design.

DT structural behavior is controlled by `--structure-interval`:

- positive value: normal DT with periodic split/merge;
- `0`: optional static-routing ablation used only for diagnosis.

Static DT is not a third primary model. It is the same DT implementation with structural updates disabled.
