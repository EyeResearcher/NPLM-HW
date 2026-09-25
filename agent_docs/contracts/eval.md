# Contract: `eval.py`

## Scope

Reconstruct a trained model from a checkpoint, evaluate every corpus split, and
write assignment-compatible metrics. Do not retrain or mutate checkpoints.

## Public API

```python
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    out_json: str | Path,
) -> dict[str, Any]:
    """Evaluate train/val/test, write metrics JSON, and return the same dict."""

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...

def main() -> None: ...
```

CLI:

```text
python -m nplm.eval --checkpoint runs/exp1/best.pt \
  --data_dir data/wikitext2_jsonl --out_json results/metrics.json
```

## Required behavior

1. Load the checkpoint on CPU and validate its schema.
2. Resolve the configured device and load the referenced `WordTokenizer`.
3. Verify tokenizer size equals checkpoint `model_config.vocab_size`.
4. Construct `NPLM` solely from `model_config` plus tokenizer `pad_id`, then
   load weights strictly.
5. Create nonshuffled train/val/test loaders using checkpoint config values.
6. Use `utils.evaluate_metrics` to compute token-weighted loss, micro top-1
   accuracy, perplexity, and target count for each split via TorchMetrics.
7. Write and return exactly the metrics schema in [README.md](README.md).

All splits are required by the chosen preprocessing contract. Missing/empty
splits, missing tokenizer artifacts, architecture mismatches, and nonfinite
metrics are errors; they must not become JSON `null`, NaN, or Infinity.

## Dependencies and outputs

Imports only public interfaces from `word_tokenizer.py`, `data.py`, `model.py`,
and `utils.py`. Output is one strict UTF-8 `metrics.json` file containing all
contracted loss, accuracy, perplexity, and run metadata fields; parent
directories are created as needed.

## Verification

- A known uniform-logit model has perplexity equal to vocabulary size.
- Uneven final batches yield token-weighted loss/perplexity and micro accuracy.
- Returned mapping exactly equals the JSON payload.
- Evaluation does not modify model parameters or checkpoint files.
