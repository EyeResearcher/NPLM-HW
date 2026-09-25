# Contract: `train.py`

## Scope

Compose the tokenizer, datasets, model, and utilities into the required training
workflow. This agent owns `configs/tiny.yaml` and `configs/medium.yaml` in
addition to `train.py`; the tiny config must be CPU-runnable in under ten minutes
on a typical CPU. Each `save_dir` is a separately comparable experiment run.

## Public API

```python
def train(
    config_path: str | Path,
    data_dir: str | Path,
    save_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    """Train and return the path to save_dir/best.pt."""

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...

def main() -> None: ...
```

CLI:

```text
python -m nplm.train --config configs/tiny.yaml \
  --data_dir data/wikitext2_jsonl --save_dir runs/exp1 \
  [--resume_from runs/exp1/last.pt]
```

## Required behavior

1. Load and validate config; seed before constructing loaders or model.
2. Load `WordTokenizer` from `tokenizer_path`; derive `V` and `padding_idx`.
3. Construct train/val/test loaders, though training uses train and val only.
4. Construct `NPLM` and Adam; restore resume state when supplied.
5. Optimize mean cross-entropy until `max_steps`. Clip gradients when configured.
6. Create a TensorBoard writer at `<save_dir>/tensorboard`, resolve `run_name`
   from config or `Path(save_dir).name`, and atomically write the fully normalized
   config to `<save_dir>/config.json` before optimization.
7. Aggregate training loss with TorchMetrics `MeanMetric` weighted by target
   count and next-token accuracy with `MulticlassAccuracy`. Every `log_every`
   steps, write/reset `train/loss`, `train/accuracy`, and
   `train/learning_rate`; also write `system/elapsed_seconds`.
8. Every `eval_every` steps, compute validation loss, perplexity, and accuracy
   with `utils.evaluate_metrics`. Write the validation scalars and save
   `best.pt` only on strict validation-loss improvement.
9. Always save `last.pt` at normal completion. Ensure at least one validation
   pass and one `best.pt`, even when `max_steps < eval_every`.
10. Print concise step, aggregated training loss/accuracy,
    validation loss/perplexity/accuracy, and elapsed-time progress. Flush after
    validation/checkpoint writes and close the TensorBoard writer in `finally`.

Resume restores model, optimizer, completed step, best loss, and prior elapsed
time. It validates model/tokenizer architecture. `max_steps` is the final global
step, not a number of additional steps.

If the train loader exhausts before `max_steps`, start another epoch with a new
shuffle and continue. Empty loaders and nonfinite loss raise `RuntimeError`.

## Dependencies and outputs

Imports only the public interfaces of `word_tokenizer.py`, `data.py`, `model.py`,
and `utils.py`, plus PyTorch and TorchMetrics.

Outputs are `best.pt`, `last.pt`, `config.json`, and TensorBoard event files
using the shared contracts. Training does not write final `metrics.json`; that
is `eval.py`'s responsibility. Separate configs must use separate `save_dir`
values; `tensorboard --logdir runs` must discover and compare all nested runs.

## Verification

- Tiny synthetic corpus overfits and produces both checkpoints.
- Resume continues from the stored global step.
- Best selection compares validation mean NLL, not batch loss or perplexity text.
- Checkpoint metadata reconstructs the model without the training config file.
- Uneven batches produce token-weighted training loss and micro token accuracy.
- Two runs with different configs/save directories produce distinct event files
  with identical required tags and their own configuration metadata.
- A resumed run logs from the restored global step without overwriting earlier
  TensorBoard events.
