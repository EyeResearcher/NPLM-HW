# Contract: `utils.py`

## Scope

Provide small shared helpers for configuration, reproducibility, devices,
checkpoint I/O, TorchMetrics-based evaluation, TensorBoard writer creation,
JSON output, and elapsed time. Do not own model construction or training policy.

## Public API

```python
def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, apply shared defaults, validate, and return a plain dict."""

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and all available CUDA devices."""

def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda; explicit unavailable CUDA raises RuntimeError."""

def perplexity(mean_nll: float) -> float:
    """Return exp(mean_nll), raising ValueError for invalid input/overflow."""

def evaluate_nll(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    """Return (total_nll, target_count) using summed cross-entropy."""

def evaluate_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, float | int]:
    """Return loss, perplexity, accuracy, and target_count for one split."""

def create_tensorboard_writer(log_dir: str | Path) -> SummaryWriter:
    """Create a TensorBoard SummaryWriter, including parent directories."""

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    step: int,
    best_val_loss: float,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    tokenizer_type: str,
    tokenizer_path: str,
    train_time_sec: float,
) -> None:
    """Atomically write the checkpoint schema."""

def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate the checkpoint schema without constructing objects."""

def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write indented strict UTF-8 JSON."""

class Timer:
    def elapsed(self) -> float: ...
```

`evaluate_nll` remains available as the low-level compatibility API.
`evaluate_metrics` is the canonical train/eval aggregation API. It temporarily
switches the model to evaluation mode, uses `torch.no_grad()`, restores the
caller's previous mode, and returns exactly:

```python
{
    "loss": float,        # token-weighted mean cross-entropy
    "perplexity": float,  # exp(loss)
    "accuracy": float,    # micro top-1 next-token accuracy in [0, 1]
    "target_count": int,
}
```

It must use TorchMetrics `MeanMetric` and `MulticlassAccuracy`, not manually
average per-batch scalar metrics. An empty loader is an error.

`load_config` validates `log_every` and `run_name` from the shared schema.
Configuration needs PyYAML; aggregation needs `torchmetrics>=1.3,<2`; and event
writing needs `tensorboard>=2.14,<3`. The implementing agent must add missing
dependencies to `pyproject.toml`.

## Dependencies and consumers

Dependencies: standard library, NumPy, PyTorch, PyYAML, TorchMetrics, and
TensorBoard through `torch.utils.tensorboard`. It has no NPLM imports.
Consumers: `train.py` and `eval.py`.

## Verification

- Deterministic seed smoke test.
- Weighted NLL differs correctly from an unweighted mean on uneven batches.
- TorchMetrics loss and accuracy match a hand-computed uneven-batch fixture,
  and repeated calls do not retain metric state.
- TensorBoard writer creates an event file that contains the required scalar
  tags after flush/close.
- Checkpoint round trip and rejection of missing/wrong-version fields.
- Atomic writers leave no partial destination on a simulated failure.
