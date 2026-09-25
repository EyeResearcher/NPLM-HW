"""Shared utilities for NPLM training and evaluation."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy


_CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 1337,
    "tokenizer_type": "word",
    "activation": "tanh",
    "dropout": 0.0,
    "optimizer": "adam",
    "clip_grad_norm": None,
    "device": "auto",
    "num_workers": 0,
    "log_every": 100,
    "run_name": None,
}
_REQUIRED_CONFIG_KEYS = {
    "tokenizer_path",
    "context_size",
    "embedding_dim",
    "hidden_dim",
    "lr",
    "batch_size",
    "max_steps",
    "eval_every",
}
_OPTIONAL_CONFIG_KEYS = {"vocab_size"}
_KNOWN_CONFIG_KEYS = (
    set(_CONFIG_DEFAULTS) | _REQUIRED_CONFIG_KEYS | _OPTIONAL_CONFIG_KEYS
)

_CHECKPOINT_KEYS = {
    "format_version",
    "model_state_dict",
    "optimizer_state_dict",
    "step",
    "best_val_loss",
    "model_config",
    "training_config",
    "tokenizer_type",
    "tokenizer_path",
    "train_time_sec",
}
_MODEL_CONFIG_KEYS = {
    "vocab_size",
    "context_size",
    "embedding_dim",
    "hidden_dim",
    "activation",
    "dropout",
}


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config[key]
    if not _is_int(value) or value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive integer")
    return int(value)


def _require_positive_float(config: Mapping[str, Any], key: str) -> float:
    value = config[key]
    if not _is_real(value) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive finite number")
    return float(value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, apply defaults, and validate the shared schema."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration in {config_path}: {error}") from error

    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration must be a YAML mapping")

    unknown = set(loaded) - _KNOWN_CONFIG_KEYS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown configuration field(s): {names}")

    missing = _REQUIRED_CONFIG_KEYS - set(loaded)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required configuration field(s): {names}")

    config = {**_CONFIG_DEFAULTS, **dict(loaded)}

    seed = config["seed"]
    if not _is_int(seed):
        raise ValueError("Config field 'seed' must be an integer")
    config["seed"] = int(seed)

    if config["tokenizer_type"] != "word":
        raise ValueError("Config field 'tokenizer_type' must be 'word'")
    if not isinstance(config["tokenizer_path"], str) or not config[
        "tokenizer_path"
    ].strip():
        raise ValueError("Config field 'tokenizer_path' must be a nonempty string")

    for key in (
        "context_size",
        "embedding_dim",
        "hidden_dim",
        "batch_size",
        "max_steps",
        "log_every",
        "eval_every",
    ):
        config[key] = _require_positive_int(config, key)

    if "vocab_size" in config:
        config["vocab_size"] = _require_positive_int(config, "vocab_size")

    if config["activation"] not in {"tanh", "relu"}:
        raise ValueError("Config field 'activation' must be 'tanh' or 'relu'")

    dropout = config["dropout"]
    if (
        not _is_real(dropout)
        or not math.isfinite(float(dropout))
        or not 0.0 <= dropout < 1.0
    ):
        raise ValueError("Config field 'dropout' must be in [0, 1)")
    config["dropout"] = float(dropout)

    if config["optimizer"] != "adam":
        raise ValueError("Config field 'optimizer' must be 'adam'")
    config["lr"] = _require_positive_float(config, "lr")

    clip_grad_norm = config["clip_grad_norm"]
    if clip_grad_norm is not None:
        if (
            not _is_real(clip_grad_norm)
            or not math.isfinite(float(clip_grad_norm))
            or clip_grad_norm <= 0
        ):
            raise ValueError(
                "Config field 'clip_grad_norm' must be null or a positive finite number"
            )
        config["clip_grad_norm"] = float(clip_grad_norm)

    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Config field 'device' must be 'auto', 'cpu', or 'cuda'")

    num_workers = config["num_workers"]
    if not _is_int(num_workers) or num_workers < 0:
        raise ValueError("Config field 'num_workers' must be a nonnegative integer")
    config["num_workers"] = int(num_workers)

    run_name = config["run_name"]
    if run_name is not None and (
        not isinstance(run_name, str) or not run_name.strip()
    ):
        raise ValueError("Config field 'run_name' must be null or a nonempty string")

    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and all available CUDA devices."""

    if not _is_int(seed):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an ``auto``, ``cpu``, or ``cuda`` device request."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("requested device must be 'auto', 'cpu', or 'cuda'")


def perplexity(mean_nll: float) -> float:
    """Return ``exp(mean_nll)`` and reject invalid or overflowing input."""

    if (
        not _is_real(mean_nll)
        or not math.isfinite(float(mean_nll))
        or mean_nll < 0
    ):
        raise ValueError("mean_nll must be a nonnegative finite number")
    try:
        result = math.exp(float(mean_nll))
    except OverflowError as error:
        raise ValueError("mean_nll is too large to compute finite perplexity") from error
    if not math.isfinite(result):
        raise ValueError("mean_nll is too large to compute finite perplexity")
    return result


def evaluate_nll(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    """Return summed negative log likelihood and target count for a loader."""

    was_training = model.training
    total_nll = 0.0
    target_count = 0
    model.eval()
    try:
        with torch.no_grad():
            for contexts, targets in dataloader:
                contexts = contexts.to(device)
                targets = targets.to(device)
                logits = model(contexts)
                total_nll += F.cross_entropy(
                    logits, targets, reduction="sum"
                ).item()
                target_count += targets.numel()
    finally:
        model.train(was_training)
    return total_nll, target_count


def evaluate_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, float | int]:
    """Compute token-weighted loss, perplexity, and micro top-1 accuracy."""

    if not _is_int(num_classes) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    loss_metric = MeanMetric().to(device)
    accuracy_metric = MulticlassAccuracy(
        num_classes=int(num_classes), average="micro"
    ).to(device)
    was_training = model.training
    target_count = 0
    model.eval()
    try:
        with torch.no_grad():
            for contexts, targets in dataloader:
                contexts = contexts.to(device)
                targets = targets.to(device)
                logits = model(contexts)
                batch_loss = F.cross_entropy(logits, targets, reduction="mean")
                loss_metric.update(batch_loss, weight=targets.numel())
                accuracy_metric.update(logits, targets)
                target_count += targets.numel()

        if target_count == 0:
            raise ValueError("Cannot evaluate metrics on an empty dataloader")

        mean_loss = float(loss_metric.compute().item())
        accuracy = float(accuracy_metric.compute().item())
        return {
            "loss": mean_loss,
            "perplexity": perplexity(mean_loss),
            "accuracy": accuracy,
            "target_count": target_count,
        }
    finally:
        loss_metric.reset()
        accuracy_metric.reset()
        model.train(was_training)


def create_tensorboard_writer(log_dir: str | Path) -> SummaryWriter:
    """Create a TensorBoard writer after ensuring its directory exists."""

    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(destination))


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Write via a sibling temporary file and atomically replace ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


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
    """Atomically write a version-one NPLM checkpoint."""

    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": tokenizer_path,
        "train_time_sec": train_time_sec,
    }
    destination = Path(path)
    _atomic_write(destination, lambda temporary: torch.save(checkpoint, temporary))


def _validate_checkpoint(checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    missing = _CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Checkpoint is missing required field(s): {names}")
    if checkpoint["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: {checkpoint['format_version']!r}"
        )
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("Checkpoint field 'model_state_dict' must be a mapping")
    optimizer_state = checkpoint["optimizer_state_dict"]
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise ValueError(
            "Checkpoint field 'optimizer_state_dict' must be a mapping or null"
        )
    if not _is_int(checkpoint["step"]) or checkpoint["step"] < 0:
        raise ValueError("Checkpoint field 'step' must be a nonnegative integer")
    if not _is_real(checkpoint["best_val_loss"]):
        raise ValueError("Checkpoint field 'best_val_loss' must be numeric")

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint field 'model_config' must be a mapping")
    missing_model_fields = _MODEL_CONFIG_KEYS - set(model_config)
    if missing_model_fields:
        names = ", ".join(sorted(missing_model_fields))
        raise ValueError(f"Checkpoint model_config is missing field(s): {names}")
    for key in ("vocab_size", "context_size", "embedding_dim", "hidden_dim"):
        if not _is_int(model_config[key]) or model_config[key] <= 0:
            raise ValueError(f"Checkpoint model_config field {key!r} must be positive")
    if model_config["activation"] not in {"tanh", "relu"}:
        raise ValueError(
            "Checkpoint model_config field 'activation' must be 'tanh' or 'relu'"
        )
    dropout = model_config["dropout"]
    if not _is_real(dropout) or not 0 <= dropout < 1:
        raise ValueError("Checkpoint model_config field 'dropout' must be in [0, 1)")

    if not isinstance(checkpoint["training_config"], Mapping):
        raise ValueError("Checkpoint field 'training_config' must be a mapping")
    if checkpoint["tokenizer_type"] != "word":
        raise ValueError("Checkpoint field 'tokenizer_type' must be 'word'")
    if (
        not isinstance(checkpoint["tokenizer_path"], str)
        or not checkpoint["tokenizer_path"]
    ):
        raise ValueError("Checkpoint field 'tokenizer_path' must be a nonempty string")
    train_time = checkpoint["train_time_sec"]
    if not _is_real(train_time) or not math.isfinite(float(train_time)) or train_time < 0:
        raise ValueError(
            "Checkpoint field 'train_time_sec' must be a nonnegative finite number"
        )
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a checkpoint without constructing model objects."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return _validate_checkpoint(checkpoint)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write an indented, strict UTF-8 JSON object."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    _atomic_write(Path(path), writer)


class Timer:
    """A monotonic elapsed-time timer."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed seconds since this timer was created."""

        return time.perf_counter() - self._started_at
