"""Evaluate a trained NPLM checkpoint on every corpus split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

from .data import create_dataloaders
from .model import NPLM
from .utils import (
    evaluate_metrics,
    load_checkpoint,
    resolve_device,
    write_json,
)
from .word_tokenizer import WordTokenizer


def _require_finite_number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    out_json: str | Path,
) -> dict[str, Any]:
    """Evaluate train/val/test, write metrics JSON, and return the payload."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    training_config = checkpoint["training_config"]

    device = resolve_device(training_config["device"])
    tokenizer = WordTokenizer.load(checkpoint["tokenizer_path"])
    vocab_size = len(tokenizer.id_to_token)
    expected_vocab_size = model_config["vocab_size"]
    if vocab_size != expected_vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match checkpoint model_config: "
            f"{vocab_size} != {expected_vocab_size}"
        )

    model = NPLM(
        vocab_size=expected_vocab_size,
        context_size=model_config["context_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        activation=model_config["activation"],
        dropout=model_config["dropout"],
        padding_idx=tokenizer.pad_id,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    loaders = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        context_size=model_config["context_size"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle_train=False,
    )

    split_metrics: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        evaluated = evaluate_metrics(
            model,
            loaders[split],
            device,
            num_classes=vocab_size,
        )
        target_count = evaluated["target_count"]
        if target_count <= 0:
            raise ValueError(f"{split} split contains no evaluation targets")
        loss = _require_finite_number(evaluated["loss"], f"{split} loss")
        accuracy = _require_finite_number(
            evaluated["accuracy"], f"{split} accuracy"
        )
        split_ppl = _require_finite_number(
            evaluated["perplexity"], f"{split} perplexity"
        )
        if loss < 0:
            raise ValueError(f"{split} loss must be nonnegative")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"{split} accuracy must be in [0, 1]")
        if split_ppl < 1.0:
            raise ValueError(f"{split} perplexity must be at least 1")
        split_metrics[split] = {
            "loss": loss,
            "accuracy": accuracy,
            "perplexity": split_ppl,
            "target_count": target_count,
        }

    train_time_sec = _require_finite_number(
        checkpoint["train_time_sec"], "checkpoint train_time_sec"
    )
    metrics: dict[str, Any] = {
        "train_loss": split_metrics["train"]["loss"],
        "val_loss": split_metrics["val"]["loss"],
        "test_loss": split_metrics["test"]["loss"],
        "train_accuracy": split_metrics["train"]["accuracy"],
        "val_accuracy": split_metrics["val"]["accuracy"],
        "test_accuracy": split_metrics["test"]["accuracy"],
        "train_ppl": split_metrics["train"]["perplexity"],
        "val_ppl": split_metrics["val"]["perplexity"],
        "test_ppl": split_metrics["test"]["perplexity"],
        "tokenizer": checkpoint["tokenizer_type"],
        "vocab_size_or_merges": vocab_size,
        "context_size": model_config["context_size"],
        "train_time_sec": train_time_sec,
    }
    write_json(out_json, metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate an NPLM checkpoint on train, validation, and test data."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing train/val/test splits"
    )
    parser.add_argument("--out_json", required=True, help="Metrics JSON output path")
    return parser.parse_args(argv)


def main() -> None:
    """Run the evaluation CLI."""
    args = parse_args()
    evaluate_checkpoint(args.checkpoint, args.data_dir, args.out_json)


if __name__ == "__main__":
    main()
