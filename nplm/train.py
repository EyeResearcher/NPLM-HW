"""Command-line training workflow for the feed-forward NPLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from .data import create_dataloaders
from .model import NPLM
from .utils import (
    Timer,
    create_tensorboard_writer,
    evaluate_metrics,
    load_checkpoint,
    load_config,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)
from .word_tokenizer import WordTokenizer


def _model_config(config: Mapping[str, Any], vocab_size: int) -> dict[str, Any]:
    """Return the architecture metadata stored in every checkpoint."""
    return {
        "vocab_size": vocab_size,
        "context_size": config["context_size"],
        "embedding_dim": config["embedding_dim"],
        "hidden_dim": config["hidden_dim"],
        "activation": config["activation"],
        "dropout": config["dropout"],
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    tokenizer_type: str,
    tokenizer_path: str,
) -> None:
    """Reject checkpoints whose token IDs or model shape cannot be resumed."""
    saved_model_config = checkpoint["model_config"]
    mismatches = [
        key
        for key, current_value in model_config.items()
        if saved_model_config.get(key) != current_value
    ]
    if mismatches:
        details = ", ".join(
            f"{key} (checkpoint={saved_model_config.get(key)!r}, "
            f"config={model_config[key]!r})"
            for key in mismatches
        )
        raise ValueError(f"Resume checkpoint architecture mismatch: {details}")

    if checkpoint["tokenizer_type"] != tokenizer_type:
        raise ValueError(
            "Resume checkpoint tokenizer type mismatch: "
            f"checkpoint={checkpoint['tokenizer_type']!r}, "
            f"config={tokenizer_type!r}"
        )
    if Path(checkpoint["tokenizer_path"]).resolve() != Path(tokenizer_path).resolve():
        raise ValueError(
            "Resume checkpoint tokenizer path mismatch: "
            f"checkpoint={checkpoint['tokenizer_path']!r}, "
            f"config={tokenizer_path!r}"
        )


def _save(
    path: Path,
    *,
    model: NPLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    tokenizer_path: str,
    train_time_sec: float,
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        best_val_loss=best_val_loss,
        model_config=model_config,
        training_config=training_config,
        tokenizer_type="word",
        tokenizer_path=tokenizer_path,
        train_time_sec=train_time_sec,
    )


def train(
    config_path: str | Path,
    data_dir: str | Path,
    save_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    """Train an NPLM and return the path to its best checkpoint."""
    config = load_config(config_path)
    destination = Path(save_dir)
    config["run_name"] = config["run_name"] or destination.name
    if not config["run_name"]:
        raise ValueError("run_name could not be derived from save_dir")
    set_seed(config["seed"])

    tokenizer_path = str(config["tokenizer_path"])
    tokenizer = WordTokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.id_to_token)
    if "vocab_size" in config and config["vocab_size"] != vocab_size:
        raise ValueError(
            "Configured vocab_size does not match the tokenizer artifact: "
            f"config={config['vocab_size']}, tokenizer={vocab_size}"
        )
    if tokenizer.pad_id is None:
        raise ValueError("The tokenizer vocabulary does not define <pad>")

    loaders = create_dataloaders(
        data_dir,
        tokenizer,
        config["context_size"],
        config["batch_size"],
        num_workers=config["num_workers"],
        shuffle_train=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty")
    if len(val_loader) == 0:
        raise RuntimeError("Validation loader is empty")

    device = resolve_device(config["device"])
    architecture = _model_config(config, vocab_size)
    model = NPLM(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        activation=config["activation"],
        dropout=config["dropout"],
        padding_idx=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    step = 0
    best_val_loss = math.inf
    prior_train_time = 0.0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        _validate_resume(
            checkpoint,
            model_config=architecture,
            tokenizer_type=config["tokenizer_type"],
            tokenizer_path=tokenizer_path,
        )
        if checkpoint["optimizer_state_dict"] is None:
            raise ValueError("Resume checkpoint does not contain optimizer state")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        prior_train_time = checkpoint["train_time_sec"]
        if step > config["max_steps"]:
            raise ValueError(
                f"Resume step {step} exceeds configured max_steps "
                f"{config['max_steps']}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best.pt"
    last_path = destination / "last.pt"
    write_json(destination / "config.json", config)
    writer = create_tensorboard_writer(destination / "tensorboard")
    timer = Timer()
    train_loss_metric = MeanMetric().to(device)
    train_accuracy_metric = MulticlassAccuracy(
        num_classes=vocab_size, average="micro"
    ).to(device)
    train_iterator = iter(train_loader)
    last_evaluated_step: int | None = None
    latest_train_metrics: tuple[float, float] | None = None
    train_target_count = 0

    def elapsed() -> float:
        return prior_train_time + timer.elapsed()

    def current_train_metrics() -> tuple[float, float] | None:
        if train_target_count == 0:
            return latest_train_metrics
        return (
            float(train_loss_metric.compute().item()),
            float(train_accuracy_metric.compute().item()),
        )

    def log_training() -> tuple[float, float]:
        nonlocal latest_train_metrics, train_target_count
        metrics = current_train_metrics()
        if metrics is None:
            raise RuntimeError("Training metrics contain no targets")
        train_loss, train_accuracy = metrics
        latest_train_metrics = metrics
        writer.add_scalar("train/loss", train_loss, step)
        writer.add_scalar("train/accuracy", train_accuracy, step)
        writer.add_scalar(
            "train/learning_rate", float(optimizer.param_groups[0]["lr"]), step
        )
        writer.add_scalar("system/elapsed_seconds", elapsed(), step)
        train_loss_metric.reset()
        train_accuracy_metric.reset()
        train_target_count = 0
        return metrics

    def evaluate_and_maybe_save(
        train_metrics: tuple[float, float] | None,
    ) -> None:
        nonlocal best_val_loss, last_evaluated_step
        validation = evaluate_metrics(
            model, val_loader, device, num_classes=vocab_size
        )
        val_loss = float(validation["loss"])
        val_ppl = float(validation["perplexity"])
        val_accuracy = float(validation["accuracy"])
        current_elapsed = elapsed()
        writer.add_scalar("validation/loss", val_loss, step)
        writer.add_scalar("validation/perplexity", val_ppl, step)
        writer.add_scalar("validation/accuracy", val_accuracy, step)
        writer.add_scalar("system/elapsed_seconds", current_elapsed, step)

        if train_metrics is None:
            train_text = "train_loss=n/a train_accuracy=n/a"
        else:
            train_text = (
                f"train_loss={train_metrics[0]:.4f} "
                f"train_accuracy={train_metrics[1]:.4f}"
            )
        print(
            f"step={step} {train_text} val_loss={val_loss:.4f} "
            f"val_ppl={val_ppl:.4f} val_accuracy={val_accuracy:.4f} "
            f"elapsed_sec={current_elapsed:.1f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(
                best_path,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                model_config=architecture,
                training_config=config,
                tokenizer_path=tokenizer_path,
                train_time_sec=current_elapsed,
            )
        last_evaluated_step = step
        writer.flush()

    try:
        writer.add_text("run/name", str(config["run_name"]), step)
        writer.add_text(
            "run/config", json.dumps(config, indent=2, sort_keys=True), step
        )

        while step < config["max_steps"]:
            try:
                contexts, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                try:
                    contexts, targets = next(train_iterator)
                except StopIteration as exc:
                    raise RuntimeError("Training loader is empty") from exc

            contexts = contexts.to(device)
            targets = targets.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(contexts)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Training loss is nonfinite at step {step + 1}: "
                    f"{loss.item()!r}"
                )
            loss.backward()
            if config["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["clip_grad_norm"]
                )
            optimizer.step()
            step += 1

            train_loss_metric.update(loss.detach(), weight=targets.numel())
            train_accuracy_metric.update(logits.detach(), targets)
            train_target_count += targets.numel()
            metrics_for_progress = current_train_metrics()
            if step % config["log_every"] == 0:
                metrics_for_progress = log_training()

            if step % config["eval_every"] == 0:
                evaluate_and_maybe_save(metrics_for_progress)

        if last_evaluated_step != step:
            evaluate_and_maybe_save(current_train_metrics())

        # A fresh run always improves on infinity. On resume, the sibling best
        # checkpoint is expected to exist unless the resumed parameters improve it.
        if not best_path.exists():
            resume_best = (
                Path(resume_from).with_name("best.pt") if resume_from else None
            )
            if resume_best is None or not resume_best.is_file():
                raise RuntimeError(
                    "No best checkpoint was produced; resume in the original save "
                    "directory or resume from a checkpoint that improves validation loss"
                )
            if resume_best.resolve() != best_path.resolve():
                best_checkpoint = load_checkpoint(resume_best, map_location="cpu")
                _validate_resume(
                    best_checkpoint,
                    model_config=architecture,
                    tokenizer_type=config["tokenizer_type"],
                    tokenizer_path=tokenizer_path,
                )
                best_model = NPLM(**architecture, padding_idx=tokenizer.pad_id)
                best_model.load_state_dict(best_checkpoint["model_state_dict"])
                best_optimizer = torch.optim.Adam(
                    best_model.parameters(), lr=config["lr"]
                )
                if best_checkpoint["optimizer_state_dict"] is not None:
                    best_optimizer.load_state_dict(
                        best_checkpoint["optimizer_state_dict"]
                    )
                _save(
                    best_path,
                    model=best_model,
                    optimizer=best_optimizer,
                    step=best_checkpoint["step"],
                    best_val_loss=best_checkpoint["best_val_loss"],
                    model_config=architecture,
                    training_config=config,
                    tokenizer_path=tokenizer_path,
                    train_time_sec=best_checkpoint["train_time_sec"],
                )

        _save(
            last_path,
            model=model,
            optimizer=optimizer,
            step=step,
            best_val_loss=best_val_loss,
            model_config=architecture,
            training_config=config,
            tokenizer_path=tokenizer_path,
            train_time_sec=elapsed(),
        )
        writer.flush()
        return best_path
    finally:
        writer.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feed-forward NPLM")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing train/val/test"
    )
    parser.add_argument("--save_dir", required=True, help="Checkpoint directory")
    parser.add_argument(
        "--resume_from", default=None, help="Optional checkpoint to resume"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    best_path = train(
        args.config,
        args.data_dir,
        args.save_dir,
        resume_from=args.resume_from,
    )
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
