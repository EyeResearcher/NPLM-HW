"""Cross-component checks for run observability and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nplm.eval import evaluate_checkpoint
from nplm.train import train
from nplm.utils import load_checkpoint
from nplm.word_tokenizer import TokenizerConfig, WordTokenizer


REQUIRED_SCALAR_TAGS = {
    "train/loss",
    "train/accuracy",
    "train/learning_rate",
    "validation/loss",
    "validation/perplexity",
    "validation/accuracy",
    "system/elapsed_seconds",
}
METRICS_KEYS = {
    "train_loss",
    "val_loss",
    "test_loss",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_ppl",
    "val_ppl",
    "test_ppl",
    "tokenizer",
    "vocab_size_or_merges",
    "context_size",
    "train_time_sec",
}


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}) + "\n")


def _make_corpus_and_tokenizer(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    documents = ["alpha beta alpha", "beta gamma alpha"]
    for split in ("train", "val", "test"):
        _write_jsonl(data_dir / split / "shard_00000.jsonl", documents)

    tokenizer = WordTokenizer.build_from_corpus(
        str(data_dir / "train"),
        TokenizerConfig(min_freq=1, max_vocab=20),
        progress=False,
    )
    tokenizer_path = root / "vocab.json"
    tokenizer.save(str(tokenizer_path))
    return data_dir, tokenizer_path


def _write_config(
    path: Path,
    tokenizer_path: Path,
    *,
    hidden_dim: int,
    learning_rate: float,
    max_steps: int,
) -> None:
    # Deliberately omit defaulted fields and run_name. The persisted snapshot
    # must contain the normalized values and derive run_name from save_dir.
    config = {
        "tokenizer_path": str(tokenizer_path),
        "context_size": 2,
        "embedding_dim": 4,
        "hidden_dim": hidden_dim,
        "lr": learning_rate,
        "batch_size": 4,
        "max_steps": max_steps,
        "log_every": 1,
        "eval_every": 1,
        "device": "cpu",
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _events(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


def test_distinct_runs_resume_and_eval_observability_contract(tmp_path: Path) -> None:
    data_dir, tokenizer_path = _make_corpus_and_tokenizer(tmp_path)
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    run_a = tmp_path / "runs" / "experiment-a"
    run_b = tmp_path / "runs" / "experiment-b"
    _write_config(
        config_a,
        tokenizer_path,
        hidden_dim=7,
        learning_rate=0.02,
        max_steps=2,
    )
    _write_config(
        config_b,
        tokenizer_path,
        hidden_dim=9,
        learning_rate=0.01,
        max_steps=1,
    )

    train(config_a, data_dir, run_a)
    train(config_b, data_dir, run_b)

    snapshot_a = json.loads((run_a / "config.json").read_text(encoding="utf-8"))
    snapshot_b = json.loads((run_b / "config.json").read_text(encoding="utf-8"))
    assert snapshot_a["run_name"] == "experiment-a"
    assert snapshot_b["run_name"] == "experiment-b"
    assert snapshot_a["hidden_dim"] == 7
    assert snapshot_b["hidden_dim"] == 9
    for snapshot in (snapshot_a, snapshot_b):
        assert snapshot["seed"] == 1337
        assert snapshot["tokenizer_type"] == "word"
        assert snapshot["activation"] == "tanh"
        assert snapshot["dropout"] == 0.0
        assert snapshot["optimizer"] == "adam"
        assert snapshot["clip_grad_norm"] is None
        assert snapshot["num_workers"] == 0

    event_files_a = set((run_a / "tensorboard").glob("events.out.tfevents.*"))
    event_files_b = set((run_b / "tensorboard").glob("events.out.tfevents.*"))
    assert event_files_a
    assert event_files_b
    assert event_files_a.isdisjoint(event_files_b)
    for run_dir in (run_a, run_b):
        events = _events(run_dir / "tensorboard")
        assert REQUIRED_SCALAR_TAGS.issubset(events.Tags()["scalars"])
        assert "run/name/text_summary" in events.Tags()["tensors"]
        assert "run/config/text_summary" in events.Tags()["tensors"]

    # max_steps is a permitted runtime-only change. Resume in the same run
    # directory and verify that newly emitted points continue global step 2.
    _write_config(
        config_a,
        tokenizer_path,
        hidden_dim=7,
        learning_rate=0.02,
        max_steps=3,
    )
    train(config_a, data_dir, run_a, resume_from=run_a / "last.pt")
    assert load_checkpoint(run_a / "last.pt")["step"] == 3
    resumed_events = _events(run_a / "tensorboard")
    train_loss_steps = [
        event.step for event in resumed_events.Scalars("train/loss")
    ]
    assert train_loss_steps == [1, 2, 3]

    metrics_path = tmp_path / "results" / "metrics.json"
    metrics = evaluate_checkpoint(run_a / "best.pt", data_dir, metrics_path)
    assert set(metrics) == METRICS_KEYS
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    for split in ("train", "val", "test"):
        assert metrics[f"{split}_loss"] >= 0.0
        assert 0.0 <= metrics[f"{split}_accuracy"] <= 1.0
        assert metrics[f"{split}_ppl"] >= 1.0
