from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from nplm.eval import evaluate_checkpoint, parse_args
from nplm.model import NPLM
from nplm.word_tokenizer import TokenizerConfig, WordTokenizer


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}) + "\n")


def _make_uniform_checkpoint(tmp_path: Path) -> tuple[Path, Path, int]:
    data_dir = tmp_path / "data"
    for split in ("train", "val", "test"):
        _write_jsonl(data_dir / split / "shard_00000.jsonl", ["alpha beta"])

    tokenizer = WordTokenizer.build_from_corpus(
        str(data_dir / "train"),
        TokenizerConfig(min_freq=1),
        progress=False,
    )
    tokenizer_path = tmp_path / "vocab.json"
    tokenizer.save(str(tokenizer_path))

    vocab_size = len(tokenizer.id_to_token)
    model_config = {
        "vocab_size": vocab_size,
        "context_size": 2,
        "embedding_dim": 3,
        "hidden_dim": 4,
        "activation": "tanh",
        "dropout": 0.0,
    }
    model = NPLM(**model_config, padding_idx=tokenizer.pad_id)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    training_config = {
        "seed": 1337,
        "tokenizer_type": "word",
        "tokenizer_path": str(tokenizer_path),
        "context_size": 2,
        "embedding_dim": 3,
        "hidden_dim": 4,
        "activation": "tanh",
        "dropout": 0.0,
        "optimizer": "adam",
        "lr": 0.001,
        "batch_size": 2,
        "max_steps": 1,
        "log_every": 1,
        "eval_every": 1,
        "clip_grad_norm": None,
        "device": "cpu",
        "num_workers": 0,
        "run_name": "test-run",
    }
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "step": 1,
        "best_val_loss": math.log(vocab_size),
        "model_config": model_config,
        "training_config": training_config,
        "tokenizer_type": "word",
        "tokenizer_path": str(tokenizer_path),
        "train_time_sec": 1.25,
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, data_dir, vocab_size


def test_evaluate_uniform_model_writes_exact_metrics_without_mutation(
    tmp_path: Path,
) -> None:
    checkpoint_path, data_dir, vocab_size = _make_uniform_checkpoint(tmp_path)
    checkpoint_before = checkpoint_path.read_bytes()
    out_path = tmp_path / "nested" / "metrics.json"

    metrics = evaluate_checkpoint(checkpoint_path, data_dir, out_path)

    assert set(metrics) == {
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
    assert metrics["train_loss"] == pytest.approx(math.log(vocab_size))
    assert metrics["val_loss"] == pytest.approx(math.log(vocab_size))
    assert metrics["test_loss"] == pytest.approx(math.log(vocab_size))
    assert metrics["train_accuracy"] == pytest.approx(0.0)
    assert metrics["val_accuracy"] == pytest.approx(0.0)
    assert metrics["test_accuracy"] == pytest.approx(0.0)
    assert metrics["train_ppl"] == pytest.approx(vocab_size)
    assert metrics["val_ppl"] == pytest.approx(vocab_size)
    assert metrics["test_ppl"] == pytest.approx(vocab_size)
    assert metrics["tokenizer"] == "word"
    assert metrics["vocab_size_or_merges"] == vocab_size
    assert metrics["context_size"] == 2
    assert metrics["train_time_sec"] == 1.25
    assert json.loads(out_path.read_text(encoding="utf-8")) == metrics
    assert checkpoint_path.read_bytes() == checkpoint_before


def test_evaluate_requires_every_split(tmp_path: Path) -> None:
    checkpoint_path, data_dir, _vocab_size = _make_uniform_checkpoint(tmp_path)
    for path in (data_dir / "test").iterdir():
        path.unlink()
    (data_dir / "test").rmdir()

    with pytest.raises(FileNotFoundError, match="test"):
        evaluate_checkpoint(checkpoint_path, data_dir, tmp_path / "metrics.json")


def test_evaluate_uses_token_weighting_for_uneven_batches(tmp_path: Path) -> None:
    checkpoint_path, data_dir, vocab_size = _make_uniform_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bias = torch.arange(vocab_size, dtype=torch.float32)
    checkpoint["model_state_dict"]["output.bias"] = bias
    torch.save(checkpoint, checkpoint_path)

    metrics = evaluate_checkpoint(
        checkpoint_path, data_dir, tmp_path / "metrics.json"
    )

    # The three targets are alpha, beta, and <eos>. With batch_size=2 the
    # final batch is smaller, so averaging batch means would give a different
    # result from this token-weighted value.
    tokenizer = WordTokenizer.load(checkpoint["tokenizer_path"])
    target_ids = [
        tokenizer.token_to_id["alpha"],
        tokenizer.token_to_id["beta"],
        tokenizer.eos_id,
    ]
    expected_mean_nll = torch.logsumexp(bias, dim=0).item() - sum(target_ids) / 3
    expected_ppl = math.exp(expected_mean_nll)
    expected_accuracy = 1 / 3
    assert metrics["train_loss"] == pytest.approx(expected_mean_nll)
    assert metrics["val_loss"] == pytest.approx(expected_mean_nll)
    assert metrics["test_loss"] == pytest.approx(expected_mean_nll)
    assert metrics["train_accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["val_accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["test_accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["train_ppl"] == pytest.approx(expected_ppl)
    assert metrics["val_ppl"] == pytest.approx(expected_ppl)
    assert metrics["test_ppl"] == pytest.approx(expected_ppl)


def test_parse_args() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "run/best.pt",
            "--data_dir",
            "data/corpus",
            "--out_json",
            "results/metrics.json",
        ]
    )

    assert args.checkpoint == "run/best.pt"
    assert args.data_dir == "data/corpus"
    assert args.out_json == "results/metrics.json"
