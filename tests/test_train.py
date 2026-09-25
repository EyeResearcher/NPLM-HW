import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nplm.train import parse_args, train
from nplm.utils import load_checkpoint
from nplm.word_tokenizer import TokenizerConfig, WordTokenizer


class _RecordingWriter:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), step))

    def add_text(self, tag: str, value: str, step: int) -> None:
        self.text.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}) + "\n")


def _fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    for split in ("train", "val", "test"):
        _write_jsonl(
            data_dir / split / "shard_00000.jsonl",
            ["alpha beta alpha", "beta alpha beta"],
        )

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
    max_steps: int,
    eval_every: int,
    log_every: int = 1,
    run_name: str | None = None,
) -> None:
    payload = {
        "seed": 7,
        "tokenizer_type": "word",
        "tokenizer_path": str(tokenizer_path),
        "context_size": 2,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "activation": "tanh",
        "dropout": 0.0,
        "optimizer": "adam",
        "lr": 0.03,
        "batch_size": 4,
        "max_steps": max_steps,
        "log_every": log_every,
        "eval_every": eval_every,
        "clip_grad_norm": 1.0,
        "device": "cpu",
        "num_workers": 0,
        "run_name": run_name,
    }
    # JSON is valid YAML and avoids platform-specific path escaping in fixtures.
    path.write_text(json.dumps(payload), encoding="utf-8")


class TrainTests(unittest.TestCase):
    def test_synthetic_training_writes_checkpoints_and_resume_uses_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, tokenizer_path = _fixture(root)
            config_path = root / "config.yaml"
            save_dir = root / "run"
            _write_config(
                config_path, tokenizer_path, max_steps=3, eval_every=10
            )

            writers: list[_RecordingWriter] = []

            def make_writer(log_dir: str | Path) -> _RecordingWriter:
                writer = _RecordingWriter(Path(log_dir))
                writers.append(writer)
                return writer

            with patch("nplm.train.create_tensorboard_writer", side_effect=make_writer):
                best_path = train(config_path, data_dir, save_dir)

            self.assertEqual(best_path, save_dir / "best.pt")
            self.assertTrue(best_path.is_file())
            self.assertTrue((save_dir / "last.pt").is_file())
            first_last = load_checkpoint(save_dir / "last.pt")
            self.assertEqual(first_last["step"], 3)
            self.assertEqual(first_last["model_config"]["vocab_size"], 6)
            self.assertEqual(first_last["model_config"]["context_size"], 2)
            normalized = json.loads(
                (save_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(normalized["run_name"], "run")
            self.assertEqual(writers[0].log_dir, save_dir / "tensorboard")
            self.assertTrue(writers[0].closed)
            self.assertGreaterEqual(writers[0].flush_count, 2)
            tags = {tag for tag, _, _ in writers[0].scalars}
            self.assertTrue(
                {
                    "train/loss",
                    "train/accuracy",
                    "train/learning_rate",
                    "validation/loss",
                    "validation/perplexity",
                    "validation/accuracy",
                    "system/elapsed_seconds",
                }.issubset(tags)
            )
            self.assertEqual({tag for tag, _, _ in writers[0].text}, {"run/name", "run/config"})

            _write_config(
                config_path, tokenizer_path, max_steps=5, eval_every=10
            )
            with patch("nplm.train.create_tensorboard_writer", side_effect=make_writer):
                train(
                    config_path,
                    data_dir,
                    save_dir,
                    resume_from=save_dir / "last.pt",
                )

            resumed_last = load_checkpoint(save_dir / "last.pt")
            self.assertEqual(resumed_last["step"], 5)
            self.assertGreaterEqual(
                resumed_last["train_time_sec"], first_last["train_time_sec"]
            )
            resumed_train_steps = {
                step
                for tag, _, step in writers[1].scalars
                if tag == "train/loss"
            }
            self.assertEqual(resumed_train_steps, {4, 5})
            self.assertTrue(writers[1].closed)

    def test_best_checkpoint_uses_weighted_validation_mean_and_strict_improvement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, tokenizer_path = _fixture(root)
            config_path = root / "config.yaml"
            save_dir = root / "run"
            _write_config(config_path, tokenizer_path, max_steps=2, eval_every=1)

            # Means are 2.0 then 3.0; the second checkpoint must not replace best.
            validation_results = [
                {"loss": 2.0, "perplexity": 7.0, "accuracy": 0.25, "target_count": 2},
                {"loss": 3.0, "perplexity": 20.0, "accuracy": 0.5, "target_count": 1},
            ]
            writer = _RecordingWriter(save_dir / "tensorboard")
            with (
                patch("nplm.train.evaluate_metrics", side_effect=validation_results),
                patch("nplm.train.create_tensorboard_writer", return_value=writer),
            ):
                train(config_path, data_dir, save_dir)

            best = load_checkpoint(save_dir / "best.pt")
            last = load_checkpoint(save_dir / "last.pt")
            self.assertEqual(best["step"], 1)
            self.assertEqual(best["best_val_loss"], 2.0)
            self.assertEqual(last["step"], 2)
            self.assertEqual(last["best_val_loss"], 2.0)
            validation_steps = [
                step for tag, _, step in writer.scalars if tag == "validation/loss"
            ]
            self.assertEqual(validation_steps, [1, 2])

    def test_parse_args_exposes_contract_flags(self) -> None:
        args = parse_args(
            [
                "--config",
                "config.yaml",
                "--data_dir",
                "data",
                "--save_dir",
                "run",
                "--resume_from",
                "old.pt",
            ]
        )
        self.assertEqual(args.config, "config.yaml")
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.save_dir, "run")
        self.assertEqual(args.resume_from, "old.pt")

    def test_readable_vocab_size_must_match_tokenizer_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, tokenizer_path = _fixture(root)
            config_path = root / "config.yaml"
            _write_config(config_path, tokenizer_path, max_steps=1, eval_every=1)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["vocab_size"] = 999
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vocab_size"):
                train(config_path, data_dir, root / "run")

    def test_separate_runs_resolve_distinct_names_and_writer_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, tokenizer_path = _fixture(root)
            config_path = root / "config.yaml"
            _write_config(config_path, tokenizer_path, max_steps=1, eval_every=1)
            train(config_path, data_dir, root / "run-a")
            train(config_path, data_dir, root / "run-b")

            names = [
                json.loads((root / name / "config.json").read_text())["run_name"]
                for name in ("run-a", "run-b")
            ]
            self.assertEqual(names, ["run-a", "run-b"])
            required_tags = {
                "train/loss",
                "train/accuracy",
                "train/learning_rate",
                "validation/loss",
                "validation/perplexity",
                "validation/accuracy",
                "system/elapsed_seconds",
            }
            event_files: list[Path] = []
            for name in ("run-a", "run-b"):
                log_dir = root / name / "tensorboard"
                event_files.extend(log_dir.glob("events.out.tfevents.*"))
                accumulator = EventAccumulator(str(log_dir))
                accumulator.Reload()
                self.assertTrue(required_tags.issubset(accumulator.Tags()["scalars"]))
                self.assertIn("run/name/text_summary", accumulator.Tags()["tensors"])
                self.assertIn("run/config/text_summary", accumulator.Tags()["tensors"])
            self.assertEqual(len(event_files), 2)

    def test_writer_closes_when_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, tokenizer_path = _fixture(root)
            config_path = root / "config.yaml"
            _write_config(config_path, tokenizer_path, max_steps=1, eval_every=1)
            writer = _RecordingWriter(root / "run" / "tensorboard")

            with (
                patch("nplm.train.create_tensorboard_writer", return_value=writer),
                patch(
                    "nplm.train.evaluate_metrics",
                    side_effect=RuntimeError("validation failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "validation failed"),
            ):
                train(config_path, data_dir, root / "run")

            self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
