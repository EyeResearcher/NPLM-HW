import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from nplm.preprocess import (
    _resolve_input_output_dirs,
    generate_documents,
    main,
    read_dataset,
    split_dataset,
    write_split_as_shards,
)


class PreprocessTests(unittest.TestCase):
    def test_read_and_generate_documents_in_stable_order(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"text": ["  First row  ", "   ", "THIRD"]}
                ),
                "test": Dataset.from_dict({"text": [" Last row\n"]}),
            }
        )
        self.assertEqual(
            generate_documents(dataset, lowercase=True),
            ["first row", "third", "last row"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset.save_to_disk(tmp_dir)
            loaded = read_dataset(tmp_dir)
            self.assertIsInstance(loaded, DatasetDict)
            self.assertEqual(generate_documents(loaded), ["First row", "THIRD", "Last row"])

    def test_generate_documents_rejects_bad_text_columns(self):
        with self.assertRaisesRegex(ValueError, "'text' column"):
            generate_documents(Dataset.from_dict({"body": ["hello"]}))
        with self.assertRaisesRegex(ValueError, "must be a string"):
            generate_documents(Dataset.from_dict({"text": [None]}))

    def test_split_is_local_deterministic_and_exhaustive(self):
        documents = [f"document-{index}" for index in range(20)]
        random.seed(8675309)
        state_before = random.getstate()

        first = split_dataset(documents, test_size=0.2, val_size=0.1, seed=7)
        second = split_dataset(documents, test_size=0.2, val_size=0.1, seed=7)
        different = split_dataset(documents, test_size=0.2, val_size=0.1, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(set(first), {"train", "val", "test"})
        self.assertTrue(all(first[name] for name in first))
        self.assertEqual(
            Counter(first["train"] + first["val"] + first["test"]),
            Counter(documents),
        )

    def test_split_validates_proportions_and_document_count(self):
        for test_size, val_size in [(0, 0.1), (1, 0.1), (0.2, 0), (0.6, 0.4)]:
            with self.subTest(test_size=test_size, val_size=val_size):
                with self.assertRaises(ValueError):
                    split_dataset(
                        ["one", "two", "three"],
                        test_size=test_size,
                        val_size=val_size,
                    )
        with self.assertRaisesRegex(ValueError, "At least three"):
            split_dataset(["one", "two"])

    def test_write_shards_respects_size_schema_and_utf8(self):
        documents = ["one", "café", "three", "four", "five"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "train"
            output_dir.mkdir(parents=True)
            stale_path = output_dir / "shard_99999.jsonl"
            stale_path.write_text("stale", encoding="utf-8")

            paths = write_split_as_shards(documents, output_dir, shard_size=2)

            self.assertEqual(
                [path.name for path in paths],
                ["shard_00000.jsonl", "shard_00001.jsonl", "shard_00002.jsonl"],
            )
            self.assertFalse(stale_path.exists())
            rows = []
            for path in paths:
                shard_rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertLessEqual(len(shard_rows), 2)
                self.assertTrue(all(set(row) == {"text"} for row in shard_rows))
                rows.extend(shard_rows)
            self.assertEqual(rows, [{"text": text} for text in documents])

            with self.assertRaises(ValueError):
                write_split_as_shards(documents, output_dir, shard_size=0)

    def test_default_output_directory(self):
        input_path, output_path = _resolve_input_output_dirs(
            Path("data") / "raw" / "wikitext2"
        )
        self.assertEqual(input_path, Path("data/raw/wikitext2"))
        self.assertEqual(output_path, Path("data/wikitext2_jsonl"))

    def test_main_runs_pipeline_with_named_cli_options(self):
        dataset = DatasetDict(
            {"source": Dataset.from_dict({"text": [f"DOC {i}" for i in range(10)]})}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "raw" / "fixture"
            output_dir = root / "processed"
            dataset.save_to_disk(input_dir)
            argv = [
                "nplm.preprocess",
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--test_size",
                "0.2",
                "--val_size",
                "0.1",
                "--seed",
                "42",
                "--shard_size",
                "2",
                "--lowercase",
            ]

            with patch.object(sys, "argv", argv):
                main()

            all_rows = []
            for split_name in ("train", "val", "test"):
                shard_paths = sorted((output_dir / split_name).glob("shard_*.jsonl"))
                self.assertTrue(shard_paths)
                for shard_path in shard_paths:
                    rows = [
                        json.loads(line)
                        for line in shard_path.read_text(encoding="utf-8").splitlines()
                    ]
                    self.assertLessEqual(len(rows), 2)
                    all_rows.extend(rows)
            self.assertEqual(
                Counter(row["text"] for row in all_rows),
                Counter(f"doc {i}" for i in range(10)),
            )


if __name__ == "__main__":
    unittest.main()
