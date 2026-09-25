"""Focused, network-free tests for the raw corpus downloader."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _Feature:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _Dataset:
    def __init__(self, rows: dict[str, list[object]]) -> None:
        self.rows = rows
        self.column_names = list(rows)
        self.features = {
            name: _Feature("string" if all(isinstance(item, str) for item in values) else "int64")
            for name, values in rows.items()
        }

    def cast_column(self, name: str, feature: _Feature) -> "_Dataset":
        self.rows[name] = [str(value) for value in self.rows[name]]
        self.features[name] = feature
        return self


class _DatasetDict(dict[str, _Dataset]):
    saved: "_DatasetDict | None" = None
    saved_path: str | None = None

    def save_to_disk(self, path: str) -> None:
        type(self).saved = self
        type(self).saved_path = path
        Path(path).mkdir(parents=True)


def _load_download_module() -> tuple[types.ModuleType, types.ModuleType]:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _Dataset
    fake_datasets.DatasetDict = _DatasetDict
    fake_datasets.Value = _Feature
    fake_datasets.load_dataset = mock.Mock()

    module_path = Path(__file__).parents[1] / "nplm" / "download.py"
    spec = importlib.util.spec_from_file_location("download_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
        spec.loader.exec_module(module)
    return module, fake_datasets


class DownloadDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.download, self.datasets = _load_download_module()
        _DatasetDict.saved = None
        _DatasetDict.saved_path = None

    def test_unsupported_alias_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dataset alias"):
            self.download.download_dataset("not-a-corpus", "unused")

    def test_wikitext2_is_saved_with_all_text_splits(self) -> None:
        source = _DatasetDict(
            {
                "train": _Dataset({"text": ["train row"]}),
                "validation": _Dataset({"text": ["val row"]}),
                "test": _Dataset({"text": ["test row"]}),
            }
        )
        self.datasets.load_dataset.return_value = source

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "nested" / "wikitext2"
            result = self.download.download_dataset("wikitext2", out_dir)

            self.assertEqual(result, out_dir)
            self.assertEqual(_DatasetDict.saved_path, str(out_dir))
            self.assertTrue(out_dir.is_dir())

        self.datasets.load_dataset.assert_called_once_with(
            "wikitext", "wikitext-2-raw-v1"
        )
        self.assertEqual(set(_DatasetDict.saved or {}), {"train", "validation", "test"})
        for split in (_DatasetDict.saved or {}).values():
            self.assertIn("text", split.column_names)
            self.assertEqual(split.features["text"].dtype, "string")

    def test_main_forwards_cli_arguments(self) -> None:
        argv = [
            "nplm.download",
            "--dataset",
            "wikitext2",
            "--out_dir",
            "raw/output",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch.object(self.download, "download_dataset") as download_dataset,
        ):
            self.download.main()

        download_dataset.assert_called_once_with("wikitext2", "raw/output")


if __name__ == "__main__":
    unittest.main()
