"""Convert a downloaded Hugging Face corpus into JSONL shards."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess a dataset for NPLM")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="DatasetDict directory produced by nplm.download",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output corpus directory (default: data/<corpus>_jsonl)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of documents assigned to the test split",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of documents assigned to the validation split",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Split random seed")
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Maximum documents per JSONL shard",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase documents during preprocessing",
    )
    return parser.parse_args()


def read_dataset(dataset_path: str | Path) -> DatasetDict:
    """Load a ``DatasetDict`` previously written by :mod:`nplm.download`."""
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict at {dataset_path!s}, got "
            f"{type(dataset).__name__}"
        )
    return dataset


def generate_documents(
    dataset: DatasetDict | HFDataset,
    *,
    lowercase: bool = False,
) -> list[str]:
    """Return trimmed, nonempty text rows in stable source order.

    Rows are kept as independent documents. For a ``DatasetDict``, splits are
    visited in their stored insertion order and rows retain their source order.
    """
    if isinstance(dataset, DatasetDict):
        source_datasets = list(dataset.values())
    elif isinstance(dataset, HFDataset):
        source_datasets = [dataset]
    else:
        raise TypeError(
            "dataset must be a datasets.Dataset or datasets.DatasetDict, got "
            f"{type(dataset).__name__}"
        )

    documents: list[str] = []
    for source in source_datasets:
        if "text" not in source.column_names:
            raise ValueError("Every source split must contain a 'text' column")
        for row_index, text in enumerate(source["text"]):
            if not isinstance(text, str):
                raise ValueError(
                    "Every 'text' value must be a string; "
                    f"row {row_index} contains {type(text).__name__}"
                )
            document = text.strip()
            if not document:
                continue
            documents.append(document.lower() if lowercase else document)
    return documents


def split_dataset(
    documents: Sequence[str],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 1337,
) -> dict[str, list[str]]:
    """Deterministically split documents into nonempty train/val/test lists."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1 (exclusive)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    shuffled = list(documents)
    if len(shuffled) < 3:
        raise ValueError("At least three documents are required for nonempty splits")

    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_size))
    val_count = max(1, int(len(shuffled) * val_size))
    train_count = len(shuffled) - test_count - val_count
    if train_count < 1:
        raise ValueError(
            "The requested proportions leave no documents for the training split"
        )

    test = shuffled[:test_count]
    val = shuffled[test_count : test_count + val_count]
    train = shuffled[test_count + val_count :]
    return {"train": train, "val": val, "test": test}


def write_split_as_shards(
    split: Sequence[str],
    output_dir: str | Path,
    *,
    shard_size: int = 1000,
) -> list[Path]:
    """Write one corpus split as UTF-8 JSONL shards and return their paths."""
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than zero")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prevent shards from a longer previous run from remaining in the corpus.
    for old_shard in output_path.glob("shard_*.jsonl"):
        old_shard.unlink()

    shard_paths: list[Path] = []
    for start in range(0, len(split), shard_size):
        shard_path = output_path / f"shard_{len(shard_paths):05d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as shard_file:
            for document in split[start : start + shard_size]:
                shard_file.write(
                    json.dumps({"text": document}, ensure_ascii=False) + "\n"
                )
        shard_paths.append(shard_path)
    return shard_paths


def _resolve_input_output_dirs(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve CLI input/output paths without creating or modifying them."""
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path.parent.parent / f"{input_path.name}_jsonl"
    else:
        output_path = Path(output_dir)
    return input_path, output_path


def main() -> None:
    """Run the complete raw-dataset to JSONL-shards preprocessing pipeline."""
    args = _parse_args()
    input_dir, output_dir = _resolve_input_output_dirs(
        args.input_dir, args.output_dir
    )

    dataset = read_dataset(input_dir)
    documents = generate_documents(dataset, lowercase=args.lowercase)
    splits = split_dataset(
        documents,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    for split_name, split_data in splits.items():
        write_split_as_shards(
            split_data,
            output_dir / split_name,
            shard_size=args.shard_size,
        )


if __name__ == "__main__":
    main()
