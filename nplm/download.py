"""Download raw corpora for the NPLM pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import datasets


_DATASETS = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
}


def _as_dataset_dict(downloaded: object) -> datasets.DatasetDict:
    """Validate and normalize a downloaded Hugging Face dataset."""
    if isinstance(downloaded, datasets.DatasetDict):
        dataset_dict = downloaded
    else:
        try:
            dataset_dict = datasets.DatasetDict(dict(downloaded))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The downloaded corpus was not a mapping of split names to datasets"
            ) from exc

    if not dataset_dict:
        raise ValueError("The downloaded corpus contains no splits")

    normalized: dict[str, datasets.Dataset] = {}
    for split_name, split in dataset_dict.items():
        if "text" not in split.column_names:
            raise ValueError(
                f"Downloaded split {split_name!r} does not contain a 'text' column"
            )

        text_feature = split.features.get("text")
        if getattr(text_feature, "dtype", None) != "string":
            try:
                split = split.cast_column("text", datasets.Value("string"))
            except Exception as exc:
                raise TypeError(
                    f"Downloaded split {split_name!r} has a non-string 'text' column"
                ) from exc
        normalized[str(split_name)] = split

    return datasets.DatasetDict(normalized)


def download_dataset(dataset: str, out_dir: str | Path) -> Path:
    """Download a supported corpus, save it to disk, and return its path.

    Args:
        dataset: Public dataset alias. Currently ``"wikitext2"`` is supported.
        out_dir: Directory in which to save the Hugging Face ``DatasetDict``.

    Raises:
        ValueError: If ``dataset`` is unsupported or its schema is invalid.
        RuntimeError: If downloading or saving the dataset fails.
    """
    try:
        dataset_name, dataset_config = _DATASETS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(_DATASETS))
        raise ValueError(
            f"Unsupported dataset alias {dataset!r}; supported aliases: {supported}"
        ) from exc

    try:
        downloaded = datasets.load_dataset(dataset_name, dataset_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset {dataset!r}") from exc

    dataset_dict = _as_dataset_dict(downloaded)
    output_path = Path(out_dir)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save dataset {dataset!r} to {output_path}"
        ) from exc

    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the dataset downloader."""
    parser = argparse.ArgumentParser(description="Download raw NPLM data")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset alias to download (currently: wikitext2)",
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        help="Directory in which to save the raw Hugging Face dataset",
    )
    
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("the following arguments are required: --out_dir")
    return args


def main() -> None:
    """Run the corpus download CLI."""
    args = _parse_args()
    download_dataset(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
