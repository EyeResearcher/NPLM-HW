"""PyTorch datasets and data loaders for preprocessed NPLM corpora.

Each nonblank JSONL row is treated as an independent document. Documents are
encoded with ``<bos>`` and ``<eos>`` and expanded into fixed-width contexts
without allowing a context to cross a document boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .word_tokenizer import WordTokenizer


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_tokenizer(tokenizer: WordTokenizer) -> tuple[int, int]:
    """Return ``(pad_id, vocab_size)`` after validating public attributes."""
    try:
        vocab_size = len(tokenizer.id_to_token)
        pad_id = tokenizer.pad_id
    except (AttributeError, TypeError) as exc:
        raise ValueError("tokenizer must expose id_to_token and a valid pad_id") from exc

    if (
        vocab_size <= 0
        or isinstance(pad_id, bool)
        or not isinstance(pad_id, int)
        or not 0 <= pad_id < vocab_size
    ):
        raise ValueError("tokenizer must have a valid pad_id within its vocabulary")
    return pad_id, vocab_size


def _jsonl_shards(jsonl_dir: Path) -> list[Path]:
    if not jsonl_dir.is_dir():
        raise FileNotFoundError(
            f"JSONL split directory does not exist or is not a directory: {jsonl_dir}"
        )

    shards = sorted(path for path in jsonl_dir.rglob("*.jsonl") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"No JSONL shards found under: {jsonl_dir}")
    return shards


class NPLMDataset(Dataset[tuple[Tensor, Tensor]]):
    """An eagerly materialized fixed-context next-token dataset."""

    def __init__(
        self,
        jsonl_dir: str | Path,
        tokenizer: WordTokenizer,
        context_size: int,
    ) -> None:
        self.context_size = _positive_integer(context_size, "context_size")
        self.tokenizer = tokenizer
        self.pad_id, vocab_size = _validate_tokenizer(tokenizer)

        self._examples: list[tuple[tuple[int, ...], int]] = []
        document_count = 0

        for shard_path in _jsonl_shards(Path(jsonl_dir)):
            with shard_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    document_count += 1
                    text = self._parse_text(line, shard_path, line_number)
                    token_ids = tokenizer.encode_text(text, with_bos_eos=True)
                    self._validate_ids(token_ids, vocab_size, shard_path, line_number)

                    document_examples = self._make_examples(token_ids)
                    if not document_examples:
                        raise ValueError(
                            "Document produced no prediction targets at "
                            f"{shard_path}:{line_number}; the tokenizer must add "
                            "boundary tokens"
                        )
                    self._examples.extend(document_examples)

        if document_count == 0:
            raise ValueError(f"JSONL split is empty: {jsonl_dir}")
        if not self._examples:
            raise ValueError(f"Documents produced no prediction targets: {jsonl_dir}")

    @staticmethod
    def _parse_text(line: str, shard_path: Path, line_number: int) -> str:
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {shard_path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(
                f"Expected an object with a string 'text' field at "
                f"{shard_path}:{line_number}"
            )
        return row["text"]

    @staticmethod
    def _validate_ids(
        token_ids: list[int],
        vocab_size: int,
        shard_path: Path,
        line_number: int,
    ) -> None:
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
            for token_id in token_ids
        ):
            raise ValueError(
                "Tokenizer emitted an ID outside its vocabulary at "
                f"{shard_path}:{line_number}"
            )

    def _make_examples(
        self, token_ids: list[int]
    ) -> list[tuple[tuple[int, ...], int]]:
        examples: list[tuple[tuple[int, ...], int]] = []
        for target_index in range(1, len(token_ids)):
            context = token_ids[max(0, target_index - self.context_size) : target_index]
            padded_context = [self.pad_id] * (self.context_size - len(context)) + context
            examples.append((tuple(padded_context), token_ids[target_index]))
        return examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        context, target = self._examples[index]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def create_dataloader(
    jsonl_dir: str | Path,
    tokenizer: WordTokenizer,
    context_size: int,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create one loader using PyTorch's default collation."""
    _positive_integer(batch_size, "batch_size")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")

    dataset = NPLMDataset(jsonl_dir, tokenizer, context_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_dataloaders(
    data_dir: str | Path,
    tokenizer: WordTokenizer,
    context_size: int,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Return loaders for the train, validation, and test split directories."""
    root = Path(data_dir)
    return {
        split: create_dataloader(
            root / split,
            tokenizer,
            context_size,
            batch_size,
            shuffle=shuffle_train if split == "train" else False,
            num_workers=num_workers,
        )
        for split in ("train", "val", "test")
    }
