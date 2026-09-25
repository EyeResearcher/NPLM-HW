# Contract: `preprocess.py`

## Scope

Load the raw Hugging Face dataset, convert rows to independent text documents,
make deterministic splits, and write JSONL shards. Do not tokenize text.

## Public API

```python
def read_dataset(dataset_path: str | Path) -> DatasetDict:
    """Load a DatasetDict previously written by download.py."""

def generate_documents(
    dataset: DatasetDict | Dataset,
    *,
    lowercase: bool = False,
) -> list[str]:
    """Return trimmed, nonempty text rows in stable source order."""

def split_dataset(
    documents: Sequence[str],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 1337,
) -> dict[str, list[str]]:
    """Deterministically return train/val/test document lists."""

def write_split_as_shards(
    split: Sequence[str],
    output_dir: str | Path,
    *,
    shard_size: int = 1000,
) -> list[Path]:
    """Write the JSONL corpus contract and return shard paths in order."""

def _resolve_input_output_dirs(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve CLI paths without creating or mutating them."""

def main() -> None:
    """Run load -> documents -> split -> shards."""
```

`generate_documents` accepts a single `Dataset` or all splits in a
`DatasetDict`. Missing/non-string `text` fields raise `ValueError`. Blank rows
are dropped. It must not concatenate neighboring rows.

`split_dataset` requires positive validation and test proportions whose sum is
less than one, and enough documents to make all three splits nonempty.

`write_split_as_shards` requires `shard_size > 0`, creates `output_dir`, writes
UTF-8 `shard_00000.jsonl` files, and returns their paths. Invalid sizes raise
`ValueError`; serialization and filesystem errors propagate.

CLI options:

```text
--input_dir PATH       required
--output_dir PATH      optional; defaults to data/<corpus>_jsonl beside raw data
--test_size FLOAT      default 0.2
--val_size FLOAT       default 0.1
--seed INT             default 1337
--shard_size INT       default 1000
--lowercase            flag
```

The starter positional path arguments should be replaced by these named options
to match `ASSIGNMENT.md`. `main()` must pass `shard_size`; the starter currently
omits it.

## Dependencies and outputs

External dependencies: `datasets`, `argparse`, `json`, `pathlib`, `random`.
No Torch dependency is needed and the unused Torch dataset import should be
removed.

Output: the JSONL corpus contract in [README.md](README.md).

## Verification

- Same seed yields identical membership and shard contents.
- Different documents never merge.
- Split sizes cover every input document exactly once.
- Shards never exceed `shard_size`, including the final shard.
