# Contract: `download.py`

## Scope

Implement the corpus download CLI and persist the raw corpus in the representation
defined in [README.md](README.md). This agent does not preprocess, tokenize, or
write JSONL.

The existing decorator and URL helper are incomplete scaffolding, not frozen
interfaces. They may be removed. Preserve `parse_args()` and `main()` as CLI
entry points, updating their options to match the public contract.

## Public API

```python
def download_dataset(dataset: str, out_dir: str | Path) -> Path:
    """Download a supported corpus, save a DatasetDict to out_dir, and return it."""

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse --dataset and --out_dir."""

def main() -> None:
    """CLI entry point."""
```

`download_dataset` requirements:

- Support the alias `wikitext2` using the WikiText-2 raw Hugging Face dataset.
- Normalize the result to a `datasets.DatasetDict` whose splits each expose a
  string `text` column.
- Create parent directories and call `save_to_disk(out_dir)`.
- Return `Path(out_dir)` after a successful save.
- Raise `ValueError` for an unsupported alias and propagate download/filesystem
  failures with context; do not silently return `None`.

CLI:

```text
python -m nplm.download --dataset wikitext2 --out_dir data/raw/wikitext2
```

For compatibility, accepting the starter spelling `--output` as a hidden alias
for `--out_dir` is allowed, but docs and tests use `--out_dir`.

## Dependencies and outputs

External dependencies: `datasets`, `pathlib`, `argparse`.

Output: the raw downloaded dataset contract only. No NPLM imports are allowed.

## Verification

- An unsupported alias raises `ValueError`.
- A mocked WikiText-2 download is saved with all splits and a `text` column.
- CLI arguments reach `download_dataset` unchanged.
