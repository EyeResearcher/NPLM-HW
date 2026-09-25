# Contract: `data.py`

## Scope

Read JSONL text, encode documents with the provided `WordTokenizer`, create
fixed-context examples, and construct PyTorch data loaders. Do not split raw
corpora and do not move batches to a device.

## Public API

```python
class NPLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_dir: str | Path,
        tokenizer: WordTokenizer,
        context_size: int,
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...


def create_dataloader(
    jsonl_dir: str | Path,
    tokenizer: WordTokenizer,
    context_size: int,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create one loader using default collation."""


def create_dataloaders(
    data_dir: str | Path,
    tokenizer: WordTokenizer,
    context_size: int,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Return loaders for train, val, and test split directories."""
```

Items and batches follow the window/batch contract in [README.md](README.md).
All emitted IDs must be in `[0, len(tokenizer.id_to_token))`.

Construction may eagerly materialize `(context, target)` ID pairs; the selected
corpus and educational scope favor clarity over a complex lazy index. It must
iterate shard paths in lexical order so non-shuffled evaluation is stable.

Raise `ValueError` for nonpositive context or batch size, a tokenizer without a
valid `pad_id`, an empty split, or documents that produce no targets. Raise
`FileNotFoundError` for missing split directories/shards.

## Dependencies and consumers

Imports: `word_tokenizer.py`, PyTorch, and standard-library JSON/path utilities.
Consumers: `train.py` and `eval.py`.

## Verification

- A two-document fixture proves contexts never cross the boundary.
- First-token context is left padded and includes `<bos>` nearest the target.
- `<eos>` is emitted as the last target for each document.
- Item and batch dtypes/shapes exactly match the shared contract.
