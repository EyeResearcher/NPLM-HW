# Contract: `word_tokenizer.py` and `build_word_vocab.py`

## Scope

Preserve and verify the supplied word tokenizer and its vocabulary-building CLI.
Do not create `build_word_tokenizer.py`; `build_word_vocab.py` is the canonical
starter filename and already matches the assignment example command.

## Public API

The frozen API and artifact are defined in the tokenizer section of
[README.md](README.md). The following module-level helpers also remain usable:

```python
def basic_tokenize(
    text: str,
    *,
    lowercase: bool = False,
    tokenizer: str = "simple",
    strip_punct: bool = False,
) -> list[str]:
    """Tokenize deterministically using simple regex or whitespace mode."""

def iter_text_from_jsonl_dir(
    jsonl_dir: str,
    text_field: str = "text",
) -> Iterator[str]:
    """Yield valid text values from .jsonl and .jsonl.gz files recursively."""
```

`build_word_vocab.parse_args()` and `build_word_vocab.main()` remain the CLI
entry points. Required command:

```text
python -m nplm.build_word_vocab \
  --jsonl_dir data/wikitext2_jsonl/train \
  --output_path artifacts/wikitext2_word_vocab.json \
  --min_freq 2 --max_vocab 20000
```

## Required invariants

- Vocabulary construction uses only the training directory supplied by the
  caller; it must not inspect sibling validation or test splits.
- Token ordering is deterministic: decreasing frequency, then lexical order.
- Mandatory special tokens are unique and precede ordinary tokens.
- Unknown tokens always encode to a valid integer `unk_id`.
- Saving then loading preserves token IDs, config, and frequencies.
- Malformed JSONL rows may be skipped as the starter does, but a missing corpus
  directory or a corpus with no valid text must raise a helpful error rather
  than create a specials-only artifact accidentally.

## Dependencies and consumers

`build_word_vocab.py` depends only on `word_tokenizer.py`. `data.py`, `train.py`,
and `eval.py` consume the public API; they must not use private file helpers.

## Verification

- Round-trip artifact test.
- Frequency cutoff and `max_vocab` exclude/retain the expected tokens.
- OOV encoding and unknown-rate calculation are correct.
- Default special IDs and deterministic tie-breaking are stable.
