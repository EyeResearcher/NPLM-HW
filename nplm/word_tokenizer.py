"""
word_tokenizer.py

Implements a simple **word-level tokenizer** and vocabulary builder, following
the approach used in Bengio et al. (2003), *A Neural Probabilistic Language Model*.

Students can use this module to:
  - Build a word-level vocabulary from a JSONL corpus (one document per line),
  - Save and load the vocabulary to/from disk,
  - Tokenize raw text into tokens,
  - Encode tokens into integer IDs,
  - Decode IDs back into tokens,
  - Estimate unknown-token (<unk>) rates on a dataset.

Typical usage inside your training code:

    from nplm.word_tokenizer import WordTokenizer, TokenizerConfig

    # Build from corpus (JSONL directory)
    config = TokenizerConfig(min_freq=2, max_vocab=20000, lowercase=True)
    tok = WordTokenizer.build_from_corpus("data/wikitext2_jsonl/train", config)

    # Save vocab
    tok.save("artifacts/wikitext2_word_vocab.json")

    # Load vocab later
    tok2 = WordTokenizer.load("artifacts/wikitext2_word_vocab.json")

    # Tokenize + encode a string
    ids = tok2.encode_text("The quick brown fox jumps over the lazy dog.", with_bos_eos=True)

    # Decode back to tokens
    tokens = tok2.decode_ids(ids)

This tokenizer is deliberately simple and word-based. You may also implement or
use a BPE/subword tokenizer (see assignment spec).
"""

from __future__ import annotations
import json
import os
import re
import gzip
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Dict, Iterator, List, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ---- Special tokens ----
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

DEFAULT_SPECIALS = [PAD, UNK, BOS, EOS]


# ---- Basic tokenization ----
_TOKEN_PATTERNS = {
    "whitespace": None,
    "simple": r"[A-Za-z0-9_'-]+|[^\sA-Za-z0-9_]",
}


def _iter_jsonl_paths(jsonl_dir: str) -> Iterator[str]:
    for root, dirs, files in os.walk(jsonl_dir):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".jsonl") or fname.endswith(".jsonl.gz"):
                yield os.path.join(root, fname)


def _open_textmaybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_text_from_jsonl_dir(jsonl_dir: str, text_field: str = "text") -> Iterator[str]:
    import json as _json
    for path in _iter_jsonl_paths(jsonl_dir):
        with _open_textmaybe_gzip(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                if text_field in obj and isinstance(obj[text_field], str):
                    yield obj[text_field]


def basic_tokenize(
    text: str,
    *,
    lowercase: bool = False,
    tokenizer: str = "simple",
    strip_punct: bool = False,
) -> List[str]:
    if lowercase:
        text = text.lower()
    pat = _TOKEN_PATTERNS.get(tokenizer)
    if pat is None:
        return text.split()
    tokens = re.findall(pat, text)
    if strip_punct:
        tokens = [t for t in tokens if re.match(r"[A-Za-z0-9_'-]+$", t)]
    return tokens


@dataclass
class TokenizerConfig:
    min_freq: int = 2
    max_vocab: Optional[int] = 20000
    lowercase: bool = False
    tokenizer: str = "simple"  # or "whitespace"
    strip_punct: bool = False
    include_bos: bool = True
    include_eos: bool = True
    specials: Optional[List[str]] = None


def _validate_config(config: TokenizerConfig) -> None:
    if config.min_freq < 1:
        raise ValueError("min_freq must be at least 1")
    if config.max_vocab is not None and config.max_vocab < 0:
        raise ValueError("max_vocab must be nonnegative or None")
    if config.tokenizer not in _TOKEN_PATTERNS:
        choices = ", ".join(sorted(_TOKEN_PATTERNS))
        raise ValueError(f"tokenizer must be one of: {choices}")


def _special_tokens(config: TokenizerConfig) -> List[str]:
    """Return unique special tokens with required boundary tokens first."""
    configured = config.specials if config.specials is not None else []
    for token in configured:
        if not isinstance(token, str) or not token:
            raise ValueError("special tokens must be nonempty strings")

    standard = [PAD, UNK]
    if config.include_bos:
        standard.append(BOS)
    if config.include_eos:
        standard.append(EOS)

    result: List[str] = []
    for token in [*standard, *configured]:
        if token in (BOS, EOS):
            if token == BOS and not config.include_bos:
                continue
            if token == EOS and not config.include_eos:
                continue
        if token not in result:
            result.append(token)
    return result


class WordTokenizer:
    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: List[str],
        config: TokenizerConfig,
        freqs: Optional[Dict[str, int]] = None,
    ):
        _validate_config(config)
        if len(token_to_id) != len(id_to_token):
            raise ValueError("token_to_id and id_to_token must have the same size")
        for expected_id, token in enumerate(id_to_token):
            if token_to_id.get(token) != expected_id:
                raise ValueError("token_to_id and id_to_token must be exact inverses")
        if PAD not in token_to_id or UNK not in token_to_id:
            raise ValueError("vocabulary must contain <pad> and <unk>")
        if config.include_bos and BOS not in token_to_id:
            raise ValueError("include_bos=True requires <bos> in the vocabulary")
        if config.include_eos and EOS not in token_to_id:
            raise ValueError("include_eos=True requires <eos> in the vocabulary")

        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.config = config
        self.freqs = freqs or {}

        self.pad_id = self.token_to_id.get(PAD)
        self.unk_id = self.token_to_id.get(UNK)
        self.bos_id = self.token_to_id.get(BOS) if config.include_bos else None
        self.eos_id = self.token_to_id.get(EOS) if config.include_eos else None

    # ---- Factory ----
    @classmethod
    def build_from_corpus(
        cls,
        jsonl_dir: str,
        config: Optional[TokenizerConfig] = None,
        text_field: str = "text",
        progress: bool = True,
    ) -> "WordTokenizer":
        config = config or TokenizerConfig()
        _validate_config(config)
        if not os.path.isdir(jsonl_dir):
            raise FileNotFoundError(
                f"JSONL corpus directory does not exist or is not a directory: {jsonl_dir}"
            )
        specials = _special_tokens(config)

        counter: Counter = Counter()
        document_count = 0
        texts = iter_text_from_jsonl_dir(jsonl_dir, text_field=text_field)
        iterator = tqdm(texts, desc="Scanning text", unit="doc") if progress else texts

        for doc in iterator:
            document_count += 1
            toks = basic_tokenize(
                doc,
                lowercase=config.lowercase,
                tokenizer=config.tokenizer,
                strip_punct=config.strip_punct,
            )
            counter.update(toks)

        if document_count == 0:
            raise ValueError(
                f"No valid string values for field {text_field!r} were found in "
                f"JSONL files under: {jsonl_dir}"
            )
        if not counter:
            raise ValueError(
                f"The corpus under {jsonl_dir} contains valid text rows but no "
                "tokenizable text"
            )

        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        base_vocab: List[str] = [tok for tok, c in items if c >= config.min_freq]
        if config.max_vocab is not None and config.max_vocab > 0:
            base_vocab = base_vocab[: max(0, config.max_vocab)]

        final_tokens: List[str] = []
        for s in specials:
            if s not in final_tokens:
                final_tokens.append(s)
        for tok in base_vocab:
            if tok not in final_tokens:
                final_tokens.append(tok)

        token_to_id = {tok: i for i, tok in enumerate(final_tokens)}
        id_to_token = final_tokens

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            config=config,
            freqs=dict(counter),
        )

    # ---- I/O ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "config": asdict(self.config),
            "freqs": self.freqs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cfg = TokenizerConfig(**obj["config"])
        return cls(
            token_to_id=obj["token_to_id"],
            id_to_token=obj["id_to_token"],
            config=cfg,
            freqs=obj.get("freqs", {}),
        )

    # ---- Encode/Decode ----
    def encode_tokens(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[i] if 0 <= i < len(self.id_to_token) else UNK for i in ids]

    def tokenize(self, text: str) -> List[str]:
        return basic_tokenize(
            text,
            lowercase=self.config.lowercase,
            tokenizer=self.config.tokenizer,
            strip_punct=self.config.strip_punct,
        )

    def encode_text(self, text: str, with_bos_eos: bool = False) -> List[int]:
        toks = self.tokenize(text)
        if with_bos_eos:
            toks = ([BOS] if self.config.include_bos else []) + toks + (
                [EOS] if self.config.include_eos else []
            )
        return self.encode_tokens(toks)

    # ---- Reporting ----
    def unk_rate_on_dir(self, jsonl_dir: str, text_field: str = "text") -> float:
        total = 0
        unk = 0
        for doc in iter_text_from_jsonl_dir(jsonl_dir, text_field=text_field):
            toks = self.tokenize(doc)
            total += len(toks)
            unk += sum(1 for t in toks if t not in self.token_to_id)
        return (unk / total) if total > 0 else 0.0
