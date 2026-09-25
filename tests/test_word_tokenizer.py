import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nplm.build_word_vocab import main as build_vocab_main
from nplm.word_tokenizer import (
    BOS,
    EOS,
    PAD,
    UNK,
    TokenizerConfig,
    WordTokenizer,
    iter_text_from_jsonl_dir,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            if isinstance(row, str):
                stream.write(row + "\n")
            else:
                stream.write(json.dumps(row) + "\n")


class WordTokenizerTests(unittest.TestCase):
    def test_frequency_cutoff_max_vocab_and_tie_breaking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory)
            _write_jsonl(
                corpus / "shard_00000.jsonl",
                [{"text": "zebra apple pear"}, {"text": "pear apple zebra kiwi"}],
            )

            tokenizer = WordTokenizer.build_from_corpus(
                str(corpus),
                TokenizerConfig(min_freq=2, max_vocab=2),
                progress=False,
            )

            self.assertEqual(tokenizer.id_to_token[:4], [PAD, UNK, BOS, EOS])
            self.assertEqual(tokenizer.id_to_token[4:], ["apple", "pear"])
            self.assertNotIn("kiwi", tokenizer.token_to_id)
            self.assertEqual(tokenizer.freqs["zebra"], 2)

    def test_round_trip_preserves_ids_config_and_frequencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_jsonl(root / "corpus" / "data.jsonl", [{"text": "One one two"}])
            config = TokenizerConfig(min_freq=1, lowercase=True, strip_punct=True)
            tokenizer = WordTokenizer.build_from_corpus(
                str(root / "corpus"), config, progress=False
            )
            artifact = root / "nested" / "vocab.json"

            tokenizer.save(str(artifact))
            loaded = WordTokenizer.load(str(artifact))

            self.assertEqual(loaded.token_to_id, tokenizer.token_to_id)
            self.assertEqual(loaded.id_to_token, tokenizer.id_to_token)
            self.assertEqual(loaded.config, tokenizer.config)
            self.assertEqual(loaded.freqs, tokenizer.freqs)

    def test_oov_encoding_and_unknown_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_jsonl(root / "train" / "data.jsonl", [{"text": "known known"}])
            _write_jsonl(root / "eval" / "data.jsonl", [{"text": "known new other known"}])
            tokenizer = WordTokenizer.build_from_corpus(
                str(root / "train"),
                TokenizerConfig(min_freq=1),
                progress=False,
            )

            self.assertEqual(
                tokenizer.encode_tokens(["known", "missing"]),
                [tokenizer.token_to_id["known"], tokenizer.unk_id],
            )
            self.assertEqual(tokenizer.unk_rate_on_dir(str(root / "eval")), 0.5)

    def test_disabled_boundaries_are_not_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory)
            _write_jsonl(corpus / "data.jsonl", [{"text": "some text"}])

            tokenizer = WordTokenizer.build_from_corpus(
                str(corpus),
                TokenizerConfig(
                    min_freq=1,
                    include_bos=False,
                    include_eos=False,
                    specials=[PAD, PAD, UNK, BOS, EOS, "<mask>"],
                ),
                progress=False,
            )

            self.assertEqual(tokenizer.id_to_token[:3], [PAD, UNK, "<mask>"])
            self.assertNotIn(BOS, tokenizer.token_to_id)
            self.assertNotIn(EOS, tokenizer.token_to_id)
            self.assertIsNone(tokenizer.bos_id)
            self.assertIsNone(tokenizer.eos_id)

    def test_jsonl_iterator_reads_gzip_and_skips_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "data.jsonl.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"text": "good"}\n')
                stream.write("not-json\n")
                stream.write('{"text": 3}\n')

            self.assertEqual(list(iter_text_from_jsonl_dir(str(root))), ["good"])

    def test_missing_or_invalid_corpus_raises_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                WordTokenizer.build_from_corpus(
                    str(root / "missing"), progress=False
                )

            _write_jsonl(root / "invalid" / "data.jsonl", ["not-json", {"other": "x"}])
            with self.assertRaisesRegex(ValueError, "No valid string values"):
                WordTokenizer.build_from_corpus(
                    str(root / "invalid"), progress=False
                )

            _write_jsonl(root / "empty" / "data.jsonl", [{"text": "   "}])
            with self.assertRaisesRegex(ValueError, "no tokenizable text"):
                WordTokenizer.build_from_corpus(str(root / "empty"), progress=False)

    def test_build_vocab_cli_writes_loadable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "train"
            artifact = root / "artifacts" / "vocab.json"
            _write_jsonl(corpus / "data.jsonl", [{"text": "Alpha alpha beta"}])
            argv = [
                "build_word_vocab",
                "--jsonl_dir",
                str(corpus),
                "--output_path",
                str(artifact),
                "--min_freq",
                "1",
                "--max_vocab",
                "1",
                "--lowercase",
            ]

            with patch("sys.argv", argv):
                build_vocab_main()

            tokenizer = WordTokenizer.load(str(artifact))
            self.assertEqual(tokenizer.id_to_token, [PAD, UNK, BOS, EOS, "alpha"])


if __name__ == "__main__":
    unittest.main()
