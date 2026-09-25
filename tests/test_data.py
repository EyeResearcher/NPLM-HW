import json
import tempfile
import unittest
from pathlib import Path

import torch

from nplm.data import NPLMDataset, create_dataloader, create_dataloaders
from nplm.word_tokenizer import TokenizerConfig, WordTokenizer


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}) + "\n")


def _tokenizer() -> WordTokenizer:
    tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma"]
    return WordTokenizer(
        token_to_id={token: index for index, token in enumerate(tokens)},
        id_to_token=tokens,
        config=TokenizerConfig(min_freq=1),
    )


class NPLMDatasetTests(unittest.TestCase):
    def test_documents_have_independent_padded_windows_and_eos_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            _write_jsonl(split / "shard_00000.jsonl", ["alpha beta", "gamma"])
            tokenizer = _tokenizer()

            dataset = NPLMDataset(split, tokenizer, context_size=3)
            actual = [
                (context.tolist(), target.item()) for context, target in dataset
            ]

            self.assertEqual(
                actual,
                [
                    ([0, 0, 2], 4),
                    ([0, 2, 4], 5),
                    ([2, 4, 5], 3),
                    ([0, 0, 2], 6),
                    ([0, 2, 6], 3),
                ],
            )

    def test_items_and_default_collated_batches_have_contract_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            _write_jsonl(split / "shard_00000.jsonl", ["alpha beta"])

            dataset = NPLMDataset(split, _tokenizer(), context_size=2)
            context, target = dataset[0]
            self.assertEqual(context.shape, torch.Size([2]))
            self.assertEqual(target.shape, torch.Size([]))
            self.assertEqual(context.dtype, torch.long)
            self.assertEqual(target.dtype, torch.long)

            contexts, targets = next(
                iter(
                    create_dataloader(
                        split,
                        _tokenizer(),
                        context_size=2,
                        batch_size=2,
                        shuffle=False,
                    )
                )
            )
            self.assertEqual(contexts.shape, torch.Size([2, 2]))
            self.assertEqual(targets.shape, torch.Size([2]))
            self.assertEqual(contexts.dtype, torch.long)
            self.assertEqual(targets.dtype, torch.long)

    def test_shards_are_read_in_lexical_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            _write_jsonl(split / "shard_00001.jsonl", ["beta"])
            _write_jsonl(split / "shard_00000.jsonl", ["alpha"])

            dataset = NPLMDataset(split, _tokenizer(), context_size=1)

            self.assertEqual(dataset[0][1].item(), 4)
            self.assertEqual(dataset[2][1].item(), 5)

    def test_create_dataloaders_builds_all_splits_without_eval_shuffle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                _write_jsonl(root / split / "shard_00000.jsonl", ["alpha"])

            loaders = create_dataloaders(
                root, _tokenizer(), context_size=2, batch_size=4, shuffle_train=False
            )

            self.assertEqual(list(loaders), ["train", "val", "test"])
            for loader in loaders.values():
                self.assertEqual(len(loader.dataset), 2)

    def test_invalid_sizes_paths_and_empty_splits_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "context_size"):
                NPLMDataset(root, _tokenizer(), context_size=0)
            with self.assertRaisesRegex(FileNotFoundError, "No JSONL shards"):
                NPLMDataset(root, _tokenizer(), context_size=2)

            empty_shard = root / "empty" / "shard_00000.jsonl"
            empty_shard.parent.mkdir()
            empty_shard.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split is empty"):
                NPLMDataset(empty_shard.parent, _tokenizer(), context_size=2)

            valid = root / "valid" / "shard_00000.jsonl"
            _write_jsonl(valid, ["alpha"])
            with self.assertRaisesRegex(ValueError, "batch_size"):
                create_dataloader(
                    valid.parent,
                    _tokenizer(),
                    context_size=2,
                    batch_size=0,
                    shuffle=False,
                )

    def test_invalid_pad_or_emitted_token_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            _write_jsonl(split / "shard_00000.jsonl", ["alpha"])

            tokenizer = _tokenizer()
            tokenizer.pad_id = None
            with self.assertRaisesRegex(ValueError, "valid pad_id"):
                NPLMDataset(split, tokenizer, context_size=2)

            tokenizer = _tokenizer()
            tokenizer.token_to_id["alpha"] = len(tokenizer.id_to_token)
            with self.assertRaisesRegex(ValueError, "outside its vocabulary"):
                NPLMDataset(split, tokenizer, context_size=2)


if __name__ == "__main__":
    unittest.main()
