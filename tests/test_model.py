import unittest

import torch

from nplm.model import NPLM


class NPLMTests(unittest.TestCase):
    def test_output_shape_for_multiple_batch_sizes(self) -> None:
        model = NPLM(17, 4, 5, 7)

        for batch_size in (1, 3, 8):
            contexts = torch.randint(0, 17, (batch_size, 4))
            self.assertEqual(model(contexts).shape, (batch_size, 17))

    def test_backward_reaches_embedding_and_linear_layers(self) -> None:
        model = NPLM(11, 3, 4, 6, activation="relu")
        logits = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        logits.sum().backward()

        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertIsNotNone(model.hidden.weight.grad)
        self.assertIsNotNone(model.output.weight.grad)

    def test_constructor_rejects_invalid_arguments(self) -> None:
        for argument in ("vocab_size", "context_size", "embedding_dim", "hidden_dim"):
            kwargs = dict(vocab_size=10, context_size=3, embedding_dim=4, hidden_dim=5)
            kwargs[argument] = 0
            with self.subTest(argument=argument), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                NPLM(**kwargs)

        with self.assertRaisesRegex(ValueError, "activation"):
            NPLM(10, 3, 4, 5, activation="gelu")
        for invalid_dropout in (-0.1, 1.0):
            with self.subTest(dropout=invalid_dropout), self.assertRaisesRegex(
                ValueError, "dropout"
            ):
                NPLM(10, 3, 4, 5, dropout=invalid_dropout)
        for invalid_padding_idx in (-1, 10):
            with self.subTest(padding_idx=invalid_padding_idx), self.assertRaisesRegex(
                ValueError, "padding_idx"
            ):
                NPLM(10, 3, 4, 5, padding_idx=invalid_padding_idx)

    def test_forward_rejects_invalid_context_shape(self) -> None:
        model = NPLM(10, 3, 4, 5)

        with self.assertRaisesRegex(ValueError, "rank 2"):
            model(torch.ones(3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "context_size=3"):
            model(torch.ones((2, 4), dtype=torch.long))

    def test_eval_mode_disables_dropout(self) -> None:
        torch.manual_seed(0)
        model = NPLM(10, 3, 4, 5, dropout=0.75)
        contexts = torch.tensor([[1, 2, 3], [3, 2, 1]])

        model.eval()
        first = model(contexts)
        second = model(contexts)

        self.assertTrue(torch.equal(first, second))

    def test_forward_returns_raw_logits(self) -> None:
        model = NPLM(7, 2, 3, 4)
        logits = model(torch.tensor([[1, 2]]))

        self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
