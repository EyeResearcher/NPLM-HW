"""Feed-forward neural probabilistic language model.

The model follows the single-hidden-layer architecture from the assignment:
context token embeddings are concatenated, transformed by a hidden layer, and
projected to unnormalized vocabulary logits.
"""

from numbers import Real

from torch import Tensor, nn


def _require_positive_int(name: str, value: int) -> None:
    """Validate a positive integer model dimension."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class NPLM(nn.Module):
    """A single-hidden-layer, fixed-context neural language model."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        activation: str = "tanh",
        dropout: float = 0.0,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()

        for name, value in (
            ("vocab_size", vocab_size),
            ("context_size", context_size),
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
        ):
            _require_positive_int(name, value)

        if activation not in ("tanh", "relu"):
            raise ValueError(
                f"activation must be 'tanh' or 'relu', got {activation!r}"
            )
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise ValueError(f"dropout must be a number in [0, 1), got {dropout!r}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if padding_idx is not None:
            if (
                isinstance(padding_idx, bool)
                or not isinstance(padding_idx, int)
                or not 0 <= padding_idx < vocab_size
            ):
                raise ValueError(
                    "padding_idx must be an integer vocabulary ID in "
                    f"[0, {vocab_size}), got {padding_idx!r}"
                )

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_ids: Tensor) -> Tensor:
        """Map token IDs of shape ``[batch, context]`` to raw logits."""
        if context_ids.ndim != 2:
            raise ValueError(
                "context_ids must have rank 2 with shape [batch, context], "
                f"got shape {tuple(context_ids.shape)}"
            )
        if context_ids.shape[1] != self.context_size:
            raise ValueError(
                f"context_ids width must equal context_size={self.context_size}, "
                f"got shape {tuple(context_ids.shape)}"
            )

        embeddings = self.embedding(context_ids)
        concatenated = embeddings.flatten(start_dim=1)
        hidden = self.dropout(self.activation(self.hidden(concatenated)))
        return self.output(hidden)
