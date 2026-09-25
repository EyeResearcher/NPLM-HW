# Contract: `model.py`

## Scope

Implement the required single-hidden-layer feed-forward NPLM. The empty starter
docstring discusses arbitrary hidden-layer tuples, but that abstraction is not
required and is deliberately excluded from the frozen interface.

## Public API

```python
class NPLM(torch.nn.Module):
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
    ) -> None: ...

    def forward(self, context_ids: Tensor) -> Tensor:
        """Map LongTensor[B,C] to raw FloatTensor[B,V] logits."""
```

Public scalar attributes must include `vocab_size`, `context_size`,
`embedding_dim`, and `hidden_dim` so checkpoint/config validation can inspect
them.

Constructor validation:

- Dimensions must be positive integers.
- `activation` must be `"tanh"` or `"relu"`.
- `dropout` must be in `[0, 1)`.
- If supplied, `padding_idx` must be a valid vocabulary ID.

`forward` requires rank 2 and width `context_size`; violations raise
`ValueError`. PyTorch should naturally reject noninteger or out-of-range IDs.
Do not apply softmax or compute loss in the model.

## Dependencies and consumers

Dependency: PyTorch only. Do not open tokenizer artifacts or config files in the
model. `train.py` and `eval.py` construct it from normalized config/checkpoint
values.

## Verification

- Output shape is `[B,V]` for multiple batch sizes.
- Backpropagation reaches embedding and both linear layers.
- Invalid dimensions, activation, dropout, and context shape fail clearly.
- Evaluation mode disables dropout through standard `nn.Module` behavior.
