# Experiments

## Setup

I used the raw WikiText-2 corpus. Empty rows were removed, and each remaining text row was treated as a separate document; windows never cross document boundaries. The source rows were shuffled with seed 1337 and split 70/10/20 into 20,385 training, 2,911 validation, and 5,823 test documents, stored in JSONL shards of at most 1,000 documents. Text was not lowercased.

The Bengio-style word tokenizer uses simple regex tokenization, preserves punctuation, requires a minimum frequency of 2, and caps ordinary vocabulary entries at 20,000. With `<pad>`, `<unk>`, `<bos>`, and `<eos>`, the final vocabulary contains 20,004 tokens. Unknown-token rates were 5.93% on validation and 5.87% on test.

All models used five preceding tokens, a `tanh` hidden layer, Adam with learning rate 0.001, gradient clipping at 1.0, seed 1337, and 20,000 training steps. The table reports each run's best-validation checkpoint; time is the total completed-run wall time.

| Config | Embedding / hidden | Parameters | Dropout | Best step | Train PPL | Val PPL | Test PPL | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 32 / 64 | 1.95M | 0.0 | 20,000 | 203.06 | 225.00 | 225.27 | 116.9 |
| Medium | 128 / 256 | 7.87M | 0.1 | 20,000 | **75.69** | **179.13** | **180.59** | 146.7 |
| Large | 512 / 1,024 | 33.37M | 0.1 | 3,000 | 104.97 | 237.66 | 238.43 | 368.0 |

## Observations

Increasing capacity from tiny to medium helped substantially: test perplexity fell by about 20%. Scaling further hurt under the shared optimization settings. The large model reached its best validation result after 3,000 steps (57.4 seconds), then validation performance degraded while training continued, suggesting that its learning rate, stopping point, or regularization needed separate tuning. The medium model therefore offered the best accuracy/compute tradeoff and was promoted to `results/metrics.json`.
