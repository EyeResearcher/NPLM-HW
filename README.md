# Neural Probabilistic Language Model

Bengio et al. (2003) replace sparse n-gram counts with a learned probability model: word IDs are mapped to distributed embeddings, a fixed window of embeddings is concatenated and passed through a nonlinear hidden layer, and a softmax predicts the next word. Learning the representation and predictor together lets the model share statistical strength among similar words. This repository implements that feed-forward architecture and evaluates it with token-level perplexity on WikiText-2.

## Quickstart

Python 3.10 or newer is required. Clone the repository, enter its directory, and install it in editable mode:

```bash
git clone <repository-url>
cd nplm
pip install -e .
python -m nplm.test_install
```

Download WikiText-2, convert nonempty text rows into case-preserving JSONL documents, and create a word-level vocabulary:

```bash
python -m nplm.download --dataset wikitext2 --out_dir data/raw/wikitext2

python -m nplm.preprocess \
  --input_dir data/raw/wikitext2 \
  --output_dir data/wikitext2_jsonl \
  --test_size 0.2 --val_size 0.1 \
  --seed 1337 --shard_size 1000

python -m nplm.build_word_vocab \
  --jsonl_dir data/wikitext2_jsonl/train \
  --output_path artifacts/wikitext2_word_vocab.json \
  --min_freq 2 --max_vocab 20000 \
  --report_unk_rate_dir data/wikitext2_jsonl/val
```

Train the selected medium model and evaluate its best validation checkpoint:

```bash
python -m nplm.train \
  --config configs/medium.yaml \
  --data_dir data/wikitext2_jsonl \
  --save_dir runs/medium

python -m nplm.eval \
  --checkpoint runs/medium/best.pt \
  --data_dir data/wikitext2_jsonl \
  --out_json results/metrics.json
```

The reported medium checkpoint obtained train/validation/test perplexities of 75.69/179.13/180.59. Full experiment details and comparisons are in [results/EXPERIMENTS.md](results/EXPERIMENTS.md). Run the test suite with `python -m pytest -q`.

## Hardware

The reported tiny, medium, and large experiments were trained with an NVIDIA GeForce RTX 4070; the selected medium run took 146.6 seconds. All configs use `device: auto`, which selects CUDA when available and otherwise uses the CPU. The 1.95M-parameter `tiny.yaml` configuration is intended for the required under-10-minute CPU reproduction, while a CUDA-capable GPU is recommended for quickly reproducing the medium or large experiments.
