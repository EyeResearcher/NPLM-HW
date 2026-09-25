# NPLM Agent Contracts

These contracts freeze the interfaces between implementation agents. They are
architecture documents, not implementations. When a starter docstring conflicts
with this document, this document wins unless `ASSIGNMENT.md` explicitly requires
otherwise.

## 1. Assignment requirement ownership

| Requirement | Owning component |
| --- | --- |
| Download a public corpus with a CLI | `download.py` |
| Read raw data, form documents, split, and write JSONL shards | `preprocess.py` |
| Build, save, load, and use a closed word vocabulary | `word_tokenizer.py`, `build_word_vocab.py` |
| Form fixed-context examples without crossing documents | `data.py` |
| Embedding, concatenated context, hidden layer, and vocabulary logits | `model.py` |
| Configs, seeds, devices, checkpoints, timing, TorchMetrics aggregation, and TensorBoard helpers | `utils.py` |
| Cross-entropy training, validation, TensorBoard run logging, best checkpoint, and resume | `train.py` |
| Best-checkpoint train/validation/test loss, accuracy, perplexity, and `metrics.json` | `eval.py` |
| `tiny.yaml` and `medium.yaml` | Training agent, using the config contract below |

Repository documentation and experiment reports are final integration work and
are outside any one component contract.

## 2. Dependency graph

```text
download.py --writes--> raw Hugging Face dataset on disk
                         |
                         v
preprocess.py --writes--> JSONL split directories
                         |
                         +--> word_tokenizer.py/build_word_vocab.py --writes--> vocab JSON
                         |                                                   |
                         +---------------------------------------------------+
                                                                             v
                                                        data.py --batches--> train.py
                                                           |                   ^
                                                           +--------------+    |
                                                                          v    |
model.py --------------------------------------------------------------> eval.py
utils.py --------------------------------------------------------------> train.py, eval.py
```

Allowed internal imports:

- `build_word_vocab.py` may import `word_tokenizer.py`.
- `data.py` may import `word_tokenizer.py`.
- `train.py` and `eval.py` may import `word_tokenizer.py`, `data.py`,
  `model.py`, and `utils.py`.
- `download.py`, `preprocess.py`, `model.py`, and `utils.py` must not depend on
  other NPLM components.
- No library component may import `train.py` or `eval.py`.

## 3. Shared data contracts

### 3.1 Raw downloaded dataset

`download.py` writes a Hugging Face `DatasetDict` using
`DatasetDict.save_to_disk(out_dir)`. Every split must contain a string column
named `text`. `preprocess.read_dataset()` loads exactly this representation with
`datasets.load_from_disk()`.

The required dataset alias is `wikitext2`, mapped to the WikiText-2 raw variant.
Supporting additional corpora is optional and must not change the on-disk
contract.

### 3.2 In-memory documents and splits

`preprocess.generate_documents()` returns `list[str]`. Each item is one trimmed,
nonempty source row and is one independent document. Optional lowercasing occurs
here. Documents from all source splits are pooled before the deterministic split
so CLI proportions control the final `train`, `val`, and `test` sizes.

`preprocess.split_dataset()` returns:

```python
dict[str, list[str]]  # exactly the keys "train", "val", and "test"
```

`test_size` and `val_size` are proportions of the whole corpus. Both must be in
`(0, 1)`, their sum must be less than 1, and all three resulting splits must be
nonempty. Splitting uses a local PRNG initialized from `seed`; it must not mutate
global random state.

### 3.3 JSONL corpus

The directory layout is:

```text
<data_dir>/
  train/shard_00000.jsonl
  val/shard_00000.jsonl
  test/shard_00000.jsonl
```

Files are UTF-8 JSON Lines. Every nonblank line has exactly the required schema:

```json
{"text": "document text here"}
```

Documents remain independent across lines and shards. Shards are enumerated in
stable lexical order and contain at most `shard_size` documents. Each contracted
split is nonempty and therefore contains at least one shard.

### 3.4 Tokenizer API and artifact

The provided `WordTokenizer` and `TokenizerConfig` are the canonical tokenizer
API and should be extended only when necessary. Consumers may rely on:

```python
WordTokenizer.build_from_corpus(jsonl_dir, config=None,
                                text_field="text", progress=True) -> WordTokenizer
WordTokenizer.save(path) -> None
WordTokenizer.load(path) -> WordTokenizer
tokenizer.tokenize(text) -> list[str]
tokenizer.encode_tokens(tokens) -> list[int]
tokenizer.encode_text(text, with_bos_eos=False) -> list[int]
tokenizer.decode_ids(ids) -> list[str]
tokenizer.unk_rate_on_dir(jsonl_dir, text_field="text") -> float
```

Consumers may also read `token_to_id`, `id_to_token`, `config`, `pad_id`,
`unk_id`, `bos_id`, and `eos_id`. `<pad>` and `<unk>` are mandatory. The chosen
pipeline also includes `<bos>` and `<eos>`. With default specials their IDs are
0, 1, 2, and 3 respectively. Vocabulary size is
`len(tokenizer.id_to_token)`.

The JSON artifact schema, already established by the starter, is:

```json
{
  "token_to_id": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
  "id_to_token": ["<pad>", "<unk>", "<bos>", "<eos>"],
  "config": {
    "min_freq": 2,
    "max_vocab": 20000,
    "lowercase": false,
    "tokenizer": "simple",
    "strip_punct": false,
    "include_bos": true,
    "include_eos": true,
    "specials": null
  },
  "freqs": {}
}
```

`token_to_id` and `id_to_token` must be exact inverses. `max_vocab` limits
ordinary tokens and excludes special tokens. `freqs` records training-corpus
counts and may contain tokens excluded from the vocabulary.

### 3.5 Window and batch semantics

Each document is encoded with boundary tokens. For every token after `<bos>`,
including `<eos>`, one example is emitted. The target is that token. The context
is the preceding `C` token IDs, left-padded with `pad_id` when fewer than `C`
tokens exist. Windows never cross JSONL document boundaries.

The dataset item is:

```python
(context, target)
```

- `context`: `torch.LongTensor` with shape `[C]`
- `target`: scalar `torch.LongTensor` with shape `[]`

Default PyTorch collation produces:

- contexts: `torch.LongTensor[B, C]`
- targets: `torch.LongTensor[B]`

### 3.6 Config schema

Configs are YAML mappings. `utils.load_config()` applies defaults and validates
the following normalized schema:

| Key | Type | Rule/default |
| --- | --- | --- |
| `seed` | `int` | default `1337` |
| `tokenizer_type` | `str` | must be `"word"` |
| `tokenizer_path` | `str` | required |
| `context_size` | `int` | required, `> 0` |
| `embedding_dim` | `int` | required, `> 0` |
| `hidden_dim` | `int` | required, `> 0` |
| `activation` | `str` | `"tanh"` or `"relu"`; default `"tanh"` |
| `dropout` | `float` | default `0.0`, in `[0, 1)` |
| `optimizer` | `str` | must be `"adam"` |
| `lr` | `float` | required, `> 0` |
| `batch_size` | `int` | required, `> 0` |
| `max_steps` | `int` | required, `> 0` |
| `log_every` | `int` | default `100`, `> 0` |
| `eval_every` | `int` | required, `> 0` |
| `clip_grad_norm` | `float | null` | default `null`; if set, `> 0` |
| `device` | `str` | `"auto"`, `"cpu"`, or `"cuda"`; default `"auto"` |
| `num_workers` | `int` | default `0`, `>= 0` |
| `run_name` | `str | null` | default `null`; when null use `Path(save_dir).name` |

`vocab_size`, if present for readability, must equal the value derived from the
tokenizer artifact; code must never truncate or pad the vocabulary to match it.
Unknown keys are configuration errors so misspellings do not silently pass.

### 3.7 Model and tensor flow

Symbols: `B` batch size, `C` context size, `E` embedding dimension, `H` hidden
dimension, and `V` vocabulary size.

```text
token IDs [B,C]
  -> shared embedding lookup [B,C,E]
  -> flatten/concatenate [B,C*E]
  -> Linear(C*E,H) + activation + dropout [B,H]
  -> Linear(H,V) [B,V] logits
  -> cross_entropy(logits, targets [B]) scalar mean NLL
  -> exp(mean NLL) scalar perplexity
```

`model.forward()` returns raw logits. It must not apply softmax because
`torch.nn.CrossEntropyLoss` expects logits.

### 3.8 Metric aggregation and run logging

TorchMetrics is the canonical stateful aggregation layer. Loss aggregation uses
`torchmetrics.MeanMetric`, updated with the mean batch cross-entropy and
`weight=targets.numel()`. Next-token top-1 accuracy uses
`torchmetrics.classification.MulticlassAccuracy` with `average="micro"` and
`num_classes=V`. Trackers are computed and reset at explicit logging boundaries;
metric state must never leak between training, validation, test, or distinct
runs.

Every training invocation is one run, identified by its unique `save_dir` and
the resolved `run_name`. It writes:

```text
<save_dir>/
  best.pt
  last.pt
  config.json
  tensorboard/
    events.out.tfevents...
```

`config.json` is the normalized configuration actually used, including defaults
and resolved `run_name`. TensorBoard logs use the global optimizer step so a
resume continues the same x-axis. At minimum, the event stream contains:

- `train/loss`, `train/accuracy`, and `train/learning_rate` every `log_every` steps;
- `validation/loss`, `validation/perplexity`, and `validation/accuracy` every
  `eval_every` steps and at final validation;
- `system/elapsed_seconds` at each logging event;
- the run name and normalized configuration as text or hparameters.

Pointing TensorBoard at the common parent compares runs created with different
configs, for example `tensorboard --logdir runs` for `runs/exp1`, `runs/exp2`,
and so on. Writers must be flushed after validation/checkpoint events and closed
in a `finally` block on success or failure.

### 3.9 Checkpoint schema

Checkpoints are dictionaries written with `torch.save`:

```python
{
    "format_version": 1,
    "model_state_dict": dict,
    "optimizer_state_dict": dict | None,
    "step": int,
    "best_val_loss": float,
    "model_config": {
        "vocab_size": int,
        "context_size": int,
        "embedding_dim": int,
        "hidden_dim": int,
        "activation": str,
        "dropout": float,
    },
    "training_config": dict,
    "tokenizer_type": "word",
    "tokenizer_path": str,
    "train_time_sec": float,
}
```

`best.pt` contains the parameters with the lowest observed validation mean NLL.
`last.pt` contains the final/resumable state. Paths are written atomically.
Architecture fields in a resume config must match the checkpoint. Runtime fields
such as `max_steps`, `eval_every`, and `device` may change.

### 3.10 Metrics schema

`eval.py` writes strict JSON (no NaN or Infinity):

```json
{
  "train_loss": 0.0,
  "val_loss": 0.0,
  "test_loss": 0.0,
  "train_accuracy": 0.0,
  "val_accuracy": 0.0,
  "test_accuracy": 0.0,
  "train_ppl": 0.0,
  "val_ppl": 0.0,
  "test_ppl": 0.0,
  "tokenizer": "word",
  "vocab_size_or_merges": 0,
  "context_size": 0,
  "train_time_sec": 0.0
}
```

Losses are token-weighted mean negative log likelihood, accuracies are micro
top-1 next-token accuracy in `[0, 1]`, and perplexities are `exp(loss)`. No
metric may be an unweighted mean of per-batch values.

## 4. Architectural decisions

These are simplifying choices, not additional assignment requirements:

1. Implement only the provided word-level path. BPE is optional in the
   assignment and would create a second tokenizer/data contract.
2. Use WikiText-2 raw as the required download target and one nonblank source
   row as one document.
3. Pool source splits and deterministically resplit them so requested CLI
   proportions have unambiguous meaning.
4. Use a single hidden layer. The aspirational multi-layer language in the
   empty `model.py` docstring is broader than the required Bengio-style model.
5. Use `<bos>` plus left `<pad>` at the boundary and train on `<eos>` as a
   target. Never cross document boundaries.
6. Support Adam and full-vocabulary softmax only. Sampled/adaptive softmax is
   optional extra credit.
7. Use validation mean NLL for checkpoint selection and recompute split
   perplexities in `eval.py`.
8. Keep the existing, working name `build_word_vocab.py`; do not create a
   duplicate `build_word_tokenizer.py`.
9. TensorBoard run logging and TorchMetrics loss/accuracy aggregation are part
   of the frozen contract so differently configured runs are directly
   comparable. Multi-layer networks, early stopping, and URL-based downloading
   remain outside the frozen minimum.

## 5. Agent context map

| Agent | Must read |
| --- | --- |
| Download | `ASSIGNMENT.md`, `download.md`, raw dataset contract |
| Preprocess | `ASSIGNMENT.md`, `preprocess.md`, raw and JSONL contracts |
| Tokenizer | `ASSIGNMENT.md`, `tokenizer.md`, JSONL and tokenizer contracts |
| Data | `ASSIGNMENT.md`, `data.md`, JSONL, tokenizer, window/batch contracts |
| Model | `ASSIGNMENT.md`, `model.md`, model/tensor and config contracts |
| Utils | `ASSIGNMENT.md`, `utils.md`, config, checkpoint, metrics contracts |
| Train | `ASSIGNMENT.md`, `train.md`, tokenizer, batch, config, model, checkpoint contracts |
| Eval | `ASSIGNMENT.md`, `eval.md`, batch, model, checkpoint, metrics contracts |

Agents must not rely on private helpers owned by another component.
