# Architecture

```
gpt2_scratch/
  model/       GPT-2: config (including the study's model), attention, MLP, block, model
  data/        tokenization, token shards, token loader, the study's corpus and orderings
  experiment/  the four document orderings
  evaluation/  multiple-choice benchmarks (HellaSwag, PIQA, ARC-Easy)
  training/    configuration, learning-rate schedule, training loop, validation, tracking
  reporting/   figures comparing the arms
  publishing.py  checkpoints to and from the Hugging Face model repository
```

## Data flow

1. `scripts/prepare_data.py` streams FineWeb-Edu. Rows with an integer score of at least 3 are
   tokenized with an end-of-text prefix, and the first 899,678,208 tokens form the training pool,
   plus one token so the last input of an arm has its target. The validation split is drawn from
   the rows after the pool: 500k tokens in each score bucket, until every bucket is full.
2. Each arm's order is computed from the pool's continuous `score` column (`experiment/arms.py`),
   and the documents are concatenated in that order (`data/corpus.py`) into one shard per arm.
3. The validation split keeps each token's score bucket, fixed at [2.5, 3.0), [3.0, 3.5),
   [3.5, 4.0) and [4.0, 5.0].
4. `TokenLoader` yields batches of inputs and targets. `training/runner.py` trains on them and,
   every 50 steps and after the last one, records the validation loss per bucket and each
   benchmark's accuracy (`evaluation/multiple_choice.py`).

## Design choices

- **Single GPU.** Distributed training was removed: the experiments need one device.
- **Arms are orders, not weights.** Every arm uses the same tokens and the same loss; only the
  order of documents differs. The model's loss is the plain mean.
- **Scores are read, not recomputed.** The FineWeb-Edu `score` column is used as published.
- **One token file per arm.** The loader is a single sequential reader, and the orders are
  precomputed.
- **Metrics are JSON Lines.** Plots and reports read the records the trainer writes.
