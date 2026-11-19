# Architecture

```
gpt2_scratch/
  model/       GPT-2: config, attention, MLP, block, model, weighted loss
  data/        tokenization, shard files, token loader, document budgeting
  experiment/  arm definitions and the educational-score classifier wrapper
  evaluation/  HellaSwag completion accuracy
  training/    configuration, learning-rate schedule, training loop, metrics
  reporting/   figures comparing runs
```

## Data flow

1. `scripts/prepare_data.py` streams FineWeb and FineWeb-Edu, tokenizes each document with an end-of-text prefix, and writes shards.
2. Every raw document gets an integer score from `EducationalScorer`. Each token stores its document's score.
3. `TokenLoader` yields batches of inputs, targets and the score of each target token.
4. The arm turns scores into loss weights; `GPT2.forward` applies them through `weighted_mean`.

## Design choices

- **Single GPU.** Distributed training was removed: the experiments need one device, and that code was untested.
- **One token file per split plus a score file.** Scores sit beside the tokens, so the loader stays a single sequential reader.
- **Arms are data, not code paths.** `ARMS` maps a name to a document source and a weight function; the training loop does not branch on the arm.
- **Metrics are JSON Lines.** Plots and reports read the same records the trainer writes.
