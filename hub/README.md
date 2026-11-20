---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- gpt2
- from-scratch
- fineweb-edu
- curriculum
---

# GPT-2 (45M) on FineWeb-Edu: ordering by classifier score

A small GPT-2 (6 layers, width 512, about 45M parameters) trained from scratch on 900M tokens of
FineWeb-Edu documents, in four orders of the same pool:

| Arm | Order |
|---|---|
| `shuffle` | Random order |
| `ascending` | Lowest score first |
| `descending` | Highest score first |
| `cooldown` | Lowest-scoring 80% of tokens shuffled, then the highest-scoring 20% shuffled |

**Live results:** the [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-ordering-trackio)
shows the validation loss per FineWeb-Edu score bucket (the primary metric), the overall validation
loss, and HellaSwag, PIQA and ARC-Easy accuracy, every 50 steps.

Score buckets: [2.5, 3.0), [3.0, 3.5), [3.5, 4.0), [4.0, 5.0].

Full method: [docs/experiment.md](https://github.com/NoeFlandre/GPT2-Fineweb-Edu-Experiments/blob/main/docs/experiment.md).

## Files

- `runs/<arm>/model.pt`: final weights
- `runs/<arm>/config.json`: arm, training and model settings
- `runs/<arm>/metrics.jsonl`: every training, validation and benchmark record
- `runs/<arm>/checkpoint.pt`: latest checkpoint (weights, optimizer and data position), published every 100 steps

## Resuming

```bash
uv run python -m scripts.train --arm cooldown --out runs/cooldown
```

If `runs/cooldown/checkpoint.pt` is missing locally, it is downloaded from this repository and
training continues on the same data.
