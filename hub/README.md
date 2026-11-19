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
- data-quality
---

# GPT-2 124M on FineWeb-Edu: uniform, hard and soft data weighting

GPT-2 (124M) trained from scratch on 100M tokens per arm, comparing three ways of using the
FineWeb-Edu quality score:

| Arm | Documents | Loss weight per token |
|---|---|---|
| `uniform` | All FineWeb documents | 1 |
| `hard` | FineWeb-Edu documents (score 3 to 5) | 1 |
| `soft` | The same documents as `uniform` | score / 5 |

Status: `uniform` and `soft` are being trained; `hard` is pending.

**Live results:** the [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/GPT2-Fineweb-Edu-Experiments-trackio)
shows validation loss and HellaSwag normalized accuracy as curves. Both are recorded every 10 steps
and after the last update (190 steps per arm).

HellaSwag is scored as in the original repository: the full validation set (10,042 examples), and an
example counts as correct when its ending with the lowest average per-token loss is the labelled one.

## Files

- `runs/<arm>/model.pt`: final weights
- `runs/<arm>/config.json`: arm, training and model settings
- `runs/<arm>/metrics.jsonl`: every training, validation and HellaSwag record
- `runs/<arm>/checkpoint.pt`: latest checkpoint (weights, optimizer and data position), published every 50 steps
- `data/arms/`: the tokenized shards for the raw and edu sources

## Resuming

```bash
uv run python -m scripts.train --arm soft --out runs/soft
```

If `runs/soft/checkpoint.pt` is missing locally, it is downloaded from this repository, and training
continues from its step on the same data.

## Source

Code, method and experiment notes:
[GPT2-Fineweb-Edu-Experiments](https://github.com/NoeFlandre/GPT2-Fineweb-Edu-Experiments), see
`docs/experiment.md`.
