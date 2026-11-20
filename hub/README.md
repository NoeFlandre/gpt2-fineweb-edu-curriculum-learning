---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- gpt2
- fineweb-edu
- data-ordering
- pretraining
---

# GPT-2 45M: does the order of documents matter?

Four small GPT-2 models trained from scratch on the same 900M tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). They differ only in the order
in which documents are seen.

> **Status:** training in progress. The runs are trained one after another, and each run's files
> appear here when it finishes.

| Run | Order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest educational score first |
| `descending` | Highest educational score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

## Model

| | |
|---|---|
| Architecture | GPT-2: 6 layers, width 512, 8 heads, context 1024 |
| Parameters | 45.2M (18.9M excluding embeddings) |
| Vocabulary | GPT-2 BPE, 50,304 (padded); input and output embeddings tied |

## Training

| | |
|---|---|
| Data | FineWeb-Edu `sample-10BT`, shuffled (see Data below) |
| Budget | 1,716 steps of 524,288 tokens, which is 899.7M tokens per run |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on weight matrices |
| Learning rate | Peak 6e-4, linear warmup over 18 steps, cosine decay to 6e-5 |
| Other | Gradient clipping at 1.0, bf16 autocast, seed 42 |

## Data

The training pool is a shuffled slice of FineWeb-Edu `sample-10BT`, about 1.3 of its 14 files. Every document in the sample scores at least 2.5. The score mix, measured over the whole sample (9,672,101 documents, 9.97B GPT-2 tokens):

| Score bucket | Share of documents | Share of tokens |
|---|---|---|
| [2.5, 3.0) | 57.2% | 56.0% |
| [3.0, 3.5) | 29.5% | 30.0% |
| [3.5, 4.0) | 11.2% | 11.9% |
| [4.0, 5.0] | 2.1% | 2.1% |

Each file's shares lie within 0.4 percentage points of this table. The pool's exact mix was not counted, because its scores are not stored when the data is built.

## Evaluation

Every 50 steps, on:
- **Validation loss** over 2M held-out tokens, reported overall and per score bucket:
  [2.5, 3.0), [3.0, 3.5), [3.5, 4.0), [4.0, 5.0].
- **Benchmarks** (normalized accuracy): HellaSwag, PIQA and ARC-Easy.

Training curves are in the
[Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-ordering-trackio).

## Files

Each run has a folder `runs/<arm>/` containing:
- `model.pt`: final weights.
- `config.json`: training and model settings.
- `metrics.jsonl`: every logged training, validation and benchmark record.
- `checkpoint.pt`: latest training checkpoint (weights, optimizer state and data position).

## Usage

```python
import torch
from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.model.gpt2 import GPT2

model = GPT2(STUDY_CONFIG)
model.load_state_dict(torch.load("runs/shuffle/model.pt", map_location="cpu"))
model.eval()
```

The model code is in [GPT2-Fineweb-Edu-Experiments](https://github.com/NoeFlandre/GPT2-Fineweb-Edu-Experiments).

## Limitations

- One seed per run, so small differences between runs may be noise.
- The validation data comes from the same FineWeb-Edu stream as the training data.
- Quality buckets use an educational classifier's scores, so they reflect that classifier's judgement.
- This is a base pretraining model. It is not instruction-tuned.
