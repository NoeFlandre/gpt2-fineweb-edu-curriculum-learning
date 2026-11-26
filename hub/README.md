---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- gpt2
- curriculum-learning
- data-ordering
- fineweb-edu
- pretraining
---

# GPT-2 45M: curriculum learning on FineWeb-Edu

Four small GPT-2 models, trained from scratch on the same 900M tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). They differ only in the
order in which documents are seen, which is set by each document's educational score.

> **Status:** training complete. All four runs are in this repository.

| Run | Order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest educational score first |
| `descending` | Highest educational score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

## Results

Final checkpoint after 1,716 steps. Validation loss is in nats per token (lower is better), overall
and per score bucket. Benchmarks are normalized accuracy, in percent.

| Run | Val loss | [2.5, 3) | [3, 3.5) | [3.5, 4) | [4, 5] | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|---|---|---|---|
| `shuffle` | **4.001** | 4.129 | **4.023** | 3.963 | 3.889 | 25.3 | **56.2** | **34.5** |
| `ascending` | 4.028 | 4.190 | 4.061 | 3.978 | 3.884 | 25.2 | 55.0 | 33.6 |
| `descending` | 4.032 | **4.127** | 4.041 | 4.007 | 3.954 | **25.4** | 55.0 | 32.7 |
| `cooldown` | 4.007 | 4.153 | 4.035 | **3.963** | **3.878** | 25.2 | 56.0 | 33.2 |

The random order has the lowest overall loss, but the gap is at most 0.03 nats. Each run was trained
once, so these differences may be noise. No run is best in every bucket. All four models are close to
chance on the benchmarks, which is expected at this size and token budget.

Training curves and comparison figures are in the
[Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-curriculum-learning) and
in `runs/figures/`.

## Model

| | |
|---|---|
| Architecture | GPT-2: 6 layers, width 512, 8 heads, context 1,024 |
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
| Hardware | One NVIDIA RTX A4500 Laptop GPU |

## Data

The training pool is a shuffled slice of FineWeb-Edu `sample-10BT`, about 1.3 of its 14 files, drawn
through a 10,000-document shuffle buffer. Every document in the sample scores at least 2.5, so the
threshold removes none of them. The score mix, measured over the whole sample (9,672,101 documents,
9.97B GPT-2 tokens):

| Score bucket | Share of documents | Share of tokens |
|---|---|---|
| [2.5, 3.0) | 57.2% | 56.0% |
| [3.0, 3.5) | 29.5% | 30.0% |
| [3.5, 4.0) | 11.2% | 11.9% |
| [4.0, 5.0] | 2.1% | 2.1% |

Each file's shares lie within 0.4 percentage points of this table. The pool's own mix was not counted,
because its scores are not stored when the data is built.

## Evaluation

Every 50 steps, and after the last step:

- **Validation loss** on 2M held-out tokens, overall and per score bucket: [2.5, 3.0), [3.0, 3.5),
  [3.5, 4.0), [4.0, 5.0]. Each bucket holds 500k tokens.
- **Benchmarks**, normalized accuracy: HellaSwag (full validation set), PIQA (validation) and
  ARC-Easy (test).

## Files

Each run has a folder `runs/<order>-seed42/`. `<order>` is the document order, and `seed42` is the
training seed. All four runs use seed 42.

- `model.pt`: final weights (`state_dict`).
- `config.json`: training and model settings.
- `metrics.jsonl`: every logged training, validation and benchmark record.
- `checkpoint.pt`: latest training checkpoint (weights, optimizer state and data position), saved
  every 100 steps. It can be earlier than the final step, so use `model.pt` for the final weights.

`runs/figures/` holds the comparison plots: `validation_loss.png`, `score_buckets.png`,
`benchmarks.png` and `training_loss.png`.

## Usage

The model is a plain PyTorch module. Its code is in the
[GitHub repository](https://github.com/NoeFlandre/gpt2-fineweb-edu-curriculum-learning), in
`gpt2_scratch/model/`.

```python
import torch
from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.model.gpt2 import GPT2

model = GPT2(STUDY_CONFIG)
model.load_state_dict(torch.load("runs/shuffle-seed42/model.pt", map_location="cpu"))
model.eval()
```

## Limitations

- **Base model only.** It is a pretraining model, not instruction-tuned, and it is not meant for chat
  or question answering.
- **One seed per run.** Small differences between runs may be noise.
- **Small scale.** At 45M parameters and 0.9B tokens, benchmark scores sit near chance.
- **Same source.** The validation data comes from the same FineWeb-Edu stream as the training data,
  so it tests generalization within that slice, not to other web text.
- **Classifier-based quality.** The score buckets reflect the educational classifier's judgement,
  not an independent measure of quality.
- **English only.**
