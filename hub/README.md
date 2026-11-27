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

![Four document orders feeding GPT-2 45M, with a falling validation-loss curve](https://huggingface.co/NoeFlandre/gpt2-fineweb-edu-curriculum-learning/resolve/main/assets/hero.svg)

# GPT-2 45M: curriculum learning on FineWeb-Edu

Four small GPT-2 models, trained from scratch on the same 900M tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). They differ only in the
order in which documents are seen, which is set by each document's educational score. Each order was
trained with six seeds, 42 to 47.

> **Status:** training complete. All 24 runs (4 orders x 6 seeds) are in this repository, in
> `runs/init<N>-data42/<order>/`.

| Run | Order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest educational score first |
| `descending` | Highest educational score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

## Results

Final checkpoint after 1,716 steps, as mean ± standard deviation over the six seeds. Validation loss is
in nats per token (lower is better), overall and per score bucket. Benchmarks are normalized accuracy,
in percent. Bold marks the best mean in each column.

| Order | Val loss | [2.5, 3) | [3, 3.5) | [3.5, 4) | [4, 5] | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|---|---|---|---|
| `shuffle` | **4.006 ± 0.008** | 4.134 ± 0.008 | **4.028 ± 0.007** | 3.969 ± 0.008 | 3.893 ± 0.008 | 25.3 ± 0.2 | 56.1 ± 0.3 | 33.8 ± 0.9 |
| `ascending` | 4.024 ± 0.008 | 4.187 ± 0.007 | 4.058 ± 0.008 | 3.975 ± 0.008 | **3.879 ± 0.007** | 25.1 ± 0.2 | 55.5 ± 0.4 | 33.5 ± 0.6 |
| `descending` | 4.033 ± 0.010 | **4.128 ± 0.011** | 4.043 ± 0.010 | 4.009 ± 0.010 | 3.954 ± 0.011 | **25.4 ± 0.1** | **56.3 ± 0.9** | 33.0 ± 0.2 |
| `cooldown` | 4.013 ± 0.006 | 4.158 ± 0.006 | 4.040 ± 0.007 | **3.969 ± 0.006** | 3.883 ± 0.006 | 25.2 ± 0.2 | 56.0 ± 0.2 | **33.9 ± 0.9** |

Validation loss per seed:

| Order | 42 | 43 | 44 | 45 | 46 | 47 | Mean |
|---|---|---|---|---|---|---|---|
| `shuffle` | 4.001 | 3.997 | 4.014 | 4.017 | 4.007 | 4.001 | 4.006 |
| `ascending` | 4.028 | 4.032 | 4.032 | 4.015 | 4.017 | 4.022 | 4.024 |
| `descending` | 4.032 | 4.023 | 4.036 | 4.032 | 4.026 | 4.052 | 4.033 |
| `cooldown` | 4.007 | 4.007 | 4.011 | 4.011 | 4.023 | 4.017 | 4.013 |

- **Shuffle has the lowest mean validation loss.** Within each seed it is lower than descending in all
  six seeds (mean difference 0.027 nats) and lower than ascending in five. Its gap to cooldown (0.006
  nats) is within seed-to-seed noise: shuffle is lower in four seeds and cooldown in two.
- **Descending has the highest loss in five of six seeds.**
- **The benchmarks are close to chance** for all four orders, and their differences are within seed
  noise. Chance is 25% on HellaSwag, 50% on PIQA and 25% on ARC-Easy.

Training curves and comparison figures are in the
[Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-curriculum-learning),
which covers all 24 runs, and in `runs/init42-data42/figures/` for seed 42.

## Model

| | |
|---|---|
| Architecture | GPT-2: 6 layers, width 512, 8 heads, context 1,024 |
| Parameters | 45.2M (18.9M excluding embeddings) |
| Vocabulary | GPT-2 BPE, 50,304 (padded); input and output embeddings tied |

## Training

| | |
|---|---|
| Data | FineWeb-Edu `sample-10BT`, shuffled (see Data below). The pool, the orders and the validation split are the same for every seed |
| Budget | 1,716 steps of 524,288 tokens, which is 899.7M tokens per run |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on weight matrices |
| Learning rate | Peak 6e-4, linear warmup over 18 steps, cosine decay to 6e-5 |
| Seeds | 42 to 47. The seed sets the model initialization only; the data is fixed at seed 42 |
| Other | Gradient clipping at 1.0, bf16 autocast |
| Hardware | Seed 42: one NVIDIA RTX A4500 Laptop GPU. Seeds 43 to 47: NVIDIA L40S and RTX A5000 GPUs on Grid'5000 |

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

Each run has a folder `runs/init<N>-data<D>/<order>/`. `<N>` is the initialization seed (42 to 47), `<D>` is
the data seed (42 for all runs so far), and `<order>` is the document order. Each folder holds:

- `model.pt`: final weights (`state_dict`).
- `config.json`: training and model settings.
- `metrics.jsonl`: every logged training, validation and benchmark record.
- `checkpoint.pt`: latest training checkpoint (weights, optimizer state and data position), saved
  every 100 steps. It can be earlier than the final step, so use `model.pt` for the final weights.

`runs/init42-data42/figures/` holds the comparison plots for seed 42: `validation_loss.png`,
`score_buckets.png`, `benchmarks.png` and `training_loss.png`. The other seeds have no figures yet.

## Usage

The model is a plain PyTorch module. Its code is in the
[GitHub repository](https://github.com/NoeFlandre/gpt2-fineweb-edu-curriculum-learning), in
`gpt2_scratch/model/`.

```python
import torch
from huggingface_hub import hf_hub_download
from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.model.gpt2 import GPT2

path = hf_hub_download(
    "NoeFlandre/gpt2-fineweb-edu-curriculum-learning",
    "runs/init42-data42/shuffle/model.pt",
)
model = GPT2(STUDY_CONFIG)
model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()
```

## Limitations

- **Base model only.** It is a pretraining model, not instruction-tuned, and it is not meant for chat
  or question answering.
- **The seed varies only the initialization.** The spread across seeds measures initialization noise.
  The data and its order are the same in every run, so variance from the data order is not measured.
  The mean differences between orders are small, from 0.006 to 0.027 nats.
- **Small scale.** At 45M parameters and 0.9B tokens, benchmark scores sit near chance.
- **Same source.** The validation data comes from the same FineWeb-Edu stream as the training data,
  so it tests generalization within that slice, not to other web text.
- **Classifier-based quality.** The score buckets reflect the educational classifier's judgement,
  not an independent measure of quality.
- **English only.**
