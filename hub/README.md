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
> `runs/lr6e-4-cosine_params45.2M_tokens899.7M_init<N>_data42/<order>/`. Earlier results were withdrawn: 16 validation documents had leaked
> into the seed-42 training pool, so all 24 runs were retrained on the corrected setup.

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
| `shuffle` | 4.013 ± 0.006 | 4.141 ± 0.007 | **4.035 ± 0.007** | 3.976 ± 0.006 | 3.900 ± 0.006 | **25.5 ± 0.2** | **56.1 ± 0.6** | **34.2 ± 1.2** |
| `ascending` | 4.022 ± 0.008 | 4.183 ± 0.007 | 4.055 ± 0.008 | 3.972 ± 0.008 | **3.876 ± 0.007** | 25.1 ± 0.3 | 55.5 ± 0.6 | 33.1 ± 0.5 |
| `descending` | 4.036 ± 0.013 | **4.130 ± 0.013** | 4.046 ± 0.013 | 4.012 ± 0.013 | 3.957 ± 0.012 | 25.3 ± 0.1 | 55.9 ± 0.3 | 33.1 ± 0.6 |
| `cooldown` | **4.010 ± 0.010** | 4.156 ± 0.010 | 4.038 ± 0.010 | **3.967 ± 0.009** | 3.881 ± 0.010 | 25.1 ± 0.2 | 55.9 ± 0.3 | 33.9 ± 0.4 |

Validation loss per seed:

| Order | 42 | 43 | 44 | 45 | 46 | 47 | Mean |
|---|---|---|---|---|---|---|---|
| `shuffle` | 4.004 | 4.013 | 4.013 | 4.010 | 4.014 | 4.024 | 4.013 |
| `ascending` | 4.022 | 4.036 | 4.013 | 4.017 | 4.020 | 4.020 | 4.022 |
| `descending` | 4.039 | 4.012 | 4.036 | 4.037 | 4.044 | 4.048 | 4.036 |
| `cooldown` | 4.010 | 4.003 | 4.030 | 4.007 | 4.007 | 4.006 | 4.010 |

- **Cooldown and shuffle are tied for the lowest mean validation loss** (4.010 and 4.013). Cooldown is lower in four seeds and shuffle in two, which is within seed-to-seed noise.
- **Shuffle beats descending in five of six seeds** (mean difference 0.023 nats) and ascending in five (0.009 nats).
- **Descending has the highest loss in five of six seeds.**
- **The benchmarks are close to chance** for all four orders, and their differences are within seed noise. Chance is 25% on HellaSwag, 50% on PIQA and 25% on ARC-Easy.

Training curves are in the
[Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-curriculum-learning),
which covers all 24 runs. Comparison plots for the rerun are not generated yet.

## Model

| | |
|---|---|
| Architecture | GPT-2: 6 layers, width 512, 8 heads, context 1,024 |
| Parameters | 45.2M (18.9M excluding embeddings) |
| Vocabulary | GPT-2 BPE, 50,304 (padded); input and output embeddings tied |

## Training

| | |
|---|---|
| Data | FineWeb-Edu `sample-10BT`, shuffled (see Data below). The validation split is the same for every run, and each training pool excludes its documents |
| Budget | 1,716 steps of 524,288 tokens, which is 899.7M tokens per run |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on weight matrices |
| Learning rate | Peak 6e-4, linear warmup over 18 steps, cosine decay to 6e-5 |
| Seeds | 42 to 47. The seed sets the model initialization only; the data seed is 42 |
| Other | Gradient clipping at 1.0, bf16 autocast |
| Hardware | NVIDIA L40S and RTX A5000 GPUs on Grid'5000 (Grenoble) |

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

Each run has a folder `runs/lr6e-4-cosine_params45.2M_tokens899.7M_init<N>_data<D>/<order>/`, named by its learning-rate schedule (`cosine`: decay to 6e-5), peak learning rate (6e-4), parameter count (45.2M), tokens trained (899.7M), initialization seed `<N>` and data seed `<D>`. `<N>` is the initialization seed (42 to 47), `<D>` is
the data seed (42 for all runs so far), and `<order>` is the document order. Each folder holds:

- `model.pt`: final weights (`state_dict`).
- `config.json`: training and model settings.
- `metrics.jsonl`: every logged training, validation and benchmark record.
- `checkpoint.pt`: latest training checkpoint (weights, optimizer state and data position), saved
  every 100 steps. It can be earlier than the final step, so use `model.pt` for the final weights.

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
    "runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42/shuffle/model.pt",
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
  The mean differences between orders are small, at most 0.026 nats.
- **Small scale.** At 45M parameters and 0.9B tokens, benchmark scores sit near chance.
- **Same source.** The validation data comes from the same FineWeb-Edu stream as the training data,
  so it tests generalization within that slice, not to other web text.
- **Classifier-based quality.** The score buckets reflect the educational classifier's judgement,
  not an independent measure of quality.
- **English only.**
