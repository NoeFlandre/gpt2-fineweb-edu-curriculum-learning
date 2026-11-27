# GPT-2 curriculum learning on FineWeb-Edu

Does the order in which a language model sees its training documents change what it learns?

We trained four small GPT-2 models (45M parameters) from scratch on the same 900M tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). They differ only in the order of the documents, which is set by each document's educational score. Each order was trained with six seeds (42 to 47).

- **Model and data:** [NoeFlandre/gpt2-fineweb-edu-curriculum-learning](https://huggingface.co/NoeFlandre/gpt2-fineweb-edu-curriculum-learning) on the Hugging Face Hub
- **Training dashboard:** [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-curriculum-learning)
- **Method and limitations:** [docs/experiment.md](docs/experiment.md)

## Results

Final checkpoint after 1,716 steps (899.7M tokens), as mean ± standard deviation over the six seeds. Validation loss is in nats per token, so lower is better. Benchmarks are normalized accuracy, in percent. Bold marks the best mean in each column.

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

- **Shuffle has the lowest mean validation loss.** Within each seed it is lower than descending in all six seeds (mean difference 0.027 nats) and lower than ascending in five. Its gap to cooldown (0.006 nats) is within seed-to-seed noise.
- **Descending has the highest loss in five of six seeds.**
- **The benchmarks are close to chance** for all four orders, and their differences are within seed noise. Chance is 25% on HellaSwag, 50% on PIQA and 25% on ARC-Easy.

## The four orders

| Order | Document order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest score first |
| `descending` | Highest score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

All four share the same tokens, model, optimizer and schedule. The loss is the plain mean, so only the order differs.

## Setup

- **Model:** GPT-2 with 6 layers, width 512, 8 heads, context 1,024 and tied embeddings (about 45M parameters)
- **Budget:** 900M tokens per arm, 524,288 tokens per step
- **Pool:** a shuffled slice of FineWeb-Edu `sample-10BT`, about 1.3 of its 14 files. Every document scores at least 2.5.
- **Seeds:** 42 to 47. The seed sets the model initialization only; the pool, the orders and the validation split are fixed at seed 42.
- **Primary metric:** validation loss on 2M held-out tokens, reported overall and per score bucket
- **Benchmarks:** HellaSwag, PIQA and ARC-Easy, every 50 steps

Full details, including the score mix of the pool and the cooldown split, are in [docs/experiment.md](docs/experiment.md).

## Quick start

```bash
uv sync
uv run pytest
bash scripts/run_experiment.sh   # needs a CUDA GPU and a Hugging Face token with write access
```

`SEED=43 bash scripts/run_experiment.sh` trains another seed; the default is 42. Each run goes to `runs/init<N>-data<D>/`, where `<N>` is the initialization seed and `<D>` the data seed.

The steps can also be run one at a time: `scripts/prepare_data.py`, `scripts/prepare_benchmarks.py`, `scripts/train.py --arm <name>`, `scripts/report.py` and `scripts/upload.py`. See [docs/usage.md](docs/usage.md).

To run the study on Grid'5000 instead (one GPU per arm, through the `abaca` or `besteffort` queue), see [docs/grid5000.md](docs/grid5000.md).

## Repository

- `gpt2_scratch/`: model, data, orders, evaluation, training and tracking
- `scripts/`: entry points; `scripts/g5k/` runs the study on Grid'5000
- `tests/`: test suite
- `docs/`: method, usage and architecture
- `data/benchmarks/`: HellaSwag, PIQA and ARC-Easy examples

## Requirements

Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU with 16 GB of memory. Seed 42 ran on a single RTX A4500 laptop GPU. Seeds 43 to 47 ran on Grid'5000, on NVIDIA L40S and RTX A5000 GPUs.

## Acknowledgements

The GPT-2 implementation follows Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU). The training data is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and the scores come from its educational classifier.

## License

MIT. See [LICENSE](LICENSE).
