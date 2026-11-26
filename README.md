# GPT-2 curriculum learning on FineWeb-Edu

Does the order in which a language model sees its training documents change what it learns?

We trained four small GPT-2 models (45M parameters) from scratch on the same 900M tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). They differ only in the order of the documents, which is set by each document's educational score.

- **Model and data:** [NoeFlandre/gpt2-fineweb-edu-curriculum-learning](https://huggingface.co/NoeFlandre/gpt2-fineweb-edu-curriculum-learning) on the Hugging Face Hub
- **Training dashboard:** [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-curriculum-learning)
- **Method and limitations:** [docs/experiment.md](docs/experiment.md)

## Results

Final checkpoint after 1,716 steps (899.7M tokens). Validation loss is in nats per token, so lower is better. Benchmarks are normalized accuracy, in percent.

| Order | Val loss | [2.5, 3) | [3, 3.5) | [3.5, 4) | [4, 5] | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|---|---|---|---|
| `shuffle` | **4.001** | 4.129 | **4.023** | 3.963 | 3.889 | 25.3 | **56.2** | **34.5** |
| `ascending` | 4.028 | 4.190 | 4.061 | 3.978 | 3.884 | 25.2 | 55.0 | 33.6 |
| `descending` | 4.032 | **4.127** | 4.041 | 4.007 | 3.954 | **25.4** | 55.0 | 32.7 |
| `cooldown` | 4.007 | 4.153 | 4.035 | **3.963** | **3.878** | 25.2 | 56.0 | 33.2 |

- The random order has the lowest overall loss, but the gap is at most 0.03 nats. Each order was trained once, so these gaps may be seed noise.
- No order wins every score bucket.
- All four models are close to chance on HellaSwag (25%), and only a little above it on PIQA (50%) and ARC-Easy (25%). That is expected at this size and token budget.

## The four orders

| Order | Document order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest score first |
| `descending` | Highest score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

All four share the same tokens, model, optimizer, schedule and seed. The loss is the plain mean, so only the order differs.

## Setup

- **Model:** GPT-2 with 6 layers, width 512, 8 heads, context 1,024 and tied embeddings (about 45M parameters)
- **Budget:** 900M tokens per arm, 524,288 tokens per step
- **Pool:** a shuffled slice of FineWeb-Edu `sample-10BT`, about 1.3 of its 14 files. Every document scores at least 2.5.
- **Primary metric:** validation loss on 2M held-out tokens, reported overall and per score bucket
- **Benchmarks:** HellaSwag, PIQA and ARC-Easy, every 50 steps

Full details, including the score mix of the pool and the cooldown split, are in [docs/experiment.md](docs/experiment.md).

## Quick start

```bash
uv sync
uv run pytest
bash scripts/run_experiment.sh   # needs a CUDA GPU and a Hugging Face token with write access
```

The steps can also be run one at a time: `scripts/prepare_data.py`, `scripts/prepare_benchmarks.py`, `scripts/train.py --arm <name>`, `scripts/report.py` and `scripts/upload.py`. See [docs/usage.md](docs/usage.md).

## Repository

- `gpt2_scratch/`: model, data, orders, evaluation, training and tracking
- `scripts/`: entry points
- `tests/`: test suite
- `docs/`: method, usage and architecture
- `data/benchmarks/`: HellaSwag, PIQA and ARC-Easy examples

## Requirements

Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU with 16 GB of memory. The study was run on a single RTX A4500 laptop GPU.

## Acknowledgements

The GPT-2 implementation follows Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU). The training data is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and the scores come from its educational classifier.

## License

MIT. See [LICENSE](LICENSE).
