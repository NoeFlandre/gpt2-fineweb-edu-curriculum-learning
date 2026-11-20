# GPT-2 on FineWeb-Edu: does the order of documents matter?

A 45M-parameter GPT-2, trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Four orders of the same 900M-token pool are compared, with documents ordered by their educational score.

**Status:** training. The four arms run one after another; the results appear as each arm finishes.

**Results:** the live [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/gpt2-fineweb-edu-ordering-trackio) and the [model repository](https://huggingface.co/NoeFlandre/gpt2-fineweb-edu-ordering) with checkpoints and the model card. Both are published by the first run.

## The experiment

| Arm | Document order |
|---|---|
| `shuffle` | Random |
| `ascending` | Lowest score first |
| `descending` | Highest score first |
| `cooldown` | Lowest-scoring 80% of tokens (shuffled), then the highest-scoring 20% (shuffled) |

All arms share the same tokens, model, optimizer, schedule and seed. The loss is the plain mean, so only the order differs.

- **Model:** 6 layers, width 512, 8 heads, context 1024, tied embeddings (about 45M parameters)
- **Budget:** 900M tokens per arm (1,716 steps of 524,288 tokens)
- **Primary metric:** validation loss per FineWeb-Edu score bucket: [2.5, 3), [3, 3.5), [3.5, 4), [4, 5]
- **Benchmarks:** HellaSwag, PIQA and ARC-Easy, normalized accuracy, every 50 steps

## The pool

The pool is a slice of FineWeb-Edu `sample-10BT`, shuffled through a 10,000-document buffer. Every document in the sample already scores at least 2.5, so the threshold removes none of them. Its score mix, measured over the whole sample (9,672,101 documents, 9.97B GPT-2 tokens):

| Score bucket | Share of documents | Share of tokens |
|---|---|---|
| [2.5, 3.0) | 57.2% | 56.0% |
| [3.0, 3.5) | 29.5% | 30.0% |
| [3.5, 4.0) | 11.2% | 11.9% |
| [4.0, 5.0] | 2.1% | 2.1% |

The pool holds 871,635 documents, roughly 1.3 of the sample's 14 files. Each file's shares lie within 0.4 percentage points of the table. The pool's exact mix was not counted, because its scores are not stored when the data is built.

Method and limitations: [docs/experiment.md](docs/experiment.md).

## Quick start

```bash
uv sync
uv run pytest
bash scripts/run_experiment.sh   # needs a CUDA GPU; uploads the results to the Hub
```

The steps are also available one at a time: `scripts/prepare_data.py`, `scripts/prepare_benchmarks.py`, `scripts/train.py --arm <name>`, `scripts/report.py` and `scripts/upload.py`.

## Repository

- `gpt2_scratch/`: model, data, orderings, evaluation and training
- `scripts/`: entry points
- `tests/`: test suite
- `docs/`: method, usage and architecture

## Requirements

Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU with 16 GB of memory. The study was designed for a single RTX A4500 laptop GPU.

## Acknowledgements

The GPT-2 implementation follows Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU).

## License

MIT. See [LICENSE](LICENSE).
