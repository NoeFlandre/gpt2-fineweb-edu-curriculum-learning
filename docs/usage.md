# Usage

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with at least 16 GB of memory (the training runs were measured on an RTX A4500 Laptop GPU)
- About 40 GB of free disk space for the data, the checkpoints and the Hugging Face cache

## Build the data

```bash
uv run python -m scripts.prepare_data --out data/arms
```

Writes `data/arms/raw/{train,val}_000000.npy` and `data/arms/edu/train_000000.npy`, each with a `.scores.npy` companion holding one score per token.

## Train one arm

```bash
uv run python -m scripts.train --arm soft --data data/arms --out runs/soft
```

Writes `metrics.jsonl`, `model.pt` and `config.json` into `runs/soft`.

## Plot and upload

```bash
uv run python -m scripts.report --runs runs --out runs/figures
uv run python -m scripts.upload --runs runs
```

## Run everything

```bash
bash scripts/run_experiment.sh
```

## Tests and quality gates

```bash
uv run pytest --cov=gpt2_scratch --cov-branch   # coverage
uv run ruff check && uv run ruff format --check  # lint and format
uv run ty check                                  # types
uv run mutmut run                                # mutation testing
uv run mkdocs build --strict                     # documentation
```
