# Usage

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with at least 16 GB of memory (measured on an RTX A4500 Laptop GPU)
- About 9 GB of free disk space for the data and the final weights; checkpoints are not kept locally

## Seeds

The seed sets the data pool, the four document orders and the model initialization. Each seed has its
own folders: `data/ordering-seed<N>/` for the data and `runs/seed<N>/<order>/` for the runs. The study uses
seed 42, and `scripts/run_experiment.sh` takes `SEED` (default 42).

## Build the data

```bash
uv run python -m scripts.prepare_data --out data/ordering-seed42 --seed 42
uv run python -m scripts.prepare_benchmarks --out data/benchmarks
```

`prepare_data` streams FineWeb-Edu and writes the training pool in each of the four orders under
`data/ordering-seed42/arms/`, and the held-out validation split with its score buckets under
`data/ordering-seed42/validation/` (`buckets.json` reports the tokens per bucket).

## Train one arm

```bash
uv run python -m scripts.train --arm cooldown --seed 42 --data data/ordering-seed42 --out runs/seed42/cooldown
```

Writes `metrics.jsonl`, `model.pt` and `config.json` into `runs/seed42/cooldown`. Each checkpoint is uploaded
to the model repository and its local copy deleted; pass `--no-hub` to keep checkpoints local
instead.

## Plot and upload

```bash
uv run python -m scripts.report --runs runs/seed42 --out runs/seed42/figures
uv run python -m scripts.upload --runs runs
```

## Run everything

```bash
bash scripts/run_experiment.sh
```

## Run on Grid'5000

The same study can run on Grid'5000's `abaca` queue, one chain of GPU jobs per arm. See [Grid'5000](grid5000.md).

## Tests and quality gates

```bash
uv run pytest --cov=gpt2_scratch --cov-branch   # coverage
uv run ruff check && uv run ruff format --check  # lint and format
uv run ty check                                  # types
uv run mutmut run                                # mutation testing
uv run mkdocs build --strict                     # documentation
```
