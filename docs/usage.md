# Usage

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with at least 16 GB of memory (measured on an RTX A4500 Laptop GPU)
- About 10 GB of free disk space for the data, the checkpoints and the Hugging Face cache

## Build the data

```bash
uv run python -m scripts.prepare_data --out data/ordering
uv run python -m scripts.prepare_benchmarks --out data/benchmarks
```

`prepare_data` streams FineWeb-Edu and writes the training pool in each of the four orders under
`data/ordering/arms/`, and the held-out validation split with its score buckets under
`data/ordering/validation/` (`buckets.json` reports the tokens per bucket).

## Train one arm

```bash
uv run python -m scripts.train --arm cooldown --out runs/cooldown
```

Writes `metrics.jsonl`, `model.pt` and `config.json` into `runs/cooldown`. Checkpoints go to the
model repository; pass `--no-hub` to keep them local.

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
