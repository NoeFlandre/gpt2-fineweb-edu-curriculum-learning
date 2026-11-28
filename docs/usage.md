# Usage

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with at least 16 GB of memory (measured on an RTX A4500 Laptop GPU)
- About 9 GB of free disk space for the data and the final weights; checkpoints are not kept locally

## Seeds

The seed sets only the model initialization. The data (the pool, the four document orders and the
validation split) is built once with a fixed seed of 42, so every training seed sees the same data. Each
training seed has its own runs folder, `runs/<run>/<order>/`. `scripts/run_experiment.sh` takes `SEED`
(default 42).

## Build the data

```bash
uv run python -m scripts.prepare_data --data-seed 42
uv run python -m scripts.prepare_benchmarks --out data/benchmarks
```

`prepare_data` streams FineWeb-Edu and writes the training pool in each of the four orders under
`data/data<data seed>/arms/`. The held-out validation split is the same for every data seed: it is
built once, from the seed-42 stream, into `data/validation/` (`buckets.json` reports the tokens per
bucket, `doc_keys.txt` lists its documents), and copied into each data folder. A data seed's pool
leaves out the validation documents. `--data-seed` (default 42) sets the pool and the orders.

## Train one arm

```bash
uv run python -m scripts.train --arm cooldown --seed 42 --data-seed 42 --out runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42/cooldown
```

`--seed` sets the model initialization. Its run is named by `run_id`, such as `lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42`.

Writes `metrics.jsonl`, `model.pt` and `config.json` into `runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42/cooldown`. Each checkpoint is uploaded
to the model repository and its local copy deleted; pass `--no-hub` to keep checkpoints local
instead.

## Plot and upload

```bash
uv run python -m scripts.report --runs runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42 --out runs/lr6e-4-cosine_params45.2M_tokens899.7M_init42_data42/figures
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
