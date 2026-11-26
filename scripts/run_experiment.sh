#!/usr/bin/env bash
# Build one seed's study data, train the four orderings one after another, then plot and upload.
# Run from the repository root. SEED defaults to 42, the study's seed.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-42}
data="data/ordering-seed$SEED"
runs="runs/seed$SEED"

if [ ! -f "$data/validation/val_000000.npy" ]; then
  uv run python -m scripts.prepare_data --out "$data" --seed "$SEED"
fi
if [ ! -f data/benchmarks/piqa.jsonl ]; then
  uv run python -m scripts.prepare_benchmarks --out data/benchmarks
fi
for arm in shuffle ascending descending cooldown; do
  uv run python -m scripts.train --arm "$arm" --seed "$SEED" --data "$data" --benchmarks data/benchmarks --out "$runs/$arm"
done
uv run python -m scripts.report --runs "$runs" --out "$runs/figures"
uv run python -m scripts.upload
