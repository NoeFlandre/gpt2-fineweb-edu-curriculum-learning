#!/usr/bin/env bash
# Build the study's data, train the four orderings one after another, then plot and upload.
# Run from the repository root.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f data/ordering/validation/val_000000.npy ]; then
  uv run python -m scripts.prepare_data --out data/ordering
fi
if [ ! -f data/benchmarks/piqa.jsonl ]; then
  uv run python -m scripts.prepare_benchmarks --out data/benchmarks
fi
for arm in shuffle ascending descending cooldown; do
  uv run python -m scripts.train --arm "$arm" --data data/ordering --benchmarks data/benchmarks --out "runs/$arm"
done
uv run python -m scripts.report --runs runs --out runs/figures
uv run python -m scripts.upload
