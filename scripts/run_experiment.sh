#!/usr/bin/env bash
# Build the study data, train the four orderings for one training seed, then plot and upload.
# Run from the repository root. SEED sets the model initialization (default 42); the data is fixed.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-42}
run=$(uv run python -c "from gpt2_scratch.experiment.naming import run_id; print(run_id($SEED))")
runs="runs/$run"

if [ ! -f data/ordering/validation/val_000000.npy ]; then
  uv run python -m scripts.prepare_data --out data/ordering
fi
if [ ! -f data/benchmarks/piqa.jsonl ]; then
  uv run python -m scripts.prepare_benchmarks --out data/benchmarks
fi
for arm in shuffle ascending descending cooldown; do
  uv run python -m scripts.train --arm "$arm" --seed "$SEED" --data data/ordering --benchmarks data/benchmarks --out "$runs/$arm"
done
uv run python -m scripts.report --runs "$runs" --out "$runs/figures"
uv run python -m scripts.upload
