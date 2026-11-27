#!/usr/bin/env bash
# Build the study data, train the four orderings for one training seed, then plot and upload.
# Run from the repository root. SEED sets the model initialization, and DATA_SEED the pool, the orders
# and the validation split (both default to 42).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-42}
DATA_SEED=${DATA_SEED:-42}
run=$(uv run python -c "from gpt2_scratch.experiment.naming import run_id; print(run_id($SEED, $DATA_SEED))")
runs="runs/$run"
data="data/data$DATA_SEED"

if [ ! -f "$data/validation/val_000000.npy" ]; then
  uv run python -m scripts.prepare_data --data-seed "$DATA_SEED" --out "$data"
fi
if [ ! -f data/benchmarks/piqa.jsonl ]; then
  uv run python -m scripts.prepare_benchmarks --out data/benchmarks
fi
for arm in shuffle ascending descending cooldown; do
  uv run python -m scripts.train --arm "$arm" --seed "$SEED" --data-seed "$DATA_SEED" --data "$data" --benchmarks data/benchmarks --out "$runs/$arm"
done
uv run python -m scripts.report --runs "$runs" --out "$runs/figures"
uv run python -m scripts.upload
