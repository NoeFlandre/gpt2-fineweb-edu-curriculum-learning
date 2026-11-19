#!/usr/bin/env bash
# Build the data, train the arms, then plot them. Run from the repository root.
# The hard arm is held back for now; add it to the list to train it.
set -euo pipefail

if [ ! -f data/arms/raw/train_000000.npy ]; then
  uv run python -m scripts.prepare_data --out data/arms
fi
for arm in uniform soft; do
  uv run python -m scripts.train --arm "$arm" --data data/arms --out "runs/$arm"
done
uv run python -m scripts.report --runs runs --out runs/figures
uv run python -m scripts.upload
