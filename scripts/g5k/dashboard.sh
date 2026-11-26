#!/usr/bin/env bash
# CPU job: rebuild the Trackio dashboard from every run and publish it. The current dashboard files
# are backed up in the bucket first (see scripts/g5k/dashboard.py).
#
# Usage: bash scripts/g5k/dashboard.sh SEED [SEED ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh
load_hf_token

scratch="$(pick_scratch 2)/gpt2-dashboard-${OAR_JOB_ID:-$$}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
zstd -dc "$GPT2_HOME/env/venv.tar.zst" | tar -x -C "$scratch"
# Trackio reads TRACKIO_DIR when it is imported: an empty folder keeps this job's database apart.
TRACKIO_DIR="$scratch/trackio" "$scratch/venv/bin/python" -m scripts.g5k.dashboard --runs runs --seeds "$@"
