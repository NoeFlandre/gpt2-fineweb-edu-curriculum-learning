#!/usr/bin/env bash
# CPU job, once per site and seed: build the training environment and that seed's study data on the
# site's NFS home. Submitted by launch.sh. It does nothing when READY already matches the code, so
# rerunning is safe.
#
# Usage: bash scripts/g5k/prepare_site.sh SEED
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh
load_hf_token

seed=$1
data="data/ordering-seed$seed"
ready="$GPT2_HOME/READY-seed$seed"

fp=$(fingerprint)
if [ "$(cat "$ready" 2>/dev/null || true)" = "$fp" ]; then
  echo "environment and seed $seed data already match this code ($fp)"
  exit 0
fi

# Build on the node's local disk, then pack the environment onto NFS: NFS is slow for thousands
# of small files, and the packed archive is what the GPU jobs unpack.
scratch="$(pick_scratch 15)/gpt2-prep-${OAR_JOB_ID:-$$}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
export UV_PROJECT_ENVIRONMENT="$scratch/venv" UV_CACHE_DIR="$scratch/uv-cache"
export UV_PYTHON_PREFERENCE=only-system HF_HOME="$scratch/hf"
uv sync --frozen --no-dev

mkdir -p "$GPT2_HOME/env"
tar -C "$scratch" -cf - venv | zstd -q -T0 -f -o "$GPT2_HOME/env/venv.tar.zst.partial"
mv "$GPT2_HOME/env/venv.tar.zst.partial" "$GPT2_HOME/env/venv.tar.zst"

rm -rf "$data"
"$scratch/venv/bin/python" -m scripts.prepare_data --out "$data" --seed "$seed"
printf '%s\n' "$fp" > "$ready"
echo "environment and seed $seed data ready ($fp)"
