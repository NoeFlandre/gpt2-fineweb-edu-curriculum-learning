#!/usr/bin/env bash
# CPU job, once per site and data seed: build the training environment and the study data for that
# data seed on the site's NFS home. Submitted by launch.sh. It does nothing when READY-data<seed>
# already matches the code, so rerunning is safe.
#
# Usage: bash scripts/g5k/prepare_site.sh [DATA_SEED]    (default 42)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh
load_hf_token

data_seed=${1:-42}
[[ "$data_seed" =~ ^[0-9]+$ ]] || { echo "DATA_SEED must be a number" >&2; exit 2; }
data="data/data$data_seed"
ready="$GPT2_HOME/READY-data$data_seed"

fp=$(fingerprint)
if [ "$(cat "$ready" 2>/dev/null || true)" = "$fp" ]; then
  echo "environment and data for data seed $data_seed already match this code ($fp)"
  exit 0
fi

# Build on the node's local disk, then pack the environment onto NFS: NFS is slow for thousands
# of small files, and the packed archive is what the GPU jobs unpack.
scratch="$(pick_scratch 15)/gpt2-prep-${OAR_JOB_ID:-$$}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
export UV_PROJECT_ENVIRONMENT="$scratch/venv" UV_CACHE_DIR="$scratch/uv-cache"
export UV_PYTHON_PREFERENCE=only-system HF_HOME="$scratch/hf"
# .python-version pins 3.12, which the sites may not have: use the system python3 (3.12 or newer).
uv sync --frozen --no-dev --python "$(command -v python3)"

mkdir -p "$GPT2_HOME/env"
tar -C "$scratch" -cf - venv | zstd -q -T0 -f -o "$GPT2_HOME/env/venv.tar.zst.partial"
mv "$GPT2_HOME/env/venv.tar.zst.partial" "$GPT2_HOME/env/venv.tar.zst"

rm -rf "$data"
"$scratch/venv/bin/python" -m scripts.prepare_data --data-seed "$data_seed" --out "$data"
printf '%s\n' "$fp" > "$ready"
echo "environment and data for data seed $data_seed ready ($fp)"
