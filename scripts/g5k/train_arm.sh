#!/usr/bin/env bash
# One link of an arm's training chain on one GPU, for one seed. Queued by launch.sh for the first link,
# then by each link for its successor. The checkpoint stays on the site's disk, so each link resumes
# from there. When training ends, the checkpoint, weights and metrics are published once to the
# Hugging Face model repository under runs/seedSEED/ARM, and deleted from the site.
#
# Usage: bash scripts/g5k/train_arm.sh ARM CLUSTER WALLTIME SEED
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh
load_hf_token

arm=$1 cluster=$2 walltime=$3 seed=$4 queue=${5:-abaca}
key="$arm-seed$seed"
run="runs/seed$seed/$arm"
data="data/ordering"
state="$GPT2_HOME/state/$key.json"
successor="$GPT2_HOME/state/$key.next"
finished="$GPT2_HOME/state/$key.done"
mkdir -p "$run" "$GPT2_HOME/state" "$GPT2_HOME/logs"

cancel_successor() {
  [ -f "$successor" ] && oar oardel "$(cat "$successor")" >/dev/null 2>&1 || true
}

# Published already: this link is a leftover successor, so stop the chain here.
if [ -f "$finished" ]; then
  cancel_successor
  echo "$key is already published"
  exit 0
fi

if [ "$(cat "$GPT2_HOME/READY" 2>/dev/null || true)" != "$(fingerprint)" ]; then
  echo "the environment or data does not match this code: run launch.sh first" >&2
  exit 1
fi

# Work on local disk: the environment is unpacked there, and the arm's shard is copied in.
scratch="$(pick_scratch 10)/gpt2-$key-${OAR_JOB_ID:-$$}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
zstd -dc "$GPT2_HOME/env/venv.tar.zst" | tar -x -C "$scratch"

# Training left its weights but was not published (a failed upload): publish them, do not retrain.
if [ ! -f "$run/model.pt" ]; then
  verdict=$(python3 -m scripts.g5k.chain begin "$run" "$state")
  if [ "$verdict" = "stop" ]; then
    echo "stopping $key: its last jobs made no progress" >&2
    exit 1
  fi

  # Queue the successor now, so it starts the moment this link ends, however it ends.
  if [ -n "${OAR_JOB_ID:-}" ]; then
    next=$(submit_job "gpt2-$key-$cluster" "$queue" "cluster='$cluster'" "gpu=1,walltime=$walltime" \
      "bash scripts/g5k/train_arm.sh $arm $cluster $walltime $seed $queue" -a "$OAR_JOB_ID") || next=""
    if [ -n "$next" ]; then
      echo "$next" > "$successor"
    else
      echo "warning: the successor of $key could not be queued" >&2
    fi
  fi

  mkdir -p "$scratch/data/arms"
  cp -r "$data/validation" "$scratch/data/"
  cp -r "$data/arms/$arm" "$scratch/data/arms/"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$scratch/venv/bin/python" -m scripts.train --arm "$arm" --seed "$seed" --data "$scratch/data" \
    --benchmarks data/benchmarks --out "$run" --no-hub --no-dashboard
fi

"$scratch/venv/bin/python" -m scripts.g5k.publish --arm "$arm" --seed "$seed" --out "$run"
touch "$finished"
cancel_successor
echo "$key finished and published"
