#!/usr/bin/env bash
# One link of an arm's training chain on one GPU. Queued by launch.sh for the first link, then by each
# link for its successor. The checkpoint stays on the site's disk, so each link resumes from there.
# When training ends, the checkpoint, weights and metrics are published once to the Hugging Face model
# repository, and deleted from the site.
#
# Usage: bash scripts/g5k/train_arm.sh ARM CLUSTER WALLTIME
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh
load_hf_token

arm=$1 cluster=$2 walltime=$3
run="runs/$arm"
state="$GPT2_HOME/state/$arm.json"
successor="$GPT2_HOME/state/$arm.next"
finished="$GPT2_HOME/state/$arm.done"
mkdir -p "$run" "$GPT2_HOME/state" "$GPT2_HOME/logs"

cancel_successor() {
  [ -f "$successor" ] && oar oardel "$(cat "$successor")" >/dev/null 2>&1 || true
}

# Published already: this link is a leftover successor, so stop the chain here.
if [ -f "$finished" ]; then
  cancel_successor
  echo "$arm is already published"
  exit 0
fi

if [ "$(cat "$GPT2_HOME/READY" 2>/dev/null || true)" != "$(fingerprint)" ]; then
  echo "the environment does not match this code: run launch.sh first" >&2
  exit 1
fi

# Work on local disk: the environment is unpacked there, and the arm's shard is copied in.
scratch="$(pick_scratch 10)/gpt2-$arm-${OAR_JOB_ID:-$$}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
zstd -dc "$GPT2_HOME/env/venv.tar.zst" | tar -x -C "$scratch"

# Training left its weights but was not published (a failed upload): publish them, do not retrain.
if [ ! -f "$run/model.pt" ]; then
  verdict=$(python3 scripts/g5k/chain.py begin "$run" "$state")
  if [ "$verdict" = "stop" ]; then
    echo "stopping $arm: its last jobs made no progress" >&2
    exit 1
  fi

  # Queue the successor now, so it starts the moment this link ends, however it ends.
  if [ -n "${OAR_JOB_ID:-}" ]; then
    next=$(oar oarsub -a "$OAR_JOB_ID" -q abaca -p "cluster='$cluster'" \
      -l "gpu=1,walltime=$walltime" -n "gpt2-$arm-$cluster" -d "$PWD" \
      -O "$GPT2_HOME/logs/%jobid%.out" -E "$GPT2_HOME/logs/%jobid%.err" \
      "bash scripts/g5k/train_arm.sh $arm $cluster $walltime" \
      | sed -n 's/^OAR_JOB_ID=//p') || next=""
    if [ -n "$next" ]; then
      echo "$next" > "$successor"
    else
      echo "warning: the successor of $arm could not be queued" >&2
    fi
  fi

  mkdir -p "$scratch/data/arms"
  cp -r data/ordering/validation "$scratch/data/"
  cp -r "data/ordering/arms/$arm" "$scratch/data/arms/"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$scratch/venv/bin/python" -m scripts.train --arm "$arm" --data "$scratch/data" \
    --benchmarks data/benchmarks --out "$run" --no-hub
fi

"$scratch/venv/bin/python" -m scripts.g5k.publish --arm "$arm" --out "$run"
touch "$finished"
cancel_successor
echo "$arm finished and published"
