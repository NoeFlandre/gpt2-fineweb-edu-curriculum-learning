#!/usr/bin/env bash
# Start the training chains of one or two arms on one cluster of this site. Run on the frontend,
# from the repository root, after transfer.sh pushed the code.
#
# Usage: bash scripts/g5k/launch.sh CLUSTER ARM [ARM]
#
# If the site's environment and data do not match this code, first queue a CPU job that rebuilds
# them. Then queue one training chain per arm on the cluster; each chain starts after that job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh

WALLTIME=${WALLTIME:-24:00}
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: bash scripts/g5k/launch.sh CLUSTER ARM [ARM]" >&2
  exit 2
fi
cluster=$1
shift
arms=("$@")
for arm in "${arms[@]}"; do
  case "$arm" in
    shuffle | ascending | descending | cooldown) ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 2
      ;;
  esac
done
if [ "$(printf '%s\n' "${arms[@]}" | sort -u | wc -l)" -ne "${#arms[@]}" ]; then
  echo "each arm may be given once" >&2
  exit 2
fi
command -v oarsub >/dev/null || { echo "run this on a frontend" >&2; exit 1; }
command -v uv >/dev/null || { echo "install uv on this site first (docs/grid5000.md)" >&2; exit 1; }

site=$(site_of_host)
usable=$(python3 scripts/g5k/clusters.py --site "$site")
if ! grep -q "^$cluster " <<<"$usable"; then
  echo "$cluster is not a usable abaca cluster at $site:" >&2
  echo "$usable" >&2
  exit 1
fi

busy_prefixes=(gpt2-prep-)
for arm in "${arms[@]}"; do
  busy_prefixes+=("gpt2-$arm-")
done
if busy=$(python3 scripts/g5k/status.py --busy "${busy_prefixes[@]}"); then
  :
else
  echo "already running (or oarstat failed); cancel these or wait for them:" >&2
  echo "$busy" >&2
  exit 1
fi

mkdir -p "$GPT2_HOME/logs" "$GPT2_HOME/state"
prep=""
if [ "$(cat "$GPT2_HOME/READY" 2>/dev/null || true)" != "$(fingerprint)" ]; then
  prep=$(oarsub -q abaca -p gpu_count=0 -l core=4,walltime=12:00 -n "gpt2-prep-$cluster" \
    -d "$PWD" -O "$GPT2_HOME/logs/%jobid%.out" -E "$GPT2_HOME/logs/%jobid%.err" \
    "bash scripts/g5k/prepare_site.sh" | sed -n 's/^OAR_JOB_ID=//p')
  echo "environment and data job: $prep"
fi

for arm in "${arms[@]}"; do
  rm -f "$GPT2_HOME/state/$arm.json" "$GPT2_HOME/state/$arm.next"
  dependency=()
  if [ -n "$prep" ]; then
    dependency=(-a "$prep")
  fi
  head=$(oarsub "${dependency[@]}" -q abaca -p "cluster='$cluster'" \
    -l "gpu=1,walltime=$WALLTIME" -n "gpt2-$arm-$cluster" -d "$PWD" \
    -O "$GPT2_HOME/logs/%jobid%.out" -E "$GPT2_HOME/logs/%jobid%.err" \
    "bash scripts/g5k/train_arm.sh $arm $cluster $WALLTIME" | sed -n 's/^OAR_JOB_ID=//p')
  echo "$arm on $cluster: job $head"
done
