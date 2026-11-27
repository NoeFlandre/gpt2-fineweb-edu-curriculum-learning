#!/usr/bin/env bash
# Start the training chains of one or two arms, for one seed, on one cluster of this site. Run on the
# frontend, from the repository root, after transfer.sh pushed the code.
#
# Usage: SEED=43 [DATA_SEED=42] bash scripts/g5k/launch.sh CLUSTER ARM [ARM]
#
# If the site's environment and that seed's data do not match this code, first queue a CPU job that
# rebuilds them. Then queue one training chain per arm on the cluster; each chain starts after that job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh

seed=${SEED:?set SEED, for example SEED=43}
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a number" >&2; exit 2; }
data_seed=${DATA_SEED:-42}
[[ "$data_seed" =~ ^[0-9]+$ ]] || { echo "DATA_SEED must be a number" >&2; exit 2; }
rid=$(run_id "$seed" "$data_seed")
WALLTIME=${WALLTIME:-24:00}
QUEUE=${QUEUE:-abaca}  # abaca, or besteffort where this account has no production access
[[ "$QUEUE" = abaca || "$QUEUE" = besteffort ]] || { echo "QUEUE must be abaca or besteffort" >&2; exit 2; }
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: SEED=43 bash scripts/g5k/launch.sh CLUSTER ARM [ARM]" >&2
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
if [ "$QUEUE" = abaca ] && ! grep -q "^$cluster " <<<"$usable"; then
  echo "$cluster is not a usable abaca cluster at $site:" >&2
  echo "$usable" >&2
  exit 1
fi

# One environment job at a time per site: it rewrites the shared environment archive.
busy_prefixes=(gpt2-prep-)
for arm in "${arms[@]}"; do
  busy_prefixes+=("gpt2-$arm-$rid-")
done
if busy=$(python3 -m scripts.g5k.status --busy "${busy_prefixes[@]}"); then
  :
else
  echo "already running (or oarstat failed); cancel these or wait for them:" >&2
  echo "$busy" >&2
  exit 1
fi

mkdir -p "$GPT2_HOME/logs" "$GPT2_HOME/state"
prep=""
if [ "$(cat "$GPT2_HOME/READY-data$data_seed" 2>/dev/null || true)" != "$(fingerprint)" ]; then
  prep=$(submit_job "gpt2-prep-$cluster" abaca "gpu_count=0 AND memnode>=131072" "core=8,walltime=12:00" \
    "bash scripts/g5k/prepare_site.sh $data_seed")
  echo "environment and data job: $prep"
fi

for arm in "${arms[@]}"; do
  rm -f "$GPT2_HOME/state/$arm-$rid.json" "$GPT2_HOME/state/$arm-$rid.next"
  dependency=()
  if [ -n "$prep" ]; then
    dependency=(-a "$prep")
  fi
  head=$(submit_job "gpt2-$arm-$rid-$cluster" "$QUEUE" "cluster='$cluster'" "gpu=1,walltime=$WALLTIME" \
    "bash scripts/g5k/train_arm.sh $arm $cluster $WALLTIME $seed $data_seed $QUEUE" ${dependency[@]+"${dependency[@]}"})
  echo "$arm seed $seed on $cluster: job $head"
done
