#!/usr/bin/env bash
# Submit the queued training chains where a cluster has room. The queue is ~/gpt2-g5k/queue.txt, one
# "SEED ARM [SCHEDULE]" per line in order (SCHEDULE defaults to cosine); a line leaves the queue once its chain is submitted. A cluster keeps at
# most the chains in limit below: its GPUs plus one waiting. Run on a frontend, from the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/g5k/common.sh

queue="$GPT2_HOME/queue.txt"
declare -A limit=([kinovis]=3 [vercors9]=2)
for cluster in kinovis vercors9; do
  [ -s "$queue" ] || break
  room=$(( limit[$cluster] - $(python3 -m scripts.g5k.status --chains "$cluster") ))
  while [ "$room" -gt 0 ] && [ -s "$queue" ]; do
    read -r seed arm schedule < <(head -1 "$queue")
    if QUEUE=besteffort SEED=$seed SCHEDULE=${schedule:-cosine} bash scripts/g5k/launch.sh "$cluster" "$arm"; then
      sed -i '1d' "$queue"
      room=$(( room - 1 ))
    else
      echo "could not submit $arm seed $seed on $cluster; it stays first in the queue" >&2
      break
    fi
  done
done
echo "chains left to submit: $(wc -l < "$queue" | tr -d ' ')"
