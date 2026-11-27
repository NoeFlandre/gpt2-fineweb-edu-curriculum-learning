# Helpers shared by the Grid'5000 scripts. Sourced, not run.
#
# Per-site state lives in $GPT2_HOME, outside the repository, so that uploaded runs stay clean:
#   env/venv.tar.zst    the packed training environment (built once per site)
#   READY-data<D>       fingerprint of the code and lockfile that the data of data seed D matches
#   state/<arm>-<run id>.json   chain progress, read by chain.py
#   state/<arm>-<run id>.next   job id of the successor queued by the last link of a chain
#   logs/               OAR stdout and stderr

export PATH="$HOME/.local/bin:$PATH"
GPT2_HOME="$HOME/gpt2-g5k"

# The name of the runs of an initialization seed and a data seed: run_id SEED DATA_SEED, such as
# init43-data42 (gpt2_scratch/experiment/naming.py).
run_id() {
  python3 -c "from gpt2_scratch.experiment.naming import run_id; print(run_id($1, $2))"
}

# Fingerprint of everything the environment and the study data depend on.
fingerprint() {
  cat uv.lock pyproject.toml scripts/prepare_data.py gpt2_scratch/data/*.py \
    gpt2_scratch/experiment/arms.py gpt2_scratch/training/config.py | sha256sum | cut -d' ' -f1
}

# The site of this host: gros-1.nancy.grid5000.fr and nancy.grid5000.fr both give nancy.
site_of_host() {
  hostname -f | awk -F. '{ print (NF > 3 ? $2 : $1) }'
}

# A local directory with at least GB free: /tmp, or /dev/shm when /tmp is small.
pick_scratch() {
  local need=$(( $1 * 1024 * 1024 )) dir avail
  for dir in /tmp /dev/shm; do
    avail=$(df -Pk "$dir" | awk 'NR == 2 { print $4 }')
    if [ "$avail" -ge "$need" ]; then
      echo "$dir"
      return 0
    fi
  done
  echo "less than $1 GB free in /tmp and /dev/shm" >&2
  return 1
}

# Run an OAR command: directly on a frontend, through the site's frontend from a node.
oar() {
  if command -v "$1" >/dev/null 2>&1; then
    "$@"
  else
    ssh "$(site_of_host).grid5000.fr" "$(printf '%q ' "$@")"
  fi
}

# Submit a job and print its id: submit_job NAME QUEUE PROPERTIES RESOURCES COMMAND [oarsub options].
# The job runs from the repository, and its output goes to $GPT2_HOME/logs.
submit_job() {
  local name=$1 queue=$2 properties=$3 resources=$4 command=$5
  shift 5
  oar oarsub "$@" -q "$queue" -p "$properties" -l "$resources" -n "$name" -d "$PWD" \
    -O "$GPT2_HOME/logs/%jobid%.out" -E "$GPT2_HOME/logs/%jobid%.err" "$command" \
    | sed -n 's/^OAR_JOB_ID=//p'
}

# The Hugging Face token, for trackio and the dataset stream. Keep it in ~/.hf_token, mode 600.
load_hf_token() {
  if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN
    HF_TOKEN=$(cat "$HOME/.hf_token")
  fi
}
