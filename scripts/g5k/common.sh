# Helpers shared by the Grid'5000 scripts. Sourced, not run.
#
# Per-site state lives in $GPT2_HOME, outside the repository, so that uploaded runs stay clean:
#   env/venv.tar.zst    the packed training environment (built once per site)
#   READY-seed<N>       fingerprint of the code and lockfile the environment and seed N data match
#   state/<arm>-seed<N>.json    chain progress, read by chain.py
#   state/<arm>-seed<N>.next    job id of the successor queued by the last link of a chain
#   logs/               OAR stdout and stderr

export PATH="$HOME/.local/bin:$PATH"
GPT2_HOME="$HOME/gpt2-g5k"

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

# The Hugging Face token, for trackio and the dataset stream. Keep it in ~/.hf_token, mode 600.
load_hf_token() {
  if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN
    HF_TOKEN=$(cat "$HOME/.hf_token")
  fi
}
