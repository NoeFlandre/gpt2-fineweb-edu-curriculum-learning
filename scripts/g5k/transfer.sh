#!/usr/bin/env bash
# Copy the code to a site's frontend. Only code: data and runs are never copied to your laptop, and
# results go to the Hugging Face model repository from the site.
#
# Usage (from the repository root on your laptop):
#   G5K_LOGIN=<login> bash scripts/g5k/transfer.sh SITE
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${G5K_LOGIN:?set G5K_LOGIN to your Grid5000 login}"

site=${1:-}
if [ -z "$site" ]; then
  echo "usage: G5K_LOGIN=<login> bash scripts/g5k/transfer.sh SITE" >&2
  exit 2
fi

rsync -az -e "ssh -J $G5K_LOGIN@access.grid5000.fr" \
  --exclude 'data/data*/' --exclude data/validation/ --exclude runs/ --exclude .venv/ --exclude .git/ \
  --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude .ty/ \
  ./ "$G5K_LOGIN@$site.grid5000.fr:gpt2-fineweb-edu-curriculum-learning/"
