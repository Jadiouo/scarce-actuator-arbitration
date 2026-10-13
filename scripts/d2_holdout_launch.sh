#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .scratch/holdout-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/holdout-tmp"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>.scratch/main-eval-launch.lock
flock -n 9 || { echo 'Another evaluation launcher is running'; exit 3; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'Another GPU compute process is present; refusing to start'
    exit 3
fi
.venv/bin/python -m pytest tests/direction2/test_d2_holdout.py -q --basetemp=.scratch/holdout-pytest
.venv/bin/python scripts/d2_holdout_eval.py run --backup
