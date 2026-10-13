#!/usr/bin/env bash
# One GPU process on the school host. CPU/GPU checks use synthetic/training data;
# test seeds are reached only by the committed production evaluation entry.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p .scratch/main-eval-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/main-eval-tmp"
export MPLCONFIGDIR="$PWD/.scratch/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec 9>.scratch/main-eval-launch.lock
flock -n 9 || { echo 'Another main evaluation launcher is running'; exit 3; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'Another GPU compute process is present; refusing to start'
    exit 3
fi
"$PYTHON" -m pytest tests/direction2/test_d2_main_eval.py -m gpu -q --basetemp=.scratch/main-eval-pytest-gpu
"$PYTHON" scripts/d2_main_eval.py run --graph --backup
