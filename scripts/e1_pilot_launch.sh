#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .scratch/e1-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/e1-tmp"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>.scratch/main-eval-launch.lock
flock -n 9 || { echo 'Another school evaluation launcher is running'; exit 3; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU occupied; refusing to interfere'
    exit 3
fi
.venv/bin/python -m pytest tests/direction2/test_e1_pilot.py -q --basetemp=.scratch/e1-pytest
for attempt in 1 2 3; do
    echo "E1 attempt $attempt"
    set +e
    timeout --signal=TERM --kill-after=60s 19h .venv/bin/python scripts/e1_pilot.py run --backup
    rc=$?
    set -e
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || [ "$rc" -eq 124 ]; then exit "$rc"; fi
    if [ "$attempt" -lt 3 ]; then
        sleep 20
        if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then exit 3; fi
    fi
done
exit "$rc"
