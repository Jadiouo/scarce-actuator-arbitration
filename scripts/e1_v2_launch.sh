#!/usr/bin/env bash
# E1 v2 launcher for the school machine (single GPU program; never concurrent).
#   bash scripts/e1_v2_launch.sh <mode>
#   audit | freeze | dry-run | status      CPU only (freeze = S0: register manifest; commit it before the next modes)
#   smoke                                  tiny budget, DT/DS/AT/AS subset, never touches the test block (uses results/direction2/e1v2_smoke)
#   s1 | s2 | sx | select | all            resumable production stages (all = s1,s2,sx,select; stops before test)
#   test                                   one-time final evaluation (needs the committed freeze_record.json)
# Exit 75 = 18 h soft cap reached at a save point: just relaunch the same mode.  Retries at most 3 times, never skips failed work.
# Extra arguments are passed to scripts/e1_v2.py.  Use PYTHON=/path/to/python to pick the interpreter.
set -u
cd "$(dirname "$0")/.." || exit 3
mode="${1:-}"; shift || true
PY="${PYTHON:-}"
if [ -z "$PY" ]; then if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi; fi
mkdir -p .scratch/e1-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/e1-tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "$mode" in
  audit|freeze|dry-run|status) exec "$PY" scripts/e1_v2.py "$mode" "$@" ;;
  smoke|s1|s2|sx|select|all|test) ;;
  *) echo "usage: $0 audit|freeze|dry-run|status|smoke|s1|s2|sx|select|all|test"; exit 2 ;;
esac
exec 9>.scratch/main-eval-launch.lock
flock -n 9 || { echo 'Another school evaluation launcher is running'; exit 3; }
busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
if busy; then echo 'GPU occupied; refusing to interfere'; exit 3; fi
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest tests/direction2/test_e1_v2.py -q --basetemp=.scratch/e1v2-pytest || exit 4
case "$mode" in
  smoke)
    "$PY" scripts/e1_v2.py freeze --smoke || exit $?
    args=(run --smoke --stage all --graph --graph-fallback) ;;
  test) args=(test --backup --graph --graph-fallback) ;;
  *)    args=(run --stage "$mode" --backup --graph --graph-fallback) ;;
esac
for attempt in 1 2 3; do
  echo "E1 v2 $mode attempt $attempt"
  timeout --signal=TERM --kill-after=60s 19h "$PY" scripts/e1_v2.py "${args[@]}" "$@" 2>&1 | tee -a "results/direction2/runlogs/e1_v2_${mode}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || [ "$rc" -eq 124 ]; then exit "$rc"; fi
  if [ "$attempt" -lt 3 ]; then sleep 20; if busy; then exit 3; fi; fi
done
exit "$rc"
