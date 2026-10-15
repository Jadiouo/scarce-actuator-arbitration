#!/usr/bin/env bash
# E1 v2 launcher for the school machine (single GPU program; never concurrent).
#   bash scripts/e1_v2_launch.sh <mode>
#   audit | freeze | dry-run | status      CPU only (freeze = S0: register manifest; commit it before the next modes)
#   smoke                                  tiny budget, DT/DS/AT/AS subset, never touches the test block (uses results/direction2/e1v2_smoke)
#   s1 | s2 | sx | select | all            resumable production stages (all = s1,s2,sx,select; stops before test)
#   testcheck                              GPU pre-check of the test code path on AS seeds (needs the committed freeze_record.json); no graph fallback
#   test                                   one-time final evaluation (needs the committed freeze_record.json and a passing testcheck.json); no graph fallback, no retry
# Exit 75 = 18 h soft cap reached at a save point: this launcher relaunches the same mode by itself (at most 30 times, one log line each).
# Other failures are retried at most 3 times (never test/testcheck), never skipping failed work.
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
  smoke|s1|s2|sx|select|all|testcheck|test) ;;
  *) echo "usage: $0 audit|freeze|dry-run|status|smoke|s1|s2|sx|select|all|testcheck|test"; exit 2 ;;
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
  testcheck) args=(testcheck --backup --graph) ;;     # no --graph-fallback: a graph error must stop the run
  test) args=(test --backup --graph) ;;
  *)    args=(run --stage "$mode" --backup --graph --graph-fallback) ;;
esac
log="results/direction2/runlogs/e1_v2_${mode}.log"
maxtry=3; case "$mode" in test|testcheck) maxtry=1 ;; esac    # never silently repeat the one-shot test
resumes=0
while :; do
  rc=0
  for attempt in $(seq 1 "$maxtry"); do
    echo "E1 v2 $mode attempt $attempt"
    timeout --signal=TERM --kill-after=60s 19h "$PY" scripts/e1_v2.py "${args[@]}" "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || [ "$rc" -eq 124 ]; then break; fi
    if [ "$attempt" -lt "$maxtry" ]; then sleep 20; if busy; then exit 3; fi; fi
  done
  if [ "$rc" -eq 75 ] && [ "$mode" != smoke ] && [ "$mode" != test ] && [ "$mode" != testcheck ]; then
    resumes=$((resumes+1))
    if [ "$resumes" -gt 30 ]; then echo "too many 18 h resumes; stopping" | tee -a "$log"; exit 75; fi
    echo "$(date -Is) E1 v2 $mode: exit 75 (18 h cap), auto-resume #$resumes" | tee -a "$log"
    sleep 30; if busy; then echo "GPU occupied by another program; stopping auto-resume" | tee -a "$log"; exit 3; fi
    continue
  fi
  exit "$rc"
done
