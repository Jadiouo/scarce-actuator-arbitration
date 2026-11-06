#!/usr/bin/env bash
# E-A2 launcher for the shared experiment machine (one GPU program of ours at a time; never compete with another account).
#   bash scripts/ea2_launch.sh <mode> [extra args for scripts/ea2.py]
#   audit | freeze | status | report --tag shared|local | smoke-clean | ledger-reset | cputests     CPU only (freeze = S0: register the manifest; commit it before the next modes)
#   smoke                  single-lane formal-size timing + graph == eager on EA2_TRAIN / EA2_SEL / EA2_AT (smoke dir results/direction2/ea2_smoke; never EA2_TEST)
#   train                  the 3 chains on the ORIGINAL setting (416 generations each, one lane), resumable, one checkpoint per generation
#   select                 per-chain checkpoint, A_fresh_orig, A_O', A_C2', SEL re-evaluations, drift check; writes selection_frozen.json + freeze_record.json
#   adapt                  needs the COMMITTED select: re-adaptation arm (A+ and A++ frozen together in selection_plus_frozen.json); resumable
#   testcheck              graph == eager on the first 64 EA2_SEL seeds for all six policy lists (needs both freezes committed); no retry
#   test --part main|plus|rest   one-time final evaluation in three parts (plus / rest need the committed result.json); no retry, no resume
#   gputests               the cuda tests of tests/direction2/test_ea2.py (EA2_ALLOW_GPU_TESTS=1)
#   bg <mode> [args]       start <mode> in a detached tmux session ea2_<mode> (reattach: tmux attach -t ea2_<mode>)
# Exit 75 = 18 h soft cap or another account's GPU program appeared (stop at a generation boundary): `train` / `adapt` relaunch themselves while the GPU
# is free (at most 30 times, one log line each).  Other failures are retried at most 3 times for train/select/adapt (never smoke/testcheck/test).
# Results are fetched with rsync and committed on the local machine; this script never commits or pushes.
# Use PYTHON=/path/to/python to pick the interpreter (default .venv/bin/python, else python3).
# The deselected E-A1 test runs scripts/ea1_launch.sh, whose frozen free-space check reads the shared root partition; it says nothing about E-A2.
set -u
cd "$(dirname "$0")/.." || exit 3
mode="${1:-}"; shift || true
PY="${PYTHON:-}"
if [ -z "$PY" ]; then if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi; fi
mkdir -p .scratch/ea2-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/ea2-tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cpu_tests() {
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" -m pytest tests/direction2/test_ea2.py tests/direction2/test_ea1.py tests/direction2/test_e1_v2.py --deselect tests/direction2/test_ea1.py::test_launcher_auto_resumes_train_on_75_and_passes_other_codes -q -rfE -p no:cacheprovider --basetemp=.scratch/ea2-pytest
}
case "$mode" in
  audit|freeze|status|report|smoke-clean|ledger-reset) exec "$PY" scripts/ea2.py "$mode" "$@" ;;
  cputests) cpu_tests; exit $? ;;
  bg)
    sub="${1:-}"; shift || true
    [ -n "$sub" ] || { echo "usage: $0 bg <mode> [args]"; exit 2; }
    command -v tmux >/dev/null || { echo 'tmux not found'; exit 3; }
    tmux new-session -d -s "ea2_$sub" "cd '$PWD' && bash scripts/ea2_launch.sh $sub $*; echo \"ea2 $sub finished rc=\$?\"; sleep 60"
    echo "started tmux session ea2_$sub (tmux attach -t ea2_$sub); log: results/direction2/runlogs/ea2_${sub}.log"
    exit 0 ;;
  smoke|train|adapt|select|testcheck|test|gputests) ;;
  *) echo "usage: $0 audit|freeze|status|report|smoke-clean|ledger-reset|cputests|smoke|train|adapt|select|testcheck|test|gputests|bg <mode>"; exit 2 ;;
esac
exec 9>.scratch/ea2-launch.lock
flock -n 9 || { echo 'Another E-A2 launcher is running'; exit 3; }
# busy = a GPU program of ANOTHER account (or one whose owner cannot be read); our own account's programs are allowed in any number
busy() {
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$(id -u)" ] || return 0
  done
  return 1
}
# own_busy = a GPU program of THIS account is running (matters only for testcheck / test: the one-shot ledger must not be put at risk)
own_busy() {
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$(id -u)" ] && return 0
  done
  return 1
}
# The project directory is a link to a dedicated space: look at the space the physical path lives on, not at the root partition.
space_ok() {
  local real mnt avail
  real=$(pwd -P)
  if [ -n "${EA2_SPACE_OVERRIDE:-}" ]; then      # recorded override: the same two checks, on the path the user names instead of the project directory
    real="$EA2_SPACE_OVERRIDE"
    echo "space check path overridden by EA2_SPACE_OVERRIDE" | tee -a results/direction2/runlogs/ea2_${mode}.log
  fi
  mnt=$(findmnt -T "$real" -n -o TARGET 2>/dev/null | head -1)
  if [ -z "$mnt" ] || [ "$mnt" = "/" ]; then
    echo 'the project directory is on the root partition (or its mount cannot be determined); the dedicated space is what counts; refusing'
    return 1
  fi
  avail=$(df --output=avail -B1 "$mnt" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$avail" ] || [ "$avail" -lt 10737418240 ]; then
    echo "the dedicated space has ${avail:-unknown} bytes free; need >= 10 GiB (10737418240)"
    return 1
  fi
  return 0
}
if busy; then echo 'GPU occupied by another program; refusing to interfere'; exit 3; fi
space_ok || exit 3
if [ "$mode" = test ] || [ "$mode" = testcheck ]; then
  if own_busy && [ "${EA2_ACCEPT_OWN_GPU_RISK:-}" != 1 ]; then
    echo 'a GPU program of this account is running: stop it first, or set EA2_ACCEPT_OWN_GPU_RISK=1 after the user has accepted the risk; refusing'
    exit 3
  fi
fi
cpu_tests || { echo 'CPU tests failed (see the FAILED lines above); nothing was started'; exit 4; }
if [ "$mode" = test ] || [ "$mode" = testcheck ]; then      # the tests take minutes: look once more right before the one-shot run starts
  if own_busy && [ "${EA2_ACCEPT_OWN_GPU_RISK:-}" != 1 ]; then echo 'a GPU program of this account appeared while the CPU tests ran; refusing'; exit 3; fi
fi
if [ "$mode" = gputests ]; then
  EA2_ALLOW_GPU_TESTS=1 "$PY" -m pytest tests/direction2/test_ea2.py -q -p no:cacheprovider -m gpu --basetemp=.scratch/ea2-pytest-gpu
  exit $?
fi
log="results/direction2/runlogs/ea2_${mode}.log"
dmon_pid=""
if [ "$mode" = smoke ]; then
  nvidia-smi dmon -s um -d 5 >> "results/direction2/runlogs/ea2_smoke_dmon.log" 2>&1 & dmon_pid=$!
fi
maxtry=1; case "$mode" in train|select|adapt) maxtry=3 ;; esac       # never silently repeat smoke / testcheck / the one-shot test
resumes=0
while :; do
  rc=0
  for attempt in $(seq 1 "$maxtry"); do
    echo "E-A2 $mode attempt $attempt"
    timeout --signal=TERM --kill-after=60s 19h "$PY" scripts/ea2.py "$mode" "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || [ "$rc" -eq 124 ]; then break; fi
    if [ "$attempt" -lt "$maxtry" ]; then sleep 20; if busy; then rc=3; break; fi; fi
  done
  if [ "$rc" -eq 75 ] && { [ "$mode" = train ] || [ "$mode" = adapt ]; }; then
    resumes=$((resumes+1))
    if [ "$resumes" -gt 30 ]; then echo "too many resumes; stopping" | tee -a "$log"; rc=75; break; fi
    echo "$(date -Is) E-A2 $mode: exit 75 (time cap or foreign GPU program), resume #$resumes" | tee -a "$log"
    sleep 30; if busy; then echo "GPU occupied by another program; stopping auto-resume (relaunch when it is free)" | tee -a "$log"; rc=3; break; fi
    continue
  fi
  break
done
[ -n "$dmon_pid" ] && kill "$dmon_pid" 2>/dev/null
exit "$rc"
