#!/usr/bin/env bash
# E-A1 launcher for astarc (Quadro RTX 4000, shared VM; one GPU program of ours at a time; never compete with another account).
#   bash scripts/ea1_launch.sh <mode> [extra args for scripts/ea1.py]
#   audit | freeze | status | report | smoke-clean | cputests      CPU only (freeze = S0: register manifest; commit it before the next modes)
#   smoke                  formal-size timing + A=B=C on EA1_TRAIN / EA1_SEL (smoke dir results/direction2/ea1_smoke; never EA1_TEST); logs nvidia-smi dmon
#   paracheck              A=B=C on the registered manifest, writes parallel_check.json and run_config.json (lanes 1 or 3)
#   train                  the 3 chains (416 generations each), resumable, one checkpoint per generation per chain
#   postcheck              lanes=3 only: single-lane recomputation of generation 200 of chain 0 against the parallel result
#   select                 per-chain checkpoint, A_fresh, A_C2', F_orig, SEL re-evaluations, drift check; writes selection_frozen.json + freeze_record.json
#   testcheck              graph == eager on the first 64 EA1_SEL seeds (needs the committed freeze_record.json); no retry
#   test                   one-time final evaluation (needs committed freeze_record.json and a passing testcheck.json); no retry
#   gputests               the cuda tests of tests/direction2/test_ea1.py (EA1_ALLOW_GPU_TESTS=1)
#   bg <mode> [args]       start <mode> in a detached tmux session ea1_<mode> (reattach: tmux attach -t ea1_<mode>)
# Exit 75 = 18 h soft cap or another account's GPU program appeared (stop at a generation boundary): `train` relaunches itself while the GPU is
# free (at most 30 times, one log line each).  Other failures are retried at most 3 times for train/select (never smoke/paracheck/test/testcheck).
# Results are fetched with rsync and committed on the local machine; this script never commits or pushes.
# Use PYTHON=/path/to/python to pick the interpreter (default .venv/bin/python).
set -u
cd "$(dirname "$0")/.." || exit 3
mode="${1:-}"; shift || true
PY="${PYTHON:-}"
if [ -z "$PY" ]; then if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi; fi
mkdir -p .scratch/ea1-tmp results/direction2/runlogs
export TMPDIR="$PWD/.scratch/ea1-tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cpu_tests() {
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" -m pytest tests/direction2/test_ea1.py tests/direction2/test_e1_v2.py -q -p no:cacheprovider --basetemp=.scratch/ea1-pytest
}
case "$mode" in
  audit|freeze|status|report|smoke-clean) exec "$PY" scripts/ea1.py "$mode" "$@" ;;
  cputests) cpu_tests; exit $? ;;
  bg)
    sub="${1:-}"; shift || true
    [ -n "$sub" ] || { echo "usage: $0 bg <mode> [args]"; exit 2; }
    command -v tmux >/dev/null || { echo 'tmux not found'; exit 3; }
    tmux new-session -d -s "ea1_$sub" "cd '$PWD' && bash scripts/ea1_launch.sh $sub $*; echo \"ea1 $sub finished rc=\$?\"; sleep 60"
    echo "started tmux session ea1_$sub (tmux attach -t ea1_$sub); log: results/direction2/runlogs/ea1_${sub}.log"
    exit 0 ;;
  smoke|paracheck|train|postcheck|select|testcheck|test|gputests) ;;
  *) echo "usage: $0 audit|freeze|status|report|smoke-clean|cputests|smoke|paracheck|train|postcheck|select|testcheck|test|gputests|bg <mode>"; exit 2 ;;
esac
exec 9>.scratch/ea1-launch.lock
flock -n 9 || { echo 'Another E-A1 launcher is running'; exit 3; }
# busy = a GPU program of ANOTHER account (or one whose owner cannot be read); our own account's programs are allowed in any number
busy() {
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$(id -u)" ] || return 0
  done
  return 1
}
if busy; then echo 'GPU occupied by another program; refusing to interfere'; exit 3; fi
avail=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$avail" ] && [ "$avail" -lt 10 ]; then echo "only ${avail}G free on /; need >= 10G"; exit 3; fi
cpu_tests || exit 4
if [ "$mode" = gputests ]; then
  EA1_ALLOW_GPU_TESTS=1 "$PY" -m pytest tests/direction2/test_ea1.py -q -p no:cacheprovider -m gpu --basetemp=.scratch/ea1-pytest-gpu
  exit $?
fi
args=("$mode")
case "$mode" in
  smoke|paracheck) args=("$mode") ;;
  train|postcheck|select|testcheck|test) args=("$mode") ;;
esac
log="results/direction2/runlogs/ea1_${mode}.log"
dmon_pid=""
if [ "$mode" = smoke ]; then
  nvidia-smi dmon -s um -d 5 >> "results/direction2/runlogs/ea1_smoke_dmon.log" 2>&1 & dmon_pid=$!
fi
maxtry=1; case "$mode" in train|select) maxtry=3 ;; esac       # never silently repeat smoke / paracheck / the one-shot test
resumes=0
while :; do
  rc=0
  for attempt in $(seq 1 "$maxtry"); do
    echo "E-A1 $mode attempt $attempt"
    timeout --signal=TERM --kill-after=60s 19h "$PY" scripts/ea1.py "${args[@]}" "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || [ "$rc" -eq 124 ]; then break; fi
    if [ "$attempt" -lt "$maxtry" ]; then sleep 20; if busy; then rc=3; break; fi; fi
  done
  if [ "$rc" -eq 75 ] && [ "$mode" = train ]; then
    resumes=$((resumes+1))
    if [ "$resumes" -gt 30 ]; then echo "too many resumes; stopping" | tee -a "$log"; rc=75; break; fi
    echo "$(date -Is) E-A1 $mode: exit 75 (time cap or foreign GPU program), resume #$resumes" | tee -a "$log"
    sleep 30; if busy; then echo "GPU occupied by another program; stopping auto-resume (relaunch when it is free)" | tee -a "$log"; rc=3; break; fi
    continue
  fi
  break
done
[ -n "$dmon_pid" ] && kill "$dmon_pid" 2>/dev/null
exit "$rc"
