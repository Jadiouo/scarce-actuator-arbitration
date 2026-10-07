#!/usr/bin/env bash
# Direction 2 acceptance check on the TARGET machine (run it once after deployment, inside the container or the venv).
#   bash scripts/d2_target_check.sh               # full check (about 30-60 min on the school GPU)
#   bash scripts/d2_target_check.sh --cpu-only    # only the CPU test (no GPU needed; for a first look)
#   bash scripts/d2_target_check.sh --quick       # whole pipeline with tiny sizes (a few minutes): checks that the script works, hour estimate NOT valid
# Result: results/direction2/target_check.json and a final line  "可以開跑" / "不可以開跑"  with the reasons.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
CPU_ONLY=0; QUICK=0
for x in "$@"; do case "$x" in --cpu-only) CPU_ONLY=1;; --quick) QUICK=1;; -h|--help) sed -n '2,6p' "$0"; exit 0;; *) echo "unknown option $x"; exit 2;; esac; done
WORK="results/direction2/target_check_work"
rm -rf "$WORK"; mkdir -p "$WORK"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONUNBUFFERED=1
T="tests/direction2"
# GPU tests that compare the new environment bit by bit with the Direction 1 frontier simulator (and the vulnerability environments with the plain one):
GPU_TESTS=(
  "$T/test_d2_env.py::test_T01_parity_m3c_per_seed"          # env == frontier, M3C, per seed, bitwise
  "$T/test_d2_env.py::test_T02_parity_m4_and_naive"          # env == frontier, M4 and NAIVE
  "$T/test_d2_env.py::test_T08_batch_invariance"             # result does not depend on the batch layout (matters for lam x n_S = 65536 rows)
  "$T/test_d2_env.py::test_T06_replay_d1_json_naive_m4"      # replay of the stored Direction 1 numbers
  "$T/test_d2_s1.py::test_T36_vuln_identity_mutation"        # vulnerability environment with knob = identity equals the plain environment
  "$T/test_d2_s1.py::test_T41_vuln_ref_policy_public_only"
)
[ "$QUICK" = 1 ] && GPU_TESTS=("$T/test_d2_env.py::test_T01_parity_m3c_per_seed")

run_step() {   # run_step NAME TIMEOUT_S CMD...   -> writes $WORK/NAME.log and the step record
  local name="$1" tmo="$2"; shift 2
  local t0=$SECONDS rc
  timeout "$tmo" "$@" > "$WORK/$name.log" 2>&1; rc=$?
  [ "$rc" = 124 ] && echo "TIMEOUT after ${tmo}s" >> "$WORK/$name.log"
  "$PY" scripts/d2_target_check.py record --dir "$WORK" --name "$name" --rc "$rc" --seconds "$((SECONDS - t0))" --log "$WORK/$name.log"
  tail -n 4 "$WORK/$name.log" | sed 's/^/    | /'
  echo "    -> exit code $rc, $((SECONDS - t0)) s   (full log: $WORK/$name.log)"
  return "$rc"
}

echo "=== [1/6] environment (nvidia-smi, driver, torch, CUDA, capability, memory) ==="
if [ "$CPU_ONLY" = 0 ]; then
  "$PY" scripts/d2_target_check.py env --out "$WORK/env.json"; ENV_RC=$?
else
  echo "(skipped: --cpu-only)"; ENV_RC=1
fi
echo
echo "=== [2/6] float64 on the GPU ==="
if [ "$CPU_ONLY" = 0 ] && [ "$ENV_RC" = 0 ]; then
  "$PY" scripts/d2_target_check.py f64 --out "$WORK/f64.json"; F64_RC=$?
else
  echo "(skipped)"; F64_RC=1
fi
echo
echo "=== [3/6] CPU tests: python -m pytest tests/direction2 -m 'not gpu' ==="
run_step cpu_tests 3000 "$PY" -m pytest tests/direction2 -m "not gpu" -q -p no:cacheprovider
echo
echo "=== [4/6] GPU tests: environment vs frontier, bit for bit (${#GPU_TESTS[@]} tests, limit 20 min) ==="
if [ "$CPU_ONLY" = 0 ] && [ "$ENV_RC" = 0 ] && [ "$F64_RC" = 0 ]; then
  run_step gpu_tests 1200 "$PY" -m pytest "${GPU_TESTS[@]}" -q -p no:cacheprovider
else
  echo "(skipped)"
fi
echo
echo "=== [5/6] speed: lam=256, n_S=256, T=2e4, 1 warm-up + 3 generations, 1 validation (T=1e5, 32 seeds) per cell ==="
if [ "$CPU_ONLY" = 0 ] && [ "$ENV_RC" = 0 ] && [ "$F64_RC" = 0 ]; then
  if [ "$QUICK" = 1 ]; then
    "$PY" scripts/d2_target_check.py perf --out "$WORK/perf.json" --gens 2 --T 2000 --T-val 5000
  else
    "$PY" scripts/d2_target_check.py perf --out "$WORK/perf.json"
  fi
else
  echo "(skipped)"
fi
echo
echo "=== [6/6] decision ==="
VERDICT_ARGS=()
[ "$CPU_ONLY" = 1 ] && VERDICT_ARGS+=(--cpu-only)
"$PY" scripts/d2_target_check.py verdict --dir "$WORK" "${VERDICT_ARGS[@]}"
exit $?
