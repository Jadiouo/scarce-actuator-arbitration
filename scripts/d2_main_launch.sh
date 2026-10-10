#!/usr/bin/env bash
# Main-experiment launcher for the school machine (and what ~/s1_watchdog.sh should run): CUDA-graph stepping with an automatic eager fallback.
#   * starts scripts/d2_run_queue.py with --graph --graph-fallback (sequential: one segment at a time);
#   * if a segment fails with a CUDA-graph error (capture / replay), the runner logs "GRAPH FALLBACK" (segment log, runlogs/queue_history.jsonl,
#     runlogs/graph_fallback.json) and runs that segment and all later ones eager from the segment's checkpoint -- bitwise the same results;
#   * any other error stops the queue as before (exit 2): no endless retry.  While runlogs/graph_fallback.json exists the runner starts eager.
# Extra arguments are passed through (e.g. --max-segments 1, --dry-run).  Use the venv's python: PYTHON=/path/to/python bash scripts/d2_main_launch.sh
set -u
cd "$(dirname "$0")/.." || exit 3
exec "${PYTHON:-python3}" scripts/d2_run_queue.py --stage main --push-every 1 --remote origin --branch direction2-freeze --graph --graph-fallback "$@"
