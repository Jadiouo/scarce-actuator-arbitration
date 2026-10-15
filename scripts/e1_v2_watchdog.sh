#!/usr/bin/env bash
# Template for ~/e1v2_watchdog.sh on the school machine (cron every 15 min + @reboot).  Mirrors main_watchdog.
# Only ever starts `all` (s1,s2,sx,select).  It NEVER runs testcheck or test: those are launched by hand after the freeze is committed.
#   pause flag: ~/E1V2_PAUSED   no-progress counter: ~/.e1v2_noprogress   log: ~/logs/e1v2_watchdog.log
set -u
REPO="${E1V2_REPO:-$HOME/scarce-actuator-arbitration}"
SESSION=e1v2
PAUSE="$HOME/E1V2_PAUSED"; CNT="$HOME/.e1v2_noprogress"; MARK="$HOME/.e1v2_last_progress"
mkdir -p "$HOME/logs"; LOG="$HOME/logs/e1v2_watchdog.log"
say() { echo "$(date -Is) $*" >> "$LOG"; }
exec 8>"$HOME/.e1v2_watchdog.lock"; flock -n 8 || exit 0
[ -e "$PAUSE" ] && exit 0
OUT="$REPO/results/direction2/e1v2"; ST="$OUT/status.json"
# finished: select is done -> nothing more for the watchdog to do (test is manual)
if [ -f "$ST" ] && python3 -c 'import json,sys; sys.exit(0 if "select" in json.load(open(sys.argv[1])).get("stages_done",[]) else 1)' "$ST"; then say "select done; watchdog idle"; exit 0; fi
# already running
if pgrep -f 'scripts/e1_v2.py' >/dev/null || pgrep -f 'e1_v2_launch.sh' >/dev/null; then exit 0; fi
# another GPU program: do not interfere, do not count
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then say "GPU busy with another program; not restarting"; exit 0; fi
# progress = status.json modification time changed since the previous restart
now=$(stat -c %Y "$ST" 2>/dev/null || echo 0); last=$(cat "$MARK" 2>/dev/null || echo -1)
n=$(cat "$CNT" 2>/dev/null || echo 0)
if [ "$now" = "$last" ]; then n=$((n+1)); else n=0; fi
echo "$n" > "$CNT"; echo "$now" > "$MARK"
if [ "$n" -ge 3 ]; then touch "$PAUSE"; say "3 restarts without progress; created $PAUSE and stopping"; exit 0; fi
tmux kill-session -t "$SESSION" 2>/dev/null
say "starting tmux $SESSION: launch all (no-progress count $n)"
tmux new-session -d -s "$SESSION" -c "$REPO" "PYTHON=$REPO/.venv/bin/python bash scripts/e1_v2_launch.sh all >> $HOME/logs/e1v2.log 2>&1"
