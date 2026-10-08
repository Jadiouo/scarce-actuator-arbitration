#!/usr/bin/env python3
"""Direction 2 resumable sequential runner (replaces `gpujob` on a machine that only has one user and no queue daemon).

It reads the frozen run plan (results/direction2/run_plan.json), produces the SAME segments as the local queue manifest
(arbitration.rl.budget.segment, max 1500 s per segment, same convention as results/direction2/s1_queue_manifest.json) and runs them one
after the other:   python -m arbitration.rl.train --plan ... --cell C --run K --g0 A --g1 B

 * a segment whose part file has status "done" is skipped (so a restart continues where it stopped, never from the beginning);
 * an interrupted segment is simply run again: train.py resumes from the checkpoint it wrote (every 50 generations and at the segment end);
 * every segment writes a log to results/direction2/runlogs/ (start, end, elapsed, exit code) and one line to runlogs/queue_history.jsonl;
 * the first failure stops the queue (nothing is skipped automatically); the reason and the command to rerun are printed;
 * after each finished segment its part file and the run's checkpoint are `git add`-ed and committed (see "git" below).

git.  train.py refuses to start unless results/direction2/run_plan.json and FREEZE.md are tracked and unchanged against HEAD and HEAD
descends from the freeze commit a72d3d7 (arbitration/rl/plan.py: verify_repo_state).  It does NOT look at any other file, so untracked result
files do not block training and committing results is not required by the guard.  The commits exist so that the git state stays clean and the
results travel with `git bundle`; they only add results/direction2/{parts,ckpt}/ files, so the guard keeps passing (HEAD stays a descendant of
the freeze commit).  Use --no-commit to leave the results untracked.

auto push (GitHub backup).  After every --push-every N finished-and-committed segments (default 1) the runner executes exactly
`git push <remote> <branch>` (defaults: origin, direction2-freeze; never any force option, never a refspec, never master/main).  It refuses
to push (and refuses to start, exit 3) unless the checked-out branch IS --branch and --branch is not master/main; --no-push turns it off.
A failed push is logged (runlogs/push.log, queue_history.jsonl) and never stops the training: it is retried after the next finished segment
(and once more at the end of the queue).  --status shows how many commits are not on <remote>/<branch> yet.

speed-up options (docs/direction2-speedup-plan.md; both off by default, the default behaviour is exactly the sequential runner above).
  --graph          every segment steps with CUDA graphs (same as D2_GRAPH=1 in the environment): bitwise identical results, faster.
  --concurrent N   advance the next segment of N DIFFERENT cells at the same time on the one GPU (CUDA-graph stepping unless --no-graph):
                   a long-lived worker process (python -m arbitration.rl.multitrain --serve) runs each segment in its own thread + CUDA stream;
                   every cell's results are bitwise what it would be alone.  The runner keeps: segments of one cell strictly in order, a part+checkpoint
                   commit (and push) per finished segment, stop at the first failure (segments already running are allowed to finish and are committed;
                   nothing new is started), skipping finished segments on restart, the same segment list (manifest) and --max-segments (= segments started).

Usage:
  python3 scripts/d2_run_queue.py --dry-run              # list the segments (done / todo)
  python3 scripts/d2_run_queue.py --status               # progress and remaining-time estimate
  python3 scripts/d2_run_queue.py                        # run stage s1 (default); rerun the same command after any interruption
  python3 scripts/d2_run_queue.py --stage main
Exit code: 0 = everything in the selection is done, 2 = a segment failed (queue stopped), 3 = refused by a pre-flight check, 130 = interrupted.
"""
import argparse
import datetime as _dt
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from arbitration.rl import budget, parts, plan as _plan  # noqa: E402  (CPU only: no torch is imported here)

PLAN_REL = _plan.DEFAULT_PLAN
MANIFEST_REL = os.path.join("results", "direction2", "s1_queue_manifest.json")
RUNLOG_REL = os.path.join("results", "direction2", "runlogs")
T_STARTUP_S = 60.0                      # same convention as the manifest (segmentation.t_startup_s)
MAX_SEG_S = 1500.0
TRAILER = ("Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01Vcin4FdSJwvFyTg4u3RSiU")
EXIT_OK, EXIT_FAIL, EXIT_REFUSED, EXIT_INTR = 0, 2, 3, 130


PROTECTED_BRANCHES = ("master", "main")
PUSH_TIMEOUT_S = 180


class Refused(Exception):
    """A pre-flight check failed (nothing was run)."""


# ------------------------------------------------------------------------------------------------ job list
def _cells_for_stage(pl: Dict[str, Any], stage: str, names: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    by_name = {c["name"]: c for c in pl["cells"]}
    if names:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise Refused(f"unknown cell(s) {missing}; known: {sorted(by_name)}")
        return [by_name[n] for n in names]
    if stage == "pilot":
        return [by_name["pilot_naive_cold"]]                  # the warm pilot is conditional on a human decision: select it with --cells
    group = {"s1": "s1", "main": "main"}[stage]
    return [c for c in pl["cells"] if c["group"] == group and c.get("wired", True) and c["name"] not in _plan.REUSED_CELLS]


def build_jobs(pl: Dict[str, Any], stage: str = "s1", cells: Optional[Sequence[str]] = None,
               runs: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """All segments of the selection in execution order: cell (plan order) -> run -> generation interval."""
    jobs = []
    for c in _cells_for_stage(pl, stage, cells):
        e = c["est_per_run"]
        B = int(c["lam"]) * int(c["n_S"])
        seg = budget.segment(lambda _B, v=e["t_step_p95_ms"]: v, int(c["T_train"]), int(c["G_gens"]), B, 0, t_val_s=e["t_val_s"],
                             t_startup_s=T_STARTUP_S, max_s=MAX_SEG_S, val_every=int(c["val_every"]))
        for k in c["run_ids"]:
            if runs is not None and k not in runs:
                continue
            for g0, g1 in seg:
                est = lambda ms: budget.segment_estimate_s(g0, g1, lambda _B: ms, int(c["T_train"]), B, 0, e["t_val_s"], T_STARTUP_S, int(c["val_every"]))
                jobs.append(dict(cell=c["name"], run=int(k), g0=g0, g1=g1, est_mean_s=est(e["t_step_mean_ms"]), est_p95_s=est(e["t_step_p95_ms"])))
    return jobs


def train_cmd(job: Dict[str, Any], python: str = sys.executable) -> List[str]:
    return [python, "-m", "arbitration.rl.train", "--plan", PLAN_REL, "--cell", job["cell"], "--run", str(job["run"]),
            "--g0", str(job["g0"]), "--g1", str(job["g1"])]


def job_label(j: Dict[str, Any]) -> str:
    return f"{j['cell']} run{j['run']} g{j['g0']}-{j['g1']}"


def part_file(root: str, j: Dict[str, Any]) -> str:
    return parts.part_path(j["cell"], j["run"], j["g0"], j["g1"], root)


def ckpt_file(root: str, j: Dict[str, Any]) -> str:
    return parts.ckpt_path(j["cell"], j["run"], root)


def is_done(root: str, j: Dict[str, Any]) -> bool:
    return parts.job_action(part_file(root, j), None) == "skip"


# ------------------------------------------------------------------------------------------------ git
def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _identity_args(root: str) -> List[str]:
    out = []
    if not _git(root, "config", "user.name").stdout.strip():
        out += ["-c", "user.name=d2-run-queue"]
    if not _git(root, "config", "user.email").stdout.strip():
        out += ["-c", "user.email=d2-run-queue@localhost"]
    return out


def commit_paths(root: str, paths: Sequence[str], message: str) -> Optional[str]:
    """`git add` + `git commit -- paths` (only these paths, whatever else is staged).  Returns the new short sha, None if there was nothing to commit.
    RuntimeError if git fails."""
    rel = [os.path.relpath(p, root) for p in paths if os.path.exists(p)]
    if not rel:
        return None
    r = _git(root, "add", "--", *rel)
    if r.returncode != 0:
        raise RuntimeError(f"git add failed: {r.stderr.strip()}")
    if _git(root, "diff", "--cached", "--quiet", "--", *rel).returncode == 0:
        return None
    r = subprocess.run(["git", "-C", root, *_identity_args(root), "commit", "-q", "-m", message + "\n\n" + TRAILER, "--", *rel],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git commit failed: {(r.stderr or r.stdout).strip()}")
    return _git(root, "rev-parse", "--short", "HEAD").stdout.strip()


def _dirty(root: str, path: str) -> bool:
    return bool(_git(root, "status", "--porcelain", "--", os.path.relpath(path, root)).stdout.strip())


# ------------------------------------------------------------------------------------------------ auto push (backup)
def current_branch(root: str, git: Callable[..., Any] = None) -> str:
    return (git or _git)(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def push_cmd(remote: str, branch: str) -> List[str]:
    """The ONLY push command this script ever builds: `git push <remote> <branch>`.  No force option, no refspec (`+x`, `a:b`, `:x` would be)."""
    for what, v in (("remote", remote), ("branch", branch)):
        if not v or v.startswith("-") or v.startswith("+") or ":" in v or any(ch.isspace() for ch in v):
            raise Refused(f"refusing to push: invalid {what} {v!r}")
    if branch in PROTECTED_BRANCHES:
        raise Refused(f"refusing to push: branch {branch!r} is protected (this runner only backs up the direction2 branch)")
    return ["git", "push", remote, branch]


def push_guard(root: str, remote: str, branch: str, git: Callable[..., Any] = None) -> None:
    """Raises Refused unless pushing <remote> <branch> from the current checkout is allowed."""
    push_cmd(remote, branch)
    cur = current_branch(root, git)
    if cur in PROTECTED_BRANCHES:
        raise Refused(f"refusing to push: the checked-out branch is {cur!r}")
    if cur != branch:
        raise Refused(f"refusing to push: the checked-out branch is {cur!r} but --branch is {branch!r}")


def _push_run(root: str, cmd: List[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND=os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=20"))
    return subprocess.run(["git", "-C", root, *cmd[1:]], capture_output=True, text=True, timeout=PUSH_TIMEOUT_S, env=env)


def unpushed_count(root: str, remote: str, branch: str, git: Callable[..., Any] = None) -> Optional[int]:
    """Commits on HEAD that <remote>/<branch> (as last fetched/pushed from here) does not have.  None if that remote ref does not exist locally yet."""
    g = git or _git
    ref = f"refs/remotes/{remote}/{branch}"
    if g(root, "rev-parse", "--verify", "-q", ref).returncode != 0:
        return None
    r = g(root, "rev-list", "--count", f"{ref}..HEAD")
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None


def try_push(root: str, remote: str, branch: str, out=print, pusher: Callable[[str, List[str]], Any] = None, git: Callable[..., Any] = None) -> bool:
    """One push attempt.  Never raises (a failed backup must not stop the training); returns True if the push succeeded."""
    pusher = pusher or _push_run
    try:
        push_guard(root, remote, branch, git)
        cmd = push_cmd(remote, branch)
        r = pusher(root, cmd)
        ok, msg = r.returncode == 0, ((r.stderr or "") + (r.stdout or "")).strip()
    except Refused as e:
        ok, msg = False, str(e)
    except Exception as e:                                            # timeout, git missing, ...
        ok, msg = False, f"{type(e).__name__}: {e}"
    line = f"[{_now()}] PUSH {remote} {branch}: " + ("ok" if ok else "FAILED (training continues; will retry after the next segment) - " + msg.replace("\n", " | ")[-400:])
    try:
        os.makedirs(os.path.join(root, RUNLOG_REL), exist_ok=True)
        with open(os.path.join(root, RUNLOG_REL, "push.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    out(line)
    return ok


# ------------------------------------------------------------------------------------------------ running one segment
class _Interrupted(BaseException):
    pass


def subprocess_runner(job: Dict[str, Any], cmd: List[str], logfile: str, root: str) -> int:
    """Run the training command with stdout+stderr appended to `logfile`; returns the exit code.  KeyboardInterrupt/SIGTERM stop the child first."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(logfile, "a") as lf:
        p = subprocess.Popen(cmd, cwd=root, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            return p.wait()
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _tail(path: str, n: int) -> List[str]:
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def run_queue(root: str, jobs: List[Dict[str, Any]], *, runner: Callable[[Dict[str, Any], List[str], str, str], int] = subprocess_runner,
              commit: bool = True, max_segments: Optional[int] = None, python: str = sys.executable, out=print,
              rerun_hint: str = "python3 scripts/d2_run_queue.py", push_every: int = 0, remote: str = "origin", branch: str = "direction2-freeze",
              pusher: Optional[Callable[[str, List[str]], Any]] = None) -> int:
    """Run the not-yet-done jobs in order.  Returns the exit code (see module doc).  push_every=N>0 (needs commit=True): push after every N committed
    segments, retrying after each later segment while a push is outstanding."""
    since_push, pending = 0, False

    def maybe_push(final: bool = False) -> None:
        nonlocal since_push, pending
        if push_every <= 0 or not commit:
            return
        if final or pending or since_push >= push_every:
            if try_push(root, remote, branch, out, pusher):
                since_push, pending = 0, False
            else:
                pending = True

    logdir = os.path.join(root, RUNLOG_REL)
    os.makedirs(logdir, exist_ok=True)
    todo = [j for j in jobs if not is_done(root, j)]
    out(f"[{_now()}] {len(jobs) - len(todo)}/{len(jobs)} segments already done; {len(todo)} to run" + (f" (at most {max_segments} this time)" if max_segments else ""))
    ran = 0
    for j in jobs:
        if is_done(root, j):
            if commit and _dirty(root, part_file(root, j)):         # finished earlier but the commit was interrupted: commit the part file now
                try:
                    commit_paths(root, [part_file(root, j)], f"Direction 2: {job_label(j)} (part, late commit)")
                except RuntimeError as e:
                    return _fail(out, j, f"late commit failed: {e}", None, rerun_hint, cmd=None, python=python)
            continue
        if max_segments is not None and ran >= max_segments:
            out(f"[{_now()}] stopping after {ran} segment(s) as requested (--max-segments); rerun the same command to continue")
            return EXIT_OK
        cmd = train_cmd(j, python)
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        logfile = os.path.join(logdir, f"{j['cell']}__run{j['run']}__g{j['g0']:04d}-{j['g1']:04d}__{stamp}.log")
        t0, start = time.time(), _now()
        with open(logfile, "w") as lf:
            lf.write(f"# START {start}\n# CMD   {' '.join(cmd)}\n# CWD   {root}\n# HEAD  {_git(root, 'rev-parse', '--short', 'HEAD').stdout.strip()}\n")
        out(f"[{start}] START {job_label(j)}  (est {j['est_mean_s'] / 60:.0f}-{j['est_p95_s'] / 60:.0f} min on the reference GPU)\n    log: {logfile}")
        try:
            rc = runner(j, cmd, logfile, root)
        except (KeyboardInterrupt, _Interrupted):
            _finish_log(logfile, root, j, start, t0, "interrupted", None)
            out(f"[{_now()}] INTERRUPTED during {job_label(j)} after {time.time() - t0:.0f} s.\n"
                f"    Nothing is lost: rerun the same command ({rerun_hint}); the segment resumes from its last checkpoint.")
            return EXIT_INTR
        elapsed = time.time() - t0
        status = None
        if rc == 0:
            try:
                status = parts.read_part(part_file(root, j)).get("status")
            except (OSError, ValueError):
                status = None
        if rc != 0 or status != "done":
            reason = f"train.py exited with code {rc}" if rc != 0 else f"train.py exited 0 but the part file is not 'done' (status={status!r})"
            _finish_log(logfile, root, j, start, t0, "failed", rc)
            return _fail(out, j, reason, logfile, rerun_hint, cmd, python)
        sha = None
        if commit:
            try:
                sha = commit_paths(root, [part_file(root, j), ckpt_file(root, j)], f"Direction 2: {job_label(j)} (part + checkpoint)")
            except RuntimeError as e:
                _finish_log(logfile, root, j, start, t0, "commit_failed", rc)
                return _fail(out, j, f"segment finished but {e}", logfile, rerun_hint, cmd, python)
        _finish_log(logfile, root, j, start, t0, "ok", rc, sha)
        ran += 1
        out(f"[{_now()}] DONE  {job_label(j)} in {elapsed / 60:.1f} min (exit 0)" + (f", committed {sha}" if sha else ""))
        if sha:
            since_push += 1
            maybe_push()
    if commit:                                                      # checkpoints of runs whose last segment was committed before a later ckpt change
        for j in jobs:
            ck = ckpt_file(root, j)
            if os.path.exists(ck) and _dirty(root, ck):
                try:
                    commit_paths(root, [ck], f"Direction 2: {j['cell']} run{j['run']} checkpoint")
                except RuntimeError as e:
                    out(f"WARNING: could not commit {ck}: {e}")
    if since_push or pending:
        maybe_push(final=True)
    out(f"[{_now()}] all {len(jobs)} segments of this selection are done")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------ concurrent cells (plan B)
class ProcWorker:
    """The long-lived `python -m arbitration.rl.multitrain --serve` process: JSON lines in (jobs), JSON lines out (one 'done' per segment)."""

    def __init__(self, cmd: List[str], root: str, logfile: str):
        import queue
        import threading
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._err = open(logfile, "a")
        self.p = subprocess.Popen(cmd, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, text=True, env=env)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                except ValueError:
                    pass
        self.events.put({"event": "eof", "returncode": self.p.wait()})

    def submit(self, jid: int, job: Dict[str, Any], logfile: str) -> None:
        msg = dict(op="run", id=jid, cell=job["cell"], run=job["run"], g0=job["g0"], g1=job["g1"], log=logfile)
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def close(self) -> None:
        try:
            self.p.stdin.write(json.dumps(dict(op="quit")) + "\n")
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=120)
        except Exception:                                                   # noqa: BLE001
            self.kill()

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self._err.close()


def run_queue_concurrent(root: str, jobs: List[Dict[str, Any]], concurrent: int, *, worker_factory: Optional[Callable[[str, int], Any]] = None,
                         commit: bool = True, max_segments: Optional[int] = None, python: str = sys.executable, out=print,
                         rerun_hint: str = "python3 scripts/d2_run_queue.py", push_every: int = 0, remote: str = "origin",
                         branch: str = "direction2-freeze", pusher: Optional[Callable[[str, List[str]], Any]] = None, graph: bool = True) -> int:
    """run_queue with up to `concurrent` segments of DIFFERENT cells in flight.  Same guarantees as run_queue (see the module doc); a worker (real:
    ProcWorker; tests inject a fake) executes the segments.  Segment order inside a cell is the order of `jobs`; the cells are served in the order
    of their first segment in `jobs`.  After a failure nothing new is started, the segments still running finish and are committed."""
    from collections import OrderedDict, deque
    since_push, pending = 0, False

    def maybe_push(final: bool = False) -> None:
        nonlocal since_push, pending
        if push_every <= 0 or not commit:
            return
        if final or pending or since_push >= push_every:
            if try_push(root, remote, branch, out, pusher):
                since_push, pending = 0, False
            else:
                pending = True

    logdir = os.path.join(root, RUNLOG_REL)
    os.makedirs(logdir, exist_ok=True)
    todo = [j for j in jobs if not is_done(root, j)]
    out(f"[{_now()}] {len(jobs) - len(todo)}/{len(jobs)} segments already done; {len(todo)} to run, up to {concurrent} cells at a time"
        + (f" (at most {max_segments} this time)" if max_segments else ""))
    for j in jobs:
        if is_done(root, j) and commit and _dirty(root, part_file(root, j)):      # finished earlier but the commit was interrupted
            try:
                commit_paths(root, [part_file(root, j)], f"Direction 2: {job_label(j)} (part, late commit)")
            except RuntimeError as e:
                return _fail(out, j, f"late commit failed: {e}", None, rerun_hint, cmd=None, python=python)
    chains: "OrderedDict[str, deque]" = OrderedDict()
    for j in todo:
        chains.setdefault(j["cell"], deque()).append(j)
    worker = None
    active: Dict[int, Dict[str, Any]] = {}                  # id -> dict(job, logfile, start, t0)
    busy: set = set()
    dispatched, nid = 0, 0
    failure: Optional[Any] = None
    stopped_by_limit = False
    try:
        while True:
            while failure is None and len(active) < concurrent and not stopped_by_limit:
                cell = next((c for c, dq in chains.items() if dq and c not in busy), None)
                if cell is None:
                    break
                if max_segments is not None and dispatched >= max_segments:
                    stopped_by_limit = True
                    out(f"[{_now()}] stopping after {dispatched} segment(s) started as requested (--max-segments); rerun the same command to continue")
                    break
                j = chains[cell].popleft()
                if worker is None:
                    cmd = [python, "-m", "arbitration.rl.multitrain", "--serve", "--max-lanes", str(concurrent)] + ([] if graph else ["--no-graph"])
                    wlog = os.path.join(logdir, "concurrent_worker.log")
                    worker = worker_factory(root, concurrent) if worker_factory else ProcWorker(cmd, root, wlog)
                nid += 1
                stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
                logfile = os.path.join(logdir, f"{j['cell']}__run{j['run']}__g{j['g0']:04d}-{j['g1']:04d}__{stamp}.log")
                start = _now()
                with open(logfile, "w") as lf:
                    lf.write(f"# START {start}\n# CMD   (concurrent worker) {j['cell']} run {j['run']} g0 {j['g0']} g1 {j['g1']}\n# CWD   {root}\n"
                             f"# HEAD  {_git(root, 'rev-parse', '--short', 'HEAD').stdout.strip()}\n")
                out(f"[{start}] START {job_label(j)}  (est {j['est_mean_s'] / 60:.0f}-{j['est_p95_s'] / 60:.0f} min on the reference GPU; "
                    f"{len(active) + 1} cell(s) running)\n    log: {logfile}")
                active[nid] = dict(job=j, logfile=logfile, start=start, t0=time.time())
                busy.add(cell)
                dispatched += 1
                worker.submit(nid, j, logfile)
            if not active:
                break
            ev = worker.events.get()
            if ev.get("event") == "eof":
                for jid, a in list(active.items()):
                    _finish_log(a["logfile"], root, a["job"], a["start"], a["t0"], "failed", ev.get("returncode"))
                    if failure is None:
                        failure = (a["job"], f"the concurrent worker process exited (code {ev.get('returncode')}) while the segment was running",
                                   a["logfile"])
                active.clear()
                break
            if ev.get("event") != "done" or ev.get("id") not in active:
                continue
            a = active.pop(ev["id"])
            j = a["job"]
            busy.discard(j["cell"])
            elapsed = time.time() - a["t0"]
            status = None
            if ev.get("ok"):
                try:
                    status = parts.read_part(part_file(root, j)).get("status")
                except (OSError, ValueError):
                    status = None
            if not ev.get("ok") or status != "done":
                reason = (f"segment failed: {ev.get('error')}" if not ev.get("ok")
                          else f"the segment returned but the part file is not 'done' (status={status!r})")
                _finish_log(a["logfile"], root, j, a["start"], a["t0"], "failed", 1)
                if failure is None:
                    failure = (j, reason, a["logfile"])
                else:
                    out(f"[{_now()}] ALSO FAILED: {job_label(j)}: {reason}")
                continue
            sha = None
            if commit:
                try:
                    sha = commit_paths(root, [part_file(root, j), ckpt_file(root, j)], f"Direction 2: {job_label(j)} (part + checkpoint)")
                except RuntimeError as e:
                    _finish_log(a["logfile"], root, j, a["start"], a["t0"], "commit_failed", 0)
                    if failure is None:
                        failure = (j, f"segment finished but {e}", a["logfile"])
                    continue
            _finish_log(a["logfile"], root, j, a["start"], a["t0"], "ok", 0, sha)
            out(f"[{_now()}] DONE  {job_label(j)} in {elapsed / 60:.1f} min (exit 0)" + (f", committed {sha}" if sha else ""))
            if sha:
                since_push += 1
                maybe_push()
    except (KeyboardInterrupt, _Interrupted):
        if worker is not None:
            worker.kill()
        for a in active.values():
            _finish_log(a["logfile"], root, a["job"], a["start"], a["t0"], "interrupted", None)
        out(f"[{_now()}] INTERRUPTED with {len(active)} segment(s) running.\n"
            f"    Nothing is lost: rerun the same command ({rerun_hint}); each segment resumes from its last checkpoint.")
        return EXIT_INTR
    if worker is not None:
        worker.close()
    if failure is not None:
        if since_push or pending:
            maybe_push(final=True)
        j, reason, logfile = failure
        return _fail(out, j, reason, logfile, rerun_hint, cmd=None, python=python)
    if commit:
        for j in jobs:
            ck = ckpt_file(root, j)
            if os.path.exists(ck) and _dirty(root, ck):
                try:
                    commit_paths(root, [ck], f"Direction 2: {j['cell']} run{j['run']} checkpoint")
                except RuntimeError as e:
                    out(f"WARNING: could not commit {ck}: {e}")
    if since_push or pending:
        maybe_push(final=True)
    if stopped_by_limit:
        return EXIT_OK
    out(f"[{_now()}] all {len(jobs)} segments of this selection are done")
    return EXIT_OK


def _finish_log(logfile: str, root: str, j: Dict[str, Any], start: str, t0: float, result: str, rc: Optional[int], sha: Optional[str] = None) -> None:
    end = _now()
    elapsed = time.time() - t0
    with open(logfile, "a") as lf:
        lf.write(f"\n# END {end}\n# ELAPSED_S {elapsed:.1f}\n# EXIT_CODE {rc}\n# RESULT {result}\n")
    rec = dict(cell=j["cell"], run=j["run"], g0=j["g0"], g1=j["g1"], start=start, end=end, elapsed_s=round(elapsed, 1), exit_code=rc,
               result=result, commit=sha, log=os.path.relpath(logfile, root))
    with open(os.path.join(root, RUNLOG_REL, "queue_history.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _fail(out, j, reason, logfile, rerun_hint, cmd, python) -> int:
    out(f"\n[{_now()}] STOPPED: {job_label(j)} failed.\n    reason: {reason}")
    if logfile:
        out(f"    log: {logfile}\n    last lines of the log:")
        for line in _tail(logfile, 15):
            out("      | " + line)
    out("    The queue does NOT skip a failed segment (later segments need its checkpoint).")
    if cmd:
        out("    segment command: " + " ".join(cmd))
    out(f"    after fixing the cause, rerun:  {rerun_hint}   (finished segments are skipped, this one resumes from its checkpoint)")
    return EXIT_FAIL


# ------------------------------------------------------------------------------------------------ status / dry run
def _hms(s: float) -> str:
    return "%dh%02dm" % (s // 3600, s % 3600 // 60)


def print_jobs(root: str, jobs: List[Dict[str, Any]], out=print) -> None:
    for i, j in enumerate(jobs, 1):
        out(f"{i:4d}  {'done' if is_done(root, j) else 'todo'}  {job_label(j):44s} est {j['est_mean_s'] / 60:5.1f} min (p95 {j['est_p95_s'] / 60:5.1f})")
    n_done = sum(is_done(root, j) for j in jobs)
    rem = [j for j in jobs if not is_done(root, j)]
    out(f"{len(jobs)} segments, {n_done} done, {len(rem)} to run; estimated {_hms(sum(j['est_mean_s'] for j in rem))} mean / "
        f"{_hms(sum(j['est_p95_s'] for j in rem))} p95 on the REFERENCE GPU (RTX 5070 Ti); see --status for the speed seen on this machine")


def print_status(root: str, jobs: List[Dict[str, Any]], out=print, remote_branch: Optional[Sequence[str]] = None) -> None:
    cells: Dict[str, Dict[str, Any]] = {}
    ratios = []
    for j in jobs:
        c = cells.setdefault(j["cell"], dict(total=0, done=0, wall=0.0))
        c["total"] += 1
        if is_done(root, j):
            c["done"] += 1
            try:
                w = parts.read_part(part_file(root, j)).get("wall_s")
            except (OSError, ValueError):
                w = None
            if w:
                ratios.append(float(w) / j["est_mean_s"])
    out(f"{'cell':22s} {'done/seg':>9s}  state")
    for name, c in cells.items():
        out(f"{name:22s} {c['done']:4d}/{c['total']:<4d}  {'DONE' if c['done'] == c['total'] else ('started' if c['done'] else 'waiting')}")
    rem = [j for j in jobs if not is_done(root, j)]
    nd = len(jobs) - len(rem)
    out(f"\nsegments done {nd}/{len(jobs)}")
    rem_mean = sum(j["est_mean_s"] for j in rem)
    out(f"remaining on the reference GPU: {_hms(rem_mean)} (mean estimate)")
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        out(f"speed on THIS machine: segments took {med:.2f}x the reference estimate (median of {len(ratios)}); "
            f"projected remaining {_hms(rem_mean * med)}  (finish about {(_dt.datetime.now() + _dt.timedelta(seconds=rem_mean * med)).strftime('%m-%d %H:%M')})")
    else:
        out("no finished segment yet: no speed measurement (run scripts/d2_target_check.sh for an estimate)")
    hist = os.path.join(root, RUNLOG_REL, "queue_history.jsonl")
    if os.path.exists(hist):
        last = [json.loads(x) for x in open(hist) if x.strip()][-3:]
        out("\nlast runs (results/direction2/runlogs/queue_history.jsonl):")
        for r in last:
            out(f"  {r['end']}  {r['cell']} run{r['run']} g{r['g0']}-{r['g1']}  {r['result']}  exit={r['exit_code']}  {r['elapsed_s'] / 60:.1f} min")
        if last and last[-1]["result"] != "ok":
            out(f"  ATTENTION: the last run ended with '{last[-1]['result']}'; log: {last[-1]['log']}")
    if remote_branch:
        n = unpushed_count(root, *remote_branch)
        out(f"\nunpushed commits (HEAD vs {remote_branch[0]}/{remote_branch[1]}): " + (str(n) if n is not None else f"unknown ({remote_branch[0]}/{remote_branch[1]} has never been pushed/fetched from this checkout)"))
    lock = os.path.join(root, RUNLOG_REL, ".queue.lock")
    if os.path.exists(lock):
        with open(lock, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out("\nrunner: not running")
            except OSError:
                out("\nrunner: RUNNING (lock held)")


# ------------------------------------------------------------------------------------------------ pre-flight
def preflight(root: str, jobs: List[Dict[str, Any]], stage: str, selection_is_default: bool, out=print, freeze_commit: Optional[str] = _plan.FREEZE_COMMIT) -> None:
    """Same checks as the train.py entry (so that a refusal shows up here once, not as a failure of segment 1)."""
    try:
        _plan.verify_repo_state(root, freeze_commit=freeze_commit)
    except RuntimeError as e:
        raise Refused(str(e))
    if stage == "s1" and selection_is_default:
        mp = os.path.join(root, MANIFEST_REL)
        if os.path.exists(mp):
            exp = [(j["cell"], j["run"], j["g0"], j["g1"]) for j in json.load(open(mp))["jobs"]]
            got = [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs]
            if exp != got:
                out("WARNING: the segmentation differs from results/direction2/s1_queue_manifest.json (the runner uses its own, see budget.segment).")


# ------------------------------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pilot", "s1", "main"], default="s1", help="which part of the plan to run (default s1)")
    ap.add_argument("--cells", default=None, help="comma separated cell names instead of the whole stage (still executed in the order given)")
    ap.add_argument("--runs", default=None, help="comma separated run ids (default: all run_ids of the cell)")
    ap.add_argument("--dry-run", action="store_true", help="only list the segments (done/todo); runs nothing")
    ap.add_argument("--status", action="store_true", help="progress, speed seen on this machine, remaining time; runs nothing")
    ap.add_argument("--max-segments", type=int, default=None, help="stop (cleanly) after this many newly run segments")
    ap.add_argument("--no-commit", action="store_true", help="do not git-commit the part/checkpoint files after each segment")
    ap.add_argument("--push-every", type=int, default=1, metavar="N", help="after every N finished+committed segments run `git push <remote> <branch>` (default 1)")
    ap.add_argument("--remote", default="origin", help="git remote for the backup push (default origin)")
    ap.add_argument("--branch", default="direction2-freeze", help="the only branch that may be pushed; must be the checked-out branch, never master/main (default direction2-freeze)")
    ap.add_argument("--no-push", action="store_true", help="never push (commits stay local)")
    ap.add_argument("--concurrent", type=int, default=1, metavar="N", help="advance the next segment of N different cells at the same time (one process, one thread + CUDA "
                    "stream per cell; bitwise identical to running them alone; implies CUDA-graph stepping).  Default 1 = the sequential runner")
    ap.add_argument("--graph", action="store_true", help="step with CUDA graphs (D2_GRAPH=1 for every segment): bitwise identical results, faster")
    ap.add_argument("--no-graph", action="store_true", help="with --concurrent N>1: eager stepping in the lanes (slower; for comparison)")
    ap.add_argument("--python", default=sys.executable, help="python used for train.py (default: the one running this script)")
    ap.add_argument("--root", default=REPO, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = os.path.realpath(a.root)
    pl = _plan.load_plan(os.path.join(root, PLAN_REL), os.path.join(root, _plan.DEFAULT_FREEZE))     # sha256 must equal FREEZE.md
    try:
        runs = [int(x) for x in a.runs.split(",")] if a.runs else None
        jobs = build_jobs(pl, a.stage, a.cells.split(",") if a.cells else None, runs)
    except Refused as e:
        print("refused:", e)
        return EXIT_REFUSED
    if a.dry_run:
        print_jobs(root, jobs)
        return EXIT_OK
    if a.status:
        print_status(root, jobs, remote_branch=(a.remote, a.branch))
        return EXIT_OK
    rerun = "python3 scripts/d2_run_queue.py " + " ".join(x for x in (argv if argv is not None else sys.argv[1:]) if x != "--max-segments").strip()
    push_every = 0 if (a.no_push or a.no_commit) else a.push_every
    if push_every < 0:
        print("REFUSED: --push-every must be >= 1 (use --no-push to disable pushing)")
        return EXIT_REFUSED
    if push_every:
        try:
            push_guard(root, a.remote, a.branch)
        except Refused as e:
            print(f"REFUSED: {e}\n(nothing was run.  Check out {a.branch} or start with --no-push.)")
            return EXIT_REFUSED
    try:
        preflight(root, jobs, a.stage, not a.cells and runs is None)
    except Refused as e:
        print(f"REFUSED: {e}\n(train.py would refuse in the same way; nothing was run.  See deploy/README_zh.md, 'git 狀態'.)")
        return EXIT_REFUSED
    os.makedirs(os.path.join(root, RUNLOG_REL), exist_ok=True)
    lockf = open(os.path.join(root, RUNLOG_REL, ".queue.lock"), "a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("REFUSED: another d2_run_queue.py is already running on this repository (two GPU jobs at once are not allowed).")
        return EXIT_REFUSED

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    if a.graph:
        os.environ["D2_GRAPH"] = "1"                                  # inherited by every train.py segment process (arbitration.rl.graphsim.from_env)
    if a.concurrent < 1:
        print("REFUSED: --concurrent must be >= 1")
        return EXIT_REFUSED
    if a.concurrent > 1:
        return run_queue_concurrent(root, jobs, a.concurrent, commit=not a.no_commit, max_segments=a.max_segments, python=a.python,
                                    rerun_hint=rerun.strip(), push_every=push_every, remote=a.remote, branch=a.branch, graph=not a.no_graph)
    return run_queue(root, jobs, commit=not a.no_commit, max_segments=a.max_segments, python=a.python, rerun_hint=rerun.strip(),
                     push_every=push_every, remote=a.remote, branch=a.branch)


if __name__ == "__main__":
    sys.exit(main())
