"""scripts/d2_run_queue.py (the gpujob-free resumable runner): segment order, skipping finished segments, resume after an interruption,
stop at the first failure, per-segment git commit, compatibility with the train.py entry guard.  CPU only; train.py is replaced by a fake runner."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from tests.direction2.conftest import ROOT, git, make_git_root

_spec = importlib.util.spec_from_file_location("d2_run_queue", os.path.join(ROOT, "scripts", "d2_run_queue.py"))
q = importlib.util.module_from_spec(_spec)
sys.modules["d2_run_queue"] = q
_spec.loader.exec_module(q)

PLAN = json.load(open(os.path.join(ROOT, "results", "direction2", "run_plan.json")))


def _fake_runner(calls, fail_at=None, interrupt_at=None, status="done"):
    """Stands in for train.py: writes the part file and the checkpoint like train_segment does."""
    def run(job, cmd, logfile, root):
        calls.append((job["cell"], job["run"], job["g0"], job["g1"]))
        if interrupt_at is not None and len(calls) == interrupt_at:
            raise KeyboardInterrupt
        with open(logfile, "a") as f:
            f.write("gen 1 fake\n")
        if fail_at is not None and len(calls) == fail_at:
            return 1
        rec = dict(cell=job["cell"], run_id=job["run"], gen0=job["g0"], gen1=job["g1"], status=status, wall_s=1.0)
        p = q.part_file(root, job)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(rec, open(p, "w"))
        c = q.ckpt_file(root, job)
        os.makedirs(os.path.dirname(c), exist_ok=True)
        open(c, "wb").write(b"ckpt-%d" % job["g1"])
        return 0
    return run


@pytest.fixture
def repo(tmp_path):
    return make_git_root(tmp_path, decision="cold")


def _quiet(*a, **k):
    pass


def test_segment_order_matches_frozen_manifest():
    jobs = q.build_jobs(PLAN, "s1")
    man = json.load(open(os.path.join(ROOT, "results", "direction2", "s1_queue_manifest.json")))
    assert [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs] == [(m["cell"], m["run"], m["g0"], m["g1"]) for m in man["jobs"]]
    assert len(jobs) == 143 and "s1_NAIVE_anchor" not in {j["cell"] for j in jobs}      # the reused anchor is never queued
    for name in man["cells"]:                                                           # contiguous cover of 0..G_gens, in order
        js = [j for j in jobs if j["cell"] == name]
        assert js[0]["g0"] == 0 and all(a["g1"] == b["g0"] for a, b in zip(js, js[1:]))
        assert js[-1]["g1"] == next(c["G_gens"] for c in PLAN["cells"] if c["name"] == name)
    assert abs(sum(j["est_mean_s"] for j in jobs) - man["total_est"]["mean_s"]) < 1e-4 * man["total_est"]["mean_s"]


def test_main_stage_and_selection():
    jobs = q.build_jobs(PLAN, "main")
    assert {j["cell"] for j in jobs} == {"main_M3C_r0.5", "main_M3C_r0.9"} and {j["run"] for j in jobs} == {0, 1, 2}
    first = [j for j in jobs if j["cell"] == "main_M3C_r0.5"]
    assert [j["run"] for j in first] == sorted(j["run"] for j in first)                  # cell -> run -> generation order
    only = q.build_jobs(PLAN, "main", ["main_M3C_r0.9"], [1])
    assert {(j["cell"], j["run"]) for j in only} == {("main_M3C_r0.9", 1)}
    with pytest.raises(q.Refused):
        q.build_jobs(PLAN, "s1", ["no_such_cell"])


def test_skips_done_segments_and_runs_in_order(repo):
    jobs = q.build_jobs(PLAN, "s1")[:6]
    calls = []
    for j in jobs[:3]:                                                   # three finished segments from an earlier session
        _fake_runner([], )(j, [], os.devnull, repo)
    assert q.run_queue(repo, jobs, runner=_fake_runner(calls), commit=False, out=_quiet) == q.EXIT_OK
    assert calls == [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs[3:]]


def test_resume_after_interruption_does_not_repeat_or_restart(repo):
    jobs = q.build_jobs(PLAN, "s1")[:6]
    calls = []
    rc = q.run_queue(repo, jobs, runner=_fake_runner(calls, interrupt_at=3), out=_quiet)
    assert rc == q.EXIT_INTR and len(calls) == 3
    assert [q.is_done(repo, j) for j in jobs] == [True, True, False, False, False, False]
    calls2 = []
    assert q.run_queue(repo, jobs, runner=_fake_runner(calls2), out=_quiet) == q.EXIT_OK
    assert calls2 == [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs[2:]]        # restarts at the interrupted segment, nothing earlier
    assert q.run_queue(repo, jobs, runner=_fake_runner([]), out=_quiet) == q.EXIT_OK    # all done: a third start runs nothing
    hist = [json.loads(x) for x in open(os.path.join(repo, q.RUNLOG_REL, "queue_history.jsonl"))]
    assert [h["result"] for h in hist].count("interrupted") == 1 and all("elapsed_s" in h and "exit_code" in h and "start" in h and "end" in h for h in hist)


def test_stops_at_first_failure_and_prints_rerun_command(repo):
    jobs = q.build_jobs(PLAN, "s1")[:6]
    calls, lines = [], []
    rc = q.run_queue(repo, jobs, runner=_fake_runner(calls, fail_at=2), out=lines.append, rerun_hint="python3 scripts/d2_run_queue.py --stage s1")
    assert rc == q.EXIT_FAIL and len(calls) == 2                                          # nothing after the failed segment was started
    assert not q.is_done(repo, jobs[1]) and not q.is_done(repo, jobs[2])
    text = "\n".join(lines)
    assert "STOPPED" in text and "exited with code 1" in text and "python3 scripts/d2_run_queue.py --stage s1" in text
    assert "--g0 %d --g1 %d" % (jobs[1]["g0"], jobs[1]["g1"]) in text
    logs = [f for f in os.listdir(os.path.join(repo, q.RUNLOG_REL)) if f.endswith(".log")]
    assert len(logs) == 2
    body = open(os.path.join(repo, q.RUNLOG_REL, sorted(logs)[-1])).read()
    assert "# START" in body and "# END" in body and "# EXIT_CODE 1" in body and "# ELAPSED_S" in body


def test_exit0_without_done_part_counts_as_failure(repo):
    jobs = q.build_jobs(PLAN, "s1")[:3]
    calls = []
    assert q.run_queue(repo, jobs, runner=_fake_runner(calls, status="partial"), out=_quiet) == q.EXIT_FAIL and len(calls) == 1


def test_commit_per_segment_and_trailer(repo):
    jobs = q.build_jobs(PLAN, "s1")[:3]
    n0 = int(git(repo, "rev-list", "--count", "HEAD"))
    assert q.run_queue(repo, jobs, runner=_fake_runner([]), out=_quiet) == q.EXIT_OK
    assert int(git(repo, "rev-list", "--count", "HEAD")) == n0 + 3
    msg = git(repo, "log", "-1", "--format=%B")
    assert "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" in msg
    assert "Claude-Session: https://claude.ai/code/session_01Vcin4FdSJwvFyTg4u3RSiU" in msg
    status = git(repo, "status", "--porcelain", "--", "results/direction2/parts", "results/direction2/ckpt")
    assert status == ""                                                                    # parts and checkpoint are all committed
    assert git(repo, "show", "--stat", "--format=", "HEAD").count("results/direction2/") == 2       # only this segment's part + the run's ckpt


def test_commit_does_not_sweep_in_other_staged_files(repo):
    jobs = q.build_jobs(PLAN, "s1")[:1]
    open(os.path.join(repo, "other.txt"), "w").write("x")
    git(repo, "add", "other.txt")
    assert q.run_queue(repo, jobs, runner=_fake_runner([]), out=_quiet) == q.EXIT_OK
    assert "other.txt" not in git(repo, "show", "--stat", "--format=", "HEAD")
    assert git(repo, "diff", "--cached", "--name-only") == "other.txt"


def test_unfinished_commit_is_made_up_on_restart(repo):
    jobs = q.build_jobs(PLAN, "s1")[:2]
    _fake_runner([])(jobs[0], [], os.devnull, repo)                                        # segment finished, process died before the commit
    assert git(repo, "status", "--porcelain", "--", "results/direction2/parts")
    assert q.run_queue(repo, jobs, runner=_fake_runner([]), out=_quiet) == q.EXIT_OK
    assert git(repo, "status", "--porcelain", "--", "results/direction2/parts", "results/direction2/ckpt") == ""


def test_max_segments(repo):
    jobs = q.build_jobs(PLAN, "s1")[:5]
    calls = []
    assert q.run_queue(repo, jobs, runner=_fake_runner(calls), commit=False, max_segments=2, out=_quiet) == q.EXIT_OK and len(calls) == 2


def test_train_entry_guard_accepts_untracked_results_but_not_a_changed_plan(repo):
    """train.py's guard (plan.verify_repo_state) looks only at run_plan.json and FREEZE.md: untracked result files and commits of results pass."""
    from arbitration.rl import plan as P
    jobs = q.build_jobs(PLAN, "s1")[:2]
    q.run_queue(repo, jobs[:1], runner=_fake_runner([]), commit=False, out=_quiet)         # untracked part + ckpt in the tree
    assert git(repo, "status", "--porcelain").count("??") >= 1
    P.verify_repo_state(repo, freeze_commit=None)
    q.run_queue(repo, jobs, runner=_fake_runner([]), commit=True, out=_quiet)               # committed results
    P.verify_repo_state(repo, freeze_commit=None)
    open(os.path.join(repo, "results", "direction2", "run_plan.json"), "a").write(" ")
    with pytest.raises(RuntimeError):
        P.verify_repo_state(repo, freeze_commit=None)


def test_dry_run_and_status_print(repo, capsys):
    jobs = q.build_jobs(PLAN, "s1")[:4]
    _fake_runner([])(jobs[0], [], os.devnull, repo)
    lines = []
    q.print_jobs(repo, jobs, out=lines.append)
    assert lines[0].split()[1] == "done" and lines[1].split()[1] == "todo" and "3 to run" in lines[-1]
    lines = []
    q.print_status(repo, jobs, out=lines.append)
    assert any("segments done 1/4" in x for x in lines)


# ------------------------------------------------------------------------------------------------ auto push (git is mocked: nothing is ever pushed)
class _Push:
    """Stands in for `git push`: records the commands, fails on the attempts listed in fail_on (1-based)."""
    def __init__(self, fail_on=()):
        self.cmds, self.fail_on = [], set(fail_on)

    def __call__(self, root, cmd):
        self.cmds.append(list(cmd))
        bad = len(self.cmds) in self.fail_on
        return subprocess.CompletedProcess(cmd, 1 if bad else 0, "", "ssh: Could not resolve hostname" if bad else "")


@pytest.fixture
def prepo(repo):
    git(repo, "checkout", "-q", "-b", "direction2-freeze")
    return repo


def _run(root, n, pusher, **kw):
    jobs = q.build_jobs(PLAN, "s1")[:n]
    lines = []
    rc = q.run_queue(root, jobs, runner=kw.pop("runner", _fake_runner([])), out=lines.append, pusher=pusher, **kw)
    return rc, lines


def test_push_every_segment_by_default_setting(prepo):
    p = _Push()
    rc, _ = _run(prepo, 3, p, push_every=1)
    assert rc == q.EXIT_OK and p.cmds == [["git", "push", "origin", "direction2-freeze"]] * 3


def test_push_frequency_every_n_and_final_flush(prepo):
    p = _Push()
    _run(prepo, 5, p, push_every=2)
    assert len(p.cmds) == 3                                   # after segments 2 and 4, and the final flush for segment 5
    p2 = _Push()
    _run(prepo, 4, p2, push_every=2, commit=True)           # segments 1-4 are already done: nothing new, nothing pushed
    assert p2.cmds == []
    p3 = _Push()
    jobs = q.build_jobs(PLAN, "s1")[:6]
    q.run_queue(prepo, jobs, runner=_fake_runner([]), out=_quiet, pusher=p3, push_every=3)
    assert len(p3.cmds) == 1                                  # segments 5,6 of 6 ran now (1-4 done): 2 < 3 -> only the final flush


def test_push_every_zero_or_no_commit_never_pushes(prepo):
    p = _Push()
    _run(prepo, 2, p, push_every=0)
    assert p.cmds == []
    p = _Push()
    jobs = q.build_jobs(PLAN, "s1")[3:5]
    q.run_queue(prepo, jobs, runner=_fake_runner([]), commit=False, out=_quiet, pusher=p, push_every=1)
    assert p.cmds == []                                       # nothing committed -> nothing to push


def test_failed_push_does_not_stop_training_and_is_retried(prepo):
    p = _Push(fail_on=(1, 2))
    calls = []
    jobs = q.build_jobs(PLAN, "s1")[:4]
    lines = []
    rc = q.run_queue(prepo, jobs, runner=_fake_runner(calls), out=lines.append, pusher=p, push_every=2)
    assert rc == q.EXIT_OK and len(calls) == 4 and all(q.is_done(prepo, j) for j in jobs)    # training ran to the end
    # segment 2: push 1 fails; segment 3: retry (push 2) fails; segment 4: retry (push 3) succeeds; nothing left for the final flush
    assert len(p.cmds) == 3
    assert sum("FAILED" in x for x in lines) == 2 and sum("PUSH origin direction2-freeze: ok" in x for x in lines) == 1
    log = open(os.path.join(prepo, q.RUNLOG_REL, "push.log")).read()
    assert log.count("FAILED") == 2 and "Could not resolve hostname" in log


def test_push_exception_is_swallowed(prepo):
    def boom(root, cmd):
        raise subprocess.TimeoutExpired(cmd, 1)
    rc, lines = _run(prepo, 2, boom, push_every=1)
    assert rc == q.EXIT_OK and any("TimeoutExpired" in x for x in lines)


def test_push_never_contains_force_or_refspec_tricks(prepo):
    p = _Push(fail_on=(2,))
    _run(prepo, 4, p, push_every=1)
    assert p.cmds
    for cmd in p.cmds:
        assert cmd == ["git", "push", "origin", "direction2-freeze"]
        assert not any(a in ("--force", "-f", "--force-with-lease", "--mirror", "--delete", "--all") or a.startswith("--force") or a.startswith("+") for a in cmd)
    for bad in ("--force", "-f", "+direction2-freeze", "a:b", ":direction2-freeze", "x y", ""):
        with pytest.raises(q.Refused):
            q.push_cmd("origin", bad)
        with pytest.raises(q.Refused):
            q.push_cmd(bad, "direction2-freeze")


def test_real_subprocess_push_command_is_plain_git_push(prepo, monkeypatch):
    seen = []
    monkeypatch.setattr(q.subprocess, "run", lambda argv, **kw: (seen.append(argv), subprocess.CompletedProcess(argv, 0, "", ""))[1])
    assert q.try_push(prepo, "origin", "direction2-freeze", out=_quiet, git=lambda root, *a: subprocess.CompletedProcess(a, 0, "direction2-freeze\n", ""))
    assert seen == [["git", "-C", prepo, "push", "origin", "direction2-freeze"]]


@pytest.mark.parametrize("branch", ["master", "main"])
def test_never_pushes_master_or_main(prepo, branch):
    p = _Push()
    git(prepo, "checkout", "-q", "-B", branch)               # now ON master/main
    rc, lines = _run(prepo, 1, p, push_every=1, branch="direction2-freeze")      # being on master/main while --branch is the direction2 branch
    assert rc == q.EXIT_OK and p.cmds == [] and any("refusing to push" in x for x in lines)      # training goes on, nothing is pushed
    git(prepo, "checkout", "-q", "-B", "direction2-freeze")
    rc, lines = _run(prepo, 2, p, push_every=1, branch=branch)                      # asking for master/main from the direction2 branch
    assert p.cmds == [] and any("protected" in x for x in lines)
    for cur in (branch, "direction2-freeze"):
        with pytest.raises(q.Refused):
            q.push_guard(prepo, "origin", branch)
        git(prepo, "checkout", "-q", "-B", cur)


def test_branch_mismatch_is_refused(prepo):
    git(prepo, "checkout", "-q", "-b", "some-other-branch")
    with pytest.raises(q.Refused, match="some-other-branch"):
        q.push_guard(prepo, "origin", "direction2-freeze")
    p = _Push()
    _, lines = _run(prepo, 2, p, push_every=1)
    assert p.cmds == [] and any("refusing to push" in x for x in lines)


def test_main_refuses_to_start_on_wrong_branch_unless_no_push(prepo, capsys):
    git(prepo, "checkout", "-q", "-b", "wip")
    rc = q.main(["--root", prepo, "--stage", "s1", "--cells", "s1_D1_s0.02", "--runs", "0", "--max-segments", "0"])
    assert rc == q.EXIT_REFUSED and "refusing to push" in capsys.readouterr().out
    rc = q.main(["--root", prepo, "--stage", "s1", "--cells", "s1_D1_s0.02", "--runs", "0", "--max-segments", "0", "--no-push"])
    assert "refusing to push" not in capsys.readouterr().out              # (the tmp repo then stops at the freeze-commit check, which is unrelated)
    rc = q.main(["--root", prepo, "--branch", "master", "--max-segments", "0"])
    assert rc == q.EXIT_REFUSED


def test_status_shows_unpushed_commits(prepo):
    jobs = q.build_jobs(PLAN, "s1")[:3]
    q.run_queue(prepo, jobs, runner=_fake_runner([]), out=_quiet)
    lines = []
    q.print_status(prepo, jobs, out=lines.append, remote_branch=("origin", "direction2-freeze"))
    assert any("unpushed commits" in x and "unknown" in x for x in lines)               # remote ref never seen
    git(prepo, "update-ref", "refs/remotes/origin/direction2-freeze", "HEAD~2")           # pretend the last push was 2 commits ago (local ref only)
    lines = []
    q.print_status(prepo, jobs, out=lines.append, remote_branch=("origin", "direction2-freeze"))
    assert any(x.endswith("origin/direction2-freeze): 2") for x in lines)


# ------------------------------------------------------------------------------------------------ --concurrent N (docs/direction2-speedup-plan.md, plan B)
class _FakeWorker:
    """Deterministic stand-in for the multitrain worker: jobs sit in flight until `events.get()` completes one (policy = which one)."""

    def __init__(self, pick="first", fail_ids=(), crash_after=None, status="done"):
        self.inflight, self.started, self.max_inflight, self.fail_ids = [], [], 0, set(fail_ids)
        self.pick, self.events, self.closed, self.killed, self.status = pick, self, False, False, status
        self.completed = 0
        self.crash_after = crash_after

    def submit(self, jid, job, logfile):
        assert all(j["cell"] != job["cell"] for _, j, _ in self.inflight), "two segments of one cell in flight"
        self.inflight.append((jid, job, logfile))
        self.started.append((job["cell"], job["run"], job["g0"], job["g1"]))
        self.max_inflight = max(self.max_inflight, len(self.inflight))

    def get(self):
        if self.crash_after is not None and self.completed >= self.crash_after:
            return {"event": "eof", "returncode": -9}
        jid, job, logfile = self.inflight.pop(0 if self.pick == "first" else -1)
        self.completed += 1
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(logfile))))
        if jid in self.fail_ids:
            return {"event": "done", "id": jid, "ok": False, "error": "boom"}
        rec = dict(cell=job["cell"], run_id=job["run"], gen0=job["g0"], gen1=job["g1"], status=self.status, wall_s=1.0)
        p = q.part_file(root, job)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(rec, open(p, "w"))
        c = q.ckpt_file(root, job)
        os.makedirs(os.path.dirname(c), exist_ok=True)
        open(c, "wb").write(b"ckpt-%d" % job["g1"])
        return {"event": "done", "id": jid, "ok": True, "status": self.status}

    def close(self):
        self.closed = True

    def kill(self):
        self.killed = True


def _conc(repo, jobs, n, worker, **kw):
    lines = []
    rc = q.run_queue_concurrent(repo, jobs, n, worker_factory=lambda root, k: worker, out=lines.append, **kw)
    return rc, lines


def test_concurrent_runs_distinct_cells_together_and_keeps_every_cell_in_order(repo):
    jobs = q.build_jobs(PLAN, "s1")[:34]
    cells = list(dict.fromkeys(j["cell"] for j in jobs))
    assert len(cells) >= 3
    w = _FakeWorker(pick="last")
    rc, _ = _conc(repo, jobs, 3, w)
    assert rc == q.EXIT_OK and w.max_inflight == 3 and w.closed
    assert sorted(w.started) == sorted((j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs)         # same segment set as the sequential runner
    for c in cells:                                                                                     # per cell: strictly in manifest order
        mine = [s for s in w.started if s[0] == c]
        assert mine == [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs if j["cell"] == c]
    assert all(q.is_done(repo, j) for j in jobs)
    assert int(git(repo, "rev-list", "--count", "HEAD")) >= 34                                           # one commit per segment
    assert git(repo, "status", "--porcelain", "--", "results/direction2/parts", "results/direction2/ckpt") == ""
    assert git(repo, "show", "--stat", "--format=", "HEAD").count("results/direction2/") == 2          # a commit holds one part + that run's ckpt


def test_concurrent_one_is_the_sequential_order_and_n1_main_path_unchanged(repo, monkeypatch):
    jobs = q.build_jobs(PLAN, "s1")[:6]
    w = _FakeWorker()
    rc, _ = _conc(repo, jobs, 1, w)
    assert rc == q.EXIT_OK and w.max_inflight == 1 and w.started == [(j["cell"], j["run"], j["g0"], j["g1"]) for j in jobs]
    called = {}
    monkeypatch.setattr(q, "run_queue", lambda *a, **k: called.setdefault("seq", 0) or 0)
    monkeypatch.setattr(q, "run_queue_concurrent", lambda *a, **k: called.setdefault("conc", 1) or 1)
    monkeypatch.setattr(q, "preflight", lambda *a, **k: None)
    q.main(["--root", repo, "--no-push"])
    assert "seq" in called and "conc" not in called                                                     # the default is the untouched sequential runner
    called.clear()
    q.main(["--root", repo, "--no-push", "--concurrent", "2"])
    assert "conc" in called


def test_concurrent_skips_finished_segments_and_late_commits(repo):
    jobs = q.build_jobs(PLAN, "s1")[:12]
    for j in jobs[:4]:
        _fake_runner([])(j, [], os.devnull, repo)
    w = _FakeWorker()
    rc, _ = _conc(repo, jobs, 2, w)
    assert rc == q.EXIT_OK and w.started and all((j["cell"], j["run"], j["g0"], j["g1"]) not in w.started for j in jobs[:4])
    assert len(w.started) == 8


def test_concurrent_failure_stops_new_segments_but_commits_the_running_ones(repo):
    jobs = q.build_jobs(PLAN, "s1")[:34]
    w = _FakeWorker(pick="last", fail_ids={2})          # the second segment started fails first (while the first one is still running)
    rc, lines = _conc(repo, jobs, 2, w, rerun_hint="python3 scripts/d2_run_queue.py --concurrent 2")
    assert rc == q.EXIT_FAIL
    text = "\n".join(lines)
    assert "STOPPED" in text and "boom" in text and "--concurrent 2" in text
    assert w.completed == 2 and len(w.started) == 2 and w.closed                         # nothing started after the failure; the running one finished
    done = [j for j in jobs if q.is_done(repo, j)]
    assert len(done) == 1 and (done[0]["cell"], done[0]["run"], done[0]["g0"], done[0]["g1"]) == w.started[0]
    assert git(repo, "status", "--porcelain", "--", "results/direction2/parts", "results/direction2/ckpt") == ""      # the finished one IS committed
    hist = [json.loads(x) for x in open(os.path.join(repo, q.RUNLOG_REL, "queue_history.jsonl"))]
    assert sorted(h["result"] for h in hist) == ["failed", "ok"]
    w2 = _FakeWorker()                                  # restart: the failed segment (and everything after it) runs, the finished one is skipped
    rc, _ = _conc(repo, jobs, 2, w2)
    assert rc == q.EXIT_OK and w.started[0] not in w2.started and w.started[1] in w2.started


def test_concurrent_part_not_done_counts_as_failure(repo):
    jobs = q.build_jobs(PLAN, "s1")[:34]
    w = _FakeWorker(status="partial")
    rc, lines = _conc(repo, jobs, 2, w)
    assert rc == q.EXIT_FAIL and "not 'done'" in "\n".join(lines) and w.completed == 2 and len(w.started) == 2


def test_concurrent_worker_crash_is_a_failure(repo):
    jobs = q.build_jobs(PLAN, "s1")[:12]
    w = _FakeWorker(crash_after=1)
    rc, lines = _conc(repo, jobs, 2, w)
    assert rc == q.EXIT_FAIL and "worker process exited" in "\n".join(lines)
    assert len([j for j in jobs if q.is_done(repo, j)]) == 1


def test_concurrent_max_segments_counts_started_segments(repo):
    jobs = q.build_jobs(PLAN, "s1")[:34]
    w = _FakeWorker()
    rc, lines = _conc(repo, jobs, 3, w, max_segments=5)
    assert rc == q.EXIT_OK and len(w.started) == 5 and sum(q.is_done(repo, j) for j in jobs) == 5
    assert "--max-segments" in "\n".join(lines) and "all 34 segments" not in "\n".join(lines)


def test_concurrent_interrupt_is_resumable(repo):
    jobs = q.build_jobs(PLAN, "s1")[:12]

    class Intr(_FakeWorker):
        def get(self):
            if self.completed >= 2:
                raise KeyboardInterrupt
            return super().get()
    w = Intr()
    rc, lines = _conc(repo, jobs, 2, w)
    assert rc == q.EXIT_INTR and w.killed and "Nothing is lost" in "\n".join(lines)
    done_now = [j for j in jobs if q.is_done(repo, j)]
    assert len(done_now) == 2
    w2 = _FakeWorker()
    assert _conc(repo, jobs, 2, w2)[0] == q.EXIT_OK and len(w2.started) == 10


def test_concurrent_pushes_after_each_committed_segment(prepo):
    jobs = q.build_jobs(PLAN, "s1")[:8]
    p = _Push()
    w = _FakeWorker()
    rc, _ = _conc(prepo, jobs, 2, w, push_every=1, pusher=p)
    assert rc == q.EXIT_OK and p.cmds == [["git", "push", "origin", "direction2-freeze"]] * 8


def test_concurrent_option_validation(repo, capsys):
    assert q.main(["--root", repo, "--no-push", "--concurrent", "0"]) in (q.EXIT_REFUSED, q.EXIT_FAIL)
