"""Plan B: several cells advance at the same time in one process (threads + CUDA streams): every cell's output must be bitwise what it is alone.

The training function, part files and checkpoints are the production ones (train.train_segment); only `train.prepare` is replaced by a small
synthetic cell (the real plan cells are lam=256 x n_S=256, far too big for a unit test; prepare's guards are covered by test_d2_plan/hardening).
cpu = GraphSim(capture=False) lanes in plain threads; gpu = real CUDA graphs on one stream per lane.
"""
import json
import os
import threading

import numpy as np
import pytest

from conftest import cuda_required
from arbitration.rl import multitrain, train

DEVS = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="gpu", marks=pytest.mark.gpu)]
CELLS = {"cellA": ("O1", 0.6), "cellB": ("D1", 0.9), "cellC": ("D4", 0.3)}
CFG = dict(p=0.5, tol=1.0, L=8.0, kref=0.5)


def _fake_prepare(dev, root, lam=6, n_S=4, T=48):
    def prepare(a, graph=None):
        op, knob = CELLS[a.cell]
        gkw = {} if graph is None else dict(graph=graph)
        sim = train.make_sim_fn("M3C", T, dict(CFG), 0.5, dev, None, 25, 2, "public", op, knob, **gkw)
        val = train.make_val_fn("M3C", 40, dict(CFG), 0.5, dev, 25, 2, "public", op, knob, **gkw)
        cell = dict(name=a.cell, n=55, lam=lam, n_S=n_S, T_score=T - 8, T_train=T, sigma0=0.3, val_every=2, patience=None)
        return dict(cell=cell, run=a.run, g0=a.g0, g1=a.g1, sim_fn=sim, val_fn=val, theta0=None, root=root)
    return prepare


def _part(root, cell, g0, g1):
    d = json.load(open(os.path.join(root, "results", "direction2", "parts", f"{cell}__run0__g{g0:04d}-{g1:04d}.json")))
    for k in ("wall_s", "t_step_ms", "git_sha"):
        d.pop(k)
    return d


def _ckpt(root, cell):
    return train._load_ckpt(os.path.join(root, "results", "direction2", "ckpt", f"{cell}__run0.npz"))


def _lanes_cfg(dev):
    return dict(graph=True, K=4, capture=(dev == "cuda"), dev=dev)


@pytest.mark.parametrize("dev", DEVS)
def test_concurrent_cells_equal_each_cell_alone_eager(dev, tmp_path, monkeypatch):
    if dev == "cuda":
        cuda_required()
    solo = str(tmp_path / "solo")
    both = str(tmp_path / "both")
    for r in (solo, both):
        os.makedirs(r)
    # references: every cell alone, eager (no graph, no threads)
    monkeypatch.setattr(train, "prepare", _fake_prepare(dev, solo))
    for cell in CELLS:
        res = multitrain.run_concurrent([dict(cell=cell, run=0, g0=0, g1=4)], graph=False, dev=dev)
        assert res[0]["ok"], res[0].get("error")
    # all three at the same time, graph stepping, one thread + stream each; two consecutive segments (the lanes persist in between)
    monkeypatch.setattr(train, "prepare", _fake_prepare(dev, both))
    lanes = {}
    for g0, g1 in ((0, 2), (2, 4)):
        res = multitrain.run_concurrent([dict(cell=c, run=0, g0=g0, g1=g1) for c in CELLS], lanes=lanes, **_lanes_cfg(dev))
        assert all(r["ok"] for r in res), [r.get("error") for r in res]
    for cell in CELLS:
        a, b = _ckpt(solo, cell), _ckpt(both, cell)
        assert a == b and a["gen"] == 4 and a["records"], cell
    # the two-segment run's cumulative fields equal the one-segment reference (part files differ only in segment bookkeeping)
    for cell in CELLS:
        ref, got = _part(solo, cell, 0, 4), _part(both, cell, 2, 4)
        for k in ("G_mean_curve", "G_best_curve", "sigma_curve", "val_records", "best", "mean_theta", "sigma", "fitness_mean", "fitness_best", "gens_done"):
            assert ref[k] == got[k], (cell, k)
    assert {n: len(l.gs.cache) for n, l in lanes.items()} == {c: 2 for c in CELLS}     # one training graph + one validation graph per lane, reused by segment 2


@pytest.mark.parametrize("dev", DEVS)
def test_one_failing_lane_does_not_disturb_the_others(dev, tmp_path, monkeypatch):
    if dev == "cuda":
        cuda_required()
    root = str(tmp_path)
    ref_root = str(tmp_path / "ref")
    os.makedirs(ref_root)
    monkeypatch.setattr(train, "prepare", _fake_prepare(dev, ref_root))
    multitrain.run_concurrent([dict(cell="cellB", run=0, g0=0, g1=4)], graph=False, dev=dev)
    good = _fake_prepare(dev, root)

    def prepare(a, graph=None):
        if a.cell == "cellA":
            raise SystemExit("refused by a guard")
        return good(a, graph)
    monkeypatch.setattr(train, "prepare", prepare)
    res = multitrain.run_concurrent([dict(cell="cellA", run=0, g0=0, g1=4), dict(cell="cellB", run=0, g0=0, g1=4)], **_lanes_cfg(dev))
    assert not res[0]["ok"] and "refused by a guard" in res[0]["error"]
    assert res[1]["ok"] and _ckpt(root, "cellB") == _ckpt(ref_root, "cellB")


def test_same_cell_twice_is_refused():
    with pytest.raises(ValueError):
        multitrain.run_concurrent([dict(cell="cellA", run=0, g0=0, g1=2), dict(cell="cellA", run=0, g0=2, g1=4)], graph=False, dev="cpu")


def test_serve_protocol_runs_segments_and_reports_failures(tmp_path):
    """The long-lived worker: JSON lines in, one 'done' line per segment out; a refused segment is reported, not fatal; one running segment per cell."""
    import io
    release = threading.Event()
    calls = []

    def fake(msg, lane, dev, log):
        calls.append(msg["cell"])
        log("started " + msg["cell"])
        if msg["cell"] == "slow":
            release.wait(5)
        if msg["cell"] == "bad":
            raise RuntimeError("boom")
        return dict(status="done")
    lines = [dict(op="run", id=1, cell="slow", run=0, g0=0, g1=2, log=str(tmp_path / "slow.log")),
             dict(op="run", id=2, cell="slow", run=0, g0=2, g1=4),                 # same cell while running: refused
             dict(op="run", id=3, cell="bad", run=0, g0=0, g1=2),
             dict(op="run", id=4, cell="ok", run=0, g0=0, g1=2)]

    class Feed:
        def __iter__(self_):
            for m in lines:
                yield json.dumps(m)
            release.set()
            yield json.dumps(dict(op="quit"))
    out = io.StringIO()
    rc = multitrain.serve(Feed(), out, graph=False, dev="cpu", run_fn=fake)
    ev = {json.loads(x)["id"]: json.loads(x) for x in out.getvalue().splitlines()}
    assert rc == 0 and ev[1]["ok"] and ev[4]["ok"]
    assert not ev[2]["ok"] and "already has a running segment" in ev[2]["error"]
    assert not ev[3]["ok"] and "boom" in ev[3]["error"]
    assert "started slow" in open(tmp_path / "slow.log").read()


# ------------------------------------------------------------------------------------------------ orphan workers (red-team M1)
def test_serve_eof_without_quit_abandons_running_segments_but_quit_waits(tmp_path):
    """stdin EOF without 'quit' = the runner died: do not finish the running segments (a restarted runner redoes them); 'quit' waits for them."""
    import io
    release = threading.Event()
    finished = []

    def slow(msg, lane, dev, log):
        release.wait(5)
        finished.append(msg["cell"])
        return dict(status="done")
    job = json.dumps(dict(op="run", id=1, cell="slow", run=0, g0=0, g1=2))
    orphan = []
    rc = multitrain.serve(io.StringIO(job + "\n"), io.StringIO(), graph=False, dev="cpu", run_fn=slow, on_orphan=lambda: orphan.append(1))
    assert orphan == [1] and rc != 0 and finished == []                      # returned at once, did not join the running segment
    release.set()
    import time
    time.sleep(0.5)                                                          # the abandoned thread (only alive in this test process) ends now
    finished.clear()
    orphan2 = []
    rc = multitrain.serve(io.StringIO(job + "\n" + json.dumps(dict(op="quit")) + "\n"), io.StringIO(), graph=False, dev="cpu", run_fn=slow,
                          on_orphan=lambda: orphan2.append(1))
    assert rc == 0 and orphan2 == [] and finished == ["slow"]                # quit: waited for the segment


def test_worker_dies_with_its_parent_sigkill(tmp_path):
    """The real mechanism: prctl(PR_SET_PDEATHSIG) in the worker.  Parent is SIGKILLed; the child must disappear within a few seconds."""
    import signal
    import subprocess
    import sys
    import time
    pidf = tmp_path / "child.pid"
    child = "import os,sys,time; sys.path.insert(0, %r); from arbitration.rl import multitrain; multitrain.die_with_parent(); open(%r,'w').write(str(os.getpid())); time.sleep(120)" % (
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), str(pidf))
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(120)" % child
    p = subprocess.Popen([sys.executable, "-c", parent])
    for _ in range(300):
        if pidf.exists() and pidf.read_text():
            break
        time.sleep(0.1)
    cpid = int(pidf.read_text())
    os.kill(cpid, 0)                                                        # alive
    p.send_signal(signal.SIGKILL)
    p.wait()
    for _ in range(100):
        try:
            os.kill(cpid, 0)
            with open(f"/proc/{cpid}/stat") as f:
                if f.read().split(")")[1].split()[0] == "Z":
                    break
        except (ProcessLookupError, FileNotFoundError):
            break
        time.sleep(0.1)
    else:
        os.kill(cpid, signal.SIGKILL)
        pytest.fail("child survived its parent")
