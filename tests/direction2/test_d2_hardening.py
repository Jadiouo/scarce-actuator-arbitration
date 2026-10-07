"""Pre-S1 hardening (T-81..T-88): locked training entry, git/registration guards, conditional cells, warm start in the cell's own environment,
NAIVE-anchor reuse, M-3 record.  All CPU.  Spec basis: REQ-S1-14, REQ-S1-22, REQ-OPT-07, REQ-MEAS-03; notes 'Red team (training wiring and plan)'."""
import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, git, make_git_root
from arbitration.rl import plan as P, train

PLAN = os.path.join(ROOT, "results", "direction2", "run_plan.json")


def _plan():
    return json.load(open(PLAN, encoding="utf-8"))


def _base(root):
    return ["--test-mode", "--root", root, "--plan", "results/direction2/run_plan.json", "--dev", "cpu"]


@pytest.fixture
def captured(monkeypatch):
    got = {}
    monkeypatch.setattr(train, "make_sim_fn", lambda *a, **k: got.__setitem__("sim", (a, k)) or (lambda th, s: None))
    monkeypatch.setattr(train, "make_val_fn", lambda *a, **k: got.__setitem__("val", (a, k)) or (lambda th: 0.0))
    monkeypatch.setattr(train, "train_segment", lambda cell, run, g0, g1, sim, val, **k: got.update(cell=cell, run=run, g0=g0, g1=g1, k=k, theta0=k.get("theta0"))
                        or dict(cell=cell["name"], run_id=run, gen0=g0, gen1=g1, status="done"))
    return got


def _cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "arbitration.rl.train", *args], cwd=cwd, capture_output=True, text=True)


# ---------------------------------------------------------------------------------------------------------------- T-81 locked entry
def test_T81_entry_without_plan_or_with_other_paths_is_refused(tmp_path):
    """T-81: no --plan -> refused (old branch removed); --plan/--freeze/--root other than the repo defaults -> refused (the red-team bypasses: old
    entry, fake plan, fake FREEZE, fake root); the budget flags of the old branch do not exist; --test-mode needs a root OUTSIDE the repo."""
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--dev", "cpu")                          # old branch: no --plan
    assert r.returncode != 0 and "--plan is required" in r.stderr
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--mech", "M3C", "--lam", "8", "--nS", "2", "--T", "100")      # old budget flags
    assert r.returncode != 0 and "unrecognized arguments" in r.stderr
    fake = make_git_root(tmp_path, decision="cold", name="fake")
    fp = os.path.join(fake, "results", "direction2", "run_plan.json")
    ff = os.path.join(fake, "results", "direction2", "FREEZE.md")
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", fp, "--dev", "cpu")           # fake plan (absolute path)
    assert r.returncode != 0 and "--plan must be" in r.stderr
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", "results/direction2/run_plan.json", "--freeze", ff, "--dev", "cpu")   # fake FREEZE
    assert r.returncode != 0 and "--freeze must be" in r.stderr
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", "results/direction2/run_plan.json", "--root", fake, "--dev", "cpu")  # fake root
    assert r.returncode != 0 and "--root must be the repository" in r.stderr
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", "../../etc/run_plan.json", "--dev", "cpu")            # relative escape
    assert r.returncode != 0 and "--plan must be" in r.stderr
    # test mode: refused inside the repo (it could write into the real results/direction2/)
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", "results/direction2/run_plan.json", "--test-mode", "--root", ROOT, "--dev", "cpu")
    assert r.returncode != 0 and "outside the repository" in r.stderr
    r = _cli("--cell", "pilot_naive_cold", "--g1", "1", "--plan", "results/direction2/run_plan.json", "--test-mode",
             "--root", os.path.join(ROOT, "results", "direction2"), "--dev", "cpu")
    assert r.returncode != 0 and "outside the repository" in r.stderr
    # a symlink into the repo does not help either (realpath)
    link = tmp_path / "link"
    os.symlink(ROOT, link)
    with pytest.raises(SystemExit):
        train.main(["--cell", "pilot_naive_cold", "--g1", "1", "--plan", "results/direction2/run_plan.json", "--test-mode", "--root", str(link)])
    # the real test-mode run on a throw-away repo never touches the real results/direction2
    before = sorted(os.listdir(os.path.join(ROOT, "results", "direction2")))
    root = make_git_root(tmp_path, name="ok")
    assert os.path.realpath(root) != os.path.realpath(ROOT)
    assert sorted(os.listdir(os.path.join(ROOT, "results", "direction2"))) == before


# ---------------------------------------------------------------------------------------------------------------- T-82 git guard
def test_T82_uncommitted_plan_or_freeze_and_non_descendant_head_are_refused(tmp_path, captured):
    """T-82: `git diff --quiet HEAD -- plan FREEZE.md` must pass (unstaged AND staged changes, untracked files refused) and HEAD must descend from the
    freeze commit."""
    for rel in ("run_plan.json", "FREEZE.md"):
        root = make_git_root(tmp_path, name="d_" + rel)
        P.verify_repo_state(root, freeze_commit=None)                                            # clean: passes
        pth = os.path.join(root, "results", "direction2", rel)
        open(pth, "a").write("\n ")                                                              # unstaged modification
        with pytest.raises(RuntimeError, match="not committed or differs from HEAD"):
            P.verify_repo_state(root, freeze_commit=None)
        with pytest.raises(SystemExit):
            train.main(_base(root) + ["--cell", "pilot_naive_cold", "--g1", "10"])
        git(root, "add", "-A")                                                                   # staged but not committed
        with pytest.raises(RuntimeError, match="not committed or differs from HEAD"):
            P.verify_repo_state(root, freeze_commit=None)
        git(root, "commit", "-qam", "now committed")
        P.verify_repo_state(root, freeze_commit=None)
    # untracked plan
    root = make_git_root(tmp_path, name="untracked", commit=False)
    with pytest.raises(RuntimeError, match="not committed"):
        P.verify_repo_state(root, freeze_commit=None)
    # freeze-commit ancestry: HEAD descends from the first commit; an unrelated commit is not an ancestor
    root = make_git_root(tmp_path, name="anc")
    first = git(root, "rev-parse", "HEAD")
    open(os.path.join(root, "x.txt"), "w").write("x")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "second")
    P.verify_repo_state(root, freeze_commit=first)
    git(root, "checkout", "-q", "--orphan", "other")
    git(root, "commit", "-qm", "unrelated root commit")
    with pytest.raises(RuntimeError, match="not a descendant"):
        P.verify_repo_state(root, freeze_commit=first)
    # production constant: the freeze commit really is an ancestor of this checkout
    assert subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", P.FREEZE_COMMIT, "HEAD"]).returncode == 0
    assert P.FREEZE_COMMIT.startswith("a72d3d7")


# ---------------------------------------------------------------------------------------------------------------- T-83 pilot decision
def test_T83_pilot_decision_must_be_committed_and_registered(tmp_path, captured):
    """T-83: without a committed decision AND its PILOT_DECISION_SHA256 registration only pilot cells run; a fake / uncommitted / modified / sha-mismatched
    decision is refused; once registered, chosen_start cannot be switched (file change, re-registration, second commit)."""
    S1 = ["--cell", "s1_D2_s0.02", "--g1", "10"]
    # (a) fake decision file present but never committed / never registered (red-team bypass)
    root = make_git_root(tmp_path, name="a")
    open(os.path.join(root, "results", "direction2", "pilot_decision.json"), "w").write(json.dumps(dict(chosen_start="cold")))
    assert P.load_pilot_decision(root) is None
    with pytest.raises(ValueError, match="pilot"):
        train.main(_base(root) + S1)
    train.main(_base(root) + ["--cell", "pilot_naive_cold", "--g1", "10"])                         # the pilot cell is still allowed
    # (b) committed but not registered in FREEZE.md
    git(root, "add", "-A")
    git(root, "commit", "-qm", "decision, no registration")
    assert P.load_pilot_decision(root) is None
    with pytest.raises(ValueError, match="pilot"):
        train.main(_base(root) + S1)
    # (c) registered AND committed -> allowed, route from the file
    root = make_git_root(tmp_path, decision="warm", name="c")
    assert P.load_pilot_decision(root) == "warm"
    # (d) registered but the file is not committed (modified after the commit) / sha mismatch
    pd = os.path.join(root, "results", "direction2", "pilot_decision.json")
    open(pd, "w").write(json.dumps(dict(chosen_start="cold")))                                    # switch the route in the working tree
    with pytest.raises(ValueError, match="not committed|sha256"):
        P.load_pilot_decision(root)
    git(root, "commit", "-qam", "switch the route (file only)")
    with pytest.raises(ValueError, match="sha256"):
        P.load_pilot_decision(root)                                                               # committed but no longer the registered sha
    # (e) switching file AND re-registering in FREEZE.md: the second commit of the decision / the changed registration is refused
    fr = os.path.join(root, "results", "direction2", "FREEZE.md")
    txt = open(fr).read()
    old = [ln for ln in txt.splitlines() if ln.startswith("PILOT_DECISION_SHA256:")]
    assert len(old) == 1
    new_sha = hashlib.sha256(open(pd, "rb").read()).hexdigest()
    open(fr, "w").write(txt.replace(old[0], "PILOT_DECISION_SHA256: " + new_sha))
    git(root, "commit", "-qam", "re-register")
    with pytest.raises(ValueError, match="committed 2 times|changed over the history"):
        P.load_pilot_decision(root)
    with pytest.raises(ValueError):
        train.main(_base(root) + S1)
    # (f) conflicting duplicate registration lines are refused
    root = make_git_root(tmp_path, decision="cold", name="f")
    fr = os.path.join(root, "results", "direction2", "FREEZE.md")
    open(fr, "a").write("\nPILOT_DECISION_SHA256: " + "0" * 64 + "\n")
    git(root, "commit", "-qam", "dup")
    with pytest.raises(ValueError, match="conflicting"):
        P.load_pilot_decision(root)
    # (g) invalid chosen_start
    root = make_git_root(tmp_path, decision="lukewarm", name="g")
    with pytest.raises(ValueError, match="chosen_start"):
        P.load_pilot_decision(root)
    # (h) a correctly registered decision: s1 cell runs with the registered route; --start contradicting it is refused
    root = make_git_root(tmp_path, decision="cold", name="h")
    train.main(_base(root) + S1)
    assert captured["cell"]["name"] == "s1_D2_s0.02"
    with pytest.raises(ValueError, match="contradicts"):
        train.main(_base(root) + S1 + ["--start", "warm"])


# ---------------------------------------------------------------------------------------------------------------- T-84 conditional cells
def test_T84_conditional_cells_are_read_and_enforced_by_gate_sizes(tmp_path, captured):
    """T-84: the three s=0.01 S1 cells carry `conditional`; the entry reads it: runs only if 0.01 >= MDE_D,plan from the REGISTERED M-3 record (S1-22);
    otherwise refused with the reason; MDE_D,plan > 0.02 is STOP-02 (no S1 cell); no / unregistered M-3 record -> refused."""
    cells = [c for c in _plan()["cells"] if c["group"] == "s1" and c.get("conditional")]
    assert sorted(c["name"] for c in cells) == ["s1_D1_s0.01", "s1_D2_s0.01", "s1_O1_s0.01"] and all(c["size"] == 0.01 for c in cells)
    c01, c02 = ["--cell", "s1_D2_s0.01", "--g1", "10"], ["--cell", "s1_D2_s0.02", "--g1", "10"]
    root = make_git_root(tmp_path, decision="cold", name="nom3")
    with pytest.raises(ValueError, match="MDE_D,plan is not available"):
        train.main(_base(root) + c01)
    train.main(_base(root) + c02)                                                                  # 0.02 cells are unconditional
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.0058, name="ok")                       # MDE_D,plan 0.0058 -> G_s = {0.01, 0.02}
    train.main(_base(root) + c01)
    assert captured["cell"]["name"] == "s1_D2_s0.01"
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.015, name="big")                       # 0.01 < 0.015 <= 0.02 -> G_s = {0.02}
    with pytest.raises(ValueError, match=r"size 0\.01 < MDE_D,plan=0\.01500"):
        train.main(_base(root) + c01)
    train.main(_base(root) + c02)
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.01, name="edge")                       # s >= MDE: 0.01 >= 0.01 is allowed
    train.main(_base(root) + c01)
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.03, name="stop")                       # STOP-02: nothing of S1 may run
    with pytest.raises(ValueError, match="STOP-02"):
        train.main(_base(root) + c01)
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.0058, register_m3=False, name="unreg")
    with pytest.raises(ValueError, match="M3_SHA256|MDE_D,plan is not available"):
        train.main(_base(root) + c01)
    pth = os.path.join(root, "results", "direction2", "measure_m3.json")                            # tampered record after registration
    root = make_git_root(tmp_path, decision="cold", m3_mde=0.0058, name="tamper")
    pth = os.path.join(root, "results", "direction2", "measure_m3.json")
    d = json.load(open(pth))
    d["rows"][0]["MDE_D_plan"] = 0.001
    json.dump(d, open(pth, "w"))
    with pytest.raises(ValueError, match="not committed|sha256"):
        train.main(_base(root) + c01)


# ---------------------------------------------------------------------------------------------------------------- T-85 warm start env
def test_T85_warm_start_is_fitted_in_the_cells_own_environment(monkeypatch):
    """T-85: best_handwritten_on_train / warm_start_theta use vulns.simulate_vuln (operator, knob) for a vulnerability cell and env.simulate with the cell's
    mech/cfg otherwise; a strategy that is best ONLY in the vulnerable environment is the one picked there; warm_obs does not change the simulation."""
    import torch
    from arbitration.rl import env, hw, vulns
    names = [n for n, _ in hw.build_s_hw()]
    k_plain, k_vuln = 3, 20
    calls = []

    def fake_plain(mech, T, cfg, pols, seeds, *, r, dev, obs_version="public", **kw):
        calls.append(("plain", mech, dict(cfg), r))
        U = torch.zeros(len(pols), len(seeds), dtype=torch.float64)
        U[1 + k_plain] = 1.0                                                                       # best strategy in the PLAIN environment
        return dict(snap=dict(util_b=torch.stack([U, U], -1)), T_score=10)

    def fake_vuln(operator, knob, mech, T, cfg, pols, seeds, *, r, dev, **kw):
        calls.append(("vuln", operator, knob, mech))
        U = torch.zeros(len(pols), len(seeds), dtype=torch.float64)
        U[1 + k_vuln] = 1.0                                                                        # best strategy in the VULNERABLE environment only
        return dict(snap=dict(util_b=torch.stack([U, U], -1)), T_score=10)
    monkeypatch.setattr(env, "simulate", fake_plain)
    monkeypatch.setattr(vulns, "simulate_vuln", fake_vuln)
    tr = list(range(4))
    cfg = env.m3c_config(0.5)
    b_plain = train.best_handwritten_on_train(cfg, "M3C", 0.5, 100, tr, "cpu")
    b_vuln = train.best_handwritten_on_train(cfg, "M3C", 0.5, 100, tr, "cpu", operator="D4", knob=0.047)
    assert b_plain["name"] == names[k_plain] and b_vuln["name"] == names[k_vuln] and names[k_plain] != names[k_vuln]
    assert calls[0][0] == "plain" and calls[1] == ("vuln", "D4", 0.047, "M3C")
    # an anchor with its own configuration: NAIVE (p=0) and M4 are forwarded as given
    naive = dict(cfg, p=0.0)
    train.best_handwritten_on_train(naive, "M3C", 0.5, 100, tr, "cpu")
    assert calls[-1][:2] == ("plain", "M3C") and calls[-1][2]["p"] == 0.0
    m4 = dict(env.anchor_config("M4"))
    train.best_handwritten_on_train({k: m4[k] for k in ("eps", "L", "rw")}, "M4", 0.99, 100, tr, "cpu")
    assert calls[-1][1] == "M4" and calls[-1][3] == 0.99
    monkeypatch.undo()
    # the real simulators (CPU, tiny): warm_obs recording is a pure observer
    cfg = env.m3c_config(0.5)
    from arbitration.rl import policy
    pol = [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=1.0))]
    seeds = [0, 1]
    a = vulns.simulate_vuln("D2", 1100, "M3C", 300, cfg, pol, seeds, r=0.5, dev="cpu")
    b = vulns.simulate_vuln("D2", 1100, "M3C", 300, cfg, pol, seeds, r=0.5, dev="cpu", warm_obs=True)
    for f in a["snap"]:
        assert torch.equal(a["snap"][f], b["snap"][f]) and torch.equal(a["final"][f], b["final"][f]), f
    assert b["warm_obs"]["obs"].shape == (2, 2, 300, 25) and b["warm_obs"]["v1"].shape == (2, 2, 300)
    # with operator None the vulnerable path reproduces env.simulate(diagnostic=True) bitwise (obs, u1, v1)
    c = vulns.simulate_vuln(None, None, "M3C", 300, cfg, pol, seeds, r=0.5, dev="cpu", warm_obs=True)
    d = env.simulate("M3C", 300, cfg, pol, seeds, r=0.5, dev="cpu", diagnostic=True)
    assert torch.equal(c["warm_obs"]["obs"], d["obs"]) and torch.equal(c["warm_obs"]["v1"], d["log"]["v1"]) and torch.equal(c["warm_obs"]["u1"], d["log"]["u1"])
    # the vulnerable environment really is a different environment for the observation-regression data
    e = vulns.simulate_vuln("D4", 0.047, "M3C", 1500, cfg, pol[1:], seeds, r=0.5, dev="cpu", warm_obs=True)
    f_ = env.simulate("M3C", 1500, cfg, pol[1:], seeds, r=0.5, dev="cpu", diagnostic=True)
    assert not torch.equal(e["warm_obs"]["obs"], f_["obs"])
    # warm_start_theta: vulnerable cell regresses on the vulnerable rollout
    th_v = train.warm_start_theta(cfg, "M3C", 0.5, dict(dz=1.0), 1500, seeds, "cpu", operator="D4", knob=0.047, n_rows=500)
    th_p = train.warm_start_theta(cfg, "M3C", 0.5, dict(dz=1.0), 1500, seeds, "cpu", n_rows=500)
    assert th_v.shape == th_p.shape == (55,) and not np.allclose(th_v, th_p)


def test_T85b_main_passes_operator_and_knob_to_the_warm_start(tmp_path, monkeypatch, captured):
    """T-85b: with the registered route 'warm', main() fits the warm start with the cell's operator/knob (vulnerability cell) and with None for the NAIVE
    pilot; the fitted theta is handed to the segment trainer."""
    got = []
    monkeypatch.setattr(train, "best_handwritten_on_train", lambda cfg, mech, r, T, tr, dev, strategies=None, operator=None, knob=None:
                        got.append(("best", mech, operator, knob, cfg.get("p"))) or dict(name="dz=0.3", scols=dict(dz=0.3), G_train=0.1))
    monkeypatch.setattr(train, "warm_start_theta", lambda cfg, mech, r, scols, T, seeds, dev, F, hidden, operator=None, knob=None:
                        got.append(("fit", mech, operator, knob)) or np.full(55, 0.25))
    root = make_git_root(tmp_path, decision="warm", name="w")
    train.main(_base(root) + ["--cell", "s1_D4_s0.02", "--g1", "10"])
    assert got == [("best", "M3C", "D4", 0.047, 0.1), ("fit", "M3C", "D4", 0.047)] and np.all(captured["theta0"] == 0.25)
    got.clear()
    train.main(_base(root) + ["--cell", "pilot_naive_warm", "--g1", "10"])
    assert got[0][2:] == (None, None, 0.0) and got[1][2:] == (None, None)
    got.clear()
    train.main(_base(root) + ["--cell", "s1_M4rw0_anchor", "--g1", "10"])
    assert got[0][1] == "M4" and got[0][2] is None


# ---------------------------------------------------------------------------------------------------------------- T-86 anchor reuse
def test_T86_naive_anchor_reuses_the_cold_pilot_only_if_every_setting_is_identical(tmp_path, captured):
    """T-86: s1_NAIVE_anchor == pilot_naive_cold in every setting (incl. algo seed, seed block, val seeds) -> the entry refuses to retrain it (CellReused);
    any difference makes the cell non-reusable; the route must be the registered one."""
    pl = _plan()
    anchor, pilot = P.get_cell(pl, "s1_NAIVE_anchor"), P.get_cell(pl, "pilot_naive_cold")
    for k in P._REUSE_FIELDS:
        assert anchor.get(k) == pilot.get(k), k
    assert anchor["algo_seeds"] == pilot["algo_seeds"] == [9000] and anchor["run_ids"] == pilot["run_ids"] == [0]
    assert P.check_reuse(pl, anchor, "cold") == "pilot_naive_cold"
    assert P.check_reuse(pl, P.get_cell(pl, "s1_D2_s0.02"), "cold") is None                                  # ordinary cells are not reused
    for k, v in (("algo_seeds", [9001]), ("run_ids", [1]), ("n_S", 128), ("G_gens", 400), ("val_seeds", [2000, 2030]), ("T_train", 40000), ("lam", 128),
                 ("train_seeds", dict(pilot["train_seeds"], pool=[0, 998])), ("r", 0.9), ("knob", 0.5), ("obs_version", "announced"), ("sigma0", 0.5)):
        bad = copy.deepcopy(pl)
        P.get_cell(bad, "s1_NAIVE_anchor")[k] = v
        with pytest.raises(ValueError, match="not identical"):
            P.check_reuse(bad, P.get_cell(bad, "s1_NAIVE_anchor"), "cold")
    bad = copy.deepcopy(pl)
    P.get_cell(bad, "s1_NAIVE_anchor")["start"] = dict(policy="warm", allowed=["warm"])
    with pytest.raises(ValueError, match="not identical"):
        P.check_reuse(bad, P.get_cell(bad, "s1_NAIVE_anchor"), "cold")
    with pytest.raises(ValueError, match="registered pilot route"):
        P.check_reuse(pl, anchor, None)
    assert P.check_reuse(pl, anchor, "warm") == "pilot_naive_warm"
    # through the entry
    root = make_git_root(tmp_path, decision="cold", name="r")
    with pytest.raises(P.CellReused, match="pilot_naive_cold"):
        train.main(_base(root) + ["--cell", "s1_NAIVE_anchor", "--g1", "10"])
    assert "cell" not in captured                                                                           # nothing was trained
    train.main(_base(root) + ["--cell", "pilot_naive_cold", "--g1", "10"])                                  # the source cell itself trains
    assert captured["cell"]["name"] == "pilot_naive_cold"
    root = make_git_root(tmp_path, name="r2")                                                                 # no registered decision -> refused anyway
    with pytest.raises(ValueError, match="pilot"):
        train.main(_base(root) + ["--cell", "s1_NAIVE_anchor", "--g1", "10"])


# ---------------------------------------------------------------------------------------------------------------- T-87 M-3 record
def test_T87_m3_record_schema_and_computation():
    """T-87: scripts/run_d2_meas.py m3 helper: sigma_hat (ddof=1) of the paired per-seed quantities over the 32 validation seeds, MDE = kappa*sigma/sqrt(32)
    (REQ-MET-06), median over the 3 proxy pairs; output rows validate with meas.validate_m3."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_d2_meas as M
    from arbitration.rl import meas, metrics, seeds as sd
    rng = np.random.default_rng(0)
    vs = list(sd.splits()["val"])
    f = rng.normal(0, 0.01, (7, 32))                                      # honest row + 6 policies (3 pairs), per-seed fitness
    f[0] = 0.0
    row = M.m3_row(0.5, f, vs)
    meas.validate_m3({k: row[k] for k in ("r", "sigma_val_G", "sigma_val_D", "MDE_G", "MDE_D_plan", "MDE_est", "rel_diff")})
    G = [f[i] - f[0] for i in range(1, 7)]
    D = [G[0] - G[1], G[2] - G[3], G[4] - G[5]]
    assert row["sigma_val_G"] == pytest.approx(float(np.median([np.std(g, ddof=1) for g in G])))
    assert row["sigma_val_D"] == pytest.approx(float(np.median([np.std(d, ddof=1) for d in D])))
    assert row["MDE_D_plan"] == pytest.approx(metrics.kappa() * row["sigma_val_D"] / np.sqrt(32))
    assert row["MDE_G"] == pytest.approx(metrics.kappa() * row["sigma_val_G"] / np.sqrt(32))
    assert row["MDE_est"] == 0.0058 and row["rel_diff"] == pytest.approx(row["MDE_G"] / 0.0058 - 1)
    with pytest.raises(ValueError):
        M.m3_row(0.5, f, list(range(32)))                                 # not validation seeds
    # the gate rule itself
    from arbitration.rl import s1
    assert s1.gate_sizes(0.0058)["G_s"] == [0.01, 0.02] and s1.gate_sizes(0.012)["G_s"] == [0.02] and s1.gate_sizes(0.03)["stop02"]


# ---------------------------------------------------------------------------------------------------------------- T-88 hours estimate
def test_T88_hours_estimate_uses_measured_vulnerability_step_time():
    """T-88: estimate(): spec 5.5 formula with the measured D4/D2 step times; diag cells and the reused NAIVE anchor are not counted; s=0.01 cells only if in
    G_s; unmeasured operators take max(D4, D2); plain cells keep the v2 step time."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import estimate_d2_hours as E
    v2 = json.load(open(os.path.join(ROOT, "results", "direction2", "measure_v2_summary.json")))
    timing = dict(cells={"s1_D4_s0.02": dict(operator="D4", t_step_ms_mean=2.0, t_step_ms_p95=3.0, t_val_s=200.0),
                         "s1_D2_s0.02": dict(operator="D2", t_step_ms_mean=1.5, t_step_ms_p95=2.5, t_val_s=100.0)})
    pl = _plan()
    full = E.estimate(pl, timing, v2, [0.01, 0.02])
    rows = {r["cell"]: r for r in full["rows"]}
    assert not any(n.startswith("diag_") for n in rows) and rows["s1_NAIVE_anchor"]["runs"] == 0
    assert rows["s1_D4_s0.02"]["mean_h"] == pytest.approx((416 * 20000 * 2.0 / 1000 + 8 * 200.0) / 3600)
    assert rows["s1_D2_s0.02"]["p95_h"] == pytest.approx((416 * 20000 * 2.5 / 1000 + 8 * 100.0) / 3600)
    assert rows["s1_O1_s0.02"]["t_source"] == "assumed=max(D4,D2)" and rows["s1_O1_s0.02"]["t_step_ms_mean"] == 2.0 and rows["s1_O1_s0.02"]["t_val_s"] == 200.0
    base = v2["run_time_lam256_T2e4_300gen"]["nS256_mean"]["t_step_ms"]
    assert rows["main_M3C_r0.5"]["mean_h"] == pytest.approx(3 * (416 * 20000 * base / 1000 + 8 * v2["validation"]["t_val_s"]) / 3600)
    assert full["totals"]["s1"]["runs"] == 11 and full["totals"]["main"]["runs"] == 6
    only02 = E.estimate(pl, timing, v2, [0.02])
    assert only02["totals"]["s1"]["runs"] == 8 and only02["totals"]["s1"]["mean_h"] < full["totals"]["s1"]["mean_h"]
