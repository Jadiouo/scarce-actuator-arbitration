"""Frozen run plan and the `--plan/--cell` training entry (freeze addendum; T-76..T-80).  All CPU: the simulator is monkeypatched."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT, git, load_json, make_git_root
from arbitration.rl import budget, plan as P, train

PLAN = os.path.join(ROOT, "results", "direction2", "run_plan.json")
FREEZE = os.path.join(ROOT, "results", "direction2", "FREEZE.md")


def _plan():
    with open(PLAN, encoding="utf-8") as f:
        return json.load(f)


def test_T76_plan_holds_the_final_5_5_values():
    """T-76 (spec 5.5, REQ-OPT-12, REQ-S1-10): every cell has lam=256, T_train=20000, n_S=256, val_every=50, explicit obs_version; G_gens 416 except
    main r=0.9 (499); 23 cells / 27 runs; S1 = 12 cells; main = 3 runs per r; knobs equal vuln_suite.json; totals match the spec 5.5 table."""
    pl = _plan()
    cells = pl["cells"]
    assert len(cells) == 23 and pl["totals"]["n_runs"] == 27
    names = [c["name"] for c in cells]
    assert len(set(names)) == len(names)
    for c in cells:
        assert (c["lam"], c["T_train"], c["n_S"], c["val_every"], c["T_val"]) == (256, 20000, 256, 50, 100000), c["name"]
        assert c["obs_version"] == "public" and c["patience"] is None and c["sigma0"] == 0.3
        assert c["G_gens"] == (499 if (c["group"] == "main" and c["r"] == 0.9) else 416), c["name"]
        assert c["algo_seeds"] == [9000 + k for k in c["run_ids"]] and c["val_seeds"] == [2000, 2031]
        assert c["est_per_run"]["mean_s"] < c["est_per_run"]["p95_s"]
    assert [c["group"] for c in cells].count("s1") == 12
    mains = [c for c in cells if c["group"] == "main"]
    assert sorted((c["r"], tuple(c["run_ids"])) for c in mains) == [(0.5, (0, 1, 2)), (0.9, (0, 1, 2))]
    anchors = {c["vuln_id"]: c for c in cells if c["group"] == "s1" and c["vuln_id"] in ("NAIVE", "M4rw0")}
    assert anchors["NAIVE"]["r"] == 0.5 and anchors["M4rw0"]["r"] == 0.99 and anchors["M4rw0"]["mechanism"] == "M4"
    suite = {(e["id"], e.get("size")): e for e in load_json("direction2/vuln_suite.json")["entries"]}
    for c in cells:
        if c["group"] == "s1" and c["vuln_id"] not in ("NAIVE", "M4rw0"):
            e = suite[(c["vuln_id"], c["size"])]
            assert c["knob"] == e["knob"] and c["operator"] == e["operator"] and c["category"] == e["category"]
            assert (c["size"] == 0.01) == bool(c["conditional"])
    assert {c["vuln_id"] for c in cells if c["group"] == "s1" and c["size"] is not None} == {"D1", "D2", "O1", "O2", "O3", "D4", "O5"}
    t = pl["totals"]
    assert abs(t["mean_h"] - 94.8) < 0.1 and abs(t["p95_h"] - 132.4) < 0.15
    pil = [c for c in cells if c["group"] == "pilot"]
    assert [c["start"]["allowed"] for c in pil] == [["cold"], ["warm"]]
    assert all(c["pilot_eval"]["always_report_1"]["policy"].startswith("v1 == 1") for c in pil)
    assert pl["start_registration"]["warm"]["warm_T"] == 4096


def test_T77_plan_is_reproducible_from_the_script(tmp_path):
    """T-77: re-running scripts/make_d2_run_plan.py reproduces results/direction2/run_plan.json byte for byte."""
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_d2_run_plan.py"), "--out", str(out)], check=True, cwd=ROOT,
                   capture_output=True, text=True)
    assert out.read_bytes() == open(PLAN, "rb").read()


def test_T78_plan_sha_matches_freeze_and_tampering_is_refused(tmp_path):
    """T-78: the sha256 of run_plan.json equals FREEZE.md's RUN_PLAN_SHA256; a modified plan, or a FREEZE without the line, is refused."""
    assert P.sha256_file(PLAN) == P.freeze_plan_sha(FREEZE)
    P.load_plan(PLAN, FREEZE)
    bad = tmp_path / "plan.json"
    d = _plan()
    d["cells"][2]["n_S"] = 64
    bad.write_text(json.dumps(d, indent=2))
    with pytest.raises(ValueError, match="sha256"):
        P.load_plan(str(bad), FREEZE)
    nofreeze = tmp_path / "FREEZE.md"
    nofreeze.write_text("no sha here\n")
    with pytest.raises(ValueError):
        P.load_plan(PLAN, str(nofreeze))
    # the plan also pins the spec, the suite and budget.py by sha256
    d = _plan()
    assert d["spec"]["sha256"] == P.sha256_file(os.path.join(ROOT, "docs", "direction2-spec.md"))
    assert d["vuln_suite"]["sha256"] == P.sha256_file(os.path.join(ROOT, "results", "direction2", "vuln_suite.json"))
    assert d["budget_file"]["sha256"] == P.sha256_file(os.path.join(ROOT, "arbitration", "rl", "budget.py"))


def test_T79_budget_py_with_v2_inputs_agrees_with_the_frozen_values():
    """T-79: decide_params with the MEAS v2 worst-case sd_CRN gives lam=256, T=20000, n_S=256, G_gens 416 (S1) / 499 (main); with the v1 sd_CRN it gives
    n_S=64 (this is why the parameters are frozen in the plan instead of being decided at run time)."""
    v1, v2 = load_json("direction2_meas.json"), load_json("direction2/measure_v2_summary.json")
    Ts = (20000, 40000, 100000)
    worst = {r: max(m["sd_crn_median"] for m in v2["m2_summary"] if m["r"] == r) for r in (0.5, 0.9)}
    sd2 = {(r, T): worst[r] for r in (0.5, 0.9) for T in Ts}
    ts = lambda B: v1["rules"]["S1"]["choice"]["t_step_ms"]
    r05 = budget.decide_params(ts, v1["B_max"], 9000.0, sd2)
    r09 = budget.decide_params(ts, v1["B_max"], 10800.0, sd2)
    assert (r05["lam"], r05["T_train"], r05["n_S"], r05["G_gens"]) == (256, 20000, 256, 416)
    assert (r09["lam"], r09["T_train"], r09["n_S"], r09["G_gens"]) == (256, 20000, 256, 499)
    sd1 = {(m["r"], m["T"]): m["sd_crn_median"] for m in v1["m2_summary"]}
    assert budget.decide_params(ts, v1["B_max"], 9000.0, sd1)["n_S"] == 64
    cc = _plan()["budget_crosscheck"]
    assert cc["v2_matches_frozen"] is True and cc["v1_would_give_n_S"] == 64


@pytest.fixture
def captured(monkeypatch):
    """Replace the simulator and the segment trainer so that main() runs on CPU and records what it was given."""
    got = {}
    monkeypatch.setattr(train, "make_sim_fn", lambda *a, **k: got.__setitem__("sim", (a, k)) or (lambda th, s: None))
    monkeypatch.setattr(train, "make_val_fn", lambda *a, **k: got.__setitem__("val", (a, k)) or (lambda th: 0.0))
    monkeypatch.setattr(train, "train_segment", lambda cell, run, g0, g1, sim, val, **k: got.update(cell=cell, run=run, g0=g0, g1=g1, k=k)
                        or dict(cell=cell["name"], run_id=run, gen0=g0, gen1=g1, status="done"))
    return got


def _base(root):
    return ["--test-mode", "--root", root, "--plan", "results/direction2/run_plan.json", "--dev", "cpu"]


def test_T80_plan_entry_takes_the_cell_parameters_from_the_plan(tmp_path, captured):
    """T-80: `--plan --cell` runs with lam/n_S/T/val_every/obs_version/knob from the plan; CLI overrides of those, a tampered plan, an unwired or
    unknown cell, a run or generation outside the plan, an undecided or contradicted start route are all refused.  (Test mode: a throw-away git
    repository outside the real repo; the git/registration guards are the production ones, see test_d2_hardening.py.)"""
    root = make_git_root(tmp_path, decision="cold")
    base = _base(root)
    train.main(base + ["--cell", "s1_D2_s0.02", "--g1", "50"])
    cell = captured["cell"]
    assert (cell["lam"], cell["n_S"], cell["T_train"], cell["val_every"], cell["patience"], cell["sigma0"]) == (256, 256, 20000, 50, None, 0.3)
    a, k = captured["sim"]
    assert a[0] == "M3C" and a[1] == 20000 and tuple(a[-3:]) == ("public", "D2", 1100)
    assert captured["val"][0][1] == 100000 and captured["g1"] == 50
    assert captured["k"]["root"] == os.path.realpath(root)
    # M4 anchor: mechanism and cfg from the plan
    train.main(base + ["--cell", "s1_M4rw0_anchor", "--g1", "10"])
    assert captured["sim"][0][0] == "M4" and captured["sim"][0][2]["L"] == 20
    # main r=0.9: 499 generations are allowed, 500 are not
    train.main(base + ["--cell", "main_M3C_r0.9", "--run", "2", "--g1", "499"])
    with pytest.raises(ValueError):
        train.main(base + ["--cell", "main_M3C_r0.9", "--g1", "500"])
    with pytest.raises(ValueError):
        train.main(base + ["--cell", "s1_D2_s0.02", "--run", "1", "--g1", "10"])
    # the budget flags no longer exist (the old command-line branch is removed): argparse refuses them
    for flag in (["--lam", "64"], ["--nS", "64"], ["--T", "40000"], ["--val-every", "25"], ["--r", "0.9"], ["--mech", "NAIVE"], ["--patience", "3"],
                 ["--T-val", "5"], ["--warm-T", "9"], ["--warm-seeds", "2"]):
        with pytest.raises(SystemExit):
            train.main(base + ["--cell", "s1_D2_s0.02", "--g1", "10"] + flag)
    # start route: contradicts the registered pilot decision
    with pytest.raises(ValueError, match="contradicts"):
        train.main(base + ["--cell", "s1_D2_s0.02", "--g1", "10", "--start", "warm"])
    # no registered decision: non-pilot cells are refused, the pilot cell runs (fixed route cold), warm on the cold pilot is refused
    undecided = make_git_root(tmp_path, name="undecided")
    with pytest.raises(ValueError, match="pilot"):
        train.main(_base(undecided) + ["--cell", "s1_D2_s0.02", "--g1", "10"])
    train.main(_base(undecided) + ["--cell", "pilot_naive_cold", "--g1", "10"])
    assert captured["sim"][0][2]["p"] == 0.0                           # NAIVE = M3C with p=0
    with pytest.raises(ValueError):
        train.main(_base(undecided) + ["--cell", "pilot_naive_cold", "--g1", "10", "--start", "warm"])
    # unwired diagnostic cell and unknown cell
    with pytest.raises(NotImplementedError):
        train.main(base + ["--cell", "diag_history64", "--g1", "10"])
    with pytest.raises(ValueError):
        train.main(base + ["--cell", "no_such_cell", "--g1", "10"])
    # tampered plan, committed (so the git guard is satisfied): the sha256 registered in FREEZE.md no longer matches
    pth = os.path.join(root, "results", "direction2", "run_plan.json")
    d = json.load(open(pth))
    d["cells"][2]["n_S"] = 64
    json.dump(d, open(pth, "w"), indent=2)
    git(root, "commit", "-qam", "tamper")
    with pytest.raises(ValueError, match="sha256"):
        train.main(base + ["--cell", "s1_D2_s0.02", "--g1", "10"])
