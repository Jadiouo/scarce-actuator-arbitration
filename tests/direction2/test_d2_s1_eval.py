"""S1 final evaluation and assembly (arbitration/rl/s1_eval.py, scripts/d2_s1_eval.py): spec REQ-MET-04/06/07, REQ-SEED-02/03/06, REQ-S1-09/10/11/17/18.

CPU tests use a fake simulator (deterministic per-seed gains) so that every number can be recomputed independently; one CPU test runs the real
simulator at a tiny horizon; the [gpu] test checks graph stepping against eager bitwise (run through gpujob)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from conftest import ROOT, cuda_required
from arbitration.rl import final_eval, hw, metrics, parts, plan as _plan, policy, s1, s1_eval, seeds, stop

NTH = policy.n_params(25, 2)
THETA = 0.01 * np.arange(1, NTH + 1)
VS = list(seeds.splits()["val"])


# ------------------------------------------------------------------------------------------------ fake simulator
class FakeRun:
    """Callable with the Runner interface.  Gain of the MLP with theta != 0: RL_EFF; of an adapter: by its dz/b value (best = dz=4.0 in the fine
    grid); per-seed noise differs between splits (so a test/val mix-up changes the numbers).  Records every seed list it is called with."""
    def __init__(self, rl_eff=0.06, hw_scale=0.05, ts=97500):
        self.calls, self.rl_eff, self.hw_scale, self.ts = [], rl_eff, hw_scale, ts

    @staticmethod
    def _noise(S):
        return 0.004 * np.sin(1.7 * S) + 0.002 * np.cos(0.31 * S * S)

    def _hw(self, scols, S):
        v = scols.get("dz", scols.get("b")) if len(scols) == 1 else None
        base = 0.0075 * min(v, 4.0) if v else 0.002                       # dz=4.0 (fine grid) is the best handwritten strategy
        return self.hw_scale / 0.05 * base + 0.0015 * self._noise(S)

    def __call__(self, pols, seeds_, use_graph=False):
        self.calls.append((list(seeds_), bool(use_graph), [type(p).__name__ for p in pols]))
        S = np.asarray(seeds_, float)
        rows = []
        for p in pols:
            if isinstance(p, policy.MLPPolicy):
                g = self.rl_eff + self._noise(S) if np.any(p.theta) else np.zeros_like(S)
            else:
                g = self._hw(p.scols, S) if p.scols else np.zeros_like(S)
            rows.append(100.0 + 0.01 * (S % 5) + g * self.ts)
        return np.asarray(rows), self.ts


def _pc(vid="O1", cat="out_of_family", size=0.02, name=None):
    return dict(name=name or f"s1_{vid}_s{size}", vuln_id=vid, category=cat, size=size, G_gens=416)


SETTINGS = dict(F=25, hidden=2, T_val=100000)


def _ck(theta=THETA, G_val=0.0):
    return dict(cell="x", run_id=0, gen=300, G_val_train=G_val, theta=np.asarray(theta), theta_sha256="abc", n_val_records=8, gens_done=416)


def _write_freeze(tmp):
    d = tmp / "docs"
    d.mkdir(exist_ok=True)
    (d / "direction2-freeze.md").write_text("freeze")
    return str(d / "direction2-freeze.md"), str(tmp / "ledger.json")


# ------------------------------------------------------------------------------------------------ best checkpoint
def _make_run_files(tmp, cell, records, best_gen, gen_done=416, last_status="done"):
    recs = [dict(gen=g, G_val=v) for g, v in records]
    sel = max(recs, key=lambda r: r["G_val"])          # earliest on ties (max keeps the first)
    best = dict(gen=best_gen, G_val=next(r["G_val"] for r in recs if r["gen"] == best_gen), theta=THETA.tolist())
    st = dict(gen=gen_done, records=recs, best=best, stopped=False, curve_G=[], curve_Gbest=[], curve_sigma=[], opt={})
    ck = parts.ckpt_path(cell, 0, str(tmp))
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    np.savez(ck, state=np.array(json.dumps(st)))
    rec = dict(cell=cell, run_id=0, gen0=408, gen1=416, status=last_status, fitness_mean=0, fitness_best=0, G_val=[], sigma=1, t_step_ms=1, wall_s=1,
               git_sha="x", spec_version="v", best=best, val_records=recs)
    parts.write_part(parts.part_path(cell, 0, 408, 416, str(tmp)), rec)
    return sel


def test_best_checkpoint_is_the_validation_argmax_not_the_last_or_training_fitness(tmp_path):
    recs = [(50, 0.10), (100, 0.30), (150, 0.30), (200, 0.12), (250, 0.05), (300, 0.0), (350, 0.2), (400, 0.01)]
    _make_run_files(tmp_path, "c", recs, best_gen=100)
    ck = s1_eval.load_best_checkpoint("c", 0, 416, str(tmp_path))
    assert ck["gen"] == 100 and ck["G_val_train"] == 0.30            # tie 100/150: the earliest
    assert np.array_equal(ck["theta"], THETA) and ck["n_val_records"] == 8
    # a checkpoint whose stored `best` is not the validation argmax is refused
    _make_run_files(tmp_path, "d", recs, best_gen=350)
    with pytest.raises(s1_eval.IncompleteRun, match="argmax"):
        s1_eval.load_best_checkpoint("d", 0, 416, str(tmp_path))
    # unfinished training / unfinished last part is refused
    _make_run_files(tmp_path, "e", recs, best_gen=100, gen_done=375)
    with pytest.raises(s1_eval.IncompleteRun, match="needs 416"):
        s1_eval.load_best_checkpoint("e", 0, 416, str(tmp_path))
    _make_run_files(tmp_path, "f", recs, best_gen=100, last_status="partial")
    with pytest.raises(s1_eval.IncompleteRun, match="finished final segment"):
        s1_eval.load_best_checkpoint("f", 0, 416, str(tmp_path))
    with pytest.raises(s1_eval.IncompleteRun, match="no checkpoint"):
        s1_eval.load_best_checkpoint("nope", 0, 416, str(tmp_path))


def test_every_real_s1_run_has_a_consistent_best_checkpoint():
    """All 12 finished runs in the repo (CPU, files only): best = argmax of the validation records, 55 parameters."""
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    cells = [c for c in s1_eval.expected_cells(pl) if c != "s1_NAIVE_anchor"] + ["pilot_naive_cold"]
    assert len(cells) == 12
    for name in cells:
        ck = s1_eval.load_best_checkpoint(name, 0, 416, ROOT)
        assert ck["theta"].size == 55 and 0 < ck["gen"] <= 416 and ck["n_val_records"] == 8, name
    assert s1_eval.source_cell("s1_NAIVE_anchor", "cold") == "pilot_naive_cold" and s1_eval.source_cell("s1_D1_s0.02", "cold") == "s1_D1_s0.02"
    with pytest.raises(ValueError):
        s1_eval.source_cell("s1_NAIVE_anchor", None)


# ------------------------------------------------------------------------------------------------ validation stage
def test_val_stage_uses_validation_seeds_only_and_matches_metrics(tmp_path):
    run = FakeRun()
    val = s1_eval.val_stage(_pc(), SETTINGS, _ck(), run, root=str(tmp_path), check_consistency=False)
    for sds, _, _ in run.calls:
        assert sds == VS, "val stage may simulate the 32 validation seeds only"
    # independent recomputation of everything from the same fake simulator
    g_rl = run(([policy.MLPPolicy(theta=np.zeros(NTH)), policy.MLPPolicy(theta=THETA)]), VS)[0]
    G_rl = (g_rl[1] - g_rl[0]) / run.ts
    s_hw = hw.build_s_hw()
    U = run([policy.AdapterPolicy({})] + [policy.AdapterPolicy(d) for _, d in s_hw], VS)[0]
    G_hw = {n: (U[i + 1] - U[0]) / run.ts for i, (n, _) in enumerate(s_hw)}
    ref = hw.select_handwritten(G_hw, G_hw, [n for n, _ in s_hw])             # val == test dict: only the SELECTION is compared
    assert val["hw"]["selected"] == ref["selected"] and val["hw"]["plus_h0"]["selected"] == ref["plus_h0"]["selected"]
    assert val["hw"]["selected"] == "dz=4.0"
    D = G_rl - G_hw[ref["selected"]]
    assert val["D_val"] == pytest.approx(metrics.difference_ci(G_rl, G_hw[ref["selected"]]), rel=1e-12)
    assert val["mde_D"] == pytest.approx(metrics.mde(D, VS), rel=1e-12) and val["mde_G"] == pytest.approx(metrics.mde(G_rl, VS), rel=1e-12)
    assert val["mde_frozen"] == val["mde_G"] and val["mde_d_frozen"] == val["mde_D"]
    assert val["G_RL_val"] == pytest.approx(metrics.mean_ci(G_rl, "t31"), rel=1e-12)
    assert len(val["hw"]["val_means"]) == 36 and val["checkpoint"]["theta_sha256"] == "abc"


def test_select_on_validation_agrees_with_hw_select_handwritten():
    names = [n for n, _ in hw.build_s_hw()]
    rng = np.random.default_rng(3)
    for trial in range(5):
        val = {n: rng.normal(0.01 * trial - 0.02, 0.01, 32) for n in names}
        sel = s1_eval.select_on_validation(val, names)
        ref = hw.select_handwritten(val, val, names)
        assert sel["selected"] == ref["selected"] and sel["plus_h0"] == ref["plus_h0"]["selected"]
    allneg = {n: -np.abs(rng.normal(0.01, 0.002, 32)) for n in names}              # H0 beats everything -> plus_h0 == 'honest'
    assert s1_eval.select_on_validation(allneg, names)["plus_h0"] == "honest"
    tie = {n: np.full(32, 0.01) for n in names}                                    # ties -> the earlier name
    assert s1_eval.select_on_validation(tie, names)["selected"] == names[0]
    with pytest.raises(ValueError):
        s1_eval.select_on_validation({n: np.zeros(32) for n in names[:35]}, names[:35])


def test_val_stage_refuses_honest_baselines_that_disagree(tmp_path):
    class Bad(FakeRun):
        def __call__(self, pols, seeds_, use_graph=False):
            U, ts = super().__call__(pols, seeds_, use_graph)
            if all(isinstance(p, policy.AdapterPolicy) for p in pols):
                U = U + 1e-3                                                     # the adapter batch has another honest baseline
            return U, ts
    with pytest.raises(RuntimeError, match="honest baselines"):
        s1_eval.val_stage(_pc(), SETTINGS, _ck(), Bad(), root=str(tmp_path), check_consistency=False)


def test_val_stage_checks_the_recomputed_G_val_against_the_training_run(tmp_path):
    run = FakeRun()
    v = s1_eval.val_stage(_pc(), SETTINGS, _ck(G_val=0.0), run, root=str(tmp_path), check_consistency=False)
    ok = _ck(G_val=v["G_val_recomputed"])
    assert s1_eval.val_stage(_pc(), SETTINGS, ok, run, root=str(tmp_path))["G_val_recompute_absdiff"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(RuntimeError, match="stored G_val"):
        s1_eval.val_stage(_pc(), SETTINGS, _ck(G_val=v["G_val_recomputed"] + 0.01), run, root=str(tmp_path))


# ------------------------------------------------------------------------------------------------ test stage
def _val_and_test(tmp, run_val, run_test, pc=None, mutate=None):
    pc = pc or _pc()
    freeze, ledger = _write_freeze(tmp)
    ck = _ck()
    val = s1_eval.val_stage(pc, SETTINGS, ck, run_val, root=str(tmp), check_consistency=False)
    if mutate:
        mutate(val)
    t = s1_eval.test_stage(pc, SETTINGS, ck, val, run_test, root=str(tmp), freeze_path=freeze, ledger_path=ledger)
    return val, t, ledger, pc, ck


def test_test_stage_only_through_final_eval_and_uses_the_frozen_mde(tmp_path):
    val_run, test_run = FakeRun(), FakeRun(rl_eff=0.07)
    val, t, ledger, pc, ck = _val_and_test(tmp_path, val_run, test_run, mutate=lambda v: v.update(mde_d_frozen=0.123, mde_frozen=0.456))
    test_seeds = list(seeds.splits()["test"])
    assert test_run.calls and all(c[0] == test_seeds for c in test_run.calls)          # the test simulator saw exactly the 32 test seeds
    assert t["MDE_D"] == 0.123 and t["mde_G"] == 0.456                                  # the frozen values of the val record, never recomputed
    assert t["detected"] is False and t["D"]["mean"] > 0.123 * 0 and t["D"]["lo"] > 0 and (t["D"]["mean"] < 0.123)   # D rule uses mde_d_frozen
    raw = json.load(open(s1_eval.eval_path(str(tmp_path), pc["name"], "test_raw")))
    G_rl, G_hw = np.array(raw["rl"]), np.array(raw["hw"])
    assert t["D"] == pytest.approx(metrics.difference_ci(G_rl, G_hw), rel=1e-12) and t["G_RL"] == pytest.approx(metrics.mean_ci(G_rl, "t31"), rel=1e-12)
    assert t["hw_selected"] == val["hw"]["selected"] and t["G_HW"] == pytest.approx(metrics.mean_ci(G_hw, "t31"), rel=1e-12)
    led = json.load(open(ledger))
    assert set(led) == {f"{pc['name']}|rl", f"{pc['name']}|hw", f"{pc['name']}|hw_plus_h0"}
    # a mismatching val record (another checkpoint) is refused before any test simulation
    other = dict(val, checkpoint=dict(val["checkpoint"], theta_sha256="zzz"))
    boom = FakeRun()
    with pytest.raises(ValueError, match="does not belong"):
        s1_eval.test_stage(pc, SETTINGS, ck, other, boom, root=str(tmp_path), freeze_path=_write_freeze(tmp_path)[0], ledger_path=str(tmp_path / "l2.json"))
    assert boom.calls == []
    no_mde = {k: v for k, v in val.items() if k != "mde_d_frozen"}
    with pytest.raises(ValueError, match="frozen mde_d_frozen"):
        s1_eval.test_stage(pc, SETTINGS, ck, no_mde, boom, root=str(tmp_path), freeze_path=_write_freeze(tmp_path)[0], ledger_path=str(tmp_path / "l2.json"))


def test_test_stage_is_once_per_policy_and_resumable_from_the_raw_file(tmp_path):
    val_run, test_run = FakeRun(), FakeRun(rl_eff=0.08)
    val, t, ledger, pc, ck = _val_and_test(tmp_path, val_run, test_run)
    n_calls = len(test_run.calls)
    # crash after the ledger was written but before the result file: rebuild from the saved raw gains, NO new test simulation, identical numbers
    again = FakeRun()
    t2 = s1_eval.test_stage(pc, SETTINGS, ck, val, again, root=str(tmp_path), freeze_path=_write_freeze(tmp_path)[0], ledger_path=ledger)
    assert again.calls == [] and t2["D"] == t["D"] and t2["G_RL"] == t["G_RL"] and t2["detected"] == t["detected"]
    os.remove(s1_eval.eval_path(str(tmp_path), pc["name"], "test_raw"))               # ledger present but raw lost -> refuse, never re-simulate
    with pytest.raises(RuntimeError, match="raw per-seed file"):
        s1_eval.test_stage(pc, SETTINGS, ck, val, again, root=str(tmp_path), freeze_path=_write_freeze(tmp_path)[0], ledger_path=ledger)
    assert again.calls == [] and n_calls >= 1
    # final_eval itself refuses a second request for the same (cell, policy)
    with pytest.raises(RuntimeError, match="already evaluated"):
        final_eval.evaluate(dict(cell=pc["name"], policy_id="rl", mde_frozen=0.1), lambda s: dict(G_s=np.zeros(32)), freeze_path=_write_freeze(tmp_path)[0],
                            ledger_path=ledger)


def test_anchor_runs_use_the_G_rule_and_the_frozen_reference(tmp_path):
    pc = _pc("NAIVE", "anchor", None, "s1_NAIVE_anchor")
    run = FakeRun(rl_eff=0.30)
    # tiny tmp "root" with the frozen reference the stage reads (NAIVE: pilot_decision.json)
    os.makedirs(tmp_path / "results" / "direction2")
    ref = dict(mean=0.2791, lo=0.2786, hi=0.2796, half=0.0005)
    (tmp_path / "results" / "direction2" / "pilot_decision.json").write_text(json.dumps(dict(G_star_ref=ref)))
    freeze, ledger = _write_freeze(tmp_path)
    val = s1_eval.val_stage(pc, SETTINGS, _ck(), run, root=str(tmp_path), check_consistency=False)
    assert val["anchor"] and val["G_star_ref"] == ref and val["G_star_ref_strategy"] == "dz=2.0" and "hw" not in val and "mde_d_frozen" not in val
    t = s1_eval.test_stage(pc, SETTINGS, _ck(), val, FakeRun(rl_eff=0.30), root=str(tmp_path), freeze_path=freeze, ledger_path=ledger)
    assert t["passed"] is s1.anchor_passes(t["G_RL"]["mean"], t["G_RL"]["lo"], 0.2791, val["mde_frozen"]) and t["passed"] is True
    assert json.load(open(ledger)).keys() == {"s1_NAIVE_anchor|rl"}
    pcw = dict(pc, name="s1_NAIVE_anchor_weak")
    valw = s1_eval.val_stage(pcw, SETTINGS, _ck(), FakeRun(rl_eff=0.10), root=str(tmp_path), check_consistency=False)
    weak = s1_eval.test_stage(pcw, SETTINGS, _ck(), valw, FakeRun(rl_eff=0.10), root=str(tmp_path), freeze_path=freeze, ledger_path=ledger)
    assert weak["passed"] is False                                                        # 0.10 < 0.8 * 0.2791
    assert s1_eval.anchor_strategy("M4rw0") == ("b=0.3", dict(b=0.3)) and s1_eval.anchor_strategy("NAIVE")[0] == "dz=2.0"
    assert s1_eval.frozen_anchor_ref("M4rw0", ROOT)["lo"] > 0 and s1.m4_anchor_status(s1_eval.frozen_anchor_ref("M4rw0", ROOT)["lo"]) == "counted"


# ------------------------------------------------------------------------------------------------ guards (REQ-SEED-02/03/06)
def test_new_code_never_names_a_test_seed():
    from arbitration.rl import seeds as sd
    files = [os.path.join(ROOT, "arbitration", "rl", "s1_eval.py"), os.path.join(ROOT, "scripts", "d2_s1_eval.py")]
    assert sd.scan_forbidden_references(files) == []
    src = open(files[0]).read() + open(files[1]).read()
    assert 'splits()["test"]' not in src and "TEST_SEEDS" not in src and "require_test_seeds" not in src
    assert "final_eval.evaluate(" in open(files[0]).read()                              # the only door
    # the whole package except final_eval stays clean (T-27 scans every arbitration/rl module)
    rl_dir = os.path.join(ROOT, "arbitration", "rl")
    every = sorted(os.path.join(rl_dir, f) for f in os.listdir(rl_dir) if f.endswith(".py") and f != "final_eval.py")
    assert sd.scan_forbidden_references(every) == []


def test_final_eval_evaluate_rejects_non_test_seeds_in_evaluator_path(tmp_path):
    freeze, ledger = _write_freeze(tmp_path)
    got = []
    final_eval.evaluate(dict(cell="c", policy_id="p", mde_frozen=0.1), lambda s: got.append(list(s)) or dict(G_s=np.zeros(len(s))), freeze_path=freeze, ledger_path=ledger)
    assert got == [list(seeds.splits()["test"])]
    with pytest.raises(ValueError, match="test seeds only"):
        final_eval.evaluate_on_test(dict(cell="c", policy_id="q", mde_frozen=0.1, seeds=VS), lambda s: dict(G_s=np.zeros(32)), freeze_path=freeze, ledger_path=ledger)
    with pytest.raises(FileNotFoundError):
        final_eval.evaluate(dict(cell="c", policy_id="r", mde_frozen=0.1), lambda s: dict(G_s=np.zeros(32)), freeze_path=str(tmp_path / "none.md"), ledger_path=ledger)


# ------------------------------------------------------------------------------------------------ assembly -> s1.json
def _fake_test(name, vid, cat, size, detected=True, anchor=False, passed=True, info=False):
    ci = lambda m, h=0.01: dict(mean=m, lo=m - h, hi=m + h, half=h)
    t = dict(schema=s1_eval.SCHEMA, stage="test", cell=name, source_cell=name, vuln_id=vid, category=cat, size=size, anchor=anchor, T_score=97500,
             test_seed_range=[0, 0], checkpoint=dict(gen=100), G_RL=ci(0.3 if anchor else 0.05), found_G=True, mde_G=0.002, git_sha="x")
    if anchor:
        t.update(G_star_ref=ci(0.279 if vid == "NAIVE" else 0.119), informational=info, passed=passed)
        if not passed:
            t["G_RL"] = ci(0.01)
    else:
        m = 0.02 if detected else 0.0
        t.update(MDE_D=0.006, D=ci(m, 0.008 if detected else 0.01), detected=bool(detected and m - 0.008 > 0 and m >= 0.006), hw_selected="dz=0.3", G_HW=ci(0.03),
                 D_plus_h0=ci(m), hw_plus_h0_selected="dz=0.3", MDE_D_plus_h0=0.006, detected_plus_h0=True, honest_batches_maxdiff=0.0)
    return t


def _populate(tmp, plan, override=None, skip=()):
    override = override or {}
    for c in plan["cells"]:
        if c["group"] != "s1" or c["name"] in skip:
            continue
        cat = c["category"]
        t = _fake_test(c["name"], c["vuln_id"], cat, c["size"], anchor=(cat == "anchor"), **override.get(c["name"], {}))
        s1_eval._write_json(s1_eval.eval_path(str(tmp), c["name"], "test"), t)


@pytest.fixture
def plan_and_root(tmp_path):
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    d = tmp_path / "results" / "direction2"
    d.mkdir(parents=True)
    (d / "vuln_suite.json").write_text(open(os.path.join(ROOT, "results", "direction2", "vuln_suite.json")).read())
    return pl, tmp_path


def test_assemble_all_pass_validates_and_classifies(plan_and_root):
    pl, tmp = plan_and_root
    _populate(tmp, pl)
    doc = s1_eval.write_s1_json(str(tmp), pl)
    assert s1.validate_s1_output(json.load(open(tmp / "results" / "direction2" / "s1.json")), str(tmp)) is True
    assert os.path.getsize(tmp / s1_eval.FIGURE) > 1000
    assert doc["gate"]["S1_pass"] is True and doc["gate"]["y"] == 3 and doc["gate"]["y_s001"] == 3 and doc["not_completed"] == []
    assert doc["verdict"]["row"] == "J7" and doc["verdict"]["incomplete"] is True            # N1: S1 passed, main experiment not done
    rows = {(r["scope"], r["size"]): (r["x"], r["n"]) for r in doc["summary"]}
    assert rows == {("out_of_family", 0.01): (3, 3), ("out_of_family", 0.02): (3, 3), ("in_class", 0.02): (6, 6)}
    assert len([r for r in doc["runs"] if r["category"] == "anchor"]) == 2 and len(doc["runs"]) == 12
    assert doc["d3"]["delta_fine"] < 0 and "delta_coarse" in doc["d3"]


def test_assemble_failure_paths(plan_and_root):
    pl, tmp = plan_and_root
    # one out-of-family vulnerability undetected at 0.02 -> S1 fails -> J2 ; in-family failures do not matter
    _populate(tmp, pl, override={"s1_D2_s0.02": dict(detected=False), "s1_O2_s0.02": dict(detected=False)})
    d = s1_eval.assemble(str(tmp), pl)
    assert d["gate"]["S1_pass"] is False and d["gate"]["y"] == 2 and d["verdict"]["row"] == "J2" and d["verdict"]["incomplete"] is False
    # only the in-family run fails -> S1 still passes (REQ-S1-17)
    _populate(tmp, pl, override={"s1_O2_s0.02": dict(detected=False)})
    assert s1_eval.assemble(str(tmp), pl)["gate"]["S1_pass"] is True
    # NAIVE anchor not passed -> fail; M4 anchor not passed but informational -> ignored
    _populate(tmp, pl, override={"s1_NAIVE_anchor": dict(passed=False)})
    assert s1_eval.assemble(str(tmp), pl)["gate"]["S1_pass"] is False
    _populate(tmp, pl, override={"s1_M4rw0_anchor": dict(passed=False, info=True)})
    assert s1_eval.assemble(str(tmp), pl)["gate"]["S1_pass"] is True
    _populate(tmp, pl, override={"s1_M4rw0_anchor": dict(passed=False, info=False)})
    assert s1_eval.assemble(str(tmp), pl)["gate"]["S1_pass"] is False


def test_assemble_unfinished_cells_count_as_undetected(plan_and_root):
    pl, tmp = plan_and_root
    _populate(tmp, pl, skip=("s1_O1_s0.02", "s1_D1_s0.01", "s1_D2_s0.01", "s1_O1_s0.01"))
    doc = s1_eval.write_s1_json(str(tmp), pl)                       # validate_s1_output inside
    assert sorted(doc["not_completed"]) == sorted(["s1_O1_s0.02", "s1_D1_s0.01", "s1_D2_s0.01", "s1_O1_s0.01"])
    assert doc["gate"]["S1_pass"] is False and doc["gate"]["y"] == 2 and doc["gate"]["y_s001"] == 0
    rows = {(r["scope"], r["size"]): r["x"] for r in doc["summary"]}
    assert rows[("out_of_family", 0.02)] == 2 and rows[("in_class", 0.02)] == 5 and ("out_of_family", 0.01) not in rows
    assert doc["verdict"]["row"] == "J2"


def test_classify_known_inputs():
    """stop.classify on inputs with a known row (REQ-STOP-03), including the N1 flag used by assemble."""
    c = stop.classify
    assert (c(True, None, None, None, None, 0).row, c(True, None, None, None, None, 0).incomplete) == ("J7", True)
    assert c(False, None, None, None, None, 0).row == "J2"
    assert c(True, 0.02, 0.01, 0.03, 0.006, 3).row == "J3" and c(True, 0.02, 0.01, 0.03, 0.006, 2).row == "J4" and c(True, 0.02, 0.01, 0.03, 0.006, 1).row == "J5"
    assert c(True, 0.004, 0.001, 0.007, 0.006, 3).row == "J6" and c(True, 0.0, -0.01, 0.01, 0.006, 0).row == "J7" and c(True, -0.02, -0.03, -0.01, 0.006, 0).row == "J8"
    assert c(False, 0.02, 0.01, 0.03, 0.006, 3).row == "J1"


# ------------------------------------------------------------------------------------------------ the real simulator, tiny horizon, CPU
def test_real_simulator_cpu_tiny_end_to_end(tmp_path):
    """Real env.simulate/vulns.simulate_vuln on CPU at T=40: both stages run, the honest baselines of the RL and the handwritten batch are the same
    numbers (the D_s pairing is valid), validation touches validation seeds only, results land in a root outside the repo."""
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    pc = _plan.get_cell(pl, "s1_O1_s0.02")
    st = _plan.cell_to_settings(pc)
    st["T_val"] = 40
    ck = s1_eval.load_best_checkpoint("s1_O1_s0.02", 0, 416, ROOT)
    run = s1_eval.make_runner(st, "cpu")
    val = s1_eval.val_stage(pc, st, ck, run, root=str(tmp_path), check_consistency=False)
    assert val["honest_batches_maxdiff"] == 0.0 and val["T_score"] == 40 and len(val["hw"]["val_means"]) == 36
    freeze, ledger = _write_freeze(tmp_path)
    t = s1_eval.test_stage(pc, st, ck, val, run, root=str(tmp_path), freeze_path=freeze, ledger_path=ledger)
    assert t["honest_batches_maxdiff"] == 0.0 and t["MDE_D"] == val["mde_d_frozen"] and t["test_seed_range"] == [5000, 5031]
    assert t["D"]["mean"] == pytest.approx(t["G_RL"]["mean"] - t["G_HW"]["mean"], abs=1e-12)


def test_cli_status_and_guards(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("d2_s1_eval", os.path.join(ROOT, "scripts", "d2_s1_eval.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit, match="only allowed with --test-mode"):
        cli.main(["status", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="outside"):
        cli.main(["status", "--test-mode", "--root", os.path.join(ROOT, "results", "direction2")])
    with pytest.raises(SystemExit, match="not S1 cells"):
        cli.cells_to_do(_plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE)), ["pilot_naive_cold"])


# ------------------------------------------------------------------------------------------------ [gpu]
@pytest.mark.gpu
def test_graph_stepping_of_the_RL_batch_equals_eager_bitwise():
    """--graph: the RL batch (MLP-only) through CUDA graphs gives bitwise the same G_s as eager, in a vulnerability environment and in M4."""
    cuda_required()
    from arbitration.rl import graphsim
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    graph = graphsim.from_env("cuda", force=True)
    for cell in ("s1_O1_s0.02", "s1_M4rw0_anchor"):
        pc = _plan.get_cell(pl, cell)
        st = _plan.cell_to_settings(pc)
        theta = s1_eval.load_best_checkpoint("s1_O1_s0.02", 0, 416, ROOT)["theta"]
        e = s1_eval.rl_gains(s1_eval.make_runner(st, "cuda", None, T=700), theta, VS, use_graph=False)
        g = s1_eval.rl_gains(s1_eval.make_runner(st, "cuda", graph, T=700), theta, VS, use_graph=True)
        assert np.array_equal(e["G_s"], g["G_s"]) and np.array_equal(e["U_hon"], g["U_hon"]), cell
