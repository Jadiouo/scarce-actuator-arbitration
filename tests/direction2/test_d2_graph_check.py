"""scripts/d2_graph_check.py (the pre-switch acceptance, red-team H1): the validation path (val_fn at real T_val, a continuation that crosses a
validation boundary and selects `best`) and the one-shot --preflight verdict.  Everything runs at a tiny size on the CPU (GraphSim(capture=False)
= the same machinery without CUDA graphs); the real thing is `gpujob python3 scripts/d2_graph_check.py --preflight`."""
import importlib.util
import json
import os
import sys

import pytest

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("d2_graph_check", os.path.join(ROOT, "scripts", "d2_graph_check.py"))
gc = importlib.util.module_from_spec(_spec)
sys.modules["d2_graph_check"] = gc
_spec.loader.exec_module(gc)

TINY = dict(lam=6, n_S=4, T_train=48, T_val=40, cfg=dict(p=0.5, tol=1.0, L=8.0, kref=0.5), knob=0.6, operator="O1", mech="M3C", r=0.5)
KW = dict(dev="cpu", capture=False, K=4, shrink=TINY)


def test_val_check_compares_val_fn_eager_vs_graph_bitwise():
    v = gc.check_val("s1_O1_s0.02", repo_root="/nonexistent", **KW)
    assert v["equal"] is True and v["T_val"] == 40 and isinstance(v["val_eager"], float) and v["val_eager"] == v["val_graph"]


def test_val_check_detects_a_difference(monkeypatch):
    real = gc._val_value
    calls = []

    def bad(S, theta, graph, dev):
        x = real(S, theta, graph, dev)
        calls.append(graph is not None)
        return x + 1e-15 if graph is not None else x
    monkeypatch.setattr(gc, "_val_value", bad)
    v = gc.check_val("s1_O1_s0.02", repo_root="/nonexistent", **KW)
    assert v["equal"] is False and calls == [False, True]


def test_cross_val_continuation_triggers_validation_and_best_and_is_bitwise():
    r = gc.check_continuation("s1_O1_s0.02", gens=2, cross_val=True, repo_root="/nonexistent", **KW)[0]
    assert r["graph_equals_eager"] is True
    assert r["validated"] is True and r["val_records_new"] >= 1 and r["best_set"] is True      # the val path really ran in the compared segment


def test_plain_continuation_does_not_validate_and_says_so():
    r = gc.check_continuation("s1_O1_s0.02", gens=2, cross_val=False, repo_root="/nonexistent", **KW)[0]
    assert r["graph_equals_eager"] is True and r["validated"] is False


def test_cross_val_from_an_existing_checkpoint(tmp_path):
    root = gc.make_checkpoint_root(tmp_path / "r", "s1_O1_s0.02", gens=3, **{k: KW[k] for k in ("dev", "shrink")})
    r = gc.check_continuation("s1_O1_s0.02", gens=2, cross_val=True, repo_root=root, **KW)[0]
    assert r["start_gen"] == 3 and r["graph_equals_eager"] is True and r["validated"] is True


def test_run_checks_verdict_includes_concurrent_and_val(tmp_path):
    v = gc.run_checks(["s1_O1_s0.02", "s1_D1_s0.02"], gens=2, cross_val=True, val_cells=["s1_O1_s0.02"], repo_root=str(tmp_path), **KW)
    assert v["ok"] is True
    assert v["cells"]["s1_O1_s0.02"]["concurrent_equals_eager"] is True and v["cells"]["s1_D1_s0.02"]["validated"] is True
    assert v["val"]["s1_O1_s0.02"]["equal"] is True


def test_cross_val_only_for_the_named_cells(tmp_path):
    v = gc.run_checks(["s1_O1_s0.02", "s1_D1_s0.02"], gens=2, cross_val=["s1_O1_s0.02"], repo_root=str(tmp_path), **KW)
    assert v["ok"] is True and v["cells"]["s1_O1_s0.02"]["validated"] is True and v["cells"]["s1_D1_s0.02"]["validated"] is False
    assert v["cells"]["s1_D1_s0.02"]["concurrent_equals_eager"] is True


def test_preflight_cross_val_is_limited_to_the_first_cell():
    cmd = next(s["cmd"] for s in gc.preflight_steps(["a", "b"], "c", python="python3") if s["name"].startswith("4_"))
    assert cmd.count("--cross-val-cell") == 1 and cmd[cmd.index("--cross-val-cell") + 1] == "a" and "--cross-val" not in cmd


# ------------------------------------------------------------------------------------------------ --preflight
def _steps_ok(name):
    return dict(ok=True, seconds=10.0, detail=name)


def test_preflight_plan_has_the_plan_9_1_steps_plus_h1():
    names = [s["name"] for s in gc.preflight_steps(["s1_D2_s0.02", "s1_D1_s0.02"], "s1_M4rw0_anchor", python="python3")]
    assert names == ["1_cpu_tests", "2_gpu_graph_tests", "3_gpu_eager_tests", "4_checkpoint_continuation_cross_val", "H1_val_fn_real_size"]


def test_preflight_full_adds_the_slow_checks():
    quick = {s["name"]: s["cmd"] for s in gc.preflight_steps(["a", "b"], "c", python="python3")}
    full = {s["name"]: s["cmd"] for s in gc.preflight_steps(["a", "b"], "c", python="python3", full=True)}
    assert "tests/direction2" in full["1_cpu_tests"] and "tests/direction2" not in quick["1_cpu_tests"]
    assert not any("T08" in c for c in quick["3_gpu_eager_tests"]) and any("T08" in c for c in full["3_gpu_eager_tests"])
    assert any("T02" in c for c in full["3_gpu_eager_tests"]) and any("T06" in c for c in full["3_gpu_eager_tests"])
    assert any("T01" in c for c in quick["3_gpu_eager_tests"]) and any("T36" in c for c in quick["3_gpu_eager_tests"])


def test_preflight_pass_and_fail_json(tmp_path):
    steps = gc.preflight_steps(["a", "b"], "c", python="python3")
    out = tmp_path / "v.json"
    ok = gc.run_preflight(steps, runner=lambda s: _steps_ok(s["name"]), school_factor=3.0, out_path=str(out))
    assert ok["verdict"] == "PASS" and set(ok["steps"]) == {s["name"] for s in steps}
    assert ok["measured_seconds"] == 50.0 and ok["school_estimate_minutes"] == pytest.approx(50 * 3 / 60) and ok["within_60_min"] is True
    assert json.load(open(out))["verdict"] == "PASS"

    def runner(s):
        return dict(ok=False, seconds=1.0, detail="boom") if s["name"].startswith("2_") else _steps_ok(s["name"])
    bad = gc.run_preflight(steps, runner=runner, school_factor=3.0)
    assert bad["verdict"] == "FAIL" and bad["steps"]["2_gpu_graph_tests"]["ok"] is False and bad["failed"] == ["2_gpu_graph_tests"]


def test_preflight_a_crashing_step_is_a_fail_not_an_exception():
    steps = gc.preflight_steps(["a"], "c", python="python3")

    def runner(s):
        raise RuntimeError("cuda exploded")
    v = gc.run_preflight(steps, runner=runner, school_factor=1.0)
    assert v["verdict"] == "FAIL" and "cuda exploded" in json.dumps(v)


def test_preflight_over_budget_flag():
    steps = gc.preflight_steps(["a"], "c", python="python3")
    v = gc.run_preflight(steps, runner=lambda s: dict(ok=True, seconds=1500.0, detail=""), school_factor=3.0)
    assert v["within_60_min"] is False and v["verdict"] == "PASS"            # over budget is reported, not a failure of the check itself


def test_default_cells_prefers_cells_with_a_checkpoint(tmp_path):
    os.makedirs(tmp_path / "results" / "direction2" / "ckpt")
    for n in ("pilot_naive_cold", "s1_D1_s0.02", "s1_O3_s0.02"):
        open(tmp_path / "results" / "direction2" / "ckpt" / f"{n}__run0.npz", "wb").write(b"x")
    assert gc.default_cells(str(tmp_path)) == ["s1_D1_s0.02", "s1_O3_s0.02"]
    assert gc.default_cells(str(tmp_path / "nowhere")) == ["s1_D2_s0.02", "s1_O1_s0.02"]
