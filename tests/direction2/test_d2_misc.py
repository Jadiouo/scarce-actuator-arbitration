"""T-48..T-55, T-57..T-60: M4 appendix, measurement schemas, GPU rules, neighbours, ablation, truncation (spec s14.7)."""
import ast
import json
import os
import sys

import numpy as np
import pytest

from conftest import ROOT, cuda_required, load_json, rel_close
from arbitration.rl import ablation, budget, gpu_rules, m4, meas, nbr, parts, policy, trunc, metrics


def test_T48_m4_inequality_vs_json():
    """T-48 (REQ-M4-01/02): LHS and prediction for stage2b consist_M4[0..17], quota main[88..91], stage2b eval_rows.M4; 'holds' => G - half <= 0;
    consist_M4[9],[12],[13] not holds; [10],[11] holds."""
    s2b, q = load_json("stage2b.json"), load_json("quota.json")["main"]
    u_bar = q[88]["honest_util_agent1"][0]
    rows = m4.compare_with_json(ROOT, u_bar=u_bar, w=u_bar, g_max=1.0)
    by = {(r["source"], r["index"]): r for r in rows}
    assert sum(r["source"] == "consist_M4" for r in rows) == 18 and sum(r["source"] == "quota_main" for r in rows) == 4
    assert sum(r["source"] == "eval_rows_M4" for r in rows) == len(s2b["eval_rows"]["M4"])
    expected_hold = {}
    for i, c in enumerate(s2b["consist_M4"]):
        r = by[("consist_M4", i)]
        gs = [(c[k][0], c[k][1]) for k in ("gain_b03", "gain_dz1", "gain_edgeO")]
        mx = max(gs)
        assert rel_close(r["G"], mx[0], 1e-12) and rel_close(r["half"], mx[1], 1e-12)
        assert (r["eps"], r["rw"], r["L"]) == (c["eps"], c["rw"], c["L"])
        # independent restatement of the stepwise inequality eps*rw*(L*u_bar + w) >= g_max (u_bar = w = honest utility, g_max = 1)
        lhs = c["eps"] * c["rw"] * (c["L"] * u_bar + u_bar)
        expected_hold[i] = bool(lhs >= 1.0)
        assert r["predicted_holds"] is expected_hold[i], (i, lhs)
        assert rel_close(m4.lhs(c["eps"], c["rw"], c["L"], u_bar, u_bar), lhs, 1e-12), i
    for i in (9, 12, 13):
        assert by[("consist_M4", i)]["predicted_holds"] is False
    for i in (10, 11):
        assert by[("consist_M4", i)]["predicted_holds"] is True
    for r in rows:
        if r["predicted_holds"]:
            assert r["G"] - r["half"] <= 0, r
        assert r["compatible"] is True, r
    assert m4.lhs(0.02, 1 / 3, 2500.0, 0.184, 0.0) == pytest.approx(0.02 / 3 * 2500 * 0.184, rel=1e-12)
    assert m4.predicts_nonpositive(0.02, 1 / 3, 2500.0, 0.184, 0.0) and not m4.predicts_nonpositive(0.2, 1.0, 20.0, 0.184, 0.0)
    assert not m4.predicts_nonpositive(0.2, 0.0, 2500.0, 0.184, 0.5)


def test_T49_m4_degeneracy_check():
    """T-49 (REQ-M4-04, HW-01): from quota.json: main[88..91].Gmax == [0,0]; all quota_* gains exactly 0.0; Gmax_nonquota values match JSON; conclusion string mentions '含空操作策略'."""
    q = load_json("quota.json")["main"]
    rep = m4.degeneracy_report(ROOT)
    for i in range(88, 92):
        assert q[i]["Gmax"] == [0.0, 0.0] and list(rep["gmax"][i]) == [0.0, 0.0]
        assert all(v == 0.0 for k, v in q[i]["gains_by_strategy"].items() if k.startswith("quota_"))
        assert rel_close(rep["gmax_nonquota"][i][0], q[i]["Gmax_nonquota"][0], 1e-12)
    assert rep["quota_gains_all_zero"] is True and "含空操作策略" in rep["conclusion"]


@pytest.mark.gpu
def test_T50_m4_break_region_signs():
    """T-50 (REQ-M4-03): >=3 near-boundary (eps,rw,L) points; simulated G sign is compatible with the stepwise-inequality prediction; at least one point predicted to fail."""
    cuda_required()
    rows = m4.check_break_region()
    assert len(rows) >= 3 and any(r["predicted_holds"] is False for r in rows)
    assert all(r["sign_compatible"] for r in rows)


def _m1_row():
    return dict(mech="M3C", r=0.5, pop=32, n_S=16, B=512, policy="mlp25-2-1", t_step_ms_median=1.0, t_step_ms_p95=1.2, steps_timed=1000,
                warmup=200, repeats=5, mem_peak_MB=100.0, elem_steps_per_s=1e5, frontier_ms_per_step=2.0, device="cuda", dtype="float64",
                torch="2.12", git_sha="abc")


def test_T51_measure_output_schema():
    """T-51 (REQ-MEAS-01..04): M-1..M-3 JSON fields/types per s11 (missing field rejected); M-4 has G_star_ref and the JSON consist_M4[12].gain_b03 side by side."""
    assert meas.validate_m1([_m1_row()]) is True
    bad = _m1_row()
    del bad["mem_peak_MB"]
    with pytest.raises(ValueError):
        meas.validate_m1([bad])
    with pytest.raises(ValueError):
        meas.validate_m1([dict(_m1_row(), B="512")])
    m2row = dict(r=0.5, T=20000, pair_kind="rand", pair_id=0, var_crn=1e-6, var_indep=4e-6, vrf=4.0, sd_crn=1e-3, n_seeds=64)
    doc = dict(rows=[m2row], summary=[dict(r=0.5, T=20000, vrf_median=4.0, vrf_min=2.0, sd_crn_median=1e-3)])
    assert meas.validate_m2(doc) is True
    with pytest.raises(ValueError):
        meas.validate_m2(dict(doc, rows=[{k: v for k, v in m2row.items() if k != "vrf"}]))
    m3 = dict(r=0.5, sigma_val_G=0.01, sigma_val_D=0.01, MDE_G=0.005, MDE_D_plan=0.005, MDE_est=0.0058, rel_diff=-0.1)
    assert meas.validate_m3(m3) is True
    with pytest.raises(ValueError):
        meas.validate_m3({k: v for k, v in m3.items() if k != "MDE_D_plan"})
    ref = load_json("stage2b.json")["consist_M4"][12]["gain_b03"]
    m4doc = dict(G_star_ref=dict(mean=0.12, lo=0.11, hi=0.13), json_ref=ref)
    assert meas.validate_m4(m4doc, ROOT) is True
    with pytest.raises(ValueError):
        meas.validate_m4(dict(m4doc, json_ref=[ref[0] + 0.01, ref[1]]), ROOT)
    with pytest.raises(ValueError):
        meas.validate_m4(dict(json_ref=ref), ROOT)


def test_T52_gpu_job_manifest():
    """T-52 (REQ-GPU-01/02/05/06): every manifest line starts with gpujob, names a part file, est <= 1500 s; no 'gpujob slots'; no direct python; [gpu] test jobs go through gpujob."""
    part = "results/direction2/parts/cell__run0__g0000-0025.json"
    py = sys.executable                                        # portable: no machine-specific venv path
    good = [dict(cmd=f"gpujob {py} scripts/d2_train.py --out {part}", est_s=1200.0)]
    assert gpu_rules.validate_manifest(good) == []
    for bad in (dict(cmd=f"{py} scripts/d2_train.py --out {part}", est_s=10.0), dict(cmd="gpujob slots 3", est_s=1.0),
                dict(cmd=f"gpujob {py} scripts/d2_train.py", est_s=10.0), dict(cmd=f"gpujob {py} x.py --out {part}", est_s=1501.0),
                dict(cmd=f"gpujob {py} x.py --out {part} && python3 y.py", est_s=10.0)):
        assert len(gpu_rules.validate_manifest(good + [bad])) >= 1, bad
    cmds = gpu_rules.gpu_test_job_commands(["tests/direction2/test_d2_env.py::test_T01_parity_m3c_per_seed"])
    assert all(c.startswith("gpujob ") for c in cmds) and len(cmds) == 1 and "pytest" in cmds[0]


def test_T53_part_files_atomic_and_idempotent(tmp_path, monkeypatch):
    """T-53 (REQ-GPU-03/04): part path format; write = temp file + os.replace; required fields enforced; done => skip, partial => overwritten/rerun, checkpoint => resume."""
    p = parts.part_path("cellA", 1, 0, 25, root=str(tmp_path))
    assert p == str(tmp_path / "results/direction2/parts/cellA__run1__g0000-0025.json")
    rec = dict(cell="cellA", run_id=1, gen0=0, gen1=25, status="partial", fitness_mean=0.1, fitness_best=0.2, G_val=[0.1], sigma=0.3,
               t_step_ms=1.0, wall_s=10.0, git_sha="abc", spec_version="v1.0")
    calls = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append((a, b)), real(a, b))[1])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    parts.write_part(p, rec)
    assert len(calls) == 1 and calls[0][1] == p and calls[0][0] != p
    assert parts.read_part(p)["status"] == "partial"
    with pytest.raises(ValueError):
        parts.write_part(p, {k: v for k, v in rec.items() if k != "sigma"})
    assert parts.job_action(p) == "run"
    ck = str(tmp_path / "ckpt.npz")
    open(ck, "w").write("x")
    assert parts.job_action(p, ck) == "resume"
    parts.write_part(p, dict(rec, status="done"))
    assert parts.job_action(p, ck) == "skip"
    parts.write_part(p, dict(rec, status="partial", fitness_mean=0.5))
    assert parts.read_part(p)["fitness_mean"] == 0.5


def test_T54_neighbor_configs_and_no_retrain(tmp_path):
    """T-54 (REQ-NBR-01/02/03): neighbours exactly {h=5,h=7,L=1250}; policy parameters unchanged by evaluation; T_score = T-L (no forced tail honesty); result file carries the limitation sentence."""
    base = dict(p=0.1, tol=6.0, L=2500.0, kref=0.5)
    cfgs = nbr.neighbor_configs(base)
    assert len(cfgs) == 3
    diffs = sorted(tuple(sorted(k for k in base if c[k] != base[k])) for c in cfgs)
    assert diffs == [("L",), ("tol",), ("tol",)]
    assert sorted(c["tol"] for c in cfgs if c["tol"] != 6.0) == [5.0, 7.0] and [c["L"] for c in cfgs if c["L"] != 2500.0] == [1250.0]
    assert nbr.neighbor_t_score(100000, 2500) == 97500 and nbr.neighbor_t_score(100000, 1250) == 98750
    theta = np.arange(55, dtype=float)
    h0 = theta.tobytes()
    seen = []

    def sim_fn(config, who, seeds, T, T_score, variant=None):
        seen.append((config, who, T, T_score, variant))
        return dict(G_s=np.zeros(len(seeds)), rt=0.01, alpha=0.03)

    out = tmp_path / "nbr.json"
    nbr.evaluate_neighbors(theta, "dz=0.3", list(range(5000, 5032)), sim_fn, str(out))
    assert theta.tobytes() == h0
    assert len(seen) == 6 and all(v is None for *_, v in seen)
    for c, who, T, Ts, v in seen:
        assert Ts == T - c["L"]
    txt = out.read_text()
    assert "機制參數沒有針對 RL 重新調參" in txt and "僅限此固定配置" in txt


def test_T55_feature_ablation_groups_partition():
    """T-55 (REQ-S1-19, ACT-03): the six leave-one-group-out index groups partition f01..f25; history-64 variant has F=134 and 273 params <= P_DIAG=300."""
    g = ablation.groups()
    assert len(g) == 6
    flat = sorted(i for grp in g for i in grp)
    assert flat == list(range(25))
    assert sorted(g, key=lambda x: x[0]) == [[0, 1], [2, 3, 4], list(range(5, 13)), list(range(13, 21)), [21, 22], [23, 24]]
    assert ablation.history64_dim() == 134 and policy.n_params(134) == 273 <= policy.param_limit(diagnostic=True)


def _is_gpu_marked(fn):
    for d in fn.decorator_list:
        if "gpu" in ast.dump(d) and "mark" in ast.dump(d):
            return True
    return False


def _assert_gpu_tests_fail_instead_of_skip(here):
    """Test-local static guard for REQ-GPU-07 (independent of gpu_rules): (1) conftest.cuda_required asserts torch.cuda.is_available() and contains
    no skip/xfail/importorskip; (2) every [gpu] test of this directory calls cuda_required() as its first statement (after the docstring), so
    without CUDA it FAILS on that assertion; (3) no test file uses skip/skipif/importorskip/xfail anywhere on a [gpu] test."""
    conf = ast.parse(open(os.path.join(here, "conftest.py")).read())
    cr = [n for n in conf.body if isinstance(n, ast.FunctionDef) and n.name == "cuda_required"]
    assert len(cr) == 1
    body_nodes = cr[0].body[1:] if isinstance(cr[0].body[0], ast.Expr) and isinstance(getattr(cr[0].body[0], "value", None), ast.Constant) else cr[0].body
    txt = " ".join(ast.dump(n) for n in body_nodes)                  # without the docstring
    assert "is_available" in txt and any(isinstance(n, ast.Assert) for n in ast.walk(cr[0])), "cuda_required must assert torch.cuda.is_available()"
    used = {n.attr for b in body_nodes for n in ast.walk(b) if isinstance(n, ast.Attribute)} | \
           {n.id for b in body_nodes for n in ast.walk(b) if isinstance(n, ast.Name)}
    assert not used & {"skip", "skipif", "xfail", "importorskip", "Skipped"}, f"cuda_required must not skip: {used}"
    n_gpu = 0
    for f in sorted(os.listdir(here)):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        tree = ast.parse(open(os.path.join(here, f)).read())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_") and _is_gpu_marked(n)]:
            n_gpu += 1
            body = list(fn.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                body = body[1:]                               # docstring
            first = body[0]
            ok = isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and getattr(first.value.func, "id", None) == "cuda_required"
            assert ok, f"{f}::{fn.name}: first statement must be cuda_required()"
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr in ("skip", "skipif", "importorskip", "xfail"):
                    raise AssertionError(f"{f}::{fn.name}: uses {node.attr}")
    assert n_gpu >= 9, n_gpu


def test_T57_gpu_tests_never_skip():
    """T-57 (REQ-GPU-07): (a) AST: no pytest.skip/skipif/importorskip/xfail in [gpu] tests; (b) with CUDA hidden every [gpu] test FAILS and none is
    skipped; (c) the report checker rejects any skipped [gpu] test."""
    here = os.path.dirname(os.path.abspath(__file__))
    _assert_gpu_tests_fail_instead_of_skip(here)
    assert gpu_rules.scan_gpu_tests(here) == []
    probe = "import pytest\n@pytest.mark.gpu\ndef test_x():\n    pytest.skip('no')\n"
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "d2_probe_gpu_scan")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "test_probe.py"), "w").write(probe)
    try:
        assert len(gpu_rules.scan_gpu_tests(d)) == 1
    finally:
        os.remove(os.path.join(d, "test_probe.py"))
        os.rmdir(d)
    res = gpu_rules.run_gpu_tests_without_cuda(here)
    assert res["skipped"] == 0 and res["xfailed"] == 0 and res["passed"] == 0 and res["failed"] > 0
    ok = '<testsuite><testcase classname="t" name="test_a"/></testsuite>'
    sk = '<testsuite><testcase classname="t" name="test_T01"><skipped/></testcase></testsuite>'
    assert gpu_rules.check_report(ok) is True and gpu_rules.check_report(sk) is False


def test_T58_gpujob_segmentation():
    """T-58 (REQ-GPU-02/08): segments are contiguous, non-overlapping, cover 0..G_gens, each estimate <= 1500 s; one generation > 1500 s is refused; test-job chunks <= 1500 s; overrun parts are flagged and re-estimated."""
    ts = lambda B: 0.002 * B
    for T, G, B in ((20000, 416, 512), (20000, 499, 512), (40000, 549, 4096), (100000, 1000, 2048)):      # v1.2: 416 (r=0.5) / 499 (r=0.9) generations
        plan = budget.segment(ts, T, G, B, 32, t_val_s=30.0, t_startup_s=20.0)
        assert plan[0][0] == 0 and plan[-1][1] == G
        assert all(a[1] == b[0] for a, b in zip(plan, plan[1:])) and all(g1 > g0 for g0, g1 in plan)
        for g0, g1 in plan:
            assert budget.segment_estimate_s(g0, g1, ts, T, B, 32, t_val_s=30.0, t_startup_s=20.0) <= 1500.0
            assert budget.segment_estimate_s(g0, g1, ts, T, B, 32, t_val_s=0.0, t_startup_s=20.0) == pytest.approx((g1 - g0) * T * ts(B) / 1000 + 20.0)
    with pytest.raises(ValueError):
        budget.segment(lambda B: 20.0, 100000, 10, 64, 32)
    chunks = budget.chunk_jobs([(f"c{i}", 400.0) for i in range(10)])
    assert [x for c in chunks for x in c] == [f"c{i}" for i in range(10)] and all(len(c) <= 3 for c in chunks)
    with pytest.raises(ValueError):
        budget.chunk_jobs([("big", 1600.0)])
    # v1.2 (REQ-OPT-06 / GPU-08): the default validation cost is one evaluation per 50 generations (v1.1: 25)
    assert budget.segment_estimate_s(0, 100, ts, 20000, 512, 32, t_val_s=30.0) == pytest.approx(100 * 20000 * ts(512) / 1000 + 2 * 30.0)
    assert budget.segment_estimate_s(0, 49, ts, 20000, 512, 32, t_val_s=30.0) == pytest.approx(49 * 20000 * ts(512) / 1000)
    assert budget.segment_estimate_s(25, 75, ts, 20000, 512, 32, t_val_s=30.0) == pytest.approx(50 * 20000 * ts(512) / 1000 + 30.0)
    plan = budget.segment(ts, 20000, 416, 512, 32)
    first = plan[0]
    part = dict(gen0=first[0], gen1=first[1], wall_s=3000.0)
    new = budget.update_after_part(part, plan, ts, 20000, 416, 512, 32)
    assert new["part"]["overrun"] is True
    measured = lambda B: 3000.0 * 1000 / ((first[1] - first[0]) * 20000)
    rest = new["plan"]
    assert rest[0][0] == first[1] and rest[-1][1] == 416
    assert all(budget.segment_estimate_s(a, b, measured, 20000, 512, 32) <= 1500.0 for a, b in rest)
    ok = budget.update_after_part(dict(gen0=first[0], gen1=first[1], wall_s=100.0), plan, ts, 20000, 416, 512, 32)
    assert not ok["part"].get("overrun")


@pytest.mark.gpu
def test_T59_replay_stage2b_m4_anchor(monkeypatch):
    """T-59 (REQ-ENV-21, ENV-11): replay stage2b consist_M4[12] (r=0.99,eps=0.2,L=20,rw=0: gain_b03, gain_dz1, gain_edgeO, waste) and consist_M4[9] (gain_b03, gain_edgeO), T=1e5, seeds 1000-1031, rel tol 1e-9, CI by 1.96."""
    cuda_required()
    from conftest import ci196, forbid_frontier
    from arbitration.rl import env
    c = load_json("stage2b.json")["consist_M4"]
    seeds = list(range(1000, 1032))
    T = 100000
    # strategies in the order of scripts/run_stage2b.py part_consist() (index 0, 5, 6, 7): honest, b0.3, dz1.0, edgeO0.05(elig) = dict(em=.05)
    strat = [("honest", {}), ("gain_b03", dict(b=0.3)), ("gain_dz1", dict(dz=1.0)), ("gain_edgeO", dict(em=0.05))]
    pols = [policy.AdapterPolicy(d) for _, d in strat]
    forbid_frontier(monkeypatch)
    for i in (12, 9):
        row = c[i]
        cfg = dict(eps=row["eps"], L=float(row["L"]), rw=row["rw"])
        out = env.simulate("M4", T, cfg, pols, seeds, r=row["r"], T_score=T, dev="cuda")["final"]
        ub = out["util_b"][:, :, 1].cpu().numpy() / T
        for j, (key, _) in enumerate(strat):
            if j == 0:
                continue
            got = ci196(ub[j] - ub[0])
            assert rel_close(got[0], row[key][0]) and rel_close(got[1], row[key][1]), (i, key)
        w = ci196(out["nraid"][0].cpu().numpy() / T)
        assert rel_close(w[0], row["waste"][0]) and rel_close(w[1], row["waste"][1]), (i, "waste")


def test_T60_truncation_bias_measure(tmp_path):
    """T-60 (REQ-ENV-20): the tail-honest variant exists only inside the measurement function; normal runs are unchanged; trunc_bias.json has policy,r,G_normal,G_variant,delta,ci with delta=G_normal-G_variant; the training/selection/classification modules never import it."""
    from arbitration.rl import env
    const = policy.MLPPolicy.from_blocks(np.zeros((25, 2)), np.zeros(2), np.zeros(2), np.array([0.3]))
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    n1 = env.simulate("M3C", 3000, cfg, [const], [0, 1], r=0.5, dev="cpu", diagnostic=True)
    n2 = env.simulate("M3C", 3000, cfg, [const], [0, 1], r=0.5, dev="cpu", diagnostic=True, diagnostic_variant=None)
    assert n1["log"]["v1"].equal(n2["log"]["v1"])
    var = env.simulate("M3C", 3000, cfg, [const], [0, 1], r=0.5, dev="cpu", diagnostic=True, diagnostic_variant="tail_honest")
    assert var["log"]["v1"][..., 2400:].equal(var["log"]["u1"][..., 2400:]) and var["log"]["v1"][..., :2400].equal(n1["log"]["v1"][..., :2400])
    out = tmp_path / "trunc_bias.json"
    row = trunc.measure(const, 0.5, 3000, 300.0, [0, 1, 2, 3], str(out), dev="cpu", name="const03")
    doc = json.loads(out.read_text())
    doc = doc[0] if isinstance(doc, list) else doc
    assert {"policy", "r", "G_normal", "G_variant", "delta", "ci"} <= set(doc)
    assert doc["delta"] == pytest.approx(doc["G_normal"] - doc["G_variant"], abs=1e-15) and len(doc["ci"]) == 2
    for mod in ("train", "cmaes", "s1", "stop", "hw", "budget", "metrics"):
        src = open(os.path.join(ROOT, "arbitration", "rl", mod + ".py")).read()
        for node in ast.walk(ast.parse(src)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any("trunc" in n for n in names), (mod, names)
        assert "tail_honest" not in src, mod
