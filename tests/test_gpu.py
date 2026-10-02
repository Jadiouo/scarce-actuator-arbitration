"""One-shot GPU/CPU-torch vs numpy equivalence check (seconds). Skipped without torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from arbitration import run, dp
from arbitration.model import make_config
from arbitration.gpu.sim_torch import simulate, METRICS, SUPPORTED
from arbitration.gpu.dp_torch import solve_full

SEEDS = [0, 1, 2]


def _err(pol, steps=300, **kw):
    o = simulate(pol, SEEDS, steps=steps, spread=0.6, **kw)
    e = 0.0
    for i, s in enumerate(SEEDS):
        r = run(pol, n=8, steps=steps, seed=s, spread=0.6, **kw)
        for m in METRICS:
            a, b = o[m][i], getattr(r, m)
            if not (np.isnan(a) and np.isnan(b)):
                e = max(e, abs(a - b))
        e = max(e, float(np.abs(o["robot_health"][i] - r.robot_health).max()))
    return e


@pytest.mark.parametrize("model", ["legacy", "v2"])
@pytest.mark.parametrize("pol", [p for p in SUPPORTED if "disp" not in p])
def test_sim_matches_run_per_seed(pol, model):
    assert _err(pol, rate=0.01, model=model, preempt=True, hyst=0.2) < 1e-9


def test_sim_variants_and_inflate():
    assert _err("honest", rate=0.01, model="v2", noise="bias", stream=2) < 1e-9
    assert _err("learned_index", rate=0.03, model="v2", service=12, precision_at="arrival") < 1e-9
    b = np.linspace(0.0, 0.35, 8)
    o = simulate("honest_index", [0, 1], steps=300, spread=0.6, rate=0.01, inflate=np.stack([b, b[::-1]]))
    for i, bb in enumerate((b, b[::-1])):
        r = run("honest_index", n=8, steps=300, seed=i, spread=0.6, rate=0.01, inflate=bb)
        assert abs(o["mean_health"][i] - r.mean_health) < 1e-9
        assert np.abs(o["robot_health"][i] - r.robot_health).max() < 1e-9


def test_sim_reproduces_pinned_baseline():
    from tests.test_model import BASELINE, KW
    for (pol, pre), (mh, ns, rt) in BASELINE.items():
        o = simulate(pol, [KW["seed"]], n=KW["n"], rate=KW["rate"], spread=KW["spread"], lam=KW["lam"],
                     preempt=pre)
        assert abs(o["mean_health"][0] - mh) < 1e-9 and o["n_services"][0] == ns
        assert abs(o["retargets_per_service"][0] - rt) < 1e-9


@pytest.mark.parametrize("model", ["legacy", "v2"])
def test_dp_batch_matches_numpy(model):
    cfg = make_config(model)
    ins = [dp.instance(s, 3, 0.004, 0.6)[:2] for s in range(3)]
    sols = solve_full(ins, cfg, K=14)
    for (P, R), s in zip(ins, sols):
        f = dp.FullDP(P, R, cfg, 14)
        assert abs(f.g - s.g) < 1e-9 and f.iters == s.iters
        assert max(np.abs(a - b).max() for a, b in zip(f.V, s.V)) < 1e-9


@pytest.mark.parametrize("pol,opts", [("fused", {}), ("fused_index", {"fuse_known_rates": True}),
                                      ("audit_index", {}),
                                      ("audit_penalty_index", {"penalty_mode": "ignore", "audit_z": 1.0}),
                                      ("audit_penalty_index", {"penalty_mode": "demote", "audit_z": 1.0, "penalty_T": 100})])
def test_game_policies_match_run(pol, opts):
    cfg = make_config("v2", **opts)
    b = np.random.default_rng(1).uniform(0, 0.6, (2, 3))
    o = simulate(pol, [0, 1], n=3, rate=0.004, steps=500, spread=0.6, model=cfg, inflate=b)
    for i in range(2):
        r = run(pol, n=3, rate=0.004, steps=500, seed=i, spread=0.6, model=cfg, inflate=b[i])
        assert np.abs(o["robot_health"][i] - r.robot_health).max() < 1e-9
        assert o["n_services"][i] == r.n_services


def test_mixed_batch_and_best_response_driver():
    from arbitration import game
    from arbitration.gpu import game_gpu as gg
    env = dict(n=3, rate=0.004, spread=0.6, steps=300, sigma=0.15, model="v2", opts={})
    grid, seeds = (0.0, 0.3, 0.6), [0, 1]
    ref = game.best_response_iteration(game.CpuEvaluator(), "fused_index", env, seeds, grid, 0.002, 4)
    got = gg.solve_games([("fused_index", {}), ("honest_index", {})], env, seeds, grid, 0.002, 4,
                         starts=(0.0,), log=lambda *a: None)[(0, 0.0)]
    assert np.array_equal(ref["b"], got["b"]) and np.array_equal(ref["converged"], got["converged"])


@pytest.mark.parametrize("pol,opts,n", [("audit_disp_index", {}, 3),
                                         ("audit_disp_penalty_index", {"penalty_mode": "ignore", "audit_z": 1.0}, 3),
                                         ("audit_disp_penalty_index", {"penalty_mode": "demote", "audit_z": 1.0, "penalty_T": 100}, 8),
                                         ("fused_index", {}, 8), ("honest_index", {}, 3)])
def test_strategy_modes_disp_audit_burn_vs_cpu_reference(pol, opts, n):
    from arbitration.gpu.ref_cpu import run_ref
    rng = np.random.default_rng(0)
    cfg = make_config("v2", **opts)
    seeds, steps, burn = [0, 1, 2], 700, 150
    infl = rng.uniform(0, 0.8, (3, n)); imode = rng.integers(0, 5, (3, n))
    ipar = np.where(imode == 2, 10.0, np.where(imode == 3, 0.3, 0.0))
    o = simulate(pol, seeds, n=n, rate=0.004, steps=steps, spread=0.6, model=cfg, inflate=infl, imode=imode,
                 ipar=ipar, burn=burn, disp_offset=0.03)
    for i, s in enumerate(seeds):
        r, p05 = run_ref(pol, n=n, rate=0.004, steps=steps, seed=s, spread=0.6, model=cfg, inflate=infl[i],
                         imode=imode[i], ipar=ipar[i], burn=burn, disp_offset=0.03)
        assert np.abs(o["robot_health"][i] - r.robot_health).max() < 1e-9
        assert abs(o["min_health_p05"][i] - p05) < 1e-9
        assert abs(o["frac_dead_robot_steps"][i] - r.frac_dead_robot_steps) < 1e-9
        assert o["n_services"][i] == r.n_services


def test_adaptive_inflation_matches_cpu_reference():
    from arbitration.gpu.ref_cpu import run_ref
    cfg = make_config("v2", penalty_mode="ignore")
    imode = np.array([[5, 0, 5], [1, 5, 4]]); ipar = np.array([[0.4, 0, 0.8], [0, 0.4, 0]])
    infl = np.array([[0, 0.3, 0], [0.5, 0, 0.4]])
    o = simulate("audit_disp_penalty_index", [0, 1], n=3, rate=0.004, steps=600, spread=0.6, model=cfg,
                 inflate=infl, imode=imode, ipar=ipar, burn=100)
    for i in range(2):
        r, p05 = run_ref("audit_disp_penalty_index", n=3, rate=0.004, steps=600, seed=i, spread=0.6, model=cfg,
                         inflate=infl[i], imode=imode[i], ipar=ipar[i], burn=100)
        assert np.abs(o["robot_health"][i] - r.robot_health).max() < 1e-9
        assert abs(o["min_health_p05"][i] - p05) < 1e-9
