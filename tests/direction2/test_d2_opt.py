"""T-20..T-25, T-64: optimizer, objective, decision rules (spec s14.3); spec v1.2: REQ-OPT-06/09/10/12/13, REQ-SEED-07."""
import ast
import json
import inspect
import os
import random

import numpy as np
import pytest

from conftest import ROOT, call_new_kw, need
from arbitration.rl import budget, cmaes, s1, train


def _sphere_run(opt, gens, f=lambda X: -np.sum((X - 0.5) ** 2, 1)):
    for _ in range(gens):
        X = opt.ask()
        opt.tell(X, f(X))


def _check_pairs(X, mean, lam, n):
    """Mirrored-sampling invariants of one generation: pairs (rows 2j, 2j+1) are exactly +-delta_j about the mean; the lam/2 deltas are
    pairwise different (also up to sign) and non-zero.  Returns the deltas."""
    assert X.shape == (lam, n)
    d_plus, d_minus = X[0::2] - mean[None, :], X[1::2] - mean[None, :]
    assert np.abs(d_plus + d_minus).max() <= 1e-15, "the two members of a pair must be exact opposites about the mean"
    assert np.abs(((X[0::2] + X[1::2]) / 2) - mean[None, :]).max() <= 1e-15
    m = lam // 2
    assert (np.abs(d_plus).max(axis=1) > 1e-9).all(), "a pair with delta=0 carries no information"
    for i in range(m):
        for j in range(i + 1, m):
            assert np.abs(d_plus[i] - d_plus[j]).max() > 1e-9 and np.abs(d_plus[i] + d_plus[j]).max() > 1e-9, f"pairs {i},{j} share the same noise"
    return d_plus


@pytest.mark.parametrize("n", [10, 55])
def test_T20_mirrored_sampling(n):
    """T-20 (REQ-OPT-04): candidates come in +/- pairs, (theta+ + theta-)/2 == mean (<=1e-15), lambda even; the deltas of the different pairs
    are pairwise different (also up to sign) and each pair is EXACTLY +-delta; deltas are not reused across generations; at generation 0
    (C=I) delta/sigma0 is standard normal-like (std in [0.7,1.3])."""
    lam = 64
    opt = cmaes.make_optimizer(n, lam=lam, seed=9000)
    prev = None
    for g in range(3):
        X = opt.ask()
        d = _check_pairs(X, opt.mean, lam, n)
        if g == 0:
            z = d / 0.3
            assert 0.7 < z.std() < 1.3 and abs(z.mean()) < 0.1, "generation-0 deltas must look like sigma0*N(0,I)"
        if prev is not None:
            assert min(np.abs(d[i] - p).max() for i in range(lam // 2) for p in prev) > 1e-9, "noise reused across generations"
        prev = list(d)
        opt.tell(X, -np.sum((X - 0.5) ** 2, 1))
    with pytest.raises(ValueError):
        cmaes.make_optimizer(n, lam=lam - 1)


def test_T21_objective_undiscounted_average():
    """T-21 (REQ-OPT-02/05): f = mean_s U/T_score (no discount); the honest baseline row (theta=0) is evaluated exactly once."""
    T_score, S, lam, n = 100, 5, 4, 7
    calls = []

    def sim_fn(thetas, seeds):
        calls.append(np.array(thetas, copy=True))
        P = thetas.shape[0]
        return np.array([[1000.0 * i + s for s in range(len(seeds))] for i in range(P)])

    rng = np.random.default_rng(0)
    thetas = rng.normal(size=(lam, n))
    res = train.evaluate_generation(sim_fn, thetas, list(range(S)), T_score)
    zero_rows = sum(int((c == 0).all(axis=1).sum()) for c in calls)
    assert zero_rows == 1 and len(calls) == 1
    f_h = np.mean([0 + s for s in range(S)]) / T_score
    assert abs(res["f_honest"] - f_h) < 1e-15
    exp = np.array([np.mean([1000.0 * (i + 1) + s for s in range(S)]) / T_score for i in range(lam)])
    assert np.abs(np.asarray(res["f"]) - exp).max() < 1e-12
    assert np.abs(np.asarray(res["G"]) - (exp - f_h)).max() < 1e-12


def test_T22_seed_blocks_rotation():
    """T-22 (REQ-OPT-03, SEED-04): S_g = consecutive block of perm_k (Random(9000+k).shuffle(range(1000))) mod 1000; deterministic;
    different run_ids differ; all seeds < 1000.  v1.2: n_S in {16,...,256}; the range check (n_S not in [1,1000] -> ValueError) is T-22b."""
    def perm(k):
        p = list(range(1000))
        random.Random(9000 + k).shuffle(p)
        return p
    for k in (0, 1, 2):
        p = perm(k)
        for g in (0, 1, 7, 40):
            nS = 32
            exp = [p[i % 1000] for i in range(g * nS, (g + 1) * nS)]
            assert list(train.seed_block(k, g, nS)) == exp
            assert max(exp) < 1000
            assert len(set(exp)) == len(exp) and len(set(train.seed_block(k, g, nS))) == nS, "duplicate seed inside a block"
        assert list(train.seed_block(k, 5, 16)) == list(train.seed_block(k, 5, 16))
    assert list(train.seed_block(0, 0, 32)) != list(train.seed_block(1, 0, 32))
    for g in range(0, 80):                                  # includes blocks that straddle the 1000 boundary (g=31)
        for nS in (16, 32, 64, 128, 256):         # v1.2: n_S up to 256 (REQ-OPT-09)
            b = list(train.seed_block(2, g, nS))
            assert len(set(b)) == nS, (g, nS)


def _sphere_converges(opt, n, gens=50, centre=0.3):
    """Acceptance check for a CMA-ES: after `gens` generations on f(x)=|x-centre|^2 (maximise -f) the mean error is < 1e-3 and the step size
    has collapsed to < 0.15 (sigma0 = 0.3).  Raises AssertionError otherwise."""
    for _ in range(gens):
        X = opt.ask()
        opt.tell(X, -np.sum((X - centre) ** 2, 1))
    err = float(np.sum((np.asarray(opt.mean) - centre) ** 2))
    assert err < 1e-3, f"sphere error {err:.3e} >= 1e-3 after {gens} generations"
    assert opt.sigma < 0.15, f"sigma {opt.sigma:.3f} >= 0.15 after {gens} generations (step size not adapted)"
    return err


class _FrozenSigmaES:
    """Counter-example mock: a mirrored (mu/mu_w, lambda)-ES whose step size is NEVER updated (sigma stays 0.3)."""
    kind = "full"

    def __init__(self, n, lam, sigma0=0.3, seed=0):
        self.n, self.lam, self.mean, self.sigma = n, lam, np.zeros(n), sigma0
        self.rng = np.random.default_rng(seed)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w, self.mu = w / w.sum(), mu

    def ask(self):
        z = self.rng.standard_normal((self.lam // 2, self.n))
        self.y = np.concatenate([z[:, None], -z[:, None]], 1).reshape(self.lam, self.n)
        return self.mean + self.sigma * self.y

    def tell(self, X, f):
        idx = np.argsort(-np.asarray(f))[:self.mu]
        self.mean = self.mean + self.sigma * (self.w[:, None] * self.y[idx]).sum(0)


@pytest.mark.parametrize("n,lam", [(10, 16), (55, 64)])
def test_T23b_cma_converges_on_sphere_and_check_rejects_frozen_sigma(n, lam):
    """T-23b (REQ-OPT-01): 50 generations on a sphere: error < 1e-3 and sigma < 0.15 (n=10 full CMA, n=55 sep-CMA).  The same check must REJECT
    a counter-example optimizer whose sigma is never updated (so a CMA without step-size adaptation cannot pass)."""
    _sphere_converges(cmaes.make_optimizer(n, lam=lam, seed=9000), n)
    mock = _FrozenSigmaES(n, lam)
    with pytest.raises(AssertionError):
        _sphere_converges(mock, n)
    assert mock.sigma == 0.3


@pytest.mark.parametrize("n", [10, 55])
def test_T23_cma_resume_determinism(n):
    """T-23 (REQ-OPT-01, GPU-04): 10 generations == 6 + checkpoint/restore + 4 (mean, sigma, covariance state, <=1e-12); initial
    mean 0 and sigma0=0.3; the state really evolves (sigma and the covariance state change), so the comparison is not trivially satisfied."""
    a = cmaes.make_optimizer(n, lam=16, seed=9000)
    assert np.all(a.mean == 0) and a.sigma == 0.3
    cov0 = np.array(a.cov_state(), copy=True)
    _sphere_run(a, 10)
    assert a.sigma != 0.3 and np.abs(np.asarray(a.cov_state()) - cov0).max() > 1e-6 and np.abs(a.mean).max() > 1e-3
    b = cmaes.make_optimizer(n, lam=16, seed=9000)
    _sphere_run(b, 6)
    sd = b.state_dict()
    c = cmaes.make_optimizer(n, lam=16, seed=1)           # different seed: everything must come from the state
    c.load_state_dict(sd)
    _sphere_run(c, 4)
    assert np.abs(a.mean - c.mean).max() <= 1e-12
    assert abs(a.sigma - c.sigma) <= 1e-12
    assert np.abs(np.asarray(a.cov_state()) - np.asarray(c.cov_state())).max() <= 1e-12


def test_T24_param_decision_rules():
    """T-24 (REQ-OPT-08..11, v1.2): decision function vs the s5.5 rule (lambda before T_train; n_S grid {16,...,256}, v1.2 cap 256 not 128),
    infeasible cases, r=0.9 flag, MDE_est.  The v1.2 final values (n_S=256, 416/499 generations) are pinned by T-24b."""
    assert budget.mde_est_defaults() == {0.5: 0.0058, 0.9: 0.0016}
    t_step = lambda B: 1e-4 * B                              # ms
    Ts = (20000, 40000, 100000)
    sd = lambda a, b: {(0.5, T): a for T in Ts} | {(0.9, T): b for T in Ts}
    res = budget.decide_params(t_step, 8192, 9000.0, sd(0.005, 0.0005))
    assert res["feasible"] and (res["lam"], res["T_train"], res["n_S"], res["G_gens"]) == (256, 40000, 16, 549)
    assert res["r09_underpowered"] is False
    assert budget.decide_params(t_step, 8192, 9000.0, sd(0.005, 0.005))["r09_underpowered"] is True

    def ref(tstep, Bmax, W, sd05):
        for lam in (256, 128, 64, 32):
            for T in (100000, 40000, 20000):
                nS = next((n for n in (16, 32, 64, 128, 256) if sd05[T] / n ** 0.5 <= 0.25 * 0.0058), None)
                if nS is None:
                    return None
                if lam * nS <= Bmax and 300 * T * tstep(lam * nS) / 1000 <= W:
                    return lam, T, nS, min(1000, int(W // (T * tstep(lam * nS) / 1000)))
        return None
    rng = np.random.default_rng(1)
    for _ in range(40):
        c = rng.uniform(1e-5, 5e-3)
        ts = lambda B, c=c: c * B
        Bmax = int(rng.choice([512, 1024, 4096, 8192]))
        W = float(rng.choice([3600.0, 9000.0, 20000.0]))
        sd05 = {T: float(rng.uniform(0.003, 0.05)) for T in Ts}      # up to 0.05: needs n_S=256 (<=0.0232) or is infeasible (>0.0232)
        got = budget.decide_params(ts, Bmax, W, {(0.5, T): sd05[T] for T in Ts} | {(0.9, T): 0.001 for T in Ts})
        exp = ref(ts, Bmax, W, sd05)
        if exp is None:
            assert got["feasible"] is False
        else:
            assert got["feasible"] and (got["lam"], got["T_train"], got["n_S"], got["G_gens"]) == exp
    assert budget.decide_params(t_step, 100, 9000.0, sd(0.005, 0.0005))["feasible"] is False      # B_max too small
    assert budget.decide_params(t_step, 8192, 9000.0, sd(1.0, 0.0005))["feasible"] is False       # even n_S=256 insufficient (v1.2 cap)


def test_T25_val_checkpoint_selection():
    """T-25 (REQ-OPT-06/07, SEED-05, S1-14): checkpoint = max validation G; no test-seed parameter; the freeze file must register BOTH
    cold and warm start plus the pilot decision, and S1 / main configs use the same start."""
    recs = [dict(gen=50, G_val=0.01), dict(gen=100, G_val=0.03), dict(gen=150, G_val=0.03), dict(gen=200, G_val=0.02)]      # v1.2: validation every 50 generations
    assert train.select_checkpoint(recs)["gen"] == 100
    assert not any("test" in p for p in inspect.signature(train.select_checkpoint).parameters)
    from conftest import make_freeze
    freeze = make_freeze(warm_start=dict(recipe="lstsq to best S_HW d-output"), pilot=dict(cold_val_G=0.1, warm_val_G=0.25, G_star_ref=0.28), chosen_start="warm")
    assert s1.check_freeze_registration(freeze) is True
    for missing in ("cold_start", "warm_start", "pilot", "chosen_start"):
        bad = {k: v for k, v in freeze.items() if k != missing}
        with pytest.raises(ValueError):
            s1.check_freeze_registration(bad)
    assert s1.load_config(freeze, "s1")["start"] == s1.load_config(freeze, "main")["start"] == "warm"


def test_T64_optimizer_selection_by_dim():
    """T-64 (REQ-OPT-01, ACT-03): n=55 and 273 -> sep (diagonal covariance); n<=50 -> full; n>64 rejected unless diagnostic (<=300)."""
    o = cmaes.make_optimizer(55)
    assert o.kind == "sep" and np.asarray(o.cov_state()).shape == (55,)
    o = cmaes.make_optimizer(273, diagnostic=True)
    assert o.kind == "sep" and np.asarray(o.cov_state()).shape == (273,)
    o = cmaes.make_optimizer(273, lam=128, diagnostic=True)               # the sep variant really optimises (step size adapts, mean moves)
    for _ in range(30):
        X = o.ask()
        o.tell(X, -np.sum((X - 0.3) ** 2, 1))
    assert o.sigma != 0.3 and np.sum((o.mean - 0.3) ** 2) < 0.25 * 273 * 0.09
    for n in (10, 50):
        o = cmaes.make_optimizer(n)
        assert o.kind == "full" and np.asarray(o.cov_state()).shape == (n, n)
    assert cmaes.make_optimizer(51).kind == "sep" and cmaes.make_optimizer(64).kind == "sep"
    with pytest.raises(ValueError):
        cmaes.make_optimizer(65)
    with pytest.raises(ValueError):
        cmaes.make_optimizer(301, diagnostic=True)


# ====================================================================================================================== spec v1.2 additions
def test_T22b_seed_block_range_check_SEED07():
    """T-22b (REQ-SEED-07): train.seed_block(run_id, g, n_S) raises ValueError for n_S not in [1, 1000] (n_S>1000 wraps the block and repeats seeds);
    n_S=1 and n_S=1000 are accepted and give blocks of exactly that many UNIQUE seeds (also the block that straddles position 1000, g=31 for
    n_S=32; and n_S=1000 for every g is a full permutation); n_S=1001 would have repeated a seed, so the check must be ON the boundary."""
    for bad in (0, -1, -16, 1001, 1024, 2000):
        with pytest.raises(ValueError):
            train.seed_block(0, 0, bad)
        with pytest.raises(ValueError):
            train.seed_block(3, 31, bad)              # the straddling-block position must not bypass the check
    assert len(list(train.seed_block(0, 0, 1))) == 1
    for g in (0, 1, 5):
        b = list(train.seed_block(1, g, 1000))
        assert sorted(b) == list(range(1000)), "n_S=1000 must give the whole training set exactly once"
    for n_S in (1, 2, 7, 256, 999, 1000):
        for g in range(0, 40):
            b = list(train.seed_block(4, g, n_S))
            assert len(b) == n_S and len(set(b)) == n_S, (n_S, g)
    # the function really would repeat seeds beyond 1000 (so the guard is not vacuous): reference arithmetic
    assert len({i % 1000 for i in range(0, 1001)}) == 1000


def test_T24b_v12_final_values_OPT09_OPT10():
    """T-24b (REQ-OPT-09/10, v1.2 s5.5): the decision function reproduces the v1.2 table.  (a) sd_CRN=0.021732 (worst-sigma median, r=0.5) with
    MDE_est 0.0058: n_S=128 does NOT satisfy (0.021732/sqrt(128)=0.00192>0.00145) but 256 does (0.00136) -> n_S=256, lam=256 (B=65536<=B_max
    2097152); with W_run=9000 s and t_step=1.0812 ms: T_train=1e5 and 4e4 time out, T_train=20000 passes; G_gens=min(1000,floor(9000/21.624))=416
    (r=0.5 S1), and W_run=10800 -> 499 (main r=0.9).  (b) sd just above the 256 limit (0.0233) -> infeasible (REQ-STOP-02); sd=0.0232 -> n_S=256.
    (c) the smaller grid steps are still chosen when sufficient (sd=0.0045 -> n_S=16 at the candidate list start)."""
    Ts = (20000, 40000, 100000)
    t_step = lambda B: 1.0812                                  # ms, independent of B (as in MEAS v1)
    sd = lambda a, b: {(0.5, T): a for T in Ts} | {(0.9, T): b for T in Ts}
    B_max = 2097152
    r05 = budget.decide_params(t_step, B_max, 9000.0, sd(0.021732, 0.020915))
    assert r05["feasible"], r05
    assert (r05["lam"], r05["T_train"], r05["n_S"], r05["G_gens"]) == (256, 20000, 256, 416)
    r09 = budget.decide_params(t_step, B_max, 10800.0, sd(0.021732, 0.020915))
    assert (r09["lam"], r09["T_train"], r09["n_S"], r09["G_gens"]) == (256, 20000, 256, 499)
    assert r09["r09_underpowered"] is True                    # r=0.9: 0.020915/16=0.0013 > 0.25*0.0016=0.0004 (REQ-OPT-11)
    # n_S boundary: sd/sqrt(256) <= 0.00145  <=>  sd <= 0.0232
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0232, 0.001))["n_S"] == 256
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0233, 0.001))["feasible"] is False
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0230, 0.001))["n_S"] == 256
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0165, 0.001))["n_S"] == 256       # 0.0165/sqrt(128)=0.001458 > 0.00145
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0164, 0.001))["n_S"] == 128       # 0.0164/sqrt(128)=0.00145
    assert budget.decide_params(t_step, B_max, 9000.0, sd(0.0045, 0.001))["n_S"] == 16
    # B_max binds: lam*n_S=65536 > 4096 -> lam drops to 16? not in grid -> lam=32 gives 32*256=8192>4096 -> infeasible
    assert budget.decide_params(t_step, 4096, 9000.0, sd(0.021732, 0.0005))["feasible"] is False
    # G_gens formula floor and 1000 cap
    ts_fast = lambda B: 0.05
    g = budget.decide_params(ts_fast, B_max, 9000.0, sd(0.005, 0.0005))
    assert g["G_gens"] == 1000 and g["feasible"]
    g2 = budget.decide_params(lambda B: 1.0, B_max, 9000.0, sd(0.005, 0.0005))     # T=1e5 -> 100 s/gen -> 90 < 300 ; 4e4 -> 40 s -> 225 <300; 2e4 -> 20 s -> 450
    assert (g2["T_train"], g2["G_gens"]) == (20000, 450)


def _val_fn_calls(tmp_path, n_gens, **cell_extra):
    """Run train.train_segment on a cheap synthetic objective; return the generations at which val_fn was called."""
    calls = []
    n, lam, n_S = 4, 4, 2

    def sim_fn(rows, seeds):
        return np.arange(rows.shape[0], dtype=float)[:, None] * np.ones((1, len(seeds)))

    def val_fn(theta):
        calls.append(len(calls))
        return 0.01 * len(calls)

    cell = dict(name="cellv", n=n, lam=lam, n_S=n_S, T_score=100, T_train=100, sigma0=0.3)
    cell.update(cell_extra)
    rec = train.train_segment(cell, 0, 0, n_gens, sim_fn, val_fn, root=str(tmp_path), log=lambda s: None)
    return [r["gen"] for r in rec["val_records"]], len(calls)


def test_T25b_validation_every_50_generations_OPT06(tmp_path):
    """T-25b (REQ-OPT-06 v1.2): validation (val_fn) runs every 50 generations, NOT every 25 (v1.1): over 100 generations with no explicit
    val_every, the records are at generations 50 and 100 and val_fn is called exactly twice; over 149 generations only once (gen 50 -> not 100).
    The checkpoint selected from those records is the max-G_val one.  The segmentation cost model (REQ-GPU-08) and the command line default
    must agree: segment_estimate_s counts one validation per 50 generations; `--val-every` defaults to 50."""
    gens, n_calls = _val_fn_calls(tmp_path / "a", 100)
    assert gens == [50, 100] and n_calls == 2, (gens, n_calls)
    gens2, n2 = _val_fn_calls(tmp_path / "b", 149)
    assert gens2 == [50, 100] and n2 == 2 and 149 not in gens2
    gens3, n3 = _val_fn_calls(tmp_path / "c", 49)
    assert gens3 == [] and n3 == 0, "no validation before generation 50"
    # cost model: t_val per 50 generations
    ts = lambda B: 1.0
    base = budget.segment_estimate_s(0, 100, ts, 100, 8, 32)
    assert budget.segment_estimate_s(0, 100, ts, 100, 8, 32, t_val_s=10.0) == pytest.approx(base + 2 * 10.0)
    assert budget.segment_estimate_s(0, 49, ts, 100, 8, 32, t_val_s=10.0) == pytest.approx(budget.segment_estimate_s(0, 49, ts, 100, 8, 32))
    assert budget.segment_estimate_s(0, 50, ts, 100, 8, 32, t_val_s=10.0) == pytest.approx(budget.segment_estimate_s(0, 50, ts, 100, 8, 32) + 10.0)
    assert budget.segment_estimate_s(50, 100, ts, 100, 8, 32, t_val_s=10.0) == pytest.approx(budget.segment_estimate_s(50, 100, ts, 100, 8, 32) + 10.0)
    # command line: the budget flags are gone (pre-S1 hardening); the 50-generation validation interval comes from the frozen plan
    tree = ast.parse(open(os.path.join(ROOT, "arbitration", "rl", "train.py"), encoding="utf-8").read())
    flags = [c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument"
             and c.args and isinstance(c.args[0], ast.Constant)]
    assert "--val-every" not in flags
    plan = json.load(open(os.path.join(ROOT, "results", "direction2", "run_plan.json"), encoding="utf-8"))
    assert {c["val_every"] for c in plan["cells"]} == {50}


# ---- REQ-OPT-12: obs_version must be passed explicitly --------------------------------------------------------------------------------
_OBS_CALLS = {"simulate", "init_state", "_Bank", "MLPPolicy"}


def _calls_without_obs_version(src):
    """Test-local scanner: calls to env.simulate / init_state / _Bank / MLPPolicy(...) / MLPPolicy.from_blocks(...) that do not carry an
    explicit `obs_version=` keyword (a **kwargs expansion or `obs_version=None` does not count).  Returns ['line:name', ...]."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        is_from_blocks = isinstance(f, ast.Attribute) and f.attr == "from_blocks" and isinstance(f.value, (ast.Name, ast.Attribute)) \
            and getattr(f.value, "id", getattr(f.value, "attr", "")) == "MLPPolicy"
        if name not in _OBS_CALLS and not is_from_blocks:
            continue
        kws = {k.arg: k.value for k in node.keywords}
        v = kws.get("obs_version")
        if v is None or (isinstance(v, ast.Constant) and v.value is None):
            out.append(f"{node.lineno}:{name}")
    return out


def test_T66_obs_version_explicit_in_training_code_OPT12(monkeypatch):
    """T-66 (REQ-OPT-12, v1.2): new training / validation / evaluation code never relies on the obs_version default.
    (1) Scanner self-check: it flags simulate()/init_state()/_Bank()/MLPPolicy()/MLPPolicy.from_blocks() without obs_version (also with
    obs_version=None or only **kw) and accepts explicit ones.  (2) Static: arbitration/rl/{train,s1,final_eval,vulns}.py contain no such call.
    (3) Run time: train.make_sim_fn / make_val_fn forward an explicit obs_version both to env.simulate and to every MLPPolicy they build
    (a policy with obs_version=None, or a simulate call without the keyword, fails; F=27 + 'announced' is passed through, not turned into 'public')."""
    bad = ("env.simulate('M3C', 10, cfg, pols, seeds, r=0.5)\n_env.init_state('M3C', cfg, 0.5, [0])\n_Bank(p, pid, dev, 25)\nMLPPolicy(F=25)\n"
           "policy.MLPPolicy.from_blocks(a, b, c, d)\nMLPPolicy(F=25, obs_version=None)\nenv.simulate('M3C', 10, cfg, pols, seeds, r=0.5, **kw)")
    assert len(_calls_without_obs_version(bad)) == 7
    good = ("env.simulate('M3C', 10, cfg, pols, seeds, r=0.5, obs_version='public')\nMLPPolicy(F=27, obs_version=ver)\n"
            "MLPPolicy.from_blocks(a, b, c, d, obs_version='announced')\n_Bank(p, pid, dev, 25, obs_version='oracle')\nsimulate_vuln(a)\nsimulate_thing(b)")
    assert _calls_without_obs_version(good) == []
    offenders = {}
    for fn in ("train.py", "s1.py", "final_eval.py", "vulns.py"):
        path = os.path.join(ROOT, "arbitration", "rl", fn)
        offenders[fn] = _calls_without_obs_version(open(path, encoding="utf-8").read()) if os.path.exists(path) else ["file missing"]
    assert all(not v for v in offenders.values()), f"calls relying on the obs_version default (REQ-OPT-12): {offenders}"

    from arbitration.rl import env, policy
    seen = dict(sim=[], pol=[])
    real_sim, real_init = env.simulate, policy.MLPPolicy.__init__

    def spy_sim(mech, T, cfg, pols, seeds, **kw):
        seen["sim"].append(kw.get("obs_version", "<missing>"))
        return real_sim(mech, T, cfg, pols, seeds, **kw)

    def spy_init(self, *a, **kw):
        real_init(self, *a, **kw)
        seen["pol"].append(self.obs_version)
    monkeypatch.setattr(env, "simulate", spy_sim)
    monkeypatch.setattr(policy.MLPPolicy, "__init__", spy_init)
    cfg = dict(env.m3c_config(0.5), L=20.0)
    for F, ver in ((25, "public"), (27, "announced"), (27, "oracle")):
        seen["sim"].clear(), seen["pol"].clear()
        sim = call_new_kw(train.make_sim_fn, dict(obs_version=ver), "M3C", 60, cfg, 0.5, dev="cpu", F=F)
        th = np.zeros((2, policy.n_params(F, 2)))
        sim(th, [0, 1])
        assert seen["sim"] == [ver] and seen["pol"] == [ver, ver], (F, ver, seen)
        seen["sim"].clear(), seen["pol"].clear()
        val = call_new_kw(train.make_val_fn, dict(obs_version=ver), "M3C", 60, cfg, 0.5, dev="cpu", F=F)
        val(np.zeros(policy.n_params(F, 2)))
        assert seen["sim"] and set(seen["sim"]) == {ver} and set(seen["pol"]) == {ver}, (F, ver, seen)


def test_T67_nS_coverage_disclosure_in_freeze_OPT13():
    """T-67 (REQ-OPT-13, v1.2): the freeze registration (REQ-S1-14) must carry the n_S coverage disclosure: key `nS_coverage_disclosure`, a text that
    names the single-pair worst case (n_S ~ 565) and says it is not covered ('not covered' or 無法涵蓋/未涵蓋).  A freeze dict without the key, with an empty text, or whose text lacks
    the 565 / 'not covered' content is rejected with ValueError; the disclosure does not change load_config's decision keys.  (The wording in the
    final report is checked by T-68.)"""
    from conftest import make_freeze
    ok_txt = "n_S=256 covers the median worst case; the single-pair worst case (sd_CRN=0.034465) needs n_S~565 and is NOT covered (REQ-OPT-13)"
    freeze = make_freeze(nS_coverage_disclosure=ok_txt)
    assert s1.check_freeze_registration(freeze) is True
    assert s1.load_config(freeze, "s1")["n_S"] == 256
    zh = dict(freeze, nS_coverage_disclosure="n_S=256 滿足三種 σ 的中位數最壞值；單一隨機對的最壞情況需 n_S≈565，目前無法涵蓋（REQ-OPT-13）")
    assert s1.check_freeze_registration(zh) is True          # Chinese wording of the same statement is accepted
    for bad in (None, "", "n_S=256 is enough", "n_S~565 would be needed", "n_S~565 is covered", "565"):   # no number / no 'not covered'
        # (the last two: 'covered' without negation, and the bare number without any statement)
        d = dict(freeze)
        if bad is None:
            d.pop("nS_coverage_disclosure")
        else:
            d["nS_coverage_disclosure"] = bad
        with pytest.raises(ValueError):
            s1.check_freeze_registration(d)
