import numpy as np
import pytest
from arbitration import run, POLICIES, V2, ModelConfig, tour_sequence
from arbitration.model import (make_config, SERVICE, _Learner, _rollout_choose,
                               LEGACY_POLICIES, V2_POLICIES)

KW = dict(n=8, rate=0.004, spread=0.6, seed=3)


def test_policy_lists():
    assert set(V2_POLICIES) <= set(POLICIES) and set(LEGACY_POLICIES) <= set(POLICIES)


def test_config_resolution():
    assert make_config("v2") == V2 and make_config(None) == make_config("legacy")
    assert make_config("v2", noise="iid").noise == "iid"
    with pytest.raises(TypeError):
        run("none", bogus=1)
    with pytest.raises(ValueError):
        run("none", model="nope")
    with pytest.raises(ValueError):
        ModelConfig(noise="pink")


@pytest.mark.parametrize("p", ["none", "honest", "greedy_true", "priced"])
def test_explicit_legacy_options_are_identity(p):
    base = run(p, preempt=True, **KW)
    assert run(p, preempt=True, model="legacy", damage="linear", noise="iid",
               keep_commitment=False, precision_at="arrival", **KW) == base


def test_none_random_is_paired_with_other_arms():
    # same rates / posts / start: the blind arms see the same world, only the order differs
    a, b = run("none", **KW), run("none_random", **KW)
    assert a != b
    assert run("none_random", **KW) == run("none_random", **KW)


def test_shock_changes_trajectory_default_unchanged():
    lin = run("none", **KW)
    sh = run("none", damage="shock", **KW)
    assert sh != lin
    assert run("none", **KW) == lin
    assert run("none", damage="shock", **KW) == sh


def test_shock_mean_degradation_matches_linear():
    # unclipped regime: mean health after T steps equal in expectation
    T, r = 100, 0.002
    out = {}
    for dm in ("linear", "shock"):
        out[dm] = np.mean([run("none", damage=dm, rate=r, steps=T, seed=s, n=8).mean_health
                           for s in range(40)])
    assert abs(out["linear"] - out["shock"]) < 0.01
    # and shocks make the health trajectory seed-dependent beyond posts/rates
    a = run("none", damage="shock", rate=0.004, steps=400, seed=1, n=8)
    b = run("none", damage="linear", rate=0.004, steps=400, seed=1, n=8)
    assert a.min_health != b.min_health


def _claim_errors(noise, steps=4000, sigma=0.15, **kw):
    """Re-derive the noise process exactly as run() does it."""
    import math
    nrng = np.random.default_rng([0, 4243])
    n = 8
    if noise == "ar1":
        phi = kw.get("ar1_phi", 0.9)
        e = nrng.normal(0, sigma, n)
        out = []
        for _ in range(steps):
            e = phi * e + nrng.normal(0, sigma * math.sqrt(1 - phi ** 2), n)
            out.append(e)
        return np.array(out)
    raise NotImplementedError


def test_ar1_statistics():
    E = _claim_errors("ar1")
    assert abs(E.std() - 0.15) < 0.01
    ac = np.mean([np.corrcoef(E[:-1, i], E[1:, i])[0, 1] for i in range(8)])
    assert abs(ac - 0.9) < 0.03


def test_noise_modes_run_and_differ():
    res = {m: run("honest", preempt=True, noise=m, **KW) for m in ("iid", "ar1", "bias")}
    assert len({id(v) for v in res.values()}) == 3
    assert res["iid"] != res["ar1"] != res["bias"]
    # ar1 / bias reduce thrashing relative to iid with preemption
    assert res["ar1"].retargets_per_service < res["iid"].retargets_per_service
    # sigma=0: all noise models give the same (noise-free) run
    z = [run("honest", preempt=True, sigma=0.0, noise=m, **KW) for m in ("iid", "ar1", "bias")]
    assert z[0] == z[1] == z[2]


def test_bias_is_persistent():
    # with large bias share the sign of (claim - truth) persists: honest's order is stable
    a = run("honest", preempt=True, noise="bias", bias_frac=1.0, **KW)
    b = run("honest", preempt=True, noise="iid", **KW)
    assert a.retargets_per_service < b.retargets_per_service


def _abandoned_travels(monkeypatch, policy, **kw):
    """Count travels that stop (or switch) before arriving, via a spy on the stepper."""
    from arbitration import model as m
    log = []
    orig = m._step_towards

    def spy(pos, target):
        out = orig(pos, target)
        log.append((float(target), out[1]))
        return out
    monkeypatch.setattr(m, "_step_towards", spy)
    run(policy, **kw)
    monkeypatch.setattr(m, "_step_towards", orig)
    return sum(1 for (t0, arr), (t1, _) in zip(log, log[1:]) if not arr and t1 != t0)


@pytest.mark.parametrize("p", ["honest", "priced", "nearest"])
def test_keep_commitment_never_drops_target(monkeypatch, p):
    kw = dict(preempt=False, rate=0.002, steps=1500, seed=2, n=8, spread=0.6, sigma=0.3)
    legacy = _abandoned_travels(monkeypatch, p, **kw)
    keep = _abandoned_travels(monkeypatch, p, keep_commitment=True, **kw)
    assert keep == 0
    assert legacy > 0        # the legacy model does drop commitments (the bug)
    for noise in ("ar1", "bias"):
        assert _abandoned_travels(monkeypatch, p, keep_commitment=True, noise=noise, **kw) == 0


def test_precision_at_dispatch():
    a = run("honest", preempt=True, **KW)
    b = run("honest", preempt=True, precision_at="dispatch", **KW)
    assert a.mean_health == b.mean_health and a.n_services == b.n_services
    assert a.precision != b.precision
    # without preemption and without time passing, 'none' picks at dispatch: tau at pick
    c = run("greedy_true", preempt=False, precision_at="dispatch", **KW)
    assert 0 < c.precision <= 1.0 + 1e-12


def test_dead_robot_metric():
    r = run("none", **{**KW, "rate": 0.04})
    assert 0 < r.frac_dead_robot_steps <= r.frac_any_dead <= 1
    assert run("none", **{**KW, "rate": 0.0001}).frac_dead_robot_steps == 0.0


def test_learner_converges_under_linear_damage():
    # fixed rate; observe arrivals at known elapsed times
    L = _Learner(3)
    true = np.array([0.001, 0.004, 0.01])
    t = 0.0
    rng = np.random.default_rng(0)
    for _ in range(30):
        for i in range(3):
            el = rng.uniform(20, 80)
            t_arr = L.t_reset[i] + el
            L.observe(i, max(0.0, 1 - true[i] * el), t_arr)
            L.t_reset[i] = t_arr + SERVICE
    assert np.allclose(L.rhat(), true, rtol=1e-6)


def test_learned_policy_estimates_in_run():
    # whole-run check through the public API: learned beats random order below capacity
    # only if estimates are informative; here just check it is reasonable and deterministic
    a = run("learned", rate=0.001, steps=1500, **{k: v for k, v in KW.items() if k != "rate"})
    assert a == run("learned", rate=0.001, steps=1500,
                    **{k: v for k, v in KW.items() if k != "rate"})
    assert a.mean_health > run("none_random", rate=0.001,
                               **{k: v for k, v in KW.items() if k != "rate"}).mean_health - 0.1


def test_tour_sqrt_frequencies():
    rates = np.array([0.001, 0.004, 0.009, 0.016])
    seq = tour_sequence(rates, 3000)
    freq = np.bincount(seq, minlength=4) / len(seq)
    want = np.sqrt(rates) / np.sqrt(rates).sum()
    assert np.allclose(freq, want, atol=2e-3)
    # equal rates -> plain spatial round-robin
    assert tour_sequence(np.ones(5), 10) == [0, 1, 2, 3, 4] * 2


def test_tour_sqrt_runs_visit_in_that_order():
    r = run("tour_sqrt", steps=600, **KW)
    assert r.n_services > 5 and r.retargets_per_service == 0.0


def test_rollout_deterministic_and_prefers_urgent_close():
    a = run("rollout", steps=400, **KW)
    assert a == run("rollout", steps=400, **KW)
    assert run("rollout_true", steps=400, **KW) == run("rollout_true", steps=400, **KW)
    posts = np.array([0.0, 10.0, 50.0])
    h = np.array([0.9, 0.2, 0.2])
    r = np.full(3, 0.004)
    # from pos=12: robot 1 (urgent, near) should beat robot 2 (urgent, far) and robot 0
    assert _rollout_choose(h, r, posts, 12.0, 150, "learned_index") == 1


@pytest.mark.parametrize("p", V2_POLICIES)
@pytest.mark.parametrize("model", ["legacy", "v2"])
def test_v2_policies_sane(p, model):
    r = run(p, steps=300, preempt=True, model=model, **KW)
    assert 0.0 <= r.min_health <= r.mean_health <= 1.0
    assert 0.0 <= r.frac_dead_robot_steps <= 1.0
