"""Correctness tests of the exact benchmark (arbitration/dp.py).

The DP is checked three independent ways: closed forms (N=1, N=2 symmetric),
a brute-force search over periodic schedules evaluated by a tiny stand-alone
step simulator (not sharing code with dp.py), and the full simulation
(model.run) -- by cumulative-reward slopes, which cancel the start-up transient.
"""
import itertools
import math
import numpy as np
import pytest
from arbitration import run, dp
from arbitration.model import make_config, ring_distance, RING, SPEED, POLICIES, DP_POLICIES

LEG = make_config("legacy")
SHK = make_config("v2")


def _L(posts, i, j, svc):
    return max(1, math.ceil(float(ring_distance(posts[i], posts[j])) / SPEED)) + svc


def periodic_value(posts, rates, seq, svc=5, warm=1200, meas=1200):
    """Average mean health of repeating `seq` (cyclic visit order) under LINEAR damage,
    measured after `warm` steps (so every robot is in steady state; 1/rate >> warm is
    NOT needed because a never-served robot is dead by then only if its rate is high --
    sequences that starve a robot are simply bad). Independent step-level simulator:
    damage, accumulate, reset served robot last."""
    n = len(posts)
    h = np.ones(n)
    pos = seq[-1]
    tot, steps, t, i, on = 0.0, 0, 0, 0, False
    while not (on and steps >= meas and i % len(seq) == 0):
        if not on and t >= warm and i % len(seq) == 0:
            on = True                     # start measuring at a period boundary
        a = seq[i % len(seq)]
        i += 1
        L = _L(posts, pos, a, svc)
        for _ in range(L):
            h = np.maximum(0.0, h - rates)
            t += 1
            if on:
                tot += h.mean()
                steps += 1
        h[a] = 1.0
        pos = a
    return tot / steps


def best_periodic(posts, rates, max_len=6):
    n = len(posts)
    best = -1.0
    for m in range(1, max_len + 1):
        for seq in itertools.product(range(n), repeat=m):
            best = max(best, periodic_value(posts, rates, list(seq), warm=800, meas=800))
    return best


# ------------------------------------------------------------- building blocks
def test_damage_closed_form_matches_monte_carlo():
    dy = dp.Dyn(0.004, "shock", 0.01, 0.8)
    rng = np.random.default_rng(1)
    n, L = 400_000, 40
    h = np.ones(n)
    means = []
    for _ in range(L):
        h = np.maximum(0, h - dy.base - (rng.random(n) < dy.p) * rng.exponential(dy.m, n))
        means.append(h.mean())
    phi = dy._phi_k(np.arange(1, L + 1), 1.0)
    assert np.allclose(means, phi, atol=3e-3)
    # starting below 1 as well
    h = np.full(n, 0.3)
    for _ in range(L):
        h = np.maximum(0, h - dy.base - (rng.random(n) < dy.p) * rng.exponential(dy.m, n))
    assert abs(h.mean() - float(dy.phi(L, 0.3))) < 3e-3


@pytest.mark.parametrize("damage,frac", [("linear", 0.5), ("shock", 0.5), ("shock", 0.9)])
def test_transition_rows_preserve_mean_and_mass(damage, frac):
    dy = dp.Dyn(0.01, damage, 0.01, frac)
    K = 25
    xs = np.linspace(0, 1, 11)
    w = dy.trans(30, xs, K)
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all()
    mean_grid = w @ (np.arange(K + 1) / K)
    assert np.allclose(mean_grid, dy.phi(30, xs), atol=1e-9)      # E[(x - S)^+] exactly


# ----------------------------------------------------------------- N = 1
@pytest.mark.parametrize("cls", [dp.FullDP, dp.AgeDP])
def test_n1_linear_closed_form(cls):
    r, svc = 0.01, 5
    posts, rates = np.array([10.0]), np.array([r])
    # serve the only robot again and again: cycle 1+svc steps, ages 1..6
    exact = np.mean([max(0.0, 1 - r * a) for a in range(1, 1 + 1 + svc)])
    sol = cls(posts, rates, LEG, 50) if cls is dp.FullDP else cls(posts, rates, LEG)
    assert sol.g == pytest.approx(exact, abs=2e-4 if cls is dp.FullDP else 1e-5)


def test_n1_shock_closed_form_matches_monte_carlo():
    r, svc = 0.01, 5
    posts, rates = np.array([10.0]), np.array([r])
    dy = dp.Dyn(r, "shock", SHK.shock_rate, SHK.shock_frac)
    exact = dy._phi_k(np.arange(1, 1 + 1 + svc), 1.0).mean()
    assert dp.AgeDP(posts, rates, SHK).g == pytest.approx(exact, abs=1e-5)
    assert dp.FullDP(posts, rates, SHK, 60).g == pytest.approx(exact, abs=2e-3)
    rng = np.random.default_rng(0)
    n = 300_000
    h, tot = np.ones(n), 0.0
    for _ in range(6):
        h = np.maximum(0, h - dy.base - (rng.random(n) < dy.p) * rng.exponential(dy.m, n))
        tot += h.mean()
    assert tot / 6 == pytest.approx(exact, abs=2e-3)


# ------------------------------------------------------- N = 2 symmetric, by hand
def test_n2_symmetric_alternation_by_hand():
    r = 0.005
    posts, rates = np.array([0.0, 50.0]), np.array([r, r])
    L = 50 + 5
    # alternate: each cycle the served robot is at age 1.., the other at age 56..
    own = sum(max(0, 1 - r * a) for a in range(1, L + 1))
    oth = sum(max(0, 1 - r * a) for a in range(L + 1, 2 * L + 1))
    by_hand = (own + oth) / (2 * L)
    assert by_hand == pytest.approx(periodic_value(posts, rates, [0, 1]), abs=1e-9)
    ja = dp.AgeDP(posts, rates, LEG, tol=1e-8).g
    assert ja == pytest.approx(by_hand, abs=2e-5)           # alternation is optimal here
    assert ja == pytest.approx(best_periodic(posts, rates, 4), abs=2e-5)
    # grid solution converges to it from above as K grows (interpolation diffusion is optimistic)
    errs = [dp.FullDP(posts, rates, LEG, K).g - by_hand for K in (30, 120, 240)]
    assert errs[0] > errs[1] > errs[2] > -1e-6 and errs[2] < 1e-3
    # symmetry: relabelling the robots leaves J* unchanged
    assert dp.AgeDP(posts[::-1].copy(), rates, LEG, tol=1e-8).g == pytest.approx(ja, abs=2e-5)


# ----------------------------- N = 3, legacy: age DP is exact; brute force agrees
@pytest.mark.parametrize("seed", [0, 3])
def test_age_dp_matches_best_periodic_schedule(seed):
    posts, rates, _ = dp.instance(seed, 3, 0.004, 0.6)
    ja = dp.AgeDP(posts, rates, LEG, tol=1e-8).g
    bp = best_periodic(posts, rates, 6)
    # optimum over ALL policies >= every periodic schedule; here it is attained
    assert ja >= bp - 2e-5
    assert ja == pytest.approx(bp, abs=1e-3)


def test_legacy_full_equals_age_only_and_k_converges():
    posts, rates, _ = dp.instance(0, 3, 0.004, 0.6)
    ja = dp.AgeDP(posts, rates, LEG, tol=1e-8).g
    errs = {K: abs(dp.FullDP(posts, rates, LEG, K).g - ja) for K in (10, 20, 40)}
    assert errs[40] < 2e-3 and errs[40] < errs[10]
    assert errs[10] < 2e-2


def test_shock_frac_zero_is_the_linear_model():
    posts, rates, _ = dp.instance(1, 3, 0.004, 0.6)
    c0 = make_config("v2", shock_frac=0.0)
    assert dp.AgeDP(posts, rates, c0, tol=1e-8).g == pytest.approx(dp.AgeDP(posts, rates, LEG, tol=1e-8).g, abs=1e-9)
    assert dp.FullDP(posts, rates, c0, 20).g == pytest.approx(
        dp.FullDP(posts, rates, LEG, 20).g, abs=1e-9)


def test_value_of_state_information_nonnegative_under_shocks():
    posts, rates, _ = dp.instance(0, 3, 0.004, 0.6)
    jf = dp.FullDP(posts, rates, SHK, 40).g
    ja = dp.AgeDP(posts, rates, SHK).g
    assert jf - ja > 0.002             # seeing h is worth something when shocks are hidden
    # more hidden-shock mass -> more to gain from seeing h
    c9 = make_config("v2", shock_frac=0.9)
    assert dp.FullDP(posts, rates, c9, 40).g - dp.AgeDP(posts, rates, c9).g > jf - ja


# ------------------------------------------------- DP policy in the simulation
def _slope(policy, seed, model, n=3, rate=0.004, spread=0.6, a=6000, b=30000, **kw):
    ma = run(policy, n=n, rate=rate, spread=spread, seed=seed, steps=a, model=model, **kw).mean_health
    mb = run(policy, n=n, rate=rate, spread=spread, seed=seed, steps=b, model=model, **kw).mean_health
    return (b * mb - a * ma) / (b - a)           # cancels the start-up transient


@pytest.mark.parametrize("policy", ["dp_full", "dp_age"])
def test_dp_policy_reproduces_predicted_value_in_simulation_legacy(policy):
    posts, rates, _ = dp.instance(0, 3, 0.004, 0.6)
    cfg = make_config("legacy", dp_k=40)
    pred = dp.build_policy(policy, posts, rates, cfg).g
    assert _slope(policy, 0, "legacy", dp_k=40) == pytest.approx(pred, abs=2.5e-3)
    # DP-age is exact in the legacy model: tight
    if policy == "dp_age":
        assert _slope(policy, 0, "legacy") == pytest.approx(pred, abs=2e-4)


def test_no_simulated_policy_beats_the_optimum_legacy():
    posts, rates, _ = dp.instance(2, 3, 0.006, 0.6)
    ja = dp.AgeDP(posts, rates, LEG, tol=1e-8).g
    for p in ("none", "none_random", "greedy_true", "index_true"):
        assert _slope(p, 2, "legacy", rate=0.006) <= ja + 5e-4, p


def test_dp_policies_are_deterministic_and_registered():
    assert set(DP_POLICIES) == {"dp_full", "dp_age"} and not set(DP_POLICIES) & set(POLICIES)
    with pytest.raises(ValueError):
        run("dp_full", n=8, steps=10)
    kw = dict(n=3, rate=0.004, spread=0.6, seed=5, steps=600, model="v2")
    for p in ("dp_full", "dp_age"):
        assert run(p, **kw) == run(p, **kw)


def test_dp_policy_n1_simulation():
    r = 0.01
    pred = np.mean([max(0.0, 1 - r * a) for a in range(1, 7)])
    for p in ("dp_full", "dp_age"):
        got = _slope(p, 0, "legacy", n=1, rate=r, spread=0.0, a=2000, b=8000)
        assert got == pytest.approx(pred, abs=5e-3)


def test_service_option_is_threaded_through():
    a = run("none", n=3, rate=0.004, seed=1, steps=800, service=5)
    b = run("none", n=3, rate=0.004, seed=1, steps=800, service=20)
    assert a == run("none", n=3, rate=0.004, seed=1, steps=800) and a != b
    posts, rates, _ = dp.instance(1, 3, 0.004, 0.6)
    assert dp.AgeDP(posts, rates, make_config(service=20)).g < dp.AgeDP(posts, rates, LEG, tol=1e-8).g


def test_dp_sweep_quick_structure():
    from arbitration import experiments as ex
    out = ex.dp_sweep(ex.QUICK)
    r0 = str(ex.QUICK["dp_rates"][0])
    leg, v2 = out["main"]["legacy"][r0], out["main"]["v2"][r0]
    assert set(ex.DP_ARMS) <= set(leg["gap_vs_J_full"])
    assert abs(leg["VoI_state"]["mean"]) < 2e-2          # legacy: age == full up to the K=10 grid
    assert v2["VoI_state"]["mean"] > leg["VoI_state"]["mean"]
    reg = out["region"]
    k = f"{ex.QUICK['dp_shock_fracs'][-1]}|{ex.QUICK['dp_spreads'][-1]}"
    assert k in reg["heat"][str(ex.QUICK["dp_region_rates"][0])]
    # nothing in the simulation beats J*(full) by more than grid + start-up slack
    assert min(leg["gap_vs_J_full"][a]["mean"] for a in ex.DP_ARMS) > -0.05
