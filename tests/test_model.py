import numpy as np
import pytest
from arbitration import run, POLICIES, ring_distance
from arbitration.model import RING

KW = dict(n=8, rate=0.004, spread=0.6, seed=3, lam=0.01)

# Captured from the repo BEFORE phase E (policy then named `oracle`), seed 3.
BASELINE = {
    ("none", False): (0.703913744979024, 84, 0.0),
    # none_random re-pinned in phase 2: its permutation now uses its own RNG stream (was
    # drawn before rates, which broke pairing with the other arms). Old: (0.61526..., 56).
    ("none_random", True): (0.5596051437741811, 47, 0.0),
    ("greedy_true", False): (0.4649193593523063, 48, 0.0),
    ("greedy_true", True): (0.31187127024485983, 44, 0.75),
    ("nearest", True): (0.4112347310254482, 122, 2.6147540983606556),
    ("honest", False): (0.4926901954492589, 49, 0.0),
    ("honest", True): (0.2568306870411002, 24, 43.625),
    ("strategic", True): (0.31472406381545676, 230, 0.3217391304347826),
    ("priced", False): (0.492922273550581, 53, 0.0),
    ("priced", True): (0.294000307520963, 29, 33.03448275862069),
}


@pytest.mark.parametrize("key", list(BASELINE))
def test_default_args_reproduce_baseline(key):
    policy, pre = key
    r = run(policy, preempt=pre, **KW)
    mh, ns, rt = BASELINE[key]
    assert r.mean_health == pytest.approx(mh, rel=1e-12)
    assert r.n_services == ns
    assert r.retargets_per_service == pytest.approx(rt, rel=1e-12)


def test_determinism():
    a = run("honest", preempt=True, **KW)
    b = run("honest", preempt=True, **KW)
    assert a == b
    assert run("honest", preempt=True, **{**KW, "seed": 4}) != a


@pytest.mark.parametrize("p", POLICIES)
def test_all_policies_run_and_are_sane(p):
    r = run(p, steps=200, preempt=True, **KW)
    assert 0.0 <= r.min_health <= r.mean_health <= 1.0
    assert 0.0 <= r.frac_below_02 <= 1.0 and 0.0 <= r.frac_any_dead <= 1.0


def test_policies_validate():
    assert "oracle" not in POLICIES and "greedy_true" in POLICIES
    for bad in ("oracle", "bogus", ""):
        with pytest.raises(ValueError):
            run(bad)


def test_ring_distance_properties():
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(0, RING, (3, 500))
    d = ring_distance
    assert np.allclose(d(a, b), d(b, a))
    assert np.all(d(a, b) >= 0) and np.all(d(a, b) <= RING / 2 + 1e-12)
    assert np.allclose(d(a, a), 0)
    assert np.all(d(a, c) <= d(a, b) + d(b, c) + 1e-9)
    assert np.allclose(d(a, a + RING), 0)          # wraps
    assert d(0.0, RING / 2) == RING / 2 and d(1.0, RING - 1.0) == 2.0


@pytest.mark.parametrize("p", ["greedy_true", "nearest", "honest", "strategic", "priced"])
def test_hyst_zero_equivalent(p):
    assert run(p, preempt=True, hyst=0.0, **KW) == run(p, preempt=True, **KW)


def test_hysteresis_reduces_thrashing():
    base = run("honest", preempt=True, **KW)
    hy = run("honest", preempt=True, hyst=0.45, **KW)
    assert hy.retargets_per_service < base.retargets_per_service
    # without preemption hysteresis is irrelevant
    assert run("honest", preempt=False, hyst=0.5, **KW) == run("honest", preempt=False, **KW)


def test_sigma_default_and_zero_noise():
    from arbitration.model import SIGMA
    assert run("honest", preempt=True, sigma=SIGMA, **KW) == run("honest", preempt=True, **KW)
    # claims are the truth at sigma=0, so honest and greedy_true agree on precision
    # up to the call threshold (both precision-1 dispatchers)
    assert run("honest", preempt=True, sigma=0.0, **KW).precision > 0.95
    # policies that do not read the claims are flat in sigma
    for p in ("none", "greedy_true"):
        assert run(p, preempt=True, sigma=0.5, **KW) == run(p, preempt=True, sigma=0.0, **KW)
