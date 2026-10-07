"""Shared helpers for the direction-2 acceptance tests (spec s14).  Direction-1 numbers are always read from results/*.json."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SPEC_T_ID = "T"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA; run only through `gpujob` (REQ-GPU-01/07); must FAIL (not skip) without CUDA")
    config.addinivalue_line("markers", "cpu: pure-CPU test (default for unmarked tests)")


def pytest_collection_modifyitems(config, items):
    for it in items:
        if it.get_closest_marker("gpu") is None:
            it.add_marker(pytest.mark.cpu)


def cuda_required():
    """REQ-GPU-07: a [gpu] test without CUDA FAILS (assertion), never skips."""
    import torch
    assert torch.cuda.is_available(), "REQ-GPU-07: [gpu] test requires CUDA (must fail, not skip, without it)"


def load_json(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def rel_close(a, b, tol=1e-9):
    """REQ-ENV-07: |a-b| <= tol*|b|; if b == 0.0 then a must be exactly 0.0."""
    a, b = float(a), float(b)
    if b == 0.0:
        return a == 0.0
    return abs(a - b) <= tol * abs(b)


def ci196(x):
    """Direction-1 CI formula: [mean, 1.96*std(ddof=1)/sqrt(n)]."""
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]


def std_samples(mean, sd, n=32):
    """Deterministic sample with exact mean and exact SD(ddof=1) (symmetric normal quantiles)."""
    from scipy.stats import norm
    z = norm.ppf((np.arange(n) + 0.5) / n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


def phi1(x):
    """Agent-1 marginal cdf (identity) clipped as in audit_fix.phi_of; independent copy for T-07/T-17."""
    import torch
    return x.clamp(1e-12, 1 - 1e-12)


STRATS_T01 = [
    ("honest", {}), ("b=0.1", dict(b=0.1)), ("dz=0.03", dict(dz=0.03)), ("dz=0.3", dict(dz=0.3)),
    ("dzcrit=1.0", dict(dzc=1.0)), ("burst dz1 W10 H90", dict(dz=1.0, W=10, H=90)),
    ("edge_oracle em=0.05", dict(em=0.05)), ("edge_uninf thr band=0.1", dict(ue_mode=1, ue_band=0.1)),
    ("edge_uninf dz band=0.1 dz=0.3", dict(ue_mode=2, ue_band=0.1, ue_dz=0.3)),
    ("stealth m=1.0 f=0.9", dict(ad_m=1.0, ad_f=0.9)),
]


@pytest.fixture
def rl():
    """The implementation package (skeleton until REQ implemented)."""
    import arbitration.rl as pkg
    return pkg


STRATS_T01_EXTRA = [
    ("edge_oracle legacy em=0.05 leg=1", dict(em=0.05, leg=1.0)),
    ("b=0.3 timing", dict(b=0.3, timing=1.0)),
    ("dz=0.1 + dzcrit=0.5", dict(dz=0.1, dzc=0.5)),
    ("edge_uninf thr band=0.1 q=0.3", dict(ue_mode=1, ue_band=0.1, ue_q=0.3)),
    ("edge_uninf dz band=0.2 q=0.7 dz=0.5", dict(ue_mode=2, ue_band=0.2, ue_q=0.7, ue_dz=0.5)),
    ("stealth m=1.0 f=0.9 c=0.5", dict(ad_m=1.0, ad_f=0.9, ad_c=0.5)),
]
# T-01/T-02 use ALL frontier SCOLS columns (leg, timing, dzc, ue_q, ad_c ... included); T-03 keeps the original 10.
STRATS_T01_FULL = STRATS_T01 + STRATS_T01_EXTRA


def parse_range(s):
    """'5000-5031' -> list(range(5000, 5032))."""
    a, b = str(s).split("-")
    return list(range(int(a), int(b) + 1))


def strategy_dict(name):
    """Direction-1 strategy dict by name (scripts/run_quota.py: strategies())."""
    from scripts.run_quota import strategies
    return dict(strategies())[name]


def frontier_ref(mech, T, cfg, r, strat_dicts, seeds, dev):
    """frontier.simulate reference reshaped to [n_strategies, n_seeds, ...].  Results are cached (key = full arguments), so a test can
    compute every expectation BEFORE it disables frontier.simulate with forbid_frontier() and still read them afterwards."""
    key = (mech, int(T), tuple(sorted(dict(cfg).items())), float(r), tuple(tuple(sorted(dict(d).items())) for d in strat_dicts), tuple(seeds), dev)
    if key in _FRONTIER_CACHE:
        return _FRONTIER_CACHE[key]
    from arbitration.gpu import frontier
    el = frontier.build([dict(cfg)], [dict(rho=r)], [dict(d) for d in strat_dicts], list(seeds), dev=dev)
    o = frontier.simulate(mech, T, el, list(seeds))
    res = {k: v.reshape(len(strat_dicts), len(seeds), *v.shape[1:]) for k, v in o.items()}
    _FRONTIER_CACHE[key] = res
    return res


_FRONTIER_CACHE = {}


def forbid_frontier(monkeypatch):
    """REQ-ENV-04 (independent implementation): after this call any attempt of the new simulator to run frontier.simulate (or
    quota.simulate) raises AssertionError.  The module attribute is replaced, and so is every alias already bound inside any
    `arbitration.rl*` module (`from ...frontier import simulate [as X]`).  Compute all expectations (frontier_ref) BEFORE calling."""
    import sys
    from arbitration.gpu import frontier, quota
    originals = {id(frontier.simulate), id(quota.simulate)}

    def boom(*a, **k):
        raise AssertionError("the new simulator must not call frontier.simulate / quota.simulate (REQ-ENV-04: independent implementation)")
    monkeypatch.setattr(frontier, "simulate", boom)
    monkeypatch.setattr(quota, "simulate", boom)
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("arbitration.rl"):
            continue
        for attr, val in list(vars(mod).items()):
            if callable(val) and id(val) in originals:
                monkeypatch.setattr(mod, attr, boom)


def need(mod, name, req):
    """New API defined by the tests (not yet in the skeleton): a missing name is a NotImplementedError (red), not an AttributeError."""
    fn = getattr(mod, name, None)
    if fn is None:
        raise NotImplementedError(f"{req}: {getattr(mod, '__name__', mod)}.{name} is not defined yet (interface fixed by the tests, see docs/direction2-tdd-map.md)")
    return fn


def call_new_kw(fn, new_kwargs, *args, **kwargs):
    """Call fn(*args, **kwargs, **new_kwargs); a skeleton that does not know the new keyword yet gives NotImplementedError, not TypeError."""
    try:
        return fn(*args, **kwargs, **new_kwargs)
    except TypeError as e:
        if "unexpected keyword argument" in str(e) and any(k in str(e) for k in new_kwargs):
            raise NotImplementedError(f"interface extension {sorted(new_kwargs)} is not implemented yet: {e}")
        raise


def assert_fields_equal(a, b, fields, what=""):
    """Bitwise (==) equality of simulator output fields a[f] vs b[f] (same shape [S, n, ...])."""
    import torch
    for f in fields:
        x, y = a[f].to("cpu"), b[f].to("cpu")
        assert x.shape == y.shape, f"{what}{f}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        assert torch.equal(x, y), f"{what}{f}: not bitwise equal (max |diff| {(x - y).abs().max().item():.3e})"


FIXED_CATEGORIES = {"D1": "out_of_family", "D2": "out_of_family", "O1": "out_of_family", "O2": "in_family", "O3": "in_family", "D4": "in_family",
                    "D3": "in_family_known", "O5": "blind", "NAIVE": "anchor", "M4rw0": "anchor", "designer_1": "contaminated_excluded"}
NS_DISCLOSURE = "n_S=256 covers the median worst case; the single-pair worst case (sd_CRN=0.034465) needs n_S~565 and is NOT covered (REQ-OPT-13)"


def make_freeze(**over):
    """A complete v1.3 freeze registration (REQ-S1-14); keyword arguments replace single items (None is kept as None)."""
    from arbitration.rl import hw
    d = dict(cold_start=dict(theta0="zeros"), warm_start=dict(recipe="lstsq"), pilot=dict(cold_val_G=0.1), chosen_start="cold",
             lam=256, n_S=256, T_train=20000, G_gens=416, sigma0=0.3, MDE_D_plan=0.0058, G_s=[0.01, 0.02], registry_sha256="0" * 64,
             nS_coverage_disclosure=NS_DISCLOSURE, S_HW=[n for n, _ in hw.build_s_hw()],
             calibration=dict(calib_seeds="val+val2", n_seeds=64, ci="t63", records=[]), categories=dict(FIXED_CATEGORIES))
    d.update(over)
    return d
