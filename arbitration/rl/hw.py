"""REQ-HW-*: handwritten baseline set S_HW and its selection."""
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def all_strategies() -> List[Tuple[str, Dict[str, float]]]:
    """The 45 direction-1 strategies (scripts/run_quota.py: strategies())."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from scripts.run_quota import strategies
    return [(n, dict(d)) for n, d in strategies()]


def _excluded(name: str) -> bool:
    return name == "honest" or "oracle" in name or name.startswith("quota_")


# REQ-HW-06 (v1.3): the fine grid.  Four extra constant-offset strategies of the own u only (public), not in scripts/run_quota.py strategies().
FINE_GRID: List[Tuple[str, Dict[str, float]]] = [("dz=2.5", dict(dz=2.5)), ("dz=3.0", dict(dz=3.0)), ("dz=4.0", dict(dz=4.0)), ("b=1.0", dict(b=1.0))]
N_COARSE, N_FINE = 32, 36


def _group_rank(name: str) -> int:
    """Position of a name's group in the spec order of REQ-HW-01 (b, dz, dzcrit, burst, edge thr, edge dz, stealth)."""
    for i, pre in enumerate(("b=", "dz=", "dzcrit=", "burst", "edge_uninf thr", "edge_uninf dz", "stealth")):
        if name.startswith(pre):
            return i
    raise ValueError(f"strategy {name!r} belongs to no S_HW group")


def build_s_hw_coarse() -> List[Tuple[str, Dict[str, float]]]:
    """The legacy 32 (v1.2 list; the basis of Delta_coarse, REQ-S1-02): strategies() without H0, oracle, quota_*."""
    s = [(n, d) for n, d in all_strategies() if not _excluded(n)]
    if len(s) != N_COARSE:
        raise RuntimeError(f"REQ-HW-01: expected {N_COARSE} coarse strategies, got {len(s)}")
    return s


def build_s_hw() -> List[Tuple[str, Dict[str, float]]]:
    """REQ-HW-01/06 (v1.3): the 36 strategies = the coarse 32 plus the fine grid (dz=2.5, 3.0, 4.0, b=1.0), each placed at the end of its group
    (spec order; ties in the selection go to the earlier entry); no H0 (honest), no name containing 'oracle' or 'quota_'."""
    merged = build_s_hw_coarse() + [(n, dict(d)) for n, d in FINE_GRID]
    order = {n: i for i, (n, _) in enumerate(merged)}
    s = sorted(merged, key=lambda nd: (_group_rank(nd[0]), order[nd[0]]))
    if len(s) != N_FINE or len({n for n, _ in s}) != N_FINE:
        raise RuntimeError(f"REQ-HW-01: expected {N_FINE} strategies, got {len(s)}")
    return s


def build_s_hw_with_h0() -> List[Tuple[str, Dict[str, float]]]:
    """REQ-HW-02(a): the 36 plus H0 (honest, appended at the end so that ties are won by the 36) = 37, reported separately, not used in the verdict."""
    return build_s_hw() + [("honest", {})]


_INFO = {
    "b=": "constant bias on the own report u1 (own u only; includes the fine-grid b=1.0)",
    "dz=": "shift of the own z by a constant (own z/u only; includes the fine-grid dz=2.5, 3.0, 4.0)",
    "dzcrit=": "z shift scaled by the own public suspicion margin (own suspension state, public)",
    "burst": "periodic burst (W on / H-W off) on the agent's own round counter (own clock, public)",
    "edge_uninf thr": "report at the edge of the own acceptance threshold, uninformed variant (own u and public suspension only)",
    "edge_uninf dz": "z shift near the own edge, uninformed variant (own u and public suspension only)",
    "stealth": "shift reset by the own 'flagged' signal liar_flag (the flag is public: being flagged implies own suspension)",
}


def public_realizability_table() -> Dict[str, Dict[str, Any]]:
    """REQ-HW-04: for each S_HW name {'public': bool, 'info': str}."""
    tab: Dict[str, Dict[str, Any]] = {}
    for n, _ in build_s_hw():
        key = next(k for k in sorted(_INFO, key=len, reverse=True) if n.startswith(k))
        tab[n] = dict(public=True, info=_INFO[key])
    return tab


def _pick(val: Dict[str, Sequence[float]], test: Dict[str, Sequence[float]], names: Sequence[str]) -> Dict[str, Any]:
    best, best_m = None, -np.inf
    for n in names:
        m = float(np.mean(val[n]))
        if best is None or m > best_m:                    # strict: ties keep the earlier name
            best, best_m = n, m
    g = np.asarray(test[best], dtype=float)
    return dict(selected=best, G_hw_s=g, G_hw=float(np.mean(g)))


def select_handwritten(val_gains: Dict[str, Sequence[float]], test_gains: Dict[str, Sequence[float]],
                       names: Sequence[str]) -> Dict[str, Any]:
    """REQ-HW-02 (v1.3): pick argmax of mean validation G over `names` (the 36; ties -> earlier in `names`); report test per-seed G of
    the pick (may be negative, untruncated).  Returns {"selected","G_hw_s","G_hw", "plus_h0": {"selected","G_hw_s","G_hw"}}
    where plus_h0 re-selects over names + ['honest' with zero gains] (37).  Anything but 36 candidates is refused (ValueError)."""
    names = list(names)
    if len(names) != N_FINE or len(set(names)) != N_FINE:
        raise ValueError(f"REQ-HW-02: the candidate set must be the {N_FINE} distinct S_HW strategies, got {len(names)}")
    res = _pick(val_gains, test_gains, names)
    some = names[0]
    val0 = dict(val_gains, honest=np.zeros(len(val_gains[some])))
    test0 = dict(test_gains, honest=np.zeros(len(test_gains[some])))
    res["plus_h0"] = _pick(val0, test0, names + ["honest"])
    return res
