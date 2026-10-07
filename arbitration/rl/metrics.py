"""REQ-MET-*: paired gains, CIs, MDE, found rules, replication, R/T and alpha."""
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

_VAL_LO, _VAL_HI = 2000, 2031          # validation seeds (the only seeds an MDE may be computed from)
_N_SEEDS = 32


def _finite_array(x: Any, what: str) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{what} contains NaN or inf")
    return a


def paired_gain(U_pol: Any, U_hon: Any, T_score: int) -> Any:
    """REQ-MET-01/02: G_s = (U_pol(s)-U_hon(s))/T_score per seed, no clipping.
    Hardened: equal shapes required (no silent broadcasting), NaN/inf rejected.  To pass a simulator output instead of raw
    utilities use paired_gain_from_out (it also checks seed order and takes the T-L snapshot itself)."""
    if isinstance(U_pol, dict):
        raise TypeError("pass raw utilities here; for an env.simulate output use paired_gain_from_out")
    a, h = _finite_array(U_pol, "U_pol"), _finite_array(U_hon, "U_hon")
    if a.shape != h.shape:
        raise ValueError(f"U_pol {a.shape} and U_hon {h.shape} must have identical shape (no broadcasting)")
    if not T_score or T_score <= 0:
        raise ValueError("T_score must be positive")
    return (a - h) / T_score


def paired_gain_from_out(out: Dict[str, Any], pol: int, hon: int = 0, *, seeds: Optional[Sequence[int]] = None,
                         hon_out: Optional[Dict[str, Any]] = None, seeds_hon: Optional[Sequence[int]] = None) -> np.ndarray:
    """REQ-MET-01/02: G_s of policy row `pol` against honest row `hon` (default: the same env.simulate output `out`), per seed, from the
    snapshot util_b[:, :, 1] at out['T_score'] (= T - L; the last L rounds are excluded) -- the caller never passes 'final'.
    If the honest rows come from another simulate output (hon_out), the seed lists of both runs must be given and identical in length and order."""
    for o in (out, hon_out):
        if o is not None and ("snap" not in o or "T_score" not in o):
            raise ValueError("expected an env.simulate output with 'snap' and 'T_score'")
    h_out = out if hon_out is None else hon_out
    if hon_out is not None:
        if seeds is None or seeds_hon is None or list(seeds) != list(seeds_hon):
            raise ValueError("policy and honest runs must use the same seeds in the same order (pass seeds and seeds_hon)")
        if h_out["T_score"] != out["T_score"]:
            raise ValueError("policy and honest runs have different T_score")
    U = _finite_array(out["snap"]["util_b"], "util_b")[:, :, 1]
    Uh = _finite_array(h_out["snap"]["util_b"], "util_b")[:, :, 1]
    if seeds is not None and len(seeds) != U.shape[1]:
        raise ValueError("seeds length does not match the simulate output")
    if U.shape[1] != Uh.shape[1]:
        raise ValueError("policy and honest outputs have different numbers of seeds")
    return paired_gain(U[pol], Uh[hon], out["T_score"])


def _tcrit(n: int) -> float:
    """t_{0.975, n-1}; for the 32 paired seeds this is t_{0.975,31} = 2.0395."""
    return float(stats.t.ppf(0.975, n - 1))


def mean_ci(x: Sequence[float], crit: str = "t31") -> Dict[str, float]:
    """REQ-MET-03: {'mean','lo','hi','half'}; crit 't31' -> t_{0.975,31}=2.0395 with SD(ddof=1)/sqrt(n); 'z196' -> 1.96."""
    a = np.asarray(x, dtype=float).reshape(-1)
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 paired values for a CI")
    if crit == "t31":
        c = _tcrit(n)
    elif crit == "z196":
        c = 1.96
    else:
        raise ValueError(f"unknown crit {crit!r}")
    m = float(a.mean())
    half = float(c * a.std(ddof=1) / math.sqrt(n))
    return {"mean": m, "lo": m - half, "hi": m + half, "half": half}


def difference_ci(G_rl_s: Sequence[float], G_hw_s: Sequence[float]) -> Dict[str, float]:
    """REQ-MET-04: CI of D_s = G_rl_s - G_hw_s (paired, t31)."""
    a, b = np.asarray(G_rl_s, dtype=float), np.asarray(G_hw_s, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same shape (same seeds)")
    return mean_ci(a - b, "t31")


def kappa() -> float:
    """REQ-MET-06: non-central t parameter with P(|T_31(delta)|>2.0395)=0.80 (=2.8922)."""
    tc = stats.t.ppf(0.975, 31)

    def power(d: float) -> float:
        return float(stats.nct.sf(tc, 31, d) + stats.nct.cdf(-tc, 31, d))

    return float(optimize.brentq(lambda d: power(d) - 0.80, 1.0, 6.0, xtol=1e-12))


def mde(paired_vals: Sequence[float], seeds: Sequence[int]) -> float:
    """REQ-MET-06: kappa*SD(ddof=1)/sqrt(32); seeds must be validation seeds (ValueError otherwise)."""
    sd = [int(s) for s in seeds]
    if not sd or any(not (_VAL_LO <= s <= _VAL_HI) for s in sd):
        raise ValueError("REQ-MET-06: the MDE may only be computed from validation seeds")
    a = np.asarray(paired_vals, dtype=float).reshape(-1)
    if a.size != len(sd):
        raise ValueError("paired_vals and seeds must have the same length")
    return float(kappa() * a.std(ddof=1) / math.sqrt(_N_SEEDS))


def found_rule(value: float, lo: float, mde: float) -> bool:
    """REQ-MET-07: lower CI bound > 0 and value >= mde."""
    return bool(lo > 0 and value >= mde)


def final_eval(record: Dict[str, Any], G_s: Sequence[float], D_s: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """REQ-MET-06: test evaluation using record['mde_frozen'] (KeyError if absent; never recomputed).
    Returns {'mde_used', 'G': ci dict, 'found_G': bool, 'D': ci dict|None, 'found_D': bool|None}.
    The D rule uses record['mde_d_frozen'] when the record carries it (separate MDE_D), else the same frozen value."""
    mde_used = record["mde_frozen"]
    g = mean_ci(G_s, "t31")
    out: Dict[str, Any] = {"mde_used": mde_used, "G": g, "found_G": found_rule(g["mean"], g["lo"], mde_used), "D": None, "found_D": None}
    if D_s is not None:
        d = mean_ci(D_s, "t31")
        mde_d = record.get("mde_d_frozen", mde_used)
        out["D"], out["found_D"], out["mde_d_used"] = d, found_rule(d["mean"], d["lo"], mde_d), mde_d
    return out


def replication_count(D_s_runs: Sequence[Sequence[float]]) -> int:
    """REQ-MET-08: number of runs whose D (t31 CI) has lo > 0."""
    return int(sum(mean_ci(r, "t31")["lo"] > 0 for r in D_s_runs))


def replication_label(k: int) -> str:
    """REQ-MET-08: 3 -> '完全複現', 2 -> '部分複現', 1 -> '未複現' (0 -> '無')."""
    return {3: "完全複現", 2: "部分複現", 1: "未複現"}.get(int(k), "無")


def cp_lower(x: int, n: int) -> float:
    """REQ-S1-16: one-sided 95% Clopper-Pearson lower bound Beta^-1(0.05; x, n-x+1); 0 when x=0."""
    if x < 0 or x > n or n < 1:
        raise ValueError("need 0 <= x <= n, n >= 1")
    if x == 0:
        return 0.0
    return float(stats.beta.ppf(0.05, x, n - x + 1))


def rt_alpha(reg_b: Any, pen: Any, T_score: int) -> Tuple[Any, Any]:
    """REQ-MET-05: (R/T per seed = reg_b/T_score, alpha per seed = pen.mean(agents)/T_score)."""
    return reg_b / T_score, pen.mean(-1) / T_score


def agent_susp_fraction(pen: Any, T_score: int, agents: Sequence[int] = (0, 2)) -> Any:
    """REQ-MET-05: pen[:, agents]/T_score per seed and agent."""
    return pen[..., list(agents)] / T_score
