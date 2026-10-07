"""REQ-M4-*: stepwise inequality appendix (CPU, from JSON).

Sufficient condition (REQ-M4-01):   eps * rw * (L * u_bar + w)  >=  g_max
  eps  per-round raid probability;  rw  probability that the raid target is the round's winner;  L  pause length (pauses do not
  stack: a suspended agent is ineligible and cannot be raided again);  u_bar  expected per-round utility lost while suspended
  (endogenous: measured by simulation, honest agent 1 ~0.184);  w  value wasted when the raided winner's item is discarded;
  g_max  largest one-round gain of a deviation (<= 1 for u in [0,1]).
Holding implies the per-deviation expected gain is <= 0 (so G <= 0); failing does NOT imply G > 0.  rw = 0 => LHS = 0 (fails).
Near the end (last L rounds) the pause cost is truncated to min(L, T-t); direction 2 does not force honesty there (ENV-20).
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence

_ROWS_NOTE = "G for consist_M4 rows = max of gain_b03, gain_dz1, gain_edgeO (edgeO is an oracle strategy)"


def lhs(eps: float, rw: float, L: float, u_bar: float, w: float) -> float:
    """REQ-M4-01: eps*rw*(L*u_bar + w)."""
    return eps * rw * (L * u_bar + w)


def predicts_nonpositive(eps: float, rw: float, L: float, u_bar: float, w: float, g_max: float = 1.0) -> bool:
    """REQ-M4-01: sufficient condition lhs >= g_max."""
    return bool(lhs(eps, rw, L, u_bar, w) >= g_max)


def critical_L(eps: float, rw: float, u_bar: float, w: float, g_max: float = 1.0) -> float:
    """Smallest pause length for which the inequality holds: (g_max/(eps*rw) - w)/u_bar (inf if eps*rw == 0 or u_bar == 0)."""
    if eps * rw <= 0 or u_bar <= 0:
        return float("inf")
    return max(0.0, (g_max / (eps * rw) - w) / u_bar)


def failure_region(u_bar: float, w: Optional[float] = None, g_max: float = 1.0,
                   eps_grid: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2), rw_grid: Sequence[float] = (0.0, 1 / 3, 1.0),
                   L_grid: Sequence[float] = (20.0, 250.0, 1250.0, 2500.0)) -> List[Dict[str, Any]]:
    """REQ-M4-03: the (eps, rw, L) grid cells in which the inequality FAILS, with the critical pause length of each (eps, rw)."""
    w = u_bar if w is None else w
    out = []
    for eps in eps_grid:
        for rw in rw_grid:
            lc = critical_L(eps, rw, u_bar, w, g_max)
            for L in L_grid:
                if not predicts_nonpositive(eps, rw, L, u_bar, w, g_max):
                    out.append(dict(eps=eps, rw=rw, L=L, lhs=lhs(eps, rw, L, u_bar, w), L_critical=lc))
    return out


def _load(root: str, name: str) -> Any:
    with open(os.path.join(root, "results", name)) as f:
        return json.load(f)


def _row(source: str, index: int, eps: float, rw: float, L: float, G: float, half: float, u_bar: float, w: float, g_max: float, **extra) -> Dict[str, Any]:
    holds = predicts_nonpositive(eps, rw, L, u_bar, w, g_max)
    return dict(source=source, index=index, eps=eps, rw=rw, L=L, G=G, half=half, lhs=lhs(eps, rw, L, u_bar, w),
                predicted_holds=holds, compatible=bool((not holds) or (G - half <= 0)), **extra)


def compare_with_json(root: str = ".", u_bar: float = None, w: float = None, g_max: float = 1.0) -> List[Dict[str, Any]]:
    """REQ-M4-02: rows {'source': 'consist_M4'|'quota_main'|'eval_rows_M4', 'index': i, 'eps','rw','L','G','half','predicted_holds',
    'compatible'} for stage2b consist_M4[0..17] (G=max of gain_b03/gain_dz1/gain_edgeO), quota main[88..91] (G=Gmax) and
    stage2b eval_rows.M4 (G=G_max); compatible = (not predicted_holds) or (G - half <= 0)."""
    s2b, qm = _load(root, "stage2b.json"), _load(root, "quota.json")["main"]
    if u_bar is None:
        u_bar = qm[88]["honest_util_agent1"][0]
    w = u_bar if w is None else w
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(s2b["consist_M4"]):
        cols = {k: (c[k][0], c[k][1]) for k in ("gain_b03", "gain_dz1", "gain_edgeO")}
        G, half = max(cols.values())
        rows.append(_row("consist_M4", i, c["eps"], c["rw"], c["L"], G, half, u_bar, w, g_max, r=c["r"], gains=cols))
    for i in range(88, 92):
        q = qm[i]
        p = q["param"]
        rows.append(_row("quota_main", i, p["eps"], p["rw"], p["L"], q["Gmax"][0], q["Gmax"][1], u_bar, w, g_max, r=q["r"]))
    for i, e in enumerate(s2b["eval_rows"]["M4"]):
        p = e["param"]
        rows.append(_row("eval_rows_M4", i, p["eps"], p["rw"], p["L"], e["G_max"][0], e["G_max"][1], u_bar, w, g_max, r=e["r"]))
    return rows


def degeneracy_report(root: str = ".") -> Dict[str, Any]:
    """REQ-M4-04: {'gmax': {88..91: [mean,half]}, 'quota_gains_all_zero': bool, 'gmax_nonquota': {...}, 'conclusion': str}."""
    qm = _load(root, "quota.json")["main"]
    idx = range(88, 92)
    gmax = {i: list(qm[i]["Gmax"]) for i in idx}
    nonq = {i: list(qm[i]["Gmax_nonquota"]) for i in idx}
    allzero = all(v == 0.0 for i in idx for k, v in qm[i]["gains_by_strategy"].items() if k.startswith("quota_"))
    vals = [nonq[i][0] for i in idx]
    concl = ("0.0000 是「最大值取自含空操作策略的集合」造成的下界（quota_* 策略在 M4 下是空操作，gain 恆為精確的 0.0）；"
             f"實際偏離策略族（排除 quota_*）中的最大值為負（約 {min(vals):.3f} 至 {max(vals):.3f}）。"
             "這不是「策略族整體退化成誠實」（其他策略的 gain 為負而非 0），但「G=0.0000」不能解讀為「測得 0」。")
    return dict(gmax=gmax, quota_gains_all_zero=bool(allzero), gmax_nonquota=nonq, conclusion=concl)


def boundary_points(u_bar: float = 0.1837185916322254, g_max: float = 1.0) -> List[Dict[str, float]]:
    """Near-boundary (eps, rw, L): for three (eps, rw) pairs, L at 0.5x and 1.5x of the critical pause length."""
    pts = []
    for eps, rw in ((0.02, 1 / 3), (0.05, 1 / 3), (0.01, 1.0)):
        lc = critical_L(eps, rw, u_bar, u_bar, g_max)
        for f in (0.5, 1.5):
            pts.append(dict(eps=eps, rw=rw, L=float(round(f * lc))))
    return pts


def check_break_region(dev: str = "cuda") -> List[Dict[str, Any]]:
    """REQ-M4-03: >=3 near-boundary (eps,rw,L) points simulated: rows {'eps','rw','L','predicted_holds','G','ci','sign_compatible'}.
    G = largest mean paired gain over a small non-oracle + oracle set (b=0.3, dz=1.0, edge oracle em=0.05) on training seeds 0..31,
    T=50000 (scored to T-L), r=0.5; the prediction concerns the sign only: holds => G - half <= 0."""
    from . import env
    from .metrics import mean_ci
    from .policy import AdapterPolicy
    u_bar = 0.1837185916322254
    strat = [{}, dict(b=0.3), dict(dz=1.0), dict(em=0.05)]
    pols = [AdapterPolicy(d) for d in strat]
    seeds_ = list(range(32))
    T = 50000
    rows = []
    for pt in boundary_points(u_bar):
        out = env.simulate("M4", T, dict(eps=pt["eps"], L=pt["L"], rw=pt["rw"]), pols, seeds_, r=0.5, dev=dev)
        ub = out["snap"]["util_b"][:, :, 1].detach().cpu().numpy() / out["T_score"]
        best = max((mean_ci(ub[j] - ub[0]) for j in range(1, len(pols))), key=lambda c: c["mean"])
        holds = predicts_nonpositive(pt["eps"], pt["rw"], pt["L"], u_bar, u_bar)
        rows.append(dict(pt, predicted_holds=holds, lhs=lhs(pt["eps"], pt["rw"], pt["L"], u_bar, u_bar), G=best["mean"],
                         ci=[best["lo"], best["hi"]], sign_compatible=bool((not holds) or best["mean"] - best["half"] <= 0)))
    return rows


def sensitivity_table(u_bars: Sequence[float] = (0.15, 0.1837185916322254, 0.22), g_max: float = 1.0,
                      pairs: Sequence[Sequence[float]] = ((0.02, 1 / 3), (0.05, 1 / 3), (0.01, 1.0), (0.2, 1.0))) -> List[Dict[str, Any]]:
    """REQ-M4-01 (u_bar is endogenous): critical pause length L_c(eps, rw) for several u_bar values (w = u_bar)."""
    return [dict(eps=e, rw=rw, u_bar=ub, L_critical=critical_L(e, rw, ub, ub, g_max)) for ub in u_bars for e, rw in pairs]
