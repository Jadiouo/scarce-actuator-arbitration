"""REQ-OPT-08..11 (parameter decision rules) and REQ-GPU-08 (job segmentation)."""
from typing import Any, Callable, Dict, List, Sequence, Tuple


import math

_LAMS = (256, 128, 64, 32)
_TS = (100000, 40000, 20000)
_NS = (16, 32, 64, 128, 256)


def mde_est_defaults() -> Dict[float, float]:
    """REQ-OPT-09/11: {0.5: 0.0058, 0.9: 0.0016}."""
    return {0.5: 0.0058, 0.9: 0.0016}


def decide_params(t_step_ms: Callable[[int], float], B_max: int, W_run_s: float,
                  sd_crn: Dict[Tuple[float, int], float], mde_est: Dict[float, float] = None) -> Dict[str, Any]:
    """REQ-OPT-08..11.  lam in {32,64,128,256}, T_train in {20000,40000,100000}, n_S in {16,32,64,128,256}.
    n_S(T) = smallest n_S with sd_crn[(0.5,T)]/sqrt(n_S) <= 0.25*mde_est[0.5]; then iterate lam desc, T desc and pick the
    first with lam*n_S<=B_max and 300*T*t_step_ms(lam*n_S)/1000 <= W_run_s.  G_gens = min(1000, floor(W_run/(T*t_step))).
    Returns {"feasible": bool, "lam","T_train","n_S","G_gens","r09_underpowered": bool, "mde_est": dict, "reason": str}."""
    mde = dict(mde_est) if mde_est is not None else mde_est_defaults()
    out = dict(feasible=False, lam=None, T_train=None, n_S=None, G_gens=None, r09_underpowered=False, mde_est=mde, reason="")
    for lam in _LAMS:
        for T in _TS:
            sd = sd_crn[(0.5, T)]
            nS = next((n for n in _NS if sd / math.sqrt(n) <= 0.25 * mde[0.5]), None)
            if nS is None:
                out["reason"] = f"REQ-OPT-09: n_S=256 insufficient for r=0.5 at T_train={T} (sd_crn={sd:.4g})"
                return out
            B = lam * nS
            if B <= B_max and 300 * T * t_step_ms(B) / 1000 <= W_run_s:
                g = min(1000, int(W_run_s // (T * t_step_ms(B) / 1000)))
                under = sd_crn[(0.9, T)] / math.sqrt(nS) > 0.25 * mde[0.9]
                out.update(feasible=True, lam=lam, T_train=T, n_S=nS, G_gens=g, r09_underpowered=bool(under),
                           reason="ok" + ("; r=0.9 underpowered (descriptive only, REQ-OPT-11)" if under else ""))
                return out
    out["reason"] = "REQ-OPT-08: no (lam, T_train) satisfies B<=B_max and 300 generations within W_run"
    return out


def _n_val_evals(g0: int, g1: int, val_every: int) -> int:
    """Validation evaluations inside generations (g0, g1]: after every `val_every` generations."""
    return g1 // val_every - g0 // val_every


def segment_estimate_s(g0: int, g1: int, t_step_ms: Callable[[int], float], T_train: int, B: int, n_val: int,
                       t_val_s: float = 0.0, t_startup_s: float = 0.0, val_every: int = 50) -> float:
    """REQ-GPU-08: (g1-g0)*T_train*t_step_ms(B)/1000 + (#validation evals in the segment)*t_val_s + t_startup_s."""
    return (g1 - g0) * T_train * t_step_ms(B) / 1000 + _n_val_evals(g0, g1, val_every) * t_val_s + t_startup_s


def _segment_from(start: int, t_step_ms, T_train, G_gens, B, n_val, t_val_s, t_startup_s, max_s, val_every) -> List[Tuple[int, int]]:
    plan, g0 = [], start
    while g0 < G_gens:
        est = lambda g1: segment_estimate_s(g0, g1, t_step_ms, T_train, B, n_val, t_val_s, t_startup_s, val_every)
        if est(g0 + 1) > max_s:
            raise ValueError(f"REQ-GPU-08: a single generation ({est(g0 + 1):.0f} s) exceeds {max_s:.0f} s: infeasible")
        g1 = g0 + 1
        while g1 < G_gens and est(g1 + 1) <= max_s:
            g1 += 1
        plan.append((g0, g1))
        g0 = g1
    return plan


def segment(t_step_ms: Callable[[int], float], T_train: int, G_gens: int, B: int, n_val: int, *, t_val_s: float = 0.0,
            t_startup_s: float = 0.0, max_s: float = 1500.0, val_every: int = 50) -> List[Tuple[int, int]]:
    """REQ-GPU-08: contiguous non-overlapping generation intervals [g0,g1) covering 0..G_gens, each estimate <= max_s;
    ValueError if a single generation alone exceeds max_s."""
    return _segment_from(0, t_step_ms, T_train, G_gens, B, n_val, t_val_s, t_startup_s, max_s, val_every)


def update_after_part(part: Dict[str, Any], plan: List[Tuple[int, int]], t_step_ms: Callable[[int], float], T_train: int,
                      G_gens: int, B: int, n_val: int, **kw) -> Dict[str, Any]:
    """REQ-GPU-08: mark part['overrun']=True if wall_s>1500; re-estimate with the measured t_step and re-segment the
    remaining generations.  Returns {"part": part, "plan": new_plan}."""
    max_s = kw.get("max_s", 1500.0)
    t_val_s, t_startup_s, val_every = kw.get("t_val_s", 0.0), kw.get("t_startup_s", 0.0), kw.get("val_every", 50)
    part = dict(part)
    g0, g1 = int(part["gen0"]), int(part["gen1"])
    over = float(part["wall_s"]) > 1500.0
    if over:
        part["overrun"] = True
        measured_ms = float(part["wall_s"]) * 1000.0 / ((g1 - g0) * T_train)       # wall-clock per round, all overheads included
        step = lambda _B, m=measured_ms: m
        new = _segment_from(g1, step, T_train, G_gens, B, n_val, 0.0, 0.0, max_s, val_every)
    else:
        new = _segment_from(g1, t_step_ms, T_train, G_gens, B, n_val, t_val_s, t_startup_s, max_s, val_every)
    return dict(part=part, plan=new)


def chunk_jobs(items: Sequence[Tuple[str, float]], max_s: float = 1500.0) -> List[List[str]]:
    """REQ-GPU-08: pack (label, est_seconds) items in order into chunks with total <= max_s (ValueError if one item > max_s)."""
    chunks: List[List[str]] = []
    cur: List[str] = []
    tot = 0.0
    for label, est in items:
        if est > max_s:
            raise ValueError(f"REQ-GPU-08: item {label!r} ({est:.0f} s) exceeds {max_s:.0f} s")
        if cur and tot + est > max_s:
            chunks.append(cur)
            cur, tot = [], 0.0
        cur.append(label)
        tot += est
    if cur:
        chunks.append(cur)
    return chunks
