"""REQ-S1-01..07: vulnerability suite operators, calibration, registry.

Design
------
A vulnerability is a *mechanism mutation*: exactly one (or two) of five hooks of the M3C audit block is replaced; every other hook keeps
the base mechanism's code path (REQ-S1-05/06).  The five hooks and their base behaviour (agent 1 = the deviator):

    audit_prob(win_streak, p)             base: p                      (called every round)
    resid(resid)                          base: resid                  (audit settlements of agent 1)
    cusum(S, resid, kref)                 base: max(S - resid - kref, 0)   (settlements of agent 1 while eligible)
    cusum_reset(S, rounds_since_win)      base: S                      (called every round, after the settlement)
    pause_len(L, clean_streak)            base: L                      (flags of agent 1)

Built-in operators O1..O6 replace one hook each (knob 0 / C=inf / g=inf = identity, "knob off").  `register_operator` is the insertion
interface for the blind designers' mutations: a mutation is a `VulnOperator` whose `hooks` map hook name -> fn(inputs, knob) -> tensor;
the framework applies it to agent 1 only and keeps all other hooks on the base path, so a designer cannot leak into another mechanism.
This module never refers to test seeds (REQ-SEED-03).
"""
import json
import math
import os
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import env as _env
from . import seeds as _seeds
from .metrics import mean_ci
from .policy import AdapterPolicy, MLPPolicy

F64 = torch.float64
BLK = _env.BLK
K = _env.K
HOOKS = ("audit_prob", "resid", "cusum", "cusum_reset", "pause_len")
M_WIN, M_CLEAN, G_RESET = 3, 50, 20            # O1 streak, O4 clean-audit streak, default O5 horizon (spec 8.2)
VAL_SEEDS = tuple(range(2000, 2032))
VAL2_SEEDS = tuple(_seeds.splits()["val2"])                 # REQ-SEED-08: calibration/selection only
CALIB_SEEDS = VAL_SEEDS + VAL2_SEEDS                         # REQ-S1-28: knob calibration and Delta on val U val2 (64 seeds)

# REQ-S1-01/26 (v1.3): the fixed `category` of every suite member (never changed by a recalibration)
CATEGORIES = ("out_of_family", "in_family", "in_family_known", "blind", "anchor", "contaminated_excluded")
CATEGORY_OF: Dict[str, str] = {
    "D1": "out_of_family", "D2": "out_of_family", "O1": "out_of_family",
    "O2": "in_family", "O3": "in_family", "D4": "in_family",
    "D3": "in_family_known",
    "O5": "blind",
    "NAIVE": "anchor", "M4rw0": "anchor",
    "designer_1": "contaminated_excluded",
}


# ------------------------------------------------------------------------------------------------ scalar operator formulas
def auto_operator_ids() -> List[str]:
    """REQ-S1-06 (v1.3): the automatic operators IN THE SUITE: ['O1','O2','O3'] (O4 and O6 keep their definitions below but are not in the suite:
    they cannot be calibrated to the smallest registered size 0.01)."""
    return ["O1", "O2", "O3"]


def blind_operator_ids() -> List[str]:
    """REQ-S1-07 (v1.3): ['O5'] (O4 left the suite)."""
    return ["O5"]


def o1_audit_prob(p: float, win_streak: int, kappa: float, m: int = 3) -> float:
    """REQ-S1-06 O1: p*(1-kappa) once the agent has won m consecutive rounds, else p."""
    return p * (1 - kappa) if win_streak >= m else p


def o2_cusum_update(S: float, resid: float, kref: float, kappa: float) -> float:
    """REQ-S1-06 O2: (1-kappa)*max(S-resid-kref, 0)."""
    return (1 - kappa) * max(S - resid - kref, 0.0)


def o3_resid_cap(resid: float, C: float) -> float:
    """REQ-S1-06 O3: max(resid, -C)."""
    return max(resid, -C)


def o4_pause_length(L: float, clean_audit_streak: int, kappa: float, m: int = 50) -> float:
    """REQ-S1-07 O4: (1-kappa)*L once the agent had m consecutive audits without a flag, else L."""
    return (1 - kappa) * L if clean_audit_streak >= m else L


def o5_cusum_reset(S: float, rounds_since_win: int, g: int = 20) -> float:
    """REQ-S1-07 O5: 0 if the agent has not won for g consecutive rounds, else S."""
    return 0.0 if rounds_since_win >= g else S


def o6_pause_length(L: float, kappa: float, flagged: bool) -> float:
    """REQ-S1-06 O6: (1-kappa)*L when flagged, else L."""
    return (1 - kappa) * L if flagged else L


# ------------------------------------------------------------------------------------------------ tensor hooks
def _base_hooks() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    return {
        "audit_prob": lambda i: i["p"],
        "resid": lambda i: i["resid"],
        "cusum": lambda i: (i["S"] - i["resid"] - i["kref"]).clamp(min=0),
        "cusum_reset": lambda i: i["S"],
        "pause_len": lambda i: i["L"],
    }


BASE_HOOKS = _base_hooks()


@dataclass
class VulnOperator:
    """A mechanism mutation: `hooks` maps hook name -> fn(inputs: Dict[str, Tensor], knob: float) -> Tensor (same shape as the base output).
    `off_knob` is the knob value for which the mutation must equal the base mechanism bitwise."""
    id: str
    hooks: Dict[str, Callable[[Dict[str, Any], float], Any]]
    off_knob: float = 0.0
    blind: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    ext: Optional[Callable[..., Any]] = None      # designer mutations only: factory(B, dev, knob, st) -> extension state object (see _Ext)
    full: bool = False                            # designer mutations only: the target hook acts on every agent, not on agent 1 only

    def __post_init__(self):
        bad = set(self.hooks) - set(HOOKS) - {"release"}
        if bad:
            raise ValueError(f"unknown hooks {sorted(bad)}; allowed {HOOKS}")


def _o1(i, k):
    return torch.where(i["win_streak"] >= M_WIN, i["p"] * (1 - k), i["p"])


def _o2(i, k):
    return (1 - k) * (i["S"] - i["resid"] - i["kref"]).clamp(min=0)


def _o3(i, C):
    return torch.clamp(i["resid"], min=-C)


def _o4(i, k):
    return torch.where(i["clean_streak"] >= M_CLEAN, (1 - k) * i["L"], i["L"])


def _o5(i, g):
    return torch.where(i["rounds_since_win"] >= g, torch.zeros_like(i["S"]), i["S"])


def _o6(i, k):
    return (1 - k) * i["L"]


_OPERATORS: Dict[str, VulnOperator] = {
    "O1": VulnOperator("O1", {"audit_prob": _o1}, 0.0),
    "O2": VulnOperator("O2", {"cusum": _o2}, 0.0),
    "O3": VulnOperator("O3", {"resid": _o3}, math.inf),
    "O4": VulnOperator("O4", {"pause_len": _o4}, 0.0, blind=True),
    "O5": VulnOperator("O5", {"cusum_reset": _o5}, math.inf, blind=True),
    "O6": VulnOperator("O6", {"pause_len": _o6}, 0.0),
}


def register_operator(op: VulnOperator) -> None:
    """Insertion interface for the blind designers' mutations (B1, B2, D1..D4): they become selectable by id in `simulate_vuln`."""
    if op.id in _OPERATORS:
        raise ValueError(f"operator {op.id} already registered")
    _OPERATORS[op.id] = op


def get_operator(op: Any) -> Optional[VulnOperator]:
    if op is None:
        return None
    if isinstance(op, VulnOperator):
        return op
    if op in _OPERATORS:
        return _OPERATORS[op]
    raise ValueError(f"unknown vulnerability operator {op!r}")


def off_knob(op: Any) -> float:
    """The knob value at which the operator is the identity (O3: C=inf, O5: g=inf, others kappa=0)."""
    return get_operator(op).off_knob


# ------------------------------------------------------------------------------------------------ designer_2 mutations (D1..D4 = V1..V4)
# docs/direction2-blind-vulns-2.md.  They need state/inputs beyond the five base hooks, so each carries an extension object (`_Ext`) that the
# simulator calls at fixed points of the round; every extension is a pure no-op for the identity knob (checked bitwise in scripts/run_d2_vulns.py).
# Interpretations (also recorded in vuln_suite.json meta): the base env uses the true slope r_true as "rh" (frontier's r_as branch), so
# V3 shrinks r_true and V4's rh_a starts at r_true ("the prior"); V4's NLMS acts on every agent's audits (rh_a[B,K]); G=5, EPS_REP=0.02.
D_G_REST, D_EPS_REP = 5, 0.02
_ONES = lambda x: torch.ones_like(x)


class _Ext:
    """No-op extension interface; `kn` is the knob (float or per-element [B] tensor)."""

    def __init__(self, B: int, dev: str, knob: Any, st: Any):
        self.B, self.dev, self.knob = B, dev, knob

    def p_inputs(self, st, idx):                       # extra inputs of the audit_prob hook
        return {}

    def release(self, hook, st, t, elig, v, susp, log):   # returns susp
        return susp

    def after_assign(self, st, idx, win_eff, elig):
        pass

    def resid_inputs(self, st, agc, zv, zw, rh):
        return {}

    def after_cusum(self, st, agc, ohag, upd, el_, zv, zw, rh):
        pass

    def after_flag(self, ohe):
        pass


def _kcol(knob, ref):
    """knob as a broadcastable column for [B,K] tensors"""
    return knob[:, None] if torch.is_tensor(knob) and knob.dim() == 1 else knob


class _ExtD1(_Ext):
    """V1 RestedAuditDiscount: lose_run[B,K] = consecutive eligible rounds without a win (reset on a win and on a flag; suspended rounds do not count)."""

    def __init__(self, B, dev, knob, st):
        super().__init__(B, dev, knob, st)
        self.lose_run = torch.zeros(B, K, dtype=F64, device=dev)

    def p_inputs(self, st, idx):
        return {"lose_run_w": self.lose_run.gather(1, idx[:, None])[:, 0]}

    def after_assign(self, st, idx, win_eff, elig):
        won = torch.nn.functional.one_hot(idx, K).bool() & win_eff[:, None]
        self.lose_run = torch.where(won, torch.zeros_like(self.lose_run), self.lose_run + elig.to(F64))

    def after_flag(self, ohe):
        self.lose_run = torch.where(ohe, torch.zeros_like(self.lose_run), self.lose_run)


def _d1_audit_prob(i, k):
    ones = _ONES(i["p"])
    return i["p"] * torch.where(i["lose_run_w"] >= D_G_REST, (1.0 - k) * ones, ones)


def _calm_next(sus, v, calm):
    return torch.where(sus & (v <= D_EPS_REP), calm + 1.0, torch.zeros_like(calm))


class _ExtD2(_Ext):
    """V2 SuspensionParole: calm[B,K] = consecutive suspended rounds with a report <= EPS_REP; Q such rounds release the agent (susp := t)."""

    def __init__(self, B, dev, knob, st):
        super().__init__(B, dev, knob, st)
        self.calm = torch.zeros(B, K, dtype=F64, device=dev)

    def release(self, hook, st, t, elig, v, susp, log):
        sus = ~elig
        tf = t if torch.is_tensor(t) else float(t)          # t: Python int (eager) or the device-side float64 time of graph stepping
        inputs = {"susp1": susp[:, 1], "v1": v[:, 1], "calm1": self.calm[:, 1], "sus1": sus[:, 1].to(F64), "t": _env._tfull(susp[:, 1], tf)}
        calm_new = _calm_next(sus, v, self.calm)
        new = _d2_release({"susp": susp, "calm_new": calm_new, "sus": sus, "t": tf}, self.knob)
        parole = new != susp
        self.calm = torch.where(parole, torch.zeros_like(calm_new), calm_new)
        if log is not None:
            rec = log.setdefault("release", {"mask": [], "out": []})
            rec["mask"].append(torch.ones(susp.shape[0], dtype=torch.bool, device=susp.device))
            rec["out"].append(new[:, 1].clone())
            for k_, v_ in inputs.items():
                rec.setdefault(k_, []).append(v_.clone())
        return new


def _d2_release(i, q):
    parole = i["sus"] & (i["calm_new"] >= _kcol(q, None))
    return torch.where(parole, _env._tfull(i["susp"], i["t"]), i["susp"])


class _ExtD3(_Ext):
    """V3 SlopeShrink: stateless; only supplies the hook's inputs."""

    def resid_inputs(self, st, agc, zv, zw, rh):
        return {"zv": zv, "zw": zw, "rh": rh}


def _d3_resid(i, k):
    rh = i["rh"] * (1.0 - k)
    sdh = torch.sqrt((1 - rh ** 2).clamp(min=1e-18))
    return (i["zw"] - rh * i["zv"]) / sdh


class _ExtD4(_Ext):
    """V4 SelfCalibratingSlope: rh_a[B,K] per-agent calibrated slope (init r_true), NLMS step eta, reset to the prior when the agent is flagged."""

    def __init__(self, B, dev, knob, st):
        super().__init__(B, dev, knob, st)
        self.rh_a = st.r_true[:, None].expand(B, K).clone()

    def resid_inputs(self, st, agc, zv, zw, rh):
        return {"zv": zv, "zw": zw, "rh": rh, "rh_a_ag": self.rh_a.gather(1, agc[:, None])[:, 0]}

    def after_cusum(self, st, agc, ohag, upd, el_, zv, zw, rh):
        eta = self.knob if torch.is_tensor(self.knob) else torch.full_like(rh, float(self.knob))
        use_cal = eta > 0
        rha = self.rh_a.gather(1, agc[:, None])[:, 0]
        rh_c = torch.where(use_cal, rha, rh)
        err = zw - rh_c * zv
        new_rh = (rha + eta * zv * err / (1 + zv * zv)).clamp(0, 0.995)
        new_rh = torch.where(el_, st.r_true, new_rh)
        self.rh_a = torch.where(ohag.bool() & (upd & use_cal)[:, None], new_rh[:, None].expand(self.B, K), self.rh_a)


def _d4_resid(i, eta):
    # a Python-number eta is filled on the device (as_tensor(float, device=cuda) is a host-to-device copy, which CUDA graph capture forbids); same value
    use = (eta if torch.is_tensor(eta) else torch.full((), float(eta), dtype=F64, device=i["rh"].device)) > 0
    rh_c = torch.where(use, i["rh_a_ag"], i["rh"])
    sdh = torch.sqrt((1 - rh_c ** 2).clamp(min=1e-18))
    return torch.where(use, (i["zw"] - rh_c * i["zv"]) / sdh, i["resid"])


_DESIGNER_OPS = {
    "D1": VulnOperator("D1", {"audit_prob": _d1_audit_prob}, 0.0, ext=_ExtD1, full=True,
                       meta=dict(name="V1_RestedAuditDiscount", source="designer_2", knob_name="s", knob_range=[0.0, 1.0], stronger="larger")),
    "D2": VulnOperator("D2", {"release": lambda i, q: _d2_release(i, q)}, math.inf, ext=_ExtD2, full=True,
                       meta=dict(name="V2_SuspensionParole", source="designer_2", knob_name="Q", knob_range=[1, "L"], stronger="smaller")),
    "D3": VulnOperator("D3", {"resid": _d3_resid}, 0.0, ext=_ExtD3, full=True,
                       meta=dict(name="V3_SlopeShrink", source="designer_2", knob_name="s", knob_range=[0.0, 0.6], stronger="larger")),
    "D4": VulnOperator("D4", {"resid": _d4_resid}, 0.0, ext=_ExtD4, full=True,
                       meta=dict(name="V4_SelfCalibratingSlope", source="designer_2", knob_name="eta", knob_range=[0.0, 0.3], stronger="larger")),
}
_OPERATORS.update(_DESIGNER_OPS)
DESIGNER_OPERATOR_IDS = ("D1", "D2", "D3", "D4")


# ------------------------------------------------------------------------------------------------ calibration / registry
def calibrate_knob(op: str, target_size: float, eval_delta: Callable[[float], float], grid: Sequence[float]) -> Dict[str, Any]:
    """REQ-S1-06: deterministic search over `grid` for the knob whose delta lies in [0.8s,1.2s]; returns
    {'status':'ok','knob':k} or {'status':'needs_commander'} (never substitutes another operator).
    Rule: the in-band grid point whose delta is closest to the target (ties: earlier grid point); grid scanned in the given order."""
    lo, hi = 0.8 * target_size, 1.2 * target_size
    best, best_err = None, None
    for k in grid:
        d = eval_delta(k)
        if lo <= d <= hi:
            err = abs(d - target_size)
            if best_err is None or err < best_err:
                best, best_err = k, err
    if best is None:
        return {"status": "needs_commander", "op": op, "target": target_size}
    return {"status": "ok", "knob": best, "op": op, "target": target_size}


_DEFAULT_REGISTRY = "results/direction2/vuln_suite.json"


def load_registry(path: str = _DEFAULT_REGISTRY) -> List[Dict[str, Any]]:
    """REQ-S1-03: frozen registry entries (id, kind, operator, knob, size, blind, ref_policy, G_star_val, G_HW_best_val,
    delta_val, delta_train64)."""
    if not os.path.exists(path) and not os.path.isabs(path):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), path)
        path = alt if os.path.exists(alt) else path
    with open(path) as f:
        doc = json.load(f)
    return doc["entries"] if isinstance(doc, dict) else doc


# ------------------------------------------------------------------------------------------------ simulator with hooks
class _HS:
    """Agent-1 bookkeeping used by the hooks (float64 [B] tensors)."""

    def __init__(self, B: int, dev: str):
        z = lambda: torch.zeros(B, dtype=F64, device=dev)
        self.win, self.clean, self.rsw = z(), z(), z()
        self.ext = None            # designer mutations: _Ext object
        self.rules = []            # in-loop rule policies: [(mask[B], loop-rule object)]
        self.wrec = None           # optional (warm-start recorder): list of (obs[B,F], u1[B], v1[B]) per round
        self.plog = None           # optional public log of agent 1: list of [B,4] (u1, v1, won1, susp1 at the start of the round)
        self.cur_susp1 = None


@torch.no_grad()
def _vstep(st, bank, op: Optional[VulnOperator], knob: float, hs: _HS, log: Optional[Dict[str, Dict[str, list]]]) -> None:
    """One round of env._step (M3C), with the five hooks.  Everything outside the hooks is a verbatim copy of the base step."""
    t = st.t + 1
    tt = t if st._feed is None else st._feed.tt          # time as a number in float64 arithmetic: Python int (eager) / device scalar (graph)
    s = (t - 1) % BLK
    dev, B = st.dev, st.B
    sidx, el = st.sidx, bank.el
    z, u = _env._next_types(st)
    obs = None
    if st.track_obs:
        ps, ln = _env._obs_state(st, z, u, st.obs_version)
        obs = st.builder.step(ps)
        st.susp_seen = st.susp[:, 1].clone()
        st.last_notice = ln
    d = bank.act(obs) if bank.has_mlp else None
    st.z, st.u = z, u
    susp, cus = st.susp, st.cus
    elig = tt > susp
    u1 = u[:, 1]
    zero = lambda *sh: torch.zeros(*sh, dtype=F64, device=dev)
    ar = torch.arange(B, device=dev)
    one = torch.ones(B, dtype=torch.bool, device=dev)
    Wd, Hd, emg, leg = el["W"], el["H"], el["em"], el["leg"]
    ue_mode, ue_band, ue_q, ue_dz = el["ue_mode"], el["ue_band"], el["ue_q"], el["ue_dz"]
    ad_m, ad_f, ad_c = el["ad_m"], el["ad_f"], el["ad_c"]
    ad_h, crit, ssh, th = st.ad_h, st.crit, st.ssh, st.th

    def hook(name: str, inputs: Dict[str, Any], mask) -> Any:
        fn = op.hooks.get(name) if op is not None else None
        out = fn(inputs, knob) if fn is not None else BASE_HOOKS[name](inputs)
        if log is not None:
            rec = log[name]
            rec["mask"].append(mask.clone())
            rec["out"].append(out.clone())
            for k, v in inputs.items():
                rec.setdefault(k, []).append(v.clone())
        return out

    mutated = lambda name: op is not None and name in op.hooks
    ex = hs.ext
    full = op is not None and op.full
    # ---- deviator's report
    if bank.has_adapter:
        dzl = el["dz"] + el["dzc"] * crit
        if bank.use_burst:
            lie_on = torch.where(Wd > 0, ((t - 1) % (Wd + Hd).clamp(min=1)) < Wd, one)
            dzl = dzl * lie_on
        if bank.use_ad:
            dzl = torch.where(ad_m > 0, torch.where(ssh < ad_f * ad_h, ad_m * crit, torch.zeros_like(crit)), dzl)
        v1 = torch.minimum(torch.special.ndtr(z[:, 1] + dzl) + el["b"], torch.ones_like(u1))
        if bank.has_mlp:
            v1 = torch.where(bank.mlp_mask, _env.report(u1, d), v1)
    else:
        v1 = _env.report(u1, d)
    if hs.rules:
        rctx = dict(t=t, u1=u1, z1=torch.special.ndtri(u1.clamp(1e-12, 1 - 1e-12)), susp1=susp[:, 1], elig1=elig[:, 1])
        for rmask, rl in hs.rules:
            v1 = torch.where(rmask, rl.report(rctx), v1)
    if hs.plog is not None:
        hs.cur_susp1 = susp[:, 1].clone()
    v = torch.cat([u[:, :1], v1[:, None], u[:, 2:]], 1)
    if bank.use_edge:
        oth = torch.where(leg[:, None] | elig, u, torch.full_like(u, -1.0))
        mo = torch.maximum(oth[:, 0], oth[:, 2])
        gap = mo - u1
        e_on = (emg > 0) & (gap > 0) & (gap < emg) & (mo >= 0)
        v1e = torch.where(e_on, torch.clamp(mo + 1e-3, max=1.0), torch.where(emg > 0, u1, v[:, 1]))
        v = torch.cat([v[:, :1], v1e[:, None], v[:, 2:]], 1)
    if bank.use_ue:
        ue_on = (ue_mode > 0) & (u1 < th) & (u1 >= th - ue_band)
        vu = torch.where(ue_on & (ue_mode == 1), torch.clamp(th, max=1.0),
                         torch.where(ue_on, torch.minimum(torch.special.ndtr(z[:, 1] + ue_dz), torch.ones_like(u1)), u1))
        v1u = torch.where(ue_mode > 0, vu, v[:, 1])
        v = torch.cat([v[:, :1], v1u[:, None], v[:, 2:]], 1)
    # ---- allocation (M3C: no raids)
    vm = torch.where(elig, v, torch.full_like(v, -1.0))
    mx, idx = vm.max(1)
    win = mx >= 0
    st.pen = st.pen + ~elig
    if ex is not None:
        susp = ex.release(hook, st, tt, elig, v, susp, log)
    umax = u.max(1).values
    uw = u.gather(1, idx[:, None])[:, 0]
    Rs = _env._row(st, "Rn", s)
    rd = torch.zeros(B, dtype=torch.bool, device=dev)
    flags, nraid = st.flags, st.nraid
    win_eff = win & ~rd
    ridx = (Rs[:, 1] * K).long().clamp(max=K - 1)
    urand = u.gather(1, ridx[:, None])[:, 0]
    ag_b = torch.where(win_eff, idx, ridx)
    uw_b = torch.where(win_eff, uw, urand) * (~rd)
    st.reg_b = st.reg_b + (umax - uw_b)
    st.util_b = st.util_b + torch.nn.functional.one_hot(ag_b, K).to(F64) * uw_b[:, None]
    if bank.use_ue:
        probe = ue_on & elig[:, 1] & win
        st.th = th + 0.02 * probe * ((idx != 1).to(F64) - (1 - ue_q))
    M = st.M
    slot = (t + st.tau) % M
    if bank.use_ad:
        st.shv[ar, slot] = torch.where(win_eff & (idx == 1), v[:, 1], -torch.ones_like(u1))
    # ---- HOOK audit_prob: audit decision of the round's winner (agent 1's probability is the hook's output)
    p_in = {"win_streak": hs.win, "p": st.p_el}
    if ex is not None:
        p_in.update(ex.p_inputs(st, idx))
    p_eff = hook("audit_prob", p_in, one)
    p_used = (p_eff if full else torch.where(idx == 1, p_eff, st.p_el)) if mutated("audit_prob") else st.p_el
    o = win_eff & (_env._row(st, "Ua", s) < p_used)
    if ex is not None:
        ex.after_assign(st, idx, win_eff, elig)
    st.audits = st.audits + o
    st.pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
    st.pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
    liar_flag = torch.zeros(B, dtype=torch.bool, device=dev)
    cs = t % M
    ag = st.pend_a[:, cs]
    has = ag >= 0
    agc = ag.clamp(min=0)
    is1 = has & (agc == 1)
    wv = u.gather(1, agc[:, None])[:, 0]
    pv = st.pend_v[:, cs]
    elig_ag = (tt > susp).gather(1, agc[:, None])[:, 0]
    ohag = torch.nn.functional.one_hot(agc, K).to(F64)
    zv = torch.special.ndtri(_env.phi_of(pv, agc, _env.C_MIN))
    zw = torch.special.ndtri(_env.phi_of(wv, agc, _env.C_MIN))
    rh = st.r_true
    sdh = torch.sqrt((1 - rh ** 2).clamp(min=1e-18))
    resid = (zw - rh * zv) / sdh
    st.sxy = st.sxy + torch.where(has, zv * zw, torch.zeros_like(zv))
    st.sxx = st.sxx + torch.where(has, zv * zv, torch.zeros_like(zv))
    # ---- HOOK resid (agent 1's settlements only)
    r_in = {"resid": resid}
    if ex is not None:
        r_in.update(ex.resid_inputs(st, agc, zv, zw, rh))
    r_h = hook("resid", r_in, has if (full and mutated("resid")) else is1)
    if mutated("resid"):
        resid = r_h if full else torch.where(is1, r_h, resid)
    upd = has & elig_ag
    Sg = cus.gather(1, agc[:, None])[:, 0]
    # ---- HOOK cusum
    Sn_b = (Sg - resid - st.kref).clamp(min=0)
    Sn_h = hook("cusum", {"S": Sg, "resid": resid, "kref": st.kref}, upd & (agc == 1))
    Sn = torch.where(agc == 1, Sn_h, Sn_b) if mutated("cusum") else Sn_b
    el_ = upd & (Sn > st.tol)
    newS = torch.where(upd, torch.where(el_, torch.zeros_like(Sn), Sn), Sg)
    cus = torch.where(ohag.bool(), newS[:, None].expand(B, K), cus)
    ohe = ohag.bool() & el_[:, None]
    if ex is not None:
        ex.after_cusum(st, agc, ohag, upd, el_, zv, zw, rh)
        ex.after_flag(ohe)
    flags = flags + ohe
    liar_flag = el_ & (agc == 1)
    # ---- HOOK pause_len (flags of agent 1)
    L_h = hook("pause_len", {"L": st.L, "clean_streak": hs.clean}, liar_flag)
    Lvec = torch.where(liar_flag, L_h, st.L) if mutated("pause_len") else st.L
    susp = torch.where(ohe, (tt + Lvec)[:, None].expand(B, K), susp)
    hs.clean = torch.where(upd & (agc == 1), torch.where(liar_flag, torch.zeros_like(hs.clean), hs.clean + 1), hs.clean)
    st.pend_a[:, cs] = -1
    if bank.use_ad:
        sv = st.shv[:, cs]
        hv = sv >= 0
        ones = torch.ones_like(idx)
        zvs = torch.special.ndtri(_env.phi_of(sv.clamp(min=0), ones, _env.C_MIN))
        zws = torch.special.ndtri(_env.phi_of(u1, ones, _env.C_MIN))
        rs_ = (zws - st.r_true * zvs) / st.sd_true
        coin = _env._row(st, "Rx", s)[:, 1] < (ad_c * st.p_el).clamp(max=1.0)
        Sn2 = torch.where(hv & coin, (ssh - rs_ - st.kref).clamp(min=0), ssh)
        st.ssh = torch.where(liar_flag | (Sn2 > ad_h), torch.zeros_like(Sn2), Sn2)
        st.shv[:, cs] = -1
    st.susp, st.cus, st.flags, st.nraid, st.v, st.t = susp, cus, flags, nraid, v, t
    won1 = win_eff & (idx == 1)
    if st.track_obs:
        st.won_hist = torch.cat([won1.to(F64)[:, None], st.won_hist[:, :-1]], 1)
        st.d_hist = torch.cat([(v[:, 1] - u1)[:, None], st.d_hist[:, :-1]], 1)
    for _m, rl in hs.rules:
        rl.update(dict(t=t, u1=u1, elig1=elig[:, 1], won1=won1))
    if hs.plog is not None:
        hs.plog.append(torch.stack([u1, v[:, 1], won1.to(F64), hs.cur_susp1], 1))
    if hs.wrec is not None:
        hs.wrec.append((obs, u1.clone(), v[:, 1].clone()))
    # ---- agent-1 streak bookkeeping and HOOK cusum_reset (every round, after the settlement)
    hs.win = torch.where(won1, hs.win + 1, torch.zeros_like(hs.win))
    hs.rsw = torch.where(won1, torch.zeros_like(hs.rsw), hs.rsw + 1)
    S_out = hook("cusum_reset", {"S": st.cus[:, 1], "rounds_since_win": hs.rsw}, one)
    if mutated("cusum_reset"):
        st.cus = torch.cat([st.cus[:, :1], S_out[:, None], st.cus[:, 2:]], 1)


def simulate_vuln(operator: Any, knob: Any, mech: str, T: int, cfg: Dict[str, float], policies: Sequence[Any],
                  seeds: Sequence[int], *, r: float, dev: str = "cuda", hook_log: bool = False,
                  T_score: Optional[int] = None, public_log: bool = False, warm_obs: bool = False, graph: Any = None) -> Dict[str, Any]:
    """REQ-S1-05: simulate in the vulnerable environment; operator=None must equal env.simulate bitwise.

    operator: None | 'O1'..'O6' | 'D1'..'D4' | registered id | VulnOperator; knob: its size parameter (kappa; C for O3; g for O5; s/Q/s/eta for D1..D4);
    a sequence of length len(policies) gives every policy its own knob (used to evaluate several knob values in one batch).
    policies may contain RulePolicy objects (history rules run in the loop; see the rule section at the end of this module).
    warm_obs=True (warm start, REQ-OPT-07) additionally returns out['warm_obs'] = dict(obs[P,S,T,F], u1[P,S,T], v1[P,S,T]): the public observation the
    policy sees INSIDE this vulnerable environment and the report it makes (obs tracking does not change the simulation).
    Returns the env.simulate fields ('final', 'snap', 'T_score') and, when hook_log=True, out['hook_log'][hook] =
    dict(mask=bool[P,S,T], out=[P,S,T], <inputs>=[P,S,T]) for the 5 hooks (inputs are those the hook received in this run).
    graph: a graphsim.GraphSim -> the step loop runs as CUDA-graph replays (bitwise identical; used only for plain training runs: MLP policies, no logs,
    no rules; anything else runs the eager loop)."""
    op = get_operator(operator)
    has_rule = any(isinstance(p, RulePolicy) for p in policies)
    if op is None and not hook_log and not has_rule and not public_log and not warm_obs:
        return _env.simulate(mech, T, cfg, policies, seeds, r=r, T_score=T_score, dev=dev, obs_version="public", graph=graph)
    if mech != "M3C":
        raise ValueError("vulnerability operators and hook logs are defined for the M3C audit block only")
    P, S = len(policies), len(seeds)
    L = float(cfg["L"])
    Ts = (_env.score_window(T, int(L)) if T > L else int(T)) if T_score is None else int(T_score)
    if not 1 <= Ts <= T:
        raise ValueError("T_score must lie in [1, T]")
    N = P * S
    seeds_a = np.asarray([int(x) for x in seeds])
    e = np.arange(N)
    pid, sd = e // S, e % S
    uniq, inv = np.unique(seeds_a[sd], return_inverse=True)
    need_obs = warm_obs or any(isinstance(p, MLPPolicy) for p in policies)
    st = _env.SimState("M3C", cfg, r, uniq.tolist(), inv, dev, "public", track_obs=need_obs)
    bank_pols = [AdapterPolicy({}) if isinstance(p, RulePolicy) else p for p in policies]
    bank = _env._Bank(bank_pols, torch.as_tensor(pid, device=dev), dev, 25, obs_version="public")
    hs = _HS(N, dev)
    if public_log:
        hs.plog = []
    if warm_obs:
        hs.wrec = []
    if isinstance(knob, (list, tuple, np.ndarray)):
        if len(knob) != P:
            raise ValueError("a knob sequence needs one entry per policy")
        knob = torch.as_tensor(np.asarray(knob, dtype=float), dtype=F64, device=dev)[torch.as_tensor(pid, device=dev)]
    if op is not None and op.ext is not None:
        hs.ext = op.ext(N, dev, knob, st)
    pid_t = torch.as_tensor(pid, device=dev)
    for name, cls in LOOP_RULES.items():
        members = [j for j, p in enumerate(policies) if isinstance(p, RulePolicy) and p.name == name]
        if members:
            tab = {k: torch.tensor([float(policies[j].params.get(k, dv)) if j in members else dv for j in range(P)], dtype=F64, device=dev)[pid_t]
                   for k, dv in RULE_DEFAULTS[name].items()}
            hs.rules.append((torch.isin(pid_t, torch.tensor(members, device=dev)), cls(tab, N, dev)))
    log = {h: {"mask": [], "out": []} for h in HOOKS} if hook_log else None
    snap = None
    if graph is not None and bank.has_mlp and not bank.has_adapter and not hs.rules and log is None and hs.plog is None and hs.wrec is None:
        holder = types.SimpleNamespace(knob=knob)
        kid = "tensor" if torch.is_tensor(knob) else repr(knob)
        ident = ("vuln", op.id if op is not None else None, id(op), kid, mech, tuple(sorted(cfg.items())), float(r), int(N), int(len(uniq)), dev)
        snap, final = graph.run(ident, dict(st=st, bank=bank, hs=hs, holder=holder), lambda: _vstep(st, bank, op, holder.knob, hs, None), T, Ts,
                                _env._FIELDS)
    else:
        for t in range(1, T + 1):
            _vstep(st, bank, op, knob, hs, log)
            if t == Ts:
                snap = {f: getattr(st, f).clone() for f in _env._FIELDS}
        final = {f: getattr(st, f) for f in _env._FIELDS}
    rs = lambda x: x.reshape(P, S, *x.shape[1:])
    out = dict(final={f: rs(final[f]) for f in _env._FIELDS}, snap={f: rs(v) for f, v in snap.items()}, T_score=Ts)
    if warm_obs:
        out["warm_obs"] = dict(obs=rs(torch.stack([x[0] for x in hs.wrec], 1)), u1=rs(torch.stack([x[1] for x in hs.wrec], 1)),
                               v1=rs(torch.stack([x[2] for x in hs.wrec], 1)))
    if public_log:
        out["public_log"] = rs(torch.stack(hs.plog, 1))
    if hook_log:
        out["hook_log"] = {h: {k: rs(torch.stack(v, 1)) for k, v in rec.items()} for h, rec in log.items()}
    return out


# ------------------------------------------------------------------------------------------------ delta measurement
def build_ref_policy(spec: Dict[str, Any]) -> Any:
    """Reference exploiting policy of a registry entry: {'kind':'adapter','scols':{...}} or {'kind':'mlp','F':25,'hidden':2,'theta':[...]}
    (all run inside the simulator loop; 'rule' policies are RulePolicy objects with an in-loop twin of the public-history rule, which is also
    what run_ref_policy_sandbox executes)."""
    kind = spec.get("kind")
    if kind == "adapter":
        return AdapterPolicy(dict(spec["scols"]))
    if kind == "mlp":
        return MLPPolicy(F=spec.get("F", 25), hidden=spec.get("hidden", 2), theta=spec["theta"], obs_version=spec.get("obs_version", "public"))
    if kind == "rule":
        return RulePolicy(spec["name"], spec.get("params", {}))
    raise ValueError(f"reference policy kind {kind!r} cannot be simulated in the loop")


def measure_delta(operator: Any, knob: float, ref_policy: Any, hw_policies: Sequence[Any], seeds: Sequence[int], *, r: float = 0.5,
                  T: int = 100000, cfg: Optional[Dict[str, float]] = None, dev: str = "cuda") -> Dict[str, Any]:
    """REQ-S1-02: Delta(knob) = G*_V - G_HW,best_V on `seeds` (paired per seed; G = (U_pol - U_honest)/T_score; honest = same seed, same V).
    hw_policies: the S_HW adapters in S_HW order (best = largest mean, ties -> earlier)."""
    cfg = dict(_env.m3c_config(r) if cfg is None else cfg)
    pols = [AdapterPolicy({}), ref_policy] + list(hw_policies)
    out = simulate_vuln(operator, knob, "M3C", T, cfg, pols, seeds, r=r, dev=dev)
    U = out["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float)
    G = (U[1:] - U[0]) / out["T_score"]
    g_star, g_hw = G[0], G[1:]
    best = int(np.argmax(g_hw.mean(1)))                       # argmax returns the first maximum: ties -> earlier
    return dict(G_star_s=g_star, G_HW_s=g_hw, best_index=best, delta=mean_ci(g_star - g_hw[best]), T_score=out["T_score"])


def recompute_delta(entry: Dict[str, Any], dev: str = "cuda") -> Dict[str, Any]:
    """REQ-S1-02/28 (v1.3): re-measure on val U val2 (64 seeds: 2000-2031 then 3000-3031) against the 36-name S_HW.  Returns
    'val_seeds' (64), 'G_star_s' [64], 'G_HW_s_by_name' {36 names: [64]} (so that a test can recompute everything itself),
    'delta_fine' / 'delta_coarse' {'mean','lo','hi','half'} (t(63) CI; fine = best of the 36, coarse = best of the legacy 32),
    'delta' (= delta_fine), 'G_HW_best_name' (fine best) and 'G_HW_best_coarse_name'."""
    from . import hw
    s_hw = hw.build_s_hw()
    names = [n for n, _ in s_hw]
    coarse = {n for n, _ in hw.build_s_hw_coarse()}
    seeds = list(CALIB_SEEDS)
    res = measure_delta(entry.get("operator"), entry["knob"], build_ref_policy(entry["ref_policy"]),
                        [AdapterPolicy(d) for _, d in s_hw], seeds, r=entry.get("r", 0.5), T=entry.get("T", 100000), dev=dev)
    g_star, g_hw = res["G_star_s"], res["G_HW_s"]
    ic = [i for i, n in enumerate(names) if n in coarse]
    bc = ic[int(np.argmax(g_hw[ic].mean(1)))]                  # ties -> earlier
    return dict(delta=res["delta"], delta_fine=res["delta"], delta_coarse=mean_ci(g_star - g_hw[bc]),
                G_HW_best_name=names[res["best_index"]], G_HW_best_coarse_name=names[bc], val_seeds=seeds, G_star_s=g_star,
                G_HW_s_by_name={n: g_hw[i] for i, n in enumerate(names)}, G_star=float(g_star.mean()))


# ------------------------------------------------------------------------------------------------ public-history sandbox
PUBLIC_RULES: Dict[str, Callable[..., float]] = {}
_PUBLIC_SCOLS = {"b", "dz", "dzc", "W", "H"}
HISTORY_COLUMNS = ("u1", "v1", "won1", "susp1")      # sandbox history: one row per round, public quantities of agent 1 only


def register_public_rule(name: str, fn: Callable[..., float]) -> None:
    """A rule fn(history[t+1, 4], **params) -> report v1 in [0,1] of round t; it receives ONLY the public history (columns HISTORY_COLUMNS;
    in the last row only u1 is current, the other columns are zeroed) and constants in params."""
    PUBLIC_RULES[name] = fn


def _adapter_report(scols: Dict[str, float], u1: float, t: int, r: float, kref: float = 0.5) -> float:
    from scipy.special import ndtr, ndtri
    bad = set(scols) - _PUBLIC_SCOLS
    if bad:
        raise ValueError(f"strategy columns {sorted(bad)} are not realizable from public history")
    z = float(ndtri(min(max(u1, 1e-12), 1 - 1e-12)))
    crit = kref * math.sqrt(1 - r * r) / max(r, 1e-9)
    dzl = scols.get("dz", 0.0) + scols.get("dzc", 0.0) * crit
    if scols.get("W", 0) > 0 and not ((t - 1) % max(scols["W"] + scols.get("H", 0), 1)) < scols["W"]:
        dzl = 0.0
    return float(min(ndtr(z + dzl) + scols.get("b", 0.0), 1.0))


def run_ref_policy_sandbox(entry: Dict[str, Any], history: Any, private: Any = None) -> Any:
    """REQ-S1-03: run entry's ref_policy through the public-history sandbox; private state is not an accepted influence.
    history[T,4] = (u1, v1, won1, susp1) per round; returns the reports v1[T].  `private` is deliberately never read."""
    del private                                              # not an accepted influence (REQ-OBS-06)
    h = np.array(history, dtype=float, copy=True)
    spec = entry["ref_policy"]
    r = float(entry.get("r", 0.5))
    out = np.zeros(h.shape[0])
    for t in range(h.shape[0]):
        view = h[: t + 1].copy()
        view[-1, 1:3] = 0.0                                  # the current round's own report/win are not yet known; the own suspension state
        #                                                      susp1 (column 3) IS known at the start of the round (obs: susp_minus_t)
        if spec["kind"] == "adapter":
            out[t] = _adapter_report(spec["scols"], view[-1, 0], t + 1, r)
        elif spec["kind"] == "rule":
            out[t] = PUBLIC_RULES[spec["name"]](view, **spec.get("params", {}))
        else:
            raise ValueError(f"reference policy kind {spec['kind']!r} has no public-history sandbox form")
    return out


# ------------------------------------------------------------------------------------------------ history rules (reference exploit policies)
# A rule is the pair (public-history function in PUBLIC_RULES, vectorised in-loop twin in LOOP_RULES).  Both see ONLY: the own type u1 (and
# z1 = ndtri(u1)), the round counter t, and the own history (reports v1, wins won1, suspension end susp1 -- the latter is public at the start of
# the round).  They never touch cus / pend_a / audit decisions.  scripts/run_d2_vulns.py checks the two forms agree on recorded runs.
from scipy.special import ndtr as _sp_ndtr, ndtri as _sp_ndtri

RULE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "d2_rested": dict(dz=1.0, G=5.0, dz0=0.0),        # V1: lie once, right after G eligible rounds without a win (dz0>0: hybrid, constant lie dz0 otherwise)
    "d2_parole": dict(dz=1.0),                        # V2: abstain (v=0) while suspended, lie otherwise
    "d2_slope": dict(s=0.0, gain=1.0, dz0=0.0),       # V3: lie only if z>0, by gain*s*z/(1-s) (dz0>0: hybrid, constant lie dz0 added in every round)
    "d2_climb": dict(kappa=0.001, dz_max=4.0, L=2500.0),   # V4/D4: lie amplitude climbs by kappa per round of the age since recovery max(since-L,0) (f05; REQ-S1-27); L = mechanism constant
    "streak_lie": dict(dz=1.0, m=3.0),                # O1: lie after m consecutive own wins
    "late_lie": dict(dz=1.0, t0=1500.0),              # O4 (blind): honest for t0 rounds after each recovery, then lie
    "launder": dict(dz=1.0, W=100.0, Z=20.0),         # O5 (blind): W rounds lying, Z rounds abstaining (v=0), repeat
}


class RulePolicy:
    """Reference policy defined by a history rule: `name` in RULE_DEFAULTS, `params` overriding its defaults."""

    def __init__(self, name: str, params: Optional[Dict[str, float]] = None):
        if name not in RULE_DEFAULTS:
            raise ValueError(f"unknown rule {name!r}")
        bad = set(params or {}) - set(RULE_DEFAULTS[name])
        if bad:
            raise ValueError(f"unknown parameters {sorted(bad)} for rule {name}")
        self.name, self.params, self.scols = name, dict(params or {}), None


def _lie(z, dz):
    return torch.clamp(torch.special.ndtr(z + dz), max=1.0)


class _LoopRule:
    def __init__(self, pt, B, dev):
        self.p, self.B, self.dev = pt, B, dev
        self.zero = lambda: torch.zeros(B, dtype=F64, device=dev)

    def update(self, c):
        pass


class _LRested(_LoopRule):
    def __init__(self, pt, B, dev):
        super().__init__(pt, B, dev)
        self.n = self.zero()

    def report(self, c):
        lie = c["elig1"] & (self.n >= self.p["G"])
        base = torch.where(self.p["dz0"] > 0, _lie(c["z1"], self.p["dz0"]), c["u1"])
        return torch.where(lie, _lie(c["z1"], self.p["dz"]), base)

    def update(self, c):
        self.n = torch.where(c["won1"], torch.zeros_like(self.n), torch.where(c["elig1"], self.n + 1, torch.zeros_like(self.n)))


class _LParole(_LoopRule):
    def report(self, c):
        return torch.where(c["elig1"], _lie(c["z1"], self.p["dz"]), torch.zeros_like(c["u1"]))


class _LSlope(_LoopRule):
    def report(self, c):
        z, s, d0 = c["z1"], self.p["s"], self.p["dz0"]
        base = torch.where(d0 > 0, _lie(z, d0), c["u1"])
        return torch.where(z > 0, _lie(z, d0 + self.p["gain"] * s * z / (1.0 - s)), base)


class _LClimb(_LoopRule):
    """D4 reference rule (REQ-S1-27, v1.3 final): the lie amplitude climbs by kappa per round of the age since RECOVERY, age = max(since - L, 0), where
    `since` is the number of rounds elapsed since the agent first OBSERVED its own susp_1 rise (the feature f05 of REQ-OBS-11, capped at 2L; never
    suspended = the cap, f05 = 1, age = L; the suspension lasts L rounds, so since - L is the time since the suspension ended).  Only the own public log is read: the
    own announced suspension end (susp1) and the own type.  The legacy rule (v1.2) used the absolute round index t, which the public log does not
    provide (red team, decision 5)."""

    def __init__(self, pt, B, dev):
        super().__init__(pt, B, dev)
        self.seen = self.zero()                                              # susp1 as observed in the previous round (the log starts at 0)
        self.since = torch.full((B,), float("inf"), dtype=F64, device=dev)   # inf = never observed a rise

    def report(self, c):
        susp1 = c["susp1"]
        self.since = torch.where(susp1 != self.seen, torch.zeros_like(self.since), self.since + 1.0)
        self.seen = susp1.clone()
        cap = 2.0 * self.p["L"]
        f05 = torch.minimum(self.since, cap) / cap                          # = feature f05 (obs.py / obs_reconstruct.py)
        age = torch.clamp(f05 * cap - self.p["L"], min=0.0)                 # age since RECOVERY = max(since - L, 0) (never suspended: L)
        dz = torch.minimum(self.p["kappa"] * age, self.p["dz_max"])
        return _lie(c["z1"], dz)


class _LStreak(_LoopRule):
    def __init__(self, pt, B, dev):
        super().__init__(pt, B, dev)
        self.ws = self.zero()

    def report(self, c):
        return torch.where(self.ws >= self.p["m"], _lie(c["z1"], self.p["dz"]), c["u1"])

    def update(self, c):
        self.ws = torch.where(c["won1"], self.ws + 1, torch.zeros_like(self.ws))


class _LLate(_LoopRule):
    """O4 (not in the suite since v1.3; code retained): honest for t0 rounds after each recovery, then lie.  Keeps the legacy absolute-t form."""

    def __init__(self, pt, B, dev):
        super().__init__(pt, B, dev)
        self.tr = self.zero()

    def report(self, c):
        return torch.where((c["t"] - self.tr) >= self.p["t0"], _lie(c["z1"], self.p["dz"]), c["u1"])

    def update(self, c):
        self.tr = torch.where(c["elig1"], self.tr, torch.full_like(self.tr, float(c["t"])))


class _LLaunder(_LoopRule):
    def report(self, c):
        W, Z = self.p["W"], self.p["Z"]
        ph = torch.remainder(float(c["t"] - 1), W + Z)
        return torch.where(ph < W, _lie(c["z1"], self.p["dz"]), torch.zeros_like(c["u1"]))


LOOP_RULES: Dict[str, Any] = {"d2_rested": _LRested, "d2_parole": _LParole, "d2_slope": _LSlope, "d2_climb": _LClimb, "streak_lie": _LStreak,
                              "late_lie": _LLate, "launder": _LLaunder}


def _hz(h):
    """z of the current own type from the last history row"""
    return float(_sp_ndtri(min(max(h[-1, 0], 1e-12), 1 - 1e-12)))


def _hlie(z, dz):
    return float(min(_sp_ndtr(z + dz), 1.0))


def _prm(name, params):
    d = dict(RULE_DEFAULTS[name]); d.update(params)
    return d


def _elig(h, j):
    """own eligibility in round j+1 (row j): t > susp1 (susp1 = suspension end announced at the start of the round)"""
    return (j + 1) > h[j, 3]


def _since_recovery(h):
    """last own suspended round index among previous rows (0 if never): the 't_rel' of V4 / O4"""
    tr = 0
    for j in range(h.shape[0] - 1):
        if not _elig(h, j):
            tr = j + 1
    return tr


def _rule_rested(h, **kw):
    p = _prm("d2_rested", kw)
    n = 0
    for j in range(h.shape[0] - 1):
        n = 0 if h[j, 2] > 0.5 else (n + 1 if _elig(h, j) else 0)
    t = h.shape[0]
    base = _hlie(_hz(h), p["dz0"]) if p["dz0"] > 0 else float(h[-1, 0])
    return _hlie(_hz(h), p["dz"]) if (t > h[-1, 3] and n >= p["G"]) else base


def _rule_parole(h, **kw):
    p = _prm("d2_parole", kw)
    t = h.shape[0]
    return _hlie(_hz(h), p["dz"]) if t > h[-1, 3] else 0.0


def _rule_slope(h, **kw):
    p = _prm("d2_slope", kw)
    z = _hz(h)
    base = _hlie(z, p["dz0"]) if p["dz0"] > 0 else float(h[-1, 0])
    return _hlie(z, p["dz0"] + p["gain"] * p["s"] * z / (1.0 - p["s"])) if z > 0 else base


def _since_public(h):
    """rounds since the own susp1 (column 3, as observed at the start of each round) last changed (REQ-OBS-11); inf if it never did.  Mirrors
    the in-loop rule: counts rounds, never reads the absolute index."""
    seen, since = 0.0, float("inf")
    for j in range(h.shape[0]):
        since = 0.0 if h[j, 3] != seen else since + 1.0
        seen = h[j, 3]
    return since


def _rule_climb(h, **kw):
    p = _prm("d2_climb", kw)
    cap = 2.0 * p["L"]
    f05 = min(_since_public(h), cap) / cap
    age = max(f05 * cap - p["L"], 0.0)                 # age since recovery = max(since - L, 0)
    return _hlie(_hz(h), min(p["kappa"] * age, p["dz_max"]))


def _rule_streak(h, **kw):
    p = _prm("streak_lie", kw)
    n = 0
    for j in range(h.shape[0] - 1):
        n = n + 1 if h[j, 2] > 0.5 else 0
    return _hlie(_hz(h), p["dz"]) if n >= p["m"] else float(h[-1, 0])


def _rule_late(h, **kw):
    p = _prm("late_lie", kw)
    t = h.shape[0]
    return _hlie(_hz(h), p["dz"]) if (t - _since_recovery(h)) >= p["t0"] else float(h[-1, 0])


def _rule_launder(h, **kw):
    p = _prm("launder", kw)
    t = h.shape[0]
    return _hlie(_hz(h), p["dz"]) if ((t - 1) % (p["W"] + p["Z"])) < p["W"] else 0.0


for _n, _f in (("d2_rested", _rule_rested), ("d2_parole", _rule_parole), ("d2_slope", _rule_slope), ("d2_climb", _rule_climb),
               ("streak_lie", _rule_streak), ("late_lie", _rule_late), ("launder", _rule_launder)):
    register_public_rule(_n, _f)


# ------------------------------------------------------------------------------------------------ batched knob / candidate measurement
def measure_blocks(operator: Any, blocks: Sequence[Dict[str, Any]], seeds: Sequence[int], *, r: float = 0.5, T: int = 100000,
                   cfg: Optional[Dict[str, float]] = None, dev: str = "cuda") -> List[Dict[str, Any]]:
    """Evaluate several knob values in ONE simulation.  blocks: [{'knob': k, 'cands': [ref-policy spec, ...]}].  For every block the environment
    runs the honest policy, the candidates and the 36 S_HW strategies (paired per seed, same V).  Returns per block
    {'knob', 'G_cands': [C,S], 'G_hw': [36,S], 'hw_names'}; G = (U_pol - U_honest)/T_score on agent 1."""
    from . import hw
    cfg = dict(_env.m3c_config(r) if cfg is None else cfg)
    s_hw = hw.build_s_hw()
    pols, knobs, idx = [], [], []
    for b in blocks:
        o = len(pols)
        specs = [build_ref_policy(c) for c in b["cands"]]
        pols += [AdapterPolicy({})] + specs + [AdapterPolicy(d) for _, d in s_hw]
        knobs += [b["knob"]] * (1 + len(specs) + len(s_hw))
        idx.append((o, len(specs)))
    out = simulate_vuln(operator, knobs, "M3C", T, cfg, pols, seeds, r=r, dev=dev)
    U = out["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float)
    res = []
    for b, (o, c) in zip(blocks, idx):
        G = (U[o + 1:o + 1 + c + len(s_hw)] - U[o]) / out["T_score"]
        res.append(dict(knob=b["knob"], G_cands=G[:c], G_hw=G[c:], hw_names=[n for n, _ in s_hw]))
    return res
