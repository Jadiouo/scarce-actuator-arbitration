"""REQ-ENV-*: batched simulator with the deviator's policy inside the loop (spec s2).

Tensor conventions: float64; batch element = [pol, seed] flattened row-major (cfg is fixed per call).  Output arrays are
shaped [P, S, ...] (P = len(policies), S = len(seeds)).

Information isolation (REQ-OBS-06): observation-based policies (MLPPolicy) only ever receive the obs tensor produced by
obs.ObsBuilder from a whitelisted PublicState/AnnouncedState/OracleState; they never see SimState.  AdapterPolicy objects are
direction-1 hand-written baselines evaluated through the frontier operation order (some read private state by design).
The frontier *simulate* function is never called here; only its constants/pure helpers are reused.
"""
import copy
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from arbitration.gpu.adaaudit import C_MIN, K, transform
from arbitration.gpu.audit_fix import phi_of
from arbitration.gpu.frontier import PCOLS, SCOLS
from . import obs as _obs
from .policy import AdapterPolicy, MLPPolicy, mlp_forward, report

F64 = torch.float64
INF = float("inf")
BLK = 1024


def m3c_config(r: float) -> Dict[str, float]:
    """REQ-ENV-10: the frozen M3C mechanism configuration {p=0.1, tol(h)=6, L=2500, kref=0.5, ad_h=inf}."""
    return dict(p=0.1, tol=6.0, L=2500.0, kref=0.5, ad_h=INF)


def anchor_config(name: str) -> Dict[str, Any]:
    """REQ-ENV-11: 'NAIVE' -> M3C with p=0, r=0.5; 'M4' -> M4 eps=0.2, L=20, rw=0, r=0.99 (anchor strategy b=0.3)."""
    if name == "NAIVE":
        return dict(mech="M3C", p=0.0, tol=6.0, L=2500.0, kref=0.5, r=0.5, strategy="dz=2.0")
    if name == "M4":
        return dict(mech="M4", eps=0.2, L=20, rw=0.0, r=0.99, strategy="b=0.3")
    raise ValueError(name)


def rng_block_size() -> int:
    """REQ-ENV-02: the fixed RNG block length (1024)."""
    return BLK


def _make_gens(seeds, dev):
    gens = ([], [], [])
    for s in seeds:
        for lst, off in zip(gens, (0, 10 ** 6, 2 * 10 ** 6)):
            g = torch.Generator(device=dev)
            g.manual_seed(int(s) + off)
            lst.append(g)
    return gens


def _draw_block(gens, dev):
    g1, g2, g3 = gens
    E = torch.stack([torch.randn(BLK, 1, K, generator=g, device=dev, dtype=F64)[:, 0] for g in g1], 1)
    Ua = torch.stack([torch.rand(BLK, 1, generator=g, device=dev, dtype=F64)[:, 0] for g in g1], 1)
    Rn = torch.stack([torch.rand(BLK, 1, K + 2, generator=g, device=dev, dtype=F64)[:, 0] for g in g2], 1)
    Rx = torch.stack([torch.rand(BLK, 1, 2, generator=g, device=dev, dtype=F64)[:, 0] for g in g3], 1)
    return E, Ua, Rn, Rx


def draw_streams(seed: int, n_blocks: int, dev: str = "cpu", consume_z0: bool = True) -> List[Dict[str, Any]]:
    """REQ-ENV-02: per-block draws {E[blk,K], Ua[blk], Rn[blk,K+2], Rx[blk,2]} from generators seeded s, s+1e6, s+2e6.
    By default z0 is drawn first from the first generator, exactly as frontier/the simulator do (REQ-ENV-02);
    consume_z0=False skips that draw (fresh-generator view, for diagnostics only)."""
    gens = _make_gens([seed], dev)
    if consume_z0:
        torch.randn(1, K, generator=gens[0][0], device=dev, dtype=F64)
    out = []
    for _ in range(n_blocks):
        E, Ua, Rn, Rx = _draw_block(gens, dev)
        out.append(dict(E=E[:, 0], Ua=Ua[:, 0], Rn=Rn[:, 0], Rx=Rx[:, 0]))
    return out


def score_window(T: int, L: int, replay: bool = False) -> int:
    """REQ-ENV-15: T_score = T - L (direction-2 mode) or T (direction-1 replay mode)."""
    return int(T) if replay else int(T - L)


def validate_T_train(T_train: int) -> int:
    """REQ-ENV-19: accept only {20000, 40000, 100000}; otherwise ValueError."""
    if T_train not in (20000, 40000, 100000):
        raise ValueError(f"T_train must be one of 20000, 40000, 100000, got {T_train}")
    return T_train


# ------------------------------------------------------------------------------------------------ policy bank
class _Bank:
    """Per-element policy dispatch.  act() receives only the obs tensor."""

    def __init__(self, policies: Sequence[Any], pid, dev: str, F_expected: int, obs_version: Optional[str] = None):
        """obs_version: the information set of the SimState this bank serves.  MLP policies must carry the same
        obs_version tag (F alone cannot tell announced from oracle, both F=27).  None = legacy callers: only F is
        checked and the tag must be consistent with it (F=25 -> public; F=27 -> announced|oracle)."""
        self.dev = dev
        P = len(policies)
        self.pid = pid
        self.is_adapter_p = [isinstance(p, AdapterPolicy) for p in policies]
        for p in policies:
            if not isinstance(p, (AdapterPolicy, MLPPolicy)):
                raise TypeError(f"unsupported policy type {type(p).__name__}")
        self.has_adapter = any(self.is_adapter_p)
        self.has_mlp = not all(self.is_adapter_p)
        mlps = [p for p in policies if isinstance(p, MLPPolicy)]
        for p in mlps:
            if p.F != F_expected:
                raise ValueError(f"policy has F={p.F} but the observation has {F_expected} features")
            ok = {obs_version} if obs_version is not None else ({"public"} if F_expected == 25 else {"announced", "oracle"})
            if p.obs_version is not None and p.obs_version not in ok:
                raise ValueError(f"policy was built for obs_version={p.obs_version!r} but the bank serves "
                                 f"{obs_version if obs_version is not None else sorted(ok)!r}")
            if not np.isfinite(p._theta).all():
                raise ValueError("policy theta contains NaN/inf")
        cols = {k: torch.tensor([float(getattr(p, "scols", {}).get(k, d)) if isinstance(p, AdapterPolicy) else d
                                 for p in policies], dtype=F64, device=dev) for k, d in SCOLS.items()}
        self.el = {k: v[pid] for k, v in cols.items()}
        for k in ("timing", "leg"):
            self.el[k] = self.el[k] > 0.5
        ad = [p.scols for p in policies if isinstance(p, AdapterPolicy)]
        anyc = lambda f: any(f(d) for d in ad)
        self.use_ad = anyc(lambda d: d.get("ad_m", 0) > 0)
        self.use_ue = anyc(lambda d: d.get("ue_mode", 0) > 0)
        self.use_edge = anyc(lambda d: d.get("em", 0) > 0)
        self.use_burst = anyc(lambda d: d.get("W", 0) > 0)
        self.mlp_mask = torch.tensor([not a for a in self.is_adapter_p], device=dev)[pid]
        if self.has_mlp:
            h = mlps[0].hidden
            for p in mlps:
                if p.hidden != h:
                    raise ValueError("all MLP policies of a run must share the hidden width")
            W1 = np.zeros((P, F_expected, h)); b1 = np.zeros((P, h)); W2 = np.zeros((P, h)); b2 = np.zeros((P, 1))
            for i, p in enumerate(policies):
                if isinstance(p, MLPPolicy):
                    W1[i], b1[i], W2[i], b2[i] = p.blocks()
            t = lambda a: torch.as_tensor(a, dtype=F64, device=dev)[pid]
            self.W1, self.b1, self.W2, self.b2 = t(W1), t(b1), t(W2), t(b2)

    def act(self, obs):
        return mlp_forward(obs, self.W1, self.b1, self.W2, self.b2)


class SimState:
    """REQ-ENV-12 stepwise state for B elements.  Attributes (tensors, batch-first): t (int rounds completed),
    M (=tau+1), z[B,K], u[B,K], v[B,K], susp[B,K], cus[B,K], pend_a[B,M] (long, -1=none), pend_v[B,M], flags[B,K], pen[B,K],
    audits[B], sxy[B], sxx[B], util_b[B,K], reg_b[B], nraid[B], mech, cfg, r.
    Convention: z,u,v are those of the last completed round; the next round's types are drawn inside step_state (or peeked,
    purely, by observe)."""

    def __init__(self, mech: str, cfg: Dict[str, float], r: float, seeds: Sequence[int], sidx=None, dev: str = "cpu",
                 obs_version: str = "public", track_obs: bool = True):
        if mech not in ("M3C", "M4"):
            raise ValueError(f"mech must be M3C or M4, got {mech}")
        seeds = [int(s) for s in seeds]
        self.mech, self.cfg, self.r, self.dev, self.obs_version = mech, dict(cfg), float(r), dev, obs_version
        self.seeds = seeds
        sidx = torch.arange(len(seeds)) if sidx is None else torch.as_tensor(sidx)
        self.sidx = sidx.long().to(dev)
        B = self.B = int(self.sidx.shape[0])
        self.t, self.M, self.tau = 0, 2, 1
        zero = lambda *s: torch.zeros(*s, dtype=F64, device=dev)
        pc = dict(PCOLS); pc.update({k: v for k, v in cfg.items() if k in PCOLS})
        full = lambda x: torch.full((B,), float(x), dtype=F64, device=dev)
        self.p_el, self.tol, self.L, self.eps, self.rw, self.kref, self.ad_h = (
            full(pc[k]) for k in ("p", "tol", "L", "eps", "rw", "kref", "ad_h"))
        self.rho = full(r)
        self.r_true = self.rho ** torch.ones(B, dtype=torch.long, device=dev).to(F64)
        self.sd_true = torch.sqrt((1 - self.r_true ** 2).clamp(min=1e-18))
        self.crit = self.kref * self.sd_true / self.r_true.clamp(min=1e-9)
        self.sq = torch.sqrt(1 - self.rho ** 2)[:, None]
        self.rho_c = self.rho[:, None]
        self.gens = _make_gens(seeds, dev)
        z0 = torch.cat([torch.randn(1, K, generator=g, device=dev, dtype=F64) for g in self.gens[0]], 0)
        self.z = z0[self.sidx]
        self.u = transform(torch.special.ndtr(self.z))
        self.v = self.u.clone()
        self.blk_idx, self.E, self.Ua, self.Rn, self.Rx = -1, None, None, None, None
        self.susp, self.cus = zero(B, K), zero(B, K)
        self.pend_a = -torch.ones(B, self.M, dtype=torch.long, device=dev)
        self.pend_v = zero(B, self.M)
        self.flags, self.pen, self.util_b = zero(B, K), zero(B, K), zero(B, K)
        self.audits, self.sxy, self.sxx, self.reg_b, self.nraid = zero(B), zero(B), zero(B), zero(B), zero(B)
        self.th = torch.full((B,), 0.5, dtype=F64, device=dev)
        self.ssh = zero(B)
        self.shv = -torch.ones(B, self.M, dtype=F64, device=dev)
        # observation bookkeeping
        self.track_obs = track_obs
        self.builder = _obs.ObsBuilder(obs_version, r, float(pc["L"]), (B,), device=dev)
        self.won_hist, self.d_hist = zero(B, _obs.HIST), zero(B, _obs.HIST)
        self.last_notice = torch.full((B,), -INF, dtype=F64, device=dev)
        self.susp_seen = zero(B)
        self.force_honest_after: Optional[int] = None
        self._bank = None
        self._feed = None            # graph stepping only (arbitration/rl/graphsim.py): per-step data supplied by the graph driver; None = eager

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "gens":
                gs = []
                for lst in v:
                    out = []
                    for g in lst:
                        g2 = torch.Generator(device=g.device)
                        g2.set_state(g.get_state())
                        out.append(g2)
                    gs.append(out)
                new.gens = tuple(gs)
            elif k == "_bank":
                new._bank = None
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new


def init_state(mech: str, cfg: Dict[str, float], r: float, seeds: Sequence[int], dev: str = "cpu",
               obs_version: str = "public") -> SimState:
    """REQ-ENV-12: fresh state (t=0)."""
    return SimState(mech, cfg, r, seeds, None, dev, obs_version)


def _ensure_block(st: SimState, tt: int) -> None:
    bi = tt // BLK
    if bi == st.blk_idx:
        return
    if bi != st.blk_idx + 1:
        raise RuntimeError("RNG blocks must be consumed sequentially")
    st.E, st.Ua, st.Rn, st.Rx = _draw_block(st.gens, st.dev)
    st.blk_idx = bi


def _row(st: SimState, name: str, s: int):
    """The RNG row of the current step for every batch element: block[name][s][sidx] (eager), or the equivalent row the graph driver put
    in the feed (same values, same layout: Ek[i][sidx] is the same indexing op on a K-row slice of the same block)."""
    fd = st._feed
    if fd is None:
        return getattr(st, name)[s][st.sidx]
    return getattr(fd, name)[fd.i][st.sidx]


def _tfull(like, t):
    """full_like(like, t) for a Python number t; for the device-side time of graph stepping (0-dim float64 tensor) the same values as an expanded view."""
    return t.expand_as(like) if torch.is_tensor(t) else torch.full_like(like, t)


def _next_types(st: SimState):
    if st._feed is None:
        _ensure_block(st, st.t)
    z = st.rho_c * st.z + st.sq * _row(st, "E", st.t % BLK)
    return z, transform(torch.special.ndtr(z))


def _obs_state(st: SimState, z, u, version: str):
    """Whitelisted obs input for the upcoming round st.t+1 with types z,u.  Returns (state_obj, last_notice_new)."""
    t = float(st.t + 1) if st._feed is None else st._feed.tt
    susp1 = st.susp[:, 1]
    changed = susp1 != st.susp_seen
    ln = torch.where(changed, _tfull(st.last_notice, t), st.last_notice)
    kw = dict(z=z[:, 1], u=u[:, 1], susp_minus_t=susp1 - t, since_rounds=t - ln, won_hist=st.won_hist,
              d_hist=st.d_hist, v_prev=st.v[:, 1])
    if version == "public":
        return _obs.PublicState(**kw), ln
    if version == "announced":
        h = float(st.cfg.get("tol", 1.0)) or 1.0
        al = (st.pend_a[:, (st.t + 1) % st.M] == 1).to(F64)
        return _obs.AnnouncedState(S=st.cus[:, 1] / h, audited_last=al, **kw), ln
    if version == "oracle":
        return _obs.OracleState(u0=u[:, 0], u2=u[:, 2], **kw), ln
    raise ValueError(version)


def observe(state: SimState, version: str = "public") -> Any:
    """REQ-OBS-01/03/04: observation vector [B,F] built from the state after round t-1 (version public|announced|oracle).
    Pure w.r.t. the simulation: peeks the upcoming round's types without advancing anything.
    NOTE (RNG): when the upcoming round opens a new 1024-block (t % 1024 == 0) this call draws that block from the
    generators now, rather than inside the next _step.  It is the same sequential draw _step would make (_ensure_block is
    idempotent per block), so results are bit-identical; it cannot be made lazier because the peek needs E[t].  The only
    visible effect is that a deepcopy/generator-state snapshot taken after observe() already contains that block."""
    z, u = _next_types(state)
    ps, _ = _obs_state(state, z, u, version)
    return state.builder.with_version(version).peek(ps)[0]


def _bank_for(st: SimState, policy: Any) -> _Bank:
    if isinstance(policy, _Bank):
        return policy
    # cache key = (object identity, weight bytes): in-place edits of policy._theta / scols invalidate the cache
    if isinstance(policy, MLPPolicy):
        wkey = policy._theta.tobytes()
    elif isinstance(policy, AdapterPolicy):
        wkey = tuple(sorted(policy.scols.items()))
    else:
        wkey = None
    if st._bank is not None and st._bank[0] is policy and st._bank[2] == wkey:
        return st._bank[1]
    F = 25 if st.obs_version == "public" else 27
    b = _Bank([policy], torch.zeros(st.B, dtype=torch.long, device=st.dev), st.dev, F, st.obs_version)
    st._bank = (policy, b, wkey)
    return b


@torch.no_grad()
def _step(st: SimState, bank: _Bank, rec: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    t = st.t + 1
    tt = t if st._feed is None else st._feed.tt          # time as a number in float64 arithmetic: Python int (eager) / device scalar (graph)
    s = (t - 1) % BLK
    dev, B, mech = st.dev, st.B, st.mech
    sidx, el = st.sidx, bank.el
    z, u = _next_types(st)
    # ---- observation (information-isolated) and policy action
    obs = None
    ln = None
    if st.track_obs:
        ps, ln = _obs_state(st, z, u, st.obs_version)
        if rec is not None:
            rec["susp1"].append(st.susp[:, 1].clone())
            rec["cus1"].append(st.cus[:, 1].clone())
            rec["audited_last"].append(st.pend_a[:, t % st.M] == 1)
            rec["u0"].append(u[:, 0].clone()); rec["u2"].append(u[:, 2].clone())
        obs = st.builder.step(ps)
        st.susp_seen = st.susp[:, 1].clone()
        st.last_notice = ln
    d = bank.act(obs) if bank.has_mlp else None
    if d is not None and st._feed is None and (t == 1 or t % BLK == 0) and not bool(torch.isfinite(d).all()):
        # isfinite costs a device sync, so it runs on t==1 and at each RNG block boundary only; NaN/inf cannot hide
        # because it persists in the EMA / d_hist state that feeds the next observation.
        raise FloatingPointError("policy action d is not finite (NaN/inf)")
    if d is not None and st._feed is not None and st._feed.last:
        st._feed.bad.logical_or_(~torch.isfinite(d).all())      # graph stepping: no sync inside the graph; the driver reads the flag once per block
    st.z, st.u = z, u
    susp, cus = st.susp, st.cus
    elig = tt > susp
    u1 = u[:, 1]
    zero = lambda *sh: torch.zeros(*sh, dtype=F64, device=dev)
    ar = torch.arange(B, device=dev)
    kk = torch.arange(K, device=dev)
    one = torch.ones(B, dtype=torch.bool, device=dev)
    Wd, Hd, emg, leg = el["W"], el["H"], el["em"], el["leg"]
    ue_mode, ue_band, ue_q, ue_dz = el["ue_mode"], el["ue_band"], el["ue_q"], el["ue_dz"]
    ad_m, ad_f, ad_c = el["ad_m"], el["ad_f"], el["ad_c"]
    ad_h, crit, ssh, th = st.ad_h, st.crit, st.ssh, st.th
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
            v1 = torch.where(bank.mlp_mask, report(u1, d), v1)
    else:
        v1 = report(u1, d)
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
    if st.force_honest_after is not None and t > st.force_honest_after:     # REQ-ENV-20 diagnostic variant only
        v = torch.cat([v[:, :1], u1[:, None], v[:, 2:]], 1)
    # ---- allocation
    vm = torch.where(elig, v, torch.full_like(v, -1.0))
    mx, idx = vm.max(1)
    win = mx >= 0
    st.pen = st.pen + ~elig
    umax = u.max(1).values
    uw = u.gather(1, idx[:, None])[:, 0]
    Rs = _row(st, "Rn", s)
    rd = torch.zeros(B, dtype=torch.bool, device=dev)
    flags, nraid = st.flags, st.nraid
    if mech == "M4":
        mask = elig & (kk[None, :] != idx[:, None])
        tg_o = torch.where(mask, Rs[:, 2:], torch.full_like(Rs[:, 2:], -1.0)).argmax(1)
        pick_w = win & (_row(st, "Rx", s)[:, 0] < st.rw)
        tg = torch.where(pick_w, idx, tg_o)
        rd = win & (Rs[:, 0] < st.eps) & (pick_w | mask.any(1))
        viol = (v - u).gather(1, tg[:, None])[:, 0] > 1e-9
        fl = rd & viol
        oht = torch.nn.functional.one_hot(tg, K).bool() & fl[:, None]
        susp = torch.where(oht, (tt + st.L)[:, None].expand(B, K), susp)
        flags = flags + oht
        nraid = nraid + rd
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
    resid = zero(B)
    has = torch.zeros(B, dtype=torch.bool, device=dev)
    if mech == "M3C":
        o = win_eff & (_row(st, "Ua", s) < st.p_el)
        st.audits = st.audits + o
        st.pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
        st.pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
    liar_flag = torch.zeros(B, dtype=torch.bool, device=dev)
    cs = t % M
    if mech == "M3C":
        ag = st.pend_a[:, cs]
        has = ag >= 0
        agc = ag.clamp(min=0)
        wv = u.gather(1, agc[:, None])[:, 0]
        pv = st.pend_v[:, cs]
        elig_ag = (tt > susp).gather(1, agc[:, None])[:, 0]
        ohag = torch.nn.functional.one_hot(agc, K).to(F64)
        zv = torch.special.ndtri(phi_of(pv, agc, C_MIN))
        zw = torch.special.ndtri(phi_of(wv, agc, C_MIN))
        rh = st.r_true
        sdh = torch.sqrt((1 - rh ** 2).clamp(min=1e-18))
        resid = (zw - rh * zv) / sdh
        st.sxy = st.sxy + torch.where(has, zv * zw, torch.zeros_like(zv))
        st.sxx = st.sxx + torch.where(has, zv * zv, torch.zeros_like(zv))
        upd = has & elig_ag
        Sg = cus.gather(1, agc[:, None])[:, 0]
        Sn = (Sg - resid - st.kref).clamp(min=0)
        el_ = upd & (Sn > st.tol)
        newS = torch.where(upd, torch.where(el_, torch.zeros_like(Sn), Sn), Sg)
        cus = torch.where(ohag.bool(), newS[:, None].expand(B, K), cus)
        ohe = ohag.bool() & el_[:, None]
        flags = flags + ohe
        susp = torch.where(ohe, (tt + st.L)[:, None].expand(B, K), susp)
        liar_flag = el_ & (agc == 1)
        st.pend_a[:, cs] = -1
    if bank.use_ad:
        sv = st.shv[:, cs]
        hv = sv >= 0
        ones = torch.ones_like(idx)
        zvs = torch.special.ndtri(phi_of(sv.clamp(min=0), ones, C_MIN))
        zws = torch.special.ndtri(phi_of(u1, ones, C_MIN))
        rs_ = (zws - st.r_true * zvs) / st.sd_true
        coin = _row(st, "Rx", s)[:, 1] < (ad_c * st.p_el).clamp(max=1.0)
        Sn = torch.where(hv & coin, (ssh - rs_ - st.kref).clamp(min=0), ssh)
        st.ssh = torch.where(liar_flag | (Sn > ad_h), torch.zeros_like(Sn), Sn)
        st.shv[:, cs] = -1
    st.susp, st.cus, st.flags, st.nraid, st.v, st.t = susp, cus, flags, nraid, v, t
    won1 = win_eff & (idx == 1)
    if st.track_obs:
        st.won_hist = torch.cat([won1.to(F64)[:, None], st.won_hist[:, :-1]], 1)
        st.d_hist = torch.cat([(v[:, 1] - u1)[:, None], st.d_hist[:, :-1]], 1)
    if rec is not None:
        rec["z1"].append(z[:, 1].clone()); rec["u1"].append(u1.clone()); rec["v1"].append(v[:, 1].clone())
        rec["won1"].append(won1)
        if obs is not None:
            rec["obs"].append(obs)
    return dict(v=v, idx=idx, win=win, rd=rd, win_eff=win_eff, won1=won1, elig=elig, resid=resid, has=has)


def step_state(state: SimState, policy: Any) -> Dict[str, Any]:
    """REQ-ENV-12: advance one round in place (round state.t+1) in the spec's event order.  Returns info dict with tensors
    v[B,K], idx[B], win[B], rd[B], win_eff[B], won1[B] (bool), elig[B,K] (eligibility used for allocation), resid[B], has[B]."""
    return _step(state, _bank_for(state, policy))


_FIELDS = ("reg_b", "util_b", "pen", "flags", "audits", "nraid", "susp")


def _consts(mech, cfg, r):
    c = dict(r=float(r), C_MIN=float(C_MIN), L=float(cfg["L"]))
    if mech == "M3C":
        c.update(p=float(cfg.get("p", 0.0)), h=float(cfg["tol"]), kref=float(cfg.get("kref", 0.5)))
    else:
        c.update(eps=float(cfg["eps"]), rw=float(cfg.get("rw", 1 / 3)))
    return c


def simulate(mech: str, T: int, cfg: Dict[str, float], policies: Sequence[Any], seeds: Sequence[int], *, r: float,
             T_score: Optional[int] = None, dev: str = "cuda", obs_version: str = "public",
             diagnostic: bool = False, diagnostic_variant: Optional[str] = None, chunk: Optional[int] = None,
             init_susp: Optional[Any] = None, graph: Any = None) -> Dict[str, Any]:
    """REQ-ENV-01/03/04/08/09/12: run T rounds for every (policy, seed).

    mech in {"M3C","M4"} (NAIVE = M3C with cfg["p"]=0).  cfg keys: p, tol, L, kref, eps, rw, ad_h (frontier names).
    policies: arbitration.rl.policy.AdapterPolicy / MLPPolicy objects.  r = rho (tau=1).
    Returns dict: "final" and "snap" (snapshot at T_score) each a dict of fields
    reg_b[P,S], util_b[P,S,K], pen[P,S,K], flags[P,S,K], audits[P,S], nraid[P,S], susp[P,S,K]; "T_score": int.
    T_score=None means the direction-2 window T - L (the snapshot at t=T-L; the last L rounds are not scored).
    graph: a graphsim.GraphSim -> CUDA-graph replays of the step loop (bitwise identical; only for plain training runs: MLP policies, no diagnostics).
    diagnostic=True adds "log" (public log, REQ-OBS-10 (a): z1,u1,v1,won1,susp1 each [P,S,T] + "consts"), "obs" [P,S,T,F]
    (b) and "private_log" (cus1, audited_last, u0, u2 at observation time, [P,S,T]).  diagnostic_variant="tail_honest"
    forces v1=u1 for t>T-2L (REQ-ENV-20; only the truncation-bias measurement may use it).  chunk = max elements per slice.
    """
    if diagnostic_variant not in (None, "tail_honest"):
        raise ValueError(diagnostic_variant)
    P, S = len(policies), len(seeds)
    L = float(cfg["L"])
    Ts = (score_window(T, int(L)) if T > L else int(T)) if T_score is None else int(T_score)   # T<=L: nothing to exclude
    if not 1 <= Ts <= T:
        raise ValueError("T_score must lie in [1, T]")
    F = 25 if obs_version == "public" else 27
    seeds_a = np.asarray([int(x) for x in seeds])
    N = P * S
    chunk = N if not chunk else max(1, int(chunk))
    susp0 = None
    if init_susp is not None:
        susp0 = torch.as_tensor(init_susp, dtype=F64).expand(P, S, K).reshape(N, K)
    need_obs = diagnostic or any(isinstance(p, MLPPolicy) for p in policies)
    parts = []
    for c0 in range(0, N, chunk):
        e = np.arange(c0, min(N, c0 + chunk))
        pid, sd = e // S, e % S
        uniq, inv = np.unique(seeds_a[sd], return_inverse=True)
        st = SimState(mech, cfg, r, uniq.tolist(), inv, dev, obs_version, track_obs=need_obs)
        if susp0 is not None:
            st.susp = susp0[e].to(dev).clone()
        if diagnostic_variant == "tail_honest":
            st.force_honest_after = int(T - 2 * L)
        bank = _Bank(policies, torch.as_tensor(pid, device=dev), dev, F, obs_version)
        rec = {k: [] for k in ("z1", "u1", "v1", "won1", "susp1", "obs", "cus1", "audited_last", "u0", "u2")} if diagnostic else None
        snap = None
        if (graph is not None and bank.has_mlp and not bank.has_adapter and rec is None and st.force_honest_after is None
                and st.track_obs):
            ident = ("env", mech, tuple(sorted(cfg.items())), float(r), obs_version, int(len(e)), int(len(uniq)), dev)
            snap, final = graph.run(ident, dict(st=st, bank=bank), lambda: _step(st, bank, None), T, Ts, _FIELDS)
        else:
            for t in range(1, T + 1):
                _step(st, bank, rec)
                if t == Ts:
                    snap = {f: getattr(st, f).clone() for f in _FIELDS}
            final = {f: getattr(st, f) for f in _FIELDS}
        parts.append((dict(final=final, snap=snap, rec=rec)))
    cat = lambda xs: torch.cat(xs, 0).reshape(P, S, *xs[0].shape[1:])
    out = dict(final={f: cat([p["final"][f] for p in parts]) for f in _FIELDS},
               snap={f: cat([p["snap"][f] for p in parts]) for f in _FIELDS}, T_score=Ts)
    if diagnostic:
        def stack(k):
            return cat([torch.stack(p["rec"][k], 1) for p in parts])
        out["log"] = {k: stack(k) for k in ("z1", "u1", "v1", "won1", "susp1")}
        out["log"]["consts"] = _consts(mech, cfg, r)
        out["obs"] = cat([torch.stack(p["rec"]["obs"], 1) for p in parts])
        out["private_log"] = {k: stack(k) for k in ("cus1", "audited_last", "u0", "u2")}
    return out


_PUBLIC_KEYS = ("z1", "u1", "v1", "won1", "susp1", "consts")


def extract_public_log(out: Dict[str, Any]) -> Dict[str, Any]:
    """REQ-OBS-10: the public log (a) of a diagnostic run: only z1,u1,v1,won1,susp1 and constants; nothing private."""
    lg = out["log"]
    return {k: lg[k] for k in _PUBLIC_KEYS}
