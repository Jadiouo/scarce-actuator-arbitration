"""Stage 2b: lying-gain vs honest-false-punish frontier simulator (see docs/stage2b_design.md).
Imports adaaudit / audit_fix helpers; modifies neither.  Mechanisms: M0, M3T (single threshold), M3C (CUSUM),
M3G (sequential cumulative-z / GLR-type), M3W (fixed-window mean), M4 (surprise raid incl. winner, parameter rw).
All elements are columns of per-element parameters, batch order [param, r, strategy, seed]."""
import math
import numpy as np
import torch
from .adaaudit import K, C_MIN, type_cdf_tables, transform  # noqa: F401
from .audit_fix import phi_of

F64 = torch.float64
INF = float("inf")
MECHS = ("M0", "M3T", "M3C", "M3G", "M3W", "M4")
PCOLS = dict(p=0.0, tol=0.0, L=INF, eps=0.0, rw=1 / 3, kref=0.5, nmax=30.0, ad_h=INF)
SCOLS = dict(b=0.0, dz=0.0, dzc=0.0, W=0.0, H=0.0, em=0.0, leg=0.0, ue_mode=0.0, ue_band=0.0, ue_q=0.5, ue_dz=0.0,
             ad_m=0.0, ad_f=0.0, ad_c=1.0, timing=0.0)
RCOLS = dict(rho=0.9, tau=1.0, ra=-1.0, online=0.0)


def build_pairs(pairs, S, seeds, dev="cuda"):
    """pairs: list of (mech-param dict, r-dict). Batch order [pair, strategy, seed]."""
    n = len(seeds)
    shp = (len(pairs), len(S), n)
    def mk(lst, cols, ax):
        out = {}
        for k, d in cols.items():
            a = np.array([x.get(k, d) for x in lst], dtype=np.float64)
            sh = [1, 1, 1]; sh[ax] = len(lst)
            out[k] = torch.tensor(np.broadcast_to(a.reshape(sh), shp).reshape(-1).copy(), dtype=F64, device=dev)
        return out
    el = {}
    el.update(mk([p for p, _ in pairs], PCOLS, 0)); el.update(mk([r for _, r in pairs], RCOLS, 0))
    el.update(mk(S, SCOLS, 1))
    sid = np.broadcast_to(np.arange(n).reshape(1, 1, n), shp).reshape(-1).copy()
    el["sid"] = torch.tensor(sid, device=dev).long()
    el["tau"] = el["tau"].long()
    for k in ("timing", "online", "leg"):
        el[k] = el[k] > 0.5
    el["shape"] = shp
    return el


def build(P, R, S, seeds, dev="cuda"):
    return build_pairs([(p, r) for p in P for r in R], S, seeds, dev)


def rho_for(r, tau):
    return r ** (1.0 / tau)


@torch.no_grad()
def simulate(mech, T, el, seeds, c=C_MIN, blk=1024):
    dev = el["rho"].device
    B = el["rho"].shape[0]
    rho, tau, sidx = el["rho"], el["tau"], el["sid"]
    bb, dzs, dzc, Wd, Hd, emg, leg = el["b"], el["dz"], el["dzc"], el["W"], el["H"], el["em"], el["leg"]
    ue_mode, ue_band, ue_q, ue_dz = el["ue_mode"], el["ue_band"], el["ue_q"], el["ue_dz"]
    ad_m, ad_f, ad_h, ad_c = el["ad_m"], el["ad_f"], el["ad_h"], el["ad_c"]
    p_el, tol, L, eps, rw, kref, nmax = el["p"], el["tol"], el["L"], el["eps"], el["rw"], el["kref"], el["nmax"]
    rho_a, onl = el["ra"], el["online"]
    r_true = rho ** tau.to(F64)
    r_as = torch.where(rho_a >= 0, rho_a, r_true)
    r0_prior = r_as.clone()
    sd_true = torch.sqrt((1 - r_true ** 2).clamp(min=1e-18))
    crit = kref * sd_true / r_true.clamp(min=1e-9)
    use_ad, use_ue = bool((ad_m > 0).any()), bool((ue_mode > 0).any())
    use_edge = bool((emg > 0).any())
    use_burst = bool((Wd > 0).any())
    M3 = mech in ("M3T", "M3C", "M3G", "M3W")
    raid = mech == "M4"
    audit_on = M3
    M = int(tau.max()) + 1
    ar = torch.arange(B, device=dev)
    kk = torch.arange(K, device=dev)
    sq = torch.sqrt(1 - rho ** 2)[:, None]
    rho_c = rho[:, None]
    gens, gens2, gens3 = [], [], []
    for s in seeds:
        for lst, off in ((gens, 0), (gens2, 10 ** 6), (gens3, 2 * 10 ** 6)):
            g = torch.Generator(device=dev); g.manual_seed(int(s) + off); lst.append(g)
    z0 = torch.cat([torch.randn(1, K, generator=g, device=dev, dtype=F64) for g in gens], 0)
    z = z0[sidx]
    zero = lambda *s: torch.zeros(*s, dtype=F64, device=dev)
    susp = zero(B, K); cus = zero(B, K); gs = zero(B, K); gn = zero(B, K)
    pend_a = -torch.ones(B, M, dtype=torch.long, device=dev)
    pend_v = zero(B, M)
    shv = -torch.ones(B, M, dtype=F64, device=dev)
    th = torch.full((B,), 0.5, dtype=F64, device=dev)
    ssh = zero(B)
    reg_b, audits, nraid = zero(B), zero(B), zero(B)
    util_b, flags, pen = zero(B, K), zero(B, K), zero(B, K)
    sxy, sxx = zero(B), zero(B)
    eta = 0.02
    one = torch.ones(B, dtype=torch.bool, device=dev)

    for t in range(1, T + 1):
        s = (t - 1) % blk
        if s == 0:
            E = torch.stack([torch.randn(blk, 1, K, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Ua = torch.stack([torch.rand(blk, 1, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Rn = torch.stack([torch.rand(blk, 1, K + 2, generator=g, device=dev, dtype=F64)[:, 0] for g in gens2], 1)
            Rx = torch.stack([torch.rand(blk, 1, 2, generator=g, device=dev, dtype=F64)[:, 0] for g in gens3], 1)
        z = rho_c * z + sq * E[s][sidx]
        u = transform(torch.special.ndtr(z))
        elig = t > susp
        u1 = u[:, 1]
        # ---- liar's report (agent 1)
        dzl = dzs + dzc * crit
        if use_burst:
            lie_on = torch.where(Wd > 0, ((t - 1) % (Wd + Hd).clamp(min=1)) < Wd, one)
            dzl = dzl * lie_on
        if use_ad:
            dzl = torch.where(ad_m > 0, torch.where(ssh < ad_f * ad_h, ad_m * crit, torch.zeros_like(crit)), dzl)
        v1 = torch.minimum(torch.special.ndtr(z[:, 1] + dzl) + bb, torch.ones_like(u1))
        v = torch.cat([u[:, :1], v1[:, None], u[:, 2:]], 1)
        if use_edge:
            oth = torch.where(leg[:, None] | elig, u, torch.full_like(u, -1.0))
            mo = torch.maximum(oth[:, 0], oth[:, 2])
            gap = mo - u1
            e_on = (emg > 0) & (gap > 0) & (gap < emg) & (mo >= 0)
            v1e = torch.where(e_on, torch.clamp(mo + 1e-3, max=1.0), torch.where(emg > 0, u1, v[:, 1]))
            v = torch.cat([v[:, :1], v1e[:, None], v[:, 2:]], 1)
        if use_ue:
            ue_on = (ue_mode > 0) & (u1 < th) & (u1 >= th - ue_band)
            vu = torch.where(ue_on & (ue_mode == 1), torch.clamp(th, max=1.0),
                             torch.where(ue_on, torch.minimum(torch.special.ndtr(z[:, 1] + ue_dz), torch.ones_like(u1)), u1))
            v1u = torch.where(ue_mode > 0, vu, v[:, 1])
            v = torch.cat([v[:, :1], v1u[:, None], v[:, 2:]], 1)
        # ---- allocation
        if mech == "M0":
            idx = torch.full((B,), (t - 1) % K, dtype=torch.long, device=dev)
            win = one
        else:
            vm = torch.where(elig, v, torch.full_like(v, -1.0))
            mx, idx = vm.max(1)
            win = mx >= 0
        pen += ~elig
        umax = u.max(1).values
        uw = u.gather(1, idx[:, None])[:, 0]
        Rs = Rn[s][sidx]
        rd = torch.zeros(B, dtype=torch.bool, device=dev)
        if raid:
            mask = elig & (kk[None, :] != idx[:, None])
            tg_o = torch.where(mask, Rs[:, 2:], torch.full_like(Rs[:, 2:], -1.0)).argmax(1)
            pick_w = win & (Rx[s][sidx][:, 0] < rw)
            tg = torch.where(pick_w, idx, tg_o)
            rd = win & (Rs[:, 0] < eps) & (pick_w | mask.any(1))
            viol = (v - u).gather(1, tg[:, None])[:, 0] > 1e-9
            fl = rd & viol
            oht = torch.nn.functional.one_hot(tg, K).bool() & fl[:, None]
            susp = torch.where(oht, (t + L)[:, None].expand(B, K), susp)
            flags += oht
            nraid += rd
        win_eff = win & ~rd
        ridx = (Rs[:, 1] * K).long().clamp(max=K - 1)
        urand = u.gather(1, ridx[:, None])[:, 0]
        ag_b = torch.where(win_eff, idx, ridx)
        uw_b = torch.where(win_eff, uw, urand) * (~rd)
        reg_b += umax - uw_b
        util_b += torch.nn.functional.one_hot(ag_b, K).to(F64) * uw_b[:, None]
        if use_ue:
            probe = ue_on & elig[:, 1] & win
            th = th + eta * probe * ((idx != 1).to(F64) - (1 - ue_q))
        slot = (t + tau) % M
        if use_ad:
            shv[ar, slot] = torch.where(win_eff & (idx == 1), v[:, 1], -torch.ones_like(u1))
        if audit_on:
            o = win_eff & (Ua[s][sidx] < p_el)
            audits += o
            pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
            pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
        cs = t % M
        liar_flag = torch.zeros(B, dtype=torch.bool, device=dev)
        if audit_on:
            ag = pend_a[:, cs]
            has = ag >= 0
            agc = ag.clamp(min=0)
            wv = u.gather(1, agc[:, None])[:, 0]
            pv = pend_v[:, cs]
            elig_ag = (t > susp).gather(1, agc[:, None])[:, 0]
            ohag = torch.nn.functional.one_hot(agc, K).to(F64)
            zv = torch.special.ndtri(phi_of(pv, agc, c))
            zw = torch.special.ndtri(phi_of(wv, agc, c))
            rh = torch.where(onl, ((sxy + r0_prior * 30.0) / (sxx + 30.0)).clamp(0, 0.995), r_as)
            sdh = torch.sqrt((1 - rh ** 2).clamp(min=1e-18))
            resid = (zw - rh * zv) / sdh
            sxy = sxy + torch.where(has, zv * zw, torch.zeros_like(zv)); sxx = sxx + torch.where(has, zv * zv, torch.zeros_like(zv))
            upd = has & elig_ag
            if mech == "M3T":
                el_ = upd & (resid < -tol)
            elif mech == "M3C":
                Sg = cus.gather(1, agc[:, None])[:, 0]
                Sn = (Sg - resid - kref).clamp(min=0)
                el_ = upd & (Sn > tol)
                newS = torch.where(upd, torch.where(el_, torch.zeros_like(Sn), Sn), Sg)
                cus = torch.where(ohag.bool(), newS[:, None].expand(B, K), cus)
            else:   # M3G (sequential) / M3W (fixed window)
                g0 = gs.gather(1, agc[:, None])[:, 0]; n0 = gn.gather(1, agc[:, None])[:, 0]
                g1 = g0 - torch.where(upd, resid, torch.zeros_like(resid)); n1 = n0 + upd.to(F64)
                Z = g1 / n1.clamp(min=1).sqrt()
                if mech == "M3G":
                    el_ = upd & (n1 >= 3) & (Z > tol)
                else:
                    el_ = upd & (n1 >= nmax) & (Z > tol)
                rst = el_ | (upd & (n1 >= nmax))
                g1 = torch.where(rst, torch.zeros_like(g1), g1); n1 = torch.where(rst, torch.zeros_like(n1), n1)
                gs = torch.where(ohag.bool(), g1[:, None].expand(B, K), gs)
                gn = torch.where(ohag.bool(), n1[:, None].expand(B, K), gn)
            ohe = ohag.bool() & el_[:, None]
            flags += ohe
            susp = torch.where(ohe, (t + L)[:, None].expand(B, K), susp)
            liar_flag = el_ & (agc == 1)
            pend_a[:, cs] = -1
        if use_ad:
            sv = shv[:, cs]
            hv = sv >= 0
            zvs = torch.special.ndtri(phi_of(sv.clamp(min=0), torch.ones_like(idx), c))
            zws = torch.special.ndtri(phi_of(u1, torch.ones_like(idx), c))
            rs_ = (zws - r_true * zvs) / sd_true
            coin = Rx[s][sidx][:, 1] < (ad_c * p_el).clamp(max=1.0)
            Sn = torch.where(hv & coin, (ssh - rs_ - kref).clamp(min=0), ssh)
            ssh = torch.where(liar_flag | (Sn > ad_h), torch.zeros_like(Sn), Sn)
            shv[:, cs] = -1
    return dict(reg_b=reg_b, util_b=util_b, flags=flags, pen=pen, audits=audits, nraid=nraid, susp=susp)
