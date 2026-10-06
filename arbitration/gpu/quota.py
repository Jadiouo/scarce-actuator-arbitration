"""Quota (linking) mechanisms Q as a baseline against audits (docs: results/quota.json["meta"]["design"]).

Mechanisms (one simulator, batch columns = per-element parameters; batch order [param, r, strategy, seed]):
  OR     oracle allocation to the true argmax (regret 0 by construction)
  NAIVE  allocate to the highest report, no audit, no quota (unprotected reference)
  M0     round robin (ignores reports)
  M3C    dispatch-time CUSUM audit  -- audit block copied from frontier.simulate (M3C branch), + reg_a
  M4     surprise raid incl. winner (rw) -- copied from frontier.simulate (M4 branch), + reg_a
  QM     Jackson-Sonnenschein linking mechanism over time (JS 2007, long version, "Linking Mechanisms" pp.20-21,
         time version Cor. 2 pp.30-31): reports are quantile BINS (finite report space); within every block of
         QW periods an agent's histogram of reported bins must match the target (budget cap_b = floor((b+1)QW/nb)
         -floor(b QW/nb)); an over-budget report is rewritten to the nearest bin that still has budget.  The
         dispatcher allocates to the highest bin-representative utility F_i^-1(mid of reported bin).
  QT     transition-quota (Escobar-Toikka 2013 spirit, credible reporting mechanism Sec. 5, eq. 5.1): for each
         class c of the agent's PREVIOUS reported bin (nc classes) the empirical frequency of the next reported
         bin must stay within a sqrt(n)-type band of the true AR(1)-copula transition probability P(b'|c).
         Over-band report -> rewritten to the nearest allowed bin.  Differences to E-T: see meta.design.
All strategies of frontier.py (b, dz, dzc, burst, edge oracle em, edge uninformed ue_*, adaptive ad_*) are copied;
the liar is agent 1; for Q the liar's *desired value* is mapped to a bin and then rewritten like any report.
Quota-specific liar strategies: qs_mode=1 'save-min' (oracle on others' submitted bins) and qs_mode=2 (uninformed).
Same RNG streams as frontier/audit_fix  =>  identical type paths per seed (paired comparisons).
"""
import math
import numpy as np
import torch
from .adaaudit import K, C_MIN, type_cdf_tables, transform  # noqa: F401
from .audit_fix import phi_of

F64 = torch.float64
INF = float("inf")
MECHS = ("OR", "NAIVE", "M0", "M3C", "M4", "QM", "QT", "QE")
PCOLS = dict(p=0.1, tol=0.0, L=INF, eps=0.0, rw=1 / 3, kref=0.5, ad_h=INF, QW=0.0, nb=10.0, nc=1.0, qh=1.0, qa=0.0)
SCOLS = dict(b=0.0, dz=0.0, dzc=0.0, W=0.0, H=0.0, em=0.0, leg=0.0, ue_mode=0.0, ue_band=0.0, ue_q=0.5, ue_dz=0.0,
             ad_m=0.0, ad_f=0.0, qs_mode=0.0, qs_lam=0.0, qs_x=0.0)
RCOLS = dict(rho=0.9, tau=1.0, ra=-1.0, online=0.0)


def build_pairs(pairs, S, seeds, dev="cuda"):
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
    for k in ("online", "leg"):
        el[k] = el[k] > 0.5
    el["shape"] = shp
    # transition tables (QT): per element [ncmax, nbmax]
    el["_pairs"] = pairs
    return el


def win_x(pi):
    """smallest x with P(others' max <= x) = F0(x)F2(x) >= pi  (stationary, known marginals)."""
    lo, hi = C_MIN, 1.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if ((m - C_MIN) / (1 - C_MIN)) * m * m >= pi:
            hi = m
        else:
            lo = m
    return hi


def trans_table(rho, nb, nc, nq=4000, dev="cuda"):
    """P[c, b'] = Pr(next bin = b' | prev quantile in class c) for the Gaussian-copula AR(1), equiprobable bins."""
    nbm = int(nb)
    out = torch.zeros(int(nc), nbm, dtype=F64, device=dev)
    edges = torch.special.ndtri(torch.linspace(0, 1, nbm + 1, dtype=F64, device=dev).clamp(1e-300, 1 - 1e-16))
    edges[0], edges[-1] = -1e9, 1e9
    s = math.sqrt(max(1 - rho * rho, 1e-18))
    for c in range(int(nc)):
        q = (c + (torch.arange(nq, dtype=F64, device=dev) + 0.5) / nq) / nc
        z = torch.special.ndtri(q)
        cdf = torch.special.ndtr((edges[None, :] - rho * z[:, None]) / s)
        out[c] = (cdf[:, 1:] - cdf[:, :-1]).mean(0)
    return out / out.sum(1, keepdim=True)


def _F(k, x):
    return torch.where(k == 0, (x - C_MIN) / (1 - C_MIN), torch.where(k == 1, x, x * x)).clamp(0, 1)


def _rep(k, q):
    return torch.where(k == 0, C_MIN + (1 - C_MIN) * q, torch.where(k == 1, q, torch.sqrt(q)))


@torch.no_grad()
def simulate(mech, T, el, seeds, c=C_MIN, blk=1024):
    dev = el["rho"].device
    B = el["rho"].shape[0]
    rho, tau, sidx = el["rho"], el["tau"], el["sid"]
    bb, dzs, dzc, Wd, Hd, emg, leg = el["b"], el["dz"], el["dzc"], el["W"], el["H"], el["em"], el["leg"]
    ue_mode, ue_band, ue_q, ue_dz = el["ue_mode"], el["ue_band"], el["ue_q"], el["ue_dz"]
    ad_m, ad_f, ad_h = el["ad_m"], el["ad_f"], el["ad_h"]
    qs_mode, qs_lam, qs_x = el["qs_mode"], el["qs_lam"], el["qs_x"]
    p_el, tol, L, eps, rw, kref = el["p"], el["tol"], el["L"], el["eps"], el["rw"], el["kref"]
    rho_a, onl = el["ra"], el["online"]
    r_true = rho ** tau.to(F64)
    r_as = torch.where(rho_a >= 0, rho_a, r_true)
    r0_prior = r_as.clone()
    sd_true = torch.sqrt((1 - r_true ** 2).clamp(min=1e-18))
    crit = kref * sd_true / r_true.clamp(min=1e-9)
    use_ad, use_ue = bool((ad_m > 0).any()), bool((ue_mode > 0).any())
    use_edge, use_burst = bool((emg > 0).any()), bool((Wd > 0).any())
    Q = mech in ("QM", "QT", "QE")
    audit_on = mech == "M3C"
    raid = mech == "M4"
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
    susp = zero(B, K); cus = zero(B, K)
    pend_a = -torch.ones(B, M, dtype=torch.long, device=dev)
    pend_v = zero(B, M)
    shv = -torch.ones(B, M, dtype=F64, device=dev)
    th = torch.full((B,), 0.5, dtype=F64, device=dev)
    ssh = zero(B)
    reg_a, reg_b, audits, nraid = zero(B), zero(B), zero(B), zero(B)
    util_a, util_b, flags, pen, nrew = zero(B, K), zero(B, K), zero(B, K), zero(B, K), zero(B, K)
    sxy, sxx = zero(B), zero(B)
    eta = 0.02
    one = torch.ones(B, dtype=torch.bool, device=dev)

    if Q:
        nb = el["nb"].long()
        nbm = int(nb.max())
        QW = el["QW"]
        binar = torch.arange(nbm, device=dev)
        valid = (binar[None, :] < nb[:, None])                                   # [B,nbm]
        validK = valid[:, None, :].expand(B, K, nbm)
        cap0 = torch.where(QW[:, None] > 0,
                           torch.floor((binar[None, :] + 1.0) * QW[:, None] / nb[:, None]) - torch.floor(binar[None, :] * QW[:, None] / nb[:, None]),
                           torch.full((B, nbm), 1e9, dtype=F64, device=dev))
        cap0 = torch.where(valid, cap0, torch.zeros_like(cap0))
        capK0 = cap0[:, None, :].expand(B, K, nbm)
        cap = capK0.clone()
        # representative utility of bin b for agent k : [B,K,nbm]
        qmid = (binar[None, None, :] + 0.5) / nb[:, None, None]
        repT = _rep(kk[None, :, None], qmid.expand(B, K, nbm).clone())
        distb = (binar[None, :] - binar[:, None]).abs().to(F64)       # [nbm(desired), nbm(candidate)]
        tieb = (binar[None, :] > binar[:, None]).to(F64) * 1e-3                   # prefer lower on ties
        if mech in ("QT", "QE"):
            ncm = int(el["nc"].max())
            nc = el["nc"].long()
            uniq = {}
            Pel = torch.zeros(B, ncm, nbm, dtype=F64, device=dev)
            for key in sorted(set(zip(rho.tolist(), nb.tolist(), nc.tolist()))):
                tt = trans_table(key[0], key[1], key[2], dev=dev)
                m = (rho == key[0]) & (nb == key[1]) & (nc == key[2])
                Pel[m, :tt.shape[0], :tt.shape[1]] = tt
            Ncnt = zero(B, K, ncm, nbm)
            nn_ = zero(B, K, ncm)
            qh, qa = el["qh"], el["qa"]
            qbin0 = (_F(kk[None, :], transform(torch.special.ndtr(z))) * nb[:, None]).floor().long()
            prev = torch.minimum(qbin0, (nb - 1)[:, None])
            failed = torch.zeros(B, K, dtype=torch.bool, device=dev)
            gens4 = []
            for sd_ in seeds:
                g4 = torch.Generator(device=dev); g4.manual_seed(int(sd_) + 3 * 10 ** 6); gens4.append(g4)

    def nearest(d, av):
        """d [B,K] desired bin; av [B,K,nbm] availability -> chosen bin [B,K]."""
        cost = distb[d] + tieb[d] + (~av).to(F64) * 1e6               # [B,K,nbm]
        return cost.argmin(2)

    for t in range(1, T + 1):
        s = (t - 1) % blk
        if s == 0:
            E = torch.stack([torch.randn(blk, 1, K, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Ua = torch.stack([torch.rand(blk, 1, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Rn = torch.stack([torch.rand(blk, 1, K + 2, generator=g, device=dev, dtype=F64)[:, 0] for g in gens2], 1)
            Rx = torch.stack([torch.rand(blk, 1, 2, generator=g, device=dev, dtype=F64)[:, 0] for g in gens3], 1)
            if mech == "QE":
                Rq = torch.stack([torch.rand(blk, 1, K, generator=g, device=dev, dtype=F64)[:, 0] for g in gens4], 1)
        z = rho_c * z + sq * E[s][sidx]
        u = transform(torch.special.ndtr(z))
        elig = t > susp
        u1 = u[:, 1]
        # ---- liar's desired value (agent 1): identical to frontier.simulate
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
        rd = torch.zeros(B, dtype=torch.bool, device=dev)
        if mech == "OR":
            idx = u.argmax(1); win = one
        elif mech == "M0":
            idx = torch.full((B,), (t - 1) % K, dtype=torch.long, device=dev); win = one
        elif Q:
            if mech == "QM":
                av = (cap > 0.5) & validK
            elif mech == "QE":
                av = validK
                cls = torch.minimum((prev * nc[:, None] // nb[:, None]), (nc - 1)[:, None])
                Pc = Pel[ar[:, None], cls]
            else:
                cls = torch.minimum((prev * nc[:, None] // nb[:, None]), (nc - 1)[:, None])           # [B,K]
                Pc = Pel[ar[:, None], cls]                                                              # [B,K,nbm]
                Nc_ = Ncnt[ar[:, None], kk[None, :], cls]                                               # [B,K,nbm]
                n1 = nn_[ar[:, None], kk[None, :], cls][:, :, None] + 1.0
                margin = Pc * n1 + qh[:, None, None] * torch.sqrt(n1 * Pc * (1 - Pc)) + qa[:, None, None] - (Nc_ + 1.0)
                margin = torch.where(validK, margin, torch.full_like(margin, -1e9))
                av = (margin >= 0) & validK
                none = ~av.any(2, keepdim=True)
                av = av | (none & (margin == margin.max(2, keepdim=True).values) & validK)
            d_h = torch.minimum((_F(kk[None, :], u) * nb[:, None]).floor().long(), (nb - 1)[:, None])
            d = d_h.clone()
            v1_bin = torch.minimum((v[:, 1].clamp(0, 1) * nb).floor().long(), nb - 1)
            d[:, 1] = v1_bin
            ch = nearest(d, av)
            rep = repT.gather(2, ch[:, :, None])[:, :, 0]                                              # [B,K]
            mo_rep = torch.maximum(rep[:, 0], rep[:, 2])
            av1 = av[:, 1, :]
            rep1 = repT[:, 1, :]
            # edge oracle under Q : smallest available bin whose representative beats the others' submitted reps
            sp = av1 & (rep1 > mo_rep[:, None])
            has_sp = sp.any(1)
            first_sp = sp.to(F64).argmax(1)
            low1 = av1.to(F64).argmax(1)
            top1 = (av1.to(F64) * (binar[None, :] + 1.0)).argmax(1)
            ch1 = ch[:, 1]
            if use_edge:
                oth_t = torch.maximum(u[:, 0], u[:, 2])
                e_q = (emg > 0) & ((oth_t - u1) > 0) & ((oth_t - u1) < emg)
                ch1 = torch.where(e_q, torch.where(has_sp, first_sp, top1), ch1)
            if bool((qs_mode > 0).any()):
                big = u1 >= qs_lam
                sp2 = av1 & (rep1 >= qs_x[:, None])
                has2 = sp2.any(1)
                fst2 = sp2.to(F64).argmax(1)
                c_or = torch.where(big & has_sp, first_sp, low1)
                c_un = torch.where(big, torch.where(has2, fst2, top1), low1)
                ch1 = torch.where(qs_mode == 1, c_or, torch.where(qs_mode == 2, c_un, ch1))
            ch = torch.cat([ch[:, :1], ch1[:, None], ch[:, 2:]], 1)
            if mech == "QE":
                own = ch                                                                                # agent's own message
                cum = Pc.cumsum(2)
                samp = (cum < Rq[s][sidx][:, :, None]).sum(2).clamp(max=nbm - 1)
                samp = torch.minimum(samp, (nb - 1)[:, None])
                ch = torch.where(failed, samp, own)
            nrew += (ch != torch.cat([d_h[:, :1], torch.minimum(d_h[:, 1:2], nb[:, None] - 1), d_h[:, 2:]], 1)).to(F64)
            rep = repT.gather(2, ch[:, :, None])[:, :, 0]
            idx = rep.argmax(1); win = one
            ohc = torch.nn.functional.one_hot(ch, nbm).to(F64)
            if mech == "QM":
                cap = cap - ohc
                rst = ((t % QW.clamp(min=1)) == 0) & (QW > 0)                                          # block ends after this round
                cap = torch.where(rst[:, None, None], capK0, cap)
            elif mech == "QE":
                ohc_own = torch.nn.functional.one_hot(own, nbm).to(F64)
                ohp = torch.nn.functional.one_hot(cls, ncm).to(F64)
                live = (~failed).to(F64)[:, :, None, None]
                Ncnt = Ncnt + live * ohp[:, :, :, None] * ohc_own[:, :, None, :]
                nn_ = nn_ + (~failed).to(F64)[:, :, None] * ohp
                Nc2 = Ncnt[ar[:, None], kk[None, :], cls]
                n2 = nn_[ar[:, None], kk[None, :], cls][:, :, None]
                bad = (Nc2 - Pc * n2).abs() > qh[:, None, None] * torch.sqrt(n2 * Pc * (1 - Pc)) + qa[:, None, None]
                failed = failed | (bad & validK).any(2)
                prev = ch
                rst = ((t % el["QW"].clamp(min=1)) == 0) & (el["QW"] > 0)
                Ncnt = torch.where(rst[:, None, None, None], torch.zeros_like(Ncnt), Ncnt)
                nn_ = torch.where(rst[:, None, None], torch.zeros_like(nn_), nn_)
                failed = failed & ~rst[:, None]
            else:
                ohp = torch.nn.functional.one_hot(cls, ncm).to(F64)                                    # [B,K,ncm]
                Ncnt = Ncnt + ohp[:, :, :, None] * ohc[:, :, None, :]
                nn_ = nn_ + ohp
                prev = ch
                rst = (((t) % el["QW"].clamp(min=1)) == 0) & (el["QW"] > 0)
                Ncnt = torch.where(rst[:, None, None, None], torch.zeros_like(Ncnt), Ncnt)
                nn_ = torch.where(rst[:, None, None], torch.zeros_like(nn_), nn_)
        else:   # NAIVE / M3C / M4
            vm = torch.where(elig, v, torch.full_like(v, -1.0))
            mx, idx = vm.max(1)
            win = mx >= 0
        pen += ~elig
        umax = u.max(1).values
        uw = u.gather(1, idx[:, None])[:, 0]
        Rs = Rn[s][sidx]
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
        reg_a += umax - torch.where(win_eff, uw, torch.zeros_like(uw))
        util_a += torch.nn.functional.one_hot(idx, K).to(F64) * (uw * win_eff)[:, None]
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
            Sg = cus.gather(1, agc[:, None])[:, 0]
            Sn = (Sg - resid - kref).clamp(min=0)
            el_ = upd & (Sn > tol)
            newS = torch.where(upd, torch.where(el_, torch.zeros_like(Sn), Sn), Sg)
            cus = torch.where(ohag.bool(), newS[:, None].expand(B, K), cus)
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
            coin = Rx[s][sidx][:, 1] < p_el
            Sn = torch.where(hv & coin, (ssh - rs_ - kref).clamp(min=0), ssh)
            ssh = torch.where(liar_flag | (Sn > ad_h), torch.zeros_like(Sn), Sn)
            shv[:, cs] = -1
    return dict(reg_a=reg_a, reg_b=reg_b, util_a=util_a, util_b=util_b, flags=flags, pen=pen, audits=audits,
                nraid=nraid, nrew=nrew, susp=susp)
