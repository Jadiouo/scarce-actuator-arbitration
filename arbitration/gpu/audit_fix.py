"""Stage-2: audit repairs under delayed verification (imports adaaudit.py, never modifies it).

One GPU simulator for mechanisms
  M0  round-robin, ignores reports            M1f fixed-p, |v-w|>delta, permanent exclusion  (== simulate_tol 'fixed')
  M1a AdaAudit, v-w>delta, permanent exclusion (== simulate_tol 'ada')
  M2  arrival-time audit (compares the UPDATED report v' with w), suspension L
  M3a/M3b/M3c dispatch-time audit with drift-corrected z-score residual; punishment = suspension / credit score / CUSUM
  M4  surprise raids (instant tau=0 check of a non-allocated agent, wastes the item)    M5 = M3a + M4

Shared with adaaudit.py: K, C_MIN, type_cdf_tables, transform (imports).  The AdaAudit q-hat estimation block in
`simulate` is a line-by-line copy of adaaudit_tol.simulate_tol (needed to use per-seed common random numbers); the
bit-identity test `check_bitwise_vs_tol` fails if adaaudit(_tol).py changes -> re-sync then.

Strategies: v = min(1, T(Phi(z+dz)) + b) for the liar (agent 1); dz=0 is the plain "u+b" family (clamped at 1, which creates
a detectable point mass at v=1), dz>0 shifts in z (copula) space and never leaves the support.
Randomness: one torch.Generator per seed (seed value s), stream order identical to adaaudit_tol for B=1; so every
(mechanism, parameter, rho, strategy) with the same seed sees the same type paths  (paired comparisons).
Regret/utility are scored under two fallbacks when nobody is eligible (or report < c in M1a):
  (a) item unallocated (utility 0; adaaudit convention)   (b) item given to a uniformly random agent.
Raided (wasted) rounds are 0 under both.   All float64 on CUDA.
"""
import math
import torch
from .adaaudit import K, C_MIN, type_cdf_tables, transform  # noqa: F401

F64 = torch.float64
INF = float("inf")
MECHS = ("M0", "M1f", "M1a", "M2", "M3a", "M3b", "M3c", "M4", "M5")


def phi_of(x, agent, c=C_MIN):
    """marginal cdf F_i(x) of agent i's utility (inverse of adaaudit.transform), clipped away from 0/1."""
    f = torch.where(agent == 0, (x - c) / (1 - c), torch.where(agent == 1, x, x * x))
    return f.clamp(1e-12, 1 - 1e-12)


def make_elements(params, rhos, strategies, seeds, dev="cuda", tau=1):
    """Batch ordering [param, rho, strategy, seed].  params: list of dict(p,tol,L,eps); strategies: list of (b, timing)."""
    P, R, S, N = len(params), len(rhos), len(strategies), len(seeds)
    def col(f):
        return torch.tensor([f(pi, ri, si, ni) for pi in range(P) for ri in range(R) for si in range(S) for ni in range(N)],
                            dtype=F64, device=dev)
    el = dict(
        rho=col(lambda pi, ri, si, ni: rhos[ri]),
        b=col(lambda pi, ri, si, ni: strategies[si][0]),
        timing=col(lambda pi, ri, si, ni: float(strategies[si][1])) > 0.5,
        dz=col(lambda pi, ri, si, ni: strategies[si][2] if len(strategies[si]) > 2 else 0.0),
        sid=col(lambda pi, ri, si, ni: ni).long(),
        p=col(lambda pi, ri, si, ni: params[pi].get("p", 0.0)),
        tol=col(lambda pi, ri, si, ni: params[pi].get("tol", 0.0)),
        L=col(lambda pi, ri, si, ni: params[pi].get("L", INF)),
        eps=col(lambda pi, ri, si, ni: params[pi].get("eps", 0.0)),
    )
    el["tau"] = torch.full_like(el["rho"], float(tau)).long()
    return el, (P, R, S, N)


@torch.no_grad()
def simulate(mech, T, el, seeds, c=C_MIN, blk=1024):
    dev = el["rho"].device
    B = el["rho"].shape[0]
    S = len(seeds)
    rho, tau, bb, timing, sidx = el["rho"], el["tau"], el["b"], el["timing"], el["sid"]
    dzs = el["dz"]
    use_dz = bool((dzs != 0).any())
    p_el, tol, L, eps = el["p"], el["tol"], el["L"], el["eps"]
    r_tau = rho ** tau.to(F64)
    sd_res = torch.sqrt((1 - r_tau ** 2).clamp(min=1e-18))
    ada, M3 = mech == "M1a", mech in ("M3a", "M3b", "M3c", "M5")
    raid = mech in ("M4", "M5")
    audit_on = mech not in ("M0", "M4")
    M = int(tau.max()) + 1
    qtab = type_cdf_tables(c, dev=dev)
    ar = torch.arange(B, device=dev)
    kk = torch.arange(K, device=dev)
    sq = torch.sqrt(1 - rho ** 2)[:, None]
    rho_c = rho[:, None]
    gens, gens2 = [], []
    for s in seeds:
        g = torch.Generator(device=dev); g.manual_seed(int(s)); gens.append(g)
        g2 = torch.Generator(device=dev); g2.manual_seed(int(s) + 10 ** 6); gens2.append(g2)
    z0 = torch.cat([torch.randn(1, K, generator=g, device=dev, dtype=F64) for g in gens], 0)  # [S,K]
    z = z0[sidx]

    susp = torch.zeros(B, K, dtype=F64, device=dev)           # eligible iff t > susp
    sc = torch.zeros(B, K, dtype=F64, device=dev)             # credit penalty (M3b)
    cus = torch.zeros(B, K, dtype=F64, device=dev)            # CUSUM (M3c)
    gam = (1 - 1 / L)[:, None] if mech == "M3b" else None
    wins = torch.zeros(B, K, dtype=F64, device=dev)
    qhat = torch.zeros(B, K, dtype=F64, device=dev)
    tl = torch.ones(B, dtype=F64, device=dev)
    pend_a = -torch.ones(B, M, dtype=torch.long, device=dev)
    pend_v = torch.zeros(B, M, dtype=F64, device=dev)
    zero = lambda *s: torch.zeros(*s, dtype=F64, device=dev)
    reg_a, reg_b, audits, nraid, n_est_reset = zero(B), zero(B), zero(B), zero(B), zero(B)
    util_a, util_b, nwin, flags, pen, n_res, n_fa = (zero(B, K) for _ in range(7))
    pw = (2 ** torch.arange(K, device=dev)).long()
    bvec = zero(B, K); bvec[:, 1] = bb
    dvec = zero(B, K); dvec[:, 1] = dzs
    kref = 0.5

    for t in range(1, T + 1):
        s = (t - 1) % blk
        if s == 0:
            E = torch.stack([torch.randn(blk, 1, K, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Ua = torch.stack([torch.rand(blk, 1, generator=g, device=dev, dtype=F64)[:, 0] for g in gens], 1)
            Rn = torch.stack([torch.rand(blk, 1, K + 2, generator=g, device=dev, dtype=F64)[:, 0] for g in gens2], 1)
        z = rho_c * z + sq * E[s][sidx]
        u = transform(torch.special.ndtr(z))
        us = transform(torch.special.ndtr(z + dvec)) if use_dz else u   # z-space-shifted ("quantile") inflation
        v = torch.minimum(us + bvec, torch.ones_like(u))
        elig = t > susp
        umax = u.max(1).values
        if mech == "M0":
            idx = torch.full((B,), (t - 1) % K, dtype=torch.long, device=dev)
            win = torch.ones(B, dtype=torch.bool, device=dev)
        elif mech == "M3b":
            idx = (v - sc).argmax(1)
            win = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            vm = torch.where(elig, v, torch.full_like(v, -1.0))
            mx, idx = vm.max(1)
            win = (mx >= c) if ada else (mx >= 0)
        if mech == "M3b":
            pen += (sc >= 0.5)
        else:
            pen += ~elig
        uw = u.gather(1, idx[:, None])[:, 0]
        Rs = Rn[s][sidx]                                          # [B,K+2]
        rd = torch.zeros(B, dtype=torch.bool, device=dev)
        if raid:
            mask = elig & (kk[None, :] != idx[:, None])
            tg = torch.where(mask, Rs[:, 2:], torch.full_like(Rs[:, 2:], -1.0)).argmax(1)
            rd = win & (Rs[:, 0] < eps) & mask.any(1)
            viol = (v - u).gather(1, tg[:, None])[:, 0] > 1e-9
            fl = rd & viol
            oht = torch.nn.functional.one_hot(tg, K).bool() & fl[:, None]
            susp = torch.where(oht, (t + L)[:, None].expand(B, K), susp)
            flags += oht
            nraid += rd
        win_eff = win & ~rd
        ridx = (Rs[:, 1] * K).long().clamp(max=K - 1)
        urand = u.gather(1, ridx[:, None])[:, 0]
        oh_a = torch.nn.functional.one_hot(idx, K).to(F64) * win_eff[:, None]
        reg_a += umax - torch.where(win_eff, uw, torch.zeros_like(uw))
        util_a += oh_a * uw[:, None]
        nwin += oh_a
        ag_b = torch.where(win_eff, idx, ridx)
        uw_b = torch.where(win_eff, uw, urand) * (~rd)
        reg_b += umax - uw_b
        util_b += torch.nn.functional.one_hot(ag_b, K).to(F64) * uw_b[:, None]
        oh = torch.nn.functional.one_hot(idx, K).to(F64) * win[:, None]
        wins += oh
        qh = qhat.gather(1, idx[:, None])[:, 0]
        if audit_on:
            if ada:
                pr = torch.clamp(4 * (1 + K * K) / ((T - t) * qh * c), max=1.0)
            else:
                pr = p_el
            o = win_eff & (Ua[s][sidx] < pr)
            audits += o
            slot = (t + tau) % M
            pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
            pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
            cs = t % M
            ag = pend_a[:, cs]
            has = ag >= 0
            agc = ag.clamp(min=0)
            wv = u.gather(1, agc[:, None])[:, 0]
            pv = pend_v[:, cs]
            elig_ag = (t > susp).gather(1, agc[:, None])[:, 0] if mech != "M3b" else torch.ones_like(has)
            ohag = torch.nn.functional.one_hot(agc, K).to(F64)
            n_res += ohag * (has & elig.all(1))[:, None]
            resid = None
            if mech == "M1f":
                mism = (pv - wv).abs() > tol
            elif mech == "M1a":
                mism = (pv - wv) > tol
            elif mech == "M2":
                bag = torch.where(agc == 1, bb, torch.zeros_like(bb))
                wl = us.gather(1, agc[:, None])[:, 0]          # what the liar's updated report would be at verification
                vp = torch.where(timing, wv, torch.where(agc == 1, torch.minimum(wl + bag, torch.ones_like(wl)), wv))
                mism = (vp - wv) > tol
            else:
                zv = torch.special.ndtri(phi_of(pv, agc, c))
                zw = torch.special.ndtri(phi_of(wv, agc, c))
                resid = (zw - r_tau * zv) / sd_res
                mism = resid < -tol
            if mech == "M3c":
                upd = has & elig_ag
                Sg = cus.gather(1, agc[:, None])[:, 0]
                Sn = (Sg - resid - kref).clamp(min=0)
                el_ = upd & (Sn > tol)
                newS = torch.where(upd, torch.where(el_, torch.zeros_like(Sn), Sn), Sg)
                cus = torch.where(ohag.bool(), newS[:, None].expand(B, K), cus)
            else:
                el_ = has & mism & elig_ag
            n_fa += ohag * (el_ & elig.all(1))[:, None]
            ohe = ohag.bool() & el_[:, None]
            flags += ohe
            if mech == "M3b":
                sc = sc + ohe.to(F64)
            else:
                susp = torch.where(ohe, (t + L)[:, None].expand(B, K), susp)
            pend_a[:, cs] = -1
            if ada:
                wins = torch.where(el_[:, None], torch.zeros_like(wins), wins)
                qhat = torch.where(el_[:, None], torch.zeros_like(qhat), qhat)
                cond = win & ~el_ & (qh == 0)
                qest = wins.gather(1, idx[:, None])[:, 0] / (t - tl + 1)
                alive = ~torch.isinf(susp)
                code = (alive.long() * pw).sum(1)
                qtrue = qtab[code, idx]
                flag = (qest > 4 * qtrue) | (qest < qtrue / 4)
                setq = cond & ~flag
                n_est_reset += (cond & flag)
                qhat = torch.where(oh.bool() & setq[:, None], qest[:, None].expand(B, K), qhat)
                tl = torch.where(el_, torch.full_like(tl, t + 1.0), tl)
        if mech == "M3b":
            sc = sc * gam
    return dict(reg_a=reg_a, reg_b=reg_b, util_a=util_a, util_b=util_b, nwin=nwin, flags=flags, pen=pen,
                audits=audits, nraid=nraid, est_reset=n_est_reset, n_res=n_res, n_fa=n_fa, susp=susp)
