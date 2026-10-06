"""Stage-1b: tolerance-delta version of the Dai-Blanchard-Jaillet audits (imports adaaudit.py, does not modify it).
Fixed-p M0 : eliminate if |w-v| > delta.   AdaAudit : eliminate if v-w > delta.   delta=0 -> stage-1 rules (exact
same random stream/rule, up to strict '>' vs '!=' on a measure-zero tie set).
Extra bookkeeping vs adaaudit.simulate: per-agent wins/utility, audits after the first elimination, and
per-agent counts of audits resolved while all agents alive (n_res) / first eliminations (n_fa) for hazard estimates."""
import math
import torch
from .adaaudit import K, C_MIN, type_cdf_tables, transform, expected_audits_idealised  # noqa: F401


@torch.no_grad()
def simulate_tol(mech, T, rho, tau, delta, dev_b, seed, p_fixed=0.1, c=C_MIN, blk=1024):
    dev = rho.device
    B = rho.shape[0]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    ada = mech == "ada"
    f64 = torch.float64
    tau_i = tau.long()
    M = int(tau_i.max()) + 1
    qtab = type_cdf_tables(c, dev=dev)
    ar = torch.arange(B, device=dev)
    sq = torch.sqrt(1 - rho ** 2)[:, None]
    rho_c = rho[:, None]
    alive = torch.ones(B, K, dtype=torch.bool, device=dev)
    z = torch.randn(B, K, generator=g, device=dev, dtype=f64)
    wins = torch.zeros(B, K, dtype=f64, device=dev)
    qhat = torch.zeros(B, K, dtype=f64, device=dev)
    tl = torch.ones(B, dtype=f64, device=dev)
    pend_a = -torch.ones(B, M, dtype=torch.long, device=dev)
    pend_v = torch.zeros(B, M, dtype=f64, device=dev)
    regret = torch.zeros(B, dtype=f64, device=dev)
    audits = torch.zeros(B, dtype=f64, device=dev)
    audits_post = torch.zeros(B, dtype=f64, device=dev)
    anyel = torch.zeros(B, dtype=torch.bool, device=dev)
    nelim = torch.zeros(B, dtype=f64, device=dev)
    elim_t = torch.full((B, K), float(T + 1), dtype=f64, device=dev)
    nwin = torch.zeros(B, K, dtype=f64, device=dev)   # rounds won (allocated)
    util = torch.zeros(B, K, dtype=f64, device=dev)   # sum of true u over rounds won
    n_res = torch.zeros(B, K, dtype=f64, device=dev)
    n_fa = torch.zeros(B, K, dtype=f64, device=dev)
    pw = (2 ** torch.arange(K, device=dev)).long()
    bvec = torch.zeros(B, K, dtype=f64, device=dev)
    bvec[:, 1] = dev_b
    dl = delta
    for t in range(1, T + 1):
        s = (t - 1) % blk
        if s == 0:
            E = torch.randn(blk, B, K, generator=g, device=dev, dtype=f64)
            Ua = torch.rand(blk, B, generator=g, device=dev, dtype=f64)
        z = rho_c * z + sq * E[s]
        u = transform(torch.special.ndtr(z))
        v = torch.minimum(u + bvec, torch.ones_like(u))
        vm = torch.where(alive, v, torch.full_like(v, -1.0))
        mx, idx = vm.max(1)
        win = (mx >= c) if ada else (mx >= 0)
        uw = u.gather(1, idx[:, None])[:, 0]
        regret += u.max(1).values - torch.where(win, uw, torch.zeros_like(uw))
        oh = torch.nn.functional.one_hot(idx, K).to(f64) * win[:, None]
        wins += oh
        nwin += oh
        util += oh * uw[:, None]
        qh = qhat.gather(1, idx[:, None])[:, 0]
        if ada:
            pr = torch.clamp(4 * (1 + K * K) / ((T - t) * qh * c), max=1.0)
        else:
            pr = torch.full((B,), p_fixed, dtype=f64, device=dev)
        o = win & (Ua[s] < pr)
        audits += o
        audits_post += o & anyel
        slot = (t + tau_i) % M
        pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
        pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
        cs = t % M
        ag = pend_a[:, cs]
        has = ag >= 0
        agc = ag.clamp(min=0)
        wv = u.gather(1, agc[:, None])[:, 0]
        pv = pend_v[:, cs]
        mism = ((pv - wv) > dl) if ada else ((pv - wv).abs() > dl)
        alive_ag = alive.gather(1, agc[:, None])[:, 0]
        el = has & mism & alive_ag
        allal = alive.all(1)
        ohag = torch.nn.functional.one_hot(agc, K).to(f64)
        n_res += ohag * (has & allal)[:, None]
        n_fa += ohag * (el & allal)[:, None]
        ohe = torch.nn.functional.one_hot(agc, K).bool() & el[:, None]
        alive = alive & ~ohe
        elim_t = torch.where(ohe, torch.full_like(elim_t, float(t)), elim_t)
        nelim += el
        anyel = anyel | el
        pend_a[:, cs] = -1
        if ada:
            wins = torch.where(el[:, None], torch.zeros_like(wins), wins)
            qhat = torch.where(el[:, None], torch.zeros_like(qhat), qhat)
            cond = win & ~el & (qh == 0)
            qest = wins.gather(1, idx[:, None])[:, 0] / (t - tl + 1)
            code = (alive.long() * pw).sum(1)
            qtrue = qtab[code, idx]
            flag = (qest > 4 * qtrue) | (qest < qtrue / 4)
            setq = cond & ~flag
            qhat = torch.where(oh.bool() & setq[:, None], qest[:, None].expand(B, K), qhat)
            tl = torch.where(el, torch.full_like(tl, t + 1.0), tl)
    return dict(regret=regret, audits=audits, audits_post=audits_post, nelim=nelim, elim_t=elim_t,
                nwin=nwin, util=util, n_res=n_res, n_fa=n_fa)


@torch.no_grad()
def theory_error(rho_tau, delta, n=4_000_000, seed=0, c=C_MIN, dev="cuda"):
    """Single-audit false-flag probabilities for the HONEST winner under the Gaussian copula, all 3 agents alive.
    z_t ~ N(0,I); z_{t+tau} = r z_t + sqrt(1-r^2) e with r = rho^tau; winner = argmax u_t (>= c, as in AdaAudit;
    for fixed-p the c-cut does not matter since u0>=c anyway makes winner>=c always).
    Returns per-winner-identity: P(winner=i), P(|u'-u|>delta | i), P(u-u'>delta | i)."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    z = torch.randn(n, K, generator=g, device=dev, dtype=torch.float64)
    e = torch.randn(n, K, generator=g, device=dev, dtype=torch.float64)
    z2 = rho_tau * z + math.sqrt(max(1 - rho_tau ** 2, 0.0)) * e
    u = transform(torch.special.ndtr(z))
    u2 = transform(torch.special.ndtr(z2))
    mx, idx = u.max(1)
    uw = mx
    uw2 = u2.gather(1, idx[:, None])[:, 0]
    out = []
    for i in range(K):
        m = idx == i
        out.append(dict(p_win=float(m.double().mean()),
                        abs=float(((uw - uw2).abs() > delta)[m].double().mean()),
                        up=float(((uw - uw2) > delta)[m].double().mean())))
    return out
