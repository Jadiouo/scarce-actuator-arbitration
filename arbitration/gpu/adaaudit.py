"""Faithful GPU re-implementation of the audit mechanisms of Dai, Blanchard, Jaillet
(arXiv:2502.08412 v3): fixed-probability baseline M0(p) (Alg. 1, p.10) and AdaAudit
(Alg. 2, p.12), run on an abstract repeated-allocation model with optional
Markov (Gaussian-copula AR(1)) types and verification delay tau.

Rules copied from the paper (no tolerance anywhere):
  * allocation : alive agent with the highest report (AdaAudit: only if report >= c).
  * M0(p)      : audit winner w.p. p; eliminate permanently if  w != v   (exact).
  * AdaAudit   : audit w.p. min(4(1+K^2)/((T-t) qhat c), 1)  (=1 while qhat=0);
                 eliminate permanently if  v > w  (markup only, exact);
                 estimation / flagging of qhat as in lines 10-14, honest flagging per Def. 9.
  * Delay      : audit decided at t, outcome revealed at t+tau (w = u_{t+tau,i}), compared with the
                 report v_t made at t.  Elimination (and epoch reset) takes effect when revealed.
                 revpast=True is a control: reveals u_{t,i} (true past type) after the delay.
All tensors are float64 on CUDA; batch dimension = (config x seed).
"""
import math
import torch

K = 3
C_MIN = 0.2  # Assumption 6 constant c: agent 0 ~ U[c,1]


def type_cdf_tables(c=C_MIN, n=2_000_000, dev="cuda"):
    """q_i(A) for all 8 subsets A of {0,1,2}: Pr[u_i>=c, u_i>u_j for j in A\\{i}] (Def. 7).
    Agents: 0 ~ U[c,1], 1 ~ U[0,1], 2 ~ Beta(2,1) (cdf x^2)."""
    x = (torch.arange(n, dtype=torch.float64, device=dev) + 0.5) / n  # midpoints on [0,1]
    F = [((x - c) / (1 - c)).clamp(0, 1), x, x ** 2]
    f = [torch.where(x >= c, 1.0 / (1 - c), 0.0 * x), torch.ones_like(x), 2 * x]
    q = torch.zeros(8, K, dtype=torch.float64, device=dev)
    for code in range(8):
        A = [i for i in range(K) if code >> i & 1]
        for i in A:
            integ = f[i] * (x >= c)
            for j in A:
                if j != i:
                    integ = integ * F[j]
            q[code, i] = integ.sum() / n
    return q


def transform(phi):
    """uniform(0,1) copula coordinates -> utilities with fixed marginals."""
    return torch.stack([C_MIN + (1 - C_MIN) * phi[:, 0], phi[:, 1], torch.sqrt(phi[:, 2])], 1)


def expected_audits_idealised(T, c=C_MIN):
    """B_T of the full-information mechanism with true q (all agents alive, honest):
    sum_t sum_i q_i min(4(1+K^2)/((T-t) q_i c), 1)  [sum_i q_i = 1 since agent0>=c]."""
    q = type_cdf_tables(c)[7]
    t = torch.arange(1, T + 1, dtype=torch.float64, device=q.device)
    tot = 0.0
    for i in range(K):
        p = torch.clamp(4 * (1 + K * K) / ((T - t) * q[i] * c), max=1.0)
        tot += float((q[i] * p).sum())
    return tot


@torch.no_grad()
def simulate(mech, T, rho, tau, revpast, dev_b, seed, p_fixed=0.1, c=C_MIN, n_chk=40, blk=1024, dev_start=1):
    """mech in {'ada','fixed'}. rho,tau,revpast,dev_b: per-element tensors [B] (cuda).
    Agent 1 reports min(1,u+dev_b) from round dev_start on (dev_b=0 -> honest); everyone else honest.
    Returns dict of per-element results."""
    dev = rho.device
    B = rho.shape[0]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    ada = mech == "ada"
    tau_i = tau.long()
    M = int(tau_i.max()) + 1
    qtab = type_cdf_tables(c, dev=dev)
    ar = torch.arange(B, device=dev)
    sq = torch.sqrt(1 - rho ** 2)[:, None]
    rho_c = rho[:, None]

    alive = torch.ones(B, K, dtype=torch.bool, device=dev)
    z = torch.randn(B, K, generator=g, device=dev, dtype=torch.float64)
    wins = torch.zeros(B, K, dtype=torch.float64, device=dev)
    qhat = torch.zeros(B, K, dtype=torch.float64, device=dev)
    tl = torch.ones(B, dtype=torch.float64, device=dev)
    pend_a = -torch.ones(B, M, dtype=torch.long, device=dev)
    pend_v = torch.zeros(B, M, dtype=torch.float64, device=dev)
    pend_u = torch.zeros(B, M, dtype=torch.float64, device=dev)
    regret = torch.zeros(B, dtype=torch.float64, device=dev)
    audits = torch.zeros(B, dtype=torch.float64, device=dev)
    elim_t = torch.full((B, K), float(T + 1), dtype=torch.float64, device=dev)
    n_est_reset = torch.zeros(B, dtype=torch.float64, device=dev)
    pw = (2 ** torch.arange(K, device=dev)).long()
    chk_ts = sorted(set(int(round(x)) for x in torch.logspace(0, math.log10(T), n_chk).tolist()))
    chk_set = set(chk_ts)
    chk_reg = {}
    bvec = torch.zeros(B, K, dtype=torch.float64, device=dev)
    bvec[:, 1] = dev_b

    for t in range(1, T + 1):
        s = (t - 1) % blk
        if s == 0:
            E = torch.randn(blk, B, K, generator=g, device=dev, dtype=torch.float64)
            Ua = torch.rand(blk, B, generator=g, device=dev, dtype=torch.float64)
        z = rho_c * z + sq * E[s]
        u = transform(torch.special.ndtr(z))
        v = torch.minimum(u + (bvec if t >= dev_start else 0.0), torch.ones_like(u))
        vm = torch.where(alive, v, torch.full_like(v, -1.0))
        mx, idx = vm.max(1)
        win = (mx >= c) if ada else (mx >= 0)
        uw = u.gather(1, idx[:, None])[:, 0]
        regret += u.max(1).values - torch.where(win, uw, torch.zeros_like(uw))
        # epoch win counts (round t belongs to current epoch)
        oh = torch.nn.functional.one_hot(idx, K).to(torch.float64) * win[:, None]
        wins += oh
        qh = qhat.gather(1, idx[:, None])[:, 0]
        if ada:
            pr = torch.clamp(4 * (1 + K * K) / ((T - t) * qh * c), max=1.0)
        else:
            pr = torch.full((B,), p_fixed, dtype=torch.float64, device=dev)
        o = win & (Ua[s] < pr)
        audits += o
        # schedule audit, reveal at t+tau
        slot = (t + tau_i) % M
        pend_a[ar, slot] = torch.where(o, idx, -torch.ones_like(idx))
        pend_v[ar, slot] = v.gather(1, idx[:, None])[:, 0]
        pend_u[ar, slot] = uw
        # resolve audits revealed this round
        cs = t % M
        ag = pend_a[:, cs]
        has = ag >= 0
        agc = ag.clamp(min=0)
        wv = torch.where(revpast, pend_u[:, cs], u.gather(1, agc[:, None])[:, 0])
        pv = pend_v[:, cs]
        mism = (pv > wv) if ada else (pv != wv)
        el = has & mism & alive.gather(1, agc[:, None])[:, 0]
        ohe = torch.nn.functional.one_hot(agc, K).bool() & el[:, None]
        alive = alive & ~ohe
        elim_t = torch.where(ohe, torch.full_like(elim_t, float(t)), elim_t)
        pend_a[:, cs] = -1
        if ada:
            # new epoch on elimination
            wins = torch.where(el[:, None], torch.zeros_like(wins), wins)
            qhat = torch.where(el[:, None], torch.zeros_like(qhat), qhat)
            # estimation (lines 10-14): winner has qhat=0 and nobody eliminated
            cond = win & ~el & (qh == 0)
            qest = wins.gather(1, idx[:, None])[:, 0] / (t - tl + 1)
            code = (alive.long() * pw).sum(1)
            qtrue = qtab[code, idx]
            flag = (qest > 4 * qtrue) | (qest < qtrue / 4)  # others flag if >4q ; winner flags if <q/4
            setq = cond & ~flag
            n_est_reset += (cond & flag)
            qhat = torch.where(oh.bool() & setq[:, None], qest[:, None].expand(B, K), qhat)
            tl = torch.where(el, torch.full_like(tl, t + 1.0), tl)
        if t in chk_set:
            chk_reg[t] = regret.clone()

    return dict(regret=regret, audits=audits, elim_t=elim_t, chk_t=chk_ts,
                chk_reg=torch.stack([chk_reg[k] for k in chk_ts]), est_reset=n_est_reset)
