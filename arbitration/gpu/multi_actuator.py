"""Exploratory, simplified A-actuator simulator (D2). Truthful reporting only, v2-type dynamics (shock damage).

A actuators share the dispatcher's information (one learner, one auditor). Each runs the SAME dispatch rule on its own position;
actuators decide in index order inside a step, and a later actuator may not pick a robot that is the target of an earlier one
(a robot is "taken" from the moment it is chosen until its service completes). No preemption (a chosen target is kept).
Policies: none (each actuator round-robins over its own contiguous arc of posts; A=1 equals sim_torch `none`),
learned_index, fused_index, audit_disp_index (disp_offset: calibration of the dispatch-audit selection bias).
Random numbers, dynamics and bookkeeping reuse sim_torch (draw_instance, _shock_draws, rhat, _index, fused_tau_t), so that with
A=1 every policy reproduces sim_torch.simulate (checked in scripts/run_d_scaling.py `check`).
"""
import numpy as np
import torch
from .. import model as _m
from ..model import RING, SPEED, make_config
from . import sim_torch as st

INF = float("inf")


class _S:
    pass


def simulate_multi(policy, seeds, n=8, rate=0.004, A=1, steps=3000, burn=500, spread=0.6, sigma=0.15, cfg=None,
                   disp_offset=0.0, device=None):
    dev = st._dev(device)
    cfg = cfg or make_config("v2")
    assert cfg.damage == "shock" and cfg.noise in ("ar1", "iid", "bias")
    seeds = [int(s) for s in seeds]
    B = len(seeds)
    f64 = torch.float64
    T = lambda a, dt=None: torch.tensor(np.asarray(a), device=dev, dtype=dt)
    posts, rates, pos0, bias = (np.empty((B, n)), np.empty((B, n)), np.empty(B), np.empty((B, n)))
    cn = []
    for b, s in enumerate(seeds):
        d = st.draw_instance(s, n, rate, spread, sigma, steps, cfg)
        posts[b], rates[b], pos0[b], bias[b] = d["posts"], d["rates"], d["pos0"], d["bias"]
        cn.append(d["cn"])
    posts_t, rates_t, bias_t = T(posts), T(rates), T(bias)
    CN = T(np.stack(cn, axis=1))                                   # (steps, B, n)
    dr = [st._shock_draws(s, steps, n, []) for s in seeds]
    SU, SE = T(np.stack([d[0] for d in dr], axis=1)), T(np.stack([d[1] for d in dr], axis=1))
    sf, sr = cfg.shock_frac, cfg.shock_rate
    base_decay = rates_t * (1.0 - sf)
    shock_size = sf * rates_t / sr
    svc = cfg.service
    S = _S()
    S.n, S.B, S.dev = n, B, dev
    S.svc = torch.full((B, 1), float(svc), dtype=f64, device=dev)
    S.drop = torch.zeros(B, n, dtype=f64, device=dev)
    S.el = torch.zeros(B, n, dtype=f64, device=dev)
    S.t_reset = torch.zeros(B, n, dtype=f64, device=dev)
    h = torch.ones(B, n, dtype=f64, device=dev)
    pos = torch.stack([(T(pos0) + a * RING / A) % RING for a in range(A)], 1)            # (B, A)
    target = torch.full((B, A), -1, dtype=torch.long, device=dev)
    serving = torch.zeros(B, A, dtype=torch.long, device=dev)
    rr = torch.zeros(B, A, dtype=torch.long, device=dev)
    arc = [list(range(a * n // A, (a + 1) * n // A)) for a in range(A)]
    arc_t = [T(x, torch.long) for x in arc]
    a_s, a_k = torch.zeros(B, n, dtype=f64, device=dev), torch.zeros(B, n, dtype=f64, device=dev)
    a_dc, a_dt = torch.zeros(B, n, dtype=f64, device=dev), torch.zeros(B, n, dtype=f64, device=dev)
    hacc = torch.zeros(B, n, dtype=f64, device=dev)
    anydead = torch.zeros(B, dtype=f64, device=dev)
    nserv = torch.zeros(B, dtype=f64, device=dev)
    ar_n = torch.arange(n, device=dev)[None, :]
    arB = torch.arange(B, device=dev)
    G = cfg.fuse_grid
    shock_flag = torch.ones(B, dtype=torch.bool, device=dev)
    p_t, f_t = torch.full((B,), sr, dtype=f64, device=dev), torch.full((B,), sf, dtype=f64, device=dev)
    sig = torch.full((B,), max(sigma, 1e-9), dtype=f64, device=dev)
    n0 = cfg.audit_n0
    use_fuse = policy in ("fused_index", "audit_disp_index")
    audit = policy == "audit_disp_index"
    learner = policy != "none"

    for t in range(steps):
        tf = torch.tensor(float(t), dtype=f64, device=dev)
        hit = (SU[t] < sr) * SE[t] * shock_size
        h = torch.clamp(h - base_decay - hit, 0.0, 1.0)
        tau = 1.0 - h
        if t >= burn:
            hacc = hacc + h
            anydead = anydead + (h <= 0).any(1).to(f64)
        claim = tau + bias_t + CN[t]
        S.h, S.t, S.tau = h, tf, tau
        reset = torch.zeros(B, n, dtype=torch.bool, device=dev)
        taken = torch.zeros(B, n, dtype=torch.bool, device=dev)
        for a in range(A):                                          # robots chosen by actuators still active (any index)
            oh = (ar_n == target[:, a:a + 1].clamp(min=0)) & (target[:, a:a + 1] >= 0)
            taken = taken | oh
        tp = None
        for a in range(A):
            srv = serving[:, a] > 0
            sv = torch.where(srv, serving[:, a] - 1, serving[:, a])
            fin = srv & (sv == 0)
            tgt_prev = target[:, a]
            fin_oh = (ar_n == tgt_prev.clamp(min=0)[:, None]) & fin[:, None]
            reset = reset | fin_oh
            act = ~srv
            need = act & (tgt_prev < 0)
            d = torch.abs(posts_t - pos[:, a:a + 1])
            d = torch.minimum(d, RING - d)
            if need.any():
                if policy == "none":
                    idx = arc_t[a][rr[:, a] % len(arc[a])]
                    new = idx
                    rr[:, a] = rr[:, a] + need.long()
                else:
                    rh = st.rhat(S)
                    if use_fuse:
                        if True:
                            age = t - S.t_reset
                            cl, sd = claim, sig[:, None].expand(B, n)
                            if audit:
                                cl = claim - a_s / (a_k + n0)
                                sd = sig[:, None] * torch.sqrt(1.0 + 1.0 / (a_k + n0))
                            rows = torch.nonzero(need).flatten()
                            tp_r = st.fused_tau_t(cl[rows].reshape(-1), age[rows].reshape(-1), rh[rows].reshape(-1),
                                                  sd[rows].reshape(-1), shock_flag[rows][:, None].expand(-1, n).reshape(-1),
                                                  p_t[rows][:, None].expand(-1, n).reshape(-1),
                                                  f_t[rows][:, None].expand(-1, n).reshape(-1), G).reshape(-1, n)
                            tpf = torch.zeros(B, n, dtype=f64, device=dev)
                            tpf[rows] = tp_r
                        score = st._index(tpf, rh, d, S.svc)
                    else:
                        th = torch.minimum(torch.ones_like(h), rh * (S.t - S.t_reset))
                        score = st._index(th, rh, d, S.svc)
                    score = torch.where(taken, torch.full_like(score, -INF), score)
                    new = score.argmax(1)
                    if audit:                                        # dispatch-time claim snapshot
                        dsp = (ar_n == new[:, None]) & need[:, None]
                        a_dc = torch.where(dsp, claim, a_dc)
                        a_dt = torch.where(dsp, tf.expand(B, n), a_dt)
                tgt_new = torch.where(need, new, tgt_prev)
                taken = taken | ((ar_n == tgt_new[:, None]) & need[:, None])
            else:
                tgt_new = tgt_prev
            tgt_new = torch.where(act, tgt_new, tgt_prev)
            tc = tgt_new.clamp(min=0)
            mover = act & (tgt_new >= 0)
            tpos = posts_t.gather(1, tc[:, None])[:, 0]
            dd = tpos - pos[:, a]
            dd = torch.where(dd > RING / 2, dd - RING, dd)
            dd = torch.where(dd < -RING / 2, dd + RING, dd)
            arrived = torch.abs(dd) <= SPEED
            newpos = torch.where(arrived, tpos, torch.remainder(pos[:, a] + torch.sign(dd) * SPEED, RING))
            pos[:, a] = torch.where(mover, newpos, pos[:, a])
            arr = mover & arrived
            serving[:, a] = torch.where(arr, torch.full_like(sv, svc), sv)
            nserv = nserv + arr.to(f64) * (1.0 if t >= burn else 0.0)
            if audit:
                m = arr[:, None] & (ar_n == tc[:, None])
                cd = a_dc.gather(1, tc[:, None])[:, 0]
                dt = tf - a_dt.gather(1, tc[:, None])[:, 0]
                rhj = st.rhat(S).gather(1, tc[:, None])[:, 0]
                tau_t = tau.gather(1, tc[:, None])[:, 0]
                samp = (cd - (tau_t - rhj * dt)) - disp_offset
                a_s = a_s + torch.where(m, samp[:, None], torch.zeros_like(a_s))
                a_k = a_k + m.to(f64)
            if learner:
                el_ = tf - S.t_reset.gather(1, tc[:, None])[:, 0]
                okm = (arr & (el_ > 0))[:, None] & (ar_n == tc[:, None])
                h_obs = h.gather(1, tc[:, None])[:, 0]
                S.drop = S.drop + torch.where(okm, (1.0 - h_obs)[:, None], torch.zeros_like(S.drop))
                S.el = S.el + torch.where(okm, el_[:, None], torch.zeros_like(S.el))
            target[:, a] = torch.where(fin, torch.full_like(tgt_new, -1), tgt_new)
        h = torch.where(reset, torch.ones_like(h), h)
        S.t_reset = torch.where(reset, tf.expand(B, n), S.t_reset)
    nb = steps - burn
    return dict(mean_health=(hacc / nb).mean(1).cpu().numpy(), robot_health=(hacc / nb).cpu().numpy(),
                frac_any_dead=(anydead / nb).cpu().numpy(), n_services=nserv.cpu().numpy(),
                audit_sum=a_s.cpu().numpy(), audit_cnt=a_k.cpu().numpy())
