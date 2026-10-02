"""Phase C (EXPLORATORY, not pre-registered): what does the lambda*d request price do?

  c1  Is priced's gain just thrash suppression?  v2 and legacy, r in {.002,.004,.008,.016}, N=8, 32 holdout seeds.
      Arms: honest(preempt), honest(no preempt + keep_commitment) = the NO-THRASH baseline, honest(preempt+best hysteresis),
      priced(lambda* | preempt), priced(lambda* | no preempt).  lambda*, hysteresis chosen on seeds 0-15, evaluated on 1000-1031.
  c2  Implied exchange rate of the exact per-step optimal policy (N=3, v2, r=.004): conditional logit of the realised next
      target on (true tau_i, distance_i) at decision points; kappa = beta_d / beta_tau (urgency per unit distance), compared to
      priced's lambda* (N=3, seeds 0-15).
  c3  magnitude comparison with the strategic loss of the reporting game (results/game.json).

    python3 scripts/run_c_pricing.py [c1|c2|c3|figs|all]      -> results/c_pricing.json, figures/fig14a/14b_*.png
"""
import json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from scipy import optimize
from arbitration.model import make_config, RING, SPEED, THETA, SERVICE
from arbitration.experiments import ci95, paired
from arbitration.gpu.sim_torch import simulate

OUT = os.path.join(ROOT, "results", "c_pricing.json")
SPREAD = 0.6
RINGI = int(RING)
BURN = 500
STEPS = 6500                      # 500 burn + 6000 scored
TUNE = list(range(16))
HOLD = list(range(1000, 1032))
RATES = [0.002, 0.004, 0.008, 0.016]
LAMS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15]
HYSTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(d):
    json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def sim_cells(policy, cells, seeds, n=8, steps=STEPS):
    """cells: list of dict(model cfg, rate, lam, preempt, hyst). One batch; returns list of per-cell metric dicts."""
    S = len(seeds)
    sd, cf, rt, lm, pe, hy = [], [], [], [], [], []
    for c in cells:
        sd += seeds; cf += [c["cfg"]] * S
        rt += [c["rate"]] * S; lm += [c.get("lam", 0.0)] * S
        pe += [c["preempt"]] * S; hy += [c.get("hyst", 0.0)] * S
    out = simulate(policy, sd, n=n, rate=np.array(rt), spread=SPREAD, steps=steps, lam=np.array(lm),
                   preempt=np.array(pe), hyst=np.array(hy), model=cf, burn=BURN)
    return [{m: out[m][i * S:(i + 1) * S] for m in ("mean_health", "retargets_per_service", "frac_dead_robot_steps",
                                                    "min_health_p05")} for i in range(len(cells))]


def cfg_of(model, keep):
    return make_config(model, keep_commitment=True) if keep else make_config(model)


def pick(policy, make, key, n=8):
    """choose argmax over grid on TUNE seeds. make(v) -> cell dict."""
    vals = list(key)
    res = sim_cells(policy, [make(v) for v in vals], TUNE, n)
    m = [float(r["mean_health"].mean()) for r in res]
    return vals[int(np.argmax(m))], dict(zip(map(str, vals), m))


# ===================================================================== C1
def stage_c1():
    t0 = time.time()
    out = {"seeds_tune": "0-15", "seeds_hold": "1000-1031", "steps": STEPS, "burn": BURN, "n": 8, "spread": SPREAD,
           "exploratory": True, "cells": {}}
    for model in ("v2", "legacy"):
        for r in RATES:
            log(f"c1 {model} r={r}")
            nt = cfg_of(model, True)                 # no-thrash: commit, keep commitment when callers vanish
            pr = cfg_of(model, False)                # preempt arms use the model's own setting
            # --- selection on TUNE
            lam_pre, tab_pre = pick("priced", lambda v: dict(cfg=pr, rate=r, lam=v, preempt=True), LAMS)
            lam_np, tab_np = pick("priced", lambda v: dict(cfg=nt, rate=r, lam=v, preempt=False), LAMS)
            hy_best, tab_hy = pick("honest", lambda v: dict(cfg=pr, rate=r, hyst=v, preempt=True), HYSTS)
            # --- evaluation on HOLD
            ev = {}
            ev["honest_preempt"] = sim_cells("honest", [dict(cfg=pr, rate=r, preempt=True)], HOLD)[0]
            ev["honest_nothrash"] = sim_cells("honest", [dict(cfg=nt, rate=r, preempt=False)], HOLD)[0]
            ev["honest_hyst"] = sim_cells("honest", [dict(cfg=pr, rate=r, preempt=True, hyst=hy_best)], HOLD)[0]
            ev["priced_preempt"] = sim_cells("priced", [dict(cfg=pr, rate=r, lam=lam_pre, preempt=True)], HOLD)[0]
            ev["priced_nopreempt"] = sim_cells("priced", [dict(cfg=nt, rate=r, lam=lam_np, preempt=False)], HOLD)[0]
            ev["priced_lam0_nopreempt"] = sim_cells("priced", [dict(cfg=nt, rate=r, lam=0.0, preempt=False)], HOLD)[0]
            ev["honest_index_ref"] = sim_cells("honest_index", [dict(cfg=nt, rate=r, preempt=False)], HOLD)[0]
            ev["learned_index_ref"] = sim_cells("learned_index", [dict(cfg=nt, rate=r, preempt=False)], HOLD)[0]
            base = ev["honest_nothrash"]["mean_health"]
            c = {"lambda_star_preempt": lam_pre, "lambda_star_nopreempt": lam_np, "hyst_best": hy_best,
                 "tune_table": {"priced_preempt": tab_pre, "priced_nopreempt": tab_np, "honest_hyst": tab_hy},
                 "arms": {}, "vs_nothrash": {}, "decomposition": {}}
            for k, v in ev.items():
                c["arms"][k] = {"health": ci95(v["mean_health"]), "median": float(np.median(v["mean_health"])),
                                "p05_min_health": ci95(v["min_health_p05"]),
                                "dead_robot_frac": ci95(v["frac_dead_robot_steps"]),
                                "retargets_per_service": ci95(v["retargets_per_service"]),
                                "per_seed": np.round(v["mean_health"], 5).tolist()}
                d = v["mean_health"] - base
                c["vs_nothrash"][k] = dict(paired(v["mean_health"], base), median=float(np.median(d)))
            hp = ev["honest_preempt"]["mean_health"]
            c["decomposition"] = {
                "thrash_cost(nothrash-honest_preempt)": paired(base, hp),
                "priced_preempt-honest_preempt": paired(ev["priced_preempt"]["mean_health"], hp),
                "priced_preempt-honest_hyst": paired(ev["priced_preempt"]["mean_health"], ev["honest_hyst"]["mean_health"]),
                "priced_nopreempt-honest_nothrash": paired(ev["priced_nopreempt"]["mean_health"], base),
                "lam0_vs_nothrash_max_abs": float(np.abs(ev["priced_lam0_nopreempt"]["mean_health"] - base).max())}
            gain = c["decomposition"]["priced_preempt-honest_preempt"]["mean"]
            thr = c["decomposition"]["thrash_cost(nothrash-honest_preempt)"]["mean"]
            c["decomposition"]["share_of_priced_preempt_gain_explained_by_nothrash"] = (thr / gain) if abs(gain) > 1e-9 else None
            out["cells"][f"{model}|{r}"] = c
            log(f"   lam*(pre)={lam_pre} lam*(nopre)={lam_np} hyst*={hy_best}  "
                f"priced_pre-nothrash {c['vs_nothrash']['priced_preempt']['mean']:+.4f} "
                f"priced_nopre-nothrash {c['vs_nothrash']['priced_nopreempt']['mean']:+.4f} "
                f"honest_pre-nothrash {c['vs_nothrash']['honest_preempt']['mean']:+.4f}")
    out["gpu_seconds"] = time.time() - t0
    d = load(); d["c1"] = out; save(d)
    log(f"c1 done {out['gpu_seconds']:.0f}s")


# ===================================================================== C2
def lam_star_n3(model_name="v2", rate=0.004, n=3):
    """priced lambda* at N=3 on seeds 0-15 (and eval on 1000-1031), preempt on/off, at the DP cell."""
    res = {}
    for pre in (True, False):
        cfg = cfg_of(model_name, not pre)
        lam, tab = pick("priced", lambda v: dict(cfg=cfg, rate=rate, lam=v, preempt=pre), LAMS, n=n)
        # refine around the optimum
        i = LAMS.index(lam)
        fine = sorted(set([round(x, 5) for x in np.linspace(LAMS[max(i - 1, 0)], LAMS[min(i + 1, len(LAMS) - 1)], 9)]))
        lam2, tab2 = pick("priced", lambda v: dict(cfg=cfg, rate=rate, lam=v, preempt=pre), fine, n=n)
        ev = sim_cells("priced", [dict(cfg=cfg, rate=rate, lam=lam2, preempt=pre)], HOLD, n)[0]
        hon = sim_cells("honest", [dict(cfg=cfg, rate=rate, preempt=pre)], HOLD, n)[0]
        res["preempt" if pre else "nopreempt"] = {
            "lambda_star": lam2, "coarse_table": tab, "fine_table": tab2,
            "priced_hold": ci95(ev["mean_health"]), "honest_hold": ci95(hon["mean_health"]),
            "priced_minus_honest": paired(ev["mean_health"], hon["mean_health"])}
    return res


def fit_logit(tau, d, chosen, spec="lin", w=None):
    """tau, d: (D, N) ; chosen (D,). u = b_tau * f(tau) - b_d * g(d). returns (b_tau, b_d, loglik, acc)."""
    tau, d = torch.tensor(tau, dtype=torch.float64), torch.tensor(d, dtype=torch.float64)
    ch = torch.tensor(chosen)
    wt = torch.ones(len(chosen), dtype=torch.float64) if w is None else torch.tensor(w, dtype=torch.float64)
    if spec == "lin":
        X1, X2 = tau, d
    else:
        X1, X2 = torch.log(tau + 0.01), torch.log(d / SPEED + SERVICE)

    def nll(p):
        u = p[0] * X1 - p[1] * X2
        return -(wt * (u.gather(1, ch[:, None])[:, 0] - torch.logsumexp(u, 1))).sum()

    best = None
    for init in ([1.0, 0.01], [5.0, 0.1], [0.5, 0.5], [10.0, 0.0]):
        p = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); l = nll(p); l.backward(); return l
        opt.step(closure)
        v = float(nll(p.detach()))
        if best is None or v < best[0]:
            best = (v, p.detach().clone())
    p = best[1]
    u = p[0] * X1 - p[1] * X2
    acc = float((u.argmax(1) == ch).double().mean())
    return float(p[0]), float(p[1]), -best[0], acc


def record_dp_decisions(seeds, rate=0.004, K=30, steps=10500, burn=500, n=3):
    """Solve the per-step MDP (K) for each seed, simulate its policy, record at every 'first free step after a service'
    the true tau of every robot, distances from the actuator, and the next robot the policy actually starts serving."""
    from arbitration.gpu import dp_step as ds
    cfg = make_config("v2")
    insts = [ds.int_instance(s, n, rate, SPREAD) for s in seeds]
    rows = []
    with ds.rounded_instances():
        cs = 16
        for a in range(0, len(seeds), cs):
            sl = slice(a, min(len(seeds), a + cs))
            res = ds.solve_steps([(p, r) for p, r, _ in insts[sl]], cfg, [K], tol=1e-4)
            _, P = res[-1]
            QM, QS = ds.policy_tables(P)
            g = P.g.copy(); P.V = None
            B = len(insts[sl])
            ss = ds.StepSim([dict(seed=seeds[sl][i], posts=insts[sl][i][0], rates=insts[sl][i][1], pos0=insts[sl][i][2])
                             for i in range(B)], [cfg] * B, steps)
            ev = run_recorded(ss, QM, QS, K, burn)
            for i in range(B):
                rows.append(dict(seed=seeds[sl][i], posts=insts[sl][i][0], g=float(g[i]), **{k: v[i] for k, v in ev.items()}))
            del QM, QS, ss, res, P
            torch.cuda.empty_cache()
    return rows


def run_recorded(self, QM, QS, K, burn):
    """ds.StepSim.run, plus event recording (a copy of its loop; the original is untouched)."""
    from arbitration.gpu.dp_step import _interp_idx
    dev, B, N, steps = self.dev, self.B, self.N, self.steps
    f = torch.float64
    ar = torch.arange(B, device=dev)
    x = self.pos0.clone()
    h = torch.ones(B, N, dtype=f, device=dev)
    serving = torch.zeros(B, dtype=torch.long, device=dev)
    tgt = torch.zeros(B, dtype=torch.long, device=dev)
    was_srv = torch.zeros(B, dtype=torch.bool, device=dev)
    QMf, QSf = QM.reshape(B, RINGI, -1), QS.reshape(B, N, -1)
    dxs = torch.tensor([-1, 0, 1], device=dev)
    R_first, R_h, R_x, R_go, R_j = [], [], [], [], []
    for t in range(steps):
        hit = (self.SU[t] < self.sr) * self.SE[t] * self.shock_size
        h = torch.clamp(h - self.base_decay - hit, 0.0, 1.0)
        srv = serving > 0
        nsv = torch.where(srv, serving - 1, serving)
        fin = srv & (nsv == 0)
        oh = (torch.arange(N, device=dev)[None, :] == tgt[:, None]) & fin[:, None]
        h = torch.where(oh, torch.ones_like(h), h)
        serving = nsv
        act = ~srv
        first = act & was_srv

        was_srv = srv
        R_first.append(first); R_h.append(h.clone()); R_x.append(x.clone())
        flat, w = _interp_idx(h, K)
        xm = (x[:, None] + dxs[None, :]) % RINGI
        qm = (QMf[ar[:, None, None], xm[:, :, None], flat[:, None, :]] * w[:, None, :]).sum(-1)
        qs = (QSf[ar[:, None, None], torch.arange(N, device=dev)[None, :, None], flat[:, None, :]] * w[:, None, :]).sum(-1)
        dpost = (self.posts - x[:, None] + RINGI // 2) % RINGI - RINGI // 2
        qs = torch.where(dpost.abs() <= 1, qs, torch.full_like(qs, -float("inf")))
        a = torch.cat([qm, qs], 1).argmax(1)
        is_serve = a >= 3
        j = (a - 3).clamp(min=0)
        move = torch.where(is_serve, dpost.gather(1, j[:, None])[:, 0], dxs[a.clamp(max=2)])
        x = torch.where(act, (x + move) % RINGI, x)
        go = act & is_serve
        serving = torch.where(go, self.svc, serving)
        tgt = torch.where(go, j, tgt)
        R_go.append(go); R_j.append(j)
    first, H, X, go, J = (torch.stack(v).cpu().numpy() for v in (R_first, R_h, R_x, R_go, R_j))
    out = {k: [] for k in ("tau", "d", "chosen", "wait")}
    for b in range(B):
        posts = self.posts[b].cpu().numpy()
        gi = np.nonzero(go[:, b])[0]
        fi = np.nonzero(first[:, b])[0]
        taus, ds_, ch, wt = [], [], [], []
        for t0 in fi:
            if t0 < burn:
                continue
            k = np.searchsorted(gi, t0)
            if k >= len(gi):
                break
            taus.append(1.0 - H[t0, b]); dd = np.abs(posts - X[t0, b]); ds_.append(np.minimum(dd, RING - dd))
            ch.append(int(J[gi[k], b])); wt.append(int(gi[k] - t0))
        out["tau"].append(np.array(taus)); out["d"].append(np.array(ds_)); out["chosen"].append(np.array(ch))
        out["wait"].append(np.array(wt))
    return out


def stage_c2(nseeds=32):
    t0 = time.time()
    d = load()
    seeds = list(range(nseeds))
    log("c2: lambda* at N=3")
    lam = lam_star_n3()
    log(f"   lambda*: {({k: v['lambda_star'] for k, v in lam.items()})}")
    log("c2: DP decisions")
    rows = record_dp_decisions(seeds)
    tau = np.concatenate([r["tau"] for r in rows]); dist = np.concatenate([r["d"] for r in rows])
    ch = np.concatenate([r["chosen"] for r in rows]); sid = np.concatenate([[i] * len(r["chosen"]) for i, r in enumerate(rows)])
    wait = np.concatenate([r["wait"] for r in rows])
    log(f"   {len(ch)} decisions from {len(rows)} seeds; median wait to start service {np.median(wait):.0f} steps")
    # decisions where the choice is non-trivial: at least two candidates with tau>0.05 (otherwise tau carries no info)
    sig = (tau > 0.05).sum(1) >= 2
    res = {"n_decisions": int(len(ch)), "n_nontrivial": int(sig.sum()), "wait_median": float(np.median(wait)),
           "wait_gt_40_frac": float((wait > 40).mean()), "seeds": nseeds, "K": 30, "rate": 0.004, "n": 3}
    def fit_boot(tau, dist, ch, sid, spec, nboot=200):
        bt, bd, ll, acc = fit_logit(tau, dist, ch, spec)
        rng = np.random.default_rng(0)
        us = np.unique(sid)
        ks = []
        for _ in range(nboot):
            pick_s = rng.choice(us, len(us))
            idx = np.concatenate([np.nonzero(sid == s)[0] for s in pick_s])
            b1, b2, _, _ = fit_logit(tau[idx], dist[idx], ch[idx], spec)
            ks.append((b1, b2, b2 / b1 if b1 > 1e-9 else np.nan))
        ks = np.array(ks)
        q = lambda c: [float(np.nanpercentile(ks[:, c], 2.5)), float(np.nanpercentile(ks[:, c], 97.5))]
        return dict(beta_tau=bt, beta_d=bd, ratio=bd / bt, ratio_ci95=q(2), beta_tau_ci95=q(0), beta_d_ci95=q(1),
                    loglik=ll, top1_acc=acc)
    for tag, m in (("all", np.ones(len(ch), bool)), ("nontrivial", sig)):
        res[f"dp_logit_lin_{tag}"] = fit_boot(tau[m], dist[m], ch[m], sid[m], "lin")
        res[f"dp_logit_logindex_{tag}"] = fit_boot(tau[m], dist[m], ch[m], sid[m], "log", nboot=100)
        r_ = res[f"dp_logit_lin_{tag}"]
        log(f"   [{tag}] kappa=beta_d/beta_tau = {r_['ratio']:.4f} {r_['ratio_ci95']}  top1 {r_['top1_acc']:.2f}")
    # reference rules on the SAME decision states (calibrates the method): what kappa does each rule imply?
    refs = {}
    tm, dm, cm = tau[sig], dist[sig], ch[sig]
    rules = {"greedy_true(argmax tau)": tm,
             "index_true(tau/(d+5))": tm / (dm / SPEED + SERVICE),
             "nearest(argmin d, tau>.05)": np.where(tm > 0.05, -dm, -1e9)}
    for lam_k, tag in ((lam["preempt"]["lambda_star"], "priced(lambda*|preempt)"),
                       (lam["nopreempt"]["lambda_star"], "priced(lambda*|nopreempt)")):
        ok = tm > THETA + lam_k * dm
        sc = np.where(ok, tm, -1.0 + 0 * tm)
        sc = np.where(ok.any(1, keepdims=True), sc, tm)       # no caller: fall back to argmax tau
        rules[tag] = sc
    for name, sc in rules.items():
        pick_ = sc.argmax(1)
        b1, b2, ll, acc = fit_logit(tm, dm, pick_, "lin")
        refs[name] = dict(beta_tau=b1, beta_d=b2, ratio=(b2 / b1 if b1 > 1e-9 else None), top1_acc=acc,
                          agreement_with_dp=float((pick_ == cm).mean()))
        log(f"   ref {name}: ratio {refs[name]['ratio']}  agree w/ DP {refs[name]['agreement_with_dp']:.2f}")
    res["reference_rules_same_states"] = refs
    res["dp_agreement_baselines"] = {"majority_chosen_is_argmax_tau": float((tm.argmax(1) == cm).mean()),
                                     "chosen_is_nearest": float((dm.argmin(1) == cm).mean())}
    res["lambda_star_n3"] = lam
    # does priced with lambda = kappa_DP do as well as lambda*?
    kap = res["dp_logit_lin_nontrivial"]["ratio"]
    cfg = cfg_of("v2", False)
    e = sim_cells("priced", [dict(cfg=cfg, rate=0.004, lam=kap, preempt=True)], HOLD, 3)[0]
    e2 = sim_cells("priced", [dict(cfg=cfg_of("v2", True), rate=0.004, lam=kap, preempt=False)], HOLD, 3)[0]
    res["priced_at_kappa_dp"] = {"kappa": kap, "preempt": ci95(e["mean_health"]), "nopreempt": ci95(e2["mean_health"])}
    res["gpu_seconds"] = time.time() - t0
    d["c2"] = res; save(d)
    log(f"c2 done {res['gpu_seconds']:.0f}s")


# ===================================================================== C3 + figure
def stage_c3():
    d = load()
    g = json.load(open(os.path.join(ROOT, "results", "game.json")))
    out = {"note": "magnitudes only; different cells (priced gain: N=8 v2 across loads; strategic loss: results/game.json)"}
    c1 = d["c1"]["cells"]
    gains = {k: v["vs_nothrash"]["priced_nopreempt"]["mean"] for k, v in c1.items() if k.startswith("v2")}
    gains_pre = {k: v["decomposition"]["priced_preempt-honest_preempt"]["mean"] for k, v in c1.items() if k.startswith("v2")}
    out["priced_nopreempt_minus_nothrash_v2"] = gains
    out["priced_preempt_minus_honest_preempt_v2"] = gains_pre
    loss = {}
    gs = g["game_sweep"]["cells"]
    for k, c in gs.items():
        for m in ("fused_index", "honest_index"):
            if m in c["mech"]:
                loss[f"{k}|{m}"] = c["mech"][m]["loss_out"]["mean"]
    for k, c in g.get("game_v2", {}).get("cells", {}).items():
        gm = c["mech"].get("fused_index", {}).get("games", {})
        for s, rec in gm.items():
            if s.startswith("full|") and "eps=0.002" in s:
                loss[f"v2cell {k}|fused_index|{s}"] = rec["loss"]["mean"]
    out["strategic_loss_naive_fusion"] = loss
    d["c3"] = out; save(d)
    log("c3", json.dumps(out)[:600])


def stage_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load()["c1"]["cells"]
    arms = [("honest_preempt", "honest (preempt)", "#999999", "o"), ("honest_hyst", "honest (preempt, best hysteresis; picks ~no switching)", "#4c78a8", "s"),
            ("priced_preempt", "priced (preempt, $\\lambda^*$)", "#e45756", "^"),
            ("priced_nopreempt", "priced (no preempt, $\\lambda^*$)", "#b02a2a", "v")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, model in zip(axs, ("v2", "legacy")):
        for j, (k, lab, col, mk) in enumerate(arms):
            xs, ys, lo, hi = [], [], [], []
            for i, r in enumerate(RATES):
                v = d[f"{model}|{r}"]["vs_nothrash"][k]
                xs.append(i + (j - 1.5) * 0.12); ys.append(v["mean"]); lo.append(v["mean"] - v["lo"]); hi.append(v["hi"] - v["mean"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, label=lab, capsize=3, ms=6)
        ax.axhline(0, color="k", lw=1)
        ax.text(0.02, 0.03, "0 = honest, no preemption (no thrash)", transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xticks(range(4)); ax.set_xticklabels([str(r) for r in RATES])
        ax.set_xlabel("degradation rate r (N=8)"); ax.set_title(f"{model} model")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("team mean health minus no-thrash baseline\n(paired, 32 holdout seeds, 95% CI)")
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Priced gain vs the no-thrash baseline (exploratory)", y=0.99)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(ROOT, "figures", "fig14a_priced_vs_nothrash.png"), dpi=150)
    plt.close(fig)
    c2 = load().get("c2")
    if c2:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        labs, vals, lo, hi = [], [], [], []
        r = c2["dp_logit_lin_nontrivial"]
        labs.append("DP per-step optimal\n(logit fit)"); vals.append(r["ratio"]); lo.append(r["ratio"] - r["ratio_ci95"][0]); hi.append(r["ratio_ci95"][1] - r["ratio"])
        for k, v in c2["reference_rules_same_states"].items():
            if v["ratio"] is not None and k.startswith("index"):
                labs.append("index_true\n(same states)"); vals.append(v["ratio"]); lo.append(0); hi.append(0)
        for k, tag in (("preempt", "priced $\\lambda^*$\n(preempt)"), ("nopreempt", "priced $\\lambda^*$\n(no preempt)")):
            labs.append(tag); vals.append(c2["lambda_star_n3"][k]["lambda_star"]); lo.append(0); hi.append(0)
        ax.bar(range(len(vals)), vals, yerr=[lo, hi], color=["#4c78a8", "#59a14f", "#e45756", "#b02a2a"][:len(vals)], capsize=4)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel("urgency per unit distance"); ax.set_title("Implied distance price, N=3, v2, r=.004 (exploratory)", fontsize=9)
        ax.text(0.98, 0.95, "priced lambda* is within noise of 0\n(flat tuning curve, paired gain vs lambda=0 n.s.)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "figures", "fig14b_implied_exchange_rate.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("c1", "all"): stage_c1()
    if st in ("c2", "all"): stage_c2()
    if st in ("c3", "all"): stage_c3()
    if st in ("figs", "all"): stage_figs()
