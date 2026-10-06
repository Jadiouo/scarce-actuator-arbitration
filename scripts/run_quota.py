"""Quota (linking) baseline vs audits.  Every GPU part must be submitted with `gpujob`.

  python3 scripts/run_quota.py --part tune --mech QM --W 200      # tune seeds 0-15, parameter grid, T=1e5   -> cache/tune_QM_200.npz
  python3 scripts/run_quota.py --part select                      # (CPU) picks parameters per (mech[,W], r) from tune files -> cache/sel.json
  python3 scripts/run_quota.py --part eval --mech QM              # selected params, eval seeds 5000-5031     -> cache/eval_QM.npz
  python3 scripts/run_quota.py --part base                        # OR, NAIVE, M0, M3C(stage2 params) on tune+eval seeds
  python3 scripts/run_quota.py --part short --T 10000             # selected params, eval seeds, shorter horizon
  python3 scripts/run_quota.py --part ctrl                        # discretisation-only control (no budget)
  python3 scripts/run_quota.py --part analyze                     # (CPU) results/quota.json + figures/fig20_*.png
--quick: T=1500, 4 seeds, separate cache dir, nothing written to results/.
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

CACHE = "/tmp/claude-1000/-home-lex-Documents-scarce-actuator-arbitration/65de289c-60a0-47e9-b793-f6f4bf98ae51/scratchpad/qcache"
TUNE = list(range(0, 16))
EVAL = list(range(5000, 5032))
RS = [0.5, 0.8, 0.9, 0.99]
WS = [50, 200, 1000]
T_MAIN = 100000
EPS_GAIN = 0.002
FIELDS = ["reg_a", "reg_b", "util_a", "util_b", "pen", "audits", "nraid", "nrew", "flags"]


def strategies():
    from arbitration.gpu.quota import win_x
    S = [("honest", {})]
    S += [(f"b={b}", dict(b=b)) for b in (0.05, 0.1, 0.2, 0.3, 0.5)]
    S += [(f"dz={d}", dict(dz=d)) for d in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0)]
    S += [(f"dzcrit={f}", dict(dzc=f)) for f in (0.5, 1.0, 1.5)]
    S += [("burst dz1 W10 H90", dict(dz=1.0, W=10, H=90)), ("burst dz1 W30 H170", dict(dz=1.0, W=30, H=170)),
          ("burst dz2 W10 H490", dict(dz=2.0, W=10, H=490))]
    S += [(f"edge_oracle em={e}", dict(em=e)) for e in (0.05, 0.1, 0.2, 0.4)]
    S += [(f"edge_uninf thr band={b}", dict(ue_mode=1, ue_band=b)) for b in (0.05, 0.1, 0.2)]
    S += [(f"edge_uninf dz band=0.1 dz={d}", dict(ue_mode=2, ue_band=0.1, ue_dz=d)) for d in (0.1, 0.3)]
    S += [(f"stealth m={m} f={f}", dict(ad_m=m, ad_f=f)) for m in (1.0, 2.0) for f in (0.5, 0.9)]
    S += [(f"quota_savemin_oracle lam={l}", dict(qs_mode=1, qs_lam=l)) for l in (0.3, 0.5, 0.7, 0.85)]
    S += [(f"quota_compress_uninf lam={l} pi={p}", dict(qs_mode=2, qs_lam=l, qs_x=win_x(p))) for l in (0.5, 0.7) for p in (0.5, 0.8)]
    return S


def sset(names_dicts, ad_h=None):
    out = []
    for n, d in names_dicts:
        d = dict(d)
        if "ad_m" in d and ad_h is not None:
            d["ad_h"] = ad_h
        out.append(d)
    return out


def qgrid(mech, W):
    if mech == "QM":
        return [dict(QW=float(W), nb=float(nb)) for nb in (5, 10, 20, 40, 80)]
    if mech == "QE":
        return [dict(QW=float(W), nb=float(nb), nc=float(nb), qh=h, qa=a) for nb in (5, 10) for h, a in ((2.5, 1.0), (3.0, 2.0), (4.0, 2.0), (5.0, 3.0))]
    out = []
    for nb in (10, 20):
        for nc in (2, 5):
            for h, a in ((1.0, 0.0), (2.0, 1.0)):
                out.append(dict(QW=float(W), nb=float(nb), nc=float(nc), qh=h, qa=a))
    return out


def m4grid():
    return [dict(eps=e, L=float(L), rw=1 / 3) for e in (0.02, 0.05, 0.1, 0.2) for L in (100, 2500)]


def m3c_params():
    js = json.load(open(os.path.join(ROOT, "results/stage2.json")))["main"]["M3c"]["100000"]
    return {r: dict(p=js[str(r)]["param"]["p"], tol=float(js[str(r)]["param"]["tol"]), L=float(js[str(r)]["param"]["L"])) for r in RS}


def run(mech, pairs, S, seeds, T, chunk_elems=40000):
    """pairs: list of (param, rdict); returns dict field -> array [pairs, strat, seeds, ...]; chunks over pairs."""
    import torch
    from arbitration.gpu import quota as Qm
    per = max(1, chunk_elems // (len(S) * len(seeds)))
    res = {f: [] for f in FIELDS}
    for i in range(0, len(pairs), per):
        pp = pairs[i:i + per]
        el = Qm.build_pairs(pp, S, seeds, dev=os.environ.get('QDEV', 'cuda'))
        shp = el["shape"]
        o = Qm.simulate(mech, T, el, seeds)
        for f in FIELDS:
            a = o[f].cpu().numpy()
            res[f].append(a.reshape(*shp, *a.shape[1:]))
        print(f"  {mech} chunk {i}/{len(pairs)} done", flush=True)
    return {f: np.concatenate(v, 0) for f, v in res.items()}


def save(name, cache, res, meta):
    os.makedirs(cache, exist_ok=True)
    np.savez(os.path.join(cache, name + ".npz"), **res)
    json.dump(meta, open(os.path.join(cache, name + ".json"), "w"), default=float)


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0]


def load(name, cache):
    z = np.load(os.path.join(cache, name + ".npz"))
    return {f: z[f] for f in FIELDS}, json.load(open(os.path.join(cache, name + ".json")))


# ------------------------------------------------------------------------------------------------ parts
def part_tune(a, cache, seeds, T):
    SN = strategies(); S = [d for _, d in SN]
    mech = a.mech
    if mech == "M4":
        grid = m4grid(); tag = "tune_M4"
    else:
        grid = qgrid(mech, a.W); tag = f"tune_{mech}_{a.W}"
    pairs = [(g, dict(rho=r)) for g in grid for r in RS]
    S = sset(SN, ad_h=6.0)
    t0 = time.time()
    res = run(mech, pairs, S, seeds, T)
    res = {k: v.reshape(len(grid), len(RS), *v.shape[1:]) for k, v in res.items()}
    save(tag, cache, res, dict(mech=mech, W=a.W, grid=grid, strat=[n for n, _ in SN], T=T, seeds=seeds))
    print(tag, f"{time.time()-t0:.0f}s", flush=True)


def bestresp(res, ri, pi, T, names, uninformed=False):
    ub = res["util_b"][pi, ri, :, :, 1] / T          # [S, N]
    g = ub[1:] - ub[:1]
    gm = g.mean(1)
    if uninformed:
        gm = np.where([("oracle" in nm) for nm in names[1:]], -9.0, gm)
    return int(gm.argmax()) + 1, g


def part_select(a, cache):
    T = T_MAIN
    sel = {}
    for mech, Ws in (("QM", WS), ("QT", WS), ("QE", WS), ("M4", [0])):
        for W in Ws:
            tag = "tune_M4" if mech == "M4" else f"tune_{mech}_{W}"
            if not os.path.exists(os.path.join(cache, tag + ".npz")):
                print("missing", tag); continue
            res, meta = load(tag, cache)
            for ri, r in enumerate(RS):
                for var in (("o",) if mech == "M4" else ("o", "u")):
                    rows = []
                    for pi, g in enumerate(meta["grid"]):
                        sb, gg = bestresp(res, ri, pi, T, meta["strat"], uninformed=(var == "u"))
                        Rbr = float(res["reg_b"][pi, ri, sb, :].mean() / T)
                        Gmax = float(gg.mean(1).max())
                        rows.append(dict(pi=pi, param=g, R_br=Rbr, Gmax=Gmax, br=meta["strat"][sb],
                                         R_hon=float(res["reg_b"][pi, ri, 0, :].mean() / T)))
                    if mech == "M4":
                        feas = [x for x in rows if x["Gmax"] <= EPS_GAIN]
                        best = min(feas, key=lambda x: x["R_br"]) if feas else min(rows, key=lambda x: x["Gmax"])
                        best = dict(best, feasible=bool(feas))
                    else:
                        best = min(rows, key=lambda x: x["R_br"])
                    sel[f"{mech}|{W}|{r}|{var}"] = dict(best=best, all=rows)
    json.dump(sel, open(os.path.join(cache, "sel.json.tmp"), "w"), default=float, indent=1)
    os.replace(os.path.join(cache, "sel.json.tmp"), os.path.join(cache, "sel.json"))
    for k, v in sel.items():
        b = v["best"]
        print(k, b["param"], "R_br %.4f Gmax_tune %.4f R_hon %.4f br=%s" % (b["R_br"], b["Gmax"], b["R_hon"], b["br"]))


def sel_pairs(mech, cache, Ws):
    sel = json.load(open(os.path.join(cache, "sel.json")))
    pairs, keys = [], []
    for var in (("o",) if mech == "M4" else ("o", "u")):
        for W in Ws:
            for r in RS:
                pairs.append((sel[f"{mech}|{W}|{r}|{var}"]["best"]["param"], dict(rho=r))); keys.append((mech, W, r, var))
    return pairs, keys


def part_eval(a, cache, seeds, T, name=None):
    SN = strategies(); S = sset(SN, ad_h=6.0)
    mech = a.mech
    pairs, keys = sel_pairs(mech, cache, [0] if mech == "M4" else WS)
    res = run(mech, pairs, S, seeds, T)
    save(name or f"eval_{mech}", cache, res, dict(mech=mech, keys=keys, strat=[n for n, _ in SN], T=T, seeds=seeds))


def part_base(a, cache, T):
    SN = strategies()
    P3 = m3c_params()
    out = {}
    for mech in ("OR", "NAIVE", "M0", "M3C"):
        pairs, keys = [], []
        for r in RS:
            p = dict(P3[r]) if mech == "M3C" else {}
            pairs.append((p, dict(rho=r))); keys.append((mech, 0, r))
        S = sset(SN, ad_h=(P3[RS[0]]["tol"] if mech == "M3C" else None))
        if mech == "M3C":     # stealth strategy needs the mechanism's own h : build per r below
            res_l = []
            for (p, rr), r in zip(pairs, RS):
                Sr = sset(SN, ad_h=P3[r]["tol"])
                res_l.append(run(mech, [(p, rr)], Sr, TUNE + EVAL, T))
            res = {f: np.concatenate([x[f] for x in res_l], 0) for f in FIELDS}
        else:
            res = run(mech, pairs, S if mech != "OR" else S[:1], TUNE + EVAL, T)
        save(f"base_{mech}", cache, res, dict(mech=mech, keys=keys, strat=[n for n, _ in SN] if mech != "OR" else ["honest"], T=T,
                                              seeds=TUNE + EVAL, params=P3 if mech == "M3C" else {}))


def part_short(a, cache, T):
    """selected params at shorter horizon (eval seeds); M3C stage2 params; M4 selected."""
    SN = strategies()
    P3 = m3c_params()
    for mech in ("QM", "QT", "QE", "M4", "M3C", "OR", "M0"):
        if mech == "M3C":
            pairs, keys = [(dict(P3[r]), dict(rho=r)) for r in RS], [("M3C", 0, r) for r in RS]
        elif mech in ("OR", "M0"):
            pairs, keys = [({}, dict(rho=r)) for r in RS], [(mech, 0, r) for r in RS]
        else:
            pairs, keys = sel_pairs(mech, cache, [0] if mech == "M4" else WS)
        if mech == "M3C":
            res_l = [run(mech, [pr], sset(SN, ad_h=P3[k[2]]["tol"]), EVAL, T) for pr, k in zip(pairs, keys)]
            res = {f: np.concatenate([x[f] for x in res_l], 0) for f in FIELDS}
        else:
            S = sset(SN, ad_h=6.0)
            res = run(mech, pairs, S[:1] if mech in ("OR",) else S, EVAL, T)
        save(f"short{T}_{mech}", cache, res, dict(mech=mech, keys=keys, strat=[n for n, _ in SN], T=T, seeds=EVAL))


def part_ctrl(a, cache, T):
    """discretisation-only control (no budget): QM with QW=0, honest only, several nb."""
    SN = strategies()
    nbs = [5, 10, 20, 40, 100]
    pairs = [(dict(QW=0.0, nb=float(nb)), dict(rho=r)) for nb in nbs for r in RS]
    res = run("QM", pairs, [SN[0][1]], EVAL, T)
    save("ctrl_disc", cache, res, dict(nbs=nbs, T=T, seeds=EVAL))


# ------------------------------------------------------------------------------------------------ analysis
def summarize(res, meta, T, seeds_sel, tune_sel=None, M0R=None, ORU=None, prefix=None):
    """res arrays [pairs,strat,seeds,..] ; returns list of dicts per pair"""
    out = []
    names = meta["strat"]
    ns = len(names)
    for pi, key in enumerate(meta["keys"]):
        r = key[2]
        ub = res["util_b"][pi, :, :, 1] / T
        n = ub.shape[1]
        gain = ub[1:] - ub[:1]
        gm = gain.mean(1)
        sm = int(gm.argmax()) + 1
        qs_mask = np.array([nm.startswith("quota_") for nm in names[1:]])
        gm_nq = np.where(qs_mask, -9, gm)
        sn = int(gm_nq.argmax()) + 1
        d = dict(mech=key[0], W=key[1], r=r, variant=(key[3] if len(key) > 3 else None))
        orc = np.array([("oracle" in nm) for nm in names[1:]])
        gu = np.where(orc, -9, gm); su = int(gu.argmax()) + 1
        d["Gmax_uninformed"] = ci(gain[su - 1]); d["Gmax_uninformed_strategy"] = names[su]
        d["R_hon_b"] = ci(res["reg_b"][pi, 0] / T); d["R_hon_a"] = ci(res["reg_a"][pi, 0] / T)
        d["Gmax"] = ci(gain[sm - 1]); d["Gmax_strategy"] = names[sm]
        d["Gmax_nonquota"] = ci(gain[sn - 1]); d["Gmax_nonquota_strategy"] = names[sn]
        d["R_at_Gmax_b"] = ci(res["reg_b"][pi, sm] / T); d["R_at_Gmax_a"] = ci(res["reg_a"][pi, sm] / T)
        if tune_sel is not None:
            sb = tune_sel[pi]
            d["BR_strategy_tune"] = names[sb]; d["Gbr"] = ci(gain[sb - 1])
            d["R_br_b"] = ci(res["reg_b"][pi, sb] / T); d["R_br_a"] = ci(res["reg_a"][pi, sb] / T)
        d["audits_per_round"] = ci(res["audits"][pi, 0] / T)
        d["raid_waste_frac"] = ci(res["nraid"][pi, 0] / T)
        d["false_punish_frac"] = ci(res["pen"][pi, 0].mean(-1) / T)
        d["rewrite_frac_honest"] = ci(res["nrew"][pi, 0].mean(-1) / T)
        d["honest_util_agent1"] = ci(ub[0])
        if ORU is not None:
            d["honest_agent1_util_loss_vs_oracle"] = ci(ORU[r] - ub[0])
        d["gains_by_strategy"] = {nm: float(g) for nm, g in zip(names[1:], gm)}
        out.append(d)
    return out


def part_analyze(a, cache, outdir):
    sel = json.load(open(os.path.join(cache, "sel.json")))
    T = T_MAIN
    base = {m: load(f"base_{m}", cache) for m in ("OR", "NAIVE", "M0", "M3C")}
    ORU = {r: base["OR"][0]["util_b"][ri, 0, len(TUNE):, 1] / T for ri, r in enumerate(RS)}
    main = []
    # base mechanisms: tune seeds slice to pick BR strategy, eval slice to evaluate
    for m in ("OR", "NAIVE", "M0", "M3C"):
        res, meta = base[m]
        if m == "OR":
            res_e = {k: v[:, :, len(TUNE):] for k, v in res.items()}
            names = meta["strat"]
            for pi, key in enumerate(meta["keys"]):
                ub = res_e["util_b"][pi, 0, :, 1] / T
                main.append(dict(mech="OR", W=0, r=key[2], R_hon_b=ci(res_e["reg_b"][pi, 0] / T), R_hon_a=ci(res_e["reg_a"][pi, 0] / T),
                                 honest_util_agent1=ci(ub), honest_agent1_util_loss_vs_oracle=[0.0, 0.0]))
            continue
        res_t = {k: v[:, :, :len(TUNE)] for k, v in res.items()}
        res_e = {k: v[:, :, len(TUNE):] for k, v in res.items()}
        tsel = []
        for pi in range(len(meta["keys"])):
            ubt = res_t["util_b"][pi, :, :, 1] / T
            tsel.append(int((ubt[1:] - ubt[:1]).mean(1).argmax()) + 1)
        main += summarize(res_e, meta, T, None, tune_sel=tsel, ORU=ORU)
    for m in ("QM", "QT", "QE", "M4"):
        if not os.path.exists(os.path.join(cache, f"eval_{m}.npz")):
            continue
        res, meta = load(f"eval_{m}", cache)
        tsel = []
        for (mm, W, r, var) in meta["keys"]:
            s = sel[f"{mm}|{W}|{r}|{var}"]
            tsel.append(meta["strat"].index(s["best"]["br"]))
        rows = summarize(res, meta, T, None, tune_sel=tsel, ORU=ORU)
        for (mm, W, r, var), row in zip(meta["keys"], rows):
            row["param"] = sel[f"{mm}|{W}|{r}|{var}"]["best"]["param"]; row["variant"] = var
        main += rows
    # short horizons
    short = {}
    for Ts in (3000, 10000):
        for m in ("QM", "QT", "QE", "M4", "M3C", "OR", "M0"):
            fn = f"short{Ts}_{m}"
            if not os.path.exists(os.path.join(cache, fn + ".npz")):
                continue
            res, meta = load(fn, cache)
            if m == "OR":
                continue
            orf = f"short{Ts}_OR"
            oru = None
            if os.path.exists(os.path.join(cache, orf + ".npz")):
                ro, _ = load(orf, cache)
                oru = {r: ro["util_b"][ri, 0, :, 1] / Ts for ri, r in enumerate(RS)}
            rows = summarize(res, meta, Ts, None, tune_sel=None, ORU=oru)
            short.setdefault(str(Ts), []).extend(rows)
    ctrl = None
    if os.path.exists(os.path.join(cache, "ctrl_disc.npz")):
        res, meta = load("ctrl_disc", cache)
        ctrl = [dict(nb=nb, r=r, R_b=ci(res["reg_b"][i * len(RS) + ri, 0] / T))
                for i, nb in enumerate(meta["nbs"]) for ri, r in enumerate(RS)]
    sel_out = {k: dict(param=v["best"]["param"], R_br_tune=v["best"]["R_br"], Gmax_tune=v["best"]["Gmax"], br_tune=v["best"]["br"],
                       feasible=v["best"].get("feasible")) for k, v in sel.items()}
    js = dict(meta=dict(T=T, rhos=RS, tau=1, tune_seeds="0-15", eval_seeds="5000-5031", Ws=WS, K=3,
                        design=DESIGN, selection_rule=("Q: per (mech,W,r) the parameter minimising regret R_b at the tune-seed best-response liar; "
                                                       "M4: feasible (tune Gmax<=0.002) then min R_br else min Gmax; M3C: stage-2 selected parameters (results/stage2.json)"),
                        G_definition="max over strategies of paired (same seed) gain of agent-1 mean utility/round over honest agent 1 in the same mechanism; "
                                     "Gmax is a max over 45 strategies (upward biased); Gbr = strategy fixed on tune seeds, evaluated on eval seeds"),
              main=main, selection=sel_out, short=short, ctrl_disc=ctrl)
    if outdir:
        json.dump(js, open(outdir, "w"), indent=1, default=float)
        figures(js)
    return js


def figures(js):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(ROOT, "figures"); os.makedirs(fd, exist_ok=True)
    main = js["main"]

    def get(mech, W, r, var=None):
        for m in main:
            if m["mech"] == mech and m["W"] == W and m["r"] == r and (var is None or m.get("variant") == var):
                return m
    cols = {"M0": "#888", "NAIVE": "#bbb", "M3C": "#2ca02c", "M4": "#9467bd", "QM": "#1f77b4", "QT": "#d62728", "QE": "#ff7f0e"}
    cats = [("M0", 0), ("NAIVE", 0), ("M3C", 0), ("M4", 0), ("QM", 50), ("QM", 200), ("QM", 1000), ("QT", 50), ("QT", 200), ("QT", 1000), ("QE", 50), ("QE", 200), ("QE", 1000)]
    fig, ax = plt.subplots(3, len(RS), figsize=(3.6 * len(RS), 9), sharex=True)
    for ri, r in enumerate(RS):
        xs = np.arange(len(cats))
        for i, (m, W) in enumerate(cats):
            row = get(m, W, r, "o" if m in ("QM", "QT", "QE", "M4") else None)
            if row is None or "R_br_b" not in row:
                continue
            ax[0, ri].bar(i, row["R_br_b"][0], yerr=row["R_br_b"][1], color=cols[m], alpha=1.0 if m != "M0" else .6)
            ax[0, ri].bar(i, row["R_hon_b"][0], color="k", alpha=.35, width=.35)
            ax[1, ri].errorbar(i, row["Gmax"][0], row["Gmax"][1], fmt="o", color=cols[m])
            ax[1, ri].errorbar(i, row["Gmax_uninformed"][0], row["Gmax_uninformed"][1], fmt="s", mfc="none", color=cols[m])
            ax[2, ri].bar(i, row["honest_agent1_util_loss_vs_oracle"][0], yerr=row["honest_agent1_util_loss_vs_oracle"][1], color=cols[m])
        ax[0, ri].set_title(f"r=rho^tau={r}", fontsize=9)
        ax[0, ri].set_yscale("log"); ax[1, ri].axhline(0, color="k", lw=.8); ax[1, ri].axhline(EPS_GAIN, color="k", lw=.5, ls=":")
        ax[2, ri].set_xticks(range(len(cats))); ax[2, ri].set_xticklabels([f"{m}{'' if W == 0 else W}" for m, W in cats], rotation=70, fontsize=7)
    ax[0, 0].set_ylabel("R/T at best-response liar (bar), honest R/T (dark)"); ax[1, 0].set_ylabel("max lying gain G  (o=all, []=non-oracle)")
    ax[2, 0].set_ylabel("honest-agent utility loss vs oracle")
    ax[1, 0].set_yscale("symlog", linthresh=1e-3)
    fig.suptitle("fig20a  quota (QM/QT, W=block) vs audits (M3C, M4) vs M0/NAIVE; T=1e5, eval seeds 5000-5031", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig20a_quota_vs_audit.png"), dpi=130); plt.close(fig)
    # fig20b : honest efficiency / rewrite fraction vs W
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for m, ls in (("QM", "-"), ("QT", "--"), ("QE", ":")):
        for ri, r in enumerate(RS):
            rows = [get(m, W, r, "o") for W in WS]
            ax[0].errorbar(WS, [x["R_hon_b"][0] for x in rows], [x["R_hon_b"][1] for x in rows], ls=ls, marker="o", ms=3, color=plt.cm.viridis(ri / 3), label=f"{m} r={r}" if m == "QM" else None)
            ax[1].plot(WS, [x["rewrite_frac_honest"][0] for x in rows], ls=ls, marker="o", ms=3, color=plt.cm.viridis(ri / 3))
    if js.get("ctrl_disc"):
        for ri, r in enumerate(RS):
            rr = [(c["nb"], c["R_b"][0]) for c in js["ctrl_disc"] if c["r"] == r]
            ax[2].plot([x[0] for x in rr], [x[1] for x in rr], marker="o", ms=3, color=plt.cm.viridis(ri / 3), label=f"r={r}")
        ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("bins nb"); ax[2].set_title("discretisation-only loss (no budget)", fontsize=8); ax[2].legend(fontsize=6)
    for a_, t_ in zip(ax[:2], ("honest R/T (solid QM, dashed QT)", "fraction of honest reports rewritten")):
        a_.set_xscale("log"); a_.set_xlabel("block length W"); a_.set_title(t_, fontsize=8)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=6)
    fig.suptitle("fig20b  where Q's efficiency loss comes from", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig20b_quota_efficiency_loss.png"), dpi=130); plt.close(fig)
    # fig20c : horizon sensitivity
    fig, ax = plt.subplots(2, len(RS), figsize=(3.6 * len(RS), 6), sharex=True)
    Tl = [3000, 10000, 100000]
    def at(Tv, mech, W, r, var):
        if Tv == 100000:
            return get(mech, W, r, var)
        for m in js["short"].get(str(Tv), []):
            if m["mech"] == mech and m["W"] == W and m["r"] == r and (var is None or m.get("variant") == var):
                return m
    for ri, r in enumerate(RS):
        for (mech, W, var, lab) in (("M0", 0, None, "M0"), ("M3C", 0, None, "M3C"), ("M4", 0, "o", "M4"), ("QM", 1000, "o", "QM W1000"), ("QM", 200, "o", "QM W200"), ("QT", 1000, "o", "QT W1000"), ("QE", 1000, "o", "QE W1000"), ("QE", 200, "o", "QE W200")):
            rows = [at(Tv, mech, W, r, var) for Tv in Tl]
            if any(x is None for x in rows):
                continue
            ax[0, ri].errorbar(Tl, [x["R_hon_b"][0] for x in rows], [x["R_hon_b"][1] for x in rows], marker="o", ms=3, label=lab)
            ax[1, ri].errorbar(Tl, [x["Gmax_uninformed"][0] for x in rows], [x["Gmax_uninformed"][1] for x in rows], marker="o", ms=3)
        ax[0, ri].set_xscale("log"); ax[0, ri].set_yscale("log"); ax[0, ri].set_title(f"r={r}", fontsize=9); ax[1, ri].axhline(0, color="k", lw=.8)
        ax[1, ri].set_xlabel("T")
    ax[0, 0].set_ylabel("honest R/T"); ax[1, 0].set_ylabel("G (non-oracle strategies, max)"); ax[0, 0].legend(fontsize=6)
    fig.suptitle("fig20c  horizon sensitivity (params fixed at T=1e5 selections)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig20c_horizon.png"), dpi=130); plt.close(fig)


DESIGN = """DESIGN (written before the full run).
Mechanism Q (linking / quota).  Reports are mapped to quantile bins with the known marginal CDFs (nb equiprobable bins per agent; finite report space as in J-S).
QM (marginal quota; Jackson-Sonnenschein 2007 long version, 'The Linking Mechanisms' pp.20-21 and the over-time version with exhaustible budgets, Corollary 2, pp.30-31):
 in every block of W periods an agent may announce bin b exactly floor((b+1)W/nb)-floor(bW/nb) times; an announcement whose budget is exhausted is rewritten to the nearest bin
 with budget (ties -> lower bin).  Dispatcher: highest F_i^-1(bin midpoint).  Honest agents announce truth until the budget binds ('approximately truthful', p.21 / Cor.2 discussion).
 Not implemented: J-S's modification of g^K that realigns decision probabilities for P^K vs P, and the n>=3 anti-collusion adjustment (p.21, proof of Thm 1); the dispatcher uses the plain argmax rule (an efficient f).
QT (transition quota; Escobar-Toikka 2013, Credible Reporting Mechanism, eq. (5.1), Lemma 5.1): for the class c (nc coarse classes) of the agent's previous announced bin the empirical frequency of the next announced
 bin must satisfy N(c,b')+1 <= P(b'|c)(n_c+1) + h*sqrt((n_c+1)P(1-P)) + a, with P from the true Gaussian-copula AR(1) transition (oracle rho, as M3C); violating announcements are rewritten to the nearest allowed bin; counts restart every W periods.
 Differences to E-T: (i) E-T test is a sup-norm band b_k->0 on transition frequencies conditioned on the previous-period report PROFILE (and, for n>=3, on others' current reports) and failure replaces the agent's reports by simulated reports FOREVER; ours conditions on the agent's own previous report class, uses a one-sided count band per cell and rewrites report-by-report with per-block reset (a softer, per-period version, closer to J-S budgets). (ii) E-T is an equilibrium result for patient players with discounting; we test a fixed-horizon loss.
 (iii) our types are independent across agents (so conditioning on own history suffices for 'independence of transitions').
Parameters tuned on seeds 0-15 per (mech,W,r) (rule in meta.selection_rule); evaluated on seeds 5000-5031 (never used for selection).
Comparators on identical seeds/type paths: OR (oracle, honest argmax, R=0), NAIVE (trust reports), M0, M3C (dispatch CUSUM, stage-2 parameters), M4 (surprise raid including the winner with prob rw=1/3, as in arbitration/gpu/frontier.py; tuned here).
Strategies: all of frontier.py (const b, dz, dz at critical multiples, burst, oracle edge, uninformed edge, stealth) plus quota-specific: save-min (oracle: win with the smallest remaining bin that beats others' submitted bins when own u>=lam, otherwise burn the lowest remaining bin) and uninformed compress.
Not done: Q+audit combination.  Everything else, including known-rho oracle for QT/M3C and one-sided tests, is noted in the report."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--mech", default="QM")
    ap.add_argument("--W", type=int, default=200)
    ap.add_argument("--T", type=int, default=T_MAIN)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    cache = a.cache + ("_quick" if a.quick else "")
    T = 1500 if a.quick else a.T
    tune_seeds, eval_seeds = (TUNE[:3], EVAL[:3]) if a.quick else (TUNE, EVAL)
    if a.quick:
        globals()["TUNE"], globals()["EVAL"] = tune_seeds, eval_seeds
        globals()["T_MAIN"] = T
    if a.part == "tune":
        part_tune(a, cache, tune_seeds, T)
    elif a.part == "select":
        part_select(a, cache)
    elif a.part == "eval":
        part_eval(a, cache, eval_seeds, T)
    elif a.part == "base":
        part_base(a, cache, T)
    elif a.part == "short":
        part_short(a, cache, T)
    elif a.part == "ctrl":
        part_ctrl(a, cache, T)
    elif a.part == "analyze":
        part_analyze(a, cache, None if a.quick else os.path.join(ROOT, "results/quota.json"))
        print("analyzed")


if __name__ == "__main__":
    main()
