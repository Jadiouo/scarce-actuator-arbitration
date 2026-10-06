"""Stage 2b driver (docs/stage2b_design.md).  GPU parts must be submitted via gpujob.
  python3 scripts/run_stage2b.py --part smoke|consist|sweep|select|eval|rho|analyze  [--mech M --chunk i --fam F]
Raw per-part results go to CACHE as npz; `analyze` (CPU) merges them into results/stage2b.json + figures/fig19_*.png."""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from pathlib import Path
CACHE = os.environ.get("ARB_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "results" / "cache" / "run_stage2b"))
TUNE, EVAL = list(range(0, 16)), list(range(5000, 5032))
RS = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
R_TAU5 = [0.8, 0.9, 0.99]
T_MAIN = 20000
ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1]
FAMS = ["M3T", "M3C", "M3G", "M3W", "M4"]
INF = float("inf")


def strategies():
    S = [("honest", "honest", {})]
    for b in (.02, .05, .1, .2, .4): S.append((f"b{b}", "const", dict(b=b)))
    for d in (.01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1.0, 2.0): S.append((f"dz{d}", "dz", dict(dz=d)))
    for f in (.5, .8, 1.0, 1.25, 2.0): S.append((f"dzc{f}", "dzc", dict(dzc=f)))
    for d, W, H in ((2.0, 5, 95), (2.0, 10, 90), (2.0, 10, 490), (2.0, 30, 470), (1.0, 10, 90), (1.0, 20, 180)):
        S.append((f"burst dz{d} W{W} H{H}", "burst", dict(dz=d, W=W, H=H)))
    for em in (.02, .05, .1, .2, .4): S.append((f"edgeO{em}", "edge_or", dict(em=em)))
    for q in (.5, .8):
        for bd in (.05, .1, .2): S.append((f"edgeU-th band{bd} q{q}", "edge_un", dict(ue_mode=1, ue_band=bd, ue_q=q)))
    for bd in (.1, .2):
        for d in (.25, .5): S.append((f"edgeU-dz{d} band{bd}", "edge_un", dict(ue_mode=2, ue_band=bd, ue_q=.7, ue_dz=d)))
    for m, f, cc in ((1.5, .8, 1.0), (1.5, .8, 3.0), (3.0, .5, 3.0), (3.0, .8, 3.0), (1.0, .9, 1.0), (2.0, .8, 2.0)):
        S.append((f"adapt m{m} f{f} c{cc}", "adapt", dict(ad_m=m, ad_f=f, ad_c=cc)))
    return S


def grids():
    G = {}
    G["M3T"] = [dict(p=p, tol=k, L=L) for p in (.1, .3) for k in (.5, 1, 1.5, 2, 2.5, 3, 3.5, 4) for L in (20, 100, 500, 2500)]
    G["M3C"] = [dict(p=.1, tol=h, L=L, kref=kr) for kr in (.25, .5, 1.0) for h in (2, 3, 4, 6, 8, 12, 16) for L in (20, 100, 500, 2500)]
    for g in G["M3C"]: g["ad_h"] = g["tol"]
    G["M3G"] = [dict(p=.1, tol=k, L=L, nmax=n, ad_h=k * math.sqrt(n / 2)) for k in (2, 2.5, 3, 3.5, 4) for n in (10, 30, 100) for L in (20, 100, 500, 2500)]
    G["M3W"] = [dict(p=.1, tol=k, L=L, nmax=n, ad_h=k * math.sqrt(n / 2)) for k in (1.5, 2, 2.5, 3) for n in (10, 30, 100) for L in (20, 100, 500, 2500)]
    G["M4"] = [dict(eps=e, L=L, rw=rw) for e in (.005, .01, .02, .05, .1, .2) for L in (20, 100, 500, 2500) for rw in (1 / 3, 1.0)]
    for fam in ("M3T",):
        for g in G[fam]: g["ad_h"] = INF
    return G


def mech_of(fam): return fam if fam != "M4" else "M4"


def rdict(r, tau=1, **kw):
    return dict(rho=r ** (1.0 / tau), tau=float(tau), **kw)


def run_pairs(mech, T, pairs, S, seeds, maxB=120000):
    import torch
    from arbitration.gpu import frontier as FR
    per = max(1, maxB // (len(S) * len(seeds)))
    outs = []
    for i in range(0, len(pairs), per):
        el = FR.build_pairs(pairs[i:i + per], [s[2] for s in S], seeds)
        o = FR.simulate(mech, T, el, seeds)
        n, ns, nn = el["shape"]
        f = lambda a: a.cpu().numpy().reshape(n, ns, nn)
        outs.append(dict(u1=f(o["util_b"][:, 1] / T), alpha=f(o["pen"].mean(1) / T), reg=f(o["reg_b"] / T),
                         nraid=f(o["nraid"] / T), audits=f(o["audits"] / T), fl1=f(o["flags"][:, 1]), fl=f(o["flags"].sum(1)), au_n=f(o["audits"])))
    return {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}


def pair_r(g, r, tau=1, **kw):
    return (g, rdict(r, tau, **kw))


# ------------------------------------------------------------------ parts
def part_smoke():
    S = strategies(); t0 = time.time()
    G = grids()
    for fam in FAMS:
        pairs = [pair_r(G[fam][0], .9), pair_r(G[fam][-1], .5), pair_r(G[fam][1], .9, 5)]
        pairs[2] = (G[fam][1], rdict(.9, 5))
        o = run_pairs(fam, 300, pairs, S, [0, 1], maxB=200000)
        print(fam, "ok", {k: float(v.mean()) for k, v in o.items()}, f"{time.time()-t0:.0f}s", flush=True)
    # speed
    t0 = time.time()
    o = run_pairs("M3C", 500, [pair_r(g, .9) for g in G["M3C"][:20]] * 1, S, TUNE)
    B = 20 * len(S) * len(TUNE)
    print(f"speed: B={B} T=500 -> {time.time()-t0:.1f}s", flush=True)
    # calibration: honest per-audit flag rate vs Phi(-kappa), M3T, r=0.9 and tau=5
    import torch
    from arbitration.gpu import frontier as FR
    for tau in (1, 5):
        pairs = [(dict(p=.5, tol=k, L=0.0), rdict(.9, tau)) for k in (1., 2., 3.)]
        o = run_pairs("M3T", 3000, pairs, [("h", "x", {})], list(range(16)))
        for i, k in enumerate((1., 2., 3.)):
            na = o["au_n"][i, 0].mean()
            nf = o["fl"][i, 0].mean() / na
            print("calib tau", tau, "kappa", k, "audits", na, "flag1 per audit(agent1 only)", nf, "(all agents n/a) Phi", 0.5 * math.erfc(k / math.sqrt(2)), flush=True)


def part_consist():
    """one-off reproduction of the red team's numbers (T=1e5, seeds 1000-1031)."""
    S = [("honest", "x", {}), ("edge0.05(legacy)", "x", dict(em=.05, leg=1.0)), ("edge0.1(legacy)", "x", dict(em=.1, leg=1.0)),
         ("dz0.25", "x", dict(dz=.25)), ("dz0.05", "x", dict(dz=.05)), ("b0.3", "x", dict(b=.3)), ("dz1.0", "x", dict(dz=1.0)),
         ("edgeO0.05(elig)", "x", dict(em=.05)), ("b0.3 timing", "x", dict(b=.3, timing=1.0))]
    seeds = list(range(1000, 1032)); out = {}
    T = 100000
    pairs = [(dict(p=.1, tol=6, L=2500, kref=.5), rdict(.9)), (dict(p=.1, tol=6, L=2500, kref=.5), rdict(.8))]
    o = run_pairs("M3C", T, pairs, S, seeds)
    out["M3C_h6_L2500"] = dict(names=[s[0] for s in S], r=[.9, .8], u1=o["u1"], alpha=o["alpha"][:, 0], reg=o["reg"])
    pairs = []
    cfg = [(.05, 2500), (.2, 20), (.2, 2500)]
    for r in (1.0, 0.99):
        for e, L in cfg:
            for rw in (0.0, 1 / 3, 1.0):
                pairs.append((dict(eps=e, L=L, rw=rw), rdict(r)))
    o = run_pairs("M4", T, pairs, S, seeds)
    out["M4"] = dict(names=[s[0] for s in S], pairs=[(p, r["rho"]) for p, r in pairs], u1=o["u1"], nraid=o["nraid"][:, 0], fl1=o["fl1"], reg=o["reg"])
    os.makedirs(CACHE, exist_ok=True)
    np.save(f"{CACHE}/consist.npy", out, allow_pickle=True)
    print("consist saved", flush=True)


def part_sweep(fam, chunk, tau5=False):
    G = grids()[fam]; S = strategies()
    rl = [(r, 5) for r in R_TAU5] if tau5 else [(r, 1) for r in RS]
    allp = [(g, rdict(r, tau)) for g in G for r, tau in rl]
    per = 26 * len(rl) if False else 24 * len(rl)
    sub = allp[chunk * per:(chunk + 1) * per]
    if not sub:
        print("empty chunk"); return
    t0 = time.time()
    o = run_pairs(fam, T_MAIN, sub, S, TUNE, maxB=len(sub) * len(S) * len(TUNE) + 1)
    os.makedirs(CACHE, exist_ok=True)
    tag = f"sweep_{fam}{'_t5' if tau5 else ''}_{chunk}"
    np.save(f"{CACHE}/{tag}.npy", dict(pairs=[(p, r) for p, r in sub], **o), allow_pickle=True)
    print(tag, f"B={len(sub)*len(S)*len(TUNE)} {time.time()-t0:.0f}s", flush=True)


def nchunks(fam, tau5=False):
    nr = 3 if tau5 else 6
    return math.ceil(len(grids()[fam]) * nr / (24 * nr))


def load_sweep(fam, tau5=False):
    ps, arr = [], {}
    for c in range(nchunks(fam, tau5)):
        d = np.load(f"{CACHE}/sweep_{fam}{'_t5' if tau5 else ''}_{c}.npy", allow_pickle=True).item()
        ps += d["pairs"]
        for k in ("u1", "alpha", "reg", "nraid", "audits", "fl1", "fl", "au_n"):
            arr.setdefault(k, []).append(d[k])
    return ps, {k: np.concatenate(v, 0) for k, v in arr.items()}


def tune_stats(fam, tau5=False):
    """per (param,r) item: x (alpha or raid cost), Gt, best strategy index (tune), array of paired gains."""
    ps, a = load_sweep(fam, tau5)
    gain = a["u1"][:, 1:, :] - a["u1"][:, :1, :]          # [n,S-1,N]
    gm = gain.mean(-1)
    x = (a["nraid"][:, 0].mean(-1) if fam == "M4" else a["alpha"][:, 0].mean(-1))
    return ps, x, gm.max(1), gm.argmax(1) + 1, gm


def select_points(fam, tau5=False):
    ps, x, Gt, sbr, gm = tune_stats(fam, tau5)
    sel = []
    keys = sorted(set((round(r["rho"] ** r["tau"], 6), int(r["tau"])) for _, r in ps))
    tgt = ALPHAS if fam != "M4" else [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    for (rr, tau) in keys:
        idx = [i for i, (_, r) in enumerate(ps) if abs(r["rho"] ** r["tau"] - rr) < 1e-9 and int(r["tau"]) == tau]
        chosen = {}
        for a in tgt:
            ok = [i for i in idx if x[i] <= a]
            if ok:
                chosen[f"a{a}"] = min(ok, key=lambda i: Gt[i])
        chosen["minG"] = min(idx, key=lambda i: Gt[i])
        z0 = [i for i in idx if Gt[i] <= 0.002]
        if z0:
            chosen["g0"] = min(z0, key=lambda i: x[i])
        for tag, i in chosen.items():
            sel.append(dict(fam=fam, tag=tag, r=rr, tau=tau, param=ps[i][0], rdict=ps[i][1], x_tune=float(x[i]), G_tune=float(Gt[i]),
                            s_br=int(sbr[i]), gm_tune=[float(v) for v in gm[i]]))
    return sel


def part_select():
    out = []
    for fam in FAMS:
        out += select_points(fam)
    for fam in ("M3C",):
        if os.path.exists(f"{CACHE}/sweep_{fam}_t5_0.npy"):
            out += select_points(fam, True)
    json.dump(out, open(f"{CACHE}/selection.json", "w"), default=float)
    print("selected", len(out))


def part_eval(fam, tau5=False):
    sel = [s for s in json.load(open(f"{CACHE}/selection.json")) if s["fam"] == fam and (s["tau"] == 5) == tau5]
    S = strategies(); t0 = time.time()
    pairs = [(s["param"], s["rdict"]) for s in sel]
    o = run_pairs(fam, T_MAIN, pairs, S, EVAL, maxB=60000)
    np.save(f"{CACHE}/eval_{fam}{'_t5' if tau5 else ''}.npy", dict(sel=sel, **o), allow_pickle=True)
    print("eval", fam, tau5, len(pairs), f"{time.time()-t0:.0f}s", flush=True)


def part_eval_cross():
    """F1 cross: evaluate tau=1 selected M3C configs at tau=5 (same r) on eval seeds."""
    sel = [s for s in json.load(open(f"{CACHE}/selection.json")) if s["fam"] == "M3C" and s["tau"] == 1 and s["r"] in R_TAU5]
    S = strategies()
    pairs = [(s["param"], rdict(s["r"], 5)) for s in sel]
    o = run_pairs("M3C", T_MAIN, pairs, S, EVAL, maxB=60000)
    np.save(f"{CACHE}/eval_cross.npy", dict(sel=sel, **o), allow_pickle=True)
    print("cross done", flush=True)


def part_evalT():
    sel = [s for s in json.load(open(f"{CACHE}/selection.json")) if s["fam"] == "M3C" and s["tau"] == 1 and s["tag"] in ("a0.01", "a0.05", "g0")]
    S = strategies()
    o = run_pairs("M3C", 100000, [(s["param"], s["rdict"]) for s in sel], S, EVAL, maxB=60000)
    np.save(f"{CACHE}/eval_M3C_T1e5.npy", dict(sel=sel, **o), allow_pickle=True)
    print("evalT done", flush=True)


def part_rho():
    sel = [s for s in json.load(open(f"{CACHE}/selection.json")) if s["fam"] == "M3C" and s["tau"] == 1 and s["tag"] in ("a0.05", "a0.02")]
    S = strategies(); pairs, meta = [], []
    for s in sel:
        r = s["r"]
        for nm, kw in (("oracle", {}), ("r-0.05", dict(ra=max(r - .05, .05))), ("r+0.05", dict(ra=min(r + .05, .999))), ("online", dict(online=1.0))):
            pairs.append((s["param"], rdict(r, 1, **kw))); meta.append(dict(tag=s["tag"], r=r, variant=nm))
    o = run_pairs("M3C", T_MAIN, pairs, S, EVAL, maxB=60000)
    np.save(f"{CACHE}/eval_rho.npy", dict(meta=meta, pairs=pairs, **o), allow_pickle=True)
    print("rho done", flush=True)


# ================================================================== analysis (CPU)
def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]


def load(name):
    return np.load(f"{CACHE}/{name}.npy", allow_pickle=True).item()


def eval_rows(d):
    """rows for each selected config on new seeds."""
    S = strategies(); names = [s[0] for s in S]; groups = [s[1] for s in S]
    rows = []
    for i, s in enumerate(d["sel"]):
        gain = d["u1"][i, 1:, :] - d["u1"][i, :1, :]       # [S-1,N]
        gm = gain.mean(-1)
        j = int(gm.argmax())
        fam = s["fam"]
        xk = d["nraid"][i, 0] if fam == "M4" else d["alpha"][i, 0]
        gb = gain[s["s_br"] - 1]
        grp = {}
        for g in set(groups[1:]):
            ids = [k for k in range(len(gm)) if groups[k + 1] == g]
            tg = np.array(s["gm_tune"])[ids]
            k = ids[int(tg.argmax())]               # group-best chosen on TUNE seeds -> unbiased on eval
            grp[g] = dict(strategy=names[k + 1], gain=ci(gain[k]), z=float(gain[k].mean() / (gain[k].std(ddof=1) / np.sqrt(gain.shape[1])))) 
        rows.append(dict(fam=fam, tag=s["tag"], r=s["r"], tau=s["tau"], param=s["param"], x_tune=s["x_tune"], G_tune=s["G_tune"],
                         x_eval=ci(xk), alpha_eval=ci(d["alpha"][i, 0]), nraid=ci(d["nraid"][i, 0]),
                         G_max=ci(gain[j]), G_max_strategy=names[j + 1], G_br=ci(gb), G_br_strategy=names[s["s_br"]],
                         R_honest=ci(d["reg"][i, 0]), R_br=ci(d["reg"][i, s["s_br"]]), groups=grp))
    return rows


def gap_density():
    """stationary-marginal MC: gap = max(u0,u2)-u1;  G0(em)=E[u1; 0<gap<em] (unpunished oracle-edge gain per round)."""
    rng = np.random.default_rng(0); n = 20_000_000
    u0 = .2 + .8 * rng.random(n); u1 = rng.random(n); u2 = np.sqrt(rng.random(n))
    gap = np.maximum(u0, u2) - u1
    out = dict(f_gap_0plus=float(((gap > 0) & (gap < .01)).mean() / .01), P_lose=float((gap > 0).mean()))
    out["G0"] = {str(em): float((u1 * ((gap > 0) & (gap < em))).mean()) for em in (.02, .05, .1, .2, .4)}
    out["P_edge"] = {str(em): float(((gap > 0) & (gap < em)).mean()) for em in (.02, .05, .1, .2, .4)}
    return out


def slope_profile(sig, G, se):
    """WLS fit G = c * sigma^s (handles G<=0 points); profile CI from delta-chi2 = 3.84."""
    sig, G, se = map(np.asarray, (sig, G, se)); se = np.maximum(se, 1e-4)
    ss = np.arange(-1.0, 6.0, 0.01); chi = []
    for sl in ss:
        x = sig ** sl; w = 1 / se ** 2
        c = (w * G * x).sum() / (w * x * x).sum()
        chi.append((w * (G - c * x) ** 2).sum())
    chi = np.array(chi); k = int(chi.argmin()); ok = ss[chi <= chi.min() + 3.84]
    x = sig ** ss[k]; w = 1 / se ** 2
    return dict(s=float(ss[k]), s_ci=[float(ok.min()), float(ok.max())], c=float((w * G * x).sum() / (w * x * x).sum()), chi2_min=float(chi.min()), dof=int(len(G) - 2))


def slope_ci(sig, G, se, nb=4000, seed=0):
    rng = np.random.default_rng(seed); ls = np.log(sig)
    b = []
    for _ in range(nb):
        g = G + se * rng.standard_normal(len(G))
        if (g <= 0).any(): continue
        b.append(np.polyfit(ls, np.log(g), 1)[0])
    b = np.array(b)
    ok = (G > 0).all()
    s0 = float(np.polyfit(ls, np.log(np.maximum(G, 1e-9)), 1)[0]) if ok else None
    return dict(slope=s0, ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 100 else None, n_boot=int(len(b)))


def analyze():
    res = dict(meta=dict(T=T_MAIN, tune_seeds="0-15", eval_seeds="5000-5031", rs=RS, alphas=ALPHAS))
    # ---- consistency
    if os.path.exists(f"{CACHE}/consist.npy"):
        c = load("consist"); m = c["M3C_h6_L2500"]; names = m["names"]
        gain = {n: ci(m["u1"][ri, i] - m["u1"][ri, 0]) for ri in range(2) for i, n in enumerate(names) if i > 0 for _ in [0]} if False else None
        res["consist_M3C"] = {str(r): {n: ci(m["u1"][ri, i] - m["u1"][ri, 0]) for i, n in enumerate(names) if i > 0} for ri, r in enumerate(m["r"])}
        res["consist_M3C_alpha_honest"] = {str(r): ci(m["alpha"][ri]) for ri, r in enumerate(m["r"])}
        q = c["M4"]; out = []
        for k, (p, rho) in enumerate(q["pairs"]):
            out.append(dict(r=rho, eps=p["eps"], L=p["L"], rw=round(p["rw"], 3), waste=ci(q["nraid"][k]),
                            gain_b03=ci(q["u1"][k, 5] - q["u1"][k, 0]), gain_dz1=ci(q["u1"][k, 6] - q["u1"][k, 0]),
                            gain_edgeO=ci(q["u1"][k, 7] - q["u1"][k, 0]), gain_timing_eq_const=bool((q["u1"][k, 8] == q["u1"][k, 5]).all())))
        res["consist_M4"] = out
    res["gap_density"] = gap_density()
    # ---- eval rows
    allrows = {}
    for fam in FAMS:
        if os.path.exists(f"{CACHE}/eval_{fam}.npy"):
            allrows[fam] = eval_rows(load(f"eval_{fam}"))
    if os.path.exists(f"{CACHE}/eval_M3C_t5.npy"):
        allrows["M3C_t5"] = eval_rows(load("eval_M3C_t5"))
    if os.path.exists(f"{CACHE}/eval_M3C_T1e5.npy"):
        allrows["M3C_T1e5"] = eval_rows(load("eval_M3C_T1e5"))
    if os.path.exists(f"{CACHE}/eval_cross.npy"):
        allrows["M3C_cross_t5"] = eval_rows(load("eval_cross"))
    res["eval_rows"] = allrows
    # ---- G*(alpha_k) per family/r and scaling
    star = {}
    for fam, rows in allrows.items():
        for rw in rows:
            if rw["tag"].startswith("a"):
                star.setdefault(fam, {}).setdefault(rw["tag"], {})[str(rw["r"])] = dict(G=rw["G_max"], G_br=rw["G_br"], alpha=rw["alpha_eval"], x=rw["x_eval"], strat=rw["G_max_strategy"], param=rw["param"])
    res["Gstar"] = star
    scal = {}
    for fam in ("M3C", "M3G", "M3T", "M3W"):
        for tag in [f"a{a}" for a in ALPHAS]:
            d = star.get(fam, {}).get(tag, {})
            rs = sorted(float(r) for r in d)
            if len(rs) < 4: continue
            G = np.array([d[str(r)]["G"][0] for r in rs]); se = np.array([d[str(r)]["G"][1] / 1.96 for r in rs])
            sig = np.sqrt(1 - np.array(rs) ** 2)
            m5 = np.array(rs) <= 0.99
            scal.setdefault(fam, {})[tag] = dict(r=rs, G=G.tolist(), se=se.tolist(), G_over_sigma=(G / sig).tolist(), fit_all=slope_profile(sig, G, se), fit_r_le_0_99=slope_profile(sig[m5], G[m5], se[m5]), **slope_ci(sig, G, se),
                                                 alpha_eval=[d[str(r)]["alpha"][0] for r in rs])
    res["scaling"] = scal
    # best family per (r, alpha_k) by TUNE G, report its eval G*
    best = {}
    for tag in [f"a{a}" for a in ALPHAS]:
        for r in RS:
            c = [(rw["G_tune"], fam, rw) for fam in ("M3T", "M3C", "M3G", "M3W") for rw in allrows.get(fam, []) if rw["tag"] == tag and abs(rw["r"] - r) < 1e-9]
            if c:
                g, fam, rw = min(c, key=lambda t: t[0])
                best.setdefault(tag, {})[str(r)] = dict(fam=fam, G=rw["G_max"], G_br=rw["G_br"], alpha=rw["alpha_eval"], strat=rw["G_max_strategy"])
    res["best_family_Gstar"] = best
    sc2 = {}
    for tag, d in best.items():
        rs = sorted(float(r) for r in d)
        if len(rs) < 4: continue
        G = np.array([d[str(r)]["G"][0] for r in rs]); se = np.array([d[str(r)]["G"][1] / 1.96 for r in rs])
        sig = np.sqrt(1 - np.array(rs) ** 2)
        m5 = np.array(rs) <= 0.99
        Gb = np.array([d[str(r)]["G_br"][0] for r in rs]); seb = np.array([d[str(r)]["G_br"][1] / 1.96 for r in rs])
        sc2[tag] = dict(r=rs, G=G.tolist(), se=se.tolist(), G_over_sigma=(G / sig).tolist(), fit_all=slope_profile(sig, G, se), fit_r_le_0_99=slope_profile(sig[m5], G[m5], se[m5]),
                        G_br=Gb.tolist(), fit_br_all=slope_profile(sig, Gb, seb), fit_br_r_le_0_99=slope_profile(sig[m5], Gb[m5], seb[m5]), **slope_ci(sig, G, se))
    res["scaling_best_family"] = sc2
    # minimal G per r (M3 families, any selected point) and its alpha
    mg = {}
    for r in RS:
        c = [(rw["G_max"][0], fam, rw) for fam in ("M3T", "M3C", "M3G", "M3W") for rw in allrows.get(fam, []) if abs(rw["r"] - r) < 1e-9]
        if c:
            g, fam, rw = min(c, key=lambda t: t[0])
            mg[str(r)] = dict(fam=fam, tag=rw["tag"], G=rw["G_max"], alpha=rw["alpha_eval"], param=rw["param"])
    res["min_G_per_r"] = mg
    g0 = {}
    for fam in ("M3T", "M3C", "M3G", "M3W", "M4"):
        for rw in allrows.get(fam, []):
            if rw["tag"] == "g0":
                g0.setdefault(fam, {})[str(rw["r"])] = dict(x=rw["x_eval"], G=rw["G_max"], G_br=rw["G_br"], strat=rw["G_max_strategy"], param=rw["param"], R_honest=rw["R_honest"])
    res["alpha_zero_gain_points"] = g0
    gd0 = res["gap_density"]["G0"]; edge = []
    for fam in ("M3C", "M3G", "M3T", "M3W"):
        for rw in allrows.get(fam, []):
            e = rw["groups"]["edge_or"]; em = e["strategy"].replace("edgeO", "")
            if em in gd0 and rw["tag"] in ("a0.01", "a0.05"):
                edge.append(dict(fam=fam, tag=rw["tag"], r=rw["r"], em=em, gain=e["gain"], G0=gd0[em], ratio=e["gain"][0] / gd0[em]))
    res["edge_vs_density"] = edge
    # F4: family comparison at same alpha
    f4 = {}
    for tag in [f"a{a}" for a in ALPHAS]:
        for fam in ("M3T", "M3C", "M3G", "M3W"):
            for r in RS:
                v = star.get(fam, {}).get(tag, {}).get(str(r))
                if v: f4.setdefault(tag, {}).setdefault(str(r), {})[fam] = [v["G"][0], v["alpha"][0]]
    res["F4_family_G_by_alpha"] = f4
    # tau comparison
    cmp = {}
    for r in R_TAU5:
        for tag in [f"a{a}" for a in ALPHAS]:
            a1 = star.get("M3C", {}).get(tag, {}).get(str(r)); a5 = star.get("M3C_t5", {}).get(tag, {}).get(str(r))
            if a1 and a5: cmp.setdefault(str(r), {})[tag] = dict(tau1=dict(G=a1["G"], alpha=a1["alpha"]), tau5=dict(G=a5["G"], alpha=a5["alpha"]))
    res["tau_cmp_own_frontier"] = cmp
    cr = {}
    t1 = {(rw["r"], rw["tag"]): rw for rw in allrows.get("M3C", [])}
    for rw in allrows.get("M3C_cross_t5", []):
        a = t1.get((rw["r"], rw["tag"]))
        if a: cr.setdefault(str(rw["r"]), {})[rw["tag"]] = dict(tau1=dict(G=a["G_max"], alpha=a["alpha_eval"]), tau5=dict(G=rw["G_max"], alpha=rw["alpha_eval"]))
    res["tau_cmp_same_config"] = cr
    # F3: uninformed edge at frontier points
    f3 = []
    for fam, rows in allrows.items():
        if fam.endswith("t5") or "T1e5" in fam or "cross" in fam: continue
        for rw in rows:
            f3.append(dict(fam=fam, tag=rw["tag"], r=rw["r"], alpha=rw["alpha_eval"][0], edge_uninformed=rw["groups"]["edge_un"], edge_oracle=rw["groups"]["edge_or"],
                           dz_best=rw["groups"]["dz"], adaptive=rw["groups"]["adapt"], G_max=rw["G_max"], G_max_strategy=rw["G_max_strategy"]))
    res["F3_uninformed"] = f3
    # rho misspec
    if os.path.exists(f"{CACHE}/eval_rho.npy"):
        d = load("eval_rho"); S = strategies(); names = [s[0] for s in S]; rr = []
        for i, mt in enumerate(d["meta"]):
            gain = d["u1"][i, 1:, :] - d["u1"][i, :1, :]; gm = gain.mean(-1); j = int(gm.argmax())
            rr.append(dict(**mt, alpha=ci(d["alpha"][i, 0]), G=ci(gain[j]), strat=names[j + 1]))
        res["rho_misspec"] = rr
    json.dump(res, open(os.path.join(ROOT, "results/stage2b.json"), "w"), indent=1, default=float)
    figures(res)
    print("analysis written", flush=True)


def figures(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(M3T="#1f77b4", M3C="#2ca02c", M3G="#d62728", M3W="#9467bd", M4="#ff7f0e")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    # fig19a: tune scatter + envelope + eval points
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for k, r in enumerate(RS):
        a = ax[k // 3, k % 3]
        for fam in FAMS:
            try:
                ps, x, Gt, _, _ = tune_stats(fam)
            except Exception:
                continue
            idx = [i for i, (_, rd) in enumerate(ps) if abs(rd["rho"] ** rd["tau"] - r) < 1e-9]
            xs = np.array([x[i] for i in idx]); gs = np.array([Gt[i] for i in idx])
            a.scatter(xs + 1e-4, gs, s=5, color=cols[fam], alpha=.25)
            o = np.argsort(xs); env, best = [], 9
            for i in o:
                if gs[i] < best: best = gs[i]; env.append((xs[i] + 1e-4, gs[i]))
            a.step([e[0] for e in env], [e[1] for e in env], where="post", color=cols[fam], lw=1.5, label=fam)
            for rw in res["eval_rows"].get(fam, []):
                if abs(rw["r"] - r) < 1e-9:
                    a.errorbar(rw["x_eval"][0] + 1e-4, rw["G_max"][0], yerr=rw["G_max"][1], fmt="D", ms=4, color=cols[fam], mec="k", capsize=2)
        a.axhline(0, color="k", lw=1); a.set_xscale("log"); a.set_title(f"r={r}  sigma={math.sqrt(1-r*r):.3f}", fontsize=9)
        a.set_xlabel("alpha (honest false-punish fraction; M4: raid cost) +1e-4"); 
        if k % 3 == 0: a.set_ylabel("G = max lying gain / round")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("fig19a  G vs alpha: dots = all params (tune seeds 0-15), line = lower envelope, diamonds = re-evaluated on new seeds 5000-5031", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig19a_frontier_G_vs_alpha.png"), dpi=130); plt.close(fig)
    # fig19b: scaling
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    sb = res.get("scaling_best_family", {})
    for tag, d in sb.items():
        sig = np.sqrt(1 - np.array(d["r"]) ** 2)
        ax[0].plot(d["r"], d["G"], "o-", ms=3, label=tag)
        ax[1].plot(d["r"], d["G_over_sigma"], "o-", ms=3, label=tag)
        ax[2].loglog(sig, np.maximum(d["G"], 1e-4), "o-", ms=3, label=f"{tag} slope={d['slope'] if d['slope'] is None else round(d['slope'],2)}")
    ax[2].loglog([.04, .9], [.04 * .15, .9 * .15], "k:", label="slope 1")
    ax[0].set_title("G*(alpha) vs r"); ax[1].set_title("G*/sigma_tau vs r (collapse => G* ~ sigma g(alpha))"); ax[2].set_title("log G* vs log sigma_tau")
    for a_ in ax: a_.legend(fontsize=6); a_.set_xlabel("r" if a_ is not ax[2] else "sigma_tau")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig19b_scaling_G_over_sigma.png"), dpi=130); plt.close(fig)
    # fig19c: F3 / density
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    f3 = [x for x in res.get("F3_uninformed", []) if x["fam"] in ("M3C", "M3G", "M3T", "M3W") and x["tag"] == "a0.05"]
    for fam in ("M3C", "M3G", "M3T", "M3W"):
        pts = sorted([(x["r"], x["edge_uninformed"]["gain"][0], x["edge_uninformed"]["gain"][1], x["edge_oracle"]["gain"][0]) for x in f3 if x["fam"] == fam])
        if pts:
            ax[0].errorbar([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], marker="o", ms=3, color=cols[fam], label=f"{fam} uninformed edge", capsize=2)
            ax[0].plot([p[0] for p in pts], [p[3] for p in pts], "--", color=cols[fam], label=f"{fam} oracle edge")
    ax[0].axhline(0, color="k"); ax[0].set_title("edge-inflation gain at alpha<=0.05 points (new seeds)"); ax[0].legend(fontsize=6); ax[0].set_xlabel("r")
    gd = res["gap_density"]
    ems = [.02, .05, .1, .2, .4]
    ax[1].plot(ems, [gd["G0"][str(e)] for e in ems], "k-o", label="G0(em)=E[u1;0<gap<em] (unpunished)")
    ax[1].plot(ems, [gd["f_gap_0plus"] * e * .5 for e in ems], "k:", label="f_gap(0+) em * ~0.5")
    ax[1].set_title("edge gain scale from margin density"); ax[1].set_xlabel("em"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig19c_edge_uninformed_density.png"), dpi=130); plt.close(fig)
    # fig19d: rho
    if "rho_misspec" in res:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, v in enumerate(("oracle", "r-0.05", "r+0.05", "online")):
            for tag, mk in (("a0.05", "o"), ("a0.02", "s")):
                pts = sorted([(x["r"], x["alpha"][0], x["G"][0]) for x in res["rho_misspec"] if x["variant"] == v and x["tag"] == tag])
                ax[0].plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", ms=3, color=f"C{k}", label=f"{v} {tag}")
                ax[1].plot([p[0] for p in pts], [p[2] for p in pts], mk + "-", ms=3, color=f"C{k}")
        ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].set_title("alpha under assumed-rho variants"); ax[1].set_title("G under assumed-rho variants")
        ax[0].legend(fontsize=6); ax[0].set_xlabel("true r"); ax[1].set_xlabel("true r"); ax[1].axhline(0, color="k")
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig19d_rho_misspec.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True); ap.add_argument("--mech", default="M3C"); ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--tau5", action="store_true")
    a = ap.parse_args()
    if a.part == "smoke": part_smoke()
    elif a.part == "consist": part_consist()
    elif a.part == "sweep": part_sweep(a.mech, a.chunk, a.tau5)
    elif a.part == "select": part_select()
    elif a.part == "eval": part_eval(a.mech, a.tau5)
    elif a.part == "cross": part_eval_cross()
    elif a.part == "rho": part_rho()
    elif a.part == "evalT": part_evalT()
    elif a.part == "nchunks": print(nchunks(a.mech, a.tau5))
    elif a.part == "analyze": analyze()


