"""Stage 2 driver: audit repairs under delayed verification (see docs/stage2_design.md).

    python3 scripts/run_stage2.py --part checks|bt|sims|analyze|all  [--cache DIR] [--quick]
Writes results/stage2.json and figures/fig18_*.png (not with --quick).  All simulation on CUDA float64.
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from arbitration.gpu import adaaudit as A
from arbitration.gpu import adaaudit_tol as X
from arbitration.gpu import audit_fix as F

DEV = "cuda"
TUNE = list(range(0, 16))
EVAL = list(range(1000, 1032))
RHOS = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]      # tau=1 => r = rho^tau (z-space); 1.0 = degenerate reference
TS = [10000, 100000]
BS = [round(0.05 * i, 2) for i in range(1, 11)]
EPS_GAIN, FP_CAP = 0.002, 0.05                 # fixed before running (design doc)
from pathlib import Path
DEFAULT_CACHE = os.environ.get("ARB_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "results" / "cache" / "run_stage2"))
INF = float("inf")


DZS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # z-space (quantile) shift strategies, never hit the clamp at v=1


def strat_full():
    return [(0.0, False)] + [(b, False) for b in BS] + [(b, True) for b in BS] + [(0.0, False, d) for d in DZS] + [(0.0, True, d) for d in DZS]


def strat_part():
    return [(0.0, False)] + [(b, False) for b in BS] + [(b, True) for b in (0.1, 0.3, 0.5)] + [(0.0, False, d) for d in DZS]


def grids():
    G = {}
    G["M0"] = (dict(strats=[(0.0, False), (0.3, False)]), [dict()])
    G["M1f"] = (dict(strats=strat_part()), [dict(p=0.1, tol=d) for d in (0, .02, .05, .1, .2, .3, .5, .7)])
    G["M1a"] = (dict(strats=strat_part()), [dict(tol=d) for d in (0, .02, .05, .1, .2, .3, .5, .7)])
    G["M2"] = (dict(strats=strat_full()), [dict(p=0.1, tol=d, L=L) for d in (0, .1, .3) for L in (20, 100, 500, 2500)])
    G["M3a"] = (dict(strats=strat_part()), [dict(p=p, tol=k, L=L) for p in (.03, .1, .3) for k in (.5, 1, 1.5, 2, 2.5, 3, 4)
                                            for L in (20, 100, 500, 2500)])
    G["M3b"] = (dict(strats=strat_part()), [dict(p=0.1, tol=k, L=H) for k in (1, 2, 3) for H in (20, 100, 500, 2500)])
    G["M3c"] = (dict(strats=strat_part()), [dict(p=0.1, tol=h, L=L) for h in (1, 2, 3, 4, 6, 8) for L in (20, 100, 500, 2500)])
    G["M4"] = (dict(strats=strat_part()), [dict(eps=e, L=L) for e in (.005, .01, .02, .05, .1, .2) for L in (20, 100, 500, 2500)])
    G["M5"] = (dict(strats=strat_part()), [dict(p=0.1, tol=k, L=100, eps=e) for k in (1, 2, 3) for e in (.01, .02, .05, .1)])
    return G


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0]


FIELDS = ["reg_a", "reg_b", "util_a", "util_b", "pen", "flags", "audits", "nraid", "est_reset"]


# ---------------------------------------------------------------- sims
def run_sims(cache, quick):
    os.makedirs(cache, exist_ok=True)
    G = grids()
    seeds = TUNE + EVAL
    Ts = [300] if quick else TS
    out = {}
    for T in Ts:
        for mech, (kw, params) in G.items():
            fn = f"{cache}/s2_{mech}_{T}.npz"
            if not quick and os.path.exists(fn):
                z = np.load(fn)
                out[(mech, T)] = {f: z[f] for f in FIELDS}
                continue
            t0 = time.time()
            strats = kw["strats"]
            sub = [params] if quick else None
            per = max(1, 110000 // (len(RHOS) * len(strats) * len(seeds)))
            chunks = [params[i:i + per] for i in range(0, len(params), per)]
            if quick:
                chunks = [params[:2]]
            res = {f: [] for f in FIELDS}
            for ch in chunks:
                el, (P, R, S, N) = F.make_elements(ch, RHOS, strats, seeds)
                o = F.simulate(mech, T, el, seeds)
                for f in FIELDS:
                    a = o[f].cpu().numpy()
                    res[f].append(a.reshape(P, R, S, N, *a.shape[1:]))
            out[(mech, T)] = {f: np.concatenate(res[f], 0) for f in FIELDS}
            if not quick:
                np.savez(fn, **out[(mech, T)])
            print(f"sim {mech} T={T} B-chunks={len(chunks)} {time.time()-t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- checks
def checks():
    res = {}
    D = DEV
    f = lambda v, dt=torch.float64: torch.full((1,), v, device=D, dtype=dt)
    bit = []
    for mech, mm in (("M1f", "fixed"), ("M1a", "ada")):
        for seed, (rho, dl, b) in zip([3, 4, 5], [(0.9, 0.1, 0.1), (0.99, 0.05, 0.3), (0.5, 0.0, 0.0)]):
            T = 2000
            o = X.simulate_tol(mm, T, f(rho), f(1, torch.long), f(dl), f(b), seed, p_fixed=0.1)
            el, _ = F.make_elements([dict(p=0.1, tol=dl)], [rho], [(b, False)], [seed])
            n = F.simulate(mech, T, el, [seed])
            bit.append(dict(mech=mech, seed=seed, d_regret=float(o["regret"] - n["reg_a"]),
                            d_util=float((o["util"] - n["util_a"]).abs().max()), d_audits=float(o["audits"] - n["audits"]),
                            same_excl=bool(((o["elim_t"] <= T) == torch.isinf(n["susp"])).all())))
    res["bitwise_vs_adaaudit_tol"] = bit
    seeds = list(range(16)); T = 3000

    def run(mech, params, rhos, strats):
        el, (P, R, S, N) = F.make_elements(params, rhos, strats, seeds)
        o = F.simulate(mech, T, el, seeds)
        return {k: v.reshape(P, R, S, N, *v.shape[1:]) for k, v in o.items()}
    cal = []
    for kap in (1.0, 2.0, 3.0):
        o = run("M3a", [dict(p=0.5, tol=kap, L=0.0)], [0.5, 0.9, 1.0], [(0.0, False)])
        for ri, r in enumerate([0.5, 0.9, 1.0]):
            n_a = float(o["audits"][0, ri, 0].sum())
            cal.append(dict(kappa=kap, r=r, audits=n_a, flag_per_audit=float(o["flags"][0, ri, 0].sum() / n_a),
                            Phi_neg_kappa=0.5 * math.erfc(kap / math.sqrt(2))))
    res["calibration_honest_flag_rate"] = cal
    o = run("M3a", [dict(p=0.3, tol=1.0, L=100)], [1.0], [(0.0, False), (0.3, False)])
    res["r1_M3"] = dict(honest_flags=float(o["flags"][0, 0, 0].sum()), liar_flags=float(o["flags"][0, 0, 1, :, 1].sum()),
                        liar_audits=float(o["audits"][0, 0, 1].sum()))
    o = run("M2", [dict(p=0.3, tol=0.0, L=100)], [0.9], [(0.0, False), (0.3, False), (0.3, True)])
    res["M2_flags_honest_const_timing"] = [float(o["flags"][0, 0, i].sum()) for i in range(3)]
    o = run("M4", [dict(eps=0.05, L=100)], [0.9], [(0.0, False), (0.3, False)])
    res["M4"] = dict(honest_flags=float(o["flags"][0, 0, 0].sum()), waste_rate=float(o["nraid"][0, 0, 0].mean() / T),
                     liar_flags_per_seed=float(o["flags"][0, 0, 1, :, 1].mean()))
    tc = {}
    for mech, pp in (("M1f", dict(p=0.1, tol=0.1)), ("M3a", dict(p=0.1, tol=1.0, L=100)), ("M4", dict(eps=0.05, L=100)),
                     ("M3b", dict(p=0.1, tol=1.0, L=100)), ("M3c", dict(p=0.1, tol=3.0, L=100)), ("M5", dict(p=0.1, tol=2.0, L=100, eps=0.02))):
        o = run(mech, [pp], [0.9], [(0.3, False), (0.3, True)])
        tc[mech] = bool((o["util_a"][0, 0, 0] == o["util_a"][0, 0, 1]).all() and (o["reg_a"][0, 0, 0] == o["reg_a"][0, 0, 1]).all())
    res["timing_equals_const_bitwise"] = tc
    o = run("M0", [dict()], [0.9], [(0.0, False), (0.3, False)])
    res["M0_max_abs_gain"] = float((o["util_a"][0, 0, 1] - o["util_a"][0, 0, 0]).abs().max())
    return res


def axis_table(n=2_000_000):
    g = torch.Generator(device=DEV); g.manual_seed(1)
    out = []
    for r in RHOS:
        z = torch.randn(n, 3, generator=g, device=DEV, dtype=torch.float64)
        z2 = r * z + math.sqrt(max(1 - r * r, 0)) * torch.randn(n, 3, generator=g, device=DEV, dtype=torch.float64)
        u, u2 = A.transform(torch.special.ndtr(z)), A.transform(torch.special.ndtr(z2))
        pe = [float(torch.corrcoef(torch.stack([u[:, i], u2[:, i]]))[0, 1]) for i in range(3)]
        out.append(dict(r=r, u_pearson=pe, u_spearman=6 / math.pi * math.asin(r / 2) if r < 1 else 1.0))
    return out


# ---------------------------------------------------------------- B_T of AdaAudit, tau=0 honest
def bt_part(cache, quick):
    os.makedirs(cache, exist_ok=True)
    rhos, S = [0.0, 0.5, 0.9, 0.99], 32
    Ts = [1000, 10000] if quick else [1000, 10000, 100000, 1000000]
    rows = {}
    for T in Ts:
        fn = f"{cache}/bt_{T}.npz"
        if not quick and os.path.exists(fn):
            z = np.load(fn); aud, est, nel = z["audits"], z["est_reset"], z["nel"]
        else:
            rho = torch.tensor([r for r in rhos for _ in range(S)], device=DEV, dtype=torch.float64)
            B = len(rho)
            zz = lambda dt, v: torch.full((B,), v, device=DEV, dtype=dt)
            o = A.simulate("ada", T, rho, zz(torch.long, 0), zz(torch.bool, False), zz(torch.float64, 0.0), seed=777 + T)
            aud, est = o["audits"].cpu().numpy().reshape(4, S), o["est_reset"].cpu().numpy().reshape(4, S)
            nel = (o["elim_t"] <= T).double().sum(1).cpu().numpy().reshape(4, S)
            if not quick:
                np.savez(fn, audits=aud, est_reset=est, regret=o["regret"].cpu().numpy().reshape(4, S), nel=nel)
        rows[T] = (aud, est, nel)
    out = dict(Ts=Ts, rhos=rhos, ideal_BT=[A.expected_audits_idealised(T) for T in Ts], by_rho={})
    lt = np.log(Ts)
    for i, r in enumerate(rhos):
        m = np.array([rows[T][0][i].mean() for T in Ts])
        out["by_rho"][str(r)] = dict(
            BT=[ci(rows[T][0][i]) for T in Ts], est_reset=[ci(rows[T][1][i]) for T in Ts],
            n_excluded_per_seed=[float(rows[T][2][i].mean()) for T in Ts],
            slope_BT_vs_lnT_fit=float(np.polyfit(lt, m, 1)[0]),
            consecutive_slopes=[float((m[j + 1] - m[j]) / (lt[j + 1] - lt[j])) for j in range(len(Ts) - 1)],
            loglog_slope=float(np.polyfit(lt, np.log(m), 1)[0]))
    return out


# ---------------------------------------------------------------- analysis
def analyze_mech(mech, T, a, params, strats, R0b):
    """a: dict of arrays [P,R,S,N,...].  R0b: M0 per-seed R_b/T  [R,N].  Returns list over (param, rho) of dicts."""
    nT = len(TUNE)
    P, R, S, N = a["reg_b"].shape
    tn, ev = slice(0, nT), slice(nT, N)
    ub = a["util_b"][..., 1] / T
    gain = ub[:, :, 1:, :] - ub[:, :, :1, :]
    rb = a["reg_b"] / T; ra = a["reg_a"] / T
    fp = a["pen"][:, :, 0].mean(-1) / T                 # honest scenario, mean over 3 agents  [P,R,N]
    rows = []
    for pi in range(P):
        for ri in range(R):
            g = gain[pi, ri]
            gt = g[:, tn].mean(-1)
            ss = int(gt.argmax())
            ge = g[:, ev]
            sm = int(ge.mean(-1).argmax())
            r_sc = 1 + ss
            lab = lambda k: (f"dz={strats[1 + k][2]}" if len(strats[1 + k]) > 2 else f"b={strats[1 + k][0]}") + ("T" if strats[1 + k][1] else "")
            isb = [len(x) < 3 for x in strats[1:]]
            gb = np.where(isb, g[:, ev].mean(-1), -9.0); smb = int(gb.argmax())
            rows.append(dict(
                pi=pi, ri=ri, param=params[pi], r=RHOS[ri],
                tune=dict(Gmax=float(gt.max()), fp=float(fp[pi, ri, :nT].mean()), R_br=float(rb[pi, ri, r_sc, :nT].mean())),
                br_strategy=lab(ss), Gmax_bfamily=ci(ge[smb]), Gmax_bfamily_strategy=lab(smb), Gbr=ci(g[ss, ev]), Gmax=ci(ge[sm]), Gmax_strategy=lab(sm),
                fp=ci(fp[pi, ri, ev]),
                R_b_honest=ci(rb[pi, ri, 0, ev]), R_b_br=ci(rb[pi, ri, r_sc, ev]),
                R_a_honest=ci(ra[pi, ri, 0, ev]), R_a_br=ci(ra[pi, ri, r_sc, ev]),
                R_b_diff_vs_M0_br=ci(rb[pi, ri, r_sc, ev] - R0b[ri, ev]),
                audits=ci(a["audits"][pi, ri, 0, ev] / T), raids=ci(a["nraid"][pi, ri, 0, ev] / T),
                honest_util1=float(ub[pi, ri, 0, ev].mean()), est_reset=float(a["est_reset"][pi, ri, 0, ev].mean()),
                gains_by_strategy_eval=[float(x) for x in ge.mean(-1)],
                no_profit=bool(ci(ge[sm])[0] - ci(ge[sm])[1] <= 0), beats_M0=bool(ci(rb[pi, ri, r_sc, ev] - R0b[ri, ev])[0] +
                                                                              ci(rb[pi, ri, r_sc, ev] - R0b[ri, ev])[1] < 0)))
    return rows


def select(rows):
    out = {}
    for ri in sorted(set(r["ri"] for r in rows)):
        rr = [r for r in rows if r["ri"] == ri]
        feas = [r for r in rr if r["tune"]["Gmax"] <= EPS_GAIN and r["tune"]["fp"] <= FP_CAP]
        if feas:
            best, ok = min(feas, key=lambda r: r["tune"]["R_br"]), True
        else:
            cap = [r for r in rr if r["tune"]["fp"] <= FP_CAP]
            best, ok = (min(cap, key=lambda r: r["tune"]["Gmax"]) if cap else min(rr, key=lambda r: r["tune"]["fp"])), False
        d = dict(best); d["feasible_on_tune"] = ok
        d["eval_pass"] = bool(ok and best["no_profit"] and best["fp"][0] <= FP_CAP)
        d["eval_pass_and_beats_M0"] = bool(d["eval_pass"] and best["beats_M0"])
        # descriptive: does ANY grid point pass on eval seeds (multiplicity: optimistic)?
        d["any_grid_pass_eval"] = int(sum(1 for r in rr if r["no_profit"] and r["fp"][0] <= FP_CAP and r["beats_M0"]))
        d["n_grid"] = len(rr)
        out[ri] = d
    return out


def figures(js):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [("M1f", "M1 fixed-p (perm.)", "#888"), ("M1a", "M1 AdaAudit (perm.)", "#444"), ("M2", "M2 arrival", "#d62728"),
            ("M3a", "M3a suspend", "#1f77b4"), ("M3b", "M3b credit", "#17becf"), ("M3c", "M3c CUSUM", "#2ca02c")]
    curves = js["curves"]
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    # --- fig18a: max gain vs false-punish
    fig, ax = plt.subplots(2, len(RHOS), figsize=(3.1 * len(RHOS), 6.2), sharey="row", squeeze=False)
    for ti, T in enumerate(TS):
        for ri, r in enumerate(RHOS):
            a = ax[ti, ri]
            for fam, name, col in fams:
                pts = sorted([(c["fp"][0], c["Gmax"][0]) for c in curves[fam][str(T)][str(r)]])
                a.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, color=col, alpha=.25)
                env, best = [], 1e9
                for x, y in pts:
                    if y < best:
                        best = y; env.append((x, y))
                a.step([e[0] for e in env], [e[1] for e in env], where="post", color=col, lw=1.6, label=name if (ti, ri) == (0, 0) else None)
            a.axhline(0, color="k", lw=1.2, ls="-"); a.axhline(EPS_GAIN, color="k", lw=.6, ls=":")
            a.axvline(FP_CAP, color="gray", lw=.6, ls=":")
            a.set_xscale("symlog", linthresh=1e-3); a.set_title(f"r=rho^tau={r}, T={T:.0e}", fontsize=8)
            if ri == 0:
                a.set_ylabel("max lying gain / round")
            a.set_xlabel("honest false-punish fraction", fontsize=7)
    ax[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle("fig18a  max lying gain vs honest false-punish (lower envelope per family; dots = all params; black line = M0)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig18a_gain_vs_falsepunish.png"), dpi=130); plt.close(fig)
    # --- fig18b: raid cost
    T = TS[-1]
    fig, ax = plt.subplots(1, len(RHOS), figsize=(3.1 * len(RHOS), 3.4), sharey=True)
    for ri, r in enumerate(RHOS):
        a = ax[ri]
        for Lv, col in zip((20, 100, 500, 2500), ("#fdae6b", "#f16913", "#a63603", "#4d1a00")):
            pts = sorted([(c["raids"][0], c["Gmax"][0]) for c in curves["M4"][str(T)][str(r)] if c["param"]["L"] == Lv])
            a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=col, label=f"M4 L={Lv}" if ri == 0 else None)
        for k, col in zip((1, 2, 3), ("#9ecae1", "#3182bd", "#08306b")):
            pts = sorted([(c["raids"][0], c["Gmax"][0]) for c in curves["M5"][str(T)][str(r)] if c["param"]["tol"] == k])
            a.plot([p[0] for p in pts], [p[1] for p in pts], "s--", ms=3, color=col, label=f"M5 kappa={k},L=100" if ri == 0 else None)
        m3 = js["main"]["M3a"][str(T)][str(r)]
        a.axhline(m3["Gmax"][0], color="#1f77b4", lw=1, ls="-.", label="M3a selected" if ri == 0 else None)
        a.axhline(0, color="k", lw=1.2); a.axhline(EPS_GAIN, color="k", lw=.6, ls=":")
        a.set_xscale("log"); a.set_xlabel("raid cost: wasted-round fraction (=eps)", fontsize=7); a.set_title(f"r={r}", fontsize=8)
    ax[0].set_ylabel("max lying gain / round"); ax[0].legend(fontsize=5.5)
    fig.suptitle(f"fig18b  surprise-raid cost vs max lying gain (T={T:.0e}; black = M0)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig18b_raid_cost.png"), dpi=130); plt.close(fig)
    # --- fig18c: operating points
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    allf = fams + [("M4", "M4 raid", "#9467bd"), ("M5", "M5 M3a+M4", "#8c564b")]
    for fam, name, col in allf:
        xs = [r for r in RHOS]
        mm = [js["main"][fam][str(T)][str(r)] for r in RHOS]
        ax[0].errorbar(xs, [m["R_b_br"][0] for m in mm], [m["R_b_br"][1] for m in mm], marker="o", ms=3, color=col, label=name, capsize=2)
        ax[1].errorbar(xs, [m["Gmax"][0] for m in mm], [m["Gmax"][1] for m in mm], marker="o", ms=3, color=col, capsize=2)
        ax[2].plot(xs, [m["fp"][0] + 1e-5 for m in mm], marker="o", ms=3, color=col)
    m0 = [js["main"]["M0"][str(T)][str(r)]["R_b_honest"][0] for r in RHOS]
    ax[0].plot(RHOS, m0, "k-", lw=1.5, label="M0 ignore reports")
    ax[1].axhline(0, color="k", lw=1.5); ax[1].axhline(EPS_GAIN, color="k", lw=.6, ls=":")
    ax[2].set_yscale("log"); ax[2].axhline(FP_CAP, color="gray", ls=":")
    for a_, t_ in zip(ax, ("R_T/T at best-response liar (fallback b)", "max lying gain / round", "honest false-punish fraction (+1e-5)")):
        a_.set_title(t_, fontsize=8); a_.set_xlabel("r = rho^tau (z-space)")
    ax[0].legend(fontsize=6)
    fig.suptitle(f"fig18c  tuned operating points (tuned on seeds 0-15, evaluated on 1000-1031), T={T:.0e}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/fig18c_operating_points.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    js = dict(meta=dict(tune_seeds="0-15", eval_seeds="1000-1031", rhos=RHOS, tau=1, Ts=TS, eps_gain=EPS_GAIN, fp_cap=FP_CAP,
                        primary_fallback="b (random allocation to all when nobody eligible); a also reported"))
    if a.part in ("checks", "all"):
        js["checks"] = checks(); js["axis"] = axis_table()
        print(json.dumps(js["checks"], indent=1)[:3000], flush=True)
    if a.part in ("bt", "all"):
        js["bt_adaaudit_tau0"] = bt_part(a.cache, a.quick)
        print(json.dumps({k: v for k, v in js["bt_adaaudit_tau0"].items() if k != "by_rho"}), flush=True)
    if a.part in ("sims", "analyze", "all"):
        raw = run_sims(a.cache, a.quick)
        if a.part == "sims":
            return
        G = grids()
        js["main"], js["curves"] = {}, {}
        for mech in G:
            js["main"][mech], js["curves"][mech] = {}, {}
        for T in ([300] if a.quick else TS):
            M0 = raw[("M0", T)]
            R0b = M0["reg_b"][0, :, 0, :] / T
            for mech, (kw, params) in G.items():
                rows = analyze_mech(mech, T, raw[(mech, T)], params, kw["strats"], R0b)
                sel = select(rows)
                js["main"][mech][str(T)] = {str(RHOS[ri]): d for ri, d in sel.items()}
                js["curves"][mech][str(T)] = {}
                for ri in range(len(RHOS)):
                    js["curves"][mech][str(T)][str(RHOS[ri])] = [
                        {k: r[k] for k in ("param", "fp", "Gmax", "Gmax_bfamily", "Gbr", "R_b_br", "R_b_honest", "audits", "raids", "no_profit", "beats_M0")}
                        for r in rows if r["ri"] == ri]
        if not a.quick:
            with open(os.path.join(ROOT, "results/stage2.json"), "w") as fh:
                json.dump(js, fh, indent=1, default=float)
            figures(js)
            print("done", flush=True)


if __name__ == "__main__":
    main()
