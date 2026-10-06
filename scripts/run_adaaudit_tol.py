"""Stage 1b: tolerance-delta audits (fixed-p M0 and AdaAudit) over (rho, tau, delta, T) x {honest, liar b=.1/.3}.
Writes results/adaaudit_tol.json and figures/fig17a/b/c_*.png.

    python3 scripts/run_adaaudit_tol.py              # sims (cached to scratchpad npz if --cache) + analysis
    python3 scripts/run_adaaudit_tol.py --quick      # smoke test, no files written
"""
import argparse, json, os, sys, time, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from arbitration.gpu import adaaudit as A
from arbitration.gpu import adaaudit_tol as X

DEV = "cuda"
S = 32
TS = [1000, 10000, 100000]
P_FIXED = 0.1
# (rho, tau): the 9 required cells + 4 extras chosen so rho^tau nearly coincides across different (rho, tau)
CELLS = [(r, t) for r in (0.5, 0.9, 0.99) for t in (1, 5, 20)] + [(0.9, 2), (0.99, 10), (0.9, 7), (0.99, 50)]
DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]   # 0 = consistency check, 0.5 = added to locate the transition
SCN = {"honest": 0.0, "b0.1": 0.1, "b0.3": 0.3}
FIELDS = ["regret", "audits", "audits_post", "nelim", "elim_t", "nwin", "util", "n_res", "n_fa"]


def run_sims(seeds, Ts, cache):
    keys = [(c, d, s) for c in CELLS for d in DELTAS for s in SCN]
    rep = lambda f, dt=torch.float64: torch.tensor([f(k) for k in keys for _ in range(seeds)], device=DEV, dtype=dt)
    rho, tau = rep(lambda k: k[0][0]), rep(lambda k: k[0][1], torch.long)
    dl, db = rep(lambda k: k[1]), rep(lambda k: SCN[k[2]])
    out = {}
    for mech in ("fixed", "ada"):
        for T in Ts:
            fn = f"{cache}_{mech}_{T}.npz" if cache else None
            if fn and os.path.exists(fn):
                z = np.load(fn)
                out[(mech, T)] = {f: z[f] for f in FIELDS}
                continue
            t0 = time.time()
            r = X.simulate_tol(mech, T, rho, tau, dl, db, seed=2000 + T, p_fixed=P_FIXED)
            out[(mech, T)] = {f: r[f].cpu().numpy().reshape(len(keys), seeds, *r[f].shape[1:]) for f in FIELDS}
            if fn:
                np.savez(fn, **out[(mech, T)])
            print(f"sim {mech} T={T} {time.time()-t0:.0f}s", flush=True)
    return keys, out


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]


def consistency(Ts):
    """delta=0 checks: (1) bit-identical to stage-1 simulate(); (2) stage-1 phenomenon at tau>=1 reproduced."""
    B = 256
    f = lambda v, dt=torch.float64: torch.full((B,), v, device=DEV, dtype=dt)
    res = {}
    for mech in ("fixed", "ada"):
        o = X.simulate_tol(mech, 1000, f(0.9), f(5, torch.long), f(0.0), f(0.0), 5)
        o1 = A.simulate(mech, 1000, f(0.9), f(5, torch.long), f(0, torch.bool), f(0.0), 5)
        o0 = X.simulate_tol(mech, 1000, f(0.0), f(0, torch.long), f(0.0), f(0.0), 6)
        res[mech] = dict(max_abs_diff_regret_vs_stage1=float((o["regret"] - o1["regret"]).abs().max()),
                         max_abs_diff_elimt_vs_stage1=float((o["elim_t"] - o1["elim_t"]).abs().max()),
                         iid_tau0_R_T_mean=float(o0["regret"].mean()),
                         iid_tau0_any_excl=float((o0["elim_t"] <= 1000).any(1).double().mean()))
    return res


def boot_slope(y_by_T_seed, Ts, rng, nb=300):
    """y_by_T_seed: [nT, n] per-seed per-agent exclusion indicator -> slope of ln(-ln(1-E)) vs ln T."""
    lt = np.log(Ts); n = y_by_T_seed.shape[1]; floor = 0.5 / n

    def sl(Y):
        E = np.clip(Y.mean(1), floor, 1 - floor)
        return float(np.polyfit(lt, np.log(-np.log(1 - E)), 1)[0])
    pt = sl(y_by_T_seed)
    bs = [sl(y_by_T_seed[:, rng.integers(0, n, n)]) for _ in range(nb)]
    return [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    seeds = 4 if a.quick else S
    Ts = [300, 1000] if a.quick else TS
    t0 = time.time()
    cons = consistency(Ts)
    print("consistency", cons, flush=True)
    keys, raw = run_sims(seeds, Ts, a.cache)
    kidx = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(0)
    rt = {c: c[0] ** c[1] for c in CELLS}
    HON = {"honest": [0, 1, 2], "b0.1": [0, 2], "b0.3": [0, 2]}

    # ---------------- per-cell summaries ----------------
    summ = {}
    seedmat = {}  # raw per-seed arrays used for bootstrap (surv per seed, catch per seed)
    for (mech, T), r in raw.items():
        for c in CELLS:
            for d in DELTAS:
                ent = {}
                for s in SCN:
                    i = kidx[(c, d, s)]
                    et = r["elim_t"][i]                       # [seeds, K]
                    hon = HON[s]
                    ex = (et[:, hon] <= T).astype(float)      # honest excluded
                    e = dict(R=ci(r["regret"][i]), B=ci(r["audits"][i]), B_post=ci(r["audits_post"][i]),
                             nelim=ci(r["nelim"][i]), honest_surv=ci(1 - ex.mean(1)),
                             honest_all_surv=ci(1 - ex.max(1)))
                    seedmat[(mech, T, c, d, s, "surv")] = 1 - ex.mean(1)
                    if s != "honest":
                        t1 = np.minimum(et[:, 1], T)
                        caught = (et[:, 1] <= T).astype(float)
                        ex1 = (r["elim_t"][kidx[(c, d, "honest")]][:, 1] <= T).astype(float)  # agent 1 excluded when honest (same cell)
                        e.update(catch_excess_over_honest_exclusion=float(caught.mean() - ex1.mean()), catch=ci(caught), liar_time=ci(t1), liar_win_util=ci(r["util"][i][:, 1]),
                                 liar_wins=ci(r["nwin"][i][:, 1]))
                        seedmat[(mech, T, c, d, s, "catch")] = caught
                    ent[s] = e
                # liar gain relative to honest-scenario same cell (agent 1's utility received when honest)
                u_h = r["util"][kidx[(c, d, "honest")]][:, 1].mean()
                for s in ("b0.1", "b0.3"):
                    ent[s]["liar_gain_vs_honest_util"] = ent[s]["liar_win_util"][0] - float(u_h)
                summ[f"{mech}|{T}|{c[0]}|{c[1]}|{d}"] = dict(rho_tau=rt[c], **ent)

    # ---------------- (i) collapse test on rho^tau ----------------
    pairs = [(c1, c2) for c1, c2 in itertools.combinations(CELLS, 2) if c1[0] != c2[0] and abs(rt[c1] - rt[c2]) < 0.03
             and 1e-3 < rt[c1] < 0.999]
    coll = []
    for mech in ("fixed", "ada"):
        for T in Ts:
            for d in DELTAS[1:]:
                for c1, c2 in pairs:
                    for s, m in (("honest", "surv"), ("b0.3", "catch"), ("b0.1", "catch")):
                        x, y = seedmat[(mech, T, c1, d, s, m)], seedmat[(mech, T, c2, d, s, m)]
                        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
                        if se < 1e-9:
                            continue  # both saturated at 0/1 -> identical, uninformative
                        coll.append(dict(mech=mech, T=T, delta=d, c1=c1, c2=c2, rt1=rt[c1], rt2=rt[c2], metric=f"{s}:{m}",
                                         diff=float(x.mean() - y.mean()), z=float((x.mean() - y.mean()) / se)))
    zs = np.array([e["z"] for e in coll])
    # also compare with the "no-collapse" yardstick: z for cells with very different rho^tau is huge by design;
    collapse = dict(pairs=[[list(a_), list(b_)] for a_, b_ in pairs], n=len(zs),
                    frac_abs_z_gt2=float((np.abs(zs) > 2).mean()), frac_abs_z_gt3=float((np.abs(zs) > 3).mean()),
                    mean_abs_diff=float(np.mean([abs(e["diff"]) for e in coll])), worst=sorted(coll, key=lambda e: -abs(e["z"]))[:8])

    # ---------------- (iv) single-audit false flag: theory vs simulation ----------------
    iv = []
    for mech in ("fixed", "ada"):
        T = 10000 if 10000 in Ts else Ts[-1]
        r = raw[(mech, T)]
        for c in CELLS:
            th0 = {}
            for d in DELTAS[1:]:
                i = kidx[(c, d, "honest")]
                nres, nfa = r["n_res"][i].sum(0), r["n_fa"][i].sum(0)
                if d not in th0:
                    th0[d] = X.theory_error(rt[c], d)
                for k in range(A.K):
                    if nres[k] < 30:
                        continue
                    pth = th0[d][k]["abs" if mech == "fixed" else "up"]
                    ps = nfa[k] / nres[k]
                    # audits are serially correlated (rho high) -> binomial se is optimistic; use as lower bound
                    se = np.sqrt(max(pth * (1 - pth), 1e-12) / nres[k])
                    iv.append(dict(mech=mech, rho=c[0], tau=c[1], rho_tau=rt[c], delta=d, agent=k, n_audits=float(nres[k]),
                                   sim=float(ps), theory=float(pth), z_binom=float((ps - pth) / se)))
    svals = np.array([e["sim"] for e in iv]); tvals = np.array([e["theory"] for e in iv])
    iv_sum = dict(n=len(iv), corr=float(np.corrcoef(svals, tvals)[0, 1]), mean_abs_err=float(np.abs(svals - tvals).mean()),
                  max_abs_err=float(np.abs(svals - tvals).max()), rows=iv)

    # ---------------- (ii) survival vs T: slopes ----------------
    fits = []
    for mech in ("fixed", "ada"):
        for c in CELLS:
            for d in DELTAS[1:]:
                Y = np.stack([1 - seedmat[(mech, T, c, d, "honest", "surv")] for T in Ts])  # [nT, seeds] mean-excl per seed
                # per-agent indicator version for finer E: use raw elim_t
                Yag = np.stack([(raw[(mech, T)]["elim_t"][kidx[(c, d, "honest")]] <= T).astype(float).reshape(-1) for T in Ts])
                E = Yag.mean(1)
                aud = np.array([float(raw[(mech, T)]["audits"][kidx[(c, d, "honest")]].mean()) for T in Ts])
                heff = [(-np.log(1 - e_) / a_ if 0 < e_ < 1 else None) for e_, a_ in zip(E, aud)]
                ent = dict(mech=mech, rho=c[0], tau=c[1], rho_tau=rt[c], delta=d, E_by_T=E.tolist(), h_eff_per_audit_by_T=heff,
                           saturated_or_zero=bool((E[-1] > 0.95) or (E[0] > 0.95) or (E[-1] == 0)),
                           audits_by_T=[float(raw[(mech, T)]["audits"][kidx[(c, d, "honest")]].mean()) for T in Ts])
                if len(Ts) >= 3 and E[-1] > 0:
                    # drop the 2-agents-per-seed correlation: resample seeds as blocks of K agents
                    Yb = np.stack([(raw[(mech, T)]["elim_t"][kidx[(c, d, "honest")]] <= T).astype(float) for T in Ts])  # [nT, seeds, K]
                    nS = Yb.shape[1]
                    lt = np.log(Ts); fl = 0.5 / (nS * A.K)

                    def fitb(Yb_):
                        Ee = np.clip(Yb_.mean((1, 2)), fl, 1 - fl); yy = -np.log(1 - Ee)
                        pw = np.polyfit(lt, np.log(yy), 1)           # power law: -ln S ~ T^beta
                        lg = np.polyfit(lt, yy, 1); res = yy - np.polyval(lg, lt)
                        pr = np.polyfit(lt, np.log(yy), 1); resp = np.log(yy) - np.polyval(pr, lt)
                        return pw[0], float((resp ** 2).sum()), float((res ** 2).sum() / max((yy ** 2).sum(), 1e-12)), lg[0]
                    pt = fitb(Yb)
                    bs = np.array([fitb(Yb[:, rng.integers(0, nS, nS)])[0] for _ in range(300)])
                    ent.update(beta=[float(pt[0]), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               loglin_slope=float(pt[3]), rel_sse_loglin=float(pt[2]), sse_logpower=float(pt[1]))
                fits.append(ent)
    # chain test: AdaAudit audits after first false exclusion
    chain = {}
    for T in Ts:
        r = raw[("ada", T)]
        rows = []
        for c in CELLS:
            for d in DELTAS[1:]:
                i = kidx[(c, d, "honest")]
                nel = r["nelim"][i]
                has = nel > 0
                if has.sum() < 3 or (~has).sum() < 3:
                    continue
                rows.append(dict(rho=c[0], tau=c[1], delta=d, frac_with_elim=float(has.mean()),
                                 audits_with=float(r["audits"][i][has].mean()), audits_without=float(r["audits"][i][~has].mean()),
                                 audits_post_with=float(r["audits_post"][i][has].mean()), nelim_with=float(nel[has].mean())))
        chain[str(T)] = rows
    ideal = {str(T): A.expected_audits_idealised(T) for T in Ts}

    # ---------------- (iii) feasibility + threshold bootstrap ----------------
    def feas_from(mech, T, b, resample):
        F = np.zeros((len(CELLS), len(DELTAS)), bool); Sv = np.zeros_like(F, float); Cv = np.zeros_like(F, float)
        for ci_, c in enumerate(CELLS):
            for di, d in enumerate(DELTAS):
                sv, cv = seedmat[(mech, T, c, d, "honest", "surv")], seedmat[(mech, T, c, d, b, "catch")]
                if resample:
                    sv, cv = sv[rng.integers(0, len(sv), len(sv))], cv[rng.integers(0, len(cv), len(cv))]
                Sv[ci_, di], Cv[ci_, di] = sv.mean(), cv.mean()
                F[ci_, di] = Sv[ci_, di] >= 0.9 and Cv[ci_, di] >= 0.9
        return F, Sv, Cv
    order = sorted(range(len(CELLS)), key=lambda i: rt[CELLS[i]])
    feas = {}
    for mech in ("fixed", "ada"):
        for T in Ts:
            for b in ("b0.1", "b0.3"):
                F, Sv, Cv = feas_from(mech, T, b, False)
                nb = 300
                thr, cellfrac = [], np.zeros_like(F, float)
                for _ in range(nb):
                    Fb, _, _ = feas_from(mech, T, b, True)
                    cellfrac += Fb / nb
                    ok = [rt[CELLS[i]] for i in order if Fb[i, 1:].any()]   # delta=0 excluded from feasible set (false-exclusion certain)
                    thr.append(min(ok) if ok else np.inf)
                thr = np.array(thr)
                feasible_cells = [rt[CELLS[i]] for i in order if F[i, 1:].any()]
                feas[f"{mech}|{T}|{b}"] = dict(
                    feasible=F.tolist(), surv=Sv.tolist(), catch=Cv.tolist(), cell_boot_frac=cellfrac.tolist(),
                    threshold_rho_tau=(min(feasible_cells) if feasible_cells else None),
                    threshold_boot_median=float(np.median(thr)) if np.isfinite(np.median(thr)) else None,
                    threshold_boot_ci=[(None if not np.isfinite(v_) else float(v_)) for v_ in (np.percentile(thr[np.isfinite(thr)], 2.5) if np.isfinite(thr).any() else np.inf, np.percentile(np.where(np.isfinite(thr), thr, 9), 97.5) if np.isfinite(thr).any() else np.inf)],
                    frac_boot_empty=float(np.isinf(thr).mean()),
                    intervals={f"{CELLS[i][0]}^{CELLS[i][1]}={rt[CELLS[i]]:.4g}": [DELTAS[j] for j in range(len(DELTAS)) if F[i, j]] for i in order})
    res = dict(meta=dict(seeds=seeds, Ts=Ts, p_fixed=P_FIXED, deltas=DELTAS, cells=CELLS, scenarios=SCN, K=A.K,
                         note="delta=0 is the consistency check; delta=0.5 added beyond the requested grid; cells (0.9,2),(0.99,10),(0.9,7),(0.99,50) are extras for rho^tau coincidences",
                         feasible_def="honest per-agent survival>=0.9 (scenario honest) AND liar caught within T >=0.9 (scenario b), delta>0"),
               consistency=cons, ideal_B_ada=ideal, cells=summ, collapse=collapse, single_audit=iv_sum, survival_fits=fits,
               ada_chain=chain, feasibility=feas)
    if a.quick:
        print(json.dumps(dict(cons=cons, collapse={k: v for k, v in collapse.items() if k != "worst"}), indent=0)[:1500]); return
    with open(os.path.join(ROOT, "results", "adaaudit_tol.json"), "w") as f:
        json.dump(res, f, default=float)
    print("analysis done", time.time() - t0, flush=True)
    make_figs(res, Ts)


def make_figs(res, Ts):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [tuple(c) for c in res["meta"]["cells"]]; D = res["meta"]["deltas"]; C = res["cells"]
    rt = lambda c: c[0] ** c[1]
    T = 10000
    col = {"fixed": "#d55e00", "ada": "#0072b2"}
    mk = {0.5: "o", 0.9: "s", 0.99: "^"}
    # ---- fig17a
    fig, ax = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for j, d in enumerate([0.1, 0.2, 0.3, 0.5]):
        for mech in ("fixed", "ada"):
            for c in cells:
                k = f"{mech}|{T}|{c[0]}|{c[1]}|{d}"; x = max(1 - rt(c), 3e-3)
                for row, (s, m) in enumerate((("honest", "honest_surv"), ("b0.3", "catch"), ("b0.1", "catch"))):
                    mu, e = C[k][s][m]
                    ax[row, j].errorbar(x, mu, yerr=e, fmt=mk[c[0]], color=col[mech], ms=5, alpha=.8, lw=1,
                                        label=f"{mech}" if (c == cells[0] and row == 0 and j == 0) else None)
        ax[0, j].set_title(f"delta={d}  (T=1e4)")
        for row in range(3):
            ax[row, j].axhline(.9, color="gray", ls=":"); ax[row, j].set_ylim(-.03, 1.03); ax[row, j].set_xscale("log"); ax[row, j].invert_xaxis()
    ax[0, 0].set_ylabel("honest survival"); ax[1, 0].set_ylabel("liar caught (b=0.3)"); ax[2, 0].set_ylabel("liar caught (b=0.1)")
    for j in range(4): ax[2, j].set_xlabel("1 - rho^tau  (log axis, informative ->; marker: rho .5 o / .9 square / .99 triangle)")
    ax[0, 0].legend()
    fig.suptitle("fig17a: outcomes against rho^tau (different (rho,tau) with equal rho^tau should overlap)")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig17a_collapse_rhotau.png"), dpi=130); plt.close(fig)
    # ---- fig17b
    order = sorted(range(len(cells)), key=lambda i: rt(cells[i]))
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for r_, b in enumerate(("b0.1", "b0.3")):
        for j, (mech, TT) in enumerate((("fixed", 10000), ("ada", 10000), ("fixed", 100000), ("ada", 100000))):
            f = res["feasibility"][f"{mech}|{TT}|{b}"]
            S_, C_, F_ = np.array(f["surv"]), np.array(f["catch"]), np.array(f["feasible"])
            Z = np.array([[F_[i, jj] for jj in range(1, len(D))] for i in order], float)
            Zb = np.array([[f["cell_boot_frac"][i][jj] for jj in range(1, len(D))] for i in order])
            a_ = ax[r_, j]; im = a_.imshow(Zb, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
            for ii in range(len(order)):
                for jj in range(1, len(D)):
                    a_.text(jj - 1, ii, f"{S_[order[ii], jj]:.2f}\n{C_[order[ii], jj]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if Zb[ii, jj - 1] < .6 else "k", fontweight="bold" if F_[order[ii], jj] else None)
            a_.set_xticks(range(len(D) - 1)); a_.set_xticklabels(D[1:])
            a_.set_yticks(range(len(order))); a_.set_yticklabels([f"{rt(cells[i]):.3g} ({cells[i][0]},{cells[i][1]})" for i in order], fontsize=7)
            thr = f["threshold_rho_tau"]
            a_.set_title(f"{mech} T={TT:.0e} {b}\nthreshold rho^tau={thr if thr is None else round(thr,3)}", fontsize=9)
            a_.set_xlabel("delta"); a_.set_ylabel("rho^tau (rho,tau)" if j == 0 else "")
    fig.subplots_adjust(wspace=.55, hspace=.35)
    fig.colorbar(im, ax=ax, label="bootstrap P(feasible: honest surv>=.9 & catch>=.9); text = surv / catch", shrink=.6)
    fig.suptitle("fig17b: feasible-delta phase diagram")
    fig.savefig(os.path.join(ROOT, "figures", "fig17b_feasible_delta_phase.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    # ---- fig17c
    fig, ax = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    show = [(0.9, 5), (0.99, 5), (0.9, 1), (0.99, 1)]
    cc = ["#1b9e77", "#7570b3", "#e6ab02", "#666666"]
    for j, d in enumerate([0.05, 0.1, 0.2, 0.3]):
        for mech, ls in (("fixed", "--"), ("ada", "-")):
            for c, co in zip(show, cc):
                S_ = [C[f"{mech}|{TT}|{c[0]}|{c[1]}|{d}"]["honest"]["honest_surv"][0] for TT in Ts]
                ax[0, j].plot(Ts, S_, ls, color=co, marker="o", ms=3, label=f"{mech} rho^tau={rt(c):.2f}" if j == 0 else None)
                E = np.clip(1 - np.array(S_), 1 / 192, None)
                ax[1, j].plot(Ts, -np.log(1 - np.clip(E, 0, 1 - 1 / 192)), ls, color=co, marker="o", ms=3)
        ax[0, j].set_xscale("log"); ax[0, j].set_title(f"delta={d}"); ax[0, j].set_ylim(-.03, 1.03)
        ax[1, j].set_xscale("log"); ax[1, j].set_yscale("log"); ax[1, j].set_xlabel("T")
    ax[0, 0].set_ylabel("honest survival"); ax[1, 0].set_ylabel("-ln(survival)  [fixed ~ T, ada ~ log T?]")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("fig17c: honest survival vs T (dashed fixed-p, solid AdaAudit)")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig17c_survival_vs_T.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
