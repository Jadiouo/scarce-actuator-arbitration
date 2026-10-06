"""Stage-1 2x2 experiment: Dai-Blanchard-Jaillet audit mechanisms (fixed-p M0 and AdaAudit)
under {iid, Markov} x {tau=0, tau>0}. Writes results/adaaudit_2x2.json, figures/fig16_adaaudit_2x2*.png.

    python3 scripts/run_adaaudit_2x2.py            # full run
    python3 scripts/run_adaaudit_2x2.py --quick    # smoke test, no files written
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from arbitration.gpu import adaaudit as A

DEV = "cuda"
S = 64
TS = [1000, 10000, 100000]
P_FIXED = 0.1
RHO_M, TAU_D = 0.9, 5  # pre-declared Markov persistence and delay for the 2x2 (the grid covers the rest)
CELLS = {  # name: (rho, tau, revpast)
    "iid_tau0": (0.0, 0, False), "markov_tau0": (RHO_M, 0, False),
    "iid_tauD": (0.0, TAU_D, False), "markov_tauD": (RHO_M, TAU_D, False),
    "ctrl_iid_tauD_revealpast": (0.0, TAU_D, True), "ctrl_markov_tauD_revealpast": (RHO_M, TAU_D, True),
}
BEH = {"honest": 0.0, "const_b0.3": 0.3, "const_b1.0": 1.0, "late_liar_b1.0": 1.0}
RHOS, TAUS = [0.0, 0.5, 0.9, 0.99], [0, 1, 5, 20]


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]


def tens(vals, rep, dt=torch.float64):
    return torch.tensor([v for v in vals for _ in range(rep)], device=DEV, dtype=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = 4 if a.quick else S
    Ts = [300, 1000] if a.quick else TS
    t0 = time.time()
    # ---------------- Part A: 2x2 (+ controls) x behaviours ----------------
    keys = [(c, b) for c in CELLS for b in BEH]
    rho = tens([CELLS[c][0] for c, b in keys], seeds)
    tau = tens([CELLS[c][1] for c, b in keys], seeds, torch.long)
    rp = tens([CELLS[c][2] for c, b in keys], seeds, torch.bool)
    db = tens([BEH[b] for c, b in keys], seeds)
    A_res = {}
    for mech in ["fixed", "ada"]:
        for T in Ts:
            ds = T - 50 if False else None
            # late liar starts at T-50; others at round 1: run the two groups with different dev_start
            out = {}
            for grp, ds in (("early", 1), ("late", max(T - 50, 1))):
                r = A.simulate(mech, T, rho, tau, rp, db, seed=1000 + T, p_fixed=P_FIXED, dev_start=ds)
                out[grp] = r
            for k, (c, b) in enumerate(keys):
                r = out["late" if b == "late_liar_b1.0" else "early"]
                sl = slice(k * seeds, (k + 1) * seeds)
                dev = b != "honest"
                hon = [0, 2] if dev else [0, 1, 2]
                el = (r["elim_t"][sl] <= T).double()
                grid_t = np.unique(np.round(np.logspace(0, np.log10(T), 30)).astype(int))
                et = r["elim_t"][sl][:, hon].cpu().numpy()
                curve = [float((et <= g).mean()) for g in grid_t]
                A_res[f"{mech}|{T}|{c}|{b}"] = dict(
                    R=ci(r["regret"][sl].cpu().numpy()), B=ci(r["audits"][sl].cpu().numpy()),
                    honest_excl_frac=ci(el[:, hon].mean(1).cpu().numpy()),
                    any_honest_excl=ci(el[:, hon].max(1).values.cpu().numpy()),
                    dev_excl=ci(el[:, 1].cpu().numpy()) if dev else None,
                    est_reset=ci(r["est_reset"][sl].cpu().numpy()), R_per_seed=r["regret"][sl].cpu().tolist(),
                    curve_t=grid_t.tolist(), curve_excl=curve)
            print(f"A {mech} T={T} done {time.time()-t0:.0f}s", flush=True)
    # slopes (log-log of mean R vs T), bootstrap CI over seeds
    rng = np.random.default_rng(0)
    slopes = {}
    for mech in ["fixed", "ada"]:
        for c, b in keys:
            Rs = np.array([A_res[f"{mech}|{T}|{c}|{b}"]["R_per_seed"] for T in Ts])  # [nT, seeds]
            lt = np.log(Ts)

            def sl_(R):
                m = R.mean(1)
                return None if (m <= 0).any() else float(np.polyfit(lt, np.log(m), 1)[0])
            pt = sl_(Rs)
            if pt is None:
                slopes[f"{mech}|{c}|{b}"] = None
                continue
            bs = [sl_(Rs[:, rng.integers(0, seeds, seeds)]) for _ in range(500)]
            slopes[f"{mech}|{c}|{b}"] = [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for v in A_res.values():
        v.pop("R_per_seed")
    # ---------------- consistency check data ----------------
    ideal = {str(T): A.expected_audits_idealised(T) for T in Ts}
    # ---------------- Part B: (rho, tau) grid at T=1e4, honest ----------------
    Tg = 1000 if a.quick else 10000
    gk = [(r_, t_) for r_ in RHOS for t_ in TAUS]
    rho = tens([g[0] for g in gk], seeds)
    tau = tens([g[1] for g in gk], seeds, torch.long)
    rp = tens([False for g in gk], seeds, torch.bool)
    db = tens([0.0 for g in gk], seeds)
    G = {}
    for mech in ["fixed", "ada"]:
        r = A.simulate(mech, Tg, rho, tau, rp, db, seed=77, p_fixed=P_FIXED)
        for k, (r_, t_) in enumerate(gk):
            sl = slice(k * seeds, (k + 1) * seeds)
            el = (r["elim_t"][sl] <= Tg).double()
            G[f"{mech}|{r_}|{t_}"] = dict(R=ci(r["regret"][sl].cpu().numpy()), B=ci(r["audits"][sl].cpu().numpy()),
                                          honest_excl_frac=ci(el.mean(1).cpu().numpy()))
    print(f"B done {time.time()-t0:.0f}s", flush=True)
    res = dict(meta=dict(seeds=seeds, Ts=Ts, p_fixed=P_FIXED, K=A.K, c=A.C_MIN, rho_markov=RHO_M, tau_delay=TAU_D,
                         grid_T=Tg, rhos=RHOS, taus=TAUS, ideal_B_ada=ideal, behaviours=BEH,
                         agents="0~U[c,1], 1~U[0,1], 2~Beta(2,1); Gaussian-copula AR(1), identical marginals",
                         deviator="agent 1: report min(1,u+b); late_liar starts at round T-50"),
               twobytwo=A_res, slopes=slopes, grid=G)
    if a.quick:
        print(json.dumps({k: v for k, v in A_res.items() if "honest" in k and "ada" in k and str(Ts[-1]) in k}, indent=0)[:1500])
        return
    json.dump(res, open(os.path.join(ROOT, "results", "adaaudit_2x2.json"), "w"))
    make_figs(res)


def make_figs(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts, tx = res["meta"]["Ts"], res["twobytwo"]
    cols = {"iid_tau0": "C0", "markov_tau0": "C1", "iid_tauD": "C2", "markov_tauD": "C3",
            "ctrl_iid_tauD_revealpast": "C4", "ctrl_markov_tauD_revealpast": "C5"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, mech in enumerate(["fixed", "ada"]):
        for c, col in cols.items():
            ls = "--" if c.startswith("ctrl") else "-"
            R = [tx[f"{mech}|{T}|{c}|honest"]["R"][0] for T in Ts]
            Bv = [tx[f"{mech}|{T}|{c}|honest"]["B"][0] for T in Ts]
            ax[i, 0].plot(Ts, np.maximum(R, 0.5), ls, color=col, marker="o", label=c)
            ax[i, 2].plot(Ts, Bv, ls, color=col, marker="o")
            e = tx[f"{mech}|{Ts[-1]}|{c}|honest"]
            ax[i, 1].plot(e["curve_t"], e["curve_excl"], ls, color=col)
        ax[i, 0].set(xscale="log", yscale="log", xlabel="T", ylabel="R_T (honest agents; floor 0.5)", title=f"{mech}: regret")
        ax[i, 1].set(xscale="log", xlabel=f"round t (T={Ts[-1]})", ylabel="fraction of honest agents eliminated", title=f"{mech}: honest exclusion")
        ax[i, 2].set(xscale="log", yscale="log", xlabel="T", ylabel="B_T", title=f"{mech}: audits")
    if res["meta"]["ideal_B_ada"]:
        ax[1, 2].plot(Ts, [res["meta"]["ideal_B_ada"][str(T)] for T in Ts], "k:", label="idealised M* (true q)")
        ax[1, 2].legend(fontsize=7)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"Dai et al. audits: 2x2 (rho={res['meta']['rho_markov']}, tau={res['meta']['tau_delay']}), all honest, "
                 f"{res['meta']['seeds']} seeds; dashed = delayed reveal of the true past type (control)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig16_adaaudit_2x2.png"), dpi=130)
    # heatmaps
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    rh, ta, Tg = res["meta"]["rhos"], res["meta"]["taus"], res["meta"]["grid_T"]
    for i, mech in enumerate(["fixed", "ada"]):
        for j, (key, ttl) in enumerate([("R", f"R_T / T (T={Tg})"), ("honest_excl_frac", "honest exclusion rate")]):
            M = np.array([[res["grid"][f"{mech}|{r}|{t}"][key][0] for t in ta] for r in rh])
            if key == "R":
                M = M / Tg
            im = ax[i, j].imshow(M, origin="lower", cmap="viridis", vmin=0)
            ax[i, j].set_xticks(range(len(ta)), ta); ax[i, j].set_yticks(range(len(rh)), rh)
            ax[i, j].set(xlabel="tau (rounds)", ylabel="rho", title=f"{mech}: {ttl}")
            for a_ in range(len(rh)):
                for b_ in range(len(ta)):
                    ax[i, j].text(b_, a_, f"{M[a_, b_]:.3f}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, ax=ax[i, j])
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig16_adaaudit_2x2_heatmap.png"), dpi=130)


if __name__ == "__main__":
    main()
