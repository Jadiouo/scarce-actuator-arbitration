"""Regenerate all figures from results/results.json (bands are 95% CIs over seeds).

    python -m arbitration.figures [--results PATH] [--outdir DIR]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiments import ci95

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"none": "#b0b0a4", "none_random": "#c8a96e", "greedy_true": "#3f6b94",
       "nearest": "#4f7a4a", "honest": "#c4593a", "strategic": "#2b2a28", "priced": "#8a5db0"}
LAB = {"none": "Blind round-robin (spatial order)", "none_random": "Blind round-robin, random order",
       "greedy_true": "Greedy on true urgency (myopic)", "nearest": "Distance-only dispatch",
       "honest": "Honest urgency reporting", "strategic": "Strategic inflation",
       "priced": "Priced requests", "index_true": "True urgency per time (index)",
       "honest_index": "Reported urgency per time (index)", "learned": "Learned-rate greedy",
       "learned_index": "Learned-rate index", "tour_sqrt": "Sqrt-law tour (true rates)",
       "tour_sqrt_learned": "Sqrt-law tour (learned)", "rollout": "Rollout (learned beliefs)",
       "rollout_claim": "Rollout (reports)", "rollout_true": "Rollout (true state)"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)


def _series(cells, keys, metric="mean_health"):
    m = np.array([cells[k][metric]["mean"] for k in keys])
    lo = np.array([cells[k][metric]["lo"] for k in keys])
    hi = np.array([cells[k][metric]["hi"] for k in keys])
    return m, lo, hi


def _band(ax, x, m, lo, hi, c, **kw):
    ax.plot(x, m, "-o", color=c, ms=4, **kw)
    ax.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)


def fig1(d, out):
    rho = d["rho"]
    ls = d["load_sweep"]
    rates = [str(r) for r in d["config"]["rates"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6))
    load = [r["load"] for r in rho]
    a.plot(load, [r["rho"] for r in rho], "-o", color="#2b2a28")
    a.fill_between(load, [r["rho_lo"] for r in rho], [r["rho_hi"] for r in rho],
                   color="#2b2a28", alpha=0.15, lw=0, label="bootstrap 95% over seeds")
    a.axhline(0, color="k", lw=1)
    a.set_ylim(-1, 1)
    a.set_xlabel("Actuator load (fraction of time >=1 robot is fully degraded)")
    a.set_ylabel("Spearman rho across the 6 policies\n(dispatch precision vs team health)")
    a.set_title("(a) Rank correlation of precision and outcome vs load")
    a.legend(loc="upper right"); _style(a)
    for p in ("none", "greedy_true", "nearest", "honest", "strategic", "priced"):
        x = [ls[p][r]["precision"]["mean"] for r in rates]
        y = [ls[p][r]["mean_health"]["mean"] for r in rates]
        b.scatter(x, y, color=COL[p], s=40, label=LAB[p])
    b.set_xlabel("Dispatch precision (true urgency served / max urgency available)")
    b.set_ylabel("Team mean health")
    b.set_title("(b) Each point is one (policy, degradation rate)")
    b.legend(fontsize=8, loc="lower left"); _style(b)
    n = d["config"]["seeds"]
    fig.suptitle(f"Dispatch precision vs team outcome  ·  N=8, {n} seeds")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig2(d, out):
    ls, rates = d["load_sweep"], d["config"]["rates"]
    keys = [str(r) for r in rates]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5))
    for p in ("none", "none_random", "greedy_true", "nearest", "honest", "strategic"):
        _band(a, rates, *_series(ls[p], keys), COL[p], label=LAB[p])
    a.set_xscale("log"); a.set_xlabel("Degradation rate r (log scale)")
    a.set_ylabel("Team mean health")
    a.set_title("Team health by policy (bands: 95% CI over seeds)")
    a.legend(fontsize=8); _style(a)
    base = {k: np.array(ls["none_random"][k]["mean_health"]["per_seed"]) for k in keys}
    for p in ("none", "greedy_true", "nearest", "honest", "strategic"):
        c = [ci95(np.array(ls[p][k]["mean_health"]["per_seed"]) - base[k]) for k in keys]
        _band(b, rates, np.array([x["mean"] for x in c]), np.array([x["lo"] for x in c]),
              np.array([x["hi"] for x in c]), COL[p], label=LAB[p])
    b.axhline(0, color="k", lw=1); b.set_xscale("log")
    b.set_xlabel("Degradation rate r (log scale)")
    b.set_ylabel("Paired gain over random-order round-robin")
    b.set_title("Paired difference on identical seeds (95% CI)")
    b.legend(fontsize=8); _style(b)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig3(d, out):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (r, name) in zip(axs, (("0.001", "Below capacity"), ("0.004", "Overloaded"))):
        x = d["lambda_sweep"][r]; ks = list(x["curve"])
        lams = [float(k) for k in ks]
        xs = np.arange(len(lams))   # categorical grid: it is geometric
        m, lo, hi = _series(x["curve"], ks)
        _band(ax, xs, m, lo, hi, COL["priced"], label="Team mean health")
        i = lams.index(x["lam_star"])
        ax.plot(xs[i], m[i], "o", ms=14, mfc="none", mec=COL["priced"], mew=2)
        ax.set_xticks(xs); ax.set_xticklabels([f"{l:g}" for l in lams], rotation=60, fontsize=8)
        ax.set_xlabel("Request price lambda (grid is roughly geometric)")
        ax.set_ylabel("Team mean health", color=COL["priced"])
        ax2 = ax.twinx()
        pm, *_ = _series(x["curve"], ks, "precision")
        ax2.plot(xs, pm, "--s", color=COL["honest"], ms=4, label="Dispatch precision")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("Dispatch precision", color=COL["honest"])
        g = x["holdout_rel"] * 100
        ax.set_title(f"{name} (r={r})  ·  lambda*={x['lam_star']:g}, "
                     f"holdout {g:+.0f}% vs honest")
        _style(ax)
    fig.suptitle("Pricing the request: team health (CI band) and dispatch precision")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig4(d, out):
    """Sigma sweep."""
    sig = d["sigma_sweep"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, r in zip(axs, ("0.001", "0.004")):
        cells = sig[r]["cells"]
        for a in ("none", "nearest", "greedy_true", "honest", "priced"):
            keys = list(cells[a]); xs = [float(k) for k in keys]
            _band(ax, xs, *_series(cells[a], keys), COL[a],
                  label=LAB[a] + (f" (lambda={sig[r]['lam_priced']:g})" if a == "priced" else ""))
        ax.set_xlabel("Noise sigma of the private urgency estimate")
        ax.set_ylabel("Team mean health")
        ax.set_title(("Below capacity" if r == "0.001" else "Overloaded") + f" (r={r})")
        ax.legend(fontsize=8); _style(ax)
    fig.suptitle("Sensitivity to estimate noise (none and greedy_true do not read claims: flat by construction)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig5(d, out):
    """Lambda bracketing: curve relative to lam=0 with paired CI, and lambda* bootstrap."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (r, c) in zip(axs, (("0.001", "#3f6b94"), ("0.004", "#c4593a"))):
        x = d["lambda_sweep"][r]; ks = list(x["curve"])
        base = np.array(x["curve"][ks[0]]["mean_health"]["per_seed"])
        cs = [ci95(np.array(x["curve"][k]["mean_health"]["per_seed"]) - base) for k in ks]
        xs = np.arange(len(ks))
        _band(ax, xs, np.array([q["mean"] for q in cs]), np.array([q["lo"] for q in cs]),
              np.array([q["hi"] for q in cs]), c, label="paired gain vs lambda=0 (95% CI)")
        h = x["lam_star_boot_hist"]
        tot = sum(h.values())
        ax.bar(xs, [h[k] / tot * max(q["hi"] for q in cs) for k in ks], color=c, alpha=0.25,
               label="bootstrap distribution of argmax (scaled)")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels([f"{float(k):g}" for k in ks], rotation=60, fontsize=8)
        ax.set_xlabel("Request price lambda")
        ax.set_ylabel("Team health gain over honest")
        lo, hi = x["lam_star_ci"]
        ax.set_title(f"r={r}: lambda*={x['lam_star']:g}, boot 95% [{lo:g}, {hi:g}], "
                     f"interior={x['interior']}")
        ax.legend(fontsize=8); _style(ax)
    fig.suptitle("lambda* is bracketed by a wider grid in both regimes")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig6(d, out):
    """Hysteresis sweep."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, r in zip(axs, ("0.001", "0.004")):
        x = d["hyst_sweep"][r]; ks = list(x["cells"]); xs = [float(k) for k in ks]
        _band(ax, xs, *_series(x["cells"], ks), COL["honest"], label="honest + preempt, hysteresis h")
        for nm, ls in (("nopreempt", ":"), ("none", "--")):
            ax.axhline(x["reference"][nm]["mean_health"]["mean"], color=COL["none" if nm == "none" else "greedy_true"],
                       ls=ls, label={"nopreempt": "honest, no preemption", "none": "blind round-robin"}[nm])
        ax.set_xlabel("Hysteresis h (switch only if claim exceeds target's by > h)")
        ax.set_ylabel("Team mean health")
        ax.set_title(("Below capacity" if r == "0.001" else "Overloaded") + f" (r={r})")
        ax.legend(fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ v2 figures
COL2 = {**COL, "index_true": "#1f9e9e", "honest_index": "#e06030", "learned": "#e0a030", "learned_index": "#b5651d",
        "tour_sqrt": "#7a7a2a", "tour_sqrt_learned": "#a0a050", "rollout": "#d36fa5",
        "rollout_claim": "#e8a5c8", "rollout_true": "#8c2f6a"}


def _best_variants(cfgd, rates):
    """policy -> (arm key, per-rate stats) using, per policy, the variant (preempt on/off/
    hysteresis) with the highest mean health averaged over the rates."""
    best = {}
    for arm, cells in cfgd["cells"].items():
        pol = arm.split("|")[0]
        score = np.mean([cells[r]["mean_health"]["mean"] for r in rates])
        if pol not in best or score > best[pol][0]:
            best[pol] = (score, arm)
    return {pol: arm for pol, (_, arm) in best.items()}


def fig7(d, out):
    """v2 load sweep: mean health vs load for every policy (best variant each) with 95% CI,
    and the paired difference to blind round-robin."""
    v = d["v2_sweep"]; c = v["configs"]["v2"]
    rates = [str(r) for r in v["rates"]]; x = [float(r) for r in rates]
    best = _best_variants(c, rates)
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.5))
    for pol, arm in best.items():
        suf = "" if arm == pol else arm.split("|")[1]
        lab = LAB.get(pol, pol) + (f" [{suf}]" if suf else "")
        _band(a, x, *_series(c["cells"][arm], rates), COL2[pol], label=lab)
        if pol != "none":
            m = [c["paired_vs_none"][r][arm]["mean_health"]["mean"] for r in rates]
            lo = [c["paired_vs_none"][r][arm]["mean_health"]["lo"] for r in rates]
            hi = [c["paired_vs_none"][r][arm]["mean_health"]["hi"] for r in rates]
            _band(b, x, np.array(m), np.array(lo), np.array(hi), COL2[pol])
    b.axhline(0, color=COL["none"], lw=2)
    for ax, t, yl in ((a, "(a) v2 model: team mean health", "Team mean health"),
                      (b, "(b) paired difference to blind round-robin (95% CI)",
                       "mean health minus `none`")):
        ax.set_xscale("log"); ax.set_xlabel("Per-robot decay rate r (load increases to the right)")
        ax.set_ylabel(yl); ax.set_title(t); _style(ax)
    a.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig8(d, out):
    """Legacy vs v2: gap of every policy (its best variant) to the best policy at each load."""
    v = d["v2_sweep"]
    rates = [str(r) for r in v["rates"]]; x = [float(r) for r in rates]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, name, t in zip(axs, ("legacy", "v2"),
                           ("(a) legacy model (linear damage, iid noise)",
                            "(b) v2 model (shock damage, AR(1) noise)")):
        c = v["configs"][name]
        best = _best_variants(c, rates)
        mean = {pol: np.array([c["cells"][arm][r]["mean_health"]["mean"] for r in rates])
                for pol, arm in best.items()}
        top = np.max(list(mean.values()), axis=0)
        for pol, m in mean.items():
            ax.plot(x, top - m, "-o", ms=3, color=COL2[pol], label=LAB.get(pol, pol),
                    lw=2.5 if pol in ("none", "greedy_true") else 1.3)
        ax.set_xscale("log"); ax.set_xlabel("Per-robot decay rate r")
        ax.set_title(t); _style(ax)
    axs[0].set_ylabel("Gap to the best policy at that load (mean health)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out, dpi=150); plt.close(fig)


DPCOL = {"none": "#7a7a6e", "none_random": "#c8a96e", "index_true": "#3f6b94", "learned_index": "#4f9a8a",
         "rollout_true": "#8a5db0", "honest": "#c4593a", "honest_index": "#e08a5a",
         "greedy_true": "#2b4a6b", "dp_full": "#000000", "dp_age": "#d4a017"}


def fig9(d, out):
    """Optimality gap J*_full - mean health, per policy, vs load; legacy and shock."""
    dpd = d["dp_sweep"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, (model, title) in zip(axs, (("legacy", "(a) legacy: linear damage (age = health)"),
                                        ("v2", "(b) shock damage (hidden)"))):
        cells = dpd["main"][model]
        rates = sorted(cells, key=float)
        x = [float(r) for r in rates]
        for a in dpd["arms"]:
            g = [cells[r]["gap_vs_J_full"][a] for r in rates]
            m = np.array([c["mean"] for c in g])
            lo = np.array([c["lo"] for c in g]); hi = np.array([c["hi"] for c in g])
            bold = a in ("none", "dp_age", "dp_full")
            ax.plot(x, m, "-o", ms=4, color=DPCOL[a], lw=2.6 if bold else 1.2,
                    label=LAB.get(a, {"dp_full": "DP, sees true h (optimal, sim)",
                                      "dp_age": "DP, sees ages only (optimal)"}.get(a, a)))
            ax.fill_between(x, lo, hi, color=DPCOL[a], alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("Per-robot decay rate r (N = 3)")
        ax.set_title(title); _style(ax)
    axs[0].set_ylabel("Optimality gap  J*(full) - mean health  (95% paired CI)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out, dpi=150); plt.close(fig)


def _heat(ax, mat, xl, yl, xt, yt, title, cmap="viridis", fmt="{:.3f}", stars=None):
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=9)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            c = "w" if (v - lo) / max(hi - lo, 1e-12) < 0.55 else "k"
            ax.text(j, i, fmt.format(v) + ("*" if stars is not None and stars[i, j] else ""),
                    ha="center", va="center", fontsize=7, color=c)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def fig10(d, out):
    """Where does information have value (VoI_state) and where is blind round-robin
    far from optimal?  Rows: two medium loads."""
    reg = d["dp_sweep"]["region"]
    fr, sp, sv = reg["shock_fracs"], reg["spreads"], reg["services"]
    rates = [str(r) for r in reg["rates"]]
    fig, axs = plt.subplots(len(rates), 4, figsize=(19, 4.6 * len(rates)), squeeze=False)

    def grid(cells, ks, key, which):
        return np.array([[cells[f"{f}|{k}"][key]["mean"] if which is None
                          else cells[f"{f}|{k}"][key][which]["mean"] for k in ks] for f in fr])

    def star(cells, ks, key, which):
        return np.array([[(lambda c: c["lo"] > 0 or c["hi"] < 0)(
            cells[f"{f}|{k}"][key] if which is None else cells[f"{f}|{k}"][key][which])
            for k in ks] for f in fr])
    for row, r in enumerate(rates):
        h, s_ = reg["heat"][r], reg["service"][r]
        _heat(axs[row, 0], grid(h, sp, "VoI_state", None), "rate heterogeneity (spread)",
              "shock_frac", sp, fr, f"r={r}: VoI_state = J*(full) - J*(age)", "magma",
              stars=star(h, sp, "VoI_state", None))
        _heat(axs[row, 1], grid(h, sp, "gap_vs_J_full", "none"), "rate heterogeneity (spread)",
              "shock_frac", sp, fr, f"r={r}: gap of blind round-robin  J*(full) - none", "viridis",
              stars=star(h, sp, "gap_vs_J_full", "none"))
        _heat(axs[row, 2], grid(s_, sv, "VoI_state", None), "SERVICE steps", "shock_frac", sv, fr,
              f"r={r}: VoI_state (spread 0.6)", "magma", stars=star(s_, sv, "VoI_state", None))
        _heat(axs[row, 3], grid(s_, sv, "gap_vs_J_full", "none"), "SERVICE steps", "shock_frac",
              sv, fr, f"r={r}: gap of blind round-robin (spread 0.6)", "viridis",
              stars=star(s_, sv, "gap_vs_J_full", "none"))
    fig.suptitle("Exact benchmark, N = 3 (mean over seeds; * = 95% paired CI excludes 0)", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


MECH_LAB = {"honest_index": "honest_index (believe claims)",
            "fused_index": "fused_index (naive fusion)",
            "learned_index": "learned_index (ignore claims)",
            "audit_index": "audit_index (arrival audit)",
            "audit_penalty_index|ignore": "audit + ignore",
            "audit_penalty_index|demote": "audit + demote",
            "audit_disp_index": "audit_disp (dispatch audit)",
            "audit_disp_penalty_index|ignore": "audit_disp + ignore",
            "audit_disp_penalty_index|demote": "audit_disp + demote",
            "audit_disp_index|cal": "audit_disp [calibrated]",
            "audit_disp_penalty_index|ignore|cal": "audit_disp + ignore [calibrated]",
            "audit_disp_penalty_index|demote|cal": "audit_disp + demote [calibrated]",
            "audit_disp_penalty_index|ignore|cal+tuned": "audit_disp + ignore [cal+tuned]",
            "audit_disp_penalty_index|demote|cal+tuned": "audit_disp + demote [cal+tuned]"}
V2_CELLS = ("N3_r0.004_f0.5", "N8_r0.002_f0.5")
GKEY = "full|truth|eps=0.002"


def _v2cells(d):
    g = d["game_v2"]["cells"]
    return [(k, g[k]) for k in V2_CELLS if k in g]


def fig11(d, out):
    """Strategic loss: truthful vs approximate best-response profile team health, game_v2 completed cells."""
    cells = _v2cells(d)
    fig, axs = plt.subplots(1, len(cells), figsize=(8.5 * len(cells), 8.2), squeeze=False)
    for ax, (name, c) in zip(axs[0], cells):
        keys = list(MECH_LAB)
        keys = [k for k in keys if k in c["mech"]]
        y = np.arange(len(keys))[::-1]
        for off, which, col, lab in ((0.2, "truthful", "#9aa5b1", "truthful"),
                                     (-0.2, "eq", "#c4593a", "approx. best-response profile")):
            m, lo, hi = [], [], []
            for k in keys:
                v = c["mech"][k]
                r = v["truthful"] if which == "truthful" else (v["games"][GKEY]["eq"] if "games" in v else None)
                if r is None:
                    m.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                else:
                    m.append(r["mean"]); lo.append(r["lo"]); hi.append(r["hi"])
            m, lo, hi = map(np.array, (m, lo, hi))
            ax.barh(y + off, m, 0.38, color=col, label=lab, xerr=[m - lo, hi - m], error_kw=dict(lw=1))
        for yi, k in zip(y, keys):
            v = c["mech"][k]
            if "games" in v:
                l = v["games"][GKEY]["loss"]
                ax.text(0.995, yi, f"loss {l['mean']:.3f} [{l['lo']:.3f}, {l['hi']:.3f}]", ha="right",
                        va="center", fontsize=7, transform=ax.get_yaxis_transform())
            else:
                ax.text(0.995, yi, "no claims used (reference)", ha="right", va="center", fontsize=7,
                        color="#555", transform=ax.get_yaxis_transform())
        ax.set_yticks(y); ax.set_yticklabels([MECH_LAB[k] for k in keys], fontsize=8)
        ax.set_xlim(0, 1.6); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xlabel("team mean health (0-1; 95% CI over seeds)", loc="left")
        ax.set_title(f"N={c['N']}, r={c['rate']}, shock_frac={c['shock_frac']}, {c['n_actions']} actions, "
                     f"{c['steps']} steps, {c['seeds']} seeds", fontsize=9)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.31, -0.07), ncol=2, frameon=False); _style(ax)
    fig.suptitle("Strategic loss = truthful - approximate best-response profile team health "
                 "(game_v2, eps=0.002, inflation b<=0.8; profiles are approximate best-response, "
                 "not guaranteed equilibria; convergence rates in fig12)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=150); plt.close(fig)


def fig12(d, out):
    """Approximate best-response profile: mean inflation b and convergence rate (game_v2, eps=0.002)."""
    cells = _v2cells(d)
    fig, axs = plt.subplots(1, 2 * len(cells), figsize=(8 * len(cells), 6.5), squeeze=False)
    for i, (name, c) in enumerate(cells):
        keys = [k for k in MECH_LAB if k in c["mech"] and "games" in c["mech"][k]]
        y = np.arange(len(keys))[::-1]
        g = [c["mech"][k]["games"][GKEY] for k in keys]
        a, b = axs[0][2 * i], axs[0][2 * i + 1]
        a.barh(y, [r["b_mean"] for r in g], color="#8a5db0")
        a.set_xlim(0, 0.85); a.set_xlabel("mean inflation b at final profile (cap 0.8)")
        b.barh(y, [r["frac_converged"] for r in g], color="#3f6b94")
        b.set_xlim(0, 1); b.set_xlabel("fraction of seeds converged")
        a.set_yticks(y); a.set_yticklabels([MECH_LAB[k] for k in keys], fontsize=8)
        b.set_yticks(y); b.set_yticklabels([])
        a.set_title(f"N={c['N']}, r={c['rate']}, f={c['shock_frac']}, {c['n_actions']} actions", fontsize=9)
        for ax in (a, b):
            _style(ax)
    fig.suptitle("Approximate best-response profiles (game_v2, eps=0.002): inflation and convergence "
                 "(low convergence = only approximate)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=150); plt.close(fig)


def fig13(d, out, cells=None):
    """Strategic loss (truthful - equilibrium team health, holdout seeds) of every mechanism against every
    strategy family tested; '*' = 95% paired CI excludes 0; small number = fraction of seeds converged."""
    g = d["game_v2"]["cells"]
    keys = cells or [k for k in g if g[k].get("n_actions", 0) >= 50] or list(g)[:2]
    cols = ["fam|const", "fam|timing", "fam|dist10", "fam|dist30", "fam|taulow", "fam|habit", "fam|adapt",
            "full|truth|eps=0.002", "full|const0.6|eps=0.002", "coll"]
    lab = ["constant", "timing", "far>10", "far>30", "tau<.3", "habitual", "adaptive", "all families\n(truth start)",
           "all families\n(b=.6 start)", "coalitions"]
    fig, axs = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 6.2), squeeze=False)
    for ax, key in zip(axs[0], keys):
        c = g[key]
        mechs = [m for m in c["mech"] if "games" in c["mech"][m]]
        M = np.full((len(mechs), len(cols)), np.nan)
        txt = [[""] * len(cols) for _ in mechs]
        for i, m in enumerate(mechs):
            for j, col in enumerate(cols):
                r = c["mech"][m]["games"].get(col)
                if r is None:
                    continue
                M[i, j] = r["loss"]["mean"]
                txt[i][j] = f"{r['loss']['mean']:+.3f}{'*' if r['loss']['significant'] else ''}\n{r['frac_converged']:.0%}"
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=max(0.05, np.nanmax(M)), aspect="auto")
        for i in range(len(mechs)):
            for j in range(len(cols)):
                if txt[i][j]:
                    ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(mechs))); ax.set_yticklabels([m.replace("audit_", "a_").replace("_penalty_index", "_pen") for m in mechs], fontsize=7)
        ax.set_title(f"{key} ({c['n_actions']} actions, {c['seeds']} seeds): strategic loss (team health), "
                     "approx. best-response profiles", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "results.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args(argv)
    d = json.load(open(a.results))
    os.makedirs(a.outdir, exist_ok=True)
    for fn, f in (("fig1_precision_stops_predicting.png", fig1),
                  ("fig2_information_has_no_value.png", fig2),
                  ("fig3_pricing_the_request.png", fig3),
                  ("fig4_sigma_sweep.png", fig4),
                  ("fig5_lambda_bracketing.png", fig5),
                  ("fig6_hysteresis.png", fig6),
                  ("fig7_v2_load_sweep.png", fig7),
                  ("fig8_legacy_vs_v2_gap.png", fig8)):
        f(d, os.path.join(a.outdir, fn))
        print("wrote", fn)
    if "dp_sweep" in d:
        for fn, f in (("fig9_optimality_gap.png", fig9), ("fig10_voi_regions.png", fig10)):
            f(d, os.path.join(a.outdir, fn))
            print("wrote", fn)
    gp = os.path.join(ROOT, "results", "game.json")
    if os.path.exists(gp):
        gd = json.load(open(gp))
        if "game_v2" in gd:
            for fn, f in (("fig11_strategic_loss_v2.png", fig11), ("fig12_equilibrium_b_v2.png", fig12),
                          ("fig13_strategy_families.png", fig13)):
                f(gd, os.path.join(a.outdir, fn))
                print("wrote", fn)


if __name__ == "__main__":
    main()
