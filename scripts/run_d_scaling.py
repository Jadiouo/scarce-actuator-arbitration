"""Phase D (EXPLORATORY, not pre-registered): do the main effects change with N?

  check   one-off: multi_actuator (A=1) vs sim_torch, per seed
  calib   load calibration per N (scheme A: none's frac_any_dead = 0.35; scheme B: r*(100+5N) constant = N=3, r=.004)
  health  N in {3,4,8,16}: mean health of none / index_true / learned_index / fused_index / audit_disp_index(cal) / honest_index
          + information-value proxies (index_true - learned_index, fused_index - learned_index)
  proxy   proxies at the exact-DP cells (N=3: r=.002..016; N=4: r=.004,.008; floored instances, 20k steps) vs dp_step VoI_step
  games   strategic loss, reduced strategy set (const / timing / dist>30, b = 0..0.8 step 0.1), eps=.002, best response from
          truthful: fused_index, audit_index, audit_disp_index(cal); N=16 (and N=3, 8 under the same rules for a like-for-like trend)
  multi   2 actuators (exploratory, truthful only)
  figs

    python3 scripts/run_d_scaling.py <stage>      -> results/d_scaling.json, figures/fig15a/15b_*.png
"""
import json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from arbitration.model import make_config
from arbitration.experiments import ci95, paired
from arbitration.gpu import sim_torch as st
from arbitration.gpu import game_gpu as gg
from arbitration.gpu.sim_torch import simulate

OUT = os.path.join(ROOT, "results", "d_scaling.json")
SPREAD, BURN = 0.6, 500
TUNE, HOLD = list(range(16)), list(range(1000, 1032))
NS = [3, 4, 8, 16]
POLS = ["none", "index_true", "learned_index", "fused_index", "audit_disp_index", "honest_index"]
CFG = make_config("v2")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(d):
    json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def put(key, val):
    """merge-safe write of one top-level key (several stages may run concurrently)."""
    d = load(); d[key] = val; save(d)


def run(policy, seeds, n, rate, steps, **kw):
    return simulate(policy, seeds, n=n, rate=rate, spread=SPREAD, steps=steps, model=CFG, preempt=False, burn=BURN, **kw)


def calib_delta(n, rate, steps, seeds=TUNE):
    """selection-bias offset of the dispatch audit: mean audit sample of a TRUTHFUL run with offset 0 (seeds 0-15)."""
    o = run("audit_disp_index", seeds, n, rate, steps, disp_offset=0.0)
    return float(o["audit_sum"].sum() / o["audit_cnt"].sum())


# ------------------------------------------------------------------ check
def stage_check():
    from arbitration.gpu.multi_actuator import simulate_multi
    worst = {}
    for pol, off in (("none", 0.0), ("learned_index", 0.0), ("fused_index", 0.0), ("audit_disp_index", 0.03)):
        a = simulate_multi(pol, [0, 1, 2], n=8, rate=0.004, A=1, steps=1500, burn=0, disp_offset=off)
        b = simulate(pol, [0, 1, 2], n=8, rate=0.004, spread=SPREAD, steps=1500, model=CFG, preempt=False, burn=0,
                     disp_offset=off)
        worst[pol] = float(max(np.abs(a["mean_health"] - b["mean_health"]).max(),
                               np.abs(a["robot_health"] - b["robot_health"]).max()))
        log(f"check A=1 vs sim_torch {pol}: max abs err {worst[pol]:.2e}  (mean_health {a['mean_health'].round(4)})")
    put("check_multi_A1_vs_sim_torch", worst)


# ------------------------------------------------------------------ calibration
def rate_for_anydead(n, target=0.35, steps=4500):
    grid = np.geomspace(0.0003, 0.02, 16)
    fr = []
    for r in grid:
        o = run("none", TUNE, n, r, steps)
        fr.append(float(o["frac_any_dead"].mean()))
    fr = np.array(fr)
    lg = np.interp(target, np.maximum.accumulate(fr), np.log(grid))
    r = float(np.exp(lg))
    o = run("none", TUNE, n, r, steps)
    return r, float(o["frac_any_dead"].mean()), dict(zip(map(lambda x: f"{x:.5f}", grid), fr.round(4)))


def stage_calib():
    out = {}
    for n in NS:
        rA, fA, tab = rate_for_anydead(n)
        rB = 0.46 / (100 + 5 * n)
        oB = run("none", TUNE, n, rB, 4500)
        oA = run("none", TUNE, n, rA, 4500)
        out[str(n)] = {"A": {"rate": rA, "frac_any_dead_tune": fA, "none_mean_health_tune": float(oA["mean_health"].mean()),
                             "n_times_r": n * rA, "scan": tab},
                       "B": {"rate": rB, "frac_any_dead_tune": float(oB["frac_any_dead"].mean()),
                             "none_mean_health_tune": float(oB["mean_health"].mean()), "n_times_r": n * rB}}
        log(f"calib N={n}: A r={rA:.5f} (any_dead {fA:.3f}, N*r={n * rA:.4f})  B r={rB:.5f} (any_dead {out[str(n)]['B']['frac_any_dead_tune']:.3f})")
    put("calibration", out)


# ------------------------------------------------------------------ health + proxies
STEPS_H = 6500


def stage_health(schemes=("A", "B")):
    d = load()
    cal = d["calibration"]
    out = d.get("health", {})
    for sc in schemes:
        for n in NS:
            if f"{sc}|{n}" in out:
                continue
            r = cal[str(n)][sc]["rate"]
            t0 = time.time()
            delta = calib_delta(n, r, 4500)
            res = {}
            for p in POLS:
                kw = dict(disp_offset=delta) if p == "audit_disp_index" else {}
                o = run(p, HOLD, n, r, STEPS_H, **kw)
                res[p] = o
            L = res["learned_index"]["mean_health"]
            cell = {"N": n, "rate": r, "steps": STEPS_H, "burn": BURN, "seeds": "1000-1031", "disp_offset": delta, "pol": {}}
            for p in POLS:
                o = res[p]
                cell["pol"][p] = {"health": ci95(o["mean_health"]), "median": float(np.median(o["mean_health"])),
                                  "p05_min_health": ci95(o["min_health_p05"]), "dead_robot_frac": ci95(o["frac_dead_robot_steps"]),
                                  "minus_learned_index": paired(o["mean_health"], L),
                                  "per_seed": np.round(o["mean_health"], 5).tolist()}
            cell["proxy"] = {"index_true-learned_index": paired(res["index_true"]["mean_health"], L),
                             "fused_index-learned_index": paired(res["fused_index"]["mean_health"], L),
                             "audit_disp_index-learned_index": paired(res["audit_disp_index"]["mean_health"], L),
                             "honest_index-learned_index": paired(res["honest_index"]["mean_health"], L),
                             "fused_index-none": paired(res["fused_index"]["mean_health"], res["none"]["mean_health"]),
                             "learned_index-none": paired(L, res["none"]["mean_health"])}
            out[f"{sc}|{n}"] = cell
            log(f"health {sc} N={n} r={r:.5f} ({time.time() - t0:.0f}s): " +
                " ".join(f"{p}={cell['pol'][p]['health']['mean']:.3f}" for p in POLS) +
                f" | VoI proxy it-li {cell['proxy']['index_true-learned_index']['mean']:+.4f} fi-li {cell['proxy']['fused_index-learned_index']['mean']:+.4f}")
            put("health", out)


def stage_proxy():
    from arbitration.gpu import dp_step as ds
    dp = json.load(open(os.path.join(ROOT, "results", "dp_step.json")))
    d = load()
    out = {}
    cells = [(3, r, dp["gap_table"]["v2"][str(r)]) for r in (0.002, 0.004, 0.008, 0.016)] + \
            [(4, r, dp["n4"][str(r)]) for r in (0.004, 0.008)]
    for n, r, c in cells:
        seeds = c["seeds"]
        t0 = time.time()
        with ds.rounded_instances():
            res = {p: simulate(p, seeds, n=n, rate=r, spread=SPREAD, steps=20500, model=CFG, preempt=False, burn=BURN)
                   for p in ("learned_index", "index_true", "fused_index")}
        L = res["learned_index"]["mean_health"]
        px = {"index_true-learned_index": res["index_true"]["mean_health"] - L,
              "fused_index-learned_index": res["fused_index"]["mean_health"] - L}
        voi_key = "VoI_step_final" if "VoI_step_final" in c else "VoI_step"
        vstep = np.array(c[voi_key]["per_seed"])
        rec = {"N": n, "rate": r, "seeds": len(seeds), "dp_VoI_step_key": voi_key, "dp_VoI_step": ci95(vstep),
               "dp_VoI_sim": c["VoI_sim"], "proxy": {k: ci95(v) for k, v in px.items()}}
        for k, v in px.items():
            rec.setdefault("corr_with_dp_VoI_step_per_seed", {})[k] = float(np.corrcoef(v, vstep)[0, 1])
        rec["fused_recovers_frac_of_dp_VoI"] = float(px["fused_index-learned_index"].mean() / vstep.mean())
        rec["index_true_over_dp_VoI"] = float(px["index_true-learned_index"].mean() / vstep.mean())
        out[f"{n}|{r}"] = rec
        log(f"proxy N={n} r={r} ({time.time() - t0:.0f}s): DP VoI_step {vstep.mean():.4f}  index_true-learned {px['index_true-learned_index'].mean():+.4f}  "
            f"fused-learned {px['fused_index-learned_index'].mean():+.4f}  corr {rec['corr_with_dp_VoI_step_per_seed']}")
    put("proxy_vs_dp", out)


# ------------------------------------------------------------------ games (reduced strategy set)
TRUTH = gg.TRUTH
BS = [round(0.1 * k, 1) for k in range(1, 9)]


def game_cell(n, rate, steps, seeds, label, log_=log):
    t0 = time.time()
    env = dict(n=n, rate=rate, spread=SPREAD, steps=steps, sigma=0.15, model="v2", burn=BURN, opts={})
    delta = calib_delta(n, rate, steps)
    mechs = {"fused_index": ("fused_index", {}), "audit_index": ("audit_index", {}),
             "audit_disp_index|cal": ("audit_disp_index", {"disp_offset": delta}), "learned_index": ("learned_index", {})}
    keys = list(mechs)
    ml = [mechs[k] for k in keys]
    midx = {k: i for i, k in enumerate(keys)}
    acts = gg.make_actions(const=BS, others=BS, fams=(1, 3))          # truthful + const + timing + dist>30 (3 families)
    singles = [(i,) for i in range(n)]
    gk = [k for k in keys if k != "learned_index"]
    games = [gg._Game2(("g", k), midx[k], acts, singles, seeds, 0.002, TRUTH) for k in gk]
    util, ne = gg.solve_games2(games, ml, env, max_rounds=12, log=lambda *a: log_("   ", *a))
    log_(f"  {label}: best response done {time.time() - t0:.0f}s, {ne} sims")
    reqs = {}
    for k in keys:
        reqs[(k, "T")] = [(midx[k], s, tuple([TRUTH] * n)) for s in seeds]
    for g in games:
        reqs[(g.key[1], "E")] = [(g.mech, s, g.profile(i, g.pa[i])) for i, s in enumerate(seeds)]
    flat = list(dict.fromkeys(q for v in reqs.values() for q in v))
    pos = {q: i for i, q in enumerate(flat)}
    ev = [gg.evaluate_full(ml, flat, env, s) for s in (1, 2)]
    rh = np.mean([e["rh"] for e in ev], axis=0).mean(axis=1)
    p05 = np.mean([e["p05"] for e in ev], axis=0)
    get = lambda key, arr: np.array([arr[pos[q]] for q in reqs[key]])
    Lh = get(("learned_index", "T"), rh)
    cell = {"N": n, "rate": rate, "steps": steps, "seeds": len(seeds), "disp_offset": delta, "n_actions": len(acts),
            "learned_index_truthful": ci95(Lh), "mech": {}}
    for g in games:
        k = g.key[1]
        T_h, E_h = get((k, "T"), rh), get((k, "E"), rh)
        loss = T_h - E_h
        fam = {}
        for i in range(len(seeds)):
            for pl, ai in zip(g.players, g.pa[i]):
                nm = "truthful" if g.actions[ai] == TRUTH else gg.FAMS[g.actions[ai][0]][0]
                fam[nm] = fam.get(nm, 0) + 1
        bvals = np.array([g.actions[ai][1] for i in range(len(seeds)) for ai in g.pa[i]])
        cell["mech"][k] = {"truthful": ci95(T_h), "eq": ci95(E_h), "loss": paired(T_h, E_h), "loss_median": float(np.median(loss)),
                           "eq_minus_learned": paired(E_h, Lh), "truthful_minus_learned": paired(T_h, Lh),
                           "p05_truthful": ci95(get((k, "T"), p05)), "p05_eq": ci95(get((k, "E"), p05)),
                           "frac_converged": float(g.converged.mean()), "frac_cycle": float(g.cycle.mean()),
                           "rounds_mean": float(g.rounds.mean()), "max_gain_max": float(g.gain.max()),
                           "frac_gain_gt_2eps": float((g.gain > 0.004).mean()), "b_mean": float(bvals.mean()),
                           "families": fam, "per_seed_loss": loss.round(5).tolist()}
        m = cell["mech"][k]
        log_(f"  {label} {k}: loss {m['loss']['mean']:+.4f} [{m['loss']['lo']:+.4f},{m['loss']['hi']:+.4f}] "
             f"conv {m['frac_converged']:.2f} cycle {m['frac_cycle']:.2f} max_gain {m['max_gain_max']:.4f} fam {fam}")
    cell["runtime_s"] = time.time() - t0
    return cell


def stage_games(ns=(16, 8, 3), scheme="A", steps=4000, nseeds=32):
    d = load()
    cal = d["calibration"]
    out = d.get("games", {})
    for n in ns:
        r = cal[str(n)][scheme]["rate"]
        key = f"{scheme}|{n}"
        if key in out:
            continue
        out[key] = game_cell(n, r, steps, HOLD[:nseeds], f"N={n} r={r:.5f}")
        put("games", out)


# ------------------------------------------------------------------ 2 actuators
def stage_multi(n=8, steps=6500):
    from arbitration.gpu.multi_actuator import simulate_multi
    d = load()
    cal = d["calibration"]
    r1 = cal[str(n)]["A"]["rate"]
    out = {"N": n, "steps": steps, "burn": BURN, "exploratory": True, "loads": {}}
    for tag, r in (("same_r (r_single)", r1), ("double_r (2*r_single)", 2 * r1)):
        res = {}
        for A in (1, 2):
            res[A] = {}
            for p in ("none", "learned_index", "fused_index", "audit_disp_index"):
                t0 = time.time()
                off = 0.0
                if p == "audit_disp_index":
                    c = simulate_multi(p, TUNE, n=n, rate=r, A=A, steps=4500, burn=BURN, disp_offset=0.0)
                    off = float(c["audit_sum"].sum() / c["audit_cnt"].sum())
                res[A][p] = simulate_multi(p, HOLD, n=n, rate=r, A=A, steps=steps, burn=BURN, disp_offset=off)
                res[A][p]["offset"] = off
                log(f"multi {tag} A={A} {p}: {res[A][p]['mean_health'].mean():.4f} ({time.time() - t0:.0f}s)")
        cell = {"rate": r, "A": {}}
        for A in (1, 2):
            cell["A"][str(A)] = {p: {"health": ci95(res[A][p]["mean_health"]), "median": float(np.median(res[A][p]["mean_health"])),
                                     "frac_any_dead": ci95(res[A][p]["frac_any_dead"]), "offset": res[A][p]["offset"],
                                     "minus_learned_index": paired(res[A][p]["mean_health"], res[A]["learned_index"]["mean_health"])}
                                 for p in res[A]}
        cell["two_minus_one"] = {p: paired(res[2][p]["mean_health"], res[1][p]["mean_health"]) for p in res[1]}
        cell["main_effects_A2"] = {"fused_index-learned_index": paired(res[2]["fused_index"]["mean_health"], res[2]["learned_index"]["mean_health"]),
                                   "audit_disp_index-learned_index": paired(res[2]["audit_disp_index"]["mean_health"], res[2]["learned_index"]["mean_health"]),
                                   "learned_index-none": paired(res[2]["learned_index"]["mean_health"], res[2]["none"]["mean_health"])}
        cell["main_effects_A1"] = {"fused_index-learned_index": paired(res[1]["fused_index"]["mean_health"], res[1]["learned_index"]["mean_health"]),
                                   "audit_disp_index-learned_index": paired(res[1]["audit_disp_index"]["mean_health"], res[1]["learned_index"]["mean_health"]),
                                   "learned_index-none": paired(res[1]["learned_index"]["mean_health"], res[1]["none"]["mean_health"])}
        out["loads"][tag] = cell
    put("multi_actuator", out)


def stage_gref():
    """strategic losses already in results/game.json (read only), for the table next to the N=16 / reduced-set runs."""
    g = json.load(open(os.path.join(ROOT, "results", "game.json")))
    ref = {}
    for k, c in g["game_sweep"]["cells"].items():
        ref[f"game_sweep|{k}"] = {"rate": c["rate"], "steps": c["steps"], "family": "constant b only (b grid 0..0.6)",
                                  **{m: {"loss_out": c["mech"][m]["loss_out"], "frac_converged": c["mech"][m]["frac_converged"]}
                                     for m in ("fused_index", "audit_index")}}
    c = g["game_v2"]["cells"]["N3_r0.004_f0.5"]
    ref["game_v2|N3_r0.004_f0.5"] = {"rate": c["rate"], "steps": c["steps"], "burn": c["burn"], "family": "full 7-family set, b 0.05..0.8",
        **{m: {"loss": c["mech"][m]["games"]["full|truth|eps=0.002"]["loss"],
               "frac_converged": c["mech"][m]["games"]["full|truth|eps=0.002"]["frac_converged"]}
           for m in ("fused_index", "audit_index", "audit_disp_index|cal")}}
    put("game_json_reference", ref)


# ------------------------------------------------------------------ figures
def stage_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load()
    H = d["health"]
    cols = {"none": "#999999", "index_true": "#59a14f", "learned_index": "#4c78a8", "fused_index": "#e45756",
            "audit_disp_index": "#b07aa1", "honest_index": "#f28e2b"}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    # (a) mean health vs N
    ax = axs[0]
    for p in POLS:
        ys = [H[f"A|{n}"]["pol"][p]["health"] for n in NS]
        ax.errorbar(range(4), [y["mean"] for y in ys], yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]],
                    marker="o", ms=4, capsize=2, color=cols[p], label=p)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"N={n}\nr={H[f'A|{n}']['rate']:.4f}" for n in NS], fontsize=8)
    ax.set_ylabel("team mean health (32 seeds, 95% CI)"); ax.set_title("(a) policies, load calibrated to\nnone: frac_any_dead = 0.35", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    # (b) VoI proxies + DP
    ax = axs[1]
    for k, col, lab in (("index_true-learned_index", "#59a14f", "index_true - learned_index"),
                        ("fused_index-learned_index", "#e45756", "fused_index - learned_index"),
                        ("audit_disp_index-learned_index", "#b07aa1", "audit_disp_index - learned_index")):
        ys = [H[f"A|{n}"]["proxy"][k] for n in NS]
        ax.errorbar(range(4), [y["mean"] for y in ys], yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]],
                    marker="o", ms=4, capsize=2, color=col, label=lab)
    pv = d.get("proxy_vs_dp", {})
    for n, xi in ((3, 0), (4, 1)):
        k = f"{n}|0.004"
        if k in pv:
            y = pv[k]["dp_VoI_step"]
            ax.errorbar([xi + 0.12], [y["mean"]], yerr=[[y["mean"] - y["lo"]], [y["hi"] - y["mean"]]], marker="D", color="k", capsize=3,
                        label="exact DP VoI_step (r=.004)" if n == 3 else None)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"N={n}" for n in NS]); ax.set_title("(b) information-value proxies", fontsize=9)
    ax.set_ylabel("health difference (paired)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    # (c) strategic loss
    ax = axs[2]
    G = d.get("games", {})
    ns_g = [n for n in (3, 8, 16) if f"A|{n}" in G]
    for k, col, lab in (("fused_index", "#e45756", "fused_index (naive trust)"), ("audit_index", "#4c78a8", "audit_index (at arrival)"),
                        ("audit_disp_index|cal", "#b07aa1", "audit_disp_index (at dispatch)")):
        ys = [G[f"A|{n}"]["mech"][k]["loss"] for n in ns_g]
        ax.errorbar(range(len(ns_g)), [y["mean"] for y in ys], yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]],
                    marker="o", ms=4, capsize=2, color=col, label=lab)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(ns_g))); ax.set_xticklabels([f"N={n}" for n in ns_g])
    ax.set_title("(c) strategic loss at the best-response\nprofile (reduced strategy set)", fontsize=9)
    ax.set_ylabel("truthful - equilibrium team health"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    # (d) 2 actuators
    ax = axs[3]
    M = d.get("multi_actuator")
    if M:
        tag = list(M["loads"])[1]
        cell = M["loads"][tag]
        pols = ["none", "learned_index", "fused_index", "audit_disp_index"]
        w = 0.38
        for j, A in enumerate(("1", "2")):
            ys = [cell["A"][A][p]["health"] for p in pols]
            ax.bar(np.arange(4) + (j - .5) * w, [y["mean"] for y in ys], w, yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]],
                   label=f"{A} actuator(s)", capsize=2, color=["#bbbbbb", "#4c78a8"][j])
        ax.set_xticks(range(4)); ax.set_xticklabels(["none", "learned", "fused", "audit_disp"], fontsize=8)
        ax.set_title(f"(d) N=8, 1 vs 2 actuators, {tag}", fontsize=9)
        lo = min(cell["A"][A][p]["health"]["lo"] for A in ("1", "2") for p in pols)
        ax.set_ylim(max(0, lo - 0.05), None); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("Scaling with N (v2, shock 0.5; exploratory)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig15a_scaling_with_N.png"), dpi=150)
    plt.close(fig)
    if "B|3" in H:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, sc, ttl in ((axs[0], "A", "scheme A (any-dead = .35)"), (axs[1], "B", "scheme B (cycle damage const.)")):
            for k, col in (("index_true-learned_index", "#59a14f"), ("fused_index-learned_index", "#e45756")):
                ys = [H[f"{sc}|{n}"]["proxy"][k] for n in NS]
                ax.errorbar(range(4), [y["mean"] for y in ys], yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]],
                            marker="o", capsize=2, color=col, label=k)
            ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels([f"N={n}\nr={H[f'{sc}|{n}']['rate']:.4f}" for n in NS], fontsize=8)
            ax.set_title(ttl, fontsize=9); ax.grid(alpha=.3)
        axs[0].legend(fontsize=7)
        fig.suptitle("VoI proxies under two load calibrations (exploratory)")
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "figures", "fig15b_voi_proxy_load_schemes.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    st_ = sys.argv[1]
    for name, fn in (("check", stage_check), ("calib", stage_calib), ("health", stage_health), ("proxy", stage_proxy),
                     ("games", stage_games), ("multi", stage_multi), ("gref", stage_gref), ("figs", stage_figs)):
        if st_ == name:
            fn()
