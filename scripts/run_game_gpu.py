"""Phase B on the GPU: reporting game (best-response equilibria, two start profiles), tuning of the
audit+penalty mechanism, and the VoI recovered by Bayesian fusion. Writes results/game.json
(layout {"game_sweep": {...}}, readable by figures.fig11 / fig12) and figures 11-12.

    python3 scripts/run_game_gpu.py            # full run
    python3 scripts/run_game_gpu.py --quick    # tiny smoke test -> results/game_quick.json
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from arbitration import run, dp
from arbitration.model import make_config
from arbitration.experiments import ci95, paired, _ratio_ci
from arbitration.gpu import game_gpu as gg
from arbitration.gpu.sim_torch import simulate

SPREAD = 0.6
MECHS = {
    "honest_index": ("honest_index", {}),
    "fused_index": ("fused_index", {}),
    "learned_index": ("learned_index", {}),
    "audit_index": ("audit_index", {}),
    "audit_penalty_index|ignore": ("audit_penalty_index", {"penalty_mode": "ignore"}),
    "audit_penalty_index|demote": ("audit_penalty_index", {"penalty_mode": "demote"}),
}
FULL = dict(cells=[(3, 0.004, 8000, 32), (8, 0.002, 3000, 32)], grid=[round(0.05 * k, 2) for k in range(13)],
            streams=[1, 2], max_rounds=12, starts=[0.0, 0.6], eps=0.002,
            tune_z=[1.5, 2.0, 3.0], tune_T=[100, 300, 1000], tune_seeds=16, tune_seeds_by_n={8: 8},
            voi_seeds=32, voi_steps=8000, dp_k=60, voi_sigmas=[0.05, 0.5])
QUICK = dict(cells=[(3, 0.004, 300, 3), (8, 0.002, 200, 2)], grid=[0.0, 0.3, 0.6], streams=[1],
             max_rounds=3, starts=[0.0, 0.6], eps=0.002, tune_z=[2.0], tune_T=[100], tune_seeds=2, tune_seeds_by_n={},
             voi_seeds=3, voi_steps=300, dp_k=10, voi_sigmas=[0.05])


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ref_job(args):
    """dp_full / dp_age (CPU) health on streams 0..2 for one N=3 instance."""
    seed, steps, K, rate, streams = args
    cfg = make_config("v2", dp_k=K)
    return {pol: {st: float(run(pol, n=3, rate=rate, spread=SPREAD, seed=seed, steps=steps, model=cfg,
                                preempt=False, stream=st).mean_health) for st in streams}
            for pol in ("dp_full", "dp_age")}


def _init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def cpu_refs(seeds, steps, K, rate, streams, pool):
    rows = list(pool.map(_ref_job, [(s, steps, K, rate, tuple(streams)) for s in seeds], chunksize=1))
    return {pol: {st: np.array([r[pol][st] for r in rows]) for st in streams} for pol in ("dp_full", "dp_age")}


def preflight(log):
    """GPU vs model.run per seed, all four game policies (N=3 and N=8)."""
    worst = {}
    rng = np.random.default_rng(7)
    for pol, opts, n, rate in (("fused", {}, 3, 0.004), ("fused_index", {}, 8, 0.002), ("audit_index", {}, 3, 0.004),
                               ("audit_penalty_index", {"penalty_mode": "ignore"}, 8, 0.004),
                               ("audit_penalty_index", {"penalty_mode": "demote", "audit_z": 1.0}, 3, 0.004)):
        cfg = make_config("v2", **opts)
        b = rng.uniform(0, 0.6, (3, n))
        o = simulate(pol, [0, 1, 2], n=n, rate=rate, steps=1500, spread=SPREAD, model=cfg, inflate=b)
        e = 0.0
        for i in range(3):
            r = run(pol, n=n, rate=rate, steps=1500, seed=i, spread=SPREAD, model=cfg, inflate=b[i])
            e = max(e, float(np.abs(o["robot_health"][i] - r.robot_health).max()), abs(o["n_services"][i] - r.n_services))
        worst[f"{pol}{opts or ''} N={n}"] = e
        log(f"  preflight {pol} {opts} N={n}: max abs err vs model.run {e:.2e}")
    return worst


def summarize_game(res, m, st, truth_in, truth_out, eq_in, eq_out, S, N, grid):
    br = res[(m, st)]
    bh = np.array([(br["b"] == g).sum() for g in grid])
    return {"truthful_in": ci95(truth_in), "eq_in": ci95(eq_in), "truthful_out": ci95(truth_out),
            "eq_out": ci95(eq_out), "loss_in": paired(truth_in, eq_in), "loss_out": paired(truth_out, eq_out),
            "b_eq": ci95(br["b"].mean(axis=1)), "b_eq_max": float(br["b"].max()),
            "b_hist": {"grid": list(grid), "count": bh.tolist()},
            "frac_converged": float(br["converged"].mean()), "frac_cycle": float(br["cycle"].mean()),
            "rounds_mean": float(br["rounds"].mean()), "max_gain": float(br["max_gain"].max()),
            "traj_mean_b": [float(x.mean()) for x in br["traj"]],
            "per_seed": {"b": br["b"].round(3).tolist(), "truthful_out": truth_out.round(5).tolist(),
                         "eq_out": eq_out.round(5).tolist()}}


def run_cell(n, rate, steps, ns, cfg, log, refs=None):
    seeds = list(range(ns))
    env = dict(n=n, rate=rate, spread=SPREAD, steps=steps, sigma=0.15, model="v2", opts={})
    keys = list(MECHS)
    mech_list = [MECHS[k] for k in keys]
    grid, starts, streams = cfg["grid"], cfg["starts"], cfg["streams"]
    t0 = time.time()
    res = gg.solve_games(mech_list, env, seeds, grid, cfg["eps"], cfg["max_rounds"], starts=starts, log=log)
    log(f"  N={n}: best-response done in {time.time() - t0:.0f}s, {res['n_eval']} sims")
    # one batched call per noise stream: truthful and every (mechanism, start) equilibrium profile
    reqs, slot = [], {}
    for m in range(len(keys)):
        for tag, prof in [("truth", np.zeros((ns, n)))] + [(st, res[(m, st)]["b"]) for st in starts]:
            slot[(m, tag)] = len(reqs)
            reqs += [(m, s, tuple(p)) for s, p in zip(seeds, prof)]
    H = {st: gg.evaluate_mixed(mech_list, reqs, env, st).mean(axis=1) for st in [0] + list(streams)}
    get = lambda st, m, tag: H[st][slot[(m, tag)]:slot[(m, tag)] + ns]
    cell = {"N": n, "rate": rate, "steps": steps, "seeds": ns, "mech": {}}
    for m, k in enumerate(keys):
        blocks = {}
        t_in = get(0, m, "truth")
        t_out = np.mean([get(st, m, "truth") for st in streams], axis=0)
        for st in starts:
            eq_in = get(0, m, st)
            eq_out = np.mean([get(x, m, st) for x in streams], axis=0)
            blocks[st] = summarize_game(res, m, st, t_in, t_out, eq_in, eq_out, ns, n, grid)
            blocks[st]["_eq_out"] = eq_out
        main = dict(blocks[starts[0]])
        main.pop("_eq_out")
        for st in starts[1:]:
            bb = dict(blocks[st]); eo = bb.pop("_eq_out")
            bs0, bs1 = res[(m, starts[0])]["b"], res[(m, st)]["b"]
            bb["vs_start0"] = {"same_profile_frac": float(np.all(bs0 == bs1, axis=1).mean()),
                               "b_mean_diff": paired(bs1.mean(1), bs0.mean(1)),
                               "eq_out_diff": paired(eo, blocks[starts[0]]["_eq_out"])}
            main[f"start{st}"] = bb
        cell["mech"][k] = main
    if refs is not None:
        full, age = refs["dp_full"], refs["dp_age"]
        rf = np.mean([full[s] for s in streams], axis=0)
        ra = np.mean([age[s] for s in streams], axis=0)
        cell["ref"] = {"dp_full": ci95(rf), "dp_age": ci95(ra)}
        cell["ref_per_seed"] = {"dp_full": rf.round(5).tolist(), "dp_age": ra.round(5).tolist()}
        voi = rf - ra
        for k, m in cell["mech"].items():
            for which in ("truthful_out", "eq_out"):
                v = np.array(m["per_seed"][which])
                m["pos_" + which] = _ratio_ci(v - ra, voi)
    return cell


def run_tuning(n, rate, steps, cfg, log, nseeds):
    seeds = list(range(nseeds))
    env = dict(n=n, rate=rate, spread=SPREAD, steps=steps, sigma=0.15, model="v2", opts={})
    names, mech_list = [], []
    for mode in ("ignore", "demote"):
        for z in cfg["tune_z"]:
            for T in cfg["tune_T"]:
                names.append(f"{mode}|z={z}|T={T}")
                mech_list.append(("audit_penalty_index", {"penalty_mode": mode, "audit_z": z, "penalty_T": T}))
    res = gg.solve_games(mech_list, env, seeds, cfg["grid"], cfg["eps"], cfg["max_rounds"], starts=(0.0,), log=log)
    S_ = len(seeds)
    reqs = [(m, sd, tuple(p)) for m in range(len(names)) for tag in ("t", "e")
            for sd, p in zip(seeds, np.zeros((S_, n)) if tag == "t" else res[(m, 0.0)]["b"])]
    H = np.mean([gg.evaluate_mixed(mech_list, reqs, env, st).mean(axis=1) for st in cfg["streams"]], axis=0)
    H = H.reshape(len(names), 2, S_)
    t_out, e_out = H[:, 0], H[:, 1]
    rows = {}
    for m, nm in enumerate(names):
        rows[nm] = {"truthful_out": ci95(t_out[m]), "eq_out": ci95(e_out[m]),
                    "loss_out": paired(t_out[m], e_out[m]), "b_eq": ci95(res[(m, 0.0)]["b"].mean(1)),
                    "frac_converged": float(res[(m, 0.0)]["converged"].mean())}
    best = max(rows, key=lambda k: rows[k]["eq_out"]["mean"])
    worst = min(rows, key=lambda k: rows[k]["eq_out"]["mean"])
    return {"N": n, "seeds": len(seeds), "configs": rows, "best_by_eq_health": best, "worst_by_eq_health": worst,
            "spread_eq_health": rows[best]["eq_out"]["mean"] - rows[worst]["eq_out"]["mean"],
            "note": "selection and evaluation use the same seeds (fresh noise streams only): optimistic for the best"}


def run_voi(cfg, refs, log):
    seeds = list(range(cfg["voi_seeds"]))
    steps = cfg["voi_steps"]
    arms = {"learned_index": ("learned_index", {}, 0.15), "fused": ("fused", {}, 0.15), "fused_index": ("fused_index", {}, 0.15),
            "fused_index_known_rates": ("fused_index", {"fuse_known_rates": True}, 0.15),
            "audit_index": ("audit_index", {}, 0.15), "honest_index": ("honest_index", {}, 0.15),
            "index_true": ("index_true", {}, 0.15)}
    for s in cfg["voi_sigmas"]:
        arms[f"fused_index@sigma={s}"] = ("fused_index", {}, s)
    mh = {}
    for k, (pol, opts, sg) in arms.items():
        r = simulate(pol, seeds, n=3, rate=0.004, steps=steps, spread=SPREAD, sigma=sg,
                     model=make_config("v2", dp_k=cfg["dp_k"], **opts), preempt=False)
        mh[k] = r["mean_health"]
        log(f"  voi arm {k}: {mh[k].mean():.4f}")
    mh["dp_full"], mh["dp_age"] = refs["dp_full"][0], refs["dp_age"][0]
    voi = mh["dp_full"] - mh["dp_age"]
    return {"N": 3, "rate": 0.004, "steps": steps, "K": cfg["dp_k"], "seeds": len(seeds),
            "VoI_sim": ci95(voi), "mean_health": {k: ci95(v) for k, v in mh.items()},
            "vs_learned_index": {k: paired(v, mh["learned_index"]) for k, v in mh.items()},
            "recovered_frac": {k: _ratio_ci(mh[k] - mh["learned_index"], voi)
                               for k in mh if k.startswith(("fused", "audit", "index_true"))},
            "position": {k: _ratio_ci(mh[k] - mh["dp_age"], voi) for k in mh},
            "per_seed": {k: v.round(6).tolist() for k, v in mh.items()}}



# ======================================================================================
# Phase 2 (pre-registered): strategy families, coalitions, adaptive threshold, eps sensitivity,
# calibrated dispatch audit, tuning on seeds 0-15, evaluation on holdout seeds 1000-1031.
# ======================================================================================
EPS_LIST = (0.001, 0.002, 0.005)
TUNE_SEEDS = list(range(16))
HOLD_SEEDS = list(range(1000, 1032))
FAM_NAMES = {f: v[0] for f, v in gg.FAMS.items()}
BURN = 500


def base_mechs(delta, tuned=None):
    """name -> (policy, opts). `delta` = calibration offsets {mechanism: offset}."""
    M = {
        "honest_index": ("honest_index", {}),
        "fused_index": ("fused_index", {}),
        "learned_index": ("learned_index", {}),
        "audit_index": ("audit_index", {}),
        "audit_penalty_index|ignore": ("audit_penalty_index", {"penalty_mode": "ignore"}),
        "audit_penalty_index|demote": ("audit_penalty_index", {"penalty_mode": "demote"}),
        "audit_disp_index": ("audit_disp_index", {}),
        "audit_disp_penalty_index|ignore": ("audit_disp_penalty_index", {"penalty_mode": "ignore"}),
        "audit_disp_penalty_index|demote": ("audit_disp_penalty_index", {"penalty_mode": "demote"}),
    }
    for k in ("audit_disp_index", "audit_disp_penalty_index|ignore", "audit_disp_penalty_index|demote"):
        pol, o = M[k]
        M[k + "|cal"] = (pol, {**o, "disp_offset": delta[k]})
    for k, o in (tuned or {}).items():          # tuned penalty parameters (selected on seeds 0-15)
        pol = M[k][0]
        M[k + "|cal+tuned"] = (pol, {**M[k][1], "disp_offset": delta[k], **o})
    return M


def cell_env(n, rate, steps, frac):
    return dict(n=n, rate=rate, spread=SPREAD, steps=steps, sigma=0.15, model="v2", burn=BURN,
                opts={"shock_frac": frac})


def calibrate(env, log):
    """Truthful runs on the CALIBRATION seeds (0-15): selection bias of the dispatch audit, per mechanism."""
    keys = ["audit_disp_index", "audit_disp_penalty_index|ignore", "audit_disp_penalty_index|demote"]
    M = base_mechs({k: 0.0 for k in keys})
    ml = [M[k] for k in keys] + [M["audit_index"], M["audit_penalty_index|ignore"]]
    names = keys + ["audit_index", "audit_penalty_index|ignore"]
    n = env["n"]
    reqs = [(m, s, tuple([gg.TRUTH] * n)) for m in range(len(ml)) for s in TUNE_SEEDS]
    r = gg.evaluate_full(ml, reqs, env, 0)
    delta, info = {}, {}
    S = len(TUNE_SEEDS)
    for m, k in enumerate(names):
        sl = slice(m * S, (m + 1) * S)
        mean = float(r["audit_sum"][sl].sum() / r["audit_cnt"][sl].sum())
        flags = r["audit_flags"][sl]
        info[k] = {"mean_sample_truthful": mean, "arrivals_per_robot": float(r["audit_cnt"][sl].mean()),
                   "frac_robots_ever_flagged_raw": float((flags > 0).mean()),
                   "flags_per_robot": float(flags.mean())}
        if k in keys:
            delta[k] = mean
    log(f"  calibration offsets {{{', '.join(f'{k}: {v:.4f}' for k, v in delta.items())}}}")
    return delta, info


def truthful_false_flags(env, M, keys, seeds, log):
    """Truthful play: fraction of robots flagged at least once (false positives), raw vs calibrated."""
    ml = [M[k] for k in keys]
    n, S = env["n"], len(seeds)
    reqs = [(m, s, tuple([gg.TRUTH] * n)) for m in range(len(ml)) for s in seeds]
    r = gg.evaluate_full(ml, reqs, env, 0)
    out = {}
    for m, k in enumerate(keys):
        fl = r["audit_flags"][m * S:(m + 1) * S]
        out[k] = {"frac_robots_ever_flagged": float((fl > 0).mean()), "flags_per_robot": float(fl.mean())}
    return out


def acts_full():
    return gg.make_actions(const=[round(0.05 * k, 2) for k in range(1, 17)], others=[round(0.1 * k, 1) for k in range(1, 9)])


def acts_reduced():
    return gg.make_actions(const=[round(0.1 * k, 1) for k in range(1, 9)], others=(0.2, 0.4, 0.8))


def summarize(rh_t, rh_e, p_t, p_e, d_t, d_e, rh_l):
    """per-seed arrays (stream-averaged): team health truthful / eq, p05, dead; learned_index truthful health."""
    loss = rh_t - rh_e
    return {"truthful": ci95(rh_t), "eq": ci95(rh_e), "loss": paired(rh_t, rh_e),
            "loss_median": float(np.median(loss)), "eq_minus_learned": paired(rh_e, rh_l),
            "truthful_minus_learned": paired(rh_t, rh_l),
            "p05_truthful": ci95(p_t), "p05_eq": ci95(p_e), "dead_truthful": ci95(d_t), "dead_eq": ci95(d_e),
            "per_seed": {"loss": loss.round(5).tolist(), "eq": rh_e.round(5).tolist(), "truthful": rh_t.round(5).tolist()}}


def run_cell2(n, rate, steps, frac, acts, adapt, delta, tuned, log, collusion=True, seeds=HOLD_SEEDS,
              eps_list=EPS_LIST, mech_fams=None):
    env = cell_env(n, rate, steps, frac)
    M = base_mechs(delta, tuned)
    keys = [k for k in M if k != "learned_index"]               # claims are ignored by learned_index
    ml = [M[k] for k in M]
    midx = {k: i for i, k in enumerate(M)}
    pen = lambda k: "penalty" in k
    singles = [(i,) for i in range(n)]
    pairs = [(0, 1)] + [(i,) for i in range(2, n)] if n == 3 else [(0, 1), (2, 3), (4, 5), (6,), (7,)]
    starts = (gg.TRUTH, (0, 0.6))
    games = []
    fam_ids = [f for f in gg.FAMS if f != 6]
    for k in keys:
        a_k = acts + [(6, c) for c in adapt] if pen(k) else acts
        if mech_fams is not None:                                 # reduced strategy set (deviation from the pre-registration)
            fk = mech_fams.get(k.replace("|cal+tuned", "").replace("|cal", ""), [0, 1, 3])
            a_k = [gg.TRUTH] + [a for a in acts if a != gg.TRUTH and a[0] in fk] + ([(6, c) for c in adapt] if pen(k) else [])
            games.append(gg._Game2(("full", k, gg.TRUTH, 0.002), midx[k], a_k, singles, seeds, 0.002, gg.TRUTH))
            continue
        for st in starts:
            for e in (eps_list if st == gg.TRUTH or len(eps_list) == 1 else (0.002,)):
                games.append(gg._Game2(("full", k, st, e), midx[k], a_k, singles, seeds, e, st))
        for f in fam_ids:                                        # family-restricted games (truthful start)
            a_f = [a for a in a_k if a == gg.TRUTH or a[0] == f]
            games.append(gg._Game2(("fam", k, f), midx[k], a_f, singles, seeds, 0.002, gg.TRUTH))
        if pen(k):
            a_f = [a for a in a_k if a == gg.TRUTH or a[0] == 6]
            games.append(gg._Game2(("fam", k, 6), midx[k], a_f, singles, seeds, 0.002, gg.TRUTH))
        if collusion and mech_fams is None and k in ("fused_index", "audit_index", "audit_disp_index|cal",
                               "audit_disp_penalty_index|ignore|cal", "audit_disp_penalty_index|demote|cal"):
            games.append(gg._Game2(("coll", k), midx[k], acts_reduced() + ([(6, c) for c in adapt] if pen(k) else []),
                                   pairs, seeds, 0.002, gg.TRUTH))
    cell_ff = truthful_false_flags(env, M, [k for k in M if "penalty" in k], seeds, log)
    log(f"  {len(games)} games, {len(seeds)} seeds, {len(acts)} actions")
    t0 = time.time()
    util, n_eval = gg.solve_games2(games, ml, env, 12, log=log)
    log(f"  best-response done in {time.time() - t0:.0f}s, {n_eval} sims")
    # final evaluation on fresh noise streams (1, 2)
    reqs = {}
    for k in M:
        reqs[(midx[k], "T")] = [(midx[k], s, tuple([gg.TRUTH] * n)) for s in seeds]
    for g in games:
        reqs[(g.key, "E")] = [(g.mech, s, g.profile(i, g.pa[i])) for i, s in enumerate(seeds)]
    flat, slot = [], {}
    for key, rl in reqs.items():
        slot[key] = len(flat); flat += rl
    uniq = list(dict.fromkeys(flat))
    pos = {q: i for i, q in enumerate(uniq)}
    ev = [gg.evaluate_full(ml, uniq, env, st) for st in (1, 2)]
    avg = lambda name: np.mean([e[name] for e in ev], axis=0)
    rh, p05, dead = avg("rh").mean(axis=1), avg("p05"), avg("dead")
    get = lambda key, arr: np.array([arr[pos[q]] for q in reqs[key]])
    Lh = get((midx["learned_index"], "T"), rh)
    cell = {"N": n, "rate": rate, "steps": steps, "shock_frac": frac, "seeds": len(seeds), "burn": BURN,
            "n_actions": len(acts), "learned_index": ci95(Lh), "mech": {}, "n_sims_br": n_eval,
            "truthful_false_flags": cell_ff}
    for k in M:
        cell["mech"][k] = {"truthful": ci95(get((midx[k], "T"), rh)),
                           "truthful_minus_learned": paired(get((midx[k], "T"), rh), Lh)}
    for g in games:
        k = g.key[1]
        T_h = get((g.mech, "T"), rh)
        rec = summarize(T_h, get((g.key, "E"), rh), get((g.mech, "T"), p05), get((g.key, "E"), p05),
                        get((g.mech, "T"), dead), get((g.key, "E"), dead), Lh)
        conv = g.converged
        pa = g.pa
        fam_count = {}
        for i in range(len(seeds)):
            for pl, ai in zip(g.players, pa[i]):
                f = g.actions[ai][0]
                fam_count[FAM_NAMES[f] if g.actions[ai] != gg.TRUTH else "truthful"] = fam_count.get(
                    FAM_NAMES[f] if g.actions[ai] != gg.TRUTH else "truthful", 0) + len(pl)
        bvals = np.array([g.actions[ai][1] for i in range(len(seeds)) for pl, ai in zip(g.players, pa[i]) for _ in pl])
        rec.update(frac_converged=float(conv.mean()), frac_cycle=float(g.cycle.mean()),
                   rounds_mean=float(g.rounds.mean()), max_gain_mean=float(g.gain.mean()),
                   max_gain_max=float(g.gain.max()), frac_gain_gt_2eps=float((g.gain > 2 * g.eps).mean()),
                   eps=g.eps, b_mean=float(bvals.mean()), families=fam_count,
                   profiles=[[list(g.actions[ai]) for ai in row] for row in pa.tolist()])
        if conv.sum() >= 3:
            rec["loss_converged_only"] = paired(T_h[conv], get((g.key, "E"), rh)[conv])
        if g.key[0] == "full":
            slotname = f"full|{gg_name(g.key[2])}|eps={g.key[3]}"
        elif g.key[0] == "fam":
            slotname = f"fam|{FAM_NAMES[g.key[2]]}"
        else:
            slotname = "coll"
        cell["mech"][k].setdefault("games", {})[slotname] = rec
    cell["oos_check"] = oos_check(games, ml, env, seeds, log)
    return cell


def run_tuning2(n, rate, steps, frac, delta, log, zs=(1.5, 2.0, 3.0), Ts=(100, 300, 1000)):
    """Penalty parameters of the dispatch audit, selected on seeds 0-15 by equilibrium team health
    (best-response over a reduced family set, fresh noise streams). Returns (table, best per mode)."""
    env = cell_env(n, rate, steps, frac)
    seeds = TUNE_SEEDS
    ml, names = [], []
    for mode in ("ignore", "demote"):
        for z in zs:
            for T in Ts:
                key = f"audit_disp_penalty_index|{mode}"
                names.append((mode, z, T))
                ml.append(("audit_disp_penalty_index", {"penalty_mode": mode, "audit_z": z, "penalty_T": T,
                                                        "disp_offset": delta[key]}))
    acts = gg.make_actions(const=[round(0.1 * k, 1) for k in range(1, 9)], others=(0.2, 0.4, 0.8), fams=(1, 3, 4, 5),
                           adapt=(0.2, 0.4, 0.8))
    singles = [(i,) for i in range(n)]
    games = [gg._Game2(("t", i), i, acts, singles, seeds, 0.002, gg.TRUTH) for i in range(len(ml))]
    t0 = time.time()
    util, ne = gg.solve_games2(games, ml, env, 12, log=log)
    log(f"  tuning N={n}: {len(games)} configs, {ne} sims, {time.time() - t0:.0f}s")
    reqs = [(i, s, tuple([gg.TRUTH] * n)) for i in range(len(ml)) for s in seeds]
    reqs += [(g.mech, s, g.profile(k, g.pa[k])) for g in games for k, s in enumerate(seeds)]
    uniq = list(dict.fromkeys(reqs))
    ev = [gg.evaluate_full(ml, uniq, env, st) for st in (1, 2)]
    rh = np.mean([e["rh"] for e in ev], axis=0).mean(axis=1)
    pos = {q: i for i, q in enumerate(uniq)}
    S = len(seeds)
    rows = {}
    for i, (mode, z, T) in enumerate(names):
        tr = np.array([rh[pos[(i, s, tuple([gg.TRUTH] * n))]] for s in seeds])
        eq = np.array([rh[pos[(g.mech, s, g.profile(k, g.pa[k]))]] for g in [games[i]] for k, s in enumerate(seeds)])
        rows[f"{mode}|z={z}|T={T}"] = {"mode": mode, "z": z, "T": T, "truthful": ci95(tr), "eq": ci95(eq),
                                       "loss": paired(tr, eq), "frac_converged": float(games[i].converged.mean())}
    best = {}
    for mode in ("ignore", "demote"):
        cand = {k: v for k, v in rows.items() if v["mode"] == mode}
        kbest = max(cand, key=lambda k: cand[k]["eq"]["mean"])
        best[f"audit_disp_penalty_index|{mode}"] = {"audit_z": cand[kbest]["z"], "penalty_T": cand[kbest]["T"], "config": kbest}
    return {"N": n, "rate": rate, "seeds": "0-15", "table": rows, "best": best}


OOS_MECHS = ("audit_index", "fused_index", "audit_disp_index|cal", "audit_disp_penalty_index|ignore|cal",
             "audit_disp_penalty_index|demote|cal")


def oos_check(games, ml, env, seeds, log):
    """Out-of-sample verification of the final profiles of the all-family games (truthful start, eps 0.002):
    every unilateral deviation is evaluated on noise stream 1 (selection) and stream 2 (evaluation).
    gain = value on stream 2 of the deviation that looks best on stream 1, minus the current action's value
    on stream 2 (selection-bias free). Per mechanism: mean over seeds of the largest per-player gain, and the
    largest per-player mean gain."""
    out = {}
    for g in games:
        if not (g.key[0] == "full" and g.key[2] == gg.TRUTH and g.key[3] == 0.002 and g.key[1] in OOS_MECHS):
            continue
        reqs = []
        for k, sd in enumerate(seeds):
            for p in range(len(g.players)):
                row = g.pa[k].copy()
                for ai in range(len(g.actions)):
                    row[p] = ai
                    reqs.append((g.mech, sd, g.profile(k, row)))
        uniq = list(dict.fromkeys(reqs))
        pos = {q: i for i, q in enumerate(uniq)}
        ev = [gg.evaluate_full(ml, uniq, env, st)["rh"] for st in (1, 2)]
        S, P, A = len(seeds), len(g.players), len(g.actions)
        V = [np.zeros((S, P, A)) for _ in range(2)]
        i = 0
        for k in range(S):
            for p in range(P):
                for ai in range(A):
                    for st in range(2):
                        V[st][k, p, ai] = ev[st][pos[reqs[i]]][list(g.players[p])].mean()
                    i += 1
        cur = g.pa[:, :, None]
        c1 = np.take_along_axis(V[0], cur, 2)[..., 0]
        c2 = np.take_along_axis(V[1], cur, 2)[..., 0]
        sel = V[0].argmax(2)
        gain = np.take_along_axis(V[1], sel[..., None], 2)[..., 0] - c2
        gain_naive = V[0].max(2) - c1
        out[g.key[1]] = {"mean_over_seeds_of_max_player_gain": float(gain.max(1).mean()),
                         "max_player_mean_gain": float(gain.mean(0).max()),
                         "naive_in_sample_gain_mean": float(gain_naive.max(1).mean()),
                         "frac_seed_players_with_gain_gt_eps": float((gain > 0.002).mean()),
                         "mean_gain_ci": ci95(gain.mean(1))}
        log(f"  oos {g.key[1]}: split-sample max-player gain {out[g.key[1]]['mean_over_seeds_of_max_player_gain']:.4f}, "
            f"max per-player mean {out[g.key[1]]['max_player_mean_gain']:.4f}")
    return out


def gg_name(a):
    return "truth" if a == gg.TRUTH else f"{FAM_NAMES[a[0]]}{a[1]}"

def top_families(cells, n):
    """For every mechanism: the 3 strategy families (ids) with the largest family-restricted loss, pooled over the
    finished shock_frac = 0.5 cells with the same N."""
    tot = {}
    for key, c in cells.items():
        if c["N"] != n or c["shock_frac"] != 0.5:
            continue
        for m, v in c["mech"].items():
            for gname, r in v.get("games", {}).items():
                if gname.startswith("fam|"):
                    f = [i for i, nm in FAM_NAMES.items() if nm == gname[4:]][0]
                    tot.setdefault(m.replace("|cal+tuned", "").replace("|cal", ""), {}).setdefault(f, []).append(r["loss"]["mean"])
    return {m: [f for f, _ in sorted(((f, np.mean(v)) for f, v in fs.items() if f != 6), key=lambda x: -x[1])[:3]]
            for m, fs in tot.items()}


def main2(a):
    """python3 scripts/run_game_gpu.py --phase2 [--only KEY ...] [--quick]"""
    path = os.path.join(ROOT, "results", "game_v2_quick.json" if a.quick else "game.json")
    log = lambda s_: print(s_, flush=True)
    d = json.load(open(path)) if os.path.exists(path) and not a.quick else {}
    v2 = d.get("game_v2", {"cells": {}, "tuning": {}, "calibration": {}})
    v2["meta"] = {"seeds_tuning": "0-15", "seeds_holdout": "1000-1031", "burn": BURN, "eps": list(EPS_LIST),
                  "families": FAM_NAMES, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if a.quick:
        cells = [("N3q", 3, 0.004, 300, 0.5, "red")]
        globals().update(HOLD_SEEDS=[1000, 1001], TUNE_SEEDS=[0, 1], BURN=100)
        gg_acts = lambda tag: gg.make_actions(const=[0.2, 0.4], others=(0.4,), fams=(1, 4))
    else:
        cells = [("N3_r0.004_f0.5", 3, 0.004, 4000, 0.5, "full"), ("N8_r0.002_f0.5", 8, 0.002, 2000, 0.5, "mid"),
                 ("N3_r0.002_f0.5", 3, 0.002, 4000, 0.5, "red"), ("N3_r0.008_f0.5", 3, 0.008, 4000, 0.5, "red"),
                 ("N8_r0.004_f0.5", 8, 0.004, 2000, 0.5, "red"), ("N8_r0.008_f0.5", 8, 0.008, 2000, 0.5, "red")]
        for f in (0.25, 0.75):
            cells += [(f"N{n}_r{r}_f{f}", n, r, 4000 if n == 3 else 2000, f, "red")
                      for n in (3, 8) for r in (0.002, 0.004, 0.008)]
        gg_acts = lambda tag: acts_full() if tag == "full" else acts_reduced()
    only = set(a.only or [])
    outp = a.out or path
    dump = lambda: (os.makedirs(os.path.dirname(outp), exist_ok=True),
                    json.dump({**d, "game_v2": v2}, open(outp, "w"), indent=1))
    tuned = v2.get("tuned_params", {})
    for key, n, rate, steps, frac, tag in cells:
        if (only and key not in only) or key in v2["cells"]:
            continue
        t0 = time.time()
        env = cell_env(n, rate, steps, frac)
        log(f"== cell {key} (actions: {tag})")
        delta, info = calibrate(env, log)
        v2["calibration"][key] = {"offsets": delta, "info": info}
        if not tuned and not a.quick and key == cells[0][0]:           # tune once, on the first cell, seeds 0-15
            tr = run_tuning2(n, rate, steps, frac, delta, log)
            v2["tuning"][key] = tr
            tuned = {k: {"audit_z": v["audit_z"], "penalty_T": v["penalty_T"]} for k, v in tr["best"].items()}
            v2["tuned_params"] = tuned
            dump()
        acts = gg_acts(tag)
        adapt = (0.2, 0.4, 0.8)
        mech_fams = None
        if frac != 0.5:                                      # shock 0.25 / 0.75: top-3 families per mechanism from the 0.5 cells
            mech_fams = top_families(v2["cells"], n)
        eps_list = EPS_LIST if tag in ("full", "mid") else (0.002,)
        cell = run_cell2(n, rate, steps, frac, acts, adapt, delta, tuned, log, collusion=(tag in ("full", "mid")),
                         seeds=HOLD_SEEDS, eps_list=eps_list, mech_fams=mech_fams)
        cell["reduced_family_set"] = mech_fams
        cell["runtime_s"] = round(time.time() - t0, 1)
        v2["cells"][key] = cell
        dump()
        log(f"  cell {key} done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing output file")
    ap.add_argument("--phase2", action="store_true", help="strategy families / audit_disp / eps sensitivity")
    ap.add_argument("--only", nargs="*", help="phase2: only these cell keys")
    a = ap.parse_args()
    if a.phase2:
        return main2(a)
    cfg = QUICK if a.quick else FULL
    path = a.out or os.path.join(ROOT, "results", "game_quick.json" if a.quick else "game.json")
    log = lambda s: print(s, flush=True)
    T0 = time.time()
    timing = {}
    out = {"grid": cfg["grid"], "eps": cfg["eps"], "streams": cfg["streams"], "max_rounds": cfg["max_rounds"],
           "starts": cfg["starts"], "mechanisms": list(MECHS), "cells": {}, "tuning": {},
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    def dump():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"game_sweep": out}, f, indent=1)

    if os.path.exists(path) and not a.quick and not a.fresh:       # resume: keep finished cells / tuning
        prev = json.load(open(path)).get("game_sweep", {})
        out["cells"], out["tuning"] = prev.get("cells", {}), prev.get("tuning", {})
        out["resumed_from"] = sorted(out["cells"]) + ["tuning:" + k for k in out["tuning"]]
        timing.update({k: v for k, v in prev.get("timing_s", {}).items() if k != "total"})
        log(f"resuming; have {out['resumed_from']}")
    log("preflight GPU vs CPU"); out["preflight_max_err"] = preflight(log)
    t = time.time()
    pool = ProcessPoolExecutor(1 if a.quick else os.cpu_count(), initializer=_init)
    rseeds = list(range(cfg["voi_seeds"]))
    refs = cpu_refs(rseeds, cfg["cells"][0][2] if a.quick else cfg["voi_steps"], cfg["dp_k"], 0.004,
                    sorted({0, *cfg["streams"]}), pool)
    pool.shutdown()
    timing["cpu_dp_refs_s"] = time.time() - t
    log(f"dp refs done ({timing['cpu_dp_refs_s']:.0f}s)")
    for n, rate, steps, ns in cfg["cells"]:
        if f"N{n}" not in out["cells"]:
            sync(); t = time.time()
            log(f"== game cell N={n} rate={rate} steps={steps} seeds={ns}")
            cref = {p: {s: v[:ns] for s, v in d.items()} for p, d in refs.items()} if n == 3 else None
            out["cells"][f"N{n}"] = run_cell(n, rate, steps, ns, cfg, log, cref)
            sync(); timing[f"game_N{n}_s"] = time.time() - t; dump()
        if f"N{n}" not in out["tuning"]:
            sync(); t = time.time()
            log(f"== tuning N={n}")
            out["tuning"][f"N{n}"] = run_tuning(n, rate, steps, cfg, log, cfg["tune_seeds_by_n"].get(n, cfg["tune_seeds"]))
            sync(); timing[f"tuning_N{n}_s"] = time.time() - t; dump()
    sync(); t = time.time()
    log("== fused VoI")
    out["fused_voi"] = run_voi(cfg, refs, log)
    sync(); timing["fused_voi_gpu_s"] = time.time() - t
    out["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    out["timing_s"]["total"] = round(time.time() - T0, 1)
    dump()
    log(f"written {path} ({time.time() - T0:.0f}s)")
    if not a.no_figs:
        from arbitration import figures
        d = json.load(open(path))
        os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
        tag = "_quick" if a.quick else ""
        if "game_v2" in d:
            figures.fig11(d, os.path.join(ROOT, "figures", f"fig11_strategic_loss_v2{tag}.png"))
            figures.fig12(d, os.path.join(ROOT, "figures", f"fig12_equilibrium_b_v2{tag}.png"))
        else:
            log("no 'game_v2' key in output; skipping fig11/fig12 (replot: python -m arbitration.figures)")


if __name__ == "__main__":
    main()
