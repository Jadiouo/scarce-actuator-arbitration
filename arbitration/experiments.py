"""Sweeps behind the figures. `python -m arbitration.experiments` runs all.

Every cell keeps its per-seed results; summaries are mean and 95% t-interval.
Policies are compared on the SAME seeds (same posts, rates, start position and
noise stream), so differences are paired and reported with their own CI.

    python -m arbitration.experiments           # full run -> results/results.json
    python -m arbitration.experiments --quick   # tiny version -> results/results_quick.json
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from .model import run, POLICIES, make_config

N, SPREAD = 8, 0.6
METRICS = ("mean_health", "min_health", "frac_below_02", "frac_any_dead",
           "precision", "n_services", "retargets_per_service", "frac_dead_robot_steps")
ARMS = ["none", "none_random", "greedy_true", "nearest", "honest", "strategic", "priced"]
RHO_ARMS = ("none", "greedy_true", "nearest", "honest", "strategic", "priced")
LOADS = (0.001, 0.004)            # below capacity / overloaded
HOLDOUT_OFFSET = 10_000           # fresh seeds for out-of-sample evaluation

FULL = dict(seeds=32, lam_seeds=64, steps=1500,
            rates=[0.0005, 0.001, 0.002, 0.004, 0.007, 0.012, 0.020, 0.040],
            lams=[0.0, 0.002, 0.004, 0.006, 0.010, 0.016, 0.025, 0.040,
                  0.065, 0.10, 0.16, 0.25, 0.40],
            sigmas=[0.0, 0.05, 0.15, 0.30, 0.50],
            hysts=[0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 1.0, 1.5, 2.0], boot=1000,
            dp_seeds=16, dp_seeds_region=12, dp_steps=8000, dp_k=60,
            dp_rates=[0.002, 0.004, 0.008, 0.016], dp_region_rates=[0.004, 0.01],
            dp_shock_fracs=[0.0, 0.25, 0.5, 0.75, 0.9], dp_spreads=[0.0, 0.3, 0.6, 1.0, 1.5],
            dp_services=[1, 5, 20])
QUICK = dict(seeds=3, lam_seeds=4, steps=300,
             rates=[0.001, 0.004, 0.02],
             lams=[0.0, 0.01, 0.04, 0.16],
             sigmas=[0.0, 0.15, 0.5],
             hysts=[0.0, 0.2, 0.6, 2.0], boot=50,
             dp_seeds=2, dp_seeds_region=2, dp_steps=300, dp_k=10,
             dp_rates=[0.004, 0.016], dp_region_rates=[0.004],
             dp_shock_fracs=[0.0, 0.9], dp_spreads=[0.0, 1.0], dp_services=[1, 5],
             game_cells=[(3, 0.004, 1500, 4), (8, 0.002, 600, 3)],
             game_grid=[0.0, 0.2, 0.4, 0.6], game_streams=[1],
             game_max_rounds=6, game_sigmas=[0.05, 0.5], voi_seeds=3)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_MAP = map   # replaced by a process pool in main() for the full run


# --------------------------------------------------------------------- runner
def _init_worker():
    """One BLAS thread per worker (the DP's small tensordots otherwise oversubscribe)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _one(args):
    policy, kw, seed, steps = args
    r = run(policy, n=N, steps=steps, seed=seed, spread=SPREAD, **dict(kw))
    return [float(getattr(r, k)) for k in METRICS]


def collect(cells, seeds, steps):
    """cells: {key: (policy, kwargs)}. Returns {key: {metric: np.array(per seed)}}."""
    jobs, index = [], []
    for key, (policy, kw) in cells.items():
        for s in seeds:
            jobs.append((policy, tuple(sorted(kw.items())), s, steps))
            index.append(key)
    res = list(_MAP(_one, jobs, chunksize=8)) if _MAP is not map else list(map(_one, jobs))
    out = {k: [] for k in cells}
    for key, row in zip(index, res):
        out[key].append(row)
    return {k: {m: np.array([r[i] for r in rows]) for i, m in enumerate(METRICS)}
            for k, rows in out.items()}


# ----------------------------------------------------------------- statistics
def ci95(x):
    """Mean and 95% t-interval of a per-seed vector (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    m = float(x.mean())
    if n < 2:
        return dict(mean=m, lo=m, hi=m, n=n)
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return dict(mean=m, lo=m - h, hi=m + h, n=n)


def paired(a, b):
    """Paired difference a - b over the same seeds (per-seed arrays), with 95% CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    out = ci95(d)
    out["significant"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def summarize(cell):
    """Per-seed arrays -> {metric: {mean, lo, hi, n, per_seed}}."""
    return {m: dict(ci95(v), per_seed=[round(float(x), 6) for x in v])
            for m, v in cell.items()}


def paired_block(cells, pairs, metrics=("mean_health", "precision")):
    return {f"{a}-{b}": {m: paired(cells[a][m], cells[b][m]) for m in metrics}
            for a, b in pairs}


# --------------------------------------------------------------------- sweeps
def load_sweep(cfg):
    """Figures 1 and 2: every policy across two decades of degradation rate."""
    seeds = range(cfg["seeds"])
    cells = {(a, r): (a, dict(rate=r, preempt=True)) for a in ARMS for r in cfg["rates"]}
    raw = collect(cells, seeds, cfg["steps"])
    sweep = {a: {str(r): summarize(raw[(a, r)]) for r in cfg["rates"]} for a in ARMS}
    pairs = [("greedy_true", "none"), ("nearest", "none"), ("honest", "none"),
             ("priced", "honest"), ("none", "none_random"), ("greedy_true", "honest"),
             ("strategic", "honest")]
    pr = {str(r): paired_block({a: raw[(a, r)] for a in ARMS}, pairs) for r in cfg["rates"]}
    return sweep, pr, raw


def precision_outcome_rho(raw, cfg, rng_seed=0):
    """Spearman rho between dispatch precision and team health across policies, per
    load, with a bootstrap-over-seeds 95% interval."""
    rng = np.random.default_rng(rng_seed)
    ns = cfg["seeds"]
    out = []
    for r in cfg["rates"]:
        P = np.array([raw[(a, r)]["precision"] for a in RHO_ARMS])      # arms x seeds
        U = np.array([raw[(a, r)]["mean_health"] for a in RHO_ARMS])

        def rho(idx):
            p = np.nanmean(P[:, idx], axis=1)
            u = U[:, idx].mean(axis=1)
            return float(spearmanr(p, u)[0])
        boots = [rho(rng.integers(0, ns, ns)) for _ in range(min(cfg["boot"], 400))]
        boots = [b for b in boots if not np.isnan(b)]
        out.append({"rate": r,
                    "load": float(raw[("none", r)]["frac_any_dead"].mean()),
                    "rho": rho(np.arange(ns)),
                    "rho_lo": float(np.percentile(boots, 2.5)) if boots else float("nan"),
                    "rho_hi": float(np.percentile(boots, 97.5)) if boots else float("nan")})
    return out


def lambda_sweep(cfg, rng_seed=1):
    """Figure 3 / bracketing: request price in both regimes, with lambda* and CI.

    lambda* = argmax over the grid of mean team health, with a bootstrap-over-seeds
    95% interval on the argmax. 'interior' means the optimum is neither end of the
    grid (so it is bracketed). The gain at lambda* over honest (lam=0, same
    threshold) is a paired difference, computed in-sample and on fresh holdout
    seeds, because choosing the argmax on the same seeds is optimistic.
    """
    lams, rng = cfg["lams"], np.random.default_rng(rng_seed)
    out = {}
    for r in LOADS:
        cells = {l: ("priced", dict(rate=r, lam=l, preempt=True)) for l in lams}
        raw = collect(cells, range(cfg["lam_seeds"]), cfg["steps"])
        M = np.array([raw[l]["mean_health"] for l in lams])              # lams x seeds
        ns = M.shape[1]
        k = int(np.argmax(M.mean(axis=1)))
        am = [int(np.argmax(M[:, rng.integers(0, ns, ns)].mean(axis=1)))
              for _ in range(cfg["boot"])]
        lam_boot = np.array([lams[i] for i in am])
        # holdout: fresh seeds, lambda* vs lam=0 (honest)
        hold = collect({"star": ("priced", dict(rate=r, lam=lams[k], preempt=True)),
                        "honest": ("honest", dict(rate=r, preempt=True))},
                       range(HOLDOUT_OFFSET, HOLDOUT_OFFSET + cfg["lam_seeds"]), cfg["steps"])
        h0 = raw[lams[0]]["mean_health"]
        out[str(r)] = {
            "lams": lams,
            "curve": {str(l): summarize(raw[l]) for l in lams},
            "lam_star": lams[k],
            "lam_star_ci": [float(np.percentile(lam_boot, 2.5)),
                            float(np.percentile(lam_boot, 97.5))],
            "lam_star_boot_hist": {str(l): int((lam_boot == l).sum()) for l in lams},
            "interior": bool(0 < k < len(lams) - 1),
            "bracketed_in_ci": bool(lam_boot.min() > lams[0] and lam_boot.max() < lams[-1]),
            "gain_vs_honest_insample": paired(raw[lams[k]]["mean_health"], h0),
            "gain_vs_honest_insample_rel": float(
                (raw[lams[k]]["mean_health"] / h0).mean() - 1) if (h0 > 0).all() else None,
            "gain_vs_honest_holdout": paired(hold["star"]["mean_health"],
                                             hold["honest"]["mean_health"]),
            "holdout_rel": float((hold["star"]["mean_health"].mean()
                                  / hold["honest"]["mean_health"].mean()) - 1),
            "precision_honest": float(raw[lams[0]]["precision"].mean()),
            "precision_star": float(raw[lams[k]]["precision"].mean()),
        }
    return out


def sigma_sweep(cfg, lam_star):
    """Noise sweep. greedy_true and none do not read the noisy claims, so they are
    flat in sigma by construction (included as references)."""
    arms = ["honest", "priced", "greedy_true", "nearest", "none"]
    out = {}
    for r in LOADS:
        cells = {(a, s): (a, dict(rate=r, sigma=s, preempt=True,
                                  **({"lam": lam_star[str(r)]} if a == "priced" else {})))
                 for a in arms for s in cfg["sigmas"]}
        raw = collect(cells, range(cfg["seeds"]), cfg["steps"])
        out[str(r)] = {
            "lam_priced": lam_star[str(r)],
            "cells": {a: {str(s): summarize(raw[(a, s)]) for s in cfg["sigmas"]} for a in arms},
            "paired": {str(s): paired_block({a: raw[(a, s)] for a in arms},
                                            [("priced", "honest"), ("honest", "none"),
                                             ("nearest", "honest"), ("greedy_true", "honest")])
                       for s in cfg["sigmas"]}}
    return out


def hyst_sweep(cfg):
    """Hysteresis on re-targeting for honest with preempt, plus non-preempt reference."""
    out = {}
    for r in LOADS:
        cells = {h: ("honest", dict(rate=r, hyst=h, preempt=True)) for h in cfg["hysts"]}
        cells["nopreempt"] = ("honest", dict(rate=r, preempt=False))
        cells["none"] = ("none", dict(rate=r, preempt=True))
        raw = collect(cells, range(cfg["seeds"]), cfg["steps"])
        out[str(r)] = {
            "hysts": cfg["hysts"],
            "cells": {str(h): summarize(raw[h]) for h in cfg["hysts"]},
            "reference": {"nopreempt": summarize(raw["nopreempt"]),
                          "none": summarize(raw["none"])},
            "paired_vs_hyst0": {str(h): {m: paired(raw[h][m], raw[0.0][m])
                                         for m in ("mean_health", "retargets_per_service")}
                                for h in cfg["hysts"][1:]},
            "paired_vs_nopreempt": {str(h): paired(raw[h]["mean_health"],
                                                   raw["nopreempt"]["mean_health"])
                                    for h in cfg["hysts"]}}
    return out


# ------------------------------------------------------------------- v2 sweep
V2_HYST = 0.45        # hysteresis used for the "no-thrash" honest/priced baselines
REACTIVE = ("greedy_true", "index_true", "honest_index", "nearest", "honest", "strategic", "priced",
            "learned", "learned_index")
STATIC = ("none", "none_random", "tour_sqrt", "tour_sqrt_learned",
          "rollout", "rollout_claim", "rollout_true")   # preempt is irrelevant
V2_ARMS = {}          # arm key -> (policy, kwargs)
for _p in REACTIVE:
    V2_ARMS[f"{_p}|pre"] = (_p, dict(preempt=True))
    V2_ARMS[f"{_p}|nopre"] = (_p, dict(preempt=False))
for _p in ("honest", "priced"):
    V2_ARMS[f"{_p}|hyst"] = (_p, dict(preempt=True, hyst=V2_HYST))
for _p in STATIC:
    V2_ARMS[_p] = (_p, dict(preempt=False))
V2_PAIRS = [("honest|nopre", "learned|nopre"), ("honest|nopre", "learned_index|nopre"),
            ("honest|hyst", "learned_index|nopre"),
            ("honest_index|nopre", "learned_index|nopre"),
            ("honest_index|nopre", "index_true|nopre"),
            ("honest_index|nopre", "none"),
            ("priced|pre", "honest|nopre"), ("priced|pre", "honest|hyst"),
            ("priced|nopre", "honest|nopre"), ("priced|hyst", "honest|hyst"),
            ("honest|pre", "honest|nopre"),
            ("learned|nopre", "greedy_true|nopre"), ("tour_sqrt", "greedy_true|nopre"),
            ("rollout_true", "greedy_true|nopre"), ("learned_index|nopre", "greedy_true|nopre"),
            ("index_true|nopre", "greedy_true|nopre"),
            ("rollout_true", "none"), ("rollout", "none"), ("learned_index|nopre", "none")]


def v2_sweep(cfg):
    """Every policy across loads under the legacy model and under the v2 model
    (shock damage, AR(1) noise, keep_commitment, precision at dispatch). Reactive
    policies run with preemption on and off (and honest/priced also with hysteresis,
    the no-thrash baselines); static ones once. Paired CIs vs `none` and a few
    targeted pairs (reports vs learned beliefs; priced vs no-thrash honest)."""
    seeds = range(cfg["seeds"])
    out = {"arms": list(V2_ARMS), "rates": cfg["rates"], "hyst": V2_HYST, "configs": {}}
    for name in ("legacy", "v2"):
        cells = {(a, r): (pol, dict(rate=r, model=name, **kw))
                 for a, (pol, kw) in V2_ARMS.items() for r in cfg["rates"]}
        raw = collect(cells, seeds, cfg["steps"])
        out["configs"][name] = {
            "cells": {a: {str(r): summarize(raw[(a, r)]) for r in cfg["rates"]}
                      for a in V2_ARMS},
            "paired_vs_none": {str(r): {a: {m: paired(raw[(a, r)][m], raw[("none", r)][m])
                                            for m in ("mean_health", "precision")}
                                        for a in V2_ARMS if a != "none"}
                               for r in cfg["rates"]},
            "pairs": {str(r): {f"{a}-{b}": {m: paired(raw[(a, r)][m], raw[(b, r)][m])
                                            for m in ("mean_health", "precision")}
                               for a, b in V2_PAIRS}
                      for r in cfg["rates"]},
            "load": {str(r): float(raw[("none", r)]["frac_any_dead"].mean())
                     for r in cfg["rates"]}}
    return out



# ------------------------------------------------------------- exact benchmark
DP_N = 3
DP_ARMS = ("none", "none_random", "index_true", "learned_index", "rollout_true",
           "honest", "honest_index", "greedy_true", "dp_full", "dp_age")
DP_REGION_ARMS = ("none", "dp_full", "dp_age")


def _dp_job(args):
    """One instance: solve both DPs, simulate the listed policies on the same seed.
    Policies run without preemption (commit until arrival), the fair baseline."""
    from . import dp
    model, opts, rate, spread, seed, steps, arms, K = args
    cfg = make_config(model, dp_k=K, **dict(opts))
    posts, rates, _ = dp.instance(seed, DP_N, rate, spread)
    jf = dp.build_policy("dp_full", posts, rates, cfg, K)
    ja = dp.build_policy("dp_age", posts, rates, cfg)
    mh = {a: float(run(a, n=DP_N, rate=rate, spread=spread, seed=seed, steps=steps,
                       model=cfg, preempt=False).mean_health) for a in arms}
    return dict(J_full=float(jf.g), J_age=float(ja.g), cap_binding=bool(ja.cap_binding),
                iters=(jf.iters, ja.iters), spans=(float(jf.span), float(ja.span)), mh=mh)


def _dp_cell(model, opts, rate, spread, seeds, steps, arms, K):
    print(f"  dp cell {model} {dict(opts)} rate={rate} spread={spread}", flush=True)
    jobs = [(model, tuple(sorted(opts.items())), rate, spread, s, steps, arms, K) for s in seeds]
    return list(_MAP(_dp_job, jobs, chunksize=1)) if _MAP is not map else list(map(_dp_job, jobs))


def _dp_summary(rows, arms):
    jf = np.array([r["J_full"] for r in rows])
    ja = np.array([r["J_age"] for r in rows])
    mh = {a: np.array([r["mh"][a] for r in rows]) for a in arms}
    out = {"J_full": ci95(jf), "J_age": ci95(ja), "VoI_state": ci95(jf - ja),
           "VoI_state_rel": ci95((jf - ja) / jf),
           "cap_binding_frac": float(np.mean([r["cap_binding"] for r in rows])),
           "max_span": float(max(max(r["spans"]) for r in rows)),
           "per_seed": {"J_full": jf.round(6).tolist(), "J_age": ja.round(6).tolist()},
           "mean_health": {a: ci95(v) for a, v in mh.items()},
           "gap_vs_J_full": {a: paired(jf, v) for a, v in mh.items()},
           "gap_vs_J_age": {a: paired(ja, v) for a, v in mh.items()}}
    if "dp_full" in mh:                      # same-horizon gap: no start-up transient
        out["gap_vs_sim_dp_full"] = {a: paired(mh["dp_full"], v) for a, v in mh.items()}
        out["pred_minus_sim"] = {"dp_full": paired(jf, mh["dp_full"]),
                                 "dp_age": paired(ja, mh["dp_age"])}
    return out


def dp_sweep(cfg):
    """Exact optimum (N=3) vs every policy, and where information has value.

    main:    legacy and shock (v2) models x loads, all arms, J*_full / J*_age.
    region:  VoI_state and the gap of `none`, at medium loads, vs shock_frac x
             spread (heatmap) and vs shock_frac x SERVICE.
    """
    K, steps = cfg["dp_k"], cfg["dp_steps"]
    seeds = list(range(cfg["dp_seeds"]))
    rseeds = list(range(cfg["dp_seeds_region"]))
    out = {"N": DP_N, "K": K, "steps": steps, "spread": SPREAD, "arms": list(DP_ARMS),
           "main": {}, "region": {}}
    for model in ("legacy", "v2"):
        out["main"][model] = {
            str(r): _dp_summary(_dp_cell(model, {}, r, SPREAD, seeds, steps, DP_ARMS, K),
                                DP_ARMS)
            for r in cfg["dp_rates"]}
    reg = {"rates": cfg["dp_region_rates"], "shock_fracs": cfg["dp_shock_fracs"],
           "spreads": cfg["dp_spreads"], "services": cfg["dp_services"],
           "heat": {}, "service": {}}
    for r in cfg["dp_region_rates"]:
        reg["heat"][str(r)] = {
            f"{f}|{sp}": _dp_summary(_dp_cell("v2", dict(shock_frac=f), r, sp, rseeds, steps,
                                              DP_REGION_ARMS, K), DP_REGION_ARMS)
            for f in cfg["dp_shock_fracs"] for sp in cfg["dp_spreads"]}
        reg["service"][str(r)] = {
            f"{f}|{sv}": _dp_summary(_dp_cell("v2", dict(shock_frac=f, service=sv), r, SPREAD,
                                              rseeds, steps, DP_REGION_ARMS, K), DP_REGION_ARMS)
            for f in cfg["dp_shock_fracs"] for sv in cfg["dp_services"]}
    out["region"] = reg
    return out


# ------------------------------------------------------- reporting game (phase B)
GAME_MECHS = {                      # mechanism key -> (policy, ModelConfig overrides)
    "honest_index": ("honest_index", {}),
    "fused_index": ("fused_index", {}),
    "learned_index": ("learned_index", {}),
    "audit_index": ("audit_index", {}),
    "audit_penalty_index|ignore": ("audit_penalty_index", {"penalty_mode": "ignore"}),
    "audit_penalty_index|demote": ("audit_penalty_index", {"penalty_mode": "demote"}),
}
VOI_RATE = 0.004


def _voi_job(args):
    """One N=3 instance (v2 model, dp_sweep setting): simulate every arm on the same seed."""
    seed, steps, K, arms, rate = args
    from . import dp
    cfg = make_config("v2", dp_k=K)
    posts, rates, _ = dp.instance(seed, DP_N, rate, SPREAD)
    jf = dp.build_policy("dp_full", posts, rates, cfg, K)
    ja = dp.build_policy("dp_age", posts, rates, cfg)
    mh = {k: float(run(pol, n=DP_N, rate=rate, spread=SPREAD, seed=seed, steps=steps, model=cfg,
                       preempt=False, **kw).mean_health) for k, (pol, kw) in arms.items()}
    return dict(J_full=float(jf.g), J_age=float(ja.g), mh=mh)


def _ratio_ci(num, den, boot=1000, seed=0):
    """mean(num)/mean(den) with a bootstrap-over-seeds 95% interval."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng, n = np.random.default_rng(seed), len(num)
    pt = float(num.mean() / den.mean())
    bs = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        if den[i].mean() != 0:
            bs.append(num[i].mean() / den[i].mean())
    return dict(mean=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def fused_voi_sweep(cfg):
    """Truthful noisy claims: how much of VoI = H(dp_full) - H(dp_age) does Bayesian fusion
    recover? N=3, v2 model, rate 0.004, spread 0.6, no preemption (the dp_sweep setting).
    Gains are paired differences vs learned_index; VoI is measured in the SAME simulation
    (dp_full - dp_age at the same horizon and noise) so the start-up transient cancels."""
    arms = {"learned_index": ("learned_index", {}), "fused": ("fused", {}),
            "fused_index": ("fused_index", {}),
            "fused_index_known_rates": ("fused_index", {"fuse_known_rates": True}),
            "audit_index": ("audit_index", {}), "honest_index": ("honest_index", {}),
            "index_true": ("index_true", {}), "dp_full": ("dp_full", {}), "dp_age": ("dp_age", {})}
    for s in cfg["game_sigmas"]:
        arms[f"fused_index@sigma={s}"] = ("fused_index", {"sigma": s})
    seeds, steps, K = list(range(cfg["voi_seeds"])), cfg["dp_steps"], cfg["dp_k"]
    jobs = [(s, steps, K, arms, VOI_RATE) for s in seeds]
    rows = list(_MAP(_voi_job, jobs, chunksize=1)) if _MAP is not map else list(map(_voi_job, jobs))
    mh = {k: np.array([r["mh"][k] for r in rows]) for k in arms}
    jf = np.array([r["J_full"] for r in rows])
    ja = np.array([r["J_age"] for r in rows])
    voi_sim = mh["dp_full"] - mh["dp_age"]
    out = {"N": DP_N, "rate": VOI_RATE, "steps": steps, "K": K, "seeds": len(seeds),
           "J_full": ci95(jf), "J_age": ci95(ja), "VoI_J": ci95(jf - ja),
           "VoI_sim": ci95(voi_sim), "mean_health": {k: ci95(v) for k, v in mh.items()},
           "vs_learned_index": {k: paired(v, mh["learned_index"]) for k, v in mh.items()},
           "recovered_frac": {k: _ratio_ci(mh[k] - mh["learned_index"], voi_sim)
                              for k in arms if k.startswith(("fused", "audit", "index_true"))},
           "position": {k: _ratio_ci(mh[k] - mh["dp_age"], voi_sim) for k in arms},
           "per_seed": {k: v.round(6).tolist() for k, v in mh.items()}}
    return out


def game_sweep(cfg):
    """Strategic reporting: equilibrium b, team health and strategic loss for dispatch rules
    (a) honest_index (b) fused_index (c) learned_index (d) audit_index (e) audit+penalty
    (ignore / demote), N=3 and N=8. Plus the VoI-recovery block (fused_voi_sweep).
    Best-response iteration: see arbitration/game.py. Uses the CPU evaluator; swap in a
    batched evaluator by passing a different callable to game.solve_game."""
    from . import game
    ev = game.CpuEvaluator(_MAP)
    grid, streams = tuple(cfg["game_grid"]), tuple(cfg["game_streams"])
    out = {"grid": list(grid), "eps": game.EPS, "streams": list(streams),
           "max_rounds": cfg["game_max_rounds"], "mechanisms": list(GAME_MECHS), "cells": {}}
    for n, rate, steps, ns in cfg["game_cells"]:
        seeds = list(range(ns))
        cell = {"N": n, "rate": rate, "steps": steps, "seeds": ns, "mech": {}}
        for key, (pol, opts) in GAME_MECHS.items():
            env = dict(n=n, rate=rate, spread=SPREAD, steps=steps, sigma=0.15, model="v2",
                       opts=opts)
            print(f"  game N={n} {key}", flush=True)
            g = game.solve_game(ev, pol, env, seeds, streams, grid, game.EPS,
                                cfg["game_max_rounds"])
            br, sy = g["br"], g["sym"]
            cell["mech"][key] = {
                "truthful_in": ci95(g["truthful_in"]), "eq_in": ci95(g["eq_in"]),
                "truthful_out": ci95(g["truthful_out"]), "eq_out": ci95(g["eq_out"]),
                "loss_in": paired(g["truthful_in"], g["eq_in"]),
                "loss_out": paired(g["truthful_out"], g["eq_out"]),
                "b_eq": ci95(br["b"].mean(axis=1)), "b_eq_max": float(br["b"].max()),
                "frac_converged": float(br["converged"].mean()),
                "frac_cycle": float(br["cycle"].mean()),
                "rounds_mean": float(br["rounds"].mean()),
                "max_gain": float(br["max_gain"].max()),
                "traj_mean_b": [float(x.mean()) for x in br["traj"]],
                "per_seed": {"b": br["b"].round(3).tolist(),
                             "truthful_out": g["truthful_out"].round(5).tolist(),
                             "eq_out": g["eq_out"].round(5).tolist()},
                "sym": {"b": sy["b"], "converged": sy["converged"], "cycle": sy["cycle"],
                        "rounds": sy["rounds"], "gain": sy["gain"], "path": sy["path"],
                        "team_in": ci95(sy["team_in"]), "team_out": ci95(sy["team_out"]),
                        "loss_out": paired(g["truthful_out"], sy["team_out"])}}
        if n == DP_N:                            # reference: optimal dispatchers (same streams)
            cfg2 = make_config("v2", dp_k=cfg["dp_k"])
            refs = {}
            for pol in ("dp_full", "dp_age"):
                refs[pol] = np.array([[run(pol, n=n, rate=rate, spread=SPREAD, seed=s, steps=steps,
                                           model=cfg2, preempt=False, stream=st).mean_health
                                       for s in seeds] for st in streams]).mean(0)
            cell["ref"] = {k: ci95(v) for k, v in refs.items()}
            cell["ref_per_seed"] = {k: v.round(5).tolist() for k, v in refs.items()}
            voi = refs["dp_full"] - refs["dp_age"]
            for key, m in cell["mech"].items():
                for which in ("truthful_out", "eq_out"):
                    v = np.array(m["per_seed"][which])
                    m["pos_" + which] = _ratio_ci(v - refs["dp_age"], voi)
        out["cells"][f"N{n}"] = cell
    out["fused_voi"] = fused_voi_sweep(cfg)
    out["evaluations"] = ev.calls
    return out


# ----------------------------------------------------------------------- main
def main(argv=None):
    global _MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny version for CI/tests")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only-v2", action="store_true",
                    help="run only v2_sweep and merge it into an existing results json")
    ap.add_argument("--only-game", action="store_true",
                    help="run only game_sweep (strategic reporting) and merge it into the results json")
    ap.add_argument("--only-dp", action="store_true",
                    help="run only dp_sweep (exact benchmark) and merge it into the results json")
    a = ap.parse_args(argv)
    cfg = QUICK if a.quick else FULL
    path = a.out or os.path.join(OUT, "results_quick.json" if a.quick else "results.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    workers = a.workers if a.workers is not None else (1 if a.quick else os.cpu_count() or 1)
    pool = ProcessPoolExecutor(workers, initializer=_init_worker) if workers > 1 else None
    if pool:
        _MAP = pool.map
    t0 = time.time()
    if a.only_v2 or a.only_dp or a.only_game:
        key, fn = (("v2_sweep", v2_sweep) if a.only_v2 else
                   ("game_sweep", game_sweep) if a.only_game else ("dp_sweep", dp_sweep))
        try:
            sub = fn(cfg)
        finally:
            _MAP = map
            if pool:
                pool.shutdown()
        res = json.load(open(path)) if os.path.exists(path) else {}
        res[key] = sub
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{key} merged into {path}  ({time.time() - t0:.0f} s)")
        return res
    try:
        sweep, pr, raw = load_sweep(cfg)
        rho = precision_outcome_rho(raw, cfg)
        lam = lambda_sweep(cfg)
        lam_star = {r: lam[r]["lam_star"] for r in lam}
        sig = sigma_sweep(cfg, lam_star)
        hy = hyst_sweep(cfg)
        v2 = v2_sweep(cfg)
        dps = dp_sweep(cfg)
    finally:
        _MAP = map
        if pool:
            pool.shutdown()
    res = {"load_sweep": sweep, "paired": pr, "rho": rho, "lambda_sweep": lam,
           "sigma_sweep": sig, "hyst_sweep": hy, "v2_sweep": v2, "dp_sweep": dps,
           "config": {**cfg, "n": N, "spread": SPREAD, "loads": list(LOADS),
                      "quick": a.quick, "ci": "95% t-interval over seeds",
                      "runtime_s": round(time.time() - t0, 1)}}
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    print(f"{'rate':>8} {'load':>6} {'rho':>7}  95% CI")
    for row in rho:
        print(f"{row['rate']:8.4f} {row['load']:6.2f} {row['rho']:+7.3f}  "
              f"[{row['rho_lo']:+.2f}, {row['rho_hi']:+.2f}]")
    for r, d in lam.items():
        print(f"\nlambda* at rate {r}: {d['lam_star']} (boot 95% {d['lam_star_ci']}), "
              f"interior={d['interior']}, holdout gain {d['holdout_rel']*100:+.1f}%")
    print(f"\nwritten to {path}  ({time.time() - t0:.0f} s)")
    return res


if __name__ == "__main__":
    main()
