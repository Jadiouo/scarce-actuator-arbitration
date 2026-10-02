"""Per-step optimal dispatcher (replanning MDP) vs every policy, VoI, and region maps -- all on the GPU.

    python3 scripts/run_dp_step.py checks            # one-off consistency checks
    python3 scripts/run_dp_step.py gap [--seeds 32]  # N=3 gap table (legacy, shock) x 4 loads
    python3 scripts/run_dp_step.py region            # VoI_step regions (shock_frac x spread, x SERVICE)
    python3 scripts/run_dp_step.py n4                # N=4 trend
    python3 scripts/run_dp_step.py figs              # fig9b / fig10b from the cache
    python3 scripts/run_dp_step.py merge             # write results/dp_step.json from the cache

Stages cache their raw per-seed numbers under ~/.cache/dp_step (resumable); `merge` assembles
results/dp_step.json. See arbitration/gpu/dp_step.py for the model.
"""
import argparse, json, os, sys, time, math, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy import stats
from arbitration import dp as _dp, model as _m
from arbitration.model import make_config
from arbitration.gpu import dp_step as ds, dp_torch as dpt, sim_torch as st

CACHE = os.path.expanduser("~/.cache/dp_step")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "dp_step.json")
os.makedirs(CACHE, exist_ok=True)
STEPS, BURN = 20500, 500
BASE_ARMS = ["none", "none_random", "index_true", "learned_index", "greedy_true", "honest", "honest_index"]
ARMS = BASE_ARMS + ["dp_full", "dp_age"]
SPREAD = 0.6


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def cache_get(key):
    p = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)


def cache_put(key, obj):
    with open(os.path.join(CACHE, key + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return dict(mean=m, lo=m, hi=m, median=float(np.median(x)), n=n)
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return dict(mean=m, lo=m - h, hi=m + h, median=float(np.median(x)), n=n)


# ---------------------------------------------------------------------------- DP pieces
def chunk_size(N, K, cap_gb=5.0):
    return max(1, int(cap_gb * 1e9 // (100 * (K + 1) ** N * 8 * 7)))


def step_dp(insts, cfg, Ks, sim=None, tol=1e-4, device=None):
    """Solve the step MDP for every instance (chunked), K nested with warm starts.
    sim: None | dict(seeds, steps, burn) -> also simulate the K=Ks[-1] policy (same instances).
    Returns dict(g {K: (S,)}, iters {K: list}, span, sim_mh (S,) or None, secs)."""
    N, S = len(insts[0][0]), len(insts)
    cs = chunk_size(N, Ks[-1])
    g = {K: np.zeros(S) for K in Ks}
    its = {K: np.zeros(S, int) for K in Ks}
    spans = {K: np.zeros(S) for K in Ks}
    mh = np.full(S, np.nan)
    rh = np.zeros((S, N))
    t0 = time.time()
    pend = []                                   # (slice, QM, QS, chunk) waiting for a batched simulation

    def flush():
        if not pend:
            return
        QM = torch.cat([p[1] for p in pend])
        QS = torch.cat([p[2] for p in pend])
        idx = np.concatenate([np.arange(S)[p[0]] for p in pend])
        ss = ds.StepSim([dict(seed=sim["seeds"][i], posts=insts[i][0], rates=insts[i][1], pos0=insts[i][2]) for i in idx],
                        [cfg] * len(idx), sim["steps"], device)
        out = ss.run(QM, QS, Ks[-1], burn=sim["burn"])
        mh[idx], rh[idx] = out["mean_health"], out["robot_health"]
        pend.clear()
        del QM, QS, ss
        free()

    for s0 in range(0, S, cs):
        sl = slice(s0, min(S, s0 + cs))
        chunk = insts[sl]
        res = ds.solve_steps([(p, r) for p, r, _ in chunk], cfg, Ks, tol=tol, device=device)
        for K, P in res:
            g[K][sl], its[K][sl], spans[K][sl] = P.g, P.iters, P.span
        if sim is not None:
            K, P = res[-1]
            QM, QS = ds.policy_tables(P)
            P.V = None
            pend.append((sl, QM, QS, chunk))
            if sum(len(p[3]) for p in pend) * 100 * (K + 1) ** N * 8 > 1.2e9:
                flush()
        del res
        free()
    flush()
    return dict(g=g, iters=its, span=spans, sim_mh=mh if sim is not None else None, robot=rh, secs=time.time() - t0)


def semi_dp(insts, cfg, Ks, tol=5e-5, tables_K=None, sim=None, n=3, device=None):
    """dp_torch semi-MDP for every K (batched); optionally GPU-simulate the K=tables_K policy."""
    t0 = time.time()
    g, mh, spans = {}, None, {}
    ins = [(p.astype(float), r) for p, r, _ in insts]
    for K in Ks:
        per = 3 * n * (K + 1) ** n * 8
        cs = max(1, int(3e9 // per))
        sols = []
        for c0 in range(0, len(ins), cs):
            sols += dpt.solve_full(ins[c0:c0 + cs], cfg, K, tol=tol, device=device)
        g[K] = np.array([s.g for s in sols])
        spans[K] = np.array([s.span for s in sols])
        if sim is not None and K == tables_K:
            tabs = torch.stack([ds.semi_tables(p.astype(float), r, x0, cfg, K, s.V, s.g, device)
                                for (p, r, x0), s in zip(insts, sols)]).reshape(len(insts), n + 1, n, -1)
            mh = ds.run_semi_policy("dp_full", sim["seeds"], sim["rates"], sim["spread"], [cfg] * len(insts),
                                    sim["steps"], n, dict(tab=tabs, K=K), sim["burn"], device)
            del tabs
        del sols
        free()
    return dict(g=g, span=spans, sim_mh=mh, secs=time.time() - t0)


def age_dp(insts, cfg, sim=None, n=3, bs=None, device=None):
    bs = bs or (16 if n == 3 else 1)
    t0 = time.time()
    S = len(insts)
    g, binding, mh = np.zeros(S), np.zeros(S, bool), np.zeros(S)
    for s0 in range(0, S, bs):
        sl = slice(s0, min(S, s0 + bs))
        A = ds.AgeBatch([(p.astype(float), r) for p, r, _ in insts[sl]], cfg, device=device)
        g[sl], binding[sl] = A.g, A.cap_binding
        if sim is not None:
            sd = sim["seeds"][sl]
            mh[sl] = ds.run_semi_policy("dp_age", sd, sim["rates"][sl] if hasattr(sim["rates"], "__len__") else sim["rates"],
                                        sim["spread"], [cfg] * len(sd), sim["steps"], n, A, sim["burn"], device)
        del A
        free()
    return dict(g=g, cap_binding=binding, sim_mh=mh if sim is not None else None, secs=time.time() - t0)


def baselines(arms, seeds, rate, spread, cfg, steps, burn, n, device=None):
    out = {}
    for a in arms:
        out[a] = ds.sim_burn(a, seeds, rate, spread, [cfg] * len(seeds), steps, n, burn, device)
    return out


def eval_cell(cfg, n, rate, spread, seeds, Kstep, Ksemi, arms=ARMS, steps=STEPS, burn=BURN, with_sim=True,
              extra_arms=BASE_ARMS):
    """Everything for one (config, rate, spread) over `seeds`: DP values (K nested) and simulated values
    of the step policy, the semi policies and the baselines, all on the same floored instances."""
    seeds = list(seeds)
    insts = [ds.int_instance(s, n, rate, spread) for s in seeds]
    sim = dict(seeds=seeds, rates=rate, spread=spread, steps=steps, burn=burn) if with_sim else None
    out = dict(seeds=seeds, n=n, rate=rate, spread=spread, Kstep=list(Kstep), Ksemi=list(Ksemi))
    with ds.rounded_instances():
        log(f"  step DP K={Kstep} ...")
        r = step_dp(insts, cfg, Kstep, sim)
        out["step"] = r
        log(f"    done {r['secs']:.0f}s  iters {[int(v.mean()) for v in r['iters'].values()]}")
        log(f"  semi DP K={Ksemi} ...")
        r = semi_dp(insts, cfg, Ksemi, tables_K=Ksemi[0], sim=sim, n=n)
        out["semi"] = r
        log(f"    done {r['secs']:.0f}s")
        log("  age DP ...")
        r = age_dp(insts, cfg, sim, n)
        out["age"] = r
        log(f"    done {r['secs']:.0f}s")
        if with_sim:
            t0 = time.time()
            out["base"] = baselines(extra_arms, seeds, rate, spread, cfg, steps, burn, n)
            out["base_secs"] = time.time() - t0
    return out


def summarize_cell(c):
    """per-seed derived quantities of an eval_cell result."""
    Ks, Ke = c["Kstep"], c["Ksemi"]
    jstep = ds.richardson(Ks[-2:], [c["step"]["g"][k] for k in Ks[-2:]]) if len(Ks) > 1 else c["step"]["g"][Ks[-1]]
    jsemi = ds.richardson(Ke[-2:], [c["semi"]["g"][k] for k in Ke[-2:]]) if len(Ke) > 1 else c["semi"]["g"][Ke[-1]]
    return dict(J_step=jstep, J_semi=jsemi, J_age=c["age"]["g"])


# ---------------------------------------------------------------------------- checks
class _Never(st.Policy):
    """never serves (no callers): the h trajectory is pure damage (dynamics check)."""
    def decide(self, S):
        return torch.zeros_like(S.h, dtype=torch.bool), S.h, "max"


def check_dynamics():
    """StepSim and sim_torch must see identical health trajectories (shock streams, clip, accounting)."""
    cfg = make_config("v2")
    seeds = list(range(6))
    st.POLICY_CLASSES["never"] = _Never
    with ds.rounded_instances():
        a = st.simulate("never", seeds, n=3, rate=0.016, spread=0.6, steps=3000, model=cfg, graph=False)
        insts = [ds.int_instance(s, 3, 0.016, 0.6) for s in seeds]
        ss = ds.StepSim([dict(seed=s, posts=p, rates=r, pos0=x0) for s, (p, r, x0) in zip(seeds, insts)], [cfg] * 6, 3000)
        K = 2
        QM = torch.zeros(6, 100, (K + 1) ** 3, dtype=torch.float64, device=ss.dev)
        QS = torch.full((6, 3, (K + 1) ** 3), -1.0, dtype=torch.float64, device=ss.dev)
        b = ss.run(QM, QS, K, burn=0)
    return dict(max_abs_diff_mean_health=float(np.abs(a["mean_health"] - b["mean_health"]).max()),
                max_abs_diff_robot_health=float(np.abs(a["robot_health"] - b["robot_health"]).max()),
                mean_health=a["mean_health"].tolist())


def check_ongrid():
    """Deterministic decay that is a multiple of the grid: the step solver's primitives (restricted semi
    built from one-step moves) must equal dp_torch's J* exactly, the full step J must not be lower, and
    the extracted policy simulated must reproduce J."""
    from arbitration.model import ModelConfig
    rows = []
    for K, svc, posts, m in [(30, 1, [10, 13, 17], [1, 1, 2]), (40, 2, [10, 16, 25], [1, 1, 1]),
                             (24, 1, [5, 9, 14], [1, 2, 1]), (30, 2, [3, 20, 50], [1, 1, 1]),
                             (60, 3, [10, 22, 35], [1, 1, 1]), (60, 2, [10, 18, 30], [1, 1, 1])]:
        cfg = ModelConfig(service=svc, dp_k=K)
        posts = np.array(posts)
        rates = np.array(m) / K
        sol = dpt.solve_full([(posts.astype(float), rates)], cfg, K, tol=1e-10, max_iter=40000)[0]
        P = ds.StepProblem([(posts, rates)], cfg, K)
        r = ds.restricted_semi(P)
        P.solve(tol=1e-9, max_iter=40000)
        QM, QS = ds.policy_tables(P)
        ss = ds.StepSim([dict(seed=0, posts=posts, rates=rates, pos0=12)], [cfg], 20000)
        out = ss.run(QM, QS, K, burn=500)
        rows.append(dict(K=K, service=svc, posts=posts.tolist(), steps_per_unit=m, J_semi_dp_torch=float(sol.g),
                         J_restricted_step_primitives=float(r[0]), J_step_full=float(P.g[0]),
                         J_sim_step_policy=float(out["mean_health"][0]),
                         diff_restricted_vs_semi=float(r[0] - sol.g), diff_step_vs_semi=float(P.g[0] - sol.g)))
        log("  on-grid", {k: (round(v, 10) if isinstance(v, float) else v) for k, v in rows[-1].items()})
        del P, QM, QS
        free()
    return rows


def check_K120():
    """K convergence of the step solver on two instances: K = 15..120 chain; compare extrapolations."""
    out = {}
    for model in ("v2", "legacy"):
        cfg = make_config(model)
        posts, rates, pos0 = ds.int_instance(0, 3, 0.004, SPREAD)
        Ks = (15, 20, 30, 40, 60, 120)
        t0 = time.time()
        res = ds.solve_steps([(posts, rates)], cfg, Ks)
        g = {K: float(P.g[0]) for K, P in res}
        its = {K: int(P.iters) for K, P in res}
        sol = {K: float(dpt.solve_full([(posts.astype(float), rates)], cfg, K, tol=5e-5)[0].g) for K in (30, 60, 120)}
        ext = lambda ks: float(ds.richardson(ks, [g[k] for k in ks])[()] if False else ds.richardson(ks, [g[k] for k in ks]))
        out[model] = dict(J_step=g, iters=its, J_semi=sol, secs=time.time() - t0,
                          ext_30_60=ext((30, 60)), ext_60_120=ext((60, 120)), ext_30_60_120=ext((30, 60, 120)),
                          ext_20_40=ext((20, 40)), ext_15_30=ext((15, 30)), ext_20_30=ext((20, 30)),
                          semi_ext_60_120=float(ds.richardson((60, 120), [sol[60], sol[120]])))
        log("  K-chain", model, {k: round(v, 6) if isinstance(v, float) else v for k, v in out[model].items() if k.startswith("ext") or k == "secs"})
        del res
        free()
    return out


def check_age():
    """(1) AgeBatch vs the CPU AgeDP; (2) age-only step MDP vs age-only semi-MDP (small age cap, exact):
    does replanning / idling add value when the only information is ages? Also the full-information step
    J on the same instance (clean VoI = J_step_full - J_step_age)."""
    cfg = make_config("v2")
    res = {}
    ins = [_dp.instance(s, 3, 0.004, SPREAD)[:2] for s in range(4)]
    A = ds.AgeBatch(ins, cfg)
    cpu = [float(_dp.AgeDP(p, r, cfg).g) for p, r in ins]
    res["gpu_vs_cpu_AgeDP"] = dict(gpu=A.g.tolist(), cpu=cpu, max_abs_diff=float(np.abs(A.g - np.array(cpu)).max()))
    del A
    rows = []
    for posts, rates in [([20, 32, 47], [0.03, 0.04, 0.035]), ([15, 30, 55], [0.035, 0.03, 0.04]),
                         ([10, 25, 60], [0.04, 0.03, 0.035])]:
        posts, rates = np.array(posts), np.array(rates)
        base = (1 - cfg.shock_frac) * rates
        Acap = int(np.ceil(1.0 / base.min())) + 2
        t0 = time.time()
        ga = ds.AgeBatch([(posts.astype(float), rates)], cfg, cap=Acap)
        gs, its, sp = ds.step_age(posts, rates, cfg, Acap)
        res_full = ds.solve_steps([(posts, rates)], cfg, (20, 40))
        jf = [float(P.g[0]) for _, P in res_full]
        semi = [float(dpt.solve_full([(posts.astype(float), rates)], cfg, K, tol=5e-5)[0].g) for K in (60, 120)]
        rows.append(dict(posts=posts.tolist(), rates=rates.tolist(), age_cap=Acap, J_age_semi=float(ga.g[0]),
                         J_age_step=float(gs), age_step_minus_semi=float(gs - ga.g[0]),
                         J_full_step_K20_40=jf, J_full_step_ext=float(ds.richardson((20, 40), jf)),
                         J_full_semi_K60_120=semi, J_full_semi_ext=float(ds.richardson((60, 120), semi)),
                         VoI_semi=float(ds.richardson((60, 120), semi) - ga.g[0]),
                         VoI_step_clean=float(ds.richardson((20, 40), jf) - gs),
                         VoI_step_vs_semi_age=float(ds.richardson((20, 40), jf) - ga.g[0]),
                         secs=time.time() - t0))
        log("  age-step", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()})
        del res_full
        free()
    res["age_step_vs_semi"] = rows
    return res


# ---------------------------------------------------------------------------- stages
def stage_smoke(a):
    cfg = make_config("v2")
    c = eval_cell(cfg, 3, 0.004, SPREAD, range(4), (30, 60), (60, 120))
    d = summarize_cell(c)
    log({k: v.round(5) for k, v in d.items()})
    log("sim step", c["step"]["sim_mh"].round(5), "semi", c["semi"]["sim_mh"].round(5), "age", c["age"]["sim_mh"].round(5))
    for k, v in c["base"].items():
        log(k, v.round(5))


def gap_summary(c):
    """Per-seed and summary statistics of one gap cell (everything vs the SIMULATED per-step policy)."""
    d = summarize_cell(c)
    ref = c["step"]["sim_mh"]
    sims = dict(c["base"])
    sims["dp_full"] = c["semi"]["sim_mh"]
    sims["dp_age"] = c["age"]["sim_mh"]
    out = dict(seeds=c["seeds"], rate=c["rate"], spread=c["spread"], n=c["n"], Kstep=c["Kstep"], Ksemi=c["Ksemi"],
               J_step_K={str(k): v.tolist() for k, v in c["step"]["g"].items()},
               J_semi_K={str(k): v.tolist() for k, v in c["semi"]["g"].items()},
               J_step_ext=d["J_step"].tolist(), J_semi_ext=d["J_semi"].tolist(), J_age=d["J_age"].tolist(),
               cap_binding_frac=float(c["age"]["cap_binding"].mean()),
               step_iters={str(k): v.tolist() for k, v in c["step"]["iters"].items()},
               step_span_max={str(k): float(v.max()) for k, v in c["step"]["span"].items()},
               semi_span_max={str(k): float(v.max()) for k, v in c["semi"]["span"].items()},
               sim=dict(dp_step_full=ref.tolist(), **{k: v.tolist() for k, v in sims.items()}),
               gap={}, gap_vs_J_step_ext={}, secs=dict(step=c["step"]["secs"], semi=c["semi"]["secs"], age=c["age"]["secs"],
                                                  base=c.get("base_secs", 0.0)))
    for k, v in sims.items():
        gp = ref - v
        out["gap"][k] = dict(per_seed=gp.tolist(), frac_arm_better=float((gp < 0).mean()), **ci95(gp))
        out["gap_vs_J_step_ext"][k] = ci95(d["J_step"] - v)
    out["VoI_step"] = dict(per_seed=(d["J_step"] - d["J_age"]).tolist(), **ci95(d["J_step"] - d["J_age"]))
    out["VoI_semi_ext"] = dict(per_seed=(d["J_semi"] - d["J_age"]).tolist(), **ci95(d["J_semi"] - d["J_age"]))
    k60 = c["semi"]["g"][c["Ksemi"][0]]
    out["VoI_semi_K%d" % c["Ksemi"][0]] = dict(per_seed=(k60 - d["J_age"]).tolist(), **ci95(k60 - d["J_age"]))
    out["VoI_sim"] = ci95(ref - c["age"]["sim_mh"])
    out["step_minus_semi_ext"] = dict(per_seed=(d["J_step"] - d["J_semi"]).tolist(), **ci95(d["J_step"] - d["J_semi"]))
    out["sim_minus_dp"] = dict(
        dp_step_full=dict(per_seed=(ref - d["J_step"]).tolist(), **ci95(ref - d["J_step"])),
        dp_step_full_vs_K=dict(per_seed=(ref - c["step"]["g"][c["Kstep"][-1]]).tolist(), **ci95(ref - c["step"]["g"][c["Kstep"][-1]])),
        dp_full=dict(per_seed=(sims["dp_full"] - d["J_semi"]).tolist(), **ci95(sims["dp_full"] - d["J_semi"])),
        dp_age=dict(per_seed=(sims["dp_age"] - d["J_age"]).tolist(), **ci95(sims["dp_age"] - d["J_age"])))
    return out


GAP_RATES = (0.002, 0.004, 0.008, 0.016)


def stage_gap(a):
    for model in ("v2", "legacy"):
        for rate in GAP_RATES:
            key = f"gap_{model}_{rate}_{a.seeds}"
            if cache_get(key + "_summary") is not None:
                continue
            log(f"=== gap cell {model} rate={rate} seeds={a.seeds}")
            c = cache_get(key)
            if c is None:
                c = eval_cell(make_config(model), 3, rate, SPREAD, range(a.seeds), (30, 60), (60, 120))
                cache_put(key, c)
            s = gap_summary(c)
            cache_put(key + "_summary", s)
            log("  VoI_step %.4f  VoI_semi %.4f  step-semi %.4f  gap none median %.4f mean %.4f" % (
                s["VoI_step"]["mean"], s["VoI_semi_ext"]["mean"], s["step_minus_semi_ext"]["mean"],
                s["gap"]["none"]["median"], s["gap"]["none"]["mean"]))


REGION_FRACS = (0.0, 0.25, 0.5, 0.75, 0.9)
REGION_SPREADS = (0.0, 0.3, 0.6, 1.0, 1.5)
REGION_SERVICES = (1, 5, 20)
REGION_RATE = 0.004


def region_cells():
    cells = []
    for f in REGION_FRACS:
        for sp in REGION_SPREADS:
            cells.append((f, sp, 5))
        for sv in REGION_SERVICES:
            if (f, 0.6, sv) not in cells:
                cells.append((f, 0.6, sv))
    return cells


def region_cell(f, sp, sv, seeds, rate=REGION_RATE, Kstep=(20, 30), Ksemi=(60, 120)):
    cfg = make_config("v2", shock_frac=f, service=sv)
    seeds = list(seeds)
    insts = [ds.int_instance(s, 3, rate, sp) for s in seeds]
    with ds.rounded_instances():
        r1 = step_dp(insts, cfg, Kstep, None, tol=2e-4)
        r2 = semi_dp(insts, cfg, Ksemi, n=3)
        r3 = age_dp(insts, cfg, None, 3)
        none = ds.sim_burn("none", seeds, rate, sp, [cfg] * len(seeds), STEPS, 3, BURN)
    jstep = ds.richardson(Kstep, [r1["g"][k] for k in Kstep])
    jsemi = ds.richardson(Ksemi, [r2["g"][k] for k in Ksemi])
    ja = r3["g"]
    return dict(shock_frac=f, spread=sp, service=sv, rate=rate, seeds=seeds, Kstep=list(Kstep), Ksemi=list(Ksemi),
                J_step_K={str(k): r1["g"][k].tolist() for k in Kstep}, J_semi_K={str(k): r2["g"][k].tolist() for k in Ksemi},
                J_step_ext=jstep.tolist(), J_semi_ext=jsemi.tolist(), J_age=ja.tolist(), sim_none=none.tolist(),
                VoI_step=ci95(jstep - ja), VoI_semi_ext=ci95(jsemi - ja),
                VoI_semi_K=ci95(r2["g"][Ksemi[0]] - ja), step_minus_semi=ci95(jstep - jsemi),
                gap_none_vs_J_step=ci95(jstep - none), gap_none_vs_J_semi=ci95(jsemi - none),
                cap_binding_frac=float(r3["cap_binding"].mean()),
                step_iters_mean={str(k): float(v.mean()) for k, v in r1["iters"].items()},
                secs=dict(step=r1["secs"], semi=r2["secs"], age=r3["secs"]))


def stage_region(a):
    seeds = range(a.seeds if a.seeds != 32 else 16)
    for (f, sp, sv) in region_cells():
        key = f"region_{f}_{sp}_{sv}_{len(list(seeds))}"
        if cache_get(key) is not None:
            continue
        log(f"=== region cell shock_frac={f} spread={sp} service={sv}")
        c = region_cell(f, sp, sv, seeds)
        cache_put(key, c)
        log("  VoI_step %.4f VoI_semi %.4f gap_none(J_step) %.4f  %.0fs" % (
            c["VoI_step"]["mean"], c["VoI_semi_ext"]["mean"], c["gap_none_vs_J_step"]["mean"], sum(c["secs"].values())))


N4_RATES = (0.004, 0.008)


def stage_n4(a):
    seeds = range(16)
    for rate in N4_RATES:
        key = f"n4_{rate}"
        if cache_get(key + "_summary") is not None:
            continue
        log(f"=== N=4 rate={rate}")
        cfg = make_config("v2")
        c = cache_get(key)
        if c is None:
            c = eval_cell(cfg, 4, rate, SPREAD, seeds, (15, 20, 30), (30, 60), with_sim=True)
            cache_put(key, c)
        s = gap_summary(c)
        # three-point alternative for the step value
        ks = c["Kstep"]
        s["J_step_ext3"] = ds.richardson(ks, [c["step"]["g"][k] for k in ks]).tolist()
        s["J_step_ext_15_20"] = ds.richardson(ks[:2], [c["step"]["g"][k] for k in ks[:2]]).tolist()
        cache_put(key + "_summary", s)
        log("  VoI_step %.4f  VoI_semi %.4f  gap none median %.4f mean %.4f" % (
            s["VoI_step"]["mean"], s["VoI_semi_ext"]["mean"], s["gap"]["none"]["median"], s["gap"]["none"]["mean"]))


def check_restricted_nongrid():
    """Real (off-grid) instances: the per-step grid projection differs from the one-shot projection of the
    semi-MDP, so the 'restricted' step solver (same policy class as dp_torch) differs from dp_torch's J* at
    discretisation level; this measures that diffusion bias directly (same policy class, same K)."""
    rows = []
    for model in ("v2", "legacy"):
        cfg = make_config(model)
        for seed in (0, 1):
            posts, rates, _ = ds.int_instance(seed, 3, 0.004, SPREAD)
            for K in (15, 20, 30):
                sol = dpt.solve_full([(posts.astype(float), rates)], cfg, K, tol=1e-6, max_iter=20000)[0]
                P = ds.StepProblem([(posts, rates)], cfg, K)
                r = ds.restricted_semi(P, tol=1e-8)
                P.solve(tol=1e-6)
                rows.append(dict(model=model, seed=seed, K=K, J_semi_dp_torch=float(sol.g),
                                 J_restricted_step=float(r[0]), J_step_full=float(P.g[0]),
                                 restricted_minus_semi=float(r[0] - sol.g)))
                log("  restricted", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()})
                del P
                free()
    return rows


# ---------------------------------------------------------------------------- figures
POL_ORDER = ["none", "none_random", "index_true", "learned_index", "greedy_true", "honest", "honest_index",
             "dp_full", "dp_age"]
POL_LAB = {"none": "none (round-robin)", "none_random": "none_random", "index_true": "index_true",
           "learned_index": "learned_index", "greedy_true": "greedy_true", "honest": "honest",
           "honest_index": "honest_index", "dp_full": "dp_full (semi-MDP)", "dp_age": "dp_age (semi-MDP)"}


def load_gap():
    out = {}
    for model in ("legacy", "v2"):
        for rate in GAP_RATES:
            s = cache_get(f"gap_{model}_{rate}_32_summary")
            if s is not None:
                out[(model, rate)] = s
    return out


def stage_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from arbitration.figures import DPCOL, _style, _heat
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gp = load_gap()
    if gp:
        fig, axs = plt.subplots(2, 4, figsize=(20, 9), sharex=False)
        for r_i, model in enumerate(("legacy", "v2")):
            ymax = max([np.max(gp[(model, r)]["gap"][p]["per_seed"]) for r in GAP_RATES if (model, r) in gp for p in POL_ORDER] + [0.01])
            ymin = min([np.min(gp[(model, r)]["gap"][p]["per_seed"]) for r in GAP_RATES if (model, r) in gp for p in POL_ORDER] + [0.0])
            for c_i, rate in enumerate(GAP_RATES):
                ax = axs[r_i, c_i]
                if (model, rate) not in gp:
                    ax.axis("off")
                    continue
                s = gp[(model, rate)]
                data = [np.array(s["gap"][p]["per_seed"]) for p in POL_ORDER]
                bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                                medianprops=dict(color="k", lw=2))
                rng = np.random.default_rng(0)
                for i, (p, d) in enumerate(zip(POL_ORDER, data)):
                    bp["boxes"][i].set(facecolor=DPCOL.get(p, "#999"), alpha=0.35)
                    ax.scatter(d, np.full(len(d), i + 1) + rng.uniform(-0.18, 0.18, len(d)), s=7, color=DPCOL.get(p, "#999"), alpha=0.8, zorder=3)
                    ax.scatter([d.mean()], [i + 1], marker="D", s=30, color="white", edgecolor="k", zorder=4)
                ax.axvline(0, color="k", lw=1)
                ax.set_yticks(range(1, len(POL_ORDER) + 1))
                ax.set_yticklabels([POL_LAB[p] for p in POL_ORDER] if c_i == 0 else [], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlim(ymin - 0.01, ymax + 0.01)
                ax.set_title(f"{'legacy' if model == 'legacy' else 'shock (v2)'}, r = {rate}", fontsize=10)
                if r_i == 1:
                    ax.set_xlabel("gap = mean health of dp_step_full - policy (per seed)")
                _style(ax)
        fig.suptitle("Gap to the per-step optimal policy (simulated, same seed / same shocks), N = 3, 32 seeds; "
                     "box = quartiles, black = median, diamond = mean", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(root, "figures", "fig9b_gap_distribution.png"), dpi=150)
        plt.close(fig)
        log("wrote fig9b")
    reg = {}
    for (f, sp, sv) in region_cells():
        c = cache_get(f"region_{f}_{sp}_{sv}_16")
        if c is not None:
            reg[(f, sp, sv)] = apply_region_cal(c, kappas()[0])
    if reg:
        fr, sps, svs = REGION_FRACS, REGION_SPREADS, REGION_SERVICES

        def mat(keyf, ks, which, a_is_spread):
            M = np.full((len(fr), len(ks)), np.nan)
            S = np.zeros((len(fr), len(ks)), bool)
            for i, f in enumerate(fr):
                for j, k in enumerate(ks):
                    c = reg.get((f, k, 5) if a_is_spread else (f, 0.6, k))
                    if c is not None:
                        d = c[which]
                        M[i, j] = d["mean"]
                        S[i, j] = d["lo"] > 0 or d["hi"] < 0
            return M, S
        fk = "VoI_step_final" if all("VoI_step_final" in c for c in reg.values()) else "VoI_step_cal"
        gk = "gap_none_final" if fk == "VoI_step_final" else "gap_none_vs_J_step_cal"
        rows = [(fk, "VoI_step = J_step(full) - J(age)  (J_semi,ext + same-K Delta, extrapolated)", "magma"),
                ("VoI_semi_ext", "VoI_semi = J_semi(full) - J(age)  (K-extrapolated)", "magma"),
                (gk, "gap of blind round-robin: J_step(full) - none", "viridis")]
        fig, axs = plt.subplots(len(rows), 2, figsize=(13, 4.4 * len(rows)))
        for r_i, (key, title, cm) in enumerate(rows):
            M, S = mat(key, sps, key, True)
            _heat(axs[r_i, 0], M, "rate heterogeneity (spread), SERVICE = 5", "shock_frac", [str(x) for x in sps],
                  [str(x) for x in fr], title, cm, stars=S)
            M, S = mat(key, svs, key, False)
            _heat(axs[r_i, 1], M, "SERVICE steps, spread = 0.6", "shock_frac", [str(x) for x in svs],
                  [str(x) for x in fr], title, cm, stars=S)
        fig.suptitle("Per-step exact benchmark, N = 3, r = 0.004, shock model (K-extrapolated; * = 95% CI excludes 0)", y=0.998)
        fig.tight_layout()
        fig.savefig(os.path.join(root, "figures", "fig10b_voi_step_regions.png"), dpi=150)
        plt.close(fig)
        log("wrote fig10b")


def kappas():
    """Extrapolation coefficients calibrated on the K = 15..60 chains of check_Kcal (2 seeds per shock_frac):
    J_inf = J(K2) + kappa (J(K2) - J(K1)). Pure 1/K error gives kappa = K1/(K2-K1) (2 for 20,30; 1 for 30,60),
    pure 1/K^2 gives K1^2/(K2^2-K1^2) (0.8; 0.33). The reference is J_age (exact) when shock_frac = 0 (damage
    deterministic: age determines health) and the 3-point (30,40,60) fit otherwise.
    Returns (kap2030 {f: kappa}, kap3060 {f: kappa}) for the f's of the chain."""
    cal = cache_get("check_Kcal")
    if cal is None:
        return None, None
    k2030, k3060 = {}, {}
    for f in sorted(set(r["shock_frac"] for r in cal)):
        rr = [r for r in cal if r["shock_frac"] == f]
        truth = [(r["J_age"] if f == 0 else r["ext"]["30_40_60"]) for r in rr]
        k2030[f] = float(np.mean([(t - r["J_step"][30]) / (r["J_step"][30] - r["J_step"][20]) for t, r in zip(truth, rr)]))
        k3060[f] = float(np.mean([(t - r["J_step"][60]) / (r["J_step"][60] - r["J_step"][30]) for t, r in zip(truth, rr)]))
    return k2030, k3060


def interp_kappa(tab, f):
    xs = sorted(tab)
    return float(np.interp(f, xs, [tab[x] for x in xs]))


def apply_region_cal(c, k2030):
    if k2030 is None:
        return c
    kap = interp_kappa(k2030, c["shock_frac"])
    j20, j30 = np.array(c["J_step_K"]["20"]), np.array(c["J_step_K"]["30"])
    jc = j30 + kap * (j30 - j20)
    ja, js, sn = np.array(c["J_age"]), np.array(c["J_semi_ext"]), np.array(c["sim_none"])
    c["kappa_20_30_used"] = kap
    c["J_step_cal"] = jc.tolist()
    c["VoI_step_cal"] = ci95(jc - ja)
    c["step_minus_semi_cal"] = ci95(jc - js)
    c["gap_none_vs_J_step_cal"] = ci95(jc - sn)
    cr = cache_get(f"consist_region_{c['shock_frac']}_{c['spread']}_{c['service']}")
    if cr is not None:
        d1, d2 = np.array(cr["delta"]["20"]), np.array(cr["delta"]["30"])
        dext = d2 + kap * (d2 - d1)
        c["Delta_K20"], c["Delta_K30"] = ci95(d1), ci95(d2)
        c["min_Delta_K"] = dict(K20=float(d1.min()), K30=float(d2.min()))
        c["Delta_ext"] = ci95(dext)
        vf = (js - ja) + dext
        c["VoI_step_final"] = ci95(vf)
        c["gap_none_final"] = ci95((js + dext) - sn)
    return c


def semi_at(posts_rates_cfg, Ks, tol=2e-5):
    """dp_torch semi J* at the given K on the floored instances (same instances as the step DP)."""
    seeds, n, rate, spread, cfg = posts_rates_cfg
    insts = [ds.int_instance(s, n, rate, spread) for s in seeds]
    ins = [(p.astype(float), r) for p, r, _ in insts]
    out = {}
    for K in Ks:
        out[K] = np.array([s.g for s in dpt.solve_full(ins, cfg, K, tol=tol)])
        free()
    return out


def same_k_table(c, cfg, Kstep):
    """Same-K, same (floored) instance comparison J_step(K) - J_semi(K) for the K of the step DP."""
    semi = semi_at((c["seeds"], c["n"], c["rate"], c["spread"], cfg), Kstep)
    return {K: np.asarray(c["step"]["g"][K] if "step" in c else c["J_step_K"][str(K)]) - semi[K] for K in Kstep}, semi


def stage_consistency(a):
    """(1) J_step(K) >= J_semi(K) at the SAME K on the SAME floored instance (both tightly converged);
    (2) lists every extrapolated step - semi < 0 and the same-K cause; (3) VoI on the same footing:
    VoI_samek(K) = (J_step(K) - J_semi(K)) + (J_semi(K) - J_age) is not comparable across K, so the
    final VoI_step is also reported as VoI_semi_ext + Delta_ext with Delta = J_step - J_semi extrapolated."""
    rows = []
    for model in ("v2", "legacy"):
        for rate in GAP_RATES:
            c = cache_get(f"gap_{model}_{rate}_32")
            if c is None:
                continue
            cfg = make_config(model)
            key = f"consist_gap_{model}_{rate}"
            r = cache_get(key)
            if r is None:
                d, semi = same_k_table(c, cfg, c["Kstep"])
                r = dict(kind="gap", model=model, rate=rate, Kstep=c["Kstep"],
                         delta={str(k): v.tolist() for k, v in d.items()},
                         J_semi_sameK={str(k): v.tolist() for k, v in semi.items()},
                         J_step_K={str(k): np.asarray(c["step"]["g"][k]).tolist() for k in c["Kstep"]},
                         J_age=np.asarray(c["age"]["g"]).tolist(), J_semi_ext=summarize_cell(c)["J_semi"].tolist(),
                         J_step_ext=summarize_cell(c)["J_step"].tolist())
                cache_put(key, r)
            rows.append(r)
    for (f, sp, sv) in region_cells():
        c = cache_get(f"region_{f}_{sp}_{sv}_16")
        if c is None:
            continue
        key = f"consist_region_{f}_{sp}_{sv}"
        r = cache_get(key)
        if r is None:
            cfg = make_config("v2", shock_frac=f, service=sv)
            c2 = dict(seeds=c["seeds"], n=3, rate=c["rate"], spread=sp, J_step_K=c["J_step_K"])
            d, semi = same_k_table(c2, cfg, c["Kstep"])
            r = dict(kind="region", shock_frac=f, spread=sp, service=sv, Kstep=c["Kstep"],
                     delta={str(k): v.tolist() for k, v in d.items()},
                     J_semi_sameK={str(k): v.tolist() for k, v in semi.items()},
                     J_step_K=c["J_step_K"], J_age=c["J_age"], J_semi_ext=c["J_semi_ext"],
                     J_step_ext=c["J_step_ext"])
            cache_put(key, r)
        rows.append(r)
    for r in rows:
        tag = (f"gap {r['model']} r={r['rate']}" if r["kind"] == "gap" else f"region f={r['shock_frac']} sp={r['spread']} sv={r['service']}")
        mins = {k: min(v) for k, v in r["delta"].items()}
        ext = np.array(r["J_step_ext"]) - np.array(r["J_semi_ext"])
        log(f"{tag}: min same-K (J_step-J_semi) per K {mins} | extrapolated step-semi mean {ext.mean():+.4f} neg {(ext < 0).sum()}/{len(ext)}")
    cache_put("consistency_rows", rows)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def stage_merge(a):
    out = dict(meta=dict(
        description="Per-step (replanning) exact MDP J_step vs the semi-MDP J* of arbitration.dp, VoI and optimality gaps; N=3 unless noted.",
        instance="posts and start position floored to the integer ring grid in DP and in every simulated policy (RING=100, SPEED=1)",
        sim=dict(steps=STEPS, burn=BURN, effective_steps=STEPS - BURN, paired="same seed, same shock stream (CRN)"),
        spread=SPREAD, gap_sign="gap = mean_health(dp_step_full simulated) - mean_health(policy) per seed",
        extrapolation="J(K) = J_inf + c/K from the two largest K (step: 30,60 gap/N4: 15,20,30 / region 20,40; semi: 60,120)",
        files=["arbitration/gpu/dp_step.py", "scripts/run_dp_step.py"]))
    checks = {}
    for n in ("dynamics_crn", "ongrid_exact", "age", "K_convergence", "restricted_nongrid", "Kcal", "longrun"):
        c = cache_get("check_" + n)
        if c is not None:
            checks[n] = c
    out["checks"] = checks
    k2030, k3060 = kappas()
    out["meta"]["kappa_calibration"] = dict(kappa_20_30=k2030, kappa_30_60=k3060,
        note="J_inf = J(K2) + kappa (J(K2)-J(K1)); calibrated on check_Kcal chains (see checks.Kcal); legacy uses kappa(f=0), shock cells kappa(f=0.5)")
    gp = load_gap()
    for (m, r), sm in gp.items():
        if k3060 is None:
            break
        kap = interp_kappa(k3060, 0.0 if m == "legacy" else 0.5)
        j60, j30 = np.array(sm["J_step_K"]["60"]), np.array(sm["J_step_K"]["30"])
        jc = j60 + kap * (j60 - j30)
        ja, js = np.array(sm["J_age"]), np.array(sm["J_semi_ext"])
        sm["kappa_30_60_used"] = kap
        cr = cache_get(f"consist_gap_{m}_{r}")
        if cr is not None:
            d1, d2 = np.array(cr["delta"]["30"]), np.array(cr["delta"]["60"])
            dext = d2 + kap * (d2 - d1)
            sm["same_K"] = dict(Delta_K30=ci95(d1), Delta_K60=ci95(d2), min_Delta_K30=float(d1.min()), min_Delta_K60=float(d2.min()),
                                n_negative_K30=int((d1 < -1e-5).sum()), n_negative_K60=int((d2 < -1e-5).sum()),
                                J_semi_K30=cr["J_semi_sameK"]["30"], J_semi_K60=cr["J_semi_sameK"]["60"])
            sm["Delta_ext"] = dict(per_seed=dext.tolist(), **ci95(dext))
            vf = np.array(sm["VoI_semi_ext"]["per_seed"]) + dext
            sm["VoI_step_final"] = dict(per_seed=vf.tolist(), **ci95(vf))
        sm["J_step_cal"] = jc.tolist()
        sm["VoI_step_cal"] = dict(per_seed=(jc - ja).tolist(), **ci95(jc - ja))
        sm["step_minus_semi_cal"] = dict(per_seed=(jc - js).tolist(), **ci95(jc - js))
    out["gap_table"] = {m: {str(r): gp[(m, r)] for r in GAP_RATES if (m, r) in gp} for m in ("legacy", "v2")}
    reg = []
    for (f, sp, sv) in region_cells():
        c = cache_get(f"region_{f}_{sp}_{sv}_16")
        if c is not None:
            apply_region_cal(c, k2030)
            reg.append(c)
    out["region"] = dict(rate=REGION_RATE, shock_fracs=list(REGION_FRACS), spreads=list(REGION_SPREADS),
                         services=list(REGION_SERVICES), cells=reg)
    n4 = {}
    for r in N4_RATES:
        c = cache_get(f"n4_{r}_summary")
        if c is not None:
            n4[str(r)] = c
    out["n4"] = n4
    secs = 0.0
    for m in out["gap_table"].values():
        for c in m.values():
            secs += sum(c["secs"].values())
    for c in reg:
        secs += sum(c["secs"].values())
    for c in n4.values():
        secs += sum(c["secs"].values())
    out["meta"]["gpu_seconds_gap_region_n4"] = secs
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(_jsonable(out), f)
    log("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.2f} MB")


def check_longrun():
    """Noise-free-ish DP vs simulation check: the K=60 step policy simulated for 200000 steps on 8 seeds
    (v2, r = 0.004; sampling sd of the mean ~ 0.0006) against J_step(K=60) and the K-extrapolated J_step."""
    cfg = make_config("v2")
    seeds = list(range(8))
    insts = [ds.int_instance(s, 3, 0.004, SPREAD) for s in seeds]
    steps, burn = 200500, 500
    with ds.rounded_instances():
        r = step_dp(insts, cfg, (30, 60), dict(seeds=seeds, steps=steps, burn=burn), tol=2e-5)
    jext = ds.richardson((30, 60), [r["g"][30], r["g"][60]])
    mh = r["sim_mh"]
    out = dict(seeds=seeds, steps=steps, J60=r["g"][60].tolist(), J_ext=jext.tolist(), sim=mh.tolist(),
               sim_minus_J60=ci95(mh - r["g"][60]), sim_minus_ext=ci95(mh - jext))
    log("  longrun sim-J60 %.4f  sim-ext %.4f [%.4f, %.4f]" % (out["sim_minus_J60"]["mean"], out["sim_minus_ext"]["mean"],
                                                         out["sim_minus_ext"]["lo"], out["sim_minus_ext"]["hi"]))
    return out


def check_Kcal():
    """Extrapolation calibration across the region map: K-chain 15..60 on 2 instances for several shock_frac
    (spread 0.6, SERVICE 5, r = 0.004); the (40, 60) / (30, 60) extrapolations serve as reference for the
    cheaper pairs used elsewhere."""
    out = []
    for f in (0.0, 0.25, 0.5, 0.9):
        cfg = make_config("v2", shock_frac=f)
        for seed in (0, 1):
            posts, rates, _ = ds.int_instance(seed, 3, REGION_RATE, SPREAD)
            Ks = (15, 20, 30, 40, 60)
            res = ds.solve_steps([(posts, rates)], cfg, Ks, tol=5e-5)
            g = {K: float(P.g[0]) for K, P in res}
            ext = {f"{a}_{b}": float(ds.richardson((a, b), [g[a], g[b]])) for a, b in ((15, 30), (20, 30), (20, 40), (30, 40), (30, 60), (40, 60))}
            ext["15_20_30"] = float(ds.richardson((15, 20, 30), [g[k] for k in (15, 20, 30)]))
            ext["20_30_40"] = float(ds.richardson((20, 30, 40), [g[k] for k in (20, 30, 40)]))
            ext["30_40_60"] = float(ds.richardson((30, 40, 60), [g[k] for k in (30, 40, 60)]))
            sol = {K: float(dpt.solve_full([(posts.astype(float), rates)], cfg, K, tol=5e-5)[0].g) for K in (60, 120)}
            ja = float(ds.AgeBatch([(posts.astype(float), rates)], cfg, device=None).g[0])
            out.append(dict(shock_frac=f, seed=seed, J_step=g, ext=ext, J_semi=sol, J_semi_ext=float(ds.richardson((60, 120), [sol[60], sol[120]])), J_age=ja))
            log("  Kcal", f, seed, {k: round(v, 5) for k, v in ext.items()}, "semi", round(out[-1]["J_semi_ext"], 5), "age", round(ja, 5))
            del res
            free()
    return out


def stage_checks(a):
    out = {}
    names = [x for x in (a.only or "dynamics_crn,ongrid_exact,age,K_convergence,restricted_nongrid").split(",")]
    fns = dict(dynamics_crn=check_dynamics, ongrid_exact=check_ongrid, age=check_age, K_convergence=check_K120,
               restricted_nongrid=check_restricted_nongrid, Kcal=check_Kcal, longrun=check_longrun)
    for name, fn in [(n, fns[n]) for n in names]:
        c = cache_get("check_" + name)
        if c is None:
            log("check", name)
            c = fn()
            cache_put("check_" + name, c)
        out[name] = c
    log(json.dumps(out, default=float)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    globals()["stage_" + a.stage](a)
