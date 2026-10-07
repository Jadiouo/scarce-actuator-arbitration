"""Direction 2 W1 measurements (spec s11): M-1 step time / throughput, M-2 CRN variance reduction.

Always run through gpujob (one GPU).  Parts (each < 25 min):
  python scripts/run_d2_meas.py m1                 -> results/direction2/measure_m1.json
  python scripts/run_d2_meas.py m2 --r 0.5         -> results/direction2/measure_m2_r0.5.json   (same for 0.9)
  python scripts/run_d2_meas.py merge              -> results/direction2_meas.json  (CPU only; also applies s5.5 rules)
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arbitration.rl import env, policy  # noqa: E402

OUT = os.path.join(ROOT, "results", "direction2")
CFG = env.m3c_config(0.5)
MDE_EST = {0.5: 0.0058, 0.9: 0.0016}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rand_mlp(rng, sigma=0.3, F=25):
    return policy.MLPPolicy.from_blocks(sigma * rng.normal(size=(F, 2)), sigma * rng.normal(size=2),
                                        sigma * rng.normal(size=2), sigma * rng.normal(size=1))


def make_bank_state(pop, nS, dev, rng, kind="mlp"):
    """State with B = pop*nS elements, element = [pol, seed] (training layout)."""
    pols = [rand_mlp(rng) for _ in range(pop)] if kind == "mlp" else [policy.AdapterPolicy({}) for _ in range(pop)]
    pid = np.repeat(np.arange(pop), nS)
    sidx = np.tile(np.arange(nS), pop)
    st = env.SimState("M3C", CFG, 0.5, list(range(nS)), sidx, dev, "public", track_obs=(kind == "mlp"))
    bank = env._Bank(pols, torch.as_tensor(pid, device=dev), dev, 25)
    return st, bank


def time_steps(st, bank, warm, n):
    for _ in range(warm):
        env._step(st, bank)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        env._step(st, bank)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def frontier_ms_per_step(dev):
    from arbitration.gpu import frontier
    seeds = list(range(32))
    el = frontier.build([dict(CFG)], [dict(rho=0.5)], [{}], seeds, dev=dev)
    res = {}
    for T in (200, 1200):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        frontier.simulate("M3C", T, el, seeds)
        torch.cuda.synchronize(); res[T] = time.perf_counter() - t0
    return (res[1200] - res[200]) / 1000 * 1e3


def part_m1(args):
    dev = "cuda"
    rng = np.random.default_rng(0)
    # B_max: double B until peak mem >= 12 GB or OOM (20 steps per probe)
    bmax, B = 0, 1024
    while B <= 2 ** 21:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            st, bank = make_bank_state(B // 64, 64, dev, rng)
            time_steps(st, bank, 0, 20)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            del st, bank
        except torch.cuda.OutOfMemoryError:
            break
        if peak > 12 * 1024:
            break
        bmax = B
        B *= 2
    fms = frontier_ms_per_step(dev)
    rows = []
    for pop in (32, 64, 128, 256):
        for nS in (16, 32, 64, 128):
            B = pop * nS
            if B > bmax:
                continue
            meds, p95s, mem = [], [], 0.0
            for _ in range(5):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                st, bank = make_bank_state(pop, nS, dev, rng)
                ts = time_steps(st, bank, 200, 1000)
                meds.append(statistics.median(ts)); p95s.append(float(np.percentile(ts, 95)))
                mem = max(mem, torch.cuda.max_memory_allocated() / 2 ** 20)
                del st, bank
            med = statistics.median(meds)
            rows.append(dict(mech="M3C", r=0.5, pop=pop, n_S=nS, B=B, policy="mlp25-2-1", t_step_ms_median=med,
                             t_step_ms_p95=statistics.median(p95s), steps_timed=1000, warmup=200, repeats=5,
                             mem_peak_MB=mem, elem_steps_per_s=B / (med / 1e3), frontier_ms_per_step=fms, device=torch.cuda.get_device_name(),
                             dtype="float64", torch=torch.__version__, git_sha=git_sha()))
            print(rows[-1]["pop"], rows[-1]["n_S"], round(med, 3), "ms", flush=True)
    # policy-free reference (adapter honest, no obs tracking)
    st, bank = make_bank_state(32, 32, dev, rng, kind="adapter")
    ts = time_steps(st, bank, 200, 1000)
    rows.append(dict(mech="M3C", r=0.5, pop=32, n_S=32, B=1024, policy="adapter", t_step_ms_median=statistics.median(ts),
                     t_step_ms_p95=float(np.percentile(ts, 95)), steps_timed=1000, warmup=200, repeats=1,
                     mem_peak_MB=torch.cuda.max_memory_allocated() / 2 ** 20, elem_steps_per_s=1024 / (statistics.median(ts) / 1e3),
                     frontier_ms_per_step=fms, device=torch.cuda.get_device_name(), dtype="float64", torch=torch.__version__,
                     git_sha=git_sha()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(B_max=bmax, rows=rows), open(os.path.join(OUT, "measure_m1.json"), "w"), indent=1)


HAND_PAIRS = [({"dz": 0.03}, {"dz": 0.05}), ({"dz": 0.1}, {"dz": 0.3}),
              (dict(ue_mode=1, ue_band=0.05), dict(ue_mode=1, ue_band=0.1))]


def part_m2(args):
    r = args.r
    rng = np.random.default_rng(1)
    seeds = list(range(64))
    pols = []
    for _ in range(20):
        pols += [rand_mlp(rng), rand_mlp(rng)]
    for a, b in HAND_PAIRS:
        pols += [policy.AdapterPolicy(a), policy.AdapterPolicy(b)]
    rows = []
    for T in (20000, 40000, 100000):
        t0 = time.time()
        out = env.simulate("M3C", T, CFG, pols, seeds, r=r, dev="cuda")
        f = (out["snap"]["util_b"][:, :, 1] / out["T_score"]).cpu().numpy().astype(np.float64)   # [P,S]
        for k in range(23):
            fa, fb = f[2 * k], f[2 * k + 1]
            vc, vi = float(np.var(fa - fb, ddof=1)), float(np.var(fa, ddof=1) + np.var(fb, ddof=1))
            rows.append(dict(r=r, T=T, pair_kind="rand" if k < 20 else "hand", pair_id=k, var_crn=vc, var_indep=vi,
                             vrf=vi / vc, sd_crn=math.sqrt(vc), n_seeds=64))
        print("T", T, "done", round(time.time() - t0), "s", flush=True)
        json.dump(dict(rows=rows), open(os.path.join(OUT, f"measure_m2_r{r}.json"), "w"), indent=1)


def interp_tstep(rows, B, pop=None, nS=None):
    exact = [x["t_step_ms_median"] for x in rows if x["policy"] == "mlp25-2-1" and x["pop"] == pop and x["n_S"] == nS]
    if exact:      # direct measurement of the (pop, n_S) cell beats interpolation through neighbouring (noisy) cells
        return exact[0]
    pts = sorted((x["B"], x["t_step_ms_median"]) for x in rows if x["policy"] == "mlp25-2-1")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(B, xs, ys))


def part_merge(args):
    m1 = json.load(open(os.path.join(OUT, "measure_m1.json")))
    rows2 = []
    for r in (0.5, 0.9):
        rows2 += json.load(open(os.path.join(OUT, f"measure_m2_r{r}.json")))["rows"]
    summ = []
    for r in (0.5, 0.9):
        for T in (20000, 40000, 100000):
            sel = [x for x in rows2 if x["r"] == r and x["T"] == T]
            v = [x["vrf"] for x in sel]
            vr = [x["vrf"] for x in sel if x["pair_kind"] == "rand"]
            vh = [x["vrf"] for x in sel if x["pair_kind"] == "hand"]
            summ.append(dict(r=r, T=T, vrf_median=statistics.median(v), vrf_min=min(v), vrf_median_rand=statistics.median(vr),
                             vrf_median_hand=statistics.median(vh), sd_crn_median=statistics.median(x["sd_crn"] for x in sel),
                             sd_crn_median_rand=statistics.median(x["sd_crn"] for x in sel if x["pair_kind"] == "rand"),
                             sd_crn_median_hand=statistics.median(x["sd_crn"] for x in sel if x["pair_kind"] == "hand")))
    # s5.5 rules (recommendation only).  n_S rule uses the median sd_crn over all pairs (random + hand).
    W_run = {"S1": 2.5 * 3600, "main": 3.0 * 3600}
    sd = {(s["r"], s["T"]): s["sd_crn_median"] for s in summ}
    nS_opts, rec = (16, 32, 64, 128), {}
    for name, W in W_run.items():
        choice = None
        trace = []
        for lam in (256, 128, 64, 32):
            for T in (100000, 40000, 20000):
                nS = next((n for n in nS_opts if sd[(0.5, T)] / math.sqrt(n) <= 0.25 * MDE_EST[0.5]), None)
                if nS is None:
                    trace.append(dict(lam=lam, T_train=T, ok=False, why="n_S=128 insufficient"))
                    continue
                B = lam * nS
                ok_b = B <= m1["B_max"]
                ts = interp_tstep(m1["rows"], B, lam, nS)
                run_s = 300 * T * ts / 1e3
                trace.append(dict(lam=lam, T_train=T, n_S=nS, B=B, ok_B=ok_b, t_step_ms=ts, run_s_300gen=run_s, ok_time=run_s <= W))
                if ok_b and run_s <= W and choice is None:
                    choice = trace[-1]
        rec[name] = dict(W_run_s=W, choice=choice, feasible=choice is not None, trace=trace)
    nS09 = {T: sd[(0.9, T)] / math.sqrt(128) <= 0.25 * 0.0016 for T in (20000, 40000, 100000)}
    doc = dict(spec="direction2-spec v1.0-rev2 s11 (M-1, M-2) + s5.5 decision rules (recommendation only)",
               git_sha=git_sha(), B_max=m1["B_max"], m1_rows=m1["rows"], m2_rows=rows2, m2_summary=summ,
               rules=rec, r09_power_ok_at_nS128=nS09)
    json.dump(doc, open(os.path.join(ROOT, "results", "direction2_meas.json"), "w"), indent=1)
    print(json.dumps(dict(summary=summ, rules={k: dict(choice=v["choice"], feasible=v["feasible"]) for k, v in rec.items()}), indent=1))


# =====================================================================================================  v2 (red-team fixes)
V2_SIG = (0.03, 0.1, 0.3)
V2_R = (0.5, 0.9)
NS_OPTS_V2 = (16, 32, 64, 128, 256, 512)


def _uptime():
    return subprocess.check_output(["uptime"], text=True).strip()


def part_m2v2(args):
    """M-2': mirrored pairs (theta0 +/- delta, theta0 = 0 = honest) sigma x r grid, T=2e4, 20 pairs, n_S=64."""
    cfg, T, seeds = CFG, 20000, list(range(64))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for r in V2_R:
        for sig in V2_SIG:
            rng = np.random.default_rng(7)
            pols = []
            for _ in range(20):
                dl = sig * rng.normal(size=55)
                pols += [policy.MLPPolicy(theta=dl), policy.MLPPolicy(theta=-dl)]
            t0 = time.time()
            out = env.simulate("M3C", T, cfg, pols, seeds, r=r, dev="cuda")
            f = (out["snap"]["util_b"][:, :, 1] / out["T_score"]).cpu().numpy().astype(np.float64)
            for k in range(20):
                a, b = f[2 * k], f[2 * k + 1]
                vc, vi = float(np.var(a - b, ddof=1)), float(np.var(a, ddof=1) + np.var(b, ddof=1))
                rows.append(dict(r=r, sigma=sig, T=T, pair_id=k, var_crn=vc, var_indep=vi, vrf=vi / vc, sd_crn=math.sqrt(vc), n_seeds=64))
            print("r", r, "sigma", sig, "done", round(time.time() - t0), "s", flush=True)
            json.dump(dict(center="theta=0 (honest warm start)", rows=rows, git_sha=git_sha()),
                      open(os.path.join(OUT, "measure_v2_m2.json"), "w"), indent=1)


def _timed(st, bank, warm, n):
    for _ in range(warm):
        env._step(st, bank)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        env._step(st, bank)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def _frontier_warm(dev):
    """frontier ms/step: warm up (T=1200 once), then slope between T=2000 and T=6000 (median of 3)."""
    from arbitration.gpu import frontier
    seeds = list(range(32))
    el = frontier.build([dict(CFG)], [dict(rho=0.5)], [{}], seeds, dev=dev)
    frontier.simulate("M3C", 1200, el, seeds)
    torch.cuda.synchronize()
    sl = []
    for _ in range(3):
        res = {}
        for T in (2000, 6000):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            frontier.simulate("M3C", T, el, seeds)
            torch.cuda.synchronize(); res[T] = time.perf_counter() - t0
        sl.append((res[6000] - res[2000]) / 4000 * 1e3)
    return statistics.median(sl), sl


def part_m1v2(args):
    dev = "cuda"
    rng = np.random.default_rng(0)
    lam = 256
    up0 = _uptime()
    fms, fall = _frontier_warm(dev)
    rows = []
    for nS in (64, 128, 256, 512):
        B = lam * nS
        means, p95s, meds, mem = [], [], [], 0.0
        for rep in range(3):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            st, bank = make_bank_state(lam, nS, dev, rng)
            ts = _timed(st, bank, 200, 2200)         # 2200 timed steps span two RNG block boundaries (amortised draw cost included)
            means.append(float(np.mean(ts))); p95s.append(float(np.percentile(ts, 95))); meds.append(float(np.median(ts)))
            mem = max(mem, torch.cuda.max_memory_allocated() / 2 ** 20)
            del st, bank
        rows.append(dict(lam=lam, n_S=nS, B=B, t_step_ms_mean=statistics.mean(means), t_step_ms_mean_reps=means,
                         t_step_ms_p95=statistics.mean(p95s), t_step_ms_p95_reps=p95s, t_step_ms_median=statistics.mean(meds),
                         steps_timed=2200, warmup=200, repeats=3, mem_peak_MB=mem, uptime=_uptime()))
        print(rows[-1], flush=True)
    # memory scaling: fixed B = 16384 with different (lam, n_S) splits, and fixed lam / n_S sweeps (peak memory of 20 steps)
    mem_rows = []
    for lam_, nS_ in [(256, 64), (128, 128), (64, 256), (32, 512), (256, 128), (512, 64), (1024, 64), (256, 256), (256, 512)]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        st, bank = make_bank_state(lam_, nS_, dev, rng)
        _timed(st, bank, 0, 20)
        mem_rows.append(dict(lam=lam_, n_S=nS_, B=lam_ * nS_, mem_peak_MB=torch.cuda.max_memory_allocated() / 2 ** 20))
        del st, bank
    # B_max: double B (n_S=64, pop=B/64) until the 12 GB limit is really reached, or OOM.  max_B_tried is reported honestly.
    bmax_rows, B, reached, cap = [], 4096, False, 2 ** 23
    B_max = 0
    while B <= cap:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            st, bank = make_bank_state(B // 64, 64, dev, rng)
            _timed(st, bank, 0, 20)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            del st, bank
        except torch.cuda.OutOfMemoryError:
            bmax_rows.append(dict(B=B, oom=True)); reached = True
            break
        bmax_rows.append(dict(B=B, mem_peak_MB=peak))
        print("B", B, round(peak), "MB", flush=True)
        if peak > 12 * 1024:
            reached = True
            break
        B_max = B
        B *= 2
    # fit memory ~ a*B for extrapolation
    pts = [(x["B"], x["mem_peak_MB"]) for x in bmax_rows if "mem_peak_MB" in x]
    slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) > 1 else None
    extrap = (12 * 1024 / slope) if slope else None
    # full validation: mean theta, 32 validation seeds (2000..2031), T=1e5, direction-2 mode (spec REQ-OPT-06)
    vp = [policy.MLPPolicy(theta=0.1 * np.random.default_rng(3).normal(size=55))]
    vs = list(range(2000, 2032))
    env.simulate("M3C", 2000, CFG, vp, vs, r=0.5, dev=dev)       # warm-up
    torch.cuda.synchronize()
    vals = []
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        env.simulate("M3C", 100000, CFG, vp, vs, r=0.5, dev=dev)
        torch.cuda.synchronize(); vals.append(time.perf_counter() - t0)
        print("validation s", vals[-1], flush=True)
    doc = dict(git_sha=git_sha(), device=torch.cuda.get_device_name(), torch=torch.__version__, uptime_start=up0, uptime_end=_uptime(),
               rows=rows, mem_scaling=mem_rows, B_max_probe=bmax_rows, B_max_reached_12GB_or_OOM=reached,
               B_max_largest_ok=B_max, B_max_note=("12 GB (or OOM) reached" if reached else
                                                   f"12 GB limit NOT reached up to B={cap // 2 if B > cap else B}; B_max >= {B_max}"),
               mem_slope_MB_per_elem=slope, B_max_extrapolated_12GB=extrap,
               frontier_ms_per_step_warm=fms, frontier_slopes=fall,
               validation_T1e5_32seeds_s=vals, validation_s_mean=float(np.mean(vals)))
    json.dump(doc, open(os.path.join(OUT, "measure_v2_m1.json"), "w"), indent=1)


def part_merge_v2(args):
    m1 = json.load(open(os.path.join(OUT, "measure_v2_m1.json")))
    m2 = json.load(open(os.path.join(OUT, "measure_v2_m2.json")))["rows"]
    summ = []
    for r in V2_R:
        for sig in V2_SIG:
            sel = [x for x in m2 if x["r"] == r and x["sigma"] == sig]
            summ.append(dict(r=r, sigma=sig, vrf_median=statistics.median(x["vrf"] for x in sel), vrf_min=min(x["vrf"] for x in sel),
                             sd_crn_median=statistics.median(x["sd_crn"] for x in sel), sd_crn_max=max(x["sd_crn"] for x in sel)))
    out = dict(m2_summary=summ)
    nS_need = {}
    for r in V2_R:
        thr = 0.25 * MDE_EST[r]
        sub = [s for s in summ if s["r"] == r]
        sd_med_worst = max(s["sd_crn_median"] for s in sub)      # worst config, median over pairs
        sd_max_worst = max(s["sd_crn_max"] for s in sub)         # worst config, worst pair
        pick = lambda sd: next((n for n in NS_OPTS_V2 if sd / math.sqrt(n) <= thr), None)
        nS_need[r] = dict(threshold=thr, sd_worst_median=sd_med_worst, sd_worst_pair=sd_max_worst,
                          nS_formula_continuous_median=(sd_med_worst / thr) ** 2, nS_formula_continuous_pair=(sd_max_worst / thr) ** 2,
                          nS_from_median=pick(sd_med_worst), nS_from_worst_pair=pick(sd_max_worst),
                          nS_from_median_cap128=pick(sd_med_worst) if (pick(sd_med_worst) or 999) <= 128 else None)
    out["nS_need"] = nS_need
    byn = {x["n_S"]: x for x in m1["rows"]}
    Tn, G, nval, t_val = 20000, 300, 12, m1["validation_s_mean"]
    runs = {}
    for nS in (64, 128, 256, 512):
        x = byn[nS]
        for stat in ("mean", "p95"):
            ts = x["t_step_ms_" + stat]
            run = G * Tn * ts / 1e3
            runs[f"nS{nS}_{stat}"] = dict(n_S=nS, t_step_ms=ts, train_s=run, validation_s=nval * t_val, total_s=run + nval * t_val,
                                          total_h=(run + nval * t_val) / 3600, mem_peak_MB=x["mem_peak_MB"],
                                          ok_S1_2p5h=run + nval * t_val <= 9000, ok_main_3h=run + nval * t_val <= 10800)
    out["run_time_lam256_T2e4_300gen"] = runs
    out["validation"] = dict(t_val_s=t_val, n_val_per_run=nval, total_s=nval * t_val)
    out["B_max_note"] = m1["B_max_note"]
    json.dump(out, open(os.path.join(OUT, "measure_v2_summary.json"), "w"), indent=1)
    meas = json.load(open(os.path.join(ROOT, "results", "direction2_meas.json")))
    meas["v2"] = dict(m1=m1, m2_rows=m2, **out, note="red-team re-measurement; v1 data above kept unchanged")
    json.dump(meas, open(os.path.join(ROOT, "results", "direction2_meas.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["m1", "m2", "merge", "m2v2", "m1v2", "merge_v2"])
    ap.add_argument("--r", type=float, default=0.5)
    a = ap.parse_args()
    {"m1": part_m1, "m2": part_m2, "merge": part_merge, "m2v2": part_m2v2, "m1v2": part_m1v2, "merge_v2": part_merge_v2}[a.part](a)
