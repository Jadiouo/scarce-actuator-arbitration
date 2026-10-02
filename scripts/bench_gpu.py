"""Benchmarks of the GPU accelerators vs the numpy CPU code (one-off; prints timings).

    python3 scripts/bench_gpu.py dp  [K ...]        # DP: single and batch of 16, torch vs numpy
    python3 scripts/bench_gpu.py dpn4 [K ...]       # N=4 feasibility (torch only, + numpy at small K)
    python3 scripts/bench_gpu.py sim [--cpu]        # v2_sweep-sized simulation workload
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from arbitration import dp
from arbitration.model import make_config


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _np_solve(args):
    P, R, name, K = args
    t, c = time.time(), time.process_time()
    f = dp.FullDP(P, R, make_config(name), K)
    return f.g, f.iters, time.time() - t, time.process_time() - c


def bench_dp(Ks, N=3, batch=16, model="v2", do_np=True, np_pool=True):
    from arbitration.gpu.dp_torch import solve_full
    cfg = make_config(model)
    ins = [dp.instance(s, N, 0.004, 0.6)[:2] for s in range(batch)]
    for K in Ks:
        solve_full(ins[:1], cfg, min(K, 12))                     # warm-up (cuda context, kernels)
        sync(); t = time.time(); s1 = solve_full(ins[:1], cfg, K); sync(); t1 = time.time() - t
        sync(); t = time.time(); sb = solve_full(ins, cfg, K); sync(); tb = time.time() - t
        line = f"N={N} K={K}: torch single {t1:.2f}s ({s1[0].iters} it)  batch{batch} {tb:.2f}s"
        if do_np:
            _init()                                              # numpy single: 1 BLAS thread, like the pool workers
            g0, it0, tn1, cn1 = _np_solve((*ins[0], model, K))
            line += f" | numpy single {tn1:.2f}s wall / {cn1:.2f}s cpu  dJ={abs(g0 - s1[0].g):.1e}"
            if np_pool:
                from concurrent.futures import ProcessPoolExecutor
                t = time.time()
                with ProcessPoolExecutor(os.cpu_count(), initializer=_init) as ex:
                    res = list(ex.map(_np_solve, [(*i, model, K) for i in ins], chunksize=1))
                tnb = time.time() - t
                dj = max(abs(r[0] - s.g) for r, s in zip(res, sb))
                cpu = sum(r[3] for r in res)
                line += (f"  numpy batch{batch} (16 procs) {tnb:.2f}s wall, cpu-sum/16 = {cpu / 16:.2f}s"
                         f"  max dJ={dj:.1e}")
        print(line, flush=True)


def _init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def bench_sim(cpu_too=True, seeds=32, steps=1500):
    from arbitration import experiments as ex
    from arbitration.gpu.sim_torch import collect_gpu, SUPPORTED
    rates = ex.FULL["rates"]
    arms = {a: v for a, v in ex.V2_ARMS.items() if v[0] in SUPPORTED}
    print(f"{len(arms)} supported arms of {len(ex.V2_ARMS)}; {len(rates)} loads; {seeds} seeds; {steps} steps")
    cells_by = {nm: {(a, r): (pol, dict(rate=r, model=nm, **kw)) for a, (pol, kw) in arms.items() for r in rates}
                for nm in ("legacy", "v2")}
    collect_gpu({("none", 0.004): ("none", dict(rate=0.004))}, range(2), 20)      # warm-up
    sync(); t = time.time()
    for nm, cells in cells_by.items():
        collect_gpu(cells, range(seeds), steps)
    sync(); tg = time.time() - t
    nruns = 2 * len(arms) * len(rates) * seeds
    print(f"GPU: {tg:.1f}s for {nruns} runs")
    if cpu_too:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(os.cpu_count(), initializer=_init)
        ex._MAP = pool.map
        t = time.time()
        for nm, cells in cells_by.items():
            ex.collect(cells, range(seeds), steps)
        tc = time.time() - t
        pool.shutdown()
        ch = os.times()
        cpu = ch.children_user + ch.children_system
        print(f"CPU ({os.cpu_count()} procs): wall {tc:.1f}s (machine was shared); total cpu {cpu:.0f}s "
              f"-> ideal 16-core {cpu / os.cpu_count():.1f}s")
        print(f"speedup: {tc / tg:.1f}x vs wall, {cpu / os.cpu_count() / tg:.1f}x vs ideal 16-core")


if __name__ == "__main__":
    what = sys.argv[1]
    nums = [int(x) for x in sys.argv[2:] if x.isdigit()]
    if what == "dp":
        bench_dp(nums or [60, 80, 160], np_pool="--nopool" not in sys.argv)
    elif what == "dpn4":
        bench_dp(nums or [20, 40], N=4, batch=1, do_np="--nonp" not in sys.argv, np_pool=False)
    elif what == "sim":
        bench_sim("--nocpu" not in sys.argv)
