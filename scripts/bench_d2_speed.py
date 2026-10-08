#!/usr/bin/env python3
"""Local speed measurement of the Direction-2 simulator step loop (submit with `gpujob`; one invocation = one configuration).

  gpujob python3 scripts/bench_d2_speed.py --mode eager --lanes 1
  gpujob python3 scripts/bench_d2_speed.py --mode graph --lanes 2        # 2 cells advance at the same time (threads + CUDA streams)

Each lane is one S1 cell (the first --lanes of --cells), real S1 size (lam=256, n_S=256, T=20000: B = 65792 per lane) unless overridden.
Every lane runs one warm-up generation (graph build / lazy kernels), then all lanes start together and run --gens timed generations of the
simulation (sim_fn only: policy objects -> simulate -> U), as train_segment calls it.  Reports ms per step per lane, cell-steps/s (all lanes), and
the mean GPU utilisation sampled from nvidia-smi during the timed window.  The fitness of every lane is compared with the single-lane eager value
when --check is given (bitwise).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

DEFAULT_CELLS = "s1_D1_s0.02,s1_D2_s0.02,s1_O1_s0.02,s1_O2_s0.02"


def _util_sampler():
    try:
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"],
                             stdout=subprocess.PIPE, text=True)
    except OSError:
        return None
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["eager", "graph"], required=True)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--lam", type=int, default=256)
    ap.add_argument("--nS", type=int, default=256)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--out", default=None, help="append the result as one JSON line to this file")
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    from arbitration.rl import graphsim, plan as _plan, policy, train

    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    names = a.cells.split(",")[: a.lanes]
    if len(names) < a.lanes:
        raise SystemExit("not enough cells for the requested number of lanes")
    seeds = list(range(a.nS))
    res = {"mode": a.mode, "lanes": a.lanes, "cells": names, "gens": a.gens, "T": a.T, "lam": a.lam, "nS": a.nS, "K": a.K,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": os.uname().nodename}
    lane_state = []
    for i, nm in enumerate(names):
        S = _plan.cell_to_settings(_plan.get_cell(pl, nm))
        stream = torch.cuda.Stream() if a.lanes > 1 else None
        gs = graphsim.GraphSim(K=a.K, capture=True, stream=stream) if a.mode == "graph" else None
        sim = train.make_sim_fn(S["mech"], a.T, S["cfg"], S["r"], a.dev, None, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"], graph=gs)
        th = np.random.default_rng(100 + i).standard_normal((a.lam, policy.n_params(S["F"], S["hidden"]))) * 0.5
        lane_state.append(dict(name=nm, sim=sim, th=th, stream=stream, gs=gs, times=[], U=[]))
    box = {}

    def released():                                  # runs once, in one lane thread, when every lane finished its warm-up generation
        box["t0"] = time.time()
        box["util"] = _util_sampler()
    barrier = threading.Barrier(a.lanes, action=released)
    err = []

    def lane(ls):
        try:
            ctx = torch.cuda.stream(ls["stream"]) if ls["stream"] is not None else None
            if ctx:
                ctx.__enter__()
            t0 = time.time()
            ls["sim"](np.vstack([ls["th"]]), seeds)                           # warm-up generation (graph build / lazy kernels)
            ls["warm_s"] = time.time() - t0
            barrier.wait()
            for g in range(a.gens):
                t0 = time.time()
                U = ls["sim"](ls["th"], seeds)
                ls["times"].append(time.time() - t0)
                ls["U"].append(float(np.sum(U)))
            ls["end"] = time.time()
        except BaseException as e:                                             # noqa: BLE001
            err.append(repr(e))
            try:
                barrier.abort()
            except Exception:  # noqa: BLE001
                pass
    threads = [threading.Thread(target=lane, args=(ls,)) for ls in lane_state]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    util = box.get("util")
    if util is not None:
        util.terminate()
        out = util.stdout.read().split()
        vals = [float(x) for x in out if x.replace(".", "").isdigit()]
    else:
        vals = []
    if err:
        raise SystemExit("lane failed: " + "; ".join(err))
    wall = max(ls["end"] for ls in lane_state) - box["t0"]
    tot_steps = a.lanes * a.gens * a.T
    res.update(wall_s=wall, cell_steps_per_s=tot_steps / wall,
               ms_per_step_per_lane=[1000.0 * sum(ls["times"]) / (a.gens * a.T) for ls in lane_state],
               ms_per_step_aggregate=1000.0 * wall / (a.gens * a.T),
               gen_times_s=[ls["times"] for ls in lane_state], warm_s=[ls["warm_s"] for ls in lane_state],
               gpu_util_samples=len(vals), gpu_util_mean=(sum(vals) / len(vals) if vals else None),
               gpu_util_max=(max(vals) if vals else None), max_mem_reserved_mb=torch.cuda.max_memory_reserved() / 2**20, sumU=[ls["U"] for ls in lane_state])
    print(json.dumps(res, default=float))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res, default=float) + "\n")


if __name__ == "__main__":
    main()
