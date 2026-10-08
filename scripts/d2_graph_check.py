#!/usr/bin/env python3
"""Pre-switch acceptance check for CUDA-graph stepping on the machine that will run the experiment (docs/direction2-speedup-plan.md, section 9).

For each --cell it takes the cell's latest checkpoint from this repository (written by the EAGER program, possibly on another machine; no checkpoint ->
generation 0, cold start) and continues it for --gens generations in a throw-away directory, twice: with the eager simulator and with CUDA graphs,
at the real S1 size (lam=256, n_S=256, T=20000).  With several --cell it runs a third time: all cells at the same time (one thread + CUDA stream
each, graphs), which is what `d2_run_queue.py --concurrent N` does.  Everything must be bitwise equal to the eager continuation: the part record
(curves, validation records, best, mean_theta, sigma) and the checkpoint.  Nothing is written into the repository.

  gpujob python3 scripts/d2_graph_check.py --cell s1_O1_s0.02 --cell s1_D2_s0.02 --gens 2        # on a school machine without gpujob: python3 ...
Exit code 0 = all equal.  Prints one JSON verdict.  Also reports the time per generation of each variant.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

TIME_FIELDS = ("wall_s", "t_step_ms", "git_sha")
SEGMENT_FIELDS = ("gen0", "gen1", "G_val", "status")


def _setup(name, root_tmp, pl):
    import numpy as np  # noqa: F401
    from arbitration.rl import env, parts, plan as _plan, policy
    S = _plan.cell_to_settings(_plan.get_cell(pl, name))
    L = int(S["cfg"]["L"])
    cell = dict(name=name, n=policy.n_params(S["F"], S["hidden"]), lam=S["lam"], n_S=S["n_S"], T_score=env.score_window(S["T_train"], L),
                T_train=S["T_train"], sigma0=S["sigma0"], val_every=S["val_every"], patience=None)
    src = parts.ckpt_path(name, 0, ROOT)
    g0 = 0
    if os.path.exists(src):
        from arbitration.rl import train
        g0 = int(train._load_ckpt(src)["gen"])
    return S, cell, g0, src


def _fresh_root(src, name, g0):
    from arbitration.rl import parts
    root = tempfile.mkdtemp(prefix="d2_graph_check_")
    if g0 > 0:
        dst = parts.ckpt_path(name, 0, root)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    return root


def _run(name, S, cell, g0, g1, src, graph, stream=None):
    from arbitration.rl import train
    root = _fresh_root(src, name, g0)
    kw = {} if graph is None else dict(graph=graph)
    sim = train.make_sim_fn(S["mech"], S["T_train"], S["cfg"], S["r"], "cuda", None, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"], **kw)
    val = train.make_val_fn(S["mech"], S["T_val"], S["cfg"], S["r"], "cuda", S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"], **kw)
    t0 = time.time()
    rec = train.train_segment(cell, 0, g0, g1, sim, val, root=root, log=lambda s: None)
    from arbitration.rl import parts
    ck = train._load_ckpt(parts.ckpt_path(name, 0, root))
    shutil.rmtree(root, ignore_errors=True)
    return dict(rec={k: v for k, v in rec.items() if k not in TIME_FIELDS + SEGMENT_FIELDS}, ckpt=ck, wall_s=time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--skip-concurrent", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from arbitration.rl import graphsim, multitrain, plan as _plan
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    cells = {n: _setup(n, None, pl) for n in a.cell}
    verdict = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), gens=a.gens, K=a.K, cells={}, ok=True)
    ref = {}
    for n, (S, cell, g0, src) in cells.items():
        e = _run(n, S, cell, g0, g0 + a.gens, src, None)
        g = _run(n, S, cell, g0, g0 + a.gens, src, graphsim.GraphSim(K=a.K))
        ref[n] = e
        same = e["rec"] == g["rec"] and e["ckpt"] == g["ckpt"]
        verdict["cells"][n] = dict(start_gen=g0, graph_equals_eager=same, eager_s_per_gen=e["wall_s"] / a.gens, graph_s_per_gen=g["wall_s"] / a.gens)
        verdict["ok"] &= same
        print(n, "start gen", g0, "graph == eager:", same, flush=True)
    if len(cells) > 1 and not a.skip_concurrent:
        out, lanes = {}, {n: multitrain.Lane(True, a.K) for n in cells}

        def work(n):
            S, cell, g0, src = cells[n]
            with torch.cuda.stream(lanes[n].stream):
                out[n] = _run(n, S, cell, g0, g0 + a.gens, src, lanes[n].gs)
                torch.cuda.current_stream().synchronize()
        t0 = time.time()
        th = [threading.Thread(target=work, args=(n,)) for n in cells]
        [t.start() for t in th]
        [t.join() for t in th]
        verdict["concurrent_wall_s"] = time.time() - t0
        for n in cells:
            same = n in out and out[n]["rec"] == ref[n]["rec"] and out[n]["ckpt"] == ref[n]["ckpt"]
            verdict["cells"][n]["concurrent_equals_eager"] = same
            verdict["ok"] &= same
            print(n, "concurrent == eager:", same, flush=True)
    print(json.dumps(verdict, indent=1, default=float))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
