#!/usr/bin/env python3
"""Pre-switch acceptance check for CUDA-graph stepping on the machine that will run the experiment (docs/direction2-speedup-plan.md, section 9).

Checks (all must be bitwise equal to the EAGER simulator; nothing is written into the repository):

 continuation   for each --cell: take the cell's latest checkpoint from this repository (written by the EAGER program, possibly on another machine; none ->
                generation 0), continue it --gens generations in a throw-away directory, once eager and once with CUDA graphs, at the real S1 size
                (lam=256, n_S=256, T=20000).  With several --cell also a third time with all cells at once (one thread + CUDA stream each, what
                `d2_run_queue.py --concurrent N` does).  Compared: part record (curves, validation records, best, mean_theta, sigma) and checkpoint.
 --cross-val    (or --cross-val-cell C, repeatable) the continuation also crosses a VALIDATION boundary (red-team H1): the cell's val_every is set so that the last of the --gens generations
                validates (G_val at the real T_val=100000, B=64, its own graph) and the best-checkpoint selection runs; the verdict records that it did.
 --val-cell C   (repeatable) val_fn(theta) of cell C at the real size, eager vs graph, theta = the checkpoint mean (or a fixed random vector).  Needs no --cell.
 --preflight    everything of plan 9.1 (1)-(4) plus the validation checks, one JSON verdict (PASS/FAIL, per-step details, measured time, estimate for the
                school machine).  Steps: 1 CPU tests (speed-up tests; --full: all), 2 GPU graph/multi tests, 3 GPU eager-path tests (T01/T36/T41; --full: + T02/T06/T08), 4 continuation with --cross-val (2 cells + concurrent),
                H1 val_fn at the real size.  --only/--skip pick steps, --json-out writes the verdict, --school-factor scales the measured time (local 5070 Ti
                -> RTX8000 is about 3).

  gpujob python3 scripts/d2_graph_check.py --preflight --school-factor 3     # on the local machine (projection)
  python3 scripts/d2_graph_check.py --preflight --json-out ~/preflight.json  # on the school machine
  gpujob python3 scripts/d2_graph_check.py --cell s1_O1_s0.02 --cell s1_D2_s0.02 --gens 2 --cross-val
Exit code 0 = all equal / PASS.  Prints one JSON verdict.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

TIME_FIELDS = ("wall_s", "t_step_ms", "git_sha")
DEFAULT_VAL_CELL = "s1_M4rw0_anchor"                    # the env._step path (the other S1 cells use vulns._vstep)
FALLBACK_CELLS = ["s1_D2_s0.02", "s1_O1_s0.02"]
BUDGET_MIN = 60.0


# ------------------------------------------------------------------------------------------------ helpers
def default_cells(repo_root=ROOT, n=2):
    """The (at most) n S1 cells that have a checkpoint in the repository (the ones the school machine is running), else FALLBACK_CELLS."""
    d = os.path.join(repo_root, "results", "direction2", "ckpt")
    try:
        names = sorted(f[:-len("__run0.npz")] for f in os.listdir(d) if f.endswith("__run0.npz"))
    except OSError:
        names = []
    names = [x for x in names if x.startswith("s1_") and "anchor" not in x]
    return names[:n] if names else list(FALLBACK_CELLS)


def _setup(name, pl, shrink=None, repo_root=ROOT, force_val_at=None):
    from arbitration.rl import env, parts, plan as _plan, policy
    S = _plan.cell_to_settings(_plan.get_cell(pl, name))
    S.update(shrink or {})
    L = int(S["cfg"]["L"])
    cell = dict(name=name, n=policy.n_params(S["F"], S["hidden"]), lam=S["lam"], n_S=S["n_S"], T_score=env.score_window(S["T_train"], L),
                T_train=S["T_train"], sigma0=S["sigma0"], val_every=S["val_every"], patience=None)
    src = parts.ckpt_path(name, 0, repo_root)
    g0 = 0
    if os.path.exists(src):
        from arbitration.rl import train
        g0 = int(train._load_ckpt(src)["gen"])
    return S, cell, g0, src


def _fns(S, graph, dev):
    from arbitration.rl import train
    kw = {} if graph is None else dict(graph=graph)
    sim = train.make_sim_fn(S["mech"], S["T_train"], S["cfg"], S["r"], dev, None, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"], **kw)
    val = train.make_val_fn(S["mech"], S["T_val"], S["cfg"], S["r"], dev, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"], **kw)
    return sim, val


def _fresh_root(src, name, g0):
    from arbitration.rl import parts
    root = tempfile.mkdtemp(prefix="d2_graph_check_")
    if g0 > 0:
        dst = parts.ckpt_path(name, 0, root)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    return root


def _run(name, S, cell, g0, g1, src, graph, dev="cuda"):
    from arbitration.rl import parts, train
    root = _fresh_root(src, name, g0)
    sim, val = _fns(S, graph, dev)
    t0 = time.time()
    rec = train.train_segment(cell, 0, g0, g1, sim, val, root=root, log=lambda s: None)
    ck = train._load_ckpt(parts.ckpt_path(name, 0, root))
    shutil.rmtree(root, ignore_errors=True)
    return dict(rec={k: v for k, v in rec.items() if k not in TIME_FIELDS}, ckpt=ck, wall_s=time.time() - t0)


def _graph(K, capture, stream=None):
    from arbitration.rl import graphsim
    return graphsim.GraphSim(K=K, capture=capture, stream=stream)


def make_checkpoint_root(root, name, gens, dev="cpu", shrink=None):
    """(tests) a repository-shaped directory holding an EAGER checkpoint of `name` after `gens` generations."""
    from arbitration.rl import plan as _plan, train
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    S, cell, _, _ = _setup(name, pl, shrink, repo_root=str(root))
    sim, val = _fns(S, None, dev)
    train.train_segment(cell, 0, 0, gens, sim, val, root=str(root), log=lambda s: None)
    return str(root)


# ------------------------------------------------------------------------------------------------ the checks
def _val_value(S, theta, graph, dev):
    _, val = _fns(S, graph, dev)
    return float(val(theta))


def check_val(name, K=32, dev="cuda", capture=None, shrink=None, repo_root=ROOT):
    """val_fn(theta) at the real T_val (B=64), eager vs graph."""
    import numpy as np
    from arbitration.rl import plan as _plan, train
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    S, cell, g0, src = _setup(name, pl, shrink, repo_root)
    theta = None
    if g0 > 0:
        theta = np.asarray(train._load_ckpt(src)["opt"]["mean"], dtype=float)
    if theta is None:
        theta = 0.5 * np.random.default_rng(12345).standard_normal(cell["n"])
    t0 = time.time()
    e = _val_value(S, theta, None, dev)
    t1 = time.time()
    g = _val_value(S, theta, _graph(K, capture), dev)
    t2 = time.time()
    return dict(T_val=S["T_val"], theta_from="checkpoint g%d" % g0 if g0 > 0 else "random(12345)", val_eager=e, val_graph=g, equal=bool(e == g),
                eager_s=t1 - t0, graph_s=t2 - t1)


def check_continuation(name, gens=2, K=32, dev="cuda", capture=None, shrink=None, cross_val=False, repo_root=ROOT, _cells=None):
    from arbitration.rl import plan as _plan
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    S, cell, g0, src = _cells[name] if _cells else _setup(name, pl, shrink, repo_root)
    if cross_val:
        cell = dict(cell, val_every=g0 + gens)                    # (g + 1) % val_every == 0 first happens at the last generation of the segment
    e = _run(name, S, cell, g0, g0 + gens, src, None, dev)
    g = _run(name, S, cell, g0, g0 + gens, src, _graph(K, capture), dev)
    new = [r for r in e["rec"]["val_records"] if r["gen"] > g0]
    same = e["rec"] == g["rec"] and e["ckpt"] == g["ckpt"]
    out = dict(start_gen=g0, graph_equals_eager=bool(same), validated=bool(new), val_records_new=len(new), best_set=e["rec"]["best"] is not None,
               eager_s_per_gen=e["wall_s"] / gens, graph_s_per_gen=g["wall_s"] / gens)
    if cross_val and not new:
        out["graph_equals_eager"] = False                         # asked to cross a validation boundary but none happened: the check did not test what it claims
        out["error"] = "no validation happened in the compared segment"
    return out, e, (S, cell, g0, src)


def run_checks(names, gens=2, K=32, dev="cuda", capture=None, shrink=None, cross_val=False, val_cells=(), concurrent=True, repo_root=ROOT):
    """cross_val: True = every cell's continuation crosses a validation boundary; a list of cell names = only those (the eager validation at T_val=100000
    is the expensive part of the whole check, ~3.5 min on the local GPU)."""
    import torch
    from arbitration.rl import multitrain, plan as _plan
    pl = _plan.load_plan(os.path.join(ROOT, _plan.DEFAULT_PLAN), os.path.join(ROOT, _plan.DEFAULT_FREEZE))
    verdict = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, dev=dev, gens=gens, K=K,
                   cross_val=sorted(cross_val) if isinstance(cross_val, (list, tuple, set)) else bool(cross_val), cells={}, val={}, ok=True)
    ref, setups = {}, {}
    cv = (lambda n: n in cross_val) if isinstance(cross_val, (list, tuple, set)) else (lambda n: bool(cross_val))      # noqa: E731
    for n in names:
        r, e, setup = check_continuation(n, gens, K, dev, capture, shrink, cv(n), repo_root)
        r["_cell"] = setup[1]
        ref[n], setups[n] = e, setup
        verdict["cells"][n] = {k: v for k, v in r.items() if k != "_cell"}
        verdict["ok"] &= r["graph_equals_eager"]
        print(n, "start gen", r["start_gen"], "graph == eager:", r["graph_equals_eager"], "| validated:", r["validated"], flush=True)
    if len(names) > 1 and concurrent:
        import torch
        out, lanes = {}, {n: multitrain.Lane(True, K, capture, dev) for n in names}

        def work(n):
            S, cell, g0, src = setups[n]
            if cv(n):
                cell = dict(cell, val_every=g0 + gens)
            try:
                if lanes[n].stream is not None:
                    with torch.cuda.stream(lanes[n].stream):
                        out[n] = _run(n, S, cell, g0, g0 + gens, src, lanes[n].gs, dev)
                        torch.cuda.current_stream().synchronize()
                else:
                    out[n] = _run(n, S, cell, g0, g0 + gens, src, lanes[n].gs, dev)
            except BaseException as ex:                           # noqa: BLE001 - a lane failure is a FAIL, not a crash of the check
                out[n] = dict(error=f"{type(ex).__name__}: {ex}")
        t0 = time.time()
        th = [threading.Thread(target=work, args=(n,)) for n in names]
        [t.start() for t in th]
        [t.join() for t in th]
        verdict["concurrent_wall_s"] = time.time() - t0
        for n in names:
            same = "rec" in out.get(n, {}) and out[n]["rec"] == ref[n]["rec"] and out[n]["ckpt"] == ref[n]["ckpt"]
            verdict["cells"][n]["concurrent_equals_eager"] = bool(same)
            if not same and "error" in out.get(n, {}):
                verdict["cells"][n]["concurrent_error"] = out[n]["error"]
            verdict["ok"] &= bool(same)
            print(n, "concurrent == eager:", same, flush=True)
    for n in val_cells:
        v = check_val(n, K, dev, capture, shrink, repo_root)
        verdict["val"][n] = v
        verdict["ok"] &= v["equal"]
        print(n, "val_fn graph == eager:", v["equal"], flush=True)
    return verdict


# ------------------------------------------------------------------------------------------------ --preflight
def preflight_steps(cells, val_cell, python=sys.executable, gens=2, K=32, full=False):
    """Step 4 crosses the validation boundary in the FIRST cell only (eager validation at the real T_val is the expensive part); the other cells do the plain
    2-generation continuation of plan 9.1 (4).  Step H1 compares val_fn on the anchor cell, which takes the other (env._step) code path.
    Default (fits ~60 min on the RTX8000): step 1 = the speed-up tests on the CPU (speed, multi, run_queue, graph_check), step 3 = T01, T36, T41 (the three
    cheap eager-path checks).  full=True (--full): step 1 = all of tests/direction2 -m "not gpu", step 3 = the six tests of plan 9.1 (3) (adds T02, T06, T08:
    about +10 min on the local GPU, i.e. +30 min on the school machine; the full CPU suite is another ~10+ min)."""
    """The acceptance steps in order.  kind 'cmd' = a subprocess whose exit code decides; steps 4 and H1 write a JSON verdict (json_out)."""
    me = os.path.join(ROOT, "scripts", "d2_graph_check.py")
    pyt = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:metadata"]
    cellargs = [x for c in cells for x in ("--cell", c)] + ["--cross-val-cell", cells[0]]
    eager = ["tests/direction2/test_d2_env.py::test_T01_parity_m3c_per_seed", "tests/direction2/test_d2_s1.py::test_T36_vuln_identity_mutation",
             "tests/direction2/test_d2_s1.py::test_T41_vuln_ref_policy_public_only"]
    if full:
        eager += ["tests/direction2/test_d2_env.py::test_T02_parity_m4_and_naive", "tests/direction2/test_d2_env.py::test_T08_batch_invariance",
                  "tests/direction2/test_d2_env.py::test_T06_replay_d1_json_naive_m4"]
    cpu = ["tests/direction2"] if full else [f"tests/direction2/{f}" for f in ("test_d2_speed.py", "test_d2_multi.py", "test_d2_run_queue.py", "test_d2_graph_check.py")]
    return [
        dict(name="1_cpu_tests", cmd=pyt + cpu + ["-m", "not gpu"], timeout=3 * 3600),
        dict(name="2_gpu_graph_tests", cmd=pyt + ["tests/direction2/test_d2_speed.py", "tests/direction2/test_d2_multi.py", "-m", "gpu"], timeout=3 * 3600),
        dict(name="3_gpu_eager_tests", cmd=pyt + eager, timeout=3 * 3600),
        dict(name="4_checkpoint_continuation_cross_val", cmd=[python, me] + cellargs + ["--gens", str(gens), "--K", str(K)], json_out=True,
             timeout=3 * 3600),
        dict(name="H1_val_fn_real_size", cmd=[python, me, "--val-cell", val_cell, "--K", str(K)], json_out=True, timeout=3 * 3600),
    ]


def _run_step(step):
    """Run one step as a subprocess; the result is dict(ok, seconds, detail[, verdict])."""
    cmd, jpath = list(step["cmd"]), None
    if step.get("json_out"):
        fd, jpath = tempfile.mkstemp(prefix="d2_preflight_", suffix=".json")
        os.close(fd)
        cmd += ["--json-out", jpath]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=step.get("timeout"))
        rc, text = p.returncode, (p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired as e:
        rc, text = -1, f"timeout after {step.get('timeout')} s\n{e.stdout or ''}"
    res = dict(ok=(rc == 0), seconds=time.time() - t0, rc=rc, detail="\n".join(text.strip().splitlines()[-6:]))
    if jpath:
        try:
            res["verdict"] = json.load(open(jpath))
            res["ok"] = res["ok"] and bool(res["verdict"].get("ok"))
        except (OSError, ValueError):
            res["ok"] = False
        finally:
            try:
                os.remove(jpath)
            except OSError:
                pass
    return res


def run_preflight(steps, runner=_run_step, school_factor=1.0, out_path=None):
    import platform
    verdict = dict(steps={}, failed=[], started=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(), school_factor=school_factor)
    try:
        import torch
        verdict.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:                                             # noqa: BLE001
        pass
    total = 0.0
    for s in steps:
        print(f"[preflight] {s['name']} ...", flush=True)
        try:
            r = runner(s)
        except BaseException as e:                                # noqa: BLE001 - a crashing step is a FAIL with the reason, the rest still runs
            r = dict(ok=False, seconds=0.0, detail=f"{type(e).__name__}: {e}")
        verdict["steps"][s["name"]] = r
        total += float(r.get("seconds", 0.0))
        if not r["ok"]:
            verdict["failed"].append(s["name"])
        print(f"[preflight] {s['name']}: {'PASS' if r['ok'] else 'FAIL'} ({r.get('seconds', 0.0):.0f} s)", flush=True)
    verdict["verdict"] = "PASS" if not verdict["failed"] else "FAIL"
    verdict["measured_seconds"] = total
    verdict["school_estimate_minutes"] = total * school_factor / 60.0
    verdict["within_60_min"] = verdict["school_estimate_minutes"] <= BUDGET_MIN
    if out_path:
        with open(out_path, "w") as f:
            json.dump(verdict, f, indent=1, default=float)
    return verdict


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", default=[])
    ap.add_argument("--val-cell", action="append", default=[], help="also compare val_fn(theta) eager vs graph at the real T_val for this cell (repeatable)")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--cross-val", action="store_true", help="make the continuation cross a validation boundary (val + best selection happen in the compared segment)")
    ap.add_argument("--cross-val-cell", action="append", default=[], help="like --cross-val but only for this cell (repeatable)")
    ap.add_argument("--skip-concurrent", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--preflight", action="store_true", help="run plan 9.1 (1)-(4) plus the validation checks and print one PASS/FAIL JSON verdict")
    ap.add_argument("--full", action="store_true", help="--preflight: the complete plan 9.1 (1) and (3) (all CPU tests, T02/T06/T08 too); much slower")
    ap.add_argument("--only", default=None, help="--preflight: comma separated step names (prefixes ok, e.g. 1,2,H1)")
    ap.add_argument("--skip", default=None, help="--preflight: comma separated step names to skip")
    ap.add_argument("--school-factor", type=float, default=1.0, help="--preflight: multiply the measured time by this for the school estimate (1 on the school machine, ~3 on the local one)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    if a.preflight:
        cells = a.cell or default_cells()
        steps = preflight_steps(cells, (a.val_cell or [DEFAULT_VAL_CELL])[0], a.python, a.gens, a.K, a.full)
        pick = lambda s, spec: any(s["name"] == x or s["name"].startswith(x + "_") for x in spec.split(","))      # noqa: E731
        if a.only:
            steps = [s for s in steps if pick(s, a.only)]
        if a.skip:
            steps = [s for s in steps if not pick(s, a.skip)]
        v = run_preflight(steps, school_factor=a.school_factor, out_path=a.json_out)
        v["cells"] = cells
        print(json.dumps(v, indent=1, default=float))
        return 0 if v["verdict"] == "PASS" else 1
    if not a.cell and not a.val_cell:
        ap.error("give --cell and/or --val-cell, or --preflight")
    v = run_checks(a.cell, a.gens, a.K, "cuda", None, None, True if a.cross_val else a.cross_val_cell, a.val_cell, not a.skip_concurrent)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(v, f, indent=1, default=float)
    print(json.dumps(v, indent=1, default=float))
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
