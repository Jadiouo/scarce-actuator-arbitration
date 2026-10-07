#!/usr/bin/env python3
"""Helper of scripts/d2_target_check.sh (acceptance check on the target GPU machine).  Sub-commands:
  env      print + record the environment (nvidia-smi, driver, torch, CUDA, capability, memory)
  f64      float64 GPU arithmetic check (against the CPU) and the f64/f32 speed ratio
  perf     step time of the training generation (lam=256, n_S=256, T=2e4; 1 warm-up + 3 timed generations) and one validation (T=1e5, 32 seeds)
           for a plain cell and two vulnerability cells, then the S1 / main hour estimate and the comparison with the local estimate
  record   store the result of a pytest step run by the shell script
  verdict  combine everything into results/direction2/target_check.json and print the decision
Nothing here writes into the training results (parts / ckpt); only the --dir work directory and target_check.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT_DEFAULT = os.path.join(ROOT, "results", "direction2", "target_check.json")
PERF_CELLS = ("main_M3C_r0.5", "s1_D4_s0.02", "s1_D2_s0.02")     # plain M3C; the two vulnerability cells timed locally (measure_vuln_timing.json)
MIN_FREE_GIB = 2.0


def _write(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(doc, open(path, "w"), indent=1, default=float)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"(failed: {e})"


def cmd_env(a):
    doc = dict(host=platform.node(), os=platform.platform(), python=sys.version.split()[0], problems=[])
    smi = _run(["nvidia-smi"])
    print(smi)
    doc["nvidia_smi"] = smi
    doc["driver_version"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip().splitlines()[:1]
    doc["git_head"] = _run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).strip()
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        doc["problems"].append(f"cannot import torch: {e}")
        print("FAIL: cannot import torch:", e)
        _write(a.out, doc)
        return 1
    doc["torch"] = torch.__version__
    doc["torch_cuda"] = torch.version.cuda
    doc["arch_list"] = torch.cuda.get_arch_list()
    print(f"driver version : {doc['driver_version']}\ntorch          : {torch.__version__} (built for CUDA {torch.version.cuda})\ncompiled archs : {doc['arch_list']}")
    if not torch.cuda.is_available():
        doc["problems"].append("torch.cuda.is_available() is False")
        print("FAIL: torch.cuda.is_available() is False (driver too old for this torch wheel? container started without --gpus all?)")
        _write(a.out, doc)
        return 1
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    doc.update(device=torch.cuda.get_device_name(0), capability=list(cap), mem_free_gib=free / 2 ** 30, mem_total_gib=total / 2 ** 30)
    print(f"device         : {doc['device']}\ncapability     : {cap}\nmemory         : free {free / 2 ** 30:.2f} GiB of {total / 2 ** 30:.2f} GiB")
    if f"sm_{cap[0]}{cap[1]}" not in doc["arch_list"] and not any(x.startswith("compute_") for x in doc["arch_list"]):
        doc["problems"].append(f"torch has no kernels for sm_{cap[0]}{cap[1]}")
    if free / 2 ** 30 < MIN_FREE_GIB:
        doc["problems"].append(f"only {free / 2 ** 30:.2f} GiB free GPU memory (< {MIN_FREE_GIB} GiB)")
    for p in doc["problems"]:
        print("FAIL:", p)
    _write(a.out, doc)
    return 1 if doc["problems"] else 0


def cmd_f64(a):
    import torch
    doc = dict(problems=[])
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2048, 2048, dtype=torch.float64, generator=g)
    y = torch.rand(2048, 2048, dtype=torch.float64, generator=g)
    c_cpu = x @ y
    c_gpu = (x.cuda() @ y.cuda()).cpu()
    err = float((c_cpu - c_gpu).abs().max() / c_cpu.abs().max())
    s_gpu = x.cuda().cumsum(1).cpu()
    ok_cum = bool(torch.allclose(s_gpu, x.cumsum(1), rtol=1e-12, atol=0))
    e1 = bool((x.cuda().pow(3).sqrt().cpu() - x.pow(3).sqrt()).abs().max() < 1e-14)
    doc.update(matmul_rel_err=err, cumsum_ok=ok_cum, pow_sqrt_ok=e1, dtype_kept=str((x.cuda() @ y.cuda()).dtype))
    if not (err < 1e-12 and ok_cum and e1 and doc["dtype_kept"] == "torch.float64"):
        doc["problems"].append(f"float64 GPU arithmetic disagrees with the CPU (matmul rel err {err:.2e}, cumsum {ok_cum}, pow/sqrt {e1})")

    def speed(dt):
        a_, b_ = x.to(dt).cuda(), y.to(dt).cuda()
        a_ @ b_
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            a_ @ b_
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10
    t64, t32 = speed(torch.float64), speed(torch.float32)
    doc.update(matmul_s_f64=t64, matmul_s_f32=t32, f64_over_f32_time=t64 / t32)
    print(f"float64 matmul rel. error vs CPU {err:.1e}; cumsum ok {ok_cum}; f64 matmul takes {t64 / t32:.1f}x the f32 time (Turing: expect ~30x)")
    print("OK: float64 GPU arithmetic works" if not doc["problems"] else "FAIL: " + doc["problems"][0])
    _write(a.out, doc)
    return 1 if doc["problems"] else 0


def _timed(fn):
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0


def cmd_perf(a):
    import numpy as np
    import torch
    from arbitration.rl import cmaes, env, plan, policy, train
    pl = json.load(open(os.path.join(ROOT, "results", "direction2", "run_plan.json")))
    nonstd = a.T is not None or a.gens != 4
    doc = dict(problems=[], nonstandard=nonstd, device=torch.cuda.get_device_name(0), torch=torch.__version__, cells={})
    free_gib = torch.cuda.mem_get_info(0)[0] / 2 ** 30
    for name in a.cells.split(","):
        c = plan.get_cell(pl, name)
        S = plan.cell_to_settings(c)
        T = a.T or S["T_train"]
        T_val = a.T_val or S["T_val"]
        sim = train.make_sim_fn(S["mech"], T, S["cfg"], S["r"], "cuda", None, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"])
        val = train.make_val_fn(S["mech"], T_val, S["cfg"], S["r"], "cuda", S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"])
        T_score = env.score_window(T, int(S["cfg"]["L"])) if T > int(S["cfg"]["L"]) else T
        opt = cmaes.make_optimizer(policy.n_params(S["F"], S["hidden"]), lam=S["lam"], sigma0=S["sigma0"], seed=9000)
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for g in range(a.gens):
                sb = list(train.seed_block(0, g, S["n_S"]))

                def one():
                    X = opt.ask()
                    res = train.evaluate_generation(sim, X, sb, T_score)
                    opt.tell(X, res["f"])
                _, dt = _timed(one)
                times.append(dt)
                print(f"{name}: generation {g} ({'warm-up, not counted' if g == 0 else 'timed'}) {dt:.1f} s  = {1000 * dt / T:.3f} ms/step", flush=True)
            _, tv = _timed(lambda: val(np.array(opt.mean)))
            print(f"{name}: validation (T={T_val}, 32 seeds) {tv:.1f} s", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            doc["problems"].append(f"{name}: GPU out of memory ({str(e)[:120]})")
            print("FAIL:", doc["problems"][-1])
            break
        timed = times[1:] if len(times) > 1 else times
        ts = [1000.0 * x / T for x in timed]
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        doc["cells"][name] = dict(operator=S["operator"], knob=S["knob"], T=T, gen_s_warmup=times[0], gen_s_timed=timed, t_step_ms_timed=ts,
                                  t_step_ms_mean=float(np.mean(ts)), t_step_ms_p95=float(np.percentile(ts, 95)), t_val_s=tv, T_val=T_val,
                                  mem_peak_MB=peak)
        if peak / 1024 > 0.85 * free_gib:
            doc["problems"].append(f"{name}: peak GPU memory {peak:.0f} MB is close to the free memory ({free_gib:.1f} GiB)")
    if not doc["problems"] and not nonstd:
        doc["estimate"] = _estimate(pl, doc)
        e = doc["estimate"]
        print("\nestimate on THIS machine vs the local estimate (results/direction2/hours_estimate.json, same formula; start-up of %d s per segment not included):" % 60)
        for g in ("s1", "main"):
            print(f"  {g:5s} mean {e['target'][g]['mean_h']:6.1f} h (local {e['local'][g]['mean_h']:6.1f} h, x{e['ratio'][g]['mean']:.2f})   "
                  f"p95 {e['target'][g]['p95_h']:6.1f} h (local {e['local'][g]['p95_h']:6.1f} h, x{e['ratio'][g]['p95']:.2f})")
        print(f"  S1 segments: {e['n_segments_s1']} -> start-up +{e['startup_s1_h']:.1f} h;  main segments: {e['n_segments_main']} -> +{e['startup_main_h']:.1f} h")
        print(f"  S1 + main (p95 + start-up): {e['s1_plus_main_p95_incl_startup_h']:.0f} h = {e['s1_plus_main_p95_incl_startup_h'] / 24:.1f} days")
    _write(a.out, doc)
    return 1 if doc["problems"] else 0


def _estimate(pl, perf):
    import numpy as np
    import estimate_d2_hours as E
    import d2_run_queue as Q
    cells = perf["cells"]
    plain_name = next(n for n in cells if cells[n]["operator"] is None)
    pl_c = cells[plain_name]
    timing = dict(cells={n: dict(operator=c["operator"], t_step_ms_mean=c["t_step_ms_mean"], t_step_ms_p95=c["t_step_ms_p95"], t_val_s=c["t_val_s"])
                         for n, c in cells.items() if c["operator"] is not None})
    v2 = {"run_time_lam256_T2e4_300gen": {"nS256_mean": {"t_step_ms": pl_c["t_step_ms_mean"]}, "nS256_p95": {"t_step_ms": pl_c["t_step_ms_p95"]}},
          "validation": {"t_val_s": pl_c["t_val_s"]}}
    G_s = [0.01, 0.02]
    tgt = E.estimate(pl, timing, v2, G_s)["totals"]
    loc = json.load(open(os.path.join(ROOT, "results", "direction2", "hours_estimate.json")))["totals"]
    s1_jobs = Q.build_jobs(pl, "s1")
    main_jobs = Q.build_jobs(pl, "main")
    ratio = {g: dict(mean=tgt[g]["mean_h"] / loc[g]["mean_h"], p95=tgt[g]["p95_h"] / loc[g]["p95_h"]) for g in ("s1", "main")}
    return dict(target={g: tgt[g] for g in ("s1", "main", "s1_plus_main")}, local={g: loc[g] for g in ("s1", "main", "s1_plus_main")}, ratio=ratio,
                n_segments_s1=len(s1_jobs), n_segments_main=len(main_jobs), startup_s1_h=len(s1_jobs) * 60 / 3600, startup_main_h=len(main_jobs) * 60 / 3600,
                s1_plus_main_p95_incl_startup_h=tgt["s1_plus_main"]["p95_h"] + (len(s1_jobs) + len(main_jobs)) * 60 / 3600,
                max_segment_s_p95_at_target=max(j["est_p95_s"] for j in s1_jobs) * ratio["s1"]["p95"],
                note="Same formula as estimate_d2_hours.py with this machine's measured step times (plain M3C for main, D4/D2 max for S1 vulnerability cells); "
                     "the local numbers are from the RTX 5070 Ti. Rule decisions do not depend on the speed.")


def cmd_record(a):
    log_tail = ""
    if a.log and os.path.exists(a.log):
        log_tail = "\n".join(open(a.log, errors="replace").read().splitlines()[-12:])
    _write(os.path.join(a.dir, f"step_{a.name}.json"), dict(name=a.name, rc=a.rc, seconds=a.seconds, log=a.log, tail=log_tail, tests=a.tests))
    return 0


def cmd_verdict(a):
    def load(n):
        p = os.path.join(a.dir, n)
        return json.load(open(p)) if os.path.exists(p) else None
    env_, f64, perf = load("env.json"), load("f64.json"), load("perf.json")
    steps = {n: load(f"step_{n}.json") for n in ("cpu_tests", "gpu_tests")}
    why_no, notes = [], []
    if not a.cpu_only:
        if env_ is None or env_["problems"]:
            why_no += ["environment: " + "; ".join((env_ or {}).get("problems", ["not run"]))]
        if f64 is None or f64["problems"]:
            why_no += ["float64: " + "; ".join((f64 or {}).get("problems", ["not run"]))]
        if steps["gpu_tests"] is None or steps["gpu_tests"]["rc"] != 0:
            why_no += ["GPU env-vs-frontier bitwise tests failed or did not run (see log: %s)" % (steps["gpu_tests"] or {}).get("log")]
        if perf is None or perf["problems"]:
            why_no += ["performance run: " + "; ".join((perf or {}).get("problems", ["not run or skipped"]))]
        elif perf.get("nonstandard"):
            notes.append("performance run used non-standard sizes (--quick): the hour estimate is NOT valid")
        else:
            e = perf["estimate"]
            if e["s1_plus_main_p95_incl_startup_h"] > 14 * 24:
                notes.append(f"S1 + main would take about {e['s1_plus_main_p95_incl_startup_h'] / 24:.0f} days (p95): discuss with the supervisor before starting")
            if max(e["ratio"]["s1"]["p95"], e["ratio"]["main"]["p95"]) > 3:
                notes.append("this GPU is more than 3x slower than the local one (float64 on Turing); the schedule must be re-estimated")
    if steps["cpu_tests"] is None or steps["cpu_tests"]["rc"] != 0:
        why_no += ["CPU tests failed or did not run (see log: %s)" % (steps["cpu_tests"] or {}).get("log")]
    ok = not why_no
    verdict = ("可以開跑" if ok else "不可以開跑") + (" (CPU part only; GPU steps were skipped)" if a.cpu_only else "")
    doc = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), verdict=verdict, can_run=ok, cpu_only=a.cpu_only, reasons_no=why_no, notes=notes,
               env=env_, float64=f64, perf=perf, steps={k: v for k, v in steps.items() if v})
    _write(a.out, doc)
    print("\n" + "=" * 70 + f"\n結果：{verdict}")
    for r in why_no:
        print("  [不可以] " + r)
    for n in notes:
        print("  [注意]   " + n)
    if ok and not notes:
        print("  所有檢查通過。")
    print(f"詳細結果：{os.path.relpath(a.out, ROOT)}\n" + "=" * 70)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("env", "f64", "perf"):
        p = sub.add_parser(n)
        p.add_argument("--out", required=True)
        if n == "perf":
            p.add_argument("--cells", default=",".join(PERF_CELLS))
            p.add_argument("--gens", type=int, default=4, help="1 warm-up + 3 timed (default)")
            p.add_argument("--T", type=int, default=None, help="quick test only: smaller T_train (the estimate is then invalid)")
            p.add_argument("--T-val", dest="T_val", type=int, default=None)
    p = sub.add_parser("record")
    p.add_argument("--dir", required=True); p.add_argument("--name", required=True); p.add_argument("--rc", type=int, required=True)  # noqa: E702
    p.add_argument("--seconds", type=float, default=0); p.add_argument("--log", default=None); p.add_argument("--tests", default="")  # noqa: E702
    p = sub.add_parser("verdict")
    p.add_argument("--dir", required=True); p.add_argument("--out", default=OUT_DEFAULT); p.add_argument("--cpu-only", action="store_true")  # noqa: E702
    a = ap.parse_args()
    sys.exit({"env": cmd_env, "f64": cmd_f64, "perf": cmd_perf, "record": cmd_record, "verdict": cmd_verdict}[a.cmd](a))


if __name__ == "__main__":
    main()
