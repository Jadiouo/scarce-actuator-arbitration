#!/usr/bin/env python3
"""S1 final evaluation and assembly (spec REQ-MET-04/06/07, REQ-S1-09/10/11/17/18).  Details: docs/direction2-s1-eval-impl.md.  GPU work: submit with gpujob.

  scripts/d2_s1_eval.py run  [--cells A B ...] [--graph]     val stage then test stage for each cell (default: all S1 cells of the plan); resumable
  scripts/d2_s1_eval.py val  [--cells ...]                     validation stage only (frozen MDEs, selected handwritten strategy)
  scripts/d2_s1_eval.py test [--cells ...]                     test stage only (needs the val file; each (cell, policy) is evaluated once)
  scripts/d2_s1_eval.py assemble                               build + validate results/direction2/s1.json and the figure (CPU, seconds)
  scripts/d2_s1_eval.py status                                 which stages are done

Outputs: results/direction2/s1_eval/<cell>__{val,test_raw,test}.json, results/direction2/final_eval_ledger.json, results/direction2/s1.json.
--graph: CUDA-graph stepping for the RL batches (bitwise identical to eager).  --test-mode/--T/--root: unit-test and smoke use only (a root outside the repo).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arbitration.rl import plan as _plan, s1_eval  # noqa: E402


def cells_to_do(plan, requested):
    allc = s1_eval.expected_cells(plan)
    if not requested:
        return allc
    bad = [c for c in requested if c not in allc]
    if bad:
        raise SystemExit(f"not S1 cells of the plan: {bad} (S1 cells: {allc})")
    return list(requested)


def status(root, plan):
    rows = {c: {st: os.path.exists(s1_eval.eval_path(root, c, st)) for st in ("val", "test")} for c in s1_eval.expected_cells(plan)}
    print(json.dumps(rows, indent=1))
    return rows


def process_cell(root, plan, pc, chosen, stages, dev, graph, T, freeze_path, ledger_path, test_mode):
    from arbitration.rl import final_eval  # noqa: F401  (the only door to the test seeds)
    if not test_mode:
        _plan.check_conditional(pc, root)                         # s=0.01 cells only if M-3 allows them (REQ-S1-22)
    settings = _plan.cell_to_settings(pc)
    if T:
        settings["T_val"] = int(T)
    src = s1_eval.source_cell(pc["name"], chosen)
    ck = s1_eval.load_best_checkpoint(src, 0, pc["G_gens"], root, settings["F"], settings["hidden"])
    run = s1_eval.make_runner(settings, dev, graph)
    vp, tp = s1_eval.eval_path(root, pc["name"], "val"), s1_eval.eval_path(root, pc["name"], "test")
    t0 = time.time()
    if "val" in stages:
        if os.path.exists(vp):
            print(f"[{pc['name']}] val exists, skipped", flush=True)
        else:
            v = s1_eval.val_stage(pc, settings, ck, run, root=root, use_graph=graph is not None, src_cell=src, check_consistency=not T)
            s1_eval._write_json(vp, v)
            print(f"[{pc['name']}] val done in {time.time() - t0:.0f}s: ckpt gen {ck['gen']}, G_RL_val {v['G_RL_val']['mean']:.5f}, mde_G {v['mde_G']:.5f}"
                  + ("" if v["anchor"] else f", HW {v['hw']['selected']} {v['hw']['G_val']['mean']:.5f}, D_val {v['D_val']['mean']:+.5f}, mde_D {v['mde_D']:.5f}"), flush=True)
    if "test" in stages:
        if os.path.exists(tp):
            print(f"[{pc['name']}] test exists, skipped", flush=True)
        else:
            if not os.path.exists(vp):
                raise SystemExit(f"{pc['name']}: no validation record {vp}: run the val stage first (the frozen MDE comes from it)")
            t1 = time.time()
            t = s1_eval.test_stage(pc, settings, ck, json.load(open(vp)), run, root=root, use_graph=graph is not None, freeze_path=freeze_path,
                                   ledger_path=ledger_path)
            s1_eval._write_json(tp, t)
            print(f"[{pc['name']}] test done in {time.time() - t1:.0f}s: G_RL {t['G_RL']['mean']:.5f} [{t['G_RL']['lo']:.5f}, {t['G_RL']['hi']:.5f}]"
                  + (f", passed={t['passed']}" if t["anchor"] else f", D {t['D']['mean']:+.5f} [{t['D']['lo']:+.5f}, {t['D']['hi']:+.5f}] MDE_D {t['MDE_D']:.5f} detected={t['detected']}"),
                  flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("cmd", choices=["run", "val", "test", "assemble", "status"])
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph stepping for the RL batches (bitwise identical to eager)")
    ap.add_argument("--root", default=None, help="only with --test-mode: a root OUTSIDE the repository (smoke tests)")
    ap.add_argument("--test-mode", action="store_true", help="smoke/unit use: root outside the repo, no git checks, --T allowed")
    ap.add_argument("--T", type=int, default=None, help="only with --test-mode: shorter horizon (smoke test)")
    a = ap.parse_args(argv)
    repo = os.path.realpath(_plan.REPO_ROOT)
    root = os.path.realpath(a.root) if a.root else repo
    if a.test_mode:
        if root == repo or root.startswith(repo + os.sep + "results"):
            raise SystemExit("--test-mode needs --root outside the repository's results/ (nothing may be written into results/direction2/)")
    else:
        if root != repo or a.T:
            raise SystemExit("--root/--T are only allowed with --test-mode")
        try:
            _plan.verify_repo_state(root, freeze_commit=_plan.FREEZE_COMMIT)
        except RuntimeError as e:
            raise SystemExit(str(e))
    plan = _plan.load_plan(os.path.join(root, _plan.DEFAULT_PLAN), os.path.join(root, _plan.DEFAULT_FREEZE))
    if a.cmd == "status":
        status(root, plan)
        return 0
    if a.cmd == "assemble":
        doc = s1_eval.write_s1_json(root, plan)
        print(json.dumps(dict(gate=doc["gate"], verdict=doc["verdict"], not_completed=doc["not_completed"]), indent=1, default=float))
        return 0
    chosen = _plan.load_pilot_decision(root) if not a.test_mode else json.load(open(os.path.join(root, _plan.PILOT_DECISION)))["chosen_start"]
    if chosen not in ("cold", "warm"):
        raise SystemExit("the NAIVE pilot decision is not registered: S1 cannot be evaluated")
    graph = None
    if a.graph:
        from arbitration.rl import graphsim
        graph = graphsim.from_env(a.dev, force=True)
    stages = {"run": ("val", "test"), "val": ("val",), "test": ("test",)}[a.cmd]
    freeze = os.path.join(root, "docs", "direction2-freeze.md")
    ledger = os.path.join(root, "results", "direction2", "final_eval_ledger.json")
    for name in cells_to_do(plan, a.cells):
        process_cell(root, plan, _plan.get_cell(plan, name), chosen, stages, a.dev, graph, a.T, freeze, ledger, a.test_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
