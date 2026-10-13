#!/usr/bin/env python3
"""Main evaluation on the school GPU: val -> test -> assemble, resumable.

Run the production entry through scripts/d2_main_eval_launch.sh. The frozen plan,
all three runs, and the original final_eval test-seed gateway are mandatory.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import main_eval as ev, plan


def backup(paths, message):
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "direction2-freeze":
        raise RuntimeError("backup is restricted to direction2-freeze")
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-m", message, "--", *paths], cwd=ROOT, check=True)
    # A failed push stops at a saved stage; never repeat a test simulation to fix a push.
    subprocess.run(["git", "push", "origin", "direction2-freeze"], cwd=ROOT, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("stage", choices=["status", "val", "test", "assemble", "run"])
    ap.add_argument("--cells", nargs="+")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--backup", action="store_true", help="commit/push only completed evaluation artifacts")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    pl = plan.load_plan(str(ROOT / plan.DEFAULT_PLAN), str(ROOT / plan.DEFAULT_FREEZE))
    pcs = [c for c in pl["cells"] if c["group"] == "main"]
    if args.cells:
        if set(args.cells) - {c["name"] for c in pcs}:
            ap.error("--cells must name registered main cells")
        pcs = [c for c in pcs if c["name"] in args.cells]
    if args.stage == "status":
        print(json.dumps({c["name"]: {s: Path(ev.path(str(ROOT), c["name"], s)).exists() for s in ["val", "test"]} for c in pcs}, indent=2))
        return
    scratch = ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    with (scratch / "main-eval.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan.verify_repo_state(str(ROOT))
        for rel in ["arbitration/rl/main_eval.py", "arbitration/rl/final_eval.py", "scripts/d2_main_eval.py"]:
            if not plan.is_committed_and_clean(str(ROOT), rel):
                raise RuntimeError(f"evaluation code must be committed and clean: {rel}")
        graph = None
        if args.stage in ("val", "test", "run"):
            import torch
            torch.set_num_threads(1)
            if args.graph:
                from arbitration.rl import graphsim
                graph = graphsim.from_env("cuda", force=True)
        stages = ["val", "test"] if args.stage == "run" else [args.stage]
        for stage in stages:
            if stage not in ("val", "test"):
                continue
            for pc in pcs:
                st = plan.cell_to_settings(pc)
                cks = ev.checkpoints(str(ROOT), pc, st)
                run = ev.make_runner(st, "cuda", graph)
                if stage == "val":
                    doc = ev.validation_stage(str(ROOT), pc, st, cks, run, use_graph=args.graph)
                    print(f"{pc['name']}: validation frozen; main run={doc['selected_run']}; HW={doc['selected_hw']}", flush=True)
                else:
                    vp = ev.path(str(ROOT), pc["name"], "val")
                    if not os.path.exists(vp):
                        raise RuntimeError("validation selection must be frozen before test evaluation")
                    doc = ev.test_stage(str(ROOT), pc, st, cks, run, ev.read(vp), use_graph=args.graph)
                    print(f"{pc['name']}: test complete; k={doc['replication_k']}/3", flush=True)
                    for r in doc["runs"]:
                        print(f"  run{r['run_id']}: G={r['G_RL']['mean']:.6f}; D={r['D']['mean']:+.6f} "
                              f"[{r['D']['lo']:+.6f}, {r['D']['hi']:+.6f}]; MDE={r['MDE_D']:.6f}; found={r['found_D']}", flush=True)
                if args.backup:
                    paths = [str(p.relative_to(ROOT)) for p in (ROOT / ev.EVAL_DIR).glob(pc["name"] + "__*.json")]
                    if stage == "test":
                        paths.append("results/direction2/final_eval_ledger.json")
                    backup(paths, f"Direction 2: {pc['name']} formal evaluation {stage}")
        if args.stage in ("assemble", "run"):
            doc = ev.assemble(str(ROOT), pl)
            print(json.dumps(doc["verdict"], ensure_ascii=False, indent=2))
            if args.backup:
                backup([ev.MAIN_JSON], "Direction 2: paired test confidence intervals and replication verdict")


if __name__ == "__main__":
    main()
