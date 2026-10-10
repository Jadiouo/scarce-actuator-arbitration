#!/usr/bin/env python3
"""POST-HOC, exploratory diagnostic for Direction 2 S1 (does the RL policy exploit the vulnerability, or just beat HW on the base mechanism?).

Pre-registered (before running) in docs/direction2-notes.md, section "2026-10-10 S1 正式結果與 D2 混淆；事後診斷（執行前登記）".
For the 10 vulnerability cells of results/direction2/s1.json: selected checkpoint (checkpoint_gen), T=100000, scored to T-L, evaluated in
(a) the cell's vulnerability env and (b) the same operator at off_knob (identity = base M3C).  Seeds: val (2000-2031) and val2 (3000-3031),
reported separately; NEVER the test split.  E = G_RL^V - G_RL^off (paired per seed, t31 CI); G_HW^off = best of the 36 S_HW (selected on the
SAME seed set, hence optimistic for HW); D^off = G_RL^off - G_HW^off.  Writes results/direction2/s1_diag_offknob.json; touches no frozen file.
GPU work: run on the school machine (never the local GPU).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arbitration.rl import hw, metrics, plan as _plan, s1_eval, seeds as _seeds, vulns  # noqa: E402

OUT = os.path.join("results", "direction2", "s1_diag_offknob.json")
NOTES = "docs/direction2-notes.md#2026-10-10 S1 正式結果與 D2 混淆；事後診斷（執行前登記）"
SPLITS = ("val", "val2")
MATCH_ATOL = 1e-9


def split_seeds(name):
    if name not in SPLITS:
        raise ValueError(f"split {name!r} not allowed: the diagnostic uses only {SPLITS}")
    s = list(_seeds.splits()[name])
    _seeds.require_split(s, name)
    return s


def paired_E(g_v, g_off):
    """E = G_RL^V - G_RL^off per seed, with the project's t31 CI."""
    return metrics.mean_ci(np.asarray(g_v, float) - np.asarray(g_off, float), "t31")


def best_hw(run, seeds, s_hw):
    """Best of the S_HW 36 on `seeds` (argmax of mean G, ties -> earlier, same rule as s1_eval.select_on_validation) with its per-seed G."""
    g = s1_eval.hw_gains(run, seeds, s_hw)
    names = [n for n, _ in s_hw]
    sel = s1_eval.select_on_validation(g["G_by_name"], names)["selected"]
    return sel, np.asarray(g["G_by_name"][sel], float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, OUT))
    ap.add_argument("--T", type=int, default=None, help="smoke tests only (disables the sanity assert and refuses the default --out)")
    a = ap.parse_args(argv)
    if a.T and os.path.realpath(a.out) == os.path.realpath(os.path.join(ROOT, OUT)):
        raise SystemExit("--T needs a different --out")
    root = ROOT
    plan = _plan.load_plan(os.path.join(root, _plan.DEFAULT_PLAN), os.path.join(root, _plan.DEFAULT_FREEZE))
    s1doc = json.load(open(os.path.join(root, s1_eval.S1_JSON)))
    runs = [r for r in s1doc["runs"] if r["category"] != "anchor"]
    if a.cells:
        runs = [r for r in runs if r["cell"] in a.cells]
    graph = None
    if a.graph:
        from arbitration.rl import graphsim
        graph = graphsim.from_env(a.dev, force=True)
    s_hw = hw.build_s_hw()
    seedsets = {sp: split_seeds(sp) for sp in SPLITS}
    t0 = time.time()
    hw_off_cache = {}                                     # (operator, split) -> (name, per-seed G): off = base M3C, identical across cells
    cells = []
    for r in runs:
        pc = _plan.get_cell(plan, r["cell"])
        st_v = _plan.cell_to_settings(pc)
        if a.T:
            st_v["T_val"] = int(a.T)
        op = st_v["operator"]
        st_o = dict(st_v, knob=vulns.off_knob(op))
        ck = s1_eval.load_best_checkpoint(r["cell"], 0, pc["G_gens"], root, st_v["F"], st_v["hidden"])
        if ck["gen"] != r["checkpoint_gen"]:
            raise RuntimeError(f"{r['cell']}: checkpoint gen {ck['gen']} != s1.json checkpoint_gen {r['checkpoint_gen']}")
        run_v, run_o = (s1_eval.make_runner(s, a.dev, graph) for s in (st_v, st_o))
        c = dict(cell=r["cell"], id=r["id"], size=r["size"], category=r["category"], operator=op, knob=st_v["knob"], off_knob=st_o["knob"],
                 checkpoint_gen=ck["gen"], theta_sha256=ck["theta_sha256"], splits={})
        for sp in SPLITS:
            sd = seedsets[sp]
            gv = s1_eval.rl_gains(run_v, ck["theta"], sd, st_v["F"], st_v["hidden"], graph is not None)
            go = s1_eval.rl_gains(run_o, ck["theta"], sd, st_o["F"], st_o["hidden"], graph is not None)
            key = (str(op), sp)
            if key not in hw_off_cache:
                hw_off_cache[key] = best_hw(run_o, sd, s_hw)
            hname, hg = hw_off_cache[key]
            d = dict(seeds=[sd[0], sd[-1]], T_score=gv["T_score"], G_RL_V=metrics.mean_ci(gv["G_s"], "t31"), G_RL_off=metrics.mean_ci(go["G_s"], "t31"),
                     E=paired_E(gv["G_s"], go["G_s"]), G_HW_off=dict(name=hname, selected_on="same seeds as evaluated (optimistic for HW)", **metrics.mean_ci(hg, "t31")),
                     D_off=metrics.mean_ci(go["G_s"] - hg, "t31"), E_ci_lower_gt_0=bool(paired_E(gv["G_s"], go["G_s"])["lo"] > 0))
            if sp == "val":                               # sanity: (a) must reproduce s1_eval's val numbers
                ev = json.load(open(s1_eval.eval_path(root, r["cell"], "val")))
                diff = abs(d["G_RL_V"]["mean"] - ev["G_RL_val"]["mean"])
                d["sanity_vs_s1_eval_val"] = dict(s1_eval=ev["G_RL_val"]["mean"], absdiff=diff, match=bool(diff <= MATCH_ATOL))
                if not a.T and diff > MATCH_ATOL:
                    raise RuntimeError(f"{r['cell']}: G_RL^V on val {d['G_RL_V']['mean']:.12f} != s1_eval {ev['G_RL_val']['mean']:.12f}")
            c["splits"][sp] = d
            print(f"[{r['cell']}|{sp}] E {d['E']['mean']:+.5f} [{d['E']['lo']:+.5f},{d['E']['hi']:+.5f}] G_RL_off {d['G_RL_off']['mean']:.5f} "
                  f"G_HW_off {d['G_HW_off']['mean']:.5f} ({hname}) D_off {d['D_off']['mean']:+.5f}  t={time.time() - t0:.0f}s", flush=True)
        cells.append(c)
    doc = dict(schema="direction2-s1-diag-offknob/1", post_hoc=True, exploratory=True, notes=NOTES, T=int(a.T or s1_eval.T_EVAL),
               rule="E CI lower bound > 0 is the only pre-registered evidence of exploitation", test_seeds_used=False, cells=cells,
               wall_s=time.time() - t0, git_sha=s1_eval._git_sha(root))
    s1_eval._write_json(a.out, doc)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
