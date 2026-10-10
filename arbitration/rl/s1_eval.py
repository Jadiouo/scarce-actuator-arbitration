"""S1 final evaluation and assembly (REQ-MET-04/06/07, REQ-S1-09/10/11/15/16/17/18, REQ-STOP-01/03).  See docs/direction2-s1-eval-impl.md.

Two stages per run (cell), both resumable and written to results/direction2/s1_eval/:
  val  : VALIDATION seeds only.  Best checkpoint of the run (largest G_val, REQ-OPT-06), its per-seed validation G, the validation-selected
         best handwritten strategy of the cell's own environment (S_HW 36, REQ-HW-02), D_s on validation, and the FROZEN MDEs (REQ-MET-06:
         mde_frozen = MDE_G from G_s, mde_d_frozen = MDE_D from D_s).  Written BEFORE any test evaluation.
  test : the test seeds, reached ONLY through final_eval.evaluate (this module never names or builds a test seed, REQ-SEED-06); the frozen
         MDEs are read from the val file, never recomputed.  Each (cell, policy) is evaluated once (final_eval ledger).
`assemble` builds results/direction2/s1.json (validate_s1_output) and the J reading (stop.classify).
"""
import glob
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import final_eval, hw, metrics, parts, policy, s1, seeds as _seeds, stop, vulns
from . import plan as _plan

SCHEMA = "direction2-s1-eval/1"
EVAL_DIR = os.path.join("results", "direction2", "s1_eval")
S1_JSON = os.path.join("results", "direction2", "s1.json")
FIGURE = os.path.join("results", "direction2", "s1_power_curve.png")
REG_PATH = os.path.join("results", "direction2", "vuln_suite.json")
T_EVAL = 100000                                            # REQ-OPT-06: validation / evaluation horizon (plan T_val)
HONEST_RTOL = 1e-12                                        # the honest baselines of the two batches must agree this closely
TRAIN_VAL_RTOL = 1e-6                                      # recomputed validation G vs the G_val stored by the training run
_ANCHOR_REF_FILES = {"NAIVE": ("results/direction2/pilot_decision.json", "G_star_ref"),
                     "M4rw0": ("results/direction2/measure_m4.json", "G_star_ref")}


class IncompleteRun(RuntimeError):
    """The run's training is not finished (or its files are inconsistent): it cannot be evaluated."""


# ---------------------------------------------------------------------------------------------------------------- io helpers
def _write_json(path: str, doc: Any) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".s1e-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, default=float)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def eval_path(root: str, cell: str, stage: str) -> str:
    """results/direction2/s1_eval/<cell>__<stage>.json with stage in {val, test, test_raw}."""
    return os.path.join(root, EVAL_DIR, f"{cell}__{stage}.json")


def _git_sha(root: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _ci(x: Sequence[float]) -> Dict[str, float]:
    return metrics.mean_ci(x, "t31")


# ---------------------------------------------------------------------------------------------------------------- checkpoint
def source_cell(cell_name: str, chosen_start: Optional[str]) -> str:
    """The cell whose training result is evaluated: the NAIVE anchor reuses the pilot of the chosen route (FREEZE.md), the others themselves."""
    routes = _plan.REUSED_CELLS.get(cell_name)
    if routes is None:
        return cell_name
    if chosen_start not in routes:
        raise ValueError(f"cell {cell_name} reuses a pilot result; the registered start route {chosen_start!r} is needed")
    return routes[chosen_start]


def load_best_checkpoint(cell: str, run_id: int, G_gens: int, root: str, F: int = 25, hidden: int = 2) -> Dict[str, Any]:
    """REQ-OPT-06: the run's final policy = the checkpoint with the largest validation G_val (earliest on ties), chosen from ALL validation
    records of the run (never by training fitness).  Cross-checks: the checkpoint file's gens == G_gens, its stored `best` equals the argmax
    of the records, and the last part file (status done) agrees.  IncompleteRun otherwise."""
    from . import train
    ck = parts.ckpt_path(cell, run_id, root)
    if not os.path.exists(ck):
        raise IncompleteRun(f"{cell}: no checkpoint {ck}")
    with np.load(ck, allow_pickle=False) as z:
        st = json.loads(str(z["state"]))
    if int(st["gen"]) != int(G_gens):
        raise IncompleteRun(f"{cell}: checkpoint at generation {st['gen']}, plan needs {G_gens}")
    recs = st["records"]
    if not recs:
        raise IncompleteRun(f"{cell}: no validation records")
    sel = train.select_checkpoint(recs)
    best = st["best"]
    if best is None or int(best["gen"]) != int(sel["gen"]) or float(best["G_val"]) != float(sel["G_val"]):
        raise IncompleteRun(f"{cell}: checkpoint 'best' (gen {None if best is None else best['gen']}) is not the validation argmax (gen {sel['gen']})")
    theta = np.asarray(best["theta"], dtype=float)
    if theta.size != policy.n_params(F, hidden):
        raise IncompleteRun(f"{cell}: theta has {theta.size} entries, expected {policy.n_params(F, hidden)}")
    pf = sorted(glob.glob(os.path.join(root, "results", "direction2", "parts", f"{cell}__run{run_id}__g*-*.json")))
    if not pf:
        raise IncompleteRun(f"{cell}: no part files")
    last = parts.read_part(pf[-1])
    if last.get("status") != "done" or int(last["gen1"]) != int(G_gens):
        raise IncompleteRun(f"{cell}: last part {os.path.basename(pf[-1])} is not the finished final segment")
    if last["best"]["gen"] != best["gen"] or last["best"]["G_val"] != best["G_val"]:
        raise IncompleteRun(f"{cell}: last part's best differs from the checkpoint's best")
    return dict(cell=cell, run_id=run_id, gen=int(best["gen"]), G_val_train=float(best["G_val"]), theta=theta, n_val_records=len(recs),
                theta_sha256=hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest(), gens_done=int(st["gen"]))


# ---------------------------------------------------------------------------------------------------------------- simulator glue
Runner = Callable[..., Tuple[np.ndarray, int]]


def make_runner(settings: Dict[str, Any], dev: str, graph: Any = None, T: Optional[int] = None) -> Runner:
    """run(pols, seeds, use_graph=False) -> (U[P,S] = util_b[:,:,1] at the score snapshot, T_score), in the cell's own environment (vulnerability
    operator/knob, M4, NAIVE p=0), exactly as the training validation (train.make_val_fn: vulns.simulate_vuln, public observation).
    The graph is only used for MLP-only batches when use_graph=True (bitwise identical to eager, see tests)."""
    T = int(settings["T_val"] if T is None else T)

    def run(pols, seeds, use_graph: bool = False):
        out = vulns.simulate_vuln(settings["operator"], settings["knob"], settings["mech"], T, settings["cfg"], pols, seeds, r=settings["r"],
                                  dev=dev, graph=graph if use_graph else None)
        return out["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float), int(out["T_score"])
    return run


def _check_honest(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.max(np.abs(a - b)))
    if d > HONEST_RTOL * max(1.0, float(np.max(np.abs(a)))):
        raise RuntimeError(f"the honest baselines of the RL batch and of the handwritten batch differ by {d:.3e}: paired D_s would be wrong")
    return d


def rl_gains(run: Runner, theta: np.ndarray, seeds: Sequence[int], F: int = 25, hidden: int = 2, use_graph: bool = False) -> Dict[str, Any]:
    """Per-seed G of the MLP policy vs the all-zero MLP (honest), REQ-MET-01 (same convention as train.make_val_fn)."""
    pols = [policy.MLPPolicy(F=F, hidden=hidden, theta=np.zeros_like(theta), obs_version="public"),
            policy.MLPPolicy(F=F, hidden=hidden, theta=theta, obs_version="public")]
    U, Ts = run(pols, list(seeds), use_graph)
    return dict(G_s=(U[1] - U[0]) / Ts, U_hon=U[0], T_score=Ts)


def hw_gains(run: Runner, seeds: Sequence[int], strategies: Sequence[Tuple[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Per-seed G of each handwritten strategy vs the honest adapter, in the cell's environment (REQ-HW-03)."""
    pols = [policy.AdapterPolicy({})] + [policy.AdapterPolicy(dict(d)) for _, d in strategies]
    U, Ts = run(pols, list(seeds), False)
    return dict(G_by_name={n: (U[i + 1] - U[0]) / Ts for i, (n, _) in enumerate(strategies)}, U_hon=U[0], T_score=Ts)


def select_on_validation(val_by_name: Dict[str, Sequence[float]], names: Sequence[str]) -> Dict[str, str]:
    """REQ-HW-02: argmax of the mean VALIDATION G over `names` (the 36; ties -> earlier name); `plus_h0` re-selects over names + ['honest']
    (H0 has zero gain and is appended last, so the 36 win ties).  Agrees with hw.select_handwritten (tested)."""
    names = list(names)
    if len(names) != hw.N_FINE or len(set(names)) != hw.N_FINE:
        raise ValueError(f"REQ-HW-02: the candidate set must be the {hw.N_FINE} distinct S_HW strategies")

    def pick(cands, vals):
        best, bm = None, -math.inf
        for n in cands:
            m = float(np.mean(vals[n]))
            if best is None or m > bm:
                best, bm = n, m
        return best
    vals0 = dict(val_by_name, honest=np.zeros(len(next(iter(val_by_name.values())))))
    return dict(selected=pick(names, val_by_name), plus_h0=pick(names + ["honest"], vals0))


def anchor_strategy(anchor_id: str) -> Tuple[str, Dict[str, float]]:
    """REQ-S1-08/09: the anchor reference strategy (NAIVE: dz=2.0; M4: b=0.3), taken from the direction-1 strategy table."""
    from . import env
    name = env.anchor_config("NAIVE" if anchor_id == "NAIVE" else "M4")["strategy"]
    table = dict(hw.all_strategies())
    if name not in table:
        raise ValueError(f"anchor strategy {name!r} is not a direction-1 strategy")
    return name, dict(table[name])


def frozen_anchor_ref(anchor_id: str, root: str) -> Dict[str, float]:
    """The W1 / pilot validation re-measurement of G*_ref (REQ-S1-08/09): NAIVE from pilot_decision.json, M4 from measure_m4.json."""
    rel, key = _ANCHOR_REF_FILES[anchor_id]
    return dict(_read_json(os.path.join(root, rel))[key])


# ---------------------------------------------------------------------------------------------------------------- val stage
def val_stage(pc: Dict[str, Any], settings: Dict[str, Any], ck: Dict[str, Any], run: Runner, *, root: str, use_graph: bool = False,
              src_cell: Optional[str] = None, check_consistency: bool = True) -> Dict[str, Any]:
    """Validation seeds ONLY.  Returns (and the caller writes) the val record containing the frozen MDEs (REQ-MET-06).
    check_consistency=False (unit tests with a tiny horizon only) skips the two comparisons with values measured at T=1e5: the run's stored
    G_val and the frozen G*_ref of the anchors."""
    t0 = time.time()
    vs = list(_seeds.splits()["val"])
    _seeds.require_split(vs, "val")
    F, hidden = settings["F"], settings["hidden"]
    rl = rl_gains(run, ck["theta"], vs, F, hidden, use_graph)
    G_val = rl["G_s"]
    diff = abs(float(G_val.mean()) - ck["G_val_train"])
    if check_consistency and diff > TRAIN_VAL_RTOL * max(1e-3, abs(ck["G_val_train"])):
        raise RuntimeError(f"{pc['name']}: recomputed validation G {G_val.mean():.9f} != the run's stored G_val {ck['G_val_train']:.9f}: "
                           f"the evaluation environment differs from the training one")
    anchor = pc["category"] == "anchor"
    rec: Dict[str, Any] = dict(
        schema=SCHEMA, stage="val", cell=pc["name"], source_cell=src_cell or pc["name"], vuln_id=pc["vuln_id"], category=pc["category"],
        size=pc["size"], anchor=anchor, T=int(settings["T_val"]), T_score=rl["T_score"],
        val_seeds=[vs[0], vs[-1]], checkpoint=dict(gen=ck["gen"], G_val_train=ck["G_val_train"], theta_sha256=ck["theta_sha256"],
                                                    n_val_records=ck["n_val_records"], run_id=ck["run_id"]),
        G_val_recomputed=float(G_val.mean()), G_val_recompute_absdiff=diff, G_RL_val=_ci(G_val),
        mde_G=metrics.mde(G_val, vs), git_sha=_git_sha(root))
    if anchor:
        name, scols = anchor_strategy(pc["vuln_id"])
        ref = hw_gains(run, vs, [(name, scols)])
        _check_honest(rl["U_hon"], ref["U_hon"])
        recheck = _ci(ref["G_by_name"][name])
        frozen = frozen_anchor_ref(pc["vuln_id"], root)
        if abs(recheck["mean"] - frozen["mean"]) > 1e-6 * max(1e-3, abs(frozen["mean"])) and check_consistency:
            raise RuntimeError(f"{pc['name']}: G*_ref recomputed ({recheck['mean']:.9f}) != the frozen W1/pilot value ({frozen['mean']:.9f})")
        rec.update(G_star_ref=frozen, G_star_ref_strategy=name, G_star_ref_recheck=recheck,
                   informational=(pc["vuln_id"] == "M4rw0" and s1.m4_anchor_status(frozen["lo"]) == "informational"))
    else:
        s_hw = hw.build_s_hw()
        names = [n for n, _ in s_hw]
        g = hw_gains(run, vs, s_hw)
        rec["honest_batches_maxdiff"] = _check_honest(rl["U_hon"], g["U_hon"])
        sel = select_on_validation(g["G_by_name"], names)
        zero = np.zeros(len(vs))
        hw_s = g["G_by_name"][sel["selected"]]
        hwh_s = zero if sel["plus_h0"] == "honest" else g["G_by_name"][sel["plus_h0"]]
        D_val, Dh_val = G_val - hw_s, G_val - hwh_s
        rec.update(hw=dict(selected=sel["selected"], G_val=_ci(hw_s), val_means={n: float(np.mean(g["G_by_name"][n])) for n in names},
                           plus_h0=dict(selected=sel["plus_h0"], G_val=_ci(hwh_s))),
                   D_val=_ci(D_val), mde_D=metrics.mde(D_val, vs), D_val_plus_h0=_ci(Dh_val), mde_D_plus_h0=metrics.mde(Dh_val, vs))
    rec["mde_frozen"] = rec["mde_G"]                                          # REQ-MET-06: the field the test evaluation reads
    if not anchor:
        rec["mde_d_frozen"] = rec["mde_D"]
    rec["wall_s"] = time.time() - t0
    return rec


# ---------------------------------------------------------------------------------------------------------------- test stage
def _test_policy_ids(anchor: bool) -> List[str]:
    return ["rl"] if anchor else ["rl", "hw", "hw_plus_h0"]


def test_stage(pc: Dict[str, Any], settings: Dict[str, Any], ck: Dict[str, Any], val: Dict[str, Any], run: Runner, *, root: str,
               use_graph: bool = False, freeze_path: Optional[str] = None, ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """Test evaluation of one run, only through final_eval.evaluate.  Reads the frozen MDEs from the val record (never recomputed); checks that
    the val record belongs to this cell and this exact checkpoint.  Raw per-seed gains are saved before the ledger can lock them."""
    if val.get("stage") != "val" or val["cell"] != pc["name"] or val["checkpoint"]["theta_sha256"] != ck["theta_sha256"] \
            or val["checkpoint"]["gen"] != ck["gen"]:
        raise ValueError(f"{pc['name']}: the validation record does not belong to this cell/checkpoint")
    for k in ("mde_frozen",) + (() if val["anchor"] else ("mde_d_frozen",)):
        if k not in val:
            raise ValueError(f"{pc['name']}: validation record lacks the frozen {k} (REQ-MET-06)")
    anchor = val["anchor"]
    raw_path = eval_path(root, pc["name"], "test_raw")
    cache: Dict[str, Any] = {}

    def compute(seeds_):
        if "raw" in cache:
            return cache["raw"]
        if os.path.exists(raw_path):
            cache["raw"] = _read_json(raw_path)
            return cache["raw"]
        rl = rl_gains(run, ck["theta"], seeds_, settings["F"], settings["hidden"], use_graph)
        raw: Dict[str, Any] = dict(cell=pc["name"], seeds=[int(s) for s in seeds_], T_score=rl["T_score"], rl=rl["G_s"].tolist())
        if not anchor:
            s_hw = dict(hw.build_s_hw())
            sel, selh = val["hw"]["selected"], val["hw"]["plus_h0"]["selected"]
            want = [(sel, s_hw[sel])] + ([] if selh in ("honest", sel) else [(selh, s_hw[selh])])
            g = hw_gains(run, seeds_, want)
            raw["honest_batches_maxdiff"] = _check_honest(rl["U_hon"], g["U_hon"])
            raw["hw"] = g["G_by_name"][sel].tolist()
            raw["hw_plus_h0"] = (np.zeros(len(seeds_)) if selh == "honest" else g["G_by_name"][selh]).tolist()
        _write_json(raw_path, raw)                                              # before final_eval's ledger locks the (cell, policy) pairs
        cache["raw"] = raw
        return raw

    def evaluator_for(pid):
        def ev(seeds_):
            raw = compute(seeds_)
            rl_s = np.asarray(raw["rl"])
            if pid == "rl":
                return dict(G_s=rl_s, D_s=None if anchor else rl_s - np.asarray(raw["hw"]))
            if pid == "hw":
                return dict(G_s=np.asarray(raw["hw"]))
            return dict(G_s=np.asarray(raw["hw_plus_h0"]), D_s=rl_s - np.asarray(raw["hw_plus_h0"]))
        return ev

    outs: Dict[str, Any] = {}
    for pid in _test_policy_ids(anchor):
        rec = dict(cell=pc["name"], policy_id=pid, mde_frozen=val["mde_frozen"])
        if not anchor:
            rec["mde_d_frozen"] = val["mde_d_frozen"]
        if final_eval.already_evaluated(pc["name"], pid, ledger_path):
            if not os.path.exists(raw_path):
                raise RuntimeError(f"{pc['name']}|{pid} is in the test ledger but the raw per-seed file {raw_path} is missing: cannot rebuild")
            raw = _read_json(raw_path)                                           # resume after a crash: rebuild from the saved raw gains
            rl_s = np.asarray(raw["rl"])
            ev = evaluator_for(pid)
            res = ev(raw["seeds"])
            outs[pid] = dict(metrics.final_eval(rec, res["G_s"], res.get("D_s")), cell=pc["name"], policy_id=pid, rebuilt_from_raw=True)
        else:
            outs[pid] = final_eval.evaluate(rec, evaluator_for(pid), freeze_path=freeze_path, ledger_path=ledger_path)
    raw = _read_json(raw_path)
    o = outs["rl"]
    res: Dict[str, Any] = dict(
        schema=SCHEMA, stage="test", cell=pc["name"], source_cell=val["source_cell"], vuln_id=val["vuln_id"], category=val["category"], size=val["size"],
        anchor=anchor, T_score=raw["T_score"], test_seed_range=[min(raw["seeds"]), max(raw["seeds"])], checkpoint=val["checkpoint"], G_RL=o["G"], found_G=o["found_G"],
        mde_G=val["mde_frozen"], git_sha=_git_sha(root))
    if anchor:
        res.update(G_star_ref=val["G_star_ref"], informational=val["informational"],
                   passed=s1.anchor_passes(o["G"]["mean"], o["G"]["lo"], val["G_star_ref"]["mean"], val["mde_frozen"]))
    else:
        res.update(MDE_D=val["mde_d_frozen"], D=o["D"], detected=o["found_D"], hw_selected=val["hw"]["selected"], G_HW=outs["hw"]["G"],
                   D_plus_h0=outs["hw_plus_h0"]["D"], hw_plus_h0_selected=val["hw"]["plus_h0"]["selected"], MDE_D_plus_h0=val["mde_D_plus_h0"],
                   detected_plus_h0=outs["hw_plus_h0"]["found_D"] if "found_D" in outs["hw_plus_h0"] else None,
                   honest_batches_maxdiff=raw.get("honest_batches_maxdiff"))
    return res


# ---------------------------------------------------------------------------------------------------------------- assemble
def _num(ci: Dict[str, float]) -> Dict[str, float]:
    return dict(mean=float(ci["mean"]), lo=float(ci["lo"]), hi=float(ci["hi"]), half=float(ci["half"]))


def expected_cells(plan: Dict[str, Any]) -> List[str]:
    """The S1 cells of the frozen plan (group 's1', REQ-S1-10) in plan order."""
    return [c["name"] for c in sorted(plan["cells"], key=lambda c: c["order"]) if c["group"] == "s1"]


def assemble(root: str, plan: Dict[str, Any], figure_fn: Optional[Callable[[Dict[str, Any], str], None]] = None) -> Dict[str, Any]:
    """Build s1.json from the per-cell test files; unfinished cells count as undetected / not passed (REQ-S1-10, STOP-01).  The caller validates and writes."""
    reg = {(e["id"], e.get("size")): e for e in _read_json(os.path.join(root, REG_PATH))["entries"]}
    tests: Dict[str, Dict[str, Any]] = {}
    for name in expected_cells(plan):
        p = eval_path(root, name, "test")
        if os.path.exists(p):
            tests[name] = _read_json(p)
    runs, missing = [], []
    for name in expected_cells(plan):
        t = tests.get(name)
        if t is None:
            missing.append(name)
            continue
        if t["anchor"]:
            runs.append(dict(id=t["vuln_id"], kind="anchor", category="anchor", size=None, cell=name, G_RL=_num(t["G_RL"]),
                             G_star_ref=float(t["G_star_ref"]["mean"]), G_star_ref_ci=_num(t["G_star_ref"]), MDE_G=float(t["mde_G"]),
                             passed=bool(t["passed"]), informational=bool(t["informational"]), source_cell=t["source_cell"]))
        else:
            e = reg[(t["vuln_id"], t["size"])]
            runs.append(dict(id=t["vuln_id"], kind=e["kind"], category=t["category"], size=t["size"], cell=name, G_RL=_num(t["G_RL"]),
                             D=_num(t["D"]), MDE_D=float(t["MDE_D"]), detected=bool(t["detected"]), G_HW=dict(name=t["hw_selected"], **_num(t["G_HW"])),
                             D_plus_h0=_num(t["D_plus_h0"]), hw_plus_h0_selected=t["hw_plus_h0_selected"], mde_G=float(t["mde_G"]),
                             checkpoint_gen=t["checkpoint"]["gen"]))
    plan_sizes = {c["size"] for c in plan["cells"] if c["group"] == "s1" and c["category"] == "out_of_family"}   # unfinished = undetected, denominator fixed
    out_sizes = sorted({r["size"] for r in runs if r["category"] == "out_of_family"})
    summary = []
    for s in out_sizes:
        x = sum(1 for r in runs if r["category"] == "out_of_family" and r["size"] == s and r["detected"])
        summary.append(dict(scope="out_of_family", **s1.summary_table({s: x}, 3)[0]))
    x6 = sum(1 for r in runs if r["category"] in ("out_of_family", "in_family") and r["size"] == 0.02 and r["detected"])
    if any(r["category"] in ("out_of_family", "in_family") and r["size"] == 0.02 for r in runs):
        summary.append(dict(scope="in_class", **s1.summary_table({0.02: x6}, 6)[0]))
    anc = {r["id"]: r for r in runs if r["category"] == "anchor"}
    naive, m4 = anc.get("NAIVE"), anc.get("M4rw0")
    m4_info = bool(m4["informational"]) if m4 else False
    det02 = {r["id"]: r["detected"] for r in runs if r["category"] == "out_of_family" and r["size"] == 0.02}
    gate = s1.s1_gate_by_id(None if naive is None else naive["passed"], None if m4 is None else m4["passed"], m4_info, det02)
    d3e = reg[("D3", None)]
    doc: Dict[str, Any] = dict(
        schema="direction2-s1/1", spec="docs/direction2-spec.md v1.3", runs=runs, summary=summary, figure=FIGURE,
        d3=dict(delta_fine=float(d3e["delta_fine"]["mean"]), delta_coarse=float(d3e["delta_coarse"]["mean"]), delta_fine_ci=_num(d3e["delta_fine"]),
                delta_coarse_ci=_num(d3e["delta_coarse"]), G_HW_best_val=d3e["G_HW_best_val"], G_star_val=_num(d3e["G_star_val"]), category=d3e["category"],
                note="D3 is registered, not run (REQ-S1-10)"),
        gate=dict(S1_pass=bool(gate), naive_pass=None if naive is None else naive["passed"], m4_pass=None if m4 is None else m4["passed"],
                  m4_informational=m4_info, out_of_family_s002=det02, y=sum(1 for v in det02.values() if v), n=3,
                  y_s001=sum(1 for r in runs if r["category"] == "out_of_family" and r["size"] == 0.01 and r["detected"]) if 0.01 in plan_sizes else None),
        not_completed=missing, git_sha=_git_sha(root))
    oc = stop.classify(bool(gate), None, None, None, None, 0)       # main experiment (r=0.5, 3 runs) is not part of S1: D does not exist (note N1)
    doc["verdict"] = dict(row=oc.row, label=oc.label, incomplete=oc.incomplete, note="D from the main experiment is absent; J-row per REQ-STOP-03/N1")
    if figure_fn is not None:
        figure_fn(doc, os.path.join(root, FIGURE))
    return doc


def power_curve_figure(doc: Dict[str, Any], path: str) -> None:
    """REQ-S1-18: detection-power curve.  x = vulnerability size, y = D (test, CI); out-of-family at both sizes (lines), the other runs as single
    points; the run's own MDE_D is drawn as a short bar at its size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cols = {"D1": "#1f77b4", "D2": "#d95f02", "O1": "#1b9e77", "O2": "#7570b3", "O3": "#e7298a", "D4": "#66a61e", "O5": "#666666"}
    ax.axhline(0, color="#999", lw=0.8)
    for vid in ("D1", "D2", "O1"):
        rr = sorted((r for r in doc["runs"] if r["id"] == vid), key=lambda r: r["size"])
        if rr:
            ax.errorbar([r["size"] for r in rr], [r["D"]["mean"] for r in rr], yerr=[[r["D"]["mean"] - r["D"]["lo"] for r in rr], [r["D"]["hi"] - r["D"]["mean"] for r in rr]],
                        marker="o", capsize=3, color=cols[vid], label=f"{vid} (out-of-family)")
    for vid in ("O2", "O3", "D4", "O5"):
        for r in (r for r in doc["runs"] if r["id"] == vid):
            ax.errorbar([r["size"]], [r["D"]["mean"]], yerr=[[r["D"]["mean"] - r["D"]["lo"]], [r["D"]["hi"] - r["D"]["mean"]]], marker="s", capsize=3,
                        color=cols[vid], ls="none", label=f"{vid} ({r['category']})")
    for r in doc["runs"]:
        if "MDE_D" in r:
            ax.plot([r["size"] - 0.0006, r["size"] + 0.0006], [r["MDE_D"]] * 2, color="#c00", lw=2)
    ax.plot([], [], color="#c00", lw=2, label="MDE_D of the run")
    ax.set_xlabel("vulnerability size s (Delta over best handwritten)")
    ax.set_ylabel("D = G_RL - G_HW (test)")
    ax.set_xticks([0.01, 0.02])
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("S1 detection power")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_s1_json(root: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """assemble + figure + validate_s1_output + atomic write of results/direction2/s1.json."""
    doc = assemble(root, plan, power_curve_figure)
    s1.validate_s1_output(doc, root)
    _write_json(os.path.join(root, S1_JSON), doc)
    return doc
