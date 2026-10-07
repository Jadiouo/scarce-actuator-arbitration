"""Frozen run plan (results/direction2/run_plan.json): loading, integrity check and per-cell parameter lookup.

The training entry point (`python -m arbitration.rl.train --plan ... --cell ...`) takes lam, n_S, T_train, G_gens, val_every, patience,
obs_version, the start route and the vulnerability knob from the plan, never from `budget.py` or from a measurement file (spec 5.5).
The plan is only accepted if its sha256 equals the one recorded in results/direction2/FREEZE.md (line `RUN_PLAN_SHA256: <hex>`).
"""
import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PLAN = os.path.join("results", "direction2", "run_plan.json")
DEFAULT_FREEZE = os.path.join("results", "direction2", "FREEZE.md")
PILOT_DECISION = os.path.join("results", "direction2", "pilot_decision.json")
M3_RECORD = os.path.join("results", "direction2", "measure_m3.json")
FREEZE_COMMIT = "a72d3d731a450fe4724f0fa118943e25c39b7577"          # the freeze commit (FREEZE.md); HEAD must descend from it
_SHA_LINE = re.compile(r"^[ \t]*RUN_PLAN_SHA256:\s*([0-9a-f]{64})\s*$", re.M)
_PILOT_SHA_LINE = re.compile(r"^[ \t]*PILOT_DECISION_SHA256:\s*([0-9a-f]{64})\s*$", re.M)
_M3_SHA_LINE = re.compile(r"^[ \t]*M3_SHA256:\s*([0-9a-f]{64})\s*$", re.M)
# Cells whose results are REUSED from another cell instead of being trained again (recorded in FREEZE.md; checked in code):
# cell -> {chosen start route -> source cell}.  Reuse is allowed only if every setting of the two cells is identical (check_reuse).
REUSED_CELLS = {"s1_NAIVE_anchor": {"cold": "pilot_naive_cold", "warm": "pilot_naive_warm"}}
_REUSE_FIELDS = ("mechanism", "variant", "operator", "knob", "r", "lam", "T_train", "T_val", "n_S", "G_gens", "val_every", "patience", "sigma0",
                 "obs_version", "F", "hidden", "run_ids", "algo_seeds", "train_seeds", "val_seeds", "mech_cfg")


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze_plan_sha(freeze_path: str) -> str:
    """The run_plan.json sha256 recorded in FREEZE.md (exactly one `RUN_PLAN_SHA256: <hex>` line)."""
    with open(freeze_path, encoding="utf-8") as f:
        found = _SHA_LINE.findall(f.read())
    if len(found) != 1:
        raise ValueError(f"{freeze_path}: expected exactly one 'RUN_PLAN_SHA256: <hex>' line, found {len(found)}")
    return found[0]


def registered_sha(freeze_path: str, pattern: "re.Pattern[str]") -> Optional[str]:
    """The sha256 registered in FREEZE.md by `pattern` (one line); None if there is no such line; ValueError if there are several different ones."""
    with open(freeze_path, encoding="utf-8") as f:
        found = pattern.findall(f.read())
    if len(set(found)) > 1:
        raise ValueError(f"{freeze_path}: conflicting registrations {sorted(set(found))}")
    return found[0] if found else None


# ------------------------------------------------------------------------------------------------ git guards
def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def is_committed_and_clean(root: str, rel: str) -> bool:
    """True iff `rel` is tracked and has no staged or unstaged change against HEAD (`git diff --quiet HEAD -- rel`)."""
    if _git(root, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return False
    return _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode == 0


def verify_repo_state(root: str, *, freeze_commit: Optional[str] = FREEZE_COMMIT) -> None:
    """Entry guard (REQ-S1-14): run_plan.json and FREEZE.md are committed with no change against HEAD, and HEAD is a descendant of the freeze commit
    (freeze_commit=None only in the test mode, where the throw-away repository has no freeze commit).  RuntimeError otherwise."""
    for rel in (DEFAULT_PLAN, DEFAULT_FREEZE):
        if not is_committed_and_clean(root, rel):
            raise RuntimeError(f"refusing to train: {rel} is not committed or differs from HEAD (git diff --quiet HEAD -- {rel} failed)")
    if _git(root, "rev-parse", "--verify", "HEAD").returncode != 0:
        raise RuntimeError("refusing to train: not a git repository with a HEAD commit")
    if freeze_commit is not None and _git(root, "merge-base", "--is-ancestor", freeze_commit, "HEAD").returncode != 0:
        raise RuntimeError(f"refusing to train: HEAD is not a descendant of the freeze commit {freeze_commit[:7]}")


def _registered_clean_file(root: str, rel: str, pattern: "re.Pattern[str]", what: str, key: str) -> str:
    """The file `rel` must be committed (clean against HEAD) and its sha256 equal the one registered in FREEZE.md under `key`; returns the path."""
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        raise ValueError(f"{what} ({rel}) does not exist yet")
    freeze = os.path.join(root, DEFAULT_FREEZE)
    reg = registered_sha(freeze, pattern)
    if reg is None:
        raise ValueError(f"{what}: FREEZE.md has no '{key}: <sha256>' registration")
    if not is_committed_and_clean(root, rel):
        raise ValueError(f"{what} ({rel}) is not committed (or differs from HEAD)")
    if sha256_file(path) != reg:
        raise ValueError(f"{what} ({rel}) sha256 {sha256_file(path)} != the one registered in FREEZE.md ({reg})")
    return path


def load_pilot_decision(root: str) -> Optional[str]:
    """chosen_start of the REGISTERED pilot decision, or None if the decision file or its FREEZE.md registration does not exist.
    A present registration is fully verified: file committed and unchanged, sha256 equal to PILOT_DECISION_SHA256, committed exactly once
    (never modified), registration never changed in FREEZE.md history; chosen_start in {cold, warm}.  Any inconsistency -> ValueError."""
    path = os.path.join(root, PILOT_DECISION)
    reg = registered_sha(os.path.join(root, DEFAULT_FREEZE), _PILOT_SHA_LINE)
    if reg is None or not os.path.exists(path):
        return None
    _registered_clean_file(root, PILOT_DECISION, _PILOT_SHA_LINE, "pilot decision", "PILOT_DECISION_SHA256")
    n = [x for x in _git(root, "log", "--format=%H", "--", PILOT_DECISION).stdout.split() if x]
    if len(n) != 1:
        raise ValueError(f"{PILOT_DECISION} was committed {len(n)} times; a registered decision is written once and never changed")
    seen = set()
    for c in [x for x in _git(root, "log", "--format=%H", "--", DEFAULT_FREEZE).stdout.split() if x]:
        txt = _git(root, "show", f"{c}:{DEFAULT_FREEZE.replace(os.sep, '/')}").stdout
        seen.update(_PILOT_SHA_LINE.findall(txt))
    if len(seen) > 1:
        raise ValueError(f"the PILOT_DECISION_SHA256 registration changed over the history of FREEZE.md: {sorted(seen)}")
    with open(path, encoding="utf-8") as f:
        chosen = json.load(f).get("chosen_start")
    if chosen not in ("cold", "warm"):
        raise ValueError(f"{PILOT_DECISION}: chosen_start {chosen!r} is not 'cold' or 'warm'")
    return chosen


def load_m3(root: str) -> Dict[str, Any]:
    """The M-3 record (results/direction2/measure_m3.json), accepted only if committed unchanged and registered in FREEZE.md (M3_SHA256)."""
    path = _registered_clean_file(root, M3_RECORD, _M3_SHA_LINE, "M-3 record", "M3_SHA256")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mde_d_plan(root: str, r: float = 0.5) -> float:
    doc = load_m3(root)
    rows = [x for x in doc["rows"] if float(x["r"]) == r]
    if len(rows) != 1:
        raise ValueError(f"M-3 record has no unique row for r={r}")
    from . import meas
    meas.validate_m3(rows[0])
    return float(rows[0]["MDE_D_plan"])


def check_conditional(cell: Dict[str, Any], root: str) -> None:
    """REQ-S1-22: a cell marked `conditional` because of its size (s=0.01) may only run if its size is in G_s = {s : s >= MDE_D,plan} from the REGISTERED
    M-3 record; 0.02 not in G_s is STOP-02 (no S1 cell at all).  No M-3 record -> refused.  The warm pilot's condition (cold pilot below 0.8 G*_ref)
    is decided by a human from the pilot record and is not machine-checked here."""
    if not cell.get("conditional") or cell.get("size") is None:
        return
    from . import s1
    try:
        mde = mde_d_plan(root)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"cell {cell['name']} is conditional ({cell['conditional']}) but MDE_D,plan is not available: {e}") from e
    g = s1.gate_sizes(mde)
    if g["stop02"]:
        raise ValueError(f"REQ-STOP-02: MDE_D,plan={mde:.5f} > 0.02, 0.02 is not in G_s; S1 cannot be run (cell {cell['name']})")
    if float(cell["size"]) not in g["G_s"]:
        raise ValueError(f"cell {cell['name']}: size {cell['size']} < MDE_D,plan={mde:.5f} (REQ-S1-22: G_s={g['G_s']}); the cell is descriptive only and is not run")


class CellReused(ValueError):
    """The cell is not trained: its result is the (identical) source cell's result."""


def check_reuse(plan: Dict[str, Any], cell: Dict[str, Any], chosen: Optional[str]) -> Optional[str]:
    """If `cell` is a reused cell (REUSED_CELLS): verify that every setting (incl. algo seed, seed block, validation seeds) is identical to the source
    cell of the chosen route and return the source cell's name (the caller must then NOT train, see train.main).  Returns None for ordinary cells.
    A difference in any setting -> ValueError('not identical'), and the cell is NOT reusable."""
    routes = REUSED_CELLS.get(cell["name"])
    if routes is None:
        return None
    if chosen not in routes:
        raise ValueError(f"cell {cell['name']}: reuse needs a registered pilot route, got {chosen!r}")
    src = get_cell(plan, routes[chosen])
    diff = [k for k in _REUSE_FIELDS if cell.get(k) != src.get(k)]
    if (src["start"]["allowed"] != [chosen]) or chosen not in cell["start"]["allowed"]:
        diff.append("start")
    if diff:
        raise ValueError(f"cell {cell['name']} is not identical to {src['name']} in {diff}: it is not reusable and must not be reported as reused")
    return src["name"]


def load_plan(plan_path: str, freeze_path: str) -> Dict[str, Any]:
    """Load the plan after checking its sha256 against FREEZE.md; ValueError on any mismatch."""
    actual = sha256_file(plan_path)
    recorded = freeze_plan_sha(freeze_path)
    if actual != recorded:
        raise ValueError(f"run plan sha256 {actual} != the one recorded in {freeze_path} ({recorded}); refusing to train")
    with open(plan_path, encoding="utf-8") as f:
        return json.load(f)


def get_cell(plan: Dict[str, Any], name: str) -> Dict[str, Any]:
    hit = [c for c in plan["cells"] if c["name"] == name]
    if len(hit) != 1:
        raise ValueError(f"cell {name!r} not found in the run plan (known: {[c['name'] for c in plan['cells']]})")
    return hit[0]


def resolve_start(cell: Dict[str, Any], requested: Optional[str], chosen: Optional[str] = None) -> str:
    """The start route of a cell.  Fixed cells (pilot) force their route; the others need the REGISTERED pilot decision `chosen`
    (load_pilot_decision; None = not registered -> refused) and --start, if given, must equal it (no switching after the decision)."""
    st = cell["start"]
    allowed = list(st["allowed"])
    if len(allowed) == 1:
        if requested not in (None, allowed[0]):
            raise ValueError(f"cell {cell['name']}: start is fixed to {allowed[0]!r} in the plan, got --start {requested!r}")
        return allowed[0]
    if chosen is None:
        raise ValueError(f"cell {cell['name']}: the start route is decided by the NAIVE pilot (REQ-OPT-07) but {PILOT_DECISION} is not committed and "
                         f"registered (PILOT_DECISION_SHA256 in FREEZE.md) yet; only pilot cells may run")
    if chosen not in allowed:
        raise ValueError(f"{PILOT_DECISION}: chosen_start {chosen!r} not in {allowed}")
    if requested not in (None, chosen):
        raise ValueError(f"cell {cell['name']}: --start {requested!r} contradicts the frozen pilot decision {chosen!r} (no switching after freeze)")
    return chosen


def check_run(cell: Dict[str, Any], run_id: int, g0: int, g1: int) -> None:
    if not cell.get("wired", True):
        raise NotImplementedError(f"cell {cell['name']} is registered in the plan but its training path is not wired yet ({cell.get('notes')})")
    if run_id not in cell["run_ids"]:
        raise ValueError(f"cell {cell['name']}: run {run_id} not in the plan's run_ids {cell['run_ids']}")
    if not (0 <= g0 < g1 <= cell["G_gens"]):
        raise ValueError(f"cell {cell['name']}: generations [{g0},{g1}) must lie within [0,{cell['G_gens']}]")


def cell_to_settings(cell: Dict[str, Any]) -> Dict[str, Any]:
    """All simulator/optimizer settings of a cell, straight from the plan (no defaults, no budget.py)."""
    from . import env
    if cell["mechanism"] == "M4":
        mc = cell["mech_cfg"]
        cfg, mech = dict(eps=float(mc["eps"]), L=int(mc["L"]), rw=float(mc["rw"])), "M4"
    else:
        cfg, mech = env.m3c_config(cell["r"]), "M3C"
        if str(cell.get("variant") or "").startswith("NAIVE") or cell.get("vuln_id") == "NAIVE":
            cfg = dict(cfg, p=0.0)
    return dict(mech=mech, cfg=cfg, r=float(cell["r"]), lam=int(cell["lam"]), n_S=int(cell["n_S"]), T_train=int(cell["T_train"]),
                T_val=int(cell["T_val"]), val_every=int(cell["val_every"]), patience=cell["patience"], sigma0=float(cell["sigma0"]),
                obs_version=cell["obs_version"], F=int(cell["F"]), hidden=int(cell["hidden"]),
                operator=cell.get("operator"), knob=cell.get("knob"))
