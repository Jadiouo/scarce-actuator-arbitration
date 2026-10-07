"""Frozen run plan (results/direction2/run_plan.json): loading, integrity check and per-cell parameter lookup.

The training entry point (`python -m arbitration.rl.train --plan ... --cell ...`) takes lam, n_S, T_train, G_gens, val_every, patience,
obs_version, the start route and the vulnerability knob from the plan, never from `budget.py` or from a measurement file (spec 5.5).
The plan is only accepted if its sha256 equals the one recorded in results/direction2/FREEZE.md (line `RUN_PLAN_SHA256: <hex>`).
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, Optional

DEFAULT_PLAN = os.path.join("results", "direction2", "run_plan.json")
DEFAULT_FREEZE = os.path.join("results", "direction2", "FREEZE.md")
PILOT_DECISION = os.path.join("results", "direction2", "pilot_decision.json")
_SHA_LINE = re.compile(r"^[ \t]*RUN_PLAN_SHA256:\s*([0-9a-f]{64})\s*$", re.M)
# CLI flags that a plan-mode run may not override (the plan is the only source of these values)
PLAN_LOCKED_FLAGS = ("--mech", "--r", "--lam", "--nS", "--T", "--T-val", "--val-every", "--patience")


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


def resolve_start(cell: Dict[str, Any], requested: Optional[str], root: str = ".") -> str:
    """The start route of a cell.  Fixed cells (pilot) force their route; the others need --start to equal the recorded pilot decision."""
    st = cell["start"]
    allowed = list(st["allowed"])
    if len(allowed) == 1:
        if requested not in (None, allowed[0]):
            raise ValueError(f"cell {cell['name']}: start is fixed to {allowed[0]!r} in the plan, got --start {requested!r}")
        return allowed[0]
    path = os.path.join(root, PILOT_DECISION)
    if not os.path.exists(path):
        raise ValueError(f"cell {cell['name']}: the start route is decided by the NAIVE pilot (REQ-OPT-07) but {PILOT_DECISION} does not exist yet")
    with open(path, encoding="utf-8") as f:
        chosen = json.load(f).get("chosen_start")
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
