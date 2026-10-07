"""REQ-MEAS-*: measurement output schemas (spec s11)."""
import json
import os
from typing import Any, Dict, Sequence

_NUM = (int, float)
_M1 = dict(mech=str, r=_NUM, pop=int, n_S=int, B=int, policy=str, t_step_ms_median=_NUM, t_step_ms_p95=_NUM, steps_timed=int,
           warmup=int, repeats=int, mem_peak_MB=_NUM, elem_steps_per_s=_NUM, frontier_ms_per_step=_NUM, device=str, dtype=str,
           torch=str, git_sha=str)
_M2_ROW = dict(r=_NUM, T=int, pair_kind=str, pair_id=int, var_crn=_NUM, var_indep=_NUM, vrf=_NUM, sd_crn=_NUM, n_seeds=int)
_M2_SUM = dict(r=_NUM, T=int, vrf_median=_NUM, vrf_min=_NUM, sd_crn_median=_NUM)
_M3 = dict(r=_NUM, sigma_val_G=_NUM, sigma_val_D=_NUM, MDE_G=_NUM, MDE_D_plan=_NUM, MDE_est=_NUM, rel_diff=_NUM)


def _check(row: Any, schema: Dict[str, Any], what: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{what}: not a dict")
    for k, t in schema.items():
        if k not in row:
            raise ValueError(f"{what}: missing field {k!r}")
        v = row[k]
        if isinstance(v, bool) or not isinstance(v, t):
            raise ValueError(f"{what}: field {k!r} has wrong type {type(v).__name__}")


def validate_m1(rows: Sequence[Dict[str, Any]]) -> bool:
    """REQ-MEAS-01: each row has the 18 listed fields with correct types; ValueError otherwise."""
    for row in rows:
        _check(row, _M1, "M-1")
    return True


def validate_m2(doc: Dict[str, Any]) -> bool:
    """REQ-MEAS-02: doc {'rows': [...], 'summary': [...]}; row fields r,T,pair_kind,pair_id,var_crn,var_indep,vrf,sd_crn,n_seeds."""
    if not isinstance(doc, dict) or "rows" not in doc or "summary" not in doc:
        raise ValueError("M-2: need 'rows' and 'summary'")
    for row in doc["rows"]:
        _check(row, _M2_ROW, "M-2 row")
    for row in doc["summary"]:
        _check(row, _M2_SUM, "M-2 summary")
    return True


def validate_m3(doc: Dict[str, Any]) -> bool:
    """REQ-MEAS-03: fields r, sigma_val_G, sigma_val_D, MDE_G, MDE_D_plan, MDE_est, rel_diff."""
    _check(doc, _M3, "M-3")
    return True


def validate_m4(doc: Dict[str, Any], root: str = ".") -> bool:
    """REQ-MEAS-04: doc has G_star_ref {'mean','lo','hi'} and json_ref equal to stage2b consist_M4[12].gain_b03."""
    g = doc.get("G_star_ref") if isinstance(doc, dict) else None
    if not isinstance(g, dict) or not all(isinstance(g.get(k), _NUM) for k in ("mean", "lo", "hi")):
        raise ValueError("M-4: G_star_ref must have numeric mean, lo, hi")
    if "json_ref" not in doc:
        raise ValueError("M-4: missing json_ref")
    with open(os.path.join(root, "results", "stage2b.json")) as f:
        ref = json.load(f)["consist_M4"][12]["gain_b03"]
    if list(doc["json_ref"]) != list(ref):
        raise ValueError("M-4: json_ref differs from stage2b.json consist_M4[12].gain_b03")
    return True
