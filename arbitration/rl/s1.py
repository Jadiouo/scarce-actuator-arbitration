"""REQ-S1-*: categories, gate rule, anchors, pilot, registry/freeze checks, calibration, designer isolation (spec v1.3).

Orchestration only (vulnerability list, sizes, run registry, freeze checks).  This module is scanned by REQ-SEED-03 (T-27): it only ever
refers to training, validation and val2 seeds (val2 = calibration only, REQ-SEED-08); the test-seed evaluation lives in final_eval.py.
"""
import hashlib
import json
import math
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import env
from . import hw
from . import seeds as _seeds
from .metrics import cp_lower, found_rule, mean_ci
from .vulns import CATEGORIES, CATEGORY_OF

SIZES = (0.01, 0.02)                                   # REQ-S1-02 (v1.3): the registered sizes (0.003 / 0.005 were never calibrated)
OUT_OF_FAMILY_IDS = ("D1", "D2", "O1")
IN_FAMILY_IDS = ("O2", "O3", "D4")
BLIND_IDS = ("O5",)
ANCHOR_IDS = ("NAIVE", "M4rw0")
N_OUT_OF_FAMILY = 3                                    # the denominator of the pass criterion (REQ-S1-17)
N_IN_CLASS = 6                                         # out_of_family + in_family (REQ-S1-15 b)
# the 64 calibration seeds (REQ-S1-28): val (2000-2031) then val2 (3000-3031); val2 is for calibration/selection only
CALIB_SEEDS = tuple(_seeds.splits()["val"]) + tuple(_seeds.splits()["val2"])

assert {i for i, c in CATEGORY_OF.items() if c == "out_of_family"} == set(OUT_OF_FAMILY_IDS)
assert {i for i, c in CATEGORY_OF.items() if c == "in_family"} == set(IN_FAMILY_IDS)


def size_basis(category: str) -> Optional[str]:
    """REQ-S1-02: the Delta that defines a vulnerability's size: 'fine' for out_of_family and blind, 'coarse' for in_family; None where no size exists."""
    return {"out_of_family": "fine", "blind": "fine", "in_family": "coarse"}.get(category)


def counts_in_detection_statistics(vuln_id: str) -> bool:
    """REQ-S1-15: True for the 6 in-class vulnerabilities (out_of_family + in_family); D3, O5, anchors and designer_1 are never counted."""
    return CATEGORY_OF.get(vuln_id) in ("out_of_family", "in_family")


def gate_sizes(mde_d_plan: float) -> Dict[str, Any]:
    """REQ-S1-22: {'G_s': [s in {.01,.02} if s>=mde_d_plan], 'stop02': bool (True iff 0.02 not in G_s)}."""
    g = [s for s in SIZES if s >= mde_d_plan]
    return {"G_s": g, "stop02": 0.02 not in g}


def run_count(G_s: Sequence[float]) -> int:
    """REQ-S1-10: N = 3*|G_s| (out_of_family) + 3 (in_family at 0.02) + 1 (blind O5 at 0.02) + 2 (anchors)."""
    return N_OUT_OF_FAMILY * len(G_s) + len(IN_FAMILY_IDS) + len(BLIND_IDS) + len(ANCHOR_IDS)


def schedule(G_s: Sequence[float]) -> List[Tuple[str, float]]:
    """REQ-S1-10: ordered runs [(vuln_or_anchor_id, size)] (anchors carry size 0.02): s=0.02 first (3 out_of_family + 3 in_family + 1 blind + 2 anchors = 9),
    then the out_of_family trio at the remaining gate sizes.  D3 (not run), designer_1, O4/O6/B1/B2 never; descriptive sizes (< MDE) are not scheduled."""
    if 0.02 not in G_s:
        raise ValueError("REQ-STOP-02: 0.02 is not in the gate size set; S1 cannot be run")
    first = [(i, 0.02) for i in OUT_OF_FAMILY_IDS + IN_FAMILY_IDS + BLIND_IDS + ANCHOR_IDS]
    rest = [(i, s) for s in sorted((s for s in G_s if s != 0.02), reverse=True) for i in OUT_OF_FAMILY_IDS]
    return first + rest


def anchor_definitions() -> Dict[str, Dict[str, Any]]:
    """REQ-S1-08/09, REQ-ENV-11: {'NAIVE': {mech,p,r}, 'M4': {mech,eps,L,rw,r,strategy}}."""
    return {"NAIVE": dict(env.anchor_config("NAIVE")), "M4": dict(env.anchor_config("M4"))}


def anchor_passes(G_rl: float, G_lo: float, G_star_ref: float, mde_G: float) -> bool:
    """REQ-S1-11: G_rl >= 0.8*G_star_ref and G_lo > 0 and G_rl >= mde_G."""
    return bool(G_rl >= 0.8 * G_star_ref and G_lo > 0 and G_rl >= mde_G)


def m4_anchor_status(G_star_ref_lo: float) -> str:
    """REQ-S1-09 (P-5): 'informational' if the W1 re-measured G*_ref CI lower bound <= 0, else 'counted'."""
    return "informational" if G_star_ref_lo <= 0 else "counted"


def s1_gate(naive_pass: Optional[bool], m4_pass: Optional[bool], m4_informational: bool,
            detections_s002: Sequence[Optional[bool]]) -> bool:
    """REQ-S1-17/STOP-01 (v1.3): S1 passes iff the (counted) anchors pass (None = unfinished = not passed; M4 ignored if informational) and ALL 3
    out-of-family vulnerabilities (D1, D2, O1) are detected at s=0.02 (None = undetected; len must be 3).  Nothing else enters."""
    if len(detections_s002) != N_OUT_OF_FAMILY:
        raise ValueError(f"exactly {N_OUT_OF_FAMILY} out-of-family detections expected (in-family, blind, D3 and designer_1 are not part of the criterion)")
    if naive_pass is not True:
        return False
    if not m4_informational and m4_pass is not True:
        return False
    return all(d is True for d in detections_s002)


def s1_gate_by_id(naive_pass: Optional[bool], m4_pass: Optional[bool], m4_informational: bool,
                  detections: Dict[str, Optional[bool]]) -> bool:
    """REQ-S1-17: like s1_gate, from a {vuln_id: detected} map at s=0.02.  Only the out_of_family ids are read (a missing one = undetected); every other
    suite member (in_family, D3, O5, anchors, designer_1) may carry any value without changing the verdict; an id outside the suite (O4, O6, B1, B2...) is an error."""
    unknown = [i for i in detections if i not in CATEGORY_OF]
    if unknown:
        raise ValueError(f"not members of the suite: {unknown}")
    return s1_gate(naive_pass, m4_pass, m4_informational, [detections.get(i) for i in OUT_OF_FAMILY_IDS])


def summary_table(x_by_size: Dict[float, int], n: int = N_OUT_OF_FAMILY) -> List[Dict[str, Any]]:
    """REQ-S1-15/16: rows {'size','x','n','p_hat','LB'} (descriptive only; n=3 out_of_family, n=6 in-class)."""
    return [{"size": s, "x": int(x), "n": n, "p_hat": x / n, "LB": cp_lower(int(x), n)} for s, x in sorted(x_by_size.items())]


def pilot_decision(cold_val_G: float, warm_val_G: Optional[float], G_star_ref: float) -> str:
    """REQ-S1-20: 'cold' if cold>=0.8*G*_ref; else 'warm' if warm>=0.8*G*_ref; else 'STOP-02'."""
    thr = 0.8 * G_star_ref
    if cold_val_G >= thr:
        return "cold"
    if warm_val_G is not None and warm_val_G >= thr:
        return "warm"
    return "STOP-02"


def require_pilot_seeds(seeds: Sequence[int]) -> None:
    """REQ-S1-20: pilot may use train/val seeds only (ValueError for test, val2 or replay seeds)."""
    for s in seeds:
        s = int(s)
        if not (0 <= s <= 999 or 2000 <= s <= 2031):
            raise ValueError(f"REQ-S1-20: seed {s} is neither a training (0-999) nor a validation (2000-2031) seed")


_FREEZE_KEYS = ("cold_start", "warm_start", "pilot", "chosen_start", "lam", "n_S", "T_train", "G_gens", "sigma0", "MDE_D_plan", "G_s",
                "registry_sha256", "nS_coverage_disclosure", "S_HW", "calibration", "categories")
_FREEZE_NOT_CONFIG = ("cold_start", "warm_start", "pilot", "chosen_start", "nS_coverage_disclosure", "S_HW", "calibration", "categories")


def _check_ns_disclosure(text: Any) -> None:
    """REQ-OPT-13: the text names the single-pair worst case (n_S ~ 565) and says it is NOT covered."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("freeze: nS_coverage_disclosure must be a non-empty text (REQ-OPT-13)")
    low = text.lower()
    if "565" not in text or not ("not covered" in low or "無法涵蓋" in text or "未涵蓋" in text):
        raise ValueError("freeze: nS_coverage_disclosure must name the n_S~565 single-pair worst case and state that it is not covered (REQ-OPT-13)")


def check_freeze_registration(freeze: Dict[str, Any]) -> bool:
    """REQ-OPT-07/OPT-13/S1-14/S1-20 (v1.3): the freeze dict must contain 'cold_start','warm_start','pilot','chosen_start'(cold|warm), the frozen
    parameters (lam,n_S,T_train,G_gens,sigma0,MDE_D_plan,G_s,registry_sha256), the n_S coverage disclosure, the 36-name S_HW list (exact order, REQ-HW-01),
    the val+val2 calibration record (REQ-S1-28) and the fixed category of every suite member (REQ-S1-26); else ValueError."""
    missing = [k for k in _FREEZE_KEYS if k not in freeze]
    if missing:
        raise ValueError(f"freeze file lacks {missing}")
    if freeze["chosen_start"] not in ("cold", "warm"):
        raise ValueError("chosen_start must be 'cold' or 'warm'")
    _check_ns_disclosure(freeze["nS_coverage_disclosure"])
    names = [n for n, _ in hw.build_s_hw()]
    if freeze["S_HW"] != names:
        raise ValueError("freeze: S_HW must be the 36-name handwritten list of REQ-HW-01 in its fixed order")
    cal = freeze["calibration"]
    if not isinstance(cal, dict) or cal.get("calib_seeds") != "val+val2" or cal.get("n_seeds") != len(CALIB_SEEDS):
        raise ValueError("freeze: calibration must record calib_seeds == 'val+val2' and n_seeds == 64 (REQ-S1-28)")
    if freeze["categories"] != dict(CATEGORY_OF):
        raise ValueError("freeze: categories must be exactly the fixed assignment of REQ-S1-01/26")
    return True


def load_config(freeze: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """REQ-OPT-07: config for stage in {'s1','main'}; both use freeze['chosen_start']."""
    if stage not in ("s1", "main"):
        raise ValueError(f"stage must be 's1' or 'main', got {stage!r}")
    check_freeze_registration(freeze)
    cfg = {k: freeze[k] for k in _FREEZE_KEYS if k not in _FREEZE_NOT_CONFIG}
    cfg.update(stage=stage, start=freeze["chosen_start"])
    return cfg


def check_registry_composition(registry: Sequence[Dict[str, Any]]) -> bool:
    """REQ-S1-01/02 (v1.3): exactly the 11 suite members (out_of_family D1,D2,O1; in_family O2,O3,D4; in_family_known D3; blind O5; anchors NAIVE,M4rw0;
    contaminated_excluded designer_1), each with its FIXED category and `blind == (category == 'blind')`; out_of_family and in_family entries carry both
    sizes {0.01, 0.02} (one entry per size), the blind O5 only 0.02; D3, anchors and designer_1 carry no size.  O4, O6, B1, B2 are not suite members.
    ValueError otherwise."""
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for e in registry:
        i = e.get("id")
        if i not in CATEGORY_OF:
            raise ValueError(f"{i!r} is not a member of the suite (O4, O6, B1, B2 left it; REQ-S1-01)")
        cat = e.get("category")
        if cat not in CATEGORIES:
            raise ValueError(f"{i}: category {cat!r} is not one of {CATEGORIES}")
        if cat != CATEGORY_OF[i]:
            raise ValueError(f"{i}: category {cat!r} contradicts the fixed assignment {CATEGORY_OF[i]!r} (REQ-S1-26: report, never reclassify)")
        if bool(e.get("blind")) != (cat == "blind"):
            raise ValueError(f"{i}: blind flag inconsistent with category {cat}")
        by_id.setdefault(i, []).append(e)
    absent = [i for i in CATEGORY_OF if i not in by_id]
    if absent:
        raise ValueError(f"suite members missing: {absent}")
    for i, es in by_id.items():
        sizes = [e.get("size") for e in es]
        cat = CATEGORY_OF[i]
        if cat in ("out_of_family", "in_family"):
            want = set(SIZES)
        elif cat == "blind":
            want = {0.02}
        else:
            want = {None}
        if sorted(sizes, key=lambda x: -1 if x is None else x) != sorted(want, key=lambda x: -1 if x is None else x):
            raise ValueError(f"{i}: sizes {sizes} != {sorted(want, key=lambda x: -1 if x is None else x)}")
    return True


_REQ_FIELDS = ("G_star_val", "G_HW_best_val", "ratio_HW_over_Gstar", "delta_fine", "delta_coarse", "size_basis", "delta_val")


def check_registry_categories(registry: Sequence[Dict[str, Any]]) -> bool:
    """REQ-S1-26 (v1.3) machine check, run after (re)calibration: composition (see check_registry_composition) plus, for every non-anchor, non-contaminated
    entry, the presence and consistency of G_star_val, G_HW_best_val, ratio_HW_over_Gstar, delta_fine, delta_coarse, size_basis, delta_val; sized entries'
    basis delta in [0.8 s, 1.2 s] with CI lower bound > 0 (out_of_family: delta_fine; in_family: delta_coarse, its delta_fine may be <= 0);
    in_family_known: both deltas negative.  Any contradiction raises ValueError listing every problem ('stop and report to the commander'); the registry is
    never modified or reclassified."""
    check_registry_composition(registry)
    problems: List[str] = []
    for e in registry:
        i, cat = e["id"], e["category"]
        if cat in ("anchor", "contaminated_excluded"):
            continue
        tag = f"{i} s={e.get('size')}"
        miss = [f for f in _REQ_FIELDS if f not in e]
        if miss:
            problems.append(f"{tag}: fields missing {miss}")
            continue
        try:
            gs, gh = e["G_star_val"]["mean"], e["G_HW_best_val"]["mean"]
            if abs(e["ratio_HW_over_Gstar"] - gh / gs) > 1e-9 * max(1.0, abs(gh / gs)):
                problems.append(f"{tag}: ratio_HW_over_Gstar != G_HW/G*")
            if abs(e["delta_fine"]["mean"] - (gs - gh)) > 1e-9:
                problems.append(f"{tag}: delta_fine != G* - G_HW")
        except (KeyError, TypeError, ZeroDivisionError) as ex:
            problems.append(f"{tag}: malformed G/delta fields ({ex!r})")
            continue
        if cat == "in_family_known":
            if not (e["delta_fine"]["mean"] < 0 and e["delta_coarse"]["mean"] < 0):
                problems.append(f"{tag}: in_family_known requires negative deltas (fine {e['delta_fine']['mean']}, coarse {e['delta_coarse']['mean']})")
            continue
        basis = size_basis(cat)
        if e["size_basis"] != basis:
            problems.append(f"{tag}: size_basis {e['size_basis']!r} != {basis!r}")
            continue
        d = e["delta_fine" if basis == "fine" else "delta_coarse"]
        dv = e["delta_val"]
        if not isinstance(dv, dict) or any(abs(dv.get(k, math.nan) - d[k]) > 1e-12 for k in ("mean", "lo", "hi")):
            problems.append(f"{tag}: delta_val is not the {basis} delta")
        s = e["size"]
        if not (0.8 * s <= d["mean"] <= 1.2 * s):
            problems.append(f"{tag}: Delta_{basis}={d['mean']} outside [{0.8 * s}, {1.2 * s}]")
        if not d["lo"] > 0:
            problems.append(f"{tag}: Delta_{basis} CI lower bound {d['lo']} <= 0")
    if problems:
        raise ValueError("REQ-S1-26: registry contradicts its categories (stop and report to the commander, no automatic reclassification): " + "; ".join(problems))
    return True


def registry_sha256(path: str) -> str:
    """REQ-S1-14: sha256 hex of the registry file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_exempt_cell(cell: str) -> bool:
    """REQ-S1-14: NAIVE pilot ('naive_pilot*') and smoke-test ('smoke*') cells are exempt from the ordering check."""
    return cell.startswith("naive_pilot") or cell.startswith("pilot_naive") or cell.startswith("smoke")


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def _adding_commit(repo: str, path: str) -> Optional[str]:
    """Earliest commit that added `path` (None if never committed)."""
    r = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", path)
    lines = [x for x in r.stdout.split() if x]
    return lines[-1] if r.returncode == 0 and lines else None


def freeze_precedes_parts(repo: str, freeze_path: str, parts_dir: str) -> Dict[str, Any]:
    """REQ-S1-14: via `git merge-base --is-ancestor`: the commit that added freeze_path must be an ancestor of the commit
    that added every non-exempt part file.  Returns {'ok': bool, 'violations': [part names]}."""
    fc = _adding_commit(repo, freeze_path)
    violations: List[str] = []
    d = os.path.join(repo, parts_dir)
    names = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
    for name in names:
        if is_exempt_cell(name.split("__")[0]):
            continue
        pc = _adding_commit(repo, os.path.join(parts_dir, name))
        if fc is None or pc is None or fc == pc or _git(repo, "merge-base", "--is-ancestor", fc, pc).returncode != 0:
            violations.append(name)            # freeze not committed, part uncommitted, same commit, or freeze not strictly earlier
    return {"ok": not violations, "violations": violations}


def check_designer_manifest(manifest: Sequence[Dict[str, str]], allowed_paths: Sequence[str],
                            forbidden_sha256: Sequence[str]) -> List[str]:
    """REQ-S1-12: manifest entries {'path','sha256'}; return violations (path not allowed or hash forbidden)."""
    out = []
    for e in manifest:
        if e["path"] not in allowed_paths:
            out.append(f"path not allowed: {e['path']}")
        if e["sha256"] in forbidden_sha256:
            out.append(f"forbidden content (sha256): {e['path']}")
    return out


_FORBIDDEN_PROMPT = ("k=8", "EMA", "λ_e", "f06", "P_MAX", "strategies()")


def check_designer_prompt(text: str) -> List[str]:
    """REQ-S1-21: forbidden phrases found in the designer prompt ('k=8','EMA','λ_e','f06','P_MAX','strategies()')."""
    return [p for p in _FORBIDDEN_PROMPT if p in text]


_FORBIDDEN_READ = re.compile(r"direction2-notes\.md|strategies\(\)|run_quota|spec_s[45](?!\d)|§\s*[45](?!\d)|obs\.py|policy\.py")


def check_designer_logs(repo: str, logs_root: str, designer_ids: Sequence[str]) -> List[str]:
    """REQ-S1-21: for each designer: prompt.txt and tool_calls.jsonl exist and are tracked in git; tool_calls.jsonl has no read
    of spec s4/s5, 'strategies()' or 'direction2-notes.md'.  Return violations."""
    out: List[str] = []
    for d in designer_ids:
        for fn in ("prompt.txt", "tool_calls.jsonl"):
            rel = os.path.join(logs_root, d, fn)
            if not os.path.exists(os.path.join(repo, rel)):
                out.append(f"{d}: {fn} missing")
            elif _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
                out.append(f"{d}: {fn} not tracked in git")
        calls = os.path.join(repo, logs_root, d, "tool_calls.jsonl")
        if os.path.exists(calls):
            with open(calls, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    m = _FORBIDDEN_READ.search(line)
                    if m:
                        out.append(f"{d}: tool_calls.jsonl line {n} touches forbidden material ({m.group(0)})")
    return out


# ------------------------------------------------------------------------------------------------------------------------ S1-18
def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_ci(c: Any, what: str) -> None:
    if not isinstance(c, dict) or not all(k in c and _isnum(c[k]) for k in ("mean", "lo", "hi")):
        raise ValueError(f"{what}: needs numeric mean, lo, hi")


_RUN_CATEGORIES = ("out_of_family", "in_family", "blind", "anchor")           # categories that have RL runs (D3 and designer_1 never)


def validate_s1_output(doc: Dict[str, Any], root: str = ".") -> bool:
    """REQ-S1-18 (v1.3): validate results/direction2/s1.json: runs (category, G_RL, D, MDE_D, detected; anchors: G_star_ref, passed), summary rows
    (scope 'out_of_family' n=3 per size run; scope 'in_class' n=6 at s=0.02), the D3 registry block (no RL run) and the figure; ValueError on any
    inconsistency.  D3, designer_1, O4, O6, B1, B2 have no runs; in_family and blind runs exist only at s=0.02."""
    for k in ("runs", "summary", "figure", "d3"):
        if k not in doc:
            raise ValueError(f"missing {k}")
    if not isinstance(doc["runs"], list) or not isinstance(doc["summary"], list) or not isinstance(doc["figure"], str):
        raise ValueError("runs/summary must be lists and figure a path string")
    if not os.path.exists(os.path.join(root, doc["figure"])):
        raise ValueError(f"figure file {doc['figure']} does not exist")
    d3 = doc["d3"]
    if not isinstance(d3, dict) or not all(k in d3 and _isnum(d3[k]) for k in ("delta_fine", "delta_coarse")):
        raise ValueError("d3 block must carry numeric delta_fine and delta_coarse (D3 is registered, not run)")
    out_x: Dict[float, int] = {}
    class_x = 0
    sizes_out = set()
    for r in doc["runs"]:
        for k in ("id", "kind", "category", "size", "G_RL"):
            if k not in r:
                raise ValueError(f"run lacks {k}: {r.get('id')}")
        i, cat = r["id"], r["category"]
        if i not in CATEGORY_OF:
            raise ValueError(f"{i}: not a member of the suite")
        if cat != CATEGORY_OF[i]:
            raise ValueError(f"{i}: category {cat!r} != fixed {CATEGORY_OF[i]!r}")
        if cat not in _RUN_CATEGORIES:
            raise ValueError(f"{i} ({cat}) has no S1 run: it is registered/reported only (REQ-S1-10, S1-23)")
        _check_ci(r["G_RL"], f"{i}.G_RL")
        if cat == "anchor":
            for k in ("G_star_ref", "passed"):
                if k not in r:
                    raise ValueError(f"anchor {i} lacks {k}")
            if not _isnum(r["G_star_ref"]) or not isinstance(r["passed"], bool):
                raise ValueError(f"anchor {i}: G_star_ref must be numeric, passed bool")
            mde_g = r.get("MDE_G", 0.0)
            if not _isnum(mde_g):
                raise ValueError(f"anchor {i}: MDE_G must be numeric")
            if r["passed"] != anchor_passes(r["G_RL"]["mean"], r["G_RL"]["lo"], r["G_star_ref"], mde_g):
                raise ValueError(f"anchor {i}: 'passed' inconsistent with G_RL >= 0.8 G*_ref, CI lo > 0, G_RL >= MDE_G")
            continue
        for k in ("D", "MDE_D", "detected"):
            if k not in r:
                raise ValueError(f"run {i} lacks {k}")
        _check_ci(r["D"], f"{i}.D")
        if r["size"] not in SIZES:
            raise ValueError(f"{i}: size {r['size']} not in {SIZES}")
        if cat in ("in_family", "blind") and r["size"] != 0.02:
            raise ValueError(f"{i}: {cat} vulnerabilities run only at s=0.02")
        if not _isnum(r["MDE_D"]) or not isinstance(r["detected"], bool):
            raise ValueError(f"{i}: MDE_D must be numeric and detected a bool")
        if r["detected"] != found_rule(r["D"]["mean"], r["D"]["lo"], r["MDE_D"]):
            raise ValueError(f"{i} s={r['size']}: 'detected' inconsistent with the D rule (D_lo>0 and D>=MDE_D)")
        if cat == "out_of_family":
            sizes_out.add(r["size"])
            if r["detected"]:
                out_x[r["size"]] = out_x.get(r["size"], 0) + 1
        if cat in ("out_of_family", "in_family") and r["size"] == 0.02 and r["detected"]:
            class_x += 1
    seen_scopes = set()
    for row in doc["summary"]:
        for k in ("scope", "size", "x", "n", "p_hat", "LB"):
            if k not in row:
                raise ValueError(f"summary row lacks {k}")
        for k in ("size", "x", "n", "p_hat", "LB"):
            if not _isnum(row[k]):
                raise ValueError(f"summary row lacks numeric {k}")
        if row["scope"] == "out_of_family":
            n, x = N_OUT_OF_FAMILY, out_x.get(row["size"], 0)
            if row["size"] not in sizes_out:
                raise ValueError(f"summary row for out_of_family size {row['size']} that was not run")
        elif row["scope"] == "in_class":
            n, x = N_IN_CLASS, class_x
            if row["size"] != 0.02:
                raise ValueError("the in_class summary row is defined at s=0.02 only")
        else:
            raise ValueError(f"unknown summary scope {row['scope']!r}")
        if row["n"] != n:
            raise ValueError(f"summary n must be {n} for scope {row['scope']}")
        if row["x"] != x:
            raise ValueError(f"summary x={row['x']} != number of detected vulnerabilities ({x}) for {row['scope']} at s={row['size']}")
        if abs(row["p_hat"] - x / n) > 1e-12 or abs(row["LB"] - cp_lower(x, n)) > 1e-9:
            raise ValueError(f"summary p_hat/LB inconsistent for {row['scope']} at s={row['size']}")
        key = (row["scope"], row["size"])
        if key in seen_scopes:
            raise ValueError(f"duplicate summary row {key}")
        seen_scopes.add(key)
    need_rows = {("out_of_family", s) for s in sizes_out}
    if any(r["category"] == "in_family" for r in doc["runs"]):
        need_rows.add(("in_class", 0.02))
    if need_rows - seen_scopes:
        raise ValueError(f"summary lacks rows {sorted(need_rows - seen_scopes)}")
    return True


# ------------------------------------------------------------------------------------------------------------------------ S1-24 / S1-25
def scan_blind_materials(paths: Sequence[str], extra_words: Sequence[str] = ()) -> Dict[str, Any]:
    """REQ-S1-24: scan EVERY file a blind designer may read (not only the prompt) for the REQ-S1-21 words (+ extra_words).  Returns
    {'clean': bool, 'hits': [(path, word)], 'files': [{'path', 'sha256'}]} - the archived scan record (file list and hashes)."""
    words = list(_FORBIDDEN_PROMPT) + list(extra_words)
    hits: List[Tuple[str, str]] = []
    files: List[Dict[str, str]] = []
    for p in paths:
        with open(p, "rb") as f:
            raw = f.read()
        files.append(dict(path=p, sha256=hashlib.sha256(raw).hexdigest()))
        text = raw.decode("utf-8")
        hits += [(p, w) for w in words if w in text]
    return dict(clean=not hits, hits=hits, files=files)


def naive_pilot_baseline(Ts: Sequence[int], seeds: Sequence[int], hw_name: str, *, cfg: Optional[Dict[str, float]] = None, r: float = 0.5,
                         dev: str = "cuda") -> Dict[str, Any]:
    """REQ-S1-25: the 'always report 1' policy (v1 == 1; adapter b=1.0, because clip(u+1)=1) against the best handwritten strategy `hw_name` on the SAME T and
    the SAME seeds (paired, same scoring window).  NAIVE = M3C with p=0 (default cfg).  Train/val seeds only.
    Returns {'by_T': {T: {T_score, seeds, always_one: {G_s, G}, hw: {name, G_s, G}, diff_s, diff, always_one_ge_hw}}, 'T_change': diff(T_last) - diff(T_first)}."""
    from . import policy
    require_pilot_seeds(seeds)
    strat = dict(hw.build_s_hw()).get(hw_name)
    if strat is None:
        raise ValueError(f"{hw_name!r} is not an S_HW strategy")
    cfg = dict(env.m3c_config(r), p=0.0) if cfg is None else dict(cfg)
    seeds = [int(x) for x in seeds]
    by_T: Dict[int, Dict[str, Any]] = {}
    for T in Ts:
        out = env.simulate("M3C", int(T), cfg, [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(b=1.0)), policy.AdapterPolicy(dict(strat))],
                           seeds, r=r, dev=dev, obs_version="public")
        U = out["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float)
        ts = out["T_score"]
        g1, ghw = (U[1] - U[0]) / ts, (U[2] - U[0]) / ts
        diff_s = g1 - ghw
        diff = float(diff_s.mean())
        by_T[int(T)] = dict(T_score=ts, seeds=list(seeds), always_one=dict(G_s=g1, G=float(g1.mean())), hw=dict(name=hw_name, G_s=ghw, G=float(ghw.mean())),
                            diff_s=diff_s, diff=diff, always_one_ge_hw=bool(diff >= 0))
    keys = [int(T) for T in Ts]
    return dict(by_T=by_T, T_change=by_T[keys[-1]]["diff"] - by_T[keys[0]]["diff"])


# ------------------------------------------------------------------------------------------------------------------------ S1-28
def calibrate_knob_v13(op: str, target_size: float, category: str, eval_delta_s: Callable[[float, Sequence[int]], Dict[str, Any]],
                       grid: Sequence[float], seeds: Sequence[int]) -> Dict[str, Any]:
    """REQ-S1-28/02 (v1.3): scan the WHOLE grid (no monotonicity assumption; D4's knob is not monotone) on exactly the 64 calibration seeds (val then val2;
    anything else is a ValueError).  eval_delta_s(knob, seeds) -> {'fine': [64], 'coarse': [64]} per-seed paired increments.  Acceptance of a knob: the mean
    of the basis increment (fine for out_of_family/blind, coarse for in_family) lies in [0.8 s, 1.2 s] AND its paired 95% CI lower bound (t(63)) is > 0.
    Among acceptors the mean closest to s wins (ties -> earlier grid point).  Returns {'status': 'ok', 'knob', ...} or {'status': 'needs_commander', ...}
    without a knob (never substitutes another operator or size).  Deterministic."""
    if category not in ("out_of_family", "in_family", "blind"):
        raise ValueError(f"category {category!r} has no size to calibrate (REQ-S1-02)")
    if target_size not in SIZES:
        raise ValueError(f"size {target_size} is not a registered size {SIZES}")
    if list(seeds) != list(CALIB_SEEDS):
        raise ValueError("REQ-S1-28: calibration uses exactly the 64 seeds val (2000-2031) then val2 (3000-3031)")
    basis = size_basis(category)
    lo_b, hi_b = 0.8 * target_size, 1.2 * target_size
    best = None
    means: List[float] = []
    for k in grid:
        d = np.asarray(eval_delta_s(k, list(seeds))[basis], dtype=float).reshape(-1)
        if d.size != len(CALIB_SEEDS):
            raise ValueError(f"eval_delta_s must return {len(CALIB_SEEDS)} per-seed values")
        ci = mean_ci(d)                                              # t(63) because n = 64
        means.append(ci["mean"])
        if lo_b <= ci["mean"] <= hi_b and ci["lo"] > 0:
            err = abs(ci["mean"] - target_size)
            if best is None or err < best[0]:
                best = (err, k, ci)
    diffs = np.diff(np.asarray(means))
    out = dict(op=op, target=target_size, size_basis=basis, n_seeds=len(CALIB_SEEDS), n_evaluated=len(means),
               knob_monotone=bool(len(diffs) == 0 or (diffs >= 0).all() or (diffs <= 0).all()))
    if best is None:
        return dict(out, status="needs_commander")
    return dict(out, status="ok", knob=best[1], delta=best[2])
