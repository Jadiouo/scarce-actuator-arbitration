"""Final evaluation on the TEST seeds.  The ONLY module allowed to reference test seeds (REQ-SEED-02/03/06/08, commander decision).

`evaluate_on_test(record, evaluator)` accepts exactly the 32 test seeds.  Seeds of any other split (train, validation, the calibration-only
split, the replay seeds, or anything else) raise ValueError BEFORE anything else is looked at, and the message names the split.  Further rules
(REQ-SEED-02, REQ-MET-06):
  * the freeze file must exist (the final evaluation happens after the freeze is signed, section 8.5);
  * every (cell, policy_id) pair is evaluated ONCE: a ledger (JSON) records finished evaluations and a second request raises RuntimeError;
  * the detection threshold is the frozen MDE of the record (`mde_frozen`, never recomputed; metrics.final_eval);
  * the per-seed gains come from `evaluator(seeds)` (a callable returning {'G_s': [32], 'D_s': [32] or None}); this module runs no simulation itself.
"""
import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import metrics, seeds as _seeds

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FREEZE = os.path.join(_ROOT, "docs", "direction2-freeze.md")
DEFAULT_LEDGER = os.path.join(_ROOT, "results", "direction2", "final_eval_ledger.json")


def _split_of(seed: int) -> str:
    for name, rng in _seeds.splits().items():
        if seed in rng:
            return name
    return "none"


def require_test_seeds(seed_list) -> list:
    """REQ-SEED-02/08: the 32 test seeds exactly (each once); ValueError naming the offending split(s) otherwise."""
    try:
        got = [int(s) for s in seed_list]
    except (TypeError, ValueError):
        raise ValueError("final_eval: seeds must be a list of integers")
    if not got:
        raise ValueError("final_eval: no seeds given")
    test = list(_seeds.splits()["test"])
    other = sorted({_split_of(s) for s in got} - {"test"})
    if other:
        raise ValueError(f"final_eval accepts test seeds only; got seeds of split(s) {other} (e.g. {[s for s in got if _split_of(s) != 'test'][:3]})")
    if len(set(got)) != len(got) or set(got) != set(test):
        raise ValueError(f"final_eval needs exactly the {len(test)} test seeds, each once (got {len(got)} seeds, {len(set(got))} distinct)")
    return sorted(got)


def _load_ledger(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def evaluate_on_test(record: Dict[str, Any], evaluator: Optional[Callable[[list], Dict[str, Any]]] = None, *,
                     freeze_path: Optional[str] = None, ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """REQ-SEED-02: evaluate one (cell, policy) on the test seeds, once, after the freeze.

    record: {'seeds': [...], 'cell': str, 'policy_id': str, 'mde_frozen': float, ['mde_d_frozen': float]}.
    Order of checks: seeds (ValueError) -> required keys (ValueError) -> freeze file (FileNotFoundError) -> ledger (RuntimeError) -> evaluator."""
    if "seeds" not in record:
        raise ValueError("final_eval: record has no 'seeds'")
    test_seeds = require_test_seeds(record["seeds"])
    missing = [k for k in ("cell", "policy_id", "mde_frozen") if k not in record]
    if missing:
        raise ValueError(f"final_eval: record lacks {missing} (the frozen MDE is never recomputed, REQ-MET-06)")
    freeze = DEFAULT_FREEZE if freeze_path is None else freeze_path
    if not os.path.exists(freeze):
        raise FileNotFoundError(f"final_eval: the freeze file {freeze} does not exist; test seeds are only used after the freeze is signed (REQ-SEED-02)")
    ledger = DEFAULT_LEDGER if ledger_path is None else ledger_path
    key = f"{record['cell']}|{record['policy_id']}"
    done = _load_ledger(ledger)
    if key in done:
        raise RuntimeError(f"final_eval: {key} was already evaluated on the test seeds (once per (cell, policy), REQ-SEED-02)")
    if evaluator is None:
        raise ValueError("final_eval: an evaluator callable is required")
    res = evaluator(list(test_seeds))
    g = np.asarray(res["G_s"], dtype=float).reshape(-1)
    d = res.get("D_s")
    d = None if d is None else np.asarray(d, dtype=float).reshape(-1)
    if g.size != len(test_seeds) or not np.isfinite(g).all() or (d is not None and (d.size != len(test_seeds) or not np.isfinite(d).all())):
        raise ValueError("final_eval: the evaluator must return finite per-seed G_s (and D_s) for every test seed")
    out = metrics.final_eval(record, g, d)
    out.update(cell=record["cell"], policy_id=record["policy_id"], seed_range=[test_seeds[0], test_seeds[-1]], n_seeds=len(test_seeds))
    done[key] = dict(seed_range=out["seed_range"], found_G=bool(out["found_G"]))
    os.makedirs(os.path.dirname(ledger) or ".", exist_ok=True)
    with open(ledger, "w") as f:
        json.dump(done, f, indent=1)
    return out


def already_evaluated(cell: str, policy_id: str, ledger_path: Optional[str] = None) -> bool:
    """True iff (cell, policy_id) is in the ledger (it was evaluated on the test seeds)."""
    return f"{cell}|{policy_id}" in _load_ledger(DEFAULT_LEDGER if ledger_path is None else ledger_path)


def evaluate(record: Dict[str, Any], evaluator: Callable[[list], Dict[str, Any]], *, freeze_path: Optional[str] = None,
             ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """Entry for the other modules (s1_eval etc.): fills in the 32 test seeds HERE, so that no other module has to name or build a test seed
    (REQ-SEED-06); everything else is evaluate_on_test (freeze file, once-per-(cell, policy) ledger, frozen MDE)."""
    return evaluate_on_test(dict(record, seeds=list(_seeds.splits()["test"])), evaluator, freeze_path=freeze_path, ledger_path=ledger_path)
