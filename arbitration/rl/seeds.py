"""REQ-SEED-*: seed splits and guards.

The test split is defined here only as a range (needed for the disjointness guards); no module other than final_eval may
reference a test-seed constant (REQ-SEED-02/03).  The numeric literal of the test range is deliberately not written out.
"""
import ast
from typing import Dict, Iterable, List, Sequence, Set

_OFFSETS = (0, 10 ** 6, 2 * 10 ** 6)          # frontier's three generator seeds: s, s+1e6, s+2e6
_RANGES = {                                    # name -> (first seed, count)
    "train": (0, 1000),
    "val": (2 * 1000, 32),
    "val2": (3 * 1000, 32),                    # REQ-SEED-08 (v1.3): calibration/selection of the vulnerability knobs only; never a test set
    "test": (5 * 1000, 32),
    "replay": (1000, 32),
}
_GUARDED_LO, _GUARDED_HI = _RANGES["test"][0], _RANGES["test"][0] + _RANGES["test"][1] - 1


def splits() -> Dict[str, Sequence[int]]:
    """REQ-SEED-01/08: {'train': 0..999, 'val': 2000..2031, 'val2': 3000..3031, 'test': 5000..5031, 'replay': 1000..1031}."""
    return {k: range(a, a + n) for k, (a, n) in _RANGES.items()}


def generator_seed_set(split: str) -> Set[int]:
    """REQ-SEED-01: actual generator seeds {s, s+1e6, s+2e6} of a split."""
    if split not in _RANGES:
        raise ValueError(f"unknown split {split!r}")
    a, n = _RANGES[split]
    return {s + o for s in range(a, a + n) for o in _OFFSETS}


def require_split(seeds: Iterable[int], split: str) -> None:
    """REQ-SEED-01/08: ValueError unless all seeds belong to `split` ('train', 'val', 'val2', 'test'; replay-only seeds belong to none)."""
    if split not in _RANGES:
        raise ValueError(f"unknown split {split!r}")
    a, n = _RANGES[split]
    bad = [int(s) for s in seeds if not a <= int(s) < a + n]
    if bad:
        raise ValueError(f"REQ-SEED-01: seeds {bad[:5]} do not belong to split {split!r}")


def scan_forbidden_references(paths: Sequence[str], names: Sequence[str] = ("TEST_SEEDS", "test_seeds")) -> List[str]:
    """REQ-SEED-02/03: AST scan; return 'path:line:name' for every import/reference of a test-seed constant,
    and for every integer literal inside the test-seed range."""
    hits: List[str] = []
    for p in paths:
        with open(p) as f:
            tree = ast.parse(f.read(), filename=str(p))
        for node in ast.walk(tree):
            what = None
            if isinstance(node, ast.alias):
                base = node.name.split(".")[-1]
                if base in names:
                    what = base
            elif isinstance(node, ast.Name) and node.id in names:
                what = node.id
            elif isinstance(node, ast.Attribute) and node.attr in names:
                what = node.attr
            elif isinstance(node, ast.Constant) and type(node.value) is int and _GUARDED_LO <= node.value <= _GUARDED_HI:
                what = str(node.value)
            if what is not None:
                line = getattr(node, "lineno", None)
                if line is None:        # ast.alias has no lineno before py3.10
                    line = 0
                hits.append(f"{p}:{line}:{what}")
    return hits
