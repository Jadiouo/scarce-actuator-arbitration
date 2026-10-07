"""REQ-MET-09 / REQ-STOP-05: number registry check."""
import json
import math
import os
import re
from typing import Any, Dict, List, Sequence

_TOKEN = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def _parse_path(path: str) -> List[Any]:
    """'main[12].Gmax_uninformed[0]' -> ['main', 12, 'Gmax_uninformed', 0]."""
    out: List[Any] = []
    pos = 0
    for m in re.finditer(r"\.?([^.\[\]]+)|\[(-?\d+)\]", path):
        if m.start() != pos:
            raise KeyError(path)
        pos = m.end()
        out.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
    if pos != len(path) or not out:
        raise KeyError(path)
    return out


def _resolve(doc: Any, path: str) -> Any:
    cur = doc
    for tok in _parse_path(path):
        if isinstance(tok, int):
            if not isinstance(cur, (list, tuple)) or not -len(cur) <= tok < len(cur):
                raise KeyError(path)
            cur = cur[tok]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                raise KeyError(path)
            cur = cur[tok]
    return cur


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    a, b = float(a), float(b)
    if a == b:
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12 * abs(b)


def check_registry(entries: Sequence[Dict[str, Any]], root: str = ".") -> List[Dict[str, Any]]:
    """Entries {'name','value','file','path'} (path like "main[12].Gmax_uninformed[0]").  Return a list of problems
    (each {'name','problem'}): 'missing_path' if the path does not resolve in the JSON file, 'mismatch' if value != JSON value."""
    cache: Dict[str, Any] = {}
    problems: List[Dict[str, Any]] = []
    for e in entries:
        name = e.get("name")
        f = os.path.join(root, e["file"])
        try:
            if f not in cache:
                with open(f) as fh:
                    cache[f] = json.load(fh)
            ref = _resolve(cache[f], e["path"])
        except (KeyError, FileNotFoundError, json.JSONDecodeError):
            problems.append({"name": name, "problem": "missing_path"})
            continue
        if not _same(e["value"], ref):
            problems.append({"name": name, "problem": "mismatch", "registry": e["value"], "json": ref})
    return problems
