"""REQ-GPU-03/04: part files and job idempotence."""
import json
import os
import tempfile
from typing import Any, Dict

REQUIRED_FIELDS = ("cell", "run_id", "gen0", "gen1", "status", "fitness_mean", "fitness_best", "G_val", "sigma", "t_step_ms",
                   "wall_s", "git_sha", "spec_version")


def part_path(cell: str, run_id: int, g0: int, g1: int, root: str = ".") -> str:
    """REQ-GPU-03: <root>/results/direction2/parts/{cell}__run{k}__g{g0:04d}-{g1:04d}.json."""
    return os.path.join(root, "results", "direction2", "parts", f"{cell}__run{run_id}__g{g0:04d}-{g1:04d}.json")


def ckpt_path(cell: str, run_id: int, root: str = ".") -> str:
    """REQ-GPU-03: <root>/results/direction2/ckpt/{cell}__run{k}.npz."""
    return os.path.join(root, "results", "direction2", "ckpt", f"{cell}__run{run_id}.npz")


def write_part(path: str, record: Dict[str, Any]) -> None:
    """REQ-GPU-03: atomic (write temp file then os.replace); ValueError if required fields are missing."""
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise ValueError(f"REQ-GPU-03: part record misses fields {missing}")
    if record["status"] not in ("done", "partial"):
        raise ValueError(f"REQ-GPU-03: status must be 'done' or 'partial', got {record['status']!r}")
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".part-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, default=float)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_part(path: str) -> Dict[str, Any]:
    """REQ-GPU-03."""
    with open(path) as f:
        return json.load(f)


def job_action(path: str, ckpt_path: str = None) -> str:
    """REQ-GPU-04: 'skip' if part exists with status 'done'; 'resume' if a checkpoint exists; else 'run'."""
    if os.path.exists(path):
        try:
            if read_part(path).get("status") == "done":
                return "skip"
        except (ValueError, OSError):
            pass                                   # unreadable/partial file: rerun
    if ckpt_path is not None and os.path.exists(ckpt_path):
        return "resume"
    return "run"
