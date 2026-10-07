"""REQ-GPU-01/02/05/06/07: gpujob manifest and [gpu]-test rules."""
import ast
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Sequence

MAX_JOB_S = 1500.0                                  # REQ-GPU-02: 25 minutes
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = re.compile(r"(^|/)python[0-9.]*$")


def job_within_limit(est_s: float, max_s: float = MAX_JOB_S) -> bool:
    """REQ-GPU-02: a job must be estimated at <= 25 minutes."""
    return float(est_s) <= max_s


def _direct_python(cmd: str) -> bool:
    """True if any shell segment of `cmd` (split on && || ; |) other than the gpujob-wrapped first one starts with python."""
    segs = [s.strip() for s in re.split(r"&&|\|\||;|\|", cmd) if s.strip()]
    for i, seg in enumerate(segs):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):          # leading VAR=value
            toks = toks[1:]
        if not toks:
            continue
        if i == 0 and toks[0] == "gpujob":
            continue
        if _PY.search(toks[0]) or (toks[0] == "env" and any(_PY.search(t) for t in toks[1:3])):
            return True
    return False


def validate_manifest(entries: Sequence[Dict[str, Any]]) -> List[str]:
    """REQ-GPU-01/02/06: entries {'cmd': str, 'est_s': float}; violations: not starting with 'gpujob ', no
    'results/direction2/parts/' path, est_s>1500, contains 'gpujob slots', or a direct python call."""
    out: List[str] = []
    for i, e in enumerate(entries):
        cmd, est = str(e.get("cmd", "")), e.get("est_s")
        if not cmd.startswith("gpujob "):
            out.append(f"entry {i}: does not start with 'gpujob '")
        if "results/direction2/parts/" not in cmd:
            out.append(f"entry {i}: no results/direction2/parts/ path")
        if est is None or float(est) > MAX_JOB_S:
            out.append(f"entry {i}: est_s {est} > {MAX_JOB_S:.0f}")
        if re.search(r"gpujob\s+slots", cmd):
            out.append(f"entry {i}: 'gpujob slots' is forbidden")
        if _direct_python(cmd):
            out.append(f"entry {i}: direct python call outside gpujob")
    return out


def gpu_test_job_commands(test_ids: Sequence[str]) -> List[str]:
    """REQ-GPU-01/07: one 'gpujob <python> -m pytest <id>' command per [gpu] test id."""
    return [f"gpujob {sys.executable} -m pytest {tid}" for tid in test_ids]


_FORBIDDEN = ("skip", "skipif", "importorskip", "xfail")


def _is_gpu_marked(fn: ast.AST) -> bool:
    return any("mark" in ast.dump(d) and "gpu" in ast.dump(d) for d in getattr(fn, "decorator_list", []))


def scan_gpu_tests(test_dir: str) -> List[str]:
    """REQ-GPU-07: AST scan of [gpu]-marked tests; violations for pytest.skip/skipif/importorskip/xfail."""
    out: List[str] = []
    for f in sorted(os.listdir(test_dir)):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        path = os.path.join(test_dir, f)
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test_") and _is_gpu_marked(fn):
                for node in ast.walk(fn):
                    name = node.attr if isinstance(node, ast.Attribute) else (node.id if isinstance(node, ast.Name) else None)
                    if name in _FORBIDDEN:
                        out.append(f"{path}:{node.lineno}:{fn.name}:{name}")
    return out


def run_gpu_tests_without_cuda(test_dir: str) -> Dict[str, int]:
    """REQ-GPU-07: run `-m gpu` in a subprocess with CUDA hidden; return {'passed','failed','skipped','xfailed','errors'}."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    p = subprocess.run([sys.executable, "-m", "pytest", test_dir, "-m", "gpu", "-q", "--tb=no", "-p", "no:cacheprovider"],
                       cwd=_ROOT, env=env, capture_output=True, text=True, timeout=900)
    res = dict(passed=0, failed=0, skipped=0, xfailed=0, errors=0)
    key = {"passed": "passed", "failed": "failed", "skipped": "skipped", "xfailed": "xfailed", "error": "errors", "errors": "errors"}
    for n, w in re.findall(r"(\d+) (passed|failed|skipped|xfailed|errors?)\b", p.stdout.splitlines()[-1] if p.stdout else ""):
        res[key[w]] += int(n)
    return res


def check_report(junit_xml: str) -> bool:
    """REQ-GPU-07: False if the JUnit report contains any skipped [gpu] test (the XML has no marker: any skip counts)."""
    root = ET.parse(junit_xml).getroot() if os.path.exists(junit_xml) else ET.fromstring(junit_xml)
    return not any(tc.find("skipped") is not None for tc in root.iter("testcase"))
