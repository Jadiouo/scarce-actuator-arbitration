"""Plan B: several Direction-2 segments of DIFFERENT cells advance at the same time inside ONE process (docs/direction2-speedup-plan.md).

Not a merged batch: every cell keeps its own simulation, its own kernel sequence (the same CUDA graph it would use alone), its own RNG
generators, CMA-ES and files.  Each cell runs in its own Python thread on its own CUDA stream, so the GPU can overlap the kernels of 2-3 cells.
Because a kernel's arithmetic does not depend on what runs next to it, every cell's results are bitwise the same as when it runs alone
(tests/direction2/test_d2_multi.py).

Two entries:
  * run_concurrent(specs, ...)  : start the given segments together, return when all are done (tests, benchmarks).
  * serve()                     : long-lived worker for `scripts/d2_run_queue.py --concurrent N`.  Reads JSON lines on stdin
        {"op": "run", "id": 7, "cell": "s1_D2_s0.02", "run": 0, "g0": 36, "g1": 69, "log": "<path>"}   (at most one running segment per cell)
        {"op": "quit"}
    and writes one JSON line per finished segment on stdout  {"event": "done", "id": 7, "ok": true, "status": "done"}  or  {"event": "done", "id": 7,
    "ok": false, "error": "..."}.  Part files and checkpoints are written by train.train_segment exactly as in the one-segment entry; committing and
    pushing stay with the runner.
"""
import argparse
import json
import os
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import train


def die_with_parent(sig: int = 9) -> bool:
    """Linux prctl(PR_SET_PDEATHSIG): the kernel sends `sig` (SIGKILL) to this process when its parent dies, so a worker never outlives a runner that
    was SIGKILLed / OOM-killed (the restarted runner would run the same segments a second time).  False where unavailable (not Linux)."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        if libc.prctl(1, sig, 0, 0, 0) != 0:                  # PR_SET_PDEATHSIG = 1
            return False
        return os.getppid() != 1                              # parent already gone before the call: nothing will ever signal us
    except (OSError, AttributeError):
        return False


class Lane:
    """The resources one cell keeps between its segments: a CUDA stream and a GraphSim (graph cache) that was built on that stream."""

    def __init__(self, graph: bool, K: int = 32, capture: Optional[bool] = None, dev: str = "cuda"):
        import torch
        from . import graphsim
        self.stream = torch.cuda.Stream() if str(dev).startswith("cuda") else None
        self.gs = graphsim.GraphSim(K=K, capture=capture, stream=self.stream) if graph else None

    def release(self) -> None:
        if self.gs is not None:
            self.gs.cache.clear()


def _args(spec: Dict[str, Any], dev: str):
    a = train.make_parser().parse_args(["--cell", spec["cell"], "--g1", str(spec["g1"])])
    a.plan = spec.get("plan") or "results/direction2/run_plan.json"
    a.run, a.g0, a.dev = int(spec.get("run", 0)), int(spec.get("g0", 0)), dev
    a.root, a.test_mode, a.start = spec.get("root"), bool(spec.get("test_mode", False)), spec.get("start")
    return a


def run_segment(spec: Dict[str, Any], lane: Lane, dev: str = "cuda", log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """One segment on `lane` (call from the lane's thread).  Same guards and same training function as `python -m arbitration.rl.train`."""
    import torch
    log = log or (lambda s: None)
    if lane.stream is not None:
        ctx = torch.cuda.stream(lane.stream)
        ctx.__enter__()
    try:
        j = train.prepare(_args(spec, dev), lane.gs)
        return train.train_segment(j["cell"], j["run"], j["g0"], j["g1"], j["sim_fn"], j["val_fn"], root=j["root"], theta0=j["theta0"], log=log)
    finally:
        if lane.stream is not None:
            torch.cuda.current_stream().synchronize()
            ctx.__exit__(None, None, None)


def _logger(path: Optional[str]) -> Callable[[str], None]:
    if not path:
        return lambda s: None
    lock = threading.Lock()

    def log(s: str) -> None:
        with lock, open(path, "a") as f:
            f.write(s + "\n")
    return log


def _guard(fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        rec = fn()
        return dict(ok=True, status=rec.get("status"), rec=rec)
    except SystemExit as e:                          # the entry's guards refuse with SystemExit(message)
        return dict(ok=False, error=f"refused: {e}")
    except BaseException as e:                       # noqa: BLE001 - a segment must never take the other lanes down
        return dict(ok=False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")


def run_concurrent(specs: Sequence[Dict[str, Any]], graph: bool = True, K: int = 32, capture: Optional[bool] = None, dev: str = "cuda",
                   lanes: Optional[Dict[str, Lane]] = None) -> List[Dict[str, Any]]:
    """Run the segments (distinct cells) at the same time, one thread + one stream each.  Returns one dict(ok, status|error, rec) per spec, in order."""
    cells = [s["cell"] for s in specs]
    if len(set(cells)) != len(cells):
        raise ValueError("concurrent segments must belong to different cells")
    lanes = {} if lanes is None else lanes
    out: List[Optional[Dict[str, Any]]] = [None] * len(specs)

    def work(i: int, spec: Dict[str, Any]) -> None:
        lane = lanes.get(spec["cell"])
        if lane is None:
            lane = lanes[spec["cell"]] = Lane(graph, K, capture, dev)
        out[i] = _guard(lambda: run_segment(spec, lane, dev, _logger(spec.get("log"))))
    threads = [threading.Thread(target=work, args=(i, s), name=f"lane-{s['cell']}") for i, s in enumerate(specs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out                                        # type: ignore[return-value]


def _hard_exit() -> None:
    os._exit(1)


def serve(inp=None, outp=None, graph: bool = True, K: int = 32, dev: str = "cuda", max_lanes: int = 3, capture: Optional[bool] = None,
          run_fn: Optional[Callable[..., Dict[str, Any]]] = None, on_orphan: Optional[Callable[[], Any]] = None) -> int:
    """The long-lived worker (see module doc).  Returns 0 after a 'quit' once every running segment has finished.  stdin EOF WITHOUT a 'quit' means the
    runner died: `on_orphan` (default: os._exit(1)) is called at once, the running segments are abandoned (a restarted runner redoes them from their
    checkpoints), and 1 is returned."""
    inp = inp or sys.stdin
    outp = outp or sys.stdout
    run_fn = run_fn or run_segment
    lanes: Dict[str, Lane] = {}
    running: Dict[str, threading.Thread] = {}
    wlock = threading.Lock()

    def emit(obj: Dict[str, Any]) -> None:
        with wlock:
            outp.write(json.dumps(obj, default=str) + "\n")
            outp.flush()

    def reap() -> None:
        for c in [c for c, t in running.items() if not t.is_alive()]:
            running.pop(c)

    quit_seen = False
    for line in inp:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg["op"] == "quit":
            quit_seen = True
            break
        if msg["op"] != "run":
            emit(dict(event="error", error=f"unknown op {msg['op']!r}"))
            continue
        reap()
        cell = msg["cell"]
        if cell in running:
            emit(dict(event="done", id=msg["id"], ok=False, error=f"cell {cell} already has a running segment"))
            continue
        if cell not in lanes:
            idle = [c for c in lanes if c not in running]
            while len(lanes) >= max_lanes and idle:
                lanes.pop(idle.pop(0)).release()                 # evict an idle cell's graphs (its next segment rebuilds them)
            lanes[cell] = Lane(graph, K, capture, dev)
        lane = lanes[cell]

        def work(m=msg, ln=lane) -> None:
            res = _guard(lambda: run_fn(m, ln, dev, _logger(m.get("log"))))
            emit(dict(event="done", id=m["id"], cell=m["cell"], ok=res["ok"], status=res.get("status"), error=res.get("error")))
        t = threading.Thread(target=work, name=f"lane-{cell}")
        running[cell] = t
        t.start()
    if not quit_seen:
        (on_orphan or _hard_exit)()
        return 1
    for t in list(running.values()):
        t.join()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="long-lived worker for scripts/d2_run_queue.py --concurrent N (JSON lines on stdin/stdout)")
    ap.add_argument("--no-graph", action="store_true", help="eager stepping in every lane (default: CUDA graphs)")
    ap.add_argument("--K", type=int, default=int(os.environ.get("D2_GRAPH_K", "32")))
    ap.add_argument("--max-lanes", type=int, default=3)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--job", action="append", default=[], metavar="CELL:RUN:G0:G1", help="without --serve: run these segments together and exit")
    a = ap.parse_args(argv)
    if a.serve:
        die_with_parent()
        return serve(graph=not a.no_graph, K=a.K, dev=a.dev, max_lanes=a.max_lanes)
    specs = []
    for j in a.job:
        c, r, g0, g1 = j.split(":")
        specs.append(dict(cell=c, run=int(r), g0=int(g0), g1=int(g1)))
    res = run_concurrent(specs, graph=not a.no_graph, K=a.K, dev=a.dev)
    for s, r in zip(specs, res):
        print(json.dumps(dict(spec=s, ok=r["ok"], status=r.get("status"), error=r.get("error"))))
    return 0 if all(r["ok"] for r in res) else 2


if __name__ == "__main__":
    sys.exit(main())
