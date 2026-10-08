"""CUDA-graph stepping of the Direction-2 simulator (docs/direction2-speedup-plan.md, plan A).

The frozen experiment must produce BITWISE the same numbers as the eager simulator (`env._step` / `vulns._vstep`).  So nothing is re-implemented
here: the graph *replays the very same eager step function*, K steps unrolled, on static buffers.

 * Randomness never enters the graph.  Every 1024 steps the driver draws the RNG block eagerly (`env._ensure_block`: same generators, same call
   order as the eager loop) and `copy_`s the K rows of the coming replay into static feed buffers; the graph reads `feed.E[i][sidx]` etc., which is
   the same indexing op the eager step applies to `st.E[s][sidx]`.
 * `t` is a Python int in the eager step.  Inside the graph it is a float64 0-dim device tensor (`feed.tt`, from `t_dev + arange(K)`); every use of
   `t` in the step is a float64 comparison/add/sub, which gives the same bits for a Python number and for a float64 scalar tensor.  The only integer
   uses (`t % M`, M=2) are static because a replay always starts at an ODD step and K is even; `st.t` is just a Python counter during capture.
 * The state (SimState / bank / builder / HS / extension tensors) lives in static buffers.  The unrolled function ends with `static.copy_(new)` for
   every tensor the step rebinds, then rebinds the attribute to the static buffer.  Each new generation builds its fresh state with the unchanged eager
   code and `copy_`s it into the statics; steps that do not fit a replay (block ends, the score-snapshot step, parity) run eagerly on the same objects.
 * `plan_events(T, Ts, K)` is the pure scheduler.

`GraphSim(capture=False)` runs the identical machinery without CUDA graphs (on any device): it tests everything except `torch.cuda.CUDAGraph`.
"""
import os
import threading
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import env as _env

BLK = _env.BLK
_ST_SKIP = frozenset({"E", "Ua", "Rn", "Rx"})          # RNG blocks: fed, not state
# Building a graph (warm-up + capture) runs device-wide synchronisations (torch.cuda.synchronize / the allocator) that are illegal while ANOTHER thread
# is capturing: lanes of one process build one at a time (replaying and eager work of the other lanes may go on meanwhile).
_BUILD_LOCK = threading.Lock()


def plan_events(T: int, Ts: int, K: int, blk: int = BLK) -> List[Tuple[str, int, int]]:
    """Pure scheduler.  Returns [(kind, first_step, n_steps)] covering steps 1..T in order.  'replay' = K steps of the graph (never across an RNG
    block boundary, never across the snapshot step Ts, always starting at an odd step so that t % 2 is static); 'eager' = one step run eagerly."""
    if K < 2 or K % 2:
        raise ValueError("K must be an even number >= 2")
    ev: List[Tuple[str, int, int]] = []
    stops = sorted({b for b in range(blk, T + 1, blk)} | {T} | ({Ts} if 1 <= Ts <= T else set()))
    a = 1
    for b in stops:
        while a <= b:
            if a % 2 == 1 and b - a + 1 >= K:
                ev.append(("replay", a, K))
                a += K
            else:
                ev.append(("eager", a, 1))
                a += 1
    return ev


def _get(c, k, item):
    return c[k] if item else getattr(c, k)


def _set(c, k, item, v):
    if item:
        c[k] = v
    else:
        setattr(c, k, v)


def _collect(objs: Dict[str, Any]) -> List[Tuple[Any, str, bool, Any]]:
    """Every tensor reachable from the simulation objects: (container, key, is_dict_item, tensor), in a deterministic order."""
    out = []

    def scan(name, obj):
        if obj is None:
            return
        d = obj if isinstance(obj, dict) else vars(obj)
        for k, v in d.items():
            if name == "st" and k in _ST_SKIP:
                continue
            if torch.is_tensor(v):
                out.append((obj, k, isinstance(obj, dict), v))
    st, bank, hs = objs["st"], objs["bank"], objs.get("hs")
    scan("st", st)
    scan("builder", st.builder)
    scan("bank", bank)
    scan("el", bank.el)
    scan("hs", hs)
    scan("ext", getattr(hs, "ext", None))
    scan("holder", objs.get("holder"))
    return out


class _Runner:
    """One captured K-step graph for one structure (operator/knob/shapes), reused by every generation."""

    def __init__(self, gs: "GraphSim", objs, step_fn, reg):
        self.gs, self.K = gs, gs.K
        self.objs, self.step_fn = objs, step_fn
        st = objs["st"]
        dev = st.dev
        self.dev = dev
        self.capture = gs.capture if gs.capture is not None else str(dev).startswith("cuda")
        self.statics = [e[3].clone() for e in reg]
        self.cap_reg = reg
        self.reg = reg
        K = self.K
        f64 = torch.float64
        self.feed = types.SimpleNamespace(
            E=torch.zeros((K,) + tuple(st.E.shape[1:]), dtype=f64, device=dev), Ua=torch.zeros((K,) + tuple(st.Ua.shape[1:]), dtype=f64, device=dev),
            Rn=torch.zeros((K,) + tuple(st.Rn.shape[1:]), dtype=f64, device=dev), Rx=torch.zeros((K,) + tuple(st.Rx.shape[1:]), dtype=f64, device=dev),
            bad=torch.zeros((), dtype=torch.bool, device=dev), i=0, last=False, tt=None)
        self.t_dev = torch.zeros((), dtype=f64, device=dev)
        self.arK = torch.arange(K, dtype=f64, device=dev)
        self.graph = None
        with _BUILD_LOCK:
            self._build()

    # ---- binding of the (per generation) objects to the static buffers
    def bind(self, reg) -> None:
        """Copy a fresh generation's tensors into the statics and make the objects use the statics."""
        if len(reg) != len(self.statics):
            raise RuntimeError("graph stepping: the simulation objects have a different set of tensors than the captured graph")
        for (c, k, item, v), s in zip(reg, self.statics):
            if v.shape != s.shape or v.dtype != s.dtype or v.device != s.device:
                raise RuntimeError(f"graph stepping: tensor {k} changed shape/dtype ({tuple(v.shape)} vs {tuple(s.shape)})")
            if v is not s:
                s.copy_(v)
                _set(c, k, item, s)
        self.reg = reg

    def sync_in(self) -> None:
        """Eager steps rebind attributes to new tensors: copy them back into the statics before a replay."""
        for (c, k, item, _), s in zip(self.reg, self.statics):
            v = _get(c, k, item)
            if v is not s:
                s.copy_(v)
                _set(c, k, item, s)

    # ---- the unrolled function (captured, or run eagerly when capture=False)
    def _unrolled(self) -> None:
        st = self.objs["st"]
        fd = self.feed
        K = self.K
        st.t = 0
        tvec = self.t_dev + self.arK
        st._feed = fd
        try:
            for i in range(K):
                fd.i, fd.tt, fd.last = i, tvec[i], (i == K - 1)
                self.step_fn()
        finally:
            st._feed = None
        for (c, k, item, _), s in zip(self.cap_reg, self.statics):
            v = _get(c, k, item)
            if v is not s and any(v is x for x in self.statics):
                raise RuntimeError(f"graph stepping: attribute {k} aliases another static buffer")
        for (c, k, item, _), s in zip(self.cap_reg, self.statics):
            v = _get(c, k, item)
            if v is not s:
                s.copy_(v)
                _set(c, k, item, s)

    def _fill_feed(self, st, s0: int) -> None:
        K, fd = self.K, self.feed
        fd.E.copy_(st.E[s0:s0 + K])
        fd.Ua.copy_(st.Ua[s0:s0 + K])
        fd.Rn.copy_(st.Rn[s0:s0 + K])
        fd.Rx.copy_(st.Rx[s0:s0 + K])

    def _build(self) -> None:
        st = self.objs["st"]
        self.bind(self.reg)
        backup = [s.clone() for s in self.statics]
        self._fill_feed(st, 0)
        self.t_dev.fill_(1.0)
        if not self.capture:
            self._restore(backup)
            self._call = self._emulated
            return
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):                                  # warm-up on a side stream (lazy kernel loading, workspaces), then restore the state
                self._unrolled()
                self._restore(backup)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode=self.gs.capture_mode):
            self._unrolled()
        torch.cuda.synchronize()
        self._restore(backup)
        self.graph = g
        self._call = g.replay

    def _emulated(self) -> None:
        """capture=False: run the unrolled function directly.  A real graph only knows device addresses; here the build-time objects must be
        pointed at the statics again (they were left on whatever tensors their last eager steps produced)."""
        for (c, k, item, _), s in zip(self.cap_reg, self.statics):
            _set(c, k, item, s)
        self._unrolled()

    def _restore(self, backup) -> None:
        for s, b in zip(self.statics, backup):
            s.copy_(b)
        self.objs["st"].t = 0
        self.feed.bad.zero_()
        for (c, k, item, _), s in zip(self.cap_reg, self.statics):
            if _get(c, k, item) is not s:
                _set(c, k, item, s)

    # ---- per replay
    def replay(self, st) -> None:
        t0 = st.t
        self.sync_in()
        self._fill_feed(st, t0 % BLK)
        self.t_dev.fill_(float(t0 + 1))
        self._call()
        st.t = t0 + self.K                       # (without capture the unrolled function itself ran on the build-time objects and reset their t)


class GraphSim:
    """Configuration + cache of captured graphs.  One instance per CUDA stream / thread (a graph and its statics belong to the stream they were
    built on).  K = steps per replay (even).  capture=None -> capture iff the simulation device is CUDA."""

    def __init__(self, K: int = 32, capture: Optional[bool] = None, capture_mode: str = "thread_local", stream=None):
        if K < 2 or K % 2:
            raise ValueError("K must be an even number >= 2")
        self.K, self.capture, self.capture_mode, self.stream = int(K), capture, capture_mode, stream
        self.cache: Dict[Any, _Runner] = {}
        self.stats = dict(replays=0, eager_steps=0, builds=0)

    def run(self, ident: Tuple, objs: Dict[str, Any], step_fn: Callable[[], Any], T: int, Ts: int, fields: Sequence[str]):
        """Run steps 1..T of the freshly initialised simulation `objs` (st, bank, hs[, holder]); `step_fn()` = one eager step on exactly these
        objects.  Returns (snap at step Ts, final) as dicts of cloned tensors, like the eager loops."""
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _Null()
        with ctx:
            return self._run(ident, objs, step_fn, T, Ts, fields)

    def _run(self, ident, objs, step_fn, T, Ts, fields):
        st = objs["st"]
        if st._feed is not None or st.t != 0:
            raise RuntimeError("graph stepping needs a fresh state (t = 0)")
        _env._ensure_block(st, 0)                       # eager order: z0 (at SimState init) -> block 0 -> ...
        reg = _collect(objs)
        sig = tuple((k, tuple(v.shape), str(v.dtype), str(v.device)) for _, k, _, v in reg)
        key = ident + (self.K, self.capture, sig, tuple(st.E.shape), tuple(st.Ua.shape), tuple(st.Rn.shape), tuple(st.Rx.shape))
        r = self.cache.get(key)
        if r is None:
            r = _Runner(self, objs, step_fn, reg)
            self.cache[key] = r
            self.stats["builds"] += 1
        r.bind(reg)
        r.feed.bad.zero_()
        snap = None
        for kind, a, n in plan_events(T, Ts, self.K):
            if (a - 1) % BLK == 0:
                _env._ensure_block(st, a - 1)
            if kind == "replay":
                r.replay(st)
                self.stats["replays"] += 1
            else:
                step_fn()
                self.stats["eager_steps"] += 1
            end = a + n - 1
            if end == Ts:
                snap = {f: getattr(st, f).clone() for f in fields}
            if end % BLK == 0 or end == T:
                if bool(r.feed.bad):
                    raise FloatingPointError("policy action d is not finite (NaN/inf)")
        final = {f: getattr(st, f).clone() for f in fields}
        return snap, final


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def from_env(dev: str, force: bool = False) -> Optional[GraphSim]:
    """The switch.  D2_GRAPH=1 (or force=True, e.g. the --graph option) enables graph stepping; D2_GRAPH_K sets K (default 32).  Off by default.
    A request for graphs on a non-CUDA device is an error (never a silent fallback)."""
    on = force or os.environ.get("D2_GRAPH", "0") not in ("", "0", "false", "False")
    if not on:
        return None
    if not str(dev).startswith("cuda"):
        raise RuntimeError("graph stepping (D2_GRAPH / --graph) needs a CUDA device")
    return GraphSim(K=int(os.environ.get("D2_GRAPH_K", "32")), capture=True)
