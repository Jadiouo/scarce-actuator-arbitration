"""REQ-OBS-*: feature construction for the public / announced / oracle information sets (spec s3, s4.4)."""
import copy
import math
from dataclasses import dataclass, fields
from typing import Any, List, Tuple

import torch

LAM = 0.99            # lambda_e (spec 4.4)
ONE_M = 1.0 - LAM
HIST = 8              # k
CLIP_LO, CLIP_HI = 1e-12, 1 - 1e-12   # phi_of clip for agent 1 (identity marginal)


@dataclass
class PublicState:
    """Whitelisted inputs of the public feature builder (REQ-OBS-06, REQ-OBS-08: no absolute time).  All tensors share the
    batch shape B.  z,u: own type at round t; susp_minus_t: susp_1 - t (signed; eligible iff <0); since_rounds: t - t_s
    (inf if never suspended); won_hist[B,8]/d_hist[B,8]: j=1..8 rounds back (0-padded); v_prev: own report at t-1."""
    z: Any
    u: Any
    susp_minus_t: Any
    since_rounds: Any
    won_hist: Any
    d_hist: Any
    v_prev: Any


@dataclass
class AnnouncedState(PublicState):
    """Public state + S=cus_1/h and audited_last (REQ-OBS-03)."""
    S: Any
    audited_last: Any


@dataclass
class OracleState(PublicState):
    """Public state + other agents' current types u0,u2 (REQ-OBS-04)."""
    u0: Any
    u2: Any


def public_state_fields() -> Tuple[str, ...]:
    """REQ-OBS-06: field names of the public whitelist."""
    return tuple(f.name for f in fields(PublicState))


def feature_names(version: str = "public") -> List[str]:
    """REQ-OBS-07: ['z','u','elig','rem','since','won_1'..'won_8','d_1'..'d_8','res_last','ema_res','ema_d','ema_win'] (+ extras)."""
    n = ["z", "u", "elig", "rem", "since"] + [f"won_{j}" for j in range(1, HIST + 1)] + \
        [f"d_{j}" for j in range(1, HIST + 1)] + ["res_last", "ema_res", "ema_d", "ema_win"]
    if version == "public":
        return n
    if version == "announced":
        return n + ["S", "audited_last"]
    if version == "oracle":
        return n + ["u0", "u2"]
    raise ValueError(version)


_CLS = {"public": PublicState, "announced": AnnouncedState, "oracle": OracleState}


class ObsBuilder:
    """REQ-OBS-01/07/08: stateful (EMA) feature builder.  step(state) takes ONLY a PublicState/AnnouncedState/OracleState
    (anything else, or objects carrying extra attributes, raises TypeError).  Its signature has no time argument."""

    def __init__(self, version: str, r: float, L: float, shape: Tuple[int, ...], dtype=None, device: str = "cpu",
                 c_min: float = 0.2, h: float = 6.0):
        if version not in _CLS:
            raise ValueError(version)
        self.version, self.r, self.L, self.c_min, self.h = version, float(r), float(L), c_min, h
        self.sd = math.sqrt(1.0 - self.r * self.r)
        self.shape = tuple(shape)
        self.dtype = dtype or torch.float64
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset EMA state to 0."""
        z = lambda: torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        self.ema_res, self.ema_d, self.ema_win = z(), z(), z()

    def with_version(self, version: str) -> "ObsBuilder":
        """Shallow copy sharing the (read-only) EMA state, for pure peeks at another information set."""
        b = copy.copy(self)
        b.version = version
        return b

    def _check(self, state: Any) -> None:
        if type(state) is not _CLS[self.version]:
            raise TypeError(f"{self.version} builder needs exactly {_CLS[self.version].__name__}, got {type(state).__name__}")
        allowed = {f.name for f in fields(type(state))}
        extra = set(vars(state)) - allowed
        if extra:
            raise TypeError(f"state carries non-whitelisted attributes {sorted(extra)}")

    def peek(self, state: Any):
        """Pure: (obs[...,F], (ema_res, ema_d, ema_win)) without touching the EMA state."""
        self._check(state)
        c = lambda x: x.contiguous()
        z, u, smt, since = c(state.z), c(state.u), c(state.susp_minus_t), c(state.since_rounds)
        won, dh, vp = c(state.won_hist), c(state.d_hist), c(state.v_prev)
        L = self.L
        f01 = torch.clamp(z / 3.0, -1.0, 1.0)
        f02 = 2.0 * u - 1.0
        f03 = (smt < 0).to(self.dtype)
        f04 = torch.clamp(smt, min=0.0) / L
        f05 = torch.clamp(since, max=2.0 * L) / (2.0 * L)
        won1 = won[..., 0]
        zv = torch.special.ndtri(torch.clamp(vp, CLIP_LO, CLIP_HI))
        zw = torch.special.ndtri(torch.clamp(u, CLIP_LO, CLIP_HI))
        res = (zw - self.r * zv) / self.sd
        f22 = torch.where(won1 > 0.5, torch.clamp(res / 6.0, -1.0, 1.0), torch.zeros_like(res))
        e_res = LAM * self.ema_res + ONE_M * f22
        e_d = LAM * self.ema_d + ONE_M * dh[..., 0]
        e_win = LAM * self.ema_win + ONE_M * won1
        parts = [torch.stack([f01, f02, f03, f04, f05], -1), won, dh, torch.stack([f22, e_res, e_d, e_win], -1)]
        if self.version == "announced":
            parts.append(torch.stack([c(state.S), c(state.audited_last)], -1))
        elif self.version == "oracle":
            parts.append(torch.stack([2.0 * c(state.u0) - 1.0, 2.0 * c(state.u2) - 1.0], -1))
        return torch.cat(parts, -1), (e_res, e_d, e_win)

    def step(self, state: Any) -> Any:
        """Return obs[..., F] in [-1,1]; F=25 (public), 27 (announced/oracle).  Advances the EMA state."""
        o, ema = self.peek(state)
        self.ema_res, self.ema_d, self.ema_win = ema
        return o
