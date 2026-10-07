"""REQ-ACT-*: policy classes.  Parameter layout is private; tests build networks through from_blocks().

Layout of theta (flat): [W1 (F*hidden, row-major [F,hidden]), b1 (hidden), W2 (hidden), b2 (1)].
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

F64 = torch.float64
P_MAX = 64
P_DIAG = 300


def n_params(F: int, hidden: int = 2) -> int:
    """REQ-ACT-02: F*hidden + hidden + hidden + 1 (F=25,hidden=2 -> 55)."""
    return F * hidden + hidden + hidden + 1


def param_limit(diagnostic: bool = False) -> int:
    """REQ-ACT-03: P_MAX=64 (P_DIAG=300 for diagnostics)."""
    return P_DIAG if diagnostic else P_MAX


def report(u1: Any, d: Any) -> Any:
    """REQ-ACT-01: v1 = clip(u1 + d, 0, 1)  (d=0 => v1==u1 bitwise)."""
    return torch.clamp(u1 + d, 0.0, 1.0)


def mlp_forward(obs, W1, b1, W2, b2):
    """d = tanh(W2 . tanh(obs W1 + b1) + b2).  Works for unbatched weights (W1[F,h], b1[h], W2[h], b2[1]) and for
    per-element weights (W1[B,F,h], b1[B,h], W2[B,h], b2[B,1]) with obs[B,F].  Elementwise multiply + sum only (no BLAS)."""
    if W1.dim() == 2:
        W1, b1, W2, b2 = W1.expand(obs.shape[:-1] + W1.shape), b1, W2, b2
    hid = torch.tanh((obs.unsqueeze(-1) * W1).sum(-2) + b1)
    o = (hid * W2).sum(-1) + b2[..., 0]
    return torch.tanh(o)


class MLPPolicy:
    """REQ-ACT-02: F -> hidden(tanh) -> 1 linear, then d = tanh(o).  Raises ValueError if n_params exceeds the limit
    (64, or 300 when diagnostic=True).  theta=None means all-zero parameters (honest, REQ-ACT-04)."""

    def __init__(self, F: int = 25, hidden: int = 2, theta: Optional[Any] = None, diagnostic: bool = False,
                 obs_version: Optional[str] = None):
        # obs_version=None: untagged (legacy); env then checks F only.  An explicit tag is enforced by env._Bank.
        if obs_version not in (None, "public", "announced", "oracle"):
            raise ValueError(f"unknown obs_version {obs_version!r}")
        n = n_params(F, hidden)
        if n > param_limit(diagnostic):
            raise ValueError(f"REQ-ACT-03: {n} parameters exceed the limit {param_limit(diagnostic)}")
        self.F, self.hidden, self.diagnostic, self.obs_version = F, hidden, diagnostic, obs_version
        if theta is None:
            th = np.zeros(n, dtype=np.float64)
        else:
            th = np.asarray(theta, dtype=np.float64).reshape(-1).copy()
            if th.size != n:
                raise ValueError(f"theta has {th.size} entries, expected {n}")
            if not np.isfinite(th).all():
                raise ValueError("theta contains NaN/inf")
        self._theta = th

    @classmethod
    def from_blocks(cls, W1: Any, b1: Any, W2: Any, b2: Any, **kw) -> "MLPPolicy":
        """Build from W1[F,hidden], b1[hidden], W2[hidden], b2[1] (layout-independent)."""
        W1 = np.asarray(W1, dtype=np.float64)
        F, h = W1.shape
        theta = np.concatenate([W1.reshape(-1), np.asarray(b1, np.float64).reshape(-1), np.asarray(W2, np.float64).reshape(-1),
                                np.asarray(b2, np.float64).reshape(-1)])
        return cls(F=F, hidden=h, theta=theta, **kw)

    @property
    def n_params(self) -> int:
        return self._theta.size

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    def blocks(self):
        """(W1[F,h], b1[h], W2[h], b2[1]) as float64 numpy arrays."""
        F, h, th = self.F, self.hidden, self._theta
        a = F * h
        return th[:a].reshape(F, h), th[a:a + h], th[a + h:a + 2 * h], th[a + 2 * h:a + 2 * h + 1]

    def action(self, obs: Any) -> Any:
        """d = tanh(o) in (-1,1) for obs[..., F]."""
        if not np.isfinite(self._theta).all():      # _theta may have been edited in place after construction
            raise ValueError("theta contains NaN/inf")
        W1, b1, W2, b2 = (torch.as_tensor(x, dtype=F64, device=obs.device) for x in self.blocks())
        lead = obs.shape[:-1]
        flat = obs.reshape(-1, self.F)
        d = mlp_forward(flat, W1, b1, W2, b2)
        if obs.device.type == "cpu" and not bool(torch.isfinite(d).all()):   # no forced sync on cuda
            raise FloatingPointError("action d is not finite (NaN/inf)")
        return d.reshape(lead)


class AdapterPolicy:
    """REQ-ENV-04: wraps a frontier SCOLS strategy dict ({} = honest); evaluated with frontier's exact operation order.
    NOTE: adapters are the hand-written baselines of direction 1; some of them (edge_oracle, stealth) read private state, by
    design.  They run through env's frontier code path, never through the observation interface."""

    def __init__(self, scols: Dict[str, float]):
        from arbitration.gpu.frontier import SCOLS
        bad = set(scols) - set(SCOLS)
        if bad:
            raise ValueError(f"unknown strategy columns {sorted(bad)} (not frontier SCOLS)")
        self.scols = dict(scols)
