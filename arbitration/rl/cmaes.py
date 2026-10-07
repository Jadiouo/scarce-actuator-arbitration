"""REQ-OPT-01/04: CMA-ES family (full covariance for n<=50, sep-CMA for n>50) with mirrored sampling and resume."""
import math
from typing import Any, Dict

import numpy as np

_N_FULL_MAX = 50
_N_SEP_MAX = 64
_N_DIAG_MAX = 300


class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES (Hansen) with mirrored sampling; maximisation.  kind 'full' keeps a full covariance
    matrix, kind 'sep' a diagonal one (Ros & Hansen 2008: learning rates c1, c_mu scaled by (n+2)/3)."""

    def __init__(self, n: int, lam: int, sigma0: float, seed: int, kind: str):
        self.n, self.lam, self.kind = n, lam, kind
        self.mean = np.zeros(n)
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)
        mu = lam // 2
        w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.mu, self.w = mu, w / w.sum()
        self.mueff = 1.0 / float((self.w ** 2).sum())
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        c1 = 2 / ((n + 1.3) ** 2 + me)
        cmu = min(1 - c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        if kind == "sep":
            f = (n + 2) / 3.0
            c1, cmu = min(1.0, c1 * f), min(cmu * f, 1.0)
            if c1 + cmu > 1.0:
                cmu = 1.0 - c1
        self.c1, self.cmu = c1, cmu
        self.damps = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C = np.eye(n) if kind == "full" else np.ones(n)
        self.gen = 0
        self._eig()

    # ---- helpers
    def _eig(self) -> None:
        if self.kind == "full":
            C = (self.C + self.C.T) / 2
            self.C = C
            d2, B = np.linalg.eigh(C)
            d = np.sqrt(np.maximum(d2, 1e-300))
            self._B, self._D = B, d
        else:
            self._D = np.sqrt(self.C)

    def _transform(self, z: np.ndarray) -> np.ndarray:
        if self.kind == "full":
            return (z * self._D) @ self._B.T
        return z * self._D

    def _inv_sqrt(self, y: np.ndarray) -> np.ndarray:
        if self.kind == "full":
            return ((y @ self._B) / self._D) @ self._B.T
        return y / self._D

    # ---- API
    def ask(self) -> np.ndarray:
        """lam x n; rows 2j = mean + delta_j, rows 2j+1 = mean - delta_j (REQ-OPT-04)."""
        m = self.lam // 2
        z = self.rng.standard_normal((m, self.n))
        delta = self.sigma * self._transform(z)
        X = np.empty((self.lam, self.n))
        X[0::2] = self.mean[None, :] + delta
        X[1::2] = self.mean[None, :] - delta
        return X

    def tell(self, X: np.ndarray, fitness) -> None:
        X = np.asarray(X, dtype=float)
        f = np.asarray(fitness, dtype=float)
        if X.shape != (self.lam, self.n) or f.shape != (self.lam,):
            raise ValueError("tell: shape mismatch")
        n = self.n
        order = np.argsort(-f, kind="stable")[: self.mu]
        Y = (X[order] - self.mean[None, :]) / self.sigma
        yw = self.w @ Y
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * self._inv_sqrt(yw)
        nps = float(np.linalg.norm(self.ps))
        hsig = 1.0 if nps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chi_n < 1.4 + 2 / (n + 1) else 0.0
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        delta_h = (1 - hsig) * self.cc * (2 - self.cc)
        old_w = 1 - self.c1 - self.cmu
        if self.kind == "full":
            rank_mu = (Y.T * self.w) @ Y
            self.C = old_w * self.C + self.c1 * (np.outer(self.pc, self.pc) + delta_h * self.C) + self.cmu * rank_mu
        else:
            rank_mu = self.w @ (Y ** 2)
            self.C = old_w * self.C + self.c1 * (self.pc ** 2 + delta_h * self.C) + self.cmu * rank_mu
        self.sigma = self.sigma * math.exp((self.cs / self.damps) * (nps / self.chi_n - 1))
        self.gen += 1
        self._eig()

    def cov_state(self) -> np.ndarray:
        return self.C.copy()

    def state_dict(self) -> Dict[str, Any]:
        return dict(n=self.n, lam=self.lam, kind=self.kind, mean=self.mean.tolist(), sigma=self.sigma, pc=self.pc.tolist(),
                    ps=self.ps.tolist(), C=self.C.tolist(), gen=self.gen, rng=self.rng.bit_generator.state)

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if (d["n"], d["lam"], d["kind"]) != (self.n, self.lam, self.kind):
            raise ValueError("state does not match this optimizer (n, lam, kind)")
        self.mean = np.array(d["mean"], dtype=float)
        self.sigma = float(d["sigma"])
        self.pc = np.array(d["pc"], dtype=float)
        self.ps = np.array(d["ps"], dtype=float)
        self.C = np.array(d["C"], dtype=float)
        self.gen = int(d["gen"])
        self.rng.bit_generator.state = d["rng"]
        self._eig()


def make_optimizer(n: int, lam: int = 32, sigma0: float = 0.3, seed: int = 9000, diagnostic: bool = False):
    """REQ-OPT-01: n<=50 -> kind 'full'; 50<n<=64 (or <=300 if diagnostic) -> kind 'sep'; larger n -> ValueError (REQ-ACT-03).
    lam must be even (else ValueError).  Initial mean 0, sigma0=0.3.  Returned object API:
      ask() -> ndarray[lam,n] with rows 2j = mean+delta_j and 2j+1 = mean-delta_j (REQ-OPT-04);
      tell(X, fitness) (maximisation); mean; sigma; kind; cov_state() (n x n matrix for 'full', length-n diagonal for 'sep');
      state_dict() / load_state_dict(d) including numpy RNG state."""
    if lam < 2 or lam % 2:
        raise ValueError("lam must be even")
    limit = _N_DIAG_MAX if diagnostic else _N_SEP_MAX
    if n < 1 or n > limit:
        raise ValueError(f"REQ-ACT-03: n={n} exceeds the limit {limit}")
    return CMAES(n, lam, sigma0, seed, "full" if n <= _N_FULL_MAX else "sep")
