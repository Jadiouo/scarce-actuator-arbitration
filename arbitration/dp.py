"""Exact optimal benchmark: a semi-MDP for the scarce-actuator problem.

Why this exists: every policy in the simulation is a heuristic. To say how far
blind round-robin (`none`) is from optimal, and when information (true health,
reports) can have positive value at all, we need J*, the best achievable long-run
mean team health. This module computes it for small N (3, optionally 4).

Model (identical to model.run; see "alignment" below)
-----------------------------------------------------
* Decision epochs: every time the actuator is idle (start, and right after each
  service completes). The action is the next post a to serve; a = current post is
  allowed ("serve again where I am", a zero-distance move).
* A cycle from post/position p to post a lasts L = max(1, ceil(d(p,a)/SPEED)) +
  service steps (the simulation's own step count). It is a semi-MDP: cycle
  lengths differ by action.
* Dynamics per robot, independent across robots, h only falls except at service:
    linear: h <- max(0, h - r) per step
    shock : h <- max(0, h - (1-f) r - Bernoulli(p) * Exp(mean f r / p)) per step
  Because h is monotone between services, the per-step clip equals one clip of the
  cumulative damage S_k, so E[max(0, x - S_k)] has a closed form (binomial number
  of Gamma sums; see Dyn._phi_k). Rewards and transitions are therefore exact in
  the damage model; the only approximation is the health grid.
* Objective: long-run average of the team mean health, with the simulation's
  accounting (clip at 0, the served robot keeps its pre-reset value during the
  cycle and is reset to 1 at the last step of the cycle).
* Alignment: the simulation applies a step's damage BEFORE the policy looks, so
  at a decision epoch the dispatcher sees h after one damage step. The DP state
  is exactly that: just-served robot has h = 1 after one damage step (random
  under shocks), age = 1. The cycle reward is  sum_{k=0..L-1} of the team mean
  health starting from the seen state.

Average reward is solved with relative value iteration after Schweitzer's
data transformation (r' = tau r / L, self-loop weight 1 - tau/L, tau = min L / 2: tau must be
STRICTLY below min L, otherwise two posts at equal cycle length can make the chain
periodic and the bracket oscillates forever; the gain is unchanged: g = g'/tau). The
stopping rule is the span of (T V - V): its max and min bracket the optimal gain
(min is also a certified lower bound on the gain of the greedy policy), so the
reported g (midpoint) is accurate to span / (2 tau); default span < 2e-4, i.e.
g within ~2e-5 (tests/verification tighten it). The max side can converge slowly
on a few transient states while the min side is nearly exact, hence the loose-looking
default; in hard instances the iteration cap (4000) is hit and the achieved span is
stored in `.span` (a sweep reports its maximum). We use the
undiscounted average criterion directly (no gamma -> 1 limit needed).

Information structures
----------------------
full (FullDP)  state (post, h_1..h_N) on a uniform grid with K+1 levels; a
               transition spreads each robot's end-of-cycle health over the two
               neighbouring grid points with weights that preserve the mean
               (exactly, from the same closed form, also for the atom at 0).
               Error is O(1/K^2) numerical diffusion; check convergence in K.
age  (AgeDP)   state (post, ages of the other robots) -- steps since each
               robot's last service. Under shocks the dispatcher cannot see h,
               its belief about robot i is a function of age_i alone, so the
               POMDP is an ordinary deterministic-transition MDP on ages with
               EXPECTED rewards: exact (no discretization) apart from an age cap
               (beyond it the robot is dead / the cap is reported). Under linear
               damage age determines h, so J*(age) == J*(full): used as an
               independent consistency test.
"""
import math
from collections import OrderedDict
import numpy as np
from scipy.special import gammainc
from scipy.stats import binom
from . import model as _m

B_MAX = 14          # max number of shocks per cycle kept (mass beyond is < 1e-15)
TOL = 2e-4          # stop when hi - lo < TOL (units of the transformed gain g' = tau g)
MAX_ITER = 4000     # safety cap; the achieved bracket is always reported (`span`)


class Dyn:
    """Damage law of one robot: S_k = base*k + sum of Bin(k,p) Exp(mean m) shocks."""

    def __init__(self, rate, damage="linear", shock_rate=0.01, shock_frac=0.5):
        if damage == "linear" or shock_frac == 0:
            self.base, self.p, self.m = float(rate), 0.0, 0.0
        else:
            self.base = (1.0 - shock_frac) * rate
            self.p = float(shock_rate)
            self.m = shock_frac * rate / shock_rate

    def _phi_k(self, ks, c):
        """E[(c - S_k)^+] for each k in ks and every entry of c -> shape (len(ks),)+c.shape."""
        ks = np.asarray(ks, int)
        c = np.asarray(c, float)
        pad = (1,) * c.ndim
        cc = c[None] - self.base * ks.reshape((-1,) + pad)
        cp = np.maximum(cc, 0.0)
        if self.m == 0.0:
            return cp
        pm = binom.pmf(np.arange(B_MAX + 1)[None, :], ks[:, None], self.p)
        z = cp / self.m
        out = pm[:, 0].reshape((-1,) + pad) * cp
        for b in range(1, B_MAX + 1):
            psi = cp * gammainc(b, z) - self.m * b * gammainc(b + 1, z)
            out = out + pm[:, b].reshape((-1,) + pad) * psi
        return np.maximum(out, 0.0)

    def phi(self, k, c):
        return self._phi_k([k], c)[0]

    def cum_at_one(self, nmax):
        """Cum[n] = sum_{k=1..n} E[h after k damages from h=1], n = 0..nmax."""
        return np.concatenate([[0.0], np.cumsum(self._phi_k(np.arange(1, nmax + 1), 1.0))])

    def rz(self, L, x):
        """Reward of one robot over a cycle of L steps from seen health x:
        x + sum_{k=1}^{L-1} E[(x - S_k)^+]  (steps 0..L-1)."""
        x = np.asarray(x, float)
        if L <= 1:
            return x.copy()
        return x + self._phi_k(np.arange(1, L), x).sum(axis=0)

    def trans(self, L, xs, K):
        """Mean-preserving grid distribution of max(0, x - S_L) for each start x in xs."""
        xs = np.atleast_1d(np.asarray(xs, float))
        g = np.arange(K + 1) / K
        P = self.phi(L, xs[:, None] - g[None, :])              # P[:, j] = E[(x - g_j - S_L)^+]
        w = np.empty_like(P)
        w[:, 0] = 1.0 - K * (P[:, 0] - P[:, 1])
        w[:, 1:K] = K * (P[:, :K - 1] - 2.0 * P[:, 1:K] + P[:, 2:K + 1])
        w[:, K] = K * (P[:, K - 1] - P[:, K])
        w = np.maximum(w, 0.0)
        return w / w.sum(axis=1, keepdims=True)


def cycle_lengths(posts, service):
    d = _m.ring_distance(posts[:, None], posts[None, :])
    return np.maximum(1, np.ceil(d / _m.SPEED)).astype(int) + service


def _dyns(rates, cfg):
    return [Dyn(r, cfg.damage, cfg.shock_rate, cfg.shock_frac) for r in rates]


def _rvi(V, step, ref, tol=TOL, max_iter=MAX_ITER):
    """Relative value iteration. step(V) -> list of arrays (transformed Bellman op)."""
    lo = hi = float("nan")
    for it in range(1, max_iter + 1):
        W = step(V)
        lo = min(float((w - v).min()) for w, v in zip(W, V))
        hi = max(float((w - v).max()) for w, v in zip(W, V))
        refval = W[ref[0]][ref[1]]
        V = [w - refval for w in W]
        if hi - lo < tol:
            break
    return V, 0.5 * (lo + hi), it, hi - lo


class FullDP:
    """Optimal dispatcher that sees the true health vector."""
    kind = "dp_full"

    def __init__(self, posts, rates, cfg, K=None, tol=TOL, max_iter=MAX_ITER):
        self.posts = np.asarray(posts, float)
        self.rates = np.asarray(rates, float)
        self.n = N = len(self.posts)
        if N > 5:
            raise ValueError("dp is for small N (<= 4 recommended)")
        self.K = K = int(K or cfg.dp_k)
        self.svc = cfg.service
        self.dyn = _dyns(self.rates, cfg)
        self.Ls = cycle_lengths(self.posts, self.svc)
        self.tau = 0.5 * float(self.Ls.min())    # strictly below min L: self-loop weight >= 1/2
        self.m1 = [dy.trans(1, [1.0], K)[0] for dy in self.dyn]     # served robot after 1 step
        self._M, self._Rw = {}, {}
        for L in set(self.Ls.ravel().tolist()):
            for i in range(N):
                self._M[i, L] = self.dyn[i].trans(L, np.arange(K + 1) / K, K)
            tot = np.zeros((K + 1,) * N)
            g = np.arange(K + 1) / K
            for i in range(N):
                shp = [1] * N
                shp[i] = K + 1
                tot = tot + self.dyn[i].rz(L, g).reshape(shp)
            self._Rw[L] = tot / N
        V = [np.zeros((K + 1,) * N) for _ in range(N)]
        self.V, gt, self.iters, self.span = _rvi(V, self._step, (0, (K,) * N), tol, max_iter)
        self.g = gt / self.tau

    def _cont(self, Va, a, L):
        t = np.tensordot(self.m1[a], Va, axes=([0], [a]))
        others = [i for i in range(self.n) if i != a]
        for idx, i in enumerate(others):
            t = np.moveaxis(np.tensordot(self._M[i, L], t, axes=([1], [idx])), 0, idx)
        return np.expand_dims(t, a)

    def _step(self, V):
        out = []
        for j in range(self.n):
            best = None
            for a in range(self.n):
                L = int(self.Ls[j, a])
                q = self._Rw[L] + self._cont(V[a], a, L)
                val = (self.tau / L) * q + (1.0 - self.tau / L) * V[j]
                best = val if best is None else np.maximum(best, val)
            out.append(best)
        return out

    def q_values(self, pos, h):
        d = _m.ring_distance(self.posts, pos)
        qs = []
        for a in range(self.n):
            L = int(max(1, math.ceil(d[a] / _m.SPEED))) + self.svc
            rew = sum(float(self.dyn[i].rz(L, h[i])) for i in range(self.n)) / self.n
            t = np.tensordot(self.m1[a], self.V[a], axes=([0], [a]))
            others = [i for i in range(self.n) if i != a]
            for idx in range(len(others) - 1, -1, -1):
                i = others[idx]
                row = self.dyn[i].trans(L, [h[i]], self.K)[0]
                t = np.tensordot(row, t, axes=([0], [idx]))
            qs.append(rew - self.g * L + float(t))
        return np.array(qs)

    def choose(self, pos, h, ages=None):
        return int(np.argmax(self.q_values(pos, np.asarray(h, float))))


class AgeDP:
    """Optimal dispatcher that knows rates and the age of every robot (steps since its
    last service) but not h. Exact for the POMDP: the belief is a function of ages."""
    kind = "dp_age"

    def __init__(self, posts, rates, cfg, cap=None, tol=TOL, max_iter=MAX_ITER):
        self.posts = np.asarray(posts, float)
        self.rates = np.asarray(rates, float)
        self.n = N = len(self.posts)
        cap_max = int(cap or cfg.dp_age_cap)
        self.svc = cfg.service
        self.dyn = _dyns(self.rates, cfg)
        self.Ls = cycle_lengths(self.posts, self.svc)
        self.tau = 0.5 * float(self.Ls.min())    # strictly below min L: self-loop weight >= 1/2
        # runtime cycles can start off-post (first decision), so cover the longest possible one
        Lmax = max(int(self.Ls.max()), math.ceil(_m.RING / 2 / _m.SPEED) + self.svc)
        self.cum, self.cap, self.cap_binding = [], [], False
        for dy in self.dyn:
            ph = dy._phi_k(np.arange(1, cap_max + 1), 1.0)
            dead = np.nonzero(ph < 1e-7)[0]
            if len(dead):
                cap_i = max(2, int(dead[0]) + 1)
            else:
                cap_i, self.cap_binding = cap_max, True
            self.cap.append(cap_i)
            self.cum.append(dy.cum_at_one(cap_i + Lmax + 1))
        # per (j, a): static reward array and gather indices
        self._rew, self._idx = {}, {}
        for j in range(N):
            shape = [1 if i == j else self.cap[i] for i in range(N)]
            ages = [np.array([1]) if i == j else np.arange(1, self.cap[i] + 1) for i in range(N)]
            for a in range(N):
                L = int(self.Ls[j, a])
                rew = np.zeros(shape)
                idx = {}
                for i in range(N):
                    ai = ages[i]
                    r = self.cum[i][ai + L - 1] - self.cum[i][ai - 1]
                    sh = [1] * N
                    sh[i] = len(ai)
                    rew = rew + r.reshape(sh)
                    if i != a:
                        idx[i] = np.minimum(self.cap[i], ai + L) - 1
                self._rew[j, a] = rew / N
                self._idx[j, a] = idx
        V = [np.zeros([1 if i == j else self.cap[i] for i in range(N)]) for j in range(N)]
        self.V, gt, self.iters, self.span = _rvi(V, self._step, (0, (0,) * N), tol, max_iter)
        self.g = gt / self.tau

    def _step(self, V):
        out = []
        for j in range(self.n):
            best = None
            for a in range(self.n):
                L = int(self.Ls[j, a])
                arr = V[a]
                for i, ix in self._idx[j, a].items():
                    arr = np.take(arr, ix, axis=i)
                val = (self.tau / L) * (self._rew[j, a] + arr) + (1.0 - self.tau / L) * V[j]
                best = val if best is None else np.maximum(best, val)
            out.append(best)
        return out

    def q_values(self, pos, ages):
        d = _m.ring_distance(self.posts, pos)
        ages = [min(max(int(x), 1), c) for x, c in zip(ages, self.cap)]
        qs = []
        for a in range(self.n):
            L = int(max(1, math.ceil(d[a] / _m.SPEED))) + self.svc
            rew = sum(self.cum[i][ages[i] + L - 1] - self.cum[i][ages[i] - 1]
                      for i in range(self.n)) / self.n
            ix = tuple(0 if i == a else min(self.cap[i], ages[i] + L) - 1 for i in range(self.n))
            qs.append(rew - self.g * L + float(self.V[a][ix]))
        return np.array(qs)

    def choose(self, pos, h, ages):
        return int(np.argmax(self.q_values(pos, ages)))


# ---------------------------------------------------------------- policy cache
_CACHE = OrderedDict()


def build_policy(kind, posts, rates, cfg, K=None):
    """Solve (or fetch) the DP for this instance. Cached per process; policies are stateless."""
    key = (kind, np.asarray(posts).tobytes(), np.asarray(rates).tobytes(), cfg.damage,
           cfg.shock_rate, cfg.shock_frac, cfg.service, K or cfg.dp_k, cfg.dp_age_cap)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    pol = (FullDP(posts, rates, cfg, K) if kind == "dp_full" else AgeDP(posts, rates, cfg))
    _CACHE[key] = pol
    while len(_CACHE) > 6:
        _CACHE.popitem(last=False)
    return pol


def instance(seed, n, rate, spread):
    """(posts, rates, start position) exactly as model.run draws them for `seed`."""
    rng = np.random.default_rng(seed)
    posts = np.sort(rng.uniform(0, _m.RING, n))
    rates = rate * np.exp(rng.normal(0, spread, n))
    return posts, rates, float(rng.uniform(0, _m.RING))


def optimal_values(posts, rates, cfg, K=None):
    """(J*_full, J*_age) for an instance."""
    return (build_policy("dp_full", posts, rates, cfg, K).g,
            build_policy("dp_age", posts, rates, cfg).g)
