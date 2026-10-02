"""Per-step ("true") optimal dispatcher on the GPU: exact-up-to-grid MDP with REPLANNING.

Why: `arbitration.dp` / `gpu.dp_torch` solve a semi-MDP that decides only when a service completes
(no mid-route change of mind, no idling). Under shock damage, replanning every step with the same
value function beats it (red-team finding), so its J* is only the optimum of the "commit until
arrival" class. This module solves the finer problem: decide EVERY step.

Model (aligned with model.run; see dp.py for the damage law)
------------------------------------------------------------
* Instance is put on the integer ring grid: posts = floor(posts), start = floor(pos0) (RING = 100,
  SPEED = 1, so every actuator position is an integer; travel time = ring distance exactly, i.e.
  the simulation's ceil(d / SPEED)). `rounded_instances()` makes sim_torch draw the same floored
  instances, so DP and every simulated policy evaluate the SAME instance.
* State at a decision step: actuator position x in Z_100 and the health vector h seen AFTER this
  step's damage (grid with K+1 levels per robot). The step's reward is mean(h) (clip at 0 included).
* Actions: move -1 / stay / +1 (one step, D = 1), or "move dir in {-1,0,+1} onto a post j and begin
  service" (non-interruptible semi-MDP transition of D = 1 + SERVICE steps: this is the simulation's
  arrival step + SERVICE service steps). Passing over a post without serving is just a move.
  Idle (stay) is also allowed. Every dispatch rule of the simulation is a policy of this MDP, so its
  optimum J_step is an upper bound for all of them (up to the grid).
* Transitions: damages are independent across robots, so each one-step (or D-step) transition is a
  K x K matrix applied along each robot axis (dp.Dyn.trans, the same mean-preserving grid
  projection as the semi-MDP). Rewards: mean(h) per step; serve: sum of dp.Dyn.rz.
* Average reward by relative value iteration with Schweitzer's transformation tau = 0.8 < min
  duration (1): self-loop weight >= 0.2 makes every policy aperiodic (position parity would
  otherwise make the chain periodic). tau/D is the weight of an action of duration D. Stopping rule:
  span(TV - V) < tol brackets the optimal gain g' = tau g; the report is the midpoint / tau.

Discretisation: the one-step projection is applied once PER STEP, so numerical diffusion is first
order in 1/K (variance Delta*drift per step) instead of the semi-MDP's one projection per cycle. J_step
therefore needs a K-extrapolation (the driver uses K = 30/60/120); the simulated value of the
extracted policy (continuous h, trilinear interpolation of the tables) is an honest achievable value
and is reported next to it.
"""
import math
import time
import numpy as np
import torch
from .. import model as _m
from .. import dp as _dp
from ..dp import Dyn, cycle_lengths
from . import sim_torch as st

RING = int(_m.RING)
TAU = 0.8


def _dev(device=None):
    return torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ integer-ring instances
def int_instance(seed, n, rate, spread):
    """(posts int array, rates, start int) exactly as sim_torch draws them under rounded_instances()."""
    posts, rates, pos0 = _dp.instance(seed, n, rate, spread)
    return np.floor(posts).astype(np.int64), rates, int(math.floor(pos0))


class rounded_instances:
    """Context manager: sim_torch.draw_instance returns floored posts and start position (all other
    draws untouched, so noise / shocks stay the CRN streams of the unrounded simulator)."""
    def __enter__(self):
        self._orig = st.draw_instance

        def patched(*a, **k):
            d = dict(self._orig(*a, **k))
            d["posts"] = np.floor(d["posts"])
            d["pos0"] = float(math.floor(d["pos0"]))
            return d
        st.draw_instance = patched
        return self

    def __exit__(self, *exc):
        st.draw_instance = self._orig


# ------------------------------------------------------------------ operators of one instance
def build_ops(posts, rates, cfg, K):
    """numpy pieces shared by the solver and the policy tables (identical grid projection as dp.py)."""
    N, D = len(posts), cfg.service + 1
    dyn = _dp._dyns(np.asarray(rates, float), cfg)
    g = np.arange(K + 1) / K
    M1 = np.stack([dy.trans(1, g, K) for dy in dyn])
    MD = np.stack([dy.trans(D, g, K) for dy in dyn])
    m1 = np.stack([dy.trans(1, [1.0], K)[0] for dy in dyn])
    RD = np.stack([dy.rz(D, g) for dy in dyn])
    return M1, MD, m1, RD


def _apply(W, M, ax):
    """out[b, ..., h_ax, ...] = sum_k M[b, h_ax, k] W[b, ..., k, ...]   (axis ax of the N health axes)."""
    d = 2 + ax
    Wm = W.movedim(d, -1)
    sh = Wm.shape
    out = torch.bmm(Wm.reshape(sh[0], -1, sh[-1]), M.transpose(1, 2)).reshape(sh)
    return out.movedim(-1, d)


def _hgrid(N, K, dev):
    """(mean-health r1 grid, per-axis views) on the (K+1)^N health grid."""
    g = torch.arange(K + 1, dtype=torch.float64, device=dev) / K
    r1 = None
    for i in range(N):
        shp = [1] * N
        shp[i] = K + 1
        t = g.reshape(shp)
        r1 = t if r1 is None else r1 + t
    return r1 / N


def _band(M, wmax=6):
    """Banded (lower-triangular, bandwidth <= wmax) form of the per-instance operator M (B, K+1, K+1):
    returns list of diagonals d = 0..w, diag[d][b, h] = M[b, h, h-d], or None if M is not that sparse.
    (Deterministic decay: every row has <= 2 neighbouring nonzeros, so a matmul is wasteful.)"""
    Mn = M.abs() > 1e-13
    K1 = M.shape[-1]
    h = torch.arange(K1, device=M.device)
    dist = h[:, None] - h[None, :]
    if bool((Mn & (dist < 0)).any()):
        return None
    w = int((dist * Mn).amax())
    if w > wmax:
        return None
    return [torch.cat([torch.zeros(M.shape[0], d, dtype=M.dtype, device=M.device),
                       torch.stack([M[:, hh, hh - d] for hh in range(d, K1)], dim=1)], dim=1) for d in range(w + 1)]


def _apply_banded(W, diags, ax):
    d_ax = 2 + ax
    K1 = W.shape[d_ax]
    out = None
    for d, dg in enumerate(diags):
        shp = [1] * W.dim()
        shp[0] = dg.shape[0]
        shp[d_ax] = K1 - d
        term = W.narrow(d_ax, 0, K1 - d) * dg[:, d:].reshape(shp)
        if out is None:
            out = torch.zeros_like(W)
        out.narrow(d_ax, d, K1 - d).add_(term)
    return out


class StepProblem:
    """Batch of B instances (same N, K, damage cfg). Holds GPU operators and runs RVI."""

    def __init__(self, instances, cfg, K, device=None, tau=TAU):
        self.dev = dev = _dev(device)
        self.cfg, self.K, self.tau = cfg, K, tau
        self.B, self.N = B, N = len(instances), len(instances[0][0])
        self.D = D = cfg.service + 1
        ops = [build_ops(p, r, cfg, K) for p, r in instances]
        T = lambda a: torch.tensor(np.stack(a), dtype=torch.float64, device=dev)
        self.M1, self.MD, self.m1, self.RD = (T([o[i] for o in ops]) for i in range(4))
        self.posts = torch.tensor(np.stack([np.asarray(p, np.int64) for p, _ in instances]), device=dev)
        self.band = [_band(self.M1[:, i]) for i in range(N)]
        self.r1 = _hgrid(N, K, dev)                                     # (K+1,)*N, mean health
        R = None
        for i in range(N):
            shp = [self.B] + [1] * N
            shp[1 + i] = K + 1
            t = self.RD[:, i].reshape(shp)
            R = t if R is None else R + t
        self.R = R / N                                                  # serve reward (B, ...)
        self.ref = (slice(None), 0) + (K,) * N

    # --- pieces of the backup -------------------------------------------------------------
    def expect1(self, V):
        W = V
        for i in range(self.N):
            W = _apply(W, self.M1[:, i], i) if self.band[i] is None else _apply_banded(W, self.band[i], i)
        return W.contiguous()

    def serve_tables(self, V):
        """C_j(h) (B, N, (K+1)^N): value after serving post j from health h (continuation only)."""
        B, N, K = self.B, self.N, self.K
        ar = torch.arange(B, device=self.dev)
        out = []
        for j in range(N):
            Vj = V[ar, self.posts[:, j]]                                 # (B, (K+1)^N)
            Vm = Vj.movedim(1 + j, -1)
            sh = Vm.shape
            t = torch.bmm(Vm.reshape(B, -1, K + 1), self.m1[:, j].unsqueeze(-1)).reshape(sh[:-1])
            others = [i for i in range(N) if i != j]
            for idx, i in enumerate(others):                             # axis idx of t (after batch)
                t = _apply_axis(t, self.MD[:, i], 1 + idx)
            out.append(t.unsqueeze(1 + j))                               # axis j of size 1 (broadcast)
        return out

    def bellman(self, V):
        tau, D = self.tau, self.D
        B, N = self.B, self.N
        W = self.expect1(V)
        Q = W.clone()                                                   # max over x-1, x, x+1 (ring)
        Q[:, 1:] = torch.maximum(Q[:, 1:], W[:, :-1])
        Q[:, 0] = torch.maximum(Q[:, 0], W[:, -1])
        Q[:, :-1] = torch.maximum(Q[:, :-1], W[:, 1:])
        Q[:, -1] = torch.maximum(Q[:, -1], W[:, 0])
        del W
        Q.add_(self.r1)
        new = torch.lerp(V, Q, tau)                                     # (1-tau) V + tau (r + E V')
        del Q
        ar = torch.arange(B, device=self.dev)
        cj = self.serve_tables(V)
        a = tau / D
        for j in range(N):
            S = (self.R + cj[j]) * a
            for dx in (-1, 0, 1):
                x = (self.posts[:, j] - dx) % RING                       # positions that can reach post j with move dx
                cand = S + (1 - a) * V[ar, x]
                new[ar, x] = torch.maximum(new[ar, x], cand)
        return new

    def solve(self, V0=None, tol=2e-5, max_iter=20000, check_every=1, verbose=False, min_iter=0):
        """RVI. tol is on the per-step gain (bracket width of g = g'/tau)."""
        K, N, B = self.K, self.N, self.B
        V = torch.zeros((B, RING) + (K + 1,) * N, dtype=torch.float64, device=self.dev) if V0 is None else V0
        it = 0
        t0 = time.time()
        while it < max_iter:
            it += 1
            new = self.bellman(V)
            if it % 5 == 0 or it >= max_iter:
                diff = (new - V).reshape(B, -1)
                lo, hi = diff.amin(1), diff.amax(1)
                del diff
                ref = new[self.ref].reshape((B,) + (1,) * (N + 1)).clone()
                V = new.sub_(ref)
                span = ((hi - lo) / self.tau)
                if it >= min_iter and bool((span < tol).all()):
                    break
                if verbose and it % 200 == 0:
                    print(f"    it {it} span {float(span.max()):.2e} g {float(((lo + hi) / 2 / self.tau).mean()):.6f} {time.time() - t0:.0f}s", flush=True)
            else:
                V = new
        self.V, self.iters = V, it
        self.g = ((lo + hi) / (2 * self.tau)).cpu().numpy()
        self.span = span.cpu().numpy()
        return self


def _apply_axis(t, M, ax):
    """like _apply but t has the batch axis first and no x axis: axis `ax` (>=1) of t."""
    Wm = t.movedim(ax, -1)
    sh = Wm.shape
    out = torch.bmm(Wm.reshape(sh[0], -1, sh[-1]), M.transpose(1, 2)).reshape(sh)
    return out.movedim(-1, ax)


# ------------------------------------------------------------------ grid refinement (warm start)
def refine(V, K1, K2):
    """Multilinear interpolation of V (B, X, (K1+1)^N) onto the (K2+1)^N grid."""
    dev = V.device
    u = torch.arange(K2 + 1, dtype=torch.float64, device=dev) * K1 / K2
    i0 = u.floor().clamp(max=K1 - 1).long()
    f = u - i0
    Mi = torch.zeros(K2 + 1, K1 + 1, dtype=torch.float64, device=dev)
    ar = torch.arange(K2 + 1, device=dev)
    Mi[ar, i0] = 1 - f
    Mi[ar, i0 + 1] += f
    N = V.dim() - 2
    out = V
    for ax in range(N):
        d = 2 + ax
        Wm = out.movedim(d, -1)
        sh = Wm.shape
        o = (Wm.reshape(-1, sh[-1]) @ Mi.T).reshape(sh[:-1] + (K2 + 1,))
        out = o.movedim(-1, d)
    return out.contiguous()


def solve_steps(instances, cfg, Ks, tol=2e-5, max_iter=20000, device=None, verbose=False, keep_last=True,
                tau=TAU):
    """Solve for each K in Ks (increasing), warm-starting every level from the previous one.
    Returns [(K, problem)] ; only the last problem keeps V (unless keep_last=False)."""
    out, V0, prev = [], None, None
    for k, K in enumerate(Ks):
        P = StepProblem(instances, cfg, K, device, tau)
        if V0 is not None:
            V0 = refine(V0, prev, K)
        P.solve(V0=V0, tol=tol, max_iter=max_iter, verbose=verbose)
        V0, prev = P.V, K
        if k < len(Ks) - 1:
            P.V = None
        out.append((K, P))
    return out


def richardson(Ks, Js):
    """Fit J(K) = a + b/K (+ c/K^2 with 3 points); return a. Js: (len(Ks), ...) arrays."""
    Ks = np.asarray(Ks, float)
    A = np.stack([Ks ** -p for p in range(len(Ks))], axis=1)
    J = np.asarray(Js, float)
    coef = np.linalg.solve(A, J.reshape(len(Ks), -1))
    return coef[0].reshape(J.shape[1:])


# ------------------------------------------------------------------ policy tables + simulator
def policy_tables(P):
    """From a solved StepProblem: QM (B, X, (K+1)^N) = value of moving/staying INTO position x' (one
    step) as a function of the seen health, QS (B, N, (K+1)^N) = value of serving post j. Both are
    'R + E V(next) - g D' so actions of different duration are comparable."""
    g = torch.tensor(P.g, dtype=torch.float64, device=P.dev)
    gb = g.reshape((P.B,) + (1,) * (P.N + 1))
    QM = P.expect1(P.V)
    QM += P.r1
    QM -= gb
    cj = P.serve_tables(P.V)
    gS = g.reshape((P.B,) + (1,) * P.N)
    QS = torch.stack([(P.R + cj[j]) - gS * P.D for j in range(P.N)], dim=1)
    return QM, QS


def _interp_idx(h, K):
    """corner flat indices (B, 2^N) and weights (B, 2^N) of the multilinear interpolation at h (B, N)."""
    B, N = h.shape
    u = h * K
    i0 = u.floor().clamp(max=K - 1)
    f = u - i0
    i0 = i0.long()
    flat = torch.zeros(B, 1, dtype=torch.long, device=h.device)
    w = torch.ones(B, 1, dtype=torch.float64, device=h.device)
    for a in range(N):
        flat = torch.stack([flat * (K + 1) + i0[:, a:a + 1], flat * (K + 1) + i0[:, a:a + 1] + 1], dim=-1).reshape(B, -1)
        w = torch.stack([w * (1 - f[:, a:a + 1]), w * f[:, a:a + 1]], dim=-1).reshape(B, -1)
    return flat, w


class StepSim:
    """Batched simulator of the per-step policy (full information): the same dynamics, shock streams
    (sim_torch._shock_draws) and accounting as sim_torch/model.run, but the actuator is controlled one
    step at a time. All sims share N, K, steps."""

    def __init__(self, insts, cfgs, steps, device=None):
        """insts: [dict(seed, posts int (N,), rates (N,), pos0 int, stream=0)], cfgs: one ModelConfig each."""
        self.dev = dev = _dev(device)
        B, N = len(insts), len(insts[0]["posts"])
        self.B, self.N, self.steps = B, N, steps
        self.insts, self.cfgs = insts, cfgs
        T = lambda a, dt=torch.float64: torch.tensor(np.asarray(a), dtype=dt, device=dev)
        self.posts = T(np.stack([i["posts"] for i in insts]), torch.long)
        rates = np.stack([i["rates"] for i in insts])
        self.rates = T(rates)
        self.pos0 = T([i["pos0"] for i in insts], torch.long)
        self.svc = T([c.service for c in cfgs], torch.long)
        self.is_shock = np.array([c.damage == "shock" and c.shock_frac != 0 for c in cfgs])
        if self.is_shock.any():
            dr = [st._shock_draws(i["seed"], steps, N, [] if i.get("stream", 0) == 0 else [int(i["stream"])])
                  if sh else (np.zeros((steps, N)), np.zeros((steps, N))) for i, sh in zip(insts, self.is_shock)]
            self.SU = T(np.stack([d[0] for d in dr], axis=1))
            self.SE = T(np.stack([d[1] for d in dr], axis=1))
            sf = T([c.shock_frac for c in cfgs])[:, None]
            sr = T([c.shock_rate for c in cfgs])[:, None]
            self.sr = sr
            self.base_decay = self.rates * (1.0 - sf)
            self.shock_size = sf * self.rates / sr
            self.shock_m = T(self.is_shock, torch.bool)[:, None]

    def run(self, QM, QS, K, burn=500, record=False):
        """Run with tables from policy_tables. Returns dict(mean_health (B,), robot_health, n_services)."""
        dev, B, N, steps = self.dev, self.B, self.N, self.steps
        f = torch.float64
        ar = torch.arange(B, device=dev)
        x = self.pos0.clone()
        h = torch.ones(B, N, dtype=f, device=dev)
        serving = torch.zeros(B, dtype=torch.long, device=dev)
        tgt = torch.zeros(B, dtype=torch.long, device=dev)
        msum = torch.zeros(B, dtype=f, device=dev)
        hsum = torch.zeros(B, N, dtype=f, device=dev)
        nserv = torch.zeros(B, dtype=f, device=dev)
        QMf = QM.reshape(B, RING, -1)
        QSf = QS.reshape(B, N, -1)
        dxs = torch.tensor([-1, 0, 1], device=dev)
        rec = []
        for t in range(steps):
            if self.is_shock.any():
                hit = (self.SU[t] < self.sr) * self.SE[t] * self.shock_size
                hs = h - self.base_decay - hit
                h = torch.clamp(torch.where(self.shock_m, hs, h - self.rates), 0.0, 1.0)
            else:
                h = torch.clamp(h - self.rates, 0.0, 1.0)
            if t >= burn:
                s = h[:, 0]
                for k in range(1, N):
                    s = s + h[:, k]
                msum += s / N
                hsum += h
            if record:
                rec.append(h.clone())
            srv = serving > 0
            nsv = torch.where(srv, serving - 1, serving)
            fin = srv & (nsv == 0)
            if bool(fin.any()):
                oh = (torch.arange(N, device=dev)[None, :] == tgt[:, None]) & fin[:, None]
                h = torch.where(oh, torch.ones_like(h), h)
            serving = nsv
            act = ~srv
            if not bool(act.any()):
                continue
            flat, w = _interp_idx(h, K)
            xm = (x[:, None] + dxs[None, :]) % RING                            # (B, 3)
            qm = (QMf[ar[:, None, None], xm[:, :, None], flat[:, None, :]] * w[:, None, :]).sum(-1)    # (B, 3)
            qs = (QSf[ar[:, None, None], torch.arange(N, device=dev)[None, :, None], flat[:, None, :]] * w[:, None, :]).sum(-1)  # (B, N)
            dpost = (self.posts - x[:, None] + RING // 2) % RING - RING // 2    # signed ring offset to each post
            valid = dpost.abs() <= 1
            qs = torch.where(valid, qs, torch.full_like(qs, -float("inf")))
            q = torch.cat([qm, qs], dim=1)
            a = q.argmax(1)                                                    # 0..2 move, 3.. serve post a-3
            is_serve = a >= 3
            j = (a - 3).clamp(min=0)
            move = torch.where(is_serve, dpost.gather(1, j[:, None])[:, 0], dxs[a.clamp(max=2)])
            x = torch.where(act, (x + move) % RING, x)
            go = act & is_serve
            serving = torch.where(go, self.svc, serving)
            tgt = torch.where(go, j, tgt)
            nserv += (go & (t >= burn)).to(f)
        n = steps - burn
        out = dict(mean_health=(msum / n).cpu().numpy(), robot_health=(hsum / n).cpu().numpy(),
                   n_services=nserv.cpu().numpy())
        if record:
            out["h_trace"] = torch.stack(rec).cpu().numpy()
        return out


# ------------------------------------------------------------------ consistency helpers
def restricted_semi(P, tol=1e-11, max_iter=20000):
    """The semi-MDP (decide only when a service completes, go straight to the target, no idling),
    but built from the SAME one-step primitives as the step solver (travel = d-1 plain one-step
    moves, then the move-and-serve step). Equals dp_torch's J* exactly whenever the one-step
    projections compose (e.g. deterministic decay that is a multiple of the grid), so it checks the
    primitives, timing, serve tables and Schweitzer weights of the step solver. B = 1 only."""
    assert P.B == 1
    N, K, svc, tau = P.N, P.K, P.cfg.service, P.tau
    posts = P.posts[0].cpu().numpy()
    dd = np.abs(posts[:, None] - posts[None, :])
    dist = np.minimum(dd, RING - dd)
    Lm = np.maximum(1, dist) + svc
    tau_s = 0.5 * float(Lm.min())
    V = torch.zeros((N,) + (K + 1,) * N, dtype=torch.float64, device=P.dev)
    for it in range(1, max_iter + 1):
        Vp = torch.zeros((1, RING) + (K + 1,) * N, dtype=torch.float64, device=P.dev)
        for a in range(N):
            Vp[0, posts[a]] = V[a]
        cj = P.serve_tables(Vp)
        newV = []
        for j in range(N):
            best = None
            for a in range(N):
                F = P.R + cj[a]
                for _ in range(int(max(dist[j, a], 1)) - 1):
                    F = P.r1 + P.expect1(F.unsqueeze(1))[:, 0]
                L = int(Lm[j, a])
                cand = (tau_s / L) * F[0] + (1 - tau_s / L) * V[j]
                best = cand if best is None else torch.maximum(best, cand)
            newV.append(best)
        W = torch.stack(newV)
        diff = (W - V).reshape(-1)
        lo, hi = float(diff.min()), float(diff.max())
        V = W - W[(0,) + (K,) * N]
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi) / tau_s, it, hi - lo


# ------------------------------------------------------------------ semi-MDP dp_full policy on the GPU
def _ring_d(a, b):
    d = abs(int(a) - int(b)) % RING
    return min(d, RING - d)


def semi_tables(posts, rates, pos0, cfg, K, Vs, g, device=None):
    """Q(c, a, h) = sum_i rz(L,h_i)/N + E V_a(next) - g L, c = decision source (post c, or c = N for the
    start position pos0), L = max(1, d(c, a)) + service. Returns (N+1, N, (K+1)^N) torch."""
    dev = _dev(device)
    N, svc = len(posts), cfg.service
    dyn = _dp._dyns(np.asarray(rates, float), cfg)
    gr = np.arange(K + 1) / K
    Mc, Rc = {}, {}
    m1 = torch.tensor(np.stack([dy.trans(1, [1.0], K)[0] for dy in dyn]), device=dev)
    out = torch.zeros((N + 1, N) + (K + 1,) * N, dtype=torch.float64, device=dev)
    Vt = [torch.tensor(v, dtype=torch.float64, device=dev) for v in Vs]
    for c in range(N + 1):
        src = posts[c] if c < N else pos0
        for a in range(N):
            L = max(1, _ring_d(src, posts[a])) + svc
            if L not in Mc:
                Mc[L] = torch.tensor(np.stack([dy.trans(L, gr, K) for dy in dyn]), device=dev)
                Rc[L] = torch.tensor(np.stack([dy.rz(L, gr) for dy in dyn]), device=dev)
            t = torch.tensordot(Vt[a], m1[a], dims=([a], [0]))             # axis a contracted
            others = [i for i in range(N) if i != a]
            for idx, i in enumerate(others):
                t = _apply_axis(t.unsqueeze(0), Mc[L][i].unsqueeze(0), 1 + idx)[0]
            R = None
            for i in range(N):
                shp = [1] * N
                shp[i] = K + 1
                term = Rc[L][i].reshape(shp)
                R = term if R is None else R + term
            out[c, a] = R / N + t.unsqueeze(a) - g * L
    return out


def interp_table(tab, h, K):
    """tab (B, ..., F) with F = (K+1)^N flat last axis; h (B, N) -> (B, ...) multilinear interpolation."""
    flat, w = _interp_idx(h, K)
    B = h.shape[0]
    sh = tab.shape[1:-1]
    t = tab.reshape(B, -1, tab.shape[-1])
    v = torch.gather(t, 2, flat[:, None, :].expand(B, t.shape[1], flat.shape[1]))
    return (v * w[:, None, :]).sum(-1).reshape((B,) + sh)


class _SemiFull(st.Policy):
    """dp_full of the semi-MDP (decides only when the actuator is idle at a post), on the same grid
    tables as dp.FullDP.choose (continuous h, multilinear interpolation of the Q tables)."""
    kind = "commit"
    CTX = None                     # set by run_semi(): dict(tab (B, N+1, N, F), K, N)

    def init(self, S):
        self.tab = _SemiFull.CTX["tab"]
        self.K = _SemiFull.CTX["K"]

    def new_target(self, S, need):
        if not bool(need.any()):
            return torch.zeros(S.B, dtype=torch.long, device=S.dev)
        eq = (S.posts == S.pos[:, None])
        c = torch.where(eq.any(1), eq.to(torch.long).argmax(1), torch.full_like(S.pos, S.n, dtype=torch.long))
        q = interp_table(self.tab, S.h.clamp(0, 1), self.K)                    # (B, N+1, N)
        q = q[torch.arange(S.B, device=S.dev), c]
        return q.argmax(1)


class AgeBatch:
    """GPU port of dp.AgeDP for a batch of instances: exact-optimal dispatcher that knows rates and ages
    (steps since last service). Pads every instance to the batch maximum cap (padded ages alias the cap
    state) so one set of gathers serves the whole batch."""

    def __init__(self, instances, cfg, cap=None, tol=2e-5, max_iter=8000, device=None, verbose=False):
        self.dev = dev = _dev(device)
        self.cfg, B = cfg, len(instances)
        N = self.N = len(instances[0][0])
        self.B = B
        svc = cfg.service
        cap_max = int(cap or cfg.dp_age_cap)
        per = []
        for posts, rates in instances:
            posts = np.asarray(posts, float)
            dyn = _dp._dyns(np.asarray(rates, float), cfg)
            Ls = cycle_lengths(posts, svc)
            Lmax = max(int(Ls.max()), math.ceil(_m.RING / 2 / _m.SPEED) + svc)
            caps, cums, binding = [], [], False
            for dy in dyn:
                ph = dy._phi_k(np.arange(1, cap_max + 1), 1.0)
                dead = np.nonzero(ph < 1e-7)[0]
                if len(dead):
                    ci = max(2, int(dead[0]) + 1)
                else:
                    ci, binding = cap_max, True
                caps.append(ci)
                cums.append(dy.cum_at_one(ci + Lmax + 1))
            per.append(dict(Ls=Ls, tau=0.5 * float(Ls.min()), caps=caps, cums=cums, binding=binding, Lmax=Lmax))
        C = [max(p["caps"][i] for p in per) for i in range(N)]
        self.C = C
        self.cap_binding = np.array([p["binding"] for p in per])
        T = lambda a, dt=torch.float64: torch.tensor(np.asarray(a), dtype=dt, device=dev)
        self.tau = T([p["tau"] for p in per]).reshape((B,) + (1,) * N)
        self.caps = T([p["caps"] for p in per], torch.long)
        self.Ls = np.stack([p["Ls"] for p in per])
        clen = max(len(c) for p in per for c in p["cums"])
        cum = np.zeros((B, N, clen))
        for b, p in enumerate(per):
            for i in range(N):
                cum[b, i, :len(p["cums"][i])] = p["cums"][i]
                cum[b, i, len(p["cums"][i]):] = p["cums"][i][-1]
        self.cum = T(cum)
        self.rew, self.idx = {}, {}
        for j in range(N):
            for a in range(N):
                big = N >= 4                    # (B, C^N) reward arrays would not fit: keep per-axis vectors
                rew = (np.zeros((B,) + tuple(1 if i == j else C[i] for i in range(N))) if not big else None)
                tv = [np.zeros((B, 1 if i == j else C[i])) for i in range(N)] if big else None
                ix = {i: [] for i in range(N) if i != a}
                for b, p in enumerate(per):
                    L = int(p["Ls"][j, a])
                    for i in range(N):
                        ai = (np.array([1]) if i == j else np.minimum(np.arange(1, C[i] + 1), p["caps"][i]))
                        r = p["cums"][i][ai + L - 1] - p["cums"][i][ai - 1]
                        sh = [1] * N
                        sh[i] = len(ai)
                        if big:
                            tv[i][b] = r
                        else:
                            rew[b] += r.reshape(sh)
                        if i != a:
                            ix[i].append(np.minimum(p["caps"][i], ai + L) - 1)
                self.rew[j, a] = T(rew / N) if not big else [T(v / N) for v in tv]
                self.idx[j, a] = {i: T(np.stack(v), torch.long) for i, v in ix.items()}
        self.tauL = {(j, a): T([p["tau"] / p["Ls"][j, a] for p in per]).reshape((B,) + (1,) * N)
                     for j in range(N) for a in range(N)}
        V = [torch.zeros((B,) + tuple(1 if i == j else C[i] for i in range(N)), dtype=torch.float64, device=dev)
             for j in range(N)]
        for it in range(1, max_iter + 1):
            W = self._step(V)
            lo = torch.stack([(w - v).reshape(B, -1).amin(1) for w, v in zip(W, V)]).amin(0)
            hi = torch.stack([(w - v).reshape(B, -1).amax(1) for w, v in zip(W, V)]).amax(0)
            ref = W[0][(slice(None),) + (0,) * N].reshape((B,) + (1,) * N)
            V = [w - ref for w in W]
            span = (hi - lo) / self.tau.reshape(B)
            if bool((span < tol).all()):
                break
            if verbose and it % 500 == 0:
                print(f"    age it {it} span {float(span.max()):.2e}", flush=True)
        self.V, self.iters, self.span = V, it, span.cpu().numpy()
        self.g = ((lo + hi) / (2 * self.tau.reshape(B))).cpu().numpy()

    def _rew(self, j, a):
        r = self.rew[j, a]
        if not isinstance(r, list):
            return r
        out = None
        for i, v in enumerate(r):
            shp = [self.B] + [1] * self.N
            shp[1 + i] = v.shape[1]
            out = v.reshape(shp) if out is None else out + v.reshape(shp)
        return out

    def _step(self, V):
        out = []
        N = self.N
        for j in range(N):
            best = None
            for a in range(N):
                arr = V[a]
                for i, ix in self.idx[j, a].items():
                    shp = list(arr.shape)
                    shp[1 + i] = ix.shape[1]
                    vs = [1] * (N + 1)
                    vs[0], vs[1 + i] = self.B, ix.shape[1]
                    arr = torch.gather(arr, 1 + i, ix.reshape(vs).expand(shp))
                tl = self.tauL[j, a]
                val = tl * (self._rew(j, a) + arr) + (1.0 - tl) * V[j]
                best = val if best is None else torch.maximum(best, val)
            out.append(best)
        return out


class _AgeFull(st.Policy):
    """dp_age on the GPU: argmax_a of dp.AgeDP.q_values (ages from the simulator's service clock)."""
    kind = "commit"
    CTX = None

    def init(self, S):
        self.A = _AgeFull.CTX

    def new_target(self, S, need):
        A = self.A
        B, N = S.B, S.n
        if not bool(need.any()):
            return torch.zeros(B, dtype=torch.long, device=S.dev)
        dev = S.dev
        tt = S.t.to(torch.float64)
        age = (tt - S.t_reset + (S.t_reset == 0).to(torch.float64)).round().long()
        age = torch.minimum(age.clamp(min=1), A.caps)
        ar = torch.arange(B, device=dev)
        posf = S.pos
        qs = []
        for a in range(N):
            d = torch.abs(S.posts[:, a] - posf)
            d = torch.minimum(d, RING - d)
            L = (torch.clamp(torch.ceil(d / st.SPEED), min=1) + S.svc[:, 0]).long()
            rew = 0.0
            ix = []
            for i in range(N):
                ci = A.cum[ar, i]
                rew = rew + ci.gather(1, (age[:, i] + L - 1)[:, None])[:, 0] - ci.gather(1, (age[:, i] - 1)[:, None])[:, 0]
                ix.append(torch.zeros_like(L) if i == a else torch.minimum(A.caps[:, i], age[:, i] + L) - 1)
            g = torch.tensor(A.g, dtype=torch.float64, device=dev)
            v = A.V[a][(ar,) + tuple(ix)]
            qs.append(rew / N - g * L + v)
        return torch.stack(qs, 1).argmax(1)


def run_semi_policy(kind, seeds, rates, spread, cfgs, steps, n, ctx, burn=500, device=None):
    """Simulate dp_full-semi / dp_age on the rounded instances (call inside rounded_instances())."""
    if kind == "dp_full":
        _SemiFull.CTX = ctx
        st.POLICY_CLASSES["dp_full_semi"] = _SemiFull
        name = "dp_full_semi"
    else:
        _AgeFull.CTX = ctx
        st.POLICY_CLASSES["dp_age_gpu"] = _AgeFull
        name = "dp_age_gpu"
    return sim_burn(name, seeds, rates, spread, cfgs, steps, n, burn, device)


def sim_burn(policy, seeds, rates, spread, cfgs, steps, n, burn=500, device=None, **kw):
    """sim_torch.simulate mean_health over steps [burn, steps): from the sums of two runs of the
    same (prefix-consistent) streams."""
    use_graph = None if policy in st.SUPPORTED else False
    if policy in ("dp_full_semi", "dp_age_gpu"):
        use_graph = False
    a = st.simulate(policy, seeds, n=n, rate=rates, spread=spread, steps=steps, model=cfgs, device=device,
                    graph=use_graph, preempt=False, **kw)
    b = st.simulate(policy, seeds, n=n, rate=rates, spread=spread, steps=burn, model=cfgs, device=device,
                    graph=use_graph, preempt=False, **kw)
    return (a["mean_health"] * steps - b["mean_health"] * burn) / (steps - burn)


# ------------------------------------------------------------------ age-only step MDP (small cap, exact)
def step_age(posts, rates, cfg, A, tol=2e-5, max_iter=20000, device=None, tau=TAU):
    """Per-step MDP when the dispatcher only knows the ages (steps since last service) of every robot:
    its belief is a deterministic function of the ages, so this is an ordinary MDP with expected
    rewards E[h | age] (exact, no health grid). State (x, ages 1..A), same actions as the full step MDP.
    `A` must be large enough that E[h | age] ~ 0 beyond it (ages are clipped at A). Returns g."""
    dev = _dev(device)
    posts = np.asarray(posts, np.int64)
    N, D = len(posts), cfg.service + 1
    dyn = _dp._dyns(np.asarray(rates, float), cfg)
    T = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
    cum = [dy.cum_at_one(A + D + 2) for dy in dyn]
    eh = [T(c[1:A + 1] - c[:A]) for c in cum]                             # E[h | age = 1..A]
    rD = [T(c[np.arange(1, A + 1) + D - 1] - c[np.arange(1, A + 1) - 1]) for c in cum]   # D-step reward from age a
    r1 = R = None
    for i in range(N):
        shp = [1] * N
        shp[i] = A
        t1, tD = eh[i].reshape(shp), rD[i].reshape(shp)
        r1 = t1 if r1 is None else r1 + t1
        R = tD if R is None else R + tD
    r1, R = r1 / N, R / N
    inc = torch.clamp(torch.arange(A, device=dev) + 1, max=A - 1)          # age a -> a+1 (index)
    incD = torch.clamp(torch.arange(A, device=dev) + D, max=A - 1)
    V = torch.zeros((RING,) + (A,) * N, dtype=torch.float64, device=dev)
    refi = (int(posts[0]),) + (0,) * N
    a_ = tau / D
    for it in range(1, max_iter + 1):
        W = V
        for ax in range(N):
            W = W.index_select(1 + ax, inc)
        W = W + r1
        Q = torch.maximum(torch.maximum(W.roll(1, 0), W), W.roll(-1, 0))
        new = Q.mul_(tau).add_(V, alpha=1 - tau)
        for j in range(N):
            t = V[int(posts[j])].select(j, 0)                               # ages of the others (axes order kept)
            others = [i for i in range(N) if i != j]
            for k in range(len(others)):
                t = t.index_select(k, incD)
            S = (R + t.unsqueeze(j)) * a_
            for dx in (-1, 0, 1):
                x = int((posts[j] - dx) % RING)
                new[x] = torch.maximum(new[x], S + (1 - a_) * V[x])
        diff = (new - V).reshape(-1)
        lo, hi = float(diff.min()), float(diff.max())
        V = new - new[refi]
        if (hi - lo) / tau < tol:
            break
    return 0.5 * (lo + hi) / tau, it, (hi - lo) / tau
