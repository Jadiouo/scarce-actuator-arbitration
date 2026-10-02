"""Batched torch (float64) relative value iteration for the full-information DP.

Same model, same Schweitzer transformation, same stopping rule, same per-iteration
arithmetic as `arbitration.dp.FullDP` (the one-step reward / transition matrices are even
built by the numpy code, `dp.Dyn`, so the discretisation is identical). The only
difference is that the Bellman backup of a whole batch of instances (different seeds /
posts / rates) runs as batched matmuls on the GPU. Instances leave the batch as soon as
THEIR OWN bracket (span < tol) closes, so each instance stops at the same iteration the
numpy solver would.

    sols = solve_full(instances, cfg, K=60)      # instances = [(posts, rates), ...]
    sols[0].g, .iters, .span, .V (list of numpy arrays), .to_numpy_policy()

AgeDP is not ported: it is a gather-only recursion that takes ~0.1-0.6 s per instance on the CPU
(measured), so it is not a bottleneck.
"""
import math
import numpy as np
import torch
from .. import dp as _dp
from .. import model as _m
from ..dp import Dyn, cycle_lengths, TOL, MAX_ITER


def _dev(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class FullDPTorch:
    """Result for one instance (mirrors the attributes of dp.FullDP)."""
    kind = "dp_full"

    def __init__(self, posts, rates, cfg, K):
        self.posts, self.rates, self.cfg, self.K = posts, rates, cfg, K
        self.n = len(posts)

    def to_numpy_policy(self):
        """A genuine dp.FullDP carrying the torch solution (for q_values/choose)."""
        f = object.__new__(_dp.FullDP)
        f.posts, f.rates, f.n, f.K, f.svc = self.posts, self.rates, self.n, self.K, self.cfg.service
        f.dyn = _dp._dyns(self.rates, self.cfg)
        f.Ls = cycle_lengths(self.posts, f.svc)
        f.tau = 0.5 * float(f.Ls.min())
        f.m1 = [dy.trans(1, [1.0], self.K)[0] for dy in f.dyn]
        f.V, f.g, f.iters, f.span = self.V, self.g, self.iters, self.span
        return f

    def choose(self, pos, h, ages=None):
        if not hasattr(self, "_np"):
            self._np = self.to_numpy_policy()
        return self._np.choose(pos, h, ages)


def _setup(posts, rates, cfg, K):
    """numpy (identical to FullDP.__init__) pieces: Ls, tau, m1, {(i,L): M}, {L: rz[i]}."""
    N = len(posts)
    dyn = _dp._dyns(rates, cfg)
    Ls = cycle_lengths(np.asarray(posts, float), cfg.service)
    g = np.arange(K + 1) / K
    m1 = np.stack([dy.trans(1, [1.0], K)[0] for dy in dyn])
    M, rz = {}, {}
    for L in sorted(set(Ls.ravel().tolist())):
        M[L] = np.stack([dyn[i].trans(L, g, K) for i in range(N)])        # (N, K+1, K+1)
        rz[L] = np.stack([dyn[i].rz(L, g) for i in range(N)])             # (N, K+1)
    return Ls, 0.5 * float(Ls.min()), m1, M, rz


def _contract(Va, a, m1, Ms, N, idx_others):
    """cont[b, ...] = sum over other axes of M_i @ (m1 . V_a); axis a left with size 1."""
    # contract axis a (position a+1) with m1 (B, K+1)
    t = torch.movedim(Va, a + 1, -1)                                  # (B, rest..., K+1)
    B = t.shape[0]
    t = torch.bmm(t.reshape(B, -1, t.shape[-1]), m1.unsqueeze(-1)).reshape(t.shape[:-1])
    for idx, i in enumerate(idx_others):                              # axes of t after removing a
        ax = idx + 1
        u = torch.movedim(t, ax, -1)
        sh = u.shape
        u = torch.bmm(u.reshape(B, -1, sh[-1]), Ms[i].transpose(1, 2)).reshape(sh)
        t = torch.movedim(u, -1, ax)
    return t.unsqueeze(a + 1)


def solve_full(instances, cfg, K=None, tol=TOL, max_iter=MAX_ITER, device=None, verbose=False):
    """Solve len(instances) full-information DPs at once. instances: [(posts, rates), ...]."""
    dev = _dev(device)
    K = int(K or cfg.dp_k)
    B = len(instances)
    N = len(instances[0][0])
    if N > 5:
        raise ValueError("dp is for small N")
    dims = (K + 1,) * N
    ax_others = {a: [i for i in range(N) if i != a] for a in range(N)}
    ctx = [_setup(np.asarray(p, float), np.asarray(r, float), cfg, K) for p, r in instances]
    Ls = np.stack([c[0] for c in ctx])                                   # (B, N, N)
    tau = np.array([c[1] for c in ctx])
    m1_all = torch.tensor(np.stack([c[2] for c in ctx]), device=dev)     # (B, N, K+1)
    # per distinct L: instances needing it, stacked M / rz
    Lset = sorted(set(Ls.ravel().tolist()))
    need = {L: np.nonzero((Ls == L).any(axis=(1, 2)))[0] for L in Lset}
    row = {L: {int(b): r for r, b in enumerate(need[L])} for L in Lset}
    M_store = {L: torch.tensor(np.stack([ctx[b][3][L] for b in need[L]]), device=dev) for L in Lset}
    rz_store = {L: torch.tensor(np.stack([ctx[b][4][L] for b in need[L]]), device=dev) for L in Lset}

    def rw_of(L, rows):            # (|rows|,)+dims  == tot / N, tot summed in the numpy order
        r = rz_store[L][rows]
        tot = None
        for i in range(N):
            shp = [r.shape[0]] + [1] * N
            shp[1 + i] = K + 1
            term = r[:, i].reshape(shp)
            tot = term if tot is None else tot + term
        # numpy starts from zeros: 0 + term0 is exact, so identical
        return tot / N

    def build_plan(act):
        """act: numpy array of global ids. -> list of per-(a, L) work items."""
        pos_of = {int(b): p for p, b in enumerate(act)}
        plan = []
        for a in range(N):
            for L in sorted(set(Ls[act][:, :, a].ravel().tolist())):
                S = [b for b in act if (Ls[b, :, a] == L).any()]
                S_loc = torch.tensor([pos_of[int(b)] for b in S], device=dev)
                rows = torch.tensor([row[L][int(b)] for b in S], device=dev)
                Sg = torch.tensor(S, device=dev)
                js = []
                for j in range(N):
                    sel = [k for k, b in enumerate(S) if Ls[b, j, a] == L]
                    if not sel:
                        continue
                    gl = torch.tensor([pos_of[int(S[k])] for k in sel], device=dev)
                    li = torch.tensor(sel, device=dev)
                    t_ = torch.tensor(tau[[S[k] for k in sel]], device=dev)
                    c1 = (t_ / L).reshape((-1,) + (1,) * N)
                    js.append((j, gl, li, c1, len(sel) == len(act) and sel == list(range(len(sel)))))
                plan.append((a, L, S_loc, rows, Sg, js, len(S) == len(act)))
        return plan

    act = np.arange(B)
    plan = build_plan(act)
    V = [torch.zeros((B,) + dims, dtype=torch.float64, device=dev) for _ in range(N)]
    out = [None] * B
    ref_idx = (slice(None),) + (K,) * N
    ar = torch.arange(1)
    it = 0
    while len(act) and it < max_iter:
        it += 1
        nb = len(act)
        best = [torch.full((nb,) + dims, -float("inf"), dtype=torch.float64, device=dev)
                for _ in range(N)]
        for a, L, S_loc, rows, Sg, js, full in plan:
            Va = V[a] if full else V[a][S_loc]
            Ms = {i: M_store[L][rows][:, i] for i in ax_others[a]}
            C = _contract(Va, a, m1_all[Sg][:, a], Ms, N, ax_others[a])
            Rw = rw_of(L, rows)
            q = Rw + C
            for j, gl, li, c1, jfull in js:
                if jfull:
                    val = c1 * q + (1.0 - c1) * V[j]
                    best[j] = torch.maximum(best[j], val)
                else:
                    val = c1 * q[li] + (1.0 - c1) * V[j][gl]
                    best[j][gl] = torch.maximum(best[j][gl], val)
        lo = torch.full((nb,), float("inf"), dtype=torch.float64, device=dev)
        hi = torch.full((nb,), -float("inf"), dtype=torch.float64, device=dev)
        for j in range(N):
            diff = (best[j] - V[j]).reshape(nb, -1)
            lo = torch.minimum(lo, diff.amin(1))
            hi = torch.maximum(hi, diff.amax(1))
        refval = best[0][ref_idx]
        V = [w - refval.reshape((-1,) + (1,) * N) for w in best]
        done = ((hi - lo) < tol) | (it >= max_iter)
        if bool(done.any()):
            dn = done.cpu().numpy()
            lo_c, hi_c = lo.cpu().numpy(), hi.cpu().numpy()
            for p in np.nonzero(dn)[0]:
                b = int(act[p])
                out[b] = (0.5 * (lo_c[p] + hi_c[p]), it, hi_c[p] - lo_c[p],
                          [v[p].cpu().numpy() for v in V])
            keep = torch.tensor(np.nonzero(~dn)[0], device=dev)
            act = act[~dn]
            if len(act):
                V = [v[keep] for v in V]
                plan = build_plan(act)
        if verbose and it % 50 == 0:
            print(f"  it {it} active {len(act)}", flush=True)
    sols = []
    for b, (p, r) in enumerate(instances):
        gt, its, span, Vn = out[b]
        s = FullDPTorch(np.asarray(p, float), np.asarray(r, float), cfg, K)
        s.tau = tau[b]
        s.g, s.iters, s.span, s.V = gt / tau[b], its, span, Vn
        sols.append(s)
    return sols


def grid_policy(sol, cfg=None):
    """Greedy action on the grid for every (post j, state): argmax_a of the transformed backup,
    computed in numpy from sol.V with the numpy FullDP building blocks. Returns (N, K+1, ...) ints."""
    f = sol.to_numpy_policy()
    N, K = f.n, f.K
    g = np.arange(K + 1) / K
    f._M, f._Rw = {}, {}
    for L in set(f.Ls.ravel().tolist()):
        for i in range(N):
            f._M[i, L] = f.dyn[i].trans(L, g, K)
        tot = np.zeros((K + 1,) * N)
        for i in range(N):
            shp = [1] * N
            shp[i] = K + 1
            tot = tot + f.dyn[i].rz(L, g).reshape(shp)
        f._Rw[L] = tot / N
    pol = []
    for j in range(N):
        vals = []
        for a in range(N):
            L = int(f.Ls[j, a])
            q = f._Rw[L] + f._cont(f.V[a], a, L)
            vals.append(np.broadcast_to((f.tau / L) * q + (1 - f.tau / L) * f.V[j], (K + 1,) * N))
        pol.append(np.argmax(np.stack(vals), axis=0))
    return np.stack(pol)
