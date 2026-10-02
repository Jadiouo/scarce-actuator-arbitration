"""Batched, seed-exact torch (float64) simulator, equivalent to `arbitration.model.run`.

Batch dimension B = independent simulations (seed x load x variant ...); all of them advance one
step at a time on the GPU. Randomness: numpy's generators cannot be reproduced on a GPU, so ALL
random numbers are pre-drawn on the CPU with exactly the rng calls of model.run (same
generators, same order, same distributions), packed into tensors and uploaded. The dynamics
are then deterministic and reproduce model.run per seed (checked to ~1e-12). Float summations
whose order matters (np.mean of 8 values, the learner's prior) emulate numpy's summation order.

Supported policies: none, none_random, greedy_true, nearest, honest, strategic, priced,
index_true, honest_index, learned, learned_index, tour_sqrt, tour_sqrt_learned.
Not supported (python loops over a planner / DP lookups): rollout*, dp_full, dp_age (and, until
ported, the game policies fused*, audit*). Model variants: any ModelConfig (shock damage,
iid/ar1/bias noise, keep_commitment, precision_at, service), preempt, hyst, sigma, lam,
`stream`, and `inflate` (per-explorer claim offset b_i; may be (n,) or (B, n), so one batch can
hold many b vectors).

Adding a policy: subclass Policy, set `kind` = "scored" (decide() returns callers, score, by)
or "commit" (new_target(S, need) returns a target for sims with need=True), register it in
POLICY_CLASSES. `S` (class _State) exposes h, tau, claim, d, pos, posts, rates, target, t,
learner arrays (drop, el, t_reset) and per-sim params. on_arrival(S, arr) is called when the
actuator arrives (before the target reset) for the sims in the boolean mask `arr`.
"""
import math, os, time
from dataclasses import fields
import numpy as np
import torch
from .. import model as _m
from ..model import RING, SPEED, THETA, THETA_STRAT, R_PRIOR, make_config, ModelConfig

METRICS = ("mean_health", "min_health", "frac_below_02", "frac_any_dead", "precision",
           "n_services", "retargets_per_service", "frac_dead_robot_steps")
INF = float("inf")


def _dev(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ------------------------------------------------------------------ numpy-order sums
def _seq(x):
    s = torch.zeros_like(x[:, 0])
    for k in range(x.shape[1]):
        s = s + x[:, k]
    return s


def _pw8(x):
    return (((x[:, 0] + x[:, 1]) + (x[:, 2] + x[:, 3])) + ((x[:, 4] + x[:, 5]) + (x[:, 6] + x[:, 7])))


def np_sum(x):
    """Row sums in numpy's order for contiguous rows of length n (exact for n <= 8)."""
    return _pw8(x) if x.shape[1] == 8 else _seq(x)


# ------------------------------------------------------------------ random numbers (CPU)
_SHOCK_CACHE = {}


def _shock_draws(seed, steps, n, sk):
    key = (seed, steps, n, tuple(sk))
    if key not in _SHOCK_CACHE:
        srng = np.random.default_rng([seed, 7919] + sk)
        U = np.empty((steps, n))
        E = np.empty((steps, n))
        for t in range(steps):
            U[t] = srng.random(n)
            E[t] = srng.exponential(1.0, n)
        if len(_SHOCK_CACHE) > 256:
            _SHOCK_CACHE.clear()
        _SHOCK_CACHE[key] = (U, E)
    return _SHOCK_CACHE[key]


_DRAW_CACHE = {}


def draw_instance(seed, n, rate, spread, sigma, steps, cfg, stream=0, order=False):
    """Everything model.run draws for one sim, with the same calls in the same order.
    Returns dict(posts, rates, pos0, cn (steps,n) additive claim noise, bias (n,), order).
    posts, start position and noise do not depend on `rate`, so they are cached per
    (seed, n, spread, sigma, steps, noise model, stream)."""
    key = (seed, n, float(spread), float(sigma), steps, cfg.noise, cfg.ar1_phi, cfg.bias_frac, int(stream))
    if key not in _DRAW_CACHE:
        rng = np.random.default_rng(seed)
        posts = np.sort(rng.uniform(0, RING, n))
        nz = rng.normal(0, spread, n)
        sk = [] if stream == 0 else [int(stream)]
        irng = rng if stream == 0 else np.random.default_rng([seed, 555] + sk)
        nrng = np.random.default_rng([seed, 4243] + sk) if cfg.noise != "iid" else None
        bias = np.zeros(n)
        e0 = None
        if cfg.noise == "ar1":
            e0 = nrng.normal(0, sigma, n)
            phi, ar_sd = cfg.ar1_phi, sigma * math.sqrt(1 - cfg.ar1_phi ** 2)
        elif cfg.noise == "bias":
            bias = nrng.normal(0, sigma * math.sqrt(cfg.bias_frac), n)
            iid_sd = sigma * math.sqrt(1 - cfg.bias_frac)
        pos0 = float(rng.uniform(0, RING))
        if cfg.noise == "iid":
            cn = irng.normal(0, sigma, (steps, n))
        elif cfg.noise == "ar1":
            w = nrng.normal(0, ar_sd, (steps, n))      # same stream as `steps` calls of normal(0, ar_sd, n)
            cn = np.empty((steps, n))
            e = e0
            for t in range(steps):
                e = phi * e + w[t]
                cn[t] = e
        else:
            cn = nrng.normal(0, iid_sd, (steps, n))
        if len(_DRAW_CACHE) > 4096:
            _DRAW_CACHE.clear()
        _DRAW_CACHE[key] = (posts, nz, pos0, cn, bias)
    posts, nz, pos0, cn, bias = _DRAW_CACHE[key]
    out = dict(posts=posts, rates=rate * np.exp(nz), pos0=pos0, cn=cn, bias=bias)
    out["order"] = np.random.default_rng([seed, 1]).permutation(n) if order else np.arange(n)
    return out


# ------------------------------------------------------------------ state + policies
class _State:
    pass


def _index(tau_hat, rh, d, service):
    tt = d / SPEED + service
    return torch.minimum(torch.ones_like(tau_hat), tau_hat + rh * d / SPEED) / tt


class Policy:
    kind = "scored"
    uses_learner = False

    def init(self, S):
        pass

    def decide(self, S):          # -> callers (B,n) bool, score (B,n) or None, by "max" | "near"
        raise NotImplementedError

    def new_target(self, S, need):   # kind == "commit"
        raise NotImplementedError

    def on_arrival(self, S, arr):
        pass


class _Static(Policy):
    kind = "commit"

    state_names = ("rr",)

    def init(self, S):
        S.rr = torch.zeros(S.B, dtype=torch.long, device=S.dev)

    def new_target(self, S, need):
        t = S.order.gather(1, (S.rr % S.n)[:, None]).squeeze(1)
        S.rr = S.rr + need.long()
        return t


class _Tour(Policy):
    kind = "commit"
    state_names = ("tour_pass",)

    def init(self, S):
        S.stride = 1.0 / torch.sqrt(S.rates)
        S.tour_pass = S.stride / 2.0

    def new_target(self, S, need):
        t = S.tour_pass.argmin(1)
        oh = _onehot(t, S.n) & need[:, None]
        S.tour_pass = S.tour_pass + torch.where(oh, S.stride, torch.zeros_like(S.stride))
        return t


class _TourLearned(Policy):
    kind = "commit"
    uses_learner = True

    def new_target(self, S, need):
        return ((S.t - S.t_reset) * torch.sqrt(rhat(S))).argmax(1)


def _onehot(idx, n):
    return torch.arange(n, device=idx.device)[None, :] == idx[:, None]


def rhat(S):
    """Learner's rate estimate (see model._Learner.rhat), prior = numpy-ordered mean of seen ratios."""
    seen = S.el > 0
    ratio = S.drop / torch.clamp(S.el, min=1e-12)
    x = torch.where(seen, ratio, torch.zeros_like(ratio))
    cnt = seen.sum(1)
    s = torch.where(cnt == S.n, np_sum(x), _seq(x)) if S.n == 8 else _seq(x)
    prior = torch.where(cnt > 0, s / cnt.clamp(min=1), torch.full_like(s, R_PRIOR))
    return torch.where(seen, ratio, prior[:, None])


def tau_hat(S):
    return torch.minimum(torch.ones_like(S.h), rhat(S) * (S.t - S.t_reset))


class _Learned(Policy):
    uses_learner = True

    def decide(self, S):
        return torch.ones_like(S.h, dtype=torch.bool), tau_hat(S), "max"


class _LearnedIndex(Policy):
    uses_learner = True

    def decide(self, S):
        rh = rhat(S)
        th = torch.minimum(torch.ones_like(S.h), rh * (S.t - S.t_reset))
        return torch.ones_like(S.h, dtype=torch.bool), _index(th, rh, S.d, S.svc), "max"


class _Greedy(Policy):
    def decide(self, S):
        return S.tau > 0.05, S.tau, "max"


class _IndexTrue(Policy):
    def decide(self, S):
        return S.tau > 0.05, S.tau / (S.d / SPEED + S.svc), "max"


class _Nearest(Policy):
    def decide(self, S):
        return S.claim > THETA, None, "near"


class _Strategic(Policy):
    def decide(self, S):
        return S.claim > THETA_STRAT, None, "near"


class _Honest(Policy):
    def decide(self, S):
        return S.claim > THETA, S.claim, "max"


class _HonestIndex(Policy):
    def decide(self, S):
        return S.claim > THETA, S.claim / (S.d / SPEED + S.svc), "max"


class _Priced(Policy):
    def decide(self, S):
        return S.claim > THETA + S.lam[:, None] * S.d, S.claim, "max"



# ------------------------------------------------------------------ fused / audit (game) policies
FUSE_BMAX = 40


def age_prior_t(agei, r, p, f, G):
    """torch twin of model.age_prior for shock damage (items already known to be non-ceiling).
    agei, r, p, f: (M,) float64 -> pmf (M, G+1)."""
    M = agei.shape[0]
    base = (1.0 - f) * r * agei
    m = f * r / p
    cell = (torch.arange(G, device=r.device, dtype=torch.float64) + 0.5) / G                # (G,)
    # binomial pmf, k = 0..FUSE_BMAX, by the stable recursion
    q = p / (1.0 - p)
    pk = [torch.exp(agei * torch.log1p(-p))]
    for k in range(FUSE_BMAX):
        pk.append(pk[-1] * ((agei - k) / (k + 1)) * q)
    pk = torch.stack(pk, dim=1)                                                              # (M, 41)
    pk = pk / pk.sum(1, keepdim=True)
    x = torch.clamp(cell[None, :] - base[:, None], min=0.0)                                  # (M, G)
    z = x / m[:, None]
    # P(k, z) = 1 - sum_{j<k} e^{-z} z^j / j!  for integer k, accumulated sequentially in k (the order
    # of the numpy axis-0 sum). Absolute accuracy ~1e-15 (scipy.gammainc is relatively exact in the far
    # tail, where the cells are below the rounding noise of the cumulative F anyway); ~10x faster than
    # torch.special.gammainc, which is the bottleneck of the posterior otherwise.
    T = torch.exp(-z)
    head = torch.zeros_like(z)
    Ssum = None
    for k in range(1, FUSE_BMAX + 1):
        head = head + T
        Pk = torch.clamp(1.0 - head, min=0.0)
        term = pk[:, k, None] * Pk
        Ssum = term if Ssum is None else Ssum + term
        T = T * z / k
    F = pk[:, 0, None] * (cell[None, :] >= base[:, None]).to(torch.float64) + Ssum
    F = torch.minimum(F, torch.ones_like(F))
    F = torch.cummin(F.flip(1), dim=1).values.flip(1)
    cum = torch.cat([F, torch.ones(M, 1, dtype=torch.float64, device=r.device)], dim=1)
    pmf = cum - torch.cat([torch.zeros(M, 1, dtype=torch.float64, device=r.device), cum[:, :-1]], dim=1)
    return torch.clamp(pmf, min=0.0)


def fused_tau_t(claim, age, r, sd, shock, p, f, G, chunk=4096):
    """torch twin of model.fused_tau, vectorised over M items (all (M,) tensors; shock: bool).
    Sync-free (fixed shapes): non-shock / ceiling items get benign dummy inputs and are masked."""
    out = torch.minimum(torch.ones_like(r), r * torch.clamp(age, min=0.0))              # linear / f == 0
    agei = torch.trunc(torch.clamp(age, min=0.0))
    base = (1.0 - f) * r * agei
    ceil_ = (agei * p > 25) | (base >= 1.0)
    g = torch.arange(G + 1, device=r.device, dtype=torch.float64) / G
    todo = shock & ~ceil_
    one = torch.ones_like(r)
    ag_s, r_s = torch.where(todo, agei, 0.0 * one), torch.where(todo, r, 0.004 * one)
    p_s, f_s = torch.where(todo, p, 0.01 * one), torch.where(todo, f, 0.5 * one)
    pm = [age_prior_t(ag_s[a:a + chunk], r_s[a:a + chunk], p_s[a:a + chunk], f_s[a:a + chunk], G)
          for a in range(0, r.shape[0], chunk)]
    ceilpmf = torch.zeros(1, G + 1, dtype=torch.float64, device=r.device)
    ceilpmf[:, G] = 1.0
    pmf = torch.where(todo[:, None], torch.cat(pm, 0), ceilpmf)
    prior_mean = (pmf * g).sum(1)
    z = (claim[:, None] - g[None, :]) / torch.clamp(sd, min=1e-9)[:, None]
    lw = torch.log(pmf + 1e-300) - 0.5 * z * z
    w = torch.exp(lw - lw.amax(1, keepdim=True))
    post = (w * g).sum(1) / w.sum(1)
    res = torch.where(torch.isfinite(sd), post, prior_mean)
    return torch.where(shock, res, out)


class _GameFused(Policy):
    """fused, fused_index, audit_index, audit_penalty_index (rules of model.run).
    Per-sim flags from the sim's own ModelConfig / policy name, so one batch can mix them and mix
    penalty parameters. The posterior (the expensive part) is evaluated only for sims that
    actually decide this step (target < 0, or preempt)."""
    uses_learner = True
    sparse = True
    state_names = ("a_s", "a_k", "a_until", "a_dc", "a_dt", "a_nf")

    def __init__(self, names):
        self.names = names

    def init(self, S):
        dev, B, n = S.dev, S.B, S.n
        nm = np.array(self.names)
        T = lambda a, dt=None: torch.tensor(np.asarray(a), device=dev, dtype=dt)
        cf = S.cfgs
        self.index = T(nm != "fused")
        self.audit = T(np.isin(nm, ["audit_index", "audit_penalty_index", "audit_disp_index", "audit_disp_penalty_index"]))
        self.adisp = T(np.isin(nm, ["audit_disp_index", "audit_disp_penalty_index"]))
        self.penal = T(np.isin(nm, ["audit_penalty_index", "audit_disp_penalty_index"]))
        self.ignore = T([c.penalty_mode == "ignore" for c in cf])
        self.demote = T([c.penalty_mode == "demote" for c in cf])
        self.factor = T([c.penalty_factor for c in cf], torch.float64)
        self.n0 = T([c.audit_n0 for c in cf], torch.float64)
        self.z = T([c.audit_z for c in cf], torch.float64)
        self.Tpen = T([float(c.penalty_T) if nn in ("audit_penalty_index", "audit_disp_penalty_index") else 0.0
                       for c, nn in zip(cf, self.names)], torch.float64)
        self.known = T([c.fuse_known_rates for c in cf])
        self.shock = T([c.damage == "shock" and c.shock_frac != 0 for c in cf])
        self.p = T([c.shock_rate for c in cf], torch.float64)
        self.f = T([c.shock_frac for c in cf], torch.float64)
        Gs = {c.fuse_grid for c in cf}
        if len(Gs) != 1:
            raise ValueError("fuse_grid must be the same for all sims of a batch")
        self.G = Gs.pop()
        self.sig_a = torch.clamp(S.sigma, min=1e-9)
        z = torch.zeros(B, n, dtype=torch.float64, device=dev)
        S.a_s, S.a_k, S.a_until = z.clone(), z.clone(), torch.full((B, n), -1.0, dtype=torch.float64, device=dev)
        S.a_dc, S.a_dt = z.clone(), z.clone()      # claim / time at the last dispatch to each robot
        S.a_nf = z.clone()                         # number of times each robot was flagged

    def decide(self, S):
        B, n = S.B, S.n
        callers = torch.ones(B, n, dtype=torch.bool, device=S.dev)
        score = torch.zeros(B, n, dtype=torch.float64, device=S.dev)
        need = S.rows & S.sel
        cnt = need.sum()
        torch.maximum(S.maxneed, cnt, out=S.maxneed)
        S.ovf.add_(torch.clamp(cnt - S.kcap, min=0))
        K = min(S.kcap, B)
        rows = torch.arange(B, device=S.dev) if K >= B else torch.topk(need.to(torch.float32), K).indices
        valid = need[rows]
        rh = rhat(S)[rows]
        age = (S.t - S.t_reset)[rows]
        r_ = torch.where(self.known[rows][:, None], S.rates[rows], rh)
        claim = S.claim[rows]
        sigma = S.sigma[rows][:, None].expand(-1, n)
        aud = self.audit[rows][:, None]
        bias = S.a_s[rows] / (S.a_k[rows] + self.n0[rows][:, None])
        sdA = self.sig_a[rows][:, None] * torch.sqrt(1.0 + 1.0 / (S.a_k[rows] + self.n0[rows][:, None]))
        cl = torch.where(aud, claim - bias, claim)
        sd = torch.where(aud, sdA, sigma)
        pen = S.a_until[rows] > S.t
        sd = torch.where((self.penal[rows] & self.ignore[rows])[:, None] & pen,
                         torch.full_like(sd, INF), sd)
        R = rows.numel()
        flat = lambda x: x.reshape(-1)
        rep = lambda v: v[:, None].expand(R, n).reshape(-1)
        tp = fused_tau_t(flat(cl), flat(age), flat(r_), flat(sd), rep(self.shock[rows]),
                         rep(self.p[rows]), rep(self.f[rows]), self.G).reshape(R, n)
        sc = torch.where(self.index[rows][:, None], _index(tp, r_, S.d[rows], S.svc[rows]), tp)
        sc = torch.where((self.penal[rows] & self.demote[rows])[:, None] & pen,
                         sc * self.factor[rows][:, None], sc)
        return callers, score.index_copy(0, rows, torch.where(valid[:, None], sc, torch.zeros_like(sc))), "max"

    def on_arrival(self, S, arr):
        n = S.n
        tc = S.tc
        ar_n = torch.arange(n, device=S.dev)[None, :]
        # dispatch-time claim snapshot (audit_disp*): the claim of the robot just chosen
        disp = S.sel & self.adisp & (S.tnew >= 0) & (S.tnew != S.prevt)
        ohd = disp[:, None] & (ar_n == tc[:, None])
        S.a_dc = torch.where(ohd, S.claim, S.a_dc)
        S.a_dt = torch.where(ohd, S.tf.expand(S.B, n), S.a_dt)
        m = (arr & S.sel & self.audit)
        claim_t = S.claim.gather(1, tc[:, None])[:, 0]
        tau_t = S.tau.gather(1, tc[:, None])[:, 0]
        oh = m[:, None] & (ar_n == tc[:, None])
        # dispatch audit: sample = claim_at_dispatch - tau_at_dispatch, tau_at_dispatch ~ tau_now - rhat * dt,
        # minus a fixed calibration offset (truthful selection bias; independent of the reports)
        cd = S.a_dc.gather(1, tc[:, None])[:, 0]
        dt = S.tf - S.a_dt.gather(1, tc[:, None])[:, 0]
        rh = rhat(S).gather(1, tc[:, None])[:, 0]
        samp_d = (cd - (tau_t - rh * dt)) - S.disp_offset
        samp = torch.where(self.adisp, samp_d, claim_t - tau_t)
        S.a_s = S.a_s + torch.where(oh, samp[:, None], torch.zeros_like(S.a_s))
        S.a_k = S.a_k + oh.to(torch.float64)
        flag = oh & (S.a_s / (self.sig_a[:, None] * torch.sqrt(S.a_k)) > self.z[:, None])
        S.a_until = torch.where(flag, S.tf + self.Tpen[:, None], S.a_until)
        S.a_nf = S.a_nf + flag.to(torch.float64)


class _Mixed(Policy):
    """Different 'max'-type scored policies per sim (one batch can hold several mechanisms)."""
    def __init__(self, groups):
        self.groups = groups          # list of (policy object, bool mask over sims) built by the factory

    @property
    def uses_learner(self):
        return any(p.uses_learner for p, _ in self.groups)

    @property
    def sparse(self):
        return any(getattr(p, "sparse", False) for p, _ in self.groups)

    @property
    def state_names(self):
        return tuple(k for p, _ in self.groups for k in getattr(p, "state_names", ()))

    def init(self, S):
        for p, m in self.groups:
            S.sel = m
            p.init(S)

    def decide(self, S):
        callers = score = None
        for p, m in self.groups:
            S.sel = m
            c, sc, by = p.decide(S)
            if by != "max":
                raise NotImplementedError("only 'max'-type policies can be mixed")
            if callers is None:
                callers, score = c, sc
            else:
                callers, score = torch.where(m[:, None], c, callers), torch.where(m[:, None], sc, score)
        return callers, score, "max"

    def on_arrival(self, S, arr):
        for p, m in self.groups:
            S.sel = m
            p.on_arrival(S, arr)


GAME_NAMES = ("fused", "fused_index", "audit_index", "audit_penalty_index",
              "audit_disp_index", "audit_disp_penalty_index")


def build_policy(names, dev):
    """names: list (per sim) of policy names -> policy object (a _Mixed if several kinds)."""
    nm = np.array(names)
    kinds = {}
    for k in dict.fromkeys(names):
        kinds.setdefault("game" if k in GAME_NAMES else k, []).append(k)
    groups = []
    for key, ks in kinds.items():
        mask = torch.tensor(np.isin(nm, ks), device=dev)
        if key == "game":
            groups.append((_GameFused([x if x in ks else "fused_index" for x in names]), mask))
        else:
            groups.append((POLICY_CLASSES[key](), mask))
    if len(groups) == 1:
        return groups[0][0], groups[0][1]
    return _Mixed(groups), None


POLICY_CLASSES = {
    "none": _Static, "none_random": _Static, "tour_sqrt": _Tour, "tour_sqrt_learned": _TourLearned,
    "greedy_true": _Greedy, "index_true": _IndexTrue, "nearest": _Nearest, "strategic": _Strategic,
    "honest": _Honest, "honest_index": _HonestIndex, "priced": _Priced,
    "learned": _Learned, "learned_index": _LearnedIndex,
}
SUPPORTED = tuple(POLICY_CLASSES) + GAME_NAMES


# ------------------------------------------------------------------ the simulator
def _vec(x, B, dtype, dev):
    a = np.broadcast_to(np.asarray(x, dtype=dtype), (B,)).copy()
    return torch.tensor(a, device=dev)


_STATE = ("h", "target", "serving", "drop", "el", "t_reset", "pos", "hacc", "acc", "num", "den",
          "d_num", "d_den", "n_serv", "retargets")


def simulate(policy, seeds, n=8, rate=0.02, steps=1500, spread=0.0, lam=0.006, preempt=False,
             hyst=0.0, sigma=_m.SIGMA, model=None, inflate=None, stream=0, device=None,
             graph=None, burn=0, imode=None, ipar=None, disp_offset=0.0, **opts):
    """Run B = len(seeds) sims at once. rate/spread/lam/preempt/hyst/sigma: scalar or length-B.
    model: None | "legacy" | "v2" | ModelConfig, or a length-B list of those (one config per sim; the
    sims then differ in damage / noise / keep_commitment / precision_at / service). inflate: None,
    (n,) or (B, n); imode / ipar (B, n) give each explorer's inflation RULE: mode 0 constant, 1 only while the
    actuator is undecided (target < 0), 2 only while the actuator is farther than ipar, 3 only while the
    robot's own tau < ipar, 4 only while the actuator is committed (target >= 0), 5 adaptive (cap ipar, stays just under the
    audit flag threshold). burn: steps excluded from
    the health metrics. disp_offset: calibration offset of the audit_disp policies. graph: capture one step in a CUDA graph and replay it (default: on for CUDA;
    removes the kernel-launch overhead that otherwise dominates). Returns dict metric -> np.array(B)
    (+ "robot_health" (B, n))."""
    _t0 = time.time()
    dev = _dev(device)
    seeds = [int(s) for s in seeds]
    B = len(seeds)
    names = [policy] * B if isinstance(policy, str) else list(policy)
    if len(names) != B:
        raise ValueError("policy list must have one entry per sim")
    for nm_ in set(names):
        if nm_ not in POLICY_CLASSES and nm_ not in GAME_NAMES:
            raise NotImplementedError(f"policy {nm_!r} not supported on GPU; have {SUPPORTED}")
    f64 = np.float64
    if isinstance(model, (list, tuple)):
        if len(model) != B:
            raise ValueError("model list must have one entry per sim")
        cfgs = [make_config(m, **opts) for m in model]
    else:
        cfgs = [make_config(model, **opts)] * B
    rate_v, spread_v, sigma_v = (np.broadcast_to(np.asarray(v, f64), (B,)) for v in (rate, spread, sigma))
    pol, sel0 = build_policy(names, dev)
    graph = (dev.type == "cuda") if graph is None else graph
    sparse = bool(getattr(pol, "sparse", False))
    is_shock = np.array([c.damage == "shock" and c.shock_frac != 0 for c in cfgs])
    any_shock, all_shock = bool(is_shock.any()), bool(is_shock.all())

    posts = np.empty((B, n)); rates = np.empty((B, n)); pos0 = np.empty(B)
    bias = np.empty((B, n)); order = np.empty((B, n), dtype=np.int64)
    cn_key, cn_list, cidx = {}, [], np.empty(B, dtype=np.int64)
    for b in range(B):
        d = draw_instance(seeds[b], n, rate_v[b], spread_v[b], sigma_v[b], steps, cfgs[b], stream,
                          order=names[b] == "none_random")
        posts[b], rates[b], pos0[b], bias[b], order[b] = d["posts"], d["rates"], d["pos0"], d["bias"], d["order"]
        k = (seeds[b], float(sigma_v[b]), cfgs[b].noise, cfgs[b].ar1_phi, cfgs[b].bias_frac)
        if k not in cn_key:
            cn_key[k] = len(cn_list)
            cn_list.append(d["cn"])
        cidx[b] = cn_key[k]
    S = _State()
    S.dev, S.B, S.n = dev, B, n
    S.cfgs = cfgs
    S.sigma = _vec(sigma, B, f64, dev)
    S.sel = sel0 if sel0 is not None else torch.ones(B, dtype=torch.bool, device=dev)
    T = lambda a: torch.tensor(a, device=dev)
    S.posts, S.rates, S.bias, S.order = T(posts), T(rates), T(bias), T(order)
    S.svc = T(np.array([c.service for c in cfgs]))[:, None].to(torch.float64)       # (B,1)
    svc_long = T(np.array([c.service for c in cfgs]))
    CN = T(np.stack(cn_list, axis=1))                 # (steps, #distinct noise paths, n)
    cidx = T(cidx)
    S.lam = _vec(lam, B, f64, dev)
    hyst_t = _vec(hyst, B, f64, dev)
    pre_t = _vec(preempt, B, bool, dev)
    keep_t = T(np.array([c.keep_commitment for c in cfgs]))
    disp_t = T(np.array([c.precision_at == "dispatch" for c in cfgs]))
    infl = None
    if inflate is not None:
        infl = T(np.broadcast_to(np.asarray(inflate, f64), (B, n)).copy())
    imode_t = None if imode is None else T(np.broadcast_to(np.asarray(imode, np.int64), (B, n)).copy())
    ipar_t = T(np.broadcast_to(np.asarray(0.0 if ipar is None else ipar, f64), (B, n)).copy())
    S.disp_offset = _vec(disp_offset, B, f64, dev)
    has_adapt = imode_t is not None and bool((imode_t == 5).any())
    adapt_z = T(np.array([c.audit_z for c in cfgs]))[:, None].to(torch.float64)
    adapt_sig = torch.clamp(S.sigma, min=1e-9)[:, None]
    zeros_bn = torch.zeros(B, n, dtype=torch.float64, device=dev)
    burn = int(burn)
    minbuf = torch.zeros(steps, B, dtype=torch.float64, device=dev)
    if any_shock:
        uniq = sorted(set(s for s, sh in zip(seeds, is_shock) if sh))
        sk = [] if stream == 0 else [int(stream)]
        dr = [_shock_draws(s, steps, n, sk) for s in uniq]
        SU = T(np.stack([d[0] for d in dr], axis=1))          # (steps, S, n)
        SE = T(np.stack([d[1] for d in dr], axis=1))
        sidx = T(np.array([uniq.index(s) if sh else 0 for s, sh in zip(seeds, is_shock)]))
        sf = T(np.array([c.shock_frac for c in cfgs]))[:, None]
        sr = T(np.array([c.shock_rate for c in cfgs]))[:, None]
        base_decay = S.rates * (1.0 - sf)
        shock_size = sf * S.rates / sr
        shock_m = T(is_shock)[:, None]

    f = torch.float64
    P = {k: None for k in _STATE}
    P["h"] = torch.ones(B, n, dtype=f, device=dev)
    P["target"] = torch.full((B,), -1, dtype=torch.long, device=dev)
    P["serving"] = torch.zeros(B, dtype=torch.long, device=dev)
    for k in ("drop", "el", "t_reset", "hacc"):
        P[k] = torch.zeros(B, n, dtype=f, device=dev)
    P["pos"] = T(pos0)
    P["acc"] = torch.zeros(B, 5, dtype=f, device=dev)
    for k in ("num", "den", "d_num", "d_den", "n_serv", "retargets"):
        P[k] = torch.zeros(B, dtype=f, device=dev)
    S.tf = torch.zeros((), dtype=f, device=dev)                 # time as device scalars (graph-safe)
    S.tidx = torch.zeros(1, dtype=torch.long, device=dev)
    S.maxneed = torch.zeros((), dtype=torch.long, device=dev)
    S.ovf = torch.zeros((), dtype=torch.long, device=dev)
    pol.init(S)
    pnames = list(getattr(pol, "state_names", ()))
    for k in pnames:
        P[k] = getattr(S, k)
    zeros = torch.zeros(B, n, dtype=f, device=dev)
    ar = torch.arange(n, device=dev)
    ring_half = RING / 2

    def step():
        for k in _STATE:
            setattr(S, k, P[k])
        for k in pnames:
            setattr(S, k, P[k])
        S.t = S.tf
        if any_shock:
            hit = (SU.index_select(0, S.tidx)[0][sidx] < sr) * SE.index_select(0, S.tidx)[0][sidx] * shock_size
            hs = S.h - base_decay - hit
            if all_shock:
                h = torch.clamp(hs, 0.0, 1.0)
            else:
                h = torch.clamp(torch.where(shock_m, hs, S.h - S.rates), 0.0, 1.0)
        else:
            h = torch.clamp(S.h - S.rates, 0.0, 1.0)
        S.h = h
        S.tau = tau = 1.0 - h
        wb = (S.tf >= burn).to(f)                       # 0 during the burn-in
        hacc = S.hacc + wb * h
        dead = h <= 0
        hmin = h.amin(1)
        acc = S.acc + wb * torch.stack([np_sum(h) / n, hmin, (h < 0.2).sum(1).to(f) / n,
                                        dead.any(1).to(f), dead.sum(1).to(f) / n], dim=1)
        minbuf.index_copy_(0, S.tidx, hmin[None])
        d = torch.abs(S.posts - S.pos[:, None])
        S.d = d = torch.minimum(d, RING - d)
        S.claim = claim = (tau + S.bias) + CN.index_select(0, S.tidx)[0][cidx]
        if infl is not None:
            if imode_t is None:
                S.claim = claim = claim + infl
            else:
                undecided = (S.target < 0)[:, None]
                on = torch.where(imode_t == 0, torch.ones_like(undecided.expand(B, n)),
                     torch.where(imode_t == 1, undecided.expand(B, n),
                     torch.where(imode_t == 2, d > ipar_t,
                     torch.where(imode_t == 3, tau < ipar_t, (~undecided).expand(B, n)))))
                eff = infl * on.to(f)
                if has_adapt:       # mode 5: largest constant inflation that keeps the NEXT audit sample under the flag threshold
                    sa = getattr(S, "a_s", zeros_bn)
                    ka = getattr(S, "a_k", zeros_bn)
                    cap = torch.clamp(torch.minimum(ipar_t, adapt_z * adapt_sig * torch.sqrt(ka + 1.0) - sa - 0.01), min=0.0)
                    eff = torch.where(imode_t == 5, cap, eff)
                S.claim = claim = claim + eff

        srv = S.serving > 0
        serving = torch.where(srv, S.serving - 1, S.serving)
        fin = srv & (serving == 0)
        prev = S.target
        tgt = prev.clamp(min=0)
        oh_t = (ar[None, :] == tgt[:, None])
        act = ~srv
        retargets = S.retargets

        if pol.kind == "commit":
            need = act & (prev < 0)
            newt = pol.new_target(S, need)
            tnew = torch.where(need, newt, prev)
            mover = act
        else:
            S.rows = act & ((prev < 0) | pre_t)
            callers, score, by = pol.decide(S)
            anyc = callers.any(1)
            if by == "max":
                best = torch.where(callers, score, torch.full_like(score, -INF)).argmax(1)
            else:
                best = torch.where(callers, d, torch.full_like(d, INF)).argmin(1)
            cur_c = prev.clamp(min=0)
            if by == "max":
                sb, sc = score.gather(1, best[:, None])[:, 0], score.gather(1, cur_c[:, None])[:, 0]
                cond = torch.where(hyst_t > 0, sb > sc + hyst_t, torch.ones_like(sb, dtype=torch.bool))
            else:
                db, dc = d.gather(1, best[:, None])[:, 0], d.gather(1, cur_c[:, None])[:, 0]
                cond = torch.where(hyst_t > 0, db < dc - hyst_t * RING, torch.ones_like(db, dtype=torch.bool))
            sw = pre_t & (prev >= 0) & (best != prev) & cond
            new = torch.where(prev < 0, best, torch.where(sw, best, prev))
            tnc = torch.where((prev >= 0) & keep_t, prev, torch.full_like(prev, -1))
            tnew = torch.where(anyc, new, tnc)
            retargets = retargets + (act & anyc & sw).to(f)
            mover = act & (tnew >= 0)
        tnew = torch.where(act, tnew, prev)
        tc = tnew.clamp(min=0)
        tau_t = tau.gather(1, tc[:, None])[:, 0]
        tau_max = torch.clamp(tau.amax(1), min=1e-9)
        chg = mover & (tnew != prev) & (tnew >= 0)
        d_num = torch.where(chg, tau_t, S.d_num)
        d_den = torch.where(chg, tau_max, S.d_den)
        tpos = S.posts.gather(1, tc[:, None])[:, 0]
        dd = tpos - S.pos
        dd = torch.where(dd > ring_half, dd - RING, dd)
        dd = torch.where(dd < -ring_half, dd + RING, dd)
        arrived = torch.abs(dd) <= SPEED
        newpos = torch.where(arrived, tpos, torch.remainder(S.pos + torch.sign(dd) * SPEED, RING))
        pos = torch.where(mover, newpos, S.pos)
        arr = mover & arrived
        serving = torch.where(arr, svc_long, serving)
        n_serv = S.n_serv + arr.to(f)
        drop, el = S.drop, S.el
        if pol.uses_learner:
            el_ = S.tf - S.t_reset.gather(1, tc[:, None])[:, 0]
            okm = (arr & (el_ > 0))[:, None] & (ar[None, :] == tc[:, None])
            h_obs = h.gather(1, tc[:, None])[:, 0]
            drop = drop + torch.where(okm, (1.0 - h_obs)[:, None].expand(B, n), zeros)
            el = el + torch.where(okm, el_[:, None].expand(B, n), zeros)
        S.tc, S.tnew, S.prevt = tc, tnew, prev
        pol.on_arrival(S, arr)
        zb = torch.zeros_like(S.num)
        num = S.num + torch.where(arr, torch.where(disp_t, d_num, tau_t), zb)
        den = S.den + torch.where(arr, torch.where(disp_t, d_den, tau_max), zb)
        fin_oh = oh_t & fin[:, None]
        h_out = torch.where(fin_oh, torch.ones_like(h), h)
        t_reset = torch.where(fin_oh, S.tf.expand(B, n), S.t_reset)
        target = torch.where(fin, torch.full_like(tnew, -1), tnew)
        new_state = dict(h=h_out, target=target, serving=serving, drop=drop, el=el, t_reset=t_reset,
                         pos=pos, hacc=hacc, acc=acc, num=num, den=den, d_num=d_num, d_den=d_den,
                         n_serv=n_serv, retargets=retargets)
        for k in pnames:
            new_state[k] = getattr(S, k)
        for k, v in new_state.items():
            P[k].copy_(v)
        S.tf.add_(1.0)
        S.tidx.add_(1)

    def make_graph():
        snap = {k: v.clone() for k, v in P.items()}
        snap_t = (S.tf.clone(), S.tidx.clone(), S.maxneed.clone(), S.ovf.clone())
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        for k, v in snap.items():
            P[k].copy_(v)
        for dst, src in zip((S.tf, S.tidx, S.maxneed, S.ovf), snap_t):
            dst.copy_(src)
        return g

    def snapshot():
        return ({k: v.clone() for k, v in P.items()}, S.tf.clone(), S.tidx.clone())

    def restore(sn):
        for k, v in sn[0].items():
            P[k].copy_(v)
        S.tf.copy_(sn[1]); S.tidx.copy_(sn[2])

    S.kcap = B
    if sparse and graph and steps > 1:
        # step 0 (every sim decides at once) runs eagerly at full capacity; the rest is replayed from a
        # CUDA graph whose decision capacity K is a guess. A sim that needs a decision beyond K would be
        # delayed, so overflow is counted and, if any, the run restarts from after step 0 with a larger K.
        step()
        S.maxneed.zero_()
        sn = snapshot()
        kcap = min(B, int(0.15 * B) + 16)
        while True:
            S.kcap = kcap
            S.ovf.zero_()
            g = make_graph()
            left, bad = steps - 1, False
            while left > 0:
                k = min(left, 400)
                for _ in range(k):
                    g.replay()
                left -= k
                if kcap < B and int(S.ovf) > 0:
                    bad = True
                    break
            if os.environ.get("SIMDBG"):
                print("graph pass kcap", kcap, "ovf", int(S.ovf), "maxneed", int(S.maxneed), time.time() - _t0, flush=True)
            if not bad:
                break
            kcap = min(B, max(2 * kcap, int(1.5 * int(S.maxneed)) + 16))
            restore(sn)
    elif graph and steps > 0:
        g = make_graph()
        for _ in range(steps):
            g.replay()
    else:
        for _ in range(steps):
            step()

    nb = max(steps - burn, 1)
    m = (P["acc"] / nb).cpu().numpy()
    mm = minbuf[burn:]
    q = torch.empty(B, dtype=torch.float64, device=dev)
    if mm.shape[0] > 0:
        pos_ = 0.05 * (mm.shape[0] - 1)
        lo_, hi_ = int(math.floor(pos_)), int(math.ceil(pos_))
        for a_ in range(0, B, 2048):
            ss = torch.sort(mm[:, a_:a_ + 2048], dim=0).values
            q[a_:a_ + 2048] = ss[lo_] + (ss[hi_] - ss[lo_]) * (pos_ - lo_)
    num_c, den_c, ns, rt = (P[k].cpu().numpy() for k in ("num", "den", "n_serv", "retargets"))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dict(mean_health=m[:, 0], min_health=m[:, 1], frac_below_02=m[:, 2], frac_any_dead=m[:, 3],
                   precision=np.where(den_c > 0, num_c / den_c, np.nan), n_services=ns,
                   retargets_per_service=np.where(ns > 0, rt / ns, np.nan), frac_dead_robot_steps=m[:, 4],
                   robot_health=(P["hacc"] / nb).cpu().numpy(), min_health_p05=q.cpu().numpy(),
                   audit_sum=P["a_s"].cpu().numpy() if "a_s" in P else None,
                   audit_cnt=P["a_k"].cpu().numpy() if "a_k" in P else None,
                   audit_flags=P["a_nf"].cpu().numpy() if "a_nf" in P else None)
    return out


# ------------------------------------------------------------------ drop-in for experiments.collect
_CFG_KEYS = {f.name for f in fields(ModelConfig)}


def collect_gpu(cells, seeds, steps, n=8, spread=0.6, device=None):
    """Same contract as experiments.collect: cells {key: (policy, kwargs)} -> {key: {metric: array}}.
    All cells of one policy (any model config, rate, preempt, hyst, lam, sigma) form one batch."""
    seeds = list(seeds)
    S_ = len(seeds)
    groups = {}
    for key, (policy, kw) in cells.items():
        kw = dict(kw)
        model = kw.pop("model", None)
        opts = {k: kw.pop(k) for k in list(kw) if k in _CFG_KEYS}
        groups.setdefault(policy, []).append((key, make_config(model, **opts), kw))
    res = {}
    defaults = dict(rate=0.02, lam=0.006, preempt=False, hyst=0.0, sigma=_m.SIGMA, spread=spread)
    for policy, items in groups.items():
        per = {name: np.concatenate([np.full(S_, kw.get(name, defaults[name]), dtype=float)
                                     for _, _, kw in items]) for name in defaults}
        cfgs = [c for _, c, _ in items for _ in range(S_)]
        out = simulate(policy, seeds * len(items), n=n, steps=steps, model=cfgs, device=device, **per)
        for i, (key, _, _) in enumerate(items):
            res[key] = {m: out[m][i * S_:(i + 1) * S_] for m in METRICS}
    return {k: res[k] for k in cells}
