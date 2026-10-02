"""
Arbitration of a scarce, non-substitutable actuator in a robot team.

N explorers sit at fixed posts on a ring. Explorer i's perception health h_i
decays at its own private rate r_i and is restored to 1 when the single
actuator `c` reaches it and completes a service. `c` moves at finite speed, so
every dispatch decision costs travel time. Each explorer observes only a noisy
estimate of its own urgency tau_i = 1 - h_i, and requesting help is free.

The model exists to compare dispatch policies that differ in how much urgency
information they use -- including one handed the ground truth (`greedy_true`).
That rule is a MYOPIC GREEDY heuristic, not an optimal oracle: it ignores travel
and service time entirely, so it is not a performance upper bound. All policies
draw from the same random stream for a given seed, so arms are comparable.

Policies
--------
none         blind round-robin over posts in spatial order; ignores all requests
none_random  the same, in a random (non-spatial) order -- the weaker baseline
greedy_true  serves argmax of the TRUE urgency (myopic greedy; ignores travel cost;
             NOT an optimal oracle)
nearest      honest request threshold, but dispatches purely by distance
honest       explorers report their private estimate; serves argmax of claims
strategic    explorers always call and claim the ceiling, so claims are tied
             and the dispatcher falls back on distance. PLACEHOLDER: this is
             bit-identical to `nearest` with theta=0.05 (a low-threshold nearest),
             not an equilibrium of any game.
priced       honest reporting, but an explorer may call only if its urgency
             exceeds THETA + lam * (distance to the actuator)

Reference policies added in v2 (they do not change any earlier policy)
----------------------------------------------------------------------
index_true      cost-aware TRUE-state baseline: serves argmax tau_i/(d_i/SPEED +
                SERVICE) among robots with tau > 0.05 (urgency per unit of actuator time).
honest_index   reports-based cost-aware rule: among callers (claim > THETA) serves
                argmax claim_i/(d_i/SPEED + SERVICE); isolates the value of reports
                from the value of distance-awareness.
learned         ignores reports. The dispatcher observes true h only on arrival,
                keeps per-robot (sum of observed drop, sum of elapsed time since the
                last service) and estimates r_i as the ratio; predicts
                tau_i = min(1, r_i_hat * elapsed_i); serves argmax predicted tau.
                (Censoring: an observation at h=0 is a lower bound and is used as
                such, so r_i_hat is biased low for robots that sit at 0.)
learned_index   same estimates, distance-aware index
                min(1, tau_hat_i + r_hat_i*d_i/SPEED) / (d_i/SPEED + SERVICE):
                expected urgency on arrival per unit of actuator time (travel +
                service). A heuristic marginal-benefit-per-time index.
tour_sqrt       static schedule, TRUE rates: visit frequency of i ~ sqrt(r_i)
                (the square-root law minimising mean staleness). Deterministic
                stride scheduling: pass_i starts at stride_i/2 with stride_i =
                1/sqrt(r_i); serve argmin pass (ties: spatial index), then
                pass_i += stride_i. Equal rates reduce to spatial round-robin.
tour_sqrt_learned  dynamic version with r_hat: serves argmax elapsed_i*sqrt(r_hat_i)
                (the robot most overdue relative to its ideal interval).
rollout         at each decision point (actuator idle) evaluates every candidate
                target by simulating H steps ahead with EXPECTED (deterministic)
                dynamics on the dispatcher's belief state (learned r_hat and elapsed
                time) and a base policy afterwards; picks the best cumulative mean
                health. Expected dynamics make the lookahead identical across
                candidates (common random numbers trivially). Decisions are made only
                when idle, so `preempt` has no effect.
rollout_claim   as rollout, but the initial state is the current noisy report
                (1 - claim), with r_hat rates: tests whether reports help a planner.
rollout_true    as rollout with the TRUE current h and TRUE rates (oracle planner).
dp_full         EXACT-OPTIMAL (up to the health grid K = cfg.dp_k) dispatcher that sees the
                true h. Solves the semi-MDP of arbitration/dp.py for the instance
                (posts, rates, damage model) and acts greedily on its value function.
dp_age          exact-optimal dispatcher that sees only rates and the age of each robot
                (steps since its last service); optimal for the POMDP because the belief
                is a function of ages. Equals dp_full under linear damage.
Reporting-game policies (arbitration/game.py; accepted by run(), kept out of POLICIES)
-------------------------------------------------------------------------------------
fused           Bayesian fusion: the prior of tau_i comes from age (steps since last service),
                the learned (or, with fuse_known_rates, true) rate and the shock parameters;
                it is updated with the claim as a Gaussian measurement (sd sigma) on a
                health grid; serves argmax of the POSTERIOR MEAN of tau. If damage is linear
                the prior is a point mass, so fused == learned.
fused_index     same posterior mean, distance-aware index as learned_index.
audit_index     fused_index plus free ex-post audit: on arrival the true h is seen and
                compared with the claim; the robot's mean residual (claim - tau, shrunk
                toward 0 with n0 pseudo-visits) is deducted from its claims and the
                measurement sd is widened by sqrt(1 + 1/(n+n0)).
audit_penalty_index  audit_index plus a limited penalty (cfg.penalty_T steps): when the
                residual z-score sum(res)/(sigma sqrt(n)) exceeds cfg.audit_z the robot is
                flagged; while flagged its claim is ignored (penalty_mode="ignore") or its
                score is multiplied by penalty_factor ("demote").
`run(..., inflate=b)` adds b_i >= 0 to robot i's claim (strategic over-reporting);
`Result.robot_health` is each robot's mean health (the explorer's own utility);
`stream` != 0 re-draws shocks and noise for the same instance (posts, rates, start).
Base policy of rollouts: `rollout_base` in {"none" (spatial round-robin),
"learned" (argmax tau), "learned_index"}; default "learned_index".
"""
from dataclasses import dataclass, field, fields, replace
import math
import numpy as np
from scipy.special import gammainc
from scipy.stats import binom
RING = 100.0        # ring circumference
SPEED = 1.0         # actuator speed, units per step
SERVICE = 5         # steps per service; not preemptible
SIGMA = 0.15        # std. dev. of the private urgency estimate
THETA = 0.30        # request threshold
THETA_STRAT = 0.05  # request threshold under strategic inflation

LEGACY_POLICIES = ("none", "none_random", "greedy_true", "nearest",
                   "honest", "strategic", "priced")
DP_POLICIES = ("dp_full", "dp_age")
V2_POLICIES = ("index_true", "honest_index", "learned", "learned_index", "tour_sqrt", "tour_sqrt_learned",
               "rollout", "rollout_claim", "rollout_true")
GAME_POLICIES = ("fused", "fused_index", "audit_index", "audit_penalty_index")
POLICIES = LEGACY_POLICIES + V2_POLICIES      # DP_POLICIES are accepted by run() too, but
                                              # only for tiny n (state space K^n); kept out of sweeps
TOURS = ("tour_sqrt", "tour_sqrt_learned")
ROLLOUTS = ("rollout", "rollout_claim", "rollout_true")
R_PRIOR = 0.005     # r_hat before any observation exists


@dataclass(frozen=True)
class ModelConfig:
    """Model-variant options. The defaults ARE the legacy model (bit-identical).

    damage:   "linear" (h_i falls by r_i per step, deterministic) or "shock"
              (hidden Poisson shocks on top of a reduced base decay).
              Shock: with prob. `shock_rate` per step each robot takes a hit of
              Exponential size with mean shock_frac*r_i/shock_rate; the linear base
              decay is (1-shock_frac)*r_i. So mean degradation per step is r_i in
              both models (before the clip at 0), but elapsed time no longer
              predicts h. Shocks come from a separate RNG stream (same for every
              policy of a seed).
    noise:    "iid" (fresh N(0,sigma) every step; legacy), "ar1" (stationary AR(1)
              error with coefficient `ar1_phi`, marginal sd sigma) or "bias"
              (persistent per-robot offset carrying `bias_frac` of the variance,
              plus iid noise carrying the rest; marginal sd sigma).
    keep_commitment: if the caller set empties mid-travel, keep the current target
              until arrival instead of silently dropping it (legacy drops it).
    precision_at: "arrival" (legacy: tau at arrival) or "dispatch" (tau, and the
              max available tau, at the moment the target was last chosen).
    rollout_h / rollout_base: horizon and base policy of the rollout policies.
    service:  steps per service (default = module SERVICE); used by every policy and by dp.
    dp_k / dp_age_cap: grid resolution (health levels) of `dp_full` and age cap of `dp_age`.
    fuse_grid / fuse_known_rates: posterior grid size and whether fused policies use the true
              rates (default: learned r_hat) in their age prior.
    audit_n0 / audit_z / penalty_T / penalty_mode / penalty_factor: audit and penalty rules
              of `audit_index` / `audit_penalty_index` (see module docstring).
    """
    damage: str = "linear"
    shock_rate: float = 0.01
    shock_frac: float = 0.5
    noise: str = "iid"
    ar1_phi: float = 0.9
    bias_frac: float = 0.8
    keep_commitment: bool = False
    precision_at: str = "arrival"
    rollout_h: int = 150
    rollout_base: str = "learned_index"
    service: int = SERVICE
    dp_k: int = 40
    dp_age_cap: int = 400
    fuse_grid: int = 100
    fuse_known_rates: bool = False
    audit_n0: float = 1.0
    audit_z: float = 2.0
    penalty_T: int = 300
    penalty_mode: str = "ignore"
    penalty_factor: float = 0.5

    def __post_init__(self):
        for name, ok in (("damage", ("linear", "shock")),
                         ("noise", ("iid", "ar1", "bias")),
                         ("precision_at", ("arrival", "dispatch")),
                         ("rollout_base", ("none", "learned", "learned_index")),
                         ("penalty_mode", ("ignore", "demote"))):
            if getattr(self, name) not in ok:
                raise ValueError(f"{name}={getattr(self, name)!r}; expected one of {ok}")
        if not 0 <= self.shock_frac <= 1 or self.shock_rate <= 0:
            raise ValueError("need 0 <= shock_frac <= 1 and shock_rate > 0")
        if self.service < 1 or self.dp_k < 2 or self.dp_age_cap < 2:
            raise ValueError("need service >= 1, dp_k >= 2, dp_age_cap >= 2")
        if self.fuse_grid < 4 or self.penalty_T < 0 or self.audit_n0 < 0:
            raise ValueError("need fuse_grid >= 4, penalty_T >= 0, audit_n0 >= 0")


LEGACY = ModelConfig()
V2 = ModelConfig(damage="shock", noise="ar1", keep_commitment=True,
                 precision_at="dispatch")
MODELS = {"legacy": LEGACY, "v2": V2}


def make_config(model=None, **opts):
    """Resolve `model` (None | "legacy" | "v2" | ModelConfig) plus field overrides."""
    if model is None:
        base = LEGACY
    elif isinstance(model, str):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; expected one of {tuple(MODELS)}")
        base = MODELS[model]
    else:
        base = model
    names = {f.name for f in fields(ModelConfig)}
    bad = set(opts) - names
    if bad:
        raise TypeError(f"unknown option(s) {sorted(bad)}; valid: {sorted(names)}")
    return replace(base, **opts) if opts else base


def ring_distance(a, b):
    d = np.abs(a - b)
    return np.minimum(d, RING - d)


def _step_towards(pos, target):
    d = target - pos
    if d > RING / 2:
        d -= RING
    if d < -RING / 2:
        d += RING
    if abs(d) <= SPEED:
        return target, True
    return (pos + np.sign(d) * SPEED) % RING, False


@dataclass
class Result:
    mean_health: float        # team objective
    min_health: float         # worst-case objective
    frac_below_02: float
    frac_any_dead: float      # steps with >=1 robot at h=0; used as a load proxy
    precision: float          # true urgency served / max urgency available
    n_services: int
    retargets_per_service: float
    frac_dead_robot_steps: float = 0.0   # mean fraction of robots sitting at h=0
    robot_health: object = field(default=None, compare=False)   # per-robot mean health (array)


# ------------------------------------------------------------ v2 helpers
class _Learner:
    """Dispatcher-side estimate of per-robot decay rates from arrival observations.

    r_hat_i = (sum of observed drops) / (sum of elapsed times) over visits to i.
    Robots never visited get the mean estimate of visited ones (R_PRIOR if none).
    """
    def __init__(self, n):
        self.drop = np.zeros(n)
        self.el = np.zeros(n)
        self.t_reset = np.zeros(n)       # time of last service completion (h=1 at 0)

    def observe(self, i, h_obs, t):
        el = t - self.t_reset[i]
        if el > 0:
            self.drop[i] += 1.0 - h_obs
            self.el[i] += el

    def rhat(self):
        seen = self.el > 0
        prior = (self.drop[seen] / self.el[seen]).mean() if seen.any() else R_PRIOR
        return np.where(seen, self.drop / np.maximum(self.el, 1e-12), prior)

    def tau_hat(self, t):
        return np.minimum(1.0, self.rhat() * (t - self.t_reset))


# ------------------------------------------------------- reporting-game helpers
FUSE_BMAX = 40      # max number of shocks kept in the age prior


def age_prior(age, r, cfg, G):
    """pmf of tau = min(1, S_age) on the grid j/G (j = 0..G) given `age` damage steps at
    rate r. Shock damage: S = (1-f) r age + sum of Bin(age, p) Exp(mean f r / p) jumps
    (cell masses from the closed-form CDF; cell j covers ((j-.5)/G, (j+.5)/G], the top
    cell absorbs the tail). Linear damage: point mass at min(1, r age) (rounded to the grid)."""
    pmf = np.zeros(G + 1)
    age = max(int(age), 0)
    f = cfg.shock_frac
    if cfg.damage == "linear" or f == 0:
        pmf[min(G, int(round(min(1.0, r * age) * G)))] = 1.0
        return pmf
    p = cfg.shock_rate
    base, m = (1.0 - f) * r * age, f * r / p
    if age * p > 25 or base >= 1.0:          # essentially certain to be at the ceiling
        pmf[G] = 1.0
        return pmf
    ks = np.arange(FUSE_BMAX + 1)
    pk = binom.pmf(ks, age, p)
    pk = pk / pk.sum()
    x = np.maximum((np.arange(G) + 0.5) / G - base, 0.0)
    F = pk[0] * ((np.arange(G) + 0.5) / G >= base)
    F = F + (pk[1:, None] * gammainc(ks[1:, None], x[None, :] / m)).sum(axis=0)
    F = np.minimum.accumulate(np.minimum(F, 1.0)[::-1])[::-1]     # monotone, <= 1
    cum = np.concatenate([F, [1.0]])
    pmf[:] = np.diff(np.concatenate([[0.0], cum]))
    return np.maximum(pmf, 0.0)


def fused_tau(claim, age, r, sd, cfg):
    """Posterior mean of tau_i from the age prior and the claim ~ N(tau, sd^2).
    sd = inf (or None) ignores the claim. Under linear damage returns min(1, r age)."""
    if cfg.damage == "linear" or cfg.shock_frac == 0:
        return min(1.0, r * max(age, 0))
    G = cfg.fuse_grid
    pmf = age_prior(age, r, cfg, G)
    g = np.arange(G + 1) / G
    if sd is None or not np.isfinite(sd):
        return float((pmf * g).sum())
    z = (claim - g) / max(sd, 1e-9)
    lw = np.log(pmf + 1e-300) - 0.5 * z * z
    w = np.exp(lw - lw.max())
    return float((w * g).sum() / w.sum())


class _Auditor:
    """Ex-post audit of each robot's claims (free: the true h is seen on arrival).

    residual = claim - true tau at arrival. bias_i = sum(res) / (n + n0) (shrunk toward 0);
    sd_i = sigma sqrt(1 + 1/(n + n0)). Flag (start a penalty of `T` steps) when
    sum(res) / (sigma sqrt(n)) > z."""
    def __init__(self, n, sigma, n0=1.0, z=2.0, T=300):
        self.sigma, self.n0, self.z, self.T = max(float(sigma), 1e-9), n0, z, T
        self.s = np.zeros(n)
        self.k = np.zeros(n)
        self.until = np.full(n, -1)       # penalty active while t < until
        self.flags = 0

    def observe(self, i, claim, tau, t):
        self.s[i] += claim - tau
        self.k[i] += 1
        if self.s[i] / (self.sigma * math.sqrt(self.k[i])) > self.z:
            self.until[i] = t + self.T
            self.flags += 1

    def bias(self):
        return self.s / (self.k + self.n0)

    def sd(self):
        return self.sigma * np.sqrt(1.0 + 1.0 / (self.k + self.n0))

    def penalized(self, t):
        return self.until > t


def tour_sequence(rates, k):
    """First k visits of the sqrt-law stride schedule (frequency of i ~ sqrt(rates_i))."""
    stride = 1.0 / np.sqrt(np.asarray(rates, float))
    pas = stride / 2.0
    out = []
    for _ in range(k):
        i = int(np.argmin(pas))
        out.append(i)
        pas[i] += stride[i]
    return out


def _seg_sum(h, r, m):
    """sum_{k=1..m} clip(h - r k, 0) per robot, summed over robots (m steps)."""
    j = np.minimum(m, np.floor(h / np.maximum(r, 1e-12)))
    return float((j * h - r * j * (j + 1) / 2).sum())


def _index(tau_hat, rh, d, service=SERVICE):
    tt = d / SPEED + service
    return np.minimum(1.0, tau_hat + rh * d / SPEED) / tt


def _base_pick(base, h, r, posts, pos, last, service=SERVICE):
    if base == "none":
        return (last + 1) % len(h)
    tau = 1.0 - h
    if base == "learned":
        return int(np.argmax(tau))
    return int(np.argmax(_index(tau, r, ring_distance(posts, pos), service)))


def _rollout_value(first, h0, r, posts, pos, H, base, service=SERVICE):
    """Cumulative mean health over H steps if `first` is served next, then `base`.
    Event-based, expected (deterministic) linear dynamics."""
    n = len(h0)
    h, t, tot, target = h0.copy(), 0, 0.0, first
    while t < H:
        d = float(ring_distance(posts[target], pos))
        dur = max(1, math.ceil(d / SPEED)) + service
        dt = min(dur, H - t)
        tot += _seg_sum(h, r, dt)
        h = np.maximum(0.0, h - r * dt)
        if dur > H - t:
            break
        h[target] = 1.0
        pos, t = posts[target], t + dur
        target = _base_pick(base, h, r, posts, pos, target, service)
    return tot / (H * n)


def _rollout_choose(h0, r, posts, pos, H, base, service=SERVICE):
    vals = [_rollout_value(c, h0, r, posts, pos, H, base, service) for c in range(len(h0))]
    return int(np.argmax(vals))          # ties -> lowest index (deterministic)


def run(policy, n=8, rate=0.02, steps=1500, seed=0, spread=0.0,
        lam=0.006, preempt=False, hyst=0.0, sigma=SIGMA, model=None,
        inflate=None, stream=0, **opts):
    """Run one policy once. Returns a Result.

    preempt: re-evaluate the target on every travel step.
    hyst:    with preempt, only switch if the new best beats the current target
             by more than `hyst` in score (claim / true urgency units), or, for
             distance-based policies, is closer by more than hyst * RING.
             hyst=0 is plain preemption (the original behaviour).
    sigma:   std. dev. of the private urgency estimate.
    model:   None / "legacy" (default, original behaviour), "v2", or a ModelConfig.
    inflate: None or per-robot vector b_i added to the claims (strategic over-reporting).
    stream:  0 (default) = the original shock/noise streams; k != 0 re-draws shocks and noise
             for the same instance (posts, rates, start position).
    **opts:  ModelConfig field overrides, e.g. damage="shock", noise="ar1".
    """
    if policy not in POLICIES + DP_POLICIES + GAME_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of "
                         f"{POLICIES + DP_POLICIES + GAME_POLICIES}")
    if policy in DP_POLICIES and n > 4:
        raise ValueError(f"{policy} solves the exact semi-MDP; supported for n <= 4 (got {n})")
    cfg = make_config(model, **opts)

    rng = np.random.default_rng(seed)
    posts = np.sort(rng.uniform(0, RING, n))
    rates = rate * np.exp(rng.normal(0, spread, n))
    svc = cfg.service
    infl = None if inflate is None else np.asarray(inflate, float)
    if infl is not None and infl.shape != (n,):
        raise ValueError(f"inflate must have shape ({n},)")
    sk = [] if stream == 0 else [int(stream)]
    irng = rng if stream == 0 else np.random.default_rng([seed, 555] + sk)

    shock = cfg.damage == "shock"
    if shock:
        srng = np.random.default_rng([seed, 7919] + sk)
        base_decay = rates * (1.0 - cfg.shock_frac)
        shock_size = cfg.shock_frac * rates / cfg.shock_rate
    nrng = np.random.default_rng([seed, 4243] + sk) if cfg.noise != "iid" else None
    if cfg.noise == "ar1":
        e = nrng.normal(0, sigma, n)
        phi, ar_sd = cfg.ar1_phi, sigma * math.sqrt(1 - cfg.ar1_phi ** 2)
    elif cfg.noise == "bias":
        bias = nrng.normal(0, sigma * math.sqrt(cfg.bias_frac), n)
        iid_sd = sigma * math.sqrt(1 - cfg.bias_frac)

    uses_learner = policy in ("learned", "learned_index", "tour_sqrt_learned",
                              "rollout", "rollout_claim") or policy in GAME_POLICIES
    learner = _Learner(n)
    auditor = (_Auditor(n, sigma, cfg.audit_n0, cfg.audit_z,
                        cfg.penalty_T if policy == "audit_penalty_index" else 0)
               if policy in ("audit_index", "audit_penalty_index") else None)
    hacc = np.zeros(n)
    if policy == "tour_sqrt":
        stride = 1.0 / np.sqrt(rates)
        tour_pass = stride / 2.0

    h = np.ones(n)
    pos = rng.uniform(0, RING)
    if policy in DP_POLICIES:
        from . import dp                  # lazy: dp imports this module
        dpol = dp.build_policy(policy, posts, rates, cfg)
        last_reset = np.full(n, -1)       # age_i = t - last_reset_i = damages since service
    # none_random's visiting order has its own stream, so posts, rates, start and
    # noise are identical across policies for a given seed (paired arms).
    order = (np.random.default_rng([seed, 1]).permutation(n)
             if policy == "none_random" else np.arange(n))
    target, serving, rr = -1, 0, 0
    acc = np.zeros(5)
    num = den = 0.0
    d_num = d_den = 0.0
    n_serv = retargets = 0

    for t in range(steps):
        if shock:
            hit = (srng.random(n) < cfg.shock_rate) * srng.exponential(1.0, n) * shock_size
            h = np.clip(h - base_decay - hit, 0.0, 1.0)
        else:
            h = np.clip(h - rates, 0.0, 1.0)
        tau = 1.0 - h
        hacc += h
        acc += (h.mean(), h.min(), (h < 0.2).mean(), float((h <= 0).any()),
                (h <= 0).mean())
        if cfg.noise == "iid":
            claim = tau + irng.normal(0, sigma, n)
        elif cfg.noise == "ar1":
            e = phi * e + nrng.normal(0, ar_sd, n)
            claim = tau + e
        else:
            claim = tau + bias + nrng.normal(0, iid_sd, n)
        if infl is not None:
            claim = claim + infl

        if serving > 0:                       # a service in progress cannot be cut
            serving -= 1
            if serving == 0:
                h[target] = 1.0
                learner.t_reset[target] = t
                if policy in DP_POLICIES:
                    last_reset[target] = t
                target = -1
            continue

        prev_target = target
        if policy in GAME_POLICIES and target >= 0 and not preempt:
            pass                              # committed; the posterior is only needed to choose
        elif policy in ("none", "none_random"):
            if target < 0:
                target = int(order[rr % n])
                rr += 1
        elif policy == "tour_sqrt":
            if target < 0:
                target = int(np.argmin(tour_pass))
                tour_pass[target] += stride[target]
        elif policy == "tour_sqrt_learned":
            if target < 0:
                target = int(np.argmax((t - learner.t_reset) * np.sqrt(learner.rhat())))
        elif policy in DP_POLICIES:
            if target < 0:
                target = dpol.choose(pos, h, t - last_reset)
        elif policy in ROLLOUTS:
            if target < 0:
                if policy == "rollout_true":
                    h0, r = h.copy(), rates
                elif policy == "rollout_claim":
                    h0, r = np.clip(1.0 - claim, 0.0, 1.0), learner.rhat()
                else:
                    h0, r = np.clip(1.0 - learner.tau_hat(t), 0.0, 1.0), learner.rhat()
                target = _rollout_choose(h0, r, posts, pos, cfg.rollout_h,
                                         cfg.rollout_base, svc)
        else:
            d = ring_distance(posts, pos)
            if policy == "greedy_true":
                callers, score, by = np.where(tau > 0.05)[0], tau, "max"
            elif policy == "nearest":
                callers, score, by = np.where(claim > THETA)[0], None, "near"
            elif policy == "honest":
                callers, score, by = np.where(claim > THETA)[0], claim, "max"
            elif policy == "priced":
                callers, score, by = np.where(claim > THETA + lam * d)[0], claim, "max"
            elif policy == "index_true":
                callers, by = np.where(tau > 0.05)[0], "max"
                score = tau / (d / SPEED + svc)
            elif policy == "honest_index":
                callers, by = np.where(claim > THETA)[0], "max"
                score = claim / (d / SPEED + svc)
            elif policy == "learned":
                callers, score, by = np.arange(n), learner.tau_hat(t), "max"
            elif policy == "learned_index":
                callers, by = np.arange(n), "max"
                score = _index(learner.tau_hat(t), learner.rhat(), d, svc)
            elif policy in GAME_POLICIES:
                callers, by = np.arange(n), "max"
                age = t - learner.t_reset
                rr_ = rates if cfg.fuse_known_rates else learner.rhat()
                cl, sd = claim, np.full(n, sigma)
                if auditor is not None:
                    cl, sd = claim - auditor.bias(), auditor.sd()
                    if policy == "audit_penalty_index" and cfg.penalty_mode == "ignore":
                        sd = np.where(auditor.penalized(t), np.inf, sd)
                tp = np.array([fused_tau(cl[i], age[i], rr_[i], sd[i], cfg) for i in range(n)])
                score = tp if policy == "fused" else _index(tp, rr_, d, svc)
                if policy == "audit_penalty_index" and cfg.penalty_mode == "demote":
                    score = np.where(auditor.penalized(t), score * cfg.penalty_factor, score)
            else:  # strategic
                callers, score, by = np.where(claim > THETA_STRAT)[0], None, "near"

            if len(callers) == 0:
                if not (cfg.keep_commitment and target >= 0):
                    target = -1
                    continue
            else:
                best = int(callers[np.argmax(score[callers])] if by == "max"
                           else callers[np.argmin(d[callers])])
                if target < 0:
                    target = best
                elif preempt and best != target:
                    if hyst > 0.0:
                        if by == "max":
                            switch = score[best] > score[target] + hyst
                        else:
                            switch = d[best] < d[target] - hyst * RING
                    else:
                        switch = True
                    if switch:
                        target, retargets = best, retargets + 1

        if target != prev_target and target >= 0:
            d_num, d_den = tau[target], max(tau.max(), 1e-9)
        pos, arrived = _step_towards(pos, posts[target])
        if arrived:
            serving = svc
            n_serv += 1
            if uses_learner:
                learner.observe(target, h[target], t)
            if auditor is not None:
                auditor.observe(target, claim[target], tau[target], t)
            if cfg.precision_at == "dispatch":
                num += d_num
                den += d_den
            else:
                num += tau[target]
                den += max(tau.max(), 1e-9)

    m, mn, b, dead, dead_r = acc / steps
    return Result(m, mn, b, dead,
                  num / den if den > 0 else float("nan"),
                  n_serv,
                  retargets / n_serv if n_serv else float("nan"),
                  dead_r, hacc / steps)
