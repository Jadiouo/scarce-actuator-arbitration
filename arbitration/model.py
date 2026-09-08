"""
Arbitration of a scarce, non-substitutable actuator in a robot team.

N explorers sit at fixed posts on a ring. Explorer i's perception health h_i
decays at its own private rate r_i and is restored to 1 when the single
actuator `c` reaches it and completes a service. `c` moves at finite speed, so
every dispatch decision costs travel time. Each explorer observes only a noisy
estimate of its own urgency tau_i = 1 - h_i, and requesting help is free.

The model exists to compare dispatch policies that differ in how much urgency
information they use -- including one handed the ground truth. All policies
draw from the same random stream for a given seed, so arms are comparable.

Policies
--------
none         blind round-robin over posts in spatial order; ignores all requests
none_random  the same, in a random (non-spatial) order -- the weaker baseline
oracle       serves argmax of the TRUE urgency
nearest      honest request threshold, but dispatches purely by distance
honest       explorers report their private estimate; serves argmax of claims
strategic    explorers always call and claim the ceiling, so claims are tied
             and the dispatcher falls back on distance
priced       honest reporting, but an explorer may call only if its urgency
             exceeds THETA + lam * (distance to the actuator)
"""
from dataclasses import dataclass
import numpy as np

RING = 100.0        # ring circumference
SPEED = 1.0         # actuator speed, units per step
SERVICE = 5         # steps per service; not preemptible
SIGMA = 0.15        # std. dev. of the private urgency estimate
THETA = 0.30        # request threshold
THETA_STRAT = 0.05  # request threshold under strategic inflation

POLICIES = ("none", "none_random", "oracle", "nearest",
            "honest", "strategic", "priced")


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


def run(policy, n=8, rate=0.02, steps=1500, seed=0, spread=0.0,
        lam=0.006, preempt=False):
    """Run one policy once. Returns a Result."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    rng = np.random.default_rng(seed)
    posts = np.sort(rng.uniform(0, RING, n))
    order = rng.permutation(n) if policy == "none_random" else np.arange(n)
    rates = rate * np.exp(rng.normal(0, spread, n))

    h = np.ones(n)
    pos = rng.uniform(0, RING)
    target, serving, rr = -1, 0, 0
    acc = np.zeros(4)
    num = den = 0.0
    n_serv = retargets = 0

    for _ in range(steps):
        h = np.clip(h - rates, 0.0, 1.0)
        tau = 1.0 - h
        acc += (h.mean(), h.min(), (h < 0.2).mean(), float((h <= 0).any()))
        claim = tau + rng.normal(0, SIGMA, n)

        if serving > 0:                       # a service in progress cannot be cut
            serving -= 1
            if serving == 0:
                h[target] = 1.0
                target = -1
            continue

        if policy in ("none", "none_random"):
            if target < 0:
                target = int(order[rr % n])
                rr += 1
        else:
            d = ring_distance(posts, pos)
            if policy == "oracle":
                callers, score, by = np.where(tau > 0.05)[0], tau, "max"
            elif policy == "nearest":
                callers, score, by = np.where(claim > THETA)[0], None, "near"
            elif policy == "honest":
                callers, score, by = np.where(claim > THETA)[0], claim, "max"
            elif policy == "priced":
                callers, score, by = np.where(claim > THETA + lam * d)[0], claim, "max"
            else:  # strategic
                callers, score, by = np.where(claim > THETA_STRAT)[0], None, "near"

            if len(callers) == 0:
                target = -1
                continue

            best = int(callers[np.argmax(score[callers])] if by == "max"
                       else callers[np.argmin(d[callers])])
            if target < 0:
                target = best
            elif preempt and best != target:
                target, retargets = best, retargets + 1

        pos, arrived = _step_towards(pos, posts[target])
        if arrived:
            serving = SERVICE
            n_serv += 1
            num += tau[target]
            den += max(tau.max(), 1e-9)

    m, mn, b, dead = acc / steps
    return Result(m, mn, b, dead,
                  num / den if den > 0 else float("nan"),
                  n_serv,
                  retargets / n_serv if n_serv else float("nan"))
