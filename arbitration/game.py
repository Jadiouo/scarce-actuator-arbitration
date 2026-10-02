"""Strategic reporting as a game, and ex-post audit mechanisms.

Explorer i's utility is its OWN long-run mean health. It reports
    claim_i = tau_i + noise + b_i ,   b_i >= 0 on a grid (over-reporting / inflation).
A dispatch rule maps claims (and, for audits, what is seen on arrival) to service
order. For each rule we look for a pure-strategy Nash equilibrium of the b-game by
best-response iteration in simulation, with common random numbers (a given seed and
noise stream fixes posts, rates, shocks and noise; only b changes), then compare the
team's mean health at equilibrium with truthful reporting (b = 0).

Games are solved per instance (per seed): every seed is an N-player game on its own
posts / rates / shock path, and the seeds are the replications. Because a best response
computed on one noise path can exploit that path, the equilibrium profile is re-evaluated
on FRESH noise streams (same instance, `stream` != 0) for the headline numbers.

Best-response iteration (asymmetric, per seed, Gauss-Seidel)
-----------------------------------------------------------
start b = 0. One pass: for i = 1..N: evaluate every grid value g for robot i with the others
fixed; move to the best g only if it beats the current value by more than `eps` (own health);
ties go to the smaller b. A seed has CONVERGED when a whole pass makes no move, i.e. no robot
can gain more than eps by a unilateral deviation on the grid (an eps-Nash equilibrium of the
gridded game; eps is the stated tolerance, not zero, because the payoff is a rugged function
of b). A profile revisited earlier flags a CYCLE; hitting max_rounds flags non-convergence.

Evaluator interface (so a batched / GPU simulator can replace the CPU one)
--------------------------------------------------------------------------
    evaluate(policy, jobs, env, stream=0) -> ndarray (len(jobs), N)
        policy: run() policy name; jobs: list of (seed, b) with b a length-N sequence;
        env: dict(n, rate, spread, steps, sigma, model, opts) (opts = ModelConfig overrides);
        returns each explorer's mean health for every job. `CpuEvaluator` is the reference
        implementation (calls model.run; `map_fn` may be a process-pool map).
"""
import numpy as np
from .model import run

B_GRID = tuple(round(0.1 * k, 1) for k in range(7))      # 0 .. 0.6
EPS = 0.002                                              # min own-health gain to move
MAX_ROUNDS = 12


def _cpu_job(args):
    policy, seed, b, env, stream = args
    e = dict(env)
    r = run(policy, n=e["n"], rate=e["rate"], spread=e["spread"], steps=e["steps"],
            seed=seed, sigma=e.get("sigma", 0.15), model=e.get("model", "v2"),
            preempt=False, inflate=np.asarray(b, float), stream=stream,
            **dict(e.get("opts", ())))
    return r.robot_health


def _freeze(env):
    e = dict(env)
    e["opts"] = tuple(sorted(dict(e.get("opts", {})).items()))
    return tuple(sorted(e.items()))


class CpuEvaluator:
    """Reference evaluator: one model.run per job. map_fn: builtin map or pool.map."""
    def __init__(self, map_fn=map, chunksize=4):
        self.map_fn, self.chunksize = map_fn, chunksize
        self.calls = 0

    def __call__(self, policy, jobs, env, stream=0):
        self.calls += len(jobs)
        fe = _freeze(env)
        args = [(policy, int(s), tuple(float(x) for x in b), fe, stream) for s, b in jobs]
        if not args:
            return np.zeros((0, env["n"]))
        if self.map_fn is map:
            out = list(map(_cpu_job, args))
        else:
            out = list(self.map_fn(_cpu_job, args, chunksize=self.chunksize))
        return np.array(out)


# ---------------------------------------------------------- best-response (per seed)
def best_response_iteration(evaluate, policy, env, seeds, grid=B_GRID, eps=EPS,
                            max_rounds=MAX_ROUNDS, stream=0):
    """Asymmetric pure-strategy equilibrium of the b-game, every seed in lockstep.

    Returns dict with per-seed arrays: b (S,N), converged, cycle, rounds, max_gain
    (largest unilateral gain left on the grid at the final profile), traj (list over rounds
    of (S,N) profiles, entry 0 = start), n_eval."""
    seeds = list(seeds)
    S, N = len(seeds), env["n"]
    b = np.zeros((S, N))
    util = {}                               # (seed, profile tuple) -> own-health vector
    seen = [{tuple(b[k])} for k in range(S)]
    active = np.ones(S, bool)
    converged = np.zeros(S, bool)
    cycle = np.zeros(S, bool)
    rounds = np.zeros(S, int)
    gain = np.zeros(S)
    traj = [b.copy()]
    n_eval = 0

    def need(jobs):
        nonlocal n_eval
        miss = [j for j in dict.fromkeys(jobs) if j not in util]
        if miss:
            res = evaluate(policy, [(s, p) for s, p in miss], env, stream)
            n_eval += len(miss)
            for j, r in zip(miss, res):
                util[j] = r

    for _ in range(max_rounds):
        if not active.any():
            break
        moved = np.zeros(S, bool)
        pass_gain = np.zeros(S)
        for i in range(N):
            jobs = []
            for k in np.nonzero(active)[0]:
                for g in grid:
                    p = b[k].copy()
                    p[i] = g
                    jobs.append((seeds[k], tuple(p)))
                jobs.append((seeds[k], tuple(b[k])))
            need(jobs)
            for k in np.nonzero(active)[0]:
                cur = util[(seeds[k], tuple(b[k]))][i]
                vals = []
                for g in grid:
                    p = b[k].copy()
                    p[i] = g
                    vals.append(util[(seeds[k], tuple(p))][i])
                vals = np.array(vals)
                j = int(np.argmax(vals))           # first max -> smallest b among ties
                pass_gain[k] = max(pass_gain[k], vals[j] - cur)
                if vals[j] > cur + eps:
                    b[k, i] = grid[j]
                    moved[k] = True
        for k in np.nonzero(active)[0]:
            rounds[k] += 1
            gain[k] = pass_gain[k]
            if not moved[k]:
                converged[k], active[k] = True, False
            else:
                key = tuple(b[k])
                if key in seen[k]:
                    cycle[k], active[k] = True, False
                seen[k].add(key)
        traj.append(b.copy())
    return dict(b=b, converged=converged, cycle=cycle, rounds=rounds, max_gain=gain,
                traj=traj, n_eval=n_eval)


# ----------------------------------------------------------- symmetric equilibrium
def symmetric_equilibrium(evaluate, policy, env, seeds, grid=B_GRID, eps=EPS,
                          max_rounds=MAX_ROUNDS, stream=0):
    """Symmetric pure strategy b (common to all explorers, pooled over seeds and robots).

    Iterates b <- argmax_g mean_{seed, robot i} u_i(g, b_{-i} = b) from b = 0; stops when the
    best deviation gains <= eps (converged), on a revisited b (cycle) or at max_rounds.
    Returns dict(b, converged, cycle, rounds, gain, path)."""
    seeds, N = list(seeds), env["n"]
    path, b, util = [0.0], 0.0, {}

    def U(g, bo):
        key = (g, bo)
        if key not in util:
            jobs = []
            for s in seeds:
                for i in range(N):
                    p = [bo] * N
                    p[i] = g
                    jobs.append((s, tuple(p)))
            res = evaluate(policy, jobs, env, stream).reshape(len(seeds), N, N)
            util[key] = float(np.mean([res[:, i, i] for i in range(N)]))
        return util[key]

    converged = cycle = False
    gain, rounds = 0.0, 0
    for _ in range(max_rounds):
        rounds += 1
        vals = np.array([U(g, b) for g in grid])
        j = int(np.argmax(vals))
        gain = float(vals[j] - U(b, b))
        if gain <= eps:
            converged = True
            break
        b = grid[j]
        if b in path:
            cycle = True
            break
        path.append(b)
    return dict(b=b, converged=converged, cycle=cycle, rounds=rounds, gain=gain, path=path)


# ------------------------------------------------------------- game summary
def team_health(evaluate, policy, env, seeds, profiles, streams):
    """Team mean health of per-seed profiles (S,N) over the given noise streams ->
    array (len(streams), S)."""
    seeds = list(seeds)
    out = []
    for st in streams:
        res = evaluate(policy, [(s, tuple(p)) for s, p in zip(seeds, profiles)], env, st)
        out.append(res.mean(axis=1))
    return np.array(out)


def solve_game(evaluate, policy, env, seeds, streams=(1, 2), grid=B_GRID, eps=EPS,
               max_rounds=MAX_ROUNDS, symmetric=True):
    """Equilibrium of the b-game for one dispatch rule, with in-sample and out-of-sample
    team health at b = 0 (truthful) and at the equilibrium profile."""
    seeds = list(seeds)
    br = best_response_iteration(evaluate, policy, env, seeds, grid, eps, max_rounds)
    zero = np.zeros_like(br["b"])
    out = dict(br=br,
               truthful_in=team_health(evaluate, policy, env, seeds, zero, (0,))[0],
               eq_in=team_health(evaluate, policy, env, seeds, br["b"], (0,))[0],
               truthful_out=team_health(evaluate, policy, env, seeds, zero, streams).mean(0),
               eq_out=team_health(evaluate, policy, env, seeds, br["b"], streams).mean(0))
    if symmetric:
        sy = symmetric_equilibrium(evaluate, policy, env, seeds, grid, eps, max_rounds)
        prof = np.full_like(br["b"], sy["b"])
        sy["team_in"] = team_health(evaluate, policy, env, seeds, prof, (0,))[0]
        sy["team_out"] = team_health(evaluate, policy, env, seeds, prof, streams).mean(0)
        out["sym"] = sy
    return out
