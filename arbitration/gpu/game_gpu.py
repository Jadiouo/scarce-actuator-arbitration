"""GPU evaluator and batched best-response solver for the reporting game (arbitration/game.py).

GpuEvaluator implements game.py's `evaluate(policy, jobs, env, stream)` interface with
sim_torch (jobs = (seed, b) pairs become one batch), so game.solve_game works unchanged.

solve_games runs many games (mechanism x start profile) x seeds in LOCKSTEP: every
Gauss-Seidel step ("robot i's best response") issues ONE simulate() call that contains the
candidate profiles of all games, whichever mechanism they belong to. The rule per game is exactly
game.best_response_iteration (grid argmax, move only if gain > eps, ties -> smaller b, cycle /
convergence flags); only the start profile is a parameter (b = 0 in game.py).
"""
import numpy as np
from ..model import make_config
from .sim_torch import simulate

CHUNK = 6000


def _env_cfg(env):
    return make_config(env.get("model", "v2"), **dict(env.get("opts", ())))


class GpuEvaluator:
    def __init__(self, device=None, chunk=CHUNK):
        self.device, self.chunk, self.calls = device, chunk, 0

    def __call__(self, policy, jobs, env, stream=0):
        self.calls += len(jobs)
        cfg = _env_cfg(env)
        n = env["n"]
        out = []
        for a in range(0, len(jobs), self.chunk):
            jb = jobs[a:a + self.chunk]
            r = simulate(policy, [int(s) for s, _ in jb], n=n, rate=env["rate"], spread=env["spread"],
                         steps=env["steps"], sigma=env.get("sigma", 0.15), model=cfg, preempt=False,
                         inflate=np.array([b for _, b in jb], float), stream=stream, device=self.device)
            out.append(r["robot_health"])
        return np.concatenate(out) if out else np.zeros((0, n))


def evaluate_mixed(mechs, requests, env, stream=0, device=None, chunk=CHUNK):
    """requests: list of (mech index, seed, profile) -> (R, N) robot health. mechs: [(policy, opts)]."""
    cfgs = [make_config(env.get("model", "v2"), **{**dict(env.get("opts", ())), **o}) for _, o in mechs]
    out = []
    for a in range(0, len(requests), chunk):
        rq = requests[a:a + chunk]
        r = simulate([mechs[m][0] for m, _, _ in rq], [int(s) for _, s, _ in rq], n=env["n"], rate=env["rate"],
                     spread=env["spread"], steps=env["steps"], sigma=env.get("sigma", 0.15),
                     model=[cfgs[m] for m, _, _ in rq], preempt=False,
                     inflate=np.array([p for _, _, p in rq], float), stream=stream, device=device)
        out.append(r["robot_health"])
    return np.concatenate(out) if out else np.zeros((0, env["n"]))


class _Game:
    """State of one best-response game (one mechanism, one start profile) over all seeds."""
    def __init__(self, mech, seeds, N, b0):
        self.mech, self.seeds, self.N = mech, list(seeds), N
        S = len(self.seeds)
        self.b = np.full((S, N), float(b0))
        self.seen = [{tuple(self.b[k])} for k in range(S)]
        self.active = np.ones(S, bool)
        self.converged = np.zeros(S, bool)
        self.cycle = np.zeros(S, bool)
        self.rounds = np.zeros(S, int)
        self.gain = np.zeros(S)
        self.traj = [self.b.copy()]
        self.moved = self.pass_gain = None

    def start_round(self):
        S = len(self.seeds)
        self.moved, self.pass_gain = np.zeros(S, bool), np.zeros(S)

    def jobs(self, i, grid):
        js = []
        for k in np.nonzero(self.active)[0]:
            for g in grid:
                p = self.b[k].copy(); p[i] = g
                js.append((self.mech, self.seeds[k], tuple(p)))
            js.append((self.mech, self.seeds[k], tuple(self.b[k])))
        return js

    def update(self, i, grid, eps, util):
        for k in np.nonzero(self.active)[0]:
            s = self.seeds[k]
            cur = util[(self.mech, s, tuple(self.b[k]))][i]
            vals = []
            for g in grid:
                p = self.b[k].copy(); p[i] = g
                vals.append(util[(self.mech, s, tuple(p))][i])
            vals = np.array(vals)
            j = int(np.argmax(vals))
            self.pass_gain[k] = max(self.pass_gain[k], vals[j] - cur)
            if vals[j] > cur + eps:
                self.b[k, i] = grid[j]
                self.moved[k] = True

    def end_round(self):
        for k in np.nonzero(self.active)[0]:
            self.rounds[k] += 1
            self.gain[k] = self.pass_gain[k]
            if not self.moved[k]:
                self.converged[k], self.active[k] = True, False
            else:
                key = tuple(self.b[k])
                if key in self.seen[k]:
                    self.cycle[k], self.active[k] = True, False
                self.seen[k].add(key)
        self.traj.append(self.b.copy())


def solve_games(mechs, env, seeds, grid, eps, max_rounds, starts=(0.0,), stream=0, device=None,
                log=print):
    """Returns {(mech index, start): dict(b, converged, cycle, rounds, max_gain, traj)}."""
    N = env["n"]
    games = [_Game(m, seeds, N, st) for m in range(len(mechs)) for st in starts]
    keys = [(g.mech, st) for g in games for st in [None]]
    util = {}
    n_eval = 0
    for rnd in range(max_rounds):
        live = [g for g in games if g.active.any()]
        if not live:
            break
        for g in live:
            g.start_round()
        for i in range(N):
            jobs = [j for g in live for j in g.jobs(i, grid)]
            miss = [j for j in dict.fromkeys(jobs) if j not in util]
            if miss:
                res = evaluate_mixed(mechs, miss, env, stream, device)
                n_eval += len(miss)
                for j, r in zip(miss, res):
                    util[j] = r
            for g in live:
                g.update(i, grid, eps, util)
            log(f"    round {rnd + 1} robot {i + 1}/{N}: {len(miss)} sims, "
                f"{sum(int(g.active.sum()) for g in live)} active (mech,start,seed)")
        for g in live:
            g.end_round()
    out = {}
    it = iter(games)
    for m in range(len(mechs)):
        for st in starts:
            g = next(it)
            out[(m, st)] = dict(b=g.b, converged=g.converged, cycle=g.cycle, rounds=g.rounds,
                                max_gain=g.gain, traj=g.traj)
    out["n_eval"] = n_eval
    return out


def team_health_mixed(mechs, env, seeds, profiles_by_mech, streams, device=None):
    """profiles_by_mech: list over mech of (S, N) profiles. -> array (len(streams), M, S) team health."""
    seeds = list(seeds)
    reqs = [(m, s, tuple(p)) for m, prof in enumerate(profiles_by_mech) for s, p in zip(seeds, prof)]
    res = []
    for st in streams:
        r = evaluate_mixed(mechs, reqs, env, st, device).mean(axis=1)
        res.append(r.reshape(len(mechs), len(seeds)))
    return np.array(res)


# =====================================================================================
# Strategy-family games (explorer strategy = (inflation rule, b)); players may be coalitions
# =====================================================================================
# family index -> (name, sim_torch imode, ipar). Action = (family, b); b is the inflation size
# (for the adaptive family it is the CAP of the inflation).
FAMS = {0: ("const", 0, 0.0), 1: ("timing", 1, 0.0), 2: ("dist10", 2, 10.0), 3: ("dist30", 2, 30.0),
        4: ("taulow", 3, 0.3), 5: ("habit", 4, 0.0), 6: ("adapt", 5, 0.0)}
TRUTH = (0, 0.0)
_MODEL_KEYS = None


def _split_opts(opts):
    global _MODEL_KEYS
    if _MODEL_KEYS is None:
        from dataclasses import fields
        from ..model import ModelConfig
        _MODEL_KEYS = {f.name for f in fields(ModelConfig)}
    mo = {k: v for k, v in opts.items() if k in _MODEL_KEYS}
    ex = {k: v for k, v in opts.items() if k not in _MODEL_KEYS}
    return mo, ex


def make_actions(const=None, others=(), fams=(1, 2, 3, 4, 5), adapt=()):
    """Ordered action list: truthful, constant b's, then (family, b) for the other families."""
    acts = [TRUTH] + [(0, round(b, 4)) for b in (const or [])]
    for f in fams:
        acts += [(f, round(b, 4)) for b in others]
    acts += [(6, round(b, 4)) for b in adapt]
    return acts


def evaluate_full(mechs, requests, env, stream=0, device=None, chunk=10000):
    """requests: [(mech idx, seed, profile)], profile = tuple over robots of (family, b).
    Returns dict(rh (R,N), p05 (R,), dead (R,)). mechs: [(policy, opts)] (opts may hold `disp_offset`)."""
    n = env["n"]
    split = [_split_opts({**dict(env.get("opts", ())), **o}) for _, o in mechs]
    cfgs = [make_config(env.get("model", "v2"), **mo) for mo, _ in split]
    off = [ex.get("disp_offset", 0.0) for _, ex in split]
    outs = {"rh": [], "p05": [], "dead": [], "audit_sum": [], "audit_cnt": [], "audit_flags": []}
    for a in range(0, len(requests), chunk):
        rq = requests[a:a + chunk]
        R = len(rq)
        infl = np.zeros((R, n)); imode = np.zeros((R, n), dtype=np.int64); ipar = np.zeros((R, n))
        for r, (_, _, prof) in enumerate(rq):
            for i, (fam, b) in enumerate(prof):
                _, md, par = FAMS[fam]
                imode[r, i] = md
                if fam == 6:
                    ipar[r, i] = b
                else:
                    infl[r, i], ipar[r, i] = b, par
        res = simulate([mechs[m][0] for m, _, _ in rq], [int(s) for _, s, _ in rq], n=n, rate=env["rate"],
                       spread=env["spread"], steps=env["steps"], sigma=env.get("sigma", 0.15),
                       model=[cfgs[m] for m, _, _ in rq], preempt=False, inflate=infl, imode=imode, ipar=ipar,
                       stream=stream, device=device, burn=env.get("burn", 0),
                       disp_offset=np.array([off[m] for m, _, _ in rq]))
        outs["rh"].append(res["robot_health"]); outs["p05"].append(res["min_health_p05"])
        outs["dead"].append(res["frac_dead_robot_steps"])
        for k in ("audit_sum", "audit_cnt", "audit_flags"):
            if res[k] is not None:
                outs[k].append(res[k])
    out = {k: (np.concatenate(v) if v else None) for k, v in outs.items()}
    return out


class _Game2:
    """One best-response game: mechanism x start x eps, all seeds in lockstep.
    players: list of tuples of robot indices (singleton = ordinary explorer, pair = coalition that plays
    one common action and maximises its members' mean health)."""
    def __init__(self, key, mech, actions, players, seeds, eps, start):
        self.key, self.mech, self.actions, self.players = key, mech, actions, players
        self.seeds, self.eps, self.N = list(seeds), eps, sum(len(p) for p in players)
        S = len(self.seeds)
        if start not in actions:        # nearest constant-b action
            start = min((a for a in actions if a[0] == start[0]), key=lambda a: abs(a[1] - start[1]))
        s0 = actions.index(start)
        self.pa = np.full((S, len(players)), s0, dtype=int)          # action index per player
        self.seen = [{tuple(self.pa[k])} for k in range(S)]
        self.active = np.ones(S, bool)
        self.converged, self.cycle = np.zeros(S, bool), np.zeros(S, bool)
        self.rounds, self.gain = np.zeros(S, int), np.zeros(S)
        self.traj = [self.pa.copy()]

    def profile(self, k, pa_row):
        prof = [TRUTH] * self.N
        for pl, ai in zip(self.players, pa_row):
            for r in pl:
                prof[r] = self.actions[ai]
        return tuple(prof)

    def start_round(self):
        S = len(self.seeds)
        self.moved, self.pass_gain = np.zeros(S, bool), np.zeros(S)

    def jobs(self, p):
        js = []
        for k in np.nonzero(self.active)[0]:
            row = self.pa[k].copy()
            for ai in range(len(self.actions)):
                row[p] = ai
                js.append((self.mech, self.seeds[k], self.profile(k, row)))
        return js

    def update(self, p, util):
        for k in np.nonzero(self.active)[0]:
            row = self.pa[k].copy()
            vals = []
            for ai in range(len(self.actions)):
                row[p] = ai
                vals.append(util[(self.mech, self.seeds[k], self.profile(k, row))][list(self.players[p])].mean())
            vals = np.array(vals)
            cur = vals[self.pa[k, p]]
            j = int(np.argmax(vals))
            self.pass_gain[k] = max(self.pass_gain[k], vals[j] - cur)
            if vals[j] > cur + self.eps:
                self.pa[k, p] = j
                self.moved[k] = True

    def end_round(self):
        for k in np.nonzero(self.active)[0]:
            self.rounds[k] += 1
            self.gain[k] = self.pass_gain[k]
            if not self.moved[k]:
                self.converged[k], self.active[k] = True, False
            else:
                key = tuple(self.pa[k])
                if key in self.seen[k]:
                    self.cycle[k], self.active[k] = True, False
                self.seen[k].add(key)
        self.traj.append(self.pa.copy())


def solve_games2(games, mechs, env, max_rounds=12, log=print, util=None, device=None):
    """Lockstep best-response for a list of _Game2 (sharing one evaluation cache `util`)."""
    util = {} if util is None else util
    n_eval = 0
    maxp = max(len(g.players) for g in games)
    for rnd in range(max_rounds):
        live = [g for g in games if g.active.any()]
        if not live:
            break
        for g in live:
            g.start_round()
        for p in range(maxp):
            lg = [g for g in live if p < len(g.players)]
            jobs = [j for g in lg for j in g.jobs(p)]
            miss = [j for j in dict.fromkeys(jobs) if j not in util]
            if miss:
                res = evaluate_full(mechs, miss, env, 0, device)["rh"]
                n_eval += len(miss)
                for j, r in zip(miss, res):
                    util[j] = r
            for g in lg:
                g.update(p, util)
            log(f"    round {rnd + 1} player {p + 1}/{maxp}: {len(miss)} new sims, "
                f"{sum(int(g.active.sum()) for g in lg)} active (game,seed)")
        for g in live:
            g.end_round()
    return util, n_eval
