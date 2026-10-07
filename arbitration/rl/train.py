"""REQ-OPT-02/03/06, REQ-SEED-03: objective, seed blocks, checkpoint selection, training guards.

Also the training loop itself (resumable, part files + checkpoints), warm start by least squares (commander decision P-7)
and a command line entry point (`python -m arbitration.rl.train`, always submitted with `gpujob`).
Only TRAIN seeds are used for fitness; VALIDATION seeds only for checkpoint / run selection and early stopping.
"""
import json
import os
import random
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import cmaes, parts, seeds as _seeds

SPEC_VERSION = "v1.0-rev2"
_PERM_CACHE: Dict[int, List[int]] = {}


def fitness(U: Any, T_score: int) -> Any:
    """REQ-OPT-02: f_i = mean_s U_i(s)/T_score (undiscounted).  U[P,S]."""
    return np.mean(np.asarray(U, dtype=float), axis=1) / T_score


def evaluate_generation(sim_fn: Callable[[Any, Sequence[int]], Any], thetas: Any, seeds: Sequence[int],
                        T_score: int) -> Dict[str, Any]:
    """REQ-OPT-02/05: call sim_fn(theta_rows[(lam+1),n], seeds) -> U[(lam+1),S] ONCE; row 0 is the all-zero honest baseline
    (computed once, shared).  Returns {"f": [lam], "f_honest": float, "G": [lam]}."""
    th = np.asarray(thetas, dtype=float)
    rows = np.vstack([np.zeros((1, th.shape[1])), th])
    U = np.asarray(sim_fn(rows, list(seeds)), dtype=float)
    if U.shape[0] != rows.shape[0]:
        raise ValueError("sim_fn must return one row per theta row")
    f_all = fitness(U, T_score)
    f_h = float(f_all[0])
    f = f_all[1:]
    return {"f": f, "f_honest": f_h, "G": f - f_h}


def seed_block(run_id: int, g: int, n_S: int) -> Sequence[int]:
    """REQ-OPT-03: block [g*n_S,(g+1)*n_S) (mod 1000) of perm_k, perm_k = Random(9000+k).shuffle(range(1000))."""
    if not 1 <= n_S <= 1000:
        raise ValueError("n_S must lie in [1, 1000] so that a block never repeats a seed")
    if run_id not in _PERM_CACHE:
        p = list(range(1000))
        random.Random(9000 + run_id).shuffle(p)
        _PERM_CACHE[run_id] = p
    p = _PERM_CACHE[run_id]
    return [p[i % 1000] for i in range(g * n_S, (g + 1) * n_S)]


def require_train_seeds(seeds: Sequence[int]) -> None:
    """REQ-SEED-03: ValueError for any seed >= 1000 (training seeds are 0-999; 2000+ forbidden)."""
    _seeds.require_split(seeds, "train")


def select_checkpoint(val_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """REQ-OPT-06: record with the largest validation G_val (earliest on ties).  Takes validation records only."""
    best = None
    for r in val_records:
        if best is None or r["G_val"] > best["G_val"]:
            best = r
    if best is None:
        raise ValueError("no validation records")
    return best


def select_run(run_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """REQ-SEED-05: pick the run (among the 3) by its best validation G_val (earliest run on ties); all runs are still reported."""
    return select_checkpoint([dict(r, G_val=r["best"]["G_val"]) for r in run_results])


# ------------------------------------------------------------------------------------------------ simulator glue
def make_sim_fn(mech: str, T: int, cfg: Dict[str, float], r: float, dev: str = "cuda", T_score: Optional[int] = None,
                F: int = 25, hidden: int = 2, obs_version: str = "public") -> Callable[[Any, Sequence[int]], Any]:
    """sim_fn(theta_rows, seeds) -> U[P,S] = util_b[:,:,1] at the score snapshot (REQ-OPT-02).  obs_version is forwarded explicitly
    to every MLPPolicy and to env.simulate (REQ-OPT-12)."""
    from . import env, policy

    def sim_fn(thetas, seeds_):
        pols = [policy.MLPPolicy(F=F, hidden=hidden, theta=t, obs_version=obs_version) for t in np.asarray(thetas)]
        out = env.simulate(mech, T, cfg, pols, seeds_, r=r, T_score=T_score, dev=dev, obs_version=obs_version)
        return out["snap"]["util_b"][:, :, 1].cpu().numpy()
    return sim_fn


def make_val_fn(mech: str, T_val: int, cfg: Dict[str, float], r: float, dev: str = "cuda", F: int = 25,
                hidden: int = 2, obs_version: str = "public") -> Callable[[np.ndarray], float]:
    """G_val(theta): paired mean gain over the 32 VALIDATION seeds (REQ-OPT-06)."""
    vs = list(_seeds.splits()["val"])
    _seeds.require_split(vs, "val")
    sim = make_sim_fn(mech, T_val, cfg, r, dev, None, F, hidden, obs_version)
    from . import env
    L = int(cfg["L"])
    Ts = env.score_window(T_val, L) if T_val > L else T_val

    def val_fn(theta):
        U = sim(np.vstack([np.zeros_like(theta), theta]), vs)
        return float(np.mean(U[1] - U[0]) / Ts)
    return val_fn


# ------------------------------------------------------------------------------------------------ warm start (P-7)
def fit_lstsq(obs: np.ndarray, target: np.ndarray, F: int = 25, hidden: int = 2, seed: int = 0, ridge: float = 1e-4,
              max_nfev: int = 200) -> np.ndarray:
    """Least squares (Levenberg-Marquardt/trf, analytic Jacobian) of d = tanh(W2.tanh(obs W1+b1)+b2) onto `target`
    (the d output of the hand-written strategy).  Returns the flat theta (policy layout)."""
    from scipy.optimize import least_squares
    obs = np.asarray(obs, float)
    tgt = np.asarray(target, float)
    N = obs.shape[0]
    nW1 = F * hidden
    rng = np.random.default_rng(seed)
    th0 = np.concatenate([0.3 * rng.standard_normal(nW1), np.zeros(hidden), 0.3 * rng.standard_normal(hidden), np.zeros(1)])
    sr = np.sqrt(ridge)

    def unpack(th):
        return th[:nW1].reshape(F, hidden), th[nW1:nW1 + hidden], th[nW1 + hidden:nW1 + 2 * hidden], th[-1]

    def res(th):
        W1, b1, W2, b2 = unpack(th)
        y = np.tanh(np.tanh(obs @ W1 + b1) @ W2 + b2)
        return np.concatenate([y - tgt, sr * th])

    def jac(th):
        W1, b1, W2, b2 = unpack(th)
        h = np.tanh(obs @ W1 + b1)
        y = np.tanh(h @ W2 + b2)
        gy = (1 - y ** 2)[:, None]
        gh = gy * W2[None, :] * (1 - h ** 2)                       # [N,hidden]
        J = np.concatenate([(obs[:, :, None] * gh[:, None, :]).reshape(N, nW1), gh, gy * h, gy], 1)
        return np.vstack([J, sr * np.eye(th.size)])

    sol = least_squares(res, th0, jac=jac, max_nfev=max_nfev)
    return sol.x


def best_handwritten_on_train(cfg: Dict[str, float], mech: str, r: float, T: int, train_seeds: Sequence[int], dev: str = "cuda",
                              strategies: Optional[Sequence] = None) -> Dict[str, Any]:
    """Pick the S_HW strategy with the largest paired G on TRAIN seeds (REQ-OPT-07); honest is the baseline row."""
    from . import env, hw, policy
    require_train_seeds(train_seeds)
    strategies = list(strategies) if strategies is not None else hw.build_s_hw()
    pols = [policy.AdapterPolicy({})] + [policy.AdapterPolicy(d) for _, d in strategies]
    out = env.simulate(mech, T, cfg, pols, list(train_seeds), r=r, dev=dev, obs_version="public")
    U = out["snap"]["util_b"][:, :, 1].cpu().numpy()
    G = (U[1:] - U[0:1]).mean(1) / out["T_score"]
    i = int(np.argmax(G))
    return dict(name=strategies[i][0], scols=dict(strategies[i][1]), G_train=float(G[i]), all_G={n: float(g) for (n, _), g in zip(strategies, G)})


def warm_start_theta(cfg: Dict[str, float], mech: str, r: float, scols: Dict[str, float], T: int, fit_seeds: Sequence[int],
                     dev: str = "cuda", F: int = 25, hidden: int = 2, n_rows: int = 20000, seed: int = 0) -> np.ndarray:
    """REQ-OPT-07 / P-7: roll the hand-written strategy out on TRAIN seeds, regress the MLP onto its d = v1 - u1."""
    from . import env, policy
    require_train_seeds(fit_seeds)
    out = env.simulate(mech, T, cfg, [policy.AdapterPolicy(scols)], list(fit_seeds), r=r, dev=dev, diagnostic=True, obs_version="public")
    d = (out["log"]["v1"] - out["log"]["u1"]).reshape(-1).cpu().numpy()
    obs = out["obs"].reshape(-1, F).cpu().numpy()
    idx = np.random.default_rng(seed).choice(len(d), size=min(n_rows, len(d)), replace=False)
    return fit_lstsq(obs[idx], d[idx], F, hidden, seed)


# ------------------------------------------------------------------------------------------------ checkpoint
def _save_ckpt(path: str, st: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, state=np.array(json.dumps(st, default=float)))
    os.replace(tmp, path)


def _load_ckpt(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["state"]))


def _git_sha(root: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def train_segment(cell: Dict[str, Any], run_id: int, g0: int, g1: int, sim_fn: Callable, val_fn: Callable, *, root: str = ".",
                  theta0: Optional[np.ndarray] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train generations [g0, g1) of one run (REQ-GPU-02/03/04).  cell: name, n, lam, n_S, T_score, sigma0, val_every, patience.
    Skips if the part file is done; resumes from the checkpoint (which must be at generation g0); writes the part file and the
    checkpoint atomically.  theta0 = initial mean for a fresh run (None -> zeros = cold start)."""
    name, n, lam, n_S = cell["name"], cell["n"], cell["lam"], cell["n_S"]
    val_every, patience = int(cell.get("val_every", 50)), cell.get("patience")
    pp = parts.part_path(name, run_id, g0, g1, root)
    ck = parts.ckpt_path(name, run_id, root)
    act = parts.job_action(pp, ck)
    if act == "skip":
        return parts.read_part(pp)
    opt = cmaes.make_optimizer(n, lam=lam, sigma0=cell.get("sigma0", 0.3), seed=9000 + run_id, diagnostic=cell.get("diagnostic", False))
    state = dict(gen=0, records=[], best=None, stopped=False, curve_G=[], curve_Gbest=[], curve_sigma=[])
    if act == "resume":
        st = _load_ckpt(ck)
        opt.load_state_dict(st["opt"])
        state = {k: st[k] for k in state}
        if state["gen"] < g0 and g0 != 0:
            raise RuntimeError(f"checkpoint at generation {state['gen']} < segment start {g0}")
    elif g0 != 0:
        raise RuntimeError("REQ-GPU-04: segment starts at g0>0 but there is no checkpoint")
    elif theta0 is not None:
        opt.mean = np.asarray(theta0, dtype=float).copy()
    start = state["gen"]
    t0 = time.time()
    f_mean = f_best = float("nan")
    t_gen: List[float] = []
    for g in range(start, g1):
        if state["stopped"]:
            break
        tg = time.time()
        sb = list(seed_block(run_id, g, n_S))
        require_train_seeds(sb)
        X = opt.ask()
        res = evaluate_generation(sim_fn, X, sb, cell["T_score"])
        opt.tell(X, res["f"])
        f_mean, f_best = float(np.mean(res["f"])), float(np.max(res["f"]))
        state["curve_G"].append(float(np.mean(res["G"])))
        state["curve_Gbest"].append(float(np.max(res["G"])))
        state["curve_sigma"].append(float(opt.sigma))
        state["gen"] = g + 1
        t_gen.append(time.time() - tg)
        log(f"gen {g + 1} G_mean {state['curve_G'][-1]:+.5f} G_best {state['curve_Gbest'][-1]:+.5f} sigma {opt.sigma:.4f} ({t_gen[-1]:.1f}s)")
        if (g + 1) % val_every == 0:
            theta = np.array(opt.mean)
            rec = dict(gen=g + 1, G_val=float(val_fn(theta)))
            state["records"].append(rec)
            if state["best"] is None or rec["G_val"] > state["best"]["G_val"]:
                state["best"] = dict(rec, theta=theta.tolist())
            log(f"  validation gen {g + 1}: G_val {rec['G_val']:+.5f} (best {state['best']['G_val']:+.5f}@{state['best']['gen']})")
            if patience:
                bi = max(range(len(state["records"])), key=lambda i: (state["records"][i]["G_val"], -i))
                if len(state["records"]) - 1 - bi >= patience:
                    state["stopped"] = True                # early stopping uses validation records only (REQ-OPT-06)
            _save_ckpt(ck, dict(state, opt=opt.state_dict()))
    done_gen = state["gen"]
    complete = state["stopped"] or done_gen >= g1
    _save_ckpt(ck, dict(state, opt=opt.state_dict()))
    rec = dict(cell=name, run_id=run_id, gen0=g0, gen1=g1, status="done" if complete else "partial", fitness_mean=f_mean,
               fitness_best=f_best, G_val=[r["G_val"] for r in state["records"] if g0 < r["gen"] <= g1], sigma=float(opt.sigma),
               t_step_ms=(1000.0 * float(np.mean(t_gen)) / cell["T_train"]) if t_gen else 0.0, wall_s=time.time() - t0,
               git_sha=_git_sha(root), spec_version=SPEC_VERSION, gens_done=done_gen, stopped_early=state["stopped"],
               G_mean_curve=state["curve_G"], G_best_curve=state["curve_Gbest"], sigma_curve=state["curve_sigma"],
               val_records=state["records"], best=state["best"], mean_theta=np.asarray(opt.mean).tolist())
    parts.write_part(pp, rec)
    return rec


# ------------------------------------------------------------------------------------------------ command line
def _cfg(args) -> Dict[str, float]:
    from . import env
    if args.mech == "NAIVE":
        return dict(env.m3c_config(args.r), p=0.0)
    return env.m3c_config(args.r)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Direction-2 training segment (submit with gpujob)")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--mech", default="NAIVE", choices=["NAIVE", "M3C"])
    ap.add_argument("--r", type=float, default=0.5)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--g0", type=int, default=0)
    ap.add_argument("--g1", type=int, required=True)
    ap.add_argument("--lam", type=int, default=32)
    ap.add_argument("--nS", type=int, default=16)
    ap.add_argument("--T", type=int, default=20000, help="training horizon T_train")
    ap.add_argument("--T-val", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--start", choices=["cold", "warm"], default="cold")
    ap.add_argument("--warm-T", type=int, default=4096)
    ap.add_argument("--warm-seeds", type=int, default=16)
    ap.add_argument("--root", default=".")
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args(argv)
    from . import env, policy
    cfg = _cfg(a)
    mech = "M3C"
    L = int(cfg["L"])
    T_score = env.score_window(a.T, L) if a.T > L else a.T
    n = policy.n_params(25, 2)
    cell = dict(name=a.cell, n=n, lam=a.lam, n_S=a.nS, T_score=T_score, T_train=a.T, sigma0=0.3, val_every=a.val_every, patience=a.patience)
    theta0 = None
    if a.start == "warm" and a.g0 == 0 and not os.path.exists(parts.ckpt_path(a.cell, a.run, a.root)):
        tr = list(_seeds.splits()["train"])[:a.warm_seeds]
        best = best_handwritten_on_train(cfg, mech, a.r, a.warm_T, tr, a.dev)
        print(f"warm start: best handwritten on train = {best['name']} (G_train {best['G_train']:+.4f})", flush=True)
        theta0 = warm_start_theta(cfg, mech, a.r, best["scols"], a.warm_T, tr, a.dev)
    sim_fn = make_sim_fn(mech, a.T, cfg, a.r, a.dev)
    val_fn = make_val_fn(mech, a.T_val or a.T, cfg, a.r, a.dev)
    rec = train_segment(cell, a.run, a.g0, a.g1, sim_fn, val_fn, root=a.root, theta0=theta0, log=lambda s: print(s, flush=True))
    print(json.dumps({k: rec[k] for k in ("cell", "run_id", "gen0", "gen1", "status", "gens_done", "wall_s", "sigma", "best")
                      if k in rec}, default=float), flush=True)


if __name__ == "__main__":
    main()
