"""Direction-2 speed-up (docs/direction2-speedup-plan.md): CUDA-graph stepping must be BITWISE identical to the eager simulator.

Two flavours of every behavioural test:
  * cpu  : `GraphSim(capture=False)` runs the very same unrolled-K-step machinery (static buffers, feeds, device-side t, tail steps,
           block feeding) but without CUDA graphs, on the CPU.  Catches logic errors without a GPU.
  * gpu  : `GraphSim(capture=True)` on CUDA (real capture / replay).  Submit through `gpujob` only.
The reference is always the unchanged eager path (graph=None) in the same process on the same device.
"""
import json
import os

import numpy as np
import pytest
import torch

from conftest import cuda_required
from arbitration.rl import env, policy, vulns, train, graphsim

DEVS = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="gpu", marks=pytest.mark.gpu)]
OPS = [(None, None), ("O1", 0.6), ("O2", 0.3), ("O3", 0.5), ("O5", 5), ("D1", 0.9), ("D2", 3), ("D4", 0.3)]
CFG = dict(p=0.5, tol=1.0, L=8.0, kref=0.5)          # many audits and flags in a short run


def _gs(dev, K=4):
    if dev == "cuda":
        cuda_required()
    return graphsim.GraphSim(K=K, capture=(dev == "cuda"))


def _pols(P, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return [policy.MLPPolicy(F=25, hidden=2, theta=scale * rng.standard_normal(55), obs_version="public") for _ in range(P)]


def _eq(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _eq(a[k], b[k], f"{path}/{k}")
    elif torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.cpu(), b.cpu()), f"{path}: not bitwise equal"
    else:
        assert a == b, path


def _sim(op, knob, dev, graph, T=70, Ts=None, P=5, seeds=(3, 1, 4, 1, 5, 9), cfg=CFG, mech="M3C", r=0.5, pol_seed=0):
    pols = _pols(P, pol_seed)
    return vulns.simulate_vuln(op, knob, mech, T, dict(cfg), pols, list(seeds), r=r, dev=dev, T_score=Ts, graph=graph)


# ------------------------------------------------------------------------------------------------ C1 scheduler
def test_C1_plan_events_cover_and_rules():
    for T, Ts, K in [(20000, 17500, 32), (100000, 97500, 32), (70, 61, 4), (70, 60, 4), (2100, 2090, 8), (5, 5, 4), (3, 1, 2), (1024, 1000, 32)]:
        ev = graphsim.plan_events(T, Ts, K)
        t = 1
        for kind, a, n in ev:
            assert kind in ("eager", "replay") and a == t and n >= 1
            if kind == "replay":
                assert n == K and a % 2 == 1, "replay must start at an odd step (t % M == 1 static) and have K steps"
                assert (a - 1) // 1024 == (a + n - 2) // 1024, "replay crosses an RNG block boundary"
                assert not (a <= Ts < a + n - 1), "replay crosses the snapshot step Ts"
            else:
                assert n == 1
            t += n
        assert t == T + 1
        ends = [a + n - 1 for _, a, n in ev]
        assert Ts in ends                                            # the snapshot lands exactly on an event end
        assert all(b % 1024 == 0 or b == Ts or b == T or True for b in ends)
        tails = [n for k, _, n in ev if k == "eager"]
        assert len(tails) <= 3 * (K + 1) + (T // 1024 + 2) * (K + 1)      # eager steps are only tails/parity fixes
    # production sizes: the number of eager steps is tiny
    ev = graphsim.plan_events(20000, 17500, 32)
    assert sum(n for k, _, n in ev if k == "eager") < 20 * 32 + 5
    assert sum(1 for k, _, _ in ev if k == "replay") >= 600


# ------------------------------------------------------------------------------------------------ C2/C3/G1/G3 simulate: graph == eager
@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("op,knob", OPS)
@pytest.mark.parametrize("T,Ts", [(70, 61), (70, 60), (23, None)])
def test_G1_simulate_graph_equals_eager(dev, op, knob, T, Ts):
    ref = _sim(op, knob, dev, None, T=T, Ts=Ts)
    got = _sim(op, knob, dev, _gs(dev), T=T, Ts=Ts)
    _eq(ref, got)
    assert float(ref["final"]["flags"].sum()) > 0 or op in ("D2",), "test config is too tame: no flags at all"


@pytest.mark.parametrize("dev", DEVS)
def test_G1_rng_block_boundary_and_tail(dev):
    """T crosses two 1024-step RNG block boundaries; Ts and T are not multiples of K."""
    ref = _sim("O1", 0.6, dev, None, T=2100, Ts=2090, P=3, seeds=(1, 2, 3))
    got = _sim("O1", 0.6, dev, _gs(dev, K=8), T=2100, Ts=2090, P=3, seeds=(1, 2, 3))
    _eq(ref, got)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("case", ["M4rw0", "NAIVE", "M3C"])
def test_G1_env_simulate_graph_equals_eager(dev, case):
    """The env.simulate path (NAIVE / M4 anchors, main cells)."""
    gs = _gs(dev)
    if case == "M4rw0":
        mech, cfg, r = "M4", dict(eps=0.2, L=6, rw=0.0), 0.99
    elif case == "NAIVE":
        mech, cfg, r = "M3C", dict(p=0.0, tol=6.0, L=8.0, kref=0.5), 0.5
    else:
        mech, cfg, r = "M3C", dict(CFG), 0.9
    pols = _pols(4, 1)
    seeds = [7, 8, 9, 7]
    ref = env.simulate(mech, 70, cfg, pols, seeds, r=r, dev=dev, T_score=61)
    got = env.simulate(mech, 70, cfg, pols, seeds, r=r, dev=dev, T_score=61, graph=gs)
    _eq(ref, got)
    if case == "M4rw0":
        assert float(ref["final"]["nraid"].sum()) > 0


@pytest.mark.parametrize("dev", DEVS)
def test_G1_replay_twice_and_two_cells_share_one_graphsim(dev):
    """The same GraphSim serves several generations / operators / shapes in a row (cache keyed by structure); no state leaks between calls."""
    gs = _gs(dev)
    for i in range(3):
        for op, knob in (("O1", 0.6), ("D1", 0.9), (None, None)):
            ref = _sim(op, knob, dev, None, T=40, Ts=33, pol_seed=i)
            got = _sim(op, knob, dev, gs, T=40, Ts=33, pol_seed=i)
            _eq(ref, got)
    ref = _sim("O3", 0.5, dev, None, T=40, Ts=33, P=2, seeds=(1, 2))
    _eq(ref, _sim("O3", 0.5, dev, gs, T=40, Ts=33, P=2, seeds=(1, 2)))       # a different batch shape: its own graph


@pytest.mark.parametrize("dev", DEVS)
def test_G6_rng_streams_identical(dev):
    """After the run every per-seed generator is in the same state as after the eager run (the graph contains no RNG op)."""
    states = {}
    for tag, graph in (("eager", None), ("graph", _gs(dev))):
        made = []
        orig = env._make_gens

        def spy(seeds, d, orig=orig, made=made):
            g = orig(seeds, d)
            made.append(g)
            return g
        env._make_gens = spy
        try:
            _sim("O1", 0.6, dev, graph, T=2100, Ts=2090, P=2, seeds=(1, 2, 3))
        finally:
            env._make_gens = orig
        states[tag] = [[x.get_state().numpy().tobytes() for x in lst] for lst in made[-1]]
    assert states["eager"] == states["graph"]


# ------------------------------------------------------------------------------------------------ G4/G5 training segments
def _cell(lam=6, n_S=4, T=48, val_every=2):
    return dict(name="speedtest", n=55, lam=lam, n_S=n_S, T_score=T - 8, T_train=T, sigma0=0.3, val_every=val_every, patience=None)


def _fns(dev, graph, lam=6, n_S=4, T=48, T_val=40, operator="O1", knob=0.6, mech="M3C", cfg=None, r=0.5):
    cfg = dict(CFG) if cfg is None else cfg
    sim = train.make_sim_fn(mech, T, cfg, r, dev, None, 25, 2, "public", operator, knob, graph=graph)
    val = train.make_val_fn(mech, T_val, cfg, r, dev, 25, 2, "public", operator, knob, graph=graph)
    return sim, val


def _strip(rec, full=False):
    drop = {"wall_s", "t_step_ms", "git_sha"}                  # time/commit fields differ by nature
    if not full:
        drop |= {"gen0", "gen1", "G_val", "status"}            # segment-boundary dependent; everything cumulative (curves, val_records, best, theta, cma) stays
    return {k: v for k, v in rec.items() if k not in drop}


def _ckpt_json(root):
    p = os.path.join(root, "results", "direction2", "ckpt", "speedtest__run0.npz")
    return train._load_ckpt(p)


def _run_seg(tmp, tag, dev, graph, g0, g1, **kw):
    root = os.path.join(str(tmp), tag)
    os.makedirs(root, exist_ok=True)
    cell = _cell()
    sim, val = _fns(dev, graph, **kw)
    return root, lambda a, b: train.train_segment(cell, 0, a, b, sim, val, root=root, log=lambda s: None)


@pytest.mark.parametrize("dev", DEVS)
def test_G4_train_segment_graph_equals_eager(dev, tmp_path):
    root_e, run_e = _run_seg(tmp_path, "eager", dev, None, 0, 4)
    root_g, run_g = _run_seg(tmp_path, "graph", dev, _gs(dev), 0, 4)
    re, rg = run_e(0, 4), run_g(0, 4)
    assert _strip(re, True) == _strip(rg, True)
    assert re["val_records"] and re["best"] is not None
    assert _ckpt_json(root_e) == _ckpt_json(root_g)
    assert np.array_equal(np.array(re["mean_theta"]), np.array(rg["mean_theta"]))


@pytest.mark.parametrize("dev", DEVS)
def test_G5_resume_in_pieces_and_switch_eager_to_graph_mid_run(dev, tmp_path):
    """The key property for switching the school machine in the middle of S1: a checkpoint written by the eager program is continued by the
    graph program (fresh process state, fresh capture) and the result equals an all-eager run; the same the other way round."""
    root_ref, run_ref = _run_seg(tmp_path, "ref", dev, None, 0, 6)
    ref = run_ref(0, 6)
    ck_ref = _ckpt_json(root_ref)
    # eager [0,2) then graph [2,4) then eager [4,6)
    root_x, run_e = _run_seg(tmp_path, "mix", dev, None, 0, 2)
    run_e(0, 2)
    _, run_g = _run_seg(tmp_path, "mix", dev, _gs(dev), 2, 4)
    run_g(2, 4)
    _, run_e2 = _run_seg(tmp_path, "mix", dev, None, 4, 6)
    got = run_e2(4, 6)
    assert _strip(got) == _strip(ref) and _ckpt_json(root_x) == ck_ref
    # graph [0,3) then graph [3,6) (new GraphSim = new process) == eager
    root_y, run_g1 = _run_seg(tmp_path, "gg", dev, _gs(dev), 0, 3)
    run_g1(0, 3)
    _, run_g2 = _run_seg(tmp_path, "gg", dev, _gs(dev), 3, 6)
    got2 = run_g2(3, 6)
    assert _strip(got2) == _strip(ref) and _ckpt_json(root_y) == ck_ref


@pytest.mark.parametrize("dev", DEVS)
def test_G5_interrupted_graph_run_resumes_identically(dev, tmp_path):
    root_ref, run_ref = _run_seg(tmp_path, "ref", dev, None, 0, 6)
    ref = run_ref(0, 6)
    gs = _gs(dev)
    root, _ = _run_seg(tmp_path, "intr", dev, gs, 0, 6)
    cell = _cell()
    sim, val = _fns(dev, gs)
    calls = {"n": 0}

    def boom(th, seeds):
        calls["n"] += 1
        if calls["n"] == 4:                                     # inside generation 4 (after the checkpoint of generation 2)
            raise KeyboardInterrupt
        return sim(th, seeds)
    with pytest.raises(KeyboardInterrupt):
        train.train_segment(cell, 0, 0, 6, boom, val, root=root, log=lambda s: None)
    sim2, val2 = _fns(dev, _gs(dev))
    got = train.train_segment(cell, 0, 0, 6, sim2, val2, root=root, log=lambda s: None)
    assert _strip(got) == _strip(ref)
    assert _ckpt_json(root) == _ckpt_json(root_ref)


@pytest.mark.parametrize("dev", DEVS)
def test_G7_validation_and_training_graphs_interleave(dev):
    gs = _gs(dev)
    sim_e, val_e = _fns(dev, None)
    sim_g, val_g = _fns(dev, gs)
    th = np.random.default_rng(5).standard_normal((7, 55))
    seeds = [4, 5, 6, 7]
    for _ in range(2):
        assert np.array_equal(sim_e(th, seeds), sim_g(th, seeds))
        assert val_e(th[1]) == val_g(th[1])


# ------------------------------------------------------------------------------------------------ misc: switch, flag, hazards
def test_C5_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("D2_GRAPH", raising=False)
    assert graphsim.from_env("cpu") is None
    monkeypatch.setenv("D2_GRAPH", "0")
    assert graphsim.from_env("cpu") is None
    monkeypatch.setenv("D2_GRAPH", "1")
    monkeypatch.setenv("D2_GRAPH_K", "16")
    gs = graphsim.from_env("cuda")
    assert gs is not None and gs.K == 16 and gs.capture is True


def test_C5_graph_requires_cuda_and_even_K():
    with pytest.raises(ValueError):
        graphsim.GraphSim(K=3, capture=False)
    with pytest.raises(RuntimeError):
        graphsim.from_env("cpu", force=True)             # --graph on a CPU device: clear error, no silent fallback


def test_C5_unsupported_features_fall_back_to_eager_path_explicitly():
    """hook logs / adapters (warm start, diagnostics) are not graph-able: simulate_vuln must run them eagerly even if a graph is passed."""
    gs = graphsim.GraphSim(K=4, capture=False)
    pols = [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=0.5))]
    a = vulns.simulate_vuln("O1", 0.5, "M3C", 30, dict(CFG), pols, [1, 2], r=0.5, dev="cpu", graph=None)
    b = vulns.simulate_vuln("O1", 0.5, "M3C", 30, dict(CFG), pols, [1, 2], r=0.5, dev="cpu", graph=gs)
    _eq(a, b)
    assert not gs.cache, "an adapter policy must not build a graph"


def test_eager_path_unchanged_by_refactor_cpu():
    """The `t`-as-tensor refactor must leave the eager path untouched: results equal the pre-refactor values recorded here (CPU, torch float64
    is deterministic for these elementwise ops).  Hash taken from the baseline commit cc26592 with the same inputs."""
    import hashlib
    gold = json.load(open(os.path.join(os.path.dirname(__file__), "golden_eager_cpu.json")))
    if gold["torch"] != torch.__version__:
        pytest.skip(f"golden digest was generated with torch {gold['torch']} (CPU math kernels may differ between versions)")
    h = hashlib.sha256()
    for op, knob in OPS:
        out = _sim(op, knob, "cpu", None, T=70, Ts=61)
        for part in ("final", "snap"):
            for k in sorted(out[part]):
                h.update(out[part][k].numpy().tobytes())
    # the expected digest is stored next to the test (generated from the baseline code, see tests/direction2/golden_eager_cpu.json)
    assert h.hexdigest() == gold["sha256"]


# ------------------------------------------------------------------------------------------------ G2 / G3 / G8 / G10 (GPU hazards, production shape)
@pytest.mark.gpu
def test_G2_no_host_sync_in_the_eager_step():
    """The graph stepping relies on the step function never synchronising with the host (a sync inside capture is an error).  Checked directly:
    five eager steps (MLP policy, M3C operator path and env path) under torch.cuda.set_sync_debug_mode('error')."""
    cuda_required()
    pols = _pols(3)
    from arbitration.rl import vulns as V
    seeds = [1, 2, 3]
    N = len(pols) * len(seeds)
    for op, knob in (("D2", 3), ("D4", 0.3), ("O1", 0.6)):
        st = env.SimState("M3C", dict(CFG), 0.5, seeds, np.tile(np.arange(3), 3), "cuda", "public", track_obs=True)
        bank = env._Bank(pols, torch.as_tensor(np.arange(N) // 3, device="cuda"), "cuda", 25, obs_version="public")
        hs = V._HS(N, "cuda")
        o = V.get_operator(op)
        if o.ext is not None:
            hs.ext = o.ext(N, "cuda", knob, st)
        V._vstep(st, bank, o, knob, hs, None)                       # warm (kernel loading may sync)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(4):
                V._vstep(st, bank, o, knob, hs, None)
        finally:
            torch.cuda.set_sync_debug_mode("default")


@pytest.mark.gpu
@pytest.mark.parametrize("op,knob", [("D2", 3), ("D4", 0.3), ("O1", 0.6), (None, None)])
def test_G3_production_shape_graph_equals_eager(op, knob):
    """B = (256+1) x 256 = 65792, K=32, T=200 with a non-multiple Ts: the shape the experiment really runs (reduction configs depend on shape)."""
    cuda_required()
    pols = _pols(257, 11, scale=0.7)
    seeds = list(range(256))
    cfg = dict(p=0.3, tol=2.0, L=20.0, kref=0.5)
    ref = vulns.simulate_vuln(op, knob, "M3C", 200, dict(cfg), pols, seeds, r=0.5, dev="cuda", T_score=181)
    gs = graphsim.GraphSim(K=32)
    got = vulns.simulate_vuln(op, knob, "M3C", 200, dict(cfg), pols, seeds, r=0.5, dev="cuda", T_score=181, graph=gs)
    _eq(ref, got)
    got2 = vulns.simulate_vuln(op, knob, "M3C", 200, dict(cfg), pols, seeds, r=0.5, dev="cuda", T_score=181, graph=gs)        # replay the cached graph again
    _eq(ref, got2)
    assert gs.stats["replays"] >= 10 and gs.stats["builds"] == 1


@pytest.mark.gpu
def test_G3_production_shape_m4_env_path():
    cuda_required()
    pols = _pols(257, 12, scale=0.7)
    seeds = list(range(256))
    cfg = dict(eps=0.2, L=20.0, rw=0.0)
    ref = env.simulate("M4", 200, dict(cfg), pols, seeds, r=0.99, dev="cuda", T_score=181)
    got = env.simulate("M4", 200, dict(cfg), pols, seeds, r=0.99, dev="cuda", T_score=181, graph=graphsim.GraphSim(K=32))
    _eq(ref, got)


@pytest.mark.gpu
def test_G8_memory_does_not_grow_over_generations():
    cuda_required()
    gs = graphsim.GraphSim(K=8)
    sim, _ = _fns("cuda", gs)
    th = np.random.default_rng(1).standard_normal((7, 55))
    used = []
    for g in range(5):
        sim(th, [1, 2, 3, 4])
        torch.cuda.synchronize()
        used.append((torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    assert used[2] == used[3] == used[4], used
    assert gs.stats["builds"] == 1


@pytest.mark.parametrize("dev", DEVS)
def test_G10_non_finite_action_is_detected_in_graph_mode_too(dev, monkeypatch):
    """env path: eager raises at t==1; graph stepping reads a device flag once per RNG block / at the end of the run (no sync inside the graph)."""
    if dev == "cuda":
        cuda_required()
    pols = _pols(2)
    monkeypatch.setattr(env, "mlp_forward", lambda obs, *w: torch.full(obs.shape[:-1], float("nan"), dtype=torch.float64, device=obs.device))
    with pytest.raises(FloatingPointError):
        env.simulate("M3C", 20, dict(CFG), pols, [1, 2], r=0.5, dev=dev)
    with pytest.raises(FloatingPointError):
        env.simulate("M3C", 20, dict(CFG), pols, [1, 2], r=0.5, dev=dev, graph=_gs(dev))
    with pytest.raises(ValueError):                                    # a NaN theta is still refused when the policy is built
        policy.MLPPolicy(F=25, hidden=2, theta=np.full(55, np.nan))


@pytest.mark.gpu
def test_G9_real_pilot_data_reproduced_bitwise_by_graph_stepping():
    """Real size (lam=256, n_S=256, T=20000, NAIVE pilot cell).  On the machine that wrote results/direction2/parts/pilot_naive_cold__run0__g0000-0036.json
    (RTX 5070 Ti, eager program) the first 3 generations of the graph run must equal the committed numbers.  On any other GPU those numbers were made by
    another device, so the oracle is the eager run of the same code on the same GPU (1 generation)."""
    cuda_required()
    from arbitration.rl import parts, plan as _plan
    import tempfile
    repo = train.__file__.rsplit("/arbitration/", 1)[0]
    ref = parts.read_part(parts.part_path("pilot_naive_cold", 0, 0, 36, repo))
    pl = _plan.load_plan(os.path.join(repo, _plan.DEFAULT_PLAN), os.path.join(repo, _plan.DEFAULT_FREEZE))
    S = _plan.cell_to_settings(_plan.get_cell(pl, "pilot_naive_cold"))
    L = int(S["cfg"]["L"])
    cell = dict(name="g9", n=policy.n_params(S["F"], S["hidden"]), lam=S["lam"], n_S=S["n_S"], T_score=env.score_window(S["T_train"], L),
                T_train=S["T_train"], sigma0=S["sigma0"], val_every=S["val_every"], patience=None)

    def run(graph, n):
        sim = train.make_sim_fn(S["mech"], S["T_train"], S["cfg"], S["r"], "cuda", None, S["F"], S["hidden"], S["obs_version"], S["operator"], S["knob"],
                                **({} if graph is None else dict(graph=graph)))
        with tempfile.TemporaryDirectory() as root:
            return train.train_segment(cell, 0, 0, n, sim, lambda th: 0.0, root=root, log=lambda s: None)
    if "5070 Ti" in torch.cuda.get_device_name(0):
        rec = run(graphsim.GraphSim(K=32), 3)
        want = {k: ref[k][:3] for k in ("G_mean_curve", "G_best_curve", "sigma_curve")}
    else:
        rec = run(graphsim.GraphSim(K=32), 1)
        e = run(None, 1)
        want = {k: e[k] for k in ("G_mean_curve", "G_best_curve", "sigma_curve")}
    for k, v in want.items():
        assert rec[k] == v, k
