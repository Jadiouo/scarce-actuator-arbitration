"""T-01..T-11: environment (spec s14.1).  [gpu] tests are meant to run through `gpujob` only."""
import itertools
import math

import numpy as np
import pytest
import torch

from conftest import (STRATS_T01, STRATS_T01_FULL, assert_fields_equal, ci196, cuda_required, forbid_frontier, frontier_ref,
                      load_json, parse_range, phi1, rel_close, strategy_dict)
from arbitration.rl import env, policy, metrics

FIELDS = ["reg_b", "util_b", "flags", "pen", "audits", "susp"]


def _adapters(strats):
    return [policy.AdapterPolicy(d) for _, d in strats]


def test_T01_strategy_set_covers_all_scols():
    """T-01 (coverage guard, REQ-ENV-04 'SCOLS 全部欄位'): the parity strategy set uses EVERY frontier SCOLS column (leg, timing, dzc,
    ue_q, ad_c ... included), so a simulator that silently ignores a column cannot pass T-01/T-02."""
    from arbitration.gpu import frontier
    used = set().union(*[set(d) for _, d in STRATS_T01_FULL])
    assert used == set(frontier.SCOLS), sorted(set(frontier.SCOLS) ^ used)


@pytest.mark.gpu
def test_T01_parity_m3c_per_seed(monkeypatch):
    """T-01 (REQ-ENV-04, ENV-10, ENV-12): adapter strategies (all SCOLS columns) == frontier.simulate(M3C), bitwise, p=0.1,h=6,L=300,kref=0.5,
    T=3000.  Expectations are computed first (cached); then frontier.simulate is replaced by a function that raises, so the new
    simulator cannot pass by delegating to it."""
    cuda_required()
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    seeds = [0, 1, 2, 5000, 5001]
    refs = {r: frontier_ref("M3C", 3000, cfg, r, [d for _, d in STRATS_T01_FULL], seeds, "cuda") for r in (0.5, 0.9)}
    forbid_frontier(monkeypatch)
    any_flag = any_audit = any_pen = False
    for r in (0.5, 0.9):
        out = env.simulate("M3C", 3000, cfg, _adapters(STRATS_T01_FULL), seeds, r=r, T_score=3000, dev="cuda")
        assert_fields_equal(out["final"], refs[r], FIELDS, f"r={r} ")
        any_flag |= bool((refs[r]["flags"] > 0).any())
        any_audit |= bool((refs[r]["audits"] > 0).any())
        any_pen |= bool((refs[r]["pen"] > 0).any())
    assert any_flag and any_audit and any_pen, "configuration does not exercise audits / CUSUM flags / suspensions"


@pytest.mark.gpu
def test_T02_parity_m4_and_naive(monkeypatch):
    """T-02 (REQ-ENV-03, ENV-04): M4 (eps=0.05,L=300,rw in {0,1/3,1}) and NAIVE (M3C,p=0) match frontier, incl. nraid; frontier.simulate
    is disabled while the new simulator runs (expectations computed beforehand)."""
    cuda_required()
    seeds = [0, 1, 2, 5000, 5001]
    fields = FIELDS + ["nraid"]
    S = [d for _, d in STRATS_T01_FULL]
    cases = []
    for r in (0.5, 0.9):
        for rw in (0.0, 1 / 3, 1.0):
            cases.append(("M4", dict(eps=0.05, L=300.0, rw=rw), r, f"M4 r={r} rw={rw} "))
        cases.append(("M3C", dict(p=0.0, tol=6.0, L=300.0, kref=0.5), r, f"NAIVE r={r} "))
    refs = [frontier_ref(mech, 3000, cfg, r, S, seeds, "cuda") for mech, cfg, r, _ in cases]
    forbid_frontier(monkeypatch)
    assert any(bool((ref["nraid"] > 0).any()) for ref, c in zip(refs, cases) if c[0] == "M4")
    for (mech, cfg, r, tag), ref in zip(cases, refs):
        out = env.simulate(mech, 3000, cfg, _adapters(STRATS_T01_FULL), seeds, r=r, T_score=3000, dev="cuda")
        assert_fields_equal(out["final"], ref, fields, tag)


@pytest.mark.gpu
def test_T03_frontier_quota_equivalence():
    """T-03 (REQ-ENV-06): precondition on EXISTING code.  frontier(M3C,p=0) == quota(NAIVE); frontier == quota on M3C and M4,
    bitwise on util_b, reg_b, pen, nraid.  (No new implementation is needed: this test is expected to be green already.)"""
    cuda_required()
    from arbitration.gpu import frontier, quota
    seeds = [0, 1, 2, 5000, 5001]
    S = [d for _, d in STRATS_T01]
    fields = ["util_b", "reg_b", "pen", "nraid"]

    def q_run(mech, cfg, r):
        el = quota.build_pairs([(dict(cfg), dict(rho=r))], [dict(d) for d in S], seeds, dev="cuda")
        o = quota.simulate(mech, 3000, el, seeds)
        return {k: v.reshape(len(S), len(seeds), *v.shape[1:]) for k, v in o.items()}

    for r in (0.5, 0.9):
        cfg0 = dict(p=0.0, tol=6.0, L=300.0, kref=0.5)
        assert_fields_equal(q_run("NAIVE", cfg0, r), frontier_ref("M3C", 3000, cfg0, r, S, seeds, "cuda"), fields, f"NAIVE r={r} ")
        cfg1 = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
        assert_fields_equal(q_run("M3C", cfg1, r), frontier_ref("M3C", 3000, cfg1, r, S, seeds, "cuda"), fields, f"M3C r={r} ")
        cfg2 = dict(eps=0.05, L=300.0, rw=1 / 3)
        assert_fields_equal(q_run("M4", cfg2, r), frontier_ref("M4", 3000, cfg2, r, S, seeds, "cuda"), fields, f"M4 r={r} ")


def test_T04_zero_deviation_is_honest_bitwise():
    """T-04 (REQ-ENV-05, ACT-01, ACT-04): theta=0 MLP, and MLP with zero output layer but random hidden layer, equal
    frontier honest bitwise; v1 sequence == u1."""
    F = 25
    rng = np.random.default_rng(0)
    pols = [policy.MLPPolicy(F=F),
            policy.MLPPolicy.from_blocks(rng.normal(size=(F, 2)), rng.normal(size=2), np.zeros(2), np.zeros(1))]
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    seeds = [0, 1, 2]
    out = env.simulate("M3C", 2000, cfg, pols, seeds, r=0.5, T_score=2000, dev="cpu", diagnostic=True)
    ref = frontier_ref("M3C", 2000, cfg, 0.5, [{}], seeds, "cpu")
    for i in range(len(pols)):
        got = {f: out["final"][f][i:i + 1] for f in FIELDS}
        assert_fields_equal(got, ref, FIELDS, f"policy {i} ")
    assert torch.equal(out["log"]["v1"], out["log"]["u1"])


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["r0.5-c0", "r0.5-c1", "r0.5-c2", "r0.5-c3", "r0.5-c4", "r0.5-c5", "r0.5-honest",
                                  "r0.9-c0", "r0.9-c1", "r0.9-c2", "r0.9-c3", "r0.9-c4", "r0.9-c5", "r0.9-honest"])
def test_T05_replay_d1_json_m3c(case, monkeypatch):
    """T-05 (REQ-ENV-07, ENV-10, MET-05, GPU-08): replay quota.json main[12] (r=0.5) / main[14] (r=0.9): 36 non-quota strategy
    gains (6 chunks of 6, each its own <=25 min job), honest_util_agent1, R_hon_b, false_punish_frac.  Relative tol 1e-9
    (exact 0.0 when JSON is 0.0); CI half-widths (1.96 formula) only for fields that have a CI; T=1e5, seeds 5000-5031."""
    cuda_required()
    forbid_frontier(monkeypatch)        # expectations come from the JSON; the simulator must not delegate to frontier.simulate
    rtag, part = case.split("-")
    r = float(rtag[1:])
    q = load_json("quota.json")
    row = q["main"][12 if r == 0.5 else 14]
    assert row["r"] == r and row["mech"] == "M3C"
    seeds = parse_range(q["meta"]["eval_seeds"])
    param = load_json("stage2.json")["main"]["M3c"]["100000"][str(r)]["param"]
    cfg = env.m3c_config(r)
    assert (cfg["p"], cfg["tol"], cfg["L"]) == (param["p"], float(param["tol"]), float(param["L"]))
    names = [n for n in row["gains_by_strategy"] if not n.startswith("quota_")]
    assert len(names) == 36
    T = q["meta"]["T"]
    if part == "honest":
        sel = []
    else:
        c = int(part[1:])
        sel = names[c * 6:(c + 1) * 6]
    pols = [policy.AdapterPolicy({})] + [policy.AdapterPolicy(strategy_dict(n)) for n in sel]
    out = env.simulate("M3C", T, cfg, pols, seeds, r=r, T_score=T, dev="cuda")["final"]
    ub = out["util_b"][:, :, 1].cpu().numpy() / T
    for j, n in enumerate(sel):
        g = float((ub[j + 1] - ub[0]).mean())
        assert rel_close(g, row["gains_by_strategy"][n]), f"{n}: {g} vs {row['gains_by_strategy'][n]}"
    if part == "honest":
        for key, vals in (("honest_util_agent1", ub[0]), ("R_hon_b", out["reg_b"][0].cpu().numpy() / T),
                          ("false_punish_frac", out["pen"][0].cpu().numpy().mean(-1) / T)):
            got = ci196(vals)
            assert rel_close(got[0], row[key][0]) and rel_close(got[1], row[key][1]), key


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["naive", "m4"])
def test_T06_replay_d1_json_naive_m4(case, monkeypatch):
    """T-06 (REQ-ENV-07, ENV-11): replay main[4] (NAIVE r=0.5: Gmax for dz=2.0, CI via 1.96) and main[88] (M4 eps/L/rw from
    its JSON `param`: raid_waste_frac, R_hon_b, non-quota gains).  Tolerances/CI rules as T-05."""
    cuda_required()
    forbid_frontier(monkeypatch)        # expectations come from the JSON; the simulator must not delegate to frontier.simulate
    q = load_json("quota.json")
    seeds = parse_range(q["meta"]["eval_seeds"])
    T = q["meta"]["T"]
    if case == "naive":
        row = q["main"][4]
        assert row["mech"] == "NAIVE"
        cfg = dict(p=0.0, tol=6.0, L=2500.0, kref=0.5)
        pols = [policy.AdapterPolicy({}), policy.AdapterPolicy(strategy_dict(row["Gmax_strategy"]))]
        out = env.simulate("M3C", T, cfg, pols, seeds, r=0.5, T_score=T, dev="cuda")["final"]
        ub = out["util_b"][:, :, 1].cpu().numpy() / T
        got = ci196(ub[1] - ub[0])
        assert rel_close(got[0], row["Gmax"][0]) and rel_close(got[1], row["Gmax"][1])
    else:
        row = q["main"][88]
        assert row["mech"] == "M4" and row["r"] == 0.5
        cfg = dict(row["param"])
        names = [n for n in row["gains_by_strategy"] if not n.startswith("quota_")]
        pols = [policy.AdapterPolicy({})] + [policy.AdapterPolicy(strategy_dict(n)) for n in names[:6]]
        out = env.simulate("M4", T, cfg, pols, seeds, r=0.5, T_score=T, dev="cuda")["final"]
        ub = out["util_b"][:, :, 1].cpu().numpy() / T
        for key, vals in (("raid_waste_frac", out["nraid"][0].cpu().numpy() / T), ("R_hon_b", out["reg_b"][0].cpu().numpy() / T)):
            got = ci196(vals)
            assert rel_close(got[0], row[key][0]) and rel_close(got[1], row[key][1]), key
        for j, n in enumerate(names[:6]):
            assert rel_close(float((ub[j + 1] - ub[0]).mean()), row["gains_by_strategy"][n]), n


@pytest.mark.parametrize("mech,cfg", [("M3C", dict(p=0.1, tol=6.0, L=300.0, kref=0.5)), ("M4", dict(eps=0.05, L=300.0, rw=1 / 3))])
def test_T01b_const_mlp_equals_frontier_b_strategy(mech, cfg, monkeypatch):
    """T-01b (REQ-ENV-04/05, ACT-01): an MLP whose output is the constant d (W2=0, b2=atanh(d), random hidden layer) must reproduce the
    hand-written frontier strategy {b: d} (d>0: v1=min(u1+d,1) in both).  Flags/pen/audits/nraid/susp bitwise, util/reg rel 1e-12 (tanh
    ulp).  frontier.simulate is disabled after the expectations are computed."""
    ds_b2 = (0.05, 0.3, 1.2)                                            # d = tanh(b2) ~ 0.05, 0.29, 0.83
    d_exact = [float(torch.tanh(torch.tensor(b, dtype=torch.float64))) for b in ds_b2]
    seeds = [0, 1, 2, 5000]
    rng = np.random.default_rng(7)
    pols = [policy.MLPPolicy.from_blocks(rng.normal(size=(25, 2)), rng.normal(size=2), np.zeros(2), np.array([b])) for b in ds_b2]
    ref = frontier_ref(mech, 3000, cfg, 0.5, [dict(b=d) for d in d_exact], seeds, "cpu")
    forbid_frontier(monkeypatch)
    out = env.simulate(mech, 3000, cfg, pols, seeds, r=0.5, T_score=3000, dev="cpu")["final"]
    exact = ["flags", "pen", "audits", "susp"] + (["nraid"] if mech == "M4" else [])
    assert_fields_equal(out, ref, exact, f"{mech} ")
    for f in ("reg_b", "util_b"):
        a, b = out[f].double().cpu(), ref[f].double().cpu()
        assert ((a - b).abs() <= 1e-12 * b.abs().clamp(min=1.0)).all(), f
    if mech == "M4":
        assert (ref["nraid"] > 0).any(), "no raid occurred: configuration does not exercise the M4 path"


@pytest.mark.parametrize("mech,cfg", [("M3C", dict(p=0.5, tol=1.0, L=15.0, kref=0.5)), ("M4", dict(eps=0.3, L=15.0, rw=1 / 3))])
def test_T07b_step_state_rolling_matches_frontier(mech, cfg, monkeypatch):
    """T-07b (REQ-ENV-12/04): the stepwise state (env.step_state), rolled round by round, equals frontier.simulate run for t=1..T rounds
    (reg_b, util_b, flags, pen, audits, nraid, susp, bitwise) after EVERY round (small scale: T=60, L=15, h=1, p=0.5).  frontier is disabled
    once the 60 prefix references are cached."""
    T, seeds, r = 60, [0, 1, 2], 0.5
    strats = [{}, dict(b=0.3), dict(dz=0.3), dict(em=0.05), dict(ue_mode=1, ue_band=0.1), dict(ad_m=1.0, ad_f=0.9)]
    refs = [frontier_ref(mech, t, cfg, r, strats, seeds, "cpu") for t in range(1, T + 1)]
    assert (refs[-1]["flags"] > 0).any() and (refs[-1]["audits"] > 0).any() or mech == "M4"
    forbid_frontier(monkeypatch)
    states = [env.init_state(mech, cfg, r, seeds, dev="cpu") for _ in strats]
    pols = [policy.AdapterPolicy(d) for d in strats]
    for t in range(1, T + 1):
        for i, (st, pol) in enumerate(zip(states, pols)):
            env.step_state(st, pol)
            assert st.t == t
            for f in ("reg_b", "util_b", "flags", "pen", "audits", "nraid", "susp"):
                got, exp = getattr(st, f).double().cpu(), refs[t - 1][f][i].double().cpu()
                assert torch.equal(got, exp), f"round {t} strategy {strats[i]} field {f}"


# ---------------------------------------------------------------- T-07: event order (CPU, constructed scenarios)
def _state(mech="M3C", cfg=None, r=0.5, seeds=(0, 1, 2, 3)):
    cfg = cfg or dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    return env.init_state(mech, cfg, r, list(seeds), dev="cpu")


def test_T07_event_order_and_susp_no_stack():
    """T-07 (REQ-ENV-12, ENV-13): audit scheduled at t settles at t+1; flagged agent ineligible for t+1..t+L (eligible again at
    t+L+1); a flag does not stack/extend an existing suspension; upd = has & elig_ag (audited agent already suspended at
    settlement => S unchanged, no flag, but sxy/sxx still accumulate on `has`); suspended agents are never raid targets."""
    h, L = 6.0, 300
    honest = policy.AdapterPolicy({})
    liar = policy.AdapterPolicy(dict(b=1.0))

    # (a) flag at settlement, suspension window exactly t+1..t+L
    st = _state()
    t0 = 10
    st.t = t0
    st.cus[:, 1] = h
    slot = (t0 + 1) % st.M
    st.pend_a[:, slot] = 1
    st.pend_v[:, slot] = 1.0
    info = env.step_state(st, liar)                       # round t0+1: settlement of the audit scheduled at t0
    assert (st.flags[:, 1] == 1).all() and (st.cus[:, 1] == 0).all()
    assert (st.susp[:, 1] == (t0 + 1) + L).all()
    for k in range(1, L + 1):                              # rounds t0+2 .. t0+1+L: still suspended
        info = env.step_state(st, honest)
        assert not info["elig"][:, 1].any(), f"agent 1 must be ineligible in round {st.t}"
    info = env.step_state(st, honest)                      # round t0+L+2: eligible again
    assert info["elig"][:, 1].all()

    # (b) audited agent already suspended at settlement: S unchanged, no flag, susp not extended, sxy/sxx accumulate
    st = _state()
    st.t = t0
    st.susp[:, 1] = t0 + 100
    st.cus[:, 1] = h
    slot = (t0 + 1) % st.M
    st.pend_a[:, slot] = 1
    st.pend_v[:, slot] = 1.0
    sxy0, sxx0 = st.sxy.clone(), st.sxx.clone()
    env.step_state(st, honest)
    assert (st.flags[:, 1] == 0).all()
    assert (st.cus[:, 1] == h).all()
    assert (st.susp[:, 1] == t0 + 100).all(), "suspension must be overwritten by flags only, never stacked/extended"
    zv = torch.special.ndtri(phi1(torch.full_like(st.sxy, 1.0)))
    zw = torch.special.ndtri(phi1(torch.special.ndtr(st.z[:, 1])))
    assert torch.allclose(st.sxy - sxy0, zv * zw, rtol=0, atol=1e-12)
    assert torch.allclose(st.sxx - sxx0, zv * zv, rtol=0, atol=1e-12)

    # (c) M4: suspended agents are not raid targets (rw=0: target among eligible non-winners only)
    st = _state("M4", dict(eps=1.0, L=300.0, rw=0.0))
    st.t = t0
    st.susp[:, 0] = t0 + 1000
    st.susp[:, 2] = t0 + 1000
    info = env.step_state(st, liar)                        # agent 1 wins (reports 1.0); no eligible non-winner => no raid
    assert not info["rd"].any() and (st.nraid == 0).all() and (st.flags == 0).all()
    assert (st.susp[:, 0] == t0 + 1000).all() and (st.susp[:, 2] == t0 + 1000).all()


def _rand_mlp(seed, F=25, scale=0.5):
    rng = np.random.default_rng(seed)
    return policy.MLPPolicy.from_blocks(scale * rng.normal(size=(F, 2)), scale * rng.normal(size=2),
                                        scale * rng.normal(size=2), scale * rng.normal(size=1))


@pytest.mark.gpu
def test_T08_batch_invariance(monkeypatch):
    """T-08 (REQ-ENV-01, ENV-09): a (policy, seed) element run alone equals the same element at the head/middle/tail of a
    batch, in reversed order, and under chunking: adapters bitwise, MLP within 1e-12."""
    cuda_required()
    forbid_frontier(monkeypatch)        # batch invariance must hold for the new simulator itself, not for a delegate
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    pols = [policy.AdapterPolicy(d) for d in ({}, dict(dz=0.3), dict(b=0.1), dict(ue_mode=1, ue_band=0.1))] + [_rand_mlp(1), _rand_mlp(2)]
    seeds = [0, 1, 2, 3, 5000, 5001]
    fields = FIELDS
    full = env.simulate("M3C", 2000, cfg, pols, seeds, r=0.5, T_score=2000, dev="cuda")["final"]
    rev = env.simulate("M3C", 2000, cfg, pols[::-1], seeds[::-1], r=0.5, T_score=2000, dev="cuda")["final"]
    chk = env.simulate("M3C", 2000, cfg, pols, seeds, r=0.5, T_score=2000, dev="cuda", chunk=7)["final"]
    for i, p in enumerate(pols):
        for j, s in enumerate(seeds):
            alone = env.simulate("M3C", 2000, cfg, [p], [s], r=0.5, T_score=2000, dev="cuda")["final"]
            for name, other, oi, oj in (("full", full, i, j), ("reversed", rev, len(pols) - 1 - i, len(seeds) - 1 - j), ("chunk", chk, i, j)):
                for f in fields:
                    a, b = alone[f][0, 0].cpu(), other[f][oi, oj].cpu()
                    if isinstance(p, policy.AdapterPolicy):
                        assert torch.equal(a, b), f"{name} {f} pol{i} seed{s}"
                    else:
                        assert (a - b).abs().max().item() <= 1e-12, f"{name} {f} pol{i} seed{s}"


def _frontier_order_streams(seed, nblocks, K=3, blk=1024):
    """Independent restatement of frontier.simulate's draw ORDER for one seed: generator 0 (seed) first yields z0 = randn(1,K), and
    only then, per block of 1024 rounds, E = randn(blk,1,K)[:,0] followed by Ua = rand(blk,1)[:,0] (same generator);  generator 1
    (seed+1e6): Rn = rand(blk,1,K+2)[:,0];  generator 2 (seed+2e6): Rx = rand(blk,1,2)[:,0]."""
    g0, g1, g2 = (torch.Generator(device="cpu") for _ in range(3))
    for g, off in ((g0, 0), (g1, 10 ** 6), (g2, 2 * 10 ** 6)):
        g.manual_seed(seed + off)
    z0 = torch.randn(1, K, generator=g0, dtype=torch.float64)          # consumed BEFORE the first block
    blocks = []
    for _ in range(nblocks):
        E = torch.randn(blk, 1, K, generator=g0, dtype=torch.float64)[:, 0]
        Ua = torch.rand(blk, 1, generator=g0, dtype=torch.float64)[:, 0]
        Rn = torch.rand(blk, 1, K + 2, generator=g1, dtype=torch.float64)[:, 0]
        Rx = torch.rand(blk, 1, 2, generator=g2, dtype=torch.float64)[:, 0]
        blocks.append(dict(E=E, Ua=Ua, Rn=Rn, Rx=Rx))
    return z0, blocks


def test_T09_crn_shared_streams():
    """T-09 (REQ-ENV-02, OPT-03): generators seeded s, s+1e6, s+2e6; blk=1024; the draw order is frontier's ACTUAL order (z0 first from the
    first generator, then per block E, Ua | Rn | Rx); the type path simulated for 1100 rounds (crossing the block boundary at t=1025) equals
    the recurrence z_t = rho z_{t-1} + sqrt(1-rho^2) E_t driven by exactly those draws; same seed => identical type path for every policy;
    identical elements give identical results."""
    assert env.rng_block_size() == 1024
    s = 7
    z0, blocks = _frontier_order_streams(s, 2)
    got = env.draw_streams(s, 2, dev="cpu")
    for b in range(2):
        for key in ("E", "Ua", "Rn", "Rx"):
            assert torch.equal(got[b][key], blocks[b][key]), f"block {b} {key} (z0 must be consumed first, then E before Ua)"
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    pols = [_rand_mlp(1), policy.AdapterPolicy(dict(dz=0.3)), _rand_mlp(1), _rand_mlp(5)]
    seeds, T, r = [3, 4], 1100, 0.5
    out = env.simulate("M3C", T, cfg, pols, seeds, r=r, dev="cpu", diagnostic=True)
    rho = torch.tensor(r, dtype=torch.float64)
    sq = torch.sqrt(1 - rho ** 2)
    for j, sd in enumerate(seeds):
        z0j, bl = _frontier_order_streams(sd, 2)
        z = z0j[0, 1].clone()
        zs = []
        for t in range(1, T + 1):
            z = rho * z + sq * bl[(t - 1) // 1024]["E"][(t - 1) % 1024, 1]
            zs.append(z.clone())
        zs = torch.stack(zs)
        for i in range(len(pols)):
            assert torch.allclose(out["log"]["z1"][i, j], zs, rtol=0, atol=1e-13), f"policy {i} seed {sd}: type path != frontier draw order"
            assert torch.allclose(out["log"]["u1"][i, j], torch.special.ndtr(zs), rtol=0, atol=1e-13)
        for i in range(1, len(pols)):
            assert torch.equal(out["log"]["u1"][0, j], out["log"]["u1"][i, j]), "type path must not depend on the policy"
            assert torch.equal(out["log"]["z1"][0, j], out["log"]["z1"][i, j])
        for f in FIELDS:
            assert torch.equal(out["final"][f][0, j], out["final"][f][2, j]), f"identical elements differ in {f}"


def test_T10_scoring_window_no_forced_honesty():
    """T-10 (REQ-ENV-15, ENV-19, ENV-14): T=3000, L=300: T_score=T-L (G divided by T_score); replay T_score=T; exactly T rounds (no geometric
    stopping); T_train in {20000,40000,100000}; no forced tail honesty (v1=clip(u1+d) for t>T-2L).  DEFAULT CALL (no T_score): the snapshot
    must equal, bitwise, the final accumulators of a full 2700-round simulation, and must differ from the 3000-round final (an
    implementation that scores the last L rounds fails)."""
    assert env.score_window(3000, 300) == 2700 and env.score_window(3000, 300, replay=True) == 3000
    for ok in (20000, 40000, 100000):
        assert env.validate_T_train(ok) == ok
    for bad in (3000, 30000, 100001):
        with pytest.raises(ValueError):
            env.validate_T_train(bad)
    d = 0.2
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    const = policy.MLPPolicy.from_blocks(np.zeros((25, 2)), np.zeros(2), np.zeros(2), np.array([math.atanh(d)]))
    pols = [policy.AdapterPolicy({}), const]
    out = env.simulate("M3C", 3000, cfg, pols, [0, 1], r=0.5, dev="cpu", diagnostic=True)
    assert out["T_score"] == 2700
    assert out["log"]["u1"].shape[-1] == 3000
    u1, v1 = out["log"]["u1"][1], out["log"]["v1"][1]
    expect = torch.clamp(u1 + torch.tanh(torch.tensor(math.atanh(d), dtype=torch.float64)), 0, 1)
    assert torch.allclose(v1, expect, rtol=0, atol=1e-15)
    tail = slice(2400, 3000)
    m = u1[:, tail] < 0.75
    assert m.any() and (v1[:, tail][m] != u1[:, tail][m]).all(), "no forced honesty in the tail"
    snap = out["snap"]["util_b"][:, :, 1]
    Gs = metrics.paired_gain(snap[1], snap[0], out["T_score"])
    assert torch.allclose(torch.as_tensor(Gs) * 2700, snap[1] - snap[0], rtol=0, atol=1e-12)
    # default-mode snapshot == full 2700-round simulation; the last L rounds are NOT scored
    full2700 = env.simulate("M3C", 2700, cfg, pols, [0, 1], r=0.5, T_score=2700, dev="cpu")["final"]
    for f in FIELDS + ["nraid"]:
        assert torch.equal(out["snap"][f], full2700[f]), f"default snapshot field {f} != 2700-round final"
    for f in ("util_b", "reg_b"):
        assert not torch.equal(out["snap"][f], out["final"][f]), f"{f}: snapshot equals the 3000-round final (last L rounds scored)"
    assert (out["final"]["util_b"][1, :, 1] > out["snap"]["util_b"][1, :, 1]).all(), "the last 300 rounds must add utility to the final"
    rep = env.simulate("M3C", 3000, cfg, pols, [0, 1], r=0.5, T_score=3000, dev="cpu")
    assert rep["T_score"] == 3000
    for f in FIELDS + ["nraid"]:
        assert torch.equal(rep["snap"][f], rep["final"][f]), f"replay mode: snapshot must be the final ({f})"
    for f in FIELDS + ["nraid"]:
        assert torch.equal(rep["final"][f], out["final"][f]), f"scoring mode must not change the simulation ({f})"


@pytest.mark.parametrize("mech,cfg", [("M3C", dict(p=0.1, tol=6.0, L=300.0, kref=0.5)),
                                      ("M4", dict(eps=0.05, L=300.0, rw=1 / 3))])
def test_T11_scoring_snapshot_causal(mech, cfg):
    """T-11 (REQ-ENV-08, ENV-15): the snapshot at t=T_score of a T-round run equals, bitwise, the final accumulators of a run of
    exactly T_score rounds (the simulation never looks ahead)."""
    pols = [_rand_mlp(3, scale=1.0), policy.AdapterPolicy(dict(dz=0.3))]
    seeds = [0, 1, 2]
    long = env.simulate(mech, 3000, cfg, pols, seeds, r=0.5, T_score=2700, dev="cpu")
    short = env.simulate(mech, 2700, cfg, pols, seeds, r=0.5, T_score=2700, dev="cpu")
    for f in FIELDS + ["nraid"]:
        assert torch.equal(long["snap"][f], short["final"][f]), f
    default = env.simulate(mech, 3000, cfg, pols, seeds, r=0.5, dev="cpu")           # T_score defaults to T-L = 2700
    assert default["T_score"] == 2700
    for f in FIELDS + ["nraid"]:
        assert torch.equal(default["snap"][f], short["final"][f]), f"default T_score: {f}"
