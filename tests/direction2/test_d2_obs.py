"""T-12..T-19, T-56: observation information sets and policy class (spec s14.2).

This file carries its OWN reference implementation of the public features f01-f25 (`ref_features`, restated from spec s4.4 with
scipy/numpy only; it imports nothing from arbitration.rl).  Hand-built sequences (pauses, wins, losses, clip cases) are compared
to the implementation with relative tolerance 1e-9.

Conventions fixed by the tests (see docs/direction2-tdd-map.md, "約定"):
  * public log series are indexed by round t=1..T at position t-1; susp1[t-1] is susp_1 AT OBSERVATION TIME of round t (= the value at the
    end of round t-1, the one used for eligibility `t > susp_1`);
  * "since" (f05): t_s is the round at which the agent first OBSERVES the rise of its own susp (spec 3.1: "由 susp[:,1] 上升事件記錄"), i.e. the round
    after the one whose settlement/raid flagged it; at that round since = 0.  (The spec text "通知輪" is ambiguous between this and the flagging
    round (since = 1/(2L)); the tests fix the first reading - reported to the commander.)
"""
import ast
import copy
import dataclasses
import inspect
import math
import os
import types

import numpy as np
import pytest
import torch
from scipy.special import ndtr as _ndtr, ndtri as _ndtri

from conftest import ROOT, phi1
from arbitration.rl import env, obs, obs_reconstruct, policy

PRIVATE = ("cus", "pend_a", "pend_v", "flags", "audits", "sxy", "sxx")
OTHERS = ("u0", "u2", "v0", "v2", "susp0", "susp2", "elig0", "elig2", "u_others", "v_others")
ALLOWED = {"z", "u", "susp_minus_t", "since_rounds", "won_hist", "d_hist", "v_prev"}
CFG = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
# spec s14.2 T-56 / REQ-OBS-10 (a): the ONLY things a public log may contain
WL_SERIES = {"z1", "u1", "v1", "won1", "susp1"}
WL_CONSTS = {"p", "h", "L", "kref", "r", "C_MIN"}
WL_CONSTS_M4 = WL_CONSTS | {"eps", "rw"}      # M4 mechanism constants (public, Kerckhoffs); spec 3/OBS-10 name only the M3C set -> flagged to the commander
LAM_E = 0.99


# ------------------------------------------------------------------------------------------ independent reference (spec s4.4)
def ref_res(v_prev, u_now, r):
    """res = (z_w - r z_v)/sqrt(1-r^2), z_v = ndtri(phi_of(v_{t-1}, agent 1)), z_w = ndtri(phi_of(u_t, agent 1)); agent-1 marginal =
    identity, clipped to [1e-12, 1-1e-12] as audit_fix.phi_of does."""
    zv = _ndtri(min(max(v_prev, 1e-12), 1 - 1e-12))
    zw = _ndtri(min(max(u_now, 1e-12), 1 - 1e-12))
    return (zw - r * zv) / math.sqrt(1 - r * r)


def ref_features(z1, u1, v1, won1, susp1, r, L):
    """Spec s4.4 f01..f25 for rounds t=1..T from the public log of ONE element -> array [T, 25]."""
    z1, u1, v1, won1, susp1 = (np.asarray(a, dtype=np.float64) for a in (z1, u1, v1, won1, susp1))
    T = len(u1)
    out = np.zeros((T, 25))
    ema_res = ema_d = ema_win = 0.0
    t_s, prev = None, None
    for t in range(1, T + 1):
        i = t - 1
        if prev is not None and susp1[i] > prev:        # a rise of susp seen at round t <=> flagged in round t-1; t_s = round of observation
            t_s = t
        prev = susp1[i]
        row = np.zeros(25)
        row[0] = min(max(z1[i] / 3.0, -1.0), 1.0)
        row[1] = 2.0 * u1[i] - 1.0
        row[2] = 1.0 if t > susp1[i] else 0.0
        row[3] = max(susp1[i] - t, 0.0) / L
        row[4] = 1.0 if t_s is None else min(t - t_s, 2 * L) / (2.0 * L)
        for j in range(1, 9):
            if t - j >= 1:
                row[5 + j - 1] = won1[i - j]
                row[13 + j - 1] = v1[i - j] - u1[i - j]
        f22 = 0.0
        if t >= 2 and won1[i - 1] == 1.0:
            f22 = min(max(ref_res(v1[i - 1], u1[i], r) / 6.0, -1.0), 1.0)
        d_last = (v1[i - 1] - u1[i - 1]) if t >= 2 else 0.0
        w_last = won1[i - 1] if t >= 2 else 0.0
        ema_res = LAM_E * ema_res + (1 - LAM_E) * f22
        ema_d = LAM_E * ema_d + (1 - LAM_E) * d_last
        ema_win = LAM_E * ema_win + (1 - LAM_E) * w_last
        row[21], row[22], row[23], row[24] = f22, ema_res, ema_d, ema_win
        out[i] = row
    return out


def _hand_log(T=60):
    """Hand-constructed public log: pauses (flag in rounds 20 and 44, L=10: susp 30 then 54), wins, losses, over/under reports, |z|>3
    (clip), and two saturated residuals (round 20: u~0 after a v=1 win; round 34: u~1 after a v~0 win)."""
    t = np.arange(1, T + 1)
    z = 1.4 * np.sin(0.7 * t) + 0.8 * np.cos(1.9 * t)
    z[3], z[7] = 3.6, -4.2
    u = _ndtr(z)
    v = np.clip(u + 0.12 * np.sin(1.3 * t), 0.0, 1.0)
    won = np.zeros(T)
    for rr in (2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 33, 34, 35, 38, 39, 40, 42, 43, 44, 56, 57, 58):
        won[rr - 1] = 1.0
    susp = np.zeros(T)
    susp[20:] = 30.0
    susp[44:] = 54.0
    v[18] = 1.0                                          # round 19 won with v=1 ...
    u[19], z[19] = 1e-6, _ndtri(1e-6)                    # ... round 20: res ~ -9.6 -> f22 = -1
    v[32] = 1e-9                                         # round 33 won with v~0 ...
    u[33], z[33] = 0.99999, _ndtri(0.99999)              # ... round 34: res ~ +8.4 -> f22 = +1
    v[41], u[42], z[42] = 0.6, 0.62, _ndtri(0.62)        # round 42 won, round 43 moderate (unsaturated)
    return dict(z1=z, u1=u, v1=v, won1=won, susp1=susp), [20, 44]


def _drive_public_builder(log, r, L, flag_rounds):
    """Feed the implementation's ObsBuilder with the whitelisted per-round relative state built by hand from the log."""
    T = len(log["u1"])
    b = obs.ObsBuilder("public", r=r, L=L, shape=(1,))
    rows = []
    for t in range(1, T + 1):
        i = t - 1
        won_h = [log["won1"][i - j] if i - j >= 0 else 0.0 for j in range(1, 9)]
        d_h = [(log["v1"][i - j] - log["u1"][i - j]) if i - j >= 0 else 0.0 for j in range(1, 9)]
        past = [f + 1 for f in flag_rounds if f + 1 <= t]           # first observation of the rise = the round after the flag
        since = float("inf") if not past else float(t - max(past))
        mk = lambda x: torch.tensor([x], dtype=torch.float64)
        st = obs.PublicState(z=mk(log["z1"][i]), u=mk(log["u1"][i]), susp_minus_t=mk(log["susp1"][i] - t), since_rounds=mk(since),
                             won_hist=torch.tensor([won_h], dtype=torch.float64), d_hist=torch.tensor([d_h], dtype=torch.float64),
                             v_prev=mk(log["v1"][i - 1] if i >= 1 else 0.0))
        rows.append(b.step(st)[0].numpy())
    return np.array(rows)


def _assert_rel(a, b, what):
    """relative 1e-9 (|a-b| <= 1e-9|b| + 1e-13)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    bad = np.abs(a - b) > 1e-9 * np.abs(b) + 1e-13
    assert not bad.any(), f"{what}: {int(bad.sum())} entries differ; first at {tuple(np.argwhere(bad)[0])}: {a[bad][0]} vs {b[bad][0]}"


def _ref_from_pub(pub, p, s):
    c = pub["consts"]
    return ref_features(*(pub[k][p, s].numpy() for k in ("z1", "u1", "v1", "won1", "susp1")), r=float(c["r"]), L=float(c["L"]))


def _rand_mlp(seed, F=25, scale=0.8, **kw):
    rng = np.random.default_rng(seed)
    return policy.MLPPolicy.from_blocks(scale * rng.normal(size=(F, 2)), scale * rng.normal(size=2),
                                        scale * rng.normal(size=2), scale * rng.normal(size=1), **kw)


def _const_policy(F, b2, version="public", seed=0):
    """MLP whose output is the constant d=tanh(b2) whatever the observation (W2=0): the trajectory cannot depend on the info set."""
    rng = np.random.default_rng(seed)
    return policy.MLPPolicy.from_blocks(rng.normal(size=(F, 2)), rng.normal(size=2), np.zeros(2), np.array([b2]), obs_version=version)


def _public_state(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    return obs.PublicState(z=r(B) * 2 - 1, u=r(B), susp_minus_t=-torch.ones(B, dtype=torch.float64), since_rounds=torch.full((B,), float("inf"), dtype=torch.float64),
                           won_hist=(r(B, 8) > 0.5).double(), d_hist=r(B, 8) * 0.2, v_prev=r(B))


# ------------------------------------------------------------------------------------------------------------------------ T-12
def test_T12_obs_whitelist_public():
    """T-12 (REQ-OBS-01/02/06): public builder takes only whitelisted fields (no other agents' susp); objects carrying cus/pend_a/
    pend_v/flags/audits/sxy/sxx/others' u,v are rejected - also when the state class is a frozen dataclass or a duck-typed object."""
    fields = set(obs.public_state_fields())
    assert fields == ALLOWED
    assert not fields & set(PRIVATE) and not fields & set(OTHERS)
    b = obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,))
    st = _public_state()
    assert b.step(st).shape == (4, 25)
    base = {f: getattr(st, f) for f in ALLOWED}
    for bad in PRIVATE + OTHERS:
        class Tainted(type(st)):                                   # plain subclass: has __dict__ even if the parent is slotted/frozen
            pass
        t = Tainted(**base)
        object.__setattr__(t, bad, torch.zeros(4, dtype=torch.float64))   # bypasses frozen-dataclass __setattr__
        assert bad in vars(t)
        with pytest.raises(TypeError):
            obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,)).step(t)
        duck = types.SimpleNamespace(**base, **{bad: torch.zeros(4, dtype=torch.float64)})
        with pytest.raises(TypeError):
            obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,)).step(duck)
        with pytest.raises(TypeError):
            obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,)).step(dict(base, **{bad: torch.zeros(4, dtype=torch.float64)}))
    # the whitelist object itself must not be able to grow private attributes silently via a dataclass field
    assert {f.name for f in dataclasses.fields(obs.PublicState)} == ALLOWED


def _rolled_state(version="public", n=60):
    st = env.init_state("M3C", CFG, 0.5, [0, 1, 2, 3], dev="cpu", obs_version=version)
    pol = _rand_mlp(1, F=25 if version == "public" else 27)
    for _ in range(n):
        env.step_state(st, pol)
    return st


# ------------------------------------------------------------------------------------------------------------------------ T-13
def test_T13_obs_invariant_to_private_state():
    """T-13 (REQ-OBS-01/03/06): replacing private state and others' u,v with random values leaves the 25 public features bitwise
    unchanged; announced f26 (S=cus/h) and f27 (audited_last) change with cus / pend_a."""
    st = _rolled_state("announced")
    base_pub = env.observe(st, "public")
    A, B = copy.deepcopy(st), copy.deepcopy(st)
    g = torch.Generator().manual_seed(5)
    for s_, cus_val, pa in ((A, 0.0, 1), (B, 3.0, -1)):
        for n in ("sxy", "sxx", "audits", "pend_v"):
            setattr(s_, n, torch.rand(getattr(s_, n).shape, generator=g, dtype=torch.float64))
        s_.flags = torch.randint(0, 5, s_.flags.shape, generator=g).to(s_.flags.dtype)
        s_.cus[:, 1] = cus_val
        s_.pend_a[:] = pa
        for k in (0, 2):
            s_.u[:, k] = torch.rand(s_.u[:, k].shape, generator=g, dtype=torch.float64)
            s_.v[:, k] = torch.rand(s_.v[:, k].shape, generator=g, dtype=torch.float64)
    pa_, pb_ = env.observe(A, "public"), env.observe(B, "public")
    assert torch.equal(pa_, base_pub) and torch.equal(pb_, base_pub)
    na, nb = env.observe(A, "announced"), env.observe(B, "announced")
    assert torch.equal(na[:, :25], base_pub) and torch.equal(nb[:, :25], base_pub)
    assert not torch.equal(na[:, 25], nb[:, 25]), "f26 (S) must follow cus"
    assert not torch.equal(na[:, 26], nb[:, 26]), "f27 (audited_last) must follow pend_a"


def test_T13b_obs_history_invariant_to_injected_private_noise():
    """T-13b (REQ-OBS-01/06, accumulated-history leak): over 300 rounds, before EVERY step the other agents' types, their CUSUM/flag/audit
    counters and previous u,v and the pair-statistics sxy/sxx are overwritten with random numbers.  A policy that always reports 1.0 wins
    whenever eligible, and agent 1's own public trajectory (types, reports, wins, suspensions) does not depend on the others, so the whole
    25-dim observation HISTORY (incl. the EMA features, which accumulate) must stay bitwise identical to the unperturbed run.  Catches a
    leak that only shows up through accumulated quantities (e.g. an EMA fed with others' u or with the audit pipeline)."""
    liar = policy.AdapterPolicy(dict(b=1.0))
    seeds = [0, 1, 2, 3]

    def roll(perturb):
        st = env.init_state("M3C", CFG, 0.5, seeds, dev="cpu")
        g = torch.Generator().manual_seed(11)
        os_, wins, flags = [], [], []
        for _ in range(300):
            if perturb:
                for k in (0, 2):
                    st.z[:, k] = torch.randn(st.z[:, k].shape, generator=g, dtype=torch.float64)
                    st.cus[:, k] = 5.0 * torch.rand(st.cus[:, k].shape, generator=g, dtype=torch.float64)
                    st.flags[:, k] = torch.randint(0, 5, st.flags[:, k].shape, generator=g).to(st.flags.dtype)
                    st.u[:, k] = torch.rand(st.u[:, k].shape, generator=g, dtype=torch.float64)
                    st.v[:, k] = torch.rand(st.v[:, k].shape, generator=g, dtype=torch.float64)
                st.sxy = torch.rand(st.sxy.shape, generator=g, dtype=torch.float64)
                st.sxx = torch.rand(st.sxx.shape, generator=g, dtype=torch.float64)
                st.audits = torch.rand(st.audits.shape, generator=g, dtype=torch.float64)
            info = env.step_state(st, liar)
            os_.append(env.observe(st, "public").clone())
            wins.append(info["won1"].clone())
            flags.append(st.flags[:, 1].clone())
        return torch.stack(os_), torch.stack(wins), torch.stack(flags)

    oa, wa, fa = roll(False)
    ob, wb, fb = roll(True)
    assert torch.equal(wa, wb) and torch.equal(fa, fb), "agent 1's public trajectory must not depend on the others"
    assert (fa[-1] > 0).all() and wa.any() and (~wa).any(), "scenario must contain wins, losses and suspensions"
    assert torch.equal(oa, ob), "observation history leaked private / other-agent state"


# ------------------------------------------------------------------------------------------------------------------------ T-14
def test_T14_obs_no_absolute_time():
    """T-14 (REQ-ENV-14, OBS-08): no t/time/round/step in the builder signature or state fields; equal relative states give equal
    features regardless of absolute time."""
    banned = ("t", "time", "round", "step", "tt", "t_abs")
    for cls in (obs.PublicState, obs.AnnouncedState, obs.OracleState):
        names = {f for f in cls.__dataclass_fields__}
        assert not names & set(banned), cls.__name__
    assert set(obs.public_state_fields()).isdisjoint(banned)
    params = set(inspect.signature(obs.ObsBuilder.step).parameters) | set(inspect.signature(obs.ObsBuilder.__init__).parameters)
    assert params.isdisjoint(banned)
    st = _public_state()
    o1 = obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,)).step(st)
    o2 = obs.ObsBuilder("public", r=0.5, L=2500.0, shape=(4,)).step(copy.deepcopy(st))
    assert torch.equal(o1, o2)


# ------------------------------------------------------------------------------------------------------------------------ T-15
@pytest.mark.parametrize("r", [0.5, 0.9])
def test_T15_obs_feature_table(r):
    """T-15 (REQ-OBS-07): F=25, names/order per s4.4, k=8; every feature f01-f25 equals the test-local reference implementation (relative
    1e-9) on a hand-built sequence with pauses / wins / losses / clipping, and on a random rollout; literal spot values; features in
    [-1,1]; zero padding of the history."""
    names = obs.feature_names("public")
    exp = ["z", "u", "elig", "rem", "since"] + [f"won_{j}" for j in range(1, 9)] + [f"d_{j}" for j in range(1, 9)] + \
          ["res_last", "ema_res", "ema_d", "ema_win"]
    assert names == exp and len(names) == 25
    # ---- hand-built sequence: implementation (fed with hand-derived relative state) vs independent reference, f01..f25
    log, flag_rounds = _hand_log(60)
    L = 10.0
    got = _drive_public_builder(log, r, L, flag_rounds)
    ref = ref_features(log["z1"], log["u1"], log["v1"], log["won1"], log["susp1"], r, L)
    assert got.shape == (60, 25)
    for k in range(25):
        _assert_rel(got[:, k], ref[:, k], f"f{k + 1:02d} ({names[k]}) r={r}")
    assert (np.abs(got) <= 1.0 + 1e-15).all()
    # ---- literal spot values (hand-computed from s4.4, independent of both implementations)
    row1 = ref[0]
    assert row1[0] == pytest.approx(min(max(log["z1"][0] / 3, -1), 1), abs=1e-15) and row1[2] == 1.0 and row1[3] == 0.0 and row1[4] == 1.0
    assert not row1[5:21].any() and not row1[21:].any(), "t=1: all history features and EMAs are zero-padded"
    assert ref[3, 0] == 1.0 and ref[7, 0] == -1.0                           # |z|>3 clipped (rounds 4, 8)
    assert ref[20, 2] == 0.0 and ref[20, 3] == pytest.approx(0.9, abs=1e-15) and ref[20, 4] == 0.0     # t=21: first observation of the flag (round 20)
    assert ref[29, 2] == 0.0 and ref[29, 3] == 0.0 and ref[30, 2] == 1.0 and ref[30, 4] == pytest.approx(10 / 20, abs=1e-15)  # t=30 (susp=30: still out), t=31
    assert ref[40, 4] == 1.0 and ref[41, 4] == 1.0                            # t=41: 20/20; t=42: min(21, 2L=20)/2L saturates
    assert ref[44, 3] == pytest.approx(0.9, abs=1e-15) and ref[44, 4] == 0.0  # t=45 first observation of the second flag (round 44)
    assert ref[2, 24] == pytest.approx(0.01, abs=1e-15) and ref[3, 24] == pytest.approx(0.99 * 0.01 + 0.01, abs=1e-15)   # won at rounds 2,3 -> ema_win
    assert ref[19, 21] == -1.0 and ref[33, 21] == 1.0                           # saturated residuals (round 20: -, round 34: +)
    assert ref[42, 21] == pytest.approx(ref_res(0.6, 0.62, r) / 6, abs=1e-15) and 0 < abs(ref[42, 21]) < 1
    # ---- random rollout of the whole simulator: obs == reference built from the public log
    out = env.simulate("M3C", 400, CFG, [_rand_mlp(2, scale=1.5)], [0, 1], r=r, dev="cpu", diagnostic=True)
    o = out["obs"]
    assert o.shape[-1] == 25 and o.shape[-2] == 400
    assert torch.isfinite(o).all() and (o.abs() <= 1.0).all()
    for t in range(1, 9):                       # round t (1-based): lags j >= t reach before round 1 -> zero padded
        for j in range(t, 9):
            assert (o[..., t - 1, 5 + (j - 1)] == 0).all(), f"won_{j} at t={t}"
            assert (o[..., t - 1, 13 + (j - 1)] == 0).all(), f"d_{j} at t={t}"
    pub = env.extract_public_log(out)
    for s_ in range(2):
        _assert_rel(o[0, s_].numpy(), _ref_from_pub(pub, 0, s_), f"rollout seed {s_}")


# ------------------------------------------------------------------------------------------------------------------------ T-16
def test_T16_obs_own_win_semantics():
    """T-16 (REQ-OBS-01): won=1 only for win_eff & idx==1: a raided winning round counts 0; a random allocation with nobody
    eligible counts 0; an unraided win counts 1.  M4 with rw=0: the raid wastes the item whenever an eligible NON-winner exists (the winner is
    never the target, but the round is lost: won=0); with no eligible non-winner there is no raid (won=1); with rw=1 the winner is raided
    even when it is the only eligible agent (won=0)."""
    liar = policy.AdapterPolicy(dict(b=1.0))
    st = env.init_state("M3C", dict(p=0.0, tol=6.0, L=300.0, kref=0.5), 0.5, [0, 1], dev="cpu")
    wins = [env.step_state(st, liar)["won1"] for _ in range(50)]
    assert torch.stack(wins).all(), "unraided win must count"
    st = env.init_state("M4", dict(eps=1.0, L=300.0, rw=1.0), 0.5, [0, 1], dev="cpu")
    info = [env.step_state(st, liar) for _ in range(50)]
    assert not torch.stack([i["won1"] for i in info]).any(), "raided winning rounds must count 0"
    st = env.init_state("M3C", CFG, 0.5, [0, 1, 2, 3], dev="cpu")
    st.susp[:] = 10 ** 9
    info = [env.step_state(st, liar) for _ in range(50)]
    assert not torch.stack([i["won1"] for i in info]).any(), "random allocation (nobody eligible) must count 0"
    # ---- M4 rw=0
    honest = policy.AdapterPolicy({})
    st = env.init_state("M4", dict(eps=1.0, L=300.0, rw=0.0), 0.5, [0, 1], dev="cpu")
    info = [env.step_state(st, honest) for _ in range(50)]
    assert not torch.stack([i["won1"] for i in info]).any(), "rw=0, others eligible: every round is raided (item wasted), won must be 0"
    assert (st.nraid == 50).all() and (st.flags == 0).all()
    st = env.init_state("M4", dict(eps=1.0, L=300.0, rw=0.0), 0.5, [0, 1], dev="cpu")
    st.susp[:, 0] = 10 ** 9
    st.susp[:, 2] = 10 ** 9
    info = [env.step_state(st, honest) for _ in range(50)]
    assert torch.stack([i["won1"] for i in info]).all(), "rw=0 and no eligible non-winner: no raid, the win counts"
    assert (st.nraid == 0).all()
    st = env.init_state("M4", dict(eps=1.0, L=300.0, rw=1.0), 0.5, [0, 1], dev="cpu")
    st.susp[:, 0] = 10 ** 9
    st.susp[:, 2] = 10 ** 9
    info = [env.step_state(st, honest) for _ in range(50)]
    assert not torch.stack([i["won1"] for i in info]).any(), "rw=1: the winner is raided even if it is the only eligible agent"
    assert (st.nraid == 50).all()


# ------------------------------------------------------------------------------------------------------------------------ T-17
RES_CASES = [(0.6, 0.62), (0.3, 0.7), (0.9, 0.2), (0.05, 0.5), (0.5, 0.01), (0.97, 0.5),          # unsaturated at r=0.5 / 0.9
             (1.0, 1e-6), (1.0, 1e-3), (0.999999, 0.001), (1e-9, 0.99999), (0.001, 0.9999), (0.0, 1.0)]      # |res| >= 6 at some r


@pytest.mark.parametrize("r", [0.3, 0.5, 0.9])
def test_T17_res_last_matches_mechanism_resid(r):
    """T-17 (REQ-OBS-01/07): f22 = clip(res/6, -1, 1) with res = (z_w - r z_v)/sqrt(1-r^2) from an independently copied formula (no
    frontier, no instrumentation): (a) hand-built pairs including |res| >= 6 truncation (+-1 exactly) and won_{t-1}=0 (f22 = 0); (b) the whole
    simulator on a rollout, saturated entries included; (c) the mechanism's own `resid` (step_state, p=1 so every win is audited) equals
    the same formula for agent 1 (tolerance 1e-12)."""
    # (a) hand-built pairs through the implementation's builder
    n_pos = n_neg = 0
    for v_prev, u_now in RES_CASES:
        res = ref_res(v_prev, u_now, r)
        for won in (1.0, 0.0):
            wh = torch.zeros(1, 8, dtype=torch.float64)
            wh[0, 0] = won
            mk = lambda x: torch.tensor([x], dtype=torch.float64)
            st = obs.PublicState(z=mk(0.1), u=mk(u_now), susp_minus_t=mk(-1.0), since_rounds=mk(float("inf")), won_hist=wh,
                                 d_hist=torch.zeros(1, 8, dtype=torch.float64), v_prev=mk(v_prev))
            f22 = float(obs.ObsBuilder("public", r=r, L=2500.0, shape=(1,)).step(st)[0, 21])
            if won == 0.0:
                assert f22 == 0.0, (v_prev, u_now)
            else:
                assert abs(f22 - min(max(res / 6.0, -1.0), 1.0)) <= 1e-12, (v_prev, u_now, r, res, f22)
                if abs(res) >= 6:
                    assert f22 == math.copysign(1.0, res), "truncation at |res|>=6 must give exactly +-1"
                    n_pos += res > 0
                    n_neg += res < 0
    assert n_pos >= 1 and n_neg >= 1, f"r={r}: the case list must contain saturated residuals of both signs"
    # (b) whole simulator
    out = env.simulate("M3C", 600, CFG, [_rand_mlp(4, scale=1.5)], [0, 1, 2], r=r, dev="cpu", diagnostic=True)
    lg, o = out["log"], out["obs"][0]
    u1, v1, won = lg["u1"][0], lg["v1"][0], lg["won1"][0]
    zv = torch.special.ndtri(phi1(v1[:, :-1]))
    zw = torch.special.ndtri(phi1(u1[:, 1:]))
    resid = (zw - r * zv) / np.sqrt(1 - r ** 2)
    f22 = o[:, 1:, 21]
    w = won[:, :-1].bool()
    assert w.any()
    assert torch.allclose(f22[w], torch.clamp(resid[w] / 6, -1, 1), rtol=0, atol=1e-12)
    assert (f22[~w] == 0).all()
    # (c) mechanism consistency: the audit residual computed by the environment for agent 1 == the quantity behind f22
    cfgp = dict(p=1.0, tol=1e9, L=300.0, kref=0.5)
    st = env.init_state("M3C", cfgp, r, [0, 1, 2, 3], dev="cpu")
    pol = policy.AdapterPolicy(dict(b=0.3))
    prev_v1 = prev_won = None
    checked = 0
    for _ in range(300):
        info = env.step_state(st, pol)
        if prev_won is not None:
            m = prev_won.bool()
            assert info["has"][m].all(), "p=1: every unraided win must be audited"
            u_now = torch.special.ndtr(st.z[:, 1]).numpy()
            exp = np.array([ref_res(float(v), float(u), r) for v, u in zip(prev_v1.numpy(), u_now)])
            got = info["resid"].numpy()
            assert np.abs(got[m.numpy()] - exp[m.numpy()]).max(initial=0.0) <= 1e-12
            checked += int(m.sum())
        prev_v1, prev_won = info["v"][:, 1].clone(), info["won1"].clone()
    assert checked > 20


# ------------------------------------------------------------------------------------------------------------------------ T-18
def test_T18_info_sets_nested():
    """T-18 (REQ-OBS-03/04/05): with observation-independent (constant-output) policies the three info sets produce the same trajectory, so
    the first 25 dims of the announced / oracle observations equal the public ones bitwise AND the reference implementation;
    oracle f26',f27' = 2u0-1, 2u2-1; announced f26 = S = cus_1/h, f27 = audited_last (= won_{t-1} when p=1, <= won_{t-1} always, 0 when p=0)."""
    kw = dict(r=0.5, dev="cpu", diagnostic=True)
    seeds = [0, 1]
    pol = lambda F, v: _const_policy(F, 0.9, v, seed=6)
    pub = env.simulate("M3C", 300, CFG, [pol(25, "public")], seeds, obs_version="public", **kw)
    ann = env.simulate("M3C", 300, CFG, [pol(27, "announced")], seeds, obs_version="announced", **kw)
    ora = env.simulate("M3C", 300, CFG, [pol(27, "oracle")], seeds, obs_version="oracle", **kw)
    for a in (ann, ora):
        assert a["obs"].shape[-1] == 27
        assert torch.equal(a["log"]["v1"], pub["log"]["v1"]) and torch.equal(a["log"]["won1"], pub["log"]["won1"]), "trajectories must coincide"
        assert torch.equal(a["obs"][..., :25], pub["obs"]), "first 25 dims must equal the public obs bitwise"
    for s_ in range(2):
        _assert_rel(ann["obs"][0, s_, :, :25].numpy(), _ref_from_pub(env.extract_public_log(ann), 0, s_), "announced first 25 vs reference")
        _assert_rel(ora["obs"][0, s_, :, :25].numpy(), _ref_from_pub(env.extract_public_log(ora), 0, s_), "oracle first 25 vs reference")
    pl, pr = ann["private_log"], ora["private_log"]
    assert torch.allclose(ann["obs"][..., 25], pl["cus1"] / CFG["tol"], rtol=0, atol=1e-15)
    assert torch.equal(ann["obs"][..., 26], pl["audited_last"].to(ann["obs"].dtype))
    assert torch.allclose(ora["obs"][..., 25], 2 * pr["u0"] - 1, rtol=0, atol=1e-15)
    assert torch.allclose(ora["obs"][..., 26], 2 * pr["u2"] - 1, rtol=0, atol=1e-15)
    assert (pl["cus1"] > 0).any() and ((pr["u0"] - pr["u2"]).abs() > 0).any()
    won = ann["log"]["won1"].to(torch.float64)
    al = pl["audited_last"].to(torch.float64)
    assert (al[..., 0] == 0).all() and (al[..., 1:] <= won[..., :-1]).all() and (al[..., 1:] < won[..., :-1]).any()
    cfg1 = dict(p=1.0, tol=6.0, L=300.0, kref=0.5)
    a1 = env.simulate("M3C", 300, cfg1, [pol(27, "announced")], seeds, obs_version="announced", **kw)
    w1 = a1["log"]["won1"].to(torch.float64)
    assert (a1["private_log"]["audited_last"].to(torch.float64)[..., 1:] == w1[..., :-1]).all() and w1.any()
    a0 = env.simulate("M3C", 300, dict(CFG, p=0.0), [pol(27, "announced")], seeds, obs_version="announced", **kw)
    assert not a0["private_log"]["audited_last"].any()
    st = _rolled_state("announced")
    assert torch.equal(env.observe(st, "announced")[:, :25], env.observe(st, "public"))
    assert torch.equal(env.observe(st, "oracle")[:, :25], env.observe(st, "public"))


# ------------------------------------------------------------------------------------------------------------------------ T-19
def test_T19_policy_param_cap_and_action_range():
    """T-19 (REQ-ACT-01/02/03): F=25 -> 55 params, F=27 -> 59, <=64; >64 rejected; diagnostic F=134 -> 273 <=300 (rejected when
    not diagnostic); v1 in [0,1]; d=0 => v1==u1."""
    assert policy.n_params(25) == 55 and policy.n_params(27) == 59 and policy.n_params(134) == 273
    assert policy.param_limit() == 64 and policy.param_limit(diagnostic=True) == 300
    assert policy.MLPPolicy(F=25).n_params == 55 and policy.MLPPolicy(F=27).n_params == 59
    with pytest.raises(ValueError):
        policy.MLPPolicy(F=40)                                  # 85 params
    assert policy.MLPPolicy(F=134, diagnostic=True).n_params == 273
    with pytest.raises(ValueError):
        policy.MLPPolicy(F=134)
    with pytest.raises(ValueError):
        policy.MLPPolicy(F=200, diagnostic=True)                # 405 > 300
    g = torch.Generator().manual_seed(0)
    u1 = torch.rand(10000, generator=g, dtype=torch.float64)
    d = torch.tanh(torch.randn(10000, generator=g, dtype=torch.float64) * 3)
    v = policy.report(u1, d)
    assert (v >= 0).all() and (v <= 1).all()
    assert torch.equal(policy.report(u1, torch.zeros_like(u1)), u1)


# ------------------------------------------------------------------------------------------------------------------------ T-56
def test_T56_e2e_info_isolation_from_logs():
    """T-56 (REQ-OBS-10, OBS-01/06): diagnostic run (T=3000, with suspensions and raids); (1) the public log has exactly the whitelisted keys
    {z1,u1,v1,won1,susp1} + constants {p,h,L,kref,r,C_MIN} and nothing private; (2) the test-local reference implementation of s4.4 reading ONLY
    that log reproduces the strategy's obs (relative 1e-9) and the independent reconstructor reproduces it bitwise (==); (3) private
    perturbations do not change the public log or the reconstruction; (4) announced extra dims rebuild from their extra inputs; (5) the
    reconstructor file does not import the simulator/feature builder."""
    cfg = dict(eps=0.1, L=300.0, rw=1 / 3)
    pol = _rand_mlp(9, scale=1.5)
    out = env.simulate("M4", 3000, cfg, [pol], [0, 1], r=0.5, dev="cpu", diagnostic=True)
    assert (out["final"]["flags"] > 0).any() and (out["final"]["nraid"] > 0).any()
    pub = env.extract_public_log(out)
    # (1) whitelist, written down from the spec
    series = set(pub) - {"consts"}
    assert series == WL_SERIES, f"public log keys {sorted(series)} != whitelist {sorted(WL_SERIES)}"
    assert "consts" in pub and set(pub["consts"]) and set(pub["consts"]) <= WL_CONSTS_M4, sorted(set(pub["consts"]) - WL_CONSTS_M4)
    m3c_out = env.simulate("M3C", 3000, dict(CFG, p=0.3), [pol], [0, 1], r=0.5, dev="cpu", diagnostic=True)     # CUSUM / audits are live here (M4 has none)
    m3c_pub = env.extract_public_log(m3c_out)
    assert set(m3c_pub) - {"consts"} == WL_SERIES and set(m3c_pub["consts"]) <= WL_CONSTS, sorted(set(m3c_pub["consts"]) - WL_CONSTS)
    assert (m3c_out["final"]["flags"] > 0).any() and (m3c_out["final"]["audits"] > 0).any()
    for s_ in range(2):
        _assert_rel(m3c_out["obs"][0, s_].numpy(), _ref_from_pub(m3c_pub, 0, s_), f"M3C reference vs obs, seed index {s_}")
    assert torch.equal(obs_reconstruct.reconstruct(m3c_pub), m3c_out["obs"])
    assert not (set(pub) | set(pub["consts"])) & (set(PRIVATE) | set(OTHERS) | {"audited_last", "S", "Ua", "Rs", "Rx", "rd"})
    # (2) independent reference + reconstructor
    for s_ in range(2):
        _assert_rel(out["obs"][0, s_].numpy(), _ref_from_pub(pub, 0, s_), f"reference vs obs, seed index {s_}")
    rebuilt = obs_reconstruct.reconstruct(pub)
    assert torch.equal(rebuilt, out["obs"])
    # (3) perturbation of the private log
    tampered = dict(out)
    tampered["private_log"] = {k: torch.rand_like(v.double()) for k, v in out["private_log"].items()}
    assert env.extract_public_log(tampered).keys() == pub.keys()
    assert torch.equal(obs_reconstruct.reconstruct(env.extract_public_log(tampered)), rebuilt)
    with pytest.raises(ValueError):
        obs_reconstruct.reconstruct(dict(pub, cus=torch.zeros(1)))
    # (4) announced
    ann = env.simulate("M3C", 600, CFG, [_rand_mlp(9, F=27, obs_version="announced")], [0], r=0.5, dev="cpu",
                       diagnostic=True, obs_version="announced")
    extra = dict(S=ann["private_log"]["cus1"] / CFG["tol"], audited_last=ann["private_log"]["audited_last"])
    ra = obs_reconstruct.reconstruct(env.extract_public_log(ann), "announced", extra)
    assert torch.equal(ra, ann["obs"]) and torch.equal(ra[..., :25], obs_reconstruct.reconstruct(env.extract_public_log(ann)))
    _assert_rel(ann["obs"][0, 0, :, :25].numpy(), _ref_from_pub(env.extract_public_log(ann), 0, 0), "announced public dims vs reference")
    # (5) static check of the reconstructor
    src = open(os.path.join(ROOT, "arbitration", "rl", "obs_reconstruct.py")).read()
    for node in ast.walk(ast.parse(src)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "")] + [a.name for a in node.names]
        for m in mods:
            assert not any(x in m for x in ("env", "arbitration.rl.obs", "frontier", "quota", "policy_sim")) or m in ("typing",), m
