"""T-36..T-44, T-61, T-63: S1 vulnerability suite, anchors, freeze, designer isolation (spec s14.5)."""
import hashlib
import inspect
import json
import math
import os
import subprocess

import numpy as np
import pytest
import torch
from scipy import stats

from conftest import ROOT, call_new_kw, cuda_required, need
from arbitration.rl import env, hw, metrics, policy, s1, vulns

# Knob that switches an operator OFF (identity).  O1/O2/O4/O6: kappa=0; O3: cap C=inf (max(resid,-inf)=resid); O5: g=inf (never triggers).
# (spec 8.2 gives O3's knob as C and O5 has no kappa: "knob=0 is identity" cannot hold literally for them -> reported to the commander)
OFF_KNOB = {"O1": 0.0, "O2": 0.0, "O3": math.inf, "O4": 0.0, "O5": math.inf, "O6": 0.0}
# O3 knob is C: smaller C = stronger, C=inf = identity.  C must be > kref (0.5 in M3C) or the CUSUM never grows, nobody is flagged and the
# operator has no effect; C=2 is used.
ON_KNOB = {"O1": 0.5, "O2": 0.5, "O3": 2.0, "O4": 0.5, "O5": 20.0, "O6": 0.5}
FIELDS_V = ["reg_b", "util_b", "flags", "pen", "audits", "susp", "nraid"]


@pytest.mark.gpu
def test_T36_vuln_identity_mutation():
    """T-36 (REQ-S1-05): the identity vulnerability environment (operator None) AND every operator O1..O6 switched off (knob 0 / C=inf / g=inf)
    equal the base environment bitwise on every output field; with the knob ON the injected operators change the outcome (the identity is not
    trivially satisfied by ignoring the operator)."""
    cuda_required()
    cfg = dict(p=0.1, tol=6.0, L=300.0, kref=0.5)
    pols = [policy.AdapterPolicy(d) for d in ({}, dict(dz=0.3), dict(b=0.1), dict(b=1.0))]
    seeds = [0, 1, 5000]
    base = env.simulate("M3C", 3000, cfg, pols, seeds, r=0.5, dev="cuda")
    a = vulns.simulate_vuln(None, 0.0, "M3C", 3000, cfg, pols, seeds, r=0.5, dev="cuda")
    for f in a["final"]:
        assert a["final"][f].equal(base["final"][f]), f
    assert (base["final"]["flags"] > 0).any()
    for op, knob in OFF_KNOB.items():
        v = vulns.simulate_vuln(op, knob, "M3C", 3000, cfg, pols, seeds, r=0.5, dev="cuda")
        for f in FIELDS_V:
            assert v["final"][f].equal(base["final"][f]), f"{op} off: {f} differs from the base environment"
    for op in ("O1", "O2", "O3", "O6"):
        v = vulns.simulate_vuln(op, ON_KNOB[op], "M3C", 3000, cfg, pols, seeds, r=0.5, dev="cuda")
        assert any(not v["final"][f].equal(base["final"][f]) for f in FIELDS_V), f"{op} on: no effect at all"


def test_T37_vuln_operators_inject_correctly():
    """T-37 (REQ-S1-06/07): hand-built states for O1,O2,O3,O6 (auto) and O4,O5 (blind); unchanged when the trigger is absent; every operator is the
    identity when switched off; a grid of hand-built states equals the test-local restatement of the formulas bitwise."""
    k, p = 0.4, 0.1
    assert vulns.o1_audit_prob(p, 3, k) == pytest.approx(p * (1 - k), abs=1e-15)
    assert vulns.o1_audit_prob(p, 7, k) == pytest.approx(p * (1 - k), abs=1e-15)
    assert vulns.o1_audit_prob(p, 2, k) == p
    assert vulns.o2_cusum_update(3.0, 0.5, 0.5, k) == pytest.approx((1 - k) * 2.0, abs=1e-15)
    assert vulns.o2_cusum_update(0.2, 1.0, 0.5, k) == 0.0
    assert vulns.o3_resid_cap(-5.0, 1.5) == -1.5 and vulns.o3_resid_cap(-1.0, 1.5) == -1.0 and vulns.o3_resid_cap(2.0, 1.5) == 2.0
    assert vulns.o4_pause_length(2500.0, 50, k) == pytest.approx(2500 * (1 - k)) and vulns.o4_pause_length(2500.0, 49, k) == 2500.0
    assert vulns.o5_cusum_reset(4.0, 20, 20) == 0.0 and vulns.o5_cusum_reset(4.0, 19, 20) == 4.0
    assert vulns.o6_pause_length(2500.0, k, True) == pytest.approx(2500 * (1 - k)) and vulns.o6_pause_length(2500.0, k, False) == 2500.0
    # ---- every operator is the identity when switched off (knob 0 / C=inf / g=inf), bitwise, over a grid of states
    for streak in (0, 2, 3, 9):
        assert vulns.o1_audit_prob(p, streak, 0.0) == p
    for S, resid in ((3.0, 0.5), (0.2, 1.0), (0.0, -2.0), (7.5, -3.0)):
        assert vulns.o2_cusum_update(S, resid, 0.5, 0.0) == max(S - resid - 0.5, 0.0)
    for resid in (-9.0, -1.0, 0.0, 4.0):
        assert vulns.o3_resid_cap(resid, math.inf) == resid
    for streak in (0, 49, 50, 80):
        assert vulns.o4_pause_length(2500.0, streak, 0.0) == 2500.0
    for rsw in (0, 19, 20, 400):
        assert vulns.o5_cusum_reset(4.0, rsw, math.inf) == 4.0
    for flagged in (True, False):
        assert vulns.o6_pause_length(2500.0, 0.0, flagged) == 2500.0
    # ---- grid against the independent restatement (bitwise)
    for kk in (0.1, 0.3, 0.77):
        for streak in range(0, 8):
            assert vulns.o1_audit_prob(p, streak, kk) == (p * (1 - kk) if streak >= 3 else p)
        for S, resid in ((3.0, 0.5), (0.2, 1.0), (0.0, -2.0), (7.5, -3.0), (1.0, -0.2)):
            assert vulns.o2_cusum_update(S, resid, 0.5, kk) == (1 - kk) * max(S - resid - 0.5, 0.0)
        for streak in (0, 49, 50, 51, 120):
            assert vulns.o4_pause_length(2500.0, streak, kk) == (2500.0 * (1 - kk) if streak >= 50 else 2500.0)
        for flagged in (True, False):
            assert vulns.o6_pause_length(300.0, kk, flagged) == (300.0 * (1 - kk) if flagged else 300.0)
    for C in (0.5, 1.5, 3.0):
        for resid in (-9.0, -3.0, -1.0, 0.0, 1.0, 6.0):
            assert vulns.o3_resid_cap(resid, C) == max(resid, -C)
    for rsw in (0, 1, 19, 20, 21, 300):
        for S in (0.0, 2.5, 5.9):
            assert vulns.o5_cusum_reset(S, rsw, 20) == (0.0 if rsw >= 20 else S)


# ---- hook-level log (REQ-S1-05/06/07): interface fixed here, see docs/direction2-tdd-map.md "約定"
#   simulate_vuln(..., hook_log=True) adds out["hook_log"][hook] = dict(mask=bool[P,S,T], out=[P,S,T], <inputs>[P,S,T] ...) for the 5 hooks
#   audit_prob(win_streak, p) | resid(resid) | cusum(S, resid, kref) | cusum_reset(S, rounds_since_win) | pause_len(L, clean_streak);
#   `mask` marks the rounds in which the hook was actually invoked for agent 1 (audit_prob/cusum_reset: every round; resid/cusum: audit settlements
#   of agent 1; pause_len: flags of agent 1).  Logged inputs are those the hook received IN THE MUTATED RUN (teacher forcing).
def _scalar(x):
    if torch.is_tensor(x):
        u = torch.unique(x)
        assert u.numel() == 1, "a constant input of a hook log must be one value"
        return float(u.item())
    return float(x)


BASE_HOOK = {
    "audit_prob": lambda i: torch.full_like(i["win_streak"], _scalar(i["p"])),
    "resid": lambda i: i["resid"],
    "cusum": lambda i: torch.clamp(i["S"] - i["resid"] - _scalar(i["kref"]), min=0.0),
    "cusum_reset": lambda i: i["S"],
    "pause_len": lambda i: torch.full_like(i["clean_streak"], _scalar(i["L"])),
}
# T-37b knobs.  O3: C is a cap on the evidence of one audit, so a SMALLER C is a STRONGER vulnerability and C=inf is the identity (REQ-S1-05).
# C=0.5 equals kref=0.5 of M3C: the CUSUM then never grows, agent 1 is never flagged and the pause_len hook is never invoked (the test was
# wrong, not the implementation).  C=2 (> kref) is a weaker cap, but lets the CUSUM grow so that agent 1 gets flagged.
T37B_KNOB = dict(ON_KNOB)
MUTATED_HOOK = {   # op -> (target hook, expected output given the logged inputs and the knob)
    "O1": ("audit_prob", lambda i, k: torch.where(i["win_streak"] >= 3, _scalar(i["p"]) * (1 - k), _scalar(i["p"]))),
    "O2": ("cusum", lambda i, k: (1 - k) * torch.clamp(i["S"] - i["resid"] - _scalar(i["kref"]), min=0.0)),
    "O3": ("resid", lambda i, k: torch.clamp(i["resid"], min=-k)),
    "O4": ("pause_len", lambda i, k: torch.where(i["clean_streak"] >= 50, (1 - k) * _scalar(i["L"]), _scalar(i["L"]))),
    "O5": ("cusum_reset", lambda i, k: torch.where(i["rounds_since_win"] >= 20, torch.zeros_like(i["S"]), i["S"])),
    "O6": ("pause_len", lambda i, k: torch.full_like(i["clean_streak"], (1 - k) * _scalar(i["L"]))),
}


@pytest.mark.parametrize("op", ["O1", "O2", "O3", "O4", "O5", "O6"])
def test_T37b_hook_level_isolation(op):
    """T-37b (REQ-S1-05/06/07; red-team 'injection leaks into another mechanism'): hook-level log of a mutated run.  Given the inputs each hook
    actually received, every NON-target hook returns exactly the base mechanism's value (bitwise), and the TARGET hook returns exactly the
    operator's formula.  O1/O2/O3/O6 must be effective (target output != base output on some call) in the scenario; O4/O5 (rare triggers)
    are only checked for isolation/formula."""
    cfg = dict(p=0.5, tol=6.0, L=100.0, kref=0.5)
    pols = [policy.AdapterPolicy(dict(b=1.0)), policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=0.3))]
    out = call_new_kw(vulns.simulate_vuln, dict(hook_log=True), op, T37B_KNOB[op], "M3C", 1500, cfg, pols, [0, 1, 2, 3], r=0.5, dev="cpu")
    log = out["hook_log"]
    assert set(log) == set(BASE_HOOK), sorted(log)
    target, formula = MUTATED_HOOK[op]
    knob = T37B_KNOB[op]
    for h, base_fn in BASE_HOOK.items():
        rec = log[h]
        m = rec["mask"].bool()
        inp = {k: v for k, v in rec.items() if k not in ("mask", "out")}
        exp = formula(inp, knob) if h == target else base_fn(inp)
        assert m.any() or h == target or op in ("O4", "O5"), f"hook {h} was never invoked"
        assert torch.equal(rec["out"][m], exp.to(rec["out"].dtype)[m]), f"{op}: hook {h} " + ("(target) != operator formula" if h == target else "differs from the base mechanism")
    if op in ("O1", "O2", "O3", "O6"):
        rec = log[target]
        m = rec["mask"].bool()
        inp = {k: v for k, v in rec.items() if k not in ("mask", "out")}
        assert m.any() and (rec["out"][m] != BASE_HOOK[target](inp)[m]).any(), f"{op}: the target hook was never effective in this scenario"
    # guard against a silently vacuous scenario (knob too weak): agent 1 must be flagged at least once, i.e. pause_len invoked
    assert log["pause_len"]["mask"].bool().any(), f"{op}: pause_len hook never invoked with knob {knob}; knob too weak, the test is vacuous"


def test_T38_auto_generator_deterministic():
    """T-38 (REQ-S1-06/01/28, v1.3): the auto operators IN THE SUITE are exactly {O1,O2,O3}, the blind operator exactly {O5}; O4 and O6 keep their
    definitions (formulas, registered operators) but are not in the suite (no `category`, composition check rejects them); calibration is
    deterministic; failure is reported, never replaced."""
    assert vulns.auto_operator_ids() == ["O1", "O2", "O3"] and vulns.blind_operator_ids() == ["O5"]
    assert set(vulns.auto_operator_ids()) | set(vulns.blind_operator_ids()) == {i for i, c in s1.CATEGORY_OF.items() if i.startswith("O")}
    assert "O4" not in s1.CATEGORY_OF and "O6" not in s1.CATEGORY_OF            # no category: not in the suite
    for op in ("O4", "O6"):                                                       # ... but the code is retained and T-37 still verifies the formulas
        assert vulns.get_operator(op) is not None, op
    assert vulns.o4_pause_length(100.0, 50, 0.5) == 50.0 and vulns.o6_pause_length(100.0, 0.5, True) == 50.0
    grid = list(np.linspace(0, 1, 101))
    f = lambda knob: 0.05 * knob
    a = vulns.calibrate_knob("O1", 0.02, f, grid)
    b = vulns.calibrate_knob("O1", 0.02, f, grid)
    assert a == b and a["status"] == "ok"
    assert 0.8 * 0.02 <= f(a["knob"]) <= 1.2 * 0.02
    bad = vulns.calibrate_knob("O3", 0.02, lambda k: 0.001 * k, grid)
    assert bad["status"] == "needs_commander" and "knob" not in bad
    reg = _registry()
    assert s1.check_registry_composition(reg) is True
    for extra_id in ("O4", "O6", "B1", "B2"):
        with pytest.raises(ValueError):
            s1.check_registry_composition(reg + [dict(id=extra_id, kind="auto", category="in_family", size=0.02, blind=False)])


@pytest.mark.gpu
def test_T39_vuln_known_delta_reproduced_ci():
    """T-39 (REQ-S1-02/03/28, v1.3): for every registry (vuln,size) the increment is RECOMPUTED IN THE TEST from the per-seed paired gains that
    `vulns.recompute_delta` returns on val U val2 (64 seeds: 2000-2031 then 3000-3031; G_star_s[64], G_HW_s_by_name{36 names: [64]}): best handwritten =
    argmax of the mean over the 36 S_HW names (ties -> earlier) for Delta_fine, over the 32 coarse names for Delta_coarse; Delta = mean, CI = t(63)=1.9983
    half-width computed here.  Basis (`size_basis`): out_of_family and blind use Delta_fine, in_family uses Delta_coarse.  Required: the basis
    Delta lies in [0.8s, 1.2s], inside the registered delta_val CI, and the test-computed CI lower bound > 0; the best handwritten name (36-list)
    equals the registered one; Delta_fine and Delta_coarse both equal the registry (rel 1e-9).  D3 (in_family_known, no size): both deltas equal
    the registry and are negative.  13 sized entries are checked: out_of_family 3x2, in_family 3x2, blind 1."""
    cuda_required()
    reg = vulns.load_registry()
    names_fine = [n for n, _ in hw.build_s_hw()]
    names_coarse = [n for n, _ in hw.build_s_hw_coarse()]
    assert len(names_fine) == 36 and len(names_coarse) == 32
    tcrit = float(stats.t.ppf(0.975, 63))
    assert abs(tcrit - 1.9983) < 1e-4
    calib = list(range(2000, 2032)) + list(range(3000, 3032))

    def best_delta(g_star, by, names):
        means = {n: float(np.mean(by[n])) for n in names}
        best = max(names, key=lambda n: (means[n], -names.index(n)))
        d_s = g_star - np.asarray(by[best], dtype=float)
        m = float(d_s.mean())
        half = tcrit * float(d_s.std(ddof=1)) / math.sqrt(len(d_s))
        return best, m, m - half, m + half

    # GPU-08: the full run needs ~5 recomputations x 64 seeds x 38 policies at T=1e5 per entry; to keep every job <= 1500 s run it in segments:
    # D2_T39_SLICE="i:j" checks registry[i:j] only (the count assertion at the end applies to the full, unsliced run).
    sl = os.environ.get("D2_T39_SLICE")
    if sl:
        i0, i1 = (int(x) for x in sl.split(":"))
        reg = reg[i0:i1]
    n_checked = 0
    for e in reg:
        if e.get("category") in ("anchor", "contaminated_excluded", None) or e["kind"] == "anchor":
            assert e.get("category") in ("anchor", "contaminated_excluded"), f"{e['id']}: registry entry without a v1.3 category"
            continue
        res = vulns.recompute_delta(e)
        assert list(res["val_seeds"]) == calib
        by = res["G_HW_s_by_name"]
        assert set(by) == set(names_fine) and all(len(np.asarray(v)) == 64 for v in by.values())
        g_star = np.asarray(res["G_star_s"], dtype=float)
        assert g_star.shape == (64,)
        bf, mf, lof, hif = best_delta(g_star, by, names_fine)
        bc, mc, loc, hic = best_delta(g_star, by, names_coarse)
        tag = f"{e['id']} s={e.get('size')}"
        assert res["G_HW_best_name"] == bf == e["G_HW_best_val"]["name"], tag
        for key, m in (("delta_fine", mf), ("delta_coarse", mc)):
            assert abs(e[key]["mean"] - m) <= 1e-9 * max(abs(m), 1e-12) + 1e-12, f"{tag}: registry {key} not reproduced"
            assert abs(res[key]["mean"] - m) <= 1e-12, f"{tag}: reported {key} != recomputed"
        if e["category"] == "in_family_known":
            assert mf < 0 and mc < 0, f"{tag}: D3 increments must be negative"
            continue
        s = e["size"]
        assert s in (0.01, 0.02), tag
        basis = "coarse" if e["category"] == "in_family" else "fine"
        assert e["size_basis"] == basis, tag
        m, lo, hi = (mc, loc, hic) if basis == "coarse" else (mf, lof, hif)
        assert 0.8 * s <= m <= 1.2 * s, f"{tag}: Delta_{basis}={m}"
        assert lo > 0, f"{tag}: recomputed CI lower bound {lo} <= 0"
        assert e["delta_val"]["lo"] <= m <= e["delta_val"]["hi"], f"{tag}: outside the registered delta_val CI"
        assert abs(m - e["delta_val"]["mean"]) <= 1e-9 * abs(e["delta_val"]["mean"]), f"{tag}: frozen registry value not reproduced"
        n_checked += 1
    if not sl:
        assert n_checked == 13, f"registry must hold 13 sized entries (3x2 out_of_family + 3x2 in_family + 1 blind), got {n_checked}"


@pytest.mark.gpu
def test_T41_vuln_ref_policy_public_only():
    """T-41 (REQ-S1-03, OBS-06): non-blind reference policies run through the public-history sandbox; perturbing private state does not change output."""
    cuda_required()
    reg = [e for e in vulns.load_registry() if not e.get("blind") and e["kind"] != "anchor" and e.get("category") != "contaminated_excluded"]   # designer_1 has no reference policy (not implemented)
    assert reg
    rng = np.random.default_rng(0)
    hist = rng.random((200, 4))
    for e in reg:
        a = vulns.run_ref_policy_sandbox(e, hist)
        b = vulns.run_ref_policy_sandbox(e, hist, private=dict(cus=rng.random(200), pend_a=rng.integers(0, 3, 200)))
        assert np.array_equal(np.asarray(a), np.asarray(b)), e["id"]


_CAT = {"D1": "out_of_family", "D2": "out_of_family", "O1": "out_of_family", "O2": "in_family", "O3": "in_family", "D4": "in_family",
        "D3": "in_family_known", "O5": "blind", "NAIVE": "anchor", "M4rw0": "anchor", "designer_1": "contaminated_excluded"}
_KIND = {"D1": "designer", "D2": "designer", "D4": "designer", "D3": "designer", "O1": "auto", "O2": "auto", "O3": "auto", "O5": "blind",
         "NAIVE": "anchor", "M4rw0": "anchor", "designer_1": "contaminated"}


def _ci_ent(m, half=None):
    half = abs(m) * 0.25 if half is None else half
    return dict(mean=m, lo=m - half, hi=m + half)


def _entry(i, size):
    """One synthetic v1.3 registry entry, consistent with REQ-S1-02/03/26."""
    cat = _CAT[i]
    e = dict(id=i, kind=_KIND[i], category=cat, size=size, blind=(cat == "blind"), source={"D1": "designer_2", "D2": "designer_2", "D3": "designer_2",
             "D4": "designer_2", "designer_1": "designer_1"}.get(i, "auto" if cat != "anchor" else "anchor"))
    if cat in ("anchor", "contaminated_excluded"):
        e["size"] = None
        return e
    g_hw, g_star = 0.02, None
    if cat == "out_of_family":
        d_fine, d_coarse, basis = size, size, "fine"
    elif cat == "blind":
        d_fine, d_coarse, basis = size, size, "fine"
    elif cat == "in_family":
        d_fine, d_coarse, basis = {"O2": -0.001, "O3": 0.0016, "D4": 0.0005}[i], size, "coarse"
    else:                                       # in_family_known: negative increments, no size
        d_fine, d_coarse, basis = -0.01, -0.02, None
        e["size"] = None
    g_star = g_hw + d_fine
    e.update(size_basis=basis, delta_fine=_ci_ent(d_fine, 0.0004 if d_fine > 0 else 0.002), delta_coarse=_ci_ent(d_coarse, abs(d_coarse) * 0.2),
             G_star_val=_ci_ent(g_star), G_HW_best_val=dict(name="dz=0.3", mean=g_hw), ratio_HW_over_Gstar=g_hw / g_star, calib_seeds="val+val2",
             honest_drift=_ci_ent(0.001), knob_monotone=(i != "D4"), delta_val_only=d_fine, delta_val2_only=d_fine, delta_train64=d_fine)
    e["delta_val"] = dict(e["delta_fine"] if basis == "fine" else e["delta_coarse"]) if basis else None
    return e


def _registry():
    """Synthetic composition of the suite (11 ids): sizes {0.01,0.02} for out_of_family and in_family, blind 0.02 only, D3 / anchors / designer_1 unsized."""
    reg = []
    for i, cat in _CAT.items():
        if cat in ("out_of_family", "in_family"):
            reg += [_entry(i, 0.01), _entry(i, 0.02)]
        elif cat == "blind":
            reg.append(_entry(i, 0.02))
        else:
            reg.append(_entry(i, None))
    return reg


def _git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout


def _mkrepo(path):
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")


def test_T42_vuln_registry_frozen_and_composition(tmp_path):
    """T-42 (REQ-S1-01/02/13/14/26, v1.3): composition = exactly 11 ids (out_of_family {D1,D2,O1}, in_family {O2,O3,D4}, in_family_known {D3}, blind {O5},
    anchor {NAIVE,M4}, contaminated_excluded {designer_1}); out_of_family and in_family carry sizes {0.01,0.02}, blind only 0.02; categories and the
    blind flag are fixed/consistent; O4/O6/B1/B2 are rejected; sha256; git ancestry (the freeze commit is an ancestor of every non-exempt S1 part
    commit; NAIVE-pilot/smoke cells exempt); the freeze registration carries lam, n_S, T_train, G_gens, sigma0, cold/warm registrations + pilot, MDE_D_plan,
    G_s, the 36-name S_HW list, the val+val2 calibration record and every vulnerability's category - a missing or wrong item is rejected."""
    reg = _registry()
    assert s1.check_registry_composition(reg) is True
    assert {e["id"] for e in reg} == set(_CAT) and len({e["id"] for e in reg}) == 11
    # structural violations
    drop = lambda pred: [e for e in reg if not pred(e)]
    for bad in (drop(lambda e: e["id"] == "O3"), drop(lambda e: e["id"] == "D3"), drop(lambda e: e["id"] == "O5"), drop(lambda e: e["id"] == "NAIVE"),
                drop(lambda e: e["id"] == "designer_1"),
                drop(lambda e: e["id"] == "O1" and e["size"] == 0.01),                                  # a size of an out_of_family vuln missing
                drop(lambda e: e["id"] == "D4" and e["size"] == 0.02),                                  # ... of an in_family one
                reg + [dict(_entry("O5", 0.01))],                                                       # blind only has 0.02
                reg + [dict(_entry("D1", 0.003))],                                                      # old sizes are gone
                reg + [dict(_entry("D2", 0.005))]):
        with pytest.raises(ValueError):
            s1.check_registry_composition(bad)
    import copy as _copy

    def mut(i, **kw):
        r = _copy.deepcopy(reg)
        for e in r:
            if e["id"] == i:
                e.update(kw)
        return r

    for bad in (mut("D1", category="in_family"), mut("O2", category="out_of_family"), mut("D3", category="in_family"), mut("O5", category="in_family"),
                mut("designer_1", category="in_family"), mut("NAIVE", category="out_of_family"), mut("D1", category=None), mut("D1", category="weird"),
                mut("O5", blind=False), mut("D1", blind=True), mut("NAIVE", blind=True)):
        with pytest.raises(ValueError):
            s1.check_registry_composition(bad)
    repo = str(tmp_path)
    _mkrepo(repo)
    (tmp_path / "vuln_suite.json").write_text(json.dumps(reg))
    assert s1.registry_sha256(str(tmp_path / "vuln_suite.json")) == hashlib.sha256((tmp_path / "vuln_suite.json").read_bytes()).hexdigest()
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "naive_pilot__run0__g0000-0025.json").write_text(json.dumps(dict(cell="naive_pilot")))
    (parts / "smoke__run0__g0000-0005.json").write_text(json.dumps(dict(cell="smoke")))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "pilot and smoke parts before freeze")
    (tmp_path / "freeze.md").write_text("frozen")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "freeze")
    (parts / "O1_s0.02__run0__g0000-0025.json").write_text(json.dumps(dict(cell="O1_s0.02")))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "s1 part")
    assert s1.is_exempt_cell("naive_pilot") and s1.is_exempt_cell("smoke") and not s1.is_exempt_cell("O1_s0.02")
    ok = s1.freeze_precedes_parts(repo, "freeze.md", "parts")
    assert ok["ok"] is True and ok["violations"] == []
    repo2 = str(tmp_path / "r2")
    os.makedirs(os.path.join(repo2, "parts"))
    _mkrepo(repo2)
    open(os.path.join(repo2, "parts", "O1_s0.02__run0__g0000-0025.json"), "w").write("{}")
    _git(repo2, "add", "-A")
    _git(repo2, "commit", "-qm", "part first")
    open(os.path.join(repo2, "freeze.md"), "w").write("frozen")
    _git(repo2, "add", "-A")
    _git(repo2, "commit", "-qm", "freeze late")
    bad = s1.freeze_precedes_parts(repo2, "freeze.md", "parts")
    assert bad["ok"] is False and len(bad["violations"]) == 1
    # ---- freeze registration contents (REQ-S1-14, v1.3)
    from conftest import make_freeze
    fz = make_freeze()
    assert s1.check_freeze_registration(fz) is True
    assert s1.load_config(fz, "s1")["n_S"] == 256 and s1.load_config(fz, "main")["start"] == "cold"
    for key in ("lam", "n_S", "T_train", "G_gens", "sigma0", "cold_start", "warm_start", "pilot", "chosen_start", "MDE_D_plan", "G_s", "registry_sha256",
                "S_HW", "calibration", "categories", "nS_coverage_disclosure"):
        with pytest.raises(ValueError):
            s1.check_freeze_registration({k: v for k, v in fz.items() if k != key})
    names = [n for n, _ in hw.build_s_hw()]
    for bad_hw in (names[:32], names[:35], names + ["honest"], list(reversed(names)), names[:-1] + ["dz=9.9"], None):
        with pytest.raises(ValueError):
            s1.check_freeze_registration(make_freeze(S_HW=bad_hw))
    for bad_cal in (dict(fz["calibration"], calib_seeds="val"), dict(fz["calibration"], n_seeds=32), "val+val2", None):
        with pytest.raises(ValueError):
            s1.check_freeze_registration(make_freeze(calibration=bad_cal))
    cats = dict(fz["categories"])
    for bad_cat in (dict(cats, D1="in_family"), {k: v for k, v in cats.items() if k != "O5"}, dict(cats, O4="in_family"), dict(cats, D3=None)):
        with pytest.raises(ValueError):
            s1.check_freeze_registration(make_freeze(categories=bad_cat))
    with pytest.raises(ValueError):
        s1.check_freeze_registration(make_freeze(chosen_start="hot"))


def test_T43_blind_design_isolation_manifest():
    """T-43 (REQ-S1-12/13/21): designer package may hold only allowed files (no feature table, strategies() table, notes, training logs - checked by
    path and sha256); the designer prompt must not contain feature details."""
    notes = os.path.join(ROOT, "docs", "direction2-notes.md")
    h = hashlib.sha256(open(notes, "rb").read()).hexdigest()
    allowed = ["docs/direction2-vuln-format.md", "arbitration/gpu/frontier.py"]          # v1.2 (REQ-S1-12): cleaned format note + frontier M3C part only
    good = [dict(path=p, sha256="a" * 64) for p in allowed]
    assert s1.check_designer_manifest(good, allowed, [h]) == []
    bad = good + [dict(path="docs/direction2-notes.md", sha256=h), dict(path="arbitration/rl/obs.py", sha256="b" * 64)]
    v = s1.check_designer_manifest(bad, allowed, [h])
    assert len(v) >= 2
    assert len(s1.check_designer_manifest(good + [dict(path="docs/direction2-spec.md", sha256="c" * 64)], allowed, [h])) == 1, \
        "v1.2: the spec itself (s8 text) is no longer readable material for blind designers"
    assert len(s1.check_designer_manifest(good + [dict(path="renamed_copy.txt", sha256=h)], allowed + ["renamed_copy.txt"], [h])) == 1
    assert s1.check_designer_prompt("Design a mutation of the audit probability hook.") == []
    for phrase in ("k=8", "EMA", "λ_e", "f06", "P_MAX", "strategies()"):
        assert s1.check_designer_prompt(f"hint: {phrase} is used") == [phrase], phrase


def test_T44_anchor_definitions():
    """T-44 (REQ-S1-08/09/10/11, ENV-11): NAIVE = M3C,p=0,r=0.5; M4 anchor = M4, eps=0.2, L=20, rw=0, r=0.99, strategy b=0.3; pass rule
    G_RL>=0.8 G*_ref & CI lo>0; W1 re-measured G*_ref CI lower bound <=0 => M4 anchor informational (not counted, only NAIVE remains)."""
    a = s1.anchor_definitions()
    assert (a["NAIVE"]["mech"], a["NAIVE"]["p"], a["NAIVE"]["r"]) == ("M3C", 0.0, 0.5)
    m = a["M4"]
    assert (m["mech"], m["eps"], m["L"], m["rw"], m["r"]) == ("M4", 0.2, 20, 0.0, 0.99) and m["strategy"] in ("b=0.3", {"b": 0.3})
    assert a["NAIVE"]["r"] == 0.5 and m["r"] == 0.99
    assert s1.anchor_passes(0.24, 0.2, 0.28, 0.01) is True and s1.anchor_passes(0.2241, 0.2, 0.28, 0.01) is True
    assert s1.anchor_passes(0.2239, 0.2, 0.28, 0.01) is False and s1.anchor_passes(0.27, 0.0, 0.28, 0.01) is False
    assert s1.anchor_passes(0.27, 0.2, 0.28, 0.5) is False
    assert s1.m4_anchor_status(0.0) == "informational" and s1.m4_anchor_status(-0.01) == "informational"
    assert s1.m4_anchor_status(0.001) == "counted"
    assert env.anchor_config("M4")["rw"] == 0.0
    assert s1.CATEGORY_OF["NAIVE"] == "anchor" and s1.CATEGORY_OF["M4rw0"] == "anchor"          # v1.3: anchors are neither out_of_family nor in_family
    assert not s1.counts_in_detection_statistics("NAIVE") and not s1.counts_in_detection_statistics("M4rw0")


def test_T61_naive_pilot_decision_rule():
    """T-61 (REQ-S1-20, OPT-07, ACT-04, STOP-02): cold>=0.8G* -> cold; else warm>=0.8G* -> warm; else STOP-02; pilot rejects test seeds; both starts are in the freeze file."""
    g = 0.28
    assert s1.pilot_decision(0.8 * g, None, g) == "cold" and s1.pilot_decision(0.30, 0.30, g) == "cold"
    assert s1.pilot_decision(0.1, 0.8 * g, g) == "warm"
    assert s1.pilot_decision(0.1, 0.1, g) == "STOP-02"
    s1.require_pilot_seeds([0, 5, 2000, 2031])
    with pytest.raises(ValueError):
        s1.require_pilot_seeds([5000])
    from arbitration.rl import stop
    assert stop.w1_stop(False, 0.0058, True) is True
    from conftest import make_freeze
    freeze = make_freeze(cold_start={}, warm_start={}, pilot={}, MDE_D_plan=0.005, G_s=[0.02],
                         nS_coverage_disclosure="single-pair worst case needs n_S~565, NOT covered (REQ-OPT-13)")
    assert s1.check_freeze_registration(freeze) is True
    with pytest.raises(ValueError):
        s1.check_freeze_registration({k: v for k, v in freeze.items() if k != "warm_start"})


def test_T63_designer_logs_saved(tmp_path):
    """T-63 (REQ-S1-21/12): designer_logs/{id}/prompt.txt and tool_calls.jsonl exist, are tracked in git, and tool_calls.jsonl has no reads of
    s4/s5, strategies() or direction2-notes.md."""
    repo = str(tmp_path)
    _mkrepo(repo)
    root = tmp_path / "designer_logs"
    for d in ("d1", "d2"):
        (root / d).mkdir(parents=True)
        (root / d / "prompt.txt").write_text("design a mutation")
        (root / d / "tool_calls.jsonl").write_text(json.dumps(dict(tool="Read", path="spec_s2.md")) + "\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "logs")
    assert s1.check_designer_logs(repo, "designer_logs", ["d1", "d2"]) == []
    (root / "d2" / "tool_calls.jsonl").write_text(json.dumps(dict(tool="Read", path="docs/direction2-notes.md")) + "\n")
    assert len(s1.check_designer_logs(repo, "designer_logs", ["d1", "d2"])) >= 1
    (root / "d3").mkdir()
    (root / "d3" / "prompt.txt").write_text("x")
    assert len(s1.check_designer_logs(repo, "designer_logs", ["d1", "d3"])) >= 1      # missing tool_calls.jsonl / untracked


# ------------------------------------------------------------------------------------------------------------------------ S1-18
def _ci(m, h):
    return dict(mean=m, lo=m - h, hi=m + h)


def _s1_doc(detected_by):
    """Synthetic results/direction2/s1.json (v1.3): runs (out_of_family at 0.01 and 0.02; in_family and the blind O5 only at 0.02; two anchors),
    summary rows (scope 'out_of_family' n=3 per size, scope 'in_class' n=6 at 0.02), the D3 registry fields (no RL run) and the figure path.
    detected_by(vuln_id, size) -> bool decides D (CI lo>0 and D>=MDE_D)."""
    runs = []
    plan = [("O1", "auto", "out_of_family", (0.01, 0.02)), ("D1", "designer", "out_of_family", (0.01, 0.02)), ("D2", "designer", "out_of_family", (0.01, 0.02)),
            ("O2", "auto", "in_family", (0.02,)), ("O3", "auto", "in_family", (0.02,)), ("D4", "designer", "in_family", (0.02,)),
            ("O5", "blind", "blind", (0.02,))]
    for i, kind, cat, sizes in plan:
        for sz in sizes:
            det = detected_by(i, sz)
            D = _ci(0.02, 0.005) if det else _ci(0.001, 0.004)
            runs.append(dict(id=i, kind=kind, category=cat, size=sz, G_RL=_ci(0.03, 0.004), D=D, MDE_D=0.0058, detected=bool(det)))
    for a_id, gref in (("NAIVE", 0.28), ("M4rw0", 0.12)):
        runs.append(dict(id=a_id, kind="anchor", category="anchor", size=None, G_RL=_ci(0.9 * gref, 0.003), G_star_ref=gref, MDE_G=0.0058, passed=True,
                         informational=False))
    lb = lambda x, n: float(stats.beta.ppf(0.05, x, n - x + 1)) if x > 0 else 0.0
    summary = []
    for sz in (0.01, 0.02):
        x = sum(r["detected"] for r in runs if r["category"] == "out_of_family" and r["size"] == sz)
        summary.append(dict(scope="out_of_family", size=sz, x=x, n=3, p_hat=x / 3, LB=lb(x, 3)))
    xc = sum(r["detected"] for r in runs if r["category"] in ("out_of_family", "in_family") and r["size"] == 0.02)
    summary.append(dict(scope="in_class", size=0.02, x=xc, n=6, p_hat=xc / 6, LB=lb(xc, 6)))
    return dict(runs=runs, summary=summary, d3=dict(delta_fine=-0.01, delta_coarse=-0.02, G_star=0.01, G_HW=0.02), figure="s1_power.png")


def test_S118_s1_output_schema(tmp_path):
    """S1-18 (REQ-S1-18, modelled on T-51; v1.3): `s1.validate_s1_output(doc, root)` accepts a synthetic s1.json (per (vuln,size): category, G_RL / D with
    mean+CI, MDE_D, detected; anchors with G_RL, G_star_ref, passed; summary rows: out_of_family (y, n=3) for every size run and in_class (x, n=6) at
    0.02, each with p_hat and the Clopper-Pearson LB; the D3 registry fields; the figure path that exists) and rejects: a missing field, a wrong
    type, `detected` inconsistent with the D rule, summary x / p_hat / LB / n inconsistent with the runs (n must be 3 resp. 6; in-family, blind
    must not be counted in y; the class count x counts out_of_family + in_family at 0.02 only), a run of D3 / designer_1 / O4 / O6 / B1 / B2, an in-family
    run at s=0.01, a category that disagrees with the id, a missing D3 block, a missing figure file."""
    validate = need(s1, "validate_s1_output", "REQ-S1-18")
    (tmp_path / "s1_power.png").write_bytes(b"\x89PNG\r\n")
    det = lambda i, sz: (i in ("O1", "O2", "O3", "D1", "D2") if sz == 0.02 else i in ("O1", "D1"))
    doc = _s1_doc(det)
    assert validate(doc, root=str(tmp_path)) is True
    assert [(r["scope"], r["size"], r["x"], r["n"]) for r in doc["summary"]] == [("out_of_family", 0.01, 2, 3), ("out_of_family", 0.02, 3, 3), ("in_class", 0.02, 5, 6)]
    assert abs(doc["summary"][1]["LB"] - 0.36840315) < 1e-7 and abs(doc["summary"][2]["LB"] - 0.41819659) < 1e-7
    import copy as _copy

    def bad(mut, what):
        d = _copy.deepcopy(doc)
        mut(d)
        with pytest.raises(ValueError):
            validate(d, root=str(tmp_path))

    first_run = lambda d: d["runs"][0]
    bad(lambda d: first_run(d).pop("G_RL"), "missing G_RL")
    bad(lambda d: first_run(d).pop("MDE_D"), "missing MDE_D")
    bad(lambda d: first_run(d).pop("detected"), "missing detected")
    bad(lambda d: first_run(d).pop("category"), "missing category")
    bad(lambda d: first_run(d).update(detected="yes"), "detected not bool")
    bad(lambda d: first_run(d)["D"].pop("lo"), "CI without lower bound")
    bad(lambda d: first_run(d).update(size=0.007), "size not in {0.01,0.02}")
    bad(lambda d: first_run(d).update(size=0.005), "old size 0.005")
    bad(lambda d: first_run(d).update(detected=not first_run(d)["detected"]), "detected inconsistent with D rule")
    bad(lambda d: first_run(d)["D"].update(lo=-0.001) or first_run(d).update(detected=True), "lo<=0 yet detected")
    bad(lambda d: first_run(d).update(category="in_family"), "category disagrees with the id")
    bad(lambda d: d["summary"][1].update(x=d["summary"][1]["x"] - 1), "out_of_family x != detected count")
    bad(lambda d: d["summary"][2].update(x=d["summary"][2]["x"] + 1), "class x != detected count")
    bad(lambda d: d["summary"][1].update(p_hat=0.5), "summary p_hat != x/n")
    bad(lambda d: d["summary"][1].update(LB=0.99), "summary LB != Clopper-Pearson")
    bad(lambda d: d["summary"][1].update(n=8), "out_of_family n != 3")
    bad(lambda d: d["summary"][2].update(n=8), "class n != 6")
    bad(lambda d: d["summary"][2].update(scope="in_family"), "unknown scope")
    bad(lambda d: d["summary"].pop(0), "a run size without a summary row")
    bad(lambda d: d["summary"].append(dict(d["summary"][0], size=0.005)), "summary row for a size that was not run")
    runs_of = lambda d, i: [r for r in d["runs"] if r["id"] == i]
    # the blind O5 and the in-family runs are never counted in y: flipping O5 / O2 must not change the out_of_family row
    flip = _copy.deepcopy(doc)
    for r in runs_of(flip, "O5") + runs_of(flip, "O2"):
        r["detected"] = not r["detected"]
        r["D"] = _ci(0.02, 0.005) if r["detected"] else _ci(0.001, 0.004)
    flip["summary"][2].update(x=4,                                    # O5 is not in the class count; flipping O2 (in-family) changes it by -1
                               p_hat=4 / 6, LB=float(stats.beta.ppf(0.05, 4, 3)))
    assert validate(flip, root=str(tmp_path)) is True
    bad(lambda d: d["runs"].append(dict(_copy.deepcopy(runs_of(d, "O5")[0]), id="O5", category="blind", size=0.01)), "blind at s=0.01")
    bad(lambda d: d["runs"].append(dict(_copy.deepcopy(runs_of(d, "O2")[0]), size=0.01)), "in_family at s=0.01")
    for i, cat in (("D3", "in_family_known"), ("designer_1", "contaminated_excluded"), ("O4", "blind"), ("O6", "in_family"), ("B1", "blind"), ("B2", "blind")):
        bad(lambda d, i=i, cat=cat: d["runs"].append(dict(_copy.deepcopy(runs_of(d, "O1")[0]), id=i, category=cat, size=0.02)), f"run of {i}")
    bad(lambda d: d.pop("d3"), "D3 block missing")
    bad(lambda d: d["d3"].pop("delta_fine"), "D3 block without delta_fine")
    bad(lambda d: d["d3"].update(delta_coarse="x"), "D3 delta_coarse not numeric")
    bad(lambda d: d.update(figure="missing.png"), "figure file missing")
    bad(lambda d: d.pop("figure"), "figure path missing")
    bad(lambda d: d.pop("summary"), "summary missing")
    anchor = lambda d: [r for r in d["runs"] if r["kind"] == "anchor"][0]
    bad(lambda d: anchor(d).pop("G_star_ref"), "anchor without G_star_ref")
    bad(lambda d: anchor(d).update(passed=False), "anchor 'passed' inconsistent with G_RL >= 0.8 G*_ref and CI lo>0")


# ====================================================================================================================== spec v1.2 additions
_DESIGNER_ROOT = os.path.join(ROOT, "results", "direction2", "designer_logs")
_S121_WORDS = ("k=8", "EMA", "λ_e", "f06", "P_MAX", "strategies()")                # REQ-S1-21 list (reference copy, not imported)


def test_S123_designer_source_and_status():
    """S1-23 (REQ-S1-23, v1.2): the 4 designer vulnerabilities counted in S1 come from designer_2; designer_1 is contaminated and is NOT counted.
    (a) designer_2's archive exists: prompt.txt, tool_calls.jsonl, isolation_check.md (states isolation holds); its tool calls are only Reads of
    arbitration/gpu/frontier.py and docs/direction2-vuln-format.md plus the output Write and the final hand-back - no Bash/Grep/Glob - and the
    prompt has no REQ-S1-21 word.  (b) designer_1's archive has contamination.md (and the tool log); its Read of the spec outside the allowed
    material is visible in the log.  (c) the checkers agree: check_designer_prompt is empty for designer_2 and the manifest check against the v1.2
    allowed list is clean for designer_2 but not for designer_1.  (d) the registry part is test_S123b."""
    d2, d1 = os.path.join(_DESIGNER_ROOT, "designer_2"), os.path.join(_DESIGNER_ROOT, "designer_1")
    for fn in ("prompt.txt", "tool_calls.jsonl", "isolation_check.md"):
        assert os.path.exists(os.path.join(d2, fn)), f"designer_2/{fn} missing"
    for fn in ("prompt.txt", "tool_calls.jsonl", "contamination.md"):
        assert os.path.exists(os.path.join(d1, fn)), f"designer_1/{fn} missing"
    iso = open(os.path.join(d2, "isolation_check.md"), encoding="utf-8").read()
    assert "隔離成立" in iso
    calls2 = [json.loads(x) for x in open(os.path.join(d2, "tool_calls.jsonl"), encoding="utf-8") if x.strip()]
    assert calls2 and {c["tool"] for c in calls2} <= {"Read", "Write", "SubagentHandback"}, "designer_2 may not use Bash/Grep/Glob"
    def _rel(fp):        # the recorded paths are absolute on the machine that made the log (/home/lex/...); keep the part inside the repository
        return fp.split("scarce-actuator-arbitration/", 1)[1] if "scarce-actuator-arbitration/" in fp else os.path.relpath(fp, ROOT)
    reads = sorted(_rel(c["args"]["file_path"]) for c in calls2 if c["tool"] == "Read")
    allowed = ["arbitration/gpu/frontier.py", "docs/direction2-vuln-format.md"]
    assert reads == allowed, f"designer_2 Read set {reads}"
    writes = [_rel(c["args"]["file_path"]) for c in calls2 if c["tool"] == "Write"]
    assert writes == ["docs/direction2-blind-vulns-2.md"]
    assert s1.check_designer_prompt(open(os.path.join(d2, "prompt.txt"), encoding="utf-8").read()) == []
    man2 = [dict(path=p, sha256="0" * 64) for p in reads]
    assert s1.check_designer_manifest(man2, allowed, []) == []
    calls1 = [json.loads(x) for x in open(os.path.join(d1, "tool_calls.jsonl"), encoding="utf-8") if x.strip()]      # designer_1 log uses name/input keys
    reads1 = {_rel(c["input"]["file_path"]) for c in calls1 if c["name"] == "Read" and "file_path" in c.get("input", {})}
    assert "docs/direction2-spec.md" in reads1, "designer_1's contaminating spec read must stay on record"
    assert s1.check_designer_manifest([dict(path=p, sha256="0" * 64) for p in sorted(reads1)], allowed, []) != []


def test_S123b_registry_designer_source():
    """S1-23 (d), v1.3: when the vulnerability suite registry exists, every kind=='designer' entry names designer_2 as its source (field `source`)
    and the ids D1..D4 stay (D3 is registered although it is not run: in_family_known).  designer_1 (contaminated) may appear ONLY as an entry with
    category 'contaminated_excluded' (kind 'contaminated'); it never carries a size, never appears under another category, and no other entry
    refers to it.  (v1.2 required it to be absent altogether; spec v1.3 REQ-S1-01 lists it as the 11th item, so the assertion is relaxed to what
    the spec says and stays strict about everything else.)  The registry file is regenerated by the recalibration step, so this is red until then."""
    try:
        reg = vulns.load_registry()
    except FileNotFoundError:
        raise NotImplementedError("REQ-S1-23: results/direction2/vuln_suite.json does not exist yet")
    des = [e for e in reg if e.get("kind") == "designer"]
    assert sorted({e["id"] for e in des}) == ["D1", "D2", "D3", "D4"]
    assert all(e.get("source") == "designer_2" for e in des), "every counted designer vulnerability must come from designer_2"
    for e in reg:
        mentions = "designer_1" in json.dumps(e)
        if mentions:
            assert e.get("category") == "contaminated_excluded" and e.get("kind") == "contaminated" and e.get("size") is None, e["id"]
        if e.get("category") == "contaminated_excluded":
            assert e["id"] == "designer_1" and e.get("kind") == "contaminated", e["id"]


def _scan_ref(paths, words):
    import hashlib as _h
    hits, files = [], []
    for p in paths:
        b = open(p, "rb").read()
        files.append(dict(path=p, sha256=_h.sha256(b).hexdigest()))
        t = b.decode("utf-8")
        hits += [(p, w) for w in words if w in t]
    return hits, files


def test_S124_blind_material_prescan(tmp_path):
    """S1-24 (REQ-S1-24, v1.2): everything a blind designer may read is scanned with the REQ-S1-21 word list BEFORE delivery, not only the prompt.
    API fixed by this test: `s1.scan_blind_materials(paths, extra_words=()) -> dict(clean: bool, hits: [(path, word)], files: [{path, sha256}])`.
    (a) the real readable material: docs/direction2-vuln-format.md and designer_2's prompt have 0 hits of the six words (the reference scan here
    is independent of the implementation).  (b) synthetic: a clean prompt + a clean format note + a THIRD material file containing 'EMA' (not
    the prompt) -> clean False and the hit names that third file (an implementation that scans only the prompt passes the prompt and misses it);
    every one of the six words is detected in a non-prompt file; (c) `files` lists every scanned file with its sha256 (the archived scan record);
    (d) extra words extend the list (e.g. a window length and a feature id); (e) all clean -> clean True and hits == []."""
    fmt = os.path.join(ROOT, "docs", "direction2-vuln-format.md")
    prm = os.path.join(_DESIGNER_ROOT, "designer_2", "prompt.txt")
    hits, _ = _scan_ref([fmt, prm], _S121_WORDS)
    assert hits == [], f"readable material for blind designers contains forbidden words: {hits}"
    scan = need(s1, "scan_blind_materials", "REQ-S1-24")
    r = scan([fmt, prm])
    assert r["clean"] is True and r["hits"] == []
    assert [f["path"] for f in r["files"]] == [fmt, prm]
    for f in r["files"]:
        assert f["sha256"] == hashlib.sha256(open(f["path"], "rb").read()).hexdigest()
    a, b, c = tmp_path / "prompt.txt", tmp_path / "format.md", tmp_path / "excerpt.md"
    a.write_text("Design a mutation of one hook.")
    b.write_text("Hook interface: audit probability, residual, CUSUM update, pause length.")
    c.write_text("Section excerpt: the features use an EMA of the own report.")
    r2 = scan([str(a), str(b), str(c)])
    assert r2["clean"] is False and [h[0] for h in r2["hits"]] == [str(c)] and r2["hits"][0][1] == "EMA"
    assert [f["path"] for f in r2["files"]] == [str(a), str(b), str(c)] and all(len(f["sha256"]) == 64 for f in r2["files"])
    for w in _S121_WORDS:
        c.write_text(f"material mentions {w} here")
        rr = scan([str(a), str(b), str(c)])
        assert rr["clean"] is False and rr["hits"] == [(str(c), w)], w
    c.write_text("the history window is 64 rounds, see f13")
    assert scan([str(a), str(b), str(c)])["clean"] is True
    rx = scan([str(a), str(b), str(c)], extra_words=("window", "f13"))
    assert rx["clean"] is False and sorted(h[1] for h in rx["hits"]) == ["f13", "window"] and {h[0] for h in rx["hits"]} == {str(c)}
    c.write_text("harmless")
    r3 = scan([str(a), str(b), str(c)])
    assert r3["clean"] is True and r3["hits"] == [] and len(r3["files"]) == 3


def test_S125_naive_pilot_always_report_one_baseline():
    """S1-25 (REQ-S1-25, v1.2): the NAIVE pilot also evaluates the 'always report 1' policy (v1 == 1) against the best handwritten strategy on the
    SAME T and SAME seeds, paired.  API fixed by this test: `s1.naive_pilot_baseline(Ts, seeds, hw_name, *, r=0.5, dev='cpu') -> dict`:
      by_T[T] = dict(T_score, seeds, always_one=dict(G_s[n_seeds], G), hw=dict(name, G_s[n_seeds], G), diff_s[n_seeds], diff, always_one_ge_hw: bool);
      T_change = diff(T_last) - diff(T_first).   NAIVE = M3C with p=0 (r=0.5; the configuration of env.anchor_config('NAIVE')).
    Reference values are computed here with env.simulate (honest adapter as the paired baseline; v1==1 via AdapterPolicy(b=1.0), because
    clip(u+1)=1).  Assertions: always_one.G_s equals the reference per seed (an honest/constant-d implementation gives 0 or another value; an
    unpaired mean would not match per seed); hw.G_s equals the reference of the named strategy; both use the same seeds and T_score (T_score =
    T - L for T > L, the direction-2 window); diff_s == always_one.G_s - hw.G_s; the flag follows diff >= 0; two T values give two entries and
    T_change; test seeds (5000..) are rejected (REQ-S1-20/SEED); the always-one gain is nonzero (so the reference is not trivially 0)."""
    from conftest import strategy_dict
    cfg = dict(env.m3c_config(0.5), p=0.0, L=200.0)       # NAIVE; short L keeps the CPU run small (the mechanism parameter L only sets the score window)
    seeds, Ts, name = [0, 1, 2, 3], [600, 900], "dz=0.3"
    ref = {}
    for T in Ts:
        out = env.simulate("M3C", T, cfg, [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(b=1.0)), policy.AdapterPolicy(strategy_dict(name))],
                           seeds, r=0.5, dev="cpu")
        U = out["snap"]["util_b"][:, :, 1].numpy()
        ref[T] = (out["T_score"], (U[1] - U[0]) / out["T_score"], (U[2] - U[0]) / out["T_score"])
        assert abs(ref[T][1].mean()) > 1e-6
    fn = need(s1, "naive_pilot_baseline", "REQ-S1-25")
    res = call_new_kw(fn, dict(cfg=cfg), Ts, seeds, name, r=0.5, dev="cpu")
    assert sorted(res["by_T"]) == sorted(Ts)
    for T in Ts:
        e = res["by_T"][T]
        ts_ref, g1, ghw = ref[T]
        assert e["T_score"] == ts_ref and list(e["seeds"]) == seeds
        assert np.allclose(e["always_one"]["G_s"], g1, rtol=0, atol=1e-12)
        assert e["hw"]["name"] == name and np.allclose(e["hw"]["G_s"], ghw, rtol=0, atol=1e-12)
        assert np.allclose(e["diff_s"], g1 - ghw, rtol=0, atol=1e-12) and abs(e["diff"] - (g1 - ghw).mean()) <= 1e-12
        assert abs(e["always_one"]["G"] - g1.mean()) <= 1e-12 and abs(e["hw"]["G"] - ghw.mean()) <= 1e-12
        assert e["always_one_ge_hw"] is bool(e["diff"] >= 0)
    assert abs(res["T_change"] - (res["by_T"][Ts[-1]]["diff"] - res["by_T"][Ts[0]]["diff"])) <= 1e-12
    for bad in ([5000, 5001], [0, 5000], [1000]):                  # test seeds / replay-only seeds are not allowed in the pilot
        with pytest.raises(ValueError):
            call_new_kw(fn, dict(cfg=cfg), Ts, bad, name, r=0.5, dev="cpu")


# ====================================================================================================================== spec v1.3 additions (T-70..T-74)
import ast
import copy
import textwrap

_RL_DIR = os.path.join(ROOT, "arbitration", "rl")
_CALIB = list(range(2000, 2032)) + list(range(3000, 3032))          # val U val2 (the 64 calibration seeds)


def _val2_refs(path):
    """Every USE of val2 in a module: a name/attribute/argument containing 'val2' (any case, VAL2...), the string 'val2' used as a subscript index, a call argument
    or a dict key (splits()['val2'], require_split(x, 'val2'), {'val2': ...}), or an integer literal in 3000..3031.  Running text and phrase tuples that merely
    mention val2 (e.g. the disclosure wording in stop.py) are not uses."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    is_v2 = lambda c: isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.lower() == "val2"
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and type(n.value) is int and 3000 <= n.value <= 3031:
            hits.append((n.lineno, repr(n.value)))
        elif isinstance(n, ast.Name) and "val2" in n.id.lower():
            hits.append((n.lineno, n.id))
        elif isinstance(n, ast.Attribute) and "val2" in n.attr.lower():
            hits.append((n.lineno, n.attr))
        elif isinstance(n, ast.arg) and "val2" in n.arg.lower():
            hits.append((n.lineno, n.arg))
        elif isinstance(n, ast.alias) and "val2" in (n.name + (n.asname or "")).lower():
            hits.append((0, n.name))
        elif isinstance(n, ast.Subscript) and is_v2(n.slice):
            hits.append((n.lineno, "['val2']"))
        elif isinstance(n, ast.Call) and any(is_v2(a) for a in list(n.args) + [k.value for k in n.keywords]):
            hits.append((n.lineno, "call(..., 'val2')"))
        elif isinstance(n, ast.Dict) and any(is_v2(k) for k in n.keys if k is not None):
            hits.append((n.lineno, "{'val2': ...}"))
    return hits


def test_T70_val2_calibration_only():
    """T-70 (REQ-SEED-08/01/03, v1.3): 3000-3031 is exactly val2; `require_split(..., "val2")` accepts it and rejects every other split's seeds (and every
    other split rejects val2); val2 is referenced ONLY by the calibration modules (seeds.py defines it, s1.py and vulns.py calibrate with it): an AST
    scan of every other arbitration/rl module (training, checkpoint selection, MDE, optimiser, budget, neighbours, ...) finds no val2 name, string or
    literal 3000..3031; at run time the training guard, the pilot guard and the MDE refuse val2, the checkpoint-selection validation function uses val
    only, and the test-evaluation entry refuses val2 seeds."""
    from arbitration.rl import final_eval, metrics, seeds, train
    sp = seeds.splits()
    assert list(sp["val2"]) == list(range(3000, 3032)) and len(sp["val2"]) == 32
    seeds.require_split(list(sp["val2"]), "val2")
    for other in ("train", "val", "test"):
        with pytest.raises(ValueError):
            seeds.require_split(list(sp["val2"]), other)
        with pytest.raises(ValueError):
            seeds.require_split(list(sp[other])[:2], "val2")
    for bad in ([2999], [3032], [1000], [3000, 3032], [5000]):
        with pytest.raises(ValueError):
            seeds.require_split(bad, "val2")
    # ---- who may reference val2
    allowed = {"seeds.py", "s1.py", "vulns.py"}
    mods = sorted(f for f in os.listdir(_RL_DIR) if f.endswith(".py") and f != "__init__.py")
    assert len(mods) >= 20 and allowed <= set(mods)
    offenders = {m: _val2_refs(os.path.join(_RL_DIR, m)) for m in mods if m not in allowed}
    offenders = {m: h for m, h in offenders.items() if h}
    assert offenders == {}, f"val2 is calibration-only (REQ-SEED-08 b): referenced in {offenders}"
    assert _val2_refs(os.path.join(_RL_DIR, "s1.py")), "the calibration module s1.py must be the one that uses val2"
    # the scanner itself: probes
    probe = lambda src: _val2_refs(_write_tmp(src))
    assert probe("x = list(range(3000, 3032))\n") and probe("v = seeds.splits()['val2']\n") and probe("from a import VAL2_SEEDS\n") and probe("def f(val2): pass\n")
    assert probe("require_split(x, 'val2')\n") and probe("d = {'val2': (3000, 32)}\n")
    assert probe("a = 2999\nb = 3032\nc = 'val'\n") == [] and probe("PH = ('val2', 'not a test set')\nTEXT = 'knob selection uses val2 seeds'\n") == []
    # ---- run-time guards
    for bad in ([3000], [3031], [0, 3000]):
        with pytest.raises(ValueError):
            train.require_train_seeds(bad)
        with pytest.raises(ValueError):
            s1.require_pilot_seeds(bad)
    with pytest.raises(ValueError):
        metrics.mde(np.ones(32), list(range(3000, 3032)))                    # the MDE may only come from validation seeds
    with pytest.raises(ValueError):
        metrics.mde(np.ones(64), _CALIB)                                      # not even val + val2 (32 validation seeds only, REQ-MET-06)
    metrics.mde(np.arange(32.0), list(range(2000, 2032)))
    val_src = inspect.getsource(train.make_val_fn)
    assert "val2" not in val_src.lower() and '"val"' in val_src
    with pytest.raises(ValueError, match="val2"):                             # REFUSES val2 (final_eval is implemented: not NotImplementedError, see T-27b)
        final_eval.evaluate_on_test(dict(seeds=list(range(3000, 3032))))


_TMP_COUNTER = [0]


def _write_tmp(src):
    import tempfile
    _TMP_COUNTER[0] += 1
    d = tempfile.mkdtemp()
    p = os.path.join(d, f"probe{_TMP_COUNTER[0]}.py")
    open(p, "w").write(src)
    return p


_SIX = {"out_of_family", "in_family", "in_family_known", "blind", "anchor", "contaminated_excluded"}


def test_T71_vuln_category_fields_and_consistency():
    """T-71 (REQ-S1-26/01/03, v1.3): every registry entry has a `category` from the six values; category=='blind' <=> blind==True; the assignment is fixed (D1, D2,
    O1 out_of_family; O2, O3, D4 in_family; D3 in_family_known; O5 blind; NAIVE, M4 anchor; designer_1 contaminated_excluded); the fields
    ratio_HW_over_Gstar, delta_fine, delta_coarse (and G_star_val, G_HW_best_val, size_basis, delta_val) are present and mutually consistent;
    out_of_family entries have a delta_fine CI lower bound > 0 and delta_fine in [0.8s, 1.2s]; in_family entries are sized by delta_coarse
    (size_basis 'coarse', delta_val == delta_coarse in [0.8s, 1.2s]) and their delta_fine may be <= 0; in_family_known has negative deltas;
    a contradiction is REPORTED (ValueError) and never repaired: the registry is untouched afterwards; contaminated_excluded never counts in runs/statistics."""
    check = need(s1, "check_registry_categories", "REQ-S1-26")
    reg = _registry()
    snapshot = copy.deepcopy(reg)
    assert check(reg) is True and reg == snapshot
    assert set(s1.CATEGORIES) == _SIX and dict(s1.CATEGORY_OF) == _CAT
    assert {e["id"]: e["category"] for e in reg} == _CAT
    assert all(e["blind"] == (e["category"] == "blind") for e in reg)
    for i in ("D1", "D2", "O1", "O2", "O3", "D4"):
        e = [x for x in reg if x["id"] == i][0]
        assert {"ratio_HW_over_Gstar", "delta_fine", "delta_coarse", "G_star_val", "G_HW_best_val", "size_basis", "delta_val"} <= set(e), i
    assert s1.size_basis("out_of_family") == "fine" and s1.size_basis("blind") == "fine" and s1.size_basis("in_family") == "coarse"
    assert s1.size_basis("in_family_known") is None
    for c in ("anchor", "contaminated_excluded"):
        assert s1.size_basis(c) is None
    # counted in the statistics: only the run categories; D3 and designer_1 never
    assert {i for i in _CAT if s1.counts_in_detection_statistics(i)} == {"D1", "D2", "O1", "O2", "O3", "D4"}
    assert not s1.counts_in_detection_statistics("D3") and not s1.counts_in_detection_statistics("designer_1") and not s1.counts_in_detection_statistics("O5")

    def mut(fn):
        r = copy.deepcopy(reg)
        fn(r)
        snap = copy.deepcopy(r)
        with pytest.raises(ValueError):
            check(r)
        assert r == snap, "a contradiction must be reported, not repaired"

    ent = lambda r, i, size=None: [e for e in r if e["id"] == i and (size is None or e["size"] == size)][0]
    mut(lambda r: ent(r, "D1", 0.02).pop("category"))
    mut(lambda r: ent(r, "D1", 0.02).update(category="weird"))
    mut(lambda r: ent(r, "D1", 0.02).update(category="in_family"))                      # fixed assignment
    mut(lambda r: ent(r, "O2", 0.02).update(category="out_of_family"))
    mut(lambda r: ent(r, "D3").update(category="in_family"))
    mut(lambda r: ent(r, "O5").update(blind=False))                                      # blind <=> category blind
    mut(lambda r: ent(r, "D1", 0.02).update(blind=True))
    for f in ("ratio_HW_over_Gstar", "delta_fine", "delta_coarse", "G_star_val", "G_HW_best_val", "size_basis", "delta_val"):
        mut(lambda r, f=f: ent(r, "O1", 0.02).pop(f))
    mut(lambda r: ent(r, "D3").pop("delta_coarse"))
    mut(lambda r: ent(r, "O1", 0.02).update(ratio_HW_over_Gstar=0.7))                    # inconsistent with G_HW / G*
    mut(lambda r: ent(r, "O1", 0.02)["delta_fine"].update(mean=0.0215, lo=0.02, hi=0.023))   # delta_fine no longer = G* - G_HW (G values untouched)
    # out_of_family: CI lower bound <= 0 -> stop and report (never reclassify)
    mut(lambda r: ent(r, "D2", 0.02)["delta_fine"].update(lo=-0.001))
    mut(lambda r: ent(r, "D2", 0.02)["delta_fine"].update(lo=0.0))
    mut(lambda r: ent(r, "D2", 0.02).update(size=0.02, delta_fine=_ci_ent(0.026, 0.001), G_star_val=_ci_ent(0.046), ratio_HW_over_Gstar=0.02 / 0.046,
                                              delta_val=_ci_ent(0.026, 0.001)))        # 0.026 > 1.2 * 0.02
    mut(lambda r: ent(r, "D2", 0.01).update(delta_fine=_ci_ent(0.007, 0.0004), G_star_val=_ci_ent(0.027), ratio_HW_over_Gstar=0.02 / 0.027,
                                              delta_val=_ci_ent(0.007, 0.0004)))       # 0.007 < 0.8 * 0.01
    mut(lambda r: ent(r, "D2", 0.02).update(size_basis="coarse"))                         # basis fixed by category
    mut(lambda r: ent(r, "D2", 0.02).update(delta_val=_ci_ent(0.0201, 0.0004)))           # delta_val must be the basis delta
    # in_family: coarse basis
    mut(lambda r: ent(r, "O2", 0.02).update(size_basis="fine"))
    mut(lambda r: ent(r, "O2", 0.02).update(delta_val=_ci_ent(-0.001, 0.0004)))
    mut(lambda r: ent(r, "O3", 0.02).update(delta_coarse=_ci_ent(0.03, 0.001), delta_val=_ci_ent(0.03, 0.001)))
    r_ok = copy.deepcopy(reg)
    ent(r_ok, "O2", 0.02)["delta_fine"].update(mean=-0.001, lo=-0.003, hi=0.001)         # delta_fine may be <= 0 for in_family (that is what in_family means)
    ent(r_ok, "O2", 0.02).update(G_star_val=_ci_ent(0.02 - 0.001), ratio_HW_over_Gstar=0.02 / 0.019)
    assert check(r_ok) is True
    mut(lambda r: ent(r, "D3")["delta_fine"].update(mean=0.01, lo=0.0, hi=0.02))          # D3: increments are negative (that is why it cannot be sized)
    # S1 statistics never contain D3 / designer_1 / O4 / O6 runs (see also S1-18)
    assert s1.schedule([0.01, 0.02]) and not {i for i, _ in s1.schedule([0.01, 0.02])} & {"D3", "designer_1"}


def test_T71b_real_registry_categories():
    """T-71b (REQ-S1-26 machine check on the REAL results/direction2/vuln_suite.json): after the GPU recalibration (val U val2, 36-name S_HW) every entry
    has its fixed category and consistent fine/coarse deltas, the 11 ids are present and the out_of_family entries pass the acceptance.  This is the
    acceptance test of the recalibration step; it is RED until that step has produced the v1.3 registry (the current file is the v1.2 one)."""
    reg = vulns.load_registry()
    assert s1.check_registry_composition(reg) is True
    assert s1.check_registry_categories(reg) is True


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["O1", "O2", "O3", "O5", "O4", "O6"])
def test_T72_vuln_identity_long_horizon_O_series(op):
    """T-72 (REQ-S1-05, GPU-08, v1.3): each O operator at its identity knob (O1/O2/O4/O6 kappa=0, O3 C=inf, O5 g=inf) in the direction-2 setting T=1e5,
    L=2500 equals the base environment BITWISE in every output field - for the direction-2 scoring window (snapshot at T-L, the default `T_score`) AND
    the direction-1 replay window (T_score=T, i.e. the final accumulators) - with at least one combination where `flags>0` (a suspension really
    happens) so that the identity is not trivial (T-36 shows the same operators ON change the outcome at short T).  One parametrised case per
    operator: run them as separate `gpujob` segments (each case <= 1500 s)."""
    cuda_required()
    cfg = dict(p=0.1, tol=6.0, L=2500.0, kref=0.5)
    pols = [policy.AdapterPolicy(d) for d in ({}, dict(dz=0.3), dict(dz=2.0), dict(b=0.1))]
    seeds = [0, 2000]
    T = 100000
    key = ("base", T, len(pols), tuple(seeds))
    if key not in _T72_BASE:
        _T72_BASE[key] = env.simulate("M3C", T, cfg, pols, seeds, r=0.5, dev="cuda")
    base = _T72_BASE[key]
    assert base["T_score"] == T - 2500
    v = vulns.simulate_vuln(op, OFF_KNOB[op], "M3C", T, cfg, pols, seeds, r=0.5, dev="cuda")
    assert v["T_score"] == T - 2500
    for f in FIELDS_V:
        assert v["snap"][f].equal(base["snap"][f]), f"{op} off, T_score=T-L: {f} differs from the base environment"
        assert v["final"][f].equal(base["final"][f]), f"{op} off, T_score=T: {f} differs from the base environment"
    assert (base["final"]["flags"] > 0).any() and (base["snap"]["flags"] > 0).any(), "no suspension in the base run: the identity check would be vacuous"
    vt = vulns.simulate_vuln(op, OFF_KNOB[op], "M3C", 2000, cfg, pols, seeds, r=0.5, dev="cuda", T_score=2000)         # explicit T_score=T path
    bt = env.simulate("M3C", 2000, cfg, pols, seeds, r=0.5, dev="cuda", T_score=2000)
    for f in FIELDS_V:
        assert vt["snap"][f].equal(bt["snap"][f]), f"{op} off, explicit T_score=T (short T): {f}"


_T72_BASE = {}


# ---------------------------------------------------------------------------------------------------------------- T-73: reference policies are public-only
_TIME_NAMES = {"t", "time", "round", "rounds", "step", "tick", "clock", "now"}
_PRIVATE_NAMES = {"cus", "pend_a", "pend_v", "flags", "audits", "sxy", "sxx", "u0", "u2", "audited_last"}
_LEGACY_LCLIMB = '''
class _LClimb(_LoopRule):
    def __init__(self, pt, B, dev):
        super().__init__(pt, B, dev)
        self.tr = self.zero()

    def report(self, c):
        dz = torch.minimum(self.p["kappa"] * (c["t"] - self.tr), self.p["dz_max"])
        return _lie(c["z1"], dz)

    def update(self, c):
        self.tr = torch.where(c["elig1"], self.tr, torch.full_like(self.tr, float(c["t"])))
'''


def _ast_violations(src):
    """Static check of REQ-S1-27(a): a parameter, name, attribute or key that is an absolute-time word or a private-state field."""
    bad = _TIME_NAMES | _PRIVATE_NAMES
    out = []
    for n in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(n, ast.arg) and n.arg in bad:
            out.append(("arg", n.arg))
        elif isinstance(n, ast.Name) and n.id in bad:
            out.append(("name", n.id))
        elif isinstance(n, ast.Attribute) and n.attr in bad:
            out.append(("attr", n.attr))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in bad:
            out.append(("key", n.value))
    return out


def _suite_rules():
    names = {e["ref_policy"]["name"] for e in vulns.load_registry() if e.get("ref_policy", {}).get("kind") == "rule"}
    return names | {"d2_rested", "d2_parole", "d2_climb", "streak_lie", "launder"}


_RULE_PARAMS = {"d2_rested": dict(dz=2.0, G=5.0, dz0=0.2), "d2_parole": dict(dz=0.5), "d2_climb": dict(kappa=0.002, dz_max=2.0, L=100.0),
                "streak_lie": dict(dz=2.0, m=3.0), "launder": dict(dz=0.7, W=100.0, Z=26.0)}
_RULE_CLASS = {"d2_rested": "out_of_family", "d2_parole": "out_of_family", "streak_lie": "out_of_family", "d2_climb": "in_family", "launder": "blind"}
# REQ-S1-27 scope (v1.3 final): a COUNTED vulnerability (out_of_family, in_family) reference rule must not use time; the blind O5 rule `launder` may use the agent's own clock
_TIME_FREE_CLASSES = ("out_of_family", "in_family")
_RULE_CASES = ["d2_rested", "d2_parole", "d2_climb", "streak_lie", "launder"]


def _hist(n, L, seed, prefix=None):
    """Random public history rows (u1, v1, won1, susp1): susp1 is the absolute end of the suspension that began at a flag (or 0)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n, 4))
    h[:, 0] = rng.uniform(0.02, 0.98, n)
    h[:, 1] = rng.uniform(0.0, 1.0, n)
    h[:, 2] = (rng.uniform(size=n) < 0.3).astype(float)
    cur, j = 0.0, 0
    flags = sorted(rng.choice(np.arange(20, n - 5), size=3, replace=False))
    for r in range(n):
        if r in flags:
            cur = float(r + 1 + L)
        h[r, 3] = cur
    return h


def _padded(h, n_p, kind):
    """Prepend n_p 'quiet' rows that change no public quantity of the original rows: never suspended (susp1 == 0), win pattern chosen so that every counter
    the rule keeps is in its initial state when the original history starts; absolute suspension ends are shifted by n_p."""
    rng = np.random.default_rng(99)
    pre = np.zeros((n_p, 4))
    pre[:, 0] = rng.uniform(0.02, 0.98, n_p)
    pre[:, 1] = rng.uniform(0.0, 1.0, n_p)
    pre[:, 2] = 1.0 if kind == "won" else 0.0
    sh = h.copy()
    sh[sh[:, 3] > 0, 3] += n_p
    return np.vstack([pre, sh])


_PREFIX_KIND = {"d2_rested": "won", "d2_parole": "any", "d2_climb": "any", "streak_lie": "none", "launder": "any"}


def _shift_variant(out_fn, h, n_p, kind):
    """True iff out_fn(h) differs from out_fn(prefix + h)[n_p:] (the rule depends on the absolute position)."""
    a = np.asarray(out_fn(h), float)
    b = np.asarray(out_fn(_padded(h, n_p, kind)), float)[n_p:]
    return not np.array_equal(a, b)


@pytest.mark.parametrize("rule", _RULE_CASES)
def test_T73_ref_policy_public_log_only_no_absolute_time(rule):
    """T-73 (REQ-S1-27, OBS-08/10/11/06, v1.3): every reference RULE of the suite (D1 rested, D2 parole, D4 climb, O1 streak, O5 launder; rule names are those of the
    registry) reads only public-log quantities; the COUNTED classes (out_of_family D1/D2/O1, in_family D4) also no absolute time, the blind O5 rule `launder` may use
    the agent's own clock (REQ-S1-27 scope, v1.3 final; its exemption is pinned: the clock use must be visible).  (a) static (AST): the loop implementation has no time word (t, time, round, step,
    tick, clock) and no private field name as parameter, name, attribute or key; the legacy absolute-t `_LClimb` source is detected (negative test);
    (b) run time: (1) private-state perturbation leaves the output unchanged (loop contexts with injected private tensors; the sandbox `private`
    argument), (2) TIME-SHIFT invariance of the public-history (sandbox) form: prepending n quiet rows (and shifting the absolute suspension ends) does
    not change the outputs of the original rounds - a rule that keys on the absolute index is flagged by the same helper (negative control);
    (3) [D4, see test_T73b] outputs equal the value rebuilt from the independent obs reconstructor's public-log inputs, bitwise."""
    assert rule in _suite_rules()
    cls = _RULE_CLASS[rule]
    counted = cls in _TIME_FREE_CLASSES
    for e in vulns.load_registry():                                  # the fixed class of the rule agrees with the registry once it carries categories
        if e.get("ref_policy", {}).get("name") == rule and e.get("category"):
            assert e["category"] == cls, (rule, e["id"], e["category"])
    # (a) static check on the loop implementation actually used inside the simulator
    src = inspect.getsource(vulns.LOOP_RULES[rule])
    viol = _ast_violations(src)
    if counted:
        assert viol == [], f"{rule} ({cls}): reads absolute time or private state: {viol}"
    else:                                                            # blind: own clock allowed, private fields never
        assert [v for v in viol if v[1] in _PRIVATE_NAMES] == [], f"{rule} ({cls}): reads private state: {viol}"
        assert any(v[1] in _TIME_NAMES for v in viol), f"{rule}: the blind exemption is stale (the rule no longer uses its clock): tighten T-73"
    # negative controls: the scanner flags the legacy absolute-t implementation and obvious private reads
    assert ("key", "t") in _ast_violations(_LEGACY_LCLIMB)
    assert _ast_violations("class X:\n    def report(self, c):\n        return c['cus'] + c.pend_a\n")
    assert _ast_violations("def f(self, t):\n    return t\n") and _ast_violations("def f(c):\n    return c['round']\n")
    assert _ast_violations("class X:\n    def report(self, c):\n        return c['elig1'] * c['z1']\n") == []
    # (b)(1) private perturbation, loop form: identical outputs with and without injected private-looking tensors in the context
    B, dev = 6, "cpu"
    tab = {k: torch.full((B,), float(v), dtype=torch.float64) for k, v in dict(vulns.RULE_DEFAULTS[rule], **_RULE_PARAMS[rule]).items()}
    rl_a, rl_b = vulns.LOOP_RULES[rule](tab, B, dev), vulns.LOOP_RULES[rule](tab, B, dev)
    g = torch.Generator().manual_seed(5)
    susp = torch.zeros(B, dtype=torch.float64)
    for t in range(1, 160):
        u = torch.rand(B, generator=g, dtype=torch.float64).clamp(0.02, 0.98)
        if t in (30, 90):
            susp = torch.full((B,), float(t + 100), dtype=torch.float64)
        ctx = dict(t=t, u1=u, z1=torch.special.ndtri(u), susp1=susp, elig1=(t > susp))
        junk = dict(ctx, cus=torch.rand(B, generator=g, dtype=torch.float64), pend_a=torch.randint(0, 3, (B,), generator=g), audits=torch.rand(B, generator=g),
                    flags=torch.rand(B, generator=g), sxy=torch.rand(B, generator=g), u0=torch.rand(B, generator=g), u2=torch.rand(B, generator=g))
        va, vb = rl_a.report(ctx), rl_b.report(junk)
        assert torch.equal(va, vb), f"{rule}: output depends on a private context field (round {t})"
        won = torch.rand(B, generator=g, dtype=torch.float64) < 0.3
        rl_a.update(dict(t=t, u1=u, elig1=ctx["elig1"], won1=won))
        rl_b.update(dict(junk, won1=won))
    h = _hist(260, 100.0, 1)
    entry = dict(ref_policy=dict(kind="rule", name=rule, params=_RULE_PARAMS[rule]), r=0.5)
    a = vulns.run_ref_policy_sandbox(entry, h)
    junk_p = dict(cus=np.random.default_rng(0).random(260), pend_a=np.random.default_rng(1).integers(0, 3, 260))
    assert np.array_equal(np.asarray(a), np.asarray(vulns.run_ref_policy_sandbox(entry, h, private=junk_p)))
    # (b)(2) time-shift invariance of the public-history form
    out_fn = lambda hh: vulns.run_ref_policy_sandbox(entry, hh)
    for n_p in (37, 211):
        if counted:
            assert not _shift_variant(out_fn, h, n_p, _PREFIX_KIND[rule]), f"{rule}: output changes when the same relative history is shifted in time by {n_p} rounds"
        else:                                                        # blind O5: the clock is the point; the shift MUST be visible (documented exemption)
            assert _shift_variant(out_fn, h, n_p, _PREFIX_KIND[rule]), f"{rule}: blind rule no longer reads its clock"
    # negative control of the helper: a rule keyed on the absolute index is flagged
    abs_rule = lambda hh: [min(0.002 * (j + 1), 0.9) for j in range(len(hh))]
    assert _shift_variant(abs_rule, h, 37, "any")


def test_T73b_d4_ref_policy_uses_since_f05_bitwise():
    """T-73b (REQ-S1-27, OBS-11, v1.3 final): the D4 reference policy `d2_climb` is a function of the age since RECOVERY max(since - L, 0), `since` = the rounds elapsed since the agent first observed its own
    susp_1 rise (never suspended: capped at 2L, i.e. f05 = 1).  Run it in the simulator (CPU, public log on) and rebuild every report from the PUBLIC
    LOG ONLY with the independent obs reconstructor (`obs_reconstruct.reconstruct`, which does not import the simulator): v1 == min(ndtr(z1 +
    min(kappa * (f05 * 2L), dz_max)), 1) bitwise for every round; the suspension happens (the climb restarts), so the check is not vacuous; the policy
    never reads absolute time (T-73); and the before/after change is visible: the old rule `min(kappa * (t - t_recovery), dz_max)` gives different reports."""
    from arbitration.rl import obs_reconstruct
    L, T = 100.0, 900
    cfg = dict(p=0.5, tol=2.0, L=L, kref=0.5)
    params = dict(kappa=0.004, dz_max=2.0, L=L)
    pols = [policy.AdapterPolicy({}), vulns.RulePolicy("d2_climb", params)]
    out = vulns.simulate_vuln(None, 0.0, "M3C", T, cfg, pols, [0, 1, 2], r=0.5, dev="cpu", public_log=True)
    plog = out["public_log"][1]                                  # [S, T, 4] = (u1, v1, won1, susp1) of the rule policy
    u1, v1, won1, susp1 = (plog[..., j] for j in range(4))
    assert (out["final"]["flags"][1, :, 1] > 0).all(), "the climbing agent must be suspended in every seed (otherwise the climb never restarts)"
    z1 = torch.special.ndtri(u1.clamp(1e-12, 1 - 1e-12))
    log = dict(z1=z1, u1=u1, v1=v1, won1=won1, susp1=susp1, consts=dict(r=0.5, L=L, p=0.5, h=2.0, kref=0.5))
    obs = obs_reconstruct.reconstruct(log, "public")
    f05 = obs[..., 4]
    assert (f05 == 1.0).any() and (f05 < 1.0).any() and (f05 == 0.0).any()
    age = torch.clamp(f05 * (2.0 * L) - L, min=0.0)                    # age since recovery = max(since - L, 0)
    assert (age == 0.0).any() and (age > 0.0).any() and (age == L).any(), "the test must exercise the suspended (age 0), recovering and never-suspended (age L) phases"
    dz = torch.minimum(params["kappa"] * age, torch.tensor(params["dz_max"], dtype=torch.float64))
    expected = torch.clamp(torch.special.ndtr(z1 + dz), max=1.0)
    assert torch.equal(expected, v1), f"D4 reports differ from the f05-based rebuild (max |diff| {(expected - v1).abs().max().item():.3e})"
    # the legacy absolute-t rule (exact re-statement: tr = last INELIGIBLE round index, dz = min(kappa * (t - tr), dz_max)) gives different reports
    t_idx = torch.arange(1, T + 1, dtype=torch.float64)[None, :].expand_as(v1)
    inelig = (t_idx <= susp1).to(torch.float64)
    tr_prev = torch.cat([torch.zeros_like(susp1[:, :1]), torch.cummax(inelig * t_idx, dim=1).values[:, :-1]], 1)
    legacy = torch.clamp(torch.special.ndtr(z1 + torch.minimum(params["kappa"] * (t_idx - tr_prev), torch.tensor(2.0, dtype=torch.float64))), max=1.0)
    assert not torch.equal(legacy, v1)
    assert (legacy[:, 0] != v1[:, 0]).all(), "round 1: the legacy rule starts the climb at 0, the recovery-age rule starts at kappa * L"
    # loop form and public-history (sandbox) form agree
    entry = dict(ref_policy=dict(kind="rule", name="d2_climb", params=params), r=0.5)
    h = np.stack([u1[0].numpy(), v1[0].numpy(), won1[0].numpy(), susp1[0].numpy()], 1)[:400]
    sb = np.asarray(vulns.run_ref_policy_sandbox(entry, h))
    assert np.abs(sb - v1[0, :400].numpy()).max() <= 1e-12


# ---------------------------------------------------------------------------------------------------------------- T-74: knob calibration on val U val2
def _eval_factory(mean_fn, sd, log, coarse_mean_fn=None):
    """Synthetic per-seed deltas for a knob: exact mean and SD (std_samples); records the (knob, seeds) of every call."""
    from conftest import std_samples

    def ev(knob, seeds):
        log.append((float(knob), list(seeds)))
        m = float(mean_fn(knob))
        c = float(coarse_mean_fn(knob)) if coarse_mean_fn is not None else m
        return dict(fine=std_samples(m, sd, n=len(seeds)), coarse=std_samples(c, sd, n=len(seeds)))
    return ev


def test_T74_knob_calibration_val_union_val2():
    """T-74 (REQ-S1-28/02, SEED-08, MET-06, v1.3): `s1.calibrate_knob_v13(op, size, category, eval_delta_s, grid, seeds)` calibrates on val U val2 (the 64 seeds
    2000-2031 + 3000-3031, in that order; anything else - train, val only, test, permuted, a different count - is a ValueError).  Acceptance: the
    basis delta's mean lies in [0.8 s, 1.2 s] AND its paired 95% CI lower bound (t(63)=1.9983, 64 seeds) is > 0; the basis is fine for out_of_family
    and blind and coarse for in_family; in_family_known (no size) and sizes outside {0.01, 0.02} are refused.  The search is a SCAN of the whole grid
    (no monotonicity assumption: a non-monotone tent function still finds the acceptor closest to the target, ties -> earlier grid point), is deterministic
    (same seeds + sequence => same knob), reports `needs_commander` instead of substituting, and records whether the knob response was monotone.
    The MDE is still computed from the 32 validation seeds only."""
    from conftest import std_samples
    from arbitration.rl import metrics
    cal = need(s1, "calibrate_knob_v13", "REQ-S1-28")
    assert list(s1.CALIB_SEEDS) == _CALIB and len(set(_CALIB)) == 64
    grid = [round(0.01 * i, 2) for i in range(101)]
    # ---- non-monotone response (tent): accepted set on both flanks, closest to the target wins, ties -> earlier
    tent = lambda k: 0.03 * math.exp(-(((k - 0.4) / (0.08 if k < 0.4 else 0.12)) ** 2))      # asymmetric: no exact ties between the flanks
    log = []
    r = cal("O1", 0.02, "out_of_family", _eval_factory(tent, 0.01, log), grid, _CALIB)
    assert [k for k, _ in log] == grid and all(sd == _CALIB for _, sd in log), "the whole grid is scanned (no bisection) with exactly the 64 calibration seeds"
    t63 = float(stats.t.ppf(0.975, 63))
    acc = [k for k in grid if 0.016 <= tent(k) <= 0.024 and tent(k) - t63 * 0.01 / 8 > 0]
    assert len(acc) >= 4 and min(acc) < 0.4 < max(acc), "the scenario must have acceptors on both flanks of the peak"
    best = min(acc, key=lambda k: (abs(tent(k) - 0.02), grid.index(k)))
    assert r["status"] == "ok" and r["knob"] == best and r["size_basis"] == "fine" and r["n_seeds"] == 64
    assert abs(r["delta"]["mean"] - tent(best)) < 1e-12 and r["delta"]["lo"] > 0
    assert r["knob_monotone"] is False
    lin = cal("O1", 0.02, "out_of_family", _eval_factory(lambda k: 0.05 * k, 0.005, []), grid, _CALIB)
    assert lin["status"] == "ok" and lin["knob_monotone"] is True and abs(0.05 * lin["knob"] - 0.02) <= 0.0005
    # deterministic
    again = cal("O1", 0.02, "out_of_family", _eval_factory(tent, 0.01, []), grid, _CALIB)
    assert again == r
    plateau = cal("O1", 0.02, "out_of_family", _eval_factory(lambda k: 0.02, 0.004, []), grid, _CALIB)
    assert plateau["status"] == "ok" and plateau["knob"] == grid[0], "exact ties go to the earlier grid point"
    # ---- the CI lower bound is part of the acceptance (merged 64-seed CI, t(63)): m/sd threshold 1.9983/sqrt(64)=0.24979
    for ratio, ok in ((0.2505, True), (0.2490, False), (0.11, False), (0.5, True)):
        sd = 0.02 / ratio
        rr = cal("O1", 0.02, "out_of_family", _eval_factory(lambda k: 0.02, sd, []), [0.3], _CALIB)
        assert (rr["status"] == "ok") is ok, (ratio, rr["status"])
        if not ok:
            assert "knob" not in rr
    # the 32-seed (val only) criterion would have decided differently for ratio 0.2505 (t(31)/sqrt(32)=0.3605): the 64-seed CI is the one used
    assert 2.0395 / math.sqrt(32) > 0.2505 > 1.9983 / math.sqrt(64)
    # ---- basis by category
    fine_bad = lambda k: 0.0                    # fine delta does not reach the size ...
    coarse_good = lambda k: 0.02                # ... the coarse delta does
    both = _eval_factory(fine_bad, 0.004, [], coarse_mean_fn=coarse_good)
    rin = cal("O2", 0.02, "in_family", both, [0.2], _CALIB)
    assert rin["status"] == "ok" and rin["size_basis"] == "coarse" and abs(rin["delta"]["mean"] - 0.02) < 1e-12
    assert cal("O1", 0.02, "out_of_family", both, [0.2], _CALIB)["status"] == "needs_commander"
    assert cal("O5", 0.02, "blind", both, [0.2], _CALIB)["status"] == "needs_commander"
    assert cal("O5", 0.02, "blind", _eval_factory(lambda k: 0.02, 0.004, []), [0.2], _CALIB)["size_basis"] == "fine"
    # ---- failure is reported, never substituted
    fail = cal("O3", 0.02, "in_family", _eval_factory(lambda k: 0.001 * k, 0.002, []), grid, _CALIB)
    assert fail["status"] == "needs_commander" and "knob" not in fail and fail["op"] == "O3"
    # ---- guards
    ev = _eval_factory(lambda k: 0.02, 0.004, [])
    for bad_seeds in (list(range(64)), list(range(2000, 2032)), list(range(3000, 3032)), _CALIB[:-1], list(reversed(_CALIB)),
                      list(range(2000, 2032)) + list(range(5000, 5032)), _CALIB + [5000]):
        with pytest.raises(ValueError):
            cal("O1", 0.02, "out_of_family", ev, grid, bad_seeds)
    for size in (0.003, 0.005, 0.015, 0.0):
        with pytest.raises(ValueError):
            cal("O1", size, "out_of_family", ev, grid, _CALIB)
    for cat in ("in_family_known", "anchor", "contaminated_excluded", "weird"):
        with pytest.raises(ValueError):
            cal("D3", 0.02, cat, ev, grid, _CALIB)
    # ---- the MDE stays on the 32 validation seeds
    with pytest.raises(ValueError):
        metrics.mde(std_samples(0.01, 0.01, n=64), _CALIB)
    assert metrics.mde(std_samples(0.01, 0.01, n=32), list(range(2000, 2032))) > 0
    sig = inspect.signature(cal)
    assert "seeds" in sig.parameters and not any("test" in p for p in sig.parameters)
