"""Acceptance for formal main evaluation; synthetic CPU data, training-only GPU preflight."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from conftest import cuda_required
from arbitration.rl import main_eval as ev, metrics, policy, seeds

SETTINGS = dict(F=25, hidden=2, T_val=100000, cfg=dict(p=.1, tol=6., L=2500., kref=.5),
                r=.5, mech="M3C", operator=None, knob=None, obs_version="public")
CELL = dict(name="main_M3C_r0.5", r=.5, group="main", run_ids=[0, 1, 2], G_gens=416)


def checkpoints():
    rows = []
    for i, v in enumerate([.06, .058, .057]):
        th = np.zeros(55)
        th[0] = i + 1
        rows.append(dict(cell=CELL["name"], run_id=i, gen=350, G_val_train=v, theta=th,
                         theta_sha256=hashlib.sha256(th.astype("<f8").tobytes()).hexdigest(), gens_done=416))
    return rows


class FakeRun:
    def __init__(self, rl=(.06, .058, .057), negative_hw=False, honest_shift=0):
        self.calls = []
        self.rl, self.negative_hw, self.honest_shift = rl, negative_hw, honest_shift

    def __call__(self, pols, sds, use_graph=False):
        self.calls.append((list(sds), use_graph))
        S = np.asarray(sds, float)
        noise = .003 * np.sin(S * .35)
        out = []
        for p in pols:
            if isinstance(p, policy.MLPPolicy):
                g = np.zeros_like(S) if not np.any(p.theta) else self.rl[int(p.theta[0]) - 1] + noise
            elif not p.scols:
                g = np.zeros_like(S)
            else:
                g = np.full_like(S, .008 * min(p.scols.get("dz", 0), 4) + .001) + .2 * noise
                if self.negative_hw:
                    g -= .08
            out.append(g)
        g = np.asarray(out)
        U = np.full((len(pols), len(S), 3), 18000.)
        U[:, :, 1] += g * 97500
        if isinstance(pols[0], policy.AdapterPolicy):
            U += self.honest_shift
        return dict(T_score=97500, snap=dict(util_b=U, reg_b=5000. - g * 1000.,
                                            pen=np.broadcast_to(np.arange(3.), U.shape).copy()))


def setup(root, run=None):
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/direction2-freeze.md").write_text("test freeze")
    cks = checkpoints()
    runner = run or FakeRun()
    val = ev.validation_stage(str(root), CELL, SETTINGS, cks, runner, check_consistency=False)
    return cks, val


def test_validation_selects_all_runs_and_freezes_paired_mdes(tmp_path):
    fake = FakeRun()
    cks, val = setup(tmp_path, fake)
    assert all(s == list(seeds.splits()["val"]) for s, _ in fake.calls)
    assert val["selected_run"] == 0
    assert val["selected_hw"]["public"] == "dz=4.0"
    assert len(val["runs"]) == 3
    rl = ev.read(ev.path(str(tmp_path), CELL["name"], "val_rl_raw"))
    hw = ev.read(ev.path(str(tmp_path), CELL["name"], "val_hw_raw"))
    gh = ev.gains(hw)[hw["contract"]["policies"].index("dz=4.0")]
    for i, r in enumerate(val["runs"]):
        d = ev.gains(rl)[i + 1] - gh
        half = stats.t.ppf(.975, 31) * np.std(d, ddof=1) / np.sqrt(32)
        assert r["D_val"]["lo"] == pytest.approx(d.mean() - half)
        assert r["mde_d_frozen"] == pytest.approx(metrics.kappa() * np.std(d, ddof=1) / np.sqrt(32))
    assert ev.validation_stage(str(tmp_path), CELL, SETTINGS, cks, FakeRun()) == val
    pub, ora = ev.candidate_strategies()
    assert len(pub) == 36 and len(ora) == 8
    assert all(not d for n, d in ora if n.startswith("quota_"))


def test_test_uses_frozen_selection_and_mde_and_paired_ci(tmp_path, monkeypatch):
    cks, val = setup(tmp_path)
    # Test ranking differs from validation; the preselected main run must remain run0.
    fake = FakeRun(rl=(.05, .10, .049))
    monkeypatch.setattr(metrics, "mde", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test must not recompute MDE")))
    result = ev.test_stage(str(tmp_path), CELL, SETTINGS, cks, fake, val)
    assert result["selected_run"] == 0 and result["runs"][1]["G_RL"]["mean"] > result["runs"][0]["G_RL"]["mean"]
    assert all(s == list(seeds.splits()["test"]) for s, _ in fake.calls)
    raw = ev.read(ev.path(str(tmp_path), CELL["name"], "test_raw"))
    for i, r in enumerate(result["runs"]):
        d = np.array(raw["rl_gains"][i]) - np.array(raw["hw_gains"]["dz=4.0"])
        half = stats.t.ppf(.975, 31) * np.std(d, ddof=1) / np.sqrt(32)
        assert r["D"]["mean"] == pytest.approx(d.mean())
        assert r["D"]["lo"] == pytest.approx(d.mean() - half)
        assert r["D"]["hi"] == pytest.approx(d.mean() + half)
        assert r["MDE_D"] == val["runs"][i]["mde_d_frozen"]
    assert result["replication_k"] == 3
    assert result["honest_system"]["mean_ineligible_fraction"]["mean"] == pytest.approx(1/97500)


def test_resume_after_ledger_write_never_resimulates(tmp_path):
    cks, val = setup(tmp_path)
    fake = FakeRun()
    first = ev.test_stage(str(tmp_path), CELL, SETTINGS, cks, fake, val)
    assert len(fake.calls) == 2
    ledger = ev.read(tmp_path / "results/direction2/final_eval_ledger.json")
    assert len(ledger) >= 4
    Path(ev.path(str(tmp_path), CELL["name"], "test")).unlink()
    def fail(*a, **k):
        raise AssertionError("resume must use raw data")
    again = ev.test_stage(str(tmp_path), CELL, SETTINGS, cks, fail, val)
    assert first == again
    assert ev.read(tmp_path / "results/direction2/final_eval_ledger.json") == ledger


def test_validation_mismatch_and_tampering_are_rejected(tmp_path):
    cks, val = setup(tmp_path)
    bad = copy.deepcopy(val)
    bad["runs"][0]["mde_d_frozen"] += .1
    fake = FakeRun()
    with pytest.raises(ValueError, match="modified"):
        ev.test_stage(str(tmp_path), CELL, SETTINGS, cks, fake, bad)
    other = copy.deepcopy(cks)
    other[0]["theta_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoints"):
        ev.test_stage(str(tmp_path), CELL, SETTINGS, other, fake, val)
    assert fake.calls == []
    p = Path(ev.path(str(tmp_path), CELL["name"], "val_hw_raw"))
    raw = ev.read(p)
    raw["contract"]["seeds"][0] = 99
    p.write_text(json.dumps(raw))
    Path(ev.path(str(tmp_path), CELL["name"], "val")).unlink()
    with pytest.raises(ValueError, match="contract mismatch"):
        ev.validation_stage(str(tmp_path), CELL, SETTINGS, cks, fake, check_consistency=False)


def test_negative_baseline_is_not_clipped_and_h0_is_separate(tmp_path):
    fake = FakeRun(negative_hw=True)
    cks, val = setup(tmp_path, fake)
    assert val["selected_hw"]["public"] == "dz=4.0"
    assert val["selected_hw"]["plus_h0"] == "honest"
    result = ev.test_stage(str(tmp_path), CELL, SETTINGS, cks, fake, val)
    assert result["G_HW"]["public"]["mean"] < 0
    assert result["G_HW"]["plus_h0"]["mean"] == 0
    assert result["runs"][0]["D"]["mean"] > result["runs"][0]["D_plus_h0"]["mean"]


def test_honest_cross_batch_mismatch_stops_before_test(tmp_path):
    with pytest.raises(RuntimeError, match="honest baselines"):
        setup(tmp_path, FakeRun(honest_shift=1.))


@pytest.mark.gpu
def test_main_snapshot_graph_matches_eager_on_training_seeds():
    cuda_required()
    from arbitration.rl import graphsim
    st = dict(SETTINGS, T_val=400, cfg=dict(SETTINGS["cfg"], L=30., tol=2.))
    rng = np.random.default_rng(19)
    pols = [policy.MLPPolicy(obs_version="public")]
    pols += [policy.MLPPolicy(theta=rng.normal(0, .1, 55), obs_version="public") for _ in range(3)]
    run = ev.make_runner(st, "cuda", graphsim.from_env("cuda", force=True))
    a, b = run(pols, [11, 12], False), run(pols, [11, 12], True)
    assert a["T_score"] == b["T_score"] == 370
    for key in ev.SNAP_FIELDS:
        assert np.array_equal(a["snap"][key], b["snap"][key]), key
