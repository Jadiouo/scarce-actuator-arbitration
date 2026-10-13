"""Post-main diagnostic checks; real simulator preflight uses training seeds only."""
import copy
import json

import numpy as np
import pytest
from scipy import stats

from conftest import cuda_required
from arbitration.rl import env, graphsim, obs_reconstruct, policy
from scripts import d2_postmain_diag as diag


def test_masking_removes_only_the_selected_input_contribution():
    import torch
    p = policy.MLPPolicy(theta=np.linspace(-.2, .4, 55), obs_version="public")
    original = p.theta.copy()
    x = torch.linspace(-1, 1, 100, dtype=torch.float64).reshape(4, 25)
    assert len(diag.GROUPS) == 7
    assert sorted(i for g in diag.GROUPS[:6] for i in g) == list(range(25))
    for group in diag.GROUPS:
        q = diag.masked(p, group)
        zeroed = x.clone()
        zeroed[:, group] = 0
        assert torch.equal(q.action(x), p.action(zeroed))
        np.testing.assert_array_equal(q.theta[50:], original[50:])
    np.testing.assert_array_equal(p.theta, original)


def test_paired_interval_and_family_correction():
    a = np.arange(32.) / 100
    b = .9 * a + .005
    d = a-b
    got = diag.interval(d)
    h = stats.t.ppf(.975,31) * d.std(ddof=1) / np.sqrt(32)
    assert got["lo"] == pytest.approx(d.mean()-h)
    corrected = diag.interval(d,21)
    h21 = stats.t.ppf(1-.05/42,31) * d.std(ddof=1) / np.sqrt(32)
    assert corrected["half"] == pytest.approx(h21)
    assert corrected["half"] > got["half"]
    with pytest.raises(ValueError):
        diag.interval([1,np.nan])


def test_cache_rejects_changed_contract_and_tampered_data(tmp_path):
    p = tmp_path / "cache.json"
    con = dict(stage="synthetic", seeds=[2000,2001])
    d = dict(contract=con, value=[.1,.2])
    d["sha256"] = diag.digest(d)
    diag.write(p,d)
    assert diag.cached(p,con)==d
    with pytest.raises(ValueError):
        diag.cached(p,dict(con, stage="another"))
    d["value"][0] = .9
    diag.write(p,d)
    with pytest.raises(ValueError):
        diag.cached(p,con)


def test_behavior_groups_use_start_of_round_eligibility_and_seed_unit():
    arrays=dict(u1=np.array([[[.2,.4,.6,.8],[.2,.4,.6,.8]]]),
                v1=np.array([[[.3,.3,.6,.9],[.3,.3,.6,.9]]]),
                won1=np.array([[[True,False,True,False],[False,True,False,True]]]),
                susp1=np.array([[[0.,3.,3.,3.],[0.,0.,0.,0.]]]))
    rows=diag.behavior_summary(arrays,["p"],0,4)
    el=next(r for r in rows if r["group"]=="eligible")
    assert [r["count"] for r in el["by_seed"]] == [2,4]
    assert el["count"] == 6
    assert el["by_seed"][0]["mean_delta"] == pytest.approx(.1)
    sus=next(r for r in rows if r["group"]=="suspended")
    assert sus["count"]==2 and sus["mean_delta"]==pytest.approx(-.05)
    assert sus["by_seed"][1]["mean_delta"] is None


@pytest.mark.gpu
def test_postmain_gpu_graph_masking_tail_boundary_and_public_trace():
    import torch
    cuda_required()
    cfg=dict(p=.1, tol=6., L=30., kref=.5)
    p=policy.MLPPolicy(theta=np.linspace(-.3,.3,55),obs_version="public")
    pols=[policy.MLPPolicy(obs_version="public"), p] + [diag.masked(p,g) for g in diag.GROUPS]
    kw=dict(mech="M3C",T=400,cfg=cfg,seeds=[11,12],r=.5,dev="cuda")
    a=env.simulate(policies=pols,**kw)
    b=env.simulate(policies=pols,graph=graphsim.from_env("cuda",force=True),**kw)
    for part in ("snap","final"):
        for key in a[part]:
            assert torch.equal(a[part][key],b[part][key]),(part,key)
    mixed=[policy.AdapterPolicy({}),p,policy.AdapterPolicy(dict(dz=.3))]
    plain=env.simulate(policies=mixed,**kw)
    log=env.simulate(policies=mixed,diagnostic=True,**kw)
    tail=env.simulate(policies=mixed,diagnostic=True,diagnostic_variant="tail_honest",**kw)
    for part in ("snap","final"):
        for key in plain[part]:
            assert torch.equal(plain[part][key],log[part][key])
            assert torch.equal(log[part][key][0],tail[part][key][0])
    assert torch.equal(log["log"]["v1"][...,:340],tail["log"]["v1"][...,:340])
    assert torch.equal(tail["log"]["v1"][...,340:],tail["log"]["u1"][...,340:])
    rebuilt=obs_reconstruct.reconstruct(env.extract_public_log(log))
    assert torch.equal(rebuilt,log["obs"])
