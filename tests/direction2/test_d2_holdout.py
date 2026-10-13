"""Independent holdout guards; synthetic CPU data and training-only GPU preflight."""
import copy

import numpy as np
import pytest
from scipy import stats

from conftest import cuda_required
from arbitration.rl import env, graphsim, policy
from scripts import d2_holdout_eval as h


def fake_raw(contrasts):
    contrasts=np.asarray(contrasts,float)
    p,n=contrasts.shape
    u=np.full((p,n,3),18000.)
    u[:,:,1]+=contrasts*97500
    return dict(T_score=97500,snap=dict(util_b=u,reg_b=np.full((p,n),500.),pen=np.zeros((p,n,3))))


def test_paired_estimator_frozen_main_run_and_three_run_family():
    # Large common noise cancels in the paired contrast; selecting the best
    # holdout run instead of the registered run would incorrectly change verdict.
    a=np.linspace(-.05,.05,64)
    hbase=.02+a
    dr=np.array([.006+.2*a,.04+.1*a,.009+.2*a])
    rl=fake_raw(np.vstack([np.zeros(64),hbase+dr]))
    hand=fake_raw(np.vstack([np.zeros(64),hbase]))
    rl['sha256']='synthetic_rl';hand['sha256']='synthetic_hw'
    m=dict(selected_run=0,delta=.005,sha256='synthetic',seeds=list(range(64)))
    r=h.summarize(m,rl,hand)
    assert r['selected_run']==0
    for i in range(3):
        se=np.std(dr[i],ddof=1)/8
        assert r['runs'][i]['D']['mean']==pytest.approx(dr[i].mean())
        assert r['runs'][i]['D']['half']==pytest.approx(stats.t.ppf(.975,63)*se)
        assert r['runs'][i]['D_simultaneous_95']['half']==pytest.approx(stats.t.ppf(1-.05/6,63)*se)
    assert r['verdict']==h.verdict(r['runs'][0]['D'],.005)
    assert r['verdict']=='confirmed_positive_only'
    hand['snap']['util_b'][0,0,0]+=1
    with pytest.raises(RuntimeError):
        h.summarize(m,rl,hand)


def test_verdict_strict_boundaries_and_inconclusive_is_not_zero():
    assert h.verdict(dict(lo=.005001,hi=.01),.005)=='confirmed_material_advantage'
    assert h.verdict(dict(lo=.005,hi=.01),.005)=='confirmed_positive_only'
    assert h.verdict(dict(lo=.001,hi=.004),.005)=='confirmed_positive_only'
    assert h.verdict(dict(lo=-.002,hi=.004),.005)=='material_advantage_ruled_out'
    assert h.verdict(dict(lo=-.01,hi=-.001),.005)=='material_advantage_ruled_out'
    assert h.verdict(dict(lo=0,hi=.005),.005)=='inconclusive'
    assert h.verdict(dict(lo=-.001,hi=.01),.005)=='inconclusive'


def test_seed_audit_detects_base_and_shifted_stream_collisions():
    h.require_disjoint([100,101],{0,1})
    for new,old in [([1],{1}),([1000001],{1}),([3000001],{1}),([1,1000001],{7}),([1,1],{0})]:
        with pytest.raises(ValueError):h.require_disjoint(new,old)
    values=h.seed_values(dict(seed_range=[5,8],nested=[dict(train_seed=17),dict(seeds=[11,13])],irrelevant=[42]))
    assert values=={5,6,7,8,11,13,17}


def test_legacy_infinity_metadata_is_finite_json_without_mutating_cfg():
    cfg=dict(ad_h=float('inf'),L=2500.)
    result=h.encode_settings(cfg)
    assert cfg['ad_h']==float('inf')
    assert result['ad_h']==dict(float_sentinel='Infinity')
    assert h.digest(result)
    with pytest.raises(ValueError):h.encode_settings(dict(x=float('nan')))


def contract():
    return dict(manifest_sha256='synthetic',stage='rl',seeds=[11,12],n_policies=2,T_score=97500)


def test_completed_or_saved_raw_is_never_resimulated(tmp_path):
    calls=[]
    def runner():
        calls.append(1)
        return fake_raw([[0,0],[.01,.02]])
    con=contract()
    original=h.once(tmp_path,'rl',con,runner)
    assert len(calls)==1
    assert h.once(tmp_path,'rl',con,runner)==original
    ledger=h.read(tmp_path/'ledger.json')
    ledger['rl']['status']='started'
    ledger['rl'].pop('raw_sha256')
    h.write(tmp_path/'ledger.json',ledger)
    assert h.once(tmp_path,'rl',con,runner)==original
    assert len(calls)==1
    changed=dict(con,seeds=[13,14])
    with pytest.raises(ValueError):h.once(tmp_path,'rl',changed,runner)


def test_interrupted_attempt_without_raw_stops(tmp_path):
    def failed():raise RuntimeError('synthetic technical interruption')
    with pytest.raises(RuntimeError,match='technical interruption'):
        h.once(tmp_path,'hw',contract(),failed)
    assert h.read(tmp_path/'ledger.json')['hw']['status']=='started'
    with pytest.raises(RuntimeError,match='already started'):
        h.once(tmp_path,'hw',contract(),lambda:pytest.fail('must not rerun'))


def test_raw_checksum_scoring_window_shape_and_nonfinite_guards(tmp_path):
    con=contract()
    raw=h.once(tmp_path,'rl',con,lambda:fake_raw([[0,0],[.01,.02]]))
    bad=copy.deepcopy(raw);bad['snap']['util_b'][1][0][1]+=1
    with pytest.raises(ValueError,match='checksum'):h.validate_raw(bad,con)
    for change in ('shape','window'):
        bad=copy.deepcopy(raw)
        if change=='shape':bad['snap']['reg_b'][0].pop()
        else:bad['T_score']-=1
        bad['sha256']=h.digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError):h.validate_raw(bad,con)
    with pytest.raises(ValueError):h.interval([1,float('nan')])


@pytest.mark.gpu
def test_gpu_frozen_policies_graph_eager_and_honest_paths_on_training_seeds():
    import torch
    cuda_required()
    st,cks,_=h.context()
    cfg=dict(st['cfg'],L=30.)
    pols=[policy.MLPPolicy(obs_version='public')]+[policy.MLPPolicy(theta=c['theta'],obs_version='public') for c in cks]
    args=dict(mech='M3C',T=400,cfg=cfg,seeds=[11,12],r=.5,dev='cuda')
    eager=env.simulate(policies=pols,**args)
    graph=env.simulate(policies=pols,graph=graphsim.from_env('cuda',force=True),**args)
    hand=env.simulate(policies=[policy.AdapterPolicy({}),policy.AdapterPolicy(dict(dz=.3))],**args)
    for part in ('snap','final'):
        for key in eager[part]:
            assert torch.equal(eager[part][key],graph[part][key]),(part,key)
            assert torch.equal(eager[part][key][0],hand[part][key][0]),(part,key)
