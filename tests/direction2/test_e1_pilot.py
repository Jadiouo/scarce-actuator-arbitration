"""E1 pilot guards and resumption; GPU checks use training seeds only."""
import copy
import numpy as np
import pytest
from conftest import cuda_required
from arbitration.rl import graphsim,policy
from scripts import e1_pilot as e


def fake(cfg,seeds):
    u=np.full((40,len(seeds),3),100.)
    u[1:,:,1]+=(.02+.002*(cfg['tol']-6))*15000
    return dict(T_score=15000,wall_s=.01,contract=dict(seeds=seeds),snap=dict(util_b=u.tolist(),
                reg_b=np.full((40,len(seeds)),100.).tolist(),pen=np.zeros((40,len(seeds),3)).tolist()))


def test_parameter_bounds_original_and_fixed_audit_budget():
    assert e.decode([0,0,-.25])==e.BASE_CFG
    assert e.decode([-99,-99,-99])==dict(p=.1,tol=4.,kref=.35,L=1000.)
    assert e.decode([99,99,99])==dict(p=.1,tol=8.,kref=.65,L=5000.)
    with pytest.raises(ValueError):e.decode([0,0,float('nan')])


def test_objective_same_mechanism_honesty_constraints_and_pool_separation():
    raw=fake(e.BASE_CFG,[1,2]);base=copy.deepcopy(raw)
    # Pooled includes high-gain frozen MLP, HW objective must ignore its ranking.
    raw['snap']['util_b'][39][0][1]+=15000*.4
    a=e.objective(raw,base,'hw');b=e.objective(raw,base,'pooled')
    assert a['worst_G']==pytest.approx(.02)
    assert b['worst_G']==pytest.approx(.22)
    assert a['feasible']
    raw['snap']['reg_b'][0]=[1000.,1000.]
    bad=e.objective(raw,base,'hw')
    assert not bad['feasible'] and bad['fitness']<a['fitness']
    raw=copy.deepcopy(base);raw['snap']['pen'][0]=[[1500.]*3]*2
    assert not e.objective(raw,base,'hw')['feasible']
    raw['contract']['seeds']=[3,4]
    with pytest.raises(ValueError):e.objective(raw,base,'hw')


def test_seed_router_rejects_old_test_holdout_and_wrong_split():
    m=dict(validation_seeds=[700,701])  # synthetic configuration only
    e.require_seeds([11,12],'train',m);e.require_seeds([700,701],'validation',m)
    for sd,stage in [([2000],'train'),([41_000_000],'train'),([11],'validation'),([11,11],'train')]:
        with pytest.raises(ValueError):e.require_seeds(sd,stage,m)


def test_raw_cache_checks_configuration_shape_window_and_hash():
    raw=fake(e.BASE_CFG,[1,2])
    con=dict(seeds=[1,2],T=20000,exclude=5000,n_policies=40,cfg=e.BASE_CFG)
    raw['contract']=con;raw['sha256']=e.digest(raw)
    assert e.validate_snapshot(raw,con)==raw
    bad=copy.deepcopy(raw);bad['T_score']=17500;bad['sha256']=e.digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):e.validate_snapshot(bad,con)
    with pytest.raises(ValueError):e.validate_snapshot(raw,dict(con,cfg=dict(e.BASE_CFG,L=1000)))


def test_defender_resumes_pending_population_without_changing_optimizer(tmp_path,monkeypatch):
    m=dict(sha256='synthetic',validation_seeds=[21,22],defender_gens=1)
    counter=[0];fail=[True]
    def evaluate(m,cfg,sd,*args,**kwargs):
        counter[0]+=1
        if fail[0] and counter[0]==5:raise RuntimeError('synthetic interrupt')
        return fake(cfg,sd),'fake/'+e.digest(cfg)
    monkeypatch.setattr(e,'evaluate',evaluate);monkeypatch.setattr(e,'publish',lambda *a:None)
    monkeypatch.setattr(e,'stamp_status',lambda *a,**k:None)
    monkeypatch.setattr(e,'OUT',tmp_path/'resumed')
    with pytest.raises(RuntimeError):e.defender(m,'hw',0,[],[])
    checkpoint=e.read(e.OUT/'defender/hw_run0.json')
    assert len(checkpoint['pending']['done'])==3
    fail[0]=False
    resumed=e.defender(m,'hw',0,[],[])
    monkeypatch.setattr(e,'OUT',tmp_path/'fresh')
    fresh=e.defender(m,'hw',0,[],[])
    assert resumed==fresh


def test_attacker_resume_preserves_population_rng_and_checkpoint(tmp_path,monkeypatch):
    m=dict(sha256='synthetic',validation_seeds=[21,22],attacker_gens=3,T_train=20000)
    counter=[0];fail=[True]
    def sim(cfg,pols,sd,T,*args,**kwargs):
        counter[0]+=1
        if fail[0] and counter[0]==3:raise RuntimeError('synthetic interruption')
        u=np.full((len(pols),len(sd),3),100.)
        for i,p in enumerate(pols):u[i,:,1]+=float(np.tanh(p.theta.sum()))*10
        return dict(T_score=15000,snap=dict(util_b=u.tolist(),reg_b=np.zeros((len(pols),len(sd))).tolist(),pen=np.zeros_like(u).tolist()))
    def evaluate(m,cfg,sd,stage,pols,*args):return sim(cfg,pols,sd,20000),'fake'
    monkeypatch.setattr(e,'simulate',sim);monkeypatch.setattr(e,'evaluate',evaluate)
    monkeypatch.setattr(e.graphsim,'from_env',lambda *a,**k:None)
    monkeypatch.setattr(e,'publish',lambda *a:None);monkeypatch.setattr(e,'stamp_status',lambda *a,**k:None)
    monkeypatch.setattr(e.time,'time',lambda:42.)
    monkeypatch.setattr(e,'OUT',tmp_path/'resumed')
    cks=[dict(theta=np.full(55,.001))]
    with pytest.raises(RuntimeError):e.adapt(m,'base',e.BASE_CFG,0,cks)
    assert e.read(e.OUT/'attacker/base_run0.json')['opt']['gen']==1
    fail[0]=False
    resumed=e.adapt(m,'base',e.BASE_CFG,0,cks)
    monkeypatch.setattr(e,'OUT',tmp_path/'fresh')
    fresh=e.adapt(m,'base',e.BASE_CFG,0,cks)
    assert resumed==fresh


@pytest.mark.gpu
def test_variable_defender_graph_and_honest_paths_common_score():
    cuda_required()
    _,_,cks=e.fixed_pool()
    pp=[policy.MLPPolicy(obs_version='public'),policy.MLPPolicy(theta=cks[0]['theta'],obs_version='public')]
    for cfg in [dict(e.BASE_CFG,L=30.),dict(e.BASE_CFG,L=70.,tol=4.,kref=.35)]:
        plain=e.simulate(cfg,pp,[11,12],400,exclude=100)
        graph=e.simulate(cfg,pp,[11,12],400,graphsim.from_env('cuda',force=True),exclude=100)
        hand=e.simulate(cfg,[policy.AdapterPolicy({}),policy.AdapterPolicy(dict(dz=.3))],[11,12],400,exclude=100)
        assert plain['T_score']==graph['T_score']==hand['T_score']==300
        for k in plain['snap']:
            np.testing.assert_array_equal(plain['snap'][k],graph['snap'][k])
            np.testing.assert_array_equal(plain['snap'][k][0],hand['snap'][k][0])
