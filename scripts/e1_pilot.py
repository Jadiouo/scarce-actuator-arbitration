#!/usr/bin/env python3
"""Bounded E1 pilot: constrained defender CMA-ES, then one attacker response."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arbitration.rl import cmaes,env,graphsim,hw,main_eval,plan,policy,train
from scripts.d2_holdout_eval import read,write,sha,digest,context,require_disjoint
from scripts.d2_main_eval import backup

OUT=ROOT/'results/direction2/e1_pilot'
BASE_CFG=dict(p=.1,tol=6.,kref=.5,L=2500.)
INPUTS=['scripts/e1_pilot.py','scripts/e1_pilot_launch.sh','tests/direction2/test_e1_pilot.py',
        'docs/direction2/design/e1-pilot.md','scripts/d2_holdout_eval.py','scripts/d2_main_eval.py']
START=None
DO_BACKUP=False


class BudgetStop(Exception):pass


def check_budget():
    if START is not None and time.time()-START>18*3600:raise BudgetStop()


def decode(x):
    x=np.asarray(x,float)
    if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('invalid defender vector')
    x=np.clip(x,-1,1)
    return dict(p=.1,tol=float(6+2*x[0]),kref=float(.5+.15*x[1]),L=float(round(3000+2000*x[2])))


def require_seeds(sd,stage,m):
    if len(sd)!=len(set(sd)):raise ValueError('duplicate environment seeds')
    allowed=set(range(1000)) if stage=='train' else set(m['validation_seeds']) if stage=='validation' else set()
    if not sd or any(type(s) is not int or s not in allowed for s in sd):raise ValueError('wrong seed split')


def metrics(raw):
    u=np.asarray(raw['snap']['util_b'],float)[:,:,1]
    ts=raw['T_score']
    return dict(gains=(u[1:]-u[0])/ts,regret=np.asarray(raw['snap']['reg_b'],float)[0]/ts,
                alpha=np.asarray(raw['snap']['pen'],float)[0].mean(-1)/ts)


def objective(raw,reference,pool):
    a,b=metrics(raw),metrics(reference)
    if raw['contract']['seeds']!=reference['contract']['seeds'] or raw['T_score']!=reference['T_score']:
        raise ValueError('constraint reference must be paired')
    rg=float(np.mean(a['regret']-b['regret']))
    al=float(np.mean(a['alpha']-b['alpha']))
    g=a['gains'][:36] if pool=='hw' else a['gains'] if pool=='pooled' else None
    if g is None:raise ValueError('unknown pool')
    worst=max(0.,float(g.mean(1).max()))
    violation=max(0.,(rg-.002)/.002)+max(0.,(al-.01)/.01)
    return dict(feasible=bool(rg<=.002 and al<=.01),worst_G=worst,delta_honest_regret=rg,delta_honest_alpha=al,
                fitness=float(-worst if violation==0 else -10-violation))


def stamp_status(stage,**kw):
    old=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    write(OUT/'status.json',dict(old,stage=stage,updated_unix=time.time(),**kw))
    print(stage,json.dumps(kw),flush=True)


def publish(message):
    if not DO_BACKUP:return
    paths=[str(p.relative_to(ROOT)) for p in OUT.rglob('*.json')]
    for attempt in range(3):
        try:
            backup(paths,message)
            return
        except subprocess.CalledProcessError:
            if attempt==2:raise
            time.sleep(5)


def freeze():
    if (OUT/'manifest.json').exists():raise RuntimeError('pilot already registered')
    _,cks,_=context()
    sd=list(range(61_000_000,61_000_000+32))
    past=read(ROOT/'results/direction2/holdout/seed_audit.json')['historical_seed_field_values']
    past+=read(ROOT/'results/direction2/holdout/manifest.json')['seeds']
    require_disjoint(sd,past)
    m=dict(schema='e1-pilot/1',validation_seeds=sd,base_cfg=BASE_CFG,r=.5,
           T_train=20000,T_val=100000,tail_exclusion=5000,defender_lam=8,defender_gens=12,defender_seeds=16,
           attacker_lam=64,attacker_gens=120,attacker_seeds=32,source_sha256=main_eval.code_hash(),
           checkpoints=main_eval.checkpoint_meta(cks),input_sha256={p:sha(ROOT/p) for p in INPUTS},
           protected_sha256={**read(ROOT/'results/direction2/holdout/manifest.json')['protected_sha256'],
                            **{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results/direction2/holdout').glob('*.json')}},
           validation_seed_audit='Disjoint from saved historical seed fields, original splits and independent holdout, including stream offsets.')
    m['sha256']=digest(m)
    write(OUT/'manifest.json',m)
    print('E1 manifest created; commit before production run.',flush=True)


def load_manifest(clean=True):
    m=read(OUT/'manifest.json')
    if m['sha256']!=digest({k:v for k,v in m.items() if k!='sha256'}):raise ValueError('manifest changed')
    for p,h in {**m['input_sha256'],**m['protected_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('frozen input changed: '+p)
    if clean:
        for p in [*m['input_sha256'],str((OUT/'manifest.json').relative_to(ROOT))]:
            if not plan.is_committed_and_clean(str(ROOT),p):raise ValueError('uncommitted input: '+p)
    if main_eval.code_hash()!=m['source_sha256']:raise ValueError('simulator changed')
    return m


def fixed_pool():
    _,cks,_=context()
    names=['honest']+[n for n,_ in hw.build_s_hw()]+[f'frozen_run{i}' for i in range(3)]
    pols=[policy.AdapterPolicy({})]+[policy.AdapterPolicy(d) for _,d in hw.build_s_hw()]
    pols += [policy.MLPPolicy(theta=c['theta'],obs_version='public') for c in cks]
    return names,pols,cks


def simulate(cfg,pols,sd,T,graph=None,dev='cuda',exclude=5000):
    out=env.simulate('M3C',T,dict(cfg,ad_h=float('inf')),pols,sd,r=.5,dev=dev,T_score=T-exclude,graph=graph)
    return dict(T_score=int(out['T_score']),snap={k:out['snap'][k].detach().cpu().numpy().tolist() for k in main_eval.SNAP_FIELDS})


def validate_snapshot(raw,con):
    if raw['contract']!=con or raw['sha256']!=digest({k:v for k,v in raw.items() if k!='sha256'}):raise ValueError('cache contract/checksum changed')
    p,n=con['n_policies'],len(con['seeds'])
    if raw['T_score']!=con['T']-con['exclude']:raise ValueError('score window changed')
    for k,shape in [('util_b',(p,n,3)),('reg_b',(p,n)),('pen',(p,n,3))]:
        a=np.asarray(raw['snap'][k],float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid snapshot '+k)
    return raw


def evaluate(m,cfg,sd,stage,pols,ids,graph=None):
    require_seeds(sd,stage,m)
    check_budget()
    T=m['T_train'] if stage=='train' else m['T_val']
    con=dict(manifest_sha256=m['sha256'],cfg=cfg,seeds=list(sd),stage=stage,T=T,exclude=5000,n_policies=len(pols),policies=ids)
    p=OUT/'evaluations'/(digest(con)+'.json')
    if p.exists():return validate_snapshot(read(p),con),str(p.relative_to(OUT))
    t=time.time();raw=dict(simulate(cfg,pols,sd,T,graph),contract=con,wall_s=time.time()-t)
    raw['sha256']=digest(raw);validate_snapshot(raw,con);write(p,raw)
    return raw,str(p.relative_to(OUT))


def state_load(path,m,initial):
    if path.exists():
        s=read(path)
        if s['manifest_sha256']!=m['sha256']:raise ValueError('checkpoint from another manifest')
        return s
    return dict(initial,manifest_sha256=m['sha256'])


def defender(m,pool,run,names,pols):
    path=OUT/'defender'/f'{pool}_run{run}.json'
    opt=cmaes.make_optimizer(3,lam=8,sigma0=.45,seed=72000+run)
    opt.mean=np.array([0.,0.,-.25])
    state=state_load(path,m,dict(opt=opt.state_dict(),records=[],curve=[],pending=None))
    opt.load_state_dict(state['opt'])
    while opt.gen<m['defender_gens']:
        if state['pending'] is None:
            state['pending']=dict(X=opt.ask().tolist(),done=[],gen=opt.gen)
            # ask consumes RNG; persist that exact state and population for resume.
            state['opt']=opt.state_dict();write(path,state)
        pending=state['pending'];X=np.asarray(pending['X'])
        sd=list(train.seed_block(88,opt.gen,16))
        reference,_=evaluate(m,BASE_CFG,sd,'train',pols,names)
        for i in range(len(pending['done']),len(X)):
            cfg=decode(X[i]);raw,rel=evaluate(m,cfg,sd,'train',pols,names)
            score=objective(raw,reference,pool)
            pending['done'].append(dict(cfg=cfg,raw=rel,**score));write(path,state)
            stamp_status('defender',pool=pool,run=run,generation=opt.gen+1,candidate=i+1,
                         candidate_seconds=raw['wall_s'],feasible=score['feasible'])
        scores=pending['done'];best=max(scores,key=lambda x:x['fitness'])
        # Validation happens before tell; a restart uses saved candidate/cache data.
        if (opt.gen+1)%3==0:
            reference_v,_=evaluate(m,BASE_CFG,m['validation_seeds'],'validation',pols,names)
            raw,rel=evaluate(m,best['cfg'],m['validation_seeds'],'validation',pols,names)
            state['records'].append(dict(gen=opt.gen+1,cfg=best['cfg'],raw=rel,**objective(raw,reference_v,pool)))
        opt.tell(X,[s['fitness'] for s in scores])
        state['curve'].append(dict(gen=opt.gen,best=best))
        state.update(opt=opt.state_dict(),pending=None);write(path,state)
        publish(f'E1 pilot: defender {pool} run{run} generation {opt.gen}')
    return state


def select_defenders(m,names,pols):
    reference,ref=evaluate(m,BASE_CFG,m['validation_seeds'],'validation',pols,names)
    selected={'original':dict(cfg=BASE_CFG,source='original',raw=ref)}
    for pool in ('hw','pooled'):
        candidates=[dict(cfg=BASE_CFG,source='original fallback',raw=ref,**objective(reference,reference,pool))]
        for run in range(2):
            s=read(OUT/'defender'/f'{pool}_run{run}.json')
            if s['opt']['gen']!=m['defender_gens']:raise ValueError('unfinished defender training')
            candidates += [dict(x,source=f'{pool}_run{run}/gen{x["gen"]}') for x in s['records']]
        winner=min((x for x in candidates if x['feasible']),key=lambda x:x['worst_G'])
        selected[pool]=dict(winner,all_candidates=candidates)
    write(OUT/'selection.json',dict(manifest_sha256=m['sha256'],selected=selected))
    publish('E1 pilot: freeze validation-selected defenders')
    return selected


def theta_id(theta):return digest(np.asarray(theta,float).tolist())


def adapt(m,label,cfg,run,cks):
    path=OUT/'attacker'/f'{label}_run{run}.json'
    opt=cmaes.make_optimizer(55,lam=64,sigma0=.15,seed=73000+run)
    opt.mean=np.asarray(cks[run]['theta']).copy()
    state=state_load(path,m,dict(cfg=cfg,opt=opt.state_dict(),best=None,curve=[],records=[]))
    if state['cfg']!=cfg:raise ValueError('defender changed under attacker')
    opt.load_state_dict(state['opt'])
    graph=graphsim.from_env('cuda',force=True)
    def validate(theta,gen):
        pols=[policy.MLPPolicy(obs_version='public'),policy.MLPPolicy(theta=theta,obs_version='public')]
        raw,rel=evaluate(m,cfg,m['validation_seeds'],'validation',pols,['honest',theta_id(theta)],graph)
        rec=dict(gen=gen,G_val=float(metrics(raw)['gains'][0].mean()),theta=np.asarray(theta).tolist(),raw=rel)
        state['records'].append(rec)
        if state['best'] is None or rec['G_val']>state['best']['G_val']:state['best']=rec
    if state['best'] is None:
        validate(opt.mean,0);write(path,state)
    while opt.gen<m['attacker_gens']:
        check_budget();g=opt.gen;sd=list(train.seed_block(99+run,g,32));require_seeds(sd,'train',m)
        t=time.time();X=opt.ask()
        pols=[policy.MLPPolicy(obs_version='public')]+[policy.MLPPolicy(theta=x,obs_version='public') for x in X]
        # Training populations are resumable from optimizer/RNG; keep scalar
        # fitness vectors, rather than retaining every population's full snapshot.
        raw=simulate(cfg,pols,sd,m['T_train'],graph)
        U=np.asarray(raw['snap']['util_b'])[:,:,1]
        if not np.isfinite(U).all():raise ValueError('nonfinite training utility')
        f=U[1:].mean(1)/raw['T_score'];gain=(U[1:]-U[0]).mean(1)/raw['T_score']
        opt.tell(X,f)
        state['curve'].append(dict(gen=opt.gen,fitness=f.tolist(),G=gain.tolist(),wall_s=time.time()-t))
        if opt.gen%20==0:validate(opt.mean,opt.gen)
        state['opt']=opt.state_dict();write(path,state)
        stamp_status('attacker',defender=label,run=run,generation=opt.gen,generation_seconds=time.time()-t)
        if opt.gen%20==0:publish(f'E1 pilot: attacker {label} run{run} generation {opt.gen}')
    del graph
    return state


def cross_validate(m,selected,cks,names,pols):
    learned=[(f'frozen_run{i}',c['theta']) for i,c in enumerate(cks)]
    for label in selected:
        for run in range(2):
            s=read(OUT/'attacker'/f'{label}_run{run}.json')
            if s['opt']['gen']!=m['attacker_gens']:raise ValueError('unfinished attacker')
            learned.append((f'{label}_adapt{run}',s['best']['theta']))
    result=[]
    pp=[policy.MLPPolicy(obs_version='public')]+[policy.MLPPolicy(theta=theta,obs_version='public') for _,theta in learned]
    ids=['honest']+[dict(name=n,theta_sha256=theta_id(th)) for n,th in learned]
    for label,sel in selected.items():
        graph=graphsim.from_env('cuda',force=True)
        raw,rel=evaluate(m,sel['cfg'],m['validation_seeds'],'validation',pp,ids,graph)
        fixed,fixedrel=evaluate(m,sel['cfg'],m['validation_seeds'],'validation',pols,names)
        main_eval._check_honest(raw,fixed)
        result.append(dict(defender=label,cfg=sel['cfg'],learned_names=[n for n,_ in learned],learned_raw=rel,
                           learned_G_per_seed=metrics(raw)['gains'].tolist(),fixed_names=names[1:],fixed_raw=fixedrel,
                           fixed_G_per_seed=metrics(fixed)['gains'].tolist()))
        del graph
    write(OUT/'cross_validation.json',dict(manifest_sha256=m['sha256'],split='selection validation; exploratory only',rows=result))


def preflight():
    m=load_manifest(clean=False);names,pols,cks=fixed_pool()
    t=time.time();raw=simulate(BASE_CFG,pols,list(range(16)),2000,exclude=500)
    actual=time.time()-t
    estimate=actual/2000*20000*(384+12+17*5)/3600
    pp=[policy.MLPPolicy(obs_version='public')]+[policy.MLPPolicy(theta=cks[i%2]['theta'],obs_version='public') for i in range(64)]
    graph=graphsim.from_env('cuda',force=True)
    simulate(BASE_CFG,pp,list(range(32)),2000,graph,exclude=500)  # capture/warmup
    t=time.time();simulate(BASE_CFG,pp,list(range(32)),2000,graph,exclude=500)
    attack_seconds=time.time()-t
    attack_hours=attack_seconds/2000*20000*720/3600
    rec=dict(schema='e1-preflight/1',manifest_sha256=m['sha256'],short_seconds=actual,
             short_rounds=2000,rough_defender_hours=estimate,attack_short_seconds=attack_seconds,
             rough_attacker_train_hours=attack_hours,rough_total_hours=estimate+attack_hours+1,
             note='Step-time extrapolation at production training batch sizes; includes approximate validation overhead, not a guarantee.',
             snapshot_finite=all(np.isfinite(np.asarray(x)).all() for x in raw['snap'].values()))
    write(OUT/'preflight.json',rec);print(json.dumps(rec),flush=True)


def run():
    m=load_manifest();names,pols,cks=fixed_pool()
    stamp_status('running',started_unix=time.time(),manifest_sha256=m['sha256'])
    for pool in ('hw','pooled'):
        for run_id in range(2):defender(m,pool,run_id,names,pols)
    selected=select_defenders(m,names,pols)
    for label,sel in selected.items():
        for run_id in range(2):adapt(m,label,sel['cfg'],run_id,cks)
    cross_validate(m,selected,cks,names,pols)
    load_manifest()
    stamp_status('complete',finished_unix=time.time())
    publish('E1 pilot: completed training and exploratory cross-validation')


def main():
    global START,DO_BACKUP
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','preflight','run']);ap.add_argument('--backup',action='store_true')
    a=ap.parse_args();os.chdir(ROOT);DO_BACKUP=a.backup
    (ROOT/'.scratch').mkdir(exist_ok=True)
    with (ROOT/'.scratch/e1-pilot.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(1)
        if a.stage=='freeze':freeze();return
        if a.stage=='preflight':preflight();return
        START=time.time()
        try:run()
        except BudgetStop:
            stamp_status('budget_stop',message='18 hour cap reached at save boundary; resume explicitly')
            publish('E1 pilot: saved at time cap');sys.exit(75)
        except Exception as exc:
            stamp_status('failed',error_type=type(exc).__name__,message='See the local run log for traceback; checkpoints preserved.')
            try:publish('E1 pilot: preserve failed run checkpoints')
            except Exception:pass
            raise


if __name__=='__main__':main()
