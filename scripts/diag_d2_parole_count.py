# D2 parole-event counter (diagnostic, CPU). Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python3 scripts/diag_d2_parole_count.py 20000 4
import os,sys,json,math
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, torch
from arbitration.rl import vulns, env as _env, policy, hw
T=int(sys.argv[1]); S=int(sys.argv[2]); seeds=list(range(S))   # ad-hoc seeds 0..S-1
z=np.load('results/direction2/ckpt/s1_D2_s0.02__run0.npz',allow_pickle=True)
st=json.loads(str(z['state'])); theta=np.asarray(st['best']['theta'])
stats={}
orig=vulns._ExtD2.release
def rel(self,hook,st_,t,elig,v,susp,log):
    new=orig(self,hook,st_,t,elig,v,susp,log)
    N=self.B; P=len(self.names); s=N//P
    par=(new!=susp)[:,1].view(P,s).sum(1); sus=(~elig)[:,1].view(P,s).sum(1)
    mc=self.calm[:,1].view(P,s)
    stats['par']=stats.get('par',0)+par.cpu().numpy(); stats['sus']=stats.get('sus',0)+sus.cpu().numpy()
    stats['maxcalm']=np.maximum(stats.get('maxcalm',0),mc.max(1).values.cpu().numpy())
    # calm before reset: track peak via calm_new unavailable; parole resets it, so also record par
    return new
vulns._ExtD2.release=rel
dz03=dict(dict(hw.build_s_hw())['dz=0.3'])
pols=[policy.AdapterPolicy({}),policy.MLPPolicy(F=25,hidden=2,theta=theta,obs_version="public"),policy.AdapterPolicy(dz03),
      vulns.RulePolicy('d2_parole',{'dz':0.3}),vulns.RulePolicy('d2_parole',{'dz':0.5})]
names=['honest','RL_ckpt','HW dz=0.3','ref d2_parole dz=.3','ref d2_parole dz=.5']
cfg=dict(_env.m3c_config(0.5))
for Q in (1100,1700,math.inf):
    stats.clear()
    orig_init=vulns._ExtD2.__init__
    def init(self,B,dev,knob,st_,_o=orig_init): _o(self,B,dev,knob,st_); self.names=names
    vulns._ExtD2.__init__=init
    out=vulns.simulate_vuln('D2',Q,'M3C',T,cfg,pols,seeds,r=0.5,dev='cpu')
    vulns._ExtD2.__init__=orig_init
    U=out['snap']['util_b'][:,:,1].numpy(); G=(U[1:]-U[0])/out['T_score']
    print('Q=',Q,'T_score',out['T_score'])
    for i,n in enumerate(names):
        print(f"  {n:22s} parole_events(sum over seeds)={int(stats['par'][i])} susp_rounds(agent1)={int(stats['sus'][i])} max_calm_end={stats['maxcalm'][i]:.0f}"+(f" G={G[i-1].mean():.5f}" if i else ""))
    np.save(f'.scratch/d2_knob_diag/G_Q{Q}.npy',G)
