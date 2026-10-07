"""NAIVE pilot evaluation, validation seeds only. usage (run from anywhere, GPU: via gpujob): scripts/eval_d2_pilot.py <out.json> <T> <kind: rl:<cell>|hw:<i0>:<i1>>"""
import os, sys, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root; paths below are relative to it
os.chdir(ROOT); sys.path.insert(0, ROOT)
from arbitration.rl import env, policy, hw, seeds as sd, s1, metrics
out, T, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
vs = list(sd.splits()["val"]); sd.require_split(vs, "val"); s1.require_pilot_seeds(vs)
cfg = dict(env.m3c_config(0.5), p=0.0)
def run(pols):
    o = env.simulate("M3C", T, cfg, pols, vs, r=0.5, dev="cuda", obs_version="public")
    U = o["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float); ts = o["T_score"]
    return [(U[i] - U[0]) / ts for i in range(1, len(pols))], ts
t0 = time.time(); res = {}
if kind.startswith("rl:"):
    cell = kind[3:]
    th = json.load(open(f"results/direction2/parts/{cell}__run0__g0408-0416.json"))["best"]
    res["best_gen"] = th["gen"]; res["G_val_train_T"] = th["G_val"]
    G, ts = run([policy.AdapterPolicy({}), policy.MLPPolicy(F=25, hidden=2, theta=np.array(th["theta"]), obs_version="public"), policy.AdapterPolicy(dict(b=1.0))])
    res["rl"] = metrics.mean_ci(G[0]); res["always1"] = metrics.mean_ci(G[1]); res["T_score"] = ts
    res["diff_rl_minus_always1"] = metrics.difference_ci(G[0], G[1])
    res["rl_G_s"] = G[0].tolist(); res["always1_G_s"] = G[1].tolist()
else:
    _, i0, i1 = kind.split(":"); S = hw.build_s_hw()[int(i0):int(i1)]
    G, ts = run([policy.AdapterPolicy({})] + [policy.AdapterPolicy(dict(d)) for _, d in S])
    res["T_score"] = ts; res["hw"] = {n: dict(ci=metrics.mean_ci(g), G_s=g.tolist()) for (n, _), g in zip(S, G)}
res["T"] = T; res["wall_s"] = time.time() - t0
json.dump(res, open(out, "w"), default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("rl_G_s", "always1_G_s", "hw")}, default=float))
if "hw" in res: print({n: round(v["ci"]["mean"], 4) for n, v in res["hw"].items()})
