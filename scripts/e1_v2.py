#!/usr/bin/env python3
"""E1 v2: pre-registered confirmatory defender repair (design: docs/direction2/design/e1-v2.md).

Stages: freeze (S0) -> run s1 / s2 / sx / select -> test (once).  Every stage is resumable.
The test block 72400000-72401023 is opened only by `test`, only after selection_frozen.json is registered
(freeze_record.json) and committed.  --smoke uses a tiny budget and a subset of DT/DS/AT/AS and has no test block.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import cmaes, graphsim, main_eval, plan, policy, seeds as seedmod
from scripts import e1_pilot as v1
from scripts.d2_holdout_eval import read, write, sha, digest, context, seed_values, require_disjoint, once
from scripts.d2_main_eval import backup

DESIGN = "docs/direction2/design/e1-v2.md"
OUT_DIR = "results/direction2/e1v2"
SMOKE_DIR = "results/direction2/e1v2_smoke"
OUT = ROOT / OUT_DIR
BASE_CFG = v1.BASE_CFG
OFFS4 = (0, 10**6, 2*10**6, 3*10**6)
BLOCKS = dict(DT=(72_000_000, 256), DS=(72_100_000, 128), AT=(72_200_000, 256), AS=(72_300_000, 128), TEST=(72_400_000, 1024))
STAGE_BLOCK = dict(dt='DT', ds='DS', **{'at': 'AT'}, **{'as': 'AS'}, test='TEST')
EXISTING = dict(holdout=(41_000_000, 64), e1_pilot=(61_000_000, 32))
EXPECT_NEW, EXPECT_OLD = 7168, 4896
INPUTS = ['scripts/e1_v2.py', 'scripts/e1_v2_launch.sh', 'tests/direction2/test_e1_v2.py', DESIGN,
          'scripts/e1_pilot.py', 'scripts/d2_holdout_eval.py', 'scripts/d2_main_eval.py']
EXTRA_PROTECTED = ['results/direction2/run_plan.json', 'results/direction2/FREEZE.md', 'results/direction2/pilot_decision.json',
                   'results/direction2/vuln_suite.json', 'docs/direction2-freeze.md', 'docs/direction2-spec.md']

PARAMS = dict(
    schema='e1-v2/1', r=.5, base_cfg=BASE_CFG, T_train=20000, T_val=100000, exclude=5000,
    delta=.005, lim_regret=.002, lim_alpha=.01, ci_level=.95, n_test=1024, test_chunk=64,
    defender=dict(lam=8, gens=12, sigma0=.45, n_dt=16, eval_gens=[4, 8, 12], seeds_r1=[91000, 91001], seeds_r2=[92000, 92001]),
    attacker=dict(lam=64, sigma0=.15, n_at=32, gens=240, ext_gens=240, val_every=20, chains=3, opt_seed=93000, sat_threshold=.001),
    search=dict(h=[4, 8], kref=[.35, .65], L=[1000, 5000]), tie_eps=1e-9)
SMOKE_PARAMS = dict(
    PARAMS, schema='e1-v2-smoke/1', T_train=1000, T_val=1500, exclude=500, lim_regret=1., lim_alpha=1., n_test=0,
    defender=dict(PARAMS['defender'], lam=4, gens=2, n_dt=4, eval_gens=[1, 2]),
    attacker=dict(PARAMS['attacker'], lam=8, n_at=4, gens=4, ext_gens=4, val_every=2, sat_threshold=1e-9))
SMOKE_N = 8

MODE = dict(smoke=False, device='cuda', graph=False, fallback=False, backup=False, start=None)


class BudgetStop(Exception):
    pass


# ---------------------------------------------------------------- seeds
def block_seeds(smoke=False):
    b = {k: list(range(a, a+n)) for k, (a, n) in BLOCKS.items()}
    if smoke:
        return {k: v[:SMOKE_N] for k, v in b.items() if k != 'TEST'}
    return b


def expand(base, offsets=OFFS4):
    return sorted({s+o for s in base for o in offsets})


def existing_seeds(root=ROOT):
    """Seven existing ranges (REQ-SEED + holdout + pilot); manifests of holdout/pilot must agree with the ranges."""
    out = {k: list(v) for k, v in seedmod.splits().items()}
    for k, (a, n) in EXISTING.items():
        out[k] = list(range(a, a+n))
    hm, pm = Path(root)/'results/direction2/holdout/manifest.json', Path(root)/'results/direction2/e1_pilot/manifest.json'
    if hm.exists() and read(hm)['seeds'] != out['holdout']:
        raise ValueError('holdout manifest seeds differ from registered range')
    if pm.exists() and read(pm)['validation_seeds'] != out['e1_pilot']:
        raise ValueError('pilot manifest seeds differ from registered range')
    return out


def seed_report(blocks, existing, extra_historical=()):
    """Raise if the blocks collide with each other, with the existing ranges, or with saved seed fields (all x 4 offsets)."""
    names = list(blocks)
    ex = {k: set(expand(v)) for k, v in blocks.items()}
    for k in names:
        if len(set(blocks[k])) != len(blocks[k]) or len(ex[k]) != 4*len(blocks[k]):
            raise ValueError('duplicate or colliding seeds inside block '+k)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if ex[a] & ex[b]:
                raise ValueError(f'blocks {a} and {b} overlap')
    new = set().union(*ex.values())
    old = set(expand([s for v in existing.values() for s in v]))
    if new & old:
        raise ValueError('new seeds overlap an existing range')
    hist = {s+o for s in extra_historical for o in OFFS4}
    if new & hist:
        raise ValueError('new seeds overlap a saved historical seed field')
    return dict(new_expanded=len(new), existing_expanded=len(old), intersection=len(new & old),
                new_expanded_sha256=digest(sorted(new)), existing_expanded_sha256=digest(sorted(old)),
                per_block={k: len(v) for k, v in ex.items()}, offsets=list(OFFS4))


def scan_saved_seeds(root=ROOT):
    """Integer seed fields of saved json/npz records (excluding this experiment's own output)."""
    vals, srcs = set(), {}
    for p in sorted((Path(root)/'results').rglob('*')):
        rel = str(p.relative_to(root))
        if rel.startswith('results/direction2/e1v2') or not p.is_file() or p.suffix not in ('.json', '.npz'):
            continue
        if p.suffix == '.json':
            vals |= seed_values(json.loads(p.read_text()))
        else:
            with np.load(p, allow_pickle=False) as z:
                if 'state' in z.files:
                    vals |= seed_values(json.loads(str(z['state'])))
        srcs[rel] = sha(p)
    return vals, srcs


def rotating(pool, key, gen, n):
    """n distinct seeds from `pool` for generation `gen`; deterministic; wraps around the pool."""
    if not 1 <= n <= len(pool):
        raise ValueError('block size must lie in [1, len(pool)]')
    start = (gen*n + key*37) % len(pool)
    return [pool[(start+j) % len(pool)] for j in range(n)]


def require_seeds(sd, stage, m, unlocked=False):
    if stage not in STAGE_BLOCK:
        raise ValueError('unknown stage '+str(stage))
    if stage == 'test' and not unlocked:
        raise ValueError('test seeds are locked until selection_frozen is registered')
    pool = m['seeds'].get(STAGE_BLOCK[stage])
    if pool is None:
        raise ValueError('this manifest has no '+STAGE_BLOCK[stage]+' block')
    if len(sd) != len(set(sd)):
        raise ValueError('duplicate environment seeds')
    allowed = set(pool)
    if not sd or any(type(s) is not int or s not in allowed for s in sd):
        raise ValueError('wrong seed split')


# ---------------------------------------------------------------- statistics / decisions
def tcrit(n):
    if n < 2:
        raise ValueError('need at least two seeds')
    return float(stats.t.ppf(.95, n-1))


def upper_bound(x):
    x = np.asarray(x, float)
    n = len(x)
    return float(x.mean()+tcrit(n)*x.std(ddof=1)/math.sqrt(n))


def lower_bound(x):
    x = np.asarray(x, float)
    n = len(x)
    return float(x.mean()-tcrit(n)*x.std(ddof=1)/math.sqrt(n))


def honest_ci(d_regret, d_alpha, lim_r=PARAMS['lim_regret'], lim_a=PARAMS['lim_alpha']):
    """One-sided 95% UB of the paired honest deltas (candidate - original); feasible iff both UB <= limit (equality passes)."""
    ur, ua = upper_bound(d_regret), upper_bound(d_alpha)
    return dict(n=len(d_regret), mean_regret=float(np.mean(d_regret)), mean_alpha=float(np.mean(d_alpha)), ub_regret=ur, ub_alpha=ua,
                feasible=bool(ur <= lim_r+1e-12 and ua <= lim_a+1e-12), t=tcrit(len(d_regret)))


def saturation(traj, val_every, gmax, threshold):
    """traj: [(gen, G)] of validation points.  prev = best over g <= gmax-2*val_every, last = best over the last two points."""
    pts = {int(g): float(v) for g, v in traj if int(g) <= gmax}
    if gmax not in pts:
        raise ValueError('trajectory lacks the final validation point')
    cut = gmax-2*val_every
    prev = [v for g, v in pts.items() if g <= cut]
    last = [v for g, v in pts.items() if g > cut]
    if not prev:
        raise ValueError('trajectory too short for the saturation rule')
    gain = round(max(last)-max(prev), 12)
    return dict(gmax=gmax, prev_best=max(prev), last_best=max(last), gain=gain, saturated=bool(gain < threshold))


def pick_run_winner(records):
    ok = [r for r in records if r['feasible']]
    return min(ok, key=lambda r: r['worst_G']) if ok else None   # min keeps the first on ties


def pick_c1(winners):
    ok = [w for w in winners if w is not None]
    return min(ok, key=lambda r: r['worst_G']) if ok else None


def final_selection(cands, g_orig, eps=PARAMS['tie_eps']):
    """cands: ordered [(label, G_astar)] of DS-feasible candidates ([C1] or [C1, C2]).  Ties (< eps) go to the earlier one."""
    if not cands:
        return dict(status='no_feasible', label=None)
    best = cands[0]
    for c in cands[1:]:
        if c[1] < best[1]-eps:
            best = c
    improvement = g_orig-best[1]
    if not improvement > 0:
        return dict(status='no_AS_improvement', label=best[0], improvement=improvement)
    return dict(status='go', label=best[0], improvement=improvement)


TEXT = dict(
    established='成立：在 M3C、r=0.5 與本攻擊者預算下，最終候選相對原設定降低最強已知攻擊者的增益超過 0.005，且未超出誠實服務上限。',
    honest_breach='誠實越線：攻擊增益下降，但誠實服務代價超出預登記上限；不構成修補。',
    both_fail='兩者皆失敗：此候選既未確認降低攻擊增益，也未確認服務維持；不採用。',
    effect_a='效果不足：排除達到 0.005 的降幅。',
    effect_b='效果不足：有降幅但未確認超過 0.005。',
    effect_c='效果不足：證據不足以判定。',
    not_tested_no_feasible='未進 test：無候選滿足誠實約束',
    not_tested_no_improvement='未進 test：無可行候選達到 AS 改善')


def power(n, mu, sigma, delta=PARAMS['delta']):
    """Design 6: P[T_{n-1}(lambda) > t_{0.95,n-1}], lambda=(mu-delta)/(sigma/sqrt(n))."""
    return float(stats.nct.sf(tcrit(n), n-1, (mu-delta)/(sigma/math.sqrt(n))))


def honest_pass_prob(n, mu, sd, limit):
    """Design 2 tables: P[mean + t*SD/sqrt(n) <= limit] for a true paired difference mu."""
    return float(stats.nct.cdf(-tcrit(n), n-1, (mu-limit)/(sd/math.sqrt(n))))


def decide(D, d_regret, d_alpha, delta=PARAMS['delta'], lim_r=PARAMS['lim_regret'], lim_a=PARAMS['lim_alpha']):
    """Section 5 decision tree (nodes 2-6).  Strict inequality for E; equality passes the honest limits."""
    D = np.asarray(D, float)
    h = honest_ci(d_regret, d_alpha, lim_r, lim_a)
    lb, ubd = lower_bound(D), upper_bound(D)
    H, E = h['feasible'], bool(lb > delta)
    if H and E:
        cat, sub = 'established', None
    elif E:
        cat, sub = 'honest_breach', None
    elif not H:
        cat, sub = 'both_fail', None
    else:
        cat = 'insufficient_effect'
        sub = 'a' if ubd < delta else 'b' if lb > 0 else 'c'
    return dict(category=cat, sub=sub, text=TEXT['effect_'+sub] if sub else TEXT[cat], H=H, E=E, n=len(D), mean_D=float(D.mean()),
                LB=lb, UB_D=ubd, LB_gt_0=bool(lb > 0), honest=h, delta=delta,
                honest_excess=dict(regret=h['ub_regret']-lim_r, alpha=h['ub_alpha']-lim_a))


def budget_consistency(budgets, final_gens, extend_decisions, base_gens):
    """budgets: {label: [per-chain initial budget dicts]}.  Every label must start with identical budgets; a label's chains may run
    base_gens*2 only if the saturation rule asked for an extension (and then all of its chains)."""
    ref = None
    for label, chains in budgets.items():
        ref = ref or chains
        if len(chains) != len(ref):
            raise ValueError(f'attacker chain count differs for {label}')
        for run, b in enumerate(chains):      # chain i uses the same optimizer seed and warm start under every defender
            if b != ref[run]:
                raise ValueError(f'attacker start budget differs for {label}')
    for label, gens in final_gens.items():
        want = base_gens*2 if extend_decisions.get(label) else base_gens
        if any(g != want for g in gens):
            raise ValueError(f'attacker generations for {label} do not follow the saturation rule')
    return True


# ---------------------------------------------------------------- manifest
def encode(cfg):
    return [(cfg['tol']-6)/2, (cfg['kref']-.5)/.15, (cfg['L']-3000)/2000]


def out_dir():
    if MODE.get('out'):
        return Path(MODE['out'])
    return ROOT/(SMOKE_DIR if MODE['smoke'] else OUT_DIR)


def git(*a):
    return subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()


def build_manifest(smoke=False, root=ROOT):
    root = Path(root)
    _, cks, _ = context()
    blocks = block_seeds(smoke)
    existing = existing_seeds(root)
    hist, srcs = (set(), {}) if smoke else scan_saved_seeds(root)
    if not smoke:
        hist |= set(read(root/'results/direction2/holdout/seed_audit.json')['historical_seed_field_values'])
    rep = seed_report(blocks, existing, hist)
    if not smoke and (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) != (EXPECT_NEW, EXPECT_OLD, 0):
        raise ValueError('seed counts differ from the registered 7168/4896/0')
    audit = dict(schema='e1-v2-seed-audit/1', report=rep, offsets=list(OFFS4), saved_seed_values=sorted(hist), scanned_sha256=srcs,
                 limitation='Saved project records only; cannot certify unrecorded external executions.')
    protected = {}
    hm = read(root/'results/direction2/holdout/manifest.json')['protected_sha256']
    protected.update(hm)
    for p in (root/'results/direction2/holdout').glob('*.json'):
        protected[str(p.relative_to(root))] = sha(p)
    for p in EXTRA_PROTECTED:
        if (root/p).exists():
            protected[p] = sha(root/p)
    v1out = {str(p.relative_to(root)): sha(p) for p in sorted((root/'results/direction2/e1_pilot').glob('*.json'))}
    v1out.update({str(p.relative_to(root)): sha(p) for sub in ('defender', 'attacker') for p in sorted((root/'results/direction2/e1_pilot'/sub).glob('*.json'))})
    params = SMOKE_PARAMS if smoke else PARAMS
    m = dict(schema=params['schema'], smoke=smoke, params=params, design=DESIGN, design_sha256=sha(root/DESIGN),
             code_git_sha=git('rev-parse', 'HEAD'), seeds=blocks, seed_expanded_sha256=rep['new_expanded_sha256'], seed_report=rep,
             checkpoints=main_eval.checkpoint_meta(cks), source_sha256=main_eval.code_hash(),
             input_sha256={p: sha(root/p) for p in INPUTS if (root/p).exists()}, protected_sha256=protected, v1_outputs_sha256=v1out,
             stage_blocks={'dt': 'DT', 'ds': 'DS', 'at': 'AT', 'as': 'AS', 'test': 'TEST'},
             test_unlock='selection_frozen.json sha256 recorded in freeze_record.json (committed) before any TEST seed is read')
    m['sha256'] = digest(m)
    return m, audit


def freeze(smoke=False):
    d = out_dir()
    if (d/'manifest.json').exists() and not smoke:
        raise RuntimeError('manifest already exists; the registered design must not be replaced')
    if smoke and d.exists():
        shutil.rmtree(d)       # the throw-away smoke directory only
    if not smoke:
        for p in INPUTS:
            if not plan.is_committed_and_clean(str(ROOT), p):
                raise ValueError('commit all inputs before registering: '+p)
    m, audit = build_manifest(smoke)
    write(d/'seed_audit.json', audit)
    m['seed_audit_sha256'] = sha(d/'seed_audit.json')
    m['sha256'] = digest({k: v for k, v in m.items() if k != 'sha256'})
    write(d/'manifest.json', m)
    print('manifest written', m['sha256'], '(commit it and seed_audit.json before running)' if not smoke else '(smoke)', flush=True)


def check_hashes(mapping, root=ROOT):
    for p, h in mapping.items():
        if not (Path(root)/p).exists() or sha(Path(root)/p) != h:
            raise ValueError('frozen input changed: '+p)


def verify_manifest_dict(m):
    if m['sha256'] != digest({k: v for k, v in m.items() if k != 'sha256'}):
        raise ValueError('manifest checksum changed')
    expected = SMOKE_PARAMS if m['smoke'] else PARAMS
    if m['params'] != json.loads(json.dumps(expected)):
        raise ValueError('registered parameters differ from the code')
    return m


def load_manifest(clean=True):
    d = out_dir()
    m = verify_manifest_dict(read(d/'manifest.json'))
    if m['smoke'] != MODE['smoke']:
        raise ValueError('smoke/official manifest mixup')
    check_hashes({**m['input_sha256'], **m['protected_sha256'], **m['v1_outputs_sha256'], DESIGN: m['design_sha256']})
    if sha(d/'seed_audit.json') != m['seed_audit_sha256']:
        raise ValueError('seed audit changed')
    audit = read(d/'seed_audit.json')
    rep = seed_report(m['seeds'], existing_seeds(), audit['saved_seed_values'])
    if rep['new_expanded_sha256'] != m['seed_expanded_sha256']:
        raise ValueError('seed set changed')
    if main_eval.code_hash() != m['source_sha256']:
        raise ValueError('simulator changed')
    if not m['smoke']:
        if subprocess.run(['git', 'diff', '--quiet', m['code_git_sha'], '--', *m['input_sha256']], cwd=ROOT).returncode:
            raise ValueError('code differs from the registered git sha')
        if clean:
            for p in [*m['input_sha256'], str((d/'manifest.json').relative_to(ROOT)), str((d/'seed_audit.json').relative_to(ROOT))]:
                if not plan.is_committed_and_clean(str(ROOT), p):
                    raise ValueError('uncommitted input: '+p)
    return m


# ---------------------------------------------------------------- infrastructure
def check_budget():
    if MODE['start'] is not None and time.time()-MODE['start'] > 18*3600:
        raise BudgetStop()


def stamp(stage, **kw):
    p = out_dir()/'status.json'
    old = read(p) if p.exists() else {}
    write(p, dict(old, stage=stage, updated_unix=time.time(), **kw))
    print(stage, json.dumps(kw), flush=True)


def mark_done(name):
    p = out_dir()/'status.json'
    old = read(p) if p.exists() else {}
    write(p, dict(old, stages_done=sorted(set(old.get('stages_done', []))|{name}), updated_unix=time.time()))


def stages_done():
    p = out_dir()/'status.json'
    return set(read(p).get('stages_done', [])) if p.exists() else set()


def publish(message):
    if not MODE['backup']:
        return
    paths = [str(p.relative_to(ROOT)) for p in out_dir().rglob('*.json')]
    for attempt in range(3):
        try:
            backup(paths, message)
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(5)


class Graph:
    def __init__(self):
        self.g = None
        self.failed = (out_dir()/'graph_fallback.json').exists()

    def get(self):
        if not MODE['graph'] or self.failed or not MODE['device'].startswith('cuda'):
            return None
        if self.g is None:
            self.g = graphsim.from_env(MODE['device'], force=True)
        return self.g


GRAPH = Graph()


def _graph_error(e):
    s = (type(e).__name__+' '+str(e)).lower()
    return any(k in s for k in ('graph', 'capture', 'replay'))


def sim(m, cfg, pols, sd, T):
    p = m['params']
    fn = lambda g: v1.simulate(cfg, pols, sd, T, g, MODE['device'], p['exclude'])
    g = GRAPH.get()
    if g is None:
        return fn(None)
    try:
        return fn(g)
    except Exception as e:
        if not (MODE['fallback'] and _graph_error(e)):
            raise
        GRAPH.failed, GRAPH.g = True, None
        write(out_dir()/'graph_fallback.json', dict(error=type(e).__name__, message=str(e)[:500], unix=time.time()))
        print('GRAPH FALLBACK: continuing eager from the saved state', flush=True)
        return fn(None)


def evaluate(m, cfg, sd, stage, pols, ids, unlocked=False):
    require_seeds(sd, stage, m, unlocked)
    check_budget()
    p = m['params']
    T = p['T_train'] if stage in ('dt', 'at') else p['T_val']
    con = dict(manifest_sha256=m['sha256'], cfg=cfg, seeds=list(sd), stage=stage, T=T, exclude=p['exclude'], n_policies=len(pols), policies=ids)
    path = out_dir()/'evaluations'/(digest(con)+'.json')
    if path.exists():
        return v1.validate_snapshot(read(path), con), str(path.relative_to(out_dir()))
    t = time.time()
    raw = dict(sim(m, cfg, pols, sd, T), contract=con, wall_s=time.time()-t)
    raw['sha256'] = digest(raw)
    v1.validate_snapshot(raw, con)
    write(path, raw)
    return raw, str(path.relative_to(out_dir()))


def state_load(path, m, initial):
    if Path(path).exists():
        s = read(path)
        if s['manifest_sha256'] != m['sha256']:
            raise ValueError('checkpoint from another manifest')
        return s
    return dict(initial, manifest_sha256=m['sha256'])


def mlp(theta):
    return policy.MLPPolicy(theta=np.asarray(theta, float), obs_version='public')


def honest_pol():
    return policy.MLPPolicy(obs_version='public')


# ---------------------------------------------------------------- defender training / selection
def ds_record(m, cfg, pols, names, gen):
    """Design 2/7: CI feasibility + worst G on the DS seeds, paired with the original setting."""
    p = m['params']
    sd = m['seeds']['DS']
    ref, _ = evaluate(m, BASE_CFG, sd, 'ds', pols, names)
    raw, rel = evaluate(m, cfg, sd, 'ds', pols, names)
    a, b = v1.metrics(raw), v1.metrics(ref)
    h = honest_ci(a['regret']-b['regret'], a['alpha']-b['alpha'], p['lim_regret'], p['lim_alpha'])
    worst = max(0., float(a['gains'].mean(1).max()))
    return dict(gen=gen, cfg=cfg, raw=rel, worst_G=worst, **{k: h[k] for k in ('feasible', 'ub_regret', 'ub_alpha', 'mean_regret', 'mean_alpha', 'n')})


def defender(m, rnd, run, names, pols, start_cfg):
    d = m['params']['defender']
    path = out_dir()/'defender'/f'r{rnd}_run{run}.json'
    opt = cmaes.make_optimizer(3, lam=d['lam'], sigma0=d['sigma0'], seed=d['seeds_r1' if rnd == 1 else 'seeds_r2'][run])
    opt.mean = np.array(encode(start_cfg), float)
    state = state_load(path, m, dict(opt=opt.state_dict(), records=[], curve=[], pending=None, start_cfg=start_cfg, names=names))
    if state['start_cfg'] != start_cfg or state['names'] != names:
        raise ValueError('defender resumed under a different start or pool')
    opt.load_state_dict(state['opt'])
    p = m['params']
    while opt.gen < d['gens']:
        if state['pending'] is None:
            state['pending'] = dict(X=opt.ask().tolist(), done=[], gen=opt.gen)
            state['opt'] = opt.state_dict()
            write(path, state)
        pending = state['pending']
        X = np.asarray(pending['X'])
        sd = rotating(m['seeds']['DT'], rnd, opt.gen, d['n_dt'])
        reference, _ = evaluate(m, BASE_CFG, sd, 'dt', pols, names)
        for i in range(len(pending['done']), len(X)):
            cfg = v1.decode(X[i])
            raw, rel = evaluate(m, cfg, sd, 'dt', pols, names)
            a, b = v1.metrics(raw), v1.metrics(reference)
            rg, al = float(np.mean(a['regret']-b['regret'])), float(np.mean(a['alpha']-b['alpha']))
            worst = max(0., float(a['gains'].mean(1).max()))
            viol = max(0., (rg-p['lim_regret'])/p['lim_regret'])+max(0., (al-p['lim_alpha'])/p['lim_alpha'])
            pending['done'].append(dict(cfg=cfg, raw=rel, worst_G=worst, feasible=bool(viol == 0), fitness=float(-worst if viol == 0 else -10-viol)))
            write(path, state)
            stamp('defender', round=rnd, run=run, generation=opt.gen+1, candidate=i+1)
        scores = pending['done']
        best = max(scores, key=lambda x: x['fitness'])
        if (opt.gen+1) in d['eval_gens']:
            state['records'].append(ds_record(m, best['cfg'], pols, names, opt.gen+1))
        opt.tell(X, [s['fitness'] for s in scores])
        state['curve'].append(dict(gen=opt.gen, best=best))
        state.update(opt=opt.state_dict(), pending=None)
        write(path, state)
        publish(f'E1 v2: defender round {rnd} run{run} generation {opt.gen}')
    return state


def theta_id(theta):
    return v1.theta_id(theta)


def candidate_from_runs(m, rnd):
    """Best feasible DS checkpoint of each run, then the smaller worst G of the two runs; None if nothing is feasible."""
    winners = []
    for run in range(2):
        s = read(out_dir()/'defender'/f'r{rnd}_run{run}.json')
        if s['opt']['gen'] != m['params']['defender']['gens']:
            raise ValueError('unfinished defender training')
        w = pick_run_winner(s['records'])
        winners.append(None if w is None else dict(w, source=f'r{rnd}_run{run}/gen{w["gen"]}'))
    return pick_c1(winners), winners


def candidates_read():
    p = out_dir()/'candidates.json'
    return read(p) if p.exists() else dict()


def candidates_update(m, **kw):
    c = dict(candidates_read(), manifest_sha256=m['sha256'], **kw)
    write(out_dir()/'candidates.json', c)
    return c


# ---------------------------------------------------------------- attacker chains
def chain_budget(m, run, cks):
    a = m['params']['attacker']
    return dict(lam=a['lam'], sigma0=a['sigma0'], n_at=a['n_at'], opt_seed=a['opt_seed']+run, val_every=a['val_every'], base_gens=a['gens'],
                warm_theta_sha256=theta_id(cks[run]['theta']), at_sha256=digest(m['seeds']['AT']), as_sha256=digest(m['seeds']['AS']),
                T_train=m['params']['T_train'], T_val=m['params']['T_val'])


def chain_path(label, run):
    return out_dir()/'attacker'/f'{label}_run{run}.json'


def attack_chain(m, label, cfg, run, cks, target):
    a = m['params']['attacker']
    path = chain_path(label, run)
    opt = cmaes.make_optimizer(55, lam=a['lam'], sigma0=a['sigma0'], seed=a['opt_seed']+run)
    opt.mean = np.asarray(cks[run]['theta'], float).copy()
    budget = chain_budget(m, run, cks)
    state = state_load(path, m, dict(label=label, cfg=cfg, budget=budget, opt=opt.state_dict(), curve=[], records=[]))
    if state['cfg'] != cfg or state['budget'] != budget:
        raise ValueError('attacker chain resumed under another defender or budget')
    opt.load_state_dict(state['opt'])

    def validate(theta, gen):
        raw, rel = evaluate(m, cfg, m['seeds']['AS'], 'as', [honest_pol(), mlp(theta)], ['honest', theta_id(theta)])
        state['records'].append(dict(gen=gen, G_AS=float(v1.metrics(raw)['gains'][0].mean()), theta=np.asarray(theta).tolist(), raw=rel))

    if not state['records']:
        validate(opt.mean, 0)
        write(path, state)
    while opt.gen < target:
        check_budget()
        g = opt.gen
        sd = rotating(m['seeds']['AT'], run, g, a['n_at'])
        require_seeds(sd, 'at', m)
        t = time.time()
        X = opt.ask()
        pols = [honest_pol()]+[mlp(x) for x in X]
        raw = sim(m, cfg, pols, sd, m['params']['T_train'])
        U = np.asarray(raw['snap']['util_b'])[:, :, 1]
        if not np.isfinite(U).all():
            raise ValueError('nonfinite training utility')
        f = U[1:].mean(1)/raw['T_score']
        opt.tell(X, f)
        state['curve'].append(dict(gen=opt.gen, fitness=f.tolist(), G=((U[1:]-U[0]).mean(1)/raw['T_score']).tolist(), wall_s=time.time()-t))
        if opt.gen % a['val_every'] == 0:
            validate(opt.mean, opt.gen)
        state['opt'] = opt.state_dict()
        write(path, state)
        stamp('attacker', defender=label, run=run, generation=opt.gen, generation_seconds=time.time()-t)
        if opt.gen % (a['val_every']*2) == 0:
            publish(f'E1 v2: attacker {label} run{run} generation {opt.gen}')
    return state


def chain_best(state, max_gen=None):
    rs = [r for r in state['records'] if max_gen is None or r['gen'] <= max_gen]
    best = rs[0]
    for r in rs[1:]:
        if r['G_AS'] > best['G_AS']:
            best = r
    return best


def attack_defender(m, label, cfg, cks):
    for run in range(m['params']['attacker']['chains']):
        attack_chain(m, label, cfg, run, cks, m['params']['attacker']['gens'])


def chains_saturation(m, label, gmax):
    a = m['params']['attacker']
    out = []
    for run in range(a['chains']):
        s = read(chain_path(label, run))
        out.append(saturation([(r['gen'], r['G_AS']) for r in s['records']], a['val_every'], gmax, a['sat_threshold']))
    return out


# ---------------------------------------------------------------- stages
def pool_with(extra):
    names, pols, cks = v1.fixed_pool()
    names, pols = list(names), list(pols)
    ids = list(names)
    for nm, theta in extra:
        ids.append(dict(name=nm, theta_sha256=theta_id(theta)))
        pols.append(mlp(theta))
    return ids, pols, cks


def cfg_of(c):
    return c['cfg']


def stage_s1(m):
    ids, pols, cks = pool_with([])
    for run in range(2):
        defender(m, 1, run, ids, pols, BASE_CFG)
    c1, winners = candidate_from_runs(m, 1)
    candidates_update(m, r1_winners=winners, C1=c1)
    publish('E1 v2: round 1 candidate C1')
    attack_defender(m, 'orig', BASE_CFG, cks)
    if c1 is not None:
        attack_defender(m, 'C1', c1['cfg'], cks)
    mark_done('s1')


def stage_s2(m):
    if 's1' not in stages_done():
        raise RuntimeError('run s1 first')
    cand = candidates_read()
    c1 = cand['C1']
    if c1 is None:
        mark_done('s2')
        return
    a = m['params']['attacker']
    if 'r2_pool' not in cand:
        pool = []
        for run in range(a['chains']):
            b = chain_best(read(chain_path('C1', run)), a['gens'])
            pool.append(dict(name=f'C1_adapt{run}', theta=b['theta'], gen=b['gen']))
        cand = candidates_update(m, r2_pool=pool)
    ids, pols, cks = pool_with([(x['name'], x['theta']) for x in cand['r2_pool']])
    for run in range(2):
        defender(m, 2, run, ids, pols, c1['cfg'])
    c2, winners = candidate_from_runs(m, 2)
    same = c2 is None
    c2 = dict(c1, source='C1 (no feasible round-2 checkpoint)') if same else c2
    candidates_update(m, r2_winners=winners, C2=c2, C2_same_as_C1=same)
    publish('E1 v2: round 2 candidate C2')
    if not same:
        attack_defender(m, 'C2', c2['cfg'], cks)
    mark_done('s2')


def defender_labels(m):
    cand = candidates_read()
    labels = [('orig', BASE_CFG)]
    if cand.get('C1') is not None:
        labels.append(('C1', cand['C1']['cfg']))
        if not cand.get('C2_same_as_C1', True):
            labels.append(('C2', cand['C2']['cfg']))
    return labels


def stage_sx(m):
    if 's2' not in stages_done():
        raise RuntimeError('run s2 first')
    a = m['params']['attacker']
    _, _, cks = pool_with([])
    path = out_dir()/'saturation.json'
    sat = read(path) if path.exists() else None
    if sat is None:
        info = {}
        for label, _ in defender_labels(m):
            ch = chains_saturation(m, label, a['gens'])
            info[label] = dict(first=ch, extend=bool(any(not c['saturated'] for c in ch)))
        sat = dict(manifest_sha256=m['sha256'], decisions=info)
        write(path, sat)     # decision is registered before any extension runs
        publish('E1 v2: saturation decisions')
    for label, cfg in defender_labels(m):
        if sat['decisions'][label]['extend']:
            for run in range(a['chains']):
                attack_chain(m, label, cfg, run, cks, a['gens']+a['ext_gens'])
    final = {}
    for label, _ in defender_labels(m):
        gmax = a['gens']+(a['ext_gens'] if sat['decisions'][label]['extend'] else 0)
        ch = chains_saturation(m, label, gmax)
        final[label] = dict(gmax=gmax, chains=ch, attack_may_be_unsaturated=bool(any(not c['saturated'] for c in ch)))
    sat['final'] = final
    write(path, sat)
    mark_done('sx')


def stage_select(m):
    if 'sx' not in stages_done():
        raise RuntimeError('run sx first')
    a = m['params']['attacker']
    cand = candidates_read()
    sat = read(out_dir()/'saturation.json')
    _, _, cks = pool_with([])
    labels = defender_labels(m)
    sel_path = out_dir()/'selection_frozen.json'
    if sel_path.exists():     # crash between writing the selection and registering it: register what was written, never recompute
        if read(sel_path)['manifest_sha256'] != m['sha256']:
            raise ValueError('selection from another manifest')
        if not (out_dir()/'freeze_record.json').exists():
            write(out_dir()/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
        mark_done('select')
        return
    # attackers: 3 frozen then each defender's 3 adapted checkpoints (AS-selected over all validation points)
    attackers = [(f'frozen_run{i}', c['theta']) for i, c in enumerate(cks)]
    budgets, finals = {}, {}
    for label, _ in labels:
        budgets[label], finals[label] = [], []
        for run in range(a['chains']):
            s = read(chain_path(label, run))
            budgets[label].append(s['budget'])
            finals[label].append(s['opt']['gen'])
            attackers.append((f'{label}_adapt{run}', chain_best(s)['theta']))
    budget_consistency(budgets, finals, {k: v['extend'] for k, v in sat['decisions'].items()}, a['gens'])
    ids = ['honest']+[dict(name=n, theta_sha256=theta_id(t)) for n, t in attackers]
    pols = [honest_pol()]+[mlp(t) for _, t in attackers]
    matrix, astar = {}, {}
    for label, cfg in labels:
        raw, rel = evaluate(m, cfg, m['seeds']['AS'], 'as', pols, ids)
        G = v1.metrics(raw)['gains'].mean(1)
        matrix[label] = dict(raw=rel, G={n: float(g) for (n, _), g in zip(attackers, G)})
        i = int(np.argmax(G))   # first maximum = list order
        astar[label] = dict(name=attackers[i][0], theta=list(map(float, attackers[i][1])), theta_sha256=theta_id(attackers[i][1]), G_AS=float(G[i]))
    g_orig = astar['orig']['G_AS']
    if cand.get('C1') is None:
        choice = final_selection([], g_orig)
    else:
        cl = [('C1', astar['C1']['G_AS'])]
        if 'C2' in astar:
            cl.append(('C2', astar['C2']['G_AS']))
        choice = final_selection(cl, g_orig)
    star = dict(choice)
    if choice['status'] == 'go':
        star['cfg'] = dict(labels)[choice['label']]
        star['L_at_upper_bound'] = bool(star['cfg']['L'] >= m['params']['search']['L'][1])
    rec = dict(schema='e1-v2-selection/1', manifest_sha256=m['sha256'], status=choice['status'], choice=star, G_AS_astar={k: v['G_AS'] for k, v in astar.items()},
               astar=astar, attackers=[n for n, _ in attackers], attacker_theta_sha256=[theta_id(t) for _, t in attackers], transfer_matrix_AS=matrix, saturation=sat, candidates=cand,
               wording=TEXT['not_tested_no_feasible'] if choice['status'] == 'no_feasible' else
               TEXT['not_tested_no_improvement'] if choice['status'] == 'no_AS_improvement' else None)
    write(sel_path, rec)
    write(out_dir()/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
    mark_done('select')
    publish('E1 v2: freeze selection (commit and push before test)')


# ---------------------------------------------------------------- test (once)
def verify_freeze(m):
    d = out_dir()
    if m['smoke']:
        raise ValueError('smoke mode never opens test')
    fr, sp = d/'freeze_record.json', d/'selection_frozen.json'
    if not fr.exists() or not sp.exists():
        raise ValueError('selection is not frozen')
    rec, sel = read(fr), read(sp)
    if rec['manifest_sha256'] != m['sha256'] or sel['manifest_sha256'] != m['sha256'] or rec['selection_frozen_sha256'] != sha(sp):
        raise ValueError('selection_frozen does not match its registered hash')
    for p in (fr, sp):
        if not plan.is_committed_and_clean(str(ROOT), os.path.relpath(p, ROOT)):
            raise ValueError('selection freeze is not committed: '+p.name)
    return sel


def simulate_chunked(m, cfg, pols, sd, chunk):
    parts = [sim(m, cfg, pols, sd[i:i+chunk], m['params']['T_val']) for i in range(0, len(sd), chunk)]
    snap = {k: np.concatenate([np.asarray(p['snap'][k], float) for p in parts], axis=1).tolist() for k in main_eval.SNAP_FIELDS}
    return dict(T_score=int(parts[0]['T_score']), snap=snap)


def compute_result(m, sel, raw_o, raw_c):
    a, b = v1.metrics(raw_o), v1.metrics(raw_c)
    if raw_o['contract']['seeds'] != raw_c['contract']['seeds'] or raw_o['T_score'] != raw_c['T_score']:
        raise ValueError('paired evaluations must share seeds and window')
    D = a['gains'][0]-b['gains'][0]
    out = decide(D, b['regret']-a['regret'], b['alpha']-a['alpha'], m['params']['delta'], m['params']['lim_regret'], m['params']['lim_alpha'])
    out.update(schema='e1-v2-result/1', manifest_sha256=m['sha256'], selection_frozen_sha256=sha(out_dir()/'selection_frozen.json'),
               candidate=sel['choice']['label'], cfg=sel['choice']['cfg'], a_star={k: sel['astar'][k]['name'] for k in ('orig', sel['choice']['label'])},
               G_orig_mean=float(a['gains'][0].mean()), G_cand_mean=float(b['gains'][0].mean()),
               attack_may_be_unsaturated={k: v['attack_may_be_unsaturated'] for k, v in sel['saturation']['final'].items()},
               L_at_upper_bound=sel['choice']['L_at_upper_bound'])
    return out


def run_test():
    m = load_manifest()
    sel = verify_freeze(m)
    d = out_dir()
    if sel['status'] != 'go':
        res = dict(schema='e1-v2-result/1', manifest_sha256=m['sha256'], category='not_tested', text=sel['wording'], test_seeds_used=False)
        write(d/'result.json', res)
        publish('E1 v2: no candidate entered test')
        return res
    sd = m['seeds']['TEST']
    require_seeds(sd, 'test', m, unlocked=True)
    if len(sd) != m['params']['n_test']:
        raise ValueError('test seed count changed')
    star = sel['choice']['label']
    cfgs = dict(orig=BASE_CFG, cand=sel['choice']['cfg'])
    names = dict(orig=sel['astar']['orig'], cand=sel['astar'][star])
    freeze_sha = sha(d/'freeze_record.json')
    raws = {}

    def go(name, which, thetas):
        pols = [honest_pol()]+[mlp(t) for t in thetas]
        con = dict(manifest_sha256=m['sha256'], freeze_record_sha256=freeze_sha, stage=name, cfg=cfgs[which], seeds=list(sd),
                   n_policies=len(pols), T_score=m['params']['T_val']-m['params']['exclude'], T=m['params']['T_val'],
                   exclude=m['params']['exclude'], theta_sha256=[theta_id(t) for t in thetas])
        raw = once(d/'test', name, con, lambda: simulate_chunked(m, cfgs[which], pols, sd, m['params']['test_chunk']))
        stamp('test', evaluation=name)
        publish(f'E1 v2: test evaluation {name}')
        return raw
    raws['o'] = go('orig_main', 'orig', [names['orig']['theta']])
    raws['c'] = go('cand_main', 'cand', [names['cand']['theta']])
    res = compute_result(m, sel, raws['o'], raws['c'])
    load_manifest(clean=False)
    write(d/'result.json', res)
    publish('E1 v2: primary decision')
    print(json.dumps({k: res[k] for k in ('category', 'sub', 'LB', 'UB_D', 'H', 'E')}), flush=True)
    # secondary (descriptive only): all attackers of the original and final defenders
    thetas = secondary_thetas(sel)
    so = go('orig_secondary', 'orig', [t for _, t in thetas])
    sc = go('cand_secondary', 'cand', [t for _, t in thetas])
    A, B = v1.metrics(so), v1.metrics(sc)
    D = A['gains']-B['gains']
    sec = dict(schema='e1-v2-secondary/1', note='descriptive only; uncorrected for multiplicity', attackers=[n for n, _ in thetas],
               G_orig=A['gains'].mean(1).tolist(), G_cand=B['gains'].mean(1).tolist(), D_mean=D.mean(1).tolist(),
               D_LB=[lower_bound(x) for x in D], honest_per_policy_note='honest = policy 0 in each evaluation')
    write(d/'secondary.json', sec)
    publish('E1 v2: secondary descriptive results')
    return res


def secondary_thetas(sel):
    """The 12 (or fewer) attackers' thetas as stored in the AS chain records are not in the selection file; reload from chains."""
    d = out_dir()
    a_names = sel['attackers']
    _, _, cks = pool_with([])
    out = [(f'frozen_run{i}', c['theta']) for i, c in enumerate(cks)]
    for n in a_names[len(cks):]:
        label, run = n.rsplit('_adapt', 1)
        out.append((n, chain_best(read(chain_path(label, int(run))))['theta']))
    if [theta_id(t) for _, t in out] != list(sel['attacker_theta_sha256']):
        raise ValueError('attacker checkpoints differ from the frozen selection')
    return out


# ---------------------------------------------------------------- CLI
def plan_summary(m):
    p = m['params']
    a = p['attacker']
    return dict(smoke=m['smoke'], manifest_sha256=m['sha256'], seeds={k: [v[0], v[-1], len(v)] for k, v in m['seeds'].items()},
                stages=['s1: round1 defenders(2) + orig/C1 attacks', 's2: round2 defenders(2) + C2 attacks', 'sx: saturation extension (0-8.8h)',
                        'select: transfer eval, a*, C*, freeze', 'test: once'],
                attacker_gens=a['gens'], extension_gens=a['ext_gens'], defender_gens=p['defender']['gens'],
                estimated_hours_without_extension=None if m['smoke'] else 24.8, estimated_hours_worst_case=None if m['smoke'] else 33.6,
                stages_done=sorted(stages_done()), test_opens_only_after='select + committed freeze_record.json')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('cmd', choices=['audit', 'freeze', 'run', 'test', 'status', 'dry-run', 'report'])
    ap.add_argument('--stage', choices=['s1', 's2', 'sx', 'select', 'all'], default='all')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--backup', action='store_true')
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--graph-fallback', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    MODE.update(smoke=a.smoke, device=a.device, graph=a.graph, fallback=a.graph_fallback, backup=a.backup)
    GRAPH.failed = (out_dir()/'graph_fallback.json').exists()
    (ROOT/'.scratch').mkdir(exist_ok=True)
    if a.cmd == 'audit':
        blocks = block_seeds(a.smoke)
        hist, _ = scan_saved_seeds()
        print(json.dumps(seed_report(blocks, existing_seeds(), hist), indent=1))
        return 0
    if a.cmd in ('status', 'dry-run', 'report'):
        m = load_manifest(clean=False)
        if a.cmd == 'report':
            sel = read(out_dir()/'selection_frozen.json')
            d = out_dir()/'test'
            res = compute_result(m, sel, read(d/'orig_main_raw.json'), read(d/'cand_main_raw.json'))
            print(json.dumps({k: res[k] for k in ('category', 'sub', 'LB', 'UB_D', 'H', 'E')}))
            same = read(out_dir()/'result.json') == res
            print('matches result.json:', same)
            return 0 if same else 1
        print(json.dumps(plan_summary(m), indent=1, ensure_ascii=False))
        if a.cmd == 'status' and (out_dir()/'status.json').exists():
            print(json.dumps(read(out_dir()/'status.json'), indent=1))
        return 0
    with (ROOT/'.scratch/e1-v2.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.cmd == 'freeze':
            freeze(a.smoke)
            return 0
        import torch
        torch.set_num_threads(1)
        MODE['start'] = time.time()
        try:
            m = load_manifest(clean=not a.smoke)
            if a.cmd == 'test':
                run_test()
            else:
                stamp('running', started_unix=time.time(), manifest_sha256=m['sha256'])
                order = ['s1', 's2', 'sx', 'select']
                fns = dict(s1=stage_s1, s2=stage_s2, sx=stage_sx, select=stage_select)
                for st in (order if a.stage == 'all' else [a.stage]):
                    if st in stages_done():
                        continue
                    fns[st](m)
                    publish(f'E1 v2: stage {st} done')
                stamp('complete' if 'select' in stages_done() else 'partial', finished_unix=time.time())
        except BudgetStop:
            stamp('budget_stop', message='18 hour cap reached at a save boundary; relaunch to resume')
            publish('E1 v2: saved at time cap')
            return 75
        except Exception as exc:
            stamp('failed', error_type=type(exc).__name__, message=str(exc)[:300])
            try:
                publish('E1 v2: preserve failed run checkpoints')
            except Exception:
                pass
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
