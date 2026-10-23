#!/usr/bin/env python3
"""E-A1: attacker-strength control on C2 (design: docs/direction2/design/e-a1.md; pre-registered, frozen at commit c66305d).

Stages: audit / freeze (S0) / smoke / paracheck / train / postcheck / select / testcheck / test (once) / report.  Every GPU stage is resumable.
Three attack chains are trained from theta=0 on C2 (lam=256, n_S=256, sigma0=0.3, 416 generations); `train` may run them as 3 lanes
(thread + CUDA stream + GraphSim each) in one process when `paracheck` showed bitwise-identical results and a throughput ratio >= 1.10.
The test block 82400000-82401023 is opened only by `test`, only after selection_frozen.json is registered (freeze_record.json) and committed
and after a passing testcheck.  Nothing here modifies the E1 v2 files; all E1 v2 hashes are re-checked on every load.
"""
import argparse
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import cmaes, graphsim, main_eval, multitrain, plan
from scripts import e1_pilot as v1
from scripts import e1_v2 as e1
from scripts.d2_holdout_eval import read, write, sha, digest, context, seed_values, once

DESIGN = "docs/direction2/design/e-a1.md"
OUT_DIR = "results/direction2/ea1"
SMOKE_DIR = "results/direction2/ea1_smoke"
E1V2_DIR = "results/direction2/e1v2"
BASE_CFG = v1.BASE_CFG
OFFS4 = e1.OFFS4
# EA1_TRAIN: attack-chain training seeds (pool of 1000, as the original chain); EA1_SEL: per-chain validation, selection, re-evaluation and
# testcheck (first 64); EA1_TEST: the one-shot test.  All three are new (design 1.3).
BLOCKS = dict(EA1_TRAIN=(82_000_000, 1000), EA1_SEL=(82_100_000, 128), EA1_TEST=(82_400_000, 1024))
STAGE_BLOCK = dict(train='EA1_TRAIN', sel='EA1_SEL', testcheck='EA1_SEL', test='EA1_TEST')
DRIFT_BLOCK = (72_300_000, 128)             # E1 v2 AS: reused only for the descriptive machine-drift check, never as a new block
EXPECT_NEW, EXPECT_OLD = 8608, 12064
NEW_SHA = "05232e843d5dd370d0c5bdfef416924990d92021d2b50c01a25ba14ec5eabe97"      # registered in design 1.3 (pre-run)
OLD_SHA = "7cdc62f47788f8c38b25b93e78adb2fdf695151693344b175835eefea1afbd85"
INPUTS = ["scripts/ea1.py", "scripts/ea1_launch.sh", "tests/direction2/test_ea1.py", DESIGN]
ASTAR_SHA256 = "88ed926ae8c1406bdb9c1d29be57935d00455b82e96ebe05861c2970fa561016"
C2_CFG = dict(p=0.1, tol=6.175461735115074, kref=0.4663589529547838, L=2412.0)
ENV_PIN = dict(device_name="Quadro RTX 4000", torch="2.6.0+cu126", numpy="2.2.6", scipy="1.15.3")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

PARAMS = dict(
    schema="ea1/1", r=.5, base_cfg=BASE_CFG, c2_cfg=C2_CFG, astar_theta_sha256=ASTAR_SHA256, T_train=20000, T_val=100000, exclude=5000,
    eps=.001, ci_level=.95, n_test=1024, test_chunk=64, testcheck_n=64,
    attacker=dict(lam=256, sigma0=.3, n_S=256, gens=416, val_every=50, chains=3, opt_seed=94000, perm_seed=94100, post_check_gen=200),
    lanes=dict(ratio_threshold=1.10, mem_limit_gb=6.0), kernel_K=32)
SMOKE_PARAMS = dict(PARAMS, schema="ea1-smoke/1", n_test=0)

MODE = dict(smoke=False, device="cuda", graph=True, out=None, start=None, strict=False, yield_check=False, K=None)
GEN_HOOK = None                    # test hook: called as GEN_HOOK(run, gen, rec) after each saved generation
_STATUS_LOCK = threading.Lock()


class BudgetStop(Exception):
    pass


class YieldStop(Exception):
    """Another account started a GPU program: finish the generation, save, exit 75 (astarc shared-machine rule)."""


class PostCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- seeds
def block_seeds(smoke=False):
    b = {k: list(range(a, a+n)) for k, (a, n) in BLOCKS.items()}
    if smoke:
        b.pop('EA1_TEST')
    return b


def expand(base, offsets=OFFS4):
    return e1.expand(base, offsets)


def existing_all(root=ROOT):
    """The seven existing ranges plus the five E1 v2 blocks (12064 expanded seeds)."""
    ex = e1.existing_seeds(root)
    for k, (a, n) in e1.BLOCKS.items():
        ex['e1v2_'+k] = list(range(a, a+n))
    return ex


def seed_report(blocks, existing, extra_historical=()):
    return e1.seed_report(blocks, existing, extra_historical)


def scan_saved_seeds(root=ROOT):
    """Integer seed fields of saved json/npz records (excluding this experiment's own output)."""
    vals, srcs = set(), {}
    for p in sorted((Path(root)/'results').rglob('*')):
        rel = str(p.relative_to(root))
        if rel.startswith('results/direction2/ea1') or not p.is_file() or p.suffix not in ('.json', '.npz'):
            continue
        if p.suffix == '.json':
            vals |= seed_values(json.loads(p.read_text()))
        else:
            with np.load(p, allow_pickle=False) as z:
                if 'state' in z.files:
                    vals |= seed_values(json.loads(str(z['state'])))
        srcs[rel] = sha(p)
    return vals, srcs


def require_seeds(sd, stage, m, unlocked=False):
    """train -> EA1_TRAIN; sel -> EA1_SEL; testcheck -> exactly EA1_SEL[:64]; test -> EA1_TEST (only unlocked); drift -> the E1 v2 AS seeds."""
    sd = list(sd)
    if stage == 'drift':
        pool = (m.get('drift') or {}).get('seeds')
    elif stage in STAGE_BLOCK:
        if stage == 'test' and not unlocked:
            raise ValueError('test seeds are locked until selection_frozen is registered')
        pool = m['seeds'].get(STAGE_BLOCK[stage])
    else:
        raise ValueError('unknown stage '+str(stage))
    if pool is None:
        raise ValueError('this manifest has no block for stage '+stage)
    if len(sd) != len(set(sd)):
        raise ValueError('duplicate environment seeds')
    if not sd or any(type(s) is not int or s not in set(pool) for s in sd):
        raise ValueError('wrong seed split')
    if stage == 'testcheck' and sd != list(m['seeds']['EA1_SEL'][:m['params']['testcheck_n']]):
        raise ValueError('testcheck must use exactly the first EA1_SEL seeds')


@functools.lru_cache(maxsize=None)
def perm_for(n, seed):
    """Training-pool order: Random(seed).shuffle(range(n)).  The pool is indexed through it (design 2.3: seed 94100+run, never 9000+run)."""
    p = list(range(n))
    random.Random(seed).shuffle(p)
    return tuple(p)


def train_block(pool, run, g, n_S, perm_seed=PARAMS['attacker']['perm_seed']):
    """Seeds of generation g: block [g*n_S, (g+1)*n_S) (mod len(pool)) of the permuted pool; same index rule as train.seed_block."""
    if not 1 <= n_S <= len(pool):
        raise ValueError('n_S must lie in [1, len(pool)] so that a block never repeats a seed')
    p = perm_for(len(pool), perm_seed+run)
    return [pool[p[i % len(pool)]] for i in range(g*n_S, (g+1)*n_S)]


# ---------------------------------------------------------------- statistics / decision
tcrit, upper_bound, lower_bound = e1.tcrit, e1.upper_bound, e1.lower_bound   # SD ddof=1, t=stats.t.ppf(.95, n-1) (design 2.1)

TEXT = dict(
    stronger='更強：SEL 選出的 A_fresh 在 C2 下比 a* 強至少 0.001（單側 95% 下界 > 0.001）。',
    weaker='沒有更強（更弱）：A_fresh 在 C2 下不如 a*（單側 95% 上界 < 0）；先驗預期本來就是 Δ<0，不是獨立證據。',
    small_gain='沒有更強（小幅提升但未達 ε）：A_fresh 顯著高於 a*（下界 > 0），但排除了高出 0.001 以上。',
    flat='沒有更強（持平）：無法排除略高於 0 的差，但排除了 A_fresh 高出 a* 達 0.001 以上。',
    insufficient='證據不足以判定：單側區間跨過 0.001，既不能宣稱更強，也不能排除更強。')


def classify(lb, ub, eps=PARAMS['eps']):
    """Design 2.2 on the two one-sided bounds: first hit wins, strict inequalities (equality with eps or 0 hits nothing)."""
    if lb > eps:
        return 'stronger', None
    if ub < eps:
        return 'not_stronger', 'weaker' if ub < 0 else 'small_gain' if lb > 0 else 'flat'
    return 'insufficient', None


def decide(D, eps=PARAMS['eps']):
    D = np.asarray(D, float)
    lb, ub = lower_bound(D), upper_bound(D)
    cat, sub = classify(lb, ub, eps)
    return dict(category=cat, sub=sub, text=TEXT[sub or cat], n=len(D), mean_delta=float(D.mean()), LB=lb, UB=ub, eps=eps,
                subject='A_fresh (the single attacker selected on EA1_SEL)')


def power_table(n, sigma, eps, mus):
    """Design 2.2 table: P(LB>eps), P(UB<eps), P(UB<0) for a true mean mu (non-central t)."""
    t, se = tcrit(n), sigma/math.sqrt(n)
    return [dict(mu=mu, p_stronger=float(stats.nct.sf(t, n-1, (mu-eps)/se)), p_ub_lt_eps=float(stats.nct.cdf(-t, n-1, (mu-eps)/se)),
                 p_weaker=float(stats.nct.cdf(-t, n-1, mu/se))) for mu in mus]


def interval(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), LB=lower_bound(x), UB=upper_bound(x), n=len(x))


def first_max(vals):
    """Index of the first maximum (ties go to the earlier / smaller-numbered one); non-finite values are an error."""
    vals = [float(v) for v in vals]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError('selection needs finite values')
    best = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[best]:
            best = i
    return best


def pick_c2_prime(g_astar, g_fresh):
    """A_C2' = whichever of a* and A_fresh has the larger EA1_SEL G_C2 (same re-evaluation); ties go to a*."""
    return 'A_fresh' if g_fresh > g_astar else 'a_star'


def decide_lanes(t1, t3, parallel_ok, graph_ok, mem_gb, threshold=PARAMS['lanes']['ratio_threshold'], mem_limit=PARAMS['lanes']['mem_limit_gb']):
    """Design 3.3: lanes=3 iff 3*t1/t3 >= 1.10 and A=B=C bitwise and GPU memory peak <= 6 GB; otherwise 1.  A!=B means the graph path is unusable."""
    if not graph_ok:
        return dict(lanes=None, reason='graph path differs from eager (A != B): report to the user, do not switch to eager')
    ratio = None if not t3 else 3*t1/t3          # no t3: the 3-lane run failed
    ok = bool(parallel_ok and ratio is not None and ratio >= threshold and mem_gb is not None and mem_gb <= mem_limit)
    why = []
    if not parallel_ok:
        why.append('3-lane run failed or B != C')
    if ratio is not None and ratio < threshold:
        why.append(f'throughput ratio {ratio:.3f} < {threshold}')
    if mem_gb is not None and mem_gb > mem_limit:
        why.append(f'GPU memory peak {mem_gb:.2f} GB > {mem_limit}')
    return dict(lanes=3 if ok else 1, ratio=ratio, reason='adopted' if ok else '; '.join(why))


# ---------------------------------------------------------------- manifest
def out_dir():
    if MODE.get('out'):
        return Path(MODE['out'])
    return ROOT/(SMOKE_DIR if MODE['smoke'] else OUT_DIR)


def git(*a):
    return subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()


def environment():
    """Machine / library record (design 1.5).  Fields that cannot be read (no GPU, no nvidia-smi) are None."""
    import torch
    d = dict(python=platform.python_version(), d2_graph_k=os.environ.get('D2_GRAPH_K'), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
             hostname=socket.gethostname(), threads={k: os.environ.get(k) for k in THREAD_VARS}, device_name=None, capability=None,
             gpu_name=None, gpu_uuid=None, driver=None)
    if torch.cuda.is_available():
        d.update(device_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            d['gpu_name'], d['gpu_uuid'], d['driver'] = [x.strip() for x in out.stdout.strip().splitlines()[0].split(',')]
    except (OSError, subprocess.SubprocessError):
        pass
    return d


ENV_CHECKED = ('device_name', 'capability', 'gpu_name', 'gpu_uuid', 'torch', 'cuda', 'numpy', 'scipy', 'threads')


def check_environment(m, env=None):
    """Every GPU stage: GPU model, torch / CUDA / numpy / scipy versions and thread variables must equal the manifest's record."""
    env = env or environment()
    if env.get('d2_graph_k') not in (None, '', str(PARAMS['kernel_K'])):
        raise ValueError(f'D2_GRAPH_K={env["d2_graph_k"]} differs from the registered kernel K {PARAMS["kernel_K"]} (graphsim.from_env would read it)')
    bad = [k for k in ENV_CHECKED if env.get(k) != m['environment'].get(k)]
    if bad:
        raise ValueError('execution environment differs from the registered manifest: '+', '.join(f'{k}: {env.get(k)!r} != {m["environment"].get(k)!r}' for k in bad))


def e1v2_manifest():
    return read(ROOT/E1V2_DIR/'manifest.json')


def e1v2_refs(root=ROOT):
    """E1 v2 results that E-A1 reads: their hashes go into the manifest (design 6, 'E1 v2 not affected')."""
    root = Path(root)
    sel = read(root/E1V2_DIR/'selection_frozen.json')
    rels = ['manifest.json', 'selection_frozen.json', 'attacker/orig_run0.json', 'result.json']
    rels += [v['raw'] for v in sel['transfer_matrix_AS'].values() if v['raw'] not in rels]
    rels += [f'attacker/{n.rsplit("_adapt", 1)[0]}_run{n.rsplit("_adapt", 1)[1]}.json' for n in sel['attackers'] if '_adapt' in n]
    return {E1V2_DIR+'/'+r: sha(root/E1V2_DIR/r) for r in dict.fromkeys(rels)}


def build_manifest(smoke=False, root=ROOT):
    root = Path(root)
    _, cks, _ = context()
    blocks = block_seeds(smoke)
    existing = existing_all(root)
    hist, srcs = (set(), {}) if smoke else scan_saved_seeds(root)
    if not smoke:
        hist |= set(read(root/'results/direction2/holdout/seed_audit.json')['historical_seed_field_values'])
    rep = seed_report(blocks, existing, hist)
    if not smoke and (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) != (EXPECT_NEW, EXPECT_OLD, 0):
        raise ValueError('seed counts differ from the registered 8608/12064/0')
    if not smoke and (rep['new_expanded_sha256'], rep['existing_expanded_sha256']) != (NEW_SHA, OLD_SHA):
        raise ValueError('seed set hashes differ from the pre-run registered ones (design 1.3)')
    audit = dict(schema='ea1-seed-audit/1', report=rep, offsets=list(OFFS4), saved_seed_values=sorted(hist), scanned_sha256=srcs,
                 limitation='Saved project records only; cannot certify unrecorded external executions.')
    e1m = e1v2_manifest()
    protected = dict(e1m['protected_sha256'])
    protected.update(e1m['input_sha256'])
    protected[e1m['design']] = e1m['design_sha256']
    sel = read(root/E1V2_DIR/'selection_frozen.json')
    if sel['astar']['orig']['theta_sha256'] != ASTAR_SHA256 or sel['choice']['cfg'] != C2_CFG:
        raise ValueError('E1 v2 selection differs from the registered a* / C2')
    drift = list(range(DRIFT_BLOCK[0], DRIFT_BLOCK[0]+DRIFT_BLOCK[1]))
    if e1m['seeds']['AS'] != drift:
        raise ValueError('E1 v2 AS seeds differ from the registered drift-check seeds')
    params = SMOKE_PARAMS if smoke else PARAMS
    m = dict(schema=params['schema'], smoke=smoke, params=params, design=DESIGN, design_sha256=sha(root/DESIGN), code_git_sha=git('rev-parse', 'HEAD'),
             seeds=blocks, drift=dict(seeds=drift, use='descriptive machine-drift check of a* only; not a new block'),
             seed_expanded_sha256=rep['new_expanded_sha256'], seed_report=rep, checkpoints=main_eval.checkpoint_meta(cks), source_sha256=main_eval.code_hash(),
             input_sha256={p: sha(root/p) for p in INPUTS if (root/p).exists()}, protected_sha256=protected, e1v2_refs_sha256=e1v2_refs(root),
             environment=environment(), stage_blocks=STAGE_BLOCK,
             test_unlock='selection_frozen.json sha256 recorded in freeze_record.json (committed) and a passing testcheck.json before any EA1_TEST seed is read')
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
    pins = {k: (m['environment'].get(k), v) for k, v in ENV_PIN.items() if m['environment'].get(k) != v}
    if pins and not smoke:
        raise ValueError(f'environment differs from the design pins (freeze only on astarc): {pins}')
    if pins:
        print('smoke: environment differs from the design pins (recorded as is):', pins, flush=True)
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
    check_hashes({**m['input_sha256'], **m['protected_sha256'], **m['e1v2_refs_sha256'], DESIGN: m['design_sha256']})
    if sha(d/'seed_audit.json') != m['seed_audit_sha256']:
        raise ValueError('seed audit changed')
    audit = read(d/'seed_audit.json')
    rep = seed_report(m['seeds'], existing_all(), audit['saved_seed_values'])
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


_YIELD = dict(event=threading.Event(), last=0.0, lock=threading.Lock())


def foreign_gpu_pids():
    """Compute processes on the GPU owned by another account (astarc rule: yield to other accounts only)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    foreign = []
    for x in out.stdout.split():
        if not x.strip().isdigit():
            continue
        try:
            if os.stat(f'/proc/{int(x)}').st_uid == os.getuid():
                continue                      # our own account's GPU programs never count (any number of them is allowed)
        except OSError:
            pass                              # cannot read the owner (gone, hidden): treat as another account
        foreign.append(int(x))
    return foreign


def check_yield(min_interval=10.0):
    """Called at generation boundaries by every lane; the first lane that sees a foreign GPU program stops all lanes."""
    if not MODE.get('yield_check'):
        return
    if _YIELD['event'].is_set():
        raise YieldStop()
    with _YIELD['lock']:
        if time.time()-_YIELD['last'] >= min_interval:
            _YIELD['last'] = time.time()
            if foreign_gpu_pids():
                _YIELD['event'].set()
    if _YIELD['event'].is_set():
        raise YieldStop()


def stamp(stage, **kw):
    with _STATUS_LOCK:
        p = out_dir()/'status.json'
        old = read(p) if p.exists() else {}
        write(p, dict(old, stage=stage, updated_unix=time.time(), **kw))
        print(stage, json.dumps(kw), flush=True)


def mark_done(name):
    with _STATUS_LOCK:
        p = out_dir()/'status.json'
        old = read(p) if p.exists() else {}
        write(p, dict(old, stages_done=sorted(set(old.get('stages_done', []))|{name}), updated_unix=time.time()))


def stages_done():
    p = out_dir()/'status.json'
    return set(read(p).get('stages_done', [])) if p.exists() else set()


class Backend:
    """What one lane simulates with: device, optional GraphSim (graph stepping) and optional CUDA stream.  `with backend:` puts the whole
    thread body on the lane's stream (multitrain.run_segment does the same)."""

    def __init__(self, dev, gs=None, stream=None):
        self.dev, self.gs, self.stream, self._ctx = dev, gs, stream, None

    def simulate(self, cfg, pols, sd, T, exclude):
        return v1.simulate(cfg, pols, sd, T, self.gs, self.dev, exclude)

    def __enter__(self):
        if self.stream is not None:
            import torch
            self._ctx = torch.cuda.stream(self.stream)
            self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        if self._ctx is not None:
            import torch
            torch.cuda.current_stream().synchronize()
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        return False

    def release(self):
        if self.gs is not None:
            self.gs.cache.clear()


def kernel_K():
    return int(MODE.get('K') or PARAMS['kernel_K'])          # D2_GRAPH_K is not read here; check_environment refuses it if it differs


def make_backend(dev=None, graph=True):
    """A fresh lane: own CUDA stream and own GraphSim on cuda; GraphSim(capture=False) (same machinery, no CUDA graph) on cpu."""
    dev = dev or MODE['device']
    if dev.startswith('cuda'):
        lane = multitrain.Lane(graph, kernel_K(), None, dev)
        return Backend(dev, lane.gs, lane.stream)
    return Backend(dev, graphsim.GraphSim(K=kernel_K(), capture=False) if graph else None, None)


_DEFAULT = {}


def default_backend():
    """The default-stream backend of test / testcheck (the E1 v2 verified path): graph on cuda when MODE['graph'], never a fallback."""
    key = (MODE['device'], bool(MODE['graph']))
    if key not in _DEFAULT:
        gs = graphsim.from_env(MODE['device'], force=True) if MODE['graph'] and MODE['device'].startswith('cuda') else None
        _DEFAULT.clear()
        _DEFAULT[key] = Backend(MODE['device'], gs, None)
    return _DEFAULT[key]


def sim(m, cfg, pols, sd, T, be=None):
    """One simulate call.  A graph error is always a hard error (no eager fallback anywhere in E-A1)."""
    return (be or default_backend()).simulate(cfg, pols, sd, T, m['params']['exclude'])


def sim_eager(m, cfg, pols, sd, T):
    """Reference path for testcheck: no graph, one call (not chunked)."""
    return v1.simulate(cfg, pols, sd, T, None, MODE['device'], m['params']['exclude'])


def simulate_chunked(m, cfg, pols, sd, chunk):
    parts = [sim(m, cfg, pols, sd[i:i+chunk], m['params']['T_val']) for i in range(0, len(sd), chunk)]
    snap = {k: np.concatenate([np.asarray(p['snap'][k], float) for p in parts], axis=1).tolist() for k in main_eval.SNAP_FIELDS}
    return dict(T_score=int(parts[0]['T_score']), snap=snap)


def evaluate(m, cfg, sd, stage, pols, ids, be=None, unlocked=False):
    """Cached evaluation at T_val (one simulate call for all `pols`; policy 0 is the honest row)."""
    require_seeds(sd, stage, m, unlocked)
    check_budget()
    p = m['params']
    con = dict(manifest_sha256=m['sha256'], cfg=cfg, seeds=list(sd), stage=stage, T=p['T_val'], exclude=p['exclude'], n_policies=len(pols), policies=ids)
    path = out_dir()/'evaluations'/(digest(con)+'.json')
    if path.exists():
        return v1.validate_snapshot(read(path), con), str(path.relative_to(out_dir()))
    t = time.time()
    raw = dict(sim(m, cfg, pols, sd, p['T_val'], be), contract=con, wall_s=time.time()-t)
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


mlp, honest_pol, theta_id = e1.mlp, e1.honest_pol, e1.theta_id


def bitwise_equal(a, b):
    """Bit-for-bit equality of two numeric arrays (np.array_equal semantics are not enough: -0.0 and NaN payloads differ)."""
    a, b = np.ascontiguousarray(np.asarray(a, float)), np.ascontiguousarray(np.asarray(b, float))
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------- attack chains
def make_opt(m, run):
    a = m['params']['attacker']
    return cmaes.make_optimizer(55, lam=a['lam'], sigma0=a['sigma0'], seed=a['opt_seed']+run)   # mean = 0: cold start from the honest policy


def chain_budget(m, run):
    a, p = m['params']['attacker'], m['params']
    return dict(lam=a['lam'], sigma0=a['sigma0'], n_S=a['n_S'], gens=a['gens'], val_every=a['val_every'], opt_seed=a['opt_seed']+run,
                perm_seed=a['perm_seed']+run, T_train=p['T_train'], T_val=p['T_val'], exclude=p['exclude'], start='zeros',
                train_sha256=digest(m['seeds']['EA1_TRAIN']), sel_sha256=digest(m['seeds']['EA1_SEL']))


def chain_path(run, name=None):
    return out_dir()/'chains'/(f'run{run}.json' if name is None else f'run{run}_{name}.json')


def chain_generation(m, cfg, run, opt, be):
    """One generation: ask, one simulate call over [honest]+lam candidates on the n_S training seeds of this generation, tell.
    fitness = mean per-step utility over the score window T-5000 (design 2.3)."""
    a = m['params']['attacker']
    sd = train_block(m['seeds']['EA1_TRAIN'], run, opt.gen, a['n_S'], a['perm_seed'])
    require_seeds(sd, 'train', m)
    X = opt.ask()
    raw = sim(m, cfg, [honest_pol()]+[mlp(x) for x in X], sd, m['params']['T_train'], be)
    U = np.asarray(raw['snap']['util_b'], float)[:, :, 1]
    if not np.isfinite(U).all():
        raise ValueError('nonfinite training utility')
    f = U[1:].mean(1)/raw['T_score']
    gen = opt.gen
    opt.tell(X, f)
    return dict(gen=gen, X=X, U=U, f=f, seeds=sd, T_score=raw['T_score'])


def run_chain(m, run, be, target=None):
    """Train (or resume) chain `run` to `target` generations (default: all).  Saved after every generation; validation every val_every."""
    a, cfg = m['params']['attacker'], m['params']['c2_cfg']
    target = a['gens'] if target is None else target
    if not 0 < target <= a['gens']:
        raise ValueError('a chain is never extended beyond the registered generations')
    path = chain_path(run)
    opt = make_opt(m, run)
    budget = chain_budget(m, run)
    state = state_load(path, m, dict(run=run, cfg=cfg, budget=budget, opt=opt.state_dict(), curve=[], records=[]))
    if state['cfg'] != cfg or state['budget'] != budget:
        raise ValueError('chain resumed under another defender or budget')
    opt.load_state_dict(state['opt'])
    if len(state['curve']) != opt.gen:
        raise ValueError('chain state is inconsistent')
    post = a['post_check_gen']
    while opt.gen < target:
        check_budget()
        check_yield()
        t = time.time()
        if opt.gen == post-1 and not chain_path(run, f'snap{post-1}').exists():
            write(chain_path(run, f'snap{post-1}'), dict(manifest_sha256=m['sha256'], opt=opt.state_dict()))     # state after generation post-1
        rec = chain_generation(m, cfg, run, opt, be)
        if rec['gen'] == post-1:
            write(chain_path(run, f'gen{post}'), dict(manifest_sha256=m['sha256'], X=rec['X'].tolist(), f=rec['f'].tolist()))
        f_h = float(rec['U'][0].mean()/rec['T_score'])
        state['curve'].append(dict(gen=opt.gen, f_honest=f_h, f_mean=float(rec['f'].mean()), f_max=float(rec['f'].max()), wall_s=time.time()-t))
        if opt.gen % a['val_every'] == 0:
            raw, rel = evaluate(m, cfg, m['seeds']['EA1_SEL'], 'sel', [honest_pol(), mlp(opt.mean)], ['honest', theta_id(opt.mean)], be)
            g = float(v1.metrics(raw)['gains'][0].mean())
            if not math.isfinite(g):
                raise ValueError('nonfinite validation gain')
            state['records'].append(dict(gen=opt.gen, G_SEL=g, theta=np.asarray(opt.mean).tolist(), raw=rel))
        state['opt'] = opt.state_dict()
        write(path, state)
        stamp('train', run=run, generation=opt.gen, generation_seconds=time.time()-t)
        if GEN_HOOK is not None:
            GEN_HOOK(run, rec['gen'], rec)
    return state


def run_lanes(m, runs, lanes, factory=None, order=None, delays=None):
    """lanes=3: one thread (own lane) per chain; lanes=1: chains one after the other on one lane.  A failing lane does not stop the others;
    the stage still fails afterwards.  `order` / `delays` only reorder thread starts (tests of interleaving)."""
    factory = factory or make_backend
    runs = list(runs)
    errs = {}

    def work(run):
        try:
            if delays and delays.get(run):
                time.sleep(delays[run])
            be = factory()
            with be:
                run_chain(m, run, be)
            be.release()
        except BaseException as exc:                  # noqa: BLE001 - one lane must never take the others down
            errs[run] = exc
    if lanes == 1:
        for r in runs:
            work(r)
    else:
        ths = [threading.Thread(target=work, args=(r,), name=f'lane-{r}') for r in (order or runs)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    if errs:
        hard = {r: x for r, x in errs.items() if not isinstance(x, (BudgetStop, YieldStop))}
        raise (next(iter(hard.values())) if hard else next(iter(errs.values())))


def chain_state(run):
    return read(chain_path(run))


def chains_complete(m):
    a = m['params']['attacker']
    for run in range(a['chains']):
        if not chain_path(run).exists():
            raise ValueError(f'chain {run} has not started')
        s = chain_state(run)
        if s['opt']['gen'] != a['gens'] or len(s['curve']) != a['gens']:
            raise ValueError(f'chain {run} is not complete (generation {s["opt"]["gen"]} of {a["gens"]})')
        want = list(range(a['val_every'], a['gens']+1, a['val_every']))
        if [r['gen'] for r in s['records']] != want or not all(math.isfinite(r['G_SEL']) for r in s['records']):
            raise ValueError(f'chain {run} has missing or non-finite validation points')
        if not all(np.isfinite(np.asarray(s['opt'][k], float)).all() for k in ('mean', 'C', 'pc', 'ps')) or not math.isfinite(s['opt']['sigma']):
            raise ValueError(f'chain {run} optimizer state is not finite')


def best_record(records):
    return records[first_max([r['G_SEL'] for r in records])]


def chain_result(state):
    """A chain's state without wall-clock fields (what must be bitwise equal across execution modes)."""
    return dict(state, curve=[{k: v for k, v in c.items() if k != 'wall_s'} for c in state['curve']])


def read_run_config(m):
    p = out_dir()/'run_config.json'
    if not p.exists():
        raise ValueError('run_config.json is missing: run the paracheck stage first')
    rc = read(p)
    if rc['manifest_sha256'] != m['sha256'] or rc['lanes'] not in (1, 3):
        raise ValueError('run_config.json does not match the manifest')
    return rc


def require_parallel_check(m):
    """Before `train`: parallel_check.json passed (A=B at least; B=C when lanes=3), matches this manifest and this code, run_config agrees."""
    d = out_dir()
    p = d/'parallel_check.json'
    if not p.exists():
        raise ValueError('parallel_check.json is missing: run the paracheck stage first')
    pc = read(p)
    if pc['manifest_sha256'] != m['sha256'] or pc['ea1_py_sha256'] != sha(ROOT/'scripts/ea1.py') or not pc['graph_equals_eager']:
        raise ValueError('parallel_check.json failed or does not match the current manifest / code')
    rc = read_run_config(m)
    if rc['lanes'] == 3 and not (pc['parallel_equals_single'] and pc['graph_equals_eager']):
        raise ValueError('lanes=3 requires a passing B=C check')
    if rc['parallel_check_sha256'] != sha(p):
        raise ValueError('run_config.json refers to another parallel_check.json')
    return rc


def stage_train(m):
    rc = require_parallel_check(m)
    if not m['smoke']:
        for n in ('run_config.json', 'parallel_check.json'):
            if not plan.is_committed_and_clean(str(ROOT), str((out_dir()/n).relative_to(ROOT))):
                raise ValueError('uncommitted: '+n)
    a = m['params']['attacker']
    runs = [r for r in range(a['chains']) if not (chain_path(r).exists() and chain_state(r)['opt']['gen'] == a['gens'])]
    stamp('train', lanes=rc['lanes'], pending_runs=runs)
    run_lanes(m, runs, rc['lanes'])
    chains_complete(m)
    mark_done('train')


# ---------------------------------------------------------------- parallel == sequential (design 3.4)
def trace_chain(m, run, be, N, validate=True):
    """N generations plus one validation (EA1_SEL, T_val) of a fresh chain: everything the design calls 'the result'."""
    cfg = m['params']['c2_cfg']
    opt, gens, walls = make_opt(m, run), [], []
    for _ in range(N):
        t = time.time()
        rec = chain_generation(m, cfg, run, opt, be)
        walls.append(time.time()-t)
        gens.append(dict(X=rec['X'], U=rec['U'], f=rec['f'], seeds=rec['seeds']))
    if not validate:
        return dict(gens=gens, wall_gen=walls)
    t = time.time()
    raw = be.simulate(cfg, [honest_pol(), mlp(opt.mean)], m['seeds']['EA1_SEL'], m['params']['T_val'], m['params']['exclude'])
    wall_val = time.time()-t
    return dict(gens=gens, opt=opt.state_dict(), val=digest(raw['snap']), G_val=float(v1.metrics(dict(raw, contract=None))['gains'][0].mean()),
                wall_gen=walls, wall_val=wall_val)


def trace_diff(x, y):
    """Names of the fields in which two traces are not bitwise equal."""
    bad = []
    for i, (p, q) in enumerate(zip(x['gens'], y['gens'])):
        for k in ('X', 'U', 'f'):
            if not bitwise_equal(p[k], q[k]):
                bad.append(f'gen{i+1}.{k}')
        if p['seeds'] != q['seeds']:
            bad.append(f'gen{i+1}.seeds')
    if len(x['gens']) != len(y['gens']):
        bad.append('n_gens')
    if digest(x['opt']) != digest(y['opt']):
        bad.append('optimizer_state')
    if x['val'] != y['val']:
        bad.append('validation_snapshot')
    if x['G_val'] != y['G_val']:
        bad.append('G_val')
    return bad


def run_parallel_traces(m, N, dev, factory=None):
    """A: each chain alone, eager, default stream.  B: each chain alone, graph, one lane.  C: all chains at once, graph, one lane each."""
    factory = factory or (lambda: make_backend(dev))
    runs = range(m['params']['attacker']['chains'])
    A = {r: trace_chain(m, r, Backend(dev, None, None), N) for r in runs}
    B = {}
    for r in runs:
        be = factory()
        with be:
            B[r] = trace_chain(m, r, be, N)
        be.release()
    C, errs, info = {}, {}, {}
    if dev.startswith('cuda'):
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def work(r):
        try:
            be = factory()
            with be:
                C[r] = trace_chain(m, r, be, N)
            be.release()
        except BaseException as exc:             # noqa: BLE001
            errs[r] = exc
    ths = [threading.Thread(target=work, args=(r,)) for r in runs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:      # registered rule: a failed 3-lane run means lanes=1; record it, never abort the measurement
        info['parallel_error'] = '; '.join(f'lane {r}: {type(x).__name__}: {x}'[:400] for r, x in sorted(errs.items()))
        C = None
    if dev.startswith('cuda'):
        info['peak_C_gb'] = None if C is None else torch.cuda.max_memory_reserved()/2**30
    return A, B, C, info


def acceptance_report(m, N, A, B, C, info=None):
    ab = {r: trace_diff(A[r], B[r]) for r in A}
    bc = None if C is None else {r: trace_diff(B[r], C[r]) for r in B}
    return dict(n_gens=N, A_vs_B=ab, B_vs_C=bc, graph_equals_eager=not any(ab.values()), parallel_equals_single=bc is not None and not any(bc.values()),
                parallel_ok=bc is not None and not any(bc.values()), parallel_error=(info or {}).get('parallel_error'),
                digests={name: {r: digest(dict(opt=t[r]['opt'], val=t[r]['val'], G_val=t[r]['G_val'])) for r in t} for name, t in (('A', A), ('B', B), ('C', C)) if t is not None})


# ---- measurement only: 3 independent processes, each on the single-lane path (does not change how `train` runs)
def mp_worker(run, N, outdir):
    """One process of the multiprocess measurement: chain `run`, N generations, single lane, no validation; saves X / fitness / timings."""
    import torch
    m = load_manifest(clean=False)
    be = make_backend(MODE['device'])
    with be:
        tr = trace_chain(m, run, be, N, validate=False)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(outdir)/f'run{run}.npz', X=np.stack([g['X'] for g in tr['gens']]), f=np.stack([g['f'] for g in tr['gens']]), wall_gen=np.array(tr['wall_gen']),
             peak_gb=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0.)


def run_multiprocess(m, N, B, smoke):
    """Start the 3 chains as 3 processes at once; compare their X / fitness with B (one process, one chain at a time) bit for bit; throughput ratio."""
    outdir = out_dir()/'mp'
    shutil.rmtree(outdir, ignore_errors=True)
    t0 = time.time()
    cmd = [sys.executable, str(ROOT/'scripts/ea1.py'), 'mpworker', '--gens', str(N), '--mp-out', str(outdir), '--device', MODE['device']] + (['--smoke'] if smoke else [])
    procs = [subprocess.Popen(cmd + ['--run', str(r)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(m['params']['attacker']['chains'])]
    outs = [p.communicate()[0] for p in procs]
    wall = time.time()-t0
    res = dict(n_gens=N, wall_s_all_three=wall, returncodes=[p.returncode for p in procs])
    if any(p.returncode for p in procs):
        res.update(error=' | '.join(o[-300:] for o in outs), bitwise_equal=None, throughput_ratio=None)
        return res
    z = [np.load(outdir/f'run{r}.npz') for r in range(len(procs))]
    t1 = mean_after_first(B[0]['wall_gen'])
    tp = float(np.mean([mean_after_first(list(x['wall_gen'])) for x in z]))
    res.update(bitwise_equal=all(bitwise_equal(z[r]['X'][g], B[r]['gens'][g]['X']) and bitwise_equal(z[r]['f'][g], B[r]['gens'][g]['f'])
                                 for r in range(len(z)) for g in range(N)),
               t_gen_per_proc_s=tp, t_gen1_s=t1, throughput_ratio=3*t1/tp, peak_vram_per_proc_gb=[float(x['peak_gb']) for x in z])
    return res


def stage_postcheck(m):
    """Design 3.4 'after training' (lanes=3): chain 0's generation `post_check_gen`, recomputed on a single lane from the snapshot saved before it,
    must equal the candidate matrix and fitness recorded during the parallel run, bit for bit."""
    rc = read_run_config(m)
    a, cfg = m['params']['attacker'], m['params']['c2_cfg']
    post = a['post_check_gen']
    rec = dict(schema='ea1-post-train-check/1', manifest_sha256=m['sha256'], lanes=rc['lanes'], required=rc['lanes'] == 3, generation=post, run=0)
    if rc['lanes'] != 3:
        rec.update(passed=True, note='lanes=1: the chains were trained sequentially on one lane; nothing to compare')
        write(out_dir()/'post_train_check.json', rec)
        return rec
    snap, saved = read(chain_path(0, f'snap{post-1}')), read(chain_path(0, f'gen{post}'))
    if snap['manifest_sha256'] != m['sha256'] or saved['manifest_sha256'] != m['sha256']:
        raise ValueError('post-train check files belong to another manifest')
    opt = make_opt(m, 0)
    opt.load_state_dict(snap['opt'])
    if opt.gen != post-1:
        raise ValueError('snapshot is not the state after the preceding generation')
    be = make_backend(MODE['device'])
    with be:
        got = chain_generation(m, cfg, 0, opt, be)
    ok_x, ok_f = bitwise_equal(got['X'], saved['X']), bitwise_equal(got['f'], saved['f'])
    rec.update(passed=bool(ok_x and ok_f), X_identical=bool(ok_x), fitness_identical=bool(ok_f), ea1_py_sha256=sha(ROOT/'scripts/ea1.py'), unix=time.time())
    write(out_dir()/'post_train_check.json', rec)
    if not rec['passed']:
        raise PostCheckFailed('single-lane recomputation of the saved generation differs from the parallel result: STOP and ask the user')
    return rec


# ---------------------------------------------------------------- E1 v2 inputs (read only)
def e1v2_entries():
    """The 12 E1 v2 attackers (thetas verified against the hashes in E1 v2's selection_frozen.json) and a*."""
    sel = read(ROOT/E1V2_DIR/'selection_frozen.json')
    _, cks, _ = context()
    out = [(f'frozen_run{i}', list(map(float, c['theta']))) for i, c in enumerate(cks)]
    for n in sel['attackers'][len(cks):]:
        label, run = n.rsplit('_adapt', 1)
        out.append((n, list(map(float, e1.chain_best(read(ROOT/E1V2_DIR/'attacker'/f'{label}_run{run}.json'))['theta']))))
    if [theta_id(t) for _, t in out] != list(sel['attacker_theta_sha256']) or [n for n, _ in out] != list(sel['attackers']):
        raise ValueError('E1 v2 attackers differ from its frozen selection')
    star = sel['astar']['orig']
    if star['theta_sha256'] != ASTAR_SHA256 or theta_id(star['theta']) != ASTAR_SHA256 or not ASTAR_SHA256.startswith('88ed926a'):
        raise ValueError('a* theta hash differs from the registered 88ed926a...')
    return dict(astar=dict(name=star['name'], theta=list(map(float, star['theta'])), theta_sha256=ASTAR_SHA256), entries=out,
                c2_cfg=sel['choice']['cfg'], G_AS_astar=sel['G_AS_astar'], attackers=list(sel['attackers']),
                transfer_raw={k: v['raw'] for k, v in sel['transfer_matrix_AS'].items()})


def drift_reference(inp):
    """E1 v2's own AS evaluations of a* (per-seed gains) for the drift check."""
    i = inp['attackers'].index(inp['astar']['name'])+1
    out = {}
    for k, rel in inp['transfer_raw'].items():
        raw = read(ROOT/E1V2_DIR/rel)
        u = np.asarray(raw['snap']['util_b'], float)[:, :, 1]
        out[k] = ((u[i]-u[0])/raw['T_score'])
    return out


# ---------------------------------------------------------------- select (freeze)
def gains_of(raw):
    return v1.metrics(raw)['gains']


def stage_select(m):
    if 'train' not in stages_done():
        raise RuntimeError('run train first')
    d, a = out_dir(), m['params']['attacker']
    sel_path = d/'selection_frozen.json'
    if sel_path.exists():     # crash between writing the selection and registering it: register what was written, never recompute
        if read(sel_path)['manifest_sha256'] != m['sha256']:
            raise ValueError('selection from another manifest')
        if not (d/'freeze_record.json').exists():
            write(d/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
        mark_done('select')
        return
    chains_complete(m)
    rc = read_run_config(m)
    if rc['lanes'] == 3:
        pt = d/'post_train_check.json'
        if not pt.exists() or not read(pt)['passed'] or read(pt)['manifest_sha256'] != m['sha256']:
            raise ValueError('lanes=3 needs a passing post_train_check.json before select')
    inp = e1v2_entries()
    if inp['c2_cfg'] != m['params']['c2_cfg']:
        raise ValueError('C2 configuration differs from E1 v2 selection')
    cfg, base, star = m['params']['c2_cfg'], m['params']['base_cfg'], inp['astar']
    SEL = m['seeds']['EA1_SEL']
    # 1. each chain's own checkpoint (chain-internal G_SEL, used only for this choice)
    chains = []
    for run in range(a['chains']):
        r = best_record(chain_state(run)['records'])
        chains.append(dict(run=run, gen=r['gen'], G_SEL_chain=r['G_SEL'], theta=r['theta'], theta_sha256=theta_id(r['theta']), name=f'fresh_run{run}'))
    # 2. one re-evaluation call on C2: [honest, a*, chain0, chain1, chain2]  ->  A_fresh, A_C2'
    pols = [honest_pol(), mlp(star['theta'])]+[mlp(c['theta']) for c in chains]
    ids = ['honest', 'a_star']+[c['name'] for c in chains]
    raw, rel = evaluate(m, cfg, SEL, 'sel', pols, ids)
    G = gains_of(raw).mean(1).tolist()
    g_star, g_ch = G[0], G[1:]
    k = first_max(g_ch)
    fresh = dict(chains[k], G_SEL_reeval=g_ch[k], index=k)
    c2p = pick_c2_prime(g_star, g_ch[k])
    reeval = dict(raw=rel, names=ids[1:], G=G)
    # 3. original setting: [honest, a*, frozen_run0..2] -> F_orig
    frozen = [(n, t) for n, t in inp['entries'] if n.startswith('frozen_run')]
    raw, rel = evaluate(m, base, SEL, 'sel', [honest_pol(), mlp(star['theta'])]+[mlp(t) for _, t in frozen], ['honest', 'a_star']+[n for n, _ in frozen])
    Go = gains_of(raw).mean(1).tolist()
    f_idx = first_max(Go[1:])
    f_orig = dict(name=frozen[f_idx][0], index=f_idx, G_SEL_orig=Go[1:][f_idx])
    reeval_orig = dict(raw=rel, names=['a_star']+[n for n, _ in frozen], G=Go)
    # 4. descriptive: 12 E1 v2 entries + 3 fresh checkpoints, both settings
    entries = [dict(name=n, theta=t, theta_sha256=theta_id(t)) for n, t in inp['entries']]+[dict(name=c['name'], theta=c['theta'], theta_sha256=c['theta_sha256']) for c in chains]
    dpols = [honest_pol()]+[mlp(e['theta']) for e in entries]
    dids = ['honest']+[e['name'] for e in entries]
    desc = {}
    for label, c in (('C2', cfg), ('orig', base)):
        raw, rel = evaluate(m, c, SEL, 'sel', dpols, dids)
        desc[label] = dict(raw=rel, names=dids[1:], G=gains_of(raw).mean(1).tolist())
    # 5. machine drift (descriptive): a* on the E1 v2 AS seeds, original and C2, against E1 v2's own numbers
    drift = dict(schema='ea1-drift/1', manifest_sha256=m['sha256'], note='descriptive; no pass/fail threshold; cross-machine comparisons are reference only if not bitwise identical')
    ref = drift_reference(inp)
    for label, c in (('orig', base), ('C2', cfg)):
        raw, rel = evaluate(m, c, m['drift']['seeds'], 'drift', [honest_pol(), mlp(star['theta'])], ['honest', 'a_star'])
        g_new, g_old = gains_of(raw)[0], ref[label]
        drift[label] = dict(raw=rel, G_new=float(g_new.mean()), G_e1v2_AS=inp['G_AS_astar'][label], mean_diff=float(g_new.mean()-g_old.mean()),
                            max_abs_per_seed=float(np.abs(g_new-g_old).max()), n_bitwise_identical=int(sum(bitwise_equal(x, y) for x, y in zip(g_new, g_old))), n=len(g_new))
    write(d/'drift.json', drift)
    rec = dict(schema='ea1-selection/1', manifest_sha256=m['sha256'], astar=star, chains=chains, A_fresh=fresh, A_C2_prime=c2p, F_orig=f_orig,
               reeval_C2=reeval, reeval_orig=reeval_orig, descriptive_SEL=desc, entries=entries, c2_cfg=cfg, drift_sha256=sha(d/'drift.json'),
               run_config_sha256=sha(d/'run_config.json'))
    write(sel_path, rec)
    write(d/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
    mark_done('select')
    print('selection registered: commit selection_frozen.json, freeze_record.json, drift.json before testcheck', flush=True)


# ---------------------------------------------------------------- testcheck / test (once)
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
    for e in sel['entries']+[sel['astar']]:
        if theta_id(e['theta']) != e['theta_sha256']:
            raise ValueError('a frozen theta does not match its hash')
    return sel


def test_pols(sel):
    """(name, which, thetas, policies) of every test evaluation: main C2 / main orig = [a*, A_fresh]; sec = [12 entries, 3 fresh checkpoints]."""
    star, fresh = sel['astar']['theta'], next(e for e in sel['entries'] if e['name'] == sel['A_fresh']['name'])['theta']
    main, sec = [star, fresh], [e['theta'] for e in sel['entries']]
    spec = [('main_C2', 'C2', main), ('main_orig', 'orig', main), ('sec_C2', 'C2', sec), ('sec_orig', 'orig', sec)]
    return [(n, w, ts, [honest_pol()]+[mlp(t) for t in ts]) for n, w, ts in spec]


def test_cfgs(m):
    return dict(C2=m['params']['c2_cfg'], orig=m['params']['base_cfg'])


def test_path_id():
    return dict(graph=bool(MODE['graph']), device=MODE['device'], chunk=None, fallback=False)


def run_testcheck():
    """GPU pre-check on the first 64 EA1_SEL seeds (never EA1_TEST): the test path (simulate_chunked, graph, test chunk) must reproduce eager exactly."""
    m = load_manifest()
    sel = verify_freeze(m)
    d = out_dir()
    if not MODE['strict']:
        raise RuntimeError('testcheck must run in strict mode')
    if MODE['device'].startswith('cuda') and not MODE['graph']:
        raise RuntimeError('testcheck/test on cuda must use the graph path (the verified one)')
    sd = m['seeds']['EA1_SEL'][:m['params']['testcheck_n']]
    require_seeds(sd, 'testcheck', m)
    chunk, cfgs, checks = m['params']['test_chunk'], test_cfgs(m), []
    for name, which, _, pols in test_pols(sel):
        got = simulate_chunked(m, cfgs[which], pols, sd, chunk)
        ref = sim_eager(m, cfgs[which], pols, sd, m['params']['T_val'])
        diff = {}
        for k in main_eval.SNAP_FIELDS:
            x, y = np.asarray(got['snap'][k], float), np.asarray(ref['snap'][k], float)
            diff[k] = dict(shape_equal=x.shape == y.shape, identical=bool(x.shape == y.shape and bitwise_equal(x, y)),
                           max_abs=float(np.max(np.abs(x-y))) if x.shape == y.shape and x.size else None)
        ok = int(got['T_score']) == int(ref['T_score']) and all(v['identical'] for v in diff.values())
        checks.append(dict(name=name, n_policies=len(pols), n_seeds=len(sd), identical=ok, fields=diff))
        print('testcheck', name, 'identical' if ok else 'MISMATCH', flush=True)
    rec = dict(schema='ea1-testcheck/1', manifest_sha256=m['sha256'], freeze_record_sha256=sha(d/'freeze_record.json'), path=dict(test_path_id(), chunk=chunk),
               seeds=[sd[0], sd[-1], len(sd)], tolerance='exact equality, no tolerance', environment=environment(),
               passed=all(c['identical'] for c in checks), checks=checks, unix=time.time())
    write(d/'testcheck.json', rec)
    if not rec['passed']:
        raise RuntimeError('testcheck failed: the test path differs from eager; test refused')
    return rec


def require_testcheck(m, sel):
    """Called before anything is written to the ledger."""
    d = out_dir()
    p = d/'testcheck.json'
    if not p.exists():
        raise ValueError('testcheck.json is missing: run the testcheck stage first')
    t = read(p)
    if not t.get('passed') or t['manifest_sha256'] != m['sha256'] or t['freeze_record_sha256'] != sha(d/'freeze_record.json'):
        raise ValueError('testcheck.json failed or does not match the current manifest/freeze_record')
    if t['path'] != dict(test_path_id(), chunk=m['params']['test_chunk']):
        raise ValueError('test execution path differs from the one testcheck verified')
    if MODE['device'].startswith('cuda') and not MODE['graph']:
        raise ValueError('test on cuda must use the graph path (the verified one)')
    if MODE['graph'] and MODE['device'].startswith('cuda'):      # every test shape must capture before the ledger is touched (EA1_SEL seeds only)
        sd = m['seeds']['EA1_SEL'][:m['params']['test_chunk']]
        require_seeds(sd, 'testcheck', m)
        for name, which, _, pols in test_pols(sel):
            sim(m, test_cfgs(m)[which], pols, sd, m['params']['T_val'])


def compute_result(m, sel, raw_c2):
    g = gains_of(raw_c2)
    if g.shape[0] != 2:
        raise ValueError('main_C2 must hold [a*, A_fresh]')
    out = decide(g[1]-g[0], m['params']['eps'])
    out.update(schema='ea1-result/1', manifest_sha256=m['sha256'], selection_frozen_sha256=sha(out_dir()/'selection_frozen.json'),
               G_C2_astar=float(g[0].mean()), G_C2_A_fresh=float(g[1].mean()), A_fresh=sel['A_fresh']['name'], A_fresh_gen=sel['A_fresh']['gen'])
    return out


def compute_secondary(m, sel, raws):
    """Design 2.6, descriptive only (uncorrected for multiplicity).  raws: main_C2, main_orig, sec_C2, sec_orig."""
    names = [e['name'] for e in sel['entries']]
    mc, mo, sc, so = (gains_of(raws[k]) for k in ('main_C2', 'main_orig', 'sec_C2', 'sec_orig'))
    i = {n: j for j, n in enumerate(names)}
    out = dict(schema='ea1-secondary/1', manifest_sha256=m['sha256'], note='descriptive only; uncorrected for multiplicity; not part of the decision')
    out['orig_type_delta'] = {k: interval(so[i[k]]-so[i[sel['astar']['name']]]) for k in ('frozen_run0', 'frozen_run1', 'frozen_run2')}
    out['orig_type_delta_F_orig'] = dict(name=sel['F_orig']['name'], **interval(so[i[sel['F_orig']['name']]]-so[i[sel['astar']['name']]]),
                                         prior='registered expectation: Delta_orig < 0 (design 2.6 item 1)')
    out['fresh_vs_astar_C2'] = {c['name']: interval(sc[i[c['name']]]-sc[i[sel['astar']['name']]]) for c in sel['chains']}
    g_prime = mc[1] if sel['A_C2_prime'] == 'A_fresh' else mc[0]
    out['D_prime'] = dict(A_C2_prime=sel['A_C2_prime'], **interval(mo[0]-g_prime), delta=.005, note='descriptive; the primary table keeps D')
    out['D_new'] = dict(**interval(mo[0]-mc[0]), cross_machine_reference_only=True)
    try:
        r1 = read(ROOT/E1V2_DIR/'result.json')
        out['D_new']['e1v2_D'] = dict(mean=r1['mean_D'], LB=r1['LB'], UB=r1['UB_D'])
    except OSError:
        pass
    out['G_C2_by_entry'] = {n: float(sc[j].mean()) for n, j in i.items()}
    out['G_orig_by_entry'] = {n: float(so[j].mean()) for n, j in i.items()}
    ref = {}
    for tag, key in (('C2', 'cand_secondary_raw.json'), ('orig', 'orig_secondary_raw.json')):
        p = ROOT/E1V2_DIR/'test'/key
        if p.exists():
            ref[tag] = dict(zip(names[:12], map(float, v1.metrics(read(p))['gains'].mean(1))))   # E1 v2 order = our first 12 entries
    out['cross_machine_vs_e1v2_test'] = {tag: {n: out['G_%s_by_entry' % tag][n]-g for n, g in r.items()} for tag, r in ref.items()}
    ra, rb = (v1.metrics(raws[k]) for k in ('sec_C2', 'sec_orig'))
    out['honest_service'] = dict(delta_regret=interval(ra['regret']-rb['regret']), delta_alpha=interval(ra['alpha']-rb['alpha']))
    return out


def chain_summaries(m, sel):
    """Per chain: SEL trajectory, best generation, saturation flag (registered rule: gmax 400, last two points vs the rest, threshold 0.001; descriptive)."""
    a, out = m['params']['attacker'], []
    for run in range(a['chains']):
        s = chain_state(run)
        traj = [[r['gen'], r['G_SEL']] for r in s['records']]
        try:
            sat = e1.saturation(traj, a['val_every'], traj[-1][0], 0.001)
        except ValueError as exc:             # only with a budget shorter than the registered one (tests)
            sat = dict(error=str(exc))
        out.append(dict(run=run, trajectory=traj, best_gen=best_record(s['records'])['gen'], saturation=sat, curve_gens=len(s['curve'])))
    return out


def run_test():
    m = load_manifest()
    sel = verify_freeze(m)
    d = out_dir()
    require_testcheck(m, sel)            # before the ledger and before test seeds are touched
    sd = m['seeds']['EA1_TEST']
    require_seeds(sd, 'test', m, unlocked=True)
    if len(sd) != m['params']['n_test']:
        raise ValueError('test seed count changed')
    cfgs, freeze_sha, raws = test_cfgs(m), sha(d/'freeze_record.json'), {}

    def go(name, which, thetas, pols):
        con = dict(manifest_sha256=m['sha256'], freeze_record_sha256=freeze_sha, stage=name, cfg=cfgs[which], seeds=list(sd), n_policies=len(pols),
                   T_score=m['params']['T_val']-m['params']['exclude'], T=m['params']['T_val'], exclude=m['params']['exclude'],
                   theta_sha256=[theta_id(t) for t in thetas])
        raw = once(d/'test', name, con, lambda: simulate_chunked(m, cfgs[which], pols, sd, m['params']['test_chunk']))
        stamp('test', evaluation=name)
        return raw
    plan_ = {n: (w, ts, pols) for n, w, ts, pols in test_pols(sel)}
    for name in ('main_C2', 'main_orig'):
        raws[name] = go(name, *plan_[name][:2], plan_[name][2])
    res = compute_result(m, sel, raws['main_C2'])
    load_manifest(clean=False)
    write(d/'result.json', res)
    print(json.dumps({k: res[k] for k in ('category', 'sub', 'mean_delta', 'LB', 'UB')}), flush=True)
    for name in ('sec_C2', 'sec_orig'):
        raws[name] = go(name, *plan_[name][:2], plan_[name][2])
    sec = compute_secondary(m, sel, raws)
    sec['chains'] = chain_summaries(m, sel)
    write(d/'secondary.json', sec)
    return res


def stage_report():
    """CPU only: recompute result and secondary from the raw files; compare with the saved ones.  Prints library versions (design 7 step 11)."""
    m = load_manifest(clean=False)
    d = out_dir()
    sel = read(d/'selection_frozen.json')
    raws = {n: read(d/'test'/f'{n}_raw.json') for n in ('main_C2', 'main_orig', 'sec_C2', 'sec_orig')}
    res = compute_result(m, sel, raws['main_C2'])
    sec = compute_secondary(m, sel, raws)
    sec['chains'] = chain_summaries(m, sel)
    same = read(d/'result.json') == res and read(d/'secondary.json') == sec
    print(json.dumps({k: res[k] for k in ('category', 'sub', 'mean_delta', 'LB', 'UB')}))
    print('matches saved result/secondary:', same, '| numpy', np.__version__, 'scipy', scipy.__version__)
    write(d/f'report_check_{socket.gethostname()}.json', dict(matches=same, numpy=np.__version__, scipy=scipy.__version__, result=res))
    return 0 if same else 1


# ---------------------------------------------------------------- smoke / paracheck (GPU)
class Meter:
    """Wall time, process CPU cores and GPU utilisation / memory (nvidia-smi sampled every 2 s) of one phase."""

    def __init__(self, name, sink):
        self.name, self.sink, self.util, self.mem, self._stop = name, sink, [], [], threading.Event()

    def _sample(self):
        while not self._stop.wait(2.0):
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                u, mem = out.stdout.strip().splitlines()[0].split(',')
                self.util.append(float(u))
                self.mem.append(float(mem))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass

    def __enter__(self):
        self.t0, self.c0 = time.time(), sum(os.times()[:2])
        self.th = threading.Thread(target=self._sample, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join()
        wall = time.time()-self.t0
        self.sink[self.name] = dict(wall_s=wall, cpu_cores=(sum(os.times()[:2])-self.c0)/wall, gpu_util_mean=float(np.mean(self.util)) if self.util else None,
                                    gpu_util_max=max(self.util) if self.util else None, gpu_mem_used_mb_max=max(self.mem) if self.mem else None)
        return False


def mean_after_first(xs):
    return float(np.mean(xs[1:])) if len(xs) > 1 else None


def run_smoke(N):
    """Before the official freeze: formal-size timing (single lane / 3 lanes), A=B=C, one validation, test-type evaluations; EA1_TRAIN / EA1_SEL only."""
    import torch
    if MODE['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('smoke needs CUDA')
    freeze(smoke=True)
    m = load_manifest(clean=False)
    check_environment(m)
    meters = {}
    with Meter('abc', meters):
        A, B, C, info = run_parallel_traces(m, N, MODE['device'])
    rep = acceptance_report(m, N, A, B, C, info)
    t1 = mean_after_first(B[0]['wall_gen'])
    t3 = None if C is None else float(np.mean([mean_after_first(C[r]['wall_gen']) for r in C]))
    mem_gb = info['peak_C_gb']          # peak reserved memory while the 3 lanes ran together (None: the 3-lane run failed)
    try:
        with Meter('multiprocess', meters):
            mp = run_multiprocess(m, N, B, True)
    except Exception as exc:             # noqa: BLE001 - measurement only; never abort the smoke
        mp = dict(error=f'{type(exc).__name__}: {exc}'[:400])
    t_val3 = None if C is None else float(np.mean([C[r]['wall_val'] for r in C]))
    inp = e1v2_entries()
    sel_seeds = m['seeds']['EA1_SEL']
    t_test = {}
    for label, ents in (('1_attacker', [inp['astar']['theta']]), ('12_attackers', [t for _, t in inp['entries']])):
        pols = [honest_pol()]+[mlp(t) for t in ents]
        t0 = time.time()
        simulate_chunked(m, m['params']['c2_cfg'], pols, sel_seeds, m['params']['test_chunk'])
        w = time.time()-t0
        t_test[label] = dict(wall_s=w, extrapolated_1024_seeds_s=w*m['params']['n_test']/len(sel_seeds))
    summary = dict(schema='ea1-smoke-summary/1', n_gens=N, acceptance=rep, t_gen1_s=t1, t_gen3_s=t3, throughput_ratio=None if t3 is None else 3*t1/t3, t_val1_s=B[0]['wall_val'],
                   t_val3_s=t_val3, parallel_error=info.get('parallel_error'), multiprocess=mp, gpu_memory_peak_gb=mem_gb, meters=meters, t_test=t_test, environment=environment(),
                   lanes_if_adopted=decide_lanes(t1, t3, rep['parallel_equals_single'], rep['graph_equals_eager'], mem_gb),
                   hours_lanes1=3*(m['params']['attacker']['gens']*t1+8*B[0]['wall_val'])/3600,
                   hours_lanes3=None if C is None else (m['params']['attacker']['gens']*t3+8*t_val3)/3600)
    write(out_dir()/'smoke_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('t_gen1_s', 't_gen3_s', 'throughput_ratio', 'gpu_memory_peak_gb', 'lanes_if_adopted', 'hours_lanes1', 'hours_lanes3')}, indent=1))
    return summary


def smoke_clean(dest=None):
    """Keep the smoke summary (the numbers go into the implementation notes), remove the smoke directory."""
    src, dst = ROOT/SMOKE_DIR/'smoke_summary.json', Path(dest) if dest else ROOT/OUT_DIR/'smoke_summary.json'
    if src.exists():
        write(dst, read(src))
    shutil.rmtree(ROOT/SMOKE_DIR, ignore_errors=True)
    print('smoke directory removed; summary kept at', dst)


def run_paracheck(N, summary_path):
    """After the official freeze: A=B=C on the registered manifest (EA1_TRAIN first generations, EA1_SEL validation), then run_config.json."""
    m = load_manifest()
    check_environment(m)
    if N < 3:
        raise ValueError('N must be at least 3')
    summ = read(summary_path)
    A, B, C, info = run_parallel_traces(m, N, MODE['device'])
    rep = acceptance_report(m, N, A, B, C, info)
    d = out_dir()
    pc = dict(schema='ea1-parallel-check/1', manifest_sha256=m['sha256'], ea1_py_sha256=sha(ROOT/'scripts/ea1.py'), environment=environment(),
              gpu_memory_peak_gb=info.get('peak_C_gb'), **rep)
    write(d/'parallel_check.json', pc)
    choice = decide_lanes(summ['t_gen1_s'], summ.get('t_gen3_s'), rep['parallel_equals_single'], rep['graph_equals_eager'], summ.get('gpu_memory_peak_gb'))
    if choice['lanes'] is None:
        raise RuntimeError(choice['reason'])
    write(d/'run_config.json', dict(schema='ea1-run-config/1', manifest_sha256=m['sha256'], lanes=choice['lanes'], decision=choice,
                                    smoke_summary_sha256=sha(summary_path), parallel_check_sha256=sha(d/'parallel_check.json'),
                                    rule='lanes=3 iff 3*t1/t3 >= 1.10 and A=B=C and GPU memory peak <= 6 GB (design 3.3)'))
    print('lanes =', choice['lanes'], '|', choice['reason'], flush=True)
    return choice


# ---------------------------------------------------------------- CLI
def status():
    d = out_dir()
    m = load_manifest(clean=False)
    a = m['params']['attacker']
    chains = {}
    for r in range(a['chains']):
        chains[r] = chain_state(r)['opt']['gen'] if chain_path(r).exists() else 0
    print(json.dumps(dict(smoke=m['smoke'], manifest_sha256=m['sha256'], chain_generations=chains, of=a['gens'], stages_done=sorted(stages_done()),
                          run_config=read(d/'run_config.json') if (d/'run_config.json').exists() else None,
                          status=read(d/'status.json') if (d/'status.json').exists() else None), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('cmd', choices=['audit', 'freeze', 'smoke', 'smoke-clean', 'paracheck', 'train', 'postcheck', 'select', 'testcheck', 'test', 'report', 'status', 'mpworker'])
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--mp-out', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gens', type=int, default=5, help='smoke / paracheck: generations per chain (>= 3)')
    ap.add_argument('--smoke-summary', default=str(ROOT/OUT_DIR/'smoke_summary.json'))
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    strict = a.cmd in ('test', 'testcheck')
    MODE.update(smoke=a.smoke or a.cmd in ('smoke',), device=a.device, graph=a.device.startswith('cuda'), strict=strict, yield_check=a.cmd == 'train' and a.device.startswith('cuda'))
    if strict and a.smoke:
        sys.exit('test/testcheck never run in smoke mode')
    if not MODE['smoke'] and a.cmd not in ('report', 'status', 'audit', 'freeze') and a.device != 'cuda':
        sys.exit(f'{a.cmd}: the official manifest runs only with --device cuda (got {a.device!r}); refusing')
    (ROOT/'.scratch').mkdir(exist_ok=True)
    if a.cmd == 'audit':
        blocks = block_seeds(a.smoke)
        hist, _ = scan_saved_seeds()
        print(json.dumps(seed_report(blocks, existing_all(), hist), indent=1))
        return 0
    if a.cmd == 'mpworker':             # child of the smoke multiprocess measurement: no lock (the parent holds it), smoke manifest only
        if not a.smoke or not a.mp_out:
            sys.exit('mpworker is internal to smoke (needs --smoke and --mp-out)')
        import torch
        torch.set_num_threads(1)
        mp_worker(a.run, max(a.gens, 3), a.mp_out)
        return 0
    if a.cmd == 'smoke-clean':
        smoke_clean()
        return 0
    if a.cmd in ('status', 'report'):
        if a.cmd == 'report':
            return stage_report()
        status()
        return 0
    with (ROOT/'.scratch/ea1.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.cmd == 'freeze':
            freeze(a.smoke)
            return 0
        import torch
        torch.set_num_threads(1)
        MODE['start'] = time.time()
        try:
            if a.cmd == 'smoke':
                run_smoke(max(a.gens, 3))
                return 0
            m = load_manifest(clean=not a.smoke)
            if MODE['device'].startswith('cuda'):
                check_environment(m)
            if a.cmd == 'paracheck':
                run_paracheck(max(a.gens, 3), a.smoke_summary)
            elif a.cmd == 'train':
                stamp('running', started_unix=time.time(), manifest_sha256=m['sha256'])
                if 'train' not in stages_done():
                    stage_train(m)
                stamp('complete', finished_unix=time.time())
            elif a.cmd == 'postcheck':
                stage_postcheck(m)
            elif a.cmd == 'select':
                stage_select(m)
            elif a.cmd == 'testcheck':
                run_testcheck()
            elif a.cmd == 'test':
                run_test()
        except (BudgetStop, YieldStop) as exc:
            stamp('budget_stop' if isinstance(exc, BudgetStop) else 'yield_stop', message='saved at a generation boundary; relaunch to resume')
            return 75
        except Exception as exc:
            stamp('failed', error_type=type(exc).__name__, message=str(exc)[:300])
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
