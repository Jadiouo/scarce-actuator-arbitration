#!/usr/bin/env python3
"""E-A2: attacker-strength control on the original setting (design: docs/direction2/design/e-a2.md; pre-registered).

Stages: audit / smoke / smoke-clean / freeze (S0) / train / select / adapt / testcheck / test (once, in three parts) / report / status.
Three attack chains are trained from theta=0 on the ORIGINAL setting (lam=256, n_S=256, sigma0=0.3, 416 generations, one lane).  `select`
freezes A_fresh_orig; `adapt` (after the committed select) runs the registered re-adaptation arm and freezes A+ / A++ together; only then may
`testcheck` and `test` touch anything.  The test block 92400000-92402047 is read by `test` only.

E-A1 is referenced by FILE HASH only (design 1.2): its history was rewritten, so ea1.load_manifest is never called here.  scripts/ea1.py is
imported for pure helpers; nothing in it is changed.  Nothing here writes into the E-A1 or E1 v2 result directories.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import cmaes, main_eval, plan
from scripts import e1_pilot as v1
from scripts import e1_v2 as e1
from scripts import ea1
from scripts.d2_holdout_eval import read, write, sha, digest, context, seed_values, once as _once

DESIGN = "docs/direction2/design/e-a2.md"
OUT_DIR = "results/direction2/ea2"
SMOKE_DIR = "results/direction2/ea2_smoke"
E1V2_DIR = ea1.E1V2_DIR
EA1_DIR = "results/direction2/ea1"
BASE_CFG = v1.BASE_CFG
C2_CFG = ea1.C2_CFG
OFFS4 = e1.OFFS4
# EA2_TRAIN: chain training; EA2_SEL: validation / selection / re-evaluation / testcheck (first 64) / re-adaptation validation;
# EA2_AT: re-adaptation TRAINING only; EA2_TEST: the one-shot test (design 1.3).
BLOCKS = dict(EA2_TRAIN=(92_000_000, 1000), EA2_SEL=(92_100_000, 128), EA2_AT=(92_200_000, 256), EA2_TEST=(92_400_000, 2048))
STAGE_BLOCK = dict(train='EA2_TRAIN', sel='EA2_SEL', at='EA2_AT', testcheck='EA2_SEL', test='EA2_TEST')
DRIFT_BLOCK = ea1.DRIFT_BLOCK
EXPECT_NEW, EXPECT_OLD, EXPECT_POOL = 13728, 20672, 15
NEW_SHA = "1940a8a5e9cd75cefe771b8d29fe2e007aded1e3b12eab5d52a375126d29ec75"      # registered in design 1.3 (pre-run)
OLD_SHA = "5de9035d2137a8ddb5f7712fb47ac2ca57635ebdb84d74568306b52f8551ffe0"
INPUTS = ["scripts/ea2.py", "scripts/ea2_launch.sh", "tests/direction2/test_ea2.py", DESIGN]
ASTAR_SHA256 = ea1.ASTAR_SHA256
# E-A1 files that E-A2 depends on, by content hash (design 1.2).  The E-A1 manifest is an anchor only: its content is never parsed.
EA1_REFS = {
    "scripts/ea1.py": "55aba50e9989087569ed2a3815311a02e83cf85441246e7a7adcf8067b0266ca",
    "scripts/ea1_launch.sh": "b4b69f9920e883dd07689f0543889066f7f2878316b81c179d4c152878491f18",
    "tests/direction2/test_ea1.py": "fb1d11c2e7d02ec4bc424d22b63442ce69c246297aadf96cdc867d00c2350e17",
    EA1_DIR+"/selection_frozen.json": "ff6b5a480249499014f319e15cceb64281fe54f4f3e48e240df90c273812c51a",
    EA1_DIR+"/result.json": "525863f9de7bcb613700d1dc63cfcf7edba2e952f9bb442fd38319ddec1b1c3b",
    EA1_DIR+"/secondary.json": "fdf89ccec103a7efd92e5a2aee0afdb80a2b1eef5ddd935e4875f20ca3265930",
    EA1_DIR+"/manifest.json": "9da1b2af819a8ce86e1c212cd5cf9e7f96a4f5b2649b2606b5eb11c2686b5cac"}
EA1_DESIGN = ("docs/direction2/design/e-a1.md", "9ff931cfcd29061c9a640f4e31303b209aa94b445185984cb8b1a3d2fc2d12c6")     # recorded, not a load condition
ENV_PIN = ea1.ENV_PIN
THREAD_VARS = ea1.THREAD_VARS
EPS, DELTA = .001, .005

PARAMS = dict(
    schema="ea2/1", r=.5, base_cfg=BASE_CFG, c2_cfg=C2_CFG, astar_theta_sha256=ASTAR_SHA256, T_train=20000, T_val=100000, exclude=5000,
    eps=EPS, delta=DELTA, ci_level=.95, n_test=2048, test_chunk=64, testcheck_n=64,
    attacker=dict(lam=256, sigma0=.3, n_S=256, gens=416, val_every=50, chains=3, opt_seed=95000, perm_seed=95100),
    plus=dict(lam=64, sigma0=.15, n_S=32, gens=240, val_every=20, opt_seed=96000, key=0, pp_min_gen=20),
    lanes=1, kernel_K=32)
SMOKE_PARAMS = dict(PARAMS, schema="ea2-smoke/1", n_test=0)

MODE = ea1.MODE                    # one shared dict: the borrowed ea1 helpers read it (device, graph, strict, K, start, yield_check)
GEN_HOOK = None                    # test hook: called as GEN_HOOK(kind, run, gen, rec) after each saved generation
_STATUS_LOCK = threading.Lock()
BudgetStop, YieldStop = ea1.BudgetStop, ea1.YieldStop
TEST_PARTS = ('main', 'plus', 'rest')


class BatchIndependenceError(RuntimeError):
    pass


# ---------------------------------------------------------------- seeds
def block_seeds(smoke=False):
    b = {k: list(range(a, a+n)) for k, (a, n) in BLOCKS.items()}
    if smoke:
        b.pop('EA2_TEST')
    return b


def expand(base, offsets=OFFS4):
    return e1.expand(base, offsets)


def existing_all(root=ROOT):
    """E-A1's existing ranges (seven + E1 v2's five) plus the three E-A1 blocks (20672 expanded seeds)."""
    ex = ea1.existing_all(root)
    for k, (a, n) in ea1.BLOCKS.items():
        ex['ea1_'+k] = list(range(a, a+n))
    return ex


def seed_report(blocks, existing, extra_historical=()):
    return e1.seed_report(blocks, existing, extra_historical)


def scan_saved_seeds(root=ROOT):
    """Integer seed fields of saved json/npz records, excluding only this experiment's own output (E-A1's output IS scanned)."""
    vals, srcs = set(), {}
    for p in sorted((Path(root)/'results').rglob('*')):
        rel = str(p.relative_to(root))
        if rel.startswith('results/direction2/ea2') or not p.is_file() or p.suffix not in ('.json', '.npz'):
            continue
        if p.suffix == '.json':
            vals |= seed_values(json.loads(p.read_text()))
        else:
            with np.load(p, allow_pickle=False) as z:
                if 'state' in z.files:
                    vals |= seed_values(json.loads(str(z['state'])))
        srcs[rel] = sha(p)
    return vals, srcs


def audit(smoke=False, root=ROOT):
    """The seed audit; an official audit must reproduce the registered counts and set hashes (design 1.3), otherwise it refuses."""
    blocks = block_seeds(smoke)
    hist, srcs = scan_saved_seeds(root)
    hist |= set(read(Path(root)/'results/direction2/holdout/seed_audit.json')['historical_seed_field_values'])
    rep = seed_report(blocks, existing_all(root), hist)
    if not smoke and (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) != (EXPECT_NEW, EXPECT_OLD, 0):
        raise ValueError('seed counts differ from the registered 13728/20672/0')
    if not smoke and (rep['new_expanded_sha256'], rep['existing_expanded_sha256']) != (NEW_SHA, OLD_SHA):
        raise ValueError('seed set hashes differ from the pre-run registered ones (design 1.3)')
    return rep, hist, srcs


def require_seeds(sd, stage, m, unlocked=False):
    """train -> EA2_TRAIN; sel -> EA2_SEL; at -> EA2_AT (re-adaptation training only); testcheck -> exactly EA2_SEL[:64];
    test -> EA2_TEST (only unlocked); drift -> the E1 v2 AS seeds."""
    sd = list(sd)
    if stage == 'drift':
        pool = (m.get('drift') or {}).get('seeds')
    elif stage in STAGE_BLOCK:
        if stage == 'test' and not unlocked:
            raise ValueError('test seeds are locked until selection_frozen and selection_plus_frozen are registered')
        pool = m['seeds'].get(STAGE_BLOCK[stage])
    else:
        raise ValueError('unknown stage '+str(stage))
    if pool is None:
        raise ValueError('this manifest has no block for stage '+stage)
    if len(sd) != len(set(sd)):
        raise ValueError('duplicate environment seeds')
    if not sd or any(type(s) is not int or s not in set(pool) for s in sd):
        raise ValueError('wrong seed split')
    if stage == 'testcheck' and sd != list(m['seeds']['EA2_SEL'][:m['params']['testcheck_n']]):
        raise ValueError('testcheck must use exactly the first EA2_SEL seeds')


perm_for = ea1.perm_for


def train_block(pool, run, g, n_S, perm_seed=PARAMS['attacker']['perm_seed']):
    return ea1.train_block(pool, run, g, n_S, perm_seed)


# ---------------------------------------------------------------- statistics / decision tree
tcrit, upper_bound, lower_bound = e1.tcrit, e1.upper_bound, e1.lower_bound
interval, first_max, classify, bitwise_equal, gains_of = ea1.interval, ea1.first_max, ea1.classify, ea1.bitwise_equal, ea1.gains_of
mlp, honest_pol, theta_id = e1.mlp, e1.honest_pol, e1.theta_id
best_record = ea1.best_record

# program key -> external label.  `not_stronger` is only the program key of classify(); (b) is never labelled "沒有更強" (design 2.2).
LABEL = dict(stronger='更強', weaker='更弱', small_gain='小幅提升（排除高出 ε 以上）', flat='持平', insufficient='證據不足以判定')
TEXT = dict(
    stronger=('更強：EA2_SEL 選出的單一從頭訓練攻擊者 A_fresh_orig（未再做適應）在原設定下比 a* 高 {mean:.6f}（單側 95% 下界 {lb:.6f} > 0.001）。'
              '在這個比較裡，a* 不是原設定下最高的攻擊者；E1 v2 的 D 在原設定一側用的是較低的一個。E1 v2 已登記的類別（對固定 a* 的 D＝0.004236，排除 0.005）不變。'
              '解釋不止一個，不得單一歸因：a* 的訓練設定較弱、新配方的偏離、從頭訓練探索到另一類策略、驗證點取最大的選取偏高；本實驗無法區分。'),
    small_gain=('小幅提升（排除高出 ε 以上）：A_fresh_orig（未再做適應）在原設定下高於 a*（Δ＝{mean:.6f}，單側 95% LB＝{lb:.6f} > 0，未校正），'
                '高出幅度排除達 0.001（UB＝{ub:.6f} < 0.001）；即檢查到小於 0.001 的超過。'),
    weaker=('更弱：A_fresh_orig（未再做適應）在原設定下不如 a*（Δ＝{mean:.6f}，單側 95% 上界 {ub:.6f} < 0）。'
            'a* 在原設定內訓練、適應並選取，使 Δ<0 本來就是先驗預期之一，不是獨立證據；檢查設計不對稱（a* 有主場優勢、A_fresh_orig 未再適應）；不能宣稱 a* 是最強。'),
    flat=('持平：排除了 A_fresh_orig（未再做適應）高出 a* 達 0.001 以上（Δ＝{mean:.6f}，單側 95% 上界 {ub:.6f} < 0.001）；不排除小於 0.001 的差（下界 {lb:.6f}）。'
          '檢查設計不對稱（a* 有主場優勢、A_fresh_orig 未再適應）；不能宣稱 a* 是最強。'),
    insufficient=('證據不足以判定：A_fresh_orig（未再做適應）相對 a* 的差為 Δ＝{mean:.6f}（單側 95% 區間 [{lb:.6f}, {ub:.6f}]），跨過預先登記的門檻 0.001，'
                  '因此既不能宣稱更強，也不能排除更強；原設定一側是否被低估維持未解決。'))
SUBJECT = 'A_fresh_orig (the single attacker selected on EA2_SEL, not re-adapted)'


def decide(D, eps=EPS):
    D = np.asarray(D, float)
    lb, ub = lower_bound(D), upper_bound(D)
    cat, sub = classify(lb, ub, eps)
    key = sub or cat
    mean = float(D.mean())
    return dict(category=cat, sub=sub, label=LABEL[key], text=TEXT[key].format(mean=mean, lb=lb, ub=ub), n=len(D), mean_delta=mean, LB=lb, UB=ub, eps=eps, subject=SUBJECT)


def power_classes(n, sigma, eps, mus):
    """Design 2.8 table: probabilities of the five classes (decision-tree order) for a true mean mu, with the sample mean ~ N(mu, sigma/sqrt(n)) and a
    fixed SD.  The registered table is a Monte-Carlo with sampled SD; this closed form is what the tests compare it with (tolerance 0.3 points)."""
    from scipy import stats
    t, se = tcrit(n), sigma/math.sqrt(n)
    w = t*se                                              # UB < x  <=>  mean < x - w ;  LB > x  <=>  mean > x + w
    out = []
    for mu in mus:
        def c(x, mu=mu):
            return float(stats.norm.cdf((x-mu)/se))      # P(sample mean < x)
        out.append(dict(mu=mu, stronger=1-c(eps+w), weaker=c(-w), small_gain=max(0., c(eps-w)-c(w)), flat=max(0., c(min(w, eps-w))-c(-w)),
                        insufficient=max(0., c(eps+w)-c(eps-w))))
    return out


# ---------------------------------------------------------------- consequence table (design 2.4): every row is a literal predicate
ROWS_MAIN = {
    'R1': lambda lb, ub: lb > EPS,
    'R2': lambda lb, ub: ub < EPS and lb > 0,
    'R3': lambda lb, ub: ub < EPS and lb <= 0,
    'R4': lambda lb, ub: ub >= EPS and lb <= EPS}
ROWS_DPOOL = {
    'R5': lambda lb, ub: lb > DELTA,
    'R6': lambda lb, ub: ub < DELTA,
    'R7': lambda lb, ub: lb <= DELTA and ub >= DELTA}
ROWS_PLUS = {
    'R8': lambda g, lb, ub: g == 0,
    'R9': lambda g, lb, ub: g > 0 and lb > EPS,
    'R10': lambda g, lb, ub: g > 0 and ub < EPS and lb > 0,
    'R11': lambda g, lb, ub: g > 0 and ub < EPS and lb <= 0,
    'R12': lambda g, lb, ub: g > 0 and ub >= EPS and lb <= EPS}
ROWS_PP = {
    'R13': lambda g, lb, ub: g > 0,
    'R14': lambda g, lb, ub: g == 0 and lb > EPS,
    'R15': lambda g, lb, ub: g == 0 and ub < EPS and lb > 0,
    'R16': lambda g, lb, ub: g == 0 and ub < EPS and lb <= 0,
    'R17': lambda g, lb, ub: g == 0 and ub >= EPS and lb <= EPS}
GROUPS = dict(main=ROWS_MAIN, d_pool=ROWS_DPOOL, plus=ROWS_PLUS, plusplus=ROWS_PP)
TREE_TO_ROW = dict(stronger='R1', small_gain='R2', weaker='R3', flat='R3', insufficient='R4')      # design 2.2 sub-label -> row
SUMMARY_ADD = dict(
    R1='「效果不足」僅對固定 a*；原設定下存在比 a* 高出 0.001 以上的單一攻擊者（單側 95%）；若要改寫判定，須新登記、新 seeds。',
    R5='「效果不足」僅對固定 a*；以本實驗 15 個條目的共同池計，描述性降幅的單側 95% 下界高於 0.005；E1 v2 類別不變；新判定須新登記（新 seeds）。')


def hits(rows, *a):
    return [k for k, f in rows.items() if f(*a)]


def the_row(rows, *a):
    """The one row of a group that a result hits; anything but exactly one is an error (the table is exclusive and exhaustive)."""
    h = hits(rows, *a)
    if len(h) != 1:
        raise ValueError(f'consequence table is not exclusive/exhaustive here: {a} hits {h}')
    return h[0]


def consequence_rows(main_lbub, d_lbub, plus_gen, plus_lbub, pp_lbub):
    """The four rows (one per group) of a complete outcome; (LB, UB) pairs are one-sided 95% bounds."""
    return dict(main=the_row(ROWS_MAIN, *main_lbub), d_pool=the_row(ROWS_DPOOL, *d_lbub), plus=the_row(ROWS_PLUS, plus_gen, *plus_lbub),
                plusplus=the_row(ROWS_PP, plus_gen, *pp_lbub))


def summary_additions(rows):
    """Union of the summary sentences that the hit rows require (only R1 and R5 have one)."""
    return [SUMMARY_ADD[r] for r in sorted(rows.values(), key=lambda s: int(s[1:])) if r in SUMMARY_ADD]


D_POOL_TEXT = dict(
    R5='本實驗 15 個條目中，兩側各自在 EA2_SEL 上最高者的描述性降幅為 {mean:.6f}，單側 95% 下界 {lb:.6f} 高於 0.005；這是描述性結果（登記前定義，非事後），'
       '不改 E1 v2 已登記的類別；要作為新判定須另行登記（新 seeds）。',
    R6='本實驗 15 個條目中，兩側各自在 EA2_SEL 上最高者的描述性降幅為 {mean:.6f}，單側 95% 上界 {ub:.6f} 仍低於 0.005。',
    R7='本實驗 15 個條目中，兩側各自在 EA2_SEL 上最高者的描述性降幅為 {mean:.6f}，單側 95% 區間 [{lb:.6f}, {ub:.6f}] 跨過 0.005，無法判斷。')
PLUS_TEXT = dict(
    R8='再適應鏈在 EA2_SEL 的 13 個驗證點上沒有選出高於起點的 checkpoint，A_fresh_orig⁺ 與 A_fresh_orig 為同一參數，本臂未提供適應後的資訊。',
    R9='a* 在原設定下被經適應的 A_fresh_orig⁺ 超過（單一、描述性、單側 95%）：Δ⁺＝{mean:.6f}，下界 {lb:.6f} > 0.001。',
    R10='A_fresh_orig⁺ 在原設定下高於 a*（單側 95% LB＝{lb:.6f} > 0，未校正，描述性），高出幅度排除達 0.001（UB＝{ub:.6f}）。',
    R11='A_fresh_orig⁺ 本身沒有被發現比 a* 強 0.001 以上（描述性）；a* 仍多了 12 取 1 的選擇，不排除更強者。',
    R12='A_fresh_orig⁺ 相對 a* 的差為 Δ⁺＝{mean:.6f}（單側 95% 區間 [{lb:.6f}, {ub:.6f}]），跨過 0.001，既不能宣稱更強也不能排除更強（描述性）。')
PP_TEXT = dict(
    R13='A_fresh_orig⁺⁺ 與 A_fresh_orig⁺ 為同一參數，不另報。',
    R14=('強制走過適應的 A_fresh_orig⁺⁺（再適應鏈在代數 ≥20 的驗證點中 EA2_SEL 最大者）在原設定下比 a* 高 {mean:.6f}（單側 95% 下界 {lb:.6f} > 0.001；描述性，單一攻擊者；'
         '來自 12 個驗證點取最大，有選取偏高，不表示適應後必然更強）。'),
    R15='A_fresh_orig⁺⁺ 在原設定下高於 a*（單側 95% LB＝{lb:.6f} > 0，未校正，描述性），高出幅度排除達 0.001；含選取偏高。',
    R16=('A_fresh_orig⁺⁺ 本身沒有被發現比 a* 強 0.001 以上（描述性）；在 collapsed 情況下 A_fresh_orig⁺⁺ 在 EA2_SEL 上的 G 不高於起點（定義使然），'
         'a* 仍多了 12 取 1 的選擇，不排除更強者。'),
    R17=('A_fresh_orig⁺⁺ 相對 a* 的差為 Δ⁺⁺＝{mean:.6f}（單側 95% 區間 [{lb:.6f}, {ub:.6f}]），跨過 0.001，既不能宣稱更強也不能排除更強（描述性）。'))


# ---------------------------------------------------------------- re-adaptation arm: A+ / A++ selection (design 2.5)
def select_plus(records, pp_min_gen):
    """(index of A+, index of A++) into `records`.  A+ = first max G_SEL over all validation points (ties -> earlier); A++ = first max over the
    points with gen >= pp_min_gen (ties -> earlier).  If A+ is not generation 0, A++ is the same record."""
    gs = [r['G_SEL'] for r in records]
    i = first_max(gs)
    late = [k for k, r in enumerate(records) if r['gen'] >= pp_min_gen]
    if not late:
        raise ValueError('no validation point at or after the forced-adaptation generation')
    return i, late[first_max([gs[k] for k in late])]


def pick_pool(G, names, astar):
    """A_O' / A_C2': the pool entry with the largest EA2_SEL G; ties go to a*, then to the list order (design 2.3)."""
    first_max(G)                                    # finite check
    top = max(G)
    tied = [i for i, g in enumerate(G) if g == top]
    return names.index(astar) if astar in names and names.index(astar) in tied else tied[0]


def dedupe(named):
    """[(name, theta)] -> distinct parameter vectors by theta hash, the first name is the entry, later names are aliases."""
    out, idx = [], {}
    for n, t in named:
        h = theta_id(t)
        if h in idx:
            out[idx[h]]['aliases'].append(n)
        else:
            idx[h] = len(out)
            out.append(dict(name=n, aliases=[], theta=[float(x) for x in t], theta_sha256=h))
    return out


# ---------------------------------------------------------------- manifest
def out_dir():
    if MODE.get('out'):
        return Path(MODE['out'])
    return ROOT/(SMOKE_DIR if MODE['smoke'] else OUT_DIR)


def git(*a):
    return subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()


def environment():
    """Library / GPU record (design 3).  No machine name is recorded.  Fields that cannot be read are None."""
    import torch
    d = dict(python=platform.python_version(), d2_graph_k=os.environ.get('D2_GRAPH_K'), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
             cuda=torch.version.cuda, threads={k: os.environ.get(k) for k in THREAD_VARS}, device_name=None, capability=None, gpu_name=None, gpu_uuid=None, driver=None)
    if torch.cuda.is_available():
        d.update(device_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            d['gpu_name'], d['gpu_uuid'], d['driver'] = [x.strip() for x in out.stdout.strip().splitlines()[0].split(',')]
    except (OSError, subprocess.SubprocessError):
        pass
    return d


def check_environment(m, env=None):
    """Every GPU stage: GPU model, torch / CUDA / numpy / scipy versions and thread variables must equal the manifest's record."""
    ea1.check_environment(m, env or environment())


def check_hashes(mapping, root=ROOT):
    ea1.check_hashes(mapping, root)


def check_ea1_refs(root=ROOT, recorded=None):
    """Design 1.2: every pinned E-A1 file still has its registered hash (and a manifest, if given, records exactly these pins)."""
    if recorded is not None and recorded != EA1_REFS:
        raise ValueError('manifest records other E-A1 hashes than the registered ones')
    check_hashes(EA1_REFS, root)


def build_manifest(smoke=False, root=ROOT):
    root = Path(root)
    check_ea1_refs(root)
    _, cks, _ = context()
    blocks = block_seeds(smoke)
    rep, hist, srcs = (seed_report(blocks, existing_all(root)), set(), {}) if smoke else audit(False, root)
    audit_rec = dict(schema='ea2-seed-audit/1', report=rep, offsets=list(OFFS4), saved_seed_values=sorted(hist), scanned_sha256=srcs,
                     limitation='Saved project records only; cannot certify unrecorded external executions.')
    e1m = ea1.e1v2_manifest()
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
             input_sha256={p: sha(root/p) for p in INPUTS if (root/p).exists()}, protected_sha256=protected, e1v2_refs_sha256=ea1.e1v2_refs(root),
             ea1_refs_sha256=dict(EA1_REFS), ea1_design_sha256_record=dict([(EA1_DESIGN[0], sha(root/EA1_DESIGN[0]))]),
             environment=environment(), stage_blocks=STAGE_BLOCK,
             test_unlock='selection_frozen.json (freeze_record.json) and selection_plus_frozen.json committed, and a passing testcheck.json, before any EA2_TEST seed is read')
    m['sha256'] = digest(m)
    return m, audit_rec


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
    m, audit_rec = build_manifest(smoke)
    pins = {k: (m['environment'].get(k), v) for k, v in ENV_PIN.items() if m['environment'].get(k) != v}
    if pins and not smoke:
        raise ValueError(f'environment differs from the design pins (freeze only on the registered machine type): {pins}')
    if pins:
        print('smoke: environment differs from the design pins (recorded as is):', pins, flush=True)
    write(d/'seed_audit.json', audit_rec)
    m['seed_audit_sha256'] = sha(d/'seed_audit.json')
    m['sha256'] = digest({k: v for k, v in m.items() if k != 'sha256'})
    write(d/'manifest.json', m)
    print('manifest written', m['sha256'], '(commit it and seed_audit.json before running)' if not smoke else '(smoke)', flush=True)


def verify_manifest_dict(m):
    if m['sha256'] != digest({k: v for k, v in m.items() if k != 'sha256'}):
        raise ValueError('manifest checksum changed')
    expected = SMOKE_PARAMS if m['smoke'] else PARAMS
    if m['params'] != json.loads(json.dumps(expected)):
        raise ValueError('registered parameters differ from the code')
    return m


def load_manifest(clean=True):
    """Never calls ea1.load_manifest.  Checks: own checksum, own input hashes, the pinned E-A1 file hashes, E1 v2's refs and protected files, the
    seed audit and the simulator hash.  `code_git_sha` is recorded but not verified; code consistency = all input hashes equal AND (clean) committed."""
    d = out_dir()
    m = verify_manifest_dict(read(d/'manifest.json'))
    if m['smoke'] != MODE['smoke']:
        raise ValueError('smoke/official manifest mixup')
    check_ea1_refs(ROOT, m['ea1_refs_sha256'])
    check_hashes({**m['input_sha256'], **m['protected_sha256'], **m['e1v2_refs_sha256'], DESIGN: m['design_sha256']})
    if sha(d/'seed_audit.json') != m['seed_audit_sha256']:
        raise ValueError('seed audit changed')
    audit_rec = read(d/'seed_audit.json')
    rep = seed_report(m['seeds'], existing_all(), audit_rec['saved_seed_values'])
    if rep['new_expanded_sha256'] != m['seed_expanded_sha256']:
        raise ValueError('seed set changed')
    if main_eval.code_hash() != m['source_sha256']:
        raise ValueError('simulator changed')
    if not m['smoke'] and clean:
        for p in [*m['input_sha256'], str((d/'manifest.json').relative_to(ROOT)), str((d/'seed_audit.json').relative_to(ROOT))]:
            if not plan.is_committed_and_clean(str(ROOT), p):
                raise ValueError('uncommitted input: '+p)
    return m


# ---------------------------------------------------------------- infrastructure
check_budget, check_yield, foreign_gpu_pids, Backend, make_backend, default_backend, Meter = (
    ea1.check_budget, ea1.check_yield, ea1.foreign_gpu_pids, ea1.Backend, ea1.make_backend, ea1.default_backend, ea1.Meter)


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


def sim(m, cfg, pols, sd, T, be=None):
    """One simulate call; a graph error is always a hard error (no eager fallback anywhere in E-A2)."""
    return (be or default_backend()).simulate(cfg, pols, sd, T, m['params']['exclude'])


def sim_eager(m, cfg, pols, sd, T):
    """Reference path for testcheck: no graph, one call (not chunked)."""
    return ea1.sim_eager(m, cfg, pols, sd, T)


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


# ---------------------------------------------------------------- attack chains on the ORIGINAL setting (cold start)
def make_opt(m, run):
    a = m['params']['attacker']
    return cmaes.make_optimizer(55, lam=a['lam'], sigma0=a['sigma0'], seed=a['opt_seed']+run)       # mean = 0: cold start from the honest policy


def chain_budget(m, run):
    a, p = m['params']['attacker'], m['params']
    return dict(lam=a['lam'], sigma0=a['sigma0'], n_S=a['n_S'], gens=a['gens'], val_every=a['val_every'], opt_seed=a['opt_seed']+run,
                perm_seed=a['perm_seed']+run, T_train=p['T_train'], T_val=p['T_val'], exclude=p['exclude'], start='zeros',
                train_sha256=digest(m['seeds']['EA2_TRAIN']), sel_sha256=digest(m['seeds']['EA2_SEL']))


def chain_path(run, name=None):
    return out_dir()/'chains'/(f'run{run}.json' if name is None else f'run{run}_{name}.json')


def chain_generation(m, cfg, run, opt, be):
    """One generation: ask, one simulate call over [honest]+lam candidates on the n_S training seeds of this generation, tell.
    fitness = mean per-step utility over the score window T-5000."""
    a = m['params']['attacker']
    sd = train_block(m['seeds']['EA2_TRAIN'], run, opt.gen, a['n_S'], a['perm_seed'])
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
    """Train (or resume) chain `run` to `target` generations (default: all) on the original setting.  Saved after every generation;
    validation (EA2_SEL, original setting) every val_every generations."""
    a, cfg = m['params']['attacker'], m['params']['base_cfg']
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
    while opt.gen < target:
        check_budget()
        check_yield()
        t = time.time()
        rec = chain_generation(m, cfg, run, opt, be)
        f_h = float(rec['U'][0].mean()/rec['T_score'])
        state['curve'].append(dict(gen=opt.gen, f_honest=f_h, f_mean=float(rec['f'].mean()), f_max=float(rec['f'].max()), wall_s=time.time()-t))
        if opt.gen % a['val_every'] == 0:
            raw, rel = evaluate(m, cfg, m['seeds']['EA2_SEL'], 'sel', [honest_pol(), mlp(opt.mean)], ['honest', theta_id(opt.mean)], be)
            g = float(v1.metrics(raw)['gains'][0].mean())
            if not math.isfinite(g):
                raise ValueError('nonfinite validation gain')
            state['records'].append(dict(gen=opt.gen, G_SEL=g, theta=np.asarray(opt.mean).tolist(), raw=rel))
        state['opt'] = opt.state_dict()
        write(path, state)
        stamp('train', run=run, generation=opt.gen, generation_seconds=time.time()-t)
        if GEN_HOOK is not None:
            GEN_HOOK('chain', run, rec['gen'], rec)
    return state


def run_chains(m, runs, factory=None):
    """lanes = 1, fixed: the chains one after the other on one lane."""
    factory = factory or make_backend
    for r in runs:
        be = factory()
        with be:
            run_chain(m, r, be)
        be.release()


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


def chain_result(state):
    """A chain's state without wall-clock fields (what must be bitwise equal across execution modes)."""
    return dict(state, curve=[{k: v for k, v in c.items() if k != 'wall_s'} for c in state['curve']])


CODE_FILES = ("scripts/ea2.py", "scripts/ea2_launch.sh", "tests/direction2/test_ea2.py")


def code_hashes(root=ROOT):
    """Content hashes of the three E-A2 code files; a smoke summary is only valid for the code it was produced with."""
    return {p: sha(Path(root)/p) for p in CODE_FILES}


def require_smoke_summary(m):
    """Before train / adapt: the smoke summary was kept (smoke-clean), graph == eager held for the main chain and the + setting, it was produced by
    exactly the current ea2.py / ea2_launch.sh / test_ea2.py, and it is committed."""
    p = out_dir()/'smoke_summary.json'
    if not p.exists():
        raise ValueError('smoke_summary.json is missing: run smoke and smoke-clean first')
    s = read(p)
    if not s.get('graph_equals_eager'):
        raise ValueError('smoke summary: graph != eager; stop and ask the user')
    if s.get('code_sha256') != code_hashes():
        raise ValueError('smoke summary was produced by other code than the current ea2.py / ea2_launch.sh / test_ea2.py: run smoke again')
    if not m['smoke'] and not plan.is_committed_and_clean(str(ROOT), os.path.relpath(p, ROOT)):
        raise ValueError('uncommitted: smoke_summary.json')
    return s


def stage_train(m):
    require_smoke_summary(m)
    a = m['params']['attacker']
    runs = [r for r in range(a['chains']) if not (chain_path(r).exists() and chain_state(r)['opt']['gen'] == a['gens'])]
    stamp('train', lanes=1, pending_runs=runs)
    run_chains(m, runs)
    chains_complete(m)
    mark_done('train')


# ---------------------------------------------------------------- re-adaptation arm (design 2.5)
def plus_path():
    return out_dir()/'chains'/'plus.json'


def make_plus_opt(m, theta):
    a = m['params']['plus']
    opt = cmaes.make_optimizer(55, lam=a['lam'], sigma0=a['sigma0'], seed=a['opt_seed'])
    opt.mean = np.asarray(theta, float).copy()
    return opt


def plus_budget(m, theta):
    a, p = m['params']['plus'], m['params']
    return dict(lam=a['lam'], sigma0=a['sigma0'], n_S=a['n_S'], gens=a['gens'], val_every=a['val_every'], opt_seed=a['opt_seed'], key=a['key'],
                start_theta_sha256=theta_id(theta), T_train=p['T_train'], T_val=p['T_val'], exclude=p['exclude'],
                at_sha256=digest(m['seeds']['EA2_AT']), sel_sha256=digest(m['seeds']['EA2_SEL']))


def plus_generation(m, cfg, opt, be):
    """One generation of the re-adaptation arm: n_S rotating seeds of EA2_AT (training only), one simulate call, tell."""
    a = m['params']['plus']
    sd = e1.rotating(m['seeds']['EA2_AT'], a['key'], opt.gen, a['n_S'])
    require_seeds(sd, 'at', m)
    X = opt.ask()
    raw = sim(m, cfg, [honest_pol()]+[mlp(x) for x in X], sd, m['params']['T_train'], be)
    U = np.asarray(raw['snap']['util_b'], float)[:, :, 1]
    if not np.isfinite(U).all():
        raise ValueError('nonfinite training utility')
    f = U[1:].mean(1)/raw['T_score']
    gen = opt.gen
    opt.tell(X, f)
    return dict(gen=gen, X=X, U=U, f=f, seeds=sd, T_score=raw['T_score'])


def run_plus(m, theta, be, target=None):
    """Train (or resume) the single re-adaptation chain from `theta` on the original setting; validate (EA2_SEL) at generation 0 and every val_every."""
    a, cfg = m['params']['plus'], m['params']['base_cfg']
    target = a['gens'] if target is None else target
    if not 0 < target <= a['gens']:
        raise ValueError('the re-adaptation chain is never extended beyond the registered generations')
    path = plus_path()
    opt = make_plus_opt(m, theta)
    budget = plus_budget(m, theta)
    state = state_load(path, m, dict(cfg=cfg, budget=budget, opt=opt.state_dict(), curve=[], records=[]))
    if state['cfg'] != cfg or state['budget'] != budget:
        raise ValueError('re-adaptation chain resumed under another defender, start or budget')
    opt.load_state_dict(state['opt'])
    if len(state['curve']) != opt.gen:
        raise ValueError('re-adaptation chain state is inconsistent')

    def validate():
        raw, rel = evaluate(m, cfg, m['seeds']['EA2_SEL'], 'sel', [honest_pol(), mlp(opt.mean)], ['honest', theta_id(opt.mean)], be)
        g = float(v1.metrics(raw)['gains'][0].mean())
        if not math.isfinite(g):
            raise ValueError('nonfinite validation gain')
        state['records'].append(dict(gen=opt.gen, G_SEL=g, theta=np.asarray(opt.mean).tolist(), raw=rel))
    if not state['records']:
        validate()
        write(path, state)
    while opt.gen < target:
        check_budget()
        check_yield()
        t = time.time()
        rec = plus_generation(m, cfg, opt, be)
        state['curve'].append(dict(gen=opt.gen, f_honest=float(rec['U'][0].mean()/rec['T_score']), f_mean=float(rec['f'].mean()), f_max=float(rec['f'].max()),
                                   wall_s=time.time()-t))
        if opt.gen % a['val_every'] == 0:
            validate()
        state['opt'] = opt.state_dict()
        write(path, state)
        stamp('adapt', generation=opt.gen, generation_seconds=time.time()-t)
        if GEN_HOOK is not None:
            GEN_HOOK('plus', 0, rec['gen'], rec)
    return state


def plus_complete(m, state):
    a = m['params']['plus']
    if state['opt']['gen'] != a['gens'] or len(state['curve']) != a['gens']:
        raise ValueError('the re-adaptation chain is not complete')
    if [r['gen'] for r in state['records']] != list(range(0, a['gens']+1, a['val_every'])) or not all(math.isfinite(r['G_SEL']) for r in state['records']):
        raise ValueError('the re-adaptation chain has missing or non-finite validation points')


def saturation_of(traj, val_every):
    try:
        return e1.saturation(traj, val_every, traj[-1][0], 0.001)
    except ValueError as exc:             # only with a budget shorter than the registered one (tests)
        return dict(error=str(exc))


def plus_record(m, sel, state):
    """Pure: the content of selection_plus_frozen.json from the finished chain (design 2.5)."""
    a = m['params']['plus']
    recs = state['records']
    i, j = select_plus(recs, a['pp_min_gen'])

    def pick(r):
        return dict(gen=r['gen'], G_SEL=r['G_SEL'], theta=r['theta'], theta_sha256=theta_id(r['theta']))
    traj = [[r['gen'], r['G_SEL']] for r in recs]
    return dict(schema='ea2-selection-plus/1', manifest_sha256=m['sha256'], start=dict(name='A_fresh_orig', theta_sha256=sel['A_fresh_orig']['theta_sha256']),
                A_plus=pick(recs[i]), A_plusplus=pick(recs[j]), collapsed=bool(recs[i]['gen'] == 0), same_theta=bool(i == j),
                trajectory=traj, saturation=saturation_of(traj, a['val_every']), pp_min_gen=a['pp_min_gen'])


def stage_adapt(m):
    """Needs a committed select.  Writes chains/plus.json and selection_plus_frozen.json (A+ and A++ together); commit both before testcheck."""
    sel = verify_freeze(m)
    sp = out_dir()/'selection_plus_frozen.json'
    if sp.exists():           # crash after writing: register what was written, never recompute
        if read(sp)['manifest_sha256'] != m['sha256']:
            raise ValueError('re-adaptation freeze from another manifest')
        mark_done('adapt')
        return
    require_smoke_summary(m)
    start = sel['A_fresh_orig']['theta']
    be = make_backend()
    with be:
        state = run_plus(m, start, be)
    be.release()
    plus_complete(m, state)
    rec = plus_record(m, sel, state)
    rec.update(selection_frozen_sha256=sha(out_dir()/'selection_frozen.json'), plus_chain_sha256=sha(plus_path()))
    write(sp, rec)
    mark_done('adapt')
    print(f're-adaptation frozen: A+ gen {rec["A_plus"]["gen"]}, A++ gen {rec["A_plusplus"]["gen"]}, collapsed={rec["collapsed"]}; '
          'commit chains/plus.json and selection_plus_frozen.json before testcheck', flush=True)


def verify_plus_freeze(m, sel):
    d = out_dir()
    sp = d/'selection_plus_frozen.json'
    if m['smoke'] or not sp.exists() or not plus_path().exists():
        raise ValueError('the re-adaptation arm is not frozen')
    rec = read(sp)
    if rec['manifest_sha256'] != m['sha256'] or rec['selection_frozen_sha256'] != sha(d/'selection_frozen.json') or rec['plus_chain_sha256'] != sha(plus_path()):
        raise ValueError('selection_plus_frozen does not match the manifest, the frozen selection or its chain')
    for p in (sp, plus_path()):
        if not plan.is_committed_and_clean(str(ROOT), os.path.relpath(p, ROOT)):
            raise ValueError('re-adaptation freeze is not committed: '+p.name)
    state = read(plus_path())
    plus_complete(m, state)
    again = plus_record(m, sel, state)
    for k in ('A_plus', 'A_plusplus', 'collapsed', 'same_theta'):
        if again[k] != rec[k]:
            raise ValueError('selection_plus_frozen differs from the selection recomputed from its chain: '+k)
    for k in ('A_plus', 'A_plusplus'):
        if theta_id(rec[k]['theta']) != rec[k]['theta_sha256']:
            raise ValueError('a frozen theta does not match its hash')
    return rec


# ---------------------------------------------------------------- E1 v2 / E-A1 inputs (read only)
def e1v2_entries():
    return ea1.e1v2_entries()


def drift_reference(inp):
    return ea1.drift_reference(inp)


def ea1_inputs(root=ROOT):
    """From E-A1's frozen selection (its file hash is pinned): the three C2-trained checkpoints and E-A1's A_fresh, hashes re-verified."""
    s = read(Path(root)/EA1_DIR/'selection_frozen.json')
    ents = {e['name']: e for e in s['entries'] if e['name'].startswith('fresh_run')}
    if sorted(ents) != ['fresh_run0', 'fresh_run1', 'fresh_run2']:
        raise ValueError('E-A1 selection lacks its three chain checkpoints')
    for e in [*ents.values(), s['astar']]:
        if theta_id(e['theta']) != e['theta_sha256']:
            raise ValueError('an E-A1 theta does not match its hash')
    if s['astar']['theta_sha256'] != ASTAR_SHA256:
        raise ValueError('E-A1 a* differs from the registered a*')
    af = ents[s['A_fresh']['name']]
    return dict(chains=[(f'ea1_{n}', [float(x) for x in ents[n]['theta']]) for n in sorted(ents)],
                A_fresh=dict(name='ea1_'+s['A_fresh']['name'], gen=s['A_fresh']['gen'], theta=[float(x) for x in af['theta']], theta_sha256=af['theta_sha256']))


# ---------------------------------------------------------------- select (freeze)
def stage_select(m):
    chains_complete(m)
    d, a = out_dir(), m['params']['attacker']
    sel_path = d/'selection_frozen.json'
    if sel_path.exists():     # crash between writing the selection and registering it: register what was written, never recompute
        if read(sel_path)['manifest_sha256'] != m['sha256']:
            raise ValueError('selection from another manifest')
        if not (d/'freeze_record.json').exists():
            write(d/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
        mark_done('select')
        return
    inp, e1a = e1v2_entries(), ea1_inputs()
    if inp['c2_cfg'] != m['params']['c2_cfg']:
        raise ValueError('C2 configuration differs from E1 v2 selection')
    cfg_c2, base, star = m['params']['c2_cfg'], m['params']['base_cfg'], inp['astar']
    SEL = m['seeds']['EA2_SEL']
    # 1. each chain's own best checkpoint (chain-internal G_SEL, used only for this choice)
    chains = []
    for run in range(a['chains']):
        r = best_record(chain_state(run)['records'])
        chains.append(dict(run=run, gen=r['gen'], G_SEL_chain=r['G_SEL'], theta=r['theta'], theta_sha256=theta_id(r['theta']), name=f'ea2_fresh_run{run}'))
    # 2. one re-evaluation in the original setting [honest, a*, chain0..2] -> A_fresh_orig (largest G_orig; ties -> smaller run)
    raw, rel = evaluate(m, base, SEL, 'sel', [honest_pol(), mlp(star['theta'])]+[mlp(c['theta']) for c in chains], ['honest', 'a_star']+[c['name'] for c in chains])
    G = gains_of(raw).mean(1).tolist()
    k = first_max(G[1:])
    fresh = dict(chains[k], G_SEL_reeval=G[1:][k], index=k)
    reeval = dict(raw=rel, names=['a_star']+[c['name'] for c in chains], G=G)
    # 3. the common pool: 15 distinct parameter vectors, re-evaluated once per setting -> A_O', A_C2'
    pool = dedupe(list(inp['entries'])+e1a['chains']+[(c['name'], c['theta']) for c in chains])
    if len(pool) != EXPECT_POOL:
        raise ValueError(f'the common pool has {len(pool)} distinct entries, registered {EXPECT_POOL}: stop and ask the user')
    names = [e['name'] for e in pool]
    if star['name'] not in names:
        raise ValueError('a* is not in the pool')
    pids = ['honest']+names
    pols = [honest_pol()]+[mlp(e['theta']) for e in pool]
    desc = {}
    for label, c in (('orig', base), ('C2', cfg_c2)):
        raw, rel = evaluate(m, c, SEL, 'sel', pols, pids)
        desc[label] = dict(raw=rel, names=names, G=gains_of(raw).mean(1).tolist())
    j_o, j_c = pick_pool(desc['orig']['G'], names, star['name']), pick_pool(desc['C2']['G'], names, star['name'])
    # 4. machine drift (descriptive): a* on the E1 v2 AS seeds, original and C2, against E1 v2's own numbers
    drift = dict(schema='ea2-drift/1', manifest_sha256=m['sha256'], note='descriptive; no pass/fail threshold; cross-machine comparisons are reference only if not bitwise identical')
    ref = drift_reference(inp)
    for label, c in (('orig', base), ('C2', cfg_c2)):
        raw, rel = evaluate(m, c, m['drift']['seeds'], 'drift', [honest_pol(), mlp(star['theta'])], ['honest', 'a_star'])
        g_new, g_old = gains_of(raw)[0], ref[label]
        drift[label] = dict(raw=rel, G_new=float(g_new.mean()), G_e1v2_AS=inp['G_AS_astar'][label], mean_diff=float(g_new.mean()-g_old.mean()),
                            max_abs_per_seed=float(np.abs(g_new-g_old).max()), n_bitwise_identical=int(sum(bitwise_equal(x, y) for x, y in zip(g_new, g_old))), n=len(g_new))
    write(d/'drift.json', drift)
    # 5. register
    rec = dict(schema='ea2-selection/1', manifest_sha256=m['sha256'], astar=star, ea1_A_fresh=e1a['A_fresh'], chains=chains, A_fresh_orig=fresh,
               A_O_prime=dict(name=names[j_o], index=j_o, G_SEL=desc['orig']['G'][j_o]), A_C2_prime=dict(name=names[j_c], index=j_c, G_SEL=desc['C2']['G'][j_c]),
               reeval_orig=reeval, pool=pool, descriptive_SEL=desc, c2_cfg=cfg_c2, drift_sha256=sha(d/'drift.json'), ea1_refs_sha256=dict(EA1_REFS))
    write(sel_path, rec)
    write(d/'freeze_record.json', dict(manifest_sha256=m['sha256'], selection_frozen_sha256=sha(sel_path), written_unix=time.time()))
    mark_done('select')
    print('selection registered: commit selection_frozen.json, freeze_record.json, drift.json before adapt', flush=True)


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
    for e in [*sel['pool'], sel['astar'], sel['ea1_A_fresh'], sel['A_fresh_orig'], *sel['chains']]:
        if theta_id(e['theta']) != e['theta_sha256']:
            raise ValueError('a frozen theta does not match its hash')
    byname = {e['name']: e for e in sel['pool']}
    if byname[sel['ea1_A_fresh']['name']]['theta_sha256'] != sel['ea1_A_fresh']['theta_sha256'] or sel['ea1_refs_sha256'] != EA1_REFS:
        raise ValueError('frozen selection is inconsistent with the E-A1 references')
    return sel


# ---------------------------------------------------------------- test plan
def test_pols(sel, plus):
    """(name, setting, thetas, policies) of the six test evaluations, in the registered order; policy 0 is always the honest row."""
    star, afo, e1a = sel['astar']['theta'], sel['A_fresh_orig']['theta'], sel['ea1_A_fresh']['theta']
    ap, app = plus['A_plus']['theta'], plus['A_plusplus']['theta']
    pool = [e['theta'] for e in sel['pool']]
    spec = [('main_orig', 'orig', [star, afo]), ('plus_orig', 'orig', [star, afo, ap, app]), ('plus_C2', 'C2', [star, ap, app, e1a]),
            ('aux_C2', 'C2', [star, afo, e1a]), ('sec_orig', 'orig', pool), ('sec_C2', 'C2', pool)]
    return [(n, w, ts, [honest_pol()]+[mlp(t) for t in ts]) for n, w, ts in spec]


def test_cfgs(m):
    return dict(C2=m['params']['c2_cfg'], orig=m['params']['base_cfg'])


def test_path_id():
    return dict(graph=bool(MODE['graph']), device=MODE['device'], chunk=None, fallback=False)


def run_testcheck():
    """Pre-check on the first 64 EA2_SEL seeds (never EA2_TEST): the test path must reproduce eager exactly, for ALL six policy combinations."""
    m = load_manifest()
    sel = verify_freeze(m)
    plus = verify_plus_freeze(m, sel)
    d = out_dir()
    if not MODE['strict']:
        raise RuntimeError('testcheck must run in strict mode')
    if MODE['device'].startswith('cuda') and not MODE['graph']:
        raise RuntimeError('testcheck/test on cuda must use the graph path (the verified one)')
    sd = m['seeds']['EA2_SEL'][:m['params']['testcheck_n']]
    require_seeds(sd, 'testcheck', m)
    chunk, cfgs, checks = m['params']['test_chunk'], test_cfgs(m), []
    for name, which, _, pols in test_pols(sel, plus):
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
    rec = dict(schema='ea2-testcheck/1', manifest_sha256=m['sha256'], freeze_record_sha256=sha(d/'freeze_record.json'),
               plus_frozen_sha256=sha(d/'selection_plus_frozen.json'), path=dict(test_path_id(), chunk=chunk), seeds=[sd[0], sd[-1], len(sd)],
               tolerance='exact equality, no tolerance', environment=environment(), passed=all(c['identical'] for c in checks), checks=checks, unix=time.time())
    write(d/'testcheck.json', rec)
    if not rec['passed']:
        raise RuntimeError('testcheck failed: the test path differs from eager; test refused')
    return rec


def require_testcheck(m, sel, plus):
    """Called before anything is written to the ledger."""
    d = out_dir()
    p = d/'testcheck.json'
    if not p.exists():
        raise ValueError('testcheck.json is missing: run the testcheck stage first')
    t = read(p)
    if (not t.get('passed') or t['manifest_sha256'] != m['sha256'] or t['freeze_record_sha256'] != sha(d/'freeze_record.json')
            or t['plus_frozen_sha256'] != sha(d/'selection_plus_frozen.json')):
        raise ValueError('testcheck.json failed or does not match the current manifest / freeze_record / selection_plus_frozen')
    if t['path'] != dict(test_path_id(), chunk=m['params']['test_chunk']):
        raise ValueError('test execution path differs from the one testcheck verified')
    if MODE['device'].startswith('cuda') and not MODE['graph']:
        raise ValueError('test on cuda must use the graph path (the verified one)')
    if MODE['graph'] and MODE['device'].startswith('cuda'):      # every test shape must capture before the ledger is touched (EA2_SEL seeds only)
        sd = m['seeds']['EA2_SEL'][:m['params']['test_chunk']]
        require_seeds(sd, 'testcheck', m)
        for name, which, _, pols in test_pols(sel, plus):
            sim(m, test_cfgs(m)[which], pols, sd, m['params']['T_val'])


def require_result_committed(m):
    """The primary result is committed before anything else is opened (design 2.8): result.json exists, is clean in git and belongs to this manifest."""
    p = out_dir()/'result.json'
    if not p.exists() or read(p)['manifest_sha256'] != m['sha256']:
        raise ValueError('result.json is missing or belongs to another manifest: run `test --part main` and commit it first')
    if not plan.is_committed_and_clean(str(ROOT), os.path.relpath(p, ROOT)):
        raise ValueError('result.json is not committed and clean: commit the primary result before the other test parts')


# ---------------------------------------------------------------- ledger (once; one authorised reset for an aborted attempt)
def aborted_path(folder):
    return Path(folder)/'ledger_aborted.json'


def ledger_once(folder, name, con, runner):
    """d2_holdout_eval.once, plus: after an authorised reset the evaluation may be run again with the SAME contract only, and only once the
    archive (ledger_aborted.json) that records the reset is committed and clean in git."""
    ab = read(aborted_path(folder)) if aborted_path(folder).exists() else []
    for x in ab:
        if x['name'] == name:
            if x['contract_sha256'] != digest(con):
                raise ValueError('an evaluation re-run after a ledger reset must use the same contract')
            if (folder/f'{name}_raw.json').exists():
                continue                                    # nothing is re-run: the raw is read back
            if not plan.is_committed_and_clean(str(ROOT), os.path.relpath(aborted_path(folder), ROOT)):
                raise ValueError('ledger_aborted.json is not committed: commit the reset record before the evaluation is run again')
    return _once(folder, name, con, runner)


def ever_in_git(path):
    """True if `path` is, or ever was, tracked in this repository (any ref) or is staged.  Paths outside the repository are never tracked."""
    rel = os.path.relpath(path, ROOT)
    if rel.startswith('..'):
        return False
    log = subprocess.run(['git', 'log', '--all', '--oneline', '--', rel], cwd=ROOT, capture_output=True, text=True)
    ls = subprocess.run(['git', 'ls-files', '--', rel], cwd=ROOT, capture_output=True, text=True)
    if log.returncode or ls.returncode:
        raise RuntimeError('cannot query git history; a ledger reset is refused without it')
    return bool(log.stdout.strip() or ls.stdout.strip())


def ledger_reset(folder, name, reason, authorized=False, confirm_started_unix=None):
    """A started attempt without a raw file (nothing was ever read out) may be run once more, with the user's authorisation (design 3).
    Refused when: no authorisation flag; the entry is not `started`; a raw file exists, or ever existed in git; the caller does not repeat the entry's
    started_unix (so the ledger entry must have been looked at); for main_orig, result.json exists.  The old entry and the reason are archived in
    ledger_aborted.json, never deleted.  Writes: archive first, then the ledger (both atomic); an interrupted pair is completed by a second call."""
    folder = Path(folder)
    if authorized is not True:
        raise PermissionError('ledger reset needs the user authorisation flag')
    if not str(reason).strip():
        raise ValueError('a reset needs a reason')
    lp = folder/'ledger.json'
    ledger = read(lp) if lp.exists() else {}
    e = ledger.get(name)
    if e is None or e.get('status') != 'started':
        raise ValueError('only a started, unfinished attempt can be reset (a completed one never is)')
    if confirm_started_unix is None or float(confirm_started_unix) != float(e['started_unix']):
        raise ValueError(f'repeat the ledger entry\'s started_unix to confirm which attempt is meant: {json.dumps(e)}')
    raw = folder/f'{name}_raw.json'
    if raw.exists() or ever_in_git(raw):
        raise ValueError('a raw file exists (or was ever committed) for this evaluation: it is never re-run')
    if name == 'main_orig' and (folder.parent/'result.json').exists():
        raise ValueError('result.json exists: the primary evaluation is never re-run')
    ab = read(aborted_path(folder)) if aborted_path(folder).exists() else []
    done = [x for x in ab if x['name'] == name]
    if done and done[0]['entry'] != e:
        raise ValueError('this evaluation was already reset once')
    if not done:
        ab.append(dict(name=name, entry=e, contract_sha256=e['contract_sha256'], reason=str(reason), reset_unix=time.time()))
        write(aborted_path(folder), ab)
    del ledger[name]
    write(lp, ledger)


# ---------------------------------------------------------------- results
def read_raw(name):
    return read(out_dir()/'test'/f'{name}_raw.json')


def batch_check(specs, raws):
    """Design 2.8: in one setting the same policy has bitwise the same per-seed G in every evaluation that contains it (also twice in one
    evaluation, e.g. A+ and A++ with equal parameters).  specs: name -> (setting, thetas); raws: name -> raw.  Returns the mismatches."""
    seen, bad = {}, []
    for name, raw in raws.items():
        which, thetas = specs[name]
        for t, row in zip(thetas, gains_of(raw)):
            key = (which, theta_id(t))
            if key not in seen:
                seen[key] = (name, row)
            elif not bitwise_equal(row, seen[key][1]):
                bad.append(dict(setting=which, theta=key[1][:12], first=seen[key][0], second=name))
    return bad


def e1v2_D():
    p = ROOT/E1V2_DIR/'result.json'
    if not p.exists():
        return None
    r = read(p)
    return dict(mean=r['mean_D'], LB=r['LB'], UB=r['UB_D'])


def compute_result(m, sel, raw_main):
    g = gains_of(raw_main)
    if g.shape[0] != 2:
        raise ValueError('main_orig must hold [a*, A_fresh_orig]')
    out = decide(g[1]-g[0], m['params']['eps'])
    ref = e1v2_D()
    row = the_row(ROWS_MAIN, out['LB'], out['UB'])
    if TREE_TO_ROW[out['sub'] or out['category']] != row:
        raise ValueError('decision tree and consequence table disagree')
    out.update(schema='ea2-result/1', manifest_sha256=m['sha256'], selection_frozen_sha256=sha(out_dir()/'selection_frozen.json'),
               G_orig_astar=float(g[0].mean()), G_orig_A_fresh_orig=float(g[1].mean()), A_fresh_orig=sel['A_fresh_orig']['name'], A_fresh_orig_gen=sel['A_fresh_orig']['gen'],
               consequence_row=row, e1v2_D=ref,
               ub_over_e1v2_D=None if not ref else out['UB']/ref['mean'], scale_note='UB as a share of the E1 v2 D; a scale only, it does not mean D changes by that much')
    return out


def compute_plus(m, sel, plus, raw_po, raw_pc):
    """Design 2.5, descriptive: Delta+ and Delta++ against a*, and the rows R8-R17 (R8 / R13 depend on `collapsed`, i.e. on the frozen generation)."""
    po, pc = gains_of(raw_po), gains_of(raw_pc)
    if po.shape[0] != 4 or pc.shape[0] != 4:
        raise ValueError('plus_orig must hold [a*, A_fresh_orig, A+, A++]; plus_C2 [a*, A+, A++, E-A1 A_fresh]')
    g = plus['A_plus']['gen']
    if (g == 0) != bool(plus['collapsed']):
        raise ValueError('collapsed flag disagrees with the selected generation')
    d_plus, d_pp = interval(po[2]-po[0]), interval(po[3]-po[0])
    row_p = the_row(ROWS_PLUS, g, d_plus['LB'], d_plus['UB'])
    row_pp = the_row(ROWS_PP, g, d_pp['LB'], d_pp['UB'])
    d_plus.update(row=row_p, text=PLUS_TEXT[row_p].format(**dict(mean=d_plus['mean'], lb=d_plus['LB'], ub=d_plus['UB'])), gen=g, collapsed=bool(plus['collapsed']))
    d_pp.update(row=row_pp, text=PP_TEXT[row_pp].format(**dict(mean=d_pp['mean'], lb=d_pp['LB'], ub=d_pp['UB'])), gen=plus['A_plusplus']['gen'])
    cols = dict(a_star=pc[0], A_plus=pc[1], A_plusplus=pc[2], ea1_A_fresh=pc[3])
    cross = {f'{r}|{c}': interval(gr-gc) for r, gr in (('A_plus', po[2]), ('A_plusplus', po[3])) for c, gc in cols.items()}
    return dict(schema='ea2-plus/1', manifest_sha256=m['sha256'], plus_frozen_sha256=sha(out_dir()/'selection_plus_frozen.json'),
                note='descriptive only; never replaces or corrects the primary result; no row of this file touches delta=0.005',
                delta_plus=d_plus, delta_plusplus=d_pp, delta_plus_minus_delta=interval(po[2]-po[1]), G_orig=dict(a_star=float(po[0].mean()), A_fresh_orig=float(po[1].mean()),
                A_plus=float(po[2].mean()), A_plusplus=float(po[3].mean())), cross_setting=cross, collapsed=bool(plus['collapsed']),
                gen_plus=g, gen_plusplus=plus['A_plusplus']['gen'])


def curve_stats(curve, block=52, last=100):
    f = np.asarray([c['f_mean'] for c in curve], float)
    blocks = []
    for i in range(0, len(f), block):
        x = f[i:i+block]
        blocks.append(dict(gens=[i+1, i+len(x)], mean=float(x.mean()), sem=float(x.std(ddof=1)/math.sqrt(len(x))) if len(x) > 1 else None))
    k = min(last, len(f))
    slope = None
    if k >= 2:
        xs = np.arange(k, dtype=float)
        xs -= xs.mean()
        y = f[-k:]
        slope = float((xs*(y-y.mean())).sum()/(xs*xs).sum())
    return dict(blocks=blocks, last_slope_per_gen=slope, last_n=k)


def chain_summaries(m):
    """Per chain: SEL trajectory, best generation, saturation flag, training-curve blocks and the late slope (descriptive); then the re-adaptation arm."""
    a, out = m['params']['attacker'], []
    for run in range(a['chains']):
        s = chain_state(run)
        traj = [[r['gen'], r['G_SEL']] for r in s['records']]
        out.append(dict(run=run, trajectory=traj, best_gen=best_record(s['records'])['gen'], saturation=saturation_of(traj, a['val_every']),
                        curve_gens=len(s['curve']), curve=curve_stats(s['curve'])))
    s = read(plus_path())
    traj = [[r['gen'], r['G_SEL']] for r in s['records']]
    pc = m['params']['plus']
    return dict(chains=out, plus=dict(trajectory=traj, selected_gen=best_record(s['records'])['gen'], saturation=saturation_of(traj, pc['val_every']),
                                      curve_gens=len(s['curve']), curve=curve_stats(s['curve'])))


def compute_secondary(m, sel, raws):
    """Design 2.3 / 2.9, descriptive only (uncorrected).  raws: main_orig, aux_C2, sec_orig, sec_C2."""
    names = [e['name'] for e in sel['pool']]
    i = {n: j for j, n in enumerate(names)}
    mo, ac, so, sc = (gains_of(raws[k]) for k in ('main_orig', 'aux_C2', 'sec_orig', 'sec_C2'))
    if so.shape[0] != len(names) or sc.shape[0] != len(names) or mo.shape[0] != 2 or ac.shape[0] != 3:
        raise ValueError('evaluation shapes differ from the registered policy lists')
    star = sel['astar']['name']
    out = dict(schema='ea2-secondary/1', manifest_sha256=m['sha256'], selection_frozen_sha256=sha(out_dir()/'selection_frozen.json'),
               note='descriptive only; uncorrected for multiplicity; not part of the decision')
    # D_pool (design 2.3): both sides pick the largest-SEL entry of the same 15-entry pool
    jo, jc = i[sel['A_O_prime']['name']], i[sel['A_C2_prime']['name']]
    dp = interval(so[jo]-sc[jc])
    row = the_row(ROWS_DPOOL, dp['LB'], dp['UB'])
    dp.update(row=row, delta=DELTA, A_O_prime=sel['A_O_prime']['name'], A_C2_prime=sel['A_C2_prime']['name'], text=D_POOL_TEXT[row].format(mean=dp['mean'], lb=dp['LB'], ub=dp['UB']),
              note='descriptive; not the E1 v2 decision tree; no honest-service gate; never labelled holds / insufficient')
    out['D_pool'] = dp
    # 2x2: rows = G_orig end {a*, A_fresh_orig} (main_orig), columns = G_C2 end {a*, E-A1 A_fresh} (aux_C2); no row of the table is triggered
    cells = {f'{r}|{c}': interval(mo[ri]-ac[ci]) for ri, r in enumerate(('a_star', 'A_fresh_orig')) for ci, c in ((0, 'a_star'), (2, 'ea1_A_fresh'))}
    out['two_by_two'] = dict(cells=cells, note='descriptive; no cell is compared with delta')
    out['D_sym'] = dict(cells['A_fresh_orig|ea1_A_fresh'], note='two single attackers, same recipe, each cold-started on its own setting; does not lift the warm-start limitation')
    out['D_new'] = dict(cells['a_star|a_star'], cross_machine_reference_only=True, e1v2_D=e1v2_D())
    try:
        out['D_new']['ea1_D_new'] = {k: read(ROOT/EA1_DIR/'secondary.json')['D_new'][k] for k in ('mean', 'LB', 'UB')}
    except (OSError, KeyError):
        pass
    # 2.9 item 1 and 2
    out['chains_vs_astar_orig'] = {c['name']: interval(so[i[c['name']]]-so[i[star]]) for c in sel['chains']}
    frozen = [n for n in names if n.startswith('frozen_run')]
    out['new_vs_old_orig'] = {a_: {f: interval(so[i[a_]]-so[i[f]]) for f in frozen} for a_ in [c['name'] for c in sel['chains']]}
    out['new_vs_old_orig']['A_fresh_orig'] = {f: interval(so[i[sel['A_fresh_orig']['name']]]-so[i[f]]) for f in frozen}
    out['ea1_checkpoints_vs_astar_orig'] = {n: interval(so[i[n]]-so[i[star]]) for n in names if n.startswith('ea1_fresh_run')}
    # item 4: every pool entry in both settings
    out['by_entry'] = {n: dict(G_orig=interval(so[j]), G_C2=interval(sc[j]), diff_mean=float((so[j]-sc[j]).mean()), aliases=sel['pool'][j]['aliases']) for n, j in i.items()}
    # item 5: honest service
    ra, rb = (v1.metrics(raws[k]) for k in ('sec_C2', 'sec_orig'))
    out['honest_service'] = dict(delta_regret=interval(ra['regret']-rb['regret']), delta_alpha=interval(ra['alpha']-rb['alpha']))
    out['chain_summaries'] = chain_summaries(m)
    p = out_dir()/'drift.json'
    out['drift'] = {k: read(p)[k] for k in ('orig', 'C2')} if p.exists() else None
    return out


def consequences(result, secondary, plus):
    """The four rows of the full outcome and the summary sentences they require (design 2.4)."""
    rows = dict(main=result['consequence_row'], d_pool=secondary['D_pool']['row'], plus=plus['delta_plus']['row'], plusplus=plus['delta_plusplus']['row'])
    if sorted(rows) != sorted(GROUPS):
        raise ValueError('one row per group is required')
    return dict(rows=rows, summary_additions=summary_additions(rows))


def load_raws(names):
    return {n: read_raw(n) for n in names if (out_dir()/'test'/f'{n}_raw.json').exists()}


def write_same(path, content):
    """Write once; a re-run may only reproduce the identical content (a committed result is never silently replaced)."""
    path = Path(path)
    if path.exists():
        if read(path) != json.loads(json.dumps(content)):
            raise ValueError(f'{path.name} exists and differs from the recomputed content; investigate')
        return
    write(path, content)


def run_test(part):
    if part not in TEST_PARTS:
        raise ValueError('test part must be one of '+', '.join(TEST_PARTS))
    m = load_manifest()
    sel = verify_freeze(m)
    plus = verify_plus_freeze(m, sel)
    d = out_dir()
    require_testcheck(m, sel, plus)           # before the ledger and before test seeds are touched
    if part != 'main':
        require_result_committed(m)
    sd = m['seeds']['EA2_TEST']
    require_seeds(sd, 'test', m, unlocked=True)
    if len(sd) != m['params']['n_test']:
        raise ValueError('test seed count changed')
    cfgs, freeze_sha, plus_sha = test_cfgs(m), sha(d/'freeze_record.json'), sha(d/'selection_plus_frozen.json')
    plan_ = {n: (w, ts, pols) for n, w, ts, pols in test_pols(sel, plus)}
    specs = {n: (w, ts) for n, (w, ts, _) in plan_.items()}

    def go(name):
        which, thetas, pols = plan_[name]
        con = dict(manifest_sha256=m['sha256'], freeze_record_sha256=freeze_sha, stage=name, cfg=cfgs[which], seeds=list(sd), n_policies=len(pols),
                   T_score=m['params']['T_val']-m['params']['exclude'], T=m['params']['T_val'], exclude=m['params']['exclude'], theta_sha256=[theta_id(t) for t in thetas])
        if name.startswith('plus_'):
            con['plus_frozen_sha256'] = plus_sha
        raw = ledger_once(d/'test', name, con, lambda: simulate_chunked(m, cfgs[which], pols, sd, m['params']['test_chunk']))
        stamp('test', evaluation=name)
        return raw

    def check(names):
        bad = batch_check(specs, load_raws(names))
        write(d/f'batch_independence_{part}.json', dict(part=part, evaluations=list(load_raws(names)), mismatches=bad))
        if bad:
            raise BatchIndependenceError(f'per-seed G of the same policy differs between evaluations {bad}: the primary result stands; STOP and ask the user')
    if part == 'main':
        go('main_orig')
        res = compute_result(m, sel, read_raw('main_orig'))
        load_manifest(clean=False)
        write_same(d/'result.json', res)
        print(json.dumps({k: res[k] for k in ('category', 'sub', 'label', 'mean_delta', 'LB', 'UB', 'consequence_row')}), flush=True)
        return res
    if not (d/'test'/'main_orig_raw.json').exists():
        raise ValueError('main_orig has not been evaluated')
    if part == 'plus':
        go('plus_orig')
        go('plus_C2')
        check(['main_orig', 'plus_orig', 'plus_C2'])
        res = compute_plus(m, sel, plus, read_raw('plus_orig'), read_raw('plus_C2'))
        write_same(d/'plus.json', res)
        return res
    for name in ('aux_C2', 'sec_orig', 'sec_C2'):
        go(name)
    check(list(plan_))
    sec = compute_secondary(m, sel, {n: read_raw(n) for n in ('main_orig', 'aux_C2', 'sec_orig', 'sec_C2')})
    write_same(d/'secondary.json', sec)
    return sec


def stage_report(tag):
    """CPU only: recompute result, plus and secondary from the raw files; compare with the saved ones.  The output file is named by --tag, never by a host."""
    if tag not in ('shared', 'local'):
        raise ValueError("--tag must be 'shared' or 'local'")
    m = load_manifest(clean=False)
    d = out_dir()
    sel, plus = read(d/'selection_frozen.json'), read(d/'selection_plus_frozen.json')
    raws = {n: read_raw(n) for n in ('main_orig', 'plus_orig', 'plus_C2', 'aux_C2', 'sec_orig', 'sec_C2')}
    res = compute_result(m, sel, raws['main_orig'])
    pl = compute_plus(m, sel, plus, raws['plus_orig'], raws['plus_C2'])
    sec = compute_secondary(m, sel, raws)
    specs = {n: (w, ts) for n, w, ts, _ in test_pols(sel, plus)}
    bad = batch_check(specs, raws)
    same = dict(result=read(d/'result.json') == json.loads(json.dumps(res)), plus=read(d/'plus.json') == json.loads(json.dumps(pl)),
                secondary=read(d/'secondary.json') == json.loads(json.dumps(sec)), batch_independent=not bad)
    cons = consequences(res, sec, pl)
    print(json.dumps({k: res[k] for k in ('category', 'sub', 'label', 'mean_delta', 'LB', 'UB')}))
    print('consequence rows:', json.dumps(cons['rows']), '| matches saved:', json.dumps(same), '| numpy', np.__version__, 'scipy', scipy.__version__)
    write(d/f'report_check_{tag}.json', dict(tag=tag, matches=all(same.values()), parts=same, numpy=np.__version__, scipy=scipy.__version__, result=res, consequences=cons))
    return 0 if all(same.values()) else 1


# ---------------------------------------------------------------- smoke (GPU)
def trace_chain(m, run, be, N, validate=True):
    """N generations plus one validation (EA2_SEL, T_val) of a fresh main chain: everything the design calls 'the result'."""
    cfg = m['params']['base_cfg']
    opt, gens, walls = make_opt(m, run), [], []
    for _ in range(N):
        t = time.time()
        rec = chain_generation(m, cfg, run, opt, be)
        walls.append(time.time()-t)
        gens.append(dict(X=rec['X'], U=rec['U'], f=rec['f'], seeds=rec['seeds']))
    return finish_trace(m, cfg, opt, gens, walls, be, validate)


def trace_plus(m, theta, be, N, validate=True):
    """The same for the re-adaptation setting (lam 64, n_S 32), starting from `theta` (smoke: a*)."""
    cfg = m['params']['base_cfg']
    opt, gens, walls = make_plus_opt(m, theta), [], []
    for _ in range(N):
        t = time.time()
        rec = plus_generation(m, cfg, opt, be)
        walls.append(time.time()-t)
        gens.append(dict(X=rec['X'], U=rec['U'], f=rec['f'], seeds=rec['seeds']))
    return finish_trace(m, cfg, opt, gens, walls, be, validate)


def finish_trace(m, cfg, opt, gens, walls, be, validate):
    if not validate:
        return dict(gens=gens, wall_gen=walls)
    t = time.time()
    raw = be.simulate(cfg, [honest_pol(), mlp(opt.mean)], m['seeds']['EA2_SEL'], m['params']['T_val'], m['params']['exclude'])
    wall_val = time.time()-t
    return dict(gens=gens, opt=opt.state_dict(), val=digest(raw['snap']), G_val=float(v1.metrics(dict(raw, contract=None))['gains'][0].mean()), wall_gen=walls, wall_val=wall_val)


trace_diff = ea1.trace_diff


def mean_after_first(xs):
    return float(np.mean(xs[1:])) if len(xs) > 1 else None


def run_smoke(N, N_plus=5, dev=None):
    """Before the official freeze: single-lane formal-size timing; eager (A) vs graph (B) bitwise for the 3 main chains (N generations) and the re-adaptation
    setting (N_plus generations, started from a*); one validation each; the six test shapes on EA2_SEL seeds.  Never EA2_TEST."""
    import torch
    dev = dev or MODE['device']
    if dev.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('smoke needs CUDA')
    freeze(smoke=True)
    m = load_manifest(clean=False)
    check_environment(m)
    inp = e1v2_entries()
    star = inp['astar']['theta']
    meters, acc, t_gen, t_val = {}, {}, [], []
    with Meter('main_chains', meters):
        for r in range(m['params']['attacker']['chains']):
            A = trace_chain(m, r, Backend(dev, None, None), N)
            be = make_backend(dev)
            with be:
                B = trace_chain(m, r, be, N)
            be.release()
            acc[f'chain{r}'] = trace_diff(A, B)
            t_gen.append(mean_after_first(B['wall_gen']))
            t_val.append(B['wall_val'])
    with Meter('plus', meters):
        A = trace_plus(m, star, Backend(dev, None, None), N_plus)
        be = make_backend(dev)
        with be:
            B = trace_plus(m, star, be, N_plus)
        be.release()
    acc['plus'] = trace_diff(A, B)
    t_pgen = mean_after_first(B['wall_gen'])
    sel_seeds, t_test = m['seeds']['EA2_SEL'], {}
    ents = [t for _, t in inp['entries']]
    shapes = dict(main_orig=2, plus_orig=4, plus_C2=4, aux_C2=3, sec_orig=EXPECT_POOL, sec_C2=EXPECT_POOL)
    for name, k in shapes.items():
        pols = [honest_pol()]+[mlp(ents[j % len(ents)]) for j in range(k)]
        cfg = m['params']['c2_cfg'] if name.endswith('C2') else m['params']['base_cfg']
        t0 = time.time()
        simulate_chunked(m, cfg, pols, sel_seeds, m['params']['test_chunk'])
        w = time.time()-t0
        t_test[name] = dict(wall_s=w, extrapolated_n_test_s=w*PARAMS['n_test']/len(sel_seeds))
    a, p = m['params']['attacker'], m['params']['plus']
    tg, tv = float(np.mean(t_gen)), float(np.mean(t_val))
    summary = dict(schema='ea2-smoke-summary/1', code_sha256=code_hashes(), n_gens=N, n_gens_plus=N_plus, acceptance=acc, graph_equals_eager=not any(acc.values()), t_gen_s=tg, t_val_s=tv, t_plus_gen_s=t_pgen,
                   meters=meters, t_test=t_test, environment=environment(), hours_train=a['chains']*(a['gens']*tg+(a['gens']//a['val_every'])*tv)/3600,
                   hours_plus=(p['gens']*t_pgen+(p['gens']//p['val_every']+1)*tv)/3600, hours_test=sum(v['extrapolated_n_test_s'] for v in t_test.values())/3600)
    write(out_dir()/'smoke_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('graph_equals_eager', 't_gen_s', 't_val_s', 't_plus_gen_s', 'hours_train', 'hours_plus', 'hours_test')}, indent=1))
    if not summary['graph_equals_eager']:
        raise RuntimeError('smoke: graph != eager; STOP and ask the user (do not change path or budget)')
    return summary


def smoke_clean(dest=None):
    """Keep the smoke summary (the numbers go into the implementation notes), remove the smoke directory."""
    src, dst = ROOT/SMOKE_DIR/'smoke_summary.json', Path(dest) if dest else ROOT/OUT_DIR/'smoke_summary.json'
    if src.exists():
        write(dst, read(src))
    shutil.rmtree(ROOT/SMOKE_DIR, ignore_errors=True)
    print('smoke directory removed; summary kept at', dst)


# ---------------------------------------------------------------- CLI
def status():
    d = out_dir()
    m = load_manifest(clean=False)
    a = m['params']['attacker']
    chains = {r: chain_state(r)['opt']['gen'] if chain_path(r).exists() else 0 for r in range(a['chains'])}
    print(json.dumps(dict(smoke=m['smoke'], manifest_sha256=m['sha256'], chain_generations=chains, of=a['gens'],
                          plus_generation=read(plus_path())['opt']['gen'] if plus_path().exists() else 0, plus_of=m['params']['plus']['gens'],
                          stages_done=sorted(stages_done()), status=read(d/'status.json') if (d/'status.json').exists() else None), indent=1))


def _no_ea1_out_dir():
    raise RuntimeError('E-A2 must never resolve an E-A1 output directory (ea1.out_dir is disabled in this process)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('cmd', choices=['audit', 'smoke', 'smoke-clean', 'freeze', 'train', 'select', 'adapt', 'testcheck', 'test', 'report', 'status', 'ledger-reset'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gens', type=int, default=3, help='smoke: generations per main chain (>= 3)')
    ap.add_argument('--plus-gens', type=int, default=5, help='smoke: generations of the re-adaptation setting (>= 3)')
    ap.add_argument('--part', choices=TEST_PARTS, default=None, help='test: main (primary result) | plus (A+/A++) | rest (D_pool, 2x2, secondary)')
    ap.add_argument('--tag', choices=['shared', 'local'], default=None, help='report: names the output file')
    ap.add_argument('--eval', dest='eval_name', default=None, help='ledger-reset: the started evaluation')
    ap.add_argument('--reason', default=None)
    ap.add_argument('--confirm-started-unix', type=float, default=None, help='ledger-reset: the started_unix of the ledger entry (shown by status / the ledger file)')
    ap.add_argument('--user-authorized', action='store_true', help='ledger-reset: the user authorised one re-run of this unfinished evaluation')
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    strict = a.cmd in ('test', 'testcheck')
    MODE.update(smoke=a.smoke or a.cmd == 'smoke', device=a.device, graph=a.device.startswith('cuda'), strict=strict,
                yield_check=a.cmd in ('train', 'adapt') and a.device.startswith('cuda'))
    if strict and a.smoke:
        sys.exit('test/testcheck never run in smoke mode')
    if a.cmd == 'test' and a.part is None:
        sys.exit('test needs --part main|plus|rest')
    if a.cmd == 'report' and a.tag is None:
        sys.exit('report needs --tag shared|local')
    if not MODE['smoke'] and a.cmd not in ('report', 'status', 'audit', 'freeze', 'smoke-clean', 'ledger-reset') and a.device != 'cuda':
        sys.exit(f'{a.cmd}: the official manifest runs only with --device cuda (got {a.device!r}); refusing')
    (ROOT/'.scratch').mkdir(exist_ok=True)
    saved_out_dir, ea1.out_dir = ea1.out_dir, _no_ea1_out_dir            # in this process only; restored below
    try:
        if a.cmd == 'audit':
            rep = audit(a.smoke)[0]
            print(json.dumps(rep, indent=1))
            return 0
        if a.cmd == 'smoke-clean':
            smoke_clean()
            return 0
        if a.cmd == 'report':
            return stage_report(a.tag)
        if a.cmd == 'status':
            status()
            return 0
        if a.cmd == 'ledger-reset' and (not a.eval_name or not a.reason):
            sys.exit('ledger-reset needs --eval and --reason')
        with (ROOT/'.scratch/ea2.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)         # every writing stage, ledger-reset included, holds the one lock
            if a.cmd == 'ledger-reset':
                ledger_reset(out_dir()/'test', a.eval_name, a.reason, authorized=a.user_authorized, confirm_started_unix=a.confirm_started_unix)
                print('ledger entry archived in ledger_aborted.json (commit it before the evaluation is run again):', a.eval_name)
                return 0
            if a.cmd == 'freeze':
                freeze(a.smoke)
                return 0
            import torch
            torch.set_num_threads(1)
            MODE['start'] = time.time()
            try:
                if a.cmd == 'smoke':
                    run_smoke(max(a.gens, 3), max(a.plus_gens, 3))
                    return 0
                m = load_manifest(clean=not a.smoke)
                if MODE['device'].startswith('cuda'):
                    check_environment(m)
                if a.cmd == 'train':
                    stamp('running', started_unix=time.time(), manifest_sha256=m['sha256'])
                    if 'train' not in stages_done():
                        stage_train(m)
                    stamp('complete', finished_unix=time.time())
                elif a.cmd == 'select':
                    stage_select(m)
                elif a.cmd == 'adapt':
                    stamp('running', started_unix=time.time(), manifest_sha256=m['sha256'])
                    stage_adapt(m)
                    stamp('complete', finished_unix=time.time())
                elif a.cmd == 'testcheck':
                    run_testcheck()
                elif a.cmd == 'test':
                    run_test(a.part)
            except (BudgetStop, YieldStop) as exc:
                stamp('budget_stop' if isinstance(exc, BudgetStop) else 'yield_stop', message='saved at a generation boundary; relaunch to resume')
                return 75
            except Exception as exc:
                stamp('failed', error_type=type(exc).__name__, message=str(exc)[:300])
                raise
        return 0
    finally:
        ea1.out_dir = saved_out_dir


if __name__ == '__main__':
    sys.exit(main())
