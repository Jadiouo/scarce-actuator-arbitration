"""E-A2: seeds, guards, statistics, decision tree, consequence table (R1-R17), selections (A_fresh_orig, A_O'/A_C2', A+/A++), ledger, hash pins,
select -> adapt -> testcheck -> test(main/plus/rest) -> report flow, launcher.
CPU only (tiny synthetic budgets; no EA2_TEST seed is ever simulated except by the fake simulator inside the synthetic flow).
GPU tests (marker gpu) run on the shared experiment machine only: EA2_ALLOW_GPU_TESTS=1."""
import ast
import copy
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import threading

import numpy as np
import pytest

from arbitration.rl import train as rl_train
from scripts import e1_v2 as e1
from scripts import ea1
from scripts import ea2 as ea

import torch
torch.set_num_threads(1)         # tiny CPU simulations: more intra-op threads only add contention (the production code does the same)

EPS, DELTA = .001, .005


def cuda_required():
    """GPU tests run only on the experiment machine: skipped unless CUDA is present AND EA2_ALLOW_GPU_TESTS=1 (the local GPU is never used)."""
    if os.environ.get('EA2_ALLOW_GPU_TESTS') != '1':
        pytest.skip('GPU tests run only on the experiment machine (EA2_ALLOW_GPU_TESTS=1)')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@pytest.fixture(scope='module', autouse=True)
def _ea1_results_untouched():
    """Design 4 'no write into E-A1': the E-A1 result directory and the pinned E-A1 files are byte-identical before and after this module."""
    before = (tree_hash(ea.ROOT/ea.EA1_DIR), {p: ea.sha(ea.ROOT/p) for p in ea.EA1_REFS})
    yield
    assert (tree_hash(ea.ROOT/ea.EA1_DIR), {p: ea.sha(ea.ROOT/p) for p in ea.EA1_REFS}) == before


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ea.MODE.update(smoke=False, device='cuda', graph=True, out=None, start=None, strict=False, yield_check=False, K=None)
    ea.GEN_HOOK = None
    ea1._DEFAULT.clear()
    ea1._YIELD['event'].clear()
    ea1._YIELD['last'] = 0.0


# ------------------------------------------------------------------ tiny synthetic manifest (CPU, a few seconds)
def tiny(tmp_path, gens=6, val_every=3, lam=6, n_S=4, plus_gens=4, plus_val=2):
    p = json.loads(json.dumps(ea.PARAMS))
    p['attacker'].update(lam=lam, n_S=n_S, gens=gens, val_every=val_every)
    p['plus'].update(lam=6, n_S=2, gens=plus_gens, val_every=plus_val, pp_min_gen=plus_val)
    p.update(T_train=16, T_val=16, exclude=4, c2_cfg=dict(p=.5, tol=1., kref=.5, L=8.), base_cfg=dict(p=.5, tol=2., kref=.5, L=8.), test_chunk=4, testcheck_n=4, n_test=8)
    seeds = dict(EA2_TRAIN=list(range(92_000_000, 92_000_016)), EA2_SEL=list(range(92_100_000, 92_100_008)), EA2_AT=list(range(92_200_000, 92_200_008)),
                 EA2_TEST=list(range(92_400_000, 92_400_008)))
    m = dict(schema='t', smoke=False, params=p, seeds=seeds, drift=dict(seeds=list(range(72_300_000, 72_300_004))), sha256='synthetic', environment={})
    ea.MODE.update(smoke=False, out=str(tmp_path), device='cpu', graph=True, start=None, yield_check=False, K=4)
    return m


def eager():
    return ea.Backend('cpu', None, None)


def lane():
    return ea.make_backend('cpu')


class Collect:
    """GEN_HOOK that records every generation's candidate matrix, utilities, fitness and seed block as raw bytes."""
    def __init__(self, stop=None):
        self.rec, self.stop = {}, stop

    def __call__(self, kind, run, gen, rec):
        self.rec[(kind, run, gen)] = (rec['X'].tobytes(), rec['U'].tobytes(), rec['f'].tobytes(), tuple(rec['seeds']))
        if self.stop and self.stop(kind, run, gen):
            raise ea.YieldStop()


def vec(mean, sd, n, seed=0):
    z = np.random.default_rng(seed).standard_normal(n)
    return mean+sd*(z-z.mean())/z.std(ddof=1)


N = 2048


# ------------------------------------------------------------------ seed audit
def test_seed_audit_counts_collisions_and_registered_hashes():
    rep = ea.seed_report(ea.block_seeds(), ea.existing_all())
    assert (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) == (13728, 20672, 0)
    assert rep['per_block'] == dict(EA2_TRAIN=4000, EA2_SEL=512, EA2_AT=1024, EA2_TEST=8192)
    assert rep['new_expanded_sha256'] == ea.NEW_SHA == '1940a8a5e9cd75cefe771b8d29fe2e007aded1e3b12eab5d52a375126d29ec75'
    assert rep['existing_expanded_sha256'] == ea.OLD_SHA == '5de9035d2137a8ddb5f7712fb47ac2ca57635ebdb84d74568306b52f8551ffe0'
    assert (ea.EXPECT_NEW, ea.EXPECT_OLD) == (13728, 20672)
    assert 'EA2_TEST' not in ea.block_seeds(smoke=True) and set(ea.block_seeds()) == {'EA2_TRAIN', 'EA2_SEL', 'EA2_AT', 'EA2_TEST'}
    assert ea.BLOCKS == dict(EA2_TRAIN=(92_000_000, 1000), EA2_SEL=(92_100_000, 128), EA2_AT=(92_200_000, 256), EA2_TEST=(92_400_000, 2048))
    allseeds = ea.expand(sum(ea.block_seeds().values(), []))
    assert min(allseeds) == 92_000_000 and max(allseeds) == 95_402_047
    ex = ea.existing_all()
    assert {'ea1_EA1_TRAIN', 'ea1_EA1_SEL', 'ea1_EA1_TEST', 'e1v2_TEST', 'e1v2_AS'} <= set(ex) and max(max(v) for v in ex.values()) == 82_401_023
    for bad in (82_400_000, 82_100_127, 82_000_000, 72_300_000, 41_000_063, 72_400_000, 61_000_000-3*10**6, 92_000_000-3*10**6):
        b = ea.block_seeds()
        b['EA2_SEL'] = b['EA2_SEL'][:-1]+[bad]
        with pytest.raises(ValueError):
            ea.seed_report(b, ea.existing_all())                                   # E-A1 blocks, E1 v2 blocks, holdout, +3e6 stream collisions
    for src, dst in (('EA2_TRAIN', 'EA2_SEL'), ('EA2_AT', 'EA2_SEL'), ('EA2_TEST', 'EA2_AT'), ('EA2_SEL', 'EA2_TEST')):
        b = ea.block_seeds()
        b[dst] = b[dst][:-1]+[b[src][0]]
        with pytest.raises(ValueError):
            ea.seed_report(b, ea.existing_all())                                   # cross-block overlap
    b = ea.block_seeds()
    b['EA2_SEL'] = b['EA2_SEL'][:-1]+[b['EA2_TRAIN'][0]+3*10**6]
    with pytest.raises(ValueError):
        ea.seed_report(b, ea.existing_all())                                       # own block, +3e6 expansion
    b = ea.block_seeds()
    b['EA2_TEST'] = b['EA2_TEST'][:-1]+[b['EA2_TEST'][0]]
    with pytest.raises(ValueError):
        ea.seed_report(b, ea.existing_all())                                       # duplicate inside a block
    with pytest.raises(ValueError):
        ea.seed_report(ea.block_seeds(), ea.existing_all(), extra_historical=[92_100_005])
    assert not set(ea.expand(ea.block_seeds()['EA2_SEL'])) & set(ea.expand(range(72_300_000, 72_300_128)))


def test_official_audit_reproduces_the_registered_hashes_and_refuses_otherwise(monkeypatch):
    rep, hist, srcs = ea.audit(False)
    assert rep['new_expanded_sha256'] == ea.NEW_SHA and rep['existing_expanded_sha256'] == ea.OLD_SHA and rep['intersection'] == 0
    assert any(k.startswith('results/direction2/ea1/') for k in srcs) and not any(k.startswith('results/direction2/ea2') for k in srcs)   # E-A1 scanned, E-A2 excluded
    monkeypatch.setattr(ea, 'NEW_SHA', '0'*64)
    with pytest.raises(ValueError, match='hashes'):
        ea.audit(False)
    monkeypatch.setattr(ea, 'EXPECT_NEW', 1)
    with pytest.raises(ValueError, match='counts'):
        ea.audit(False)


# ------------------------------------------------------------------ stage / seed mismatch
def test_stages_only_accept_their_own_block_test_locked_testcheck_is_first_64_of_sel_at_only_for_training():
    m = dict(seeds=ea.block_seeds(), params=dict(testcheck_n=64), drift=dict(seeds=list(range(72_300_000, 72_300_128))))
    S = m['seeds']
    ea.require_seeds(S['EA2_TRAIN'][:256], 'train', m)
    ea.require_seeds(S['EA2_SEL'], 'sel', m)
    ea.require_seeds(S['EA2_AT'][:32], 'at', m)
    ea.require_seeds(S['EA2_SEL'][:64], 'testcheck', m)
    ea.require_seeds(S['EA2_TEST'], 'test', m, unlocked=True)
    ea.require_seeds(m['drift']['seeds'], 'drift', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_TEST'][:4], 'test', m)                                         # locked
    for stage, own in (('train', 'EA2_TRAIN'), ('sel', 'EA2_SEL'), ('at', 'EA2_AT'), ('test', 'EA2_TEST')):
        for other in S:
            if other != own:
                with pytest.raises(ValueError):
                    ea.require_seeds(S[other][:4], stage, m, unlocked=True)                    # EA2_AT trains the re-adaptation arm and nothing else
    for bad in ([0], [2000], [41_000_000], [72_300_000], [82_400_000], [92_400_000-1]):
        for stage in ('train', 'sel', 'at', 'testcheck'):
            with pytest.raises(ValueError):
                ea.require_seeds(bad, stage, m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_TRAIN'][:2]*2, 'train', m)                                     # duplicates
    with pytest.raises(ValueError):
        ea.require_seeds([], 'train', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_SEL'][:63]+[S['EA2_TEST'][0]], 'testcheck', m)                  # any TEST seed in testcheck fails
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_SEL'][:63], 'testcheck', m)                                    # must be exactly the first 64
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_SEL'][1:65], 'testcheck', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA2_SEL'][:8], 'drift', m)
    smoke = dict(seeds=ea.block_seeds(smoke=True), params=dict(testcheck_n=64))
    with pytest.raises(ValueError):
        ea.require_seeds([92_400_000], 'test', smoke, unlocked=True)                           # smoke manifests have no test block
    with pytest.raises(ValueError):
        ea.require_seeds([1], 'nonsense', m)


def test_evaluate_and_training_refuse_a_wrong_stage_block(tmp_path):
    m = tiny(tmp_path)
    cfg = m['params']['base_cfg']
    pols = [ea.honest_pol(), ea.mlp(np.zeros(55))]
    for stage, sd in (('sel', m['seeds']['EA2_TEST']), ('sel', m['seeds']['EA2_TRAIN'][:4]), ('sel', m['seeds']['EA2_AT'][:4]), ('train', m['seeds']['EA2_SEL'][:4]),
                      ('testcheck', m['seeds']['EA2_SEL'][:3]), ('at', m['seeds']['EA2_SEL'][:2])):
        with pytest.raises(ValueError):
            ea.evaluate(m, cfg, sd, stage, pols, ['honest', 'x'])
    with pytest.raises(ValueError):
        ea.evaluate(m, cfg, m['seeds']['EA2_TEST'][:4], 'test', pols, ['honest', 'x'])         # test seeds are locked in evaluate()
    assert not (tmp_path/'evaluations').exists()                                               # nothing was simulated or cached


# ------------------------------------------------------------------ permutation / optimizer seeds / training blocks
def test_perm_block_rules_and_seed_independence():
    pool = list(range(92_000_000, 92_001_000))
    for n in (1, 16, 256, 999, 1000):
        for g in range(8):
            blk = ea.train_block(pool, 0, g, n)
            assert len(blk) == n == len(set(blk)) and set(blk) <= set(pool) and blk == ea.train_block(pool, 0, g, n)
    with pytest.raises(ValueError):
        ea.train_block(pool, 0, 0, 1001)
    for run in range(3):
        expected = list(range(1000))
        random.Random(95100+run).shuffle(expected)
        assert list(ea.perm_for(1000, 95100+run)) == expected                                  # the registered permutation seed 95100+run
        for old_seed in (94100+run, 9000+run):                                                # not E-A1's, not the original chain's order
            old = list(range(1000))
            random.Random(old_seed).shuffle(old)
            assert expected != old
    idx = list(range(1000))
    for run in range(3):                                                                       # same index rule as train.seed_block
        for g in (0, 3, 5):
            assert ea.train_block(idx, run, g, 256, perm_seed=9000) == list(rl_train.seed_block(run, g, 256))
    a, pl = ea.PARAMS['attacker'], ea.PARAMS['plus']
    assert (a['opt_seed'], a['perm_seed'], a['lam'], a['n_S'], a['gens'], a['sigma0'], a['val_every'], a['chains']) == (95000, 95100, 256, 256, 416, .3, 50, 3)
    assert (pl['opt_seed'], pl['lam'], pl['n_S'], pl['sigma0'], pl['gens'], pl['val_every'], pl['key'], pl['pp_min_gen']) == (96000, 64, 32, .15, 240, 20, 0, 20)
    assert {94000, 94100, 9000, 91000, 91001, 92000, 92001, 93000}.isdisjoint({a['opt_seed']+r for r in range(3)} | {a['perm_seed']+r for r in range(3)} | {pl['opt_seed']})
    reg = ea.PARAMS
    assert (reg['T_train'], reg['T_val'], reg['exclude'], reg['eps'], reg['delta'], reg['n_test'], reg['lanes']) == (20000, 100000, 5000, .001, .005, 2048, 1)
    assert len(range(0, pl['gens']+1, pl['val_every'])) == 13 and len(range(a['val_every'], a['gens']+1, a['val_every'])) == 8      # 13 re-adaptation points, 8 chain points
    assert reg['base_cfg'] == ea.BASE_CFG and reg['c2_cfg'] == dict(p=.1, tol=6.175461735115074, kref=.4663589529547838, L=2412.0)


# ------------------------------------------------------------------ cold-start chains on the ORIGINAL setting
def test_cold_start_chain_runs_on_the_original_setting_budget_comes_from_manifest_and_resume_checks_it(tmp_path):
    m = tiny(tmp_path)
    assert not np.any(ea.make_opt(m, 0).mean) and ea.make_opt(m, 2).lam == 6
    b = ea.chain_budget(m, 1)
    assert b['opt_seed'] == 95001 and b['perm_seed'] == 95101 and b['start'] == 'zeros' and b['exclude'] == 4 and b['gens'] == 6 and b['val_every'] == 3
    ea.run_chain(m, 0, eager(), target=2)
    st = ea.chain_state(0)
    assert st['opt']['gen'] == 2 and len(st['curve']) == 2 and st['cfg'] == m['params']['base_cfg'] != m['params']['c2_cfg']       # original setting, not C2
    for edit in (lambda q: q['params']['attacker'].update(n_S=2), lambda q: q['params']['base_cfg'].update(L=9.), lambda q: q['params'].update(T_val=20),
                 lambda q: q['params']['attacker'].update(opt_seed=1), lambda q: q['params']['base_cfg'].update(tol=3.)):
        bad = copy.deepcopy(m)
        edit(bad)
        with pytest.raises(ValueError):
            ea.run_chain(bad, 0, eager(), target=3)                                            # resumed under another defender / budget
    with pytest.raises(ValueError):
        ea.run_chain(m, 0, eager(), target=7)                                                  # never extended beyond the registered generations
    ea.run_chain(m, 0, eager())
    assert ea.chain_state(0)['opt']['gen'] == 6 and [r['gen'] for r in ea.chain_state(0)['records']] == [3, 6]
    assert not np.array_equal(ea.chain_state(0)['opt']['mean'], np.zeros(55))


def test_training_uses_only_train_seeds_validation_only_sel_seeds_and_resume_is_bitwise_identical(tmp_path, monkeypatch):
    runs = [0, 1, 2]
    ref_hook = Collect()
    monkeypatch.setattr(ea, 'GEN_HOOK', ref_hook)
    m = tiny(tmp_path/'ref')
    ea.run_chains(m, runs, eager)
    ref = {r: ea.chain_result(ea.chain_state(r)) for r in runs}
    for r in runs:
        for g in range(6):
            assert ref_hook.rec[('chain', r, g)][3] == tuple(ea.train_block(m['seeds']['EA2_TRAIN'], r, g, 4, 95100))
    assert len({ref_hook.rec[('chain', r, 0)][3] for r in runs}) == 3 and len({ref_hook.rec[('chain', r, 0)][0] for r in runs}) == 3
    assert not any(set(v[3]) & set(m['seeds']['EA2_SEL']+m['seeds']['EA2_TEST']+m['seeds']['EA2_AT']) for v in ref_hook.rec.values())
    # interrupted after generations 1 / 3 / 5 (like a yield to another account), then resumed on a fresh lane
    m2 = tiny(tmp_path/'res')
    monkeypatch.setattr(ea, 'GEN_HOOK', Collect(stop=lambda k, r, g: g == 1+2*r))
    with pytest.raises(ea.YieldStop):
        ea.run_chains(m2, [0], eager)
    with pytest.raises(ea.YieldStop):
        ea.run_chains(m2, [1], eager)
    with pytest.raises(ea.YieldStop):
        ea.run_chains(m2, [2], eager)
    assert [ea.chain_state(r)['opt']['gen'] for r in runs] == [2, 4, 6]
    monkeypatch.setattr(ea, 'GEN_HOOK', None)
    ea.run_chains(m2, runs, lane)
    assert ea.digest({r: ea.chain_result(ea.chain_state(r)) for r in runs}) == ea.digest(ref)
    ea.chains_complete(m2)


def test_chains_complete_refuses_incomplete_or_nonfinite_chains(tmp_path):
    m = tiny(tmp_path)
    ea.run_chain(m, 0, eager())
    ea.run_chain(m, 1, eager(), target=4)
    with pytest.raises(ValueError):
        ea.chains_complete(m)                                                                  # chain 1 stopped at 4, chain 2 never started
    bad = ea.chain_state(0)
    bad['records'][0]['G_SEL'] = float('nan')
    with pytest.raises(ValueError):
        ea.write(ea.chain_path(0), bad)                                                        # non-finite values cannot even be written (allow_nan=False)


# ------------------------------------------------------------------ re-adaptation chain
def start_theta(seed=7):
    return list(np.random.default_rng(seed).normal(0, .02, 55))


def test_plus_chain_starts_from_the_frozen_theta_uses_at_for_training_sel_for_validation_and_resumes_bitwise(tmp_path, monkeypatch):
    th = start_theta()
    hook = Collect()
    monkeypatch.setattr(ea, 'GEN_HOOK', hook)
    m = tiny(tmp_path/'ref')
    st = ea.run_plus(m, th, eager())
    ref = ea.chain_result(st)
    assert st['opt']['gen'] == 4 and [r['gen'] for r in st['records']] == [0, 2, 4] and st['cfg'] == m['params']['base_cfg']
    assert np.array_equal(np.asarray(st['records'][0]['theta']), np.asarray(th)) and st['budget']['start_theta_sha256'] == ea.theta_id(th)
    b = st['budget']
    assert (b['lam'], b['n_S'], b['sigma0'], b['gens'], b['val_every'], b['opt_seed'], b['key']) == (6, 2, .15, 4, 2, 96000, 0)
    for g in range(4):
        assert hook.rec[('plus', 0, g)][3] == tuple(e1.rotating(m['seeds']['EA2_AT'], 0, g, 2))        # rotating AT seeds, key 0
    assert all(set(v[3]) <= set(m['seeds']['EA2_AT']) for v in hook.rec.values())                       # training seeds come from EA2_AT only
    assert all(set(json.loads(json.dumps(ea.read(tmp_path/'ref'/'evaluations'/f.name)))['contract']['seeds']) <= set(m['seeds']['EA2_SEL'])
               for f in (tmp_path/'ref'/'evaluations').iterdir())                                       # validation only on EA2_SEL
    for edit in (lambda q: q['params']['plus'].update(n_S=3), lambda q: q['params']['base_cfg'].update(L=9.), lambda q: q['params']['plus'].update(opt_seed=1)):
        bad = copy.deepcopy(m)
        edit(bad)
        with pytest.raises(ValueError):
            ea.run_plus(bad, th, eager(), target=4)
    with pytest.raises(ValueError):
        ea.run_plus(m, start_theta(8), eager(), target=4)                                       # another start point is refused on resume
    with pytest.raises(ValueError):
        ea.run_plus(m, th, eager(), target=5)                                                   # never extended beyond the registered generations
    # interrupt after generation 2, resume: bitwise identical
    m2 = tiny(tmp_path/'res')
    monkeypatch.setattr(ea, 'GEN_HOOK', Collect(stop=lambda k, r, g: g == 1))
    with pytest.raises(ea.YieldStop):
        ea.run_plus(m2, th, eager())
    assert ea.read(ea.plus_path())['opt']['gen'] == 2
    monkeypatch.setattr(ea, 'GEN_HOOK', None)
    assert ea.digest(ea.chain_result(ea.run_plus(m2, th, lane()))) == ea.digest(ref)


# ------------------------------------------------------------------ decision tree / statistics
def test_decision_tree_every_node_labels_and_texts():
    c = ea.decide(vec(.003, .0090, N))
    assert (c['category'], c['sub'], c['label']) == ('stronger', None, '更強') and c['text'].startswith('更強')
    c = ea.decide(vec(-.003, .0090, N))
    assert (c['category'], c['sub'], c['label']) == ('not_stronger', 'weaker', '更弱')
    c = ea.decide(vec(.0005, .0001*np.sqrt(N), N))                                            # SE 0.0001: LB > 0, UB < eps
    assert (c['category'], c['sub']) == ('not_stronger', 'small_gain') and c['LB'] > 0 and c['UB'] < .001
    assert c['label'] == '小幅提升（排除高出 ε 以上）' and '沒有更強' not in c['label'] + c['text'] and '沒有發現' not in c['text'] and '沒有被發現' not in c['text']
    assert '檢查到小於 0.001 的超過' in c['text'] and '未再做適應' in c['text']
    c = ea.decide(vec(0., .0090, N))
    assert (c['category'], c['sub'], c['label']) == ('not_stronger', 'flat', '持平') and c['LB'] <= 0 <= c['UB'] < .001
    assert '不能宣稱 a* 是最強' in c['text'] and '不是獨立證據' in ea.decide(vec(-.003, .009, N))['text']
    c = ea.decide(vec(.001, .0090, N))
    assert c['category'] == 'insufficient' and c['label'] == '證據不足以判定' and '未解決' in c['text']
    assert ea.decide(np.full(8, .002))['category'] == 'stronger' and ea.decide(np.zeros(8))['sub'] == 'flat'      # sigma = 0
    assert 'A_fresh_orig' in ea.decide(vec(.001, .009, N))['subject'] and '未再做適應' in ea.decide(vec(.0016, .009, N))['text']
    assert all('沒有更強' not in v for v in ea.TEXT.values()) and all('沒有更強' not in v for v in ea.LABEL.values())


def test_decision_strict_boundaries_equal_threshold_hits_nothing():
    D = vec(.0016, .0090, N)
    lb, ub = ea.lower_bound(D), ea.upper_bound(D)
    assert ea.decide(D, eps=lb)['category'] != 'stronger' and ea.decide(D, eps=lb-1e-9)['category'] == 'stronger'
    assert ea.decide(D, eps=ub)['category'] != 'not_stronger' and ea.decide(D, eps=ub+1e-9)['category'] == 'not_stronger'
    c = ea.classify
    assert c(EPS, 1.) == ('insufficient', None) and c(EPS+1e-12, 1.)[0] == 'stronger'
    assert c(-1., EPS) == ('insufficient', None) and c(-1., EPS-1e-12)[0] == 'not_stronger'
    assert c(-1., 0.) == ('not_stronger', 'flat') and c(-1., -1e-12) == ('not_stronger', 'weaker')
    assert c(0., EPS-1e-12) == ('not_stronger', 'flat') and c(1e-12, EPS-1e-12) == ('not_stronger', 'small_gain')
    assert ea.tcrit(2048) == pytest.approx(1.6456, abs=1e-4)
    with pytest.raises(ValueError):
        ea.decide(np.array([.0]))


# the registered table of design 2.8 (n = 2048, sigma = 0.0090; Monte-Carlo with sampled SD), in percent: stronger, a, b, c, insufficient
POWER_TABLE = {-.0005: (0, 80.8, 0, 19.2, 0), 0: (0, 5.0, 4.9, 90.0, 0), .0003: (0, .1, 41.5, 55.4, 3.0), .0005: (0, 0, 61.5, 19.2, 19.3),
               .0007: (.1, 0, 41.6, 3.0, 55.3), .0010: (5.1, 0, 5.0, 0, 89.9), .0015: (80.6, 0, 0, 0, 19.4), .0020: (100, 0, 0, 0, 0)}


def test_power_table_reproduces_the_design_table_for_n_2048_including_a_b_c():
    rows = {r['mu']: r for r in ea.power_classes(2048, .0090, EPS, list(POWER_TABLE))}
    for mu, want in POWER_TABLE.items():
        got = tuple(100*rows[mu][k] for k in ('stronger', 'weaker', 'small_gain', 'flat', 'insufficient'))
        assert got == pytest.approx(want, abs=.3), (mu, got, want)
        assert sum(got) == pytest.approx(100, abs=.01)
    nc = {r['mu']: r for r in ea1.power_table(2048, .0090, EPS, [0, .0005, .001, .0015, .002])}          # non-central t for the two tails
    assert nc[.001]['p_stronger'] == pytest.approx(.051, abs=.004) and nc[.0015]['p_stronger'] == pytest.approx(.806, abs=.006) and nc[.002]['p_stronger'] > .999
    assert nc[0]['p_weaker'] == pytest.approx(.05, abs=.003)
    t, se = ea.tcrit(2048), .0090/np.sqrt(2048)
    assert (t*se, EPS-t*se, EPS+t*se) == pytest.approx((.000327, .000673, .001327), abs=2e-6)         # the (b) window of design 2.8
    assert (ea.PARAMS['n_test'], ea.PARAMS['eps']) == (2048, .001)


# ------------------------------------------------------------------ selection rules
def test_selection_rules_ties_a_fresh_orig_pool_picks_and_dedupe():
    assert ea.first_max([.1, .3, .3, .2]) == 1 and ea.first_max([.3, .3]) == 0
    with pytest.raises(ValueError):
        ea.first_max([.1, float('nan')])
    recs = [dict(gen=50, G_SEL=.01), dict(gen=100, G_SEL=.03), dict(gen=150, G_SEL=.03), dict(gen=200, G_SEL=.02)]
    assert ea.best_record(recs)['gen'] == 100                                                  # chain checkpoint: earliest of the equal maxima
    # A_O' / A_C2': ties go to a*, then to the list order
    names = ['x0', 'x1', 'astar', 'x3', 'x4']
    assert ea.pick_pool([.1, .5, .5, .5, .2], names, 'astar') == 2                              # a* among the ties
    assert ea.pick_pool([.1, .5, .4, .5, .2], names, 'astar') == 1                              # a* not tied: smallest list index
    assert ea.pick_pool([.1, .2, .9, .5, .2], names, 'astar') == 2 and ea.pick_pool([.1, .2, .3, .9, .2], names, 'astar') == 3
    assert ea.pick_pool([.6, .6, .1, .6, .6], names, 'astar') == 0
    with pytest.raises(ValueError):
        ea.pick_pool([.1, float('nan'), .1, .1, .1], names, 'astar')
    # dedupe by theta hash, first name is the entry
    t = [list(np.full(55, v)) for v in (.1, .2, .3)]
    pool = ea.dedupe([('a', t[0]), ('b', t[1]), ('c', t[0]), ('d', t[2]), ('e', t[1])])
    assert [e['name'] for e in pool] == ['a', 'b', 'd'] and pool[0]['aliases'] == ['c'] and pool[1]['aliases'] == ['e'] and pool[2]['aliases'] == []
    assert all(e['theta_sha256'] == ea.theta_id(e['theta']) for e in pool)
    # the real inputs: 12 E1 v2 entries -> 9 distinct, plus 3 E-A1 checkpoints plus 3 new = 15 distinct
    inp = ea.e1v2_entries()
    e1a = ea.ea1_inputs()
    assert inp['astar']['theta_sha256'].startswith('88ed926a') and inp['astar']['name'] == 'orig_adapt0' and len(inp['entries']) == 12
    d9 = ea.dedupe(inp['entries'])
    assert len(d9) == 9 and len(ea.dedupe(inp['entries']+e1a['chains'])) == 12 and [e['name'] for e in d9][:4] == ['frozen_run0', 'frozen_run1', 'frozen_run2', 'orig_adapt0']
    assert [n for n, _ in e1a['chains']] == ['ea1_fresh_run0', 'ea1_fresh_run1', 'ea1_fresh_run2'] and e1a['A_fresh']['name'] == 'ea1_fresh_run0' and e1a['A_fresh']['gen'] == 300
    assert ea.EXPECT_POOL == 15


def test_plus_selection_a_plus_all_points_a_plusplus_from_generation_20_ties_to_the_earlier():
    def recs(gs, step=20):
        return [dict(gen=i*step, G_SEL=g, theta=[float(i)]*55) for i, g in enumerate(gs)]
    r = recs([.03, .031, .0305, .032, .03, .02, .02, .02, .02, .02, .02, .02, .02])           # 13 points, best at generation 60
    i, j = ea.select_plus(r, 20)
    assert (i, j) == (3, 3)                                                                    # not collapsed: A++ is A+
    r = recs([.05, .031, .0305, .032, .03, .02, .02, .02, .02, .02, .02, .02, .02])           # start is the best: collapsed
    i, j = ea.select_plus(r, 20)
    assert (i, j) == (0, 3) and r[j]['gen'] == 60 and r[j]['G_SEL'] <= r[0]['G_SEL']          # A++ = best of the points from generation 20 on; not above the start
    r = recs([.05, .03, .04, .04, .01])
    assert ea.select_plus(r, 20) == (0, 2)                                                     # ties among the late points: earlier
    r = recs([.03, .03, .01, .03])
    assert ea.select_plus(r, 20) == (0, 1)                                                     # tie with the start: A+ stays at generation 0 (earlier wins) -> collapsed
    r = recs([.01, .03, .02, .03])
    assert ea.select_plus(r, 20) == (1, 1)
    assert ea.select_plus(r, 40) == (1, 3)                                                     # pp_min_gen = 40 drops generations 0 and 20
    with pytest.raises(ValueError):
        ea.select_plus(r, 1000)
    with pytest.raises(ValueError):
        ea.select_plus(recs([.01, float('nan')]), 20)
    for seed in range(50):                                                                      # random: if A+ is not at 0 then A++ == A+; if it is, A++ is a late point
        g = np.random.default_rng(seed).integers(0, 4, 13)/1000.
        i, j = ea.select_plus(recs(list(g)), 20)
        assert j >= 1 and (i == j if i > 0 else g[j] <= g[0]) and g[i] == g.max() and g[j] == g[1:].max()


# ------------------------------------------------------------------ consequence table R1-R17: every row is a literal predicate; mechanical checks
def tree(lb, ub):
    if lb > EPS:
        return 'stronger'
    if ub < EPS:
        return 'weaker' if ub < 0 else 'small_gain' if lb > 0 else 'flat'
    return 'insufficient'


def test_every_outcome_hits_exactly_one_row_per_group_including_the_boundaries():
    vals = sorted({-0.003, -0.001, -0.0005, 0.0, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.004, 0.005, 0.006, 0.008})                 # eps, 0 and delta are in the grid
    assert {EPS, 0.0, DELTA} <= set(vals) and (ea.EPS, ea.DELTA) == (EPS, DELTA)
    pts = [(lb, ub) for lb in vals for ub in vals if ub >= lb]
    for lb, ub in pts:
        assert len(ea.hits(ea.ROWS_MAIN, lb, ub)) == 1 and len(ea.hits(ea.ROWS_DPOOL, lb, ub)) == 1
        assert ea.TREE_TO_ROW[tree(lb, ub)] == ea.hits(ea.ROWS_MAIN, lb, ub)[0]               # the rows R1-R4 are the decision tree of section 2.2
        assert ea.classify(lb, ub)[0] == ('stronger' if tree(lb, ub) == 'stronger' else 'insufficient' if tree(lb, ub) == 'insufficient' else 'not_stronger')
        for g in (0, 1, 20, 240):
            assert len(ea.hits(ea.ROWS_PLUS, g, lb, ub)) == 1 and len(ea.hits(ea.ROWS_PP, g, lb, ub)) == 1
    assert len(pts) == 91
    # exact boundary rows, by hand
    assert ea.hits(ea.ROWS_MAIN, EPS, 1.) == ['R4'] and ea.hits(ea.ROWS_MAIN, EPS+1e-12, 1.) == ['R1'] and ea.hits(ea.ROWS_MAIN, -1., EPS) == ['R4']
    assert ea.hits(ea.ROWS_MAIN, -1., EPS-1e-12) == ['R3'] and ea.hits(ea.ROWS_MAIN, 0., EPS-1e-12) == ['R3'] and ea.hits(ea.ROWS_MAIN, 1e-12, EPS-1e-12) == ['R2']
    assert ea.hits(ea.ROWS_DPOOL, DELTA, 1.) == ['R7'] and ea.hits(ea.ROWS_DPOOL, DELTA+1e-12, 1.) == ['R5'] and ea.hits(ea.ROWS_DPOOL, -1., DELTA) == ['R7']
    assert ea.hits(ea.ROWS_DPOOL, -1., DELTA-1e-12) == ['R6']
    assert ea.hits(ea.ROWS_PLUS, 20, EPS, 1.) == ['R12'] and ea.hits(ea.ROWS_PLUS, 20, EPS+1e-12, 1.) == ['R9'] and ea.hits(ea.ROWS_PLUS, 20, 0., EPS-1e-12) == ['R11']
    assert ea.hits(ea.ROWS_PLUS, 20, 1e-12, EPS-1e-12) == ['R10'] and ea.hits(ea.ROWS_PLUS, 0, 1., 2.) == ['R8']
    assert ea.hits(ea.ROWS_PP, 0, 1e-12, EPS-1e-12) == ['R15'] and ea.hits(ea.ROWS_PP, 0, EPS+1e-12, 1.) == ['R14'] and ea.hits(ea.ROWS_PP, 0, EPS, 1.) == ['R17']
    assert ea.hits(ea.ROWS_PP, 0, -1., EPS-1e-12) == ['R16'] and ea.hits(ea.ROWS_PP, 40, 1., 2.) == ['R13']
    for bad_rows in ({'a': lambda x: True, 'b': lambda x: True}, {'a': lambda x: False}):
        with pytest.raises(ValueError, match='exclusive'):
            ea.the_row(bad_rows, 0)                                                            # an empty or a doubled hit is an error, never a silent pick


def test_all_300_combinations_hit_exactly_four_rows_and_the_summary_sentence_is_required_iff_r1_or_r5():
    main_rep = {'stronger': (0.0015, 0.0030), 'small_gain': (0.0002, 0.0008), 'weaker': (-0.0010, -0.0002), 'flat': (-0.0002, 0.0008), 'insufficient': (0.0003, 0.0015)}
    d_rep = {'LB>delta': (0.0055, 0.0065), 'UB<delta': (0.0035, 0.0045), 'cross': (0.0040, 0.0060)}
    plus_rep = {'gen0': (0, 0.0002, 0.0006), 'stronger': (60, 0.0015, 0.0030), 'small_gain': (60, 0.0002, 0.0008), 'UB<eps,LB<=0': (60, -0.0002, 0.0008),
                'insufficient': (60, 0.0003, 0.0015)}
    pp_rep = {'LB>eps': (0.0015, 0.0030), 'small_gain': (0.0002, 0.0008), 'UB<eps,LB<=0': (-0.0002, 0.0008), 'insufficient': (0.0003, 0.0015)}
    n = 0
    for (mk, mv), (dk, dv), (pk, pv), (qk, qv) in itertools.product(main_rep.items(), d_rep.items(), plus_rep.items(), pp_rep.items()):
        rows = ea.consequence_rows(mv, dv, pv[0], pv[1:], qv)
        n += 1
        assert set(rows) == {'main', 'd_pool', 'plus', 'plusplus'} and rows['main'] == ea.TREE_TO_ROW[mk] and tree(*mv) == mk
        assert rows['d_pool'] == {'LB>delta': 'R5', 'UB<delta': 'R6', 'cross': 'R7'}[dk]
        r = list(rows.values())
        assert len(r) == len(set(r)) == 4
        collapsed = pv[0] == 0
        assert (rows['plus'] == 'R8') == collapsed and (rows['plusplus'] == 'R13') == (not collapsed)       # R8 <-> A++ group is R14-R17; R13 <-> not collapsed
        assert (rows['plusplus'] in ('R14', 'R15', 'R16', 'R17')) == collapsed
        if collapsed:
            assert rows['plusplus'] == {'LB>eps': 'R14', 'small_gain': 'R15', 'UB<eps,LB<=0': 'R16', 'insufficient': 'R17'}[qk]
        else:
            assert rows['plus'] == {'stronger': 'R9', 'small_gain': 'R10', 'UB<eps,LB<=0': 'R11', 'insufficient': 'R12'}[pk]
        add = ea.summary_additions(rows)
        assert (('R1' in r) or ('R5' in r)) == bool(add) and len(add) == int('R1' in r)+int('R5' in r)    # both sentences when both rows hit
    assert n == 300
    assert ea.summary_additions(dict(main='R3', d_pool='R6', plus='R8', plusplus='R16')) == []
    both = ea.summary_additions(dict(main='R1', d_pool='R5', plus='R9', plusplus='R13'))
    assert both == [ea.SUMMARY_ADD['R1'], ea.SUMMARY_ADD['R5']] and '僅對固定 a*' in both[0] and '僅對固定 a*' in both[1]
    assert set(ea.SUMMARY_ADD) == {'R1', 'R5'}
    # the texts of every row exist and carry the registered qualifiers
    assert set(ea.PLUS_TEXT) == {'R8', 'R9', 'R10', 'R11', 'R12'} and set(ea.PP_TEXT) == {'R13', 'R14', 'R15', 'R16', 'R17'} and set(ea.D_POOL_TEXT) == {'R5', 'R6', 'R7'}
    assert '未提供適應後的資訊' in ea.PLUS_TEXT['R8'] and '經適應' not in ea.PLUS_TEXT['R8'].replace('未提供適應後的資訊', '') and '未發現' not in ea.PLUS_TEXT['R10']
    assert '選取偏高' in ea.PP_TEXT['R14'] and '選取偏高' in ea.PP_TEXT['R15'] and '未發現' not in ea.PP_TEXT['R15']
    assert '15 個條目' in ea.D_POOL_TEXT['R6'] and all('穩健' not in v and '效果不足' not in v.replace('不改 E1 v2 已登記的類別', '') for v in ea.D_POOL_TEXT.values())


# ------------------------------------------------------------------ ledger
def test_ledger_is_one_shot_atomic_and_detects_tampering(tmp_path):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    calls = [0]

    def runner():
        calls[0] += 1
        return dict(T_score=10, snap=dict(util_b=np.ones((2, 2, 3)).tolist(), reg_b=np.ones((2, 2)).tolist(), pen=np.zeros((2, 2, 3)).tolist()))
    raw = ea.ledger_once(tmp_path, 'main_orig', con, runner)
    assert ea.ledger_once(tmp_path, 'main_orig', con, runner) == raw and calls[0] == 1
    assert ea.read(tmp_path/'ledger.json')['main_orig']['status'] == 'completed' and not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError):
        ea.ledger_once(tmp_path, 'main_orig', dict(con, seeds=[1, 3]), runner)
    r = ea.read(tmp_path/'main_orig_raw.json')
    r['snap']['util_b'][0][0][0] = 5.
    ea.write(tmp_path/'main_orig_raw.json', r)
    with pytest.raises(ValueError):
        ea.ledger_once(tmp_path, 'main_orig', con, runner)                                      # tampered raw
    ea.write(tmp_path/'ledger.json', dict(crash=dict(status='started', contract_sha256=ea.digest(con))))
    with pytest.raises(RuntimeError):
        ea.ledger_once(tmp_path, 'crash', con, runner)                                          # started, no raw: stop
    assert calls[0] == 1


def ok_raw():
    return dict(T_score=10, snap=dict(util_b=np.ones((2, 2, 3)).tolist(), reg_b=np.ones((2, 2)).tolist(), pen=np.zeros((2, 2, 3)).tolist()))


def started_ledger(folder, con, name='plus_orig'):
    """A ledger left behind by an attempt that died before any raw was written."""
    def dying():
        raise OSError('machine went away')
    with pytest.raises(OSError):
        ea.ledger_once(folder, name, con, dying)
    return ea.read(folder/'ledger.json')[name]


def test_ledger_reset_needs_authorisation_confirmation_archives_and_allows_one_rerun_with_the_same_contract(tmp_path, monkeypatch):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: False)
    old = started_ledger(tmp_path, con)
    assert old['status'] == 'started' and not (tmp_path/'plus_orig_raw.json').exists()
    with pytest.raises(RuntimeError):
        ea.ledger_once(tmp_path, 'plus_orig', con, ok_raw)                                      # stuck: never silently repeats
    t = old['started_unix']
    for kw in (dict(), dict(authorized=False), dict(authorized='yes')):
        with pytest.raises(PermissionError):
            ea.ledger_reset(tmp_path, 'plus_orig', 'host restarted', confirm_started_unix=t, **kw)          # no user authorisation: refused
    with pytest.raises(ValueError):
        ea.ledger_reset(tmp_path, 'plus_orig', '  ', authorized=True, confirm_started_unix=t)   # a reason is required
    with pytest.raises(ValueError):
        ea.ledger_reset(tmp_path, 'never_started', 'x', authorized=True, confirm_started_unix=t)
    for wrong in (None, t+1.):
        with pytest.raises(ValueError, match='started_unix'):
            ea.ledger_reset(tmp_path, 'plus_orig', 'host restarted', authorized=True, confirm_started_unix=wrong)     # the entry must have been looked at
    assert not ea.aborted_path(tmp_path).exists()
    ea.ledger_reset(tmp_path, 'plus_orig', 'host restarted', authorized=True, confirm_started_unix=t)
    ab = ea.read(ea.aborted_path(tmp_path))
    assert len(ab) == 1 and ab[0]['name'] == 'plus_orig' and ab[0]['entry'] == old and ab[0]['reason'] == 'host restarted' and 'plus_orig' not in ea.read(tmp_path/'ledger.json')
    assert not list(tmp_path.glob('*.tmp'))
    # the reset record must be committed before the run is repeated; another contract is refused whatever the git state
    with pytest.raises(ValueError, match='not committed'):
        ea.ledger_once(tmp_path, 'plus_orig', con, ok_raw)
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    with pytest.raises(ValueError, match='same contract'):
        ea.ledger_once(tmp_path, 'plus_orig', dict(con, seeds=[1, 3]), ok_raw)
    raw = ea.ledger_once(tmp_path, 'plus_orig', con, ok_raw)                                    # same contract, archive committed: runs once more
    assert raw['T_score'] == 10 and ea.read(tmp_path/'ledger.json')['plus_orig']['status'] == 'completed'
    with pytest.raises(ValueError, match='never is'):
        ea.ledger_reset(tmp_path, 'plus_orig', 'again', authorized=True, confirm_started_unix=t)   # completed: never reset
    assert ea.ledger_once(tmp_path, 'plus_orig', con, ok_raw) == raw                            # and never run again


def test_ledger_reset_never_touches_a_finished_evaluation_a_committed_raw_or_the_primary_result(tmp_path, monkeypatch):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    d = tmp_path/'test'
    # a raw file on disk next to a 'started' entry (forged or half-written): refused
    t = started_ledger(d, con, 'aux_C2')['started_unix']
    ea.write(d/'aux_C2_raw.json', dict(x=1))
    with pytest.raises(ValueError, match='never re-run'):
        ea.ledger_reset(d, 'aux_C2', 'x', authorized=True, confirm_started_unix=t)
    assert not ea.aborted_path(d).exists() and ea.read(d/'ledger.json')['aux_C2']['status'] == 'started'
    # a raw file that was ever committed (and then deleted) is refused too: real history of a throw-away git repository
    real_root = tmp_path/'repo'
    real_root.mkdir()
    for cmd in (['init', '-q'], ['config', 'user.email', 'a@b.c'], ['config', 'user.name', 'n']):
        subprocess.run(['git', *cmd], cwd=real_root, check=True)
    monkeypatch.setattr(ea, 'ROOT', real_root)
    d2 = real_root/'results'/'test'
    t2 = started_ledger(d2, con, 'sec_orig')['started_unix']
    assert ea.ever_in_git(d2/'sec_orig_raw.json') is False
    (d2/'sec_orig_raw.json').write_text('{}')
    subprocess.run(['git', 'add', '-f', 'results/test/sec_orig_raw.json'], cwd=real_root, check=True)
    assert ea.ever_in_git(d2/'sec_orig_raw.json') is True                                       # staged
    subprocess.run(['git', 'commit', '-q', '-m', 'raw'], cwd=real_root, check=True)
    subprocess.run(['git', 'rm', '-q', '-f', 'results/test/sec_orig_raw.json'], cwd=real_root, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'gone'], cwd=real_root, check=True)
    assert not (d2/'sec_orig_raw.json').exists() and ea.ever_in_git(d2/'sec_orig_raw.json') is True            # deleted, but it is in the history
    with pytest.raises(ValueError, match='never re-run'):
        ea.ledger_reset(d2, 'sec_orig', 'x', authorized=True, confirm_started_unix=t2)
    # the primary evaluation: refused once result.json exists
    t3 = started_ledger(tmp_path/'o'/'test', con, 'main_orig')['started_unix']
    ea.write(tmp_path/'o'/'result.json', dict(x=1))
    with pytest.raises(ValueError, match='primary'):
        ea.ledger_reset(tmp_path/'o'/'test', 'main_orig', 'x', authorized=True, confirm_started_unix=t3)


def test_ledger_reset_interrupted_between_the_two_writes_is_completed_by_a_second_call(tmp_path, monkeypatch):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    old = started_ledger(tmp_path, con)
    real_write = ea.write
    calls = []

    def dying_write(p, x):
        if str(p).endswith('ledger.json') and not str(p).endswith('aborted.json'):
            calls.append(1)
            raise OSError('power cut between the two writes')
        return real_write(p, x)
    monkeypatch.setattr(ea, 'write', dying_write)
    with pytest.raises(OSError):
        ea.ledger_reset(tmp_path, 'plus_orig', 'power', authorized=True, confirm_started_unix=old['started_unix'])
    monkeypatch.setattr(ea, 'write', real_write)
    assert len(ea.read(ea.aborted_path(tmp_path))) == 1 and ea.read(tmp_path/'ledger.json')['plus_orig'] == old        # archive written, ledger not yet
    ea.ledger_reset(tmp_path, 'plus_orig', 'power', authorized=True, confirm_started_unix=old['started_unix'])           # completes, does not archive twice
    assert len(ea.read(ea.aborted_path(tmp_path))) == 1 and 'plus_orig' not in ea.read(tmp_path/'ledger.json') and calls


def test_ledger_reset_runs_under_the_one_lock_and_is_refused_while_a_test_holds_it(tmp_path, monkeypatch):
    import fcntl
    ea.MODE.update(out=str(tmp_path), smoke=False)
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    old = started_ledger(tmp_path/'test', con)
    argv = ['ledger-reset', '--eval', 'plus_orig', '--reason', 'x', '--user-authorized', '--confirm-started-unix', repr(old['started_unix']), '--device', 'cpu']
    (ea.ROOT/'.scratch').mkdir(exist_ok=True)
    with (ea.ROOT/'.scratch/ea2.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)                                        # a running stage holds the lock
        with pytest.raises(BlockingIOError):
            ea.main(argv)
        assert not ea.aborted_path(tmp_path/'test').exists()
    assert ea.main(argv) == 0 and ea.read(ea.aborted_path(tmp_path/'test'))[0]['name'] == 'plus_orig'


# ------------------------------------------------------------------ manifest / environment / hash pins
def fake_manifest(smoke=True):
    m = dict(schema='x', smoke=smoke, params=json.loads(json.dumps(ea.SMOKE_PARAMS if smoke else ea.PARAMS)), seeds={'a': [1, 2]})
    m['sha256'] = ea.digest(m)
    return m


def test_manifest_changes_are_detected_and_params_must_match_code(tmp_path):
    m = fake_manifest()
    assert ea.verify_manifest_dict(m) is m
    for edit in (lambda x: x['params'].update(eps=.002), lambda x: x['seeds'].update(a=[1, 3]), lambda x: x.update(design_sha256='0'),
                 lambda x: x['params']['attacker'].update(gens=500), lambda x: x['params']['plus'].update(lam=32)):
        bad = copy.deepcopy(m)
        edit(bad)
        with pytest.raises(ValueError):
            ea.verify_manifest_dict(bad)
    reg = fake_manifest(False)
    reg['params']['attacker']['lam'] = 64
    reg['sha256'] = ea.digest({k: v for k, v in reg.items() if k != 'sha256'})
    with pytest.raises(ValueError):                                                              # self-consistent hash, but not the coded design
        ea.verify_manifest_dict(reg)
    f = tmp_path/'a.txt'
    f.write_text('x')
    h = {'a.txt': ea.sha(f)}
    ea.check_hashes(h, tmp_path)
    f.write_text('y')
    with pytest.raises(ValueError):
        ea.check_hashes(h, tmp_path)
    with pytest.raises(ValueError):
        ea.check_hashes({'missing.txt': '0'}, tmp_path)


def test_ea1_files_are_pinned_by_content_hash_and_any_change_is_refused(tmp_path):
    assert set(ea.EA1_REFS) == {'scripts/ea1.py', 'scripts/ea1_launch.sh', 'tests/direction2/test_ea1.py', 'results/direction2/ea1/selection_frozen.json',
                                'results/direction2/ea1/result.json', 'results/direction2/ea1/secondary.json', 'results/direction2/ea1/manifest.json'}
    ea.check_ea1_refs()                                                                          # the real files match the registered values
    assert ea.EA1_REFS['scripts/ea1.py'].startswith('55aba50e') and ea.EA1_REFS['results/direction2/ea1/selection_frozen.json'].startswith('ff6b5a48')
    assert ea.EA1_DESIGN[0] == 'docs/direction2/design/e-a1.md' and ea.EA1_DESIGN[1] not in ea.EA1_REFS.values()       # recorded only, not a load condition
    root = tmp_path/'root'
    for rel in ea.EA1_REFS:
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ea.ROOT/rel, root/rel)
    ea.check_ea1_refs(root)
    for rel in ea.EA1_REFS:
        orig = (root/rel).read_bytes()
        (root/rel).write_bytes(orig+b' ')
        with pytest.raises(ValueError, match='frozen input changed'):
            ea.check_ea1_refs(root)                                                              # any one of the seven files changed
        (root/rel).write_bytes(orig)
        (root/rel).unlink()
        with pytest.raises(ValueError):
            ea.check_ea1_refs(root)                                                              # or missing
        (root/rel).write_bytes(orig)
    ea.check_ea1_refs(root)
    other = dict(ea.EA1_REFS, **{'scripts/ea1.py': '0'*64})
    with pytest.raises(ValueError, match='registered'):
        ea.check_ea1_refs(root, recorded=other)                                                  # a manifest that records other pins is refused


def test_e1v2_protected_and_input_files_still_match_their_manifest():
    e1m = e1.read(ea.ROOT/ea.E1V2_DIR/'manifest.json')
    files = {**e1m['protected_sha256'], **e1m['input_sha256'], e1m['design']: e1m['design_sha256']}
    assert len(files) >= 90
    for rel, h in files.items():
        assert hashlib.sha256((ea.ROOT/rel).read_bytes()).hexdigest() == h, rel
    assert not any(p.startswith(('scripts/ea1', 'scripts/ea2', 'tests/direction2/test_ea')) for p in files)


def poison_ea1_manifest_and_load(monkeypatch):
    """ea1.load_manifest must never be called, and the E-A1 manifest content must never be parsed."""
    def boom(*a, **k):
        raise AssertionError('ea1.load_manifest was called')
    monkeypatch.setattr(ea1, 'load_manifest', boom)
    for mod in (ea, ea1):
        orig = mod.read
        monkeypatch.setattr(mod, 'read', lambda p, orig=orig: (_ for _ in ()).throw(AssertionError('E-A1 manifest parsed')) if str(p).endswith('ea1/manifest.json') else orig(p))


def test_smoke_manifest_build_records_pins_inputs_protected_hashes_and_no_host_name(tmp_path, monkeypatch):
    poison_ea1_manifest_and_load(monkeypatch)
    monkeypatch.setattr(ea, 'environment', lambda: dict(torch='t', numpy='n', scipy='s', device_name='d', threads={}))
    m, audit_rec = ea.build_manifest(smoke=True)
    e1m = e1.read(ea.ROOT/ea.E1V2_DIR/'manifest.json')
    assert m['smoke'] and set(m['seeds']) == {'EA2_TRAIN', 'EA2_SEL', 'EA2_AT'} and 'EA2_TEST' not in m['seeds']
    assert all(m['protected_sha256'][k] == v for k, v in {**e1m['protected_sha256'], **e1m['input_sha256']}.items())
    assert m['protected_sha256']['docs/direction2/design/e1-v2.md'] == e1m['design_sha256']
    assert m['ea1_refs_sha256'] == ea.EA1_REFS and m['ea1_design_sha256_record'] == {ea.EA1_DESIGN[0]: ea.sha(ea.ROOT/ea.EA1_DESIGN[0])}
    assert any(k.endswith('selection_frozen.json') for k in m['e1v2_refs_sha256']) and any(k.endswith('attacker/orig_run0.json') for k in m['e1v2_refs_sha256'])
    assert set(m['input_sha256']) == set(ea.INPUTS) and ea.DESIGN == 'docs/direction2/design/e-a2.md'
    assert m['environment']['torch'] == 't' and m['drift']['seeds'] == e1m['seeds']['AS']
    ea.check_hashes({**m['protected_sha256'], **m['e1v2_refs_sha256'], **m['ea1_refs_sha256']})
    assert m['params']['astar_theta_sha256'].startswith('88ed926a') and m['sha256'] == ea.digest({k: v for k, v in m.items() if k != 'sha256'})
    assert 'host' + 'name' not in json.dumps(m) and 'host' + 'name' not in json.dumps(ea.environment.__code__.co_names)
    # borrowed ea1 helpers read these manifest keys
    assert {'exclude', 'T_val', 'T_train', 'test_chunk', 'kernel_K'} <= set(m['params'])


def test_smoke_freeze_and_load_manifest_never_depend_on_the_e_a1_git_or_design_hash(tmp_path, monkeypatch):
    poison_ea1_manifest_and_load(monkeypatch)
    monkeypatch.setattr(ea, 'environment', lambda: dict(torch='t', numpy='n', scipy='s', device_name='d', threads={}, d2_graph_k=None))
    ea.MODE.update(out=str(tmp_path/'s'), smoke=True)
    ea.freeze(True)
    m = ea.load_manifest(clean=False)
    assert m['smoke'] and m['ea1_refs_sha256'] == ea.EA1_REFS
    p = tmp_path/'s'/'manifest.json'
    raw = ea.read(p)

    def rewrite(**kw):
        x = dict(copy.deepcopy(raw), **kw)
        x['sha256'] = ea.digest({k: v for k, v in x.items() if k != 'sha256'})
        ea.write(p, x)
    rewrite(code_git_sha='0'*40)                                                                 # recorded only: neither a missing commit nor a rewritten history matters
    assert ea.load_manifest(clean=False)['code_git_sha'] == '0'*40
    rewrite(ea1_design_sha256_record={ea.EA1_DESIGN[0]: '0'*64})                                 # E-A1 design hash: recorded only
    ea.load_manifest(clean=False)
    rewrite(ea1_refs_sha256=dict(ea.EA1_REFS, **{'scripts/ea1.py': '0'*64}))
    with pytest.raises(ValueError, match='registered'):
        ea.load_manifest(clean=False)
    rewrite(input_sha256=dict(raw['input_sha256'], **{'scripts/ea2.py': '0'*64}))
    with pytest.raises(ValueError, match='frozen input changed'):
        ea.load_manifest(clean=False)
    rewrite()
    ea.MODE.update(smoke=False)
    with pytest.raises(ValueError, match='mixup'):
        ea.load_manifest(clean=False)


def test_official_load_requires_inputs_committed_and_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, 'environment', lambda: dict(torch='t', numpy='n', scipy='s', device_name='d', threads={}, d2_graph_k=None))
    ea.MODE.update(out=str(tmp_path/'s'), smoke=True)
    ea.freeze(True)
    m = ea.read(tmp_path/'s'/'manifest.json')
    m['smoke'] = False
    m['params'] = json.loads(json.dumps(ea.PARAMS))
    m['sha256'] = ea.digest({k: v for k, v in m.items() if k != 'sha256'})
    ea.write(tmp_path/'s'/'manifest.json', m)
    ea.MODE.update(smoke=False)
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: False)
    with pytest.raises(ValueError, match='uncommitted'):
        ea.load_manifest()
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    assert ea.load_manifest()['sha256'] == m['sha256']                                           # every input hash matches and everything is committed and clean
    ea.write(tmp_path/'s'/'seed_audit.json', dict(ea.read(tmp_path/'s'/'seed_audit.json'), scanned_sha256={'x': 'y'}))
    with pytest.raises(ValueError, match='seed audit'):
        ea.load_manifest()


def test_gpu_stages_refuse_when_the_environment_differs_and_no_host_name_is_recorded():
    env = dict(device_name='Quadro RTX 4000', capability=[7, 5], gpu_name='Quadro RTX 4000', gpu_uuid='GPU-x', torch='2.6.0+cu126', cuda='12.6', numpy='2.2.6',
               scipy='1.15.3', threads=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), driver='595', d2_graph_k=None)
    m = dict(environment=env)
    ea.check_environment(m, dict(env, driver='596'))                                             # the driver is recorded, not checked
    for k, v in (('torch', '2.7.0'), ('numpy', '2.0.0'), ('scipy', '1.14.0'), ('device_name', 'Other GPU model'), ('cuda', '12.1'), ('gpu_uuid', 'GPU-y'), ('capability', [7, 0]),
                 ('threads', dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')), ('gpu_name', 'Other GPU')):
        with pytest.raises(ValueError):
            ea.check_environment(m, dict(env, **{k: v}))
    with pytest.raises(ValueError, match='D2_GRAPH_K'):
        ea.check_environment(m, dict(env, d2_graph_k='8'))
    ea.check_environment(m, dict(env, d2_graph_k='32'))
    e = ea.environment()
    assert not any('host' in k or 'name' == k for k in e) and set(e) == {'python', 'd2_graph_k', 'numpy', 'scipy', 'torch', 'cuda', 'threads', 'device_name', 'capability',
                                                                          'gpu_name', 'gpu_uuid', 'driver'}
    assert (ea.ENV_PIN['device_name'], ea.ENV_PIN['torch'], ea.ENV_PIN['numpy'], ea.ENV_PIN['scipy']) == ('Quadro RTX 4000', '2.6.0+cu126', '2.2.6', '1.15.3')


def test_no_machine_or_account_name_in_code_and_no_host_name_call():
    banned = ['gethost' + 'name', 'socket.' + 'gethost', 'host' + 'name']
    for rel in ('scripts/ea2.py', 'scripts/ea2_launch.sh'):
        text = (ea.ROOT/rel).read_text()
        assert not any(b in text for b in banned), rel
    tree_ = ast.parse((ea.ROOT/'scripts/ea2.py').read_text())
    calls = {f'{n.func.value.id}.{n.func.attr}' for n in ast.walk(tree_) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)}
    assert 'ea1.load_manifest' not in calls and 'ea1.build_manifest' not in calls and 'ea1.freeze' not in calls and 'socket.gethost' + 'name' not in calls
    assert 'ea1.out_dir' not in calls and 'ea1.stamp' not in calls and 'ea1.evaluate' not in calls and 'ea1.run_chain' not in calls                # the E-A1-bound functions are not borrowed


def test_freeze_raises_when_environment_differs_from_pins(tmp_path, monkeypatch):
    good = dict(ea.ENV_PIN)
    ea.MODE.update(out=str(tmp_path/'o'), smoke=False)
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    for k in ea.ENV_PIN:
        env = dict(good, **{k: 'other'})
        monkeypatch.setattr(ea, 'build_manifest', lambda smoke=False, env=env: (dict(environment=env, sha256='x'), {}))
        with pytest.raises(ValueError, match='design pins'):
            ea.freeze(False)
        assert not (tmp_path/'o/manifest.json').exists()
    monkeypatch.setattr(ea, 'build_manifest', lambda smoke=False: (dict(environment=good, sha256='x'), {}))
    ea.freeze(False)
    assert (tmp_path/'o/manifest.json').exists()
    with pytest.raises(RuntimeError, match='already exists'):
        ea.freeze(False)                                                                         # the registered manifest is never replaced
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: False)
    ea.MODE.update(out=str(tmp_path/'o2'))
    with pytest.raises(ValueError, match='commit all inputs'):
        ea.freeze(False)


# ------------------------------------------------------------------ guards against writing into E-A1 / using E-A1 code paths
def test_main_disables_ea1_out_dir_only_inside_the_process_call_and_restores_it(monkeypatch):
    orig = ea1.out_dir
    seen = {}

    def probe(smoke=False, root=ea.ROOT):
        try:
            ea1.out_dir()
        except RuntimeError as exc:
            seen['err'] = str(exc)
        return {}, set(), {}
    monkeypatch.setattr(ea, 'audit', probe)
    monkeypatch.setattr(ea, 'seed_report', lambda *a, **k: {})
    assert ea1.out_dir is orig                                                                   # nothing at import time
    ea.main(['audit'])
    assert 'E-A1' in seen['err'] and ea1.out_dir is orig                                         # disabled while main runs, restored afterwards


def test_chain_and_report_code_never_resolve_the_e_a1_directory(tmp_path, monkeypatch):
    def boom():
        raise AssertionError('ea1.out_dir was resolved')
    monkeypatch.setattr(ea1, 'out_dir', boom)
    m = tiny(tmp_path)
    ea.run_chains(m, [0], eager)
    ea.stamp('x')
    ea.mark_done('x')
    assert ea.stages_done() == {'x'} and ea.chain_path(0).parent.parent == tmp_path
    th = start_theta()
    ea.run_plus(m, th, eager())
    assert ea.plus_path().parent.parent == tmp_path


# ------------------------------------------------------------------ select -> adapt -> testcheck -> test -> report, with a deterministic fake simulator
class FakeSim:
    """Deterministic stand-in: a policy's gain grows with tanh(sum(theta)), shrinks with tol; independent of the other policies in the call.  Records every call."""
    def __init__(self):
        self.log, self.stage, self.perturb = [], '', None

    def __call__(self, m, cfg, pols, sd, T, be=None):
        self.log.append((self.stage, len(pols), cfg['tol'], list(sd)))
        ts = T-m['params']['exclude']
        base = np.array([float(np.tanh(np.asarray(getattr(q, 'theta', np.zeros(1))).sum())) for q in pols])
        u = np.full((len(pols), len(sd), 3), 100.)
        u[:, :, 1] += ((base-base[0])[:, None]*(1.0-.05*(cfg['tol']-1.))*.02+.001*np.sin(np.asarray(sd, float))[None, :])*ts*(np.arange(len(pols))[:, None] > 0)
        reg = np.ones((len(pols), len(sd)))*(1+cfg['tol'])
        if self.perturb is not None:
            self.perturb(self, pols, cfg, u)
        return dict(T_score=ts, snap=dict(util_b=u.tolist(), reg_b=reg.tolist(), pen=np.zeros((len(pols), len(sd), 3)).tolist()))

    def seeds(self, stage):
        return {s for st, _, _, sd in self.log if st == stage for s in sd}


def fake_inputs():
    """12 E1 v2 entries as in the real data: nine distinct parameter vectors, three aliases (orig_adapt1, orig_adapt2, C2_adapt1)."""
    rng = np.random.default_rng(3)
    d = [list(rng.normal(0, .02, 55)) for _ in range(9)]
    pairs = [('frozen_run0', d[0]), ('frozen_run1', d[1]), ('frozen_run2', d[2]), ('orig_adapt0', d[3]), ('orig_adapt1', d[1]), ('orig_adapt2', d[2]), ('C1_adapt0', d[4]),
             ('C1_adapt1', d[5]), ('C1_adapt2', d[6]), ('C2_adapt0', d[7]), ('C2_adapt1', d[1]), ('C2_adapt2', d[8])]
    star = dict(name='orig_adapt0', theta=d[3], theta_sha256=ea.theta_id(d[3]))
    inp = dict(astar=star, entries=pairs, c2_cfg=dict(p=.5, tol=1., kref=.5, L=8.), G_AS_astar=dict(orig=.03, C2=.02), attackers=[n for n, _ in pairs], transfer_raw=dict(orig='x', C2='y'))
    ch = [(f'ea1_fresh_run{i}', list(rng.normal(0, .02, 55))) for i in range(3)]
    e1a = dict(chains=ch, A_fresh=dict(name='ea1_fresh_run0', gen=300, theta=ch[0][1], theta_sha256=ea.theta_id(ch[0][1])))
    return inp, e1a


def patch_flow(mp, m, fake):
    inp, e1a = fake_inputs()
    mp.setattr(ea, 'e1v2_entries', lambda: inp)
    mp.setattr(ea, 'ea1_inputs', lambda root=ea.ROOT: e1a)
    mp.setattr(ea, 'drift_reference', lambda i: dict(orig=np.full(4, .03), C2=np.full(4, .02)))
    mp.setattr(ea, 'sim', fake)
    mp.setattr(ea, 'sim_eager', lambda m_, cfg, pols, sd, T: fake(m_, cfg, pols, sd, T))
    mp.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    mp.setattr(ea, 'load_manifest', lambda clean=True: m)
    mp.setattr(ea, 'environment', lambda: dict(torch='t'))
    mp.setattr(ea1, 'check_budget', lambda: None)


def flow_gen(tmp_path_factory, name, lift=False):
    """One tiny official-style run, built once: 3 tiny chains trained (real CPU simulator), then select / adapt / testcheck / test(main, plus, rest) with the
    fake simulator.  Snapshots of the directory are kept after testcheck and after the primary result (copies are used by the mutating tests).
    lift=True makes every re-adaptation validation point after generation 0 clearly better than the start, so that A+ is NOT collapsed."""
    root = tmp_path_factory.mktemp(name)
    d = root/'live'
    mp = pytest.MonkeyPatch()
    m = tiny(d, gens=6)
    ea.run_chains(m, [0, 1, 2], eager)
    fake = FakeSim()
    patch_flow(mp, m, fake)
    try:
        fake.stage = 'select'
        ea.stage_select(m)
        ea.write(d/'smoke_summary.json', dict(graph_equals_eager=True, code_sha256=ea.code_hashes()))
        if lift:
            start = np.asarray(ea.read(d/'selection_frozen.json')['A_fresh_orig']['theta'])

            def lifter(f, pols, cfg, u):
                if f.stage == 'adapt' and len(pols) == 2 and not np.array_equal(np.asarray(pols[1].theta), start):
                    u[1, :, 1] += 1.0
            fake.perturb = lifter
        fake.stage = 'adapt'
        ea.stage_adapt(m)
        fake.perturb = None
        ea.MODE.update(strict=True, graph=False)
        fake.stage = 'testcheck'
        ea.run_testcheck()
        shutil.copytree(d, root/'pre_test')
        fake.stage = 'test'
        ea.run_test('main')
        shutil.copytree(d, root/'after_main')
        ea.run_test('plus')
        ea.run_test('rest')
        yield dict(root=root, dir=d, m=m, fake=fake, pre_test=root/'pre_test', after_main=root/'after_main')
    finally:
        mp.undo()
        ea.MODE.update(smoke=False, device='cuda', graph=True, out=None, start=None, strict=False, yield_check=False, K=None)


@pytest.fixture(scope='module')
def base_flow(tmp_path_factory):
    yield from flow_gen(tmp_path_factory, 'flow')


@pytest.fixture(scope='module')
def nc_flow(tmp_path_factory):
    yield from flow_gen(tmp_path_factory, 'flow_nc', lift=True)


def use(base):
    """Point the module state at the live flow directory (the autouse fixture resets MODE after every test)."""
    ea.MODE.update(out=str(base['dir']), smoke=False, device='cpu', graph=False, strict=True, K=4)


def clone(base, which, tmp_path, monkeypatch):
    d = tmp_path/'clone'
    shutil.copytree(base[which], d)
    m = copy.deepcopy(base['m'])
    ea.MODE.update(out=str(d), smoke=False, device='cpu', graph=False, strict=True, K=4)
    fake = FakeSim()
    patch_flow(monkeypatch, m, fake)
    return m, d, fake


def test_select_registers_one_frozen_selection_from_one_reevaluation_per_setting(base_flow):
    use(base_flow)
    d, m, fake = base_flow['dir'], base_flow['m'], base_flow['fake']
    sel = ea.read(d/'selection_frozen.json')
    assert ea.read(d/'freeze_record.json')['selection_frozen_sha256'] == ea.sha(d/'selection_frozen.json')
    calls = [(n, t) for st, n, t, _ in fake.log if st == 'select']
    assert [n for n, _ in calls] == [5, 16, 16, 2, 2]                                           # orig re-eval; pool in orig, pool in C2; a* drift in orig, C2
    assert [t for _, t in calls] == [2., 2., 1., 2., 1.]
    assert all(set(sd) <= set(m['seeds']['EA2_SEL']) for st, _, _, sd in fake.log if st == 'select' and len(sd) == 8) and fake.seeds('select') - set(m['seeds']['EA2_SEL']) <= set(m['drift']['seeds'])
    assert not fake.seeds('select') & set(m['seeds']['EA2_TEST']) | fake.seeds('select') & set(m['seeds']['EA2_AT']) | fake.seeds('select') & set(m['seeds']['EA2_TRAIN'])
    G = sel['reeval_orig']['G']
    assert sel['reeval_orig']['names'] == ['a_star', 'ea2_fresh_run0', 'ea2_fresh_run1', 'ea2_fresh_run2']
    assert sel['A_fresh_orig']['index'] == ea.first_max(G[1:]) and sel['A_fresh_orig']['G_SEL_reeval'] == G[1:][sel['A_fresh_orig']['index']]
    assert sel['A_fresh_orig']['name'] == f'ea2_fresh_run{sel["A_fresh_orig"]["index"]}'
    for c in sel['chains']:                                                                       # chain checkpoint = best of its own validation points
        assert c['G_SEL_chain'] == max(r['G_SEL'] for r in ea.chain_state(c['run'])['records']) and c['theta_sha256'] == ea.theta_id(c['theta'])
    names = [e['name'] for e in sel['pool']]
    assert len(names) == 15 == len(set(sel['pool'][i]['theta_sha256'] for i in range(15))) and names[:4] == ['frozen_run0', 'frozen_run1', 'frozen_run2', 'orig_adapt0']
    assert names[-6:] == ['ea1_fresh_run0', 'ea1_fresh_run1', 'ea1_fresh_run2', 'ea2_fresh_run0', 'ea2_fresh_run1', 'ea2_fresh_run2']
    assert sel['pool'][1]['aliases'] == ['orig_adapt1', 'C2_adapt1'] and sel['pool'][2]['aliases'] == ['orig_adapt2']
    for label, key in (('orig', 'A_O_prime'), ('C2', 'A_C2_prime')):
        assert sel[key]['index'] == ea.pick_pool(sel['descriptive_SEL'][label]['G'], names, 'orig_adapt0') and sel[key]['name'] == names[sel[key]['index']]
        assert sel['descriptive_SEL'][label]['names'] == names
    assert sel['ea1_A_fresh']['name'] == 'ea1_fresh_run0' and sel['ea1_refs_sha256'] == ea.EA1_REFS
    assert (d/'drift.json').exists() and set(ea.read(d/'drift.json')) >= {'orig', 'C2'}
    assert 'ea1_selection_frozen' not in json.dumps(sel)


def test_select_is_idempotent_and_refuses_incomplete_chains(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    n = len(fake.log)
    ea.stage_select(m)                                                                           # registers what exists, never recomputes
    assert len(fake.log) == n
    m2 = tiny(tmp_path/'inc')
    ea.run_chain(m2, 0, eager())
    ea.run_chain(m2, 1, eager(), target=4)
    patch_flow(monkeypatch, m2, fake)
    with pytest.raises(ValueError):
        ea.stage_select(m2)                                                                      # chain 1 stopped at 4, chain 2 never started
    assert not (tmp_path/'inc'/'selection_frozen.json').exists()


def test_adapt_needs_a_committed_select_and_freezes_a_plus_and_a_plusplus_together(tmp_path, monkeypatch, base_flow):
    use(base_flow)
    d0, m0, fake0 = base_flow['dir'], base_flow['m'], base_flow['fake']
    plus = ea.read(d0/'selection_plus_frozen.json')
    state = ea.read(d0/'chains'/'plus.json')
    sel = ea.read(d0/'selection_frozen.json')
    assert [r['gen'] for r in state['records']] == [0, 2, 4] and state['budget']['start_theta_sha256'] == sel['A_fresh_orig']['theta_sha256']
    assert np.array_equal(np.asarray(state['records'][0]['theta']), np.asarray(sel['A_fresh_orig']['theta']))                  # starts from the frozen A_fresh_orig
    i, j = ea.select_plus(state['records'], m0['params']['plus']['pp_min_gen'])
    assert plus['A_plus']['theta'] == state['records'][i]['theta'] and plus['A_plusplus']['theta'] == state['records'][j]['theta']
    assert plus['collapsed'] == (state['records'][i]['gen'] == 0) and plus['same_theta'] == (i == j) and (plus['A_plus']['theta'] == plus['A_plusplus']['theta']) == (not plus['collapsed'])
    assert plus['selection_frozen_sha256'] == ea.sha(d0/'selection_frozen.json') and plus['plus_chain_sha256'] == ea.sha(d0/'chains'/'plus.json')
    assert fake0.seeds('adapt') <= set(m0['seeds']['EA2_AT']) | set(m0['seeds']['EA2_SEL']) and not fake0.seeds('adapt') & set(m0['seeds']['EA2_TEST'])
    train_calls = [sd for st, n, t, sd in fake0.log if st == 'adapt' and n == 7]                  # lam 6 + honest
    assert train_calls and all(set(sd) <= set(m0['seeds']['EA2_AT']) for sd in train_calls)
    ea.verify_plus_freeze(m0, ea.read(d0/'selection_frozen.json'))
    # adapt on a directory where select is not committed / not registered
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    (d/'selection_plus_frozen.json').unlink()
    (d/'chains'/'plus.json').unlink()
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: False)
    with pytest.raises(ValueError, match='not committed'):
        ea.stage_adapt(m)
    assert not (d/'chains'/'plus.json').exists() and not fake.log                                  # nothing was simulated
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    (d/'freeze_record.json').unlink()
    with pytest.raises(ValueError, match='not frozen'):
        ea.stage_adapt(m)
    shutil.copy(base_flow['dir']/'freeze_record.json', d/'freeze_record.json')
    (d/'smoke_summary.json').write_text(json.dumps(dict(graph_equals_eager=False)))
    with pytest.raises(ValueError, match='graph != eager'):
        ea.stage_adapt(m)
    good = dict(graph_equals_eager=True, code_sha256=ea.code_hashes())
    for bad_code in (None, {}, dict(ea.code_hashes(), **{'scripts/ea2.py': '0'*64}), dict(ea.code_hashes(), **{'scripts/ea2_launch.sh': '0'*64}),
                     dict(ea.code_hashes(), **{'tests/direction2/test_ea2.py': '0'*64})):
        (d/'smoke_summary.json').write_text(json.dumps(dict(good, code_sha256=bad_code)))
        with pytest.raises(ValueError, match='smoke again'):
            ea.stage_adapt(m)                                                                    # a smoke of other code (any of the three files) does not pass train / adapt
        with pytest.raises(ValueError, match='smoke again'):
            ea.require_smoke_summary(m)
    (d/'smoke_summary.json').write_text(json.dumps(good))
    ea.stage_adapt(m)                                                                            # the fake simulator is deterministic: same outcome as the original run
    again = ea.read(d/'selection_plus_frozen.json')
    assert {k: v for k, v in again.items() if k != 'plus_chain_sha256'} == {k: v for k, v in plus.items() if k != 'plus_chain_sha256'}     # chain file carries wall-clock times
    n = len(fake.log)
    ea.stage_adapt(m)                                                                            # idempotent
    assert len(fake.log) == n and 'adapt' in ea.stages_done()


def test_plus_freeze_verification_recomputes_the_selection_from_its_chain(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    sel = ea.read(d/'selection_frozen.json')
    plus = ea.read(d/'selection_plus_frozen.json')
    ea.verify_plus_freeze(m, sel)
    other = copy.deepcopy(plus)
    other['A_plus']['theta'][0] += 1e-9
    other['A_plus']['theta_sha256'] = ea.theta_id(other['A_plus']['theta'])
    ea.write(d/'selection_plus_frozen.json', other)
    with pytest.raises(ValueError, match='recomputed|chain'):
        ea.verify_plus_freeze(m, sel)                                                            # a theta that is not the one the chain selects
    ea.write(d/'selection_plus_frozen.json', dict(plus, collapsed=not plus['collapsed']))
    with pytest.raises(ValueError):
        ea.verify_plus_freeze(m, sel)
    ea.write(d/'selection_plus_frozen.json', plus)
    chain = ea.read(d/'chains'/'plus.json')
    chain['records'][1]['G_SEL'] += 1.
    ea.write(d/'chains'/'plus.json', chain)
    with pytest.raises(ValueError):
        ea.verify_plus_freeze(m, sel)                                                            # chain changed after the freeze
    ea.write(d/'chains'/'plus.json', ea.read(base_flow['dir']/'chains'/'plus.json'))
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: not rel.endswith('selection_plus_frozen.json'))
    with pytest.raises(ValueError, match='not committed'):
        ea.verify_plus_freeze(m, sel)


def test_testcheck_covers_all_six_policy_lists_only_on_the_first_sel_seeds_and_fails_on_any_difference(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    rec = ea.read(d/'testcheck.json')
    assert rec['passed'] and [c['name'] for c in rec['checks']] == ['main_orig', 'plus_orig', 'plus_C2', 'aux_C2', 'sec_orig', 'sec_C2']
    assert [c['n_policies'] for c in rec['checks']] == [3, 5, 5, 4, 16, 16] and rec['tolerance'].startswith('exact')
    assert base_flow['fake'].seeds('testcheck') == set(m['seeds']['EA2_SEL'][:4]) and not base_flow['fake'].seeds('testcheck') & set(m['seeds']['EA2_TEST'])
    assert rec['plus_frozen_sha256'] == ea.sha(d/'selection_plus_frozen.json') and rec['freeze_record_sha256'] == ea.sha(d/'freeze_record.json')
    inner = fake.__call__
    monkeypatch.setattr(ea, 'sim_eager', lambda *a: (lambda r: dict(r, snap=dict(r['snap'], util_b=(np.asarray(r['snap']['util_b'])+1e-12).tolist())))(inner(*a)))
    with pytest.raises(RuntimeError, match='failed'):
        ea.run_testcheck()
    assert ea.read(d/'testcheck.json')['passed'] is False
    n = len(fake.log)
    with pytest.raises(ValueError):
        ea.run_test('main')
    assert not (d/'test').exists() and len(fake.log) == n                                        # no ledger, no test seed touched
    ea.MODE.update(strict=False)
    with pytest.raises(RuntimeError, match='strict'):
        ea.run_testcheck()


def test_test_refuses_stale_testcheck_unfrozen_plus_and_a_graph_failure_precedes_the_ledger(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    sel = ea.read(d/'selection_frozen.json')
    plus = ea.read(d/'selection_plus_frozen.json')
    for key, val in (('manifest_sha256', 'other'), ('freeze_record_sha256', 'other'), ('plus_frozen_sha256', 'other')):
        t = ea.read(d/'testcheck.json')
        ea.write(d/'testcheck.json', dict(t, **{key: val}))
        with pytest.raises(ValueError):
            ea.require_testcheck(m, sel, plus)
        ea.write(d/'testcheck.json', t)
    t = ea.read(d/'testcheck.json')
    ea.write(d/'testcheck.json', dict(t, path=dict(t['path'], graph=True)))
    with pytest.raises(ValueError):
        ea.require_testcheck(m, sel, plus)                                                       # different execution path
    ea.write(d/'testcheck.json', t)
    # the plus freeze changes after testcheck: the test refuses before the ledger
    ea.write(d/'selection_plus_frozen.json', dict(plus, extra=1))
    n = len(fake.log)
    with pytest.raises(ValueError):
        ea.run_test('main')
    ea.write(d/'selection_plus_frozen.json', plus)
    assert not (d/'test').exists() and len(fake.log) == n
    ea.MODE.update(device='cuda', graph=True)                                                    # pretend cuda graph path; the pre-capture raises
    ea.write(d/'testcheck.json', dict(t, path=dict(t['path'], graph=True, device='cuda')))
    monkeypatch.setattr(ea, 'sim', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('CUDA graph capture failed')))
    with pytest.raises(RuntimeError, match='graph'):
        ea.run_test('main')
    assert not (d/'test').exists()                                                               # failed before the ledger; no eager fallback exists


def test_test_runs_in_three_parts_each_evaluation_once_with_plus_hash_in_the_plus_contracts(tmp_path, monkeypatch, base_flow):
    use(base_flow)
    m, d0, fake0 = base_flow['m'], base_flow['dir'], base_flow['fake']
    led = ea.read(d0/'test'/'ledger.json')
    assert set(led) == {'main_orig', 'plus_orig', 'plus_C2', 'aux_C2', 'sec_orig', 'sec_C2'} and all(v['status'] == 'completed' for v in led.values())
    test_calls = [(n, sd) for st, n, t, sd in fake0.log if st == 'test']
    assert len(test_calls) == 6*2 and [n for n, _ in test_calls] == [3, 3, 5, 5, 5, 5, 4, 4, 16, 16, 16, 16]          # chunks of 4 seeds, in the registered order
    assert {s for _, sd in test_calls for s in sd} == set(m['seeds']['EA2_TEST']) and all(len(sd) == 4 for _, sd in test_calls)
    plus_sha = ea.sha(d0/'selection_plus_frozen.json')
    for name in led:
        con = ea.read(d0/'test'/f'{name}_raw.json')['contract']
        assert (con.get('plus_frozen_sha256') == plus_sha) == name.startswith('plus_') and con['freeze_record_sha256'] == ea.sha(d0/'freeze_record.json')
        assert con['seeds'] == m['seeds']['EA2_TEST'] and con['n_policies'] == {'main_orig': 3, 'plus_orig': 5, 'plus_C2': 5, 'aux_C2': 4, 'sec_orig': 16, 'sec_C2': 16}[name]
    sel, plus = ea.read(d0/'selection_frozen.json'), ea.read(d0/'selection_plus_frozen.json')
    th = lambda ts: [ea.theta_id(t) for t in ts]       # noqa: E731
    sp = {e['name']: e['theta'] for e in sel['pool']}
    assert ea.read(d0/'test'/'main_orig_raw.json')['contract']['theta_sha256'] == th([sel['astar']['theta'], sel['A_fresh_orig']['theta']])
    assert ea.read(d0/'test'/'plus_orig_raw.json')['contract']['theta_sha256'] == th([sel['astar']['theta'], sel['A_fresh_orig']['theta'], plus['A_plus']['theta'], plus['A_plusplus']['theta']])
    assert ea.read(d0/'test'/'plus_C2_raw.json')['contract']['theta_sha256'] == th([sel['astar']['theta'], plus['A_plus']['theta'], plus['A_plusplus']['theta'], sel['ea1_A_fresh']['theta']])
    assert ea.read(d0/'test'/'aux_C2_raw.json')['contract']['theta_sha256'] == th([sel['astar']['theta'], sel['A_fresh_orig']['theta'], sel['ea1_A_fresh']['theta']])
    assert ea.read(d0/'test'/'sec_orig_raw.json')['contract']['theta_sha256'] == th([sp[n] for n in sp]) == ea.read(d0/'test'/'sec_C2_raw.json')['contract']['theta_sha256']
    assert ea.read(d0/'test'/'plus_C2_raw.json')['contract']['cfg'] == m['params']['c2_cfg'] and ea.read(d0/'test'/'plus_orig_raw.json')['contract']['cfg'] == m['params']['base_cfg']
    # nothing is simulated again, whatever part is run
    m2, d, fake = clone(base_flow, 'dir', tmp_path, monkeypatch)
    for part in ('main', 'plus', 'rest'):
        ea.run_test(part)
    assert fake.log == []
    with pytest.raises(ValueError):
        ea.run_test('everything')


def test_plus_and_rest_need_the_committed_primary_result_first_and_main_orig_exists(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'pre_test', tmp_path, monkeypatch)
    for part in ('plus', 'rest'):
        with pytest.raises(ValueError, match='result.json'):
            ea.run_test(part)                                                                    # no result.json yet
    assert not (d/'test').exists() and fake.log == []
    ea.run_test('main')
    res = ea.read(d/'result.json')
    assert res['schema'] == 'ea2-result/1' and res['consequence_row'] in ('R1', 'R2', 'R3', 'R4') and res['n'] == 8 and res['A_fresh_orig'].startswith('ea2_fresh_run')
    assert set(ea.read(d/'test'/'ledger.json')) == {'main_orig'} and not (d/'plus.json').exists() and not (d/'secondary.json').exists()
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: not rel.endswith('result.json'))
    n = len(fake.log)
    for part in ('plus', 'rest'):
        with pytest.raises(ValueError, match='committed'):
            ea.run_test(part)                                                                    # result.json is not committed and clean
    assert len(fake.log) == n and set(ea.read(d/'test'/'ledger.json')) == {'main_orig'}
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    ea.write(d/'result.json', dict(res, manifest_sha256='other'))
    with pytest.raises(ValueError, match='another manifest'):
        ea.run_test('plus')
    ea.write(d/'result.json', res)
    ea.run_test('plus')
    assert set(ea.read(d/'test'/'ledger.json')) == {'main_orig', 'plus_orig', 'plus_C2'} and (d/'plus.json').exists() and not (d/'secondary.json').exists()
    # a re-run of the primary part may only reproduce the identical result
    ea.write(d/'result.json', dict(res, mean_delta=res['mean_delta']+1))
    with pytest.raises(ValueError, match='differs'):
        ea.run_test('main')


def test_result_secondary_and_plus_content_follow_the_registered_definitions(base_flow):
    use(base_flow)
    d, m = base_flow['dir'], base_flow['m']
    sel, plus = ea.read(d/'selection_frozen.json'), ea.read(d/'selection_plus_frozen.json')
    raw = {n: ea.read(d/'test'/f'{n}_raw.json') for n in ('main_orig', 'plus_orig', 'plus_C2', 'aux_C2', 'sec_orig', 'sec_C2')}
    res, sec, pl = ea.read(d/'result.json'), ea.read(d/'secondary.json'), ea.read(d/'plus.json')
    g = ea.gains_of(raw['main_orig'])
    dd = g[1]-g[0]
    assert res['mean_delta'] == float(dd.mean()) and res['LB'] == ea.lower_bound(dd) and res['UB'] == ea.upper_bound(dd) and res['eps'] == EPS
    assert ea.classify(res['LB'], res['UB']) == (res['category'], res['sub']) and res['consequence_row'] in ea.ROWS_MAIN
    assert ea.TREE_TO_ROW[res['sub'] or res['category']] == res['consequence_row'] and res['label'] in ea.LABEL.values()
    assert res['e1v2_D']['mean'] == pytest.approx(0.004236, abs=1e-6) and res['ub_over_e1v2_D'] == pytest.approx(res['UB']/res['e1v2_D']['mean'])
    names = [e['name'] for e in sel['pool']]
    so, sc = ea.gains_of(raw['sec_orig']), ea.gains_of(raw['sec_C2'])
    jo, jc = names.index(sel['A_O_prime']['name']), names.index(sel['A_C2_prime']['name'])
    dp = so[jo]-sc[jc]
    assert sec['D_pool']['mean'] == float(dp.mean()) and sec['D_pool']['LB'] == ea.lower_bound(dp) and sec['D_pool']['row'] == ea.the_row(ea.ROWS_DPOOL, sec['D_pool']['LB'], sec['D_pool']['UB'])
    assert sec['D_pool']['A_O_prime'] == sel['A_O_prime']['name'] and '15 個條目' in sec['D_pool']['text'] and 'delta' in sec['D_pool']
    mo, ac = ea.gains_of(raw['main_orig']), ea.gains_of(raw['aux_C2'])
    assert set(sec['two_by_two']['cells']) == {'a_star|a_star', 'a_star|ea1_A_fresh', 'A_fresh_orig|a_star', 'A_fresh_orig|ea1_A_fresh'}
    assert sec['two_by_two']['cells']['A_fresh_orig|ea1_A_fresh']['mean'] == float((mo[1]-ac[2]).mean()) and sec['D_sym']['mean'] == sec['two_by_two']['cells']['A_fresh_orig|ea1_A_fresh']['mean']
    assert sec['D_new']['mean'] == float((mo[0]-ac[0]).mean()) and sec['D_new']['cross_machine_reference_only'] and sec['D_new']['e1v2_D']['mean'] == pytest.approx(0.004236, abs=1e-6)
    assert sec['D_new']['ea1_D_new']['mean'] == pytest.approx(0.003970, abs=1e-6)
    for blk in ('two_by_two', 'D_sym', 'D_new'):
        assert 'row' not in sec[blk] and 'row' not in json.dumps(sec[blk])                       # 2x2 cells, D_sym, D_new never trigger a row
    assert set(sec['chains_vs_astar_orig']) == {'ea2_fresh_run0', 'ea2_fresh_run1', 'ea2_fresh_run2'} and set(sec['ea1_checkpoints_vs_astar_orig']) == {f'ea1_fresh_run{i}' for i in range(3)}
    assert set(sec['new_vs_old_orig']['A_fresh_orig']) == {'frozen_run0', 'frozen_run1', 'frozen_run2'} and len(sec['by_entry']) == 15 and 'delta_regret' in sec['honest_service']
    cs = sec['chain_summaries']
    assert len(cs['chains']) == 3 and cs['chains'][0]['curve']['blocks'][0]['gens'] == [1, 6] and [g_ for g_, _ in cs['plus']['trajectory']] == [0, 2, 4]
    po, pc = ea.gains_of(raw['plus_orig']), ea.gains_of(raw['plus_C2'])
    assert pl['delta_plus']['mean'] == float((po[2]-po[0]).mean()) and pl['delta_plusplus']['mean'] == float((po[3]-po[0]).mean()) and pl['delta_plus_minus_delta']['mean'] == float((po[2]-po[1]).mean())
    assert pl['collapsed'] == plus['collapsed'] and pl['delta_plus']['row'] == ea.the_row(ea.ROWS_PLUS, plus['A_plus']['gen'], pl['delta_plus']['LB'], pl['delta_plus']['UB'])
    assert pl['delta_plusplus']['row'] == ea.the_row(ea.ROWS_PP, plus['A_plus']['gen'], pl['delta_plusplus']['LB'], pl['delta_plusplus']['UB'])
    assert (pl['delta_plus']['row'] == 'R8') == plus['collapsed'] and (pl['delta_plusplus']['row'] == 'R13') == (not plus['collapsed'])
    assert set(pl['cross_setting']) == {f'{r}|{c}' for r in ('A_plus', 'A_plusplus') for c in ('a_star', 'A_plus', 'A_plusplus', 'ea1_A_fresh')}
    assert pl['cross_setting']['A_plus|ea1_A_fresh']['mean'] == float((po[2]-pc[3]).mean()) and 'row' not in pl['cross_setting']['A_plus|a_star']
    cons = ea.consequences(res, sec, pl)
    assert set(cons['rows']) == {'main', 'd_pool', 'plus', 'plusplus'} and len(set(cons['rows'].values())) == 4


def test_batch_independence_check_detects_one_bit_and_equal_theta_rows():
    def raw(g):
        u = np.zeros((len(g)+1, 3, 3))
        u[1:, :, 1] = np.asarray(g)
        return dict(T_score=1, snap=dict(util_b=u.tolist(), reg_b=np.zeros((len(g)+1, 3)).tolist(), pen=np.zeros((len(g)+1, 3, 3)).tolist()))
    a, b = [[1., 2., 3.], [4., 5., 6.]], [[1., 2., 3.], [7., 8., 9.]]
    ta, tb, tc = np.full(55, .1), np.full(55, .2), np.full(55, .3)
    specs = dict(g1=('orig', [ta, tb]), g2=('orig', [ta, tc]), g3=('C2', [ta, tc]), g4=('orig', [tb, tb]))
    assert ea.batch_check(specs, dict(g1=raw(a), g2=raw(b))) == []
    c = copy.deepcopy(b)
    c[0][0] = np.nextafter(1., 2.)                                                               # one unit in the last place in the shared policy
    bad = ea.batch_check(specs, dict(g1=raw(a), g2=raw(c)))
    assert len(bad) == 1 and bad[0]['first'] == 'g1' and bad[0]['second'] == 'g2' and bad[0]['setting'] == 'orig'
    assert ea.batch_check(specs, dict(g1=raw(a), g3=raw(c))) == []                               # another setting is another evaluation
    assert ea.batch_check(specs, dict(g4=raw([[1., 2., 3.], [1., 2., 3.]]))) == []              # A+ and A++ with equal parameters: identical rows pass
    assert len(ea.batch_check(specs, dict(g4=raw([[1., 2., 3.], c[0]])))) == 1                   # ... and must be identical


def test_a_batch_independence_mismatch_stops_but_the_primary_result_stands(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'after_main', tmp_path, monkeypatch)
    res = ea.read(d/'result.json')

    def bump_a_star(f, pols, cfg, u):
        if len(pols) == 5 and cfg['tol'] == 2.:                                                  # plus_orig only: the a* row moves by one tiny step
            u[1, :, 1] += 1e-9
    fake.perturb = bump_a_star
    with pytest.raises(ea.BatchIndependenceError, match='STOP'):
        ea.run_test('plus')
    bi = ea.read(d/'batch_independence_plus.json')
    assert bi['mismatches'] and bi['mismatches'][0]['first'] == 'main_orig' and bi['mismatches'][0]['second'] == 'plus_orig'
    assert ea.read(d/'result.json') == res and not (d/'plus.json').exists()                      # the primary result stands, nothing else is published


def test_an_interrupted_evaluation_blocks_until_an_authorised_reset_and_is_then_run_once_more(tmp_path, monkeypatch, base_flow):
    m, d, fake = clone(base_flow, 'after_main', tmp_path, monkeypatch)
    real = fake.__call__
    state = {'n': 0}

    def dying(m_, cfg, pols, sd, T, be=None):
        state['n'] += 1
        if state['n'] == 2:
            raise OSError('connection lost')                                                    # the machine goes away in the middle of plus_orig
        return real(m_, cfg, pols, sd, T, be)
    monkeypatch.setattr(ea, 'sim', dying)
    with pytest.raises(OSError):
        ea.run_test('plus')
    assert ea.read(d/'test'/'ledger.json')['plus_orig']['status'] == 'started'
    with pytest.raises(RuntimeError, match='already started'):
        ea.run_test('plus')
    t = ea.read(d/'test'/'ledger.json')['plus_orig']['started_unix']
    with pytest.raises(PermissionError):
        ea.main(['ledger-reset', '--eval', 'plus_orig', '--reason', 'connection lost', '--confirm-started-unix', repr(t), '--device', 'cpu'])    # no authorisation flag
    with pytest.raises(ValueError, match='started_unix'):
        ea.main(['ledger-reset', '--eval', 'plus_orig', '--reason', 'connection lost', '--user-authorized', '--device', 'cpu'])                    # not confirmed
    assert ea.main(['ledger-reset', '--eval', 'plus_orig', '--reason', 'connection lost', '--user-authorized', '--confirm-started-unix', repr(t), '--device', 'cpu']) == 0
    assert ea.read(d/'test'/'ledger_aborted.json')[0]['reason'] == 'connection lost'
    monkeypatch.setattr(ea, 'sim', real)
    ea.run_test('plus')
    assert set(ea.read(d/'test'/'ledger.json')) == {'main_orig', 'plus_orig', 'plus_C2'} and (d/'plus.json').exists()


def test_report_recomputes_everything_from_the_raw_files_and_flags_any_difference(tmp_path, monkeypatch, base_flow, capsys):
    m, d, fake = clone(base_flow, 'dir', tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ea.stage_report('some-host')                                                             # the tag is never a machine name
    assert ea.stage_report('local') == 0
    out = capsys.readouterr().out
    rc = ea.read(d/'report_check_local.json')
    assert rc['matches'] and rc['parts'] == dict(result=True, plus=True, secondary=True, batch_independent=True) and set(rc['consequences']['rows']) == set(ea.GROUPS)
    assert rc['consequences']['rows']['main'] in out and not list(d.glob('report_check_*_*'))
    # change a saved result: the recomputation from raw disagrees
    res = ea.read(d/'result.json')
    ea.write(d/'result.json', dict(res, mean_delta=res['mean_delta']+1e-6))
    assert ea.stage_report('shared') == 1
    assert ea.read(d/'report_check_shared.json')['parts']['result'] is False and ea.read(d/'report_check_shared.json')['matches'] is False
    ea.write(d/'result.json', res)
    # change a raw file: the recomputed numbers differ from the saved ones (the report works from raw, not from the saved result)
    r = ea.read(d/'test'/'main_orig_raw.json')
    r['snap']['util_b'][2][0][1] += .5
    ea.write(d/'test'/'main_orig_raw.json', r)
    assert ea.stage_report('local') == 1 and ea.read(d/'report_check_local.json')['parts']['result'] is False
    r = ea.read(base_flow['dir']/'test'/'main_orig_raw.json')
    ea.write(d/'test'/'main_orig_raw.json', r)
    ea.write(d/'plus.json', dict(ea.read(d/'plus.json'), collapsed=not ea.read(d/'plus.json')['collapsed']))
    assert ea.stage_report('local') == 1 and ea.read(d/'report_check_local.json')['parts']['plus'] is False


# ------------------------------------------------------------------ CLI
def test_official_manifest_refuses_non_cuda_device_for_stage_commands_and_unknown_stages(monkeypatch):
    for argv in (['train'], ['select'], ['adapt'], ['testcheck'], ['test', '--part', 'main']):
        with pytest.raises(SystemExit) as e:
            ea.main(argv+['--device', 'cpu'])
        assert 'cuda' in str(e.value)
    for argv in (['paracheck'], ['postcheck'], ['bogus']):
        with pytest.raises(SystemExit):
            ea.main(argv)                                                                        # no paracheck / postcheck: lanes = 1 is fixed
    with pytest.raises(SystemExit, match='--part'):
        ea.main(['test'])
    with pytest.raises(SystemExit, match='--tag'):
        ea.main(['report'])
    with pytest.raises(SystemExit):
        ea.main(['test', '--part', 'main', '--smoke'])
    monkeypatch.setattr(ea, 'stage_report', lambda tag: 0)
    assert ea.main(['report', '--tag', 'local', '--device', 'cpu']) == 0


# ------------------------------------------------------------------ launcher
def launcher_root(tmp_path, space='/data', avail=50*2**30, smi='exit 0'):
    root = tmp_path/'repo'
    (root/'scripts').mkdir(parents=True)
    shutil.copy(ea.ROOT/'scripts/ea2_launch.sh', root/'scripts/ea2_launch.sh')
    (root/'bin').mkdir()
    tools = {'nvidia-smi': f'#!/bin/sh\n{smi}\n', 'nice': '#!/bin/sh\n[ "$1" = "-n" ] && shift 2\nexec "$@"\n', 'findmnt': f'#!/bin/sh\necho {space}\n',
             'df': f'#!/bin/sh\necho Avail\necho {avail}\n'}
    for tool, body in tools.items():
        (root/'bin'/tool).write_text(body)
        (root/'bin'/tool).chmod(0o755)
    (root/'tests/direction2').mkdir(parents=True)
    fake = root/'fakepy'
    fake.write_text('#!/bin/sh\n[ "$1" = "-m" ] && { touch cpu_tests_ran; [ -n "${CPU_FAIL:-}" ] && { echo "FAILED tests/x.py::t - boom"; exit 1; }; exit 0; }\necho "$@" >> args\nn=$(cat count 2>/dev/null || echo 0); n=$((n+1)); echo $n > count\n[ "$n" -lt 3 ] && exit 75\nexit ${FINAL_RC:-0}\n')
    fake.chmod(0o755)
    return root, dict(os.environ, PATH=f'{root}/bin:'+os.environ['PATH'], PYTHON=str(fake))


def run_launch(root, env, *args):
    return subprocess.run(['bash', 'scripts/ea2_launch.sh', *args], cwd=root, env=env, capture_output=True, text=True)


def test_launcher_auto_resumes_train_and_adapt_on_75_passes_other_codes_and_never_retries_test(tmp_path):
    root, env = launcher_root(tmp_path)
    for mode in ('train', 'adapt'):
        (root/'count').write_text('0')
        r = run_launch(root, env, mode)
        assert r.returncode == 0 and (root/'count').read_text().strip() == '3'
        assert (root/f'results/direction2/runlogs/ea2_{mode}.log').read_text().count('resume #') == 2
    (root/'count').write_text('2')
    r = run_launch(root, env.copy() | dict(FINAL_RC='7'), 'select')
    assert r.returncode == 7                                                                    # other exit codes pass through unchanged
    (root/'count').write_text('2')
    r = run_launch(root, dict(env, FINAL_RC='7'), 'test', '--part', 'main')
    assert r.returncode == 7 and (root/'count').read_text().strip() == '3'                      # test: no retry, no resume
    assert (root/'args').read_text().strip().splitlines()[-1] == 'scripts/ea2.py test --part main'
    (root/'count').write_text('2')
    r = run_launch(root, dict(env, FINAL_RC='7'), 'testcheck')
    assert r.returncode == 7 and (root/'count').read_text().strip() == '3'
    for mode in ('paracheck', 'postcheck', 'nonsense'):
        r = run_launch(root, env, mode)
        assert r.returncode == 2 and 'usage' in r.stdout                                         # E-A1's lane stages do not exist here
    (root/'bin'/'nvidia-smi').write_text('#!/bin/sh\necho 1234\n')
    r = run_launch(root, env, 'train')
    assert r.returncode == 3 and 'GPU occupied' in r.stdout                                      # another program on the GPU: refuse


def test_launcher_looks_at_the_dedicated_space_not_at_the_root_partition(tmp_path):
    root, env = launcher_root(tmp_path, space='/')
    r = run_launch(root, env, 'train')
    assert r.returncode == 3 and 'root partition' in r.stdout and not (root/'count').exists()   # the project directory is on the root partition: refused
    root, env = launcher_root(tmp_path/'b', space='/data', avail=10*2**30-1)
    r = run_launch(root, env, 'train')
    assert r.returncode == 3 and '10 GiB' in r.stdout and not (root/'count').exists()           # one byte under 10 GiB on the dedicated space: refused (no rounding up)
    root, env = launcher_root(tmp_path/'c', space='', avail=50*2**30)
    r = run_launch(root, env, 'select')
    assert r.returncode == 3 and 'root partition' in r.stdout                                    # unknown mount: refused
    root, env = launcher_root(tmp_path/'d', space='/data', avail=10*2**30)
    (root/'count').write_text('2')
    assert run_launch(root, env, 'select').returncode == 0 and (root/'count').read_text().strip() == '3'
    root, env = launcher_root(tmp_path/'e', space='/')
    assert run_launch(root, env, 'status').returncode != 3 and (root/'count').exists()          # CPU-only modes do not look at the space (the stub python ran)
    # the physical path (pwd -P) is what findmnt is asked about: a link into another space
    real = tmp_path/'real'
    root, env = launcher_root(real)
    (root/'bin'/'findmnt').write_text('#!/bin/sh\n[ "$2" = "' + str(root.resolve()) + '" ] && echo /data || echo /\n')
    link = tmp_path/'link'
    link.symlink_to(root)
    (root/'count').write_text('2')
    r = subprocess.run(['bash', 'scripts/ea2_launch.sh', 'select'], cwd=link, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_launcher_busy_means_another_account_only_and_test_modes_also_wait_for_our_own_gpu_programs(tmp_path):
    (tmp_path/'nvidia-smi').write_text(f'#!/bin/sh\necho {os.getpid()}\n')
    (tmp_path/'nvidia-smi').chmod(0o755)
    src = (ea.ROOT/'scripts/ea2_launch.sh').read_text()

    def fn(name):
        return src[src.index(name+'() {'):src.index('\n}\n', src.index(name+'() {'))+3]
    env = dict(os.environ, PATH=f'{tmp_path}:'+os.environ['PATH'])
    res = {}
    for label, body in (('own', f'echo {os.getpid()}'), ('other', 'echo 1'), ('gone', 'echo 999999999'), ('none', 'true')):
        (tmp_path/'nvidia-smi').write_text(f'#!/bin/sh\n{body}\n')
        res[label] = (subprocess.run(['bash', '-c', fn('busy')+'\nbusy'], env=env).returncode, subprocess.run(['bash', '-c', fn('own_busy')+'\nown_busy'], env=env).returncode)
    assert res == dict(own=(1, 0), other=(0, 1), gone=(0, 1), none=(1, 1))                      # busy: others only; own_busy: our own programs only
    root, env2 = launcher_root(tmp_path/'r', smi=f'echo {os.getpid()}')
    for mode in ('test', 'testcheck'):
        r = run_launch(root, env2, mode, '--part', 'main')
        assert r.returncode == 3 and 'this account' in r.stdout and not (root/'count').exists()  # our own GPU program is running: the one-shot ledger is not put at risk
    (root/'count').write_text('2')
    assert run_launch(root, dict(env2, EA2_ACCEPT_OWN_GPU_RISK='1'), 'test', '--part', 'main').returncode == 0
    (root/'count').write_text('2')
    assert run_launch(root, env2, 'select').returncode == 0                                      # train / select / adapt are not blocked by our own programs


# ------------------------------------------------------------------ GPU (shared experiment machine only)
@pytest.mark.gpu
def test_gpu_small_acceptance_main_and_plus_graph_equals_eager(tmp_path):
    """T=48, K=4: graphs ARE captured and replayed for the main chain setting and the re-adaptation setting."""
    cuda_required()
    m = tiny(tmp_path, gens=3)
    m['params'].update(T_train=48, T_val=48, exclude=8)
    ea.MODE.update(device='cuda', K=4)
    th = start_theta()
    A = ea.trace_chain(m, 0, ea.Backend('cuda', None, None), 3)
    be = ea.make_backend('cuda')
    with be:
        B = ea.trace_chain(m, 0, be, 3)
    assert be.gs.stats['builds'] >= 1 and be.gs.stats['replays'] >= 1 and ea.trace_diff(A, B) == []
    A = ea.trace_plus(m, th, ea.Backend('cuda', None, None), 3)
    be = ea.make_backend('cuda')
    with be:
        B = ea.trace_plus(m, th, be, 3)
    assert ea.trace_diff(A, B) == []


@pytest.mark.gpu
def test_gpu_graph_path_has_no_fallback_and_chunked_equals_eager(tmp_path):
    cuda_required()
    m = tiny(tmp_path)
    ea.MODE.update(device='cuda', graph=True, strict=True)
    pols = [ea.honest_pol(), ea.mlp(np.full(55, .01))]
    sd = m['seeds']['EA2_SEL'][:4]
    got = ea.simulate_chunked(m, m['params']['base_cfg'], pols, sd, 2)
    ref = ea.sim_eager(m, m['params']['base_cfg'], pols, sd, m['params']['T_val'])
    for k in ('util_b', 'reg_b', 'pen'):
        assert ea.bitwise_equal(got['snap'][k], ref['snap'][k])
    assert ea.default_backend().gs is not None and ea.default_backend().gs.stats['builds'] >= 1


# ------------------------------------------------------------------ identities, not indices: an oracle simulator keyed by the policy's own parameters
POOL_NAMES = ['frozen_run0', 'frozen_run1', 'frozen_run2', 'orig_adapt0', 'C1_adapt0', 'C1_adapt1', 'C1_adapt2', 'C2_adapt0', 'C2_adapt2',
              'ea1_fresh_run0', 'ea1_fresh_run1', 'ea1_fresh_run2', 'ea2_fresh_run0', 'ea2_fresh_run1', 'ea2_fresh_run2']
ID = {n: i+1 for i, n in enumerate(POOL_NAMES)}                      # parameter identity 1..15; A+ = 20, A++ = 21
GO = {i: .0300+.00041*((3*i) % 13) for i in list(range(1, 16))+[20, 21]}
GC = {i: .0260+.00037*((5*i) % 11) for i in list(range(1, 16))+[20, 21]}


def ident(i):
    v = np.zeros(55)
    v[0] = i/1000.
    return [float(x) for x in v]


def oracle_sim(m, cfg, pols, sd, T, be=None):
    """G of a policy depends on its own parameters (identity in theta[0]) and the setting only - never on its position in the list."""
    ts = T-m['params']['exclude']
    table = GO if cfg['tol'] == m['params']['base_cfg']['tol'] else GC
    u = np.zeros((len(pols), len(sd), 3))
    for k, q in enumerate(pols[1:], 1):
        u[k, :, 1] = table[int(round(float(np.asarray(q.theta)[0])*1000))]*ts
    return dict(T_score=ts, snap=dict(util_b=u.tolist(), reg_b=np.ones((len(pols), len(sd))).tolist(), pen=np.zeros((len(pols), len(sd), 3)).tolist()))


def oracle_world(tmp_path, monkeypatch, collapsed):
    m = tiny(tmp_path)
    ent = lambda n, i: dict(name=n, theta=ident(i), theta_sha256=ea.theta_id(ident(i)))       # noqa: E731
    sel = dict(astar=ent('orig_adapt0', ID['orig_adapt0']), ea1_A_fresh=ent('ea1_fresh_run0', ID['ea1_fresh_run0']), A_fresh_orig=dict(ent('ea2_fresh_run1', ID['ea2_fresh_run1']), gen=150),
               pool=[dict(ent(n, ID[n]), aliases=[]) for n in POOL_NAMES], A_O_prime=dict(name='C1_adapt2'), A_C2_prime=dict(name='C2_adapt0'),
               chains=[ent(f'ea2_fresh_run{i}', ID[f'ea2_fresh_run{i}']) for i in range(3)])
    ap = ent('x', ID['ea2_fresh_run1']) if collapsed else ent('x', 20)
    app = ent('x', 21) if collapsed else ent('x', 20)
    plus = dict(A_plus=dict(gen=0 if collapsed else 40, theta=ap['theta']), A_plusplus=dict(gen=60 if collapsed else 40, theta=app['theta']), collapsed=collapsed)
    ea.write(tmp_path/'selection_frozen.json', sel)
    ea.write(tmp_path/'selection_plus_frozen.json', plus)
    monkeypatch.setattr(ea, 'sim', oracle_sim)
    monkeypatch.setattr(ea, 'chain_summaries', lambda m_: {})
    raws = {n: ea.simulate_chunked(m, {'orig': m['params']['base_cfg'], 'C2': m['params']['c2_cfg']}[w], pols, m['seeds']['EA2_TEST'], 4)
            for n, w, _, pols in ea.test_pols(sel, plus)}
    return m, sel, plus, raws


@pytest.mark.parametrize('collapsed', [True, False])
def test_every_reported_quantity_is_the_difference_of_the_named_policies_by_identity(tmp_path, monkeypatch, collapsed):
    m, sel, plus, raws = oracle_world(tmp_path, monkeypatch, collapsed)
    near = lambda x, y: x == pytest.approx(y, abs=1e-12)           # noqa: E731
    a, f, e = ID['orig_adapt0'], ID['ea2_fresh_run1'], ID['ea1_fresh_run0']
    p1, p2 = (f, 21) if collapsed else (20, 20)
    res = ea.compute_result(m, sel, raws['main_orig'])
    assert near(res['mean_delta'], GO[f]-GO[a]) and near(res['G_orig_astar'], GO[a]) and near(res['G_orig_A_fresh_orig'], GO[f])
    pl = ea.compute_plus(m, sel, plus, raws['plus_orig'], raws['plus_C2'])
    assert near(pl['delta_plus']['mean'], GO[p1]-GO[a]) and near(pl['delta_plusplus']['mean'], GO[p2]-GO[a]) and near(pl['delta_plus_minus_delta']['mean'], GO[p1]-GO[f])
    for r_, ri in (('A_plus', p1), ('A_plusplus', p2)):
        for c_, ci in (('a_star', a), ('A_plus', p1), ('A_plusplus', p2), ('ea1_A_fresh', e)):
            assert near(pl['cross_setting'][f'{r_}|{c_}']['mean'], GO[ri]-GC[ci]), (r_, c_)
    assert (pl['delta_plus']['row'] == 'R8') == collapsed and (pl['delta_plusplus']['row'] == 'R13') == (not collapsed)
    sec = ea.compute_secondary(m, sel, raws)
    assert near(sec['D_pool']['mean'], GO[ID['C1_adapt2']]-GC[ID['C2_adapt0']])
    cells = sec['two_by_two']['cells']
    assert near(cells['a_star|a_star']['mean'], GO[a]-GC[a]) and near(cells['a_star|ea1_A_fresh']['mean'], GO[a]-GC[e])
    assert near(cells['A_fresh_orig|a_star']['mean'], GO[f]-GC[a]) and near(cells['A_fresh_orig|ea1_A_fresh']['mean'], GO[f]-GC[e])
    assert near(sec['D_sym']['mean'], GO[f]-GC[e]) and near(sec['D_new']['mean'], GO[a]-GC[a])
    assert near(sec['chains_vs_astar_orig']['ea2_fresh_run2']['mean'], GO[ID['ea2_fresh_run2']]-GO[a])
    assert near(sec['new_vs_old_orig']['A_fresh_orig']['frozen_run1']['mean'], GO[f]-GO[ID['frozen_run1']])
    assert near(sec['new_vs_old_orig']['ea2_fresh_run0']['frozen_run2']['mean'], GO[ID['ea2_fresh_run0']]-GO[ID['frozen_run2']])
    assert near(sec['ea1_checkpoints_vs_astar_orig']['ea1_fresh_run2']['mean'], GO[ID['ea1_fresh_run2']]-GO[a])
    assert near(sec['by_entry']['C1_adapt1']['G_orig']['mean'], GO[ID['C1_adapt1']]) and near(sec['by_entry']['C1_adapt1']['G_C2']['mean'], GC[ID['C1_adapt1']])
    specs = {n: (w, ts) for n, w, ts, _ in ea.test_pols(sel, plus)}
    assert ea.batch_check(specs, raws) == []                                                    # also: A+ == A++ rows and A+ == A_fresh_orig rows are the same numbers


# ------------------------------------------------------------------ a flow whose A+ is not collapsed
def test_flow_with_a_plus_that_moved_away_from_the_start_goes_through_the_non_collapsed_rows(tmp_path, monkeypatch, nc_flow):
    use(nc_flow)
    d = nc_flow['dir']
    plus = ea.read(d/'selection_plus_frozen.json')
    state = ea.read(d/'chains'/'plus.json')
    assert plus['collapsed'] is False and plus['A_plus']['gen'] > 0 and plus['same_theta'] is True and plus['A_plus']['theta'] == plus['A_plusplus']['theta']
    assert plus['A_plus']['G_SEL'] == max(r['G_SEL'] for r in state['records']) and state['records'][0]['G_SEL'] < plus['A_plus']['G_SEL']
    pl = ea.read(d/'plus.json')
    assert pl['collapsed'] is False and pl['delta_plus']['row'] in ('R9', 'R10', 'R11', 'R12') and pl['delta_plusplus']['row'] == 'R13'
    assert pl['delta_plusplus']['text'] == ea.PP_TEXT['R13'] and pl['delta_plus']['text'] == ea.PLUS_TEXT[pl['delta_plus']['row']].format(
        mean=pl['delta_plus']['mean'], lb=pl['delta_plus']['LB'], ub=pl['delta_plus']['UB'])
    raw = ea.read(d/'test'/'plus_orig_raw.json')
    po = ea.gains_of(raw)
    assert ea.bitwise_equal(po[2], po[3]) and not ea.bitwise_equal(po[1], po[2])                # A++ is A+ (same numbers); A+ is not A_fresh_orig
    assert (d/'test'/'plus_orig_raw.json').exists() and raw['contract']['theta_sha256'][2] == raw['contract']['theta_sha256'][3] != raw['contract']['theta_sha256'][1]
    cons = ea.consequences(ea.read(d/'result.json'), ea.read(d/'secondary.json'), pl)
    assert cons['rows']['plus'] in ('R9', 'R10', 'R11', 'R12') and cons['rows']['plusplus'] == 'R13' and len(set(cons['rows'].values())) == 4
    m, d2, fake = clone(nc_flow, 'dir', tmp_path, monkeypatch)
    assert ea.stage_report('local') == 0 and ea.read(d2/'report_check_local.json')['consequences']['rows']['plus'] == cons['rows']['plus']


# ------------------------------------------------------------------ launcher: override, cpu-test failure, second own-GPU look
def test_launcher_space_override_is_logged_cpu_test_failure_stops_and_own_gpu_program_is_looked_for_again_after_the_tests(tmp_path):
    root, env = launcher_root(tmp_path/'a', space='/')
    r = run_launch(root, env, 'select')
    assert r.returncode == 3 and 'root partition' in r.stdout                                  # the default path is on the root partition
    (root/'bin'/'findmnt').write_text('#!/bin/sh\n[ "$2" = "/named/space" ] && echo /data || echo /\n')
    (root/'count').write_text('2')
    r = run_launch(root, dict(env, EA2_SPACE_OVERRIDE='/named/space'), 'select')
    assert r.returncode == 0 and 'overridden' in r.stdout and 'overridden' in (root/'results/direction2/runlogs/ea2_select.log').read_text()
    (root/'count').write_text('2')
    r = run_launch(root, dict(env, EA2_SPACE_OVERRIDE='/named/other'), 'select')
    assert r.returncode == 3 and (root/'count').read_text() == '2'                              # the override is checked, not trusted
    root, env = launcher_root(tmp_path/'b')
    r = run_launch(root, dict(env, CPU_FAIL='1'), 'select')
    assert r.returncode == 4 and 'FAILED tests/x.py::t' in r.stdout and 'nothing was started' in r.stdout and not (root/'count').exists()
    own = f'[ -f cpu_tests_ran ] && echo {os.getpid()}'                                          # our own GPU program shows up only while/after the CPU tests ran
    root, env = launcher_root(tmp_path/'c', smi=own)
    r = run_launch(root, env, 'test', '--part', 'main')
    assert r.returncode == 3 and 'appeared' in r.stdout and not (root/'count').exists()
    (root/'count').write_text('2')
    assert run_launch(root, dict(env, EA2_ACCEPT_OWN_GPU_RISK='1'), 'test', '--part', 'main').returncode == 0
