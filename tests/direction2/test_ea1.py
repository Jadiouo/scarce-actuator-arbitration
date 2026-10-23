"""E-A1: seeds, guards, statistics, decision tree, ledger, lanes == sequential (bitwise), post-train check, selection and test flow.
CPU only (tiny synthetic budgets; no EA1_TEST seed is ever simulated).  GPU tests (marker gpu) run on astarc only: EA1_ALLOW_GPU_TESTS=1."""
import copy
import hashlib
import json
import os
import random
import threading
import time

import numpy as np
import pytest

from arbitration.rl import train as rl_train
from scripts import e1_v2 as e1
from scripts import ea1 as ea

import torch
torch.set_num_threads(1)         # tiny CPU simulations: more intra-op threads only add contention (the production code does the same)


def cuda_required():
    """GPU tests run only on the GPU machine: skipped unless CUDA is present AND EA1_ALLOW_GPU_TESTS=1 (the local GPU is never used)."""
    if os.environ.get('EA1_ALLOW_GPU_TESTS') != '1':
        pytest.skip('GPU tests run only on astarc (EA1_ALLOW_GPU_TESTS=1)')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ea.MODE.update(smoke=False, device='cuda', graph=True, out=None, start=None, strict=False, yield_check=False, K=None)
    ea.GEN_HOOK = None
    ea._DEFAULT.clear()
    ea._YIELD['event'].clear()
    ea._YIELD['last'] = 0.0


# ------------------------------------------------------------------ tiny synthetic manifest (CPU, a few seconds)
def tiny(tmp_path, gens=6, val_every=3, post=3, lam=6, n_S=4):
    p = json.loads(json.dumps(ea.PARAMS))
    p['attacker'].update(lam=lam, n_S=n_S, gens=gens, val_every=val_every, post_check_gen=post)
    p.update(T_train=16, T_val=16, exclude=4, c2_cfg=dict(p=.5, tol=1., kref=.5, L=8.), base_cfg=dict(p=.5, tol=2., kref=.5, L=8.), test_chunk=4, testcheck_n=4, n_test=8)
    seeds = dict(EA1_TRAIN=list(range(82_000_000, 82_000_016)), EA1_SEL=list(range(82_100_000, 82_100_008)), EA1_TEST=list(range(82_400_000, 82_400_008)))
    m = dict(schema='t', smoke=False, params=p, seeds=seeds, drift=dict(seeds=list(range(72_300_000, 72_300_004))), sha256='synthetic', environment={})
    ea.MODE.update(smoke=False, out=str(tmp_path), device='cpu', graph=True, start=None, yield_check=False, K=4)
    return m


def eager():
    return ea.Backend('cpu', None, None)


def lane():
    return ea.make_backend('cpu')


def run_solo(m, runs, factory=eager, lanes=1):
    ea.run_lanes(m, runs, lanes, factory=factory)
    return {r: ea.chain_result(ea.chain_state(r)) for r in runs}


class Collect:
    """GEN_HOOK that records every generation's candidate matrix, utilities, fitness and seed block as raw bytes."""
    def __init__(self, sleep=None, fail=None, stop=None):
        self.rec, self.sleep, self.fail, self.stop, self.lock = {}, sleep, fail, stop, threading.Lock()

    def __call__(self, run, gen, rec):
        with self.lock:
            self.rec[(run, gen)] = (rec['X'].tobytes(), rec['U'].tobytes(), rec['f'].tobytes(), tuple(rec['seeds']))
        if self.sleep:
            time.sleep(self.sleep(run, gen))
        if self.fail and self.fail(run, gen):
            raise RuntimeError('synthetic lane failure')
        if self.stop and self.stop(run, gen):
            raise ea.YieldStop()


# ------------------------------------------------------------------ seed audit
def test_seed_audit_counts_collisions_and_registered_hashes():
    rep = ea.seed_report(ea.block_seeds(), ea.existing_all())
    assert (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) == (8608, 12064, 0)
    assert rep['per_block'] == dict(EA1_TRAIN=4000, EA1_SEL=512, EA1_TEST=4096)
    assert rep['new_expanded_sha256'] == '05232e843d5dd370d0c5bdfef416924990d92021d2b50c01a25ba14ec5eabe97'
    assert rep['existing_expanded_sha256'] == '7cdc62f47788f8c38b25b93e78adb2fdf695151693344b175835eefea1afbd85'
    assert 'EA1_TEST' not in ea.block_seeds(smoke=True)
    assert min(ea.expand(sum(ea.block_seeds().values(), []))) == 82_000_000 and max(ea.expand(sum(ea.block_seeds().values(), []))) == 85_401_023
    for bad in (72_300_000, 41_000_063, 72_400_000, 61_000_000-3*10**6, 72_000_000-2*10**6):     # E1 v2 AS / holdout / E1 v2 TEST / +3e6 stream / offset collision
        b = ea.block_seeds()
        b['EA1_SEL'] = b['EA1_SEL'][:-1]+[bad]
        with pytest.raises(ValueError):
            ea.seed_report(b, ea.existing_all())
    b = ea.block_seeds()
    b['EA1_SEL'] = b['EA1_SEL'][:-1]+[b['EA1_TRAIN'][0]]
    with pytest.raises(ValueError):
        ea.seed_report(b, ea.existing_all())                                   # cross-block overlap
    b = ea.block_seeds()
    b['EA1_TRAIN'] = b['EA1_TRAIN'][:-1]+[b['EA1_TRAIN'][0]]
    with pytest.raises(ValueError):
        ea.seed_report(b, ea.existing_all())                                   # duplicate inside a block
    with pytest.raises(ValueError):
        ea.seed_report(ea.block_seeds(), ea.existing_all(), extra_historical=[82_100_005])
    assert not set(ea.expand(ea.block_seeds()['EA1_SEL'])) & set(ea.expand(range(*[72_300_000, 72_300_128])))      # drift seeds are not a new block


# ------------------------------------------------------------------ stage / seed mismatch
def test_stages_only_accept_their_own_block_test_locked_testcheck_is_first_64_of_sel():
    m = dict(seeds=ea.block_seeds(), params=dict(testcheck_n=64), drift=dict(seeds=list(range(72_300_000, 72_300_128))))
    S = m['seeds']
    ea.require_seeds(S['EA1_TRAIN'][:256], 'train', m)
    ea.require_seeds(S['EA1_SEL'], 'sel', m)
    ea.require_seeds(S['EA1_SEL'][:64], 'testcheck', m)
    ea.require_seeds(S['EA1_TEST'], 'test', m, unlocked=True)
    ea.require_seeds(m['drift']['seeds'], 'drift', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_TEST'][:4], 'test', m)                                         # locked
    for stage, own in (('train', 'EA1_TRAIN'), ('sel', 'EA1_SEL'), ('test', 'EA1_TEST')):
        for other in S:
            if other != own:
                with pytest.raises(ValueError):
                    ea.require_seeds(S[other][:4], stage, m, unlocked=True)
    for bad in ([0], [2000], [41_000_000], [72_300_000], [72_400_000]):
        for stage in ('train', 'sel', 'testcheck'):
            with pytest.raises(ValueError):
                ea.require_seeds(bad, stage, m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_TRAIN'][:2]*2, 'train', m)                                     # duplicates
    with pytest.raises(ValueError):
        ea.require_seeds([], 'train', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_SEL'][:63]+[S['EA1_TEST'][0]], 'testcheck', m)                  # any TEST seed in testcheck fails
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_SEL'][:63], 'testcheck', m)                                    # must be exactly the first 64
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_SEL'][1:65], 'testcheck', m)
    with pytest.raises(ValueError):
        ea.require_seeds(S['EA1_SEL'][:8], 'drift', m)
    smoke = dict(seeds=ea.block_seeds(smoke=True), params=dict(testcheck_n=64))
    with pytest.raises(ValueError):
        ea.require_seeds([82_400_000], 'test', smoke, unlocked=True)                           # smoke manifests have no test block
    with pytest.raises(ValueError):
        ea.require_seeds([1], 'nonsense', m)


def test_testcheck_runs_only_on_first_64_sel_seeds_and_fails_on_any_difference(tmp_path, monkeypatch):
    m, sel, used = flow_env(tmp_path, monkeypatch)
    rec = ea.run_testcheck()
    flat = {s for u in used for s in u}
    assert rec['passed'] and [c['name'] for c in rec['checks']] == ['main_C2', 'main_orig', 'sec_C2', 'sec_orig']
    assert flat == set(m['seeds']['EA1_SEL'][:4]) and not flat & set(m['seeds']['EA1_TEST'])
    assert ea.read(tmp_path/'testcheck.json')['tolerance'].startswith('exact')
    inner = ea.sim_eager
    monkeypatch.setattr(ea, 'sim_eager', lambda *a: (lambda r: dict(r, snap=dict(r['snap'], util_b=(np.asarray(r['snap']['util_b'])+1e-12).tolist())))(inner(*a)))
    with pytest.raises(RuntimeError):
        ea.run_testcheck()
    assert ea.read(tmp_path/'testcheck.json')['passed'] is False
    n = len(used)
    with pytest.raises(ValueError):
        ea.run_test()
    assert not (tmp_path/'test').exists() and len(used) == n                                   # no ledger, no test seed touched


# ------------------------------------------------------------------ permutation / training blocks
def test_perm_block_function_rules_and_seed_independence():
    pool = list(range(82_000_000, 82_001_000))
    seen = set()
    for n in (1, 16, 256, 999, 1000):
        for g in range(8):
            blk = ea.train_block(pool, 0, g, n)
            assert len(blk) == n == len(set(blk)) and set(blk) <= set(pool)                   # never repeats a seed inside a block (also across the pool edge)
            assert blk == ea.train_block(pool, 0, g, n)                                        # reproducible
            seen.add(tuple(blk))
    with pytest.raises(ValueError):
        ea.train_block(pool, 0, 0, 1001)
    with pytest.raises(ValueError):
        ea.train_block(pool, 0, 0, 0)
    for run in range(3):
        expected = list(range(1000))
        random.Random(94100+run).shuffle(expected)
        assert list(ea.perm_for(1000, 94100+run)) == expected                                  # the registered permutation seed 94100+run
        old = list(range(1000))
        random.Random(9000+run).shuffle(old)
        assert expected != old                                                                 # not the original chain's order
    idx = list(range(1000))
    for run in range(3):                                                                       # same index rule as train.seed_block on the 0-999 pool
        for g in (0, 3, 5):
            assert ea.train_block(idx, run, g, 256, perm_seed=9000) == list(rl_train.seed_block(run, g, 256))
    p = ea.PARAMS['attacker']
    assert (p['opt_seed'], p['perm_seed'], p['lam'], p['n_S'], p['gens'], p['sigma0'], p['val_every']) == (94000, 94100, 256, 256, 416, .3, 50)


# ------------------------------------------------------------------ cold start chain
def test_cold_start_chain_budget_comes_from_manifest_and_resume_checks_it(tmp_path):
    m = tiny(tmp_path)
    assert not np.any(ea.make_opt(m, 0).mean) and ea.make_opt(m, 2).lam == 6
    reg = ea.PARAMS
    assert (reg['T_train'], reg['T_val'], reg['exclude'], reg['eps'], reg['n_test']) == (20000, 100000, 5000, .001, 1024)
    assert reg['c2_cfg'] == dict(p=.1, tol=6.175461735115074, kref=.4663589529547838, L=2412.0)
    b = ea.chain_budget(m, 1)
    assert b['opt_seed'] == 94001 and b['perm_seed'] == 94101 and b['start'] == 'zeros' and b['exclude'] == 4 and b['gens'] == 6 and b['val_every'] == 3
    ea.run_chain(m, 0, eager(), target=2)
    st = ea.chain_state(0)
    assert st['opt']['gen'] == 2 and len(st['curve']) == 2 and st['cfg'] == m['params']['c2_cfg']
    for edit in (lambda q: q['params']['attacker'].update(n_S=2), lambda q: q['params']['c2_cfg'].update(L=9.), lambda q: q['params'].update(T_val=20),
                 lambda q: q['params']['attacker'].update(opt_seed=1)):
        bad = copy.deepcopy(m)
        edit(bad)
        with pytest.raises(ValueError):
            ea.run_chain(bad, 0, eager(), target=3)                                            # resumed under another budget / defender
    with pytest.raises(ValueError):
        ea.run_chain(m, 0, eager(), target=7)                                                  # never extended beyond the registered generations
    ea.run_chain(m, 0, eager())                                                                # resume to the end
    assert ea.chain_state(0)['opt']['gen'] == 6 and [r['gen'] for r in ea.chain_state(0)['records']] == [3, 6]
    assert not np.array_equal(ea.chain_state(0)['opt']['mean'], np.zeros(55))


# ------------------------------------------------------------------ decision tree / statistics
def vec(mean, sd, n, seed=0):
    z = np.random.default_rng(seed).standard_normal(n)
    return mean+sd*(z-z.mean())/z.std(ddof=1)


N = 1024


def test_decision_tree_every_node():
    c = ea.decide(vec(.003, .0095, N))
    assert (c['category'], c['sub']) == ('stronger', None) and c['text'].startswith('更強')
    c = ea.decide(vec(-.003, .0095, N))
    assert (c['category'], c['sub']) == ('not_stronger', 'weaker')
    c = ea.decide(vec(.0005, .0003*np.sqrt(N), N))                                             # SE 0.0003: LB > 0, UB < eps -> small gain
    assert (c['category'], c['sub']) == ('not_stronger', 'small_gain') and c['LB'] > 0 and c['UB'] < .001
    c = ea.decide(vec(0., .0095, N))
    assert (c['category'], c['sub']) == ('not_stronger', 'flat') and c['LB'] <= 0 <= c['UB'] < .001
    c = ea.decide(vec(.001, .0095, N))
    assert c['category'] == 'insufficient'
    assert ea.decide(np.full(8, .002))['category'] == 'stronger' and ea.decide(np.zeros(8))['sub'] == 'flat'      # sigma = 0
    assert 'A_fresh' in ea.decide(vec(.001, .0095, N))['subject']


def test_decision_strict_boundaries_equal_threshold_hits_nothing():
    D = vec(.0016, .0095, N)
    lb, ub = ea.lower_bound(D), ea.upper_bound(D)
    assert ea.decide(D, eps=lb)['category'] != 'stronger'                                      # LB exactly eps is not 'stronger'
    assert ea.decide(D, eps=lb-1e-9)['category'] == 'stronger'
    assert ea.decide(D, eps=ub)['category'] != 'not_stronger'                                  # UB exactly eps is not 'not stronger'
    assert ea.decide(D, eps=ub+1e-9)['category'] == 'not_stronger'
    c = ea.classify
    eps = ea.PARAMS['eps']
    assert c(eps, 1.) == ('insufficient', None) and c(eps+1e-12, 1.)[0] == 'stronger'          # LB exactly eps
    assert c(-1., eps) == ('insufficient', None) and c(-1., eps-1e-12)[0] == 'not_stronger'     # UB exactly eps
    assert c(-1., 0.) == ('not_stronger', 'flat') and c(-1., -1e-12) == ('not_stronger', 'weaker')   # UB exactly 0: not 'weaker'
    assert c(0., eps-1e-12) == ('not_stronger', 'flat') and c(1e-12, eps-1e-12) == ('not_stronger', 'small_gain')   # LB exactly 0: not 'small_gain'
    assert c(2e-3, 3e-3) == ('stronger', None) and c(-3e-3, -2e-3) == ('not_stronger', 'weaker')
    assert ea.tcrit(1024) == pytest.approx(1.6463, abs=1e-4)
    with pytest.raises(ValueError):
        ea.decide(np.array([.0]))


def test_power_table_reproduces_the_design_table():
    rows = {r['mu']: r for r in ea.power_table(1024, .0095, .001, [-.001, -.0005, 0, .0005, .001, .0015, .002, .003])}
    assert rows[-.001]['p_ub_lt_eps'] == pytest.approx(1.0, abs=1e-3) and rows[-.001]['p_weaker'] == pytest.approx(.957, abs=.005)
    assert rows[-.0005]['p_weaker'] == pytest.approx(.515, abs=.005)
    assert rows[0]['p_ub_lt_eps'] == pytest.approx(.957, abs=.005) and rows[0]['p_weaker'] == pytest.approx(.05, abs=.003)
    assert rows[.0005]['p_ub_lt_eps'] == pytest.approx(.515, abs=.005)
    assert rows[.001]['p_stronger'] == pytest.approx(.05, abs=.003) and rows[.001]['p_ub_lt_eps'] == pytest.approx(.05, abs=.003)
    assert rows[.0015]['p_stronger'] == pytest.approx(.515, abs=.005)
    assert rows[.002]['p_stronger'] == pytest.approx(.957, abs=.005) and rows[.003]['p_stronger'] > .999


# ------------------------------------------------------------------ selection rules
def test_selection_rules_ties_fresh_c2_prime_and_astar_hash():
    assert ea.first_max([.1, .3, .3, .2]) == 1 and ea.first_max([.3, .3]) == 0                 # ties: smaller number / earlier
    assert ea.first_max([-1., -2.]) == 0
    with pytest.raises(ValueError):
        ea.first_max([.1, float('nan')])
    with pytest.raises(ValueError):
        ea.first_max([])
    recs = [dict(gen=50, G_SEL=.01), dict(gen=100, G_SEL=.03), dict(gen=150, G_SEL=.03), dict(gen=200, G_SEL=.02)]
    assert ea.best_record(recs)['gen'] == 100                                                  # chain checkpoint: earliest of the equal maxima
    assert ea.pick_c2_prime(.03, .0301) == 'A_fresh' and ea.pick_c2_prime(.03, .03) == 'a_star' and ea.pick_c2_prime(.03, .02) == 'a_star'
    inp = ea.e1v2_entries()                                                                     # reads the E1 v2 files and verifies all 12 thetas + a*
    assert inp['astar']['theta_sha256'].startswith('88ed926a') and ea.theta_id(inp['astar']['theta']) == inp['astar']['theta_sha256']
    assert len(inp['entries']) == 12 and inp['astar']['name'] == 'orig_adapt0' and inp['c2_cfg'] == ea.C2_CFG
    sel = e1.read(ea.ROOT/ea.E1V2_DIR/'selection_frozen.json')
    sel['astar']['orig']['theta'][0] += 1e-9
    orig = ea.read
    try:
        ea.read = lambda p: sel if str(p).endswith('selection_frozen.json') else orig(p)
        with pytest.raises(ValueError):
            ea.e1v2_entries()                                                                   # a tampered a* theta is refused
    finally:
        ea.read = orig


def test_original_setting_delta_uses_the_same_rule_as_a_fresh():
    G = [.0343, .0333, .0334]                                                                   # frozen_run0-2 on EA1_SEL in the original setting
    assert ea.first_max(G) == 0 and ea.first_max([.03, .0335, .0335]) == 1
    star, f = vec(.0348, .01, N, 1), vec(.0343, .01, N, 2)
    d = ea.interval(f-star)
    assert d['mean'] == pytest.approx(-.0005) and d['LB'] == ea.lower_bound(f-star) and d['UB'] == ea.upper_bound(f-star) and d['n'] == N


def test_lane_adoption_rule_is_fixed():
    assert ea.decide_lanes(10., 25., True, True, 2.)['lanes'] == 3                              # ratio 1.2
    assert ea.decide_lanes(10., 27.3, True, True, 2.)['lanes'] == 1                             # 3*10/27.3 = 1.0989 < 1.10
    assert ea.decide_lanes(10., 27.27, True, True, 2.)['lanes'] == 3                            # 1.1001
    assert ea.decide_lanes(10., 20., False, True, 2.)['lanes'] == 1 and ea.decide_lanes(10., 20., True, True, 6.5)['lanes'] == 1
    assert ea.decide_lanes(10., 20., True, True, 6.0)['lanes'] == 3
    assert ea.decide_lanes(10., 20., False, False, 2.)['lanes'] is None                         # A != B: graph unusable, report, no eager switch
    assert (ea.PARAMS['lanes']['ratio_threshold'], ea.PARAMS['lanes']['mem_limit_gb']) == (1.10, 6.0)


# ------------------------------------------------------------------ ledger
def test_ledger_is_one_shot_atomic_and_detects_tampering(tmp_path):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    calls = [0]

    def runner():
        calls[0] += 1
        return dict(T_score=10, snap=dict(util_b=np.ones((2, 2, 3)).tolist(), reg_b=np.ones((2, 2)).tolist(), pen=np.zeros((2, 2, 3)).tolist()))
    raw = ea.once(tmp_path, 'main_C2', con, runner)
    assert ea.once(tmp_path, 'main_C2', con, runner) == raw and calls[0] == 1
    assert ea.read(tmp_path/'ledger.json')['main_C2']['status'] == 'completed' and not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError):
        ea.once(tmp_path, 'main_C2', dict(con, seeds=[1, 3]), runner)
    r = ea.read(tmp_path/'main_C2_raw.json')
    r['snap']['util_b'][0][0][0] = 5.
    ea.write(tmp_path/'main_C2_raw.json', r)
    with pytest.raises(ValueError):
        ea.once(tmp_path, 'main_C2', con, runner)                                               # tampered raw
    ea.write(tmp_path/'ledger.json', dict(crash=dict(status='started', contract_sha256=ea.digest(con))))
    with pytest.raises(RuntimeError):
        ea.once(tmp_path, 'crash', con, runner)                                                 # started, no raw: stop
    assert calls[0] == 1


# ------------------------------------------------------------------ manifest / environment
def fake_manifest(smoke=True):
    m = dict(schema='x', smoke=smoke, params=json.loads(json.dumps(ea.SMOKE_PARAMS if smoke else ea.PARAMS)), seeds={'a': [1, 2]})
    m['sha256'] = ea.digest(m)
    return m


def test_manifest_changes_are_detected_and_params_must_match_code(tmp_path):
    m = fake_manifest()
    assert ea.verify_manifest_dict(m) is m
    for edit in (lambda x: x['params'].update(eps=.002), lambda x: x['seeds'].update(a=[1, 3]), lambda x: x.update(design_sha256='0'),
                 lambda x: x['params']['attacker'].update(gens=500)):
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


def test_smoke_manifest_build_records_inputs_protected_hashes_and_environment(monkeypatch):
    monkeypatch.setattr(ea, 'environment', lambda: dict(torch='t', numpy='n', scipy='s', device_name='d', threads={}))
    m, audit = ea.build_manifest(smoke=True)
    e1m = e1.read(ea.ROOT/ea.E1V2_DIR/'manifest.json')
    assert m['smoke'] and set(m['seeds']) == {'EA1_TRAIN', 'EA1_SEL'} and 'EA1_TEST' not in m['seeds']
    assert all(m['protected_sha256'][k] == v for k, v in {**e1m['protected_sha256'], **e1m['input_sha256']}.items())
    assert m['protected_sha256']['docs/direction2/design/e1-v2.md'] == e1m['design_sha256']
    assert any(k.endswith('selection_frozen.json') for k in m['e1v2_refs_sha256']) and any(k.endswith('attacker/orig_run0.json') for k in m['e1v2_refs_sha256'])
    assert m['environment']['torch'] == 't' and m['drift']['seeds'] == e1m['seeds']['AS']
    ea.check_hashes({**m['protected_sha256'], **m['e1v2_refs_sha256']})
    assert m['params']['astar_theta_sha256'].startswith('88ed926a') and m['sha256'] == ea.digest({k: v for k, v in m.items() if k != 'sha256'})


def test_gpu_stages_refuse_when_the_environment_differs():
    env = dict(device_name='Quadro RTX 4000', capability=[7, 5], gpu_name='Quadro RTX 4000', gpu_uuid='GPU-x', torch='2.6.0+cu126', cuda='12.6', numpy='2.2.6',
               scipy='1.15.3', threads=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), hostname='astarc', driver='595')
    m = dict(environment=env)
    ea.check_environment(m, dict(env, hostname='other', driver='596'))                         # host and driver are recorded, not checked
    for k, v in (('torch', '2.7.0'), ('numpy', '2.0.0'), ('scipy', '1.14.0'), ('device_name', 'RTX 8000'), ('cuda', '12.1'), ('gpu_uuid', 'GPU-y'), ('capability', [7, 0]),
                 ('threads', dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))):
        with pytest.raises(ValueError):
            ea.check_environment(m, dict(env, **{k: v}))


# ------------------------------------------------------------------ lanes == sequential (design 3.4), CPU
@pytest.fixture(scope='module')
def ref6(tmp_path_factory):
    """Reference for gens=6: the three chains one after the other on the default stream, eager (computed once per module)."""
    d = tmp_path_factory.mktemp('ref6')
    m = tiny(d, gens=6)
    hook = Collect()
    ea.GEN_HOOK = hook
    try:
        res = run_solo(m, [0, 1, 2], eager, 1)
    finally:
        ea.GEN_HOOK = None
    return res, hook.rec

def test_parallel_lanes_equal_sequential_and_eager_bitwise(tmp_path, monkeypatch):
    """(1) 3 chains one after the other (eager) vs one after the other on a lane vs 3 threads at once: every candidate matrix, utility matrix, fitness vector,
    seed block, validation gain and optimizer state is bitwise identical (>= 20 generations)."""
    runs = [0, 1, 2]
    res, hooks = {}, {}
    for tag, factory, lanes in (('eager_seq', eager, 1), ('lane_seq', lane, 1), ('lanes3', lane, 3)):
        m = tiny(tmp_path/tag, gens=20, val_every=10, post=10)
        hooks[tag] = Collect()
        monkeypatch.setattr(ea, 'GEN_HOOK', hooks[tag])
        res[tag] = run_solo(m, runs, factory, lanes)
    assert len(hooks['lanes3'].rec) == 60
    for tag in ('lane_seq', 'lanes3'):
        assert hooks[tag].rec == hooks['eager_seq'].rec
        assert ea.digest(res[tag]) == ea.digest(res['eager_seq'])
    assert [r['G_SEL'] for r in res['lanes3'][1]['records']] == [r['G_SEL'] for r in res['eager_seq'][1]['records']] and len(res['lanes3'][1]['records']) == 2
    for r in runs:
        assert ea.chain_state(r) is not None
        for k in ('mean', 'sigma', 'pc', 'ps', 'C', 'gen', 'rng'):
            assert ea.digest(res['lanes3'][r]['opt'][k]) == ea.digest(res['eager_seq'][r]['opt'][k])
    assert len({ea.digest(res['lanes3'][r]['opt']['mean']) for r in runs}) == 3                # the chains really differ from each other


@pytest.mark.parametrize('variant', range(5))
def test_thread_interleaving_does_not_change_results(tmp_path, monkeypatch, variant, ref6):
    """(2) start order, start delays and sleeps injected between generations change the interleaving; results stay bitwise equal to the reference."""
    runs = [0, 1, 2]
    ref, ref_rec = ref6
    orders = [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1], [1, 0, 2]]
    delays = [dict(), {0: .15}, {1: .1, 2: .2}, {2: .12}, {0: .05, 1: .1}]
    sleepers = [None, lambda r, g: .01*((r+g) % 3), lambda r, g: .02*(r == 0), lambda r, g: .015*((g+1) % 2)*(r+1), lambda r, g: .005*(3-r)]
    m = tiny(tmp_path/'par', gens=6)
    hook = Collect(sleep=sleepers[variant])
    monkeypatch.setattr(ea, 'GEN_HOOK', hook)
    ea.run_lanes(m, runs, 3, factory=lane, order=orders[variant], delays=delays[variant])
    got = {r: ea.chain_result(ea.chain_state(r)) for r in runs}
    assert ea.digest(got) == ea.digest(ref) and hook.rec == ref_rec


def test_a_failing_lane_does_not_disturb_the_others(tmp_path, monkeypatch, ref6):
    """(3) one lane raises in generation 3: the other two finish and are bitwise what they are when run alone; the stage still fails."""
    runs = [0, 1, 2]
    ref = ref6[0]
    m = tiny(tmp_path/'par', gens=6)
    monkeypatch.setattr(ea, 'GEN_HOOK', Collect(fail=lambda r, g: r == 1 and g == 2))
    with pytest.raises(RuntimeError, match='synthetic lane failure'):
        ea.run_lanes(m, runs, 3, factory=lane)
    assert ea.digest(ea.chain_result(ea.chain_state(0))) == ea.digest(ref[0]) and ea.digest(ea.chain_result(ea.chain_state(2))) == ea.digest(ref[2])
    assert ea.chain_state(1)['opt']['gen'] == 3                                                  # the failed chain stopped where it failed
    with pytest.raises(ValueError):
        ea.chains_complete(m)                                                                    # select refuses an incomplete chain
    monkeypatch.setattr(ea, 'GEN_HOOK', None)
    ea.run_lanes(m, [1], 1, factory=lane)                                                        # resume the failed chain: identical to the reference
    assert ea.digest(ea.chain_result(ea.chain_state(1))) == ea.digest(ref[1])
    ea.chains_complete(m)


def test_interrupted_parallel_run_resumed_on_one_lane_equals_one_shot(tmp_path, monkeypatch):
    """(4) all lanes stop at different generations (like a yield to another account), then a single lane finishes: bitwise equal to an uninterrupted run."""
    runs = [0, 1, 2]
    ref = run_solo(tiny(tmp_path/'ref', gens=8, val_every=4, post=4), runs, eager, 1)
    m = tiny(tmp_path/'par', gens=8, val_every=4, post=4)
    monkeypatch.setattr(ea, 'GEN_HOOK', Collect(stop=lambda r, g: g == 1+2*r))
    with pytest.raises(ea.YieldStop):
        ea.run_lanes(m, runs, 3, factory=lane)
    assert [ea.chain_state(r)['opt']['gen'] for r in runs] == [2, 4, 6]
    monkeypatch.setattr(ea, 'GEN_HOOK', None)
    ea.run_lanes(m, runs, 1, factory=lane)
    assert ea.digest({r: ea.chain_result(ea.chain_state(r)) for r in runs}) == ea.digest(ref)
    ea.chains_complete(m)


def test_seed_blocks_and_optimizer_streams_are_the_same_in_parallel_and_sequential(tmp_path, monkeypatch):
    """(5) per chain, the seed block of generation g is the registered permuted block, and the candidate matrices (= optimizer sampling) do not change."""
    runs = [0, 1, 2]
    seq, par = Collect(), Collect()
    m = tiny(tmp_path/'seq', gens=5)
    monkeypatch.setattr(ea, 'GEN_HOOK', seq)
    run_solo(m, runs, eager, 1)
    m2 = tiny(tmp_path/'par', gens=5)
    monkeypatch.setattr(ea, 'GEN_HOOK', par)
    ea.run_lanes(m2, runs, 3, factory=lane)
    for r in runs:
        for g in range(5):
            assert par.rec[(r, g)][3] == seq.rec[(r, g)][3] == tuple(ea.train_block(m['seeds']['EA1_TRAIN'], r, g, 4)) and par.rec[(r, g)][0] == seq.rec[(r, g)][0]
    assert len({seq.rec[(r, 0)][3] for r in runs}) == 3 and len({seq.rec[(r, 0)][0] for r in runs}) == 3     # chains use different blocks and different samples
    assert not any(set(par.rec[k][3]) & set(m['seeds']['EA1_SEL'] + m['seeds']['EA1_TEST']) for k in par.rec)


# ------------------------------------------------------------------ post-train check (item 8)
def write_run_config(m, tmp_path, lanes):
    ea.write(tmp_path/'run_config.json', dict(manifest_sha256=m['sha256'], lanes=lanes, parallel_check_sha256='x'))


def test_post_train_check_matches_and_detects_one_flipped_bit(tmp_path, monkeypatch):
    m = tiny(tmp_path, gens=6, val_every=3, post=4)
    ea.run_lanes(m, [0, 1, 2], 3, factory=lane)
    write_run_config(m, tmp_path, 3)
    rec = ea.stage_postcheck(m)
    assert rec['passed'] and rec['X_identical'] and rec['fitness_identical'] and rec['generation'] == 4 and (tmp_path/'post_train_check.json').exists()
    saved = ea.read(ea.chain_path(0, 'gen4'))
    x = np.array(saved['X'])
    x[3, 7] = np.nextafter(x[3, 7], np.inf)                                                      # one unit in the last place
    ea.write(ea.chain_path(0, 'gen4'), dict(saved, X=x.tolist()))
    with pytest.raises(ea.PostCheckFailed):
        ea.stage_postcheck(m)
    assert ea.read(tmp_path/'post_train_check.json')['passed'] is False and ea.read(tmp_path/'post_train_check.json')['X_identical'] is False
    f = np.array(saved['f'])
    f[0] = np.nextafter(f[0], np.inf)
    ea.write(ea.chain_path(0, 'gen4'), dict(saved, f=f.tolist()))
    with pytest.raises(ea.PostCheckFailed):
        ea.stage_postcheck(m)
    assert ea.read(tmp_path/'post_train_check.json')['fitness_identical'] is False
    assert not ea.bitwise_equal([0.0], [-0.0]) and ea.bitwise_equal([1., 2.], [1., 2.]) and not ea.bitwise_equal([1.], [1., 1.])
    write_run_config(m, tmp_path, 1)
    one = ea.stage_postcheck(m)
    assert one['passed'] and one['required'] is False                                            # lanes=1: nothing to compare


def test_a_foreign_gpu_program_stops_all_lanes_at_a_generation_boundary(tmp_path, monkeypatch, ref6):
    runs = [0, 1, 2]
    ref = ref6[0]
    m = tiny(tmp_path/'par', gens=6)
    ea.MODE['yield_check'] = True
    calls = []
    monkeypatch.setattr(ea, 'foreign_gpu_pids', lambda: (calls.append(1), [4242] if len(calls) > 1 else [])[1])
    monkeypatch.setattr(ea, 'check_yield', functools_partial_interval())
    with pytest.raises(ea.YieldStop):
        ea.run_lanes(m, runs, 3, factory=lane)
    gens = [ea.chain_state(r)['opt']['gen'] if ea.chain_path(r).exists() else 0 for r in runs]
    assert all(g < 6 for g in gens) and ea._YIELD['event'].is_set()
    ea._YIELD['event'].clear()
    monkeypatch.setattr(ea, 'foreign_gpu_pids', lambda: [])
    ea.run_lanes(m, runs, 3, factory=lane)                                                       # relaunch when the GPU is free: identical to the reference
    assert ea.digest({r: ea.chain_result(ea.chain_state(r)) for r in runs}) == ea.digest(ref)


def functools_partial_interval():
    orig = ea.check_yield
    return lambda: orig(min_interval=0.0)


# ------------------------------------------------------------------ select + testcheck + test flow (synthetic E1 v2 inputs, no TEST seed outside the fake sim)
class FakeSim:
    """Deterministic stand-in: gain grows with the sum of theta, shrinks with tol; records the seeds it was asked for."""
    def __init__(self):
        self.used = []

    def __call__(self, m, cfg, pols, sd, T, be=None):
        self.used.append(list(sd))
        ts = T-m['params']['exclude']
        base = np.array([float(np.tanh(np.asarray(getattr(q, 'theta', np.zeros(1))).sum())) for q in pols])
        u = np.full((len(pols), len(sd), 3), 100.)
        u[:, :, 1] += ((base-base[0])[:, None]*(1.0-.05*(cfg['tol']-1.))*.02+.001*np.sin(np.asarray(sd, float))[None, :])*ts*(np.arange(len(pols))[:, None] > 0)
        reg = np.ones((len(pols), len(sd)))*(1+cfg['tol'])
        return dict(T_score=ts, snap=dict(util_b=u.tolist(), reg_b=reg.tolist(), pen=np.zeros((len(pols), len(sd), 3)).tolist()))


def fake_inputs():
    rng = np.random.default_rng(3)
    ents = [(n, list(rng.normal(0, .02, 55))) for n in ['frozen_run0', 'frozen_run1', 'frozen_run2', 'orig_adapt0', 'orig_adapt1', 'orig_adapt2',
                                                         'C1_adapt0', 'C1_adapt1', 'C1_adapt2', 'C2_adapt0', 'C2_adapt1', 'C2_adapt2']]
    star = dict(name='orig_adapt0', theta=ents[3][1], theta_sha256=ea.theta_id(ents[3][1]))
    return dict(astar=star, entries=ents, c2_cfg=dict(p=.5, tol=1., kref=.5, L=8.), G_AS_astar=dict(orig=.03, C2=.02), attackers=[n for n, _ in ents],
                transfer_raw=dict(orig='x', C2='y'))


def flow_env(tmp_path, monkeypatch):
    """Tiny official-style run: 3 tiny chains trained, select run with fake E1 v2 inputs, freeze registered; sim replaced by FakeSim afterwards."""
    m = tiny(tmp_path, gens=6)
    ea.run_lanes(m, [0, 1, 2], 1, factory=lane)
    ea.mark_done('train')
    write_run_config(m, tmp_path, 1)
    inp = fake_inputs()
    monkeypatch.setattr(ea, 'e1v2_entries', lambda: inp)
    monkeypatch.setattr(ea, 'drift_reference', lambda i: dict(orig=np.full(4, .03), C2=np.full(4, .02)))
    fake = FakeSim()
    monkeypatch.setattr(ea, 'sim', fake)
    ea.stage_select(m)
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    monkeypatch.setattr(ea, 'load_manifest', lambda clean=True: m)
    sel = ea.verify_freeze(m)
    fake.used.clear()
    monkeypatch.setattr(ea, 'sim_eager', lambda m_, cfg, pols, sd, T: fake(m_, cfg, pols, sd, T))
    ea.MODE.update(strict=True, graph=False)
    return m, sel, fake.used


def test_select_registers_one_frozen_selection_from_one_reevaluation(tmp_path, monkeypatch):
    m = tiny(tmp_path, gens=6)
    ea.run_lanes(m, [0, 1, 2], 1, factory=lane)
    with pytest.raises(RuntimeError):
        ea.stage_select(m)                                                                       # train not marked done
    ea.mark_done('train')
    write_run_config(m, tmp_path, 3)
    with pytest.raises(ValueError):
        ea.stage_select(m)                                                                       # lanes=3 needs a passing post_train_check
    ea.write(tmp_path/'post_train_check.json', dict(manifest_sha256='synthetic', passed=True))
    inp = fake_inputs()
    monkeypatch.setattr(ea, 'e1v2_entries', lambda: inp)
    monkeypatch.setattr(ea, 'drift_reference', lambda i: dict(orig=np.full(4, .03), C2=np.full(4, .02)))
    used = []
    real = ea.sim

    def spy(m_, cfg, pols, sd, T, be=None):
        used.append((len(pols), tuple(sd)))
        return real(m_, cfg, pols, sd, T, be)
    monkeypatch.setattr(ea, 'sim', spy)
    ea.stage_select(m)
    sel = ea.read(tmp_path/'selection_frozen.json')
    assert ea.read(tmp_path/'freeze_record.json')['selection_frozen_sha256'] == ea.sha(tmp_path/'selection_frozen.json')
    assert [u[0] for u in used] == [5, 5, 16, 16, 2, 2]                                           # C2 re-eval, orig re-eval, 15 attackers x 2 settings, a* drift x 2
    assert all(set(u[1]) <= set(m['seeds']['EA1_SEL']) for u in used[:4]) and all(set(u[1]) <= set(m['drift']['seeds']) for u in used[4:])
    assert not any(set(u[1]) & set(m['seeds']['EA1_TEST']) for u in used)
    G = sel['reeval_C2']['G']
    assert sel['reeval_C2']['names'] == ['a_star', 'fresh_run0', 'fresh_run1', 'fresh_run2']
    assert sel['A_fresh']['index'] == ea.first_max(G[1:]) and sel['A_fresh']['G_SEL_reeval'] == G[1:][sel['A_fresh']['index']]
    assert sel['A_C2_prime'] == ea.pick_c2_prime(G[0], max(G[1:])) and sel['F_orig']['name'].startswith('frozen_run')
    assert len(sel['entries']) == 15 and [e['name'] for e in sel['entries'][12:]] == ['fresh_run0', 'fresh_run1', 'fresh_run2']
    for c in sel['chains']:                                                                       # chain checkpoint = best of its own validation points
        assert c['G_SEL_chain'] == max(r['G_SEL'] for r in ea.chain_state(c['run'])['records']) and c['theta_sha256'] == ea.theta_id(c['theta'])
    assert (tmp_path/'drift.json').exists() and 'ea1_selection_frozen' not in json.dumps(sel)
    n = len(used)
    ea.stage_select(m)                                                                            # idempotent: registers what exists, never recomputes
    assert len(used) == n


def test_select_refuses_incomplete_chains(tmp_path, monkeypatch):
    m = tiny(tmp_path, gens=6)
    ea.run_chain(m, 0, eager())
    ea.run_chain(m, 1, eager(), target=4)
    ea.mark_done('train')
    write_run_config(m, tmp_path, 1)
    monkeypatch.setattr(ea, 'e1v2_entries', lambda: fake_inputs())
    with pytest.raises(ValueError):
        ea.stage_select(m)                                                                       # chain 1 stopped at 4, chain 2 never started
    assert not (tmp_path/'selection_frozen.json').exists()
    bad = ea.chain_state(0)
    bad['records'][0]['G_SEL'] = float('nan')
    with pytest.raises(ValueError):
        ea.write(ea.chain_path(0), bad)                                                          # non-finite values cannot even be written (allow_nan=False)


def test_test_stage_unlocks_only_via_freeze_and_testcheck_and_runs_each_evaluation_once(tmp_path, monkeypatch):
    m, sel, used = flow_env(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ea.run_test()                                                                            # no testcheck yet
    assert not (tmp_path/'test').exists() and used == []
    ea.run_testcheck()
    del used[:]
    res = ea.run_test()
    led = ea.read(tmp_path/'test/ledger.json')
    assert set(led) == {'main_C2', 'main_orig', 'sec_C2', 'sec_orig'} and all(v['status'] == 'completed' for v in led.values())
    assert set(s for u in used for s in u) == set(m['seeds']['EA1_TEST']) and all(len(u) == 4 for u in used)
    assert res['category'] in ('stronger', 'not_stronger', 'insufficient') and res['n'] == 8 and res['A_fresh'].startswith('fresh_run')
    n = len(used)
    ea.run_test()                                                                                # nothing is simulated again
    assert len(used) == n
    sec = ea.read(tmp_path/'secondary.json')
    assert set(sec['orig_type_delta']) == {'frozen_run0', 'frozen_run1', 'frozen_run2'} and set(sec['fresh_vs_astar_C2']) == {'fresh_run0', 'fresh_run1', 'fresh_run2'}
    assert sec['D_prime']['A_C2_prime'] == sel['A_C2_prime'] and sec['D_new']['n'] == 8 and len(sec['chains']) == 3 and 'honest_service' in sec
    assert ea.read(tmp_path/'result.json')['category'] == res['category']
    assert ea.stage_report() == 0                                                                # recomputed from raw: identical to the saved files
    r = ea.read(tmp_path/'test/main_C2_raw.json')
    assert r['contract']['theta_sha256'] == [ea.theta_id(sel['astar']['theta']), ea.theta_id(next(e for e in sel['entries'] if e['name'] == sel['A_fresh']['name'])['theta'])]


def test_test_refuses_stale_testcheck_and_a_graph_failure_precedes_the_ledger(tmp_path, monkeypatch):
    m, sel, used = flow_env(tmp_path, monkeypatch)
    ea.run_testcheck()
    for key, val in (('manifest_sha256', 'other'), ('freeze_record_sha256', 'other')):
        t = ea.read(tmp_path/'testcheck.json')
        ea.write(tmp_path/'testcheck.json', dict(t, **{key: val}))
        with pytest.raises(ValueError):
            ea.require_testcheck(m, sel)
        ea.write(tmp_path/'testcheck.json', t)
    t = ea.read(tmp_path/'testcheck.json')
    ea.write(tmp_path/'testcheck.json', dict(t, path=dict(t['path'], graph=True)))
    with pytest.raises(ValueError):
        ea.require_testcheck(m, sel)                                                             # different execution path
    ea.write(tmp_path/'testcheck.json', t)
    ea.MODE.update(device='cuda', graph=True)                                                    # pretend cuda graph path; the pre-capture raises
    ea.write(tmp_path/'testcheck.json', dict(t, path=dict(t['path'], graph=True, device='cuda')))
    monkeypatch.setattr(ea, 'sim', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('CUDA graph capture failed')))
    with pytest.raises(RuntimeError, match='graph'):
        ea.run_test()
    assert not (tmp_path/'test').exists()                                                        # failed before the ledger; no eager fallback exists


def test_verify_freeze_requires_files_matching_hash_and_commit(tmp_path, monkeypatch):
    m = dict(smoke=False, sha256='synthetic')
    ea.MODE.update(out=str(tmp_path), smoke=False)
    with pytest.raises(ValueError):
        ea.verify_freeze(m)
    th = list(np.zeros(55))
    sel = dict(manifest_sha256='synthetic', status='go', astar=dict(name='a', theta=th, theta_sha256=ea.theta_id(th)), entries=[])
    ea.write(tmp_path/'selection_frozen.json', sel)
    ea.write(tmp_path/'freeze_record.json', dict(manifest_sha256='synthetic', selection_frozen_sha256=ea.sha(tmp_path/'selection_frozen.json')))
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: False)
    with pytest.raises(ValueError):
        ea.verify_freeze(m)                                                                      # not committed
    monkeypatch.setattr(ea.plan, 'is_committed_and_clean', lambda root, rel: True)
    assert ea.verify_freeze(m)['status'] == 'go'
    ea.write(tmp_path/'selection_frozen.json', dict(sel, x=1))
    with pytest.raises(ValueError):
        ea.verify_freeze(m)                                                                      # tampered after registration
    ea.write(tmp_path/'selection_frozen.json', sel)
    with pytest.raises(ValueError):
        ea.verify_freeze(dict(smoke=True, sha256='synthetic'))                                   # smoke never opens test


def test_train_stage_requires_a_passing_parallel_check_for_this_code(tmp_path):
    m = tiny(tmp_path)
    with pytest.raises(ValueError):
        ea.require_parallel_check(m)
    pc = dict(manifest_sha256='synthetic', ea1_py_sha256=ea.sha(ea.ROOT/'scripts/ea1.py'), graph_equals_eager=True, parallel_equals_single=True)
    ea.write(tmp_path/'parallel_check.json', pc)
    ea.write(tmp_path/'run_config.json', dict(manifest_sha256='synthetic', lanes=3, parallel_check_sha256=ea.sha(tmp_path/'parallel_check.json')))
    assert ea.require_parallel_check(m)['lanes'] == 3
    for bad in (dict(pc, ea1_py_sha256='0'), dict(pc, graph_equals_eager=False), dict(pc, manifest_sha256='x')):
        ea.write(tmp_path/'parallel_check.json', bad)
        ea.write(tmp_path/'run_config.json', dict(manifest_sha256='synthetic', lanes=3, parallel_check_sha256=ea.sha(tmp_path/'parallel_check.json')))
        with pytest.raises(ValueError):
            ea.require_parallel_check(m)
    ea.write(tmp_path/'parallel_check.json', dict(pc, parallel_equals_single=False))
    ea.write(tmp_path/'run_config.json', dict(manifest_sha256='synthetic', lanes=3, parallel_check_sha256=ea.sha(tmp_path/'parallel_check.json')))
    with pytest.raises(ValueError):
        ea.require_parallel_check(m)                                                             # lanes=3 needs B == C
    ea.write(tmp_path/'run_config.json', dict(manifest_sha256='synthetic', lanes=1, parallel_check_sha256=ea.sha(tmp_path/'parallel_check.json')))
    assert ea.require_parallel_check(m)['lanes'] == 1


# ------------------------------------------------------------------ E1 v2 is untouched
def test_e1v2_protected_and_input_files_still_match_their_manifest():
    e1m = e1.read(ea.ROOT/ea.E1V2_DIR/'manifest.json')
    files = {**e1m['protected_sha256'], **e1m['input_sha256'], e1m['design']: e1m['design_sha256']}
    assert len(files) >= 90
    for rel, h in files.items():
        assert hashlib.sha256((ea.ROOT/rel).read_bytes()).hexdigest() == h, rel
    assert not any(p.startswith('scripts/ea1') or p.startswith('tests/direction2/test_ea1') for p in files)      # E-A1 files are new, not E1 v2 inputs


def test_launcher_auto_resumes_train_on_75_and_passes_other_codes(tmp_path):
    import shutil
    import subprocess
    root = tmp_path/'repo'
    (root/'scripts').mkdir(parents=True)
    shutil.copy(ea.ROOT/'scripts/ea1_launch.sh', root/'scripts/ea1_launch.sh')
    (root/'bin').mkdir()
    for tool in ('nvidia-smi', 'nice'):
        (root/'bin'/tool).write_text('#!/bin/sh\n' + ('exit 0\n' if tool == 'nvidia-smi' else '[ "$1" = "-n" ] && shift 2\nexec "$@"\n'))
        (root/'bin'/tool).chmod(0o755)
    (root/'tests/direction2').mkdir(parents=True)
    fake = root/'fakepy'
    fake.write_text('#!/bin/sh\n[ "$1" = "-m" ] && exit 0\nn=$(cat count 2>/dev/null || echo 0); n=$((n+1)); echo $n > count\n[ "$n" -lt 3 ] && exit 75\nexit ${FINAL_RC:-0}\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f'{root}/bin:'+os.environ['PATH'], PYTHON=str(fake))
    r = subprocess.run(['bash', 'scripts/ea1_launch.sh', 'train'], cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and (root/'count').read_text().strip() == '3'
    assert (root/'results/direction2/runlogs/ea1_train.log').read_text().count('resume #') == 2
    (root/'count').write_text('2')
    r = subprocess.run(['bash', 'scripts/ea1_launch.sh', 'select'], cwd=root, env=dict(env, FINAL_RC='7'), capture_output=True, text=True)
    assert r.returncode == 7                                                                    # other exit codes pass through unchanged
    (root/'count').write_text('2')
    r = subprocess.run(['bash', 'scripts/ea1_launch.sh', 'test'], cwd=root, env=dict(env, FINAL_RC='7'), capture_output=True, text=True)
    assert r.returncode == 7 and (root/'count').read_text().strip() == '3'                      # test: no retry, no resume
    (root/'bin'/'nvidia-smi').write_text('#!/bin/sh\necho 1234\n')
    r = subprocess.run(['bash', 'scripts/ea1_launch.sh', 'train'], cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 3 and 'GPU occupied' in r.stdout                                      # another program on the GPU: refuse


# ------------------------------------------------------------------ GPU (astarc only)
@pytest.mark.gpu
def test_gpu_small_acceptance_graph_equals_eager_and_lanes_equal_single(tmp_path):
    """Scale matters: with T < K (the old tiny budget, K=32) no graph is ever captured or replayed, so the 3-lane problem seen in the formal-size smoke
    (torch 2.6 'Offset increment outside graph capture') could not show.  Here T=48, K=4: graphs ARE captured and replayed.  A failed 3-lane run is
    a recorded result (lanes=1), never an abort; A == B must hold either way."""
    cuda_required()
    m = tiny(tmp_path, gens=3)
    m['params'].update(T_train=48, T_val=48, exclude=8)
    ea.MODE.update(device='cuda', K=4)
    be = ea.make_backend('cuda')
    with be:
        ea.trace_chain(m, 0, be, 3)
    assert be.gs.stats['builds'] >= 1 and be.gs.stats['replays'] >= 1
    A, B, C, info = ea.run_parallel_traces(m, 3, 'cuda')
    rep = ea.acceptance_report(m, 3, A, B, C, info)
    assert rep['graph_equals_eager'], rep
    if C is None:
        assert not rep['parallel_ok'] and rep['parallel_error'] and ea.decide_lanes(1., None, False, True, None)['lanes'] == 1
    else:
        assert rep['parallel_equals_single'], rep


@pytest.mark.gpu
def test_gpu_graph_path_has_no_fallback_and_chunked_equals_eager(tmp_path):
    cuda_required()
    m = tiny(tmp_path)
    ea.MODE.update(device='cuda', graph=True, strict=True)
    pols = [ea.honest_pol(), ea.mlp(np.full(55, .01))]
    sd = m['seeds']['EA1_SEL'][:4]
    got = ea.simulate_chunked(m, m['params']['c2_cfg'], pols, sd, 2)
    ref = ea.sim_eager(m, m['params']['c2_cfg'], pols, sd, m['params']['T_val'])
    for k in ('util_b', 'reg_b', 'pen'):
        assert ea.bitwise_equal(got['snap'][k], ref['snap'][k])
    assert ea.default_backend().gs is not None and ea.default_backend().gs.stats['builds'] >= 1


# ------------------------------------------------------------------ review fixes
def test_foreign_gpu_pids_ignores_own_account_only(monkeypatch):
    class R:
        stdout = '11\n22\n33\nx\n'
    monkeypatch.setattr(ea.subprocess, 'run', lambda *a, **k: R())
    monkeypatch.setattr(ea.os, 'getuid', lambda: 1000)

    def stat(path, *a, **k):
        uid = {'/proc/11': 1000, '/proc/22': 2000}.get(str(path))
        if uid is None:
            raise FileNotFoundError(path)                      # unreadable owner counts as another account
        return type('S', (), {'st_uid': uid})()
    monkeypatch.setattr(ea.os, 'stat', stat)
    assert ea.foreign_gpu_pids() == [22, 33]


def test_launcher_busy_means_another_account_only(tmp_path):
    import subprocess
    (tmp_path/'nvidia-smi').write_text(f'#!/bin/sh\necho {os.getpid()}\n')
    (tmp_path/'nvidia-smi').chmod(0o755)
    src = (ea.ROOT/'scripts/ea1_launch.sh').read_text()
    fn = src[src.index('busy() {'):src.index('\n}\n', src.index('busy() {'))+3]
    env = dict(os.environ, PATH=f'{tmp_path}:'+os.environ['PATH'])
    own = subprocess.run(['bash', '-c', fn+'\nbusy'], env=env).returncode
    (tmp_path/'nvidia-smi').write_text('#!/bin/sh\necho 1\n')                    # pid 1 is root's
    other = subprocess.run(['bash', '-c', fn+'\nbusy'], env=env).returncode
    (tmp_path/'nvidia-smi').write_text('#!/bin/sh\necho 999999999\n')            # unreadable
    gone = subprocess.run(['bash', '-c', fn+'\nbusy'], env=env).returncode
    assert (own, other, gone) == (1, 0, 0)


def test_official_manifest_refuses_non_cuda_device_for_stage_commands(tmp_path, monkeypatch):
    for cmd in ('train', 'paracheck', 'postcheck', 'select', 'testcheck', 'test', 'smoke-clean-x'):
        if cmd == 'smoke-clean-x':
            continue
        with pytest.raises(SystemExit) as e:
            ea.main([cmd, '--device', 'cpu'])
        assert 'cuda' in str(e.value)
    # report / status / audit / freeze are not blocked by the device check (they fail later, for other reasons, not SystemExit about cuda)
    monkeypatch.setattr(ea, 'stage_report', lambda: 0)
    assert ea.main(['report', '--device', 'cpu']) == 0


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
    ea.MODE.update(out=str(tmp_path/'s'), smoke=True)
    monkeypatch.setattr(ea, 'build_manifest', lambda smoke=False: (dict(environment=dict(good, torch='x'), sha256='x'), {}))
    ea.freeze(True)                                                              # smoke: recorded, not refused
    assert (tmp_path/'s/manifest.json').exists()


def test_d2_graph_k_is_ignored_for_k_and_refused_when_different(monkeypatch):
    monkeypatch.setenv('D2_GRAPH_K', '8')
    ea.MODE['K'] = None
    assert ea.kernel_K() == 32
    env = dict(ea.environment(), d2_graph_k='8')
    m = dict(environment=dict(env, d2_graph_k=None))
    with pytest.raises(ValueError, match='D2_GRAPH_K'):
        ea.check_environment(m, env)
    ea.check_environment(m, dict(env, d2_graph_k=None))
    ea.check_environment(m, dict(env, d2_graph_k='32'))                          # equal to the pin: allowed


def test_failed_three_lane_run_is_recorded_not_raised_and_means_lanes_one(tmp_path):
    m = tiny(tmp_path)
    main = threading.current_thread()

    def factory():
        if threading.current_thread() is not main:
            raise RuntimeError('Offset increment outside graph capture encountered unexpectedly.')
        return lane()
    A, B, C, info = ea.run_parallel_traces(m, 3, 'cpu', factory=factory)
    assert C is None and 'Offset increment' in info['parallel_error'] and set(B) == {0, 1, 2}        # single-lane measurements still complete
    rep = ea.acceptance_report(m, 3, A, B, C, info)
    assert rep['graph_equals_eager'] and rep['parallel_ok'] is False and rep['parallel_equals_single'] is False and rep['B_vs_C'] is None
    d = ea.decide_lanes(mean := 1.0, None, rep['parallel_equals_single'], rep['graph_equals_eager'], None)
    assert d['lanes'] == 1 and d['ratio'] is None and 'failed' in d['reason'] and mean == 1.0
    assert ea.decide_lanes(1., None, False, False, None)['lanes'] is None                              # a broken single-lane graph path is still reported, not hidden
    A2, B2, C2, info2 = ea.run_parallel_traces(m, 3, 'cpu', factory=lane)                            # healthy: unchanged behaviour
    assert C2 is not None and 'parallel_error' not in info2 and ea.acceptance_report(m, 3, A2, B2, C2, info2)['parallel_equals_single']


def test_multiprocess_worker_matches_single_process_trace(tmp_path, monkeypatch):
    m = tiny(tmp_path)
    monkeypatch.setattr(ea, 'load_manifest', lambda clean=True: m)
    ea.mp_worker(1, 3, tmp_path/'mp')
    z = np.load(tmp_path/'mp/run1.npz')
    be = lane()
    with be:
        tr = ea.trace_chain(m, 1, be, 3, validate=False)
    assert all(ea.bitwise_equal(z['X'][g], tr['gens'][g]['X']) and ea.bitwise_equal(z['f'][g], tr['gens'][g]['f']) for g in range(3)) and len(z['wall_gen']) == 3
    with pytest.raises(SystemExit):
        ea.main(['mpworker'])                                                                          # internal: needs --smoke and --mp-out
