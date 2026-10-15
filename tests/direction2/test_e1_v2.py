"""E1 v2: seeds, guards, statistics, decision tree, ledger, resumable pipeline on synthetic data (CPU only; no test-seed evaluation)."""
import copy
import json

import numpy as np
import pytest

from arbitration.rl import policy
from scripts import e1_v2 as e


# ------------------------------------------------------------------ seeds
def test_blocks_expand_to_registered_counts_and_are_disjoint_from_existing():
    rep = e.seed_report(e.block_seeds(), e.existing_seeds())
    assert (rep['new_expanded'], rep['existing_expanded'], rep['intersection']) == (7168, 4896, 0)
    assert rep['per_block'] == dict(DT=1024, DS=512, AT=1024, AS=512, TEST=4096)
    assert min(e.expand(sum(e.block_seeds().values(), []))) == 72_000_000
    assert max(e.expand(sum(e.block_seeds().values(), []))) == 75_401_023


@pytest.mark.parametrize('bad', [61_000_000, 41_000_063, 5_000, 61_000_031-3_000_000+0, 41_000_000-10**6])
def test_seed_report_rejects_collision_with_existing_ranges_and_offsets(bad):
    blocks = e.block_seeds()
    blocks['DT'] = blocks['DT'][:-1]+[bad] if bad < 72_000_000 else blocks['DT']
    if bad == 41_000_000-10**6:      # 40,000,000 + 1e6 = 41,000,000 (holdout): collision only through the offsets
        blocks['DT'][-1] = bad
    with pytest.raises(ValueError):
        e.seed_report(blocks, e.existing_seeds())


def test_seed_report_rejects_plus_3e6_stream_and_cross_block_and_saved_fields():
    ex = e.existing_seeds()
    b = e.block_seeds()
    b2 = dict(b, DT=b['DT'][:-1]+[61_000_000-3*10**6])    # 58,000,000 + 3e6 = 61,000,000
    with pytest.raises(ValueError):
        e.seed_report(b2, ex)
    b3 = dict(b, DS=b['DS'][:-1]+[b['AT'][0]])
    with pytest.raises(ValueError):
        e.seed_report(b3, ex)
    with pytest.raises(ValueError):
        e.seed_report(b, ex, extra_historical=[b['AS'][3]])


def test_require_seeds_each_stage_only_its_own_block_and_test_locked():
    m = dict(seeds=e.block_seeds())
    for stage, block in e.STAGE_BLOCK.items():
        if stage == 'test':
            continue
        e.require_seeds(m['seeds'][block][:5], stage, m)
        for other, ob in e.STAGE_BLOCK.items():
            if ob != block:
                with pytest.raises(ValueError):
                    e.require_seeds(m['seeds'][ob][:5], stage, m)
    for old in ([0], [2000], [5000], [41_000_000], [61_000_000], [72_400_000]):
        with pytest.raises(ValueError):
            e.require_seeds(old, 'dt', m)
    with pytest.raises(ValueError):
        e.require_seeds(m['seeds']['DT'][:2]*2, 'dt', m)
    with pytest.raises(ValueError):
        e.require_seeds([], 'dt', m)
    with pytest.raises(ValueError):
        e.require_seeds(m['seeds']['TEST'][:4], 'test', m)                    # locked
    e.require_seeds(m['seeds']['TEST'][:4], 'test', m, unlocked=True)
    with pytest.raises(ValueError):
        e.require_seeds(m['seeds']['DT'][:4], 'test', m, unlocked=True)       # still only its own block
    smoke = dict(seeds=e.block_seeds(smoke=True))
    assert 'TEST' not in smoke['seeds']
    with pytest.raises(ValueError):
        e.require_seeds([72_400_000], 'test', smoke, unlocked=True)
    with pytest.raises(ValueError):
        e.require_seeds([72_000_100], 'dt', smoke)                             # outside the smoke subset


@pytest.mark.parametrize('n', [1, 16, 32, 128, 256])
def test_rotating_blocks_distinct_reproducible_and_wrap(n):
    pool = list(range(72_000_000, 72_000_256))
    seen = set()
    for g in range(40):
        blk = e.rotating(pool, 1, g, n)
        assert len(blk) == len(set(blk)) == n and set(blk) <= set(pool)
        assert blk == e.rotating(pool, 1, g, n)
        seen |= set(blk)
    assert len(seen) == min(256, 40*n) or n == 256
    with pytest.raises(ValueError):
        e.rotating(pool, 0, 0, 257)
    with pytest.raises(ValueError):
        e.rotating(pool, 0, 0, 0)


# ------------------------------------------------------------------ honest constraint, saturation, selection
def vec(mean, sd, n, seed=0):
    z = np.random.default_rng(seed).standard_normal(n)
    return mean+sd*(z-z.mean())/z.std(ddof=1)


def test_honest_ci_uses_one_sided_t_upper_bound_and_boundaries():
    n = 128
    dr, da = vec(.001, .0042, n), vec(.004, .0146, n, 1)
    h = e.honest_ci(dr, da)
    assert h['ub_regret'] == pytest.approx(.001+1.6569*.0042/np.sqrt(n), abs=2e-6) and h['feasible']
    assert h['t'] == pytest.approx(1.6569, abs=1e-4)
    # point estimate under the limit but UB over it is infeasible; equality of UB with the limit is feasible
    assert not e.honest_ci(vec(.0018, .0042, n), da)['feasible']
    ub = e.upper_bound(dr)
    assert e.honest_ci(dr, da, lim_r=ub)['feasible']
    assert not e.honest_ci(dr, da, lim_r=ub-1e-6)['feasible']
    ub_a = e.upper_bound(da)
    assert not e.honest_ci(dr, da, lim_a=ub_a-1e-6)['feasible']
    # SD = 0 -> UB equals the mean; t depends on n
    assert e.honest_ci(np.full(8, .002), np.zeros(8))['feasible']
    assert not e.honest_ci(np.full(8, .0021), np.zeros(8))['feasible']
    assert e.tcrit(1024) == pytest.approx(1.6463, abs=1e-4)
    with pytest.raises(ValueError):
        e.honest_ci(np.array([.0]), np.array([.0]))


def test_honest_pass_probabilities_and_power_reproduce_design_tables():
    assert e.honest_pass_prob(1024, .00179, .0042, .002) == pytest.approx(.482, abs=.01)
    assert e.honest_pass_prob(1024, .0015, .0042, .002) == pytest.approx(.985, abs=.01)
    assert e.honest_pass_prob(1024, .0020, .0042, .002) == pytest.approx(.05, abs=.005)
    assert e.honest_pass_prob(1024, .009, .0146, .01) == pytest.approx(.707, abs=.01)
    assert e.power(1024, .006, .0135) == pytest.approx(.765, abs=.01)
    assert e.power(1024, .006, .015) == pytest.approx(.687, abs=.01)
    assert e.power(1024, .007, .0135) > .998
    assert e.power(1024, .005, .0135) == pytest.approx(.05, abs=.002)


def traj(vals, step=20):
    return [(i*step, v) for i, v in enumerate(vals)]


@pytest.mark.parametrize('gain,sat', [(.00099, True), (.001, False), (.00101, False), (-.01, True), (0., True)])
def test_saturation_boundary_240(gain, sat):
    pts = [(g, .01+.0001*g/20) for g in range(0, 201, 20)]
    prev = max(v for _, v in pts)
    pts += [(220, prev+gain), (240, prev+gain-.0005)]
    r = e.saturation(pts, 20, 240, .001)
    assert r['saturated'] is sat and r['gain'] == pytest.approx(gain, abs=1e-9)
    assert r['prev_best'] == pytest.approx(prev)


def test_saturation_extension_rejudged_with_480_and_same_rule():
    pts = traj([.01*i for i in range(25)])                      # strictly rising through 480
    assert not e.saturation(pts[:13], 20, 240, .001)['saturated']
    assert not e.saturation(pts, 20, 480, .001)['saturated']
    flat = traj([.02]*25)
    assert e.saturation(flat, 20, 480, .001)['saturated']
    with pytest.raises(ValueError):
        e.saturation(pts[:12], 20, 240, .001)                   # final point missing
    r = e.saturation(pts, 20, 480, .001)
    assert r['gmax'] == 480 and r['prev_best'] == pytest.approx(.01*22)   # best over g <= 440


def test_budget_consistency_and_extension_only_by_rule():
    b = dict(lam=64, n_at=32)
    budgets = {k: [dict(b, run=0), dict(b, run=1), dict(b, run=2)] for k in ('orig', 'C1')}
    assert e.budget_consistency(budgets, dict(orig=[240]*3, C1=[480]*3), dict(orig=False, C1=True), 240)
    bad = copy.deepcopy(budgets)
    bad['C1'][1]['lam'] = 32
    with pytest.raises(ValueError):
        e.budget_consistency(bad, dict(orig=[240]*3, C1=[240]*3), {}, 240)
    with pytest.raises(ValueError):       # extended without the rule asking for it
        e.budget_consistency(budgets, dict(orig=[480]*3, C1=[240]*3), dict(orig=False, C1=False), 240)
    with pytest.raises(ValueError):       # only some chains extended
        e.budget_consistency(budgets, dict(orig=[240]*3, C1=[480, 240, 480]), dict(orig=False, C1=True), 240)


def test_candidate_selection_rules():
    recs = [dict(feasible=False, worst_G=.001), dict(feasible=True, worst_G=.03), dict(feasible=True, worst_G=.02), dict(feasible=True, worst_G=.02, tag='later')]
    assert e.pick_run_winner(recs)['worst_G'] == .02 and 'tag' not in e.pick_run_winner(recs)
    assert e.pick_run_winner([dict(feasible=False, worst_G=0.)]) is None
    assert e.pick_c1([None, None]) is None
    assert e.pick_c1([dict(worst_G=.03, id=0), dict(worst_G=.02, id=1)])['id'] == 1
    assert e.pick_c1([dict(worst_G=.02, id=0), dict(worst_G=.02, id=1)])['id'] == 0
    assert e.final_selection([], .04)['status'] == 'no_feasible'
    s = e.final_selection([('C1', .03), ('C2', .02)], .04)
    assert s['status'] == 'go' and s['label'] == 'C2' and s['improvement'] == pytest.approx(.02)
    assert e.final_selection([('C1', .03), ('C2', .03+5e-10)], .04)['label'] == 'C1'       # tie -> C1
    assert e.final_selection([('C1', .03), ('C2', .03-5e-10)], .04)['label'] == 'C1'
    assert e.final_selection([('C1', .03)], .03)['status'] == 'no_AS_improvement'           # strictly greater than zero
    assert e.final_selection([('C1', .05), ('C2', .06)], .04)['status'] == 'no_AS_improvement'
    assert e.TEXT['not_tested_no_feasible'] != e.TEXT['not_tested_no_improvement']


# ------------------------------------------------------------------ decision tree
N = 1024


def case(dmean, dsd, rmean=0., amean=0.):
    return e.decide(vec(dmean, dsd, N), vec(rmean, .0042, N, 1), vec(amean, .0146, N, 2))


def test_decision_tree_every_branch():
    c = case(.0086, .0108)
    assert (c['category'], c['H'], c['E']) == ('established', True, True) and c['text'].startswith('成立')
    c = case(.0086, .0108, rmean=.003)
    assert c['category'] == 'honest_breach' and c['honest_excess']['regret'] > 0
    c = case(.0086, .0108, amean=.012)
    assert c['category'] == 'honest_breach'
    c = case(.002, .0135, rmean=.003)
    assert c['category'] == 'both_fail'
    c = case(.002, .0135)
    assert (c['category'], c['sub']) == ('insufficient_effect', 'a') and c['LB_gt_0'] is True      # (a) and LB>0 hold together: label (a)
    c = case(.0045, .0135)
    assert (c['category'], c['sub']) == ('insufficient_effect', 'b')
    c = case(.0, .15)
    assert (c['category'], c['sub']) == ('insufficient_effect', 'c') and not c['LB_gt_0']
    for sub in 'abc':
        assert e.TEXT['effect_'+sub].startswith('效果不足')


def test_decision_strict_inequalities():
    D = vec(.0086, .0108, N)
    lb = e.lower_bound(D)
    dr, da = vec(0., .0042, N, 1), vec(0., .0146, N, 2)
    assert e.decide(D, dr, da, delta=lb)['E'] is False               # LB equal to delta is not E
    assert e.decide(D, dr, da, delta=lb-1e-9)['E'] is True
    ub = e.upper_bound(dr)
    assert e.decide(D, dr, da, lim_r=ub)['H'] is True                # UB equal to the limit passes
    assert e.decide(D, dr, da, lim_r=ub-1e-6)['H'] is False


# ------------------------------------------------------------------ manifest
def fake_manifest(smoke=True):
    m = dict(schema='x', smoke=smoke, params=json.loads(json.dumps(e.SMOKE_PARAMS if smoke else e.PARAMS)), seeds={'a': [1, 2]})
    m['sha256'] = e.digest(m)
    return m


def test_manifest_changes_are_detected_and_params_must_match_code(tmp_path):
    m = fake_manifest()
    assert e.verify_manifest_dict(m) is m
    for edit in (lambda x: x['params'].update(delta=.01), lambda x: x['seeds'].update(a=[1, 3]), lambda x: x.update(design_sha256='0')):
        bad = copy.deepcopy(m)
        edit(bad)
        with pytest.raises(ValueError):
            e.verify_manifest_dict(bad)
    reg = fake_manifest(False)
    reg['params']['attacker']['gens'] = 120
    reg['sha256'] = e.digest({k: v for k, v in reg.items() if k != 'sha256'})
    with pytest.raises(ValueError):                                 # self-consistent hash, but not the coded design
        e.verify_manifest_dict(reg)
    f = tmp_path/'a.txt'
    f.write_text('x')
    h = {'a.txt': e.sha(f)}
    e.check_hashes(h, tmp_path)
    f.write_text('y')
    with pytest.raises(ValueError):
        e.check_hashes(h, tmp_path)
    with pytest.raises(ValueError):
        e.check_hashes({'missing.txt': '0'}, tmp_path)


def test_registered_parameters_match_the_design():
    p = e.PARAMS
    assert (p['delta'], p['lim_regret'], p['lim_alpha'], p['n_test']) == (.005, .002, .01, 1024)
    assert (p['T_train'], p['T_val'], p['exclude']) == (20000, 100000, 5000)
    assert p['attacker'] | dict() == dict(lam=64, sigma0=.15, n_at=32, gens=240, ext_gens=240, val_every=20, chains=3, opt_seed=93000, sat_threshold=.001)
    assert p['defender']['lam'] == 8 and p['defender']['gens'] == 12 and p['defender']['eval_gens'] == [4, 8, 12] and p['defender']['n_dt'] == 16
    assert e.encode(e.BASE_CFG) == [0, 0, -.25] and e.v1.decode(e.encode(e.BASE_CFG)) == e.BASE_CFG
    assert e.SMOKE_PARAMS['n_test'] == 0 and 'TEST' not in e.block_seeds(smoke=True)


# ------------------------------------------------------------------ ledger
def test_ledger_is_one_shot_atomic_and_detects_tampering(tmp_path):
    con = dict(seeds=[1, 2], n_policies=2, T_score=10, stage='x')
    calls = [0]

    def runner():
        calls[0] += 1
        return dict(T_score=10, snap=dict(util_b=np.ones((2, 2, 3)).tolist(), reg_b=np.ones((2, 2)).tolist(), pen=np.zeros((2, 2, 3)).tolist()))
    raw = e.once(tmp_path, 'orig_main', con, runner)
    again = e.once(tmp_path, 'orig_main', con, runner)
    assert calls[0] == 1 and raw == again                                   # completed evaluations are never re-simulated
    led = e.read(tmp_path/'ledger.json')
    assert led['orig_main']['status'] == 'completed' and not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError):
        e.once(tmp_path, 'orig_main', dict(con, seeds=[1, 3]), runner)       # changed contract
    r = e.read(tmp_path/'orig_main_raw.json')
    r['snap']['util_b'][0][0][0] = 5.
    e.write(tmp_path/'orig_main_raw.json', r)
    with pytest.raises(ValueError):
        e.once(tmp_path, 'orig_main', con, runner)                           # tampered raw
    # started but no raw (crash): stop, never silently repeat
    e.write(tmp_path/'ledger.json', dict(crash=dict(status='started', contract_sha256=e.digest(con))))
    with pytest.raises(RuntimeError):
        e.once(tmp_path, 'crash', con, runner)
    assert calls[0] == 1


# ------------------------------------------------------------------ synthetic pipeline
class Fake:
    """Deterministic stand-in for the simulator: MLP thetas give a gain that shrinks with a larger audit horizon."""
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def __call__(self, m, cfg, pols, sd, T):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError('synthetic interruption')
        p = m['params']
        n = len(sd)
        base = np.array([float(np.tanh(np.asarray(getattr(q, 'theta', np.zeros(1))).sum())) for q in pols])
        gain = np.maximum(base-base[0], 0)*(1.0-.05*(cfg['tol']-6.))
        ts = T-p['exclude']
        u = np.full((len(pols), n, 3), 100.)
        noise = .02*np.sin(np.asarray(sd, float))
        u[:, :, 1] += (gain[:, None]*.02+noise[None, :]*0+0)*ts
        reg = np.zeros((len(pols), n))
        reg[0] = 10.+(cfg['tol']-6.)*(1+.1*noise)
        return dict(T_score=ts, snap=dict(util_b=u.tolist(), reg_b=reg.tolist(), pen=np.zeros((len(pols), n, 3)).tolist()))


def smoke_env(tmp_path, monkeypatch, fake):
    m = dict(schema='s', smoke=True, params=json.loads(json.dumps(e.SMOKE_PARAMS)), seeds=e.block_seeds(smoke=True), sha256='synthetic')
    e.MODE.update(smoke=True, out=str(tmp_path), backup=False, start=None, graph=False, fallback=False, device='cpu')
    monkeypatch.setattr(e, 'sim', fake)
    rng = np.random.default_rng(7)
    cks = [dict(theta=rng.normal(0, .01, 55)) for _ in range(3)]

    def pool_with(extra):
        names = ['honest']+[f'hw{i}' for i in range(4)]
        pols = [e.honest_pol()]+[e.mlp(np.full(55, .001*(i+1))) for i in range(4)]
        for nm, th in extra:
            names.append(dict(name=nm, theta_sha256=e.theta_id(th)))
            pols.append(e.mlp(th))
        return names, pols, cks
    monkeypatch.setattr(e, 'pool_with', pool_with)
    return m


@pytest.fixture(autouse=True)
def _reset_mode():
    yield
    e.MODE.pop('out', None)
    e.MODE.update(smoke=False, backup=False, start=None)


def run_all(m):
    for st, fn in (('s1', e.stage_s1), ('s2', e.stage_s2), ('sx', e.stage_sx), ('select', e.stage_select)):
        if st not in e.stages_done():
            fn(m)


def test_pipeline_runs_resumes_and_never_reads_test_seeds(tmp_path, monkeypatch):
    seen = []
    base_require = e.require_seeds

    def spy(sd, stage, m, unlocked=False):
        seen.append((stage, tuple(sd)))
        return base_require(sd, stage, m, unlocked)
    fake = Fake()
    m = smoke_env(tmp_path/'fresh', monkeypatch, fake)
    monkeypatch.setattr(e, 'require_seeds', spy)
    run_all(m)
    sel = e.read(tmp_path/'fresh/selection_frozen.json')
    assert sel['status'] in ('go', 'no_AS_improvement') and sel['manifest_sha256'] == 'synthetic'
    rec = e.read(tmp_path/'fresh/freeze_record.json')
    assert rec['selection_frozen_sha256'] == e.sha(tmp_path/'fresh/selection_frozen.json')
    assert {s for s, _ in seen} <= {'dt', 'ds', 'at', 'as'} and all(not (set(sd) & set(range(72_400_000, 72_401_024))) for _, sd in seen)
    sat = e.read(tmp_path/'fresh/saturation.json')
    assert set(sat['final']) == set(sat['decisions']) and 'orig' in sat['final']
    for label, info in sat['decisions'].items():          # extension happens exactly when the rule asked for it
        want = 2*m['params']['attacker']['gens'] if info['extend'] else m['params']['attacker']['gens']
        assert all(e.read(e.chain_path(label, r))['opt']['gen'] == want for r in range(3))
    # idempotent: a second pass does no simulation and cannot refreeze
    n = fake.calls
    run_all(m)
    assert fake.calls == n
    # an interruption part-way resumes to the same selection
    interrupted = Fake(fail_at=fake.calls//3)
    m2 = smoke_env(tmp_path/'resumed', monkeypatch, interrupted)
    with pytest.raises(RuntimeError):
        run_all(m2)
    monkeypatch.setattr(e, 'sim', Fake())
    run_all(m2)
    sel2 = e.read(tmp_path/'resumed/selection_frozen.json')
    for k in ('status', 'G_AS_astar', 'choice', 'attacker_theta_sha256'):
        assert sel2[k] == sel[k]


def test_select_refuses_before_sx_and_registers_existing_selection_without_recompute(tmp_path, monkeypatch):
    m = smoke_env(tmp_path, monkeypatch, Fake())
    with pytest.raises(RuntimeError):
        e.stage_s2(m)
    with pytest.raises(RuntimeError):
        e.stage_select(m)
    run_all(m)
    (tmp_path/'freeze_record.json').unlink()
    e.state_load  # noqa
    st = e.read(tmp_path/'status.json')
    st['stages_done'].remove('select')
    e.write(tmp_path/'status.json', st)
    before = e.sha(tmp_path/'selection_frozen.json')
    e.stage_select(m)
    assert e.sha(tmp_path/'selection_frozen.json') == before and (tmp_path/'freeze_record.json').exists()


def test_chain_resume_rejects_other_budget_or_defender(tmp_path, monkeypatch):
    m = smoke_env(tmp_path, monkeypatch, Fake())
    _, _, cks = e.pool_with([])
    e.attack_chain(m, 'orig', e.BASE_CFG, 0, cks, 2)
    with pytest.raises(ValueError):
        e.attack_chain(m, 'orig', dict(e.BASE_CFG, L=1000.), 0, cks, 4)
    m3 = copy.deepcopy(m)
    m3['params']['attacker']['lam'] = 4
    with pytest.raises(ValueError):
        e.attack_chain(m3, 'orig', e.BASE_CFG, 0, cks, 4)


# ------------------------------------------------------------------ test stage (synthetic, unlock guard + ledger)
def official_env(tmp_path, monkeypatch, status='go'):
    m = dict(schema='o', smoke=False, params=json.loads(json.dumps(e.PARAMS)), seeds=e.block_seeds(), sha256='synthetic')
    e.MODE.update(smoke=False, out=str(tmp_path), backup=False, start=None, device='cpu')
    seen = []

    def fake(m_, cfg, pols, sd, T):
        seen.append(len(sd))
        shift = .01 if cfg == e.BASE_CFG else .0
        u = np.full((len(pols), len(sd), 3), 100.)
        u[1:, :, 1] += (shift+.001*np.sin(np.asarray(sd, float)))[None, :]*(T-5000)
        return dict(T_score=T-5000, snap=dict(util_b=u.tolist(), reg_b=np.ones((len(pols), len(sd))).tolist(), pen=np.zeros((len(pols), len(sd), 3)).tolist()))
    monkeypatch.setattr(e, 'sim', fake)
    th = [0.001]*55
    sel = dict(schema='x', manifest_sha256='synthetic', status=status, wording='未進 test：無候選滿足誠實約束',
               choice=dict(status=status, label='C1', cfg=dict(e.BASE_CFG, L=1000.), L_at_upper_bound=False),
               astar=dict(orig=dict(name='frozen_run0', theta=th), C1=dict(name='C1_adapt0', theta=th)),
               saturation=dict(final=dict(orig=dict(attack_may_be_unsaturated=False), C1=dict(attack_may_be_unsaturated=True))))
    e.write(tmp_path/'selection_frozen.json', sel)
    e.write(tmp_path/'freeze_record.json', dict(manifest_sha256='synthetic', selection_frozen_sha256=e.sha(tmp_path/'selection_frozen.json')))
    monkeypatch.setattr(e, 'load_manifest', lambda clean=True: m)
    monkeypatch.setattr(e, 'verify_freeze', lambda m_: sel)
    monkeypatch.setattr(e, 'secondary_thetas', lambda s: [('frozen_run0', th), ('C1_adapt0', th)])
    return m, seen


def test_test_stage_unlocks_only_via_freeze_and_runs_each_evaluation_once(tmp_path, monkeypatch):
    m, seen = official_env(tmp_path, monkeypatch)
    res = e.run_test()
    assert res['category'] == 'established' and res['attack_may_be_unsaturated'] == dict(orig=False, C1=True)
    assert sum(seen[:16]) == 1024 and set(seen) == {64}
    led = e.read(tmp_path/'test/ledger.json')
    assert set(led) == {'orig_main', 'cand_main', 'orig_secondary', 'cand_secondary'}
    n = len(seen)
    e.run_test()                                    # nothing is simulated again
    assert len(seen) == n
    assert e.read(tmp_path/'result.json')['category'] == 'established'
    assert e.read(tmp_path/'secondary.json')['attackers'] == ['frozen_run0', 'C1_adapt0']
    del led


def test_test_stage_without_candidate_uses_no_test_seeds(tmp_path, monkeypatch):
    m, seen = official_env(tmp_path, monkeypatch, status='no_feasible')
    res = e.run_test()
    assert res['category'] == 'not_tested' and res['test_seeds_used'] is False and seen == []
    assert not (tmp_path/'test').exists()


def test_verify_freeze_requires_files_matching_hash_and_commit(tmp_path, monkeypatch):
    m = dict(smoke=False, sha256='synthetic')
    e.MODE.update(out=str(tmp_path), smoke=False)
    with pytest.raises(ValueError):
        e.verify_freeze(m)
    e.write(tmp_path/'selection_frozen.json', dict(manifest_sha256='synthetic', status='go'))
    e.write(tmp_path/'freeze_record.json', dict(manifest_sha256='synthetic', selection_frozen_sha256=e.sha(tmp_path/'selection_frozen.json')))
    monkeypatch.setattr(e.plan, 'is_committed_and_clean', lambda root, rel: False)
    with pytest.raises(ValueError):
        e.verify_freeze(m)                                       # not committed
    monkeypatch.setattr(e.plan, 'is_committed_and_clean', lambda root, rel: True)
    assert e.verify_freeze(m)['status'] == 'go'
    e.write(tmp_path/'selection_frozen.json', dict(manifest_sha256='synthetic', status='go', x=1))
    with pytest.raises(ValueError):
        e.verify_freeze(m)                                       # tampered after registration
    with pytest.raises(ValueError):
        e.verify_freeze(dict(smoke=True, sha256='synthetic'))    # smoke never opens test
