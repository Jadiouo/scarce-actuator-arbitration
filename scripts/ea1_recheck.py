#!/usr/bin/env python3
"""E-A1 independent recomputation and post-hoc numbers (read-only on results/; writes only results/direction2/ea1/posthoc.json).

Run from the repository root:   python3 scripts/ea1_recheck.py

It imports nothing from this repository's scripts/ or arbitration/ packages.  Everything is derived from the raw files
under results/direction2/ea1 and results/direction2/e1v2.  G = (util_b[k,:,1] - util_b[0,:,1]) / T_score (policy 0 is the
honest policy; same formula as scripts/e1_pilot.py::metrics).  One-sided 95% bounds: mean -/+ t(0.95,n-1)*sd(ddof=1)/sqrt(n).

Sections of posthoc.json
  check       registered numbers (result.json, secondary.json) recomputed and compared
  post_hoc    every number cited in docs/direction2/results/ea1.md that is NOT in result.json/secondary.json.
              These are post hoc, uncorrected for multiplicity, and never enter the registered decision.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EA, E1 = ROOT/'results/direction2/ea1', ROOT/'results/direction2/e1v2'
OUT = EA/'posthoc.json'


def rd(p):
    return json.loads(Path(p).read_text())


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gains(raw):
    u = np.asarray(raw['snap']['util_b'], float)[:, :, 1]
    return (u[1:]-u[0])/raw['T_score']


def honest(raw):
    ts = raw['T_score']
    return np.asarray(raw['snap']['reg_b'], float)[0]/ts, np.asarray(raw['snap']['pen'], float)[0].mean(-1)/ts


def iv(x, conf=.95):
    x = np.asarray(x, float)
    n, m, s = len(x), float(x.mean()), float(x.std(ddof=1))
    t = float(stats.t.ppf(conf, n-1))
    se = s/math.sqrt(n)
    return dict(mean=m, LB=m-t*se, UB=m+t*se, SD=s, SE=se, n=n)


def welch(a, b):
    """b - a for two independent samples (different seed sets)."""
    d = float(np.mean(b)-np.mean(a))
    se = math.sqrt(np.var(a, ddof=1)/len(a)+np.var(b, ddof=1)/len(b))
    return dict(diff=d, SE=se, z=d/se, n_a=len(a), n_b=len(b))


def main():
    res, sec, sel = rd(EA/'result.json'), rd(EA/'secondary.json'), rd(EA/'selection_frozen.json')
    raw = {k: rd(EA/'test'/f'{k}_raw.json') for k in ('main_C2', 'main_orig', 'sec_C2', 'sec_orig')}
    e1res, e1sel = rd(E1/'result.json'), rd(E1/'selection_frozen.json')
    e1raw = {k: rd(E1/'test'/f'{k}_raw.json') for k in ('orig_main', 'cand_main', 'orig_secondary', 'cand_secondary')}
    names = [e['name'] for e in sel['entries']]
    ix = {n: i for i, n in enumerate(names)}
    STAR, F0, C0, FZ0 = ix['orig_adapt0'], ix['fresh_run0'], ix['C2_adapt0'], ix['frozen_run0']
    gmc, gmo, gsc, gso = (gains(raw[k]) for k in ('main_C2', 'main_orig', 'sec_C2', 'sec_orig'))
    ei = {n: i for i, n in enumerate(names[:12])}
    gesc, geso = gains(e1raw['cand_secondary']), gains(e1raw['orig_secondary'])
    gemc, gemo = gains(e1raw['cand_main'])[0], gains(e1raw['orig_main'])[0]
    seeds_ea, seeds_e1 = raw['main_C2']['contract']['seeds'], e1raw['cand_secondary']['contract']['seeds']
    assert seeds_ea == list(range(82400000, 82401024)) and raw['main_C2']['T_score'] == 95000
    assert raw['main_C2']['contract']['theta_sha256'] == [sel['astar']['theta_sha256'], sel['A_fresh']['theta_sha256']]
    out = dict(schema='ea1-posthoc/1', note=('read-only recomputation from raw; "check" compares registered numbers, '
                                            '"post_hoc" numbers are not registered, not corrected for multiplicity, not part of the decision'),
               env=dict(numpy=np.__version__, scipy=scipy.__version__, t_0p95_df1023=float(stats.t.ppf(.95, 1023))))

    # ---------------------------------------------------------------- check: registered numbers
    d = gmc[1]-gmc[0]
    p = iv(d)
    eps = res['eps']
    cat = ('stronger' if p['LB'] > eps else 'not_stronger:' + ('weaker' if p['UB'] < 0 else 'small_gain' if p['LB'] > 0 else 'flat')
           if p['UB'] < eps else 'insufficient')
    pairs = []

    def cmp(tag, saved, mine):
        pairs.append((tag, saved == mine, abs(saved-mine)))

    for k in ('mean', 'LB', 'UB'):
        cmp('result.' + k, res[{'mean': 'mean_delta'}.get(k, k)], p[k])
    cmp('result.G_C2_astar', res['G_C2_astar'], float(gmc[0].mean()))
    cmp('result.G_C2_A_fresh', res['G_C2_A_fresh'], float(gmc[1].mean()))
    for k in ('frozen_run0', 'frozen_run1', 'frozen_run2'):
        for f, v in iv(gso[ix[k]]-gso[STAR]).items():
            if f in ('mean', 'LB', 'UB'):
                cmp(f'orig_type_delta.{k}.{f}', sec['orig_type_delta'][k][f], v)
    for k in ('fresh_run0', 'fresh_run1', 'fresh_run2'):
        for f, v in iv(gsc[ix[k]]-gsc[STAR]).items():
            if f in ('mean', 'LB', 'UB'):
                cmp(f'fresh_vs_astar_C2.{k}.{f}', sec['fresh_vs_astar_C2'][k][f], v)
    dprime, dnew = iv(gmo[0]-gmc[1]), iv(gmo[0]-gmc[0])
    for f in ('mean', 'LB', 'UB'):
        cmp('D_prime.' + f, sec['D_prime'][f], dprime[f])
        cmp('D_new.' + f, sec['D_new'][f], dnew[f])
    for n in names:
        cmp('G_C2_by_entry.' + n, sec['G_C2_by_entry'][n], float(gsc[ix[n]].mean()))
        cmp('G_orig_by_entry.' + n, sec['G_orig_by_entry'][n], float(gso[ix[n]].mean()))
    rc2, ac2 = honest(raw['sec_C2'])
    ro, ao = honest(raw['sec_orig'])
    hr, ha = iv(rc2-ro), iv(ac2-ao)
    for f in ('mean', 'LB', 'UB'):
        cmp('honest_service.delta_regret.' + f, sec['honest_service']['delta_regret'][f], hr[f])
        cmp('honest_service.delta_alpha.' + f, sec['honest_service']['delta_alpha'][f], ha[f])
    t_saved = e1res['honest']['t']
    out['check'] = dict(
        n_compared=len(pairs), n_not_bitwise=sum(not b for _, b, _ in pairs), max_abs_diff=max(a for _, _, a in pairs),
        not_bitwise_fields=[t for t, b, _ in pairs if not b], all_not_bitwise_are_LB_or_UB=all(t.endswith(('.LB', '.UB')) for t, b, _ in pairs if not b),
        primary=dict(p, decision_recomputed=cat, decision_saved=f"{res['category']}:{res['sub']}", margin_eps_minus_UB=eps-p['UB'], t=float(stats.t.ppf(.95, 1023))),
        main_equals_sec_bitwise=dict(
            a_star_C2=bool(np.array_equal(gmc[0], gsc[STAR])), A_fresh_C2=bool(np.array_equal(gmc[1], gsc[F0])),
            a_star_orig=bool(np.array_equal(gmo[0], gso[STAR])), A_fresh_orig=bool(np.array_equal(gmo[1], gso[F0]))),
        t_ulp=dict(t_e1v2_saved=t_saved, t_here=float(stats.t.ppf(.95, 1023)), ulp=float(np.spacing(t_saved))),
        seeds=dict(ea1_test=[seeds_ea[0], seeds_ea[-1], len(seeds_ea)], e1v2_test=[seeds_e1[0], seeds_e1[-1], len(seeds_e1)],
                   intersection=len(set(seeds_ea) & set(seeds_e1))),
        honest_absolute=dict(regret_C2=float(rc2.mean()), regret_orig=float(ro.mean()), alpha_C2=float(ac2.mean()), alpha_orig=float(ao.mean())))
    # LB with the t value of the astarc scipy (1.15.3) vs this environment
    se_ = p['SE']
    out['check']['primary_LB_with_t_astarc'] = float(p['mean']-t_saved*se_)
    out['check']['primary_UB_with_t_astarc'] = float(p['mean']+t_saved*se_)
    out['check']['primary_UB_planned_SE_0p000297'] = float(p['mean']+stats.t.ppf(.95, 1023)*0.000297)
    out['check']['primary_UB_planned_SD_0p0095'] = float(p['mean']+stats.t.ppf(.95, 1023)*0.0095/32)

    # report_check files (astarc copy retrieved into the repo; local copy)
    rc = {}
    for f in sorted(EA.glob('report_check_*.json')):
        r = rd(f)
        rc[f.name] = dict(sha256=fsha(f), matches=r['matches'], numpy=r['numpy'], scipy=r['scipy'],
                          result_equals_saved=(r['result'] == res),
                          differing_result_keys={k: dict(file=r['result'][k], saved=res[k]) for k in res if r['result'].get(k) != res[k]})
    out['check']['report_check'] = rc

    # ---------------------------------------------------------------- post hoc
    ph = {}
    # robustness of the registered interval (descriptive)
    ph['primary_robustness'] = dict(UB_onesided_0p99=iv(d, .99)['UB'], UB_onesided_0p995=iv(d, .995)['UB'],
                                    trimmed_mean_1pct=float(stats.trim_mean(d, .01)), lag1_autocorr=float(np.corrcoef(d[:-1], d[1:])[0, 1]),
                                    excess_kurtosis=float(stats.kurtosis(d)),
                                    UB_over_D_new=p['UB']/dnew['mean'], UB_over_D_e1v2=p['UB']/e1res['mean_D'], eps_over_D_e1v2=eps/e1res['mean_D'])
    # entry table
    th = {}
    for e in sel['entries']:
        th.setdefault(e['theta_sha256'], []).append(e['name'])
    dup = [v for v in th.values() if len(v) > 1]
    ph['entries'] = dict(
        table={n: dict(G_orig=float(gso[ix[n]].mean()), G_C2=float(gsc[ix[n]].mean()), G_orig_minus_G_C2=float((gso[ix[n]]-gsc[ix[n]]).mean()),
                       G_C2_minus_astar=float((gsc[ix[n]]-gsc[STAR]).mean())) for n in names},
        D_range=[min(float((gso[ix[n]]-gsc[ix[n]]).mean()) for n in names), max(float((gso[ix[n]]-gsc[ix[n]]).mean()) for n in names)],
        D_max_entry_ci=iv(gso[ix['frozen_run2']]-gsc[ix['frozen_run2']]),
        theta_duplicate_groups=dup, n_distinct_among_15=len(th), n_distinct_among_first_12=len({e['theta_sha256'] for e in sel['entries'][:12]}),
        rank_by_G_C2_top5=sorted(names, key=lambda n: -gsc[ix[n]].mean())[:5], rank_by_G_orig_top5=sorted(names, key=lambda n: -gso[ix[n]].mean())[:5])
    # E1 v2 TEST vs EA1_TEST (different seeds)
    a1c, a1o = gesc[ei['orig_adapt0']], geso[ei['orig_adapt0']]
    dc = {n: float(gsc[ix[n]].mean()-gesc[ei[n]].mean()) for n in names[:12]}
    do = {n: float(gso[ix[n]].mean()-geso[ei[n]].mean()) for n in names[:12]}
    ph['ea1_vs_e1v2_test'] = dict(
        a_star_G_C2=dict(e1v2=float(a1c.mean()), ea1=float(gsc[STAR].mean()), **welch(a1c, gsc[STAR])),
        a_star_G_orig=dict(e1v2=float(a1o.mean()), ea1=float(gso[STAR].mean()), **welch(a1o, gso[STAR])),
        a_star_D=dict(e1v2=float((a1o-a1c).mean()), ea1=float((gso[STAR]-gsc[STAR]).mean()), **welch(a1o-a1c, gso[STAR]-gsc[STAR])),
        e1v2_a_star_saved_check=dict(G_C2=float(a1c.mean()), saved=e1res['G_cand_mean'], G_orig=float(a1o.mean()), saved_orig=e1res['G_orig_mean']),
        entry_diff_C2_range=[min(dc.values()), max(dc.values())], entry_diff_orig_range=[min(do.values()), max(do.values())],
        all_24_diffs_negative=all(v < 0 for v in list(dc.values())+list(do.values())),
        n_distinct_params_among_the_24=2*len({e['theta_sha256'] for e in sel['entries'][:12]}),
        e1v2_D_recomputed=iv(geso[ei['orig_adapt0']]-gesc[ei['orig_adapt0']]))
    # D samples incl. EA1_SEL and AS
    names_sel = sel['descriptive_SEL']['C2']['names']
    gs_c2, gs_o = gains(rd(EA/sel['descriptive_SEL']['C2']['raw'])), gains(rd(EA/sel['descriptive_SEL']['orig']['raw']))
    D_sel = gs_o[names_sel.index('orig_adapt0')]-gs_c2[names_sel.index('orig_adapt0')]
    dr = rd(EA/'drift.json')
    raw_do, raw_dc = rd(EA/dr['orig']['raw']), rd(EA/dr['C2']['raw'])
    D_as = gains(raw_do)[0]-gains(raw_dc)[0]
    D_e1 = a1o-a1c
    D_ea = gso[STAR]-gsc[STAR]
    D_pool = np.r_[D_ea, D_e1]
    ph['D_samples'] = dict(AS_n128=iv(D_as), EA1_SEL_n128=iv(D_sel), E1v2_TEST_n1024=iv(D_e1), EA1_TEST_n1024=iv(D_ea), pooled_tests_n2048=iv(D_pool),
                           AS_vs_EA1_SEL=welch(D_as, D_sel), AS_vs_EA1_TEST=welch(D_as, D_ea), AS_vs_E1v2_TEST=welch(D_as, D_e1),
                           AS_vs_pooled_tests=welch(D_as, D_pool), EA1_SEL_vs_pooled_tests=welch(D_sel, D_pool),
                           note='AS values come from the astarc drift raw files; they equal the E1 v2 AS values bitwise (see drift)')
    # C2_adapt0 against a*, A_fresh and its parent
    ea, eb = gsc[C0]-gsc[STAR], gesc[ei['C2_adapt0']]-gesc[ei['orig_adapt0']]
    pooled = np.r_[ea, eb]
    k = len(th)-1      # distinct parameter vectors other than a* among the 15 entries
    mm, se2 = float(ea.mean()), float(ea.std(ddof=1)/math.sqrt(len(ea)))
    tb = float(stats.t.ppf(1-.05/k, len(ea)-1))
    atk = rd(E1/'attacker/C2_run0.json')
    ent = {e['name']: e for e in sel['entries']}
    as_c2 = e1sel['transfer_matrix_AS']['C2']['G']
    G_SEL_C2 = dict(zip(names_sel, sel['descriptive_SEL']['C2']['G']))
    lam_f, ns_f = rd(EA/'manifest.json')['params']['attacker']['lam'], rd(EA/'manifest.json')['params']['attacker']['n_S']
    lam_a, n_a = rd(E1/'manifest.json')['params']['attacker']['lam'], rd(E1/'manifest.json')['params']['attacker']['n_at']
    orig0 = rd(E1/'attacker/orig_run0.json')
    ph['C2_adapt0'] = dict(
        n_other_distinct_params=k,
        above_A_fresh_in_EA1_TEST_G_C2=[n for n in names if gsc[ix[n]].mean() > gsc[F0].mean()],
        above_astar_in_EA1_TEST_G_C2=[n for n in names if gsc[ix[n]].mean() > gsc[STAR].mean()],
        EA1_TEST_vs_astar=iv(ea), E1v2_TEST_vs_astar=iv(eb), pooled_2048_vs_astar=iv(pooled),
        EA1_TEST_vs_A_fresh=iv(gsc[C0]-gsc[F0]), EA1_TEST_vs_parent_frozen_run0_C2=iv(gsc[C0]-gsc[FZ0]),
        E1v2_TEST_vs_parent_frozen_run0_C2=iv(gesc[ei['C2_adapt0']]-gesc[ei['frozen_run0']]),
        astar_vs_parent_frozen_run0_orig_EA1_TEST=iv(gso[STAR]-gso[FZ0]),
        bonferroni_EA1_TEST_vs_astar=dict(k=k, LB=mm-tb*se2, UB=mm+tb*se2),
        descriptive_D_with_C2_adapt0=iv(gso[STAR]-gsc[C0]),
        G_C2_EA1_SEL=dict(C2_adapt0=G_SEL_C2['C2_adapt0'], A_fresh=G_SEL_C2['fresh_run0'], a_star=G_SEL_C2['orig_adapt0'], argmax=max(G_SEL_C2, key=G_SEL_C2.get)),
        EA1_SEL_vs_astar=iv(gs_c2[names_sel.index('C2_adapt0')]-gs_c2[names_sel.index('orig_adapt0')]),
        G_C2_E1v2_TEST=dict(C2_adapt0=float(gesc[ei['C2_adapt0']].mean()), a_star=float(a1c.mean()), argmax_of_12=max(ei, key=lambda n: gesc[ei[n]].mean())),
        G_orig_EA1_TEST=dict(C2_adapt0=float(gso[C0].mean()), a_star=float(gso[STAR].mean()), A_fresh=float(gso[F0].mean()),
                             C2_adapt0_minus_astar=iv(gso[C0]-gso[STAR]), argmax=max(names, key=lambda n: gso[ix[n]].mean())),
        G_C2_AS=dict(C2_adapt0=as_c2['C2_adapt0'], a_star=as_c2['orig_adapt0'], diff=as_c2['C2_adapt0']-as_c2['orig_adapt0']),
        C2_adapt0_theta_is_C2_run0_gens=[r['gen'] for r in atk['records'] if r.get('theta') == ent['C2_adapt0']['theta']],
        astar_theta_is_orig_run0_gens=[r['gen'] for r in orig0['records'] if r.get('theta') == sel['astar']['theta']],
        frozen_run0_is_orig_run0_g0=orig0['records'][0].get('theta') == ent['frozen_run0']['theta'],
        frozen_run0_is_C2_run0_g0=atk['records'][0].get('theta') == ent['frozen_run0']['theta'],
        sample_budgets=dict(A_fresh_gen300=300*lam_f*ns_f, a_star_parent_350_gens=350*lam_f*ns_f, a_star_adapt_60_gens=60*lam_a*n_a,
                            C2_adapt0_adapt_40_gens=40*lam_a*n_a, note='generations x lambda x n_S (cold-start chains) or x n_at (E1 v2 adaptation chains)'))
    # second-round additions (post hoc): transfer between settings, difference-in-differences, increments over the parent
    def distinct(ns):
        seen, keep = set(), []
        for n in ns:
            h = ent[n]['theta_sha256']
            if h not in seen:
                seen.add(h)
                keep.append(n)
        return keep

    def slopes(go_, gc_):
        a = stats.linregress(go_, gc_)
        b = stats.linregress(go_, np.asarray(go_)-np.asarray(gc_))
        return dict(n_params=len(go_), slope_G_C2_on_G_orig=float(a.slope), se=float(a.stderr), r=float(a.rvalue),
                    slope_D_on_G_orig=float(b.slope), se_D=float(b.stderr), r_D=float(b.rvalue), p_D=float(b.pvalue))

    d15, d12 = distinct(names), distinct(names[:12])
    ph['transfer_between_settings'] = dict(
        EA1_TEST=dict(params=d15, **slopes([gso[ix[n]].mean() for n in d15], [gsc[ix[n]].mean() for n in d15])),
        E1v2_TEST=dict(params=d12, **slopes([geso[ei[n]].mean() for n in d12], [gesc[ei[n]].mean() for n in d12])),
        max_G_orig_above_astar_among_15=max(float(gso[ix[n]].mean()-gso[STAR].mean()) for n in names),
        note='across-entry slopes; entries share seeds and are not independent; descriptive only')
    frz = [ix[f'frozen_run{i}'] for i in range(3)]
    frs = [ix[f'fresh_run{i}'] for i in range(3)]
    frz12 = [ix['frozen_run1'], ix['frozen_run2']]
    did = lambda idx: iv((gsc[idx].mean(0)-gsc[STAR])-(gso[idx].mean(0)-gso[STAR]))
    ph['difference_in_differences'] = dict(
        frozen_mean_minus_astar=dict(orig=iv(gso[frz].mean(0)-gso[STAR]), C2=iv(gsc[frz].mean(0)-gsc[STAR]), C2_minus_orig=did(frz)),
        frozen_run1_2_mean_minus_astar=dict(orig=iv(gso[frz12].mean(0)-gso[STAR]), C2=iv(gsc[frz12].mean(0)-gsc[STAR])),
        fresh_mean_minus_astar=dict(orig=iv(gso[frs].mean(0)-gso[STAR]), C2=iv(gsc[frs].mean(0)-gsc[STAR]), C2_minus_orig=did(frs)),
        fresh_C2_minus_frozen_orig_both_vs_astar=iv((gsc[frs].mean(0)-gsc[STAR])-(gso[frz].mean(0)-gso[STAR])),
        fresh_minus_frozen_in_orig=iv(gso[frs].mean(0)-gso[frz].mean(0)),
        note='paired on the EA1_TEST seeds; frozen_run0 (parent of a*) is inside the frozen mean')
    a_o_e1 = geso[ei['orig_adapt0']]-geso[ei['frozen_run0']]
    a_o_ea = gso[STAR]-gso[FZ0]
    c_e1 = gesc[ei['C2_adapt0']]-gesc[ei['frozen_run0']]
    c_ea = gsc[C0]-gsc[FZ0]
    rec_c = {r['gen']: r['G_AS'] for r in atk['records']}
    rec_o = {r['gen']: r['G_AS'] for r in orig0['records']}
    ph['parent_increments'] = dict(
        astar_minus_frozen_run0_orig=dict(EA1_TEST=iv(a_o_ea), E1v2_TEST=iv(a_o_e1), pooled_2048=iv(np.r_[a_o_ea, a_o_e1])),
        C2_adapt0_minus_frozen_run0_C2=dict(EA1_TEST=iv(c_ea), E1v2_TEST=iv(c_e1), pooled_2048=iv(np.r_[c_ea, c_e1])),
        AS_chain_records=dict(C2_run0_g0=rec_c[0], C2_run0_g40=rec_c[40], C2_run0_gain_g0_to_g40=rec_c[40]-rec_c[0],
                              orig_run0_g0=rec_o[0], orig_run0_g60=rec_o[60], orig_run0_gain_g0_to_g60=rec_o[60]-rec_o[0]),
        note='the C2 chain starts from a checkpoint trained for the original setting (warm-start asymmetry); descriptive only')
    ph['selection_rules_for_C2_adapt0'] = dict(
        AS_registered_rule_argmax_over_12=max(as_c2, key=as_c2.get), AS_top2=sorted(as_c2.items(), key=lambda kv: -kv[1])[:2],
        EA1_SEL_argmax=max(G_SEL_C2, key=G_SEL_C2.get), E1v2_TEST_argmax_of_12=max(ei, key=lambda n: gesc[ei[n]].mean()),
        note='only the AS rule was registered (E1 v2); the other two are not registered')
    # symmetry of the baselines (post hoc)
    fr = [ix[f'fresh_run{i}'] for i in range(3)]
    fz = [ix[f'frozen_run{i}'] for i in range(3)]
    ph['baseline_symmetry'] = dict(
        C2_mean_3_fresh_minus_astar=iv(gsc[fr].mean(0)-gsc[STAR]),
        orig_mean_3_frozen_minus_astar=iv(gso[fz].mean(0)-gso[STAR]),
        orig_mean_frozen1_2_minus_astar=iv(gso[fz[1:]].mean(0)-gso[STAR]),
        orig_mean_3_fresh_minus_mean_3_frozen=iv(gso[fr].mean(0)-gso[fz].mean(0)),
        orig_fresh_minus_astar={f'fresh_run{i}': iv(gso[fr[i]]-gso[STAR]) for i in range(3)},
        orig_fresh_minus_frozen_same_index={f'run{i}': iv(gso[fr[i]]-gso[fz[i]]) for i in range(3)},
        orig_A_fresh_minus_astar=iv(gso[F0]-gso[STAR]),
        note='frozen_run0 is the parent of a* (a* = orig_run0 generation 60 started from frozen_run0)')
    ph['orig_side_margin'] = dict(
        e1v2_extra_G_orig_needed_for_UB_D_to_reach_0p005=0.005-e1res['UB_D'], e1v2_extra_G_orig_needed_for_LB_to_exceed_0p005=0.005-e1res['LB'],
        ea1_D_new_extra_needed_UB=0.005-dnew['UB'], ea1_D_new_extra_needed_LB=0.005-dnew['LB'],
        note='shift of the original-setting attacker gain that would change the E1 v2 sub-label; SD of D assumed unchanged (inference)')
    # selection effects
    gsel = sel['reeval_C2']['G']
    ph['selection'] = dict(
        n_validation_points_behind_A_fresh=sum(len(rd(EA/f'chains/run{r}.json')['records']) for r in range(3)),
        SEL_A_fresh_minus_astar=iv(gs_c2[names_sel.index('fresh_run0')]-gs_c2[names_sel.index('orig_adapt0')]),
        SEL_to_TEST_level_shift=dict(a_star=float(gmc[0].mean()-gsel[0]), A_fresh=float(gmc[1].mean()-gsel[1])),
        A_fresh_SEL_chain_equals_reeval=sel['A_fresh']['G_SEL_chain'] == sel['A_fresh']['G_SEL_reeval'])
    # drift (all stored fields)
    ev1 = {n: rd(E1/e1sel['transfer_matrix_AS'][n]['raw']) for n in ('orig', 'C2')}
    drift = {}
    for tag, new in (('orig', raw_do), ('C2', raw_dc)):
        old = ev1[tag]
        uo, un = np.asarray(old['snap']['util_b'], float)[:, :, 1], np.asarray(new['snap']['util_b'], float)[:, :, 1]
        idx = 1+e1sel['attackers'].index('orig_adapt0')
        drift[tag] = dict(a_star_util_bitwise=bool(np.array_equal(uo[idx], un[1])), honest_util_bitwise=bool(np.array_equal(uo[0], un[0])),
                          honest_reg_bitwise=bool(np.array_equal(np.asarray(old['snap']['reg_b'])[0], np.asarray(new['snap']['reg_b'])[0])),
                          honest_pen_bitwise=bool(np.array_equal(np.asarray(old['snap']['pen'])[0], np.asarray(new['snap']['pen'])[0])),
                          raw_files_differ=old.get('sha256') != new.get('sha256'), n_seeds=int(un.shape[1]),
                          drift_json=dict(n_bitwise_identical=dr[tag]['n_bitwise_identical'], max_abs_per_seed=dr[tag]['max_abs_per_seed']))
    ph['drift_all_fields'] = drift
    # chains: trajectories, saturation, training curve
    ch = []
    for r in range(3):
        c = rd(EA/f'chains/run{r}.json')
        vals = [x['G_SEL'] for x in c['records']]
        best = max(range(len(vals)), key=lambda i: (vals[i], -i))
        a = np.array([x['f_mean'] for x in c['curve']], float)
        sl = stats.linregress(np.arange(100), a[-100:])
        blocks = [(float(a[i:i+52].mean()), float(a[i:i+52].std(ddof=1)/math.sqrt(52))) for i in range(208, 416, 52)]
        ch.append(dict(run=r, n_curve=len(c['curve']), finite=bool(all(math.isfinite(v) for v in vals) and np.isfinite(a).all()),
                       gens=[x['gen'] for x in c['records']], G_SEL=vals, best_gen=c['records'][best]['gen'],
                       saturation_gain=max(vals[6:])-max(vals[:6]),
                       selected_theta_equals_record=next(x for x in c['records'] if x['gen'] == sel['chains'][r]['gen'])['theta'] == sel['chains'][r]['theta'],
                       f_mean_block_means_sem_gens_209_416=blocks, f_mean_slope_last_100=float(sl.slope), f_mean_slope_last_100_se=float(sl.stderr)))
    ph['chains'] = ch
    # hashes / integrity
    led = rd(EA/'test/ledger.json')
    man1 = rd(E1/'manifest.json')
    changed = [q for grp in ('input_sha256', 'protected_sha256') for q, h in man1[grp].items() if fsha(ROOT/q) != h]
    ph['integrity'] = dict(
        design_sha256=fsha(ROOT/'docs/direction2/design/e-a1.md'), selection_frozen_sha256=fsha(EA/'selection_frozen.json'),
        selection_matches_result=fsha(EA/'selection_frozen.json') == res['selection_frozen_sha256'],
        ledger_raw_sha_ok={k: fsha(EA/f'test/{k}_raw.json') == v['raw_sha256'] for k, v in led.items()},
        e1v2_manifest_input_or_protected_files_changed=changed,
        e1v2_results_md_in_manifest=any('results/e1-v2.md' in q for grp in ('input_sha256', 'protected_sha256') for q in man1[grp]))
    out['post_hoc'] = ph

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False)+'\n')
    c = out['check']
    print(f"numpy {np.__version__} scipy {scipy.__version__}")
    print(f"primary: mean {p['mean']:.6f} LB {p['LB']:.6f} UB {p['UB']:.6f} -> {cat} (saved {res['category']}:{res['sub']})")
    print(f"registered numbers compared {c['n_compared']}, not bitwise {c['n_not_bitwise']}, max abs diff {c['max_abs_diff']:.3g}, all LB/UB: {c['all_not_bitwise_are_LB_or_UB']}")
    print(f"integrity: {ph['integrity']['ledger_raw_sha_ok']} E1 v2 manifest changed files: {changed}")
    print('wrote', OUT.relative_to(ROOT))
    return 0 if (cat == f"{res['category']}:{res['sub']}" and c['all_not_bitwise_are_LB_or_UB'] and c['max_abs_diff'] < 1e-15 and not changed) else 1


if __name__ == '__main__':
    sys.exit(main())
