# Number audit: paper.tex and README.md against results/*.json

Keys: `GV2` = `results/game.json` -> `game_v2.cells[<cell>].mech[<mech>].games["full|truth|eps=0.002"]`, quantity `.loss` unless noted. `DP` = `results/dp_step.json`. `RES` = `results/results.json`. `CP` = `results/c_pricing.json`. `DS` = `results/d_scaling.json`. "Fix" means the text was changed to the JSON value; "removed" means no JSON source existed and the statement was deleted or made qualitative. Values are rounded as printed in the paper. Pre-existing correct values are marked "yes".

## A. Value of information (C2, C3, H1)

| value | where | JSON key path | JSON value | ok |
|---|---|---|---|---|
| VoI_step 0.012/0.019/0.023/0.018 (r=.002/.004/.008/.016) | abstract, C2, README | `DP.gap_table.v2[r].VoI_step.mean` | 0.0120 / 0.0188 / 0.0232 / 0.0181 | yes |
| CIs of the four VoI values (new) | C2 table, README | `DP.gap_table.v2[r].VoI_step.{lo,hi}` | [0.0104,0.0135] [0.0170,0.0206] [0.0213,0.0251] [0.0154,0.0207] | added |
| 32 seeds, K-extrapolated from (30,60), spread 0.6 | C2 | `DP.gap_table.v2[r].Kstep`, `.seeds`, `meta.spread` | [30,60]; 32; 0.6 | yes |
| "about twice the non-re-targetable DP" | C2 | `DP.gap_table.v2[r].VoI_semi_ext.mean` | 0.0059/0.0096/0.0120/0.0100 (ratio 1.8-2.0) | yes (values added) |
| legacy estimator bias -0.0014 to -0.0025 | Sec. 4, H1 | `DP.gap_table.legacy[r].VoI_step.mean` | -0.0014/-0.0025/-0.0024/-0.0016 | yes |
| re-planning worth +0.012 (8/8 seeds) | Sec. 4 | none (only in decisions.md); `DP.gap_table.v2[r].step_minus_semi_ext.mean` | 0.0061/0.0092/0.0112/0.0080 (32 seeds) | fix: now 0.006-0.011, 32 seeds |
| region map: 0.006 at f=.25, 0.045 at f=.9 (r=.004, K=(20,30), 16 seeds, spread .6) | C2, README | `DP.region.cells[f,spread=0.6,service=5].VoI_step.mean` | 0.00584; 0.04548 | yes |
| region map f=0 / .5 / .75 values, spread trend (new) | C2 | same | -0.0027; 0.0195; 0.0351; f=.5: 0.0214 (spread 0) to 0.0136 (1.5) | added |
| service=1 refutation 0.024 [0.020,0.028] | H1, prereg, README | `DP.region.cells[f=.5,spread=.6,service=1].VoI_step` | 0.0239 [0.0203,0.0276]; `VoI_step_final` lower 0.01997, `VoI_step_cal` lower 0.0200 | yes, caveat added (borderline, region-map settings) |
| H1 cell 0.019 [0.017,0.021] | H1, prereg | `DP.gap_table.v2["0.004"].VoI_step` | 0.0188 [0.0170,0.0206] | yes |
| median gap of `none`, N=3: 0.021 ... 0.082 | C3, README | `DP.gap_table.v2[r].gap.none.median` | 0.0213/0.0382/0.0568/0.0819 | yes (middle values added) |
| mean gap of `none` (new) | C3 | `...gap.none.mean` | 0.0244/0.0415/0.0584/0.0884 | added |
| N=4 median gap 0.039/0.058 (r=.004/.008) | C3 | `DP.n4[r].gap.none.median` | 0.0392 / 0.0579 | yes; mean 0.0435/0.0678, 16 seeds, K=(15,20,30) added |
| N=4 VoI_step (new) | C3 | `DP.n4[r].VoI_step` | 0.0205 [0.0181,0.0229]; 0.0197 [0.0168,0.0225] | added |
| GPU vs CPU "~1e-12" | Sec. 3, README | `game.game_sweep.preflight_max_err`; `DS.check_multi_A1_vs_sim_torch`; `DP.checks.age.gpu_vs_cpu_AgeDP.max_abs_diff`; `DP.checks.ongrid_exact[].diff_step_vs_semi` | simulator <=1.2e-15; step DP <=3.7e-10; age DP 2.7e-5 | fix |

## B. Strategic game (C4-C6)

Cells: `N3_r0.004_f0.5` (57 actions, 4000 steps, 32 seeds) and `N8_r0.002_f0.5` (24 actions, 2000 steps, 32 seeds). **`audit_index` and `audit_disp_index` are in the same cell and the same game key (`full|truth|eps=0.002`), same seeds 1000-1031.**

| value | where | JSON key path | JSON value | ok |
|---|---|---|---|---|
| fused loss N=3: 0.247 [0.223,0.272] | abstract, table, README | `GV2[N3_r0.004_f0.5].fused_index.loss` | 0.2474 [0.2232,0.2717] | yes |
| fused loss N=8: 0.40 [0.36,0.44] | same | `GV2[N8_r0.002_f0.5].fused_index.loss` | 0.4005 [0.3614,0.4397] | yes |
| fused 9 N=3 cells 0.11-0.27 | table, C4 | `GV2[N3_*].fused_index.loss.mean` | 0.1123 ... 0.2700 | yes (0.112-0.270) |
| 57 / 24 actions | table | `cells[..].n_actions` | 57; 24 (others 24) | yes |
| fused convergence (was "---") | table | `.frac_converged` | 0.875 (N3), 0.875 (N8), nine cells 0.875-1.00 | filled |
| audit_index N=3: 0.024 [0.010,0.038], conv 0.62 | abstract, C5 | `GV2[N3...].audit_index.loss`, `.frac_converged` | 0.0237 [0.0100,0.0375]; 0.625 | yes |
| audit_index N=8: 0.042 [0.025,0.059], conv 0.31 | same | `GV2[N8...].audit_index` | 0.0417 [0.0245,0.0589]; 0.3125 | yes |
| audit_disp N=3: 0.0027 [0.0001,0.0052] | abstract, C6 | `GV2[N3...].audit_disp_index.loss` | 0.00267 [0.00010,0.00523]; conv 0.25 | yes; conv filled |
| audit_disp N=8: 0.0058 [0.002,0.010] | same | `GV2[N8...].audit_disp_index.loss` | 0.00579 [0.00179,0.00979]; conv 0.0 | yes |
| ratio dispatch/arrival "1/7-1/9" | C6, README | `audit_index.loss.mean / audit_disp_index.loss.mean`, same cell | 8.9 (N=3), 7.2 (N=8); all ten cells 4.6-48 | recomputed from same cell; demoted to side remark |
| "one order of magnitude" | abstract, intro, conclusion | same | 7-9 in two cells only | fix |
| H2: eq - learned CI excludes 0 in all cells | C4, H2 | `GV2[*].fused_index.eq_minus_learned.hi` | all < 0 (10/10) | yes |
| loss / VoI "8-25 times" | C4 | `GV2[N3_r*_f0.5].fused_index.loss.mean / DP VoI_step` | 17.4 / 13.2 / 8.3 (f=.5); 33.6 (f=.25), 7.7 (f=.75) at r=.004 | fix: 25 not supported; now side remark with these values |
| dead fraction 0.029 -> 0.067 (audit_index, N=3) | C5 | `.dead_truthful.mean`, `.dead_eq.mean` | 0.0287 -> 0.0671 | yes |
| CI includes 0 "in a minority of cells", e.g. N3 r.008 f.25 | C5 | `GV2[N3_r0.008_f0.25].audit_index.loss` | 0.0110 [-0.0033,0.0253]; the only such cell of 10 | fix: "one of ten" |
| audit_disp loss CI >0 "at both N" | C6 | `GV2[*].audit_disp_index.loss` | >0 in 5 of 10 cells (N3 r.004 f.5, r.002 f.5, r.004 f.25, r.004 f.75; N8); contains 0 in 5; N3 r.004 f.5 at eps=.005: 0.0018 [-0.0010,0.0045] | caveat added |
| calibrated variant N=3 | C6 | `GV2[N3...]["audit_disp_index\|cal"].loss` | 0.0041 [0.0016,0.0065] | added |
| max_gain: in-sample 0.075-0.10, out-of-sample 0.004-0.007 | C6, README | `game_v2.cells.N8_r0.002_f0.5.oos_check.*.{naive_in_sample_gain_mean,max_player_mean_gain}` | audit_disp\|cal 0.0753 / 0.0072; audit_index 0.1002 / 0.0038; penalty variants 0.078-0.097 / 0.0104-0.0178. Only N=8 stored | fix |
| max_gain in-sample mean (H4 test) | C6 | `.max_gain_mean` | N3 0.0197, N8 0.0769 (> 2 eps=0.004) | yes |
| false trigger "about 40%" | C6 | `cells[..].truthful_false_flags[*].frac_robots_ever_flagged` | N3 0.61-0.64 raw, 0.35-0.40 cal; N8 0.30-0.34 raw, 0.07-0.11 cal | fix |
| 10 of 27 cells | abstract, prereg | `game_v2.completed_cells` | 10 (9 at N=3, N8_r0.002_f0.5) | yes |
| strategy family list | Sec. 6 | `game_v2.meta.families`; `arbitration/gpu/game_gpu.py` FAMS, `scripts/run_game_gpu.py` acts_full/acts_reduced | const,timing,dist10,dist30,taulow,habit,adapt; 57 = 1+16+5x8; 24 = 1+8+5x3 | filled (definitions from code, not JSON) |
| deviation (2) details | prereg | `game_v2.notes` | see text | filled |
| eps in {0.001,0.002,0.005} run only in two cells | Sec. 6 | `notes`, `games` keys | yes | added |

## C. Exploratory (C7, C8)

| value | where | JSON key path | JSON value | ok |
|---|---|---|---|---|
| priced - no-thrash +0.015/+0.045/+0.082/+0.092 | C7, README | `CP.c1.cells["v2\|r"].vs_nothrash.priced_nopreempt.mean`; `CP.c3.priced_nopreempt_minus_nothrash_v2` | 0.0148/0.0452/0.0819/0.0916 | yes |
| seeds 0-15 / 1000-1031 | C7 | `CP.c1.seeds_tune`, `seeds_hold` | yes | yes; "tuned in one cell only" removed (lambda* tuned per load; 6500 steps) |
| "0.03-0.07 below honest_index and learned_index" | C7, README | `CP.c1.cells[v2\|r].arms.{priced_nopreempt,honest_index_ref,learned_index_ref}.health.mean` | vs honest_index 0.029/0.068/0.077/0.052; vs learned_index 0.058/0.084/0.087/0.057 | fix: 0.03-0.08 and 0.06-0.09 |
| N=3 lambda* ~ 0 | C7 | `CP.c2.lambda_star_n3` | nopreempt 0.0; preempt 0.00219, priced-honest -0.0021 [-0.0067,0.0025] | refined |
| index tau/(d+5)^1.14, top-1 0.77 | C7 | `CP.c2.dp_logit_logindex_all.{ratio,ratio_ci95,top1_acc}`, `n_decisions` | 1.138 [1.073,1.211]; 0.7705; 16552 decisions; argmax-tau baseline 0.5061 | yes ("agreement" renamed accuracy) |
| thrash-suppression statement | C7 | `CP.c1.cells[legacy\|0.004 / v2\|0.004].decomposition.share_of_priced_preempt_gain_explained_by_nothrash` | 1.264 / 1.352 | replaced the unsupported "fixed lambda" sentence |
| scaling loss 0.165/0.405/0.413 | C4, C8 | `DS.games["A\|3","A\|8","A\|16"].mech.fused_index.loss.mean` | 0.1649/0.4053/0.4125; r 0.0098/0.0032/0.0022; conv 1.0/0.875/0.5625; 4000 steps | yes, cell details added |
| proxy "differs from DP by -5% to +41%" | C8, claims.md | `DS.proxy_vs_dp[*].{fused_recovers_frac_of_dp_VoI,index_true_over_dp_VoI}` | fused -0.05..0.34; index_true 0.10..0.41 (fraction OF the DP value) | fix: wording was wrong (it recovers -5%..41%, i.e. underestimates) |
| round-robin beats heuristics for N>=8 | C8 | `DS.health["A\|8","A\|16","B\|8","B\|16"].pol.*.health.mean` | none 0.763/0.768/0.759/0.739 vs best heuristic 0.746/0.722/0.743/0.692 | yes, values added |
| two-actuator uninformative | C8 | `DS.multi_actuator.loads["same_r (r_single)"].A["1"].fused_index.minus_learned_index` | 0.0012, not significant | yes |

## D. C1 (results.json)

Cell: `RES.v2_sweep.configs.v2`, N=8 (`RES.config.n`), 1500 steps (`RES.config.steps`), 32 seeds, spread 0.6, shock fraction 0.5 (ModelConfig default), loads `RES.v2_sweep.rates`; no burn-in.

| value | JSON key path | JSON value | ok |
|---|---|---|---|
| greedy_true loses at every load | `configs.v2.paired_vs_none[r]["greedy_true\|nopre"/"\|pre"].mean_health` | all negative, all CIs exclude 0 (-0.019 ... -0.15) | yes |
| distance-aware index wins only at r>=0.02 | `...["index_true\|nopre"]` | loses at .0005/.004/.007, ns at .001/.002/.012, +0.0154/+0.0271 at .02/.04. With preemption `index_true\|pre` also +0.0054/+0.0095/+0.0139 at .001/.002/.012. `learned_index\|nopre`, `honest_index\|nopre` significant only at .04 | claim restricted to no-preemption, exception stated |
| `none` health 0.956 -> 0.173 | `configs.v2.cells.none[r].mean_health.mean` | yes | added |

## E. Removed or rewritten

- "re-planning +0.012 (8/8 seeds)": no JSON source; replaced by `step_minus_semi_ext`.
- "8-25 times": upper bound unsupported; replaced by computed ratios as a side remark.
- "(tuned in one cell only)" in C7: wrong for C7 (per-load tuning); removed.
- "noise i.i.d. unless stated; one AR variant exists": wrong; v2 config uses AR(1) phi=0.9 (`arbitration/model.py` V2). Fixed.
- "actuator may re-target each step unless stated": wrong; default `preempt=False` in all game/scaling runs. Fixed.
- "burn-in 500 for all": not for C1. Fixed.
- "thrash ... correct for legacy at fixed lambda": replaced by `decomposition` values.

## F. Still unverifiable from JSON

Strategy-family definitions (code), Sec. 2 literature statements (docs/literature.md; only abstract-level checks), the claim that no published index exists for decaying arms with travel (negative search), pre-registration dates.
