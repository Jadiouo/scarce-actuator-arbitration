# Number audit: direction1-paper/paper.tex

Every number in the paper body that comes from an experiment is substituted into `paper.tex` by a script directly from `results/*.json` (formatting: mean ± half-width, 4 decimals unless noted; pct = value×100; `R/T` for `adaaudit_2x2` R = JSON value ×1e-5). The column "consistent?" is the script check that the formatted JSON value equals the string printed in the paper. Rows marked (param) are the cell parameters printed in the table. The table rows are listed once per displayed value; the same displayed number may appear in more than one place in the text and shares one row here. a duplicate check against `docs/direction1-number-audit.md` is in the final note below.

| # | Location in paper | Value | Source (JSON key path) | Consistent? |
|---|---|---|---|---|
| 1 | Sec: Model | 0.1608 | `results/stage2.json` : `checks.calibration_honest_flag_rate.0.flag_per_audit` = 0.16077518090326873 | yes |
| 2 | Sec: Model | 0.1587 | `results/stage2.json` : `checks.calibration_honest_flag_rate.0.Phi_neg_kappa` = 0.15865525393145707 | yes |
| 3 | Sec: Model | 1.0 | `results/stage2.json` : `checks.calibration_honest_flag_rate.0.kappa` = 1.0 | yes |
| 4 | Sec: Model | 0.5 | `results/stage2.json` : `checks.calibration_honest_flag_rate.0.r` = 0.5 | yes |
| 5 | Sec: 4 (stage 1 zero tolerance) | 5 | `results/adaaudit_2x2.json` : `meta.tau_delay` = 5 | yes |
| 6 | Sec: 4 (stage 1 zero tolerance) | 1.00 | `results/adaaudit_2x2.json` : `twobytwo.fixed|100000|iid_tauD|honest.honest_excl_frac.0` = 1.0 | yes |
| 7 | Sec: 4 (stage 1 zero tolerance) | 0.812 | `results/adaaudit_2x2.json` : `twobytwo.fixed|100000|iid_tauD|honest.R.0` = 81227.39165900918 | yes |
| 8 | Sec: 4 (stage 1 zero tolerance) | 0.812 | `results/adaaudit_2x2.json` : `twobytwo.fixed|100000|markov_tauD|honest.R.0` = 81235.0461188572 | yes |
| 9 | Sec: 4 (stage 1 zero tolerance) | 0.0 | `results/adaaudit_2x2.json` : `twobytwo.fixed|100000|iid_tau0|honest.R.0` = 0.0 | yes |
| 10 | Sec: 4 (stage 1 zero tolerance) | 2.63 | `results/adaaudit_2x2.json` : `twobytwo.ada|100000|iid_tau0|late_liar_b1.0.R.0` = 2.6300361870373936 | yes |
| 11 | Sec: 4 (stage 1b AdaAudit more tolerant) | 0.00 | `results/adaaudit_tol.json` : `cells.fixed|100000|0.9|1|0.1.honest.honest_surv.0` = 0.0 | yes |
| 12 | Sec: 4 (stage 1b AdaAudit more tolerant) | 0.00 | `results/adaaudit_tol.json` : `cells.ada|100000|0.9|1|0.1.honest.honest_surv.0` = 0.0 | yes |
| 13 | Sec: 4 (stage 1b AdaAudit more tolerant) | 16.7 | `results/adaaudit_tol.json` : `cells.ada|100000|0.9|1|0.1.honest.B.0` = 16.71875 | yes |
| 14 | Sec: 4 (stage 1b AdaAudit more tolerant) | 9.4 | `results/adaaudit_tol.json` : `cells.fixed|100000|0.9|1|0.1.honest.B.0` = 9.4375 | yes |
| 15 | Sec: 4 (stage 1b AdaAudit more tolerant) | true | `results/adaaudit_tol.json` : `feasibility.ada|100000|b0.1.feasible.6.4` = true | yes |
| 16 | Sec: 4 (stage 1b AdaAudit more tolerant) | 3645.9 | `results/stage2.json` : `bt_adaaudit_tau0.ideal_BT.2` = 3645.909494694089 | yes |
| 17 | Sec: 4 (stage 1b AdaAudit more tolerant) | 5626.1 | `results/stage2.json` : `bt_adaaudit_tau0.by_rho.0.9.BT.2.0` = 5626.125 | yes |
| 18 | Sec: 4 (stage 2 M3c, edge inflation) | -0.0118 ± 0.0028 | `results/stage2.json` : `main.M3c.100000.0.9.Gmax` = [-0.01177532731935538, 0.002829372795761977] | yes |
| 19 | Sec: 4 (stage 2 M3c, edge inflation) | 0.0067 ± 0.0015 | `results/stage2.json` : `main.M3c.100000.0.9.R_b_honest` = [0.006708961159344852, 0.001537032855914119] | yes |
| 20 | Sec: 4 (stage 2 M3c, edge inflation) | 0.2795 ± 0.0004 | `results/stage2.json` : `main.M2.100000.0.5.Gmax` = [0.2794880284199983, 0.0004205750367242674] | yes |
| 21 | Sec: 4 (stage 2 M3c, edge inflation) | 0.2805 ± 0.0022 | `results/stage2.json` : `main.M2.100000.0.99.Gmax` = [0.28048819561516036, 0.0021518647500347494] | yes |
| 22 | Sec: 4 (stage 2 M3c, edge inflation) | 0.0211 ± 0.0033 | `results/quota.json` : `main.14.Gmax` = [0.021104115773860545, 0.003304642560485526] | yes |
| 23 | Sec: 4 (stage 2 M3c, edge inflation) | 0.0870 ± 0.0053 | `results/quota.json` : `main.12.Gmax` = [0.086962112771764, 0.005295154442673111] | yes |
| 24 | Sec: 4 (stage 2 M3c, edge inflation) | 0.0021 ± 0.0025 | `results/stage2.json` : `main.M3c.100000.0.8.Gmax` = [0.0021185843561695797, 0.0024900939082519776] | yes |
| 25 | Sec: 4 (stage 2 M3c, edge inflation) | false | `results/stage2.json` : `main.M3c.10000.0.8.eval_pass` = false | yes |
| 26 | Sec: 4 (stage 2 M3c, edge inflation) | false | `results/stage2.json` : `main.M3c.10000.0.9.eval_pass` = false | yes |
| 27 | Sec: 4 (M4 loophole) | 0.1811 ± 0.0721 | `results/stage2.json` : `main.M4.100000.1.0.Gmax` = [0.18105352017215026, 0.07206902628243358] | yes |
| 28 | Sec: 4 (M4 loophole) | 0.0356 ± 0.0008 | `results/stage2b.json` : `consist_M4.9.gain_edgeO` = [0.03556691436913642, 0.0007521353638302052] | yes |
| 29 | Sec: 4 (M4 loophole) | 0.0 | `results/stage2b.json` : `consist_M4.9.rw` = 0.0 | yes |
| 30 | Sec: 4 (M4 loophole) | -0.1450 ± 0.0039 | `results/stage2b.json` : `consist_M4.11.gain_edgeO` = [-0.14497351692990867, 0.003948775355400535] | yes |
| 31 | Sec: 4 (M4 loophole) | 1.0 | `results/stage2b.json` : `consist_M4.11.rw` = 1.0 | yes |
| 32 | Sec: 4 (F1-F4) | 1.26 | `results/stage2b.json` : `scaling_best_family.a0.005.fit_all.s` = 1.260000000000002 | yes |
| 33 | Sec: 4 (F1-F4) | 1.13 | `results/stage2b.json` : `scaling_best_family.a0.005.fit_all.s_ci.0` = 1.1300000000000017 | yes |
| 34 | Sec: 4 (F1-F4) | 1.39 | `results/stage2b.json` : `scaling_best_family.a0.005.fit_all.s_ci.1` = 1.3900000000000023 | yes |
| 35 | Sec: 4 (F1-F4) | 2.18 | `results/stage2b.json` : `scaling_best_family.a0.05.fit_all.s` = 2.180000000000003 | yes |
| 36 | Sec: 4 (F1-F4) | 1.77 | `results/stage2b.json` : `scaling_best_family.a0.05.fit_all.s_ci.0` = 1.7700000000000022 | yes |
| 37 | Sec: 4 (F1-F4) | 2.63 | `results/stage2b.json` : `scaling_best_family.a0.05.fit_all.s_ci.1` = 2.6300000000000034 | yes |
| 38 | Sec: 4 (F1-F4) | 0.0227 ± 0.0060 | `results/stage2b.json` : `tau_cmp_own_frontier.0.9.a0.05.tau1.G` = [0.02268993697697485, 0.0060403626694569305] | yes |
| 39 | Sec: 4 (F1-F4) | 0.0221 ± 0.0069 | `results/stage2b.json` : `tau_cmp_own_frontier.0.9.a0.05.tau5.G` = [0.02211527187927447, 0.006882338707465279] | yes |
| 40 | Sec: 4 (F1-F4) | 0.0287 | `results/stage2b.json` : `F3_uninformed.43.edge_uninformed.gain.0` = 0.028723287686942968 | yes |
| 41 | Sec: 4 (F1-F4) | 0.0058 | `results/stage2b.json` : `F3_uninformed.43.edge_uninformed.gain.1` = 0.0057878206503488225 | yes |
| 42 | Sec: 4 (F1-F4) | 0.1235 ± 0.0077 | `results/stage2b.json` : `F3_uninformed.43.edge_oracle.gain` = [0.12349621787238228, 0.007730479086694218] | yes |
| 43 | Sec: 4 (F1-F4) | 0.0011 ± 0.0015 | `results/stage2b.json` : `F3_uninformed.56.edge_uninformed.gain` = [0.0011362486228142891, 0.0014981321301022745] | yes |
| 44 | Sec: 4 (F1-F4) | 1.49 | `results/stage2b.json` : `F3_uninformed.56.edge_uninformed.z` = 1.486549320962745 | yes |
| 45 | Sec: 4 (F1-F4) | -0.0108 ± 0.0064 | `results/stage2b.json` : `F3_uninformed.70.edge_uninformed.gain` = [-0.010817061823506715, 0.0064465743912310155] | yes |
| 46 | Sec: 4 (F1-F4) | -0.0443 ± 0.0094 | `results/stage2b.json` : `Gstar.M4.a0.01.0.5.G` = [-0.04428507698001881, 0.009436728953495954] | yes |
| 47 | Sec: 4 (F1-F4) | 0.0000 ± 0.0000 | `results/stage2b.json` : `Gstar.M4.a0.01.0.5.alpha` = [0.0, 0.0] | yes |
| 48 | Sec: 5 (findings) | 0.2793 ± 0.0004 | `results/quota.json` : `main.4.Gmax` = [0.27930049154850023, 0.0003786066909699181] | yes |
| 49 | Sec: 5 (findings) | 0.2795 ± 0.0004 | `results/stage2.json` : `main.M2.100000.0.5.Gmax` = [0.2794880284199983, 0.0004205750367242674] | yes |
| 50 | Sec: 5 (findings) | 0.1875 ± 0.0007 | `results/quota.json` : `main.4.honest_util_agent1` = [0.18745347021268274, 0.0006672699115254098] | yes |
| 51 | Sec: 5 (findings) | 0.0070 ± 0.0016 | `results/quota.json` : `main.14.R_hon_b` = [0.0069707399001550895, 0.0016374238390795821] | yes |
| 52 | Sec: 5 (findings) | 0.0303 ± 0.0060 | `results/quota.json` : `main.14.false_punish_frac` = [0.030255625, 0.006032201336841113] | yes |
| 53 | Sec: 5 (findings) | 0.0213 ± 0.0039 | `results/quota.json` : `main.12.Gmax_uninformed` = [0.021327654113202883, 0.0039011376120500494] | yes |
| 54 | Sec: 5 (findings) | 0.0211 ± 0.0033 | `results/quota.json` : `main.14.Gmax` = [0.021104115773860545, 0.003304642560485526] | yes |
| 55 | Sec: 5 (findings) | 0.0870 ± 0.0053 | `results/quota.json` : `main.12.Gmax` = [0.086962112771764, 0.005295154442673111] | yes |
| 56 | Sec: 5 (findings) | 0.0460 ± 0.0015 | `results/quota.json` : `main.15.Gmax` = [0.04601827255506004, 0.0014688955181857205] | yes |
| 57 | Sec: 5 (findings) | 0.0001 ± 0.0001 | `results/quota.json` : `main.15.false_punish_frac` = [9.375000000000002e-05, 6.711605092758883e-05] | yes |
| 58 | Sec: 5 (findings) | 0.3073 ± 0.0212 | `results/stage2b.json` : `rho_misspec.18.alpha` = [0.3072546875, 0.021157987755285438] | yes |
| 59 | Sec: 5 (findings) | 0.9531 ± 0.0032 | `results/stage2b.json` : `rho_misspec.26.alpha` = [0.9530708333333333, 0.003198593922015889] | yes |
| 60 | Sec: 5 (findings) | 0.0044 ± 0.0059 | `results/stage2b.json` : `rho_misspec.32.G` = [0.004363548968191942, 0.005875346405022896] | yes |
| 61 | Sec: 5 (findings) | 0.0556 ± 0.0019 | `results/stage2b.json` : `rho_misspec.33.G` = [0.05562505849202838, 0.0019310726015046944] | yes |
| 62 | Sec: 5 (findings) | 0.0048 ± 0.0014 | `results/quota.json` : `main.27.Gmax_uninformed` = [0.004849327961781592, 0.001414774967156068] | yes |
| 63 | Sec: 5 (findings) | 0.0897 ± 0.0003 | `results/quota.json` : `main.24.Gmax` = [0.08967161764263046, 0.00032121729712398215] | yes |
| 64 | Sec: 5 (findings) | 4.3% | `results/quota.json` : `main.24.rewrite_frac_honest.0` = 0.04254322916666667 | yes |
| 65 | Sec: 5 (findings) | 27.0% | `results/quota.json` : `main.27.rewrite_frac_honest.0` = 0.27003916666666666 | yes |
| 66 | Sec: 5 (findings) | 0.0114 ± 0.0001 | `results/quota.json` : `main.24.R_hon_b` = [0.011358985509924945, 0.0001278198640795332] | yes |
| 67 | Sec: 5 (findings) | 0.0585 ± 0.0011 | `results/quota.json` : `main.27.R_hon_b` = [0.0585007504952225, 0.0010935331212447714] | yes |
| 68 | Sec: 6 (raids) | 0.0000 ± 0.0000 | `results/quota.json` : `main.88.Gmax` = [0.0, 0.0] | yes |
| 69 | Sec: 6 (raids) | 0.0162 ± 0.0001 | `results/quota.json` : `main.88.R_hon_b` = [0.016222466622181183, 0.00013373540540860911] | yes |
| 70 | Sec: 6 (raids) | 0.0162 ± 0.0002 | `results/quota.json` : `main.91.R_hon_b` = [0.016242242420091327, 0.00015193809282280473] | yes |
| 71 | Sec: 6 (raids) | 0.0070 ± 0.0016 | `results/quota.json` : `main.14.R_hon_b` = [0.0069707399001550895, 0.0016374238390795821] | yes |
| 72 | Sec: 6 (raids) | 0.0082 ± 0.0002 | `results/stage2b.json` : `eval_rows.M4.0.R_honest` = [0.008203655647187037, 0.00016272771173547943] | yes |
| 73 | Sec: 6 (raids) | 0.0200 ± 0.0002 | `results/quota.json` : `main.88.raid_waste_frac` = [0.019975625, 0.00015828944490058554] | yes |
| 74 | Sec: Limitations | -0.0297 ± 0.0078 | `results/stage2b.json` : `F3_uninformed.56.adaptive.gain` = [-0.02965509025450029, 0.00783556914736919] | yes |
| 75 | Sec: Limitations | 0.0021 ± 0.0025 | `results/stage2.json` : `main.M3c.100000.0.8.Gmax` = [0.0021185843561695797, 0.0024900939082519776] | yes |
| 76 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.4.R_hon_b` = [0.0, 0.0] | yes |
| 77 | Table 1 (main map) | 0.2793 ± 0.0004 | `results/quota.json` : `main.4.Gmax_uninformed` = [0.27930049154850023, 0.0003786066909699181] | yes |
| 78 | Table 1 (main map) | 0.2793 ± 0.0004 | `results/quota.json` : `main.4.Gmax` = [0.27930049154850023, 0.0003786066909699181] | yes |
| 79 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.5.R_hon_b` = [0.0, 0.0] | yes |
| 80 | Table 1 (main map) | 0.2793 ± 0.0006 | `results/quota.json` : `main.5.Gmax_uninformed` = [0.2793018980631612, 0.0005998432879280821] | yes |
| 81 | Table 1 (main map) | 0.2793 ± 0.0006 | `results/quota.json` : `main.5.Gmax` = [0.2793018980631612, 0.0005998432879280821] | yes |
| 82 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.6.R_hon_b` = [0.0, 0.0] | yes |
| 83 | Table 1 (main map) | 0.2792 ± 0.0009 | `results/quota.json` : `main.6.Gmax_uninformed` = [0.27922695968122613, 0.0008732689130059621] | yes |
| 84 | Table 1 (main map) | 0.2792 ± 0.0009 | `results/quota.json` : `main.6.Gmax` = [0.27922695968122613, 0.0008732689130059621] | yes |
| 85 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.7.R_hon_b` = [0.0, 0.0] | yes |
| 86 | Table 1 (main map) | 0.2806 ± 0.0028 | `results/quota.json` : `main.7.Gmax_uninformed` = [0.28055676737104984, 0.0027847341758342925] | yes |
| 87 | Table 1 (main map) | 0.2806 ± 0.0028 | `results/quota.json` : `main.7.Gmax` = [0.28055676737104984, 0.0027847341758342925] | yes |
| 88 | Table 1 (main map) | 0.2236 ± 0.0002 | `results/quota.json` : `main.8.R_hon_b` = [0.2236292718177209, 0.00024121174865107035] | yes |
| 89 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.8.Gmax_uninformed` = [0.0, 0.0] | yes |
| 90 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.8.Gmax` = [0.0, 0.0] | yes |
| 91 | Table 1 (main map) | 0.2237 ± 0.0003 | `results/quota.json` : `main.9.R_hon_b` = [0.2237241769805047, 0.00029827952847249963] | yes |
| 92 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.9.Gmax_uninformed` = [0.0, 0.0] | yes |
| 93 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.9.Gmax` = [0.0, 0.0] | yes |
| 94 | Table 1 (main map) | 0.2239 ± 0.0004 | `results/quota.json` : `main.10.R_hon_b` = [0.223899874487951, 0.00039507173591193495] | yes |
| 95 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.10.Gmax_uninformed` = [0.0, 0.0] | yes |
| 96 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.10.Gmax` = [0.0, 0.0] | yes |
| 97 | Table 1 (main map) | 0.2240 ± 0.0014 | `results/quota.json` : `main.11.R_hon_b` = [0.22400934627083816, 0.0013689274233931333] | yes |
| 98 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.11.Gmax_uninformed` = [0.0, 0.0] | yes |
| 99 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.11.Gmax` = [0.0, 0.0] | yes |
| 100 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/stage2.json` : `main.M2.100000.0.5.R_b_honest` = [0.0, 0.0] | yes |
| 101 | Table 1 (main map) | 0.2795 ± 0.0004 | `results/stage2.json` : `main.M2.100000.0.5.Gmax` = [0.2794880284199983, 0.0004205750367242674] | yes |
| 102 | Table 1 (main map) | 0.2795 ± 0.0004 | `results/stage2.json` : `main.M2.100000.0.5.Gmax` = [0.2794880284199983, 0.0004205750367242674] | yes |
| 103 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M2.100000.0.5.param.p` = 0.1 | yes |
| 104 | Table 1 (main map) (param) | 20 | `results/stage2.json` : `main.M2.100000.0.5.param.L` = 20 | yes |
| 105 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/stage2.json` : `main.M2.100000.0.8.R_b_honest` = [0.0, 0.0] | yes |
| 106 | Table 1 (main map) | 0.2794 ± 0.0005 | `results/stage2.json` : `main.M2.100000.0.8.Gmax` = [0.27941372069528836, 0.0005471637218847936] | yes |
| 107 | Table 1 (main map) | 0.2794 ± 0.0005 | `results/stage2.json` : `main.M2.100000.0.8.Gmax` = [0.27941372069528836, 0.0005471637218847936] | yes |
| 108 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M2.100000.0.8.param.p` = 0.1 | yes |
| 109 | Table 1 (main map) (param) | 20 | `results/stage2.json` : `main.M2.100000.0.8.param.L` = 20 | yes |
| 110 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/stage2.json` : `main.M2.100000.0.9.R_b_honest` = [0.0, 0.0] | yes |
| 111 | Table 1 (main map) | 0.2792 ± 0.0009 | `results/stage2.json` : `main.M2.100000.0.9.Gmax` = [0.2792442333713937, 0.0008969882724184369] | yes |
| 112 | Table 1 (main map) | 0.2792 ± 0.0009 | `results/stage2.json` : `main.M2.100000.0.9.Gmax` = [0.2792442333713937, 0.0008969882724184369] | yes |
| 113 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M2.100000.0.9.param.p` = 0.1 | yes |
| 114 | Table 1 (main map) (param) | 20 | `results/stage2.json` : `main.M2.100000.0.9.param.L` = 20 | yes |
| 115 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/stage2.json` : `main.M2.100000.0.99.R_b_honest` = [0.0, 0.0] | yes |
| 116 | Table 1 (main map) | 0.2805 ± 0.0022 | `results/stage2.json` : `main.M2.100000.0.99.Gmax` = [0.28048819561516036, 0.0021518647500347494] | yes |
| 117 | Table 1 (main map) | 0.2805 ± 0.0022 | `results/stage2.json` : `main.M2.100000.0.99.Gmax` = [0.28048819561516036, 0.0021518647500347494] | yes |
| 118 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M2.100000.0.99.param.p` = 0.1 | yes |
| 119 | Table 1 (main map) (param) | 20 | `results/stage2.json` : `main.M2.100000.0.99.param.L` = 20 | yes |
| 120 | Table 1 (main map) (param) | 6 | `results/stage2.json` : `main.M3c.100000.0.5.param.tol` = 6 | yes |
| 121 | Table 1 (main map) (param) | 2500 | `results/stage2.json` : `main.M3c.100000.0.5.param.L` = 2500 | yes |
| 122 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M3c.100000.0.5.param.p` = 0.1 | yes |
| 123 | Table 1 (main map) | 0.0071 ± 0.0018 | `results/quota.json` : `main.12.R_hon_b` = [0.007078929144348073, 0.0017511954756897544] | yes |
| 124 | Table 1 (main map) | 0.0213 ± 0.0039 | `results/quota.json` : `main.12.Gmax_uninformed` = [0.021327654113202883, 0.0039011376120500494] | yes |
| 125 | Table 1 (main map) | 0.0870 ± 0.0053 | `results/quota.json` : `main.12.Gmax` = [0.086962112771764, 0.005295154442673111] | yes |
| 126 | Table 1 (main map) | 0.0285 ± 0.0061 | `results/quota.json` : `main.12.false_punish_frac` = [0.02845197916666667, 0.006071740203791877] | yes |
| 127 | Table 1 (main map) (param) | 6 | `results/stage2.json` : `main.M3c.100000.0.8.param.tol` = 6 | yes |
| 128 | Table 1 (main map) (param) | 2500 | `results/stage2.json` : `main.M3c.100000.0.8.param.L` = 2500 | yes |
| 129 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M3c.100000.0.8.param.p` = 0.1 | yes |
| 130 | Table 1 (main map) | 0.0081 ± 0.0012 | `results/quota.json` : `main.13.R_hon_b` = [0.008076755633298922, 0.0011787897816710386] | yes |
| 131 | Table 1 (main map) | 0.0037 ± 0.0019 | `results/quota.json` : `main.13.Gmax_uninformed` = [0.0037020866106143094, 0.0018700871104188519] | yes |
| 132 | Table 1 (main map) | 0.0335 ± 0.0044 | `results/quota.json` : `main.13.Gmax` = [0.03353653062447393, 0.0043961176575323295] | yes |
| 133 | Table 1 (main map) | 0.0339 ± 0.0042 | `results/quota.json` : `main.13.false_punish_frac` = [0.0338728125, 0.004243578980961265] | yes |
| 134 | Table 1 (main map) (param) | 6 | `results/stage2.json` : `main.M3c.100000.0.9.param.tol` = 6 | yes |
| 135 | Table 1 (main map) (param) | 2500 | `results/stage2.json` : `main.M3c.100000.0.9.param.L` = 2500 | yes |
| 136 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M3c.100000.0.9.param.p` = 0.1 | yes |
| 137 | Table 1 (main map) | 0.0070 ± 0.0016 | `results/quota.json` : `main.14.R_hon_b` = [0.0069707399001550895, 0.0016374238390795821] | yes |
| 138 | Table 1 (main map) | 0.0020 ± 0.0011 | `results/quota.json` : `main.14.Gmax_uninformed` = [0.0019867888685107483, 0.0010604507245275906] | yes |
| 139 | Table 1 (main map) | 0.0211 ± 0.0033 | `results/quota.json` : `main.14.Gmax` = [0.021104115773860545, 0.003304642560485526] | yes |
| 140 | Table 1 (main map) | 0.0303 ± 0.0060 | `results/quota.json` : `main.14.false_punish_frac` = [0.030255625, 0.006032201336841113] | yes |
| 141 | Table 1 (main map) (param) | 8 | `results/stage2.json` : `main.M3c.100000.0.99.param.tol` = 8 | yes |
| 142 | Table 1 (main map) (param) | 100 | `results/stage2.json` : `main.M3c.100000.0.99.param.L` = 100 | yes |
| 143 | Table 1 (main map) (param) | 0.1 | `results/stage2.json` : `main.M3c.100000.0.99.param.p` = 0.1 | yes |
| 144 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.15.R_hon_b` = [2.824222351231168e-05, 2.9721996165525657e-05] | yes |
| 145 | Table 1 (main map) | 0.0059 ± 0.0004 | `results/quota.json` : `main.15.Gmax_uninformed` = [0.005911867723934272, 0.0003917633898750441] | yes |
| 146 | Table 1 (main map) | 0.0460 ± 0.0015 | `results/quota.json` : `main.15.Gmax` = [0.04601827255506004, 0.0014688955181857205] | yes |
| 147 | Table 1 (main map) | 0.0001 ± 0.0001 | `results/quota.json` : `main.15.false_punish_frac` = [9.375000000000002e-05, 6.711605092758883e-05] | yes |
| 148 | Table 1 (main map) (param) | 0.02 | `results/quota.json` : `main.88.param.eps` = 0.02 | yes |
| 149 | Table 1 (main map) (param) | 2500.0 | `results/quota.json` : `main.88.param.L` = 2500.0 | yes |
| 150 | Table 1 (main map) | 0.0162 ± 0.0001 | `results/quota.json` : `main.88.R_hon_b` = [0.016222466622181183, 0.00013373540540860911] | yes |
| 151 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.88.Gmax_uninformed` = [0.0, 0.0] | yes |
| 152 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.88.Gmax` = [0.0, 0.0] | yes |
| 153 | Table 1 (main map) (param) | 0.02 | `results/quota.json` : `main.89.param.eps` = 0.02 | yes |
| 154 | Table 1 (main map) (param) | 2500.0 | `results/quota.json` : `main.89.param.L` = 2500.0 | yes |
| 155 | Table 1 (main map) | 0.0162 ± 0.0001 | `results/quota.json` : `main.89.R_hon_b` = [0.01622567234568157, 0.00013439634325190946] | yes |
| 156 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.89.Gmax_uninformed` = [0.0, 0.0] | yes |
| 157 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.89.Gmax` = [0.0, 0.0] | yes |
| 158 | Table 1 (main map) (param) | 0.02 | `results/quota.json` : `main.90.param.eps` = 0.02 | yes |
| 159 | Table 1 (main map) (param) | 2500.0 | `results/quota.json` : `main.90.param.L` = 2500.0 | yes |
| 160 | Table 1 (main map) | 0.0162 ± 0.0001 | `results/quota.json` : `main.90.R_hon_b` = [0.016234143048632705, 0.00013399928726487472] | yes |
| 161 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.90.Gmax_uninformed` = [0.0, 0.0] | yes |
| 162 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.90.Gmax` = [0.0, 0.0] | yes |
| 163 | Table 1 (main map) (param) | 0.02 | `results/quota.json` : `main.91.param.eps` = 0.02 | yes |
| 164 | Table 1 (main map) (param) | 2500.0 | `results/quota.json` : `main.91.param.L` = 2500.0 | yes |
| 165 | Table 1 (main map) | 0.0162 ± 0.0002 | `results/quota.json` : `main.91.R_hon_b` = [0.016242242420091327, 0.00015193809282280473] | yes |
| 166 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.91.Gmax_uninformed` = [0.0, 0.0] | yes |
| 167 | Table 1 (main map) | 0.0000 ± 0.0000 | `results/quota.json` : `main.91.Gmax` = [0.0, 0.0] | yes |
| 168 | Table 1 (main map) (param) | 1000.0 | `results/quota.json` : `main.24.param.QW` = 1000.0 | yes |
| 169 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.24.param.nb` = 5.0 | yes |
| 170 | Table 1 (main map) | 0.0114 ± 0.0001 | `results/quota.json` : `main.24.R_hon_b` = [0.011358985509924945, 0.0001278198640795332] | yes |
| 171 | Table 1 (main map) | 0.0006 ± 0.0001 | `results/quota.json` : `main.24.Gmax_uninformed` = [0.0005548073657416974, 0.00011965296395798535] | yes |
| 172 | Table 1 (main map) | 0.0897 ± 0.0003 | `results/quota.json` : `main.24.Gmax` = [0.08967161764263046, 0.00032121729712398215] | yes |
| 173 | Table 1 (main map) | 4.3% | `results/quota.json` : `main.24.rewrite_frac_honest.0` = 0.04254322916666667 | yes |
| 174 | Table 1 (main map) (param) | 1000.0 | `results/quota.json` : `main.25.param.QW` = 1000.0 | yes |
| 175 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.25.param.nb` = 5.0 | yes |
| 176 | Table 1 (main map) | 0.0159 ± 0.0002 | `results/quota.json` : `main.25.R_hon_b` = [0.015930196903729623, 0.00023077750583777611] | yes |
| 177 | Table 1 (main map) | 0.0008 ± 0.0002 | `results/quota.json` : `main.25.Gmax_uninformed` = [0.0008276905697242399, 0.00020821705344426016] | yes |
| 178 | Table 1 (main map) | 0.0896 ± 0.0005 | `results/quota.json` : `main.25.Gmax` = [0.08962559922422259, 0.00046046016652284536] | yes |
| 179 | Table 1 (main map) | 6.4% | `results/quota.json` : `main.25.rewrite_frac_honest.0` = 0.06449520833333333 | yes |
| 180 | Table 1 (main map) (param) | 1000.0 | `results/quota.json` : `main.26.param.QW` = 1000.0 | yes |
| 181 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.26.param.nb` = 5.0 | yes |
| 182 | Table 1 (main map) | 0.0212 ± 0.0003 | `results/quota.json` : `main.26.R_hon_b` = [0.021175868521411057, 0.00033907639320009353] | yes |
| 183 | Table 1 (main map) | 0.0011 ± 0.0003 | `results/quota.json` : `main.26.Gmax_uninformed` = [0.0010603440309085114, 0.00031681921694933965] | yes |
| 184 | Table 1 (main map) | 0.0903 ± 0.0006 | `results/quota.json` : `main.26.Gmax` = [0.0903073434594557, 0.0006269234984294104] | yes |
| 185 | Table 1 (main map) | 9.0% | `results/quota.json` : `main.26.rewrite_frac_honest.0` = 0.09008177083333332 | yes |
| 186 | Table 1 (main map) (param) | 1000.0 | `results/quota.json` : `main.27.param.QW` = 1000.0 | yes |
| 187 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.27.param.nb` = 5.0 | yes |
| 188 | Table 1 (main map) | 0.0585 ± 0.0011 | `results/quota.json` : `main.27.R_hon_b` = [0.0585007504952225, 0.0010935331212447714] | yes |
| 189 | Table 1 (main map) | 0.0048 ± 0.0014 | `results/quota.json` : `main.27.Gmax_uninformed` = [0.004849327961781592, 0.001414774967156068] | yes |
| 190 | Table 1 (main map) | 0.1001 ± 0.0015 | `results/quota.json` : `main.27.Gmax` = [0.10007054933253469, 0.0014907376917003765] | yes |
| 191 | Table 1 (main map) | 27.0% | `results/quota.json` : `main.27.rewrite_frac_honest.0` = 0.27003916666666666 | yes |
| 192 | Table 1 (main map) (param) | 10.0 | `results/quota.json` : `main.48.param.nb` = 10.0 | yes |
| 193 | Table 1 (main map) (param) | 2.0 | `results/quota.json` : `main.48.param.nc` = 2.0 | yes |
| 194 | Table 1 (main map) | 0.0043 ± 0.0000 | `results/quota.json` : `main.48.R_hon_b` = [0.004319971072439137, 4.284686863300064e-05] | yes |
| 195 | Table 1 (main map) | 0.0224 ± 0.0004 | `results/quota.json` : `main.48.Gmax_uninformed` = [0.0223526428685449, 0.0004149203674855355] | yes |
| 196 | Table 1 (main map) | 0.1220 ± 0.0003 | `results/quota.json` : `main.48.Gmax` = [0.12199235301935166, 0.0003090831051183191] | yes |
| 197 | Table 1 (main map) | 10.9% | `results/quota.json` : `main.48.rewrite_frac_honest.0` = 0.10889864583333334 | yes |
| 198 | Table 1 (main map) (param) | 10.0 | `results/quota.json` : `main.49.param.nb` = 10.0 | yes |
| 199 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.49.param.nc` = 5.0 | yes |
| 200 | Table 1 (main map) | 0.0058 ± 0.0000 | `results/quota.json` : `main.49.R_hon_b` = [0.005786655351158127, 4.8939717410680985e-05] | yes |
| 201 | Table 1 (main map) | 0.0629 ± 0.0005 | `results/quota.json` : `main.49.Gmax_uninformed` = [0.06291523897832882, 0.000548849205656857] | yes |
| 202 | Table 1 (main map) | 0.0745 ± 0.0005 | `results/quota.json` : `main.49.Gmax` = [0.0744598778058058, 0.0004942700199051453] | yes |
| 203 | Table 1 (main map) | 15.4% | `results/quota.json` : `main.49.rewrite_frac_honest.0` = 0.15429437499999998 | yes |
| 204 | Table 1 (main map) (param) | 10.0 | `results/quota.json` : `main.50.param.nb` = 10.0 | yes |
| 205 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.50.param.nc` = 5.0 | yes |
| 206 | Table 1 (main map) | 0.0067 ± 0.0001 | `results/quota.json` : `main.50.R_hon_b` = [0.0066805923933157315, 5.205910502310007e-05] | yes |
| 207 | Table 1 (main map) | 0.0765 ± 0.0006 | `results/quota.json` : `main.50.Gmax_uninformed` = [0.07650078524317439, 0.0005957692391144835] | yes |
| 208 | Table 1 (main map) | 0.0801 ± 0.0006 | `results/quota.json` : `main.50.Gmax` = [0.08009899445853377, 0.0006125842420478615] | yes |
| 209 | Table 1 (main map) | 16.2% | `results/quota.json` : `main.50.rewrite_frac_honest.0` = 0.1622882291666667 | yes |
| 210 | Table 1 (main map) (param) | 20.0 | `results/quota.json` : `main.51.param.nb` = 20.0 | yes |
| 211 | Table 1 (main map) (param) | 2.0 | `results/quota.json` : `main.51.param.nc` = 2.0 | yes |
| 212 | Table 1 (main map) | 0.0067 ± 0.0001 | `results/quota.json` : `main.51.R_hon_b` = [0.006749752668118491, 0.00014312182899345896] | yes |
| 213 | Table 1 (main map) | 0.0987 ± 0.0021 | `results/quota.json` : `main.51.Gmax_uninformed` = [0.09870150726100962, 0.0021413623164054903] | yes |
| 214 | Table 1 (main map) | 0.1091 ± 0.0020 | `results/quota.json` : `main.51.Gmax` = [0.10905831409827854, 0.0020305895093121807] | yes |
| 215 | Table 1 (main map) | 34.4% | `results/quota.json` : `main.51.rewrite_frac_honest.0` = 0.34390166666666666 | yes |
| 216 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.72.param.nb` = 5.0 | yes |
| 217 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.72.param.nc` = 5.0 | yes |
| 218 | Table 1 (main map) | 0.0049 ± 0.0001 | `results/quota.json` : `main.72.R_hon_b` = [0.004910970024591075, 0.00014981565438407799] | yes |
| 219 | Table 1 (main map) | 0.0330 ± 0.0006 | `results/quota.json` : `main.72.Gmax_uninformed` = [0.032959501997645566, 0.0005716478473326057] | yes |
| 220 | Table 1 (main map) | 0.0330 ± 0.0006 | `results/quota.json` : `main.72.Gmax` = [0.032959501997645566, 0.0005716478473326057] | yes |
| 221 | Table 1 (main map) | 0.2% | `results/quota.json` : `main.72.rewrite_frac_honest.0` = 0.0024165625 | yes |
| 222 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.73.param.nb` = 5.0 | yes |
| 223 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.73.param.nc` = 5.0 | yes |
| 224 | Table 1 (main map) | 0.0043 ± 0.0000 | `results/quota.json` : `main.73.R_hon_b` = [0.004314312487605542, 3.054287368123385e-05] | yes |
| 225 | Table 1 (main map) | 0.0657 ± 0.0008 | `results/quota.json` : `main.73.Gmax_uninformed` = [0.06567336490465482, 0.0008464937797874927] | yes |
| 226 | Table 1 (main map) | 0.0657 ± 0.0008 | `results/quota.json` : `main.73.Gmax` = [0.06567336490465482, 0.0008464937797874927] | yes |
| 227 | Table 1 (main map) | 0.0% | `results/quota.json` : `main.73.rewrite_frac_honest.0` = 1.9062500000000003e-05 | yes |
| 228 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.74.param.nb` = 5.0 | yes |
| 229 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.74.param.nc` = 5.0 | yes |
| 230 | Table 1 (main map) | 0.0043 ± 0.0000 | `results/quota.json` : `main.74.R_hon_b` = [0.004348394883455883, 4.513116816536332e-05] | yes |
| 231 | Table 1 (main map) | 0.0654 ± 0.0005 | `results/quota.json` : `main.74.Gmax_uninformed` = [0.06542912303231828, 0.0004884332112432074] | yes |
| 232 | Table 1 (main map) | 0.0654 ± 0.0005 | `results/quota.json` : `main.74.Gmax` = [0.06542912303231828, 0.0004884332112432074] | yes |
| 233 | Table 1 (main map) | 0.0% | `results/quota.json` : `main.74.rewrite_frac_honest.0` = 0.00023645833333333334 | yes |
| 234 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.75.param.nb` = 5.0 | yes |
| 235 | Table 1 (main map) (param) | 5.0 | `results/quota.json` : `main.75.param.nc` = 5.0 | yes |
| 236 | Table 1 (main map) | 0.0094 ± 0.0005 | `results/quota.json` : `main.75.R_hon_b` = [0.009401969459927242, 0.0005295217495693913] | yes |
| 237 | Table 1 (main map) | 0.1040 ± 0.0028 | `results/quota.json` : `main.75.Gmax_uninformed` = [0.1040211681464083, 0.0027658914872585767] | yes |
| 238 | Table 1 (main map) | 0.1040 ± 0.0028 | `results/quota.json` : `main.75.Gmax` = [0.1040211681464083, 0.0027658914872585767] | yes |
| 239 | Table 1 (main map) | 2.2% | `results/quota.json` : `main.75.rewrite_frac_honest.0` = 0.022110625 | yes |

## Cross-check against the summary

All 239 value strings above were also searched, as exact strings, in `docs/direction1-summary.md` plus `docs/direction1-number-audit.md`: 239 of 239 found (script check; "consistent" = JSON formatting matches AND the string appears there).

## Design and setting numbers (not result values)

| Location | Value | Source | Consistent? |
|---|---|---|---|
| Sec 3, 5 | K=3, tau=1, T=10^5, r grid {0.5,0.8,0.9,0.99} | `quota.json` : `meta.K`, `meta.tau`, `meta.T`, `meta.rhos` | yes |
| Sec 3, 5 | eval seeds 5000-5031, tune seeds 0-15 | `quota.json` : `meta.eval_seeds`, `meta.tune_seeds` | yes |
| Sec 4(3), 5 (M2, M3c rows) | seeds 1000-1031 | `stage2.json` : `meta.eval_seeds` | yes |
| Sec 4(1) | 64 seeds, p=0.1, rho=0.9 (Markov), tau_delay=5 | `adaaudit_2x2.json` : `meta.seeds`, `meta.p_fixed`, `meta.rho_markov`, `meta.tau_delay` | yes |
| Sec 4(2) | 32 seeds; 13 cells x 7 deltas; delta=0.1, 0.2 | `adaaudit_tol.json` : `meta.seeds`, `meta.cells`, `meta.deltas` | yes |
| Sec 3, 4(4)(6), 7 (stage 2b) | T=20000, eval seeds 5000-5031 | `stage2b.json` : `meta.T`, `meta.eval_seeds` | yes (see note 1) |
| Sec 3, 5 | 45 strategies | `quota.json` : `meta.G_definition` ("max over 45 strategies") | yes |
| Sec 4(1) | K^2 = 9 | `adaaudit_2x2.json` : `meta.K`=3 | yes |
| Sec 4(5) | alpha targets 0.005, 0.01, 0.02, 0.05 | `stage2b.json` : `meta.alphas` (tags a0.005, a0.01, a0.02, a0.05) | yes |
| Sec 4(4), 6 | M4 eps=0.05 / 0.02 / 0.01, rw=0 / 1/3 / 1, L=2500 | `stage2b.json` : `consist_M4[9,11]`; `quota.json` : `main[88].param`; `stage2b.json` : `Gstar.M4.a0.01.0.5.param` | yes |
| Sec 4(4) | stage-2 raid at r=1: eps=0.2, L=20 | `stage2.json` : `main.M4["100000"]["1.0"].param` | yes |
| Sec 5 (misspec) | alpha target 0.02, variants r+0.05 / r-0.05 / oracle | `stage2b.json` : `rho_misspec[18,26,32,33]` (`tag`, `variant`, `strat`) | yes |
| Sec 4(5) F3 | alpha target 0.01, M3C | `stage2b.json` : `F3_uninformed[43,56,70]` (`fam`, `tag`) | yes |
| Sec 3 | kappa=1.0, r=0.5 | `stage2.json` : `checks.calibration_honest_flag_rate[0]` | yes |

## Notes

1. `stage2b.json` has `meta.T = 20000`; `docs/direction1-summary.md` does not state T for stage-2b cells (it lists T=1e5 only for the main table). The paper states T=20000 for those cells; the summary should be checked for the same omission.
2. Items deliberately NOT in the paper (no JSON source): winner-selection bias 0.573/0.522; old ~40% false-trigger rate; red-team scratchpad numbers; F4 "within 1.5x"; (the r=0.95 z=2.57 is now in the paper with a source; see the revision table below).
3. M4 parameters differ between files: `quota.json` rw=1/3, eps=0.02; `stage2b.json` rw=1. The paper labels them separately.
4. Figure 4 (`fig18b`) is a pre-fix stage-2 figure and carries no number from the paper.

## Numbers added in the pre-delivery revision

All checked against the JSON (mean ± 95% half-width; z = gain / (half-width/1.96)). Source files: S2B = `results/stage2b.json`, Q = `results/quota.json`.

| # | Location | Value | Source | Consistent? |
|---|---|---|---|---|
| R1 | Sec 4(5) F3, r=0.5 | z=9.7 | S2B `eval_rows.M3C[tag=a0.01,r=0.5].groups.edge_un.z` = 9.7269 | yes |
| R2 | Sec 4(5) F3, r=0.8 | 0.0158 ± 0.0044, z=6.95 | same path, r=0.8: gain [0.015762, 0.004442], z=6.9542 | yes |
| R3 | Sec 4(5) F3, r=0.9 | z=1.49 | same path, r=0.9: z=1.4865 | yes |
| R4 | Sec 4(5) F3, r=0.95 | 0.0047 ± 0.0036, z=2.57 | same path, r=0.95: gain [0.004739, 0.003607], z=2.5749 | yes |
| R5 | Sec 4(5) F3, r=0.95, alpha target 0.02 | -0.0013 ± 0.0031, z=-0.79 | S2B `eval_rows.M3C[tag=a0.02,r=0.95].groups.edge_un`: gain [-0.001267, 0.003135], z=-0.7922 | yes |
| R6 | Sec 4(5) F3, r=0.99 | z=-3.29 | S2B `eval_rows.M3C[tag=a0.01,r=0.99].groups.edge_un.z` = -3.2888 | yes |
| R7 | Sec 6 (M4 best deviation, r=0.5) | -0.0443 ± 0.0094 | S2B `eval_rows.M4[tag=a0.01,r=0.5].G_max` = [-0.044285, 0.009437] (eps=0.01, rw=1, T=20000) | yes |
| R8 | Sec 6 (M4 best deviation, r=0.99) | -0.0544 ± 0.0112 | S2B `eval_rows.M4[tag=a0.01,r=0.99].G_max` = [-0.054366, 0.011233] | yes |
| R9 | Abstract, Sec 5, Sec 7 (QT uninformed gain, range) | 0.0224 ± 0.0004 to 0.0987 ± 0.0021 | Q `main` QT rows, Table 1 (r=0.5 and r=0.99) | yes |
| R10 | Abstract, Sec 5, Sec 7 (QE uninformed gain, range) | 0.0330 ± 0.0006 to 0.1040 ± 0.0028 | Q `main` QE rows, Table 1 (r=0.5 and r=0.99) | yes |
| R11 | Sec 6 (ratio vs M3C) | about 2.3 | derived: M4 0.0162 / M3C 0.0070 (r=0.9, Q) = 2.31 | yes |
| R12 | Sec 6, Sec 7 (ratio vs QT) | 2.4 to 3.8 | derived: 0.0162 / QT 0.0067 and 0.0162 / QT 0.0043 (Q) | yes |
| R13 | Sec 6, Sec 7 (ratio vs QE) | 1.7 to 3.8 | derived: 0.0162 / QE 0.0094 and 0.0162 / QE 0.0043 (Q) | yes |
| R14 | Sec 6 (QT honest R/T range) | 0.0043 to 0.0067 | Q Table 1 rows (W=1000, optimal tuning) | yes |
| R15 | Sec 6 (QE honest R/T range) | 0.0043 to 0.0094 | Q Table 1 rows | yes |
| R16 | Sec 5 (M3C r=0.99 oracle gain, two cells) | 0.0044 ± 0.0059 (T=2e4, alpha target 0.02) vs 0.0460 ± 0.0015 (T=1e5 map, h=8, L=100) | S2B `rho_misspec[tag=a0.02,r=0.99,variant=oracle].G`; Q `main` M3C r=0.99 (Table 1) | yes |

Revision note: 16 rows added (R1--R16); the T=2e4 label now appears wherever stage-2b numbers are used.
