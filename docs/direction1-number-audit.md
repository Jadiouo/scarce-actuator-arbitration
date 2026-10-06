# 方向一數字稽核表

對應文件：`docs/direction1-summary.md`。每列＝summary 中出現的一個結果數字。「JSON 值」為原始浮點（pair 表示 [平均, 半寬]）；「一致?」檢查 summary 顯示值是否等於 JSON 值四捨五入到所示位數（衍生值註明換算）。

不列入本表：設計參數與格點（r 網格、W=1000、ε 與 α 目標格點等，出處為對應 JSON 的 meta 或 param）、時間點與 seeds 範圍、章節編號、文獻年份、百分比欄位中的「0」（`false_punish_frac` 等 JSON 值確為 0.0 者，只在主表以「0」表示）。未寫入 summary 的數字：紅方 scratchpad 的邊緣灌水得利、贏家偏誤 P(v>w|贏家) 實測值、notes 中舊版約 40% 的誤觸發率，它們在 `results/*.json` 找不到出處。

| # | 數字 | 出現位置（summary 章節） | JSON 檔與鍵路徑 | JSON 值 | 一致? |
|---|---|---|---|---|---|
| 1 | 3 | §1 | `results/quota.json` : `meta.K` | 3 | 是 |
| 2 | 1 | §1 | `results/quota.json` : `meta.tau` | 1 | 是 |
| 3 | 1.00 | §3 | `results/adaaudit_2x2.json` : `twobytwo["fixed\|100000\|iid_tauD\|honest"].honest_excl_frac[0]` | 1.0 | 是 |
| 4 | 0.812 | §3 | `results/adaaudit_2x2.json` : `twobytwo["fixed\|100000\|iid_tauD\|honest"].R[0]` | 81227.39165900918（衍生：JSON 值×1e-05） | 是 |
| 5 | 0.812 | §3 | `results/adaaudit_2x2.json` : `twobytwo["fixed\|100000\|markov_tauD\|honest"].R[0]` | 81235.0461188572（衍生：JSON 值×1e-05） | 是 |
| 6 | 5 | §3 | `results/adaaudit_2x2.json` : `meta.tau_delay` | 5 | 是 |
| 7 | 0.0 | §3 | `results/adaaudit_2x2.json` : `twobytwo["fixed\|100000\|iid_tau0\|honest"].R[0]` | 0.0 | 是 |
| 8 | 2.63 | §3 | `results/adaaudit_2x2.json` : `twobytwo["ada\|100000\|iid_tau0\|late_liar_b1.0"].R[0]` | 2.6300361870373936 | 是 |
| 9 | 9 | §3 | `results/adaaudit_2x2.json` : `K² = (meta.K)²` | meta.K=3 | 是 |
| 10 | 0.00 | §3 | `results/adaaudit_tol.json` : `cells["fixed\|100000\|0.9\|1\|0.1"].honest.honest_surv[0]` | 0.0 | 是 |
| 11 | 0.00 | §3 | `results/adaaudit_tol.json` : `cells["ada\|100000\|0.9\|1\|0.1"].honest.honest_surv[0]` | 0.0 | 是 |
| 12 | 16.7 | §3 | `results/adaaudit_tol.json` : `cells["ada\|100000\|0.9\|1\|0.1"].honest.B[0]` | 16.71875 | 是 |
| 13 | 9.4 | §3 | `results/adaaudit_tol.json` : `cells["fixed\|100000\|0.9\|1\|0.1"].honest.B[0]` | 9.4375 | 是 |
| 14 | 13 | §3 | `results/adaaudit_tol.json` : `len(meta.cells)` | 13 | 是 |
| 15 | 7 | §3 | `results/adaaudit_tol.json` : `len(meta.deltas)` | 7 | 是 |
| 16 | true | §3 | `results/adaaudit_tol.json` : `feasibility["ada\|100000\|b0.1"].feasible[6][4]` | True | 是 |
| 17 | 3645.9 | §3 | `results/stage2.json` : `bt_adaaudit_tau0.ideal_BT[2]` | 3645.909494694089 | 是 |
| 18 | 5626.1 | §3 | `results/stage2.json` : `bt_adaaudit_tau0.by_rho["0.9"].BT[2][0]` | 5626.125 | 是 |
| 19 | -0.0118 ± 0.0028 | §3 | `results/stage2.json` : `main.M3c["100000"]["0.9"].Gmax` | [-0.01177532731935538, 0.002829372795761977] | 是 |
| 20 | 0.0067 ± 0.0015 | §3 | `results/stage2.json` : `main.M3c["100000"]["0.9"].R_b_honest` | [0.006708961159344852, 0.001537032855914119] | 是 |
| 21 | 0.2795 ± 0.0004 | §3 | `results/stage2.json` : `main.M2["100000"]["0.5"].Gmax` | [0.2794880284199983, 0.0004205750367242674] | 是 |
| 22 | 0.2805 ± 0.0022 | §3 | `results/stage2.json` : `main.M2["100000"]["0.99"].Gmax` | [0.28048819561516036, 0.0021518647500347494] | 是 |
| 23 | 0.0211 ± 0.0033 | §3 | `results/quota.json` : `main[14].Gmax` | [0.021104115773860545, 0.003304642560485526] | 是 |
| 24 | 0.0870 ± 0.0053 | §3 | `results/quota.json` : `main[12].Gmax` | [0.086962112771764, 0.005295154442673111] | 是 |
| 25 | 0.0021 ± 0.0025 | §3 | `results/stage2.json` : `main.M3c["100000"]["0.8"].Gmax` | [0.0021185843561695797, 0.0024900939082519776] | 是 |
| 26 | false | §3 | `results/stage2.json` : `main.M3c["10000"]["0.8"].eval_pass` | False | 是 |
| 27 | false | §3 | `results/stage2.json` : `main.M3c["10000"]["0.9"].eval_pass` | False | 是 |
| 28 | 0.1811 ± 0.0721 | §3 | `results/stage2.json` : `main.M4["100000"]["1.0"].Gmax` | [0.18105352017215026, 0.07206902628243358] | 是 |
| 29 | 0.05 | §3 | `results/stage2b.json` : `consist_M4[9].eps` | 0.05 | 是 |
| 30 | 0.0 | §3 | `results/stage2b.json` : `consist_M4[9].rw` | 0.0 | 是 |
| 31 | 0.0356 ± 0.0008 | §3 | `results/stage2b.json` : `consist_M4[9].gain_edgeO` | [0.03556691436913642, 0.0007521353638302052] | 是 |
| 32 | 1.0 | §3 | `results/stage2b.json` : `consist_M4[11].rw` | 1.0 | 是 |
| 33 | -0.1450 ± 0.0039 | §3 | `results/stage2b.json` : `consist_M4[11].gain_edgeO` | [-0.14497351692990867, 0.003948775355400535] | 是 |
| 34 | 1.26 | §3 | `results/stage2b.json` : `scaling_best_family["a0.005"].fit_all.s` | 1.260000000000002 | 是 |
| 35 | 1.13 | §3 | `results/stage2b.json` : `scaling_best_family["a0.005"].fit_all.s_ci[0]` | 1.1300000000000017 | 是 |
| 36 | 1.39 | §3 | `results/stage2b.json` : `scaling_best_family["a0.005"].fit_all.s_ci[1]` | 1.3900000000000023 | 是 |
| 37 | 2.18 | §3 | `results/stage2b.json` : `scaling_best_family["a0.05"].fit_all.s` | 2.180000000000003 | 是 |
| 38 | 1.77 | §3 | `results/stage2b.json` : `scaling_best_family["a0.05"].fit_all.s_ci[0]` | 1.7700000000000022 | 是 |
| 39 | 2.63 | §3 | `results/stage2b.json` : `scaling_best_family["a0.05"].fit_all.s_ci[1]` | 2.6300000000000034 | 是 |
| 40 | 0.0227 ± 0.0060 | §3 | `results/stage2b.json` : `tau_cmp_own_frontier["0.9"]["a0.05"].tau1.G` | [0.02268993697697485, 0.0060403626694569305] | 是 |
| 41 | 0.0221 ± 0.0069 | §3 | `results/stage2b.json` : `tau_cmp_own_frontier["0.9"]["a0.05"].tau5.G` | [0.02211527187927447, 0.006882338707465279] | 是 |
| 42 | 0.0287 ± 0.0058 | §3 | `results/stage2b.json` : `F3_uninformed[43].edge_uninformed.gain` | [0.028723287686942968, 0.0057878206503488225] | 是 |
| 43 | 0.1235 ± 0.0077 | §3 | `results/stage2b.json` : `F3_uninformed[43].edge_oracle.gain` | [0.12349621787238228, 0.007730479086694218] | 是 |
| 44 | 0.0011 ± 0.0015 | §3 | `results/stage2b.json` : `F3_uninformed[56].edge_uninformed.gain` | [0.0011362486228142891, 0.0014981321301022745] | 是 |
| 45 | 1.49 | §3 | `results/stage2b.json` : `F3_uninformed[56].edge_uninformed.z` | 1.486549320962745 | 是 |
| 46 | -0.0108 ± 0.0064 | §3 | `results/stage2b.json` : `F3_uninformed[70].edge_uninformed.gain` | [-0.010817061823506715, 0.0064465743912310155] | 是 |
| 47 | -0.0443 ± 0.0094 | §3 | `results/stage2b.json` : `Gstar.M4["a0.01"]["0.5"].G` | [-0.04428507698001881, 0.009436728953495954] | 是 |
| 48 | 0.0000 ± 0.0000 | §3 | `results/stage2b.json` : `Gstar.M4["a0.01"]["0.5"].alpha` | [0.0, 0.0] | 是 |
| 49 | 100000 | §4 | `results/quota.json` : `meta.T` | 100000 | 是 |
| 50 | 1 | §4 | `results/quota.json` : `meta.tau` | 1 | 是 |
| 51 | 3 | §4 | `results/quota.json` : `meta.K` | 3 | 是 |
| 52 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[4].R_hon_b` | [0.0, 0.0] | 是 |
| 53 | 0.2793 ± 0.0004 | §4 主表 | `results/quota.json` : `main[4].Gmax_uninformed` | [0.27930049154850023, 0.0003786066909699181] | 是 |
| 54 | 0.2793 ± 0.0004 | §4 主表 | `results/quota.json` : `main[4].Gmax` | [0.27930049154850023, 0.0003786066909699181] | 是 |
| 55 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[5].R_hon_b` | [0.0, 0.0] | 是 |
| 56 | 0.2793 ± 0.0006 | §4 主表 | `results/quota.json` : `main[5].Gmax_uninformed` | [0.2793018980631612, 0.0005998432879280821] | 是 |
| 57 | 0.2793 ± 0.0006 | §4 主表 | `results/quota.json` : `main[5].Gmax` | [0.2793018980631612, 0.0005998432879280821] | 是 |
| 58 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[6].R_hon_b` | [0.0, 0.0] | 是 |
| 59 | 0.2792 ± 0.0009 | §4 主表 | `results/quota.json` : `main[6].Gmax_uninformed` | [0.27922695968122613, 0.0008732689130059621] | 是 |
| 60 | 0.2792 ± 0.0009 | §4 主表 | `results/quota.json` : `main[6].Gmax` | [0.27922695968122613, 0.0008732689130059621] | 是 |
| 61 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[7].R_hon_b` | [0.0, 0.0] | 是 |
| 62 | 0.2806 ± 0.0028 | §4 主表 | `results/quota.json` : `main[7].Gmax_uninformed` | [0.28055676737104984, 0.0027847341758342925] | 是 |
| 63 | 0.2806 ± 0.0028 | §4 主表 | `results/quota.json` : `main[7].Gmax` | [0.28055676737104984, 0.0027847341758342925] | 是 |
| 64 | 0.2236 ± 0.0002 | §4 主表 | `results/quota.json` : `main[8].R_hon_b` | [0.2236292718177209, 0.00024121174865107035] | 是 |
| 65 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[8].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 66 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[8].Gmax` | [0.0, 0.0] | 是 |
| 67 | 0.2237 ± 0.0003 | §4 主表 | `results/quota.json` : `main[9].R_hon_b` | [0.2237241769805047, 0.00029827952847249963] | 是 |
| 68 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[9].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 69 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[9].Gmax` | [0.0, 0.0] | 是 |
| 70 | 0.2239 ± 0.0004 | §4 主表 | `results/quota.json` : `main[10].R_hon_b` | [0.223899874487951, 0.00039507173591193495] | 是 |
| 71 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[10].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 72 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[10].Gmax` | [0.0, 0.0] | 是 |
| 73 | 0.2240 ± 0.0014 | §4 主表 | `results/quota.json` : `main[11].R_hon_b` | [0.22400934627083816, 0.0013689274233931333] | 是 |
| 74 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[11].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 75 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[11].Gmax` | [0.0, 0.0] | 是 |
| 76 | 0.2795 ± 0.0004 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.5"].Gmax` | [0.2794880284199983, 0.0004205750367242674] | 是 |
| 77 | 0.0000 ± 0.0000 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.5"].R_b_honest` | [0.0, 0.0] | 是 |
| 78 | 0.1 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.5"].param.p` | 0.1 | 是 |
| 79 | 20 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.5"].param.L` | 20 | 是 |
| 80 | 0.2794 ± 0.0005 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.8"].Gmax` | [0.27941372069528836, 0.0005471637218847936] | 是 |
| 81 | 0.0000 ± 0.0000 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.8"].R_b_honest` | [0.0, 0.0] | 是 |
| 82 | 0.1 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.8"].param.p` | 0.1 | 是 |
| 83 | 20 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.8"].param.L` | 20 | 是 |
| 84 | 0.2792 ± 0.0009 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.9"].Gmax` | [0.2792442333713937, 0.0008969882724184369] | 是 |
| 85 | 0.0000 ± 0.0000 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.9"].R_b_honest` | [0.0, 0.0] | 是 |
| 86 | 0.1 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.9"].param.p` | 0.1 | 是 |
| 87 | 20 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.9"].param.L` | 20 | 是 |
| 88 | 0.2805 ± 0.0022 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.99"].Gmax` | [0.28048819561516036, 0.0021518647500347494] | 是 |
| 89 | 0.0000 ± 0.0000 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.99"].R_b_honest` | [0.0, 0.0] | 是 |
| 90 | 0.1 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.99"].param.p` | 0.1 | 是 |
| 91 | 20 | §4 主表 | `results/stage2.json` : `main.M2["100000"]["0.99"].param.L` | 20 | 是 |
| 92 | 0.0071 ± 0.0018 | §4 主表 | `results/quota.json` : `main[12].R_hon_b` | [0.007078929144348073, 0.0017511954756897544] | 是 |
| 93 | 0.0213 ± 0.0039 | §4 主表 | `results/quota.json` : `main[12].Gmax_uninformed` | [0.021327654113202883, 0.0039011376120500494] | 是 |
| 94 | 0.0870 ± 0.0053 | §4 主表 | `results/quota.json` : `main[12].Gmax` | [0.086962112771764, 0.005295154442673111] | 是 |
| 95 | 0.0285 ± 0.0061 | §4 主表 | `results/quota.json` : `main[12].false_punish_frac` | [0.02845197916666667, 0.006071740203791877] | 是 |
| 96 | 0.1 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.5"].param.p` | 0.1 | 是 |
| 97 | 6 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.5"].param.tol` | 6 | 是 |
| 98 | 2500 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.5"].param.L` | 2500 | 是 |
| 99 | 0.0081 ± 0.0012 | §4 主表 | `results/quota.json` : `main[13].R_hon_b` | [0.008076755633298922, 0.0011787897816710386] | 是 |
| 100 | 0.0037 ± 0.0019 | §4 主表 | `results/quota.json` : `main[13].Gmax_uninformed` | [0.0037020866106143094, 0.0018700871104188519] | 是 |
| 101 | 0.0335 ± 0.0044 | §4 主表 | `results/quota.json` : `main[13].Gmax` | [0.03353653062447393, 0.0043961176575323295] | 是 |
| 102 | 0.0339 ± 0.0042 | §4 主表 | `results/quota.json` : `main[13].false_punish_frac` | [0.0338728125, 0.004243578980961265] | 是 |
| 103 | 0.1 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.8"].param.p` | 0.1 | 是 |
| 104 | 6 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.8"].param.tol` | 6 | 是 |
| 105 | 2500 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.8"].param.L` | 2500 | 是 |
| 106 | 0.0070 ± 0.0016 | §4 主表 | `results/quota.json` : `main[14].R_hon_b` | [0.0069707399001550895, 0.0016374238390795821] | 是 |
| 107 | 0.0020 ± 0.0011 | §4 主表 | `results/quota.json` : `main[14].Gmax_uninformed` | [0.0019867888685107483, 0.0010604507245275906] | 是 |
| 108 | 0.0211 ± 0.0033 | §4 主表 | `results/quota.json` : `main[14].Gmax` | [0.021104115773860545, 0.003304642560485526] | 是 |
| 109 | 0.0303 ± 0.0060 | §4 主表 | `results/quota.json` : `main[14].false_punish_frac` | [0.030255625, 0.006032201336841113] | 是 |
| 110 | 0.1 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.9"].param.p` | 0.1 | 是 |
| 111 | 6 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.9"].param.tol` | 6 | 是 |
| 112 | 2500 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.9"].param.L` | 2500 | 是 |
| 113 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[15].R_hon_b` | [2.824222351231168e-05, 2.9721996165525657e-05] | 是 |
| 114 | 0.0059 ± 0.0004 | §4 主表 | `results/quota.json` : `main[15].Gmax_uninformed` | [0.005911867723934272, 0.0003917633898750441] | 是 |
| 115 | 0.0460 ± 0.0015 | §4 主表 | `results/quota.json` : `main[15].Gmax` | [0.04601827255506004, 0.0014688955181857205] | 是 |
| 116 | 0.0001 ± 0.0001 | §4 主表 | `results/quota.json` : `main[15].false_punish_frac` | [9.375000000000002e-05, 6.711605092758883e-05] | 是 |
| 117 | 0.1 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.99"].param.p` | 0.1 | 是 |
| 118 | 8 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.99"].param.tol` | 8 | 是 |
| 119 | 100 | §4 主表 | `results/stage2.json` : `main.M3c["100000"]["0.99"].param.L` | 100 | 是 |
| 120 | 0.0162 ± 0.0001 | §4 主表 | `results/quota.json` : `main[88].R_hon_b` | [0.016222466622181183, 0.00013373540540860911] | 是 |
| 121 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[88].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 122 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[88].Gmax` | [0.0, 0.0] | 是 |
| 123 | 0.02 | §4 主表 | `results/quota.json` : `main[88].param.eps` | 0.02 | 是 |
| 124 | 2500 | §4 主表 | `results/quota.json` : `main[88].param.L` | 2500.0 | 是 |
| 125 | 0.0200 ± 0.0002 | §4 主表 | `results/quota.json` : `main[88].raid_waste_frac` | [0.019975625, 0.00015828944490058554] | 是 |
| 126 | 0.0162 ± 0.0001 | §4 主表 | `results/quota.json` : `main[89].R_hon_b` | [0.01622567234568157, 0.00013439634325190946] | 是 |
| 127 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[89].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 128 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[89].Gmax` | [0.0, 0.0] | 是 |
| 129 | 0.02 | §4 主表 | `results/quota.json` : `main[89].param.eps` | 0.02 | 是 |
| 130 | 2500 | §4 主表 | `results/quota.json` : `main[89].param.L` | 2500.0 | 是 |
| 131 | 0.0200 ± 0.0002 | §4 主表 | `results/quota.json` : `main[89].raid_waste_frac` | [0.019975625, 0.00015828944490058554] | 是 |
| 132 | 0.0162 ± 0.0001 | §4 主表 | `results/quota.json` : `main[90].R_hon_b` | [0.016234143048632705, 0.00013399928726487472] | 是 |
| 133 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[90].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 134 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[90].Gmax` | [0.0, 0.0] | 是 |
| 135 | 0.02 | §4 主表 | `results/quota.json` : `main[90].param.eps` | 0.02 | 是 |
| 136 | 2500 | §4 主表 | `results/quota.json` : `main[90].param.L` | 2500.0 | 是 |
| 137 | 0.0200 ± 0.0002 | §4 主表 | `results/quota.json` : `main[90].raid_waste_frac` | [0.019975625, 0.00015828944490058554] | 是 |
| 138 | 0.0162 ± 0.0002 | §4 主表 | `results/quota.json` : `main[91].R_hon_b` | [0.016242242420091327, 0.00015193809282280473] | 是 |
| 139 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[91].Gmax_uninformed` | [0.0, 0.0] | 是 |
| 140 | 0.0000 ± 0.0000 | §4 主表 | `results/quota.json` : `main[91].Gmax` | [0.0, 0.0] | 是 |
| 141 | 0.02 | §4 主表 | `results/quota.json` : `main[91].param.eps` | 0.02 | 是 |
| 142 | 2500 | §4 主表 | `results/quota.json` : `main[91].param.L` | 2500.0 | 是 |
| 143 | 0.0200 ± 0.0002 | §4 主表 | `results/quota.json` : `main[91].raid_waste_frac` | [0.019975625, 0.00015828944490058554] | 是 |
| 144 | 0.0114 ± 0.0001 | §4 主表 | `results/quota.json` : `main[24].R_hon_b` | [0.011358985509924945, 0.0001278198640795332] | 是 |
| 145 | 0.0006 ± 0.0001 | §4 主表 | `results/quota.json` : `main[24].Gmax_uninformed` | [0.0005548073657416974, 0.00011965296395798535] | 是 |
| 146 | 0.0897 ± 0.0003 | §4 主表 | `results/quota.json` : `main[24].Gmax` | [0.08967161764263046, 0.00032121729712398215] | 是 |
| 147 | 4.3% | §4 主表 | `results/quota.json` : `main[24].rewrite_frac_honest` | 0.04254322916666667（衍生：JSON 值×100（百分比）） | 是 |
| 148 | 1000 | §4 主表 | `results/quota.json` : `main[24].param.QW` | 1000.0 | 是 |
| 149 | 5 | §4 主表 | `results/quota.json` : `main[24].param.nb` | 5.0 | 是 |
| 150 | 0.0159 ± 0.0002 | §4 主表 | `results/quota.json` : `main[25].R_hon_b` | [0.015930196903729623, 0.00023077750583777611] | 是 |
| 151 | 0.0008 ± 0.0002 | §4 主表 | `results/quota.json` : `main[25].Gmax_uninformed` | [0.0008276905697242399, 0.00020821705344426016] | 是 |
| 152 | 0.0896 ± 0.0005 | §4 主表 | `results/quota.json` : `main[25].Gmax` | [0.08962559922422259, 0.00046046016652284536] | 是 |
| 153 | 6.4% | §4 主表 | `results/quota.json` : `main[25].rewrite_frac_honest` | 0.06449520833333333（衍生：JSON 值×100（百分比）） | 是 |
| 154 | 1000 | §4 主表 | `results/quota.json` : `main[25].param.QW` | 1000.0 | 是 |
| 155 | 5 | §4 主表 | `results/quota.json` : `main[25].param.nb` | 5.0 | 是 |
| 156 | 0.0212 ± 0.0003 | §4 主表 | `results/quota.json` : `main[26].R_hon_b` | [0.021175868521411057, 0.00033907639320009353] | 是 |
| 157 | 0.0011 ± 0.0003 | §4 主表 | `results/quota.json` : `main[26].Gmax_uninformed` | [0.0010603440309085114, 0.00031681921694933965] | 是 |
| 158 | 0.0903 ± 0.0006 | §4 主表 | `results/quota.json` : `main[26].Gmax` | [0.0903073434594557, 0.0006269234984294104] | 是 |
| 159 | 9.0% | §4 主表 | `results/quota.json` : `main[26].rewrite_frac_honest` | 0.09008177083333332（衍生：JSON 值×100（百分比）） | 是 |
| 160 | 1000 | §4 主表 | `results/quota.json` : `main[26].param.QW` | 1000.0 | 是 |
| 161 | 5 | §4 主表 | `results/quota.json` : `main[26].param.nb` | 5.0 | 是 |
| 162 | 0.0585 ± 0.0011 | §4 主表 | `results/quota.json` : `main[27].R_hon_b` | [0.0585007504952225, 0.0010935331212447714] | 是 |
| 163 | 0.0048 ± 0.0014 | §4 主表 | `results/quota.json` : `main[27].Gmax_uninformed` | [0.004849327961781592, 0.001414774967156068] | 是 |
| 164 | 0.1001 ± 0.0015 | §4 主表 | `results/quota.json` : `main[27].Gmax` | [0.10007054933253469, 0.0014907376917003765] | 是 |
| 165 | 27.0% | §4 主表 | `results/quota.json` : `main[27].rewrite_frac_honest` | 0.27003916666666666（衍生：JSON 值×100（百分比）） | 是 |
| 166 | 1000 | §4 主表 | `results/quota.json` : `main[27].param.QW` | 1000.0 | 是 |
| 167 | 5 | §4 主表 | `results/quota.json` : `main[27].param.nb` | 5.0 | 是 |
| 168 | 0.0043 ± 0.0000 | §4 主表 | `results/quota.json` : `main[48].R_hon_b` | [0.004319971072439137, 4.284686863300064e-05] | 是 |
| 169 | 0.0224 ± 0.0004 | §4 主表 | `results/quota.json` : `main[48].Gmax_uninformed` | [0.0223526428685449, 0.0004149203674855355] | 是 |
| 170 | 0.1220 ± 0.0003 | §4 主表 | `results/quota.json` : `main[48].Gmax` | [0.12199235301935166, 0.0003090831051183191] | 是 |
| 171 | 10.9% | §4 主表 | `results/quota.json` : `main[48].rewrite_frac_honest` | 0.10889864583333334（衍生：JSON 值×100（百分比）） | 是 |
| 172 | 1000 | §4 主表 | `results/quota.json` : `main[48].param.QW` | 1000.0 | 是 |
| 173 | 10 | §4 主表 | `results/quota.json` : `main[48].param.nb` | 10.0 | 是 |
| 174 | 2 | §4 主表 | `results/quota.json` : `main[48].param.nc` | 2.0 | 是 |
| 175 | 0.0058 ± 0.0000 | §4 主表 | `results/quota.json` : `main[49].R_hon_b` | [0.005786655351158127, 4.8939717410680985e-05] | 是 |
| 176 | 0.0629 ± 0.0005 | §4 主表 | `results/quota.json` : `main[49].Gmax_uninformed` | [0.06291523897832882, 0.000548849205656857] | 是 |
| 177 | 0.0745 ± 0.0005 | §4 主表 | `results/quota.json` : `main[49].Gmax` | [0.0744598778058058, 0.0004942700199051453] | 是 |
| 178 | 15.4% | §4 主表 | `results/quota.json` : `main[49].rewrite_frac_honest` | 0.15429437499999998（衍生：JSON 值×100（百分比）） | 是 |
| 179 | 1000 | §4 主表 | `results/quota.json` : `main[49].param.QW` | 1000.0 | 是 |
| 180 | 10 | §4 主表 | `results/quota.json` : `main[49].param.nb` | 10.0 | 是 |
| 181 | 5 | §4 主表 | `results/quota.json` : `main[49].param.nc` | 5.0 | 是 |
| 182 | 0.0067 ± 0.0001 | §4 主表 | `results/quota.json` : `main[50].R_hon_b` | [0.0066805923933157315, 5.205910502310007e-05] | 是 |
| 183 | 0.0765 ± 0.0006 | §4 主表 | `results/quota.json` : `main[50].Gmax_uninformed` | [0.07650078524317439, 0.0005957692391144835] | 是 |
| 184 | 0.0801 ± 0.0006 | §4 主表 | `results/quota.json` : `main[50].Gmax` | [0.08009899445853377, 0.0006125842420478615] | 是 |
| 185 | 16.2% | §4 主表 | `results/quota.json` : `main[50].rewrite_frac_honest` | 0.1622882291666667（衍生：JSON 值×100（百分比）） | 是 |
| 186 | 1000 | §4 主表 | `results/quota.json` : `main[50].param.QW` | 1000.0 | 是 |
| 187 | 10 | §4 主表 | `results/quota.json` : `main[50].param.nb` | 10.0 | 是 |
| 188 | 5 | §4 主表 | `results/quota.json` : `main[50].param.nc` | 5.0 | 是 |
| 189 | 0.0067 ± 0.0001 | §4 主表 | `results/quota.json` : `main[51].R_hon_b` | [0.006749752668118491, 0.00014312182899345896] | 是 |
| 190 | 0.0987 ± 0.0021 | §4 主表 | `results/quota.json` : `main[51].Gmax_uninformed` | [0.09870150726100962, 0.0021413623164054903] | 是 |
| 191 | 0.1091 ± 0.0020 | §4 主表 | `results/quota.json` : `main[51].Gmax` | [0.10905831409827854, 0.0020305895093121807] | 是 |
| 192 | 34.4% | §4 主表 | `results/quota.json` : `main[51].rewrite_frac_honest` | 0.34390166666666666（衍生：JSON 值×100（百分比）） | 是 |
| 193 | 1000 | §4 主表 | `results/quota.json` : `main[51].param.QW` | 1000.0 | 是 |
| 194 | 20 | §4 主表 | `results/quota.json` : `main[51].param.nb` | 20.0 | 是 |
| 195 | 2 | §4 主表 | `results/quota.json` : `main[51].param.nc` | 2.0 | 是 |
| 196 | 0.0049 ± 0.0001 | §4 主表 | `results/quota.json` : `main[72].R_hon_b` | [0.004910970024591075, 0.00014981565438407799] | 是 |
| 197 | 0.0330 ± 0.0006 | §4 主表 | `results/quota.json` : `main[72].Gmax_uninformed` | [0.032959501997645566, 0.0005716478473326057] | 是 |
| 198 | 0.0330 ± 0.0006 | §4 主表 | `results/quota.json` : `main[72].Gmax` | [0.032959501997645566, 0.0005716478473326057] | 是 |
| 199 | 0.2% | §4 主表 | `results/quota.json` : `main[72].rewrite_frac_honest` | 0.0024165625（衍生：JSON 值×100（百分比）） | 是 |
| 200 | 1000 | §4 主表 | `results/quota.json` : `main[72].param.QW` | 1000.0 | 是 |
| 201 | 5 | §4 主表 | `results/quota.json` : `main[72].param.nb` | 5.0 | 是 |
| 202 | 5 | §4 主表 | `results/quota.json` : `main[72].param.nc` | 5.0 | 是 |
| 203 | 0.0043 ± 0.0000 | §4 主表 | `results/quota.json` : `main[73].R_hon_b` | [0.004314312487605542, 3.054287368123385e-05] | 是 |
| 204 | 0.0657 ± 0.0008 | §4 主表 | `results/quota.json` : `main[73].Gmax_uninformed` | [0.06567336490465482, 0.0008464937797874927] | 是 |
| 205 | 0.0657 ± 0.0008 | §4 主表 | `results/quota.json` : `main[73].Gmax` | [0.06567336490465482, 0.0008464937797874927] | 是 |
| 206 | 0.0% | §4 主表 | `results/quota.json` : `main[73].rewrite_frac_honest` | 1.9062500000000003e-05（衍生：JSON 值×100（百分比）） | 是 |
| 207 | 1000 | §4 主表 | `results/quota.json` : `main[73].param.QW` | 1000.0 | 是 |
| 208 | 5 | §4 主表 | `results/quota.json` : `main[73].param.nb` | 5.0 | 是 |
| 209 | 5 | §4 主表 | `results/quota.json` : `main[73].param.nc` | 5.0 | 是 |
| 210 | 0.0043 ± 0.0000 | §4 主表 | `results/quota.json` : `main[74].R_hon_b` | [0.004348394883455883, 4.513116816536332e-05] | 是 |
| 211 | 0.0654 ± 0.0005 | §4 主表 | `results/quota.json` : `main[74].Gmax_uninformed` | [0.06542912303231828, 0.0004884332112432074] | 是 |
| 212 | 0.0654 ± 0.0005 | §4 主表 | `results/quota.json` : `main[74].Gmax` | [0.06542912303231828, 0.0004884332112432074] | 是 |
| 213 | 0.0% | §4 主表 | `results/quota.json` : `main[74].rewrite_frac_honest` | 0.00023645833333333334（衍生：JSON 值×100（百分比）） | 是 |
| 214 | 1000 | §4 主表 | `results/quota.json` : `main[74].param.QW` | 1000.0 | 是 |
| 215 | 5 | §4 主表 | `results/quota.json` : `main[74].param.nb` | 5.0 | 是 |
| 216 | 5 | §4 主表 | `results/quota.json` : `main[74].param.nc` | 5.0 | 是 |
| 217 | 0.0094 ± 0.0005 | §4 主表 | `results/quota.json` : `main[75].R_hon_b` | [0.009401969459927242, 0.0005295217495693913] | 是 |
| 218 | 0.1040 ± 0.0028 | §4 主表 | `results/quota.json` : `main[75].Gmax_uninformed` | [0.1040211681464083, 0.0027658914872585767] | 是 |
| 219 | 0.1040 ± 0.0028 | §4 主表 | `results/quota.json` : `main[75].Gmax` | [0.1040211681464083, 0.0027658914872585767] | 是 |
| 220 | 2.2% | §4 主表 | `results/quota.json` : `main[75].rewrite_frac_honest` | 0.022110625（衍生：JSON 值×100（百分比）） | 是 |
| 221 | 1000 | §4 主表 | `results/quota.json` : `main[75].param.QW` | 1000.0 | 是 |
| 222 | 5 | §4 主表 | `results/quota.json` : `main[75].param.nb` | 5.0 | 是 |
| 223 | 5 | §4 主表 | `results/quota.json` : `main[75].param.nc` | 5.0 | 是 |
| 224 | 0.2793 ± 0.0004 | §5 | `results/quota.json` : `main[4].Gmax` | [0.27930049154850023, 0.0003786066909699181] | 是 |
| 225 | 0.2795 ± 0.0004 | §5 | `results/stage2.json` : `main.M2["100000"]["0.5"].Gmax` | [0.2794880284199983, 0.0004205750367242674] | 是 |
| 226 | 0.1875 ± 0.0007 | §5 | `results/quota.json` : `main[4].honest_util_agent1` | [0.18745347021268274, 0.0006672699115254098] | 是 |
| 227 | 0.0070 ± 0.0016 | §5 | `results/quota.json` : `main[14].R_hon_b` | [0.0069707399001550895, 0.0016374238390795821] | 是 |
| 228 | 0.0303 ± 0.0060 | §5 | `results/quota.json` : `main[14].false_punish_frac` | [0.030255625, 0.006032201336841113] | 是 |
| 229 | 0.0213 ± 0.0039 | §5 | `results/quota.json` : `main[12].Gmax_uninformed` | [0.021327654113202883, 0.0039011376120500494] | 是 |
| 230 | 0.0211 ± 0.0033 | §5 | `results/quota.json` : `main[14].Gmax` | [0.021104115773860545, 0.003304642560485526] | 是 |
| 231 | 0.0870 ± 0.0053 | §5 | `results/quota.json` : `main[12].Gmax` | [0.086962112771764, 0.005295154442673111] | 是 |
| 232 | 0.0460 ± 0.0015 | §5 | `results/quota.json` : `main[15].Gmax` | [0.04601827255506004, 0.0014688955181857205] | 是 |
| 233 | 8 | §5 | `results/stage2.json` : `main.M3c["100000"]["0.99"].param.tol` | 8 | 是 |
| 234 | 0.0001 ± 0.0001 | §5 | `results/quota.json` : `main[15].false_punish_frac` | [9.375000000000002e-05, 6.711605092758883e-05] | 是 |
| 235 | 0.3073 ± 0.0212 | §5 | `results/stage2b.json` : `rho_misspec[18].alpha` | [0.3072546875, 0.021157987755285438] | 是 |
| 236 | 0.9531 ± 0.0032 | §5 | `results/stage2b.json` : `rho_misspec[26].alpha` | [0.9530708333333333, 0.003198593922015889] | 是 |
| 237 | 0.0044 ± 0.0059 | §5 | `results/stage2b.json` : `rho_misspec[32].G` | [0.004363548968191942, 0.005875346405022896] | 是 |
| 238 | 0.0556 ± 0.0019 | §5 | `results/stage2b.json` : `rho_misspec[33].G` | [0.05562505849202838, 0.0019310726015046944] | 是 |
| 239 | 0.0048 ± 0.0014 | §5 | `results/quota.json` : `main[27].Gmax_uninformed` | [0.004849327961781592, 0.001414774967156068] | 是 |
| 240 | 0.0897 ± 0.0003 | §5 | `results/quota.json` : `main[24].Gmax` | [0.08967161764263046, 0.00032121729712398215] | 是 |
| 241 | 4.3% | §5 | `results/quota.json` : `main[24].rewrite_frac_honest` | 0.04254322916666667（衍生：JSON 值×100（百分比）） | 是 |
| 242 | 27.0% | §5 | `results/quota.json` : `main[27].rewrite_frac_honest` | 0.27003916666666666（衍生：JSON 值×100（百分比）） | 是 |
| 243 | 0.0114 ± 0.0001 | §5 | `results/quota.json` : `main[24].R_hon_b` | [0.011358985509924945, 0.0001278198640795332] | 是 |
| 244 | 0.0585 ± 0.0011 | §5 | `results/quota.json` : `main[27].R_hon_b` | [0.0585007504952225, 0.0010935331212447714] | 是 |
| 245 | 0.0000 ± 0.0000 | §5 | `results/quota.json` : `main[88].Gmax` | [0.0, 0.0] | 是 |
| 246 | 0.0162 ± 0.0001 | §5 | `results/quota.json` : `main[88].R_hon_b` | [0.016222466622181183, 0.00013373540540860911] | 是 |
| 247 | 0.0162 ± 0.0002 | §5 | `results/quota.json` : `main[91].R_hon_b` | [0.016242242420091327, 0.00015193809282280473] | 是 |
| 248 | 0.0070 ± 0.0016 | §5 | `results/quota.json` : `main[14].R_hon_b` | [0.0069707399001550895, 0.0016374238390795821] | 是 |
| 249 | 0.0200 ± 0.0002 | §5 | `results/quota.json` : `main[88].raid_waste_frac` | [0.019975625, 0.00015828944490058554] | 是 |
| 250 | 0.0082 ± 0.0002 | §5 | `results/stage2b.json` : `eval_rows.M4[0].R_honest` | [0.008203655647187037, 0.00016272771173547943] | 是 |
| 251 | 45 | §6 | `results/quota.json` : `quota.meta.G_definition 文字 "max over 45 strategies"` | max over 45 strategies | 是 |
| 252 | -0.0297 ± 0.0078 | §6 | `results/stage2b.json` : `F3_uninformed[56].adaptive.gain` | [-0.02965509025450029, 0.00783556914736919] | 是 |
| 253 | 3 | §6 | `results/quota.json` : `meta.K` | 3 | 是 |
| 254 | 2500 | §6 | `results/stage2.json` : `main.M3c["100000"]["0.9"].param.L` | 2500 | 是 |
| 255 | 6 | §6 | `results/stage2.json` : `main.M3c["100000"]["0.9"].param.tol` | 6 | 是 |
| 256 | 8 | §6 | `results/stage2.json` : `main.M3c["100000"]["0.99"].param.tol` | 8 | 是 |
| 257 | 2500 | §6 | `results/quota.json` : `main[88].param.L` | 2500.0 | 是 |
| 258 | 0.0021 ± 0.0025 | §6 | `results/stage2.json` : `main.M3c["100000"]["0.8"].Gmax` | [0.0021185843561695797, 0.0024900939082519776] | 是 |
| 259 | 1.0 | §6 | `results/stage2.json` : `checks.calibration_honest_flag_rate[0].kappa` | 1.0 | 是 |
| 260 | 0.1608 | §6 | `results/stage2.json` : `checks.calibration_honest_flag_rate[0].flag_per_audit` | 0.16077518090326873 | 是 |
| 261 | 0.1587 | §6 | `results/stage2.json` : `checks.calibration_honest_flag_rate[0].Phi_neg_kappa` | 0.15865525393145707 | 是 |

| 262 | z=2.57 (0.0047 ± 0.0036) | §4(5) F3 | `results/stage2b.json` : `eval_rows.M3C[tag=a0.01, r=0.95].groups.edge_un` | gain [0.004739012147929266, 0.00360728772407663], z=2.574916258535757（邊際；α=0.02 時 gain −0.0013 ± 0.0031，z=−0.79，`eval_rows.M3C[tag=a0.02, r=0.95].groups.edge_un`）。先前「無出處」的註記有誤 | 是 |

共 262 筆，不一致 0 筆。
