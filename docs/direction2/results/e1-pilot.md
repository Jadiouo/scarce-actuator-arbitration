# E1 pilot 結果分析（探索性；2026-10-15）

來源簡寫：設計＝`docs/direction2/design/e1-pilot.md`；sel＝`results/direction2/e1_pilot/selection.json`；cv＝`.../cross_validation.json`（`rows[].learned_G_per_seed`，9 攻擊者×32 seeds；`fixed_G_per_seed`，39 固定策略×32 seeds）；man＝`.../manifest.json`；st＝`.../status.json`；pre＝`.../preflight.json`。表中數字由上述欄位以 CPU 計算，未重跑任何評估。

## 1. 協定與 seeds

設計文件：M3C 規則、r=0.5、float64、p=0.1 固定；防守方以 full CMA-ES 調 h∈[4,8]、κ_ref∈[0.35,0.65]、L∈[1000,5000]（HW 與 pooled 目標各 2 個 optimizer seeds，λ=8、12 代，man `defender_lam/defender_gens`）。誠實約束：R/T 增加 ≤0.002、平均不合格比例增加 ≤0.01，可行者再最小化 worst G。選出的原設定／hw／pooled 各做 2 條攻擊再適應（warm start、sep-CMA-ES、λ=64、120 代、n_S=32，man `attacker_lam/attacker_gens/attacker_seeds`），再與 3 個原攻擊 checkpoint 交叉放到三個防守。T_train=20000、T_val=100000、共同記分窗 T−5000（man `T_train/T_val/tail_exclusion`）。

- train seeds 0–999（design；`scripts/e1_pilot.py::require_seeds` 的 train 允許集合 range(1000)）；validation 61000000–61000031，共 32 個（man `validation_seeds`，首尾已核對）。
- **沒有 test**：`require_seeds` 在 stage 非 train／validation 時允許集合為空，會拒絕；cv 標註 `split: "selection validation; exploratory only"`。validation 同時用於選模與本交叉表。
- 完成狀態：st `stage=complete`。

## 2. 防守選取結果（sel `selected.*`，validation seeds）

| 設定 | h (tol) | κ_ref | L | worst_G | 誠實 regret 變化 | 不合格比例變化 | 可行 |
|---|---|---|---|---|---|---|---|
| 原設定 | 6.0 | 0.5 | 2500 | hw 池 0.02145／pooled 池 0.03532（`all_candidates[0].worst_G`） | 0（基準） | 0（基準） | 是 |
| hw（hw_run0/gen6） | 7.466 | 0.4618 | 5000 | 0.01749 | -0.00393 | -0.01600 | 是 |
| pooled（pooled_run1/gen6） | 6.770 | 0.4561 | 4479 | 0.01938 | +0.00179 | +0.00653 | 是 |

欄位：`cfg.tol/kref/L`、`worst_G`、`delta_honest_regret`、`delta_honest_alpha`、`feasible`。注意 `worst_G` 是對**未再適應的固定池**（hw：36 個手寫策略；pooled：36＋3 個 frozen）取最大平均 G（`scripts/e1_pilot.py::objective`），不含再適應攻擊者。每目標僅 9 個候選（2 runs×4 個驗證點，gen 3、6、9、12，＋fallback）可選，故 worst_G 的改善含選擇偏誤。

## 3. 交叉驗證（cv；validation 32 seeds）

平均 G 與 95% CI：函式 `arbitration.rl.metrics.mean_ci(x, crit="t31")`，t_{0.975,31}=2.0395、SD(ddof=1)/√32。

| 攻擊者 | 原設定 | hw | pooled |
|---|---|---|---|
| frozen_run0 | 0.0353 [0.0301,0.0405] | 0.0318 [0.0248,0.0388] | 0.0194 [0.0114,0.0273] |
| frozen_run1 | 0.0345 [0.0297,0.0392] | 0.0297 [0.0222,0.0372] | 0.0175 [0.0100,0.0250] |
| frozen_run2 | 0.0334 [0.0281,0.0386] | 0.0274 [0.0200,0.0348] | 0.0181 [0.0108,0.0254] |
| original_adapt0 | 0.0356 [0.0304,0.0408] | 0.0315 [0.0246,0.0383] | 0.0211 [0.0139,0.0282] |
| original_adapt1 | 0.0356 [0.0321,0.0391] | 0.0337 [0.0275,0.0399] | 0.0229 [0.0164,0.0294] |
| hw_adapt0 | 0.0350 [0.0299,0.0402] | 0.0325 [0.0251,0.0399] | 0.0192 [0.0113,0.0270] |
| hw_adapt1 | 0.0343 [0.0305,0.0381] | 0.0344 [0.0283,0.0405] | 0.0213 [0.0154,0.0272] |
| pooled_adapt0 | 0.0363 [0.0326,0.0399] | 0.0356 [0.0288,0.0424] | 0.0276 [0.0219,0.0334] |
| pooled_adapt1 | 0.0327 [0.0297,0.0357] | 0.0321 [0.0269,0.0373] | 0.0261 [0.0195,0.0326] |
| **9 個學習型攻擊者最大值** | 0.0363（pooled_adapt0） | 0.0356（pooled_adapt0） | 0.0276（pooled_adapt0） |
| 39 個固定策略最大值 | 0.0353（frozen_run0） | 0.0318（frozen_run0） | 0.0194（frozen_run0） |

「最大值」取自 9（或 39）個平均的 argmax，有**贏家詛咒**：最大值的點估計偏高，其 CI 並未對選取作校正，列中 CI 只對固定的某一攻擊者有效。三個防守的最大值產生者都是同一個攻擊者 pooled_adapt0，故下面的配對比較同攻擊者、配對 seeds（G 已各自減去同防守的 honest 效用，seed 相同）。

配對差 D_s = G_原 − G_def（正值＝防守降低 G），CI 用 `mean_ci(D, "t31")`（與 `difference_ci` 同式）：

| 攻擊者 | 原設定−hw | 原設定−pooled |
|---|---|---|
| frozen_run0 | 0.0035 [-0.0014,0.0084] | 0.0159 [0.0114,0.0204] |
| frozen_run1 | 0.0048 [-0.0021,0.0118] | 0.0170 [0.0117,0.0222] |
| frozen_run2 | 0.0060 [0.0004,0.0115] | 0.0152 [0.0107,0.0198] |
| original_adapt0 | 0.0041 [-0.0012,0.0094] | 0.0145 [0.0101,0.0190] |
| original_adapt1 | 0.0019 [-0.0041,0.0078] | 0.0127 [0.0079,0.0174] |
| hw_adapt0 | 0.0026 [-0.0031,0.0082] | 0.0159 [0.0111,0.0207] |
| hw_adapt1 | -0.0001 [-0.0055,0.0053] | 0.0130 [0.0085,0.0175] |
| **pooled_adapt0（最大 G 者）** | 0.0006 [-0.0046,0.0059] | **0.0086 [0.0047,0.0125]** |
| pooled_adapt1 | 0.0006 [-0.0040,0.0052] | 0.0067 [0.0013,0.0121] |

九個比較未做多重比較校正。

## 4. 依協定門檻判讀（全部為探索性）

協定門檻（設計「尺與否證條件」）：再適應或交叉攻擊後 G 降幅 <0.005，記為修補效果薄弱訊號；誠實約束 regret ≤+0.002、不合格比例 ≤+0.01。

- **hw：薄弱。** 最大 G 由 0.0363 降到 0.0356，降幅 0.0006（<0.005）；對 pooled_adapt0／1 的配對差 CI 皆含 0。hw 對未再適應的 frozen_run2 有降幅 0.0060，其餘配對差 CI 皆含 0。hw 的誠實成本為負（regret -0.0039、不合格 -0.016），在上限內。
- **pooled：未被判為薄弱，但降幅隨攻擊者適應而縮小。** 最大 G 降幅 0.0086（CI [0.0047,0.0125]）>0.005；對未再適應者降幅 0.015–0.017，對 original／hw 適應者 0.013–0.016，對 pooled 適應者縮到 0.0086／0.0067。pooled_adapt1 的降幅 0.0067，CI 下界 0.0013 低於 0.005，點估計過線但 CI 不排除薄弱。（推論）降幅隨「攻擊者對該防守的再適應」而縮小，多一輪適應是否還會縮、是否縮至 0.005 以下，本 pilot 無法回答。
- 尚存的最大 G 0.0276 仍遠大於 0；「降到門檻以下」的語意若是殘餘 G 而非降幅，協定未定義，需正式確認時先固定。
- **誠實代價：** pooled regret +0.00179（上限 0.002，已用 89%）、不合格比例 +0.00653（上限 0.01，已用 65%）；兩者皆為均值、單一 seed 集、沒有 CI，且 pooled 剛好被選在接近 regret 上限處（另一個 pooled 候選 pooled_run0 gen6 為 -0.0060 regret 但 worst_G 0.0486，見 sel `pooled.all_candidates[2]`）。協定本來就只做均值約束，不能宣稱非劣性。regret 是否真的在上限內，在 validation 重抽時可能越線（推論）。
- pooled 的 L=4479 與 tol=6.77 距邊界（L 上限 5000）不遠；hw 的 L 在上限 5000（sel `hw.cfg.L`），搜尋空間被邊界截斷，可能還有更好值（推論）。
- 學習型攻擊者在三個防守下的最大值都不超過固定策略最大值很多（原設定 0.0363 vs 0.0353），顯示 warm-start 再適應在原設定下只多出約 0.001，但 pooled 下多出約 0.008（0.0276 vs 0.0194）。

## 5. 能說／不能說

可說（探索性，限本 pilot 的 validation 32 seeds）：
1. 在這組 9 個攻擊者與 39 個固定策略內，pooled 防守的最大平均 G 低於原設定（0.0276 vs 0.0363），配對 CI 下界 0.0047；誠實成本在協定均值上限內。
2. hw 防守對再適應攻擊者沒有可辨識的降幅。
3. 攻擊者對 pooled 再適應後，降幅由約 0.016 縮到約 0.007–0.009。

不可說：
- 不宣稱 Nash、最強攻擊、收斂；只做單次 defender→attacker 最佳回應、120 代 warm-start，非從零訓練（設計）。
- 不宣稱「修補成功」：validation 同時用於選模與評估，pooled 是 9 個候選中挑出，降幅有選擇偏誤；沒有 test；CI 未校正多重比較與最大值選取。
- 不宣稱 S1 已重新標定；**改機制後的 planted-exploit 檢測力未檢驗**（設計明載），即不知道這套攻擊者池能不能偵測到新機制下真正存在的漏洞，G 低也可能是攻擊者池不足。
- 不宣稱誠實代價「確認」在上限內（均值約束、無 CI、同 seeds 選取）。
- 不與舊 97500 記分窗的絕對值混表（設計）。

## 6. 若做正式確認：需固定的事項與 GPU 粗估

需固定（在看任何新資料前寫入新設計文件）：
1. 候選：原設定 vs pooled（sel `pooled.cfg`，tol=6.77005290146592、κ_ref=0.4560583957673218、L=4479、p=0.1）；hw 因無效果可捨棄或僅列為對照。不再重新選模。
2. 攻擊者池：固定 9 個 checkpoint 的雜湊（man `checkpoints`），並另定是否對兩個防守各以新 optimizer seeds 重新做 warm-start 再適應（訓練僅用 train seeds 0–999）。
3. 主要比較：同攻擊者、同 seeds 的配對差 D_s，用 `mean_ci(.., "t31")` 或預先指定的 CI；主攻擊者定為「對各防守**各自再適應後**最佳者」，避免事後取最大；明定多重比較處理。
4. 效應判準：預先寫明例如「配對差 CI 下界 >0.005」，並區分降幅與殘餘 G；誠實成本用配對 CI 上界對 0.002／0.01 做非劣性，不只看均值。
5. 全新 test seeds：不得使用 41000000–41000063（舊 holdout）與 61000000–61000031（本 pilot validation），也不得與其 generator offsets 重疊；需先做 seed 稽核並擴充 `require_seeds` 的 test 階段（現為拒絕）。seeds 數量需以功效計算決定（`arbitration.rl.metrics.kappa()` 已有 32 seeds 的功效參數）。
6. planted-exploit 檢測力：在新機制下補做檢測，否則「低 G」解讀受限。

GPU 粗估（算法與假設：以 st 的 `finished_unix − started_unix`＝48133.5 s≈13.37 h 為總時間；pre 的預估為 14.84 h；攻擊階段取 git 提交時間：10-14 04:15 凍結防守→07:00 結束≈2.75 h／6 條≈0.46 h／條，含最後交叉驗證；防守訓練≈總時間−攻擊≈10.6 h（推論）。提交時間為本機時區，只取差值）：
- A 最小版（固定 9 個 checkpoint 只做新 seeds 評估）：交叉驗證在 pilot 末端只佔數分鐘（06:57→07:00 提交，推論）；新 seeds 若為 128 個（4 倍）、2 個防守，約 <0.5 GPU 小時（線性外推，推論）。
- B 含再適應版：每防守 4 條新 optimizer seeds 再適應＝8 條×0.46 h≈3.7 h，加 A≈4 h。
- C 再加一輪防守重訓（對再適應後的攻擊者做 pooled 最佳回應，驗證降幅是否持續）：+≈10.6 h，合計≈15 h。
- 以上為學校單 GPU 時間；依 `gpu-center` 規則，不用本機 GPU。建議先做 A／B，C 視 B 結果。
