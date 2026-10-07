# 方向二凍結登記檔（REQ-S1-14）

本檔是規格 v1.3 REQ-S1-14 / REQ-OPT-07 / REQ-S1-20 / REQ-S1-22 要求的登記檔，於**凍結補登 commit**（見 `results/direction2/FREEZE.md` 的「更正紀錄」）補上；晚於凍結 commit（a72d3d7、5165666），但**早於任何正式訓練**。尚未決定的項目明寫「待 …」及其規則，不預填數字。

## 1. 憑證（sha256）

| 項目 | 檔案 | sha256 |
|---|---|---|
| 凍結訓練計畫 | `results/direction2/run_plan.json` | `2ae25e33223074c987684dd68f2bfea1b4d0b1a4bba9a7f4e2da10616b8c421f` |
| 漏洞套件 | `results/direction2/vuln_suite.json` | `1ec74ae331e59a3d6692055d5939e3eb59373eb1d4929a9b214cdde9014fada0` |
| 規格 v1.3 | `docs/direction2-spec.md` | `b0963760b23f92a7ac3438f7b6c428bdcaf6920063c3650690c338cee4eb2d2d` |
| 判定表程式（`classify`） | `arbitration/rl/stop.py` | `59895f67d477f5c4fbeee008e6c932d5ff01d38e5a265cfd7c4735a8e8830e5e` |
| 預算規則（只作交叉檢查） | `arbitration/rl/budget.py` | `bf195bab453a03f20a3a8473eaacc12f4bc25bed07485822aac01fd9282765a0` |

run_plan.json 由 `scripts/make_d2_run_plan.py` 產生（可重現，測試 T-77 逐位元比對）；訓練入口 `python -m arbitration.rl.train --plan results/direction2/run_plan.json --cell <name>` 啟動時驗證其 sha256 等於 FREEZE.md 的 `RUN_PLAN_SHA256`，且參數只取自計畫檔（不讀 budget.py、不讀量測檔）。

## 2. 預算（規格 §5.5 定稿，已更正 FREEZE.md 的舊值）

λ=256、T_train=20000、n_S=256、G_gens＝r=0.5 為 416／r=0.9 為 499（S1、pilot、診斷、M4 錨點用 416）、每 50 世代驗證一次（T_val=1e5、32 個驗證 seeds）、σ0=0.3、patience 無（規格只定「取 G_val 最大的檢查點」）、obs_version="public"（REQ-OPT-12，每個 cell 明確寫入）。鍵路徑見 FREEZE.md。每個 run 預估：mean 3.436 小時（r=0.9 為 4.099）、p95 4.797 小時（r=0.9 為 5.73）；run_plan.json 總計 23 個 cell、27 個 run、mean 約 94.8 小時、p95 約 132.3 小時。

## 3. MDE

規格 REQ-OPT-09／OPT-11 的規劃用估計 `MDE_est`（由方向一 `results/quota.json` 的半寬換算，κ=2.8922，REQ-MET-06）：

| r | MDE_est | 出處與算式 |
|---|---|---|
| 0.5 | **0.0058** | `results/quota.json: main[12].Gmax_uninformed[1]`＝0.0039011376（半寬）；0.0039011376/1.96×2.8922＝0.005757 |
| 0.9 | **0.0016** | `results/quota.json: main[14].Gmax_uninformed[1]`＝0.0010604507（半寬）；0.0010604507/1.96×2.8922＝0.001565 |

`MDE_D,plan`（REQ-MEAS-03 / M-3，以代理策略對在驗證 seeds 量測）**尚未量測**（`results/direction2/` 內沒有 M-3 檔）：**待 M-3 量測後依規則填入**，不得以上表的估計代替。r=0.9 在 n_S≤256 下功效不足（需 n_S≈2734），只作描述性報告（REQ-OPT-11）。

## 4. 閘門大小集合 G_s

定義（REQ-S1-22）：`G_s = { s ∈ {0.01, 0.02} : s ≥ MDE_D,plan }`；`0.02 ∉ G_s`（`MDE_D,plan > 0.02`）⇒ 觸發 REQ-STOP-02，S1 不進行。**G_s 待 M-3 量測 MDE_D,plan 後依此規則填入。** 規劃用估計 MDE_est(r=0.5)=0.0058 ⇒ G_s={0.01, 0.02}，run 數 N=3·|G_s|+6＝12；因此 run_plan.json 登記 12 個 S1 run，其中 s=0.01 的 3 個標為 `conditional`（只在 0.01∈G_s 時跑）。族內與盲區固定只跑 s=0.02。

## 5. 暖啟動與冷啟動（REQ-OPT-07，兩條路都登記）

- **冷啟動**：θ=0，σ0=0.3，sep-CMA-ES（n=55>50）。
- **暖啟動**：以最小平方回歸把 MLP（F=25、隱藏 2）初始化成「在訓練 seeds 上 G 最大的 S_HW 手寫策略」的輸出：`train.fit_lstsq`（ridge=1e-4）、`warm_T=4096`、`warm_seeds=16`（訓練 seeds 的前 16 個）。參數同樣寫入 run_plan.json 的 `start_registration.warm`。
- **採用哪一條**：待 NAIVE 學習性 pilot（REQ-S1-20）決定：冷啟動 pilot（`pilot_naive_cold`）驗證 G ≥ 0.8·G*_ref → 凍結「冷」；否則跑 `pilot_naive_warm`，達標 → 「暖」；仍未達標 → STOP-02。決定寫入 `results/direction2/pilot_decision.json`（`chosen_start`）；在該檔存在前，訓練入口拒絕啟動 pilot 以外的任何 cell；之後全部 S1 與主實驗只准用同一條路（`--start` 與決定不符即拒絕）。pilot 結果與「永遠報 1」基準（REQ-S1-25；T∈{20000, 100000}、驗證 seeds、與手寫最佳配對比較）待 pilot 後記入本檔。pilot 只用訓練與驗證 seeds。

## 6. 通過標準與判定表

- S1 通過 ⇔ REQ-S1-17（v1.3）：**s=0.02 時族外 3 個（D1、D2、O1）全部被偵測（3/3，CP 單側 95% 下界 0.3684）**，且計入的錨點通過（NAIVE；M4 若依 S1-09 降為資訊性則只要求 NAIVE）。族內（O2、O3、D4）、D3、盲區 O5、s=0.01 只描述。找不到 → 走 J1／J2，不延長、不補跑、不放寬。
- 判定表：規格 v1.3 §9.2（J1–J8，共 8 列，窮舉且互斥），實作 `arbitration/rl/stop.py: classify`（sha256 見上表）。

## 7. 其他凍結項目

- **S_HW 36 個清單**（REQ-HW-01，順序固定，`arbitration/rl/hw.py: build_s_hw`）：b=0.05、b=0.1、b=0.2、b=0.3、b=0.5、b=1.0、dz=0.01、dz=0.02、dz=0.03、dz=0.05、dz=0.07、dz=0.1、dz=0.15、dz=0.2、dz=0.3、dz=0.5、dz=1.0、dz=2.0、dz=2.5、dz=3.0、dz=4.0、dzcrit=0.5、dzcrit=1.0、dzcrit=1.5、burst dz1 W10 H90、burst dz1 W30 H170、burst dz2 W10 H490、edge_uninf thr band=0.05、edge_uninf thr band=0.1、edge_uninf thr band=0.2、edge_uninf dz band=0.1 dz=0.1、edge_uninf dz band=0.1 dz=0.3、stealth m=1.0 f=0.5、stealth m=1.0 f=0.9、stealth m=2.0 f=0.5、stealth m=2.0 f=0.9。
- **每個漏洞的 category**（REQ-S1-26，固定）：族外 D1、D2、O1；族內 O2、O3、D4；族內已知 D3（不跑 RL）；盲區 O5；錨點 NAIVE、M4rw0；受汙染未計入 designer_1。knob 與校準紀錄（val∪val2 共 64 seeds，REQ-S1-28）見 `vuln_suite.json`（sha256 見上）。
- **n_S 涵蓋範圍的揭露**（REQ-OPT-13）：n_S=256 依「σ 三種的中位數最壞值」滿足 REQ-OPT-09，但**單一隨機策略對的最壞情況**（`sd_CRN`=0.034465）需要 n_S≈565（`measure_v2_summary.json: nS_need["0.5"].nS_formula_continuous_pair`＝564.95），**目前無法涵蓋（not covered）**；r=0.9 需 n_S 數千，只作描述。此揭露不改變任何判定規則。
- **CRN 降幅有限**（VRF 約 1–1.7）：見規格 §5.5 的 VRF 列（REQ-MEAS-02）。
- 凍結記錄、特徵表、k=8、P_MAX=64、σ0：見 `results/direction2/FREEZE.md`。
