# Direction 2 凍結紀錄（REQ-S1-13、REQ-S1-14、T-42）

- 凍結 commit：`a72d3d731a450fe4724f0fa118943e25c39b7577`（分支 direction2-freeze，本機，未 push；本檔為其後的單獨 commit）
- 時間：2026-10-07T14:18:04+08:00
- 規格：docs/direction2-spec.md 版本 v1.3，sha256 `b0963760b23f92a7ac3438f7b6c428bdcaf6920063c3650690c338cee4eb2d2d`
- 漏洞套件：results/direction2/vuln_suite.json sha256 `1ec74ae331e59a3d6692055d5939e3eb59373eb1d4929a9b214cdde9014fada0`
  （vuln_suite_v12.json：`f338eddc255a48005b308db33e4e7a5e068a588f2979f1be080b5afc9a6b5e89`）
- 實測：results/direction2_meas.json sha256 `edb88a5aefd6a51642643eb237f3d3f3a01f9a494f652d9179652a137a170046`
- 凍結項目（S1-13）：
  - 特徵表：規格 §4（以 spec sha256 與 arbitration/rl/obs.py sha256 `27e84accc1824da355975a8c6c21224e340ac1d5b4e22476d1a2674d65b2099e` 涵蓋）
  - k = 8（arbitration/rl/obs_reconstruct.py `_K = 8`）
  - P_MAX = 64（arbitration/rl/policy.py）；P_DIAG = 300
  - ~~預算規則：規格 §5／§5.5（λ=256、T_train=20000、n_S=64、300 世代；arbitration/rl/budget.py sha256 `bf195bab453a03f20a3a8473eaacc12f4bc25bed07485822aac01fd9282765a0`）~~
  （原記錄，**有誤**，保留供稽核；更正見文末「更正紀錄」）
  - 預算規則（**更正後**，規格 v1.3 §5.5 定稿值）：**λ=256、T_train=20000、n_S=256、G_gens＝r=0.5 為 416／r=0.9 為 499、每 50 世代驗證一次**；JSON 鍵路徑：`results/direction2_meas.json: rules.S1.choice.lam`、`rules.S1.choice.T_train`；`results/direction2/measure_v2_summary.json: nS_need["0.5"].nS_from_median`（＝256）；G_gens＝⌊W_run／(T_train·t_step)⌋，`results/direction2_meas.json: rules.S1.W_run_s`（9000）、`rules.main.W_run_s`（10800）、`rules.S1.choice.t_step_ms`（1.0812）；驗證間隔 REQ-OPT-06。arbitration/rl/budget.py sha256 `bf195bab453a03f20a3a8473eaacc12f4bc25bed07485822aac01fd9282765a0`（只作交叉檢查，不在訓練時被讀取）
  - 凍結訓練計畫：results/direction2/run_plan.json（scripts/make_d2_run_plan.py 產生；訓練入口 `--plan ... --cell ...` 以此檔為準）
    RUN_PLAN_SHA256: 2ae25e33223074c987684dd68f2bfea1b4d0b1a4bba9a7f4e2da10616b8c421f

- 程式：arbitration/rl/*.py 與 scripts/run_d2_*.py 合併 sha256 `2bf0ca7a1a6c05cdc600524863097fa3577f7e7059f2dac40f253861aa71fce4`
- 測試：tests/direction2/*.py 合併 sha256 `67e4d897154b77959f1c153bae8d5375fb7d090a32edbe3c1ea5b56587c77677`；CPU 測試（-m "not gpu"）98 passed、31 deselected
- 注意：規格 S1-14 原訂另立 docs/direction2-freeze.md（含 MDE_D,plan、G_s、暖冷啟動登記等）；本檔為 pre-training 的最小憑證，上述尚未決定項目須於 S1 訓練前補登。（原記錄；已於「更正紀錄」所載的補登 commit 補上 docs/direction2-freeze.md，pilot 結果與採用的路仍待 NAIVE pilot 後記入。）

## 更正紀錄（凍結補登，早於任何正式訓練）

- **補登 commit**：`f2fe70b73b23cbfe7b6815123f68915a328e0bb1`（分支 direction2-freeze，本機，未 push）。補登晚於上述凍結 commit（a72d3d7、5165666），但**早於任何 S1／主實驗的正式訓練**（本分支沒有任何正式的 part 檔、檢查點或 pilot 結果；只有凍結前的煙霧測試 results/direction2/smoke_d2）；本節與下列檔案均為補登，照實記錄。
- **更正內容**：上方「預算規則」一行原寫 n_S=64、300 世代，已更正為規格 v1.3 §5.5 的定稿值（λ=256、T_train=20000、n_S=256、G_gens 416／499、每 50 世代驗證）；原文以刪除線保留。
- **原寫成舊值的原因**：舊值取自 MEAS v1（`results/direction2_meas.json`）的決定規則推薦（`rules.S1.choice.n_S`＝64、300 世代）。v1 的 `sd_CRN` 以 σ0=0.3 的隨機策略對量得（約 0.0104），低估訓練後期接近誠實時的變異；規格 v1.2 起改取 MEAS v2 最壞情況（`measure_v2_summary.json: nS_need["0.5"]`，σ=0.03 的 `sd_CRN` 中位數 0.021732）→ n_S=256，世代數改依 OPT-10 公式取 416／499（v1.1 的 300 與規則不一致）。凍結紀錄撰寫時誤把 v1 的推薦值抄入，規格 v1.3 本身（§5.5）一直是定稿值。**沒有任何正式訓練用到舊值**（凍結前只有管線煙霧測試 smoke_d2，依 REQ-S1-14 例外）。
- **為何不能依賴執行時讀到的檔案**：`arbitration/rl/budget.py` 的 `decide_params` 依輸入即時算出 n_S；以 v1 輸入會得到 n_S=64，以 v2 最壞情況才得到 256（交叉檢查見 run_plan.json 的 `budget_crosscheck`：v2 輸入 → λ=256、T_train=20000、n_S=256、G_gens=416／499，與定稿值一致；v1 輸入 → n_S=64）。因此訓練參數寫死在 run_plan.json，訓練入口只讀計畫檔，並在啟動時驗證計畫檔 sha256 與本檔 `RUN_PLAN_SHA256` 一致。
- **補登的檔案**：results/direction2/run_plan.json、scripts/make_d2_run_plan.py、arbitration/rl/plan.py、arbitration/rl/train.py（`--plan`／`--cell` 入口，另讓漏洞環境與 M4 錨點可由計畫檔訓練）、docs/direction2-freeze.md（規格 S1-14 要求的登記檔）、tests/direction2/test_d2_plan.py（T-76～T-80）、tdd-map 的登記、tests/direction2/test_d2_misc.py 的可攜寫法。**規格 v1.3、漏洞套件、判定規則與 budget.py 均未改動**（sha256 與上方凍結記錄相同）。
- 登記檔：docs/direction2-freeze.md（MDE、G_s 規則、暖冷啟動、通過標準、判定表版本、run_plan.json 與 vuln_suite.json 的 sha256）。

## S1 開跑前強化補登（pre-S1 hardening；早於任何 S1／主實驗訓練，本節為新增、前面各節保留）

- **補登 commit**：見本分支標題為「Direction 2: pre-S1 hardening … — before S1」的 commit（分支 direction2-freeze，本機，未 push）。凍結 commit 仍為 `a72d3d731a450fe4724f0fa118943e25c39b7577`；run_plan.json、vuln_suite.json、規格、判定規則**未改動**（RUN_PLAN_SHA256 不變）。
- **程式與測試的新合併 sha**（舊的一筆保留於上）：程式（`arbitration/rl/*.py` 與 `scripts/run_d2_*.py`，檔名排序後串接）sha256 `8ca14ac31cb61699535ab896a4bd2078a626bda3616fe206e95fa55314396b76`；測試（`tests/direction2/*.py`）sha256 `849b3224a26a62de2551aaa5bb6b9c53347e0bc9865249ac4cf73410979b252d`；CPU 測試（-m "not gpu"）112 passed（含 S123 於主 repo 路徑）。
- **入口鎖定**：`train` 沒有 `--plan` 即拒絕（舊分支移除，預算旗標不存在）；`--plan`／`--freeze`／`--root` 只接受 repo 預設路徑；啟動時 `git diff --quiet HEAD -- run_plan.json FREEZE.md` 必須通過，HEAD 必須是凍結 commit 的後代；`pilot_decision.json` 必須已 commit、只 commit 一次、sha256 等於本檔 `PILOT_DECISION_SHA256`（登記前只允許 pilot cell；登記後 chosen_start 不得更動）；`--test-mode` 只能搭配 repo 外的 root。
- **M-3 記錄登記**（REQ-MEAS-03；`results/direction2/measure_m3.json`，gpujob #143）：
  M3_SHA256: a82d91921ba7fc411e167384f64af4687e051757406930c5fd81355f729156e5
  s=0.01 的 conditional cell 只在此記錄（已 commit、sha 相符）顯示 `0.01 ≥ MDE_D,plan` 時才可執行（REQ-S1-22）。
- **單步耗時量測**：`results/direction2/measure_vuln_timing.json`（gpujob #144）sha256 `88a0dcce8ca485b242b04f173e45a532350563579a2aabc0f1999b0b008749b2`；時數估算 `results/direction2/hours_estimate.json`（`scripts/estimate_d2_hours.py`）sha256 `fceba3761bf59b37f52231a5b666929892f1a0c87715240c129d8066c2967fb3`。
- **NAIVE 錨點重用紀錄**：`s1_NAIVE_anchor` 與 `pilot_naive_cold` 的設定逐項相同（mechanism、variant、r、λ=256、T_train=20000、n_S=256、G_gens=416、val_every、σ0、obs_version、F、hidden、run_ids=[0]、algo_seeds=[9000]、train_seeds 區塊規則、val_seeds），**不再重跑**，直接引用 pilot_naive_cold 的結果（測試 seeds 評估仍依 S1 流程對該 pilot 的驗證選出檢查點做一次）。程式 `plan.REUSED_CELLS`／`check_reuse` 在啟動 `s1_NAIVE_anchor` 時逐項比對，並拒絕重跑；僅在採用冷啟動時重用 pilot_naive_cold（若決定為暖，改引用 pilot_naive_warm，同樣逐項檢查）。省下約 3.4 小時（mean）。
- **揭露（訓練時間尺度）**：訓練用 T_train=2e4，而漏洞大小（knob）是在 T=1e5 下校準的，兩者時間尺度不同；訓練時漏洞的相對得利可能與校準值不同，驗證（T_val=1e5）與測試則在 T=1e5 進行。此項只揭露，不改變任何規則或參數。
