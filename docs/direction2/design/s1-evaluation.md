# 方向二 S1 最終評估與組裝：實作說明

程式：`arbitration/rl/s1_eval.py`（邏輯）、`scripts/d2_s1_eval.py`（指令列）、`arbitration/rl/final_eval.py`（新增 `evaluate`、`already_evaluated` 兩個小函式）。測試：`tests/direction2/test_d2_s1_eval.py`（17 個 CPU＋1 個 [gpu]）。未改動凍結檔（run_plan.json、vuln_suite.json、FREEZE.md、規格、stop.py、budget.py）。

## 1. 流程與對應條文

每個 S1 run（計畫檔 group `s1` 的 12 個 cell；NAIVE 錨點依 FREEZE.md 重用 `pilot_naive_cold` 的檢查點）分兩段，輸出在 `results/direction2/s1_eval/`：

| 段 | 內容 | 條文 |
|---|---|---|
| val | 從該 run 的全部驗證記錄取 G_val 最大者（同分取較早）為最佳檢查點，與檢查點檔及最後一個 part 交叉核對；在該 cell 自己的環境（漏洞算子＋knob、M4、NAIVE p=0；`vulns.simulate_vuln`，公開觀察，T=1e5）於驗證 seeds 2000–2031 重算 RL 的逐 seed G；對 36 個 S_HW（加 H0 為 37）算驗證 G，選最大者（平手取較前）；算驗證集 D_s、`MDE_G`、`MDE_D`（κ=2.8922，`metrics.mde`，只收驗證 seeds）並寫入 `mde_frozen`、`mde_d_frozen`；與訓練時存的 G_val 比對（差 >1e-6 相對即拒絕，確保評估環境＝訓練環境） | REQ-OPT-06、REQ-HW-02、REQ-MET-06、REQ-SEED-05 |
| test | 唯一入口 `final_eval.evaluate`（測試 seeds 在 final_eval 內才產生，s1_eval 與腳本不出現任何測試 seed；T-27 掃描與新測試守門）；MDE 一律讀 val 檔；評估三個 policy_id：`rl`（G、D=G_RL−G_HW）、`hw`（被選手寫策略的 G_手寫）、`hw_plus_h0`（D^{+H0}，附帶，不入判定）；錨點只評 `rl`。原始逐 seed 增益先存 `*__test_raw.json`，再由 ledger 鎖定，中途當機可由 raw 重建，不重新模擬 | REQ-MET-04、REQ-MET-07、REQ-SEED-02/06、REQ-S1-10 |
| assemble | 彙整為 `results/direction2/s1.json`（每 run 的 G_RL、D 與 CI、MDE_D、detected；錨點 G_star_ref、MDE_G、passed、informational；彙總 (y,3,p̂,LB)、(x,6,p̂,LB)；D3 的 delta_fine／delta_coarse；`gate`；檢測力曲線圖 `results/direction2/s1_power_curve.png`），過 `validate_s1_output`，再以 `stop.classify` 得 J 判讀 | REQ-S1-11/15/16/17/18、REQ-STOP-01/03 |

偵測判定：漏洞用 D 規則（`D_lo>0` 且 `D≥MDE_D`，CI 為 t₃₁），錨點用 G 規則與 `s1.anchor_passes`（`G_RL≥0.8·G*_ref`、CI 下界>0、`G_RL≥MDE_G`）。未完成的 cell 記為未偵測／未通過，分母不變（STOP-01）。

`--graph`：RL 批次（全是 MLP）走 `graphsim`，與 eager 逐位元相同（[gpu] 測試 `test_graph_stepping_of_the_RL_batch_equals_eager_bitwise`，含漏洞環境與 M4；本機 gpujob #1 通過）。手寫批次含 adapter，沿用現有行為，一律 eager。

## 2. 歧義與選擇（供指揮官審）

沒有任何一項會改變判讀方向的歧義需要停下；以下逐項列出。

1. **「驗證集最佳手寫策略」用哪些 seeds**：只用驗證 2000–2031（32 個），不用 val∪val2（登錄檔的 `G_HW_best_val` 是 64 seeds 的校準值）。依據：REQ-SEED-08(b)、REQ-HW-02 仍只用 val。保守（val2 不進任何選擇）。
2. **手寫基準在哪個環境**：每個漏洞 cell 在自己的漏洞環境（plan 的 operator／knob，等於登錄檔的 knob）重新模擬 36 個策略，不引用登錄檔或方向一的數字（REQ-HW-03、REQ-S1-02）。
3. **honest 基準**：RL 批次用 θ=0 的 MLP（與訓練的 val_fn 相同），手寫批次用 adapter honest；兩者必須數值一致（相對 1e-12，否則中止）。實測 CPU 測試差為 0、GPU 全尺寸煙霧測試通過此檢查，D_s 的配對有效。
4. **錨點的 G\*_ref**：採規格字面的參考策略（NAIVE：dz=2.0；M4：b=0.3），數值取 FREEZE 登記的驗證集重測（NAIVE：`pilot_decision.json` 的 0.27909；M4：`measure_m4.json` 的 0.11895，CI 下界 0.1171>0 ⇒ 「counted」，不降級），評估時再重算一次核對（相對 1e-6）。若改用 pilot 的「手寫最佳 b=1.0」（0.3121）當 G\*_ref，閾值會更嚴，NAIVE 錨點結果（0.3124）仍通過；我們依規格 REQ-S1-08 與凍結檔用 dz=2.0。
5. **M4 的 S1-09 降級**：判斷依據是 W1 登記的驗證重測（上項），不是測試結果；`informational` 寫入 s1.json，降級時 M4 照實記錄但不入 `s1_gate`。
6. **MDE 的兩種用途**：`mde_frozen`＝MDE_G（來自 RL 的 G_s，用於錨點與 G 規則）；`mde_d_frozen`＝MDE_D（來自驗證集 D_s＝G_RL,s−G_HW,s，用於漏洞偵測），照 REQ-MET-06 字面；`metrics.final_eval` 已有此分流。D^{+H0} 另用自己的驗證集 MDE，只報告。
7. **最佳檢查點的認定**：用「所有驗證記錄的 argmax，同分取較早」（REQ-OPT-06 的字面），不用「最後一個」或訓練 fitness；檢查點檔的 `best` 若與 argmax 不一致就拒絕。
8. **一個 (cell, policy) 只評估一次**的粒度：`rl`、`hw`、`hw_plus_h0` 三個 policy_id 各一次；一次模擬同時供三者使用（同一批測試 seeds）。
9. **J 判讀**：S1 階段主實驗（r=0.5 三個 run）尚不存在，故以 D 不存在餵 `classify`：S1 通過 ⇒ J7＋`incomplete=True`（N1，不得寫「未找到」），未通過 ⇒ J2。J1、J3–J6、J8 需要主實驗的 D，不在此階段產生。
10. **y′（s=0.01）與缺 cell**：計畫檔有 s=0.01 的 cell 時 y′ 照計（未完成算未偵測，分母 3）；某大小完全沒有 run 時 `validate_s1_output` 不允許該大小的彙總列，所以不列該列，但 `gate.y_s001=0` 與 `not_completed` 如實記載。
11. **不計算的項目**：REQ-MET-05 的 R/T 與 α、REQ-STOP-06 揭露數字的填寫（屬最終報告腳本）不在 S1-18 的輸出清單內，未做。錨點在 s1.json 的 `size` 為 null。
12. **s=0.01 的條件 cell**：正式模式下先過 `plan.check_conditional`（M-3 記錄已 commit 且 sha 相符）。
13. **G_HW 的 CI／D 的 CI**：皆 t₃₁（32 個測試 seeds），與 REQ-MET-03／04 一致。

## 3. 執行（學校 RTX8000；本機未跑正式評估）

前置：學校機同步 direction2-freeze 分支（含 parts／ckpt 與本次 commit），工作樹乾淨（`run_plan.json`、`FREEZE.md` 與 HEAD 一致；入口會檢查）。

```bash
# 逐段送出（每個 cell：val 與 test 兩段，皆可續跑；已完成的段會跳過）
gpujob python3 scripts/d2_s1_eval.py run --graph --cells s1_NAIVE_anchor s1_M4rw0_anchor s1_O1_s0.02 s1_D1_s0.02
gpujob python3 scripts/d2_s1_eval.py run --graph --cells s1_D2_s0.02 s1_O2_s0.02 s1_O3_s0.02 s1_D4_s0.02
gpujob python3 scripts/d2_s1_eval.py run --graph --cells s1_O5_s0.02 s1_D1_s0.01 s1_D2_s0.01 s1_O1_s0.01
python3 scripts/d2_s1_eval.py assemble        # CPU，秒級；寫 results/direction2/s1.json 與圖並驗證
python3 scripts/d2_s1_eval.py status          # 看各 cell 的 val／test 是否完成
```

產物（都要 commit）：`results/direction2/s1_eval/*.json`、`results/direction2/final_eval_ledger.json`、`results/direction2/s1.json`、`results/direction2/s1_power_curve.png`。ledger 一旦寫入，同一 (cell, policy) 不能重評；若要重來必須由指揮官決定（規格要求測試 seeds 只評一次）。

## 4. 耗時估計

本機 RTX 5070 Ti 的 T=1e5 實測（gpujob #3，`--graph`）：漏洞 cell val 154 s＋test 111 s≈4.4 分；M4 錨點 80＋15 s；NAIVE 錨點 109＋20 s。全部 12 個 cell（10 個漏洞 cell＋2 個錨點）合計約 10×4.4＋3.3 ≈ 47 分（約 0.8 小時）。RTX8000 的單步速度與訓練實測（t_step≈1.4 ms）同級，保守估 **1–2 小時**（含三個 job 的啟動），每個 job（4 個 cell）約 20–40 分鐘；若要求每工作 ≤25 分鐘，改成每個 job 2 個 cell。

## 5. 驗證紀錄

- `pytest tests/direction2 -m "not gpu"`：238 passed。[gpu] 圖形對 eager 逐位元測試：1 passed（gpujob #1）。
- 煙霧測試：`--test-mode --T 2000` 對 12 個 cell 全流程（gpujob #2）與 `assemble`；`T=1e5` 對 s1_O1_s0.02、兩個錨點全流程（gpujob #3；重算的驗證 G 與訓練時的 G_val 一致，G\*_ref 重算與凍結值一致）。煙霧輸出在 `.scratch/`（git 排除），非正式結果。
