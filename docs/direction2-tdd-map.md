# 方向二 TDD 對照表（紅燈階段；2026-10-07 依紅方審查加強；同日依 spec v1.2 補寫測試，見文末）

產生日期：2026-10-06。規格：`docs/direction2-spec.md`（v1.0-rev1＋rev2 小修）。測試：`tests/direction2/`；骨架：`arbitration/rl/`（原為全部 `raise NotImplementedError("REQ-xxx")`；2026-10-07 起 env／obs／policy 與部分 metrics 已由另一個 agent 實作，下表「目前狀態」照實記錄，未為了通過而放寬任何斷言）。

執行方式：CPU 部分 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/direction2 -m "not gpu"`；GPU 部分一律 `gpujob python3 -m pytest tests/direction2 -m gpu`（本次 job #0；既有測試回歸 job #1）。

## 紅燈結果（初版，2026-10-06）

- 現行測試函式 63 個（T-40 withdrawn 不寫）；pytest 案例 80 個（T-05 展開 14 個、T-06 展開 2 個、T-11 與 T-20/T-23 各展開 2 個）。
- 當時 CPU：55 案例，55 失敗，全部為 NotImplementedError；GPU：25 案例，24 失敗（NotImplementedError），1 通過：T-03（既有 frontier／quota 前置等價，REQ-ENV-06）。既有測試（`tests --ignore=tests/direction2`）：148 passed（gpujob #1）。

## 目前結果（加強後，2026-10-07）

- 現行測試函式 72 個；pytest 案例 100 個（CPU 75、GPU 25）。語法錯誤 0、fixture 錯誤 0（`--collect-only` 乾淨）。
- **CPU（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/direction2 -m "not gpu" -q`）：75 案例 → 24 通過、51 失敗。** 51 個失敗中 50 個是 NotImplementedError（尚未實作的模組，含「測試新定義的 API 缺席」），**1 個是斷言失敗：T-09**（`env.draw_streams` 沒有先消耗 z0，與 frontier 實際抽取順序不同；模擬路徑本身的 z 路徑與 frontier 一致）。通過的 24 個是 T-01c、T-01b×2、T-04、T-07、T-07b×2、T-10、T-11×2、T-12、T-13、T-13b、T-14、T-15×2、T-16、T-17×3、T-18、T-19、T-56 與 OBJ-06（其中 env／obs 部分對的是另一個 agent 剛實作的版本）。
- 對目前實作的通過，是在斷言**加強之後**取得的；已用 monkeypatch／shim 注入錯誤實作驗證這些斷言會失敗（見「錯誤實作 → 抓到它的測試」）。
- GPU：見「GPU 執行紀錄」。

## GPU 執行紀錄（2026-10-07）

GPU 隊列被實作 agent 的量測 job（m1、m2）佔用，本次加強後的 [gpu] 測試只完成下列部分；全部走 `gpujob`，未改 slots。

- **已通過**：T-06（2 案例）與 T-59（gpujob #6，在本次修訂後的檔案上跑，含 `forbid_frontier`、T-59 的 em=0.05 修正；JSON 重播相對 1e-9 通過）；T-05 的 r=0.5 各塊與部分 r=0.9 塊（gpujob #3–#5，由實作 agent 送出，時間點可能早於 `forbid_frontier` 的加入）。
- **排隊中（gpujob #13）**：T-01、T-02、T-03、T-08、T-36、T-39、T-41、T-50。送出時 env 已實作，T-36／T-39／T-41／T-50 預期為 NotImplementedError。
- **等價 CPU 驗證**：T-01、T-02、T-08 的測試邏輯（含 `forbid_frontier` 與新增的 16 策略全 SCOLS 集合）以 `dev="cpu"` 對目前實作跑過，3 passed（64 秒）；GPU 上的逐位元結果仍待 #13。
- **未跑**：T-05 其餘 r=0.9 塊（需依 GPU-08 分塊，每塊 ≤25 分鐘）。

## 約定（測試自行定義、規格未明寫之處）

- [gpu] 測試開頭呼叫 `cuda_required()`，沒有 CUDA 時斷言失敗（REQ-GPU-07），不 skip。
- 狀態物件欄位（`SimState`、`PublicState` 等）、模擬器回傳鍵（`final`／`snap`／`log`／`obs`／`private_log`）、`MLPPolicy.from_blocks` 等介面見 `arbitration/rl/*.py` 的 docstring；實作須符合，否則視為介面變更並同步改測試。
- 測試中的數值（`results/*.json`）一律讀檔；T-24、T-47 內有獨立重述的規則（供交叉驗證），不含方向一數字。
- NAIVE pilot／煙霧測試的 part 檔以 `cell` 開頭 `naive_pilot`／`smoke` 判定（T-42）。

## 紅方審查後的加強（2026-10-07）

測試範圍：只動 `tests/direction2/`（含 `conftest.py`）與本檔；`arbitration/rl/`、`docs/idae/`、既有測試皆未改。

### 修改與新增

- **修改的既有測試（27 個函式）**：T-01、T-02、T-05、T-06、T-08（frontier 抽離）、T-09（z0 先抽、frontier 實際順序）、T-10、T-11（預設 `T_score`）、T-12（frozen dataclass／duck-typed）、T-15、T-17（參考實作、|resid|≥6 截斷）、T-16（M4 rw=0 突襲）、T-18（刪自比自）、T-20、T-23、T-64（對偶／σ／收斂）、T-26、T-27（互換、字面量、別名）、T-29（`test["s20"]=9.9`）、T-36、T-37、T-39（hook／knob=0／CI 在測試內重算）、T-45（LB 數值）、T-48（刪自比自）、T-56（白名單＋參考實作）、T-57（新增靜態檢查）、T-59（em=0.05，刪死碼）。
- **新增 9 個函式（17 個 pytest 案例）**：T-01c、T-01b（×2）、T-07b（×2）、T-13b、T-23b（×2）、T-37b（×6）、S1-18、OBS-09、OBJ-06。pytest 案例由 80 增為 100（CPU 75、GPU 25）；現行測試函式由 63 增為 72。
- **「無測試」→「有測試」**：REQ-S1-18、REQ-OBS-09、REQ-OBJ-06 移出下方「沒有寫成測試的 REQ」（8 條 → 5 條）。

### 約定（測試新定義、規格未明寫）

- **新 API 缺席時為 NotImplementedError，不是 AttributeError**：`conftest.need(mod, name, req)`、`conftest.call_new_kw(fn, {...})`；實作補上後自動轉為真實檢查。測試新定義的 API：`s1.validate_s1_output(doc, root)`（S1-18）、`simulate_vuln(..., hook_log=True)`（T-37b）、`vulns.recompute_delta` 另回傳 `val_seeds / G_star_s[32] / G_HW_s_by_name{32 名: [32]}`（T-39，驗證集 paired G，供測試重算 Δ 與 CI）。
- **frontier 抽離**：`conftest.forbid_frontier(monkeypatch)` 把 `frontier.simulate`、`quota.simulate` 及所有已綁入 `arbitration.rl*` 模組的別名換成會 raise 的函式；期望值由 `conftest.frontier_ref`（有快取）在 monkeypatch 之前先算。
- **公開日誌慣例**：序列以 `[...,T]` 儲存、位置 t−1 為第 t 輪；`susp1[t−1]` 是第 t 輪觀察時（第 t−1 輪結束）的 susp；`log["consts"]` 的鍵必須是 `{p,h,L,kref,r,C_MIN}` 的子集（M4 另允許 `eps, rw`）。
- **f05 `since`**：t_s 取「首次觀察到自己 susp 上升的那一輪」（`since=0`），不是被標記的那一輪（見下節疑義 1）。
- **`env.simulate` 預設 `T_score`**：不帶 `T_score` ＝ 方向二模式 `T−L`（`snap` ＝ 2700 輪完整模擬的 final）；重播模式必須顯式傳 `T_score=T`。
- **knob 關閉值（T-36/T-37）**：O1、O2、O4、O6：κ=0；O3：C=inf；O5：g=inf。開啟值：κ=0.5、C=0.5、g=20。
- **hook 級日誌（T-37b）**：5 個 hook：`audit_prob`（O1）、`resid`（O3）、`cusum`（O2）、`cusum_reset`（O5）、`pause_len`（O4、O6）；每個 `out["hook_log"][hook] = dict(mask, out, <輸入>)`，輸入為該 hook 在變異環境中實際收到的值（teacher forcing）；非目標 hook 的 `out` 必須逐位元等於 base 公式，目標 hook 等於算子公式。輸入鍵：`audit_prob(win_streak, p)`、`resid(resid)`、`cusum(S, resid, kref)`、`cusum_reset(S, rounds_since_win)`、`pause_len(L, clean_streak)`。
- **S1-18 schema**：`runs[]`（漏洞：`id, kind∈{auto,designer,blind}, size, G_RL{mean,lo,hi}, D{mean,lo,hi}, MDE_D, detected`；錨點：`kind="anchor", G_RL, G_star_ref, MDE_G, passed, informational`）、`summary[]`（`size, x, n=8, p_hat, LB`，只計類內漏洞）、`figure`（檔案必須存在）。`detected` 必須等於 D 規則；`LB`＝Clopper–Pearson。
- **OBJ-06**：規格沒有指定登記檔名，測試檢查 `docs/direction2-spec.md`（REQ-OBJ-06）與 `docs/direction2-notes.md`（「期望（推論）」）兩處存在且數字一致。

### 錯誤實作 → 抓到它的測試

「實測」欄為本次以 monkeypatch／shim 實際注入該錯誤後的結果（不修改 `arbitration/rl/`）；未標「實測」者為依斷言的推演。

| 錯誤實作 | 抓到它的測試 | 驗證 |
|---|---|---|
| (a) obs 洩漏 CUSUM 或稽核 | T-13（私有狀態置換）、T-13b（累積歷史）、T-15／T-18／T-56（測試內參考實作逐特徵比對；T-56 另含 M3C 診斷跑）、T-12 | 實測 4 種洩漏（cus→d_hist、pend_a→won_hist、他人 cusum 累積、他人 u）各 ≥4 個測試失敗；T-13b 唯一能抓「他人 cusum 累積」類（T-13 單步擾動抓不到） |
| (b) 對偶取樣重用同一條雜訊 | T-20（δ 兩兩不同且各對正好 ±、世代間不重用）、T-23b、T-64 | 實測 shim：T-20×2、T-23b×2、T-64 失敗 |
| (c) 評估時未排除最後 L 輪 | T-10（預設呼叫 snap＝2700 輪 final 且 ≠ 3000 輪 final）、T-11（預設呼叫） | 實測預設 `T_score=T`：T-10、T-11×2 失敗 |
| (d) 最大值包含 H0 | T-29（全負時選最不負者、G_手寫<0）、T-28、T-49 | 實測 shim（主選取含 honest）：T-29 失敗 |
| (e) seeds 混用 | T-26（6 種互換、混合清單、邊界）、T-27（s1／vulns、字面量 5000–5031、別名匯入）、T-22、T-25 | 實測 shim：`require_split` 只檢查聯集 → T-26 失敗；掃描器漏字面量／別名 → T-27 失敗 |
| (f) 漏洞注入改到其他機制 | T-36（每個算子關閉時逐位元等於 base）、T-37（恆等＋公式）、T-37b（hook 級）、T-39（Δ 與 CI 測試內重算） | 實測假 `simulate_vuln`：非目標 hook 被改 → T-37b×6 失敗；目標 hook 沒作用 → ×6 失敗 |
| (g) G 相對 0 而非相對誠實 | T-30、T-10、T-21、T-05／T-06／T-59（逐 seed 對 honest） | 實測 `paired_gain=Up/T`：T-10、T-30 失敗 |
| (h) σ 不更新 | T-23b（σ<0.15、誤差<1e-3，附反例 mock）、T-23（σ／共變異數必須變動）、T-64 | 實測 shim：T-23b×2、T-23×2、T-64 失敗；參考 CMA 通過 |
| (i) GPU 測試 skip | T-57（AST、無 CUDA 子行程全失敗、報告檢查器、新增 (d)：`cuda_required` 必須 assert 且不含 skip；每個 [gpu] 測試第一句必為 `cuda_required()`） | 實測副本：`cuda_required` 改成 skip／某 [gpu] 測試改成 skip → (d) 抓到 |

### 規格本身可能有問題（需 commander 決定）

1. **f05 `since` 的 t_s**：「最近一次被暫停的通知輪」可讀成被標記那一輪（首次觀察 since=1/(2L)）或首次觀察到 susp 上升的那一輪（since=0）。§3.1「由 susp 上升事件記錄」支持後者；測試採後者（實作亦同）。
2. **公開日誌的 `susp1` 時點**（觀察時 vs 該輪結束）規格未寫；測試採觀察時。
3. **`env.simulate` 的預設 `T_score`**：骨架 docstring 寫「None ＝ final」，規格與 T-10 要求預設為 `T−L`；測試依規格。
4. **O3／O5 的旋鈕**：§8.2 寫 O3「旋鈕為 C（越小越弱）」，但 `resid←max(resid,−C)` 是 C 越小越強；O5 沒有 κ。「knob=0 即恆等」無法逐字成立，測試改用 C=inf、g=inf。
5. **REQ-OBJ-06 沒有指定登記檔名**（OBJ-06 測 spec＋notes）。
6. **T-56 的常數白名單**只列 M3C 的 `p,h,L,kref,r,C_MIN`；M4 診斷跑的 `eps, rw` 屬同類公開常數，測試額外允許。
7. **REQ-OBS-09 的措辭**：「僅限此特徵類別」不在 §9.3 範本內；測試接受該句、共同限制句（J3／J4）或 J7 的「特徵 k=8」。
8. **REQ-SEED-03 與 s1.py**：若 S1 的最終測試評估必須寫在 `s1.py`，會與「s1／vulns 不得引用測試 seeds」衝突，需把最終評估放在別的模組。
9. **實作端已知偏差（非規格）**：`env.draw_streams` 沒有先消耗 z0（與 frontier 的實際抽取順序不同），T-09 因此失敗；模擬路徑本身（z 路徑）與 frontier 一致。

## 對照表

| T 編號 | 測試函式 | REQ | cpu/gpu | 目前狀態 |
|---|---|---|---|---|
| T-01 | `test_d2_env.py::test_T01_parity_m3c_per_seed` | REQ-ENV-04, ENV-10, ENV-12 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-01c | `test_d2_env.py::test_T01_strategy_set_covers_all_scols` | REQ-ENV-04 | cpu | PASS（對目前實作）（新增：T-01/T-02 的策略集涵蓋 frontier SCOLS 全部欄位） |
| T-01b | `test_d2_env.py::test_T01b_const_mlp_equals_frontier_b_strategy`（M3C／M4） | REQ-ENV-04/05, ACT-01 | cpu | PASS（對目前實作）（新增：常數輸出 MLP（d=tanh(b2)）對 frontier {b:d} 手寫策略） |
| T-02 | `test_d2_env.py::test_T02_parity_m4_and_naive` | REQ-ENV-03, ENV-04 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-03 | `test_d2_env.py::test_T03_frontier_quota_equivalence` | REQ-ENV-06 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-04 | `test_d2_env.py::test_T04_zero_deviation_is_honest_bitwise` | REQ-ENV-05, ACT-01, ACT-04 | cpu | PASS（對目前實作） |
| T-05 | `test_d2_env.py::test_T05_replay_d1_json_m3c （參數化）` | REQ-ENV-07, ENV-10, MET-05, GPU-08 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-06 | `test_d2_env.py::test_T06_replay_d1_json_naive_m4 （參數化）` | REQ-ENV-07, ENV-11 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-07 | `test_d2_env.py::test_T07_event_order_and_susp_no_stack` | REQ-ENV-12, ENV-13 | cpu | PASS（對目前實作） |
| T-07b | `test_d2_env.py::test_T07b_step_state_rolling_matches_frontier`（M3C／M4） | REQ-ENV-12, ENV-04 | cpu | PASS（對目前實作）（新增：step_state 逐輪滾動 60 輪，每輪與 frontier（T=1..60）逐位元比對） |
| T-08 | `test_d2_env.py::test_T08_batch_invariance` | REQ-ENV-01, ENV-09 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-09 | `test_d2_env.py::test_T09_crn_shared_streams` | REQ-ENV-02, OPT-03 | cpu | FAIL：AssertionError（AssertionError: block 0 E (z0 must be consumed first, then E） |
| T-10 | `test_d2_env.py::test_T10_scoring_window_no_forced_honesty` | REQ-ENV-15, ENV-19, ENV-14 | cpu | PASS（對目前實作） |
| T-11 | `test_d2_env.py::test_T11_scoring_snapshot_causal （參數化）` | REQ-ENV-08, ENV-15 | cpu | PASS（對目前實作） |
| T-12 | `test_d2_obs.py::test_T12_obs_whitelist_public` | REQ-OBS-01/02/06 | cpu | PASS（對目前實作） |
| T-13 | `test_d2_obs.py::test_T13_obs_invariant_to_private_state` | REQ-OBS-01/03/06 | cpu | PASS（對目前實作） |
| T-13b | `test_d2_obs.py::test_T13b_obs_history_invariant_to_injected_private_noise` | REQ-OBS-01/06 | cpu | PASS（對目前實作）（新增：累積歷史洩漏：300 輪每步注入他人／私有雜訊，整段觀察歷史（含 EMA）逐位元不變） |
| T-14 | `test_d2_obs.py::test_T14_obs_no_absolute_time` | REQ-ENV-14, OBS-08 | cpu | PASS（對目前實作） |
| T-15 | `test_d2_obs.py::test_T15_obs_feature_table` | REQ-OBS-07 | cpu | PASS（對目前實作） |
| T-16 | `test_d2_obs.py::test_T16_obs_own_win_semantics` | REQ-OBS-01 | cpu | PASS（對目前實作） |
| T-17 | `test_d2_obs.py::test_T17_res_last_matches_mechanism_resid` | REQ-OBS-01/07 | cpu | PASS（對目前實作） |
| T-18 | `test_d2_obs.py::test_T18_info_sets_nested` | REQ-OBS-03/04/05 | cpu | PASS（對目前實作） |
| T-19 | `test_d2_obs.py::test_T19_policy_param_cap_and_action_range` | REQ-ACT-01/02/03 | cpu | PASS（對目前實作） |
| T-20 | `test_d2_opt.py::test_T20_mirrored_sampling （參數化）` | REQ-OPT-04 | cpu | FAIL：NotImplementedError（REQ-OPT-01） |
| T-21 | `test_d2_opt.py::test_T21_objective_undiscounted_average` | REQ-OPT-02/05 | cpu | FAIL：NotImplementedError（REQ-OPT-05） |
| T-22 | `test_d2_opt.py::test_T22_seed_blocks_rotation` | REQ-OPT-03, SEED-04, SEED-07（區塊內 seed 唯一部分） | cpu | PASS（v1.2：n_S 迴圈含 256；範圍檢查見 T-22b） |
| T-23 | `test_d2_opt.py::test_T23_cma_resume_determinism （參數化）` | REQ-OPT-01, GPU-04 | cpu | FAIL：NotImplementedError（REQ-OPT-01） |
| T-23b | `test_d2_opt.py::test_T23b_cma_converges_on_sphere_and_check_rejects_frozen_sigma`（n=10／55） | REQ-OPT-01 | cpu | FAIL：NotImplementedError（REQ-OPT-01）（新增：球函數 50 代 σ<0.15、誤差<1e-3；σ 不更新的 mock 必須被拒絕） |
| T-24 | `test_d2_opt.py::test_T24_param_decision_rules` | REQ-OPT-08..11 | cpu | FAIL（v1.2 同步後）：斷言不符——`budget._NS` 仍是 (16,…,128)，參考實作改為含 256 後不一致（尚未實作 n_S=256）。已用 `_NS=(16,…,256)` 暫換驗證測試本身正確（通過） |
| T-25 | `test_d2_opt.py::test_T25_val_checkpoint_selection` | REQ-OPT-06/07, SEED-05, S1-14 | cpu | PASS（v1.2：記錄世代改為 50 的倍數；freeze 例子改為 n_S=256、T_train=20000、G_gens=416 並加入 `nS_coverage_disclosure`） |
| T-26 | `test_d2_stats.py::test_T26_seed_splits_disjoint` | REQ-SEED-01 | cpu | FAIL：NotImplementedError（REQ-SEED-01） |
| T-27 | `test_d2_stats.py::test_T27_test_seeds_unreachable_from_training` | REQ-SEED-02/03 | cpu | FAIL：NotImplementedError（REQ-SEED-03） |
| T-28 | `test_d2_stats.py::test_T28_hw_set_definition` | REQ-HW-01/03/04 | cpu | FAIL：NotImplementedError（REQ-HW-01） |
| T-29 | `test_d2_stats.py::test_T29_hw_selection_val_only_untruncated` | REQ-HW-02, SEED-05, MET-02/04 | cpu | FAIL：NotImplementedError（REQ-HW-02） |
| T-30 | `test_d2_stats.py::test_T30_paired_gain_and_ci` | REQ-MET-01/02/03 | cpu | FAIL：NotImplementedError（REQ-MET-03） |
| T-31 | `test_d2_stats.py::test_T31_difference_ci_paired` | REQ-MET-04 | cpu | FAIL：NotImplementedError（REQ-MET-04） |
| T-32 | `test_d2_stats.py::test_T32_mde_formula` | REQ-MET-06 | cpu | FAIL：NotImplementedError（REQ-MET-06） |
| T-33 | `test_d2_stats.py::test_T33_found_rule_truth_table` | REQ-MET-07/08 | cpu | FAIL：NotImplementedError（REQ-MET-07） |
| T-34 | `test_d2_stats.py::test_T34_rt_and_false_punish_definitions` | REQ-MET-05 | cpu | FAIL：NotImplementedError（REQ-MET-05） |
| T-35 | `test_d2_stats.py::test_T35_number_audit_registry` | REQ-MET-09, STOP-05 | cpu | FAIL：NotImplementedError（REQ-MET-09） |
| T-36 | `test_d2_s1.py::test_T36_vuln_identity_mutation` | REQ-S1-05 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-37 | `test_d2_s1.py::test_T37_vuln_operators_inject_correctly` | REQ-S1-06/07 | cpu | FAIL：NotImplementedError（REQ-S1-06） |
| T-37b | `test_d2_s1.py::test_T37b_hook_level_isolation`（O1..O6 各一） | REQ-S1-05/06/07 | cpu | FAIL：NotImplementedError（interface）（新增：hook 級日誌：非目標 hook 逐位元等於 base、目標 hook 等於算子公式；O3 開啟用 C=2（>kref=0.5；C 越小越強，C=∞ 為恆等），並斷言 pause_len 至少被呼叫一次以防旋鈕過弱而空轉） |
| T-38 | `test_d2_s1.py::test_T38_auto_generator_deterministic` | REQ-S1-06/01 | cpu | FAIL：NotImplementedError（REQ-S1-06） |
| T-39 | `test_d2_s1.py::test_T39_vuln_known_delta_reproduced_ci` | REQ-S1-02/03 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-41 | `test_d2_s1.py::test_T41_vuln_ref_policy_public_only` | REQ-S1-03, OBS-06 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-42 | `test_d2_s1.py::test_T42_vuln_registry_frozen_and_composition` | REQ-S1-01/02/13/14 | cpu | FAIL：NotImplementedError（REQ-S1-01） |
| T-43 | `test_d2_s1.py::test_T43_blind_design_isolation_manifest` | REQ-S1-12/13/21 | cpu | FAIL：NotImplementedError（REQ-S1-12） |
| T-44 | `test_d2_s1.py::test_T44_anchor_definitions` | REQ-S1-08/09/10/11, ENV-11 | cpu | FAIL：NotImplementedError（REQ-S1-09） |
| T-45 | `test_d2_gate.py::test_T45_clopper_pearson_lower_bound` | REQ-S1-16 | cpu | FAIL：NotImplementedError（REQ-S1-16） |
| T-46 | `test_d2_gate.py::test_T46_s1_gate_rule` | REQ-S1-10/15/17, STOP-01 | cpu | FAIL：NotImplementedError（REQ-S1-17） |
| T-47 | `test_d2_gate.py::test_T47_outcome_classifier` | REQ-STOP-01..04 | cpu | FAIL：NotImplementedError（REQ-STOP-03） |
| T-48 | `test_d2_misc.py::test_T48_m4_inequality_vs_json` | REQ-M4-01/02 | cpu | FAIL：NotImplementedError（REQ-M4-02） |
| T-49 | `test_d2_misc.py::test_T49_m4_degeneracy_check` | REQ-M4-04, HW-01 | cpu | FAIL：NotImplementedError（REQ-M4-04） |
| T-50 | `test_d2_misc.py::test_T50_m4_break_region_signs` | REQ-M4-03 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-51 | `test_d2_misc.py::test_T51_measure_output_schema` | REQ-MEAS-01..04 | cpu | FAIL：NotImplementedError（REQ-MEAS-01） |
| T-52 | `test_d2_misc.py::test_T52_gpu_job_manifest` | REQ-GPU-01/02/05/06 | cpu | FAIL：NotImplementedError（REQ-GPU-06） |
| T-53 | `test_d2_misc.py::test_T53_part_files_atomic_and_idempotent` | REQ-GPU-03/04 | cpu | FAIL：NotImplementedError（REQ-GPU-03） |
| T-54 | `test_d2_misc.py::test_T54_neighbor_configs_and_no_retrain` | REQ-NBR-01/02/03 | cpu | FAIL：NotImplementedError（REQ-NBR-01） |
| T-55 | `test_d2_misc.py::test_T55_feature_ablation_groups_partition` | REQ-S1-19, ACT-03 | cpu | FAIL：NotImplementedError（REQ-S1-19） |
| T-56 | `test_d2_obs.py::test_T56_e2e_info_isolation_from_logs` | REQ-OBS-10, OBS-01/06 | cpu | PASS（對目前實作） |
| T-57 | `test_d2_misc.py::test_T57_gpu_tests_never_skip` | REQ-GPU-07 | cpu | FAIL：NotImplementedError（REQ-GPU-07） |
| T-58 | `test_d2_misc.py::test_T58_gpujob_segmentation` | REQ-GPU-02/08, OPT-06 | cpu | FAIL（v1.2 同步後）：斷言不符——`segment_estimate_s` 預設 `val_every=25`，v1.2 要 50（`2168.0 != 2108.0`）。G 值改為 416／499 |
| T-59 | `test_d2_misc.py::test_T59_replay_stage2b_m4_anchor` | REQ-ENV-21, ENV-11 | gpu | （gpu）本次未執行，見下方 GPU 小節 |
| T-60 | `test_d2_misc.py::test_T60_truncation_bias_measure` | REQ-ENV-20 | cpu | FAIL：NotImplementedError（REQ-ENV-20） |
| T-61 | `test_d2_s1.py::test_T61_naive_pilot_decision_rule` | REQ-S1-20, OPT-07, ACT-04, STOP-02 | cpu | FAIL：NotImplementedError（REQ-S1-20） |
| T-62 | `test_d2_stats.py::test_T62_s1_gate_sizes_and_run_count` | REQ-S1-10/22, MET-06, STOP-02 | cpu | FAIL：NotImplementedError（REQ-S1-22） |
| T-63 | `test_d2_s1.py::test_T63_designer_logs_saved` | REQ-S1-21/12 | cpu | FAIL：NotImplementedError（REQ-S1-21） |
| T-64 | `test_d2_opt.py::test_T64_optimizer_selection_by_dim` | REQ-OPT-01, ACT-03 | cpu | FAIL：NotImplementedError（REQ-OPT-01） |
| S1-18 | `test_d2_s1.py::test_S118_s1_output_schema` | REQ-S1-18 | cpu | FAIL：NotImplementedError（REQ-S1-18）（新增：仿 T-51，合成 s1.json 的 schema／一致性驗證） |
| OBS-09 | `test_d2_gate.py::test_OBS09_limitation_wording` | REQ-OBS-09 | cpu | FAIL：NotImplementedError（REQ-STOP-03）（新增：J3／J4／J7 措辭含特徵類別限制；N1 文字不含「未找到」） |
| OBJ-06 | `test_d2_gate.py::test_OBJ06_expectation_registered_before_work` | REQ-OBJ-06 | cpu | PASS（對目前實作）（新增：事前期望已登記於 spec 與 notes，數字一致） |

## 沒有寫成測試的 REQ（5 條；REQ-OBJ-06、REQ-OBS-09、REQ-S1-18 已於 2026-10-07 移入「有測試」）

| REQ | 理由 |
|---|---|
| REQ-OBJ-01 | 研究問題（Q-main）本身，非可驗證行為；其結果由 STOP-03 判定表（T-47）與指標（T-30～T-33）間接承接。 |
| REQ-OBJ-02 | 格（r=0.5/0.9）的角色定位，屬設計說明；功效格設定由 ENV-10、S1 流程的測試涵蓋。 |
| REQ-OBJ-03 | 次要問題清單（Q-ann、Q-nbr、Q-M4、Q-suff），無獨立行為；對應 REQ 各自有測試（T-54、T-48/49、T-55）。 |
| REQ-OBJ-04 | 延伸目標，明列不屬本規格交付。 |
| REQ-OBJ-05 | 硬性關卡與保底的策略性陳述，行為面由 T-46、T-47 涵蓋。 |

另有部分 REQ 只有「測試涵蓋其可自動驗證的一面」：REQ-S1-19（只測分組與維度，不測實際消融結果）、REQ-GPU-05（只測 manifest 規則，不測 GPU 時數預算）、REQ-S1-12／S1-21（以檔案與雜湊清單檢查，無法證明設計者真的沒有看到資訊，另有 prompt／工具紀錄的人工審查）。

以下三條原本無測試，現改為「可自動驗證的一面有測試」：REQ-OBJ-06（OBJ-06：登記檔存在且數字一致；「開工前告知使用者」仍屬流程）、REQ-OBS-09（OBS-09：J3／J4／J7 措辭；「所有對外文字都註明」仍無法自動檢查）、REQ-S1-18（S1-18：以合成文件驗 schema；真實 `s1.json` 待 W2 產生後補驗收）。

## spec v1.2 新增或修改的 REQ 與測試對應（2026-10-07，TDD 補寫）

測試範圍：只動 `tests/direction2/` 與本檔。**新增 11 個測試函式**（T-22b、T-24b、T-25b、T-66、T-67、T-68、T-69、S1-23、S1-23b、S1-24、S1-25）；新增 API 缺席時為 NotImplementedError（`conftest.need`）。

### CPU 結果（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/direction2 -m "not gpu" -q`，約 20 分鐘）

86 案例 → **74 通過、12 失敗**（跑完後修正 OBJ-06 一個測試自身的錯誤：`spec.index("REQ-OBJ-06")` 在 v1.2 先命中修訂紀錄，改為找「**REQ-OBJ-06」定義處；該案例改為通過）。12 個失敗全是尚未實作或斷言不符，沒有測試自身的錯誤：

- 既有、非 v1.2：T-51（REQ-MEAS-01 NotImplementedError）、T-55（REQ-S1-19 NotImplementedError）。
- 既有、與 v1.2 同步後轉紅：T-24（n_S 上限 256 未實作）、T-58（驗證間隔預設 25，應為 50）。
- 新增失敗（8 個）：見下表。新增通過（3 個）：T-22b、T-68、S1-23。

### 新增測試

| # | 測試 | REQ | 狀態 | 失敗原因／抓到的錯誤實作 |
|---|---|---|---|---|
| T-22b | `test_d2_opt.py::test_T22b_seed_block_range_check_SEED07` | SEED-07 | PASS | n_S∈{0,−1,1001,1024,2000} 都要 ValueError（含 g=31）；n_S=1、1000 通過且全唯一、n_S=1000 為完整排列；抓「沒檢查」「上界差一」「只檢查 g=0」 |
| T-24b | `test_d2_opt.py::test_T24b_v12_final_values_OPT09_OPT10` | OPT-09/10/11 | FAIL | 斷言不符：`decide_params` 不含 n_S=256（sd=0.021732 回傳不可行；應為 λ=256、T=20000、n_S=256、G_gens=416／499）；邊界 sd 0.0232／0.0233、0.0164／0.0165 區分 128／256／不可行；B_max 約束；G_gens 上限 1000 |
| T-25b | `test_d2_opt.py::test_T25b_validation_every_50_generations_OPT06` | OPT-06, GPU-08 | FAIL | 斷言不符：`train_segment` 預設 25（記錄在 25、50、75、100；應為 50、100）；`segment_estimate_s` 的驗證次數；`--val-every` 預設要為 50（AST） |
| T-66 | `test_d2_opt.py::test_T66_obs_version_explicit_in_training_code_OPT12` | OPT-12 | FAIL | 斷言不符：`train.py`（simulate×3、MLPPolicy×1）與 `vulns.py`（_Bank、simulate、MLPPolicy）未傳 obs_version；`s1.py`、`final_eval.py` 乾淨。掃描器自檢（無／None／僅 **kw 都算缺）；執行期：`make_sim_fn`／`make_val_fn` 須接受並轉送 obs_version 給 simulate 與每個 MLPPolicy（F=27 的 announced／oracle 不得變 public）。已用參考實作驗證執行期部分 |
| T-67 | `test_d2_opt.py::test_T67_nS_coverage_disclosure_in_freeze_OPT13` | OPT-13 | FAIL | 斷言不符：`check_freeze_registration` 不檢查 `nS_coverage_disclosure`（缺鍵、空字串、沒有 565 或沒有「無法涵蓋／not covered」都應 ValueError）；**測試新定義的鍵名**，規格只說「凍結檔必須揭露」 |
| T-68 | `test_d2_stats.py::test_T68_burst_uses_own_clock_and_is_public_HW05` | HW-05 | PASS | 3 個 burst 名稱、public=True、info 含 clock、其他策略不含；行為：偏離輪集合恰為 t mod (W+H) < W、各 seed 相同（抓以機制狀態或共用時鐘實作 burst） |
| T-69 | `test_d2_gate.py::test_T69_v12_disclosures_in_conclusion_wording_OPT13_HW05_S123` | OPT-13, HW-05, S1-23 | FAIL | 斷言不符：J3／J4／J7 措辭（`stop._COMMON_LIMIT`）沒有 (a) 565／n_S 涵蓋限制、(b) burst 時鐘＋保守、(c) designer_1 受汙染、不計入 |
| S1-23 | `test_d2_s1.py::test_S123_designer_source_and_status` | S1-23, S1-12, S1-21 | PASS | designer_2 日誌與 isolation_check.md 存在，只讀 frontier.py 與 vuln-format.md、無 Bash／Grep／Glob、prompt 0 禁字；designer_1 有 contamination.md 且讀過 spec（日誌可見）、manifest 檢查對它報違規。**不讀** blind-vulns 檔 |
| S1-23b | `test_d2_s1.py::test_S123b_registry_designer_source` | S1-23 | FAIL | NotImplementedError：`results/direction2/vuln_suite.json` 尚未產生；產生後 designer 項須 `source=="designer_2"`、D1..D4、registry 不得出現 designer_1（**測試新定義 `source` 欄位**） |
| S1-24 | `test_d2_s1.py::test_S124_blind_material_prescan` | S1-24 | FAIL | NotImplementedError：`s1.scan_blind_materials(paths, extra_words=())` 未定義（**測試新定義**，回傳 `clean/hits/files[path,sha256]`）。含獨立參考掃描：真實的 vuln-format.md 與 designer_2 prompt 0 命中；第三個非 prompt 檔含 EMA 必須被抓；六個字都測；extra_words；sha256 檔案清單 |
| S1-25 | `test_d2_s1.py::test_S125_naive_pilot_always_report_one_baseline` | S1-25, S1-20 | FAIL | NotImplementedError：`s1.naive_pilot_baseline(Ts, seeds, hw_name, *, r, dev, cfg)` 未定義（**測試新定義**）。參考值以 `env.simulate` 在 NAIVE 上算：`AdapterPolicy(b=1.0)`（v1≡1）對 honest 的逐 seed G、手寫策略 G、同 seeds 同 T_score、diff_s、`T_change`、拒絕測試／重播 seeds。已用參考實作驗證測試本身正確（通過） |

### 與 v1.2 同步修改的既有測試（只改成與規格一致，未放寬）

- **T-22**：n_S 迴圈加入 256。
- **T-24**：參考實作的 n_S 格 (16,…,128) → (16,…,256)；隨機 sd 範圍擴到 0.05（涵蓋 256 與不可行）；「n_S=128 不足」註解改為 256。
- **T-25**：驗證記錄世代 25/50/75/100 → 50/100/150/200（選第 100 世代）；freeze 例子改為 v1.2 值並含 `nS_coverage_disclosure`。
- **T-43**：設計者可讀清單改為 v1.2 的 `docs/direction2-vuln-format.md` 與 `arbitration/gpu/frontier.py`；新增「spec 本身不再可讀」斷言。
- **T-58**：G 值 300 → 416／499；`segment_estimate_s` 預設每 50 世代驗證（加 3 條斷言）；update_after_part 的 300 → 416。
- **T-61**：freeze 例子改為 n_S=256、G_gens=416（含揭露鍵）；`test_d2_gate.py` 的 `s1_description` 範例預算字串改為 v1.2 值。
- **OBJ-06**：修正 `spec.index` 命中修訂紀錄的測試自身脆弱性（見上）。
- 其他 v1.2 修改的 REQ：GPU-02／GPU-05／GPU-08 的 ≤1500 秒與預算：T-58、T-52 已涵蓋 ≤1500 秒；**REQ-GPU-05 的時數預算**仍只測 manifest 規則（無法以單元測試驗證 GPU 時數）。S1-12（可讀材料）→ T-43、S1-23；S1-20（pilot 附永遠報 1）→ S1-25；OPT-10 → T-24b；§9.3 共同限制句 → T-69。
- 檢查過未寫死舊值：T-51（`n_S=16` 為合成 schema 範例）、T-53（gen=25 為 part 檔範例）、T-55、T-64（診斷 λ=128 為最佳化器測試）。

### 無法完整自動測試的部分

- REQ-OPT-13、HW-05 的「最終報告必須揭露」：只能測措辭函式（T-69）與凍結檔鍵（T-67）；實際報告全文由人工審查。
- REQ-S1-23「designer_1 不計入統計」：registry 部分待 `vuln_suite.json`（S1-23b）；「最終報告揭露」靠 T-69。
- REQ-GPU-05 的總時數：只能由量測 JSON 與 part 檔 `wall_s` 事後核對。

## spec v1.3 的 TDD 一輪（2026-10-07：HW 細網格、val2、漏洞分類、族外 3/3、參考策略公開性、O 系列 T=1e5）

範圍：`tests/direction2/`、本檔、`arbitration/rl/{hw,seeds,s1,stop,vulns}.py`（`vulns.py` 只動 D4 參考策略與分類欄位／Δ 重算所需處）、`scripts/run_d2_vulns.py`（只修 36 個策略與 t(63) 的重算）。未動 `env/obs/obs_reconstruct/policy.py`、`results/` 既有檔、`scripts/run_quota.py`；未 commit；未讀 designer_1。

### 測試異動

- **新增 T-70…T-75**（規格列）＋兩個輔助：T-71b（對**真實** `vuln_suite.json` 的分類驗收，重新校準後才會綠）、T-73b（D4 參考策略以獨立重建器重建，逐位元相等）。新增 pytest 案例 17 個：T-70、T-71、T-71b、T-72×6（O1、O2、O3、O5、O4、O6）、T-73×5（d2_rested、d2_parole、d2_climb、streak_lie、launder）、T-73b、T-74、T-75。
- **依 v1.3 改寫的既有測試（斷言與規格一致，沒有放寬）**：T-26（四個切分＋val2）、T-27（掃描全部 `arbitration/rl` 模組、val2 被訓練端拒絕、`final_eval` 拒絕 val2）、T-28（36 個／37／32 舊清單、分組個數、`run_quota.py` 未改）、T-29（36 個候選、+H0 為 37、候選數不對拒絕）、T-38（O1–O3、O5；O4／O6 保留定義但不在套件）、T-39（val∪val2 的 64 個 seeds、36／32 名單、t(63)、size_basis、Δ_fine／Δ_coarse 皆重現、D3 為負；`D2_T39_SLICE=i:j` 分段）、T-42（11 項組成、固定 category、blind 一致、凍結檔內容含 S_HW 36、val+val2 校準紀錄、categories）、T-44（錨點屬 anchor）、T-45（(3,3)、(2,3)、(6,6)、(5,6)）、T-46（族外 3/3；其他任意變化不影響）、T-47（新措辭、J1／J2「族外 y/3，門檻 3/3」、禁舊門檻）、T-62（G_s⊆{0.01,0.02}、N=3|G_s|+6）、S1-18（新 s1.json schema）、S1-23b（designer_1 只能以 contaminated_excluded 出現）。連帶的 fixture：T-25、T-61（`conftest.make_freeze`）、T-67（加嚴：只有 565 或只有「covered」都拒絕）、OBS-09。
- 對規格措辭的一處調整：§9.3 範本含「也不推論對其他漏洞的檢測率」，而 T-47 禁止「檢測率」，實作改寫為「也不推論其他漏洞被找回的比例」。

### 新定義的 API（由測試固定）

`hw.build_s_hw()`（36）、`hw.build_s_hw_coarse()`（32）、`hw.build_s_hw_with_h0()`（37）、`seeds.splits()["val2"]`、`s1.CATEGORY_OF/CATEGORIES/CALIB_SEEDS/size_basis/counts_in_detection_statistics`、`s1.s1_gate_by_id`、`s1.check_registry_composition/check_registry_categories`、`s1.calibrate_knob_v13`、`s1.scan_blind_materials`、`s1.naive_pilot_baseline`、`stop.disclosure_text/missing_disclosures/unfilled_numbers`、`vulns.recompute_delta`（64 seeds、Δ_fine／Δ_coarse）、規則 `d2_climb` 多一個參數 `L`（機制常數，預設 2500）。

### 紅燈（改測試後、未改實作；CPU `-m "not gpu"`）

97 案例 → 22 失敗、74 通過、1 xfail。失敗原因全為「尚未實作」（NotImplementedError／AttributeError）或「斷言不符」：
- 尚未實作：T-75（`stop.missing_disclosures`）、T-71（`s1.check_registry_categories`）、T-74（`s1.calibrate_knob_v13`）、S1-24（`scan_blind_materials`）、S1-25（`naive_pilot_baseline`）、T-44（`s1.CATEGORY_OF`）、T-45（`s1.N_OUT_OF_FAMILY`）、T-26／T-70（val2 的 KeyError）。
- 斷言不符：T-28（32≠36）、T-29（不拒絕非 36 個）、T-62（G_s 仍含舊大小）、T-38（O6／O4 仍在套件）、T-46（要 8 個偵測）、T-47（J1 仍寫 6/8）、T-42（kind 與組成）、S1-18（n 必須為 8）、T-67（freeze 未檢查揭露文字）、T-73[d2_climb]（`_LClimb` 讀 `c["t"]`）、T-73b（規則沒有 `L` 參數）、S1-23b／T-71b（registry 為 v1.2，無 category、無 D3）。
- T-73[launder]：以 `xfail(strict=True)` 標記（見下「規格與範圍的衝突」）。T-27、T-73[d2_rested／d2_parole／streak_lie] 在紅燈時就通過（現行實作已滿足）。

### 綠燈（實作後）

- **CPU**：97 案例 → **94 通過、2 失敗、1 xfail**。2 個失敗是對**舊 registry**（`results/direction2/vuln_suite.json` 為 v1.2：無 category、無 D3）的驗收，只有重新校準產生 v1.3 registry 後才會轉綠：`test_S123b_registry_designer_source`、`test_T71b_real_registry_categories`。xfail：`test_T73[launder]`。
- **GPU**（全部經 `gpujob`，每個 job ≤25 分鐘）：31 案例 → **30 通過、1 失敗**。失敗：T-39（#81，預期：v1.2 registry 沒有 category；`recompute_delta` 的 64 seeds／36 名單路徑已另以 #90 驗證可用，O3@0.02 得 Δ_fine=+0.00046（最佳 dz=2.5）、Δ_coarse=+0.02014（dz=2.0），與紅方偵察相符）。
  - T-72（O 系列 T=1e5、L=2500，snap＝T−L 與 final＝T 兩種記分窗，逐位元恆等，且 base `flags>0`）：O1、O2、O3 #80（351 秒）、O5、O4、O6 #89（408 秒）→ 6/6 通過。
  - T-36、T-41（#81）；T-01、T-02、T-03、T-08、T-50（#82，417 秒）；T-06×2、T-59（#83，248 秒）；T-05 r0.5-c0..c2＋r0.9-c0..c2（#84，516 秒）、其餘 8 個（#85，817 秒）→ 全部通過。
  - 既有 148 個測試（`tests --ignore=tests/direction2`）：#86 → 148 passed。
- **D4 參考策略修正前後**（驗證 seeds 32 個、T=1e5、registry 原 knob 與參數；#87、#88）：

| 大小 | 舊（絕對 t）G* | 新（`since`＝f05）G* | 舊 Δ_fine | 新 Δ_fine | 最佳手寫（36） |
|---|---|---|---|---|---|
| 0.02（knob 0.052） | 0.2440 | 0.1917 | +0.0186 | −0.0337 | stealth m=2.0 f=0.5：0.2254 |
| 0.01（knob 0.086） | 0.2455 | 0.2357 | +0.0098 | 0.0000 | dz=2.0：0.2357 |

  原因：舊規則的振幅從「恢復那一輪」起爬升；新規則的 `since` 從「首次看到 susp 上升」起算，恢復時已約 L=2500，振幅立刻到上限。s=0.01 時 kappa·L>dz_max，策略退化成常數 dz=2.0（與手寫逐位元相同，Δ=0）。以 f05 與 L 導出的「恢復後年齡」`max(since−L, 0)`（草稿，不在 repo；#91）在原參數下 G*=0.2378、Δ_fine=+0.0124，保留大部分得利。**D4 必須用新的參考策略參數重新校準；採用哪一種相對量由 commander 決定。**

### 規格與範圍的衝突（需 commander 決定）

1. **O5 的參考規則 `launder` 用絕對 t**（`ph = (t−1) mod (W+Z)`），違反 REQ-S1-27／T-73（規格明列 O5）；本輪 `vulns.py` 只准改 D4，故 T-73[launder] 為 strict xfail。可用「自己報告歷史的連續 0 報告／非 0 報告長度」改寫成與原規則輸出完全相同的公開形式（待核准）。
2. **S1-23b**：v1.2 要求 registry 完全不出現 designer_1；v1.3 的 REQ-S1-01 把 designer_1 列為第 11 項（受汙染未計入）。測試改為「只能以 category=contaminated_excluded、kind=contaminated、無 size 出現」。
3. **`final_eval.py`** 仍是骨架（對任何輸入都拋 NotImplementedError），不在本輪可改檔案內，所以 T-27／T-70 的「`final_eval` 拒絕 val2」目前無法區分「拒絕」與「未實作」。實作 `final_eval` 時須補成 ValueError。
4. 規格 §9.3 範本的「檢測率」與 T-47 的禁字衝突（見上）。
5. 錨點 id：規格寫 `M4`，registry 與既有測試用 `M4rw0`；沿用 `M4rw0`。

## 凍結補登（2026-10-07，早於任何正式訓練）：run_plan.json 與 `--plan/--cell` 入口

測試檔 `tests/direction2/test_d2_plan.py`（5 個 CPU 測試；模擬器與 `train_segment` 以 monkeypatch 取代）。規格依據：§5.5（定稿值）、REQ-OPT-07、REQ-OPT-12、REQ-S1-10、REQ-S1-14、REQ-S1-20／22／25。

| T | 測試 | 對應 REQ | 驗證內容 |
|---|---|---|---|
| T-76 | `test_T76_plan_holds_the_final_5_5_values` | §5.5、OPT-12、S1-10 | 23 個 cell／27 個 run；λ=256、T_train=20000、n_S=256、每 50 世代驗證；G_gens 416（主實驗 r=0.9 為 499）；obs_version 明確為 public；S1 12 個 cell、主實驗每個 r 3 個 run；knob、operator、category 與 `vuln_suite.json` 相同；總時數與 §5.5 估算表相符（mean 約 94.8、p95 約 132.4 小時）；pilot 的冷／暖與「永遠報 1」評估設定 |
| T-77 | `test_T77_plan_is_reproducible_from_the_script` | S1-14 | `scripts/make_d2_run_plan.py` 重新產生的檔案與 run_plan.json 逐位元相同 |
| T-78 | `test_T78_plan_sha_matches_freeze_and_tampering_is_refused` | S1-13／14 | run_plan.json 的 sha256 ＝ FREEZE.md 的 `RUN_PLAN_SHA256`；改動計畫檔或 FREEZE 缺該行即拒絕；計畫檔內記錄的 spec、vuln_suite、budget.py sha256 與檔案相同 |
| T-79 | `test_T79_budget_py_with_v2_inputs_agrees_with_the_frozen_values` | OPT-08／09／10 | `budget.decide_params` 搭配 v2 最壞情況輸入 → (256, 20000, 256, 416／499)；搭配 v1 輸入 → n_S=64（說明為何要寫死） |
| T-80 | `test_T80_plan_entry_takes_the_cell_parameters_from_the_plan` | OPT-07、OPT-12、S1-10、S1-14 | `train --plan --cell`：參數取自計畫檔（含漏洞 operator／knob、M4 錨點）；覆寫 `--lam/--nS/--T/--val-every/--r/--mech` 被拒；改過的計畫檔被拒（sha256）；run 或世代超出計畫被拒；未登記 pilot 決定或與決定矛盾的暖／冷啟動被拒；未接線的診斷 cell 拋 NotImplementedError |

- 另：`test_d2_misc.py::test_T52_gpu_job_manifest` 的 python 路徑改為 `sys.executable`（可攜，斷言不變）。
- 限制：診斷 cell（REQ-S1-19，7 個）在計畫中登記為 `wired=false`，入口拒絕執行（T-55 仍為 NotImplementedError 的既有狀態）。`pilot_decision.json` 在 pilot 完成後才會產生；之前入口只允許 pilot 兩個 cell。

## S1 開跑前的強化（2026-10-07，早於任何 S1 訓練）：鎖定入口、conditional、暖啟動環境、錨點重用、M-3、時數

依據：notes「紅方（訓練接線與計畫）與 commander 決定」。測試檔 `tests/direction2/test_d2_hardening.py`（CPU）；`test_d2_plan.py::T-80` 改為測試模式（`--test-mode --root <repo 外的暫存 git repo>`）。

| T | 測試 | 對應 REQ | 驗證內容 |
|---|---|---|---|
| T-81 | `test_T81_entry_without_plan_or_with_other_paths_is_refused` | S1-14、OPT-07 | 沒有 `--plan` 拒絕（舊分支移除）；舊的預算旗標（`--lam/--nS/--T/--mech…`）不存在；`--plan`／`--freeze`／`--root` 不是 repo 預設路徑即拒絕（假 plan、假 FREEZE、假 root）；`--test-mode` 的 root 必須在 repo 之外（含 symlink），不得寫入 results/direction2 |
| T-82 | `test_T82_uncommitted_plan_or_freeze_and_non_descendant_head_are_refused` | S1-14 | `git diff --quiet HEAD -- plan FREEZE.md`：未暫存、已暫存未 commit、未追蹤都拒絕；HEAD 必須是凍結 commit a72d3d7 的後代 |
| T-83 | `test_T83_pilot_decision_must_be_committed_and_registered` | OPT-07、S1-14、S1-20 | pilot_decision.json 未 commit／未在 FREEZE.md 登記 `PILOT_DECISION_SHA256` ⇒ 只允許 pilot cell；sha 不符、事後修改、第二次 commit、登記被改、重複衝突登記、chosen_start 非法皆拒絕；已登記後 `--start` 與決定矛盾拒絕 |
| T-84 | `test_T84_conditional_cells_are_read_and_enforced_by_gate_sizes` | S1-22、MEAS-03、STOP-02 | 計畫中 s=0.01 的 3 個 conditional cell 被程式讀取：依已登記（`M3_SHA256`）且已 commit 的 M-3 記錄，`0.01 ≥ MDE_D,plan` 才可執行（邊界相等可），否則拒絕並說明；`MDE_D,plan > 0.02` ⇒ STOP-02；無或未登記的 M-3 記錄拒絕 |
| T-85 | `test_T85_warm_start_is_fitted_in_the_cells_own_environment`、`test_T85b_main_passes_operator_and_knob_to_the_warm_start` | OPT-07、S1-05 | `best_handwritten_on_train`／`warm_start_theta` 在該 cell 自己的環境擬合：漏洞 cell 用 `simulate_vuln(operator, knob)`，只在漏洞環境中最佳的手寫策略被挑出；NAIVE／M4 錨點用各自設定；`warm_obs` 只是觀察者（不改變模擬，operator=None 時與 `env.simulate(diagnostic=True)` 的 obs 逐位元相同）；`main()` 把 operator／knob 傳給暖啟動 |
| T-86 | `test_T86_naive_anchor_reuses_the_cold_pilot_only_if_every_setting_is_identical` | S1-08、S1-20 | `s1_NAIVE_anchor` 與 `pilot_naive_cold` 逐項相同（含 algo seed、run_ids、train_seeds 區塊規則、val seeds）才允許重用，入口拒絕重跑；任何一項不同即不可重用；路線須為已登記的路線 |
| T-87 | `test_T87_m3_record_schema_and_computation` | MEAS-03、MET-06、S1-22 | M-3 的 σ̂（ddof=1）、MDE=κσ/√32、三對取中位數、輸出欄位符合 `validate_m3`；只接受驗證 seeds |
| T-88 | `test_T88_hours_estimate_uses_measured_vulnerability_step_time` | OPT-10、GPU-05 | 時數估算：規格 5.5 公式搭配實測的 D4／D2 單步耗時；不計 7 個 diag；NAIVE 錨點重用不計；s=0.01 只在 G_s 內計入 |
