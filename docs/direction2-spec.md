# 方向二規格（SDD）：學習型對手對 M3C 的可利用度評估與檢測力標定

版本：v1.3（凍結前最後版；併入漏洞套件完成後的 commander 決定與紅方（漏洞套件）審查後的 commander 決定第 1–7 項）　日期：2026-10-07
依據：`docs/direction2-notes.md` 的「SDD 初稿與 commander 決定」、「紅方（SDD）判定與 commander 決定」（第 1–11 項與 GPU 項；衝突時以後者為準）、「使用者新增：攻防雙方學習的 2×2 消融」；v1.3 另依「漏洞套件完成與 commander 決定」與「紅方（漏洞套件）與 commander 決定」兩節（第 1–7 項與執行順序；與先前決定衝突時以紅方節為準）。
參考：`docs/direction1-summary.md`、`docs/direction1-number-audit.md`、`arbitration/gpu/frontier.py`、`arbitration/gpu/quota.py`、`results/quota.json`、`results/stage2.json`、`results/stage2b.json`。

**修訂紀錄**
- 2026-10-06；依據：上列 notes 三節（逐項處理紅方第 1–11 項與 GPU 項）。REQ／測試編號全部保留，withdrawn 者標「(withdrawn)」，修改者標「(revised)」，新增者接在各類最大號之後。
- 尾端強制誠實刪除（ENV-16/17/18 withdrawn）；改為量測截斷暫停偏差（ENV-20）。觀察仍不含時間。
- 漏洞大小改為「超過該漏洞環境中手寫最佳的增量」，偵測改用 D 規則（S1-02、S1-03、S1-04 withdrawn）；MDE 改為 80% 檢定力（非中心 t，κ=2.8922）；低於 MDE 的大小只作描述，S1 run 由 42 改為約 22（S1-10、S1-22）。
- 結果判定改為 8 列窮舉表（§9.2）；措辭改為「對本套件的描述」，不作二項推論（§9.3）。
- 策略參數上限 ≤64（MLP F→2→1＝55 參數，>50 維改用 sep-CMA）；暖／冷啟動於凍結前登記並由 NAIVE 學習性 pilot 決定（S1-20）。
- M4 錨點更正：原 0.0356 是 oracle 邊緣灌水；改用 `stage2b.json: consist_M4[12]`（r=0.99、ε=0.2、L=20、rw=0、b=0.3，+0.1209）；S1-09、S1-10 的 r 一致化。
- 手寫基準排除 H0（32 策略），含 H0 另報；O4、O5 改列特徵盲區並新增 O6 以維持類內 n=8；設計者看不到 k=8／EMA，並保存 prompt 與工具紀錄（S1-21）。
- 新增端對端資訊隔離（OBS-10、T-56）、[gpu] 測試不得 skip（GPU-07、T-57）、gpujob 分段（GPU-08、T-58）；容差改相對 1e-9；T-17、T-42、T-03 改法；新增 T-56…T-64；新增 §18 延伸階段 E1。
- 2026-10-06（rev2，commander 對 P-1～P-9 的決定與最終 4 處小修）：(1) M4 錨點重測 CI 下界 ≤0 時降為資訊性、不計入通過條件（S1-09、S1-11、S1-17、T-44、T-46、§17 P-5）；(2) §17.2 的 P-1～P-9 全部移為已決定，S1-20 門檻明寫「≥0.8·G*_ref（驗證重測，預期約 0.28）」；(3) MEAS-01 隱藏寬度 6 改 2（mlp25-2-1），T-42 的 part 檔祖先檢查排除 NAIVE pilot／煙霧測試；(4) 判定表 J7 加註「S1 通過但主實驗未完成」並於 §9.2 補處理條目。
- 2026-10-07（v1.1 定稿；依據：notes 的「測試加強」、「W1 第一批實作與 MEAS」、「TDD 完成」P-8，及 `results/direction2_meas.json`）：(1) §5.5 填入 MEAS 定稿值 λ=256、T_train=20000、n_S=64、300 世代、每個 run 約 1.8 小時（每步約 1.08 ms），並記錄「CRN 降幅有限（VRF 1.2–1.7）」；(2) REQ-OPT-11：r=0.9 在 T_train=2e4 下功效不足，只作描述；(3) 7 項規格疑義入規：REQ-OBS-11（f05 的 t_s）、REQ-OBS-12（susp1 時點）、REQ-ENV-15 revised（T_score 預設 T−L，T≤L 退回 T）、REQ-S1-05／S1-06 revised（O3／O5 恆等基準 C=∞、g=∞）、REQ-OBJ-06 revised（以 spec 與 notes 為事前登記）、REQ-OBS-10 revised（M4 的 ε、rw 為公開常數）、REQ-SEED-03 revised＋REQ-SEED-06（測試 seeds 只准在 `arbitration/rl/final_eval.py`）；(4) P-8：S1-10、GPU-05 的「有餘裕才補跑」改為「不補跑」；(5) 以 S1 run 數 × 1.8 小時重算 GPU 時程（核心合計約 66.6 小時，原規劃 99 小時）；(6) 版本號與各處「待 v1.1」字樣更新。REQ／測試編號仍全部保留。
- 2026-10-07（v1.2，凍結前最後版；依據：notes 的「spec v1.1 與世代數更正」、「MEAS v2 與 commander 決定」（6 項決定）、「訓練側實作與 NAIVE smoke」、「盲設計資訊洩漏與重做」、「designer_2 完成，隔離成立」，及 `results/direction2/measure_v2_summary.json`）：(1) §5.5：n_S 上限 256、所需 n_S 依最壞中位數取 256（單一對最壞需約 565，不涵蓋，須揭露，REQ-OPT-13）；世代數依 OPT-10 為 r=0.5 416、r=0.9 499；驗證每 50 世代一次（OPT-06）；每個 run 預算約 4 小時，mean／p95 兩版重算各階段總時數（REQ-GPU-05）；(2) REQ-GPU-02／GPU-08：[gpu] 測試與訓練 job 一律每段 ≤25 分鐘（#26 曾超過）；(3) 新增 REQ-OPT-12（訓練程式碼必須明確傳入 obs_version）、REQ-OPT-13（n_S 涵蓋範圍揭露）、REQ-SEED-07（seed_block 的範圍與唯一性）、REQ-HW-05（burst 時鐘揭露）、REQ-S1-23（設計者漏洞來源與狀態）、REQ-S1-24（盲設計者材料禁字掃描）、REQ-S1-25（NAIVE pilot「永遠報 1」基準）；(4) REQ-S1-12、S1-20、OPT-10、OPT-06 修訂；(5) §9.3 共同限制句、§15、§17 同步更新。REQ／測試編號仍全部保留。
- 2026-10-07（v1.3，凍結前最後版；依據：notes 的「漏洞套件完成與 commander 決定」與「紅方（漏洞套件）與 commander 決定」第 1–7 項與執行順序，及 `results/direction2/vuln_suite.json`）：(1) **HW**（REQ-HW-01／02／06、MET-04）：手寫基準加入細網格 dz∈{2.5,3,4}、b=1，S_HW 由 32 增為 **36** 個（含 H0 為 37）；主實驗 G_手寫、漏洞增量 Δ 的 G_HW 都用新清單；T-28 的「32 個」改為 36；(2) **套件組成**（REQ-S1-01、S1-26 新增、S1-06、S1-07）：每個漏洞有 `category` 欄位：族外 D1、D2、O1；族內 O2、O3、D4；族內已知 D3；盲區 O5；錨點 NAIVE、M4；受汙染未計入 designer_1；O4、O6（無法校準）與 B1、B2（未實作）移出套件；分類依據為細網格下 G_HW/G* 比值與 Δ；(3) **S1 通過標準**（REQ-S1-10、S1-15、S1-16、S1-17、S1-18、S1-22、STOP-01、§9.2 J1–J8 的 S1 欄與備註、§9.3）：s=0.02 時族外 3 個必須全部找回（3/3；CP 單側 95% 下界 0.05^{1/3}≈0.37）並兩個（計入的）錨點通過；族內只描述；舊標準（8 個找回 ≥6；中間版 6 個找回 ≥5）作廢；S1 run 數由 22 降為 **12**（範圍 9–12），§5.5 與 REQ-GPU-05 重算（核心 27 個 run，mean 約 94.8／p95 約 132.4 小時）；(4) **措辭**：「6 個類內漏洞中 RL 找回 x 個；其中族外 3 個找回 y 個」，不得外推；盲區措辭限定為對單一範例 O5 的描述；(5) **seeds**（REQ-SEED-01／02／08）：新增 val2＝3000–3031，與 train、val、test 兩兩互斥，只能用於校準與選擇，永遠不得作為測試集；`require_split` 涵蓋 val2；(6) **校準**（REQ-S1-02、S1-28 新增）：knob 以 val∪val2 合併後選擇，判定區間沿用 [0.8s,1.2s]，並要求合併後 CI 下界 >0；登錄檔只含 s∈{0.01,0.02}（原 {0.003,0.005} 未校準，與登錄檔現況一致）；(7) **參考策略的公開性**（REQ-S1-03、S1-27 新增、OBS-08）：只能使用公開日誌欄位、不得用絕對時間；D4 改用 f05 可重建的相對量；以靜態與執行期檢查驗證；(8) **恆等性**（REQ-S1-05）：O 系列亦在 T=1e5 驗證；(9) **揭露清單**（REQ-STOP-06 新增、§9.3）；(10) §14 同步（既有測試的斷言改寫、新增 T-70…T-75 的規格列，尚無測試檔）、§15、§17（R-1…R-7、V-1…V-6）。commander 在「漏洞套件完成」節的決定 1（類內 n 由 8 降為 6、通過標準 ≥5/6）已被紅方節決定 2 取代。REQ／測試編號仍全部保留。
- 2026-10-07（v1.3 凍結前小修；依據：notes 的「v1.3 TDD 完成與 commander 決定」第 2、3 項）：REQ-S1-27 的適用範圍改為只適用於計入統計的漏洞（族外、族內）：參考策略不得使用時間、須在 RL 觀察範圍內可實作；盲區 O5 的參考策略可使用自身時鐘、手寫基準可使用自身時鐘（須揭露）；D4 參考策略定為恢復後年齡 `max(since−L, 0)`；`final_eval.py` 實作為只接受測試 seeds（拿到 val、val2、train 等一律拋錯）。其餘條文不變。

本文件是後續 TDD 與實作的唯一依據。每條需求有編號 `REQ-<區>-<nn>`，第 14 節的每個測試都對應到現行（非 withdrawn）的 REQ。「必須」＝可驗證；「待量測」＝第 11 節量測後填入第 5.5 節的表（v1.1 填入、v1.2 依 MEAS v2 更新）；「待決定」＝需要 commander 決定，列於第 17 節，規格預設值只是讓測試可以先寫，不代表已決定。已由 commander 決定者在第 17 節標「已決定」。

記號：K=3 位 agent，偏離者為 agent 1（frontier.py 的欄位 index 1），其餘誠實；τ=1；z 空間相關 r=ρ^τ（τ=1 時 r=ρ）；C_MIN=c=0.2；T 為總輪數；L 為暫停長度；h 為 CUSUM 門檻（frontier 的欄位名為 `tol`）。

---

## 1. 目的與問題

### 1.1 動機
方向一的每個「通過」都只是「在手寫策略集下通過」（summary §6：oracle 策略主導、自適應偏弱）。方向二用學習出來的對手取代手寫策略集，並用「人造漏洞檢測力標定」說明這個紅隊抓得到什麼、抓不到什麼。

### 1.2 主問題與次要問題

- **REQ-OBJ-01（Q-main）**：在 M3C（派遣時 CUSUM，方向一選定參數，見 §2.3）上，使用**公開版觀察資訊集**（§3.1）、策略參數 ≤ 64 個（§4.2）的學習型對手，能否找到「最佳手寫無資訊策略」（G_手寫，§6.3 定義）找不到的得利？
  - 主指標：測試 seeds 上的未截斷 G_RL、G_手寫，以及 D = G_RL − G_手寫（paired，95% CI，§7）。
- **REQ-OBJ-02（格的角色）**：
  - **r=0.5 為功效主格**：方向一的無資訊手寫得利在此格明顯大於 0（主表設定下 0.0213 ± 0.0039，`quota.json: main[12].Gmax_uninformed`）。所有檢測力標定（S1）、特徵充分性檢查都在 r=0.5 做（唯一例外：M4 錨點固定在 r=0.99，REQ-S1-09）。
  - **r=0.9 為探索性**：方向一的無資訊手寫得利在此格約 0.0020 ± 0.0011（`main[14].Gmax_uninformed`），已接近雜訊。把它當重點，等於預先把測試場設在 RL 最難發揮的位置（commander 在第 3 輪承認的偏誤），所以改為探索性；此格的任何結論只能寫「探索性、未經 S1 標定」。
- **REQ-OBJ-03（次要問題，結果不影響主結論的成立與否）**：
  - Q-ann：公告版（§3.2，含真實 CUSUM 值 S）與公開版的 G 差距，解讀為「機制私有狀態的資訊價值」上界對照。
  - Q-nbr：學好的策略在鄰近參數上的遷移（§13）。
  - Q-M4：M4 的逐次不等式附錄與 G=0.0000 退化檢查（§10）。
  - Q-suff：特徵充分性（去一組特徵消融、64 步歷史對照），只在 r=0.5 做一次，不是硬性關卡（§8.7）。
- **REQ-OBJ-04（延伸目標，不屬於本規格的交付）**：oracle 版（§3.3，用公開版解暖啟動、策略類別嵌套）、QM、r=0.99、GRU、共謀／多偏離者、接回環形空間模擬（`sim_torch.py`）、機制共同演化。不得因延伸目標而延後 W1–W3 的交付。
- **REQ-OBJ-05（硬性關卡與保底）**：S1（§8）是硬性關卡。W2 結束時未通過，走 §9.2 判定表的 J1／J2（結果 (iii)）。不論結果為何，「可重複的 RL 紅隊流程＋檢測力曲線」是保底產出。
- **REQ-OBJ-06（期望的事前登記）(revised)**：最可能的結果為 (ii) 或 (iii)（合計預期 >80%），(i) 預期 <15%（第 3 輪白方，屬推論，非實測）。此期望必須在開工前告知使用者（記憶中的「explain before big deliverables」規則）。
  - **登記方式（commander，v1.1）**：以本規格（本條）與 `docs/direction2-notes.md`（「期望（推論）」）本身作為事前登記，不另開新檔；兩處的數字必須一致（OBJ-06 測試檢查）。

### 1.3 範圍外
S1 只做 M3C（判定 8）；M3C 以外的機制只有兩個錨點（NAIVE、M4 rw=0，§8.3）會用到。不改方向一的機制參數、不針對 RL 重新調參（§13）。

---

## 2. 環境規格

### 2.1 「策略在迴圈內」的批次模擬器

frontier.simulate 的偏離者策略是由欄位（b、dz、em…）在迴圈內即時算出，沒有策略回呼，所以不能直接放入 MLP。需新增模擬器（建議模組名 `arbitration/gpu/policy_sim.py`，不修改 frontier.py／quota.py）。

- **REQ-ENV-01（批次維度）**：batch 元素以 `[cfg, pol, seed]` 展平（與 frontier 的 `[param, r, strategy, seed]` 同為 row-major；`pol` 取代 strategy 軸）。訓練時 `cfg` 通常為 1，`pol` 為 ES 族群（λ 個候選，含鏡像對），`seed` 為該世代的環境 seeds，即 batch 維度為「ES 族群×seeds」。所有張量 float64、device cuda（測試可用 cpu，但與 JSON 比對的測試必須 cuda，因為 JSON 由 cuda RNG 產生）。
- **REQ-ENV-02（隨機數與 CRN）**：沿用 frontier 的三組 generator：`seed`、`seed+10^6`、`seed+2·10^6`，每 `blk=1024` 輪一批抽樣（blk 寫死為 1024，因為抽樣順序依 blk 而定），元素以 `sid`（seed 索引）取用。結果：同 seed 的所有 `pol` 元素看到完全相同的型別路徑、稽核抽籤與突襲抽籤（CRN）。任何元素的結果不得依賴同一批次的其他元素。
- **REQ-ENV-03（機制）**：支援 `mech ∈ {M3C, M4}`。NAIVE 以 `M3C` 且 `p=0` 表示（無稽核時分配分支與 quota.py 的 NAIVE 相同；此等價由 REQ-ENV-06 驗證）。M0、M3T、M3G、M3W、QM、QT、QE 不在範圍內。
- **REQ-ENV-04（與 frontier.simulate 的逐 seed 一致）**：對任何以 frontier 欄位表達的手寫策略（SCOLS 全部欄位），「轉接策略」（adapter：以和 frontier 完全相同的運算順序算出 v1）在本模擬器下，輸出欄位 `reg_b, util_b, flags, pen, audits, nraid, susp` 與 frontier.simulate **逐 seed 逐位元相等**（`==`，不是近似）。MLP 策略只要求與自身在不同批次組成下的一致性（REQ-ENV-09）。
- **REQ-ENV-05（零偏離＝誠實）**：θ=0 的 MLP 策略（§4.2）輸出 d=0.0，v1 = u1 精確；結果與 frontier.simulate 的 honest 欄位（策略 `{}`）**逐位元相等**。此外，只要輸出層權重與偏置為 0（隱藏層任意），也必須逐位元相等。
- **REQ-ENV-06（前置等價）(revised)**：`frontier.simulate`（M3C，`p=0`）與 `quota.simulate`（NAIVE）在 `util_b, reg_b, pen` 上逐 seed 逐位元一致；`frontier.simulate` 與 `quota.simulate` 在 M3C、M4 上同參數同 seeds 也逐位元一致（quota 的這些分支是複製來的，方向一的 JSON 由 quota.simulate 產生；若不一致，後續與 JSON 的比對就無法歸因）。
- **REQ-ENV-07（重播方向一 JSON）(revised)**：以 `results/quota.json` 為準，模擬器在「方向一重播」（記分到 T，見 §2.5）下重現：
  - `main[12]`（M3C，r=0.5）與 `main[14]`（M3C，r=0.9）的 `gains_by_strategy` 中全部 36 個非 quota 策略（含 oracle 邊緣灌水；此 36 為方向一 JSON 的項目數，與 v1.3 的 S_HW 36 個無關；`gains_by_strategy` 共 44 項，扣除 8 個 `quota_*`）、`honest_util_agent1`、`R_hon_b`、`false_punish_frac`；
  - `main[4]`（NAIVE，r=0.5）的 `Gmax`（0.27930049154850023，策略 dz=2.0）；
  - `main[88]`（M4，r=0.5，ε=0.02、L=2500、rw=1/3）的 `raid_waste_frac`、`R_hon_b` 與非 quota 策略的 gains。
  - **容許誤差（revised）**：數值一律用**相對容差** `|a−b| ≤ 1e-9·|b|`；b 為精確 0.0 時要求 a 精確等於 0.0（預期多數情形為逐位元相等）。
  - **CI 只比對 JSON 中有 CI 的欄位**：`[mean, half_width]` 形式者（`Gmax*`、`R_hon_b`、`false_punish_frac`、`honest_util_agent1`、`raid_waste_frac`）以方向一的公式 `1.96·std(ddof=1)/√n` 重算比對半寬；`gains_by_strategy` 的值是純量、沒有 CI，只比對均值。
  - seeds 為 5000–5031（`quota.meta.eval_seeds`），T=100000。
- **REQ-ENV-08（模擬器輸出）(revised)**：每個元素回傳：`util_b[:,K]`、`reg_b`、`pen[:,K]`、`flags[:,K]`、`audits`、`nraid`、`susp[:,K]`，以及在記分輪 `T_score`（§2.5）的快照（上列累計量在 t=T_score 的值）。另可選回傳每輪 v1 序列與每輪的公開日誌（診斷用，預設關閉；OBS-10 需要）。**快照因果性**：T 輪模擬在 t=T_score 的快照，必須與 T'=T_score 輪的完整模擬的最終累計量逐位元相等（模擬因果、不看未來；此條吸收原 ENV-18）。
- **REQ-ENV-09（批次不變性）**：同一個（θ, seed, cfg）單獨跑，與放在任意大批次、任意位置、任意切塊（chunk）下跑，結果一致：轉接策略逐位元相等；MLP 策略允許 |差| ≤ 1e-12（float64 矩陣乘法的批次相依捨入）。

### 2.2 與 frontier.simulate 的欄位對應

| 概念 | frontier 欄位／變數 | 本模擬器 |
|---|---|---|
| 機制參數 | PCOLS：`p, tol, L, eps, rw, kref, nmax, ad_h` | 同名；M3C 用 `p, tol(=h), L, kref, ad_h`；M4 用 `eps, L, rw` |
| 型別參數 | RCOLS：`rho, tau, ra, online` | 同名；`ra=-1`（稽核者使用真實 r，即「已知 ρ」）、`online=0` |
| 偏離者策略 | SCOLS：`b, dz, dzc, W, H, em, leg, ue_*, ad_*, timing` | 由策略回呼 `policy(obs)→d` 取代；SCOLS 只在轉接策略內使用 |
| 型別 | `z`（AR(1)），`u = transform(ndtr(z))` | 同；agent 1 的 `u1 = ndtr(z1)` |
| 稽核私有狀態 | `cus`、`pend_a`、`pend_v`、`flags`、`audits`、`sxy/sxx` | 同，**不得**進入公開版觀察（§3.1） |
| 公開狀態 | `susp`（暫停截止時間） | 觀察以相對量 `susp−t` 提供（§4.4） |
| 計分 | `util_b[:,1]`（得標得 u，無人合格時隨機分配）與 `reg_b`、`pen` | 同 |

### 2.3 方向一選定的 M3C 參數（精確值）

來源：`results/stage2.json: main.M3c["100000"][r].param` 與 `results/quota.json`（`scripts/run_quota.py: m3c_params()`）；`kref` 取 frontier 預設 0.5（`param` 未覆寫）。

- **REQ-ENV-10（固定配置）**：M3C 參數寫死為下表，機制不針對 RL 重新調參。

| 參數 | 值 | 出處 |
|---|---|---|
| mech | M3C（派遣時 CUSUM＋可寬恕暫停） | |
| K, τ, C_MIN | 3, 1, 0.2 | `quota.meta.K, tau`；`adaaudit.C_MIN` |
| r（=ρ） | **0.5（主格）、0.9（探索）** | `quota.meta.rhos` |
| p（稽核機率） | 0.1 | `stage2.json … param.p` |
| h（`tol`） | 6 | `stage2.json … param.tol`（r=0.5、0.9 皆為 6） |
| L（暫停長度） | 2500 | `stage2.json … param.L` |
| kref | 0.5 | frontier `PCOLS` 預設 |
| ad_h, ra, online | INF, −1, 0 | 預設 |
| T（評估） | 100000 | `quota.meta.T` |

  此配置下的方向一參考值（僅供對照，驗收門檻不是這些數字，除了 REQ-ENV-07 的重播）：r=0.5：誠實 R/T 0.0071 ± 0.0018、誤罰 α 0.0285 ± 0.0061、G(無資訊) 0.0213 ± 0.0039（策略 dz=0.3）、G(含 oracle) 0.0870 ± 0.0053；r=0.9：R/T 0.0070 ± 0.0016、α 0.0303 ± 0.0060、G(無資訊) 0.0020 ± 0.0011（dz=0.03）、G(含 oracle) 0.0211 ± 0.0033。這些「最大值」取自評估 seeds，上偏；本規格的 G_手寫 在驗證集選、測試集評，預期較低。
  **已決定（D-1，commander）**：使用本表（主表）配置；r=0.5 的無資訊手寫得利為 0.0213 ± 0.0039（`quota.json: main[12].Gmax_uninformed`，已核對 JSON：[0.021327654113202883, 0.0039011376120500494]；number-audit #93）。notes 的 0.0287 出自 `stage2b.json: F3_uninformed[43].edge_uninformed.gain`（number-audit #42），是另一組參數（h=16、kref=0.25、T=20000、α 目標 0.01），**不使用**。

- **REQ-ENV-11（錨點機制參數）(revised)**：
  - NAIVE 錨點：M3C 且 `p=0`，r=0.5（G 與 r 幾乎無關：`quota.json main[4..7].Gmax` 依序為 0.27930、0.27930、0.27923、0.28056）。已知 G＝0.2793 ± 0.0004（`main[4].Gmax`＝[0.27930049154850023, 0.0003786066909699181]，策略 dz=2.0）。
  - **M4 錨點（commander 更正後）**：M4，`eps=0.2, L=20, rw=0`（舊版：不突襲得標者），**r=0.99**，錨點策略為常數灌水 `b=0.3`（只用自己的 u，屬公開資訊可實現）。JSON 出處：`results/stage2b.json: consist_M4[12].gain_b03`＝[0.12093494445548655, 0.0014286323549844095]（+0.1209 ± 0.0014；該列 `r=0.99, eps=0.2, L=20, rw=0.0`）。同格其他策略：`gain_dz1`＝0.1126 ± 0.0014、`gain_edgeO`＝0.0300 ± 0.0006（oracle）。注意 `consist_M4` 的條件為 T=100000、seeds 1000–1031（`scripts/run_stage2b.py: part_consist()` 的程式碼；JSON 的 `meta.T=20000` 屬其他部分，不適用於此區塊）。
  - **更正說明**：原規格曾引用 `consist_M4[9]`（r=0.99、ε=0.05、L=2500、rw=0）的 `gain_edgeO`＝0.0356（oracle 邊緣灌水）；同一格的常數灌水為 `gain_b03`＝−0.1475 ± 0.0039（`consist_M4[9].gain_b03`＝−0.14747848180358147），不能當錨點。
  - 錨點的 G*_ref 在 W1 以驗證 seeds 於公開資訊下重新量測（REQ-S1-09）；上列 JSON 數字只作對照，不是驗收值（seeds 不同）。

### 2.4 時序（每輪 t=1..T，與 frontier.simulate 的運算順序一致）

- **REQ-ENV-12（事件順序，必須逐步照此，否則 REQ-ENV-04 不成立）**：
  1. 型別更新：`z ← ρ z + √(1−ρ²) ε_t`；`u = transform(ndtr(z))`。
  2. 合格：`elig_k = (t > susp_k)`（暫停截止時間 `susp_k` 為上一輪結束時的值）。
  3. **觀察**：由「第 t−1 輪結束時的狀態」加上本輪的 `z_t, u_t` 建構觀察向量（§4.4），**不含**任何第 t 輪才產生的量（本輪別人的報告、本輪的稽核抽籤、本輪的突襲抽籤）。
  4. 偏離者報告 `v1 = clip(u1 + d, 0, 1)`；其他 agent 報告 `v=u`。（轉接策略則照 frontier 公式。）
  5. 分配：M3C：在合格者中取最高報告（`argmax`，平手取最小 index），無人合格時 `win=False` 並隨機分配（`ridx`）；M4：同，另有突襲（`rd`）使該輪物品浪費。
  6. 計分累加（`reg_b, util_b, pen`）。
  7. 稽核排程：`o = win_eff & (Ua < p)`，結果寫入槽 `(t+τ)%M`（`M=τ+1`），記錄被稽核的 agent 與其報告。
  8. 稽核結算：結算槽 `t%M`：以本輪真實 `u_t` 與當時報告 `v` 算殘差 `resid=(z_w − r·z_v)/√(1−r²)`（`z=ndtri(phi_of(·))`）。**`upd = has & elig_ag`（revised）**：只有「該槽有排定的稽核（`has`）」且「被稽核者在結算當下合格（`elig_ag = t > susp_ag`）」時才更新 CUSUM `S ← max(S − resid − kref, 0)`；若 `S>h` 則標記、`S←0`、`susp ← t+L`。被稽核者在結算時已被暫停（`elig_ag` 為假）→ `S` 不變、不標記（與 frontier.simulate 一致；`sxy/sxx` 的累計不受 `elig_ag` 影響，只受 `has` 影響）。
  9. 本輪結束。被標記者自第 t+1 輪起不合格，共 L 輪（`t+1 … t+L`）。
- **REQ-ENV-13（暫停不疊加）**：`susp` 是「覆寫為 t+L」，不是累加；暫停中的 agent 不合格，因此既不能得標，在 M4 中也不會成為非得標者的突襲目標，也不會再被稽核標記。測試以構造情境驗證。
- **REQ-ENV-14（不提供絕對時間）(revised)**：策略可見的任何量都不得含有 t 或由 t 推得的絕對量（`rem=(susp−t)/L`、`since` 等為相對量，允許）。開頭 8 輪的零填充（歷史窗不足）與 EMA 暖機（約 100 輪）會隱含少量時間資訊，視為可接受並於結果中註明。環境不洩漏「還剩幾輪」。**因為觀察不含時間，所以環境不強制尾端誠實**（commander 決定 2；REQ-ENV-16 withdrawn）；末端暫停成本被截斷造成的偏差改由 REQ-ENV-20 量測並報告。

### 2.5 horizon：只把最後 L 輪排除在評估之外（不強制誠實）

**不使用幾何終止；不強制尾端誠實；只把最後 L 輪排除在記分之外**（commander 決定 2）。理由：幾何終止會改變隨機數對齊與期望長度，破壞與 frontier／JSON 的逐 seed 比對與 CRN 配對；觀察不含時間（REQ-ENV-14），策略無法蓄意利用末端。

- **REQ-ENV-15（記分窗）(revised)**：總模擬輪數 T；「方向二」記分輪 `T_score = T − L`；所有「方向二指標」（G、R/T、α）只用第 1..T_score 輪的累計量（取 REQ-ENV-08 的快照），並除以 T_score。「方向一重播」（REQ-ENV-07、REQ-ENV-21、REQ-ENV-04 的一致性測試）的記分輪為 `T_score = T`（不排除最後 L 輪）。除記分窗外，兩者的模擬完全相同；策略輸出在任何一輪都不被環境改寫。 **預設與退化（v1.1）**：模擬器未明示 `T_score` 時，預設為方向二模式 `T_score = T − L`（不是「整段」；骨架原 docstring「None ＝ final」有誤，以本條為準）；方向一重播須顯式傳 `T_score=T`。當 `T ≤ L`（無可排除之輪）時，`T_score` 退回 `T`。
- **REQ-ENV-16（尾端保護，tail guard）(withdrawn)**：原「t > T−2L 強制誠實」。依 commander 決定 2 刪除。
- **REQ-ENV-17（兩種模式）(withdrawn)**：併入 REQ-ENV-15（兩種模式現在只差記分窗）。
- **REQ-ENV-18（因果性）(withdrawn)**：快照因果性併入 REQ-ENV-08。
- **REQ-ENV-19（長度）(revised)**：評估與驗證 T=100000（記分 T−L 輪）。訓練 T_train 待量測，允許值 ∈ {20000, 40000, 100000}（20000 為下限＝8L）。
- **REQ-ENV-20（截斷暫停偏差的量測，新增）**：在 `T=100000、L=2500` 下，對被選的 RL 策略與被選的手寫策略，另跑一個**診斷變體**（只供量測，不用於任何判定、訓練或選取）：變體中 t>T−2L 的輪次強制 `v1=u1`，其餘相同；回報 `ΔG_trunc = G(正常) − G(變體)`（paired，95% CI），並寫入 `results/direction2/trunc_bias.json`，欄位 `policy, r, G_normal, G_variant, delta, ci`。解讀：`ΔG_trunc` 是末端暫停成本被截斷（最後 L 輪沒被計入）造成的 G 偏差上界的近似。**必須報告**，若其 CI 下界 >0 且 ≥ MDE，須在結果文字中註明「G 含截斷偏差」。
- **REQ-ENV-21（重播 stage2b 的 M4 錨點，新增）**：以 `frontier` 的機制設定，模擬器在方向一重播記分窗（`T_score=T=100000`）、seeds 1000–1031（重播專用，不屬任何切分，REQ-SEED-01）下重現 `results/stage2b.json: consist_M4[12]`（r=0.99、ε=0.2、L=20、rw=0）的 `gain_b03`、`gain_dz1`、`gain_edgeO` 與 `waste`（皆為 `[mean, half_width]`，CI 以 1.96 公式重算），並重現 `consist_M4[9]` 的 `gain_b03`（−0.1475）與 `gain_edgeO`（+0.0356）。容差同 REQ-ENV-07（相對 1e-9）。

---

## 3. 觀察資訊集

對手知道機制規則與參數（Kerckhoffs 原則）：p、h、L、kref、r、C_MIN、各 agent 的邊際分布。這些是**常數**，不是每輪的觀察特徵。M4 的 ε（每輪突襲機率）與 rw（突襲對象為得標者的機率）同屬公開常數（REQ-OBS-10）。以下列的是**每輪變動、對手可由機制互動得知**的量。

### 3.1 公開版（主要資訊集，REQ-OBS-01）

frontier.py 的稽核是**靜默**的（`pend_a` 不通知）、CUSUM 值 `cus` 是機制私有，對手唯一看得到稽核結果的管道是自己被暫停（`susp`）。

| 欄位 | frontier 變數 | 為什麼對手看得到 |
|---|---|---|
| 自己的當期型別 z_t、u_t | `z[:,1]`、`u[:,1]` | 自己的私有型別，本來就知道 |
| 自己的報告歷史 v_{t−j} | `v[:,1]` 的歷史 | 自己提交的 |
| 自己是否得標 | `win_eff & (idx==1)`（得標且未被突襲） | 得標者會拿到物品（M4 中被突襲者拿不到）。注意這會洩漏「對手最大報告低於自己」的資訊，視為合法並需註明 |
| 自己是否合格、暫停剩餘 | `elig[:,1]`、`susp[:,1]−t` | 被暫停時機制通知／無法得標，是稽核結果唯一的公開訊號 |
| 距上次被暫停多久 | 由 `susp[:,1]` 上升事件記錄 | 自己被暫停的事件，自己知道 |
| 自己可自行計算的殘差 | `resid` 公式的輸入（自己的 `v_{t−1}`、`u_t`、r、kref） | 全是自己的量與公開常數；對手**不知道**該輪是否被稽核（`Ua<p` 靜默），也不知道 `cus` |
| **不含** | `cus`、`pend_a`、`pend_v`、`flags`、`audits`、`sxy/sxx`、別人的 `u`/`v`、`Ua/Rs/Rx` 抽籤、`rd`（突襲抽籤本身） | 機制私有或他人私有 |

- **REQ-OBS-02（別人的暫停）(revised)**：**已決定（D-5，commander）：他人的 `susp` 不公開**。公開版**不含**其他 agent 的 `susp` 或 `elig`；只有自己的 `susp`。

### 3.2 稽核公告版（上界對照，REQ-OBS-03）
公開版再加兩個特徵：真實 CUSUM 值 `S_t = cus[:,1]/h`、上一輪是否被稽核（`pend_a` 在 t−1 輪排定且 agent=1）。解讀為「若機制公告稽核結果」的資訊價值上界，**不是**可實現的威脅。

### 3.3 oracle 版（延伸目標，REQ-OBS-04）
公開版再加其他兩位 agent 的當期型別 `u_0, u_2`（不是報告；誠實時兩者相同）。策略類別嵌套於公開版，可用公開版解暖啟動。只在延伸目標做。

- **REQ-OBS-05（嵌套）**：oracle ⊃ 公開版、公告 ⊃ 公開版；前 25 維特徵的定義與順序完全相同（§4.4）。
- **REQ-OBS-06（私有量隔離，可斷言）**：公開版特徵函式的輸入只能是「公開白名單」物件，不得接觸模擬器完整狀態。測試方法：在同一步，把私有狀態（`cus, pend_a, pend_v, flags, audits, sxy, sxx`、別人的 `u`、別人的報告）換成隨機值，公開版特徵向量必須逐位元不變；公告版特徵必須因 `cus`、`pend_a` 改變而改變。

---

## 4. 動作與策略類別

### 4.1 動作
- **REQ-ACT-01**：偏離者每輪直接報告 `v1 ∈ [0,1]`。參數化為殘差形式 `v1 = clip(u1 + d, 0, 1)`，`d = tanh(o)`，`o` 為策略輸出（保證 `d=0 ⇒ v1=u1` 逐位元；任何 v∈[0,1] 都可由 d∈(−1,1) 加截斷達到）。不輸出 d≥0 的灌水量（notes 第 1 輪：d≥0 表達不了策略族，需允許少報）。

### 4.2 策略類別
- **REQ-ACT-02（策略）(revised)**：單隱藏層 MLP：`F → 2 → 1`，隱藏層 tanh，輸出線性再套 tanh（REQ-ACT-01）。參數量：`F·2 + 2 + 2 + 1`；公開版 F=25 → **55**；公告版／oracle 版 F=27 → **59**。（隱藏寬度 2 是為了滿足 REQ-ACT-03 的 ≤64 上限的預設值，見 §17 P-2。）
- **REQ-ACT-03（參數上限，寫死）(revised)**：策略參數量上限 **P_MAX = 64**。超過必須拒絕建構。優化器依維度選擇：**n ≤ 50 用完整共變異數 CMA-ES；n > 50 用 sep-CMA-ES（對角共變異數）**（REQ-OPT-01）；公開版 55 維因此用 sep-CMA-ES。64 步歷史對照（§8.7）屬診斷用途，使用獨立上限 **P_DIAG = 300**（F=134、隱藏 2 → 273，同樣用 sep-CMA-ES）。
- **REQ-ACT-04（零參數＝誠實）(revised)**：θ=0（含所有偏置）對應 d≡0。冷啟動的初始平均為 θ=0。**暖啟動（以手寫策略回歸初始化）與冷啟動兩條路都在凍結前登記**（寫入凍結檔），由 REQ-S1-20 的 NAIVE 學習性 pilot 在凍結前決定採用哪一條；採用的那一條對 S1 與主實驗一視同仁（REQ-OPT-07）。

### 4.3 手寫策略的角色
手寫策略以「轉接策略」在同一模擬器內跑（REQ-ENV-04）。它們與 RL 策略受同一個環境與記分窗約束（REQ-ENV-15），不強制尾端誠實。

### 4.4 特徵清單（公開版 F=25，k=8，寫死）

所有特徵縮放到約 [−1,1]。`λ_e = 0.99`（EMA 衰減）。第 t 輪的觀察只用到第 t−1 輪結束時的狀態與本輪 `z_t, u_t`。歷史窗不足（t−j<1）補 0。

| # | 名稱 | 定義 | 備註 |
|---|---|---|---|
| f01 | `z` | `clip(z_t/3, −1, 1)` | 自己型別 |
| f02 | `u` | `2u_t − 1` | |
| f03 | `elig` | `1[t > susp_1]` | 是否合格 |
| f04 | `rem` | `max(susp_1 − t, 0)/L` | 暫停剩餘比例 |
| f05 | `since` | `min(t − t_s, 2L)/(2L)`，`t_s` 為「**自己首次觀察到 `susp_1` 上升的那一輪**」（即被標記那一輪的下一輪；該輪 `since=0`）；從未被暫停＝1.0（REQ-OBS-11） | 相對時間 |
| f06–f13 | `won_{t−j}`，j=1..8 | 第 t−j 輪自己得標＝1，否則 0 | **k=8** |
| f14–f21 | `d_{t−j}`，j=1..8 | `v_{t−j} − u_{t−j}`（自己報告減自己真值） | **k=8** |
| f22 | `res_last` | 若 `won_{t−1}=1`：`clip(res/6, −1, 1)`，`res=(z_w − r·z_v)/√(1−r²)`，`z_v=ndtri(phi_of(v_{t−1},1,C_MIN))`，`z_w=ndtri(phi_of(u_t,1,C_MIN))`；否則 0 | 「若該輪被稽核會得到的殘差」，與機制的 `resid` 公式一致 |
| f23 | `ema_res` | `λ_e·ema_res + (1−λ_e)·f22`，初始 0 | CUSUM 的自行估計代理 |
| f24 | `ema_d` | `λ_e·ema_d + (1−λ_e)·d_{t−1}`，初始 0 | |
| f25 | `ema_win` | `λ_e·ema_win + (1−λ_e)·won_{t−1}`，初始 0 | |
| f26（公告版） | `S` | `cus[:,1]/h`（∈[0,1]） | 私有，僅公告版 |
| f27（公告版） | `audited_last` | 上一輪自己的得標是否被稽核 | 私有，僅公告版 |
| f26'（oracle 版） | `u0` | `2u_{0,t} − 1` | 僅 oracle |
| f27'（oracle 版） | `u2` | `2u_{2,t} − 1` | 僅 oracle |

- **REQ-OBS-07（特徵表）**：特徵順序、定義、縮放必須與上表完全一致；公開版維度恰為 25；所有特徵在任意 rollout 上取值於 [−1,1]。
- **REQ-OBS-08（特徵函式簽章無時間）**：特徵建構函式的簽章與其接收的狀態物件不得含 `t`、`time`、`round`、`step` 之類欄位（相對量由環境事先算好）。**漏洞套件的參考策略（`ref_policy`）亦適用同一限制（REQ-S1-27）：不得使用絕對時間。**
- **REQ-OBS-09（限制必須明寫）(revised)**：特徵是手工充分統計量的近似，遺漏的特徵會漏掉依賴更長歷史或絕對時間的漏洞；套件中的 1 個「特徵盲區」範例（§8.2：O5；B1、B2 未實作，故對絕對時間與長記憶盲區沒有證據）刻意量測這個限制。所有「未找到」的結論只限於此特徵類別（§9.3）。
- **REQ-OBS-10（端對端資訊隔離，新增）(revised v1.1)**：模擬器在診斷模式（REQ-ENV-08）下，每一輪輸出兩份紀錄：(a) **公開日誌**，只含：`z_1,u_1`（自己的型別）、`v_1`（自己的報告）、`won_1`（自己是否得標）、`susp_1`（自己的暫停截止時間）、以及機制常數（p、h、L、kref、r、C_MIN；M4 另含 ε、rw。**commander 決定：M4 的 ε、rw 依 Kerckhoffs 原則列為公開常數，T-56 的常數白名單允許**）；(b) **策略實際收到的 obs 向量**（逐輪、逐元素）。另有一個**獨立重建器**（獨立模組，只依 §3.1、§4.4 的文字撰寫，不得 import 模擬器或特徵建構函式；T-56 以檔案層級靜態檢查）只讀 (a) 並依 §4.4 重建 obs。**必須逐位元相等**（`==`）。意義：obs 完全由公開日誌決定，沒有任何私有量（`cus`、`pend_a`、`pend_v`、`flags`、`audits`、`sxy`、`sxx`、他人的 `u`/`v`/`susp`、抽籤）從特徵管線之外的路徑滲入。公告版／oracle 版（延伸）的重建器另外接受其額外輸入，且公開版的前 25 維仍須逐位元相同。
- **REQ-OBS-11（f05 的 t_s，新增，v1.1）**：f05 的 `t_s` ＝自己首次觀察到 `susp_1` 上升的那一輪（屬公開資訊；在該輪 `since=0`；被標記那一輪本身不可見，故不採用）。獨立重建器與特徵建構函式都必須採此定義。
- **REQ-OBS-12（susp1 的時點，新增，v1.1）**：公開日誌的 `susp1[t]` 記錄的是第 t 輪**觀察時**（即第 t−1 輪結算後）的 `susp_1`，不是第 t 輪結算後的值。序列以 `[...,T]` 儲存、位置 t−1 對應第 t 輪。

---

## 5. 優化器

### 5.1 演算法
- **REQ-OPT-01 (revised)**：CMA-ES 家族（Hansen 的標準 (μ/μ_w, λ) 版；ask／tell 介面，CMA 的更新在 CPU、適應值評估在 GPU）。**維度 n ≤ 50 用完整共變異數 CMA-ES；n > 50 用 sep-CMA-ES（對角共變異數，Ros & Hansen 2008）**；公開版 n=55 ⇒ sep-CMA-ES，診斷用 273 維同樣如此。初始平均 θ=0（冷啟動；暖啟動見 REQ-OPT-07），初始步長 σ0=0.3（寫死；不在 S1 或主實驗之間調整）。不使用 PPO、不使用折扣。
- **REQ-OPT-02（目標）**：每世代，個體 i 的適應值為 **無折扣平均報酬** `f_i = (1/|S_g|) Σ_{s∈S_g} U_i(s)/T_score`，`U_i(s)` 為 REQ-ENV-08 在 T_score 的 `util_b[:,1]` 快照。誠實基準 `U_0(s)` 對所有個體相同，不影響排名，但每世代記錄 `G_i = f_i − f_honest` 供日誌。

### 5.2 CRN 與對偶（鏡像）取樣
- **REQ-OPT-03（CRN）**：同一世代所有個體使用同一組環境 seeds `S_g`（REQ-ENV-02 保證同 seed 同隨機數）。`S_g` 為訓練 seeds 0–999 的輪替區塊：世代 g 取 `perm_k` 的第 `[g·n_S, (g+1)·n_S) mod 1000` 個，`perm_k` 為以 `run_id` k 為種子的 0..999 的固定排列（`Random(9000+k).shuffle`）；`n_S`（每世代 seeds 數）待量測。
- **REQ-OPT-04（鏡像取樣）**：候選以 ± 成對產生（`θ_m ± δ_j`），λ 取偶數；每一對共享同一個 `δ_j`。
- **REQ-OPT-05（誠實基準共用）**：同世代的誠實基準只算一次（θ=0 個體），所有個體共用。

### 5.3 驗證與選取
- **REQ-OPT-06 (revised v1.2)**：每 **50** 世代（v1.1 為 25；MEAS v2 後 commander 決定，驗證成本減半）以當前平均 `θ_m` 在**驗證 seeds**（2000–2031）評估 `G_val`（方向二模式、T=100000）；訓練結束時，選取 `G_val` 最大的檢查點作為該 run 的最終策略（早停亦由此決定）。測試 seeds 在此流程中不得出現（§6）。

### 5.4 暖啟動與冷啟動
- **REQ-OPT-07 (revised)**：**暖啟動與冷啟動兩條路都在凍結前登記**（寫入 `docs/direction2-freeze.md`，REQ-S1-14）：冷啟動＝θ=0；暖啟動＝以回歸把 MLP 初始化成「在訓練 seeds 上 G 最大的 S_HW 手寫策略」的 d 輸出（回歸步數與損失凍結前定稿，待量測）。二選一由 REQ-S1-20 的 NAIVE 學習性 pilot 在凍結前決定（pilot 只用訓練與驗證 seeds）。決定後，S1 全部與主實驗**一律**採用同一條路，不得只對失敗的 cell 改用另一條，也不得在凍結後切換。此規則取代原「冷啟動失敗後再改暖啟動」的作法（原規則與 REQ-S1-14 的「凍結後禁止修改」互斥）。

- **REQ-OPT-12（obs_version 必須明確傳入，新增，v1.2）**：所有新的訓練、驗證、評估程式碼（`train.py`、`s1.py`、`final_eval.py` 等）在建構 `_Bank`、`simulate`、`init_state` 與 `MLPPolicy` 時，**必須明確傳入 `obs_version`**（`"public"`、`"announced"`、`"oracle"` 之一）；不得依賴預設值。`obs_version=None` 的預設行為只為了相容舊測試（此時只以 F 推斷資訊集：F=25 視為公開版，F=27 視為公告版或 oracle 版皆可，無法區分），**新程式碼不得走這條路**。理由：F=27 無法區分公告版與 oracle 版，靠預設值會讓錯配的資訊集靜默通過。

### 5.5 待量測後定稿的參數與決定規則

下表為 **v1.2 定稿值**。來源兩份 JSON：**MEAS v1**＝`results/direction2_meas.json`（git_sha `eae1f3a`，簡稱 `meas.json`）；**MEAS v2**＝`results/direction2/measure_v2_summary.json`（簡稱 `v2.json`，含最壞情況 M-2′ 與 M-1′）。規則本身寫死，修訂不得改規則。數值後的括號為 JSON 鍵路徑；決定規則的逐步追蹤見 `meas.json: rules.S1.trace`、`rules.main.trace`。v1.1 的 n_S=64、300 世代、每個 run 約 1.8 小時已被本表取代（原因：紅方指出 v1 的 `sd_crn` 以 σ0=0.3 的隨機策略對量得，低估訓練後期接近誠實時的變異；1.8 小時也未計入驗證成本與 p95）。

| 參數 | 符號 | 定稿值（JSON 鍵路徑） | 決定規則 |
|---|---|---|---|
| 族群大小、訓練長度 | λ、T_train | **λ=256、T_train=20000**（`meas.json: rules.S1.choice.lam`、`rules.S1.choice.T_train`；`rules.main.choice` 相同）；B=λ·n_S=256×256=65536 ≤ `B_max`=2097152（`meas.json: B_max`）。n_S=256 的峰值顯存約 140.7 MB（`v2.json: run_time_lam256_T2e4_300gen.nS256_mean.mem_peak_MB`） | **REQ-OPT-08（聯合規則）**：依 λ 由大到小、每個 λ 內依 T_train 由大到小，取第一個同時滿足 (a) `λ·n_S ≤ B_max` 與 (b) `300·T_train·t_step(B=λ·n_S) ≤ W_run` 的組合（λ 優先於 T_train）；無組合滿足 → 判定不可行（REQ-STOP-02）。T_train=4e4、1e5 在 λ=256 時超時（`meas.json: rules.S1.trace[0..1].ok_time=false`），20000 通過（`trace[2].ok_time=true`）；v2 的 t_step（mean 1.359 ms）下 4e4 亦超過 4 小時（300×40000×1.359 ms＝16308 s＞14400 s），結論不變 |
| 每世代 seeds 數 | n_S | **256**（`v2.json: nS_need["0.5"].nS_from_median`＝256；上限由 128 提高為 256，commander 決定）。依據：最壞情況取 σ=0.03（訓練後期接近誠實）的 `sd_CRN` 中位數 0.021732（`v2.json: m2_summary[0].sd_crn_median`，亦即 `nS_need["0.5"].sd_worst_median`）；門檻 0.00145＝0.25·0.0058（`nS_need["0.5"].threshold`）；連續解 n_S≥224.6（`nS_need["0.5"].nS_formula_continuous_median`）→ 取 256；128 不滿足（`nS_need["0.5"].nS_from_median_cap128`＝null） | **REQ-OPT-09 (revised v1.2)**：取最小的 n_S ∈ {16, 32, 64, 128, 256}，使 `sd_CRN/√n_S ≤ 0.25·MDE_est(r=0.5)`；其中 `sd_CRN` 取 σ∈{0.03, 0.1, 0.3} 三種隨機策略對中**位數的最壞值**（不是 σ=0.3 單一值）；`MDE_est(r=0.5)=0.0058`（由 `main[12].Gmax_uninformed` 半寬 0.0039011 換算：`0.0039011/1.96×2.8922=0.005757`，κ 見 REQ-MET-06）；n_S=256 仍不滿足 → r=0.5 判為不可行（REQ-STOP-02）。**單一隨機對的最壞值**（`sd_CRN`=0.034465，`nS_need["0.5"].sd_worst_pair`）需 n_S≈565（`nS_need["0.5"].nS_formula_continuous_pair`＝564.95；`nS_from_worst_pair`＝null），**目前無法涵蓋，須揭露**（REQ-OPT-13） |
| 世代數 | G_gens | **r=0.5：416；r=0.9：499**（OPT-10 公式；`meas.json: rules.S1.W_run_s`＝9000、`rules.main.W_run_s`＝10800、`rules.S1.choice.t_step_ms`＝1.0812，即 T_train·t_step＝21.62 s：⌊9000/21.62⌋＝416、⌊10800/21.62⌋＝499）。S1、NAIVE pilot、診斷、M4 錨點（r=0.99）與其他 r=0.5 的 run 用 416；主實驗 r=0.5 用 416、r=0.9 用 499 | **REQ-OPT-10 (revised v1.2)**：`min(1000, ⌊W_run/(T_train·t_step)⌋)`，下限 300；W_run、t_step 取事前登記值（S1 2.5 小時、主實驗 3 小時；v1 的 t_step 1.0812 ms）。**v1.2 更正**：v1.1 取 300 與規則不一致；commander 改依公式（世代不足會把結果誤判為 (iii)「學不起來」而汙染結論）。**實際耗時與 W_run 不同**：v2 實測 t_step 較高（見下列），每個 run 預算放寬到約 4 小時（使用者同意 GPU 不設上限）；公式的輸入不隨之更動 |
| 每步耗時 | t_step | **mean 約 1.359 ms、p95 約 1.948 ms**（n_S=256：`v2.json: run_time_lam256_T2e4_300gen.nS256_mean.t_step_ms`＝1.35901、`nS256_p95.t_step_ms`＝1.94808）；p95 反映機器高負載時的耗時。n_S=64、128 幾乎相同（`nS64_mean.t_step_ms`＝1.3797、`nS128_mean.t_step_ms`＝1.3317）；n_S=512 為 1.759 ms（`nS512_mean.t_step_ms`） | REQ-MEAS-01；內插規則同前 |
| 驗證 | — | **每 50 世代一次**（REQ-OPT-06）；每次驗證（T=1e5、32 seeds）約 132.7 秒（`v2.json: validation.t_val_s`＝132.724）；每個 run 驗證次數＝⌊G_gens/50⌋：416 世代 8 次、499 世代 9 次 | v1 的 25 世代一次改為 50（成本減半，commander 決定） |
| 單次 run 時間 | W_run | **每個 run 預算約 4 小時**（commander 決定；v1.1 的 1.8 小時作廢）。每個 run＝G_gens×T_train×t_step＋⌊G_gens/50⌋×t_val（不含啟動開銷，記於 part 檔 `wall_s`）：**r=0.5（416 世代）：mean 3.44 小時（12369 s＝11307＋1062）、p95 4.80 小時（17270 s）；r=0.9（499 世代）：mean 4.10 小時（14757 s＝13563＋1195）、p95 5.73 小時（20636 s）**。mean 版大致落在 4 小時內（r=0.9 略超）；p95 版超過 4 小時，因此靠 REQ-GPU-08 分段、不以單一 job 跑完。對照 v2.json 的 300 世代／25 世代驗證版：`run_time_lam256_T2e4_300gen.nS256_mean.total_h`＝2.707、`nS256_p95.total_h`＝3.689（舊 W_run 2.5／3 小時皆不通過：`ok_S1_2p5h=false`） | 估算式見 REQ-GPU-05；**已決定 D-4**：先量測再優化；t_step 實測 1.36 ms，CUDA graph／`torch.compile` 優化不再是必要工作 |
| CRN 降幅 | VRF | **CRN 降幅有限（VRF 約 1–1.7）**：v1 中位數 r=0.5：1.22（T=2e4）、1.32、1.54；r=0.9：1.68、1.71、1.55（`meas.json: m2_summary[*].vrf_median`）；v2（σ=0.03／0.1／0.3，T=2e4）r=0.5：1.25、1.22、1.28；r=0.9：0.98、1.08、1.58（`v2.json: m2_summary[*].vrf_median`）；個別隨機對最低 0.83（`v2.json: m2_summary[4].vrf_min`）；手寫對較高（`meas.json: m2_summary[0].vrf_median_hand`＝18.0）。MLP 策略對的 CRN 效益小，須記於凍結檔（REQ-MEAS-02），n_S 依 REQ-OPT-09 處理，不另調規則 | 見 REQ-MEAS-02 |
| r=0.9 可行性 | — | **功效不足，僅作描述**：r=0.9 最壞 σ=0.03 的 `sd_CRN` 中位數 0.020915（`v2.json: nS_need["0.9"].sd_worst_median`），門檻 0.0004（`nS_need["0.9"].threshold`）；需 n_S≈2734（`nS_need["0.9"].nS_formula_continuous_median`），單一對最壞需約 5954（`nS_formula_continuous_pair`），遠超上限 256（`nS_from_median`＝null） | **REQ-OPT-11 (revised v1.2)**：以 M-2 在 r=0.9 的 `sd_CRN`，若 `sd_CRN/√n_S > 0.25·0.0016`（`MDE_est(r=0.9)=0.0016`，由 `main[14]` 半寬 0.00106045 換算：`0.00106045/1.96×2.8922=0.001565`）則 r=0.9 標示「功效不足，僅描述」。**本規格 T_train=2e4、n_S≤256 下 r=0.9 屬此情形：只作描述性報告、不作判定依據**，與 REQ-OBJ-02 的探索性定位一致；仍訓練（499 世代）與報告，並照實記錄 |
| 暖啟動 | — | 由 NAIVE pilot 決定（REQ-S1-20） | 見 REQ-OPT-07 |

- **REQ-OPT-13（n_S 涵蓋範圍的揭露，新增，v1.2）**：最終報告與凍結檔（REQ-S1-14）必須揭露：n_S=256 依「σ 三種的中位數最壞值」滿足 REQ-OPT-09，但**單一隨機策略對的最壞情況**（`sd_CRN`=0.034465）需要 n_S≈565（`v2.json: nS_need["0.5"].nS_formula_continuous_pair`），**目前無法涵蓋**；因此個別訓練 run 的 CMA-ES 排名雜訊在最壞情形下可能高於 0.25·MDE_est 的設計目標。r=0.9 需 n_S 數千，只作描述（REQ-OPT-11）。此揭露不改變任何判定規則。

**GPU 時數估算（v1.2，每個 run 依上列 mean／p95 重算；單位：小時，四捨五入到 0.1）**

| 階段 | run 數 | 世代 | mean 合計 | p95 合計 |
|---|---|---|---|---|
| 每個 run（參考） | 1 | r=0.5：416；r=0.9：499 | 3.4（r=0.9：4.1） | 4.8（r=0.9：5.7） |
| S1（REQ-S1-10，v1.3） | 12（範圍 9–12） | 416 | 41.2（30.9–41.2） | 57.6（43.2–57.6） |
| NAIVE pilot（REQ-S1-20） | ≤2 | 416 | 6.9 | 9.6 |
| 主實驗 | 6（r=0.5 ×3、r=0.9 ×3） | 416／499 | 22.6 | 31.6 |
| 診斷（REQ-S1-19 等） | 7 | 416 | 24.1 | 33.6 |
| **核心合計** | 27 | — | **約 94.8 小時**（S1 範圍下 84.5–94.8） | **約 132.4 小時**（118.0–132.4） |

計算式：run 小時＝(G_gens×20000×t_step＋⌊G_gens/50⌋×132.724 s)/3600，t_step 取 1.35901 ms（mean）或 1.94808 ms（p95）；階段小時＝run 數×該值，主實驗＝3×(416 世代)＋3×(499 世代)；M4 錨點（r=0.99）以 416 世代計入 S1 的 12 個 run。**v1.3**：S1 由 22 個 run 降為 12 個（族外 3×{0.01, 0.02}＋族內 3＋盲區 1＋錨點 2；run 數 N＝3·|G_s|＋6，|G_s|∈{1,2} ⇒ 9–12；每個 run 取 mean 3.436／p95 4.797 小時），核心由 37 個降為 27 個 run；v1.2 的「約 129／約 180 小時」與 S1 的「75.6／105.5 小時」作廢。校準（REQ-S1-28）只用參考策略與手寫策略的模擬，不含 RL 訓練，不計入上表。p95 為保守上界（機器高負載）。v1.1 的 66.6 小時與 notes 中「約 120 小時」皆為較粗的估算，以本表為準。啟動開銷、續跑重做與失敗重跑未計。

---

## 6. Seeds 規範與手寫基準

### 6.1 Seeds
- **REQ-SEED-01（分離）(revised v1.3)**：環境 seeds：訓練 `0–999`（1000 個）、驗證 `2000–2031`（32 個）、**val2 `3000–3031`（32 個，v1.3 新增，REQ-SEED-08）**、測試 `5000–5031`（32 個）。frontier 的三組 generator 實際種子為 `s, s+10^6, s+2·10^6`；**四個**切分在三組下的實際種子集合必須兩兩不相交（寫成測試）。`seeds.require_split(seeds, split)` 的 `split` 必須接受 `"train"、"val"、"val2"、"test"`，並對不屬於該切分的 seed 拋 `ValueError`（含 val2 對其他三者的互斥）。**重播專用 seeds** `1000–1031`（stage2b 的 `consist_M4`，REQ-ENV-21）不屬任何切分，只用於重播，不得用於訓練、選取或報告 G。
- **REQ-SEED-02（用途）**：訓練 seeds 只用於 CMA-ES 的適應值、漏洞套件的校準、CRN 量測；驗證 seeds 只用於選超參數、選 run、選檢查點、選手寫策略、算 MDE；val2 seeds 只用於漏洞套件的 knob 校準與選擇（REQ-S1-28、SEED-08）；測試 seeds 只用於最終報告，且每個（cell, 策略）組合的最終評估只做一次，在凍結檔簽署（§8.5）之後。
- **REQ-SEED-03（程式強制）(revised v1.1)**：訓練、驗證、選取模組**以及 `s1.py`、`vulns.py` 等其餘所有模組**不得匯入或引用測試 seeds 常數（唯一例外見 REQ-SEED-06）；訓練函式遇到 ≥2000 的 seed 立即拋錯（靜態檢查＋執行期檢查）。
- **REQ-SEED-08（val2，新增，v1.3）**：`3000–3031` 登記為 **val2**，來源：紅方（漏洞套件）決定 4（換新 seeds 後 7 組校準中有 2 組落在區間外，屬贏家詛咒）。規則：(a) 與 train、val、test 兩兩互斥（REQ-SEED-01）；(b) **只能用於校準與選擇**：漏洞 knob 的校準與挑選、校準階段的 Δ 與 G_HW 計算（REQ-S1-28）；不用於 RL 訓練、檢查點選取（REQ-OPT-06 仍只用 val）、MDE（REQ-MET-06 仍只用 val 的 32 個 seeds）、主實驗的手寫策略選取（REQ-HW-02 仍只用 val）；(c) **永遠不得作為測試集**：`final_eval` 與任何「測試評估」函式遇到 val2 seeds 必須拋錯；val2 不得出現在任何報告為「測試結果」的數字中；(d) 紅方的實驗（gpujob #77–#79）已用過這些 seeds 偵察，因此它們不再是未見資料，更加不能當測試集。
- **REQ-SEED-04（訓練 seed 與 run）**：每個設定 3 個訓練 run：`run_id ∈ {0,1,2}`，演算法種子 `9000+run_id`（與環境 seeds 無關）。3 個 run **全部報告**，不得只報最好的。
- **REQ-SEED-05（驗證選、測試評）**：RL 與手寫策略一律「在驗證集選、在測試集評」。
- **REQ-SEED-06（測試 seeds 的唯一入口，新增，v1.1）**：測試 seeds 的評估**只准寫在專用模組 `arbitration/rl/final_eval.py`**；該檔不列入 T-27 的 AST 掃描，其他模組（包含 `s1.py`）一律禁止匯入或引用測試 seeds。`s1.py` 需要最終評估時，呼叫 `final_eval`。
- **REQ-SEED-07（seed_block 的範圍與唯一性，新增，v1.2）**：`train.seed_block(run_id, g, n_S)` 在 `n_S` 不在 `[1, 1000]` 時必須拋出 `ValueError`（n_S>1000 會使區塊繞回而重複）；且**同一區塊內的 seed 必須唯一**（含橫跨第 1000 個位置邊界的區塊，如 g=31）。唯一性由 T-22 斷言；範圍檢查需另有測試（見 tdd-map）。

### 6.2 方向一的歷史 seeds
方向一的調參用 seeds 0–15、評估用 5000–5031（`quota.meta`）。因此測試 seeds 5000–5031 與方向一的評估 seeds 相同：方向一的數字可與本方向的測試結果直接對照，但**方向一在這些 seeds 上選了 G 最大的策略**，上偏，不得當作 G_手寫 的無偏值。

### 6.3 G_手寫的精確定義
- **REQ-HW-01（候選集 S_HW）(revised v1.3)**：自方向一的策略表（`scripts/run_quota.py: strategies()`，共 45 個，含 honest）中取「**非 oracle、對 M3C 有作用、且不是 honest**」者 32 個（**排除 H0**，commander 決定 6；H0 的得利恆為 0，納入會把手寫基準的下限人為壓在 0），**再加入細網格 4 個**（REQ-HW-06）：`dz=2.5`、`dz=3.0`、`dz=4.0`、`b=1.0`，共 **36 個**。完整清單與個數：
  - `b=`：0.05、0.1、0.2、0.3、0.5、**1.0**（6 個）
  - `dz=`：0.01、0.02、0.03、0.05、0.07、0.1、0.15、0.2、0.3、0.5、1.0、2.0、**2.5、3.0、4.0**（15 個）
  - `dzcrit=`：0.5、1.0、1.5（3 個）
  - burst：`burst dz1 W10 H90`、`burst dz1 W30 H170`、`burst dz2 W10 H490`（3 個）
  - `edge_uninf thr band=`：0.05、0.1、0.2（3 個）
  - `edge_uninf dz band=0.1 dz=`：0.1、0.3（2 個）
  - stealth：m∈{1.0, 2.0}×f∈{0.5, 0.9}（4 個）
  - 合計 6＋15＋3＋3＋3＋2＋4＝**36**；含 H0 的版本（REQ-HW-02(a)）為 37。粗體的 4 個不在 `strategies()` 內，須由 `arbitration/rl/hw.py` 另行定義（不修改 `scripts/run_quota.py`）。
  - 排除：honest（H0）1 個、`edge_oracle` 4 個（oracle）、`quota_savemin_oracle` 4 個（oracle）、`quota_compress_uninf` 4 個。**排除 quota_* 的原因**：它們在非配額機制（M3C、M4）下是空操作，gain 恆為精確的 0.0（見 `quota.json: main[88].gains_by_strategy` 與 `main[14].gains_by_strategy` 中全為 0.0），等價於把 H0 偷渡進來（見 §10 的 G=0.0000 檢查）。名稱集合＝`strategies()` 的 45 個扣除 13 個（H0＋8 oracle＋4 `quota_compress_uninf`）再加 4 個細網格。
  - `stealth` 以 `liar_flag`（被標記）重置內部狀態，而被標記必然使自己被暫停（`susp` 上升），因此屬於公開資訊可實現（REQ-HW-04）。
- **REQ-HW-02（選取規則）(revised v1.3)**：在驗證 seeds（2000–2031），方向二模式（T=100000、記分到 T−L）下，對 S_HW 的 **36** 個策略（REQ-HW-01）算 paired 平均 G；取最大者（平手取清單順序較前者）。`G_手寫` 為被選策略在**測試 seeds** 上的 paired G；由於 H0 不在候選集，**G_手寫 可以是負值**（未截斷，REQ-MET-02）。**另外報告**：(a) `G_手寫^{+H0}`：候選集加入 H0（37 個）後同樣在驗證集選、測試集評，以及由它算出的 `D^{+H0}`（REQ-MET-04）；(b)「含 oracle 的最佳手寫」（再加 8 個 oracle 策略）作為參考。(a)、(b) 只作附帶報告，不用於 §9.2 的判定。
- **REQ-HW-03（含 v1.3 的 36 個）**：S_HW 的策略在本模擬器中須以轉接策略重跑（記分窗同 REQ-ENV-15），不得直接引用方向一的數字。
- **REQ-HW-04（無資訊可實現性）**：S_HW 每個策略都需在文件中標註其使用的資訊是否屬於公開版（附對照表）；不屬於者須剔除並回報。
- **REQ-HW-06（細網格，新增，v1.3）**：紅方（漏洞套件）實驗（gpujob #77–#79，新 seeds 3000–3031）顯示：粗網格的 dz 最大只到 2.0、b 最大只到 0.5，手寫族在更大的灌水量上仍有得利；加入 dz∈{2.5, 3, 4}、b=1 後，O2 的 Δ 變為 −0.001、O3 為 +0.0016、D4 的 G_HW/G* 達 0.92–0.96（此為紅方的偵察值；登錄值以重新校準後的 `vuln_suite.json` 為準）。規則：(a) 新增的 4 個策略皆為「只用自己的 u 的常數偏移」，屬公開資訊可實現（REQ-HW-04 的對照表須列入）；(b) S_HW 的 36 個清單用於**所有**手寫基準：主實驗的 G_手寫（REQ-HW-02）、漏洞增量 Δ 的 G_HW^V（REQ-S1-02）、漏洞分類（REQ-S1-26）；舊的 32 個清單不再作為任何主結果；(c) 含 H0 的版本（37 個）另外報告（REQ-HW-02(a)），不用於判定；(d) 清單凍結（REQ-S1-14）後不得再增減；(e) 細網格使手寫族更強，使「RL 對手寫最佳」的比較**對 RL 更保守**，最終報告必須揭露（REQ-STOP-06）。
- **REQ-HW-05（burst 時鐘的揭露，新增，v1.2）**：S_HW 中的 `burst` 策略（3 個）使用 **agent 自己的回合計數**（自己的時鐘）。agent 本來就知道自己已經玩了幾輪，這不是機制洩漏，因此 commander 判定列為公開策略（REQ-HW-04 的可實現性成立）。但 RL 的觀察不含時間（REQ-ENV-14），手寫族因此比 RL **多了時間資訊**；這使「RL 對手寫最佳」的比較**對 RL 較為保守**（若 RL 不贏手寫，部分原因可能是少了時間資訊）。最終報告必須揭露此點（同 §9.3 共同限制句）。

---

## 7. 指標

- **REQ-MET-01（G，paired）**：對每個 seed s，`G_s = (U_pol(s) − U_hon(s))/T_score`，`U` 為 `util_b[:,1]` 在 T_score 的快照；`U_hon` 為同一 seed 的誠實（θ=0）結果。`G = mean_s G_s`（32 個測試 seeds）。
- **REQ-MET-02（未截斷）**：G 不得截斷於 0、不得取 max(·,0)；負值照實報告。（「未截斷」＝不 clip。與「尾端截斷」無關。）
- **REQ-MET-03（95% CI）**：`G ± t_{0.975,31}·SD(G_s)/√32`，`t_{0.975,31}=2.0395`。與方向一比對（REQ-ENV-07）時使用方向一的 1.96 公式。
- **REQ-MET-04（G_RL、G_手寫、差值）(revised v1.3)**：
  - `G_RL`：該 cell 的 3 個 run 各自在測試 seeds 上的 G 全部報告；**主值**為驗證集選出的那個 run（REQ-SEED-05）的測試 G。
  - `G_手寫`：REQ-HW-02（36 個策略，排除 H0）。
  - `D = G_RL − G_手寫`：逐 seed 取 `D_s = G_{RL,s} − G_{HW,s}`（同 seed 配對），`D ± t_{0.975,31}·SD(D_s)/√32`，CI 為 `[D_lo, D_hi]`。
  - 另報告 `D^{+H0}`（以 `G_手寫^{+H0}` 為基準，同式計算），不用於判定。
- **REQ-MET-05（R/T 與誤罰率）**：
  - `R/T`：誠實 R/T＝誠實（θ=0）時 `reg_b/T_score`（對照 JSON `R_hon_b`）；另報告被選策略下 `reg_b/T_score`（系統遭偏離者造成的後悔）。
  - 誤罰率 α：沿用方向一定義 `false_punish_frac`＝誠實運行中 `pen`（不合格輪數）對三位 agent 平均後除以 T_score（對照 JSON `false_punish_frac`）；另報告偏離者策略下 agent 0、2 的 `pen/T_score`（誠實者被連累的暫停比例）。
- **REQ-MET-06（MDE，80% 檢定力）(revised)**：`MDE` 定義為「在 32 個 paired seeds、雙尾 α=0.05（CI 下界 >0 的檢定）下，檢定力為 80% 的真實效果大小」。計算：
  `MDE = κ · σ̂_val / √32`，其中 `κ = 2.8922` 是使 `P(|T_{31}(δ)| > t_{0.975,31}) = 0.80` 的非中心參數 δ（`t_{0.975,31}=2.0395`；以 `scipy.stats.nct` 數值求解；正態近似 `z_{0.975}+z_{0.80}=1.9600+0.8416=2.8016` 低估約 3%，只在規劃時使用）。`σ̂_val` 為驗證 seeds 上逐 seed 配對量的樣本標準差（ddof=1）：偵測用取 `G_s`（得 `MDE_G`）、Q-main 與漏洞偵測取 `D_s = G_{RL,s} − G_{HW,s}`（得 `MDE_D`），其中 RL 為驗證集選出的檢查點、HW 為驗證集選出的手寫策略。MDE 在看到測試結果之前由驗證集算出並寫入該 run 的結果檔（欄位 `mde_frozen`）；測試評估函式必須讀取該值，不得重算。**低於 MDE 的漏洞大小只作描述，不納入通過判定**（§8.4、REQ-S1-22）。
- **REQ-MET-07（「找到」的定義）(revised)**：
  - **G 規則**（錨點、「找到任何得利」）：測試 seeds 上 `G 的 CI 下界 > 0` 且 `G_RL ≥ MDE_G`。
  - **D 規則**（漏洞偵測、Q-main）：測試 seeds 上 `D 的 CI 下界 > 0`（`D_lo > 0`）且 `D ≥ MDE_D`。漏洞偵測時 `G_HW` 取該漏洞環境中的驗證集最佳手寫策略（REQ-S1-02）。
- **REQ-MET-08（複現，S3）(revised)**：複現數 `k` ＝ 3 個 run（`run_id` 0、1、2）中，測試 seeds 上 `D_lo > 0` 的個數（`D` 以各 run 自己的 paired 差值計，基準同為被選手寫策略）。`k` 與主值一起進入 §9.2 的判定表：`k=3` 為完全複現，`k=2` 為部分複現（可稱找到但須附註），`k=1` 為未複現。
- **REQ-MET-09（結果登錄）**：所有報告的數字由腳本自 JSON 取值並登錄 `docs/direction2-number-audit.md`（格式與方向一的 number-audit 相同：數字、出處、JSON 路徑、是否一致）。

---

## 8. S1：檢測力標定（硬性關卡）

目的：回答「這個紅隊抓得到什麼？」。之後任何「沒找到漏洞」的結論，只能寫成對本套件的描述（§9.3）。S1 只做 M3C、只做 r=0.5（功效主格；r=0.9 的結論標示為未標定，§1.2）。**唯一例外**：M4 錨點依 REQ-S1-09 固定在 stage2b 的格點 r=0.99（ε=0.2、L=20），因為已知得利只在該格被量到。S1-09 與 S1-10 對 r 的規定一致（均如此）。

### 8.1 漏洞套件組成
- **REQ-S1-01（組成與分類）(revised v1.3)**：套件登記 11 項，每項有一個 `category` 欄位（取值固定為下表，REQ-S1-26 說明依據）。「族內／族外」是相對於**手寫族**（S_HW 36 個，REQ-HW-01）；「類內」是相對於**特徵類別**（公開版觀察可見）。**類內漏洞**＝族外＋族內共 6 個。

  | id | 來源 | `category`（中文） | 計入偵測統計？ |
  |---|---|---|---|
  | D1、D2、O1 | designer_2（D1、D2）、自動算子（O1） | `out_of_family`（族外） | 是（通過標準的分母，n=3） |
  | O2、O3、D4 | 自動算子（O2、O3）、designer_2（D4） | `in_family`（族內） | 只描述，不計入通過判定 |
  | D3 | designer_2 | `in_family_known`（族內已知） | 否，單獨報告（不跑 RL，見 REQ-S1-10） |
  | O5 | 自動算子 | `blind`（盲區） | 否，只描述 |
  | NAIVE、M4 | 方向一已知得利 | `anchor`（錨點） | 錨點通過條件（REQ-S1-11） |
  | designer_1 的漏洞 | designer_1 | `contaminated_excluded`（受汙染未計入） | 否，只在最終報告揭露（REQ-S1-23） |

  數量固定，不得擴大、不得補招設計者。**移出套件**（commander，notes「漏洞套件完成與 commander 決定」）：O4（最高約 0.0027）、O6（最高約 0.0036）無法校準到最小登錄大小 0.01；B1、B2 規格只給例子、未實作，**不補**，列為限制（REQ-STOP-06）。`blind` 布林欄位保留，且 `blind=true ⇔ category=="blind"`。
- **REQ-S1-02（大小＝超過手寫最佳的增量）(revised v1.3)**：對漏洞環境 V（機制變異＋旋鈕 κ），令 `G_HW^V` 為 S_HW（**36 個**，排除 H0，REQ-HW-01）在 V 中的 paired 最佳 G，`G*^V` 為參考利用策略在 V 的 paired G。**漏洞大小 `s := Δ(κ) = G*^V − G_HW^V`**（即「超過手寫最佳的部分」）。登錄的大小為 **s ∈ {0.01, 0.02}**（v1.3：原 {0.003, 0.005, 0.01, 0.02} 中，0.003、0.005 低於 MDE，從未校準也不會跑，從規格中移除，與 `vuln_suite.json` 現況一致；`G_s ⊆ {0.01, 0.02}`，見 REQ-S1-22）；盲區 O5 只登錄 s=0.02。
  - **兩種 Δ**：`Δ_fine` ＝以 36 個細網格算的 Δ（正式定義）；`Δ_coarse` ＝以原 32 個清單算的 Δ（舊登錄的基準）。**登錄時兩者都記錄**。
  - **大小的判定基準（`size_basis`）**：`out_of_family`、`blind` 以 `Δ_fine` 為準；`in_family` 以 `Δ_coarse` 為準（理由：其 `Δ_fine` 在細網格下可為零或負——O2 為 −0.001——此即「族內」的定義，無法以 `Δ_fine` 校準大小；`Δ_coarse` 僅作為大小的標籤，不代表手寫族打不過它）。`in_family_known`（D3）的 Δ 為負，無法校準大小，只登錄 `Δ_fine`、`Δ_coarse`、G_HW 與 G*。**此基準的選擇為 v1.3 的規格解讀，列於 §17 待 commander 確認的註記**。
  - **驗收**：校準在 **val∪val2（64 個 seeds）** 上進行（REQ-S1-28）；基準 Δ 的合併均值須落在 `[0.8s, 1.2s]` 內，且其合併後 paired 95% CI 下界 >0。
- **REQ-S1-03（登錄檔欄位）(revised v1.3)**：每個（漏洞, 大小）在凍結登錄檔 `results/direction2/vuln_suite.json` 中須有：`id, kind, category, source, operator, knob, size, blind, size_basis, ref_policy, G_star_val (mean, CI), G_HW_best_val (策略名, mean；36 個清單), ratio_HW_over_Gstar, delta_fine (mean, CI), delta_coarse (mean, CI), delta_val (＝size_basis 對應者, mean, CI), delta_val_only, delta_val2_only, calib_seeds ("val+val2"), honest_drift (mean, CI), knob_monotone (bool；D4 預期 false), delta_train64`。`val`／`val2` 欄位分別以驗證 seeds、val2 seeds 計，僅作透明度（不另設驗收）；`honest_drift`＝漏洞環境中誠實（θ=0）的 `util_b` 減去基礎環境的（paired，REQ-STOP-06）。`ref_policy` **只使用公開日誌欄位且不得使用絕對時間**（REQ-S1-27；機器檢驗：靜態檢查＋執行期檢查，並通過 REQ-OBS-06 式的私有量擾動不變性檢查）；盲區漏洞（O5）的參考策略可使用公開版之外的**更長歷史**，其欄位照常登錄並標示 `blind=true`，但同樣不得使用絕對時間與私有狀態。
- **REQ-S1-04（手寫排除，`G_HW < 0.5·G*`）(withdrawn)**：被 REQ-S1-02 的「增量」定義取代（漏洞大小本身就是超過手寫最佳的部分，不再另設 0.5 比例規則）。
- **REQ-S1-05（恆等）(revised v1.3)**：漏洞算子取「無變異」時，漏洞環境與 §2 的環境逐位元相等。**「無變異」的旋鈕值**：O1、O2、O4、O6 為 κ=0；**O3 為 C=∞、O5 為 g=∞**（commander 決定：O3 的 C 越小越強、O5 無 κ，故不能用「旋鈕=0」）；設計者漏洞 D1–D4 取各自的恆等 knob（登錄於 `vuln_suite.json`）。**v1.3**：**O 系列（O1、O2、O3、O5，以及程式碼仍保留的 O4、O6）除短 T 外，必須另在 T=1e5、L=2500（方向二記分窗與方向一重播窗兩種 `T_score`）驗證恆等**（紅方決定 6）；D 系列沿用 v1.2 的驗證長度並建議同步補驗。[gpu] 測試，依 REQ-GPU-08 分塊（每塊 ≤25 分鐘）。

### 8.2 自動生成與盲區
- **REQ-S1-06（自動生成器）(revised v1.3)**：進入套件的自動漏洞為算子 **O1、O2、O3（`category` 依 REQ-S1-01）與 O5（盲區）**；O4、O6 的定義與程式碼保留（T-37 仍驗證其公式），但**因無法校準到 0.01 而不在套件內**（見 REQ-S1-01）。每個算子在旋鈕範圍內校準出登錄的大小；校準程序是確定性的（相同 seeds、相同搜尋序列 ⇒ 相同旋鈕，REQ-S1-28）。若某算子無法在旋鈕範圍內達到登錄大小，須回報 commander 決定（不自動遞補）。算子定義（對 M3C 的稽核區塊）：
  - O1 連勝免稽：自己已連續得標 m=3 輪時，`p_el ← p·(1−κ)`。
  - O2 CUSUM 洩漏：`S ← (1−κ)·max(S − resid − kref, 0)`。
  - O3 殘差封頂：`resid ← max(resid, −C)`（單次稽核最多貢獻 C 的證據）。旋鈕為 C（**C 越小，封頂越強、漏洞越大；C=∞ 為無變異**）。
  - O4 乾淨後縮短暫停（未入套件）：定義見 REQ-S1-07。
  - O6 標記後縮短暫停（未入套件）：被標記時暫停長度 `L ← (1−κ)L`。
- **REQ-S1-07（特徵盲區漏洞，1 個）(revised v1.3)**：標示 `blind=true`、`category="blind"`，用來標出「類內／類外」界線；不計入偵測統計與通過判定，結果只描述。套件內只有 **O5**：久未得標歸零——自己連續 g=20 輪未得標時 `S ← 0`（超出歷史窗 k=8；旋鈕為 g，g=∞ 為無變異；紅方在新 seeds 上測得 Δ=+0.0233）。O5 只在 s=0.02 跑（REQ-S1-10、§17 P-4）。
  - 原 B1（絕對時間相依）、B2（長記憶相依）**未實作、不補**（commander 決定 2）；因此**本套件對「絕對時間」與「長記憶」型盲區沒有任何證據**，只有一個盲區範例 O5（REQ-STOP-06、§9.3）。
  - O4（乾淨後縮短暫停：自己連續 m'=50 次被稽核且未被標記後，下一次暫停長度 `L ← (1−κ)L`；稽核是靜默的，公開版看不到「被稽核」事件）原列盲區，因最高只能校準到約 0.0027（<0.01），已移出套件；定義保留於程式碼。

### 8.3 已知錨點
- **REQ-S1-08（NAIVE 錨點）**：M3C，`p=0`，r=0.5。已知 G＝方向一最佳（dz=2.0）0.2793 ± 0.0004（`quota.json main[4].Gmax`）。以驗證集重新量測 `G*_ref`（本模擬器）。錨點以 G 規則（REQ-MET-07）判定，不用增量定義（NAIVE 的已知得利本來就在手寫族內）。
- **REQ-S1-09（M4 rw=0 錨點）(revised)**：M4，`eps=0.2, L=20, rw=0`，**r=0.99**（格點出自 `stage2b.json: consist_M4[12]`，見 REQ-ENV-11），錨點策略為公開資訊的常數灌水 `b=0.3`。W1 在驗證 seeds 以本模擬器**重新量測** `G*_ref`（paired，CI）；JSON 的 +0.1209 ± 0.0014（`consist_M4[12].gain_b03`）只作對照，不是驗收值（seeds 不同）。若重測的 `G*_ref` CI 下界 ≤ 0，W1 重測 `G*_ref` 的 CI 下界 ≤0 時，M4 錨點降為資訊性，不計入 S1-17 的通過條件（只剩 NAIVE 錨點），並照實記錄（§17 P-5）。錨點以 G 規則判定。
  - 此錨點是 S1 中唯一不在 r=0.5 的 run（REQ-S1-10 同）；M4 的 `T=100000、L=20`，記分窗 `T_score=99980`。

### 8.4 訓練與偵測
- **REQ-S1-10（流程與 run 數）(revised v1.3)**：每個要跑的（漏洞, 大小）與每個錨點訓練 **1 個 run**（`run_id=0`，公開版；漏洞與 NAIVE 錨點為 r=0.5，M4 錨點為 r=0.99；第 5 節定稿的預算），以驗證集選檢查點，在測試 seeds 評估一次。偵測成立 ⇔ REQ-MET-07 的 D 規則（漏洞）或 G 規則（錨點）。**要跑的集合**（`G_s` 見 REQ-S1-22，預估 {0.01, 0.02}）：
  - **族外**（D1、D2、O1，3 個）× 閘門大小集合 `G_s`；
  - **族內**（O2、O3、D4，3 個）× s=0.02；
  - **盲區**（O5，1 個）× s=0.02；
  - **2 個錨點**（NAIVE、M4）；
  - **不跑**：D3（族內已知，Δ 為負、無大小可校準，只登錄與報告 REQ-S1-03 的欄位）、designer_1（受汙染）、O4、O6、B1、B2、以及低於 MDE 的大小（P-8 不補跑）。
  - **run 數 `N = 3·|G_s| + 3 + 1 + 2 = 3·|G_s| + 6`；預估 |G_s|=2 → N=12**（|G_s|=1 → 9）。GPU 時間見 §5.5 估算表與 REQ-GPU-05（12 個 run：mean 41.2／p95 57.6 小時）。
  - **優先順序**：先 s=0.02（族外 3＋族內 3＋盲區 1＋錨點 2＝9 個 run），再 s=0.01（族外 3 個，此即全部要跑的 run）。第 14 天結束時尚未完成者記為「未完成（計為未偵測）」。
- **REQ-S1-11（錨點通過）(revised)**：錨點通過 ⇔ 測試 G_RL ≥ 0.8·G*_ref 且 CI 下界 >0（且 ≥ MDE_G）。M4 錨點若依 REQ-S1-09 降為資訊性，則不納入通過判定（只剩 NAIVE 錨點），其結果仍照實記錄。
- **REQ-S1-20（NAIVE 學習性 pilot，新增；revised v1.2）**：W1 在凍結前，於 NAIVE 錨點（M3C、p=0、r=0.5）上以第 5.5 節定出的預算（λ、T_train、n_S、G_gens）做 1 個**冷啟動** run，只用訓練 seeds 與驗證 seeds：若驗證 G ≥ 0.8·`G*_ref`（驗證重測，預期約 0.28）→ 凍結「冷啟動」；否則做 1 個**暖啟動** run（同預算）：達標 → 凍結「暖啟動」；仍未達標 → 判定 S1 不可行（REQ-STOP-02），走 §9.2 的 J2。pilot 結果與兩條路的登記都寫入凍結檔；pilot 不使用測試 seeds。 pilot 另須附上「永遠報 1」基準（REQ-S1-25）。
- **REQ-S1-22（閘門大小集合，新增；revised v1.3）**：W1 的 M-3（REQ-MEAS-03）以代理策略對量測 `MDE_D,plan`（r=0.5）並寫入凍結檔。**閘門大小集合 `G_s = { s ∈ {0.01, 0.02} : s ≥ MDE_D,plan }`**（v1.3：登錄大小只有 0.01、0.02，REQ-S1-02）；`0.02 ∉ G_s`（即 `MDE_D,plan > 0.02`）⇒ S1 無法以本規格的通過標準判定，觸發 REQ-STOP-02。`G_s` 只決定**族外**漏洞要跑哪些大小；族內與盲區固定只跑 s=0.02。每個 run 實際的偵測仍用它自己的 `MDE_D`（REQ-MET-06；若某 run 的 `MDE_D` 大於其大小，該 run 自然判為未偵測）。目前的估計：`MDE_est(r=0.5)=0.0058`（REQ-OPT-09）⇒ `G_s={0.01, 0.02}`。

### 8.4b 套件分類、參考策略公開性與校準（v1.3 新增）
- **REQ-S1-26（套件分類與依據，新增，v1.3）**：`category` 的指派**固定如 REQ-S1-01 的表**（commander，紅方決定 1–3），不得因重新校準而更動；其**依據**是細網格（REQ-HW-01）下兩個量：(i) `ratio_HW_over_Gstar = G_HW^V / G*^V`；(ii) `Δ_fine = G*^V − G_HW^V`。紅方偵察值（新 seeds，僅作依據，登錄值以重新校準為準）：
  - 族內：O2 的 `Δ_fine` ≈ −0.001、O3 ≈ +0.0016（相對於 s=0.02 可忽略）、D4 的 `ratio` ≈ 0.92–0.96——手寫細網格已能拿到參考策略的絕大部分得利；
  - 族內已知：D3 的 G_HW 為基準的 3–5 倍（手寫族已大量利用），參考策略的 Δ 為負；
  - 族外：D1、D2、O1 在細網格下仍有 `Δ_fine>0`、且 `ratio` 明顯低於族內三者（紅方：真正屬於族外的只有這三個）。
  - **機器檢查（凍結前，重新校準之後）**：(a) 族外項目須滿足 REQ-S1-02 的驗收（`Δ_fine` 於 `[0.8s,1.2s]`、合併 CI 下界 >0）；(b) 登錄檔每一項必有 `category` 且取值屬於六種之一，`category=="blind" ⇔ blind==true`，`contaminated_excluded` 項不得出現在任何 S1 run 或統計；(c) 若重新校準後的登錄值與 `category` 矛盾（例如族外項目的 `Δ_fine` CI 下界 ≤0），**停止並回報 commander**，不自動改分類、不自動遞補。
- **REQ-S1-27（參考策略只能使用公開日誌欄位，新增，v1.3；適用範圍於 v1.3 凍結前修訂）**：**適用範圍：只適用於計入統計的漏洞（族外 D1、D2、O1 與族內 O2、O3、D4）：其參考策略不得使用時間，必須在 RL 的觀察範圍內就能實作。盲區漏洞（O5）本來就是在測 RL 觀察範圍以外的東西，其參考策略可以使用 agent 自身的時鐘（例如 `t mod (W+Z)`），但仍不得接觸私有狀態（`cus`、`pend_a` 等）；手寫基準亦可使用自身時鐘，須揭露（REQ-HW-05）。**以下規定對計入統計的漏洞全部適用；`ref_policy` 的輸入只能是 REQ-OBS-10(a) 的**公開日誌欄位**：`z_1、u_1`（自己的型別）、`v_1`（自己的報告）、`won_1`、`susp_1`（自己的暫停截止時間）以及機制常數（p、h、L、kref、r、C_MIN；M4 另含 ε、rw）；由這些欄位可導出的**相對量**（`since`、`rem`、窗內計數、EMA 等）允許。**不得使用絕對時間**（`t`、輪數、`t mod P` 等；REQ-OBS-08、ENV-14），亦不得接觸 `cus`、`pend_a`、`pend_v`、`flags`、`audits`、`sxy`、`sxx`、他人的 `u`／`v`／`susp`、抽籤。**D4 的參考策略**原用到絕對 t（`vulns.py` 的 `_LClimb`，紅方決定 5），**改用 f05 可重建的相對量**（`since`＝自己首次觀察到 `susp_1` 上升後經過的輪數，REQ-OBS-11）。驗證方式（兩者都要）：(a) **靜態檢查**：對 `ref_policy` 的實作做 AST 掃描，函式簽章與其讀取的狀態物件不含 `t/time/round/step/tick/clock` 之類欄位，也不含上列私有欄位名；(b) **執行期檢查**：(1) 私有狀態擾動不變性（同 REQ-OBS-06）；(2) **時間平移不變性**——兩個「公開日誌的相對內容相同、絕對時間不同」的狀態（例如在日誌前端補入不改變任何公開量的等長前綴，或於不同 t 重現同一相對歷史）得到相同輸出；(3) 以 REQ-OBS-10 的獨立重建器只讀公開日誌重建 `ref_policy` 的輸入，輸出逐位元相同。盲區 O5 的參考策略可用更長歷史（>k=8）與自身時鐘，但不得接觸私有狀態，並須通過 (b)(1) 的私有擾動不變性；(a) 的時間字樣掃描與 (b)(2) 的時間平移不變性對 O5 不適用（O5 launder 以 `t mod (W+Z)` 為相位，屬已揭露的合規例外）。D4 的參考策略改用恢復後年齡 `max(since−L, 0)`（`since` 為 f05 可重建的相對量，上限 2L），不使用絕對時間。
- **REQ-S1-28（knob 校準，新增，v1.3）**：
  - **校準資料**：knob 的選擇用 **val∪val2（驗證 2000–2031 與 val2 3000–3031 合併，共 64 個 seeds）** 上的 paired Δ（REQ-S1-02 的 `size_basis`）；粗搜尋可用訓練 seeds 0–63，但最終挑選與驗收只看 val∪val2。校準只用參考策略與手寫策略，**不用 RL**。從相鄰 knob 中挑選屬驗證資料的合法用途（commander 決定 5）。
  - **驗收**：基準 Δ 的合併均值落在 `[0.8s, 1.2s]`（沿用）**且**合併後 paired 95% CI（`t_{0.975,63}=1.9983`，64 個 seeds）下界 >0。
  - **掃描而非二分**：D4 的 knob 不單調（紅方決定 6），搜尋不得假設 Δ 對 knob 單調；登錄 `knob_monotone`，並於 REQ-STOP-06 揭露。
  - **確定性**：相同 seeds、相同搜尋序列 ⇒ 相同 knob；挑出的 knob 在凍結檔簽署後不得再改。
  - **範圍界定**：val2 只參與 knob 校準與 G_HW 的計算；MDE 仍只由驗證集的 32 個 seeds 算（REQ-MET-06）；RL 檢查點選取仍只用驗證集（REQ-OPT-06）；val2 不得進入任何測試結果（REQ-SEED-08）。
  - **失敗處理**：任一族外／盲區項目無法於登錄大小通過驗收 → 回報 commander，不自動遞補、不放寬區間（REQ-S1-06）。

### 8.5b 設計者漏洞來源、禁字掃描與 NAIVE 基準（v1.2 新增）
- **REQ-S1-23（設計者漏洞的來源與狀態，新增，v1.2）**：登記於 S1 的 4 個設計者漏洞（D1–D4）**一律以 designer_2 為準**，來源 `docs/direction2-blind-vulns-2.md`；其隔離驗證紀錄為 `results/direction2/designer_logs/designer_2/isolation_check.md`（隔離成立：只讀 `frontier.py` 與 `docs/direction2-vuln-format.md`，無 Bash／Grep／Glob 呼叫，禁字檢查 0 命中）。**designer_1**（`docs/direction2-blind-vulns.md`）**已確認受汙染**（contaminated：其可讀材料含特徵細節），`category="contaminated_excluded"`，**不計入 S1 檢測統計、不跑任何 S1 run，只在最終報告中作揭露**。兩位設計者的 prompt 與工具紀錄依 REQ-S1-21 保存。（此條只記來源與狀態，不描述漏洞內容。）
- **REQ-S1-24（盲設計者材料的禁字預掃，新增，v1.2）**：任何提供給盲設計者（或其他不得看到特徵細節的 agent）閱讀的材料（規格節錄、格式說明、程式碼片段、prompt），**交付前必須先用 REQ-S1-21 的禁字清單掃描並確認 0 命中**，掃描結果存檔（含掃描的檔案清單與雜湊）。**不得只檢查設計者的 prompt**：prompt 以外的可讀材料同樣要掃（第一次盲設計的洩漏即來自可讀規格 §8 本身的文字，而非 prompt）。禁字清單至少涵蓋 REQ-S1-21 所列字樣，並依材料實際內容補充（例如歷史窗長度、EMA 時間尺度、特徵編號）。
- **REQ-S1-25（NAIVE pilot 的「永遠報 1」基準，新增，v1.2）**：NAIVE 學習性 pilot（REQ-S1-20）須同時評估一個「永遠報 1」策略（`v1 ≡ 1`，adapter 形式、公開資訊可實現），並與**手寫最佳**（REQ-HW-02 在 NAIVE 錨點上選出者，dz=2.0 的 0.2793 為方向一的參考）在**相同的 T 與相同的 seeds** 下比較（配對、同一記分窗；pilot 只用訓練與驗證 seeds）。目的：解釋 smoke run 中 RL 得利 0.311 高於手寫最佳 0.279 的原因——(A) 方向一的手寫族最多只灌水 b≤0.8，RL 可直接報 v=1；(B) smoke 的 T=1e4 記分窗較短。判讀：若「永遠報 1」≥ 手寫最佳且接近 RL，則 0.311 是手寫族的上限所致，不是 RL 找到新漏洞；兩者差異在 T 間的變化亦須報告。此基準只用於解釋 pilot，不改變 S1-11 的錨點通過標準（仍用 `G*_ref` 的 0.8 倍）。

### 8.5 盲設計流程與資訊隔離
- **REQ-S1-12（設計者）(revised v1.2)**：4 個設計者漏洞（D1–D4；其中 D1、D2、D4 計入偵測統計，D3 為族內已知、單獨報告，見 REQ-S1-01）由獨立 agent（各自獨立 context、不互通）設計。
  - **能看到**：本規格 §1.3、§2（環境、M3C 參數、時序）、§3.1（公開版資訊表，變數層級）、變異介面（§8.1 的鉤子：稽核機率、殘差、CUSUM 更新、暫停長度）、模擬器 API、沙盒公開歷史介面。
  - **不能看到**：§4 的特徵清單、k=8、EMA（λ_e）等特徵細節、策略類別與參數量、§5 優化器與預算、方向一的手寫策略表（`strategies()`）、`docs/direction2-notes.md`、任何 RL 訓練日誌或結果。
  - 設計者須交付：變異程式（符合鉤子介面）、各登錄大小的旋鈕、`ref_policy`（以公開歷史介面寫成）、自述的增量。驗收由 REQ-S1-03、S1-05 的自動檢查完成（驗證集增量與 CI 判定）。
  - 每個設計者的 **prompt 原文與工具呼叫紀錄必須保存**（REQ-S1-21）。
  - **盲設計者可讀的材料必須事先掃描**：見 REQ-S1-24。v1.2 起，設計者可讀的是清除過的格式說明 `docs/direction2-vuln-format.md` 與 `frontier.py` 的 M3C 部分，**不再是本規格 §8.1 原文**（第一次盲設計因 §8 的 REQ-S1-07 文字含特徵細節而受汙染，見 REQ-S1-23）。
- **REQ-S1-13（特徵與規格先凍結）**：§4 的特徵表、k、P_MAX 與 §5 的規則在設計者交付前即已寫定；凍結後禁止依漏洞內容修改特徵、k、P_MAX 或訓練預算（否則等於對測試答案調參）。
- **REQ-S1-14（凍結時點）(revised)**：W1 結束前（第 7 天）簽署凍結檔 `docs/direction2-freeze.md`，內容：`vuln_suite.json` 的 sha256、程式 commit、第 5.5 節定稿參數（λ、n_S、T_train、G_gens）、σ0、**暖啟動與冷啟動兩條路的登記與 NAIVE pilot 結果及採用的路**（REQ-OPT-07、REQ-S1-20）、`MDE_D,plan` 與 `G_s`（REQ-S1-22）、MDE 計算程式版本、**S_HW 36 個清單（REQ-HW-01）與 val∪val2 的校準紀錄（REQ-S1-28）、每個漏洞的 `category`（REQ-S1-26）**。**任何 S1 RL 訓練必須在凍結之後開始**（NAIVE pilot 與錨點的管線煙霧測試例外，但這些不得調整超參數、不得使用測試 seeds）。「在凍結之後」以 **git commit 順序**認定（REQ-S1-14 的凍結 commit 必須是第一個 S1 part 檔 commit 的祖先），不以檔案時間為準。凍結後若發現實作 bug，須記錄原因，且受影響的所有 run 重跑。
- **REQ-S1-21（設計者紀錄，新增）**：`results/direction2/designer_logs/{designer_id}/prompt.txt` 與 `tool_calls.jsonl` 必須存在並進 git；自動檢查：prompt 文字不含「k=8」「EMA」「λ_e」「f06」「P_MAX」「strategies()」等特徵細節與手寫策略表的字樣，且 `tool_calls.jsonl` 中沒有讀取 §4、§5、`strategies()` 或 `direction2-notes.md` 的紀錄。

### 8.6 偵測描述與通過標準
- **REQ-S1-15（偵測統計）(revised v1.3)**：以 D 規則（REQ-MET-07）判定每個漏洞是否被偵測。計算：(a) `y`＝s=0.02 時族外 3 個（D1、D2、O1）中被偵測者數（分母恆為 3）；`y'`＝s=0.01 時族外被偵測者數（若 0.01∈G_s）；(b) `x`＝s=0.02 時 6 個類內漏洞（族外 3＋族內 3）中被偵測者數（分母恆為 6）；(c) 族內 3 個（O2、O3、D4）逐一報告；(d) 盲區 O5 與 2 個錨點單獨報告；(e) D3 與 designer_1 不進入任何計數。未跑或未完成者計為未偵測（分母不變）。
- **REQ-S1-16（Clopper–Pearson 數字，僅作描述）(revised v1.3)**：族外的 `(y, 3, p̂, LB)` 與類內的 `(x, 6, p̂, LB)` 仍報告，`LB = Beta⁻¹(0.05; x, n−x+1)`（單側 95%；x=0 時 LB=0；x=n 時 LB=0.05^{1/n}；例：3/3 → **0.3684**；2/3 → 0.1353；6/6 → 0.6070；5/6 → 0.4182；舊 8/8 → 0.6877、6/8 → 0.4003）。**LB 只用來說明通過門檻 3/3 的來源（對應 CP 單側 95% 下界約 0.37），不得在結果措辭中作為對「其他漏洞的檢測率」的二項推論**（§9.3、commander 決定 6）。
- **REQ-S1-17（通過標準，統計定義，寫死）(revised v1.3)**：S1 **通過** ⇔ 同時滿足
  1. 兩個錨點通過（REQ-S1-11）；但若 M4 錨點依 REQ-S1-09 降為資訊性（W1 重測 `G*_ref` 的 CI 下界 ≤0），則只要求 NAIVE 錨點通過，M4 錨點照實記錄但不計入；
  2. 在 s=0.02，**族外 3 個漏洞（D1、D2、O1）全部被偵測（y=3，即 3/3）**（紅方決定 1：對應 CP 單側 95% 下界 0.05^{1/3}≈0.37；屬誠實但偏弱的標準）。
  族內（O2、O3、D4）、族內已知（D3）、盲區（O5）與 s=0.01 的結果**只描述，不影響通過與否**；描述性大小（低於 MDE）不納入通過判定，須在圖上標出 MDE。**找不到的情況**（y<3，或 NAIVE 錨點未過，或 W1 停損、第 14 天到期未完成）依停損規則（REQ-STOP-01／02）處理：S1 記為未通過，走 §9.2 的 J1／J2，**不延長、不補跑、不放寬**；族內結果不能補救也不能加重未通過。舊標準（8 個類內找回 ≥6；中間版 6 個類內找回 ≥5）作廢。
- **REQ-S1-18（輸出）(revised v1.3)**：`results/direction2/s1.json`：每個（漏洞, 大小）的 `category, x（或 detected）, G_RL (mean, CI), D (mean, CI), MDE_D, detected`；錨點的 `G_RL, G_star_ref, MDE_G, passed, informational`；彙總：族外 `(y, 3, p̂, LB)`（s=0.02 與 s=0.01）、類內 `(x, 6, p̂, LB)`（s=0.02）、族內逐項、盲區 O5、D3 的登錄欄位（不跑 RL）；檢測力曲線圖（族外兩個大小；其餘為單點，並標出 MDE）。

### 8.7 特徵充分性檢查（非硬性）
- **REQ-S1-19**：在 r=0.5、公開版、各 1 個 run，各做一次：(a) 去一組特徵消融（6 組：{f01,f02}、{f03,f04,f05}、{f06–f13}、{f14–f21}、{f22,f23}、{f24,f25}），(b) 64 步歷史對照（`won_{t−j}`、`d_{t−j}`，j=1..64，加 f01–f05、f22，F=134，P_DIAG 上限）。比較測試 G_RL 與主 run。未做或未通過時，結論只寫「在此特徵類別內」；**不阻擋流程**（判定 5）。

---

## 9. 停損與結果判定

### 9.1 W2 結束時 S1 的判準
- **REQ-STOP-01 (revised v1.3)**：第 14 天結束時以 REQ-S1-17 判定 S1；**與 REQ-S1-10 一致**：未完成的 run 記為「未偵測」（族外的 y 不增加，分母恆為 3；類內的 x 不增加，分母恆為 6）；未完成的錨點記為「未通過」。判定只看 s=0.02 的 **3 個族外漏洞（D1、D2、O1）** 與（計入的）錨點（REQ-S1-17）；族內、盲區、s=0.01 的 run 未完成只影響描述，不影響通過與否。
  - 通過 → 進入 W3 的主實驗，結果依 §9.2 判定表（J3–J8）。
  - 未通過 → 走 §9.2 的 J1／J2，**不延長**。主實驗（r=0.5、0.9 的 3 個 run）若已跑完，其結果仍報告，但只能稱為「候選結果（未經檢測力背書）」。
- **REQ-STOP-02 (revised)**：W1 關卡：下列任一成立，W1 結束時即停損，回報 commander，不進入 S1：(a) REQ-OPT-08 或 REQ-OPT-09 判定 r=0.5 不可行；(b) `MDE_D,plan > 0.02`（REQ-S1-22）；(c) NAIVE 學習性 pilot 的冷、暖啟動都未達標（REQ-S1-20）。停損後的結果走 §9.2 的 J2。

### 9.2 結果判定表（REQ-STOP-03，窮舉且互斥；決策函式 `classify` 須可測試）(revised)

輸入：`S1`（通過／未通過；**通過 ⇔ REQ-S1-17(v1.3)：s=0.02 時族外 3/3 被偵測且（計入的）錨點通過**；族外 y<3、錨點未過、W1 停損與到期未完成都算未通過）、主值 D（r=0.5，測試 seeds，驗證集選出的 run，paired；CI 為 `[D_lo, D_hi]`）、`MDE_D`（該 run 的驗證集計算值）、複現數 `k`（REQ-MET-08，含主 run）。D 不存在（主實驗未跑）時，視為「D 不顯著」。判定依 J1→J8 的條件欄，**恰有一列成立**。

| 列 | S1 | D 的條件（測試 seeds） | k | 結果 | 動作 |
|---|---|---|---|---|---|
| **J1** | 未通過 | `D_lo > 0` 且 `D ≥ MDE_D` | 不論 | (iii-c) 候選漏洞，未經檢測力背書（S1 未通過＝族外 y<3、錨點未過或停損） | 人工紅方查證；不得稱 (i)；不作「機制安全」主張 |
| **J2** | 未通過 | 其餘（D 不顯著、顯著但 <MDE_D、顯著為負、未跑） | 不論 | (iii) 學不起來／標定失敗（族外 y<3 或錨點未過） | 退回參數化策略族＋網格搜尋；**不算獨立結果，只算方法學上的負面結果**；不得作「機制安全」或「可利用度下界」的主張 |
| **J3** | 通過 | `D_lo > 0` 且 `D ≥ MDE_D` | 3 | (i) 找到（完全複現） | 以 §2 的逐 seed 單元比對排除模擬器 bug（ENV-04/05/06/07 已綠）；描述漏洞形態；更正方向一短文（如涉及）；列入作品集 |
| **J4** | 通過 | `D_lo > 0` 且 `D ≥ MDE_D` | 2 | (i-b) 找到（部分複現 2/3） | 同 J3，但結果文字須註明另 1 個 run 的 D CI 下界 ≤0（§17 P-3） |
| **J5** | 通過 | `D_lo > 0` 且 `D ≥ MDE_D` | 1 | (iv) 未複現（只有 1/3） | 不稱「找到」；只稱「單一 run 的候選結果」；須增加 run 或人工查證 |
| **J6** | 通過 | `D_lo > 0` 且 `0 < D < MDE_D` | 不論 | (v) 顯著但小於 MDE_D（描述性） | 只作描述，不作「找到」也不作「未找到」的主張 |
| **J7** | 通過 | `D_lo ≤ 0 ≤ D_hi`（D 不顯著） | 不論 | (ii) 在此策略類別、預算、MDE 下未找到 | 依 §9.3 (ii) 措辭；不構成安全證明。**若 S1 通過但主實驗尚未完成（D 不存在），結果應標為「未完成」，不得以 J7 的「未找到」措辭報告**（見下方 §9.2 註記 N1） |
| **J8** | 通過 | `D_hi < 0`（D 顯著為負） | 不論 | (vi) 優化失敗類：學習型對手顯著弱於最佳手寫 | 不得用 (ii) 措辭；檢查優化與特徵並如實報告；RL 比手寫弱不是「機制較強」的證據 |

- **v1.3 同步說明**：判定表的 S1 欄只依 REQ-S1-17(v1.3) 的**二值**結果（通過／未通過），欄位數與各列條件不變，因此窮舉性與互斥性不受通過標準由 6/8（再改 5/6）改為族外 3/3 的影響；族內／盲區／D3 的結果不進入 `classify` 的輸入。`classify` 的簽章不變。
- 窮舉性：S1 兩種取值 × D 的 CI 位置（`D_lo>0`；`D_lo≤0≤D_hi`；`D_hi<0`）× `D≥MDE_D` 與否 × `k`。`D_lo>0` 時主 run 自己滿足複現條件，故 `k∈{1,2,3}`。互斥性：條件欄兩兩不相交（`D_lo>0`、`D_lo≤0≤D_hi`、`D_hi<0` 互斥且涵蓋全部實數情形；`D_hi<0` 蘊含 `D_lo<0`）。
- **N1（S1 通過但主實驗未完成，維持窮舉與互斥）**：此情況仍只落在 J7 一列（D 不存在視為「D 不顯著」，J3–J6、J8 的條件皆不成立，故不新增列、不破壞互斥）；`classify` 須另回傳旗標 `incomplete=True`（僅當 S1 通過且 D 不存在時），此時報告的結果標為「未完成」，輸出措辭不得含 J7 的「未找到」「優於手寫策略的得利」等語，改為「S1 已通過，但主實驗（r=0.5 的 3 個 run）尚未完成，無法判定 D」，也不得附 J7 的「此結果不構成機制安全的證明」以外的結論性主張。S1 未通過且 D 不存在者走 J2，不受此條影響。
- 共 8 列（J1–J8）。r=0.9 的結果（探索性）不影響上表的判定，單獨列出並標示「探索性、未經 S1 標定」，並可套用同一函式但結果一律加註「探索性」。
- `D^{+H0}`（REQ-MET-04）以同一函式另算一次作附帶報告，不影響判定；兩者列不同時，兩者都報告。
- 3 次複現中只成功 1 次或 2 次的情形，分別對應 J5、J4（S1 通過且主值為 D 規則成立時）。

### 9.3 措辭範本（REQ-STOP-04）(revised)

措辭是「對本套件的描述」，**不作二項推論**（commander 決定 6）：不把 x/n 外推成對其他漏洞的檢測率或其信賴下界。數字一律由腳本填入。

- **S1 描述句（附在 J3–J8；revised v1.3）**：「**6 個類內漏洞中 RL 找回 x 個；其中族外 3 個找回 y 個**（s=0.02；族外在 s=0.01 找回 y'/3；固定預算：T_train=…、λ=…、n_S=…、G_gens=…，各 1 個 run；MDE_D=…）。6 個類內漏洞＝族外 3 個（D1、D2、O1）＋族內 3 個（O2、O3、D4）；族內漏洞手寫細網格（S_HW 36 個）已能取得大部分得利，只作描述，不計入通過判定。族內已知的 D3 單獨報告、不計入。對單一盲區範例（O5）的描述：…（預期不被找到）；本套件沒有 B1／B2（絕對時間、長記憶）的證據，不得由 O5 的結果推論對這類盲區的檢測能力。此為對本套件的描述，不外推到其他漏洞，也不推論對其他漏洞的檢測率。」
- **J3（(i)）**：「在 M3C（h=6、L=2500、p=0.1、kref=0.5、r=0.5）上，學習型公開資訊對手（MLP ≤64 參數）在測試 seeds 上的得利 G_RL=…（CI …）超過最佳手寫無資訊策略 G_手寫=…（CI …），差值 D=…（CI …，≥MDE_D=…），3 個 run 中 3 個複現。此為此固定配置、此策略類別與預算下的可利用度下界。」＋S1 描述句＋共同限制句。
- **J4（(i-b)）**：同 J3，但改為「3 個 run 中 2 個複現；另 1 個 run 的 D 的 CI 下界 ≤0」，並註明「部分複現」。
- **J5（(iv)）**：「僅 1 個 run（共 3 個）達到 D 規則（D=…，CI …）；未複現，不構成找到。」
- **J6（(v)）**：「D=…（CI …）統計上顯著，但小於本流程在 80% 檢定力下可確認的最小大小 MDE_D=…；僅作描述，不作為找到或機制可利用的主張。」
- **J7（(ii)）**：「在此策略類別（公開資訊、MLP ≤64 參數、特徵 k=8）、預算（…）與 MDE_D=… 下，未找到優於手寫策略的得利（D=…，CI …）。」＋S1 描述句。「此結果不構成機制安全的證明。」
- **J8（(vi)）**：「D=…（CI …，上界 <0）：學習型對手顯著弱於最佳手寫策略，歸為優化失敗，不是機制較強的證據。」
- **J1（(iii-c)）**：「D=…（CI …，≥MDE_D=…）但 S1 未通過（s=0.02：族外 y/3，門檻 3/3；錨點 …），為候選漏洞，未經檢測力背書，須人工查證。」
- **J2（(iii)）**：「此訓練流程的檢測力標定（S1）未達標準（s=0.02：族外 y/3，門檻 3/3；錨點 NAIVE …、M4 …），因此無法對 M3C 的可利用度作出任何下界或否定結論。」
- **共同限制句**：「機制參數沒有針對 RL 重新調參，結論僅限此固定配置（M3C：p=0.1、h=6、L=2500、kref=0.5、r=0.5／0.9）；觀察資訊集為公開版，特徵為手工充分統計量的近似，遺漏的特徵會漏掉依賴更長歷史或絕對時間的漏洞。」 另須附上 REQ-STOP-06 的**揭露清單**（所有揭露項目，v1.2 的 (a)(b)(c) 併入其中）。
- **REQ-STOP-06（揭露清單，新增，v1.3）**：最終報告（及 J3–J8 的結果文字）必須逐項揭露下列事項；數字由腳本自 `vuln_suite.json` 與 `s1.json` 填入（REQ-STOP-05）：
  1. **D1 的混合參考策略**：D1 的參考策略是「字面策略＋固定灌水 dz0」的混合，混合是為了湊足登錄大小（commander 已接受，`meta` 已標示）。
  2. **誠實基準的漂移量**：漏洞環境中誠實（θ=0）的得利與基礎環境不同；此漂移不影響 paired 的 G 與 Δ（同一環境內配對），但須逐項記錄並報告 `honest_drift`（REQ-S1-03）。
  3. **D4 的 knob 不單調**：Δ 對 knob 不單調，校準以掃描完成（REQ-S1-28），`knob_monotone=false` 須報告。
  4. **倖存者偏誤**：套件只保留 Δ>0 且可校準的漏洞（D3 的 Δ 為負、O4／O6 校準不到 0.01 而被移出；B1／B2 未實作），因此套件不代表「所有可能的漏洞」；移出的原因與數字須列出。
  5. **designer_1 受汙染**：designer_1 的材料含特徵細節，`contaminated_excluded`，不計入任何統計（REQ-S1-23）。
  6. **盲區只有 O5**：只有一個盲區範例，沒有 B1／B2（絕對時間、長記憶）的證據；盲區結果只能寫成「對單一盲區範例（O5）的描述」。
  7. **細網格與手寫族**：S_HW 擴為 36 個（REQ-HW-06）使手寫族更強，「RL 對手寫最佳」的比較對 RL 更保守；族內三者（O2、O3、D4）因細網格已被手寫族涵蓋而只描述（REQ-S1-26）。
  8. **val2 與校準**：knob 以 val∪val2 選擇（贏家詛咒的對策，REQ-S1-28），val2 不是測試集（REQ-SEED-08）。
  9. **v1.2 延續**：(a) 單一隨機對的最壞情況需 n_S≈565，目前無法涵蓋（REQ-OPT-13）；(b) 手寫的 burst 策略使用 agent 自己的回合時鐘，RL 的觀察中沒有時間，因此比較對 RL 較為保守（REQ-HW-05）。
- **REQ-STOP-05**：結果文字的數字須由腳本填入，並通過 REQ-MET-09 的登錄。

---

## 10. M4 附錄任務

此部分寫成方向一短文的附錄草稿，交由使用者決定是否放入；方向二只引用。可全程在 CPU 上以 JSON 數字與短模擬完成，不阻擋 W1。

- **REQ-M4-01（推導）(revised)**：推導逐次不等式 `ε · rw · (L · ū + w) ≥ g_max`，其中：ε＝每輪突襲機率；rw＝突襲的對象選為得標者的機率；L＝暫停長度；ū＝被暫停期間每輪的期望損失效用（內生，等於偏離者若誠實可得的每輪期望效用，需由模擬量測，誠實 agent 1 約 0.184，`quota.json main[88].honest_util_agent1`＝0.1837 ± 0.0007）；w＝被突襲那一輪物品被浪費的成本（期望等於該輪得標效用）；g_max＝一次偏離在一輪內可得的最大得利（以 u∈[0,1] 的效用上界給出）。必須明寫：
  - 這是**充分條件**（不等式成立 ⇒ 該逐次偏離的期望得利 ≤0，所以 G≤0）；不成立不等於 G>0。
  - **暫停不疊加**（REQ-ENV-13：暫停中不合格，無法被突襲，重複突襲不延長）。
  - **末端截斷（revised）**：最後 L 輪內暫停成本被截斷（`L` 以剩餘輪數 `min(L, T−t)` 取代），使「逐次不等式」在末端略為偏離；方向二**不**強制尾端誠實（REQ-ENV-16 withdrawn），改以 REQ-ENV-20 量測並報告截斷偏差。
  - **ū 為內生**：ū 依賴機制本身，須用模擬值代入，並給出敏感度。
  - **rw=0 時失效**：得標者永不被突襲，左式為 0（偏離者只在得標時獲利）。
- **REQ-M4-02（與 JSON 逐項對照）(revised)**：對 `results/stage2b.json: consist_M4[0..17]`（r∈{1, 0.99}，ε∈{0.05, 0.2}，L∈{20, 2500}，rw∈{0, 0.333, 1}；T=100000、seeds 1000–1031）、`results/quota.json: main[88..91]`（ε=0.02、L=2500、rw=1/3）、`stage2b.json: eval_rows.M4`（ε=0.01、rw=1）逐項：計算 `LHS`、指出預測（成立／不成立）、列出 JSON 的 G（含 CI），檢查「成立 ⇒ G 與 ≤0 相容（G − 半寬 ≤ 0）」。對 `consist_M4[i]`，**G 取 `gain_b03`、`gain_dz1`、`gain_edgeO` 三欄的最大值**，並逐欄列出（`gain_edgeO` 為 oracle 策略）。已知樣本：`consist_M4[9]`（rw=0）`gain_edgeO`＝0.0356 ± 0.0008、`gain_b03`＝−0.1475，不等式不成立，符合；`consist_M4[10,11]`（rw=1/3, 1）三欄皆 <0，成立，符合；`consist_M4[13]`（ε=0.2、L=20、rw=1/3、r=0.99）`gain_edgeO`＝+0.0119 > 0，須檢查為不成立；`consist_M4[12]`（rw=0，錨點格）`gain_b03`＝+0.1209，不等式因 rw=0 不成立，符合。
- **REQ-M4-03（會破的參數區）**：給出 `(ε, rw, L)` 平面（固定 ū 為量測值、g_max≤1）上不等式不成立的區域；並以模擬在邊界附近至少 3 個點驗證預測的符號。預期量級（待腳本驗證）：ε=0.02、rw=1/3、ū≈0.184 時，需 `L ≳ g_max/(ε·rw·ū)≈800`（g_max=1）才成立。
- **REQ-M4-04（G=0.0000 的退化檢查，優先序提高）**：確認方向一 M4 的 `G=0.0000 ± 0.0000` 是否為退化。已由 JSON 讀出的事實須由腳本重現並寫入附錄：
  - `main[88..91].Gmax = [0.0, 0.0]`，其策略為 `quota_savemin_oracle lam=0.3`；
  - `gains_by_strategy` 中所有 `quota_*` 策略在 M4 下恆為精確的 0.0（空操作，等價於誠實），所以 `Gmax` 被它們壓在 0；
  - 排除 `quota_*` 後的 `Gmax_nonquota` 為 −0.0406 ± 0.0033（r=0.5）、−0.0392 ± 0.0032（r=0.8）、−0.0392 ± 0.0035（r=0.9）、−0.0369 ± 0.0041（r=0.99），策略 `burst dz2 W10 H490`。
  結論的措辭須為：「0.0000 是『最大值取自含空操作策略的集合』造成的下界；實際偏離策略族中的最大值為負（約 −0.037 至 −0.041）」。此退化不是「策略族整體退化成誠實」（其他策略的 gain 為負而非 0），但「G=0.0000」不能解讀為「測得 0」。此檢查同時規範 G_手寫 必須排除空操作（REQ-HW-01）。

---

## 11. 開工前量測（約 1 小時級，W1 第 3–4 天）

量測以 `gpujob` 排隊（§12），輸出到 `results/direction2/measure_*.json`，須先於第 5.5 節定稿。

### 11.1 M-1：單步耗時與吞吐
- **REQ-MEAS-01**：對 `pop ∈ {32,64,128,256}`、`n_S ∈ {16,32,64,128}`（`B=pop·n_S` 超過 `B_max` 的組合略過；`B_max` 以逐步加倍直到顯存峰值達 12 GB 或 OOM 前一檔決定），MLP 策略（F=25、隱藏 2）在迴圈內，M3C、r=0.5，暖機 200 步、計時 1000 步（每步 `cuda.synchronize`），重複 5 次取中位數；另量 `frontier.simulate`（`1 pair × 1 strategy × 32 seeds`，同 T）作為基準（notes 記載約 2 ms／步）。
- **輸出格式**（每筆一列）：`{"mech","r","pop","n_S","B","policy":"mlp25-2-1|none|adapter","t_step_ms_median","t_step_ms_p95","steps_timed","warmup","repeats","mem_peak_MB","elem_steps_per_s","frontier_ms_per_step","device","dtype","torch","git_sha"}`。
- **決定**：`t_step(B)` 以最近量測點內插；`B_max` 與 `t_step` 代入 REQ-OPT-08、REQ-OPT-10。若 CUDA graph／`torch.compile` 可使 `t_step` 降低 ≥30%，須先採用並重量（優化只能在 W1 內做；凍結後不得改）。

### 11.2 M-2：CRN 的變異數降幅
- **REQ-MEAS-02**：在訓練 seeds 0–63（64 個）上，對 `T ∈ {20000, 40000, 100000}`、`r ∈ {0.5, 0.9}`，量測候選對 `(a,b)` 的適應值差在 CRN（同 seed）與獨立（不同 seed 組，即 a 用 seeds 0–31、b 用 32–63）兩種做法下的變異數：
  - 候選對：(a) 20 對隨機 MLP（`θ ~ N(0, σ0²I)`，σ0=0.3，兩個策略各抽一次）；(b) 3 個手寫轉接策略對：{dz=0.03, dz=0.05}、{dz=0.1, dz=0.3}、{edge_uninf thr band=0.05, edge_uninf thr band=0.1}。
  - `VRF = Var_indep / Var_CRN`，`Var_CRN = Var_s[f_a(s) − f_b(s)]`，`Var_indep = Var_s[f_a(s)] + Var_s[f_b(s)]`（兩者都用同一個 `f = U/T_score`）。
- **輸出格式**：`{"r","T","pair_kind":"rand|hand","pair_id","var_crn","var_indep","vrf","sd_crn","n_seeds":64}`；彙總：每個 `(r, T)` 的 `vrf` 中位數與最小值、`sd_crn` 中位數。
- **決定**：`sd_crn` 代入 REQ-OPT-09、REQ-OPT-11。若 `vrf` 中位數 <2（r=0.5），須在凍結檔記錄「CRN 降幅有限」，並以 REQ-OPT-09 的 n_S 規則處理（不另調規則）。 **（v1.1 實測：VRF 中位數 1.22–1.71 <2，已記錄「CRN 降幅有限（VRF 1.2–1.7）」，見 §5.5。）**

### 11.3 其他量測
- **REQ-MEAS-03 (revised)**：M-3（MDE 預估）：在驗證 seeds 以三個手寫轉接策略對（同 M-2 的 (b) 組）為代理，量測 `σ̂_val`（`G_s` 與 `D_s`）並依 REQ-MET-06 算 `MDE_G`、`MDE_D,plan`（r=0.5、0.9），與 REQ-OPT-09 的 `MDE_est` 對照；若實測 `MDE_est(r=0.5)` 與 0.0058 相差超過 ±30%，以實測值取代並記錄，規則不變。`MDE_D,plan` 寫入凍結檔，供 REQ-S1-22 決定閘門大小集合。
- **REQ-MEAS-04 (revised)**：M-4（M4 錨點重測）：REQ-S1-09 的 `G*_ref` 量測（驗證 seeds、M4、r=0.99、ε=0.2、L=20、rw=0、b=0.3），輸出含 `G_star_ref`（mean, CI），並與 JSON 的 `consist_M4[12].gain_b03` 並列。

---

## 12. GPU 規則

- **REQ-GPU-01**：所有會用到 GPU 的工作（量測、訓練、驗證、評估、S1、與 JSON 重播的測試）一律以 `gpujob` 送入共用隊列，不直接執行；不得改 `gpujob slots`；`gpujob wait N` 放在背景。不需 GPU 的短指令（CPU 的單元測試、數字登錄、M4 附錄腳本）可直接跑。
- **REQ-GPU-02（切成小 job）(revised v1.2)**：單一 job 的預計時間 ≤ 25 分鐘（以 M-1 的 `t_step` 換算成「每 job 世代數」，含每 50 世代一次的驗證評估與啟動開銷）。訓練 run 與 [gpu] 測試都切成多個 job，依序提交（訓練的後一個 job 讀前一個的檢查點）。分段規則見 REQ-GPU-08。 **v1.2**：[gpu] 測試必須拆成每段 ≤25 分鐘的多個 job；MEAS v2 期間 GPU 測試 job #26 耗時 35 分鐘，超過此上限（違規實例）。不得把一個長測試整包送成單一 job。
- **REQ-GPU-03（part 檔）**：每個 job 寫一個 part 檔：`results/direction2/parts/{cell}__run{k}__g{g0:04d}-{g1:04d}.json`，內容：`cell, run_id, gen0, gen1, status("done"|"partial"), fitness_mean, fitness_best, G_val(每 50 世代), sigma, t_step_ms, wall_s, git_sha, spec_version`；寫檔採「寫暫存檔再 rename」（原子）。檢查點 `results/direction2/ckpt/{cell}__run{k}.npz`：CMA 狀態（平均、σ、共變異數、演化路徑、世代數、計數）、numpy／torch RNG 狀態、seed 排列指標。
- **REQ-GPU-04（可續跑）**：job 啟動時若對應 part 檔存在且 `status="done"` → 直接跳過；若存在 checkpoint → 由它續跑。續跑與不中斷的結果須相同（以微型設定測試）。
- **REQ-GPU-05（預算）(revised v1.3)**：總 GPU 時數以 MEAS v2 實測重算，見 §5.5 的估算表：每個 run 預算約 4 小時（r=0.5、416 世代：mean 3.44／p95 4.80 小時；r=0.9、499 世代：mean 4.10／p95 5.73 小時，含每 50 世代的驗證評估）。**描述性大小不補跑（P-8）**。**S1：12 個 run（REQ-S1-10；範圍 9–12）＝mean 41.2／p95 57.6 小時（範圍 30.9–41.2／43.2–57.6）**；NAIVE 學習性 pilot ≤2 個 run＝6.9／9.6；主實驗 6 個 run＝22.6／31.6；診斷 7 個 run＝24.1／33.6；**核心合計 27 個 run，約 94.8 小時（mean）／約 132.4 小時（p95）**（v1.2 的 129／180 小時、v1.1 的 66.6／99 小時皆作廢；使用者同意 GPU 不設上限）。上述不含啟動開銷與重跑（以 part 檔 `wall_s` 實測回填），也不含 REQ-S1-28 的校準與細網格手寫策略的重新評估（皆不含 RL 訓練，每個 [gpu] job 仍 ≤25 分鐘）。超出者依 REQ-S1-10 的優先順序截止。
- **REQ-GPU-06**：每個 job 清單（manifest）須是命令列字串 `gpujob <完整 python 路徑> <腳本> <參數>`；不得有直接呼叫 CUDA 的條目。
- **REQ-GPU-07（[gpu] 測試不得 skip，新增）**：所有標記 [gpu] 的測試，在沒有 CUDA 的環境下必須**失敗（fail）**，不得 skip、xfail 或被靜默跳過。[gpu] 測試中不得出現 `pytest.skip`、`skipif`、`importorskip`、`xfail`；測試報告只要出現任何被 skip 的 [gpu] 測試，整體視為失敗。[gpu] 測試一律透過 `gpujob` 執行。
- **REQ-GPU-08（gpujob 分段，新增；revised v1.2）**：存在一個分段函式 `segment(t_step_ms(B), T_train, G_gens, B, n_val)`，把一個訓練 run 切成連續、不重疊、剛好涵蓋 0..G_gens 的世代區間；每段的預計時間 `est = (g1−g0)·T_train·t_step_ms(B)/1000 + (驗證評估次數)·t_val + t_startup` **≤ 1500 秒（25 分鐘）**。單一世代預計 >1500 秒時拒絕分段並回報不可行。[gpu] 測試同樣以此規則切成 job（長測試如 T-05 依 r、策略分塊，每塊 ≤1500 秒）。每段 part 檔記錄實際 `wall_s`；`wall_s > 1500` 者標 `overrun=true`，之後各段以實測 `t_step` 重新估計。 **v1.2**：此上限同樣適用於 [gpu] 測試 job（#26 曾耗時 35 分鐘而超過）；每個 [gpu] 測試 job 的預計時間須先以量測值估算，超過 1500 秒者必須再分塊。

---

## 13. 鄰近參數遷移檢查

- **REQ-NBR-01 (revised)**：對 r=0.5（與 r=0.9）的最終 RL 策略（驗證集選出的 run）與被選手寫策略，**不重新訓練**，直接在以下鄰近機制設定上以測試 seeds 評估 G（記分窗同 REQ-ENV-15，不強制尾端誠實；`L` 改變時 `T_score=T−L` 對應改變）：
  - h=5（其餘不變）、h=7、L=1250（減半）。
  - 每個設定同時報告該設定下的誠實 R/T、α，與 G_RL、G_手寫（固定策略）、D。
- **REQ-NBR-02（限制措辭，必須隨附）**：「機制參數沒有針對 RL 重新調參（h=6、L=2500、p=0.1、kref=0.5 為方向一選定值），結論僅限此固定配置。鄰近參數遷移為描述性結果（策略未重新訓練），不構成對參數空間的任何主張。」
- **REQ-NBR-03**：遷移結果不影響 §9 的判定。

---

## 14. TDD 測試清單

只列測試名稱與斷言，不寫程式。**v1.3 備註**：下列各列已與 v1.3 同步；`(revised v1.3)` 者的斷言已改寫（對應的既有測試檔需配合更新，見 `docs/direction2-tdd-map.md`）；T-70…T-75 為 v1.3 新增的規格列，**尚無測試檔**；T-66…T-69 為 v1.2 補寫的測試，定義在 tdd-map（T-65 空號）。標記 `[gpu]` 者必須經 `gpujob` 在 GPU 上跑（REQ-GPU-07：沒有 CUDA 時必須失敗，不得 skip；每個 job ≤25 分鐘，REQ-GPU-08）；其餘可在 CPU 上直接跑（短 T）。所有測試先寫、先確認紅燈（因尚無實作而失敗），實作後轉綠。**編號全部保留**：`(revised)` 為修改、`(withdrawn)` 為作廢（保留編號、不再執行）、T-56 起為新增。每個現行測試都對應到現行（非 withdrawn）的 REQ。

### 14.1 環境
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-01 (revised) | `test_parity_m3c_per_seed` [gpu] | **機制參數指定為 `p=0.1, h(tol)=6, L=300, kref=0.5`**（L=300 使 T=3000 內會發生暫停）；轉接策略 {honest, b=0.1, dz=0.03, dz=0.3, dzcrit=1.0, burst dz1 W10 H90, edge_oracle em=0.05, edge_uninf thr band=0.1, edge_uninf dz band=0.1 dz=0.3, stealth m=1.0 f=0.9} × r∈{0.5, 0.9} × seeds{0,1,2,5000,5001} × T=3000（記分到 T）：本模擬器與 frontier.simulate(M3C) 的 `reg_b, util_b, flags, pen, audits, susp` 逐 seed 以 `==` 相等；並確認至少一個組合 `flags>0`（暫停真的發生） | ENV-04, ENV-10, ENV-12 |
| T-02 | `test_parity_m4_and_naive` [gpu] | M4（ε=0.05、L=300、rw∈{0, 1/3, 1}）與 NAIVE（M3C、p=0）同上比對，另含 `nraid` | ENV-03, ENV-04 |
| T-03 (revised) | `test_frontier_quota_equivalence` [gpu] | **frontier.simulate（M3C，p=0）對 quota.simulate（NAIVE）**；以及 frontier 對 quota 在 M3C、M4：同參數同 seeds 下 `util_b, reg_b, pen, nraid` 逐位元相等 | ENV-06 |
| T-04 | `test_zero_deviation_is_honest_bitwise` | θ=0 的 MLP；以及「輸出層權重、偏置為 0、隱藏層隨機」的 MLP：全部輸出欄位與 frontier honest 逐位元相等；v1 序列 `==u1` | ENV-05, ACT-01, ACT-04 |
| T-05 (revised) | `test_replay_d1_json_m3c` [gpu] | 重播 `quota.json main[12]`、`main[14]` 的 36 個非 quota 策略 gains、`honest_util_agent1`、`R_hon_b`、`false_punish_frac`；**相對容差 1e-9（b=0 時要求精確等於 0.0）**；**CI 半寬只比對有 CI 的欄位**（1.96 公式；`gains_by_strategy` 為純量只比均值）；T=1e5、seeds 5000–5031；依 r、策略分塊成多個 ≤25 分鐘的 job | ENV-07, ENV-10, MET-05, GPU-08 |
| T-06 (revised) | `test_replay_d1_json_naive_m4` [gpu] | 重播 `main[4]`（Gmax=0.27930049154850023，半寬 0.0003786…）、`main[88]`（`raid_waste_frac`、`R_hon_b`、非 quota gains）；容差與 CI 規則同 T-05 | ENV-07, ENV-11 |
| T-07 | `test_event_order_and_susp_no_stack` | 構造情境：稽核於 t 排定、t+1 結算；被標記者 t+1..t+L 不合格；暫停期間再被標記不延長（`susp` 取覆寫）；暫停者不被當作突襲目標；**`upd=has&elig_ag`：結算時被稽核者已被暫停 ⇒ `S` 不變、不標記，但 `sxy/sxx` 照 `has` 累計** | ENV-12, ENV-13 |
| T-08 | `test_batch_invariance` [gpu] | 同一（θ, seed）單獨跑、置於批次首／中／尾、切成不同 chunk：轉接策略逐位元；MLP |差|≤1e-12 | ENV-01, ENV-09 |
| T-09 | `test_crn_shared_streams` | 同 seed、不同 θ 的元素看到相同 `E, Ua, Rn, Rx`；同 θ 同 seed 結果相同；generator 種子為 `s, s+1e6, s+2e6`；blk=1024 | ENV-02, OPT-03 |
| T-10 (revised) | `test_scoring_window_no_forced_honesty` | T=3000、L=300：`T_score=T−L`、G 除以 `T_score`；重播模式 `T_score=T`；未傳 `T_score` 時預設 `T−L`；`T≤L` 時 `T_score=T`；沒有幾何終止（總輪數恆為 T）；T_train 只接受 {20000, 40000, 100000}；**不強制尾端誠實**：對常數 d>0 的策略，t>T−2L 的 v1 仍為 `clip(u1+d,0,1)`（≠u1） | ENV-15, ENV-19, ENV-14 |
| T-11 (revised) | `test_scoring_snapshot_causal` | T 輪模擬在 t=T_score 的快照與 T'=T_score 輪模擬的最終累計量逐位元相等（不看未來） | ENV-08, ENV-15 |

### 14.2 觀察資訊集
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-12 | `test_obs_whitelist_public` | 公開版特徵建構函式只接受白名單物件；其欄位集合恰為 §3.1 表所列（不含別人的 `susp`）；傳入含 `cus`/`pend_a`/`pend_v`/`flags`/`audits`/`sxy`/`sxx`/別人 u 或 v 的物件拋錯 | OBS-01, OBS-02, OBS-06 |
| T-13 | `test_obs_invariant_to_private_state` | 同一步把私有狀態與別人的 u、v 換成隨機值：公開版特徵逐位元不變；公告版 f26、f27 隨 `cus`、`pend_a` 改變 | OBS-01, OBS-03, OBS-06 |
| T-14 | `test_obs_no_absolute_time` | 特徵函式簽章與狀態物件不含 `t/time/round/step`；兩個「相對狀態相同、絕對 t 不同」的狀態得到相同特徵（commander 決定 2：確認觀察不含時間） | ENV-14, OBS-08 |
| T-15 | `test_obs_feature_table` | 維度 25；順序與名稱同 §4.4；k=8；隨機 rollout 上全部特徵 ∈[−1,1]；歷史窗零填充 | OBS-07 |
| T-16 | `test_obs_own_win_semantics` | `won` 僅在 `win_eff & idx==1` 為 1；被突襲的得標輪為 0；無人合格的隨機分配輪為 0 | OBS-01 |
| T-17 (revised) | `test_res_last_matches_mechanism_resid` | **以獨立複製的實作**（測試目錄內自行依 §2.4 步驟 8 的公式複製 `resid=(z_w−r·z_v)/√(1−r²)`，不 import frontier、不插樁）算出被稽核輪的 `resid`，與 f22（縮放前）|差|≤1e-12 | OBS-01, OBS-07 |
| T-18 | `test_info_sets_nested` | oracle、公告版前 25 維與公開版逐位元相同；oracle 的 f26'、f27' 為他人 u；公告版為 S、audited_last | OBS-03, OBS-04, OBS-05 |
| T-19 (revised) | `test_policy_param_cap_and_action_range` | F=25→**55** 個參數、F=27→**59** 個參數，皆 ≤P_MAX=**64**；建構 >64 拒絕；診斷 F=134→273 ≤P_DIAG=**300**；輸出 v1∈[0,1] 恆成立，d=0⇒v1==u1 | ACT-01, ACT-02, ACT-03 |
| T-56 | `test_e2e_info_isolation_from_logs` | 跑一段短模擬（診斷模式，T=3000，含暫停與突襲）；**只用公開日誌**（REQ-OBS-10 (a)）以獨立重建器（獨立模組；靜態檢查它不 import 模擬器或特徵建構模組）依 §4.4 重建每輪 obs，與策略實際收到的 obs（(b)）**逐位元 `==`**；變化他人的報告／私有狀態使公開日誌不變時，重建值不變；公告版的額外兩維由其額外輸入重建 | OBS-10, OBS-01, OBS-06 |

### 14.3 優化器
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-20 | `test_mirrored_sampling` | 候選成 ± 對，`(θ⁺+θ⁻)/2` 等於當前平均（|差|≤1e-15）；λ 為偶數 | OPT-04 |
| T-21 | `test_objective_undiscounted_average` | 以常數報酬的替身環境驗證 `f=ΣU/T_score` 無折扣；誠實基準對所有個體共用且只算一次 | OPT-02, OPT-05 |
| T-22 | `test_seed_blocks_rotation` | 世代 g 的 `S_g` 取自訓練 seeds 且為 `perm_k` 的連續區塊；同 run 重跑相同；不同 run_id 排列不同；全部 <1000；**同一區塊內 seed 唯一**（含橫跨邊界的區塊）；n_S∉[1,1000] 拋 `ValueError`（v1.2，REQ-SEED-07；範圍檢查部分尚無測試） | OPT-03, SEED-04, SEED-07 |
| T-23 (revised) | `test_cma_resume_determinism` | 微型設定：跑 10 世代，與「跑 6 世代存檔、再續 4 世代」的 `θ_m, σ` 與共變異數狀態（完整版為矩陣、sep 版為對角）相同（|差|≤1e-12）；初始平均為 0、σ0=0.3 | OPT-01, GPU-04 |
| T-24 | `test_param_decision_rules` | 以合成的 M-1／M-2 表呼叫決定函式：λ、T_train、n_S、G_gens 符合 §5.5 規則（λ 優先於 T_train）；邊界情形（無滿足者）回傳「不可行」；r=0.9 功效不足旗標；`MDE_est` 取 0.0058／0.0016 | OPT-08, OPT-09, OPT-10, OPT-11 |
| T-25 (revised) | `test_val_checkpoint_selection` | 檢查點由驗證 G 最大者選出；函式簽章不接受測試 seeds；**暖啟動與冷啟動兩條路的登記與 pilot 決定必須同時出現在凍結檔**，S1 與主實驗的設定檔採用同一條路 | OPT-06, OPT-07, SEED-05, S1-14 |
| T-64 | `test_optimizer_selection_by_dim` | n=55 與 n=273 建構 sep-CMA-ES（共變異數為對角）；n≤50 建構完整共變異數 CMA-ES；n>P_MAX（非診斷）拒絕 | OPT-01, ACT-03 |

### 14.4 Seeds、手寫基準、指標
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-26 (revised v1.3) | `test_seed_splits_disjoint` | 個數 1000/32/32/32；範圍 0–999／2000–2031／**3000–3031（val2）**／5000–5031；三組 generator 的實際種子集合（s、s+1e6、s+2e6）在**四個**切分間兩兩不相交；`require_split` 接受 `"train"、"val"、"val2"、"test"`，並對其他切分的 seed 拋 `ValueError`（含 val2 對另三者）；重播專用 seeds 1000–1031 不屬任何切分，且訓練／驗證／val2／測試的函式拒絕它 | SEED-01, SEED-08 |
| T-27 (revised v1.3) | `test_test_seeds_unreachable_from_training` | AST 掃描：訓練／驗證／選取模組不含測試 seeds 常數的匯入；訓練函式遇 seed≥2000 拋錯；`arbitration/rl/final_eval.py` 排除於掃描之外，且除它之外的所有 `arbitration/rl` 模組（含 `s1.py`、`vulns.py`）都不得含測試 seeds 常數或匯入；**v1.3：`final_eval` 遇 val2 seeds（3000–3031）拋錯；訓練函式遇 val2 拋錯** | SEED-02, SEED-03, SEED-06, SEED-08 |
| T-28 (revised v1.3) | `test_hw_set_definition` | S_HW 恰 **36** 個（**不含 H0**），不含任何名稱含 `oracle` 或 `quota_` 者；其名稱集合＝`strategies()` 的 45 個扣除 13 個（H0＋8 oracle＋4 `quota_compress_uninf`）再加 4 個細網格（`dz=2.5`、`dz=3.0`、`dz=4.0`、`b=1.0`）；分組個數 b=6、dz=15、dzcrit=3、burst=3、edge thr=3、edge dz=2、stealth=4；含 H0 版本為 37；公開可實現性標註表涵蓋全部 36（含新增 4 個）；`scripts/run_quota.py` 未被修改 | HW-01, HW-03, HW-04, HW-06 |
| T-29 (revised v1.3) | `test_hw_selection_val_only_untruncated` | 以合成 gains：選取只依驗證 gains（**36 個**，無 H0）；平手取較前者；全為負時選最不負者，**測試 G_手寫 可為負**；D 可為負且不被截斷；`G_手寫^{+H0}` 與 `D^{+H0}`（**37 個**含 H0）另外輸出，且不進入 §9.2 判定 | HW-02, SEED-05, MET-02, MET-04 |
| T-30 | `test_paired_gain_and_ci` | `G_s` 以同 seed 配對；CI 用 t(31)=2.0395；重播模式用 1.96；與手算一致 | MET-01, MET-02, MET-03 |
| T-31 (revised) | `test_difference_ci_paired` | `D_s=G_{RL,s}−G_{HW,s}` 逐 seed；CI 與手算一致；3 個 run 全部列出、主值為驗證集選出者；`D^{+H0}` 另列 | MET-04 |
| T-32 (revised) | `test_mde_formula` | `κ=2.8922`（以 `scipy.stats.nct` 求 `P(|T31(δ)|>2.0395)=0.80` 的 δ，|差|≤1e-4）；`MDE=κ·SD(ddof=1)/√32`；以合成資料驗證（`G_s` 與 `D_s` 各一）；**以此 MDE 為真效果的合成 paired 資料做蒙地卡羅，檢定力在 0.80±0.02**；MDE 只由驗證 seeds 資料計算（傳入測試資料拋錯）；測試評估函式必須讀取 `mde_frozen` 而不是重算 | MET-06 |
| T-33 (revised) | `test_found_rule_truth_table` | G 規則與 D 規則的「找到」真值表（下界>0、值≥MDE 四種組合）；複現數 k 的計算（3 個 run 中 `D_lo>0` 的個數）；`k` 對應 §9.2（3→完全複現、2→部分、1→未複現） | MET-07, MET-08 |
| T-34 | `test_rt_and_false_punish_definitions` | 與 T-05 重播中的 `R_hon_b`、`false_punish_frac` 定義一致；偏離策略下 agent 0、2 的暫停比例計算正確 | MET-05 |
| T-35 | `test_number_audit_registry` | 報告的每個數字都有 JSON 路徑，且登錄值與 JSON 值一致 | MET-09, STOP-05 |
| T-62 (revised v1.3) | `test_s1_gate_sizes_and_run_count` | 給合成的 `MDE_D,plan`：`G_s` 為 `{s∈{0.01,0.02}: s≥MDE_D,plan}`；0.02∉G_s ⇒ 回傳 STOP-02 觸發；run 數 `N=3·|G_s|+3+1+2`（|G_s|=2→12、1→9）；描述性大小（<MDE）不進入通過判定；以 `MDE_est=0.0058` 輸入得 |G_s|=2、N=12；N 的各項（族外×G_s、族內×1、盲區×1、錨點 2）與 REQ-S1-10 一致 | S1-10, S1-22, MET-06, STOP-02 |

### 14.5 S1 漏洞套件
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-36 | `test_vuln_identity_mutation` [gpu] | 無變異的漏洞環境與 §2 環境逐位元相等（短 T）；長 T 的 O 系列恆等見 T-72 | S1-05 |
| T-37 (revised) | `test_vuln_operators_inject_correctly` | 對 O1、O2、O3、O6、O4、O5 以手工構造狀態逐一驗證（程式碼保留全部六個算子；v1.3 起 O4、O6 不在套件內，但公式仍驗證）：O1（連勝時 `p_el=p(1−κ)`）、O2（`S'=(1−κ)·max(S−resid−kref,0)`）、O3（殘差封頂）、O4（連續 50 次乾淨後暫停為 `(1−κ)L`）、O5（連續 20 輪未得標 S 歸零）、O6（被標記時暫停為 `(1−κ)L`）；非觸發條件下不變 | S1-06, S1-07 |
| T-38 (revised v1.3) | `test_auto_generator_deterministic` | 套件內自動算子恰為 {O1, O2, O3}、盲區算子恰為 {O5}；O4、O6 的定義保留但不在套件內（`category` 登錄中無）；相同 seeds、相同搜尋序列 ⇒ 相同旋鈕（兩次相同）；校準失敗時回報而不自動遞補 | S1-06, S1-01, S1-28 |
| T-39 (revised v1.3) | `test_vuln_known_delta_reproduced_ci` [gpu] | 對凍結登錄檔中每個（漏洞, 大小）：**用 CI 判定**——在 **val∪val2（64 個 seeds）** 重算的 `size_basis` 對應之 Δ（`out_of_family`、`blind` 用 `Δ_fine`，`in_family` 用 `Δ_coarse`）落在登錄的 `delta_val` 的 95% CI 內、落在 `[0.8s, 1.2s]` 內，且其合併 CI 下界 >0（t(63)=1.9983）；`G_HW_best_val` 的策略名（36 個清單）與登錄一致；`Δ_fine`、`Δ_coarse` 皆與登錄一致；依 GPU-08 分塊 | S1-02, S1-03, S1-28 |
| T-40 | `test_vuln_handwritten_exclusion` | (withdrawn)：原斷言 S_HW 最佳 G <0.5·G*；已由增量定義（T-39）取代，保留編號、不再執行 | (S1-04 withdrawn) |
| T-41 | `test_vuln_ref_policy_public_only` | 參考策略（O5 除可用更長歷史外）以公開歷史沙盒介面執行；私有狀態擾動下輸出不變；絕對時間與時間平移不變性、靜態檢查見 T-73 | S1-03, OBS-06, S1-27 |
| T-42 (revised v1.3) | `test_vuln_registry_frozen_and_composition` | `vuln_suite.json` 的 sha256 與凍結檔一致；組成恰為 11 項：族外 {D1, D2, O1}、族內 {O2, O3, D4}、族內已知 {D3}、盲區 {O5}、錨點 {NAIVE, M4}、受汙染未計入 {designer_1}；族外／族內各含大小 {0.01, 0.02}，盲區只含 0.02；**凍結以 git commit 順序認定：凍結檔的 commit 是任何 S1 訓練 part 檔（`cell` 為 NAIVE pilot 或煙霧測試者除外）commit 的祖先**（`git merge-base --is-ancestor`，不用檔案時間）；凍結檔含 λ、n_S、T_train、G_gens、σ0、暖／冷啟動登記與 pilot 結果、`MDE_D,plan`、`G_s`、S_HW 36 個清單與 val∪val2 校準紀錄 | S1-01, S1-02, S1-13, S1-14, S1-26 |
| T-43 (revised) | `test_blind_design_isolation_manifest` | 設計者交付包的資訊清單：只含允許的檔案（§8.5 能看到者），不含特徵表檔、`strategies()` 手寫表、notes、訓練日誌（以檔案雜湊清單檢查）；**設計者 prompt 不含 k=8、EMA 等特徵細節字樣** | S1-12, S1-13, S1-21 |
| T-44 (revised) | `test_anchor_definitions` | NAIVE 錨點＝M3C、p=0、r=0.5；**M4 錨點＝M4、ε=0.2、L=20、rw=0、r=0.99、策略 b=0.3**；錨點通過規則 `G_RL≥0.8·G*_ref` 且 CI 下界>0；W1 重測 `G*_ref` 的 CI 下界≤0 時，M4 錨點降為資訊性、不計入 S1-17 的通過條件（只剩 NAIVE 錨點）並照實記錄；S1-09 與 S1-10 的 r 一致（僅 M4 錨點為 0.99，其餘 0.5）；v1.3：錨點不屬於 `category` 的族外／族內判定 | S1-08, S1-09, S1-10, S1-11, ENV-11 |
| T-61 | `test_naive_pilot_decision_rule` | 決定函式：冷啟動驗證 G≥0.8·`G*_ref` → 凍結冷啟動；否則暖啟動達標 → 凍結暖啟動；兩者皆未達 → 觸發 STOP-02；pilot 程式碼不接受測試 seeds；兩條路都出現在凍結檔 | S1-20, OPT-07, ACT-04, STOP-02 |
| T-63 | `test_designer_logs_saved` | `designer_logs/{id}/prompt.txt` 與 `tool_calls.jsonl` 存在、已進 git；`tool_calls.jsonl` 無讀取 §4、§5、`strategies()`、notes 的紀錄 | S1-21, S1-12 |

### 14.6 統計、關卡、結果判定
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-45 (revised v1.3) | `test_clopper_pearson_lower_bound` | `LB(x,n)` 單尾 95%：**(3,3)=0.36840315、(2,3)=0.13535036、(6,6)=0.60696223、(5,6)=0.41819659**、(6,8)=0.40031061、(7,8)=0.52932059、(8,8)=0.68765602、(5,8)=0.28924082、(10,10)=0.74113445、(0,n)=0；與 `scipy.stats.beta.ppf(0.05, x, n−x+1)` 一致；並滿足 `P(Bin(n,LB)≥x)=0.05`（|差|≤1e-9）；對 x 單調；**LB 只出現在描述欄位，結果措辭不得把它當作對其他漏洞檢測率的推論** | S1-16 |
| T-46 (revised v1.3) | `test_s1_gate_rule` | 真值表：兩個（計入的）錨點通過＋s=0.02 時族外 y=3 → 通過；y≤2（含 2/3）或任一（計入的）錨點未過 → 未通過；M4 錨點降為資訊性時，只看 NAIVE 錨點；**族內 3 個（O2、O3、D4）、盲區 O5、D3、designer_1、s=0.01 的結果任意變化都不改變通過與否**；未完成的 run 計為未偵測（族外分母恆為 3、類內分母恆為 6）；描述性大小不進入判定；優先順序 s=0.02 先跑（9 個 run：族外 3＋族內 3＋盲區 1＋錨點 2）；與 STOP-01 一致 | S1-10, S1-15, S1-17, STOP-01 |
| T-47 (revised v1.3) | `test_outcome_classifier` | `classify(S1, D, D_lo, D_hi, MDE_D, k)` → J1…J8：對 (S1 兩種)×(D_lo>0 / D_lo≤0≤D_hi / D_hi<0)×(D≥MDE_D 與否)×(k=1,2,3) 的完整網格 + 邊界值（D_lo=0、D=MDE_D、D_hi=0）**恰好一列成立**（窮舉且互斥；簽章與 v1.2 相同，S1 為二值）；W1 停損與到期未通過 → 未通過；D 不存在 → D 不顯著；輸出的措辭含必要片語（「對本套件的描述」「僅限此固定配置」「固定預算」；**「6 個類內漏洞中 RL 找回 x 個；其中族外 3 個找回 y 個」「對單一盲區範例（O5）的描述」「沒有 B1／B2」**；J5「未複現」；J6「小於 MDE_D」；J8「優化失敗」；J1「候選漏洞」；J1／J2 含「族外 y/3，門檻 3/3」）；措辭不含二項推論片語（「檢測率」「Clopper」「CP 95% 下界」）與舊門檻（「6/8」「≥5/6」） | STOP-01, STOP-02, STOP-03, STOP-04, S1-15 |

### 14.6b v1.3 新增測試（只列規格；**尚無測試檔**）
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-70 | `test_val2_calibration_only` | 3000–3031 恰為 val2；`require_split(…, "val2")` 通過、對其他切分拋錯；val2 不出現在 RL 訓練、檢查點選取、MDE 的呼叫路徑（AST／執行期）；`final_eval` 與任何測試評估函式遇 val2 拋錯；val2 只由校準模組（`vulns.py` 校準函式／`s1.py` 校準）匯入 | SEED-08, SEED-01, SEED-03 |
| T-71 | `test_vuln_category_fields_and_consistency` | 登錄檔每項有 `category` 且屬六種之一；`category=="blind" ⇔ blind==true`；固定指派（D1、D2、O1＝族外；O2、O3、D4＝族內；D3＝族內已知；O5＝盲區；NAIVE、M4＝錨點；designer_1＝受汙染未計入）；`contaminated_excluded` 不出現在 S1 run 與統計；族外項目的 `Δ_fine` CI 下界 >0 且在 `[0.8s,1.2s]`；矛盾時回報而不自動改分類；`ratio_HW_over_Gstar`、`delta_fine`、`delta_coarse` 欄位齊全 | S1-26, S1-01, S1-03 |
| T-72 | `test_vuln_identity_long_horizon_O_series` [gpu] | O1、O2、O3、O5（及 O4、O6）取恆等 knob（κ=0／C=∞／g=∞），在 **T=1e5、L=2500**，`T_score=T−L` 與 `T_score=T` 兩種記分窗下，與 §2 環境逐位元相等；至少一個組合 `flags>0`；依 GPU-08 分塊（每塊 ≤1500 秒） | S1-05, GPU-08 |
| T-73 | `test_ref_policy_public_log_only_no_absolute_time` | 靜態（AST）：每個 `ref_policy`（含 D4、O5）的簽章與讀取欄位不含 `t/time/round/step/tick/clock`，不含私有欄位名；執行期：私有擾動不變、**時間平移不變**、以獨立重建器只讀公開日誌重建輸入後輸出逐位元相同；D4 的參考策略改用 `since`（f05 可重建的相對量），舊 `_LClimb` 的絕對 t 版本須被偵測為違規（負向測試） | S1-27, OBS-08, OBS-10, OBS-11, OBS-06 |
| T-74 | `test_knob_calibration_val_union_val2` | 校準只用 val∪val2（64 個 seeds）；驗收＝合併 Δ 在 `[0.8s,1.2s]` 且合併 paired CI（t(63)=1.9983）下界 >0；`size_basis` 對 `out_of_family`／`blind` 為 fine、`in_family` 為 coarse；對非單調 knob（合成）搜尋仍找到驗收者；相同 seeds 與搜尋序列 ⇒ 相同 knob；MDE 仍只由驗證集 32 個 seeds 計；校準函式簽章不接受測試 seeds；無法達成時回報而不遞補 | S1-28, S1-02, SEED-08, MET-06 |
| T-75 | `test_disclosure_list_v13` | J3–J8 的結果文字與凍結檔／報告模板含 REQ-STOP-06 的全部揭露項（D1 混合策略、`honest_drift`、D4 knob 不單調、倖存者偏誤、designer_1 受汙染、盲區只有 O5 且無 B1／B2 證據、細網格使比較對 RL 較保守、val2、n_S≈565、burst 時鐘）；缺任一項 → 失敗；數字來自 JSON（STOP-05） | STOP-06, STOP-05, OPT-13, HW-05, S1-23 |

### 14.7 M4 附錄、量測、GPU、鄰近遷移、重播
| # | 測試名稱 | 斷言 | REQ |
|---|---|---|---|
| T-48 (revised) | `test_m4_inequality_vs_json` | 對 REQ-M4-02 列出的 JSON 項目計算 LHS 與預測；G 取 `gain_b03/gain_dz1/gain_edgeO` 最大值；「成立」者的 G 與 ≤0 相容（G−半寬≤0）；`consist_M4[9]`（rw=0）判為不成立；`[10]`、`[11]` 判為成立；`[13]` 判為不成立；`[12]`（rw=0，G=+0.1209）判為不成立 | M4-01, M4-02 |
| T-49 | `test_m4_degeneracy_check` | 自 `quota.json` 讀出：`main[88..91].Gmax==[0,0]`；全部 `quota_*` gains 精確為 0.0；`Gmax_nonquota` 為 −0.0406／−0.0392／−0.0392／−0.0369（允差 1e-4）；腳本輸出結論字串含「含空操作策略」 | M4-04, HW-01 |
| T-50 | `test_m4_break_region_signs` [gpu] | 邊界附近 3 個 `(ε, rw, L)` 點：預測不成立的點，模擬 G 的符號與預測相容 | M4-03 |
| T-51 (revised) | `test_measure_output_schema` | M-1、M-2、M-3 輸出 JSON 欄位與型別符合 §11 格式；缺欄位拒絕；M-4 輸出含 `G_star_ref` 並與 `consist_M4[12].gain_b03` 並列 | MEAS-01, MEAS-02, MEAS-03, MEAS-04 |
| T-52 (revised) | `test_gpu_job_manifest` | manifest 每條以 `gpujob` 開頭、含 part 檔路徑、預計時間 ≤25 分鐘；不含 `gpujob slots`；不含直接 python 呼叫；[gpu] 測試 job 同樣走 `gpujob` | GPU-01, GPU-02, GPU-05, GPU-06 |
| T-53 | `test_part_files_atomic_and_idempotent` | part 檔經暫存＋rename；`status="done"` 的 job 重送被跳過；`partial` 重送會覆寫；欄位齊全 | GPU-03, GPU-04 |
| T-54 (revised) | `test_neighbor_configs_and_no_retrain` | 鄰近設定恰為 {h=5, h=7, L=1250}，其餘參數不變；評估前後策略參數雜湊不變；`T_score` 隨 L 調整（`T−L`）、無強制尾端誠實；結果檔含 REQ-NBR-02 的限制句 | NBR-01, NBR-02, NBR-03 |
| T-55 (revised) | `test_feature_ablation_groups_partition` | 6 組去一消融的特徵索引恰好分割 f01–f25；64 步歷史對照 F=134、參數 273 ≤P_DIAG=300 | S1-19, ACT-03 |
| T-57 | `test_gpu_tests_never_skip` | (a) AST 掃描所有 [gpu] 測試：不含 `pytest.skip`、`skipif`、`importorskip`、`xfail`；(b) 以 `torch.cuda.is_available()` 為假的環境（子行程、monkeypatch）執行 [gpu] 測試：**全部失敗、skip 數為 0**；(c) 報告檢查器：出現任何被 skip 的 [gpu] 測試即整體失敗 | GPU-07 |
| T-58 | `test_gpujob_segmentation` | `segment(...)` 輸出的區間連續、不重疊、剛好涵蓋 0..G_gens；每段預計時間（含驗證評估與啟動開銷）≤1500 秒；單一世代 >1500 秒時拒絕；[gpu] 測試 job 的分塊同樣每塊 ≤1500 秒；part 檔 `wall_s>1500` 時標 `overrun=true` 並重新估計後續分段 | GPU-02, GPU-08 |
| T-59 | `test_replay_stage2b_m4_anchor` [gpu] | 重播 `stage2b.json: consist_M4[12]`（r=0.99、ε=0.2、L=20、rw=0；`gain_b03`＝0.12093494445548655、`gain_dz1`、`gain_edgeO`、`waste`）與 `consist_M4[9]`（`gain_b03`＝−0.14747848180358147、`gain_edgeO`＝0.03556691436913642）；T=1e5、seeds 1000–1031；相對容差 1e-9、CI 以 1.96 公式重算 | ENV-21, ENV-11 |
| T-60 | `test_truncation_bias_measure` | 診斷變體只在量測函式內啟用（t>T−2L 強制 `v1=u1`），正常模式不受影響（正常輸出與 T-10 一致）；輸出 `trunc_bias.json` 欄位齊全（`policy, r, G_normal, G_variant, delta, ci`）；`delta=G_normal−G_variant`；變體不被訓練、選取、判定模組匯入（AST 掃描） | ENV-20 |

測試數量：規格列編號 T-01 至 T-64（共 64 個，T-40 為 (withdrawn)，現行 63 個）；v1.2 補寫 T-66…T-69（定義於 tdd-map，T-65 空號）；v1.3 新增 T-70…T-75（6 個，尚無測試檔）。

---

## 15. 里程碑與時程

3 週主線＋1 週緩衝。每週交付物如下；週結束時未達成者依 §9 停損。

### W1（第 1–7 天）：規格、測試、環境、量測、pilot、凍結
- 第 1 天：本規格 v1.0-rev1 定稿並交由 commander 確認第 17 節；啟動設計者 agent（只給 §8.5 允許的資料，保存 prompt 與工具紀錄）；寫出全部測試（T-01…T-64）並確認紅燈。
- 第 2–4 天：實作模擬器，使 T-01…T-11（環境）、T-12…T-19 與 T-56（觀察）轉綠；M4 附錄的 CPU 腳本（T-48、T-49）可平行進行。
- 第 3–4 天：M-1、M-2、M-3、M-4 量測（`gpujob`，每 job ≤25 分鐘）。
- 第 5 天：規格修訂 v1.1（填入 §5.5 的 λ、T_train、n_S、G_gens）；MEAS v2 後再修訂為 v1.2（n_S=256、世代 416／499、驗證每 50 世代、每個 run 約 4 小時）；若 REQ-OPT-08／09 判 r=0.5 不可行，觸發 REQ-STOP-02；NAIVE 學習性 pilot（REQ-S1-20，含「永遠報 1」基準 REQ-S1-25）決定暖／冷啟動。
- 第 6–7 天：設計者與自動生成器交付漏洞；通過 T-36…T-44、T-61…T-64、T-70…T-75；**第 7 天簽署凍結檔**（REQ-S1-14）。
- **v1.3 執行順序（紅方決定「執行順序」）**：(1) 規格 v1.3（本文件）→ (2) 細網格 HW（`hw.py` 36 個）、既有測試與規格對齊（T-26…T-29、T-36…T-47、T-62、S1-18 等）、`s1.py` 的 T-67／S1-24／S1-25、D4 參考策略改用相對量（REQ-S1-27）、新增 T-70…T-75 → (3) **GPU 重新校準**（val∪val2，REQ-S1-28；不含 RL，每 job ≤25 分鐘）→ (4) **紅方快速核對** → (5) **凍結**（REQ-S1-14）。
- **交付物**：`docs/direction2-spec.md`（v1.3）、`tests/test_d2_*.py`、`arbitration/gpu/policy_sim.py`、`results/direction2/measure_*.json`、`results/direction2/vuln_suite.json`、`results/direction2/designer_logs/`、`docs/direction2-freeze.md`、M4 附錄腳本與草稿。

### W2（第 8–14 天）：S1
- 第 8–9 天：s=0.02 的 9 個 run（族外 3、族內 3、盲區 1、錨點 2）。
- 第 10–12 天：s=0.01（族外 3 個）；描述性大小不補跑（P-8）。預估 run 總數 12（REQ-S1-10；範圍 9–12）＝mean 約 41.2／p95 約 57.6 GPU 小時（§5.5）。
- 第 13 天：特徵充分性診斷（非硬性，若時間允許）；截斷偏差量測（REQ-ENV-20）。
- **第 14 天結束：S1 判定（REQ-STOP-01；s=0.02 族外 3/3＋錨點，REQ-S1-17）**。
- **交付物**：`results/direction2/s1.json`、檢測力曲線圖（含各 s 的 `(x, n, p̂, LB)` 與 MDE 標示）、S1 通過／未通過的書面判定。

### W3（第 15–21 天）：主實驗與紅方審查
- 第 15–17 天：r=0.5 的 3 個 run；r=0.9 的 3 個 run（探索性）。
- 第 18 天：手寫基準在驗證集選取、測試集評估；每個 cell 只評估一次（REQ-SEED-02）；鄰近參數遷移（§13）。
- 第 19 天：數字登錄（REQ-MET-09）、依 §9.2 判定表套用 §9.3 措辭範本。
- 第 20–21 天：紅方審查（攻擊 S1 的標定、特徵限制、手寫基準的選取、M4 附錄）。
- **交付物**：`results/direction2/main.json`、`results/direction2/trunc_bias.json`、`docs/direction2-number-audit.md`、`docs/direction2-summary.md`（含判定列 J1–J8 與措辭）、紅方審查紀錄。

### W4（緩衝，第 22–28 天）
重跑失敗的 job、補完 S1 未完成的 run 與診斷、處理 r=0.9 的功效不足、整理作品集。不新增範圍；超出即收斂成短文，不開新框架。

---

## 16. 規格之外的備註
- 不提供絕對時間（REQ-ENV-14）與手工特徵的限制，須在所有對外文字中註明（REQ-OBS-09、§9.3 共同限制句）。
- 方向二不修改 `arbitration/gpu/frontier.py`、`quota.py`、`results/*.json`；新增檔案另置。
- 紅方審查已確認的引用：RegretNet（1706.03459）、AI Economist（2004.13332）、Gleave 等（1905.10615，ICLR 2020）、PSRO（1711.00832，NeurIPS 2017）、Brero 等（2202.07106，NeurIPS 2022）、planted bug 類比 LAVA（IEEE S&P 2016）；新意定位為「評估方法學＋實證」。

---

## 17. 待 commander 決定的點（規格預設值只為讓測試可先寫）

### 17.1 已決定（commander；已反映在本規格）
| 編號 | 決定 | 反映處 |
|---|---|---|
| D-1 | M3C 沿用方向一主表（h=6、L=2500、p=0.1、kref=0.5）；r=0.5 手寫無資訊得利 0.0213（0.0287 為另一組參數，不用） | ENV-10 |
| D-2 | 觀察不含時間，不強制尾端誠實，只把最後 L 輪排除在評估外；量測並報告截斷偏差 | ENV-14、15、16(withdrawn)、20 |
| D-3 | M4 rw=0 錨點在公開資訊下重測；紅方更正：改用 `consist_M4[12]`（r=0.99、ε=0.2、L=20、b=0.3，+0.1209） | ENV-11、S1-09、MEAS-04 |
| D-4 | S1 算力：W1 先量測、再做優化 | GPU-05、MEAS-01 |
| D-5 | 他人的 susp 不公開 | OBS-02 |
| D-6 | （**已被 R-2 取代**）原：s=0.02 時 8 個找回 6 個（CP 下界 0.40）。仍保留的部分：屬誠實但偏弱的標準；措辭為對本套件的描述，不作二項推論 | S1-16、S1-17、§9.3 |
| D-8 | 手寫基準排除 H0；含 H0 版本另外報告（v1.3：清單由 32 個擴為 36 個，含 H0 為 37，見 R-1） | HW-01、HW-02 |
| P-1 | 接受 O6（標記後縮短暫停），類內維持 n=8（**v1.3：O6 已因無法校準移出套件、類內 n=6，見 V-1、R-2**）；某算子校準失敗時回報 commander，不自動遞補 | S1-06、T-37、T-38 |
| P-2 | 策略 `F→2→1`（55 參數，sep-CMA） | ACT-02、ACT-03、OPT-01、T-19、T-64 |
| P-3 | 3 次複現中 2 次成功算「找到」（J4），但一律以 x/3 報告；1 次（J5）不算 | MET-08、§9.2 |
| P-4 | 盲區漏洞只跑 s=0.02（v1.3：盲區只剩 O5） | S1-07、S1-10 |
| P-5 | M4 錨點若在 W1 重測時 `G*_ref` 的 CI 下界 ≤0，降為資訊性，不計入 S1 通過判定（只剩 NAIVE 錨點），並照實記錄 | S1-09、S1-11、S1-17、T-44、T-46 |
| P-6 | 閘門大小集合依 `MDE_D,plan`（代理量測）決定 | S1-22、MEAS-03 |
| P-7 | 暖啟動以最小平方法擬合最佳手寫無資訊策略；NAIVE pilot 門檻＝在 NAIVE 上、預算內達到 0.8·G*_ref（驗證重測，預期約 0.28） | OPT-07、S1-20 |
| P-8 | 低於 MDE 的大小不補跑，只描述已有結果；S1-10、GPU-05、§15 的「有餘裕才補跑」措辭已於 v1.1 改為「不補跑」 | S1-10、GPU-05、§15 |
| P-9 | MDE 用精確非中心 t 的 κ=2.8922 | MET-06、OPT-09、OPT-11、T-32 |
| C-1 | f05 的 t_s ＝自己首次觀察到 susp 上升的那一輪 | OBS-11 |
| C-2 | 公開日誌 susp1 在觀察時記錄 | OBS-12 |
| C-3 | T_score 預設為 T−L（骨架 docstring 有誤）；T≤L 時退回 T | ENV-15 |
| C-4 | O3／O5 的恆等基準改用 C=∞、g=∞ | S1-05、S1-06 |
| C-5 | OBJ-06 以 spec 與 notes 作為事前登記，不另開新檔 | OBJ-06 |
| C-6 | M4 的 ε、rw 為公開常數（Kerckhoffs 原則），T-56 白名單允許 | OBS-10、§3 |
| C-7 | 測試 seeds 的評估只准寫在 `arbitration/rl/final_eval.py`（不列入 T-27 掃描）；其餘模組含 `s1.py` 一律禁止 | SEED-03、SEED-06、T-27 |
| M-1 | W1 MEAS v1：λ=256、T_train=20000；CRN 降幅有限；r=0.9 在 T_train=2e4 只作描述（n_S=64、300 世代、1.8 小時已被 M-2 取代） | §5.5、OPT-08..11、MEAS-02 |
| M-2 | MEAS v2（紅方重大 1、2 採納）：n_S 上限 256、依最壞中位數取 256；單一對最壞約 565 無法涵蓋，須揭露 | §5.5、OPT-09、OPT-13 |
| M-3 | 世代數依 OPT-10：r=0.5 為 416、r=0.9 為 499（更正 v1.1 的 300）；驗證每 50 世代一次 | §5.5、OPT-06、OPT-10 |
| M-4 | 每個 run 預算約 4 小時；總時數 mean 約 129／p95 約 180 小時，使用者同意 GPU 不設上限 | §5.5、GPU-05 |
| M-5 | [gpu] 測試拆成每段 ≤25 分鐘的 job（#26 曾耗 35 分鐘） | GPU-02、GPU-08 |
| M-6 | 新訓練程式碼必須明確傳入 obs_version；預設 None 只為相容舊測試 | OPT-12 |
| M-7 | seed_block：n_S∉[1,1000] 拋 ValueError，同區塊 seed 唯一（T-22） | SEED-07、T-22 |
| M-8 | burst 使用 agent 自己的回合時鐘，列為公開策略；比較對 RL 較保守，須揭露 | HW-05、§9.3 |
| M-9 | check_report：任何 skipped 一律回傳 False | GPU-07、T-57 |
| M-10 | 設計者漏洞以 designer_2 為準；designer_1 受汙染、只作揭露；盲設計者可讀材料須事先禁字預掃 | S1-12、S1-23、S1-24 |
| M-11 | NAIVE pilot 加「永遠報 1」基準，與手寫最佳在相同 T、seeds 下比較（解釋 smoke 0.311 對 0.279） | S1-20、S1-25 |
| V-1 | 漏洞套件完成：designer_2 的 V1–V4 實作為 D1–D4；校準成功 13 組；D3、O6、O4 無法校準，B1、B2 未實作；類內 n 由 8 降為 6（**通過標準 ≥5/6 已被 R-2 取代**） | S1-01、S1-06、S1-07 |
| V-2 | 盲區只用 O5；B1、B2 不補，列為限制 | S1-07、STOP-06 |
| V-3 | D1 使用混合參考策略（字面策略＋固定灌水 dz0）：接受，meta 已標示 | STOP-06 |
| V-4 | 「D4 的設計主張不成立（手寫策略也能利用）」：接受，Δ 的定義已處理 | S1-02、S1-26 |
| V-5 | 驗證 seeds 用來從相鄰 knob 中挑選：接受（v1.3 擴為 val∪val2） | S1-28 |
| V-6 | rh 取 r_true；V1 的折扣對任何得標者生效；沙盒保留當輪的 susp1：接受 | S1-03、S1-27 |
| R-1 | 紅方 1（致命，採納）：手寫基準改用細網格（dz∈{2.5,3,4}、b=1），S_HW＝36；O2、O3、D4 標為族內 | HW-01、HW-02、HW-06、S1-26 |
| R-2 | 紅方 2（致命，採納）：族外只有 D1、D2、O1；S1 通過＝s=0.02 族外 3/3（CP 單側 95% 下界約 0.37）＋錨點；族內只描述；措辭「6 個類內漏洞中 RL 找回 x 個；其中族外 3 個找回 y 個」 | S1-15、S1-17、§9.2、§9.3 |
| R-3 | 紅方 3（重大，採納）：D3 為族內已知，單獨報告；揭露倖存者偏誤 | S1-01、S1-10、STOP-06 |
| R-4 | 紅方 4（重大，採納，贏家詛咒）：val∪val2 合併重新校準 knob；3000–3031 登記為 val2，永遠不得作為測試集 | SEED-01、SEED-08、S1-28 |
| R-5 | 紅方 5（重大，採納）：D4 的參考策略用到絕對 t，改用 f05 可重建的相對量；新增「參考策略只能用公開日誌欄位」的 REQ | S1-27、OBS-08 |
| R-6 | 紅方 6（次要）：D1 混合策略照實說明；誠實基準漂移只記錄；O 系列補驗 T=1e5 恆等；D4 knob 不單調只記錄 | STOP-06、S1-05、S1-28 |
| R-7 | 紅方 7：盲區只有 O5（新 seeds 上 Δ=+0.0233）；措辭限定為「對單一盲區範例（O5）的描述」，明講沒有 B1／B2 的證據 | S1-07、§9.3 |
| 紅方 1–11、GPU 項 | 幾乎全部採納（見修訂紀錄） | 各 REQ 的 (revised)／(withdrawn) |

### 17.2 仍待 commander 決定
目前沒有待決項（原 P-1～P-9 已全部決定，見 17.1）。**v1.3 規格解讀，凍結前請 commander 確認一次**：(i) REQ-S1-02 對 `in_family` 以 `Δ_coarse`（32 個清單）作為大小標籤、`out_of_family`／`blind` 以 `Δ_fine`——因族內漏洞的 `Δ_fine` 可為零或負，無法以其校準大小；(ii) D3 不跑 RL（REQ-S1-10；若要加跑，增加 1 個 run＝mean 3.4／p95 4.8 小時）；(iii) 登錄大小只剩 {0.01, 0.02}（移除從未校準的 0.003、0.005）。D-7 為沿用的已知風險，非待決事項：類內漏洞由「看不到特徵細節」的設計者設計，可能有部分落在特徵類別之外而拉低 x，照實計入分母（盲區漏洞才排除）。

---

## 18. 延伸階段 E1（不在核心範圍）：攻防雙方學習的 2×2 消融

本節只記目標與待定事項，**不含任何細節 REQ**，不影響 §1–§17 的交付與時程。核心（攻擊方學習 × 防守方固定）完成之後才做。

### 18.1 目標
使用者提議讓防守方也變成學習型，並做 2×2：

| | 防守方固定 | 防守方學習 |
|---|---|---|
| 攻擊方手寫 | 方向一已完成 | 機制針對手寫族調參 |
| 攻擊方學習 | 方向二核心 | 共同演化 |

使用者的態度：「試試看，有中就有中，沒中就當經驗值。」目的是補上「機制只針對手寫族調過」這一批評，並看共同演化下結論是否改變。

### 18.2 初步建議（commander，待攻防檢驗，不是規格）
- 「防守方學習」限定為在小範圍內調參數（h、κ_ref、L、p，必要時加 M4 的 ε），用 CMA-ES，不學整套新規則。
- 共同演化採交替最佳回應或 PSRO 式的作法，並事先定義循環或不收斂的判定與報告方式。
- 評估時每一格都要對「該格以外的對手」做交叉評估，避免只報告在自己訓練對手上的成績。

### 18.3 待定事項（開工前必須先決定；E1 開始前另做紅方／白方攻防）
- 事前否證條件（什麼結果算 E1 失敗／無效）。
- 防守方參數範圍與目標函數（例如以誠實 R/T、α 為約束的前提下，最小化對手得利）。
- 共同演化的演算法選擇與收斂／循環判定。
- 交叉評估的設計與統計（沿用本規格的 MDE、paired CI、驗證選取、測試只評一次的原則是否足夠）。
- 算力與時程（核心完成後再估，不得延後 W1–W3）。
- 與 S1 檢測力標定的關係（防守方學習後，planted-exploit 套件是否須重新標定）。