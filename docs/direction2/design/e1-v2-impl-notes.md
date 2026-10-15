# E1 v2 實作說明：對設計的詮釋與偏離

本文件不修改已登記設計（`e1-v2.md`），只揭露實作怎麼詮釋、哪裡偏離。行號指 `docs/direction2/design/e1-v2.md`；程式位置指 `scripts/e1_v2.py`。須在正式 freeze 前 commit。

## A. 詮釋與偏離

1. **freeze_record.json 取代「雜湊寫入 manifest」。** 設計 L57、L147 寫 selection 雜湊進 manifest；但 manifest 在 S0 就凍結，selection 要到 select 階段才產生，無法事先寫入。實作把 `selection_frozen.json` 的雜湊寫在 `freeze_record.json`（`stage_select` 末段），須 commit 後 test 才開（`verify_freeze`，約 L813）。效果等價：test 前雜湊與 commit 都被檢查。
2. **訓練 fitness 用 n=16 點估計的誠實約束，只有 DS 選模用 CI。** 設計 L24–L26 把 CI 約束寫在選模；實作的 DT 訓練（`defender`，約 L513–L545）沿用 v1 的 DT n=16 逐代點估計（ΔR≤0.002、Δα≤0.01）決定 fitness，DS（n=128）才做 UB 判定（`ds_record`）。理由：n=16 的 CI 過寬，無法導引搜尋。風險：點估計可行的訓練結果在 DS 的 CI 下可能全被刷掉，結果變「未進 test：無候選滿足誠實約束」（設計 L79 已涵蓋此分支）。
3. **Round 2 池的 C1 適應攻擊者取自 240 代、延長之前。** 設計 L111–L112 未說明與延長的先後。實作 `stage_s2`（約 L696）以 `chain_best(…, gens)` 取 240 代以內最佳，因 s2 在 sx 之前跑。理由：流程順序與預算。後果：若 C1 其後被延長，Round 2 用的是較早的攻擊者。
4. **飽和規則以驗證點實作。** 設計 L65 以「前 200 代／最後 40 代」描述；`saturation`（約 L177）用「g≤gmax−2·val_every」對「最後兩個驗證點」。驗證間隔為 20 時兩者等價（200／{220,240}；延長後 440／{460,480}）；間隔改變時不等價，故間隔寫入 manifest 不可改。
5. **次要結果 5、6 未實作。** 設計 L16–L17 的轉移矩陣（C1 適應攻擊者放到 C2／原設定）與 36 手寫池 test 重評未做；`secondary.json` 只有各攻擊者逐一的 G_orig／G_cand／D（`run_test` 後段）。AS 上的轉移矩陣有存（`selection_frozen.json`），但不是設計所指的 test 上矩陣。這些項目只描述、不參與判定，故不影響主判定。
6. **防守 eager、攻擊者與 test 用 graph。** 設計 L121 只寫「graph 模式」。實作的 `sim`（約 L444）對所有階段依 `--graph` 一視同仁，launcher 一律帶 `--graph`；審查所述「防守 eager」未在程式中看到分流，實際路徑以 log 的 `GRAPH_FALLBACK_USED` 與 `status.json` 為準（防守若走 fallback 即為 eager），攻擊者鏈與 test 走 graph。graph 與 eager 位元相同是依主實驗 preflight 的假設，故新增 testcheck（見 B）直接驗證 test 路徑。graph 失敗在非 test 階段可 fallback（旗標 `graph_fallback_used`，見 B）。
7. **誠實策略的兩條路徑。** DT／DS 的誠實策略為 `AdapterPolicy({})`（v1 的 `fixed_pool`），test 為 `MLPPolicy` 零參數（`honest_pol`，約 L496）。理論上輸出相同，但數值路徑不同，可能使 ΔR 的絕對水準略異（推論，未驗證）。訓練與選模的誠實約束因此與 test 的約束不嚴格同尺；testcheck 只比 graph 對 eager，不比這兩個策略。
8. **smoke 用 DT／DS／AT／AS 前 8 個 seeds。** `block_seeds(smoke=True)`（約 L64）。這些 seeds 與正式 DT／DS／AT／AS 前 8 個重疊，但 smoke 輸出在獨立目錄 `e1v2_smoke`、無 TEST 區段、永不開 test；正式 freeze 前會被清除。

## B. 本次新增（依紅方審查）

9. **testcheck stage**（`run_testcheck`）。用 AS 區段前 64 seeds（非 test seeds），以 test 完全相同的路徑（`simulate_chunked`、`test_chunk`=64、graph、同樣的策略組合：原設定與候選的主要與次要共 4 組）評估，並與 `sim_eager`（不分塊、不用 graph）比對。**容差：無，要求 snap 三欄位逐元素完全相同**（因而逐 seed 相同）；任何差異即失敗。理由：graph 重播與 eager 在固定隨機流下應位元相同（主實驗 preflight 的 graph==eager 檢查），一旦出現差異，代表 test 路徑可能與已驗證路徑不同，不應用容差遮蓋。結果寫 `testcheck.json`（含 manifest 與 freeze_record 雜湊、路徑識別）。
10. **test 的前置檢查**（`require_testcheck`）。在 ledger 寫 started 之前，要求 testcheck.json 存在、通過、雜湊與目前一致、執行路徑相同；cuda 上並用 AS seeds 對四組 test 評估各跑一次 graph 預熱，graph 擷取失敗會在 ledger 與 test seeds 被碰到之前報錯。
11. **graph fallback 規則。** test 與 testcheck 為 strict 模式：忽略 `--graph-fallback`、忽略先前的 `graph_fallback.json`，graph 錯誤直接中止，launcher 對這兩個模式不重試。其他階段保留 fallback，並在 `status.json`、status 輸出與各 result JSON 以 `graph_fallback_used` 標示。
12. **自動續跑。** launcher 對 s1／s2／sx／select／all，退出碼 75（18 h 上限）時自動以同模式重跑，每次在 log 寫一行，至多 30 次；其他退出碼原樣傳出。`scripts/e1_v2_watchdog.sh` 為學校 watchdog 範本（flock、pgrep、tmux `e1v2`、連續 3 次無進度即停並建立 `~/E1V2_PAUSED`、遇其他 GPU 程式不重啟），只負責 `all`，絕不啟動 testcheck 或 test。
