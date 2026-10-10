# 方向二 S1 最終評估：紅隊審查（正式 test 評估前）

審查對象：分支 direction2-freeze，commit b0f3b0a（`arbitration/rl/s1_eval.py`、`scripts/d2_s1_eval.py`、`arbitration/rl/final_eval.py` 新增的 `evaluate`／`already_evaluated`、`tests/direction2/test_d2_s1_eval.py`、`docs/direction2-s1-eval-impl.md`）。依據：`docs/direction2-spec.md` v1.3、`docs/direction2-freeze.md`、`results/direction2/FREEZE.md`、`run_plan.json`、`vuln_suite.json`、`docs/direction2-notes.md`。

做了什麼：逐行讀上列程式與規格條文；跑 `pytest tests/direction2/test_d2_s1_eval.py -m "not gpu"`（17 passed，用 `-p no:cacheprovider`，未寫檔）；用 `gpujob log 3` 讀實作者的煙霧測試紀錄；核對 S_HW 36 清單與凍結檔逐項相同（順序一致）、`stop.py` sha256 與 freeze.md 登記相同、`metrics.kappa()`＝2.8922015。沒有跑任何 test 評估、沒有 GPU、沒有寫 ledger；`git status` 與開始時相同（只有原本就有的 `docs/direction2-notes.md` 修改，加上本報告）。

## 結論

統計與程式邏輯沒有找到會改變判讀的偏離。沒有 BLOCKER。有 1 個 HIGH：實作者的煙霧測試已經用**真實最終檢查點**在**真實 test seeds** 上做過完整的 test 評估（3 個 cell），結果也寫進了 commit 的文件。這件事已經發生、不可逆，必須在正式評估前由使用者明示接受並登記揭露。另有 3 個 MEDIUM 與數個 LOW，其中兩個 MEDIUM 屬於執行程序（命令不能在學校機器直接用、val 與 test 沒有分開），執行前改掉即可。

**能否執行正式評估：可以，條件是先處理 H1（使用者接受並登記）、M1（命令改 tmux＋venv python）、M2（先只跑全部 val 並 commit）、M3（關掉學校的 watchdog）。**

---

## HIGH

### H1　煙霧測試已在真實 test seeds 上評估過 3 個真實 cell，並把結果寫進文件（違反 REQ-SEED-02「每個 (cell, 策略) 最終評估只做一次」的字面）

證據：
- `gpujob ls` 的 job #3：`scripts/d2_s1_eval.py run --test-mode --root .scratch/s1eval_full --graph --cells s1_O1_s0.02 s1_M4rw0_anchor s1_NAIVE_anchor`，T=1e5（未縮短）。`gpujob log 3` 輸出：
  - `[s1_O1_s0.02] test done ... G_RL 0.03636 [0.03166, 0.04105], D +0.01092 [+0.00645, +0.01539] MDE_D 0.00633 detected=True`
  - `[s1_M4rw0_anchor] test done ... G_RL 0.25102 ... passed=True`
  - `[s1_NAIVE_anchor] test done ... G_RL 0.31244 [0.31206, 0.31282], passed=True`
- 這三個 cell 讀的是真實檢查點（val 輸出的 ckpt gen 與 G_RL_val 與真實 run 相符），`--test-mode` 只是把 ledger 與輸出放到 `.scratch/`，所以**沒有寫入真正的 ledger**，但 test seeds 5000–5031 已被用來評估這三個最終策略。job #2（`--T 2000`，12 個 cell 全部）也在 test seeds 上跑過真實檢查點，但 horizon 不同，嚴重性較低。
- `docs/direction2-s1-eval-impl.md:26` 把「NAIVE 錨點結果（0.3124）」寫進 commit 的文件，是 test 數字外洩進入版本庫的直接證據。
- `.scratch/` 現在是空的（已清除），所以沒有殘留檔案，但紀錄在 gpujob 日誌與該 doc 裡。

影響評估：
- 這是確定性評估（同檢查點、同 seeds、同程式），正式跑會得到相同數字；沒有證據顯示實作者因為看到結果而改過程式（b0f3b0a 是單一 commit，內容與 doc 一致）。所以**不構成選擇性偏差，不改變任何判讀**。
- 但規格字面（REQ-SEED-02、§6.1）是「只評一次」，現在對 O1_s0.02、NAIVE、M4 這三格已經不是了；指揮官與實作者也已看過這三格的 test 結果。
- 不可逆，只能揭露。

建議：
1. 使用者在執行前明確決定接受（或不接受）；接受的話，在 `docs/direction2-notes.md` 與最終報告的揭露清單（REQ-STOP-06 同層級）寫明：O1_s0.02、NAIVE 錨點、M4 錨點的 test 評估在正式 ledger 之前被煙霧測試跑過一次，結果一致；其他 9 格是第一次。
2. 從現在起**凍結評估程式**：b0f3b0a（或審查後的修正 commit）之後，任何改動都會讓這三格變成「看到結果後改程式」。若 M1–M3 與 L 項需要修改，修改僅限不影響數值的部分（命令、檢查、記錄），並在 commit 訊息說明。
3. 之後所有煙霧測試只准用 `--T` 很短且不得載入真實檢查點，或乾脆用假資料（測試檔本來就有 FakeRun）。

---

## MEDIUM

### M1　文件裡的執行命令在學校機器不能用

`docs/direction2-s1-eval-impl.md:42-48` 寫 `gpujob python3 scripts/d2_s1_eval.py run ...`。學校沒有 `gpujob`；而且學校用 `.venv`（系統 python3 沒有 torch，`scripts/d2_s1_launch.sh` 也明寫 `PYTHON=<venv python>`）。三個「job」在學校必須在同一個 tmux 內**依序**跑（同時只一個 GPU 程式）。

建議的學校命令（工作樹乾淨、已 `git pull --ff-only` 到含 b0f3b0a 的 commit 之後）：
```bash
tmux new -s s1eval
cd ~/scarce-actuator-arbitration
PY=.venv/bin/python
$PY scripts/d2_s1_eval.py val  --graph 2>&1 | tee -a runlogs/s1_eval_val.log      # 步驟 1：全部 12 格只跑 val（見 M2）
# 檢查 val 輸出、commit 並 push results/direction2/s1_eval/*__val.json，再做步驟 2
$PY scripts/d2_s1_eval.py test --graph 2>&1 | tee -a runlogs/s1_eval_test.log     # 步驟 2：12 格 test（已完成的格自動跳過）
$PY scripts/d2_s1_eval.py status
$PY scripts/d2_s1_eval.py assemble
```
`--graph` 失敗（capture 錯誤）時不會自動退回 eager，直接重跑不加 `--graph`（兩者逐位相同）；之後的 `--cells` 範圍同理。`runlogs/` 目錄需存在（`mkdir -p`）。

### M2　val 與 test 兩段綁在同一個 cell 迴圈裡，MDE 凍結沒有「先於全部 test」的證據，且 val 階段的失敗會在部分 cell 已 test 之後才出現

- `scripts/d2_s1_eval.py:116-117`、`process_cell` 對每個 cell 依序做 val 再 test（`run`）。規格 REQ-MET-06 要求 MDE「在看到測試結果之前由驗證集算出並寫入結果檔」；現在的順序在**單格內**成立，但沒有 commit 層級的證據，且若第 k 格的 val 檢查（`s1_eval.py:211-213` 的 G_val 重算一致性、`s1_eval.py:228-229` 的 G*_ref 一致性、`_check_honest`）失敗，前 k−1 格的 test 已經被消耗。
- 這些檢查是 fail-closed（val 失敗不會寫 ledger），所以不會造成錯誤結果，但會造成「部分格已評、部分格被擋住」的半成品，容易誘發臨場放寬容差的決定。
- 建議：先 `val`（全部 12 格，不碰 test seeds，約 1 小時內），確認 12 個 val 檔都產生且檢查全過，commit 並 push val 檔（取得「MDE 凍結早於任何 test」的 git 順序證據），再 `test`。這樣所有可能中止的檢查都在 test 之前被排除。`val` 與 `test` 子命令已存在，不需改程式。

### M3　學校機器的 watchdog 可能在評估期間重啟訓練佇列

`docs/direction2-notes.md`（2026-10-08 一節）：`~/s1_watchdog.sh` 由 cron `@reboot` 與每 15 分鐘執行，沒有行程就重啟 `d2_s1_launch.sh`。若佇列已完成它應該什麼都不做，但「重啟路徑未實測」。評估期間若 watchdog 啟動第二個 GPU 程式，違反「同時只一個 GPU 程式」；若它同時 `git commit/push`，也會與評估中的工作樹互相干擾。

建議：評估前 `crontab -l` 確認並暫時註解掉該列（或確認佇列 manifest 已完成時 watchdog 會直接退出），`tmux ls` 與 `nvidia-smi` 確認沒有其他程序，再開跑。

---

## LOW

### L1　`detected_plus_h0` 與 `MDE_D_plus_h0` 不一致（只報告，不入判定）

`s1_eval.py:304-306` 對 `hw_plus_h0` 這個 policy 的 record 用 `mde_d_frozen`（即主要的 MDE_D），`s1_eval.py:328-329` 卻同時輸出 `MDE_D_plus_h0=val["mde_D_plus_h0"]`（另一個驗證 MDE）。輸出的 `detected_plus_h0` 是以主 MDE_D 判的，標示的 MDE 卻是另一個。REQ-MET-04 只說 D^{+H0}「同式計算」，不用於判定，所以不影響結論，但文件中的「這一格 D^{+H0} 是否偵測」會與其標示的 MDE 對不上。建議其中之一改用一致的 MDE，或不輸出 `detected_plus_h0`。

### L2　REQ-HW-02(b)（含 oracle 的最佳手寫）沒有實作，且 doc §2 第 11 項沒有列入「不計算的項目」

REQ-HW-02 要求另外報告 (a) `G_手寫^{+H0}`（已做）與 (b) 含 8 個 oracle 的最佳手寫（附帶、不入判定）。實作未含 (b)，`docs/direction2-s1-eval-impl.md:33` 的清單只列 R/T、α、STOP-06 揭露。僅屬文件漏列；(b) 不在 REQ-S1-18 的輸出清單內，所以可留給最終報告腳本，但要補列。

### L3　`test_raw` 檔沒有綁定檢查點與程式版本

`s1_eval.py:274-276` 若 `*__test_raw.json` 已存在就直接採用。若有人在 raw 寫入後（ledger 寫入前）發現 bug 並改了程式，舊的 raw 會被靜默沿用。raw 內只有 cell、seeds、T_score 與增益，沒有 `theta_sha256`、git sha。建議 raw 加上 `theta_sha256` 與 git sha，讀回時比對。目前的風險只存在於「raw 已寫、ledger 未寫」這個極窄的視窗，且是 fail-open 方向（沿用舊值），所以標 LOW。

### L4　ledger 寫入不是原子的

`final_eval.py:86-88` 以 `open(ledger, "w")` 直接覆寫。極端情況（寫到一半斷電或被 kill）會留下被截斷的 JSON；`_load_ledger` 之後會丟 `JSONDecodeError`（fail-closed，不會讓同一個 (cell, policy) 被重評），但 ledger 內的紀錄會遺失，只能靠 raw 與 test 檔重建。建議改成與 `s1_eval._write_json` 相同的 tmp＋`os.replace`。注意這會改動 `final_eval.py`（已是 commit 內的檔案），屬 H1 第 2 點允許的「不影響數值」改動。

### L5　`process_cell` 不呼叫 `plan.check_reuse`

`scripts/d2_s1_eval.py:49` 對 NAIVE 錨點只做 `source_cell` 的名稱映射，沒有重做 `plan.check_reuse`（`plan.py:162`）的逐項比對。實際上由 val 階段的 G_val 重算一致性（`s1_eval.py:211`）間接保證環境相同，且煙霧測試（job #3）通過，所以標 LOW。建議補呼叫以對齊 FREEZE.md 的「NAIVE 錨點重用紀錄」。

### L6　`verify_repo_state` 只檢查 run_plan.json 與 FREEZE.md 乾淨，不檢查評估程式本身

`d2_s1_eval.py:95` → `plan.py:65`。`s1_eval.py`、`final_eval.py`、`stop.py` 等有未提交修改時照樣開跑；輸出的 `git_sha`（`s1_eval.py:69-73`）是短 sha 且沒有 dirty 標記。建議執行前人工 `git status` 乾淨、`git log -1` 與預期 commit 相符，並把完整 sha 記在 notes。`vuln_suite.json` 的 sha 在評估時也沒有重驗（`assemble` 只讀取，`s1_eval.py:346`）；目前核對過 sha 與 freeze.md 相同。

### L7　`assemble` 在 cell 未完成時照樣寫 `s1.json`，且內含 J 判讀

`s1_eval.py:392-393`：S1 未通過 → `classify` 回 J2。未完成的 cell 計為未偵測是規格（REQ-S1-10、REQ-STOP-01）要求的，但 `s1.json` 的 `verdict` 在中途 assemble 時會是「J2」，容易被誤讀成最終判定。建議：`not_completed` 非空時在 `verdict` 加 `provisional=True`，或拒絕寫檔。

### L8　per-cell 立即印出 test 的 D 與 detected

`d2_s1_eval.py:71-73`。不違反規格（assemble 前看到結果不影響已凍結的流程），但配合 M2 的做法（先 val 全部）可以降低「看到前面幾格的結果之後在後面幾格臨場決定」的壓力。可不處理。

### L9　`scripts/run_d2_vulns.py:50` 有 test seed 5000

這是既有的恆等性（bitwise identity）檢查，只比較兩個環境輸出是否相同，不涉及增益或選取；不在 T-27 掃描範圍（只掃 `arbitration/rl`）。不影響本次評估，只記錄。

---

## 十三項歧義的逐項判定

| # | 實作選擇 | 判定 | 說明 |
|---|---|---|---|
| 1 | 手寫最佳只用 val（2000–2031），不用 val∪val2 | **正確，且是規格唯一讀法** | REQ-SEED-08(b)：「主實驗的手寫策略選取（REQ-HW-02 仍只用 val）」；REQ-HW-02「在驗證 seeds（2000–2031）」。`vuln_suite.json` 的 `G_HW_best_val` 是 64 seeds 的校準值，用於定義漏洞大小 s，不是評估基準。選到的策略可能與登錄檔不同（例如 D1 登錄為 dz=0.3），不構成問題。會不會改變判讀：不會（只影響 D 的基準策略，仍是 val 選、test 評）。 |
| 2 | 每格在自己的漏洞環境重新模擬 36 個 S_HW | **正確** | REQ-HW-03、REQ-S1-02。S_HW 清單與 freeze.md §7 逐項相同（已驗）。 |
| 3 | honest 基準：RL 批次用 θ=0 MLP、手寫批次用 adapter honest，需數值一致 | **正確，且更嚴** | `_check_honest`（`s1_eval.py:142-146`）相對 1e-12；失敗發生在 raw 寫入之前，不消耗 ledger。與 train.make_val_fn（`train.py:109-111`）同一慣例。 |
| 4 | 錨點 `G*_ref` 取凍結登記值（NAIVE：`pilot_decision.json` 0.27909；M4：`measure_m4.json` 0.11895）並於評估時重算核對 | **正確** | REQ-S1-08「以驗證集重新量測」、REQ-S1-20「0.8·G*_ref（驗證重測，預期約 0.28）」、REQ-S1-09/11。NAIVE 這個值就是 pilot 決策所用的門檻 0.2233 的基礎，必須一致；改用「手寫最佳 b=1.0」會與凍結檔不一致。判讀不變（NAIVE RL≈0.312 對兩種門檻都過；M4 RL 約 0.25 對 0.8×0.119=0.095 大幅通過），但這句話的依據是 H1 的煙霧數字，不得當作額外證據。M4 的 `counted`（CI 下界 0.1171>0）由 `m4_anchor_status` 判定，不看 test。 |
| 5 | M4 降級判斷依據是 W1 登記的驗證重測，不是 test | **正確** | REQ-S1-09。`s1_eval.py:231`。 |
| 6 | `mde_frozen`＝MDE_G（來自 RL 的 G_s）、`mde_d_frozen`＝MDE_D（來自驗證集 D_s）分流 | **正確** | REQ-MET-06「偵測用取 G_s（MDE_G）、漏洞偵測取 D_s（MDE_D）」；`metrics.final_eval`（`metrics.py:114-125`）已分流；`metrics.mde`（`metrics.py:98-106`）只接受 val seeds，κ=2.8922015，除以 √32。 |
| 7 | 最佳檢查點＝所有驗證記錄的 argmax、平手取較早，並與檢查點檔的 `best` 與最後一個 part 交叉核對 | **正確** | REQ-OPT-06。`train.select_checkpoint`（`train.py:59-67`）嚴格大於才換，平手取先；`load_best_checkpoint`（`s1_eval.py:91-122`）三處交叉核對。NAIVE pilot 的 G_val 在各 gen 完全相同（σ 坍縮），取 gen 50，與規則一致。 |
| 8 | 一個 (cell, policy) 的粒度：`rl`、`hw`、`hw_plus_h0` 各一次，同一次模擬 | **合理** | 一次模擬供三者使用，ledger 三個 key（`s1_eval.py:253-254`、`final_eval.py:71`）。比「每 policy 各模擬一次」更不容易出現 CRN 不一致。 |
| 9 | J 判讀：D 不存在，S1 通過→J7＋`incomplete=True`、未通過→J2 | **正確，沒有混淆 S1 閘門與 J 表** | S1 通過與否由 `s1.s1_gate_by_id`（`s1_eval.py:381`；`s1.py:82-103`）獨立判定，只看兩個錨點（M4 資訊性時略過）與 s=0.02 的 D1/D2/O1 三者，族內、盲區、s=0.01 都不進入。J 表只是把這個二值餵進 `classify`。規格 §9.2 的輸入欄位就是「S1（通過／未通過）」，N1 明寫 D 不存在時通過→J7＋incomplete、未通過→J2。所以「S1 通過判 J7 標未完成」是規格字面。唯一疑慮是 L7（中途 assemble 的 J2 容易被誤讀）。 |
| 10 | s=0.01 缺 cell：y′ 照計、分母 3、`gate.y_s001` 與 `not_completed` 記載 | **正確** | REQ-S1-15(a)、REQ-STOP-01。`s1_eval.py:370-376, 390`。`check_conditional` 在正式模式下先驗 M-3（`d2_s1_eval.py:44`）：三個 s=0.01 cell 的 `conditional` 欄位非空，M-3 記錄 `MDE_D_plan_max_pair`=0.00598，0.01≥0.00598，所以要跑；已用 `plan.mde_d_plan` 的邏輯核對（`plan.py:125-159`）。 |
| 11 | 不算 REQ-MET-05 的 R/T、α 與 STOP-06 揭露 | **可接受** | 不在 REQ-S1-18 的輸出清單內；但 REQ-HW-02(b) 也未做且未列入，見 L2。 |
| 12 | s=0.01 條件 cell 在正式模式先過 `check_conditional` | **正確** | REQ-S1-22。 |
| 13 | G_HW 與 D 的 CI 皆 t31 | **正確** | REQ-MET-03/04；`metrics.mean_ci`、`difference_ci`。 |

## 統計核對

- G_s、D_s：逐 seed 配對，D_s＝G_{RL,s}−G_{HW,s}（`s1_eval.py:297, 241`），同 seeds、同記分窗 T_score；CI 為 t_{0.975,31}=2.0395、ddof=1、/√32（`metrics.py:63-77`）。
- 偵測：漏洞用 `D_lo>0 and D≥MDE_D`（`metrics.found_rule`，REQ-MET-07 D 規則），MDE_D 取自 val 檔的 `mde_d_frozen`；錨點用 `s1.anchor_passes`（`G≥0.8·G*_ref、CI 下界>0、G≥MDE_G`，REQ-S1-11）。`validate_s1_output` 會再核對 `detected` 與 `passed` 與數字一致（`s1.py:375-391`）。
- MDE：只來自驗證集（`metrics.mde` 會拒絕非 val seeds），test 階段只讀 val 檔欄位（`s1_eval.py:261-266`），`final_eval.evaluate_on_test` 要求 record 帶 `mde_frozen`（`final_eval.py:64-66`）。
- CP LB：`metrics.cp_lower` 在 x=0 為 0、x=n 為 0.05^{1/n}（3/3→0.3684）；族外 n=3、類內 n=6（`s1.summary_table`），類內只計 s=0.02 的族外加族內（`s1_eval.py:374`）。

## 資料洩漏核對

- test seeds 只在 `final_eval.py` 內產生（`final_eval.evaluate` 在 `final_eval.py:101`）。`grep` 全庫：`s1_eval.py`、`d2_s1_eval.py` 沒有 test seed 常數或 `splits()["test"]`；測試 `test_new_code_never_names_a_test_seed` 與 T-27 掃描守門。
- 最佳檢查點只看驗證記錄（`load_best_checkpoint`）；S_HW 選取只看 val（`val_stage` 中 `select_on_validation`）。
- assemble 只讀 `*__test.json` 與 `vuln_suite.json`（`s1_eval.py:344-351`），沒有任何路徑在 assemble 前讀取 test 結果並回饋到訓練、選取或 MDE。
- 例外：H1（煙霧測試已碰過 test seeds）。

## 類別核對

- 計入 out-of-family：D1、D2、O1（`s1.py:23`，s=0.02 三個為通過標準，s=0.01 三個為 y′）。
- 族內 O2、O3、D4（`run_plan.json` 中 category=in_family，與 `vuln_suite.json`、`CATEGORY_OF` 一致；D4 的重分類已撤回，notes 502-513 行）：只描述，進入 x/6 但不進通過閘門。
- 盲區 O5：只描述。錨點：NAIVE（重用 `pilot_naive_cold`，`REUSED_CELLS`，`plan.py:25`）、M4rw0。D3、designer_1 不在計畫的 12 個 S1 cell 內，D3 只輸出登錄欄位。與 `gate`、`summary`、`validate_s1_output` 一致。

## 續跑與 ledger

- 當機在 raw 寫入前：沒有任何 ledger 項目，重跑安全。
- raw 已寫、ledger 未寫：重跑直接讀 raw（`s1_eval.py:274-276`），不重新模擬，安全。
- 部分 policy 已入 ledger：已入者由 raw 重建（`s1_eval.py:307-314`），未入者經 `final_eval.evaluate` 正常入帳；測試 `test_test_stage_is_once_per_policy_and_resumable_from_the_raw_file` 覆蓋。
- ledger 有、raw 遺失：明確拒絕、不重新模擬（`s1_eval.py:308-309`）。
- 風險點：L3（raw 未綁版本）、L4（ledger 非原子寫）。另外，ledger 與 raw 在 commit 之前只存在學校機器的工作樹；每個階段完成後立刻 commit＋push `results/direction2/final_eval_ledger.json` 與 `s1_eval/*`，避免 `git clean` 或重新 clone 後等於「重評」。
- `--graph`：RL 批次（全 MLP）走 graphsim，`[gpu]` 測試 `test_graph_stepping_of_the_RL_batch_equals_eager_bitwise`（本機 gpujob #1 通過）驗證與 eager 逐位相同；學校 RTX8000 上 `d2_graph_check.py --preflight` 的真實大小驗證路徑也已通過（notes 2026-10-08）。val 階段另有「重算 G_val 與訓練時存的 G_val 相對 1e-6」把關（`s1_eval.py:211-213`），本機煙霧對學校訓練的檢查點通過，表示跨機器重現性足夠。手寫批次含 adapter，一律 eager。

## 學校機器可用性

見 M1–M3。另外：`matplotlib>=3.7` 在 `requirements.txt`，所以學校 venv 應有；`assemble` 只需要 CPU，萬一缺套件也不會損失任何 test 結果（可在本機做）。
