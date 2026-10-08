# 方向二加速（CUDA Graph + 多 cell 並行）紅隊報告

日期：2026-10-08。範圍：分支 `direction2-speedup`（89ef840、d090b98、a3c13d3，基於 cc26592）。
目標：找出任何會讓 S1 結果與 eager 程式不是 bitwise 相同、或讓學校機器上無人看顧的 S1 跑壞的問題。
方法：讀程式碼並附行號、跑 CPU 測試、在變異（mutation）複本上驗證測試的偵測力、用 `gpujob` 在本機 GPU（RTX 5070 Ti、torch 2.12.0+cu130）做補充實驗。repo 本身沒有任何修改（所有變異只在 scratchpad 的複本上做）；本檔是唯一新增檔案。

## 0. 結論

**沒有找到 BLOCKER。HIGH 兩項，都是「切換前必須補的驗收」，不是已知的位元差異。**

- 在本機（同機同 torch）上，我找不到任何 graph 與 eager 不同的情況：S1 全部 operator 類型、兩個 anchor、條件 cell（s=0.01）、main r=0.9，都在**正式大小**（B=65792、T=20000）上逐位相同；驗證路徑（B=64、T_val=100000）也逐位相同。
- 變異測試（CPU 與 GPU）都被測試抓到，測試不是在比兩次同一條路徑。
- **能不能切到學校：可以，但有條件。** 前提是先在學校的 torch 2.6.0+cu126／sm_75 上跑完第 9.1 節的 (1)–(4)，並且把驗收補上兩點（見 H1、H2）。沒有補就切換，等於把「從未在 torch 2.6 上跑過 graph」的風險押在無人看顧的 S1 上。
- 建議第一階段只用 `--graph`（不併行）。`--concurrent` 另有一項 MEDIUM（孤兒 worker），先不要開。

## 1. 發現一覽

| # | 等級 | 標題 |
|---|---|---|
| H1 | HIGH | 驗收腳本不會跑到驗證路徑（每 50 代一次，決定 best checkpoint）的正式大小比對 |
| H2 | HIGH（條件式） | graph 從未在 torch 2.6／sm_75／vGPU 上跑過；失敗時 watchdog 會在 3 次重啟後讓 S1 停擺 |
| M1 | MEDIUM | `--concurrent`：runner 被 SIGKILL 時 worker 變孤兒，仍會把手上的段跑完，與 watchdog 重啟的新 runner 重複寫同一份 checkpoint |
| M2 | MEDIUM（已緩解） | T39 沒跑完：覆蓋範圍是 eager 的 hook／adapter 路徑；風險低；已拆片補跑，全部通過（第 7 節） |
| M3 | MEDIUM | 多執行緒 capture 的安全性依賴 `thread_local` 與建圖鎖，只在 torch 2.12 驗過 |
| M4 | MEDIUM | 沒有逐段逾時：graph 建圖或 worker 卡死時，runner 與 watchdog 都看不出來 |
| L1 | LOW | `serve` 的 `reap()` 與 `already has a running segment` 的極小競態 |
| L2 | LOW | worker 的 stdout 同時是 JSON 通道；多執行緒 `print` 理論上可吞掉一行事件 |
| L3 | LOW | S1 不會用到的路徑（O4、O6、D3、adapter、warm start）沒有 graph 覆蓋，會靜默走 eager（設計如此） |
| L4 | LOW | `_d4_resid` 在 eta 為張量時不再轉型別／裝置 |
| L5 | LOW | 凍結 golden hash 測試在 torch 版本不同時 skip，學校上沒有這一道 |
| L6 | LOW | 其他小事（`isfinite` 時機、`print_status` 的倍率會混合 eager 與 graph 的段） |

## 2. 攻擊面 1：多執行緒（B）

**RNG 隔離：通過。**
- 每個 `SimState` 自己建 generator：`arbitration/rl/env.py:48-55`（`_make_gens`，每個種子 3 個 `torch.Generator`，各自 `manual_seed(s + off)`）。抽取只用 `generator=g`：`env.py:60-63`、`env.py:178`。
- 我 grep 了 `arbitration/` 與 `scripts/`：沒有 `torch.manual_seed`、沒有不帶 generator 的 `torch.rand/randn`、沒有 `np.random.seed`、沒有全域 `random.seed`。numpy 用的是實例化的 `default_rng`：`cmaes.py:20`（每個 optimizer 一個）、`train.py:125,186`（只有 warm start）。`random.Random(9000+run_id)` 是實例：`train.py:48`。
- 全域設定：`torch.set_*`、`torch.backends.*`、`use_deterministic_algorithms`、`set_num_threads`、`set_default_dtype` 在 `arbitration/`、`scripts/` 都是 0 筆。沒有 cuBLAS／matmul（MLP 是 `(obs*W1).sum(-2)`），所以 cuBLAS workspace 不會在執行緒間共用。
- 唯一的共用可變狀態：`train.py:19` 的 `_PERM_CACHE`（`train.py:46-50`）。多個執行緒同時填同一個 `run_id` 是「檢查再設定」的競態，但內容只由 `random.Random(9000+run_id)` 決定，所以是良性的。
- 執行緒本地：每段在新執行緒跑，`torch` 的 grad mode 預設為開，但所有張量都不需要梯度，`_step` 另有 `@torch.no_grad()`。結果不變。

**CMA-ES 的 numpy 在多執行緒下是否與單執行緒逐位相同：本機通過。** 我跑了 3 個 `cmaes` 最佳化器各 60 代（n=55、lam=256，含 `eigh`）：單執行緒一次，與 3 條執行緒同時各 3 次，hash 全部相同（本機 numpy 2.4.6／scipy-openblas 0.3.31）。學校的 numpy／OpenBLAS 版本與核心數不同，這一點只有 `d2_graph_check.py` 的 concurrent 比較能在學校驗證（它有跑 `tell`）。

**capture 期間另一執行緒在 launch：**
- 設計：`capture_error_mode="thread_local"`（`graphsim.py:222`、`graphsim.py:185`），一次只允許一個 lane 建圖（`graphsim.py:32`、`graphsim.py:111`）。
- capture 是以 stream 為單位的；其他 lane 在自己的 `torch.cuda.Stream` 上 launch，不會被捕捉。`torch.cuda.graph` 預設使用的 capture stream 是行程內共用的一條，但被建圖鎖序列化。
- **壓力實驗（我加的，job 41）：** 4 條執行緒，每條 40 輪，每輪都用全新的 `GraphSim` 重新建圖（warm-up＋capture），同時另外 3 條在 replay、跑 eager 尾段、抽 RNG block、`.cpu()`，6 種 operator 輪替，結果都與事先算好的 eager 結果逐位比對：**0 筆不同、0 個例外**（80 秒）。上一輪 6 輪的版本（job 40）同樣通過。
- 殘餘風險（M3）：這只在 torch 2.12／sm_120 上驗過。torch 2.6 的 caching allocator 在別的執行緒 capture 時如果遇到記憶體壓力，`release_cached_blocks` 不能執行，可能出現假性 OOM（12 GB vGPU，估計總用量約 2 GB，不太會碰到）。

## 3. 攻擊面 2：A（t 改成裝置端 float64）

**eager 路徑的數值有沒有被改變：沒有，有四重證據。**
1. 程式碼：eager 時 `tt = t`（Python int），`env.py:311`、`vulns.py:386`；`_row()` 在 `_feed is None` 時回傳原本的 `getattr(st, name)[s][st.sidx]`（`env.py:238-244`）；`_tfull` 在 t 不是張量時就是原本的 `torch.full_like`（`env.py:247-249`）；D2 的 `tf = float(t)` 與原本相同（`vulns.py` 的 `_ExtD2.release`）。所有 `t` 的使用都是 float64 的比較、加、減，`int→double` 在 `t ≤ 1e5` 時是精確的。
2. 對照 cc26592 原始碼樹的一次性雜湊：41 組（plan 8.5 節），CPU 與 GPU 相同。
3. 永久測試 `test_eager_path_unchanged_by_refactor_cpu`（`test_d2_speed.py:281`）。
4. 我重跑 CPU 測試：`test_d2_speed.py`＋`test_d2_multi.py` 45 項全過（204 秒）。

**graph 與 eager 在正式大小：**我寫了一支比較腳本（`scratchpad/prodcmp.py`，不在 repo 內），對下列 cell 用它們在 run_plan 裡的真實設定（λ=256、n_S=256、B=65792、T=20000、真實 knob，包含 `D2=1100`、`O5=27` 這兩個整數 knob），用 257 列隨機 θ，完整比較 `final` 與 `snap` 的每個欄位（`torch.equal`）：

| 路徑 | cell | 結果 |
|---|---|---|
| `vulns._vstep` | s1_O1_s0.02、O2、O3、O5、D1（s0.02） | 全部相同（job 42） |
| `vulns._vstep` | s1_D1_s0.01、s1_D2_s0.01、s1_O1_s0.01（條件 cell） | 全部相同（job 43） |
| `env._step` | s1_NAIVE_anchor、s1_M4rw0_anchor（r=0.99、M4）、main_M3C_r0.9 | 全部相同（job 42、43） |
| D2_s0.02、D4_s0.02 | 已有 `graph_check_local.json`（學校 eager checkpoint 續跑） | 相同（既有） |

也就是 S1 的 7 個 operator、2 個 anchor、3 個條件 cell 與 main 的型態都有在正式大小驗過。（`main_M3C_r0.5` 與 r0.9 只差 `r`／G_gens，同一條 `env._step` 路徑。）

**K=32 尾段、RNG 邊界：**
- 我手算了正式 T 的事件表（`plan_events`）：T=20000、Ts=17500、K=32 時，每個 1024 區塊從奇數步開始、正好是 32 個 replay，Ts 前的區段 92 步＝2 個 replay＋28 步 eager，之後從 17501（奇數）接回 replay，最後一個區塊 544＝17×32。每代只有 28 步 eager。驗證 T=100000、Ts=97500 同理（28 步 eager）。與 plan 第 2 節吻合。
- 小形狀測試用 K=4 且 T=70，涵蓋「Ts 為奇數／偶數」「T 不是 K 的倍數」「跨 1024 邊界」（`test_G1_rng_block_boundary_and_tail`）。K=32 的尾段只在 G3（T=200、Ts=181，B=65792）、G9（NAIVE pilot 前 3 代）與我上面的正式大小比較涵蓋，都通過。
- 變異測試 Mu8（只在 t0≥1024 之後把 t 偏移 1e-6）只有跨區塊的測試會抓到，確實被 `test_G1_rng_block_boundary_and_tail` 抓到，代表該測試不是擺設。

**驗證路徑：**見 H1。

**warm start 路徑：**`pilot_decision.json` 已決定 `chosen_start = cold`（`results/direction2/pilot_decision.json`），所有 S1 cell 的 `theta0 = None`（`train.py:355` 的條件不成立），warm start 與 adapter 路徑不會被執行。

## 4. 攻擊面 3：測試本身（mutation 測試）

做法：把 repo 複製到 scratchpad（`mut/`、`mutgpu/`），在複本上改 graph 路徑，跑測試，確認會失敗。原 repo 完全沒動。

| 變異 | 內容 | 結果 |
|---|---|---|
| Mu1 | `graphsim.py` `tvec = t_dev + arK + 1e-9` | CPU：`test_G1_simulate_graph_equals_eager[70-61-None-None-cpu]` 失敗 |
| Mu1（GPU 真 capture） | 同上，複本 `mutgpu`，`gpujob` | **28 項 GPU 測試失敗**，訊息為 `/final/reg_b`、`/final/susp: not bitwise equal`（是不等，不是例外）（job 45） |
| Mu2 | feed 的 `E[0,0,0]` 加 1 ulp | CPU 失敗（G1） |
| Mu3 | 尾段條件 `b-a+1 >= K` 改 `>= K-2` | `test_C1_plan_events_cover_and_rules` 失敗 |
| Mu4 | 只在 graph 路徑 `tt * (1+2.3e-16)` | G1[O1] 失敗 |
| Mu6 | feed 的 `Rn` 最後一列複製倒數第二列 | G1[D4] 失敗 |
| Mu7 | 只在 graph 路徑把第 3 步的 `Ua` 乘 0.9 | G1[None] 失敗 |
| Mu8 | 只在 t0≥1024 的 replay 把 t 偏移 1e-6 | `test_G1_rng_block_boundary_and_tail` 失敗 |
| Mu5 | 只在 graph 路徑把第 3 步的 `Ua` 加 1 ulp | **通過（沒有被抓到）**。這是等價變異：`Ua` 只用來做 `Ua < p` 的比較，1 ulp 的差不會翻轉任何決定，所以沒有任何輸出會變。不是測試的缺陷。 |

結論：測試真的比較了 eager 與 graph，旗標有生效（GPU 變異會讓 `g.replay` 的結果改變）。其他的證據：`graphsim.stats["replays"] >= 10 and builds == 1`（G3）。

## 5. 攻擊面 4：runner

- **失敗即停：**`run_queue_concurrent`（`d2_run_queue.py:439-496`）失敗後不再派新段，已在跑的段跑完並 commit，exit 2。測試 `test_concurrent_failure_stops_new_segments_but_commits_the_running_ones`、`..._worker_crash_is_a_failure`。
- **續跑跳過已完成段：**`todo` 以 `is_done`（part status=done）過濾（`:419`），late commit 在 `:422-427`。
- **同 cell 段序：**`chains` 以 deque 保序，`busy` 集合保證同 cell 同時只有一段（`:428-440`）。段間的 checkpoint 依賴：下一段只在上一段 `done` 事件、commit 完成之後才派（commit 在 `:497-505`，派工在迴圈開頭 `:439`）。
- **兩個 cell 同時完成時 git 競爭：沒有。**commit／push 全在 runner 主執行緒依事件序進行（`:478-510`），worker 不碰 git；`commit_paths` 只 `add`／`commit` 該段自己的 part＋ckpt 路徑。worker 側的 `verify_repo_state`（`plan.py:65-76`）只做唯讀的 `git ls-files`／`diff`／`merge-base`，不需要 index.lock。
- **條件 cell 的順序：**`check_conditional`（`plan.py:140`）讀的是已登記的 M-3 紀錄，不依賴其他 S1 cell 的結果，所以併行改變順序不會讓條件 cell 提早或延後得到不同的答案；若檔案不可用會丟例外使該段失敗（失敗即停，安全）。
- **manifest 語意：**`build_jobs` 與切分只看 `est_per_run`（`:73-88`），不看速度；段數與檔名不變。`preflight` 的 manifest 比對只是警告。`print_status` 的「速度比」會把 eager 與 graph 的段混在一起取中位數（L6）。
- **與 watchdog 相容：**我看不到 `~/s1_watchdog.sh`（不在 repo，也不碰學校機器），依文字描述（`pgrep d2_run_queue --stage s1`、固定 eager 指令、flock、連續 3 次重啟上限）判斷：
  - 加了 `--graph`／`--concurrent N` 之後，runner 的命令列仍含 `d2_run_queue` 與 `--stage s1`，pgrep 仍找得到。worker（`python -m arbitration.rl.multitrain --serve`）的命令列不含 `d2_run_queue`，不會被誤判成第二個 runner。
  - **watchdog 用沒有 `--graph` 的舊指令重啟，結果仍 bitwise 相同**：checkpoint 與 part 檔沒有任何 graph 相關內容（`train.py:191-196` 的 `_save_ckpt`；測試 `test_G5_resume_in_pieces_and_switch_eager_to_graph_mid_run`），且 eager 與 graph 逐位相同，段與段之間可以混用。代價只是變慢。
  - 注意 `pgrep -f` 的字串要實際比對 `d2_run_queue.py --stage s1`（有 `.py`）；請在切換時用 `pgrep -af` 實際看一次，不要憑印象。
- **M1（孤兒 worker）：**`multitrain.serve` 對 stdin EOF 與 `quit` 的處理相同（`multitrain.py:128-133`、`:156`）：兩者都 break，然後 `join()` 所有執行中的段（最長一段可達 1500 秒量級）。runner 被 SIGKILL／OOM killer 殺掉時，worker 的 stdin 會 EOF，但 worker 會把手上的段繼續跑完。此時 watchdog 15 分鐘內看到沒有 runner，就啟動新的 runner，新 runner 會對同一 cell 的同一段從 checkpoint 重跑。兩個程序寫同一個 ckpt／part：寫入是原子的（`train.py:191-196`、`parts.write_part` 用 tmp＋`os.replace`），而且兩邊算出的內容逐位相同（只差 `wall_s`），所以**不會造成數值錯誤**，但 GPU 時間被浪費、記憶體加倍、`.queue.lock` 擋不住（flock 只在 runner 上）。eager 的循序 runner 也有同樣的暴露面（孤兒 `train.py` 子程序），但最多 1 個，併行時最多 N 個。
  - 建議：`serve` 區分 `quit` 與 EOF；EOF（沒有 quit）就 `os._exit(1)`，或在 `ProcWorker` 啟動時設 `prctl(PR_SET_PDEATHSIG, SIGKILL)`。
- **M4（沒有逐段逾時）：**`worker.events.get()` 沒有 timeout（`:467`）；graph 建圖卡住（例如驅動問題）時 runner 永遠等待，watchdog 只看程序是否活著，不會處理。eager 也沒有逾時，但 graph 多了「建圖」這個新的卡住點。建議給 `events.get(timeout=...)`，超過（例如預估時間的 5 倍）就視為失敗。
- **L1：**`serve` 的 `reap()` 靠 `Thread.is_alive()`（`multitrain.py:124-126`），`done` 事件是在執行緒結束之前 emit 的（`multitrain.py:152`）。runner 收到事件後還要讀 part、`git add/commit`（毫秒～秒），所以同一 cell 的下一段到達時執行緒早已結束；只有 `--no-commit` 且極快時才有機會看到 `already has a running segment`，結果是整個佇列誤停（不是數值問題）。
- **L2：**`ProcWorker._read` 只接受以 `{` 開頭的行（`d2_run_queue.py:362-370`）。worker 的 stdout 同時是 JSON 通道，而 `train.prepare` 在 warm start 時會 `print`（`train.py:357`）。多執行緒時 `print` 的文字與換行是兩次寫入，可能與 `emit` 交錯而弄壞一行事件，runner 會永遠等不到 `done`。S1 全部是 cold start，不會 `print`，所以目前不會發生；改成把 worker 的 stdout 重導到 stderr、只用專用 fd 傳事件更穩。

## 6. 攻擊面 5：從 eager checkpoint 切到 graph 續跑

- checkpoint 內容：`train.py:191-196` 存 `state`（`gen`、records、best、三條曲線）加 `opt.state_dict()`（`cmaes.py:104-106`：`mean, sigma, pc, ps, C, gen, rng`）。`load_state_dict` 之後會 `_eig()` 重算特徵分解（`cmaes.py:118`），而 `tell` 本來每代都 `_eig()`（`cmaes.py:99`），所以續跑與不中斷等價。JSON 的浮點 round-trip 是精確的。
- 模擬器端**沒有需要還原的 RNG 狀態**：每一代都用 `seed_block(run_id, g, n_S)` 重新建 generator（`env.py:48-55`），所以 checkpoint 只需要 CMA 狀態與代數。
- 實測：`graph_check_local.json` 以學校 eager 產生的 `s1_D2_s0.02` g36 checkpoint 續跑 2 代，graph、並行都與 eager 逐位相同；測試 G5 做了 eager→graph→eager。
- 結論：完整。唯一沒有涵蓋的是**那 2 代不含驗證**（H1）。

## 7. 攻擊面 6：T39

- T39（`tests/direction2/test_d2_s1.py:182`）對登記表每個（漏洞，大小）用 `vulns.recompute_delta` 重算增量（64 個校準種子×36 個手寫策略 S_HW＋最佳 θ，T=1e5），要求在 1e-9 內重現**已凍結**的 `vuln_suite.json`，並檢查 CI、最佳手寫策略名稱等。它走的是 **eager 的 adapter＋hook 路徑**（AdapterPolicy，沒有 graph）。
- 它覆蓋什麼：本次重構改到 eager 路徑的三處（`_row`、`tt`／`_tfull`、`_d4_resid`）在真實 knob、T=1e5、36 個策略下，是否還重現凍結的數字。小形狀的雜湊與其他 GPU 測試（T01/T02/T08/T36/T41）覆蓋同一份程式，但不是同一個資料量。
- 風險：低。理由是 eager 時 `tt` 就是原來的 Python int、`_row` 回傳原表達式，而且 41 組雜湊 CPU／GPU 都相同。
- 為什麼超過 25 分鐘：我量到**一個登記項約 3 分鐘**（`D2_T39_SLICE=11:12`，job 46：3 分 3 秒通過），整個登記表 14 個項目約 42–45 分鐘。測試本身已經支援 `D2_T39_SLICE="i:j"`（`test_d2_s1.py:207-212`），切片就不會超時。
- 補跑結果：
  - `11:12`（D4 s0.02）：通過，183 秒（job 46）。
  - `0:5`（D1×2、D2×2、O1 s0.01）：通過，873 秒（job 47）。
  - `5:10`：通過，780 秒（job 48）。
  - `10:17`（D4×2、O5、D3、anchors）：通過，687 秒（job 49）。
  - 合計 `0:17` 全部覆蓋，**整個登記表在 eager 路徑上重現凍結數字**（各片 ≤ 15 分鐘；只缺最後的 `n_checked == 13` 總數斷言，它只在不切片時才檢查，13 項由各片逐項檢查）。- 建議：在學校機上也用切片補跑（每片 ≤ 5 項，約 15–25 分鐘；RTX8000 較慢，建議每片 ≤ 3 項），或在本機用 `gpujob` 切成三片（`0:5`、`5:10`、`10:17`）。學校機器上要先確認凍結 registry 在 RTX8000 上的容許範圍：既有紀錄寫「T77 放寬到浮點最後一位」，T39 的 1e-9 在不同 GPU 上本來就不保證，所以學校上跑 T39 的失敗不代表 graph 有問題。

## 8. 發現的詳細說明

### H1（HIGH）驗收不會跑到驗證路徑的正式大小比對
- 證據：驗證在 `(g + 1) % val_every == 0` 時才執行（`train.py:255`，val_every=50），`d2_graph_check.py` 預設 `--gens 2`（`:71`），從 checkpoint 的 g36（D2）或 g0（D4）往後 2 代，不會碰到第 50 代。所以驗收看不到驗證路徑：B=64、T_val=100000、約 98 個 RNG 區塊、Ts=97500（28 步 eager 尾段）、另一張獨立的圖。S1 每個 cell 做 8 次驗證，G_val 決定 best checkpoint 與所有 S1 判讀。本機的 G7 只是小形狀（T_val=40）。
- 本機補驗（job 44，我的 `prodcmp.py val`）：對 D2（int knob 1100）、O5（int knob 27）、M4 anchor 各用一個隨機 θ，在正式的 T_val=100000 上算 `val_fn`，graph 與 eager **逐位相同**（例：D2 `-0.11922351474436142`＝`-0.11922351474436142`）。
- 修正：切換前在學校加跑同樣的比較，或讓 `d2_graph_check.py` 支援「跨過第 50 的倍數」（例如 `--gens` 要涵蓋 val，或新增 `--val` 直接比 `val_fn(θ)`）。成本：學校 eager 約 3.5 分鐘＋graph 約 0.5 分鐘每個 cell。

### H2（HIGH，條件式）torch 2.6／sm_75／vGPU 從未驗證；失敗模式
- 證據：plan 8.7 第 1 點自己承認。本機是 2.12.0+cu130／sm_120。差異來源：graph 的 capture API 在 2.6 存在（`capture_error_mode` 參數也在），但 allocator、`index_put_`、`ndtri`、`one_hot` 在 capture 中的行為、vGPU（GRID 12Q）對 graph 的支援都沒有測。
- 好消息：任何 capture 失敗是**大聲的**（例外→該段 failed→runner exit 2），`from_env` 不會靜默退回 eager（`graphsim.py` `from_env`），測試 C5 鎖住了這點。靜默的位元差異則由 9.1 的比較擋下。
- 壞消息：失敗後 runner 退出，watchdog 若已被改成帶 `--graph` 的指令，會用同一個壞指令重啟 3 次然後放棄，S1 在無人看顧下停擺。
- 修正：(1) 9.1 的 (1)–(4) 必須在學校全過才切換；(2) watchdog 的指令改成「先 graph、失敗就 eager」：`d2_run_queue.py ... --graph || d2_run_queue.py ...`，或在 runner 加 `--graph-fallback`（graph 建圖或例外失敗時，同一段改用 eager 重跑；因為兩者逐位相同，這是安全的）。

### M1（MEDIUM）孤兒 worker：見第 5 節。
### M2（MEDIUM）T39：見第 7 節。
### M3（MEDIUM）多執行緒 capture 只在 torch 2.12 驗過：見第 2 節。壓力實驗已通過；在學校請重跑 `d2_graph_check.py` 的 concurrent 比較（它會同時建圖）。
### M4（MEDIUM）無逐段逾時：見第 5 節。

### L3
S1 用不到：O4、O6、D3（`OPS` 測試表只有 None/O1/O2/O3/O5/D1/D2/D4，`test_d2_speed.py:20`）、adapter policy、hook_log、public_log、warm_obs。`simulate_vuln`／`env.simulate` 遇到這些會走 eager（`vulns.py:624`、`env.py:534`），不是靜默壞掉，只是沒有加速。
### L4
`_d4_resid`（`vulns.py:314-318`）原本是 `torch.as_tensor(eta, dtype=F64, device=...)`，現在 eta 為張量時直接使用。`simulate_vuln` 建張量 knob 時已是 F64 且在同一裝置（`vulns.py:612`），所以等價；S1 的 D4 knob 是 Python float（0.047），走 `torch.full`。
### L5
`test_eager_path_unchanged_by_refactor_cpu`（`test_d2_speed.py:281-287`）在 torch 版本與 golden 不同時 skip。學校是 2.6.0，所以學校上只剩 T01/T02/T08/T36/T41（9.1 的 (3)）擋 eager 路徑。這已足夠，但請確認 (3) 真的在學校跑過。
### L6
- graph 模式的 `isfinite` 檢查晚一個區塊（plan 已記載），結果不變。
- `print_status` 的速度倍率（`d2_run_queue.py:593-606`）把 eager 與 graph 的段一起取中位數，切換後預測剩餘時間會不準，但只是顯示。
- runner 的 commit trailer（`d2_run_queue.py:46-47`）寫的是 "Claude Opus 5.5 (1M context)"，與目前模型不同，只是署名，與結果無關。

## 9. 建議的切換前清單（在 plan 第 9.1 節之上加）

1. 在學校 torch 2.6.0 上跑 plan 9.1 的 (1)–(4)。
2. **補 H1：**同時比較 `val_fn(θ)` 的 eager／graph（B=64、T_val=100000），至少 D2 與 M4 兩個 cell。
3. **補 H2：**watchdog 的指令做 graph→eager 後備，或用 `--graph-fallback`。
4. 第一階段只用 `--graph`；`--concurrent` 等修掉 M1（EOF 不 `quit` 就結束）再開。
5. 用 `pgrep -af d2_run_queue` 實際確認 watchdog 的比對字串能匹配新的命令列。
6. T39 本機已用切片全部跑過；學校上可選。

## 10. 本次實驗清單（可重現）

| 項目 | 內容 | 結果 |
|---|---|---|
| CPU | `pytest tests/direction2/test_d2_speed.py tests/direction2/test_d2_multi.py -m "not gpu" -p no:metadata`（CUDA 隱藏） | 45 passed |
| CPU | 變異測試 Mu1–Mu8（複本） | 見第 4 節 |
| CPU | CMA-ES 多執行緒雜湊（`scratchpad/cma_threads.py`） | 相同 |
| job 40/41 | 4 執行緒壓力（`scratchpad/stress_threads.py`，6 輪／40 輪） | 通過 |
| job 42/43 | 正式大小 graph vs eager（`scratchpad/prodcmp.py prod`） | 11 個 cell 全部相同 |
| job 44 | 正式大小驗證（`prodcmp.py val`） | 3 個 cell 全部相同 |
| job 45 | GPU 變異（`mutgpu`） | 28 項失敗（符合預期） |
| job 46–49 | T39 切片（11:12、0:5、5:10、10:17） | 4 片全部通過 |

scratchpad 的腳本不在 repo 內，建議把 `prodcmp.py` 整理後放進 `scripts/`（例如併入 `d2_graph_check.py`），讓學校上也能跑。

`git status` 結束時與開始時相同（只有原本就有的 `docs/direction2-notes.md` 修改），加上本檔 `docs/direction2-speedup-redteam.md`（未追蹤）。
