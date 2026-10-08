# 方向二訓練加速計畫（結果必須與現行程式 bitwise 相同）

狀態（2026-10-08）：**A（CUDA Graphs）與 B（同程序多 cell 並行）已實作、已在本機測過**，在分支 `direction2-speedup`（從 cc26592 分出，未 push）。
第 1–7 節是實作前的計畫（含當時的推論數字，已被第 8 節的實測取代）；**第 8 節是實作與量測記錄，第 9 節是學校機器切換步驟草案**。
實驗凍結不變：預設行為完全是舊的 eager 程式（旗標關閉），打開旗標後 θ、CMA 狀態、適應度、驗證輸出、part 檔與 checkpoint 必須逐位元相同。

## 0. 結論（先看這裡）

| 項目 | 判斷 |
|---|---|
| 瓶頸 | 很可能是每步約 175–215 個小 kernel 的 launch／Python 開銷（B=65792，每個張量 0.5–1.6 MB，一個 kernel 只有幾 µs 的實際工作）。學校機 3.2 ms 對本機 1.1 ms 的比值（約 2.9）大於兩張卡的頻寬比（約 1.3），支持「學校機的 CPU 側 launch 更慢」。但 GPU 本身的 busy 比例還沒量，所以 A 的上限是推論。 |
| A. CUDA Graphs | **可行，建議先做**。RNG 不進 graph（每 1024 步在 eager 抽一次），沒有 Philox offset 問題。要改的是 `t`、`s`、`t%M` 這三個 Python 整數（改成裝置端張量或固定奇偶的兩份 graph），以及把狀態改成靜態緩衝區。同形狀、同 kernel，bitwise 應相同。預估倍數（推論）：本機 1.4–2.4 倍、學校機 2–3 倍；驗證（B=64，純 launch-bound）約 4–8 倍。 |
| B. 同程序合併多 cell | **不建議**。（1）S1 的 9 個 cell 幾乎各自使用不同 operator，要合併就得重寫成「逐元素選 operator」的聯集步驟，等於重寫凍結的模擬器；（2）現有 T-08 對 MLP 只保證 1e-12，不是 bitwise，批次大小一變，`sum(-2)` 的 reduction 設定可能變，bitwise 沒有保證；（3）加了 A 之後 GPU 已接近飽和，合併的邊際效益趨近 0。 |
| B'（零程式改動的替代）| 同一張卡同時開 2 個**獨立程序**（各跑不同 cell）。每個程序與單獨跑完全相同，bitwise 自動成立。預估 1.3–1.8 倍（推論，用 MPS 較好）。需要改 queue runner 的全域鎖，且與 CLAUDE.md 的 gpujob 規則衝突，只適合學校機、要使用者決定。 |
| 凍結守門 | 純效能修改**不會被擋**：`--plan`、FREEZE、HEAD 為 a72d3d7 後代、pilot sha 都只看 plan／FREEZE／git 狀態，不看程式內容。`frozen_budget` 在執行時根本不被讀取，代數與步數不變，不受影響。要注意的是 FREEZE.md 登記的「程式合併 sha」本來就已過期，要補登一筆（見第 5 節）。 |
| 主要風險 | (1) GPU 其實已經很忙，A 只有 1.2 倍；(2) 把 `t` 改成張量的重構在 eager 路徑也生效，必須用 CPU bitwise 測試鎖住；(3) 步驟快照點 Ts 不是 K 的倍數，要處理事件切分；(4) 驗證時 `isfinite` 檢查的時機改變；(5) 「bitwise」只在同一台機器、同一版 torch 內成立。 |

## 1. 現行熱路徑（每個 generation、每一步做什麼）

### 1.1 呼叫鏈與位置

```
train_segment                     arbitration/rl/train.py:238-265   (逐代迴圈)
  seed_block                      train.py:42-51                    (numpy/random, CPU)
  opt.ask()                       cmaes.py:~75                      (numpy, CPU)
  evaluate_generation             train.py:27-39
    sim_fn                        train.py:84-94
      MLPPolicy x (lam+1)         train.py:85                       (257 個 numpy 物件, CPU)
      simulate_vuln (operator)    vulns.py:566-631   迴圈 618-621   <- S1 的 7 個 operator cell 走這裡
      env.simulate (operator None) env.py:471-538   迴圈 514-516   <- NAIVE 錨點、M4 錨點、main 走這裡
      .cpu().numpy()              train.py:93                       (每代 1 次 D2H 同步)
  opt.tell                        cmaes.py:~95                      (numpy, CPU, n=55 的 eigh)
  val_fn (每 50 代)               train.py:108-111  B=2x32=64 個元素, T_val=100000
```

每代的 batch 是 B = (lam+1) x n_S = 257 x 256 = 65792 個元素（第 0 列是全零的誠實基準）。種子用 `np.unique` 去重，所以只有 256 個獨立隨機串流，`sidx` 把元素對應回種子。

### 1.2 每一步（t = 1..20000）的熱路徑

`vulns._vstep`（vulns.py:380-563）與 `env._step`（env.py:293-450）是同一套邏輯，順序如下：

1. **型別更新**：`_next_types`（env.py:237-240）：`z = rho*z + sq*E[t%1024][sidx]`，`u = transform(ndtr(z))`（adaaudit.py:42-44，`stack`+`sqrt`）。
2. **觀測**：`_obs_state`（env.py:243-259，每步建一個 dataclass）→ `ObsBuilder.step`（obs.py:458-462，`_check` 用 `dataclasses.fields`＋`vars`＋集合運算，純 Python 每步約數十 µs）→ `peek`（obs.py:431-456：約 25 個 clamp／where／stack／cat，含 2 個 `ndtri`）。
3. **策略**：`bank.act`（env.py:144-145）→ `mlp_forward`（policy.py:254-261）：`(obs.unsqueeze(-1) * W1).sum(-2)`，W1 是逐元素展開的 `[B,25,2]`（26 MB）。這是每步最大的記憶體流量：讀 W1、寫乘積、再讀乘積，約 80 MB／步，**這是約 0.1 ms 的硬下限**（推論，兩張卡都是）。
4. **報告與配置**：`report`＝`clamp(u1+d)`，`cat`，`torch.where(elig, v, -1)`，`max(1)`，`u.max(1)`，`gather`。
5. **統計累計**：`one_hot(ag_b,K)` 乘 `uw_b` 加進 `util_b`（vulns.py:471，env.py:386）。
6. **審計**：`audit_prob` hook、`pend_a[ar,slot]=...`、`pend_v[ar,slot]=...`（index_put_）。
7. **結算**：`pend_a[:,cs]`、`gather`、`one_hot`（vulns.py:500）、`phi_of`×2＋`ndtri`×2（audit_fix.py:31-34）、`resid`、`sxy/sxx`、`cusum` 更新、`flags`、`susp`（`t+L`）。
8. **hook／extension**：resid、cusum、pause_len、cusum_reset 等 hook；D1/D2/D4 另有 `_Ext*` 狀態（lose_run／calm／rh_a）。
9. **歷史**：`won_hist`、`d_hist` 的 `cat`；`hs.win/clean/rsw` 的 `where`。

### 1.3 量到的 kernel 數（CPU 上用 `TorchDispatchMode` 計 aten op，T=1030 步，B 很小）

每步 aten op 總數（含 view 類，這些不啟動 kernel）：

| operator | aten ops／步 |
|---|---|
| None（env._step 路徑，MLP） | 236 |
| O1 | 258 |
| O3 | 255 |
| O5 | 260 |
| D2 | 273 |
| D1 | 277 |
| D4 | 290 |

扣掉 view 類（`select` 約 28–34、`unsqueeze` 16、`slice` 4、`expand` 2、`lift_fresh` 1，合計約 52–60）與只在 CPU 上才有的 `one_hot` 檢查（見 1.4），**真正的 CUDA kernel 約 175（plain）到 215（D4）個／步**（`index_put_`、`one_hot`、`cat` 等少數 op 可能是 1–2 個 kernel，這個數是粗估）。1.08–1.36 ms／步 ÷ 約 200 ≈ 5–7 µs／kernel，正好落在 PyTorch eager 每個 op 的 CPU 派發成本（約 4–8 µs）上，所以「launch-bound」是合理的第一假設。

### 1.4 同步點與 Python 分支

- **每步的 `.item()`／CPU 同步：沒有**（CUDA 上）。dispatch 計數在 CPU 上看到每步 4 個（D1 為 6 個）`_local_scalar_dense`，來源是 `one_hot` 的 `min().item()`／`max().item()` 範圍檢查（env.py:386、410；vulns.py:471、500、234）。ATen 對 CUDA tensor 跳過這個檢查，所以這是 CPU 專屬現象。**這點要用 `torch.cuda.set_sync_debug_mode("error")` 在 GPU 上驗證**（計畫中的測試 G2）。
- `env._step` 有一個 `bool(torch.isfinite(d).all())`（env.py:313-316），只在 `t==1` 與 `t%1024==0` 執行（每 1024 步 1 次同步）。`_vstep` **沒有**這個檢查。
- 每代同步：`.cpu().numpy()`（train.py:93）、`val_fn` 的結果。
- **Python 分支依賴 GPU 值：沒有。** 所有 `if` 都依賴 Python 整數 `t`、`M`、`bank.use_*`、`op.hooks`（靜態），或 `mech == "M4"`。這是 A 可行的關鍵。
- **Python 整數隨步數變化的地方**（graph 化必須處理）：
  - `t`：`elig = t > susp`（vulns.py:396）、`susp = where(ohe, (t+Lvec)...)`（vulns.py:533）、`_obs_state` 的 `float(st.t+1)`（env.py:245-249，含 `torch.full_like(..., t)`）、D2 的 `float(t)`（vulns.py:259、261）、`(t-1) % (Wd+Hd)`（只在 adapter，MLP 訓練不會進）。
  - `s=(t-1)%BLK`：`st.Rn[s]`、`st.Ua[s]`（vulns.py:462、485），`E[st.t % BLK]`（env.py:239）。
  - `cs = t % M`（M=2）：`pend_a[:, cs]`、`pend_v[:, cs]`、`shv[:, cs]`（vulns.py:492-493、533）；`slot=(t+tau)%M`（vulns.py:476）。

### 1.5 每步新配置的張量

有，而且很多：`zeros`（約 6）、`zeros_like`（4）、`full_like`（2–4）、`arange`（2，`ar`、`kk` 每步重建，env.py:322-323）、`ones`、`clone`（4）、`cat`（4）、`stack`（3）、`_to_copy`（7）。走 PyTorch caching allocator，穩態沒有 `cudaMalloc`，不影響結果，只增加 CPU 時間。整個程序峰值記憶體只有約 154 MB（measure_vuln_timing.json），所以 graph 的私有記憶體池沒有壓力。

### 1.6 RNG 如何產生

- 每個獨立種子 s 有 3 個 CUDA `torch.Generator`，種子為 s、s+1e6、s+2e6（env.py:48-55）。
- 每代開頭：`z0 = randn(1,K)` 從第 1 個 generator 抽（env.py SimState.__init__，約 178 行），**必須先於第一個 block**。
- 每 1024 步（`_ensure_block`，env.py:227-234 → `_draw_block`，env.py:58-64）：對每個種子各一次 `randn(1024,1,3)`（E）、`rand(1024,1)`（Ua，**同一個 generator，在 E 之後**）、`rand(1024,1,5)`（Rn，第 2 個 generator）、`rand(1024,1,2)`（Rx，第 3 個 generator），再 `stack`。每個 block 約 4x256 = 1024 次 kernel＋4 次 stack，每代約 20 個 block，約 2 萬次 launch，推論約佔每代時間的 1% 以下（22–27 秒的一代裡 0.1–0.3 秒）。
- **步驟迴圈內完全沒有隨機數呼叫**；CMA-ES 的隨機數是 numpy（`default_rng(9000+run_id)`），在 CPU。
- 觀察：M3C 且沒有 adapter 時 `Rx` 不會被使用（只有 raid 與 `use_ad` 用到），而 Rx 有自己獨立的 generator，所以跳過 Rx 的抽取不會影響 E/Ua/Rn 的串流。這是可選的小優化（A0），不是必要。

## 2. 方案 A：CUDA Graphs

### 2.1 為什麼 RNG 不是問題

標準 CUDA graph 的 RNG 難題是：capture 時 Philox 的 seed／offset 要在 replay 時遞增，PyTorch 對預設 generator 與 `register_generator_state` 登記過的 generator 做了處理，但這裡有 768 個自訂 generator，逐個登記不實際。**解法是不把抽取放進 graph**：維持現行的「每 1024 步在 eager 抽一個 block」，抽完後 `copy_` 進 graph 讀取的靜態緩衝區（`E_buf`、`Ua_buf`、`Rn_buf`、`Rx_buf`）。這樣 graph 內沒有任何 RNG op，各 generator 的狀態推進方式與現行完全相同（含 z0 先於 block、E 先於 Ua 的順序）。測試 G6 會直接比對 replay 後每個 generator 的 `get_state()`。

### 2.2 設計

1. **圖的單位**：展開 K 步（建議 K=32，可調，必須是偶數且整除 1024），一次 `replay()` 走 K 步。
2. **靜態緩衝區**：所有狀態（`z,u,v,susp,cus,pend_a,pend_v,flags,pen,util_b,audits,sxy,sxx,reg_b,nraid,th,ssh,shv`、ObsBuilder 的三個 EMA、`won_hist,d_hist,last_notice,susp_seen`、`hs.win/clean/rsw`、extension 狀態）、逐元素權重（`W1,b1,W2,b2`）、`sidx`、E/Ua/Rn/Rx 的 block 緩衝區、`t_dev`、`j_dev`。每代開頭用現行的 eager 初始化碼（`SimState(...)`、`_Bank(...)`、`_HS`、`ext`）建出新狀態，再逐個 `static.copy_(fresh)`。這樣初始化語意與現行逐字相同，沒有重寫。
3. **`t` 變成裝置端 float64 純量**（`t_dev`）。`t > susp`、`t + L`、`susp1 - t`、`t - ln` 都是 float64 運算，與現行「Python 整數轉 double」逐位相同（t ≤ 1e5，精確可表示）。需要改的點：`torch.full_like(x, t)` 改成 `t_dev.expand_as(x)`（或 `where`），D2 的 `float(t)` 同理。每次 replay 結束在圖內 `t_dev += K`。
4. **`s`（block 內位置）**：把 block 緩衝區視為 `[1024/K, K, N, ...]`，圖內用 `index_select(0, j_dev)` 取出本次 replay 的 K 步，再用 `[:, :, sidx]` 一次 gather 成 `[K, B, ...]`，每步取 `[i]`（view，不啟動 kernel）。純資料搬移，bitwise 不受影響，並且把現行每步 3 個 `index.Tensor` 變成每 K 步 3 個。`j_dev += 1` 在圖內，block 換新時由 eager 歸零。
5. **`cs = t % M`、`slot`**：M=2，所以奇偶是靜態的。每個展開的步驟 i，奇偶由「replay 起點 t 的奇偶」決定，**捕捉兩份 graph**（起點奇數、起點偶數）。這樣 `pend_a[:, cs]` 這種 Python 整數切片可以原樣保留。
6. **狀態回寫**：展開 K 步後，Python 層的屬性已被改綁到新張量，圖的最後在圖內對每個狀態做 `static.copy_(new)`（約 30 個 copy，K=32 時相對 6000 個 kernel 可忽略），並把屬性改綁回 `static`。
7. **事件切分**（排程器，純 Python，CPU 可測）：`[1, T]` 被「block 邊界（1024 的倍數）」與「快照點 Ts」切成區段；每個區段內盡可能用 K 步 replay，剩下不足 K 步的尾巴用**同一個步驟函式 eager 跑**。S1 的 T=20000、Ts=T-L=17500（L=2500），17500 不是 32 的倍數，所以快照前會有 1 個 28 步的 eager 尾巴；驗證 T_val=100000、Ts=97500 同理。eager 尾巴每個事件最多 K-1 步，成本可忽略。快照 `clone()` 發生在兩段之間（eager）。
8. **`isfinite` 檢查（只有 env._step 路徑）**：原本在 t==1 與 t%1024==0 同步檢查。graph 內改為在每個 block 結尾那一步把 `isfinite(d).all()` 寫進一個裝置端旗標，block 結束時（原本就有一次同步的頻率）讀一次。語意差別：NaN 時丟例外的時機最多晚一個 block；結果不受影響。因為策略建構時已拒絕非有限的 theta（policy.py:283-284、env.py:118-119），NaN 本來就極不可能出現。
9. **warm-up 與 capture**：capture 前在側串流跑 3 步 eager（用丟棄用的狀態），避免 lazy kernel 載入發生在 capture 中；每個訓練 B 一份（B=65792，T_train），驗證一份（B=64，T_val）。capture 在程序啟動時做一次（每個 segment 一次，約 143 次，每次數秒以內，推論）。
10. **開關**：預設關閉；`D2_GRAPH=1` 或 `--graph` 才啟用；任何 capture 失敗（例外）就退回 eager 並印警告。上線前不改預設。

### 2.3 要寫的程式（限縮改動面）

- 新檔 `arbitration/rl/graphsim.py`（排程器、靜態狀態包裝、capture／replay）。
- `vulns._vstep` 與 `env._step` 的步驟邏輯需要一個「`t` 為張量、block 資料由呼叫端供給」的版本。**做法**：把步驟邏輯抽成一個內部函式 `_vstep_core(st, bank, op, knob, hs, t, t_dev, E_s, Ua_s, Rn_s, Rx_s, parity)`，舊的 `_vstep` 只是把 `t=st.t+1`、`E_s=E[s][sidx]` 等算好再呼叫它。這樣只有一份邏輯，現有 T-01..T-08、T-36 等 bitwise 測試就是回歸網。
- 第一階段只做 `_vstep`（S1 的 7 個 operator cell＋條件 cell），第二階段再做 `env._step`（NAIVE 錨點、M4 錨點、main 的 6 個 run）。

### 2.4 bitwise 風險清單

| 風險 | 說明 | 緩解 |
|---|---|---|
| kernel 設定隨形狀改變 | graph 不改形狀，replay 的 kernel 與 capture 時的 eager kernel 相同 | 測試 G4 在**正式形狀**（B=65792）比對 |
| reduction 順序 | `sum(-2)`、`max(1)` 的 reduction 設定由形狀決定，不由位址決定；同形狀同設定 | 同上 |
| 向量化對齊 | caching allocator 與 graph 私有池都是 512 B 對齊 | 同上 |
| `t` 改張量的型別提升 | float64 0-dim 與 float64 張量運算，與「Python 純量→double」逐位相同；若誤用 int64 張量則要確認 | CPU 測試 C2 逐欄位比對 |
| 重構改到 eager 路徑 | 舊 `_vstep` 呼叫新核心函式 | 既有測試＋C2 |
| 非確定性 kernel | `index_put_` 無 accumulate、索引唯一；`scatter_` 唯一；沒有 atomicAdd 累加 | G3 重複 replay 兩次比對 |
| capture 中的隱式同步 | 若有，capture 會直接報錯 | G2 |
| 跨機器 | 本機與學校機 GPU 架構不同，eager 對 eager 都不保證位元相同 | 驗收只比「同機 eager vs graph」，不跨機 |

### 2.5 預估倍數（**推論**）

- GPU 側下限：約 200 個小 kernel 各 1–3 µs（L2 命中，5070 Ti 的 L2 為 48 MB，RTX8000 為 6 MB）＋ MLP 的 80 MB 流量（約 0.1 ms）＋ fp64 超越函數（`ndtri`/`ndtr`/`tanh`，兩張卡 fp64 都約 0.4–0.5 TFLOPs）。粗估 5070 Ti 下限 0.5–0.9 ms／步，RTX8000 下限 1.0–1.6 ms／步。
- 結果：5070 Ti 約 1.1–1.36 → 0.5–0.9 ms，**約 1.4–2.4 倍**；RTX8000 約 3.2 → 1.0–1.6 ms，**約 2–3 倍**。若 Step 0 量到 GPU busy 已超過 70%，A 只有約 1.3 倍，要重新評估。
- 驗證（B=64）：純 launch-bound，推論 4–8 倍；每 cell 驗證約 8 次 x 130 秒 ≈ 17 分鐘，約佔 8%。
- 保守估：S1 約 97 小時 → 約 35–50 小時（學校機）。

### 2.6 A0：不改數值、可選的小優化（排在 A 之後，視 Step 0 結果）

預先配置 `ar/kk/one`，不再每步重建；每個 block 預先 gather `E[s][sidx]`（在 eager 路徑也能省 3 個 kernel／步）；`_check` 的集合運算只在除錯時做。預估再省 5–15%（推論）。每一項都只是搬移運算，仍須通過同一組 bitwise 測試。**不要用 `torch.compile` 融合**（會改變浮點結果）。

## 3. 方案 B：同程序合併多個 cell／seed

### 3.1 要做哪些事（若硬要做）

1. **模擬器**：把 2–4 個 cell 的 `(lam+1) x n_S` 列疊在 batch 維度。S1 的 operator 逐 cell 不同（O1, O2, O3, O5, D1, D2, D4＋2 個錨點），hook 集合、`full` 旗標、extension 狀態（lose_run／calm／rh_a）都不同，必須改成逐元素的 operator 遮罩並計算聯集，等於重寫 `_vstep`。只有 D1/D2/O1 各有一個條件 cell（s=0.01）可以與同 operator 的主 cell 直接合併（knob 可用逐元素張量，vulns.py:603-606 已支援）。
2. **M4 錨點**走 `env._step`，機制不同，不能與 M3C 疊。`main_*` 的 r=0.5／0.9 需要逐元素 `rho`（`SimState.rho = full(r)`、`ObsBuilder.r` 是 Python float），要改 obs 的純量。
3. **RNG 與種子**：各 cell 的 `run_ids=[0]`、`algo_seed=9000`，所以所有 S1 cell **用同一組 256 個種子**（seed_block 只取決於 run_id 與 g）。合併後獨立種子數仍是 256，`sidx` 把各 cell 的列對回去，RNG 串流與單獨跑完全一致，抽取成本還會因共用而下降。每個 cell 自己的 CMA-ES（`default_rng(9000+run_id)`）互相獨立，不受影響。
4. **checkpoint／續跑**：每個 cell 保留自己的 part／ckpt（檔名不變）。合併跑的每一代：各 cell 的 `ask()` → 疊起來模擬一次 → 切開 → 各自 `evaluate_generation`／`tell`。各 cell 的 segment 切分目前依 `t_step_p95` 逐 cell 不同（`d2_run_queue.build_jobs`），合併後要用共同的切分。
5. **`d2_run_queue.py`**：現在一個 job＝一個 (cell, run, g0, g1)，鎖檔 `.queue.lock` 是全域的（d2_run_queue.py:419-426、493-495）。要新增「job 群」概念，一次 commit 多個 part／ckpt，push 頻率與 `test_segment_order_matches_frozen_manifest`（tests/direction2/test_d2_run_queue.py:51）要一起改。
6. **train.py 入口**：新增多 cell 入口，並保持 `_locked_paths`／`verify_repo_state`／`check_reuse`／`check_conditional` 對**每個**被合併的 cell 都執行。

### 3.2 為什麼不建議

- **bitwise 沒有保證**：tests/direction2/test_d2_env.py 的 T-08（`test_T08_batch_invariance`）對 adapter 是 bitwise，對 **MLP 只要求 1e-12**。這代表專案自己就沒有主張「批次組成改變時 MLP 位元相同」。MLP 的 `(obs*W1).sum(-2)` 是 reduction，其設定隨輸出元素數而變的可能性不能排除。合併後 B 從 65792 變 131584 以上，必須 GPU 實測才知道，而且測試必須在正式形狀跑。
- **改動面大**：重寫凍結的 vulns 步驟，審查成本高於 A。
- **邊際效益**：eager 下合併 2 個 cell 約 1.3–1.6 倍（聯集步驟的 kernel 數約為單 cell 的 1.3 倍，推論），4 個 cell 約 2 倍但 B 變成 26 萬，GPU 先飽和。有了 A 之後 GPU 已接近飽和，合併約只剩 1.0–1.2 倍。

### 3.3 B'：多程序並行（零程式改動的替代）

- 做法：同時開 2 個 `d2_run_queue.py`，用 `--cells` 分成兩組不相交的 cell。每個程序跑的 kernel 與單獨跑相同，**bitwise 自動成立**（唯一要驗的是 GPU 資源競爭不影響數值，不會）。
- 需要改的：`.queue.lock` 要改成每組一個鎖；兩個 runner 同時 `git commit`／`git push` 會搶 `index.lock`，要加重試或讓其中一個 `--no-commit`、由另一個統一 commit。
- 預估（推論）：預設 time-slicing 約 1.2–1.5 倍；開 MPS（`nvidia-cuda-mps-control -d`）約 1.5–1.8 倍。與 A 疊加後效益遞減（約 +0–15%）。
- **限制**：CLAUDE.md 規定本機 GPU 工作一律走 gpujob 且不得自行改同時執行上限（本機同時負責顯示，多程序曾經造成當機）。所以這只適用於學校機，且要使用者明確同意。

## 4. 建議順序

Step 0（量測，不改凍結程式）→ A（先 `_vstep`，再 `env._step`）→ A0（視結果）→ B'（只在學校機、使用者同意時）。B 不做。

## 5. 凍結守門檢查

| 守門 | 位置 | 會擋純效能修改嗎？ |
|---|---|---|
| `--plan` 必須是 `results/direction2/run_plan.json` | train.py:280-300（`_locked_paths`） | 不會。只比路徑。 |
| run_plan.json 的 sha256 要等於 FREEZE.md 的 `RUN_PLAN_SHA256` | plan.py:181-185（`load_plan`） | 不會，只要不碰 run_plan.json。 |
| run_plan.json 與 FREEZE.md 已 commit 且與 HEAD 無差異 | plan.py:65-70（`verify_repo_state`） | 不會，但**不能改這兩個檔而不 commit**。若為了補登而改 FREEZE.md，必須先 commit，否則所有 segment 都會拒絕啟動。 |
| HEAD 必須是 a72d3d7 的後代 | plan.py:73-74 | 不會。效能 commit 疊在目前分支上就是後代。**不要 rebase／squash 掉 a72d3d7。** |
| pilot_decision.json 的 sha256＝`PILOT_DECISION_SHA256`，且只 commit 過一次 | plan.py:93-113 | 不會，不要碰該檔與該登記行。 |
| `check_run`（g0、g1 要在 G_gens 之內） | plan.py:216-222 | 不會。代數不變。 |
| `check_reuse`／`check_conditional` | train.py:333-337 | 不會。 |

**不看程式內容**：整個守門鏈沒有任何一步檢查 arbitration/rl/*.py 的雜湊。FREEZE.md 第 19、35 行登記的「程式合併 sha」本來就已因 plan.py／conftest.py 的補登而過期（第 56 行寫明「以 git 為準」）。**建議**：效能修改完成、測試通過後，在 FREEZE.md 末尾以「補登」格式附加一筆（改動內容、新合併 sha、bitwise 測試名稱、不改 RUN_PLAN_SHA256／PILOT_DECISION_SHA256），先 commit 再開跑。這是凍結語意的決定，請使用者確認；如果不想動 FREEZE.md，也可以只寫在 docs/direction2-notes.md。

**frozen_budget**：只出現在 `scripts/make_d2_run_plan.py:194`（產生計畫時寫入），訓練時不被讀取。內容（lam=256、T_train=20000、n_S=256、G_gens=416／499、val_every=50、T_val=100000）都不隨速度改變。`est_per_run`（t_step 與秒數）只是估計，用在 `d2_run_queue.build_jobs` 與 manifest 的 segment 切分（上限 1500 秒）。**建議保持切分不變**（143 段、#158–#300、檔名與 test_segment_order_matches_frozen_manifest 都不動）；加速後每段只是提早結束。**不要因為變快而增加 G_gens**，那會改變實驗。

**part 檔比對**：part 檔裡的 `wall_s`、`t_step_ms`、`git_sha` 一定會不同（git_sha 是 HEAD，效能 commit 會改變它）。「bitwise 相同」只比較 θ（`mean_theta`、`best.theta`）、`G_mean_curve`、`G_best_curve`、`sigma_curve`、`val_records`、`fitness_mean/best`、CMA 狀態（`mean,sigma,pc,ps,C,gen,rng`）。已用 eager 跑完的 segment 可以與之後用 graph 跑的 segment 混用，前提是測試通過。

## 6. TDD 計畫

原則：先寫會失敗的測試（紅燈），再實作；所有 GPU 測試用 `@pytest.mark.gpu`，一律走 gpujob（見 conftest.py:17-30）；CPU 測試能在一般 `-m "not gpu"` 下跑。新檔 `tests/direction2/test_d2_speed.py`。小規模設定：`lam=8~16`、`n_S=4~8`、`T=70~200`、`L=10~20`、`K∈{2,4,8}`，每個 GPU 測試目標在 1–2 分鐘內，整組幾分鐘。

### 6.1 CPU 測試（不需要 GPU）

| 編號 | 內容 | 紅燈原因 |
|---|---|---|
| C1 | **事件排程器**：給定 `(T, Ts, K, blk)`，產生 `[(replay|eager, t_start, n_steps)]`；檢查覆蓋 1..T 無重疊、replay 不跨 block 邊界與 Ts、replay 起點奇偶正確、尾巴 < K。包含 T=20000/Ts=17500/K=32 與 T=100000/Ts=97500。 | 模組不存在 |
| C2 | **`t` 張量化不改結果**：新核心函式（`t_dev` float64 張量、block 資料外供）與舊 `_vstep`，在 CPU 上對 operator ∈ {None, O1, O2, O3, O5, D1, D2, D4}、隨機 MLP 權重、T=300、L=20、跑完整模擬，逐欄位 `torch.equal`（`final`、`snap`，並逐步比對 `util_b`、`susp`、`cus`）。 | 核心函式不存在 |
| C3 | **靜態狀態包裝（eager 模式，不 capture）**：以「展開 K 步＋copy_ 回寫」的包裝跑，與舊路徑逐位相同，涵蓋非 K 倍數的 T（走 eager 尾巴）與快照點。這一步在 CPU 就能抓到邏輯錯誤。 | 包裝不存在 |
| C4 | **治理不變**：既有 `test_d2_plan.py`、`test_d2_run_queue.py`、`test_d2_gate.py` 全綠；新增一個測試斷言 `run_plan.json` 與 FREEZE.md 的雜湊行在效能 commit 前後相同。 | 無（回歸網） |
| C5 | **開關**：未設 `D2_GRAPH` 時 `train.main` 走舊路徑（用 monkeypatch 確認 graphsim 未被呼叫）；設了但 CUDA 不可用時給明確錯誤而不是默默退回。 | 旗標不存在 |

### 6.2 GPU 測試（gpujob）

| 編號 | 內容 | 規模 |
|---|---|---|
| G1 | **graph 與 eager 逐位相同**：operator ∈ {None, O1, O2, O3, O5, D1, D2, D4}，K∈{2,4,8}，T=70（含 Ts 為奇數），`final`/`snap` 全欄位 `torch.equal`；replay 兩次（重置狀態後）結果相同。 | B 約 2000，數秒 |
| G2 | **零同步**：replay 區段內 `torch.cuda.set_sync_debug_mode("error")` 不報錯；capture 成功。這同時驗證 1.4 節「`one_hot` 在 CUDA 上不同步」的假設。 | 小 |
| G3 | **正式形狀煙霧測試**：B=257x256、T=200、K=32，graph vs eager 全欄位 `torch.equal`。T-08 顯示 MLP 對批次組成並非 bitwise，所以必須在正式形狀驗證，不能只靠小形狀。 | 約 10 秒 |
| G4 | **數代訓練比對**：`train_segment` 以 `lam=16, n_S=8, T=128, L=16, val_every=2, T_val=64` 跑 4 代，eager 與 graph 比較：`opt.state_dict()`（mean、sigma、pc、ps、C、gen、rng 狀態）、`curve_G/Gbest/sigma`、`val_records`、`best`、`mean_theta`，以及 ckpt 的 JSON（去掉時間欄位）。 | 約 1 分鐘 |
| G5 | **中斷續跑**：(a) graph 跑 [0,2) 再跑 [2,4)（第二段重新建立程序狀態與重新 capture）等於 eager 一次跑 [0,4)；(b) 在第 3 代的 `sim_fn` 丟例外模擬中斷，再用 graph 續跑，結果相同；(c) ckpt 內不含任何 graph 相關狀態，eager 與 graph 的 ckpt 可互相續跑（eager 寫、graph 讀，反之亦然）。 | 約 1 分鐘 |
| G6 | **RNG 串流**：跑完後每個種子的 3 個 generator 的 `get_state()` 與 eager 相同；預設 generator 的 offset 沒有被動到（證明 graph 內沒有 RNG op）。 | 小 |
| G7 | **驗證 graph**：`val_fn`（B=64、T_val 為 1024 的非倍數）graph vs eager 逐位相同，並且在同一程序內訓練 graph 與驗證 graph 交替執行不互相干擾。 | 小 |
| G8 | **記憶體**：連跑 5 代，`torch.cuda.memory_allocated()` 與 `memory_reserved()` 不增長。 | 小 |
| G9 | **真實資料回歸（同機）**：用 `pilot_naive_cold`（`results/direction2/parts/pilot_naive_cold__run0__g0000-0036.json`）的前 3 代 `G_mean_curve`、`G_best_curve`、`sigma_curve`，以 graph 模式在正式設定下重跑 3 代，必須與已 commit 的數字逐位相同。只在產生該檔的同一台機器（本機 5070 Ti）有意義，學校機用「同機 eager vs graph」代替。 | 3 代，約 30 秒 |
| G10 | **NaN 守門**：theta 含 NaN 仍在 `MLPPolicy` 建構時被拒絕（policy.py:283）；`env._step` 路徑的 `isfinite` 旗標在 block 結尾能觸發 `FloatingPointError`。 | 小 |
| G11 | **加速量測（不是正確性）**：印出 eager 與 graph 的 ms／步；graph 慢於 eager 時測試只警告，不失敗。 | 正式形狀 |

### 6.3 實作順序（每步先有紅燈）

1. Step 0：寫一支**拋棄式**量測腳本（不放進 tests、不改凍結程式）：(a) `torch.profiler` 取一代的前 200 步，算 kernel 總時間／wall 時間＝GPU busy 比例，以及 kernel 數；(b) 用標準 recipe 對單步（或 K 步）做 capture，量 replay ms／步。用 gpujob 送出，兩台機器各跑一次。若 replay 只比 eager 快不到 1.3 倍，停止，改議 B'。
2. 寫 C1 → 實作排程器。
3. 寫 C2 → 抽出 `_vstep_core`（`t` 張量化，eager 路徑）→ 既有測試全綠。
4. 寫 C3 → 實作靜態狀態包裝（eager）。
5. 寫 G1–G3、G6 → 加上 capture／replay。
6. 寫 G4、G5、G7、G8 → 接上 `train_segment`、`make_sim_fn`、`make_val_fn`（旗標預設關）。
7. 寫 C5、G9、G10 → 對 `env._step`（NAIVE、M4、main）重複 3–6。
8. 更新 deploy 驗收腳本：`scripts/d2_target_check.sh` 的 `GPU_TESTS` 加入 G1、G3、G5，並在 `d2_target_check.py perf` 同時量 eager 與 graph 的 ms／步，驗收通過才可以開 `D2_GRAPH=1`。
9. 補登 FREEZE.md（見第 5 節，需使用者確認），commit，再開跑。

## 7. 未解問題（要使用者或 Step 0 回答）

1. GPU busy 比例到底多少？（Step 0）
2. 學校機的 torch／CUDA 版本（setup_venv.sh 會依驅動選 cu118–cu126），capture 的行為要在該版本上跑一次 G1–G3。
3. 是否同意在 FREEZE.md 補登程式變更？
4. 是否要在學校機用 B'（多程序）？若是，需要改 runner 的鎖。
5. S1 已經有任何 segment 用 eager 跑完嗎？（目前看起來只有 pilot；混用沒有問題，只要測試通過。）

## 8. 實作與量測記錄（2026-10-08，本機 RTX 5070 Ti、torch 2.12.0+cu130）

### 8.1 做了什麼（分支 `direction2-speedup`，從 cc26592 分出，沒有 push）

| 檔案 | 內容 |
|---|---|
| `arbitration/rl/graphsim.py`（新） | `plan_events`（純排程）、`GraphSim`（K 步 CUDA graph 的建立／快取／replay）、`from_env`（`D2_GRAPH=1`、`D2_GRAPH_K`）。 |
| `arbitration/rl/env.py`、`vulns.py` | 步驟函式只做三種「不改數值」的改動：時間 `t` 可以是裝置端 float64 純量（`tt`）；RNG 列經 `_row()` 取得（eager 時就是原本的 `block[s][sidx]`）；D4 的 `as_tensor(eta)` 改成裝置端 `full`（CUDA graph 不允許 host→device 複製，值相同）。`simulate` / `simulate_vuln` 多一個 `graph=` 參數（預設 None）。 |
| `arbitration/rl/train.py` | `main` 拆成 `make_parser` / `prepare` / `main`（守門與呼叫順序不變），`make_sim_fn` / `make_val_fn` 多 `graph=`；`--graph` 或 `D2_GRAPH=1` 才啟用。 |
| `arbitration/rl/multitrain.py`（新） | 同程序多 cell：每個 cell 一條執行緒＋一條 CUDA stream＋自己的 GraphSim；`--serve` 是給 runner 用的常駐 worker（stdin/stdout JSON 行）。 |
| `scripts/d2_run_queue.py` | 新增 `--concurrent N`（預設 1 = 原本的循序 runner，程式路徑完全沒動）、`--graph`、`--no-graph`。 |
| `scripts/bench_d2_speed.py`、`scripts/d2_graph_check.py`（新） | 本機量測；切換前驗收（見第 9 節）。 |
| `tests/direction2/test_d2_speed.py`、`test_d2_multi.py`、`test_d2_run_queue.py`（追加）、`golden_eager_cpu.json` | 測試。 |

**與第 2 節計畫的差異**（都是為了讓「bitwise」更有把握）：
1. **只用一張 graph**（不是奇偶兩張）：replay 一律從奇數步開始、K 為偶數，所以 `t % 2`（M=2）永遠是靜態的。不是 K 的倍數、跨 1024 邊界、跨快照步 Ts 的部分用同一個步驟函式 eager 跑（S1 的 T=20000：約 10 步 eager 對 600+ 次 replay）。
2. **不批次 gather 整個 K 步的 RNG 列**：圖內每步照原樣做 `Ek[i][sidx]`，和 eager 的 `E[s][sidx]` 是同一種 index 運算、同樣的記憶體排列（省掉的 kernel 不到 2%，換來沒有排列差異的疑慮）。
3. **狀態不用手列**：`graphsim._collect` 掃描 SimState / bank / ObsBuilder / _HS / 擴充物件 / knob 的所有張量，建立靜態緩衝區並在每代把新建（沿用原本 eager 初始化碼）的狀態 `copy_` 進去。形狀、dtype 不同會直接報錯。
4. **`isfinite`**：圖內不同步；每個 replay 的最後一步把 `~isfinite(d).all()` 累進一個裝置旗標，driver 每個 RNG block 結尾與整段結束時讀一次（只影響 `env._step` 路徑，丟 `FloatingPointError` 的時機最多晚一個 block；NaN 會留在 EMA／d_hist，不會漏掉）。`_vstep` 本來就沒有這個檢查。
5. **建圖鎖**：同程序多 lane 時，一次只允許一個 lane 建圖（warm-up＋capture 內有 device 範圍的同步，別的執行緒正在 capture 時不合法；實測會出 `operation not permitted when stream is capturing`）。已建好的 lane 的 replay 不受影響。
6. RNG 的 generator 完全不進圖：逐 seed 的 3 個 generator 呼叫順序與 eager 完全相同（測試比對 `get_state()`）。

### 8.2 bitwise 測試（全部是 `torch.equal` / `==`，對象永遠是同一程序、同一 GPU 上未改動的 eager 路徑）

| 類別 | 內容 | 結果 |
|---|---|---|
| CPU（無 GPU 的整套機制：靜態緩衝區、feed、裝置端 t、尾段、跨代重用，只是不 capture） | `plan_events`（C1）；8 個 operator（None/O1/O2/O3/O5/D1/D2/D4）× 3 組 (T,Ts) 的 `simulate_vuln`；M4／NAIVE／M3C 的 `env.simulate`；跨 1024 邊界；同一 GraphSim 連跑多代、多 operator、多形狀；generator 狀態；`train_segment` graph vs eager（part、ckpt）；eager→graph→eager 接續；graph→graph 接續；中斷續跑；驗證 graph 與訓練 graph 交錯；旗標預設關；graph 在 CPU 上的明確錯誤；NaN 守門；eager 路徑不變（CPU 雜湊，對照 cc26592 的程式） | 見 8.5 |
| GPU（真 capture / replay，gpujob） | 同上每一項的 GPU 版；另有 G2（eager 步驟零 host 同步）、G3（正式形狀 B=65792，K=32，D2/D4/O1/None 與 M4，重複 replay）、G8（連跑 5 代記憶體不增長）、G9（正式大小，NAIVE pilot 前 3 代等於 repo 內 committed 的 eager 數字）；多 cell 並行（3 個 cell 同時，兩段連續，part 與 ckpt 與各自單獨 eager 逐位相同） | 見 8.5 |

「用 eager 產生的 checkpoint 接續」：`test_G5_resume_in_pieces_and_switch_eager_to_graph_mid_run`（eager [0,2) → graph [2,4)（新 GraphSim＝新程序）→ eager [4,6)，part 與 ckpt JSON 全部等於一次 eager 跑完 [0,6)）；加上 `scripts/d2_graph_check.py` 用 repo 內學校機器產生的真實 checkpoint（s1_D2_s0.02 g36）在正式大小上做同樣的比較。

### 8.3 速度（本機，S1 正式大小 λ=256、n_S=256、T=20000，B=65792；只量 sim_fn，不含 CMA-ES；`scripts/bench_d2_speed.py`，原始資料 `results/direction2_speedup/bench_local*.jsonl`）

**注意**：量測時這台機器的 CPU 被別的專案佔滿，eager 是 CPU 派發受限，所以 eager 的數字被放大（學校機器的 eager 是 2.75–3.45 ms／步）；graph 幾乎不吃 CPU，數字較穩。倍數只能說明本機。

| 設定 | cell | ms／步（每個 cell） | cell-steps／s（全部 cell 合計） | GPU util（nvidia-smi 平均） |
|---|---|---|---|---|
| eager | D1 s0.02 | 1.878 | 532 | 44% |
| eager | D2 s0.02 | 2.249 | 445 | 38% |
| eager | O1 s0.02 | 1.885 | 531 | 45% |
| graph K=32 | D1 | 0.718 | 1393 | 99% |
| graph K=32 | D2 | 0.711 | 1407 | 99% |
| graph K=32 | O1 | 0.697 | 1436 | 99% |
| graph + 2 個 cell 並行（D1,D2） | | 1.21 | 1654 | 100% |
| graph + 3 個 cell 並行（D1,D2,O1） | | 1.70 | 1760 | 100% |

- **eager → graph：2.6–3.2 倍**（eager 的 GPU util 44% 與學校機器實測的 43% 一致，確認是 launch／Python 受限；graph 之後 GPU 已接近 100%，剩下的是 GPU 真正在算的時間）。NAIVE pilot 的 `env._step` 路徑 eager 本來就比較輕（1.07 ms／步），真實訓練 3 代（含 CMA-ES、policy 物件）從 21.5 s／代變 13.3 s／代（約 1.6 倍）。
- **並行 2 個 cell：1654 / 1400（單獨 graph 的平均）= +18%；並行 3 個：1760 / 1412 = +25%**（相對 eager 的 3.3×）。GPU 已被單一 cell 的 graph 吃到 99%，並行只是把 kernel 之間的空隙與小 kernel 的低佔用率填起來，所以增益有限；每個 cell 各自變慢（1.2 / 1.7 ms／步），只有總產出增加。
- 記憶體：每個 graph lane 的 `max_memory_reserved` 約 0.53 GB（eager 0.22 GB）。學校機器是 12 GB 的 vGPU 切片，3 條 lane 約 1.6 GB，沒有壓力。
- K（每次 replay 的步數）：見 8.4。驗證（B=64、T=100000）：見 8.4。

### 8.4 其他量測

- **驗證**（B=64＝2 列×32 seeds，T_val=100000；`bench_local_val.jsonl`）：eager 2.09 ms／步（GPU util 18%）→ graph 0.228 ms／步（99%），**9.2 倍**（一次驗證約 209 秒 → 23 秒）。S1 每個 cell 有 8 次驗證，驗證時間從約 28 分鐘降到約 3 分鐘。
- **K**：K=16 / 32 / 64 的每步時間是 0.721 / 0.718 / 0.724 ms（`bench_local_K.jsonl`），沒有差別，預設 K=32（`D2_GRAPH_K` 可改）。
- **建圖時間**：第一代多約 1–2 秒（warm-up＋capture）；之後每代重用同一張圖。驗證用另一張圖（B 不同），整個程序共兩張。
- **端到端**（本機，`d2_run_queue.py --concurrent 2`，在 repo 的複本上，不動真正的 results）：s1_O1_s0.02 與 s1_O2_s0.02 的 g0–36 同時跑，**兩段都在 14.5 分鐘內完成**（每段 36 代 × 約 24 秒），各自 commit（0bc5e10 / 09a3d54 在複本裡），part 檔 status=done。相對 eager 單跑一段約 23 分鐘（36 代 × 1.88 ms × 20000 步，8.3 的量測值），這兩段合計 14.5 分鐘＝每段有效 7.3 分鐘（約 3.1 倍）。
- **正式大小的比較**（`scripts/d2_graph_check.py`，結果 `results/direction2_speedup/graph_check_local.json`）：用 repo 內學校機器（RTX8000、eager）產生的 s1_D2_s0.02 g36 checkpoint，與無 checkpoint 的 s1_D4_s0.02，在本機各續跑 2 代：**graph 與 eager、並行（兩個 cell 同時）與 eager 的 part 與 checkpoint 全部逐位相同**。

### 8.5 測試帳（本機；CPU 測試在 CUDA 被隱藏的環境下跑，GPU 測試全部經 gpujob）

| 範圍 | 項數 | 結果 |
|---|---|---|
| CPU 全部（`pytest tests -m "not gpu"`，含舊測試與新增的 55 項） | 339 | 全過（338 + 後來追加的 1 項） |
| 新增 CPU：`test_d2_speed.py` 41、`test_d2_multi.py` 4、`test_d2_run_queue.py` 的 `--concurrent` 10 | 55 | 全過；其中 graph/並行 vs eager 的逐位比對約 37 項 |
| 新增 GPU（真 capture）：`test_d2_speed.py` 43、`test_d2_multi.py` 2 | 45 | 全過（含 D4 一度因 `as_tensor(float)` 在 capture 中失敗，已修） |
| 既有 GPU：env T01/T02/T03/T08（4）、T05 r0.5（7）、T05 r0.9（7）、T06（2）、misc T50/T59（2）、s1 T36/T41（2）、T72 O1–O3（3）、T72 O4–O6（3）、`tests/test_gpu.py`（50） | 80 | 全過 |
| 既有 GPU：s1 `T39`（重算整個漏洞登記表的 Δ） | 1 | **沒跑完**：單獨跑超過 25 分鐘，依 gpujob 規則中止。它只走 eager 的 hook／adapter 路徑，那條路徑改動由下面兩項涵蓋 |
| eager 路徑逐位不變（一次性比對，對照 cc26592 的原始碼樹；`scratchpad` 腳本，不進 repo）：11 個 operator（含 O4/O6/D3）× {MLP、9 種 adapter＋hook_log＋public_log、warm_obs} + M4／NAIVE／M3C 的 `env.simulate`（diagnostic 與 1100 步） | 41 組雜湊 | **CPU 與 GPU 都完全相同** |
| eager 路徑逐位不變（永久測試）：`test_eager_path_unchanged_by_refactor_cpu`（對照 cc26592 的 CPU 雜湊；torch 版本不同時 skip） | 1 | 過 |
| 正式大小、真 checkpoint：`scripts/d2_graph_check.py`（學校 eager checkpoint g36 與無 checkpoint 的 cell，各 2 代） | 2 cell | graph == eager、並行 == eager |
| 正式大小真實資料：G9，NAIVE pilot 前 3 代等於 repo 內 committed 的 eager 數字（學校外的機器）；本機 eager 單獨重跑也等於 committed 數字 | 3 代 | 全部逐位相同 |

### 8.6 B（同程序多 cell 並行）的結論

- 增益：2 個 cell +18%、3 個 cell +25%（相對各自單獨用 graph 跑；相對 eager 是 3.3 倍）。達到了「>15%」的門檻，bitwise 有測試與正式大小的比對，所以**已接到 runner，但預設關閉**（`--concurrent N`，預設 1）。
- 增益不大的原因：單一 cell 用 graph 之後 GPU 已 99% 忙；並行只能填 kernel 之間的空隙。再多 cell（4+）預期遞減，也沒有量。
- 取捨：並行時每個 cell 的牆鐘變慢（graph 單獨一段約 9 分鐘，2 條並行各 14.5 分鐘），但總產出較高；斷電或失敗時同時有 N 段在跑，各自最多損失一個 checkpoint 區間（50 代）的進度。建議學校機器先只用 `--graph`（確定、最單純），確認穩定之後再考慮 `--concurrent 2`。
- 未做：合併 batch（第 3 節的 B，維持否決）；多程序（B'）。

### 8.7 還沒解決的風險

1. **bitwise 只在「同一台機器、同一版 torch」內驗證**。本機是 torch 2.12.0+cu130／RTX 5070 Ti，學校是 torch 2.6.0+cu126／RTX8000（12 GB vGPU）。CUDA graph 的 capture API 在 2.6 都有，但**沒有在 2.6 上跑過**；切換前必須在學校機器上重跑第 9 節的測試。若 graph 在學校不 bitwise，就不要切換（舊程式不受影響）。
2. 「時間 t 用裝置端 float64 純量」假設 `t > susp`、`t + L`、`susp1 - t` 與 Python 數字逐位相同（IEEE 單一運算，理論上必然；本機 CPU＋GPU 全部測試通過）。
3. 同程序並行依賴 `capture_error_mode="thread_local"` 與建圖鎖；多 lane 同時建圖的失敗模式（`operation not permitted when stream is capturing`）已觀察到並修掉，但不同驅動／vGPU 的行為沒有驗證。
4. `isfinite` 檢查的時機在 graph 模式晚一個 block（結果不受影響）。
5. FREEZE.md 沒有改。「程式合併 sha」的補登（第 5 節）要由使用者決定；這個分支沒有碰任何凍結檔。
6. eager 的速度數字是在 CPU 被其他專案佔滿的環境下量的（倍數偏高）；學校機器的實際倍數要看它自己的 GPU 時間（RTX8000 的 fp64 與頻寬都比 5070 Ti 低，graph 後的 GPU 時間可能更接近 eager 的 CPU 時間，倍數可能比本機低，約 1.5–2.5 倍是合理猜測，**這是推論，不是量到的**）。
7. 目前沒有做 A0（預先配置 `ar/kk`、批次 gather 等）：graph 之後 GPU 已 99% 忙，剩下的要靠減少 kernel 數，每一項都要重新證明 bitwise。

## 9. 學校機器切換步驟（草案，需要使用者決定後才執行）

背景：學校機器（aics-st05-u06，RTX8000-12Q，torch 2.6.0+cu126）正在 tmux `s1` 裡跑
`d2_run_queue.py --stage s1 --push-every 1 --remote origin --branch direction2-freeze`，每段 commit＋push；`~/s1_watchdog.sh`（cron `@reboot`＋每 15 分鐘）會在 runner 死掉時重啟它。
分支 `direction2-speedup` 目前只在本機（沒有 push）；要由使用者 push 成**另一個分支名**（runner 只會 push `direction2-freeze`，不會碰它），或用 `git bundle` 帶過去。

**原則**：舊程式（eager）永遠是預設；graph 只在加了 `--graph`（或 `D2_GRAPH=1`）時才用；checkpoint／part 檔格式完全沒變，所以隨時可以退回 eager，也可以在同一個 cell 的段與段之間混用。

### 9.1 切換前：在學校的 torch 2.6.0+cu126 上驗收（全部通過才切換）

可以在 runner 還在跑的時候做（不改正在跑的檔案，另 clone 一份到 `~/d2-speedup`，用同一個 `.venv`）。bitwise 測試的結果不受同時有別的 GPU 工作影響，只是兩邊都會變慢；GPU 記憶體不是問題（本機峰值約 1 GB）。比較保守的作法是等 runner 在兩段之間停下來再做（9.2 的第 1–2 步）。

```bash
git clone <origin> ~/d2-speedup && cd ~/d2-speedup && git checkout direction2-speedup     # 或從 bundle
source ~/scarce-actuator-arbitration/.venv/bin/activate        # torch 2.6.0+cu126
# (1) CPU 全部（含 CPU 版的整套 graph 機制測試與 --concurrent 的排程測試；golden 雜湊在不同 torch 版本會自動 skip）
python3 -m pytest tests/direction2 -m "not gpu" -q -p no:cacheprovider
# (2) GPU：graph 與 eager 逐位相同（真 capture）＋同程序多 cell（45 項；G9 在非 5070 Ti 上自動改成「同一台 GPU 的 eager 對 graph」）
python3 -m pytest tests/direction2/test_d2_speed.py tests/direction2/test_d2_multi.py -m gpu -q -p no:cacheprovider
# (3) GPU：eager 路徑沒被重構弄壞（原本的驗收項目，d2_target_check.sh 的 GPU_TESTS）
python3 -m pytest tests/direction2/test_d2_env.py::test_T01_parity_m3c_per_seed tests/direction2/test_d2_env.py::test_T02_parity_m4_and_naive \
    tests/direction2/test_d2_env.py::test_T08_batch_invariance "tests/direction2/test_d2_env.py::test_T06_replay_d1_json_naive_m4" \
    tests/direction2/test_d2_s1.py::test_T36_vuln_identity_mutation tests/direction2/test_d2_s1.py::test_T41_vuln_ref_policy_public_only -q -p no:cacheprovider
# (4) 正式大小＋真正的 eager checkpoint：從學校自己的 checkpoint 續跑 2 代，eager、graph、兩個 cell 同時，三者必須逐位相同
python3 scripts/d2_graph_check.py --cell <正在跑的 cell> --cell <另一個 cell> --gens 2        # 結尾印 "ok": true，exit 0
```
（4）是「中途切換」的直接驗收：checkpoint 是 eager 在學校機器上寫的，續跑的 graph 版與 eager 版必須相同。它在 `/tmp` 底下工作，不寫進 repo。

**一鍵版（紅隊 H1 之後的標準作法）**：`--preflight` 把 (1)–(4) 與驗證路徑檢查一次跑完，輸出一份 PASS／FAIL 的 JSON（每步的結果、耗時、學校時間估計）：

```bash
python3 scripts/d2_graph_check.py --preflight --json-out ~/preflight.json     # 在學校；cell 預設取 repo 裡有 checkpoint 的前兩個 S1 cell
python3 scripts/d2_graph_check.py --preflight --only 4,H1 --json-out ~/pf_41.json      # 只跑某幾步（名稱前綴：1、2、3、4、H1）
```
步驟：`1_cpu_tests`、`2_gpu_graph_tests`、`3_gpu_eager_tests`、`4_checkpoint_continuation_cross_val`、`H1_val_fn_real_size`。
- 步驟 4 用 `--cross-val`：續跑的那 2 代把 `val_every` 設成「最後一代剛好驗證」，所以被比較的 part／checkpoint 內含**一次正式大小的驗證**（B=64、T_val=100000、另一張圖）與 `best` 的挑選；verdict 會記 `validated`、`val_records_new`、`best_set`，沒有真的驗證到就算失敗。它是 (4) 的超集合。
- 步驟 H1 用 `--val-cell s1_M4rw0_anchor`（走 `env._step`，其餘 S1 cell 走 `vulns._vstep`）：以固定亂數 θ（或 checkpoint 的 mean）比較 `val_fn` 的 eager／graph 輸出，必須逐位相同。
- 時間估計見 9.1b。總 verdict 在 JSON 的 `verdict`（PASS/FAIL）、`failed`（失敗步驟名）、`school_estimate_minutes`、`within_60_min`。
任何一項失敗：**不要切換**，把輸出帶回來（舊程式不受影響）。

### 9.1b `--preflight` 的時間估計（本機實測 × 3 ≈ RTX8000）

本機（RTX 5070 Ti；量測時機器負載很高，CPU 被別的專案佔滿，所以 eager 的數字偏大、偏保守）各步驟牆鐘秒數，與乘 3 的學校估計：

| 步驟 | 內容 | 本機 | 學校估計（×3） |
|---|---|---|---|
| 1_cpu_tests | speed／multi／run_queue／graph_check 的 CPU 測試（109 項；torch 多執行緒，CPU 時間約 1230 s，學校核心數少的話會更久） | 310 s | 15.5 分 |
| 2_gpu_graph_tests | test_d2_speed＋test_d2_multi 的 GPU 測試（45 項） | 73 s | 3.7 分 |
| 3_gpu_eager_tests | T01、T36、T41（eager 路徑沒被重構弄壞的最便宜三項） | 75 s | 3.8 分 |
| 4_checkpoint_continuation_cross_val | 2 個 cell 各續跑 2 代（eager／graph／兩 cell 同時）；第一個 cell 的最後一代做正式大小驗證（eager 驗證約 180 s） | 434 s | 21.7 分 |
| H1_val_fn_real_size | `s1_M4rw0_anchor` 的 `val_fn`，T_val=100000，eager 約 182 s、graph 約 15 s | 200 s | 10 分 |
| 合計 | | 約 1090 s（18 分） | **約 55 分（≤ 60）** |

- `--full` 另加：全部 CPU 測試（`tests/direction2 -m "not gpu"`，220 項；本機在高負載下 43 分鐘，無負載約 10+ 分鐘）與 T02／T06／T08（本機 +10～15 分，學校 +30～45 分）。學校上這會超過 60 分鐘，建議只在 runner 兩段之間、有空時跑。
- 預設版刻意把「eager 驗證」只做兩次（一次 vulns 路徑在步驟 4 的第一個 cell，一次 env 路徑在 H1），因為它是最貴的部分。
- 學校機若比本機慢超過 3 倍，`within_60_min` 會在 JSON 裡反映實測；單步可用 `--only` 分開跑（每步最長約 22 分鐘）。

### 9.2 切換（在兩段之間）

1. 先暫停 watchdog：`crontab -l > ~/crontab.s1.bak`，把 s1_watchdog 兩行註解掉（否則 runner 被停下後 15 分鐘內會被它用舊參數重啟）；確認沒有 `s1_watchdog.sh` 在跑。
2. 等 runner 印出一行 `DONE ... committed <sha>` 之後立刻停它（tmux `s1` 裡 Ctrl-C；runner 會終止 train 子程序、印 INTERRUPTED、結束）。**盡量不要在段的中間停**：中間停也不會壞（下次從上一個 checkpoint 續跑，每 50 代與每段結束時寫一次），只是白白損失那之後的進度。確認 `git status` 只剩 untracked 的 runlogs，`git log origin/direction2-freeze..HEAD` 是空的（都已 push）。
3. 把程式帶進學校的 `direction2-freeze`：`git fetch origin direction2-speedup && git merge --no-edit origin/direction2-speedup`。這個分支只動 `arbitration/rl/*.py`、`scripts/*.py`、`tests/direction2/`、`docs/direction2-speedup-plan.md`、`results/direction2_speedup/`，沒有碰 `run_plan.json`、`FREEZE.md`、`pilot_decision.json`、`vuln_suite.json`、parts／ckpt，所以不會有衝突，也不會讓 `verify_repo_state` 拒絕（HEAD 仍是 a72d3d7 的後代）。合併前先用 `git diff --stat HEAD...origin/direction2-speedup` 確認清單。
   - 注意：之後 runner 的 `git push origin direction2-freeze` 會把這些程式 commit 一併推上去。如果希望 `direction2-freeze` 只有結果，就改在新分支（例如 `direction2-run2`）上合併、並用 `--branch direction2-run2` 跑；這會讓結果分散在兩個分支，要事後合併。**由使用者決定**；FREEZE.md 的「程式合併 sha」補登（第 5 節）也由使用者決定。
4. `python3 scripts/d2_run_queue.py --stage s1 --dry-run`：必須列出同樣的 143 段，done／todo 與切換前一致（只看狀態，不跑任何東西）。
5. 啟動（第一階段只用 `--graph`，不並行；**用包裝 `scripts/d2_s1_launch.sh`**，它等於 `d2_run_queue.py --stage s1 --push-every 1 --remote origin --branch direction2-freeze --graph --graph-fallback`）：
   `tmux new -s s1 'bash scripts/d2_s1_launch.sh'`（要指定 venv 的 python 時：`PYTHON=$HOME/scarce-actuator-arbitration/.venv/bin/python bash scripts/d2_s1_launch.sh`；其餘參數照原樣傳給 runner，例如 `--max-segments 1`）。
   - **watchdog 該用的重啟指令**：把 `~/s1_watchdog.sh` 裡啟動 runner 的那一行換成 `bash scripts/d2_s1_launch.sh`（在 repo 目錄下；`pgrep -af d2_run_queue` 仍然找得到，因為包裝是 `exec` 進 `scripts/d2_run_queue.py --stage s1 ...`）。**不要**再用沒有後備的 `... --graph`：graph 失敗時 watchdog 會連續重啟 3 次然後放棄。
   - 後備行為（紅隊 H2）：某一段因 CUDA graph 例外（capture／replay，由 log 最後一個 traceback 判斷）失敗時，runner 在該段 log 與 `runlogs/queue_history.jsonl` 記一筆 `graph_fallback`、寫 `runlogs/graph_fallback.json`，然後**同一段從它的 checkpoint 起改用 eager 重跑，之後的段也都用 eager**（結果逐位相同，只是慢）。一般訓練錯誤（NaN、CMA-ES、非 graph 的 OOM…）**不會**重試，仍然失敗即停（exit 2）。只重試一次，不會無限迴圈。
   - `runlogs/graph_fallback.json` 存在時，包裝一開始就用 eager（watchdog 重啟不會再撞同一個 graph 錯誤）；想再試 graph 就刪掉這個檔案。
   - `--concurrent` 不能與 `--graph-fallback` 同用（會被拒絕）；第一階段不開 `--concurrent`。`--concurrent` 的 worker 在 runner 被 SIGKILL 時會自行結束（`prctl(PR_SET_PDEATHSIG)`；stdin EOF 但沒有 `quit` 也立刻結束），不再變孤兒（紅隊 M1）。
   - 同時恢復 crontab。
6. 第一段完成後檢查：`--status`；新 part 的 `t_step_ms` 應明顯小於之前的段（本機 eager→graph 為 2.6–3.2 倍；學校的倍數要實測）；`G_mean_curve` 的第一點接著上一段的最後一點（曲線沒有斷點，σ 連續）；`git log` 顯示該段的 commit＋push 正常。
7. 穩定幾段之後才考慮 `--concurrent 2`（加在同一行；需要 12 GB vGPU 上約 1–2 GB 額外記憶體，增益本機 +18%）。

### 9.3 退回

在任一段完成後停下 runner，拿掉 `--graph`／`--concurrent` 重啟即可：checkpoint 與 part 檔沒有任何 graph 相關內容，退回後的結果與一直用 eager 的結果相同。中途斷線、watchdog 重啟、segment 被中斷也都是從 checkpoint 續跑，和以前一樣。

### 9.5 紅隊後續修補記錄（2026-10-08）

- H1：`scripts/d2_graph_check.py` 新增 `--cross-val`／`--cross-val-cell`（續跑跨過驗證邊界並挑 best）、`--val-cell`（正式大小 `val_fn` eager vs graph）、`--preflight`（9.1 (1)–(4)＋H1，JSON verdict）。測試：`tests/direction2/test_d2_graph_check.py`（CPU 小尺寸，13 項）。本機 GPU 實測全部 PASS：D1（checkpoint g416）／D2（g36）續跑 2 代、第一個 cell 內含正式大小驗證，graph == eager、並行 == eager；`s1_M4rw0_anchor` 的 `val_fn` 兩邊同為 0.14500079156464982。
- H2：`d2_run_queue.py --graph-fallback`＋`scripts/d2_s1_launch.sh`（見 9.2 第 5 點）。測試在 `test_d2_run_queue.py`（分類器、fallback、只重試一次、非 graph 錯誤仍停、marker 檔、包裝腳本）。
- M1：`multitrain.die_with_parent()`（`prctl(PR_SET_PDEATHSIG, SIGKILL)`，`--serve` 啟動時呼叫）；`serve` 在 stdin EOF 而沒有 `quit` 時立刻結束（不再把手上的段跑完）。測試在 `test_d2_multi.py`（含真的 SIGKILL 父程序）。
