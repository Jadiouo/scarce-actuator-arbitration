# E-A1 設計（凍結版候選 v3）：攻擊者強度對照（預先登記，2026-10-21）

狀態：凍結版候選 v3（已套用第二輪審查修改；所有待決項已定案，見 §10）。設計 commit 後即凍結：判準、門檻、選取規則、seeds、預算、偏離清單、執行環境釘選，都不得再改；凍結後才能碰任何新 seeds。本文只寫設計，不實作、不跑 GPU。

標記規則：「推論」＝由數字或程式推得、未直接量測；「重算」＝用 python 讀 json 或 CPU 計算；找不到的寫「查不到」。來源簡寫：e1v2設計＝`docs/direction2/design/e1-v2.md`；e1v2結果＝`docs/direction2/results/e1-v2.md`；plan＝`results/direction2/run_plan.json`；man＝`results/direction2/e1v2/manifest.json`；sel＝`results/direction2/e1v2/selection_frozen.json`；atk＝`results/direction2/e1v2/attacker/*.json`；parts＝`results/direction2/parts/main_M3C_r0.5__run{k}__*.json`；speedup＝`docs/direction2/design/speedup.md`。

---

## 第二輪審查修改 → 位置

| # | 修改 | 位置 |
|---|---|---|
| 1 | 清草稿語氣與待決項，已定項目列定案值；§2.8 改「不在本登記範圍」 | 頂端狀態列、紅方對照表第 10/11 列、§2.2、§2.3、§2.4、§3.3、§4、§2.8、§10 |
| 2 | 主表仍列 D；D′ 另標描述性；「D′≤D」改「預期」並註明 test 上不保證 | §0、§2.6 第 4 項、§2.7（一） |
| 3 | A_fresh 與 A_C2′ 一律用 select 階段同一次重評的 G_SEL，策略順序固定 | §2.4「SEL 上必做的評估」第 1、2 項、§2.6 第 4 項、§7 步驟 8 |
| 4 | 3 條鏈須跑滿 416 代且無 NaN 才能進 select，否則停下來問 | §2.4、§3.3、§7 步驟 7、§10 |
| 5 | SD ddof=1、t＝scipy.stats.t.ppf(0.95,n−1)、用 e1_v2.py 的 lower_bound／upper_bound（已查相符） | §2.1 |
| 6 | 五個檔在 input_sha256 而非 protected_sha256，兩者皆由 load_manifest 檢查 | §1.4 |
| 7 | 結果文件完整路徑，並註明 design/e1-v2.md 在 input_sha256 內不可改 | §7 步驟 12 |
| 8 | 訓練後檢查：第 200 代 checkpoint 單 lane 重算與平行結果逐位比對 | §3.4「訓練後檢查」、§6 測試表、§7 步驟 7 |
| 9 | smoke 在正式 manifest 凍結前執行並清除，§5 與 §7 一致 | §5、§7 步驟 4–6 |

---

## 紅方意見 → 修改位置

| # | 等級 | 意見 | 修改位置 |
|---|---|---|---|
| 1 | BLOCKER | 執行機器改 astarc；主判準只用同機配對；跨機只描述；機器漂移檢查；計時待 smoke | §1.5（機器與環境釘選）、§2.1（同機配對）、§2.5（評估與 testcheck 只在同機）、§2.6 第 2 項（機器漂移檢查）、§5（算力待量測＋smoke 量測方法）、§6（manifest 欄位與環境檢查測試）、§7（流程）、§8（限制）。學校 RTX8000／katago／學校規則／學校原環境回歸的前提已全部刪除 |
| 2 | HIGH | 不稱「同配方」；逐項偏離表；記分窗與驗證 seeds 數／頻率寫死 | §2.3（偏離表，含記分窗、驗證 n 與頻率）、§2.4（鏈設計）、§0 第 1 點 |
| 3 | HIGH | A_fresh 限「SEL 選出的單一攻擊者」；3 條鏈各自 checkpoint 對 a\* 的 Δ 為描述性 | §2.1（A_fresh 定義與結論模板限制）、§2.6 第 3 項 |
| 4 | HIGH | 「更強」模板加入替代解釋（C2 對已知池過配），不可單一歸因 | §2.7（一） |
| 5 | MEDIUM | 決策樹補節點 LB>0 且 UB<ε；UB<0 未校正多重比較 | §2.2 |
| 6 | MEDIUM | 登記「原設定同型 Δ」基準，只讀不判；先驗預期 Δ<0 | §2.6 第 1 項、§2.7（二） |
| 7 | MEDIUM | 訓練池排列種子寫死（不用 Random(9000+run_id)）；測試比對 | §2.3、§2.4、§6 測試表 |
| 8 | MEDIUM | testcheck 用 SEL 前 64 個 seeds；新 seeds 與 simulate 內部串流的關係 | §1.3（串流關係，已查程式）、§2.5 |
| 9 | MEDIUM | §2.3 明列需在 SEL 重評 a\* | §2.4 最後一段「SEL 上必做的評估」 |
| 10, 11 | — | 已定：維持現狀、措辭依提醒調整；本稿定稿，不再列為待確認 | 全文 |
| 12 | LOW | 摘要不可寫成「驗證了 C2 結論的穩健性」；寫明 C2 一側不可能翻轉 E1 v2 判定 | §0 摘要、§8 不能宣稱 |
| 新 | 需求 | 把 GPU 與 CPU 用滿；平行化不得改變隨機串流或結果；要有逐位相同測試 | §3（執行方式）、§6 測試表（平行逐位相同）、§7、§10 |

---

## 0. 摘要與三件需要知道的事

**摘要（登記時寫死，論文與摘要不得超出）。** E1 v2 的主判定（C2 對 a\* 的降幅 D＝0.004236，排除 δ＝0.005，判「效果不足」）是對**固定的轉移攻擊者 a\*** 的結果。本實驗在 C2 上另外從頭訓練 3 條攻擊鏈（同 λ／n_S／σ0／代數，目標窗與驗證不同，見 §2.3），在全新 seeds、同一台新機上與 a\* 逐 seed 配對。**本實驗在 C2 一側不可能翻轉 E1 v2 的判定**：更強的 C2 攻擊者只會讓 C2 的降幅更小（預期 D′≤D；A_C2′ 是在 SEL 上選取，test 上不保證成立），使「效果不足」更成立；更弱或持平則不改變任何數字。因此本實驗不是「驗證了 C2 結論的穩健性」，它的價值是**完整性聲明**：回答「E1 v2 的 C2 適應攻擊者沒有超過 a\*，是不是只因為 C2 一側的搜尋偏弱」這個被指出的缺口，並如實報告。

1. **「隨機初始化」其實是「冷啟動」。** 主實驗攻擊鏈從 θ=0（等於誠實策略）冷啟動，由 sep-CMA-ES（σ0=0.3）自己撒開（plan `start_registration.chosen`；`results/direction2/pilot_decision.json` 的 `chosen_start="cold"`）。本設計沿用冷啟動 θ=0，稱「從頭訓練的攻擊鏈」，不引入隨機權重。
2. **E1 v2 的適應鏈預算遠小於原鏈。** 原鏈每代 λ=256、n_S=256、416 代；E1 v2 適應鏈每代 λ=64、n_at=32、240 代。以樣本數算約差 55 倍（重算：416×256×256＝27.3M vs 240×64×32＝0.49M）。本實驗用原鏈的 λ／n_S／σ0／代數，不照抄 E1 v2 的 240 代。
3. **warm-start 不對稱的方向，只會讓 E1 v2 的 D 偏大，不會偏小。** a\* 是針對原設定訓練的，對 C2 只是轉移；若 C2 下有更強的攻擊者，G_C2 更高、真實降幅更小（e1v2結果 §6.1 同寫法）。能翻盤的是**原設定一側**被低估；本設計用已有的「原設定同型 Δ」基準（§2.6 第 1 項）描述它；「原設定一側不重訓」已定（§2.8、§10）。

---

## 第一部分：事實盤點

### 1.1 主實驗原攻擊鏈（a\* 的起點 `orig_run0`＝`main_M3C_r0.5` run0）

| 項目 | 值 | 來源 |
|---|---|---|
| 設定 | M3C、r=0.5、h=6、κ_ref=0.5、L=2500、p=0.1；T_train=20000、T_val=100000 | plan `frozen_budget`；`docs/direction2/results/main.md` |
| 初始化 | 冷啟動 θ=0（`train_segment` 預設 theta0=None 即 zeros） | `arbitration/rl/train.py:211` 起；`pilot_decision.json` |
| 優化器 | sep-CMA-ES（55 參數＞50），σ0=0.3，optimizer seed＝9000+run_id，mirrored sampling | `arbitration/rl/cmaes.py:make_optimizer`；`train.py:223` |
| 每代 | λ=256 個候選＋1 列誠實基準，各在 n_S=256 個訓練 seeds 上評估；fitness＝每步個人效用均值，記分窗 T−L（L=2500） | plan（lam=256、n_S=256）；`train.py:evaluate_generation`；`env.score_window` |
| 總代數 | r=0.5：**416 代**；patience=None，不早停 | plan `frozen_budget.G_gens`；parts `gens_done=416` |
| 訓練 seeds | 0–999 池；perm_k＝`random.Random(9000+run_id).shuffle(range(1000))`；每代取 256 個區塊（mod 1000） | `train.py:42 seed_block` |
| 驗證／選模 | 每 50 代在 val seeds 2000–2031（n=32）、T=1e5、記分窗 T−L 驗證 optimizer mean 的 G_val（8 點：50,…,400）；取最大者（平手取早） | `train.py:98 make_val_fn`、`select_checkpoint` |
| 結果 | 三個 run 皆選第 350 代；G_val＝0.03306／0.03059／0.03162 | `docs/direction2/history/main-status-2026-10-13.md`；`results/direction2/ckpt/main_M3C_r0.5__run*.npz` |
| 歷史機器與耗時（僅供參考，**不是本實驗的機器**） | 學校機 GRID RTX8000-12Q，graph 模式；穩態 t_step＝1.41 ms/步（≈28.2 s/代）；每次驗證 132.7 s | parts `t_step_ms`、`wall_s`；`results/direction2/measure_v2_summary.json` |

**a\*（`orig_adapt0`）的來歷：** 主實驗 run0 第 350 代 checkpoint（冷啟動 350 代）→ 作為 E1 v2 `orig_run0` 適應鏈的 warm start（λ=64、n_at=32、σ0=0.15）→ 該鏈在 AS 上第 60 代 optimizer mean 勝出（atk `orig_run0.json`：g=0 為 0.03590，g=60 為 0.03647）；再於 12 個條目中取 AS 上 G 最大者。即 a\* ＝ 350 代冷啟動 ＋ 60 代小預算適應 ＋ 12 取 1 的選擇。

### 1.2 E1 v2 適應鏈

| 項目 | 值 | 來源 |
|---|---|---|
| 鏈數／防守 | 每防守 3 條（optimizer seed 93000+run），三個防守（原設定、C1、C2）共 9 條 | man `params.attacker`；`scripts/e1_v2.py:48` |
| 起點 | warm start：`opt.mean`＝主實驗 run k 第 350 代 checkpoint | `scripts/e1_v2.py:597` |
| 每代預算 | λ=64、n_at=32（自 AT 256 seeds 輪替）、σ0=0.15、T_train=20000，**記分窗 T−5000** | man `params`；`scripts/e1_pilot.py:127 simulate` |
| 代數與驗證 | 240 代；每 20 代在 AS（128 seeds、T=1e5）驗證，共 13 點；AS 選 checkpoint | man；e1v2結果 §4 |
| C2 的訓練對手池 | Round 2 池＝36 手寫策略＋frozen_run0–2＋C1 適應攻擊者 3 條（共 42）。**a\* 本身不在池中，但其起點 frozen_run0 與 a\* 同源，且 C1_adapt0 由 frozen_run0 適應而來**（池與 a\* 高度相關） | `scripts/e1_v2.py:stage_s2`（`pool_with`）；e1v2設計 §7 Round 2 |
| 歷史機器（僅供參考） | 學校機，graph，`graph_fallback_used=false`；每代 11.0–11.5 s | man／`status.json`；atk `curve[].wall_s` |

### 1.3 已使用 seeds 與 simulate 內部串流

manifest `seed_report`：新集合 7168、既有 4896、交集 0。範圍一律展開 `+{0,1,2,3}·10^6` 比對。

| 範圍 | 用途 | 來源 |
|---|---|---|
| 0–999 | train | `arbitration/rl/seeds.py:11` |
| 1000–1031 / 2000–2031 / 3000–3031 / 5000–5031 | replay / val / val2 / 原 test | 同上 |
| 41000000–41000063 | holdout | `scripts/e1_v2.py:37` |
| 61000000–61000031 | E1 v1 pilot | 同上 |
| 72000000–72000255 / 72100000–72100127 / 72200000–72200255 / 72300000–72300127 | E1 v2 DT / DS / AT / AS | man `seeds` |
| 72400000–72401023 | E1 v2 TEST（已用一次） | 同上；ledger 四筆 completed |

保存紀錄的 seed 欄位掃描在 `results/direction2/e1v2/seed_audit.json`；只證明保存紀錄無重疊，不能證明沒有未保存的外部執行（e1v2設計 §3「推論」）。

**本設計的新 seeds（重算：已用 `scripts/e1_v2.py` 的 `seed_report`／`existing_seeds` 實跑，把 E1 v2 五塊也算進既有）：**

| 區塊 | 範圍 | 數量 | 用途 |
|---|---|---|---|
| EA1_TRAIN | 82000000–82000999 | 1000 | 新攻擊鏈的訓練 seeds（池大小 1000 與原鏈一致；E1 v2 的 AT 只有 256，n_S=256 時每代看到同一組） |
| EA1_SEL | 82100000–82100127 | 128 | 鏈內驗證選 checkpoint、選 A_fresh、a\* 的 SEL 重評、testcheck（前 64 個）。**全新，不重用 AS**（a\* 是在 AS 上被選出的，重用會讓 a\* 佔便宜） |
| EA1_TEST | 82400000–82401023 | 1024 | 一次性 test（n 與 E1 v2 相同） |

預跑結果：新展開 8608（＝(1000+128+1024)×4）、既有展開 12064（＝4896+7168）、交集 0；新集合 sha256 `05232e843d5dd370d0c5bdfef416924990d92021d2b50c01a25ba14ec5eabe97`，既有集合 sha256 `7cdc62f47788f8c38b25b93e78adb2fdf695151693344b175835eefea1afbd85`。預跑在設計階段做；正式凍結時須重跑並寫入 manifest。

**新 seeds 與 simulate 內部衍生串流的關係（紅方 8；已查程式）：**
- `env._make_gens`（`arbitration/rl/env.py:48–56`）對每個環境 seed s 建 **三個** `torch.Generator`，種子為 s、s+1e6、s+2e6（`seeds.py:_OFFSETS` 同）。`simulate` 路徑上，我在 `arbitration/rl/*.py`（不含 `vulns.py`，M3C 一般模擬不走它）搜尋 `manual_seed`／`default_rng`／`Random(`，環境端只有這一處；CMA-ES 的 `np.random.default_rng(seed)`（`cmaes.py:20`）是獨立串流，不吃環境 seed。
- 所以新 seeds 在環境中實際用到 82000000+{0,1e6,2e6}＝82M／83M／84M 系列；不碰 72M–75M（E1 v2 的 72M+{0..3e6}）與 41M／61M 系列。第 4 個偏移 +3e6 在 simulate 中**沒有用到**，但稽核仍保守展開四個偏移比對（與 E1 v2 同）。
- 限制：這個「只有三個衍生串流」的結論來自我讀 `env.py` 與 grep，不是執行期的串流追蹤；若有我沒讀到的路徑（例如 `policy.py` 內部抽樣），grep 結果顯示沒有，但屬「查程式」層級的證據，列入 §8 限制。

### 1.4 現有程式能否直接支援？

**結論：不能「直接」支援，但差距小；全部用新檔案實作，不修改任何已凍結檔案。**

- 可複用（只 import、不改）：`scripts/e1_v2.py:593 attack_chain` 的結構；`scripts/e1_pilot.py:127 simulate`、`metrics`；`e1_v2.py` 的 `upper_bound/lower_bound/tcrit/saturation/seed_report/existing_seeds/expand/mlp/honest_pol/theta_id`；`scripts/d2_holdout_eval.py` 的 `once`（一次性 ledger）、`read/write/sha/digest`；`arbitration/rl/multitrain.py` 的 `Lane`（§3）。
- 必須新增（原因）：
  1. `attack_chain` 的預算來自 E1 v2 manifest，訓練 seeds 由 `rotating(AT, …)` 取，`require_seeds` 只認 E1 v2 五個階段區塊。
  2. `train.seed_block` 只支援 0–999 池且 `require_train_seeds` 強制 <1000；需要等價的「perm 洗牌＋區塊」函式作用在 EA1_TRAIN，且 perm 種子不同（§2.3）。
  3. `train.make_val_fn` 寫死 val seeds 2000–2031 與記分窗 T−L；本設計要在 EA1_SEL（n=128）驗證、記分窗 T−5000。故不走 `train.py --plan` 入口（它鎖在凍結計畫與 FREEZE.md），改用 `e1_v2` 的 `sim`＋`evaluate` 風格自寫。
  4. 新的 manifest／stage／ledger／選模凍結／test／report，路徑 `results/direction2/ea1`。
- **不能動的檔案：** (i) man `protected_sha256` 涵蓋 `arbitration/rl/*.py` 全部（含 `graphsim.py`、`multitrain.py`）與 `arbitration/gpu/*.py`（已查 manifest 鍵）；(ii) man `input_sha256`（不是 protected_sha256）涵蓋 `scripts/e1_v2.py`、`scripts/e1_v2_launch.sh`、`scripts/e1_pilot.py`、`scripts/d2_holdout_eval.py`、`scripts/d2_main_eval.py`、`tests/direction2/test_e1_v2.py`，以及 E1 v2 設計檔 `docs/direction2/design/e1-v2.md`（另有 `design_sha256`）。兩組都由 `load_manifest` 的 `check_hashes` 檢查，另對 `input_sha256` 的檔案做 `git diff --quiet code_git_sha`；改任一個，`e1_v2.py report` 就無法重現 E1 v2 結果。**本設計的平行化不需要改上述任何檔案**（§3）。
- 工作量（推論）：`scripts/ea1.py` 約 550–750 行（含 lane 驅動），`tests/direction2/test_ea1.py` 約 300–400 行，啟動腳本約 60 行。無新演算法。

### 1.5 執行機器與環境釘選（紅方 1）

**執行機器：astarc（學長的共用 VM）。**

| 項目 | 值 | 來源 |
|---|---|---|
| GPU | Quadro RTX 4000，8GB（可用約 7.5GB），Turing sm_75；**實體卡，不是 vGPU** | gpu-center `platforms/astarc.md` §1、§7 |
| CPU／RAM | Xeon Silver 4110，20 CPU，125GB RAM | 同上 |
| 驅動 | 595.71.05（CUDA 13.2）；`/usr/local` 有 cuda-12.1 與 13.2 | 同上（2026-10-21） |
| Python／套件（本實驗環境，須在 astarc 建 venv 後實測確認） | Python 3.10；`torch==2.6.0+cu126`、`numpy==2.2.6`、`scipy==1.15.3`（對齊學校環境） | 任務指定；`astarc.md` §6 |
| 共用機規則 | 只動 `ucc` 家目錄；開跑前先查 GPU 有無他人程式（有就排後面、不可停掉別人）；同一時間我們只跑一個 GPU 程式；`df -h /` 扣掉用量後須剩 ≥10G；VM 可能被關機，需定期 `rsync` 取回 checkpoint | 同上 §4、§5、§6 |

**manifest 必記（S0 凍結時由 `freeze` 在 astarc 上讀取並寫入）：** `nvidia-smi` 的 GPU 型號與 UUID、驅動版本、`torch.__version__`、`torch.version.cuda`、`torch.cuda.get_device_capability()`、`torch.cuda.get_device_name()`、Python 版本、numpy／scipy 版本、`OMP_NUM_THREADS` 等執行緒環境變數、主機名。**每個階段啟動時檢查**：GPU 型號、torch／CUDA／numpy／scipy 版本與 manifest 不一致即拒絕開跑（只有純 CPU 的 `report` 階段可在別台機器跑，且須在輸出裡記下當時的版本）。

**同機配對原則（紅方 1）：**
- 主判準的兩個比較對象 a\* 與 A_fresh 在 **astarc 上、同一次 `simulate` 呼叫、同一批 seeds** 一起重評（`[誠實, a*, A_fresh]`）。Δ_s 的兩端永遠在同一台機器、同一版 torch、同一個 kernel 序列算出。
- 跨機數字**只作描述，不參與任何判定**：EA1_TEST 上的 D_new 對 E1 v2 的 0.004236；12 個既有條目在 EA1_TEST 的 G 對 E1 v2 test；SEL 上 a\* 的 G 對 E1 v2 AS 的 G_AS。
- graph 對 eager 的逐位比對，**只在 astarc 本機內**要求（§2.5、§6）。不要求 astarc 與學校（或本機）之間逐位相同。

---

## 第二部分：設計

### 2.1 研究問題與唯一主要判準

**研究問題。** E1 v2 的 a\*（`orig_adapt0`）是「針對原設定訓練後、拿去打 C2」的轉移攻擊者；C2 下重新適應的攻擊者（warm start 自原設定攻擊者、λ=64／n=32／240 代）沒有一條在 AS 上超過它。這是否只因為 C2 一側的搜尋偏弱？本實驗在 C2 上以 **同 λ／n_S／σ0／代數（目標窗與驗證不同，§2.3）** 從頭訓練攻擊鏈，看它們能否超過 a\*。

**唯一主要判準（全新 test seeds EA1_TEST，n=1024，astarc 上同機逐 seed 配對）。**

- **A_fresh（寫死的選取規則）：** 3 條新鏈各自在 EA1_SEL 上選出最佳 checkpoint（§2.4）；再從這 3 個 checkpoint 中取 **EA1_SEL 上 G_C2 最大者**為 A_fresh（平手取編號小的 run）。A_fresh 與 a\* 的 theta 在 test 前凍結並寫進 `selection_frozen.json`，test 不做任何 argmax。
- Δ_s ＝ G_C2(A_fresh; s) − G_C2(a\*; s)，s 為 EA1_TEST 的 1024 個 seeds；G 的定義與 E1 v2 相同（相對同 seed 誠實策略的每步個人效用增量，記分窗 T−5000，C2 設定）。
- LB＝mean(Δ)−t·SD(Δ)/√n，UB＝mean(Δ)+t·SD(Δ)/√n，t＝t_{0.95,1023}＝1.6463（重算），皆為**單側 95%**；兩者合起來是 90% 雙側區間（沿用 e1v2設計 §5 口徑）。**計算寫死：** SD 用 ddof=1；t＝`scipy.stats.t.ppf(0.95, n−1)`；直接呼叫 `scripts/e1_v2.py` 的 `lower_bound`／`upper_bound`（已查：兩函式存在，內含 `tcrit(n)`＝`stats.t.ppf(.95, n-1)` 與 `x.std(ddof=1)`，公式與上式相符，不另寫第二份實作）。
- **門檻 ε＝0.001。**
- **結論模板限制（紅方 3）：** 主判準的所有結論句，主詞只能是「SEL 選出的單一攻擊者 A_fresh」。不得寫成「從頭訓練的攻擊者（泛指）」「3 條鏈」「從頭訓練這個方法」。因為被檢定的是 3 取 1 之後的單一對象；對「這個方法的整體能力」沒有登記檢定。

### 2.2 決策樹（依序判斷，第一個命中即為結果；嚴格不等號，恰等於門檻不算命中）

1. LB > ε → **更強**：「SEL 選出的 A_fresh 在 C2 下比 a\* 強至少 0.001。」
2. 否則若 UB < ε → **沒有更強（排除 A_fresh 強過 a\* 達 ε 以上）**；再依序給子標籤（第一個命中者為準）：
   - (a) UB < 0 → **更弱**。
   - (b) 否則若 LB > 0 → **有小幅提升但未達 ε**：「A_fresh 顯著高於 a\*（單側 95% 下界＞0），但排除了高出 ε 以上。」
   - (c) 否則 → **持平**：「無法排除略高於 0 的差，但排除了高出 ε 以上。」
3. 否則 → **證據不足以判定**（單側區間跨過 ε）。

**注意事項（登記時寫死）：**
- (b) 節點在規劃的 σ_Δ＝0.0095、n＝1024 下幾乎不可達：它要求 mean 落在 (t·SE, ε−t·SE)＝(0.00049, 0.00051)（SE＝0.000297，重算），寬度僅約 0.00002；只有當樣本 SD 明顯小於規劃值時才有實質機率（推論）。納入是為了決策樹對所有 (LB, UB) 組合完備。
- 子標籤 (a)(b)(c) 與節點 1 是對**同一個 Δ** 的多個單側 95% 檢定（對 ε、對 0）。**未做多重比較校正。** 只有節點 1（對 ε 的 LB）是登記的主判定；「更弱」（UB<0）與「小幅提升」（LB>0）是措辭層級的分類，不得單獨拿來當作顯著性主張，也不得在論文中當作獨立發現。

**ε＝0.001 的理由（三個來源，皆為既有數字）：**
(i) 與 E1 v2 的飽和門檻同值（e1v2設計 §4 `sat_threshold`＝0.001，約為 AS 平均 SE 0.0012 的量級）。
(ii) 約為 E1 v2 實測降幅 D＝0.004236 的 24%、δ＝0.005 的 20%：若 C2 一側真的強了 0.001，降幅由 0.0042 降為約 0.0032，是可在論文裡明確說「攻擊集合不完整」的量。
(iii) 12 個既有條目在 C2 下的 test G 為 0.02902–0.03107（全距 0.00205，重算自 `test/cand_secondary_raw.json`），與 a\* 的逐條差在 −0.0015 到 +0.00055 之間；ε 約為這個自然散佈的一半。
**ε 是設計選擇，不是從資料推出的唯一值**；且 ε 的選擇受到 E1 v2 已觀察到的數字影響（§8 限制 7）。已定（§10）。

**檢定力（n=1024；重算，非中心 t；σ_Δ＝0.0095，來源：既有 test 資料中其他攻擊者與 a\* 在 C2 下的逐 seed 配對差 SD 為 0.0067–0.0102）：**

| 真實 mean Δ | P(更強：LB>ε) | P(UB<ε) | P(更弱：UB<0) |
|---|---|---|---|
| −0.0010 | 0 | 100% | 95.7% |
| −0.0005 | 0 | 100% | 51.5% |
| 0 | 0 | 95.7% | 5.0% |
| +0.0005 | 0 | 51.5% | 0 |
| +0.0010 | 5.0% | 5.0% | 0 |
| +0.0015 | 51.5% | 0 | 0 |
| +0.0020 | 95.7% | 0 | 0 |
| +0.0030 | 100% | 0 | 0 |

（本稿已重跑 nct 驗證此表。）power≥0.80 的最小真實 Δ：判「更強」約 0.00174。**固有限制：** 真實 Δ 落在 +0.0005 到 +0.0015 之間時，多半落入「證據不足」節點；這是 ε 與 n 的取捨，登記時接受。

### 2.3 與原鏈（主實驗）的逐項偏離表（紅方 2、7）

**不稱「同配方」。** 下表是完整清單；表外的項目視為與原鏈相同。**同**＝與原鏈相同；**異**＝不同（列方向或理由）。

| 項目 | 原鏈（主實驗 run k） | E-A1 新鏈 | 同／異 | 說明與方向 |
|---|---|---|---|---|
| 候選數 λ | 256 | 256 | 同 | |
| 每代訓練 seeds 數 n_S | 256 | 256 | 同 | |
| 初始步長 σ0 | 0.3 | 0.3 | 同 | |
| 代數 | 416，不早停 | 416，不早停、不延長 | 同 | |
| 初始化 | θ=0 冷啟動 | θ=0 冷啟動 | 同 | |
| 優化器 | sep-CMA-ES、mirrored、55 維 | 同 | 同 | |
| T_train／T_val | 20000／100000 | 20000／100000 | 同 | |
| 防守設定 | 原設定（h=6、κ_ref=0.5、L=2500） | C2（h=6.175…、κ_ref=0.4664、L=2412） | **異** | 這正是要測的對象 |
| **訓練 fitness 記分窗** | T−L＝T−2500＝17500 | **T−5000＝15000** | **異** | 見下方「記分窗的選擇」 |
| **評估記分窗（SEL 驗證、SEL 重評、test）** | 驗證用 T−L；原鏈 test 另有其窗 | **一律 T−5000** | **異** | 與 a\* 在 E1 v2 的 AS／test 評估同尺 |
| 訓練 seeds 池 | train 0–999 | **EA1_TRAIN**（82000000–82000999，1000 個） | **異** | 全新環境 seeds，無已知方向性影響（推論） |
| 訓練池排列種子 | `Random(9000+run_id)` | **`random.Random(94100+run).shuffle(range(1000))`**，再以排列結果索引 EA1_TRAIN | **異** | 寫死；不得沿用 `9000+run_id` |
| optimizer seed | 9000+run_id | **94000+run** | **異** | 與所有既有 seed（9000+k、93000+k、91000/92000 系列）不同 |
| **驗證 seeds** | val 2000–2031，n=**32** | **EA1_SEL 82100000–82100127，n=128** | **異** | 選模雜訊較小，對新鏈略有利（§8 限制 3 的殘餘不對稱） |
| **驗證頻率** | 每 50 代；8 點（50,…,400） | **每 50 代；8 點（50,…,400）**，驗證 optimizer mean | 同 | 寫死；第 416 代不驗證（與原鏈一致） |
| 選模規則 | G_val 最大者，平手取早 | EA1_SEL 上 G 最大者，平手取早 | 同 | |
| 三個 run 再取最佳 | val 上取最佳 | EA1_SEL 上取 G_C2 最大者，平手取小編號 | 同（驗證集不同） | 即 A_fresh |
| 執行機器 | 學校 RTX8000-12Q | **astarc Quadro RTX 4000** | **異** | 所有比較只用 astarc 同機配對（§1.5） |
| 執行排程 | 單 cell 依序 | **3 條鏈同程序 3 lane 並行（若 smoke 判定採用，§3）** | **異（不改結果）** | 有逐位相同測試保證與依序執行結果相同（§3.4、§6） |
| 與 a\* 的預算關係 | — | a\*＝350 代冷啟動＋60 代小預算適應＋12 取 1；新鏈＝416 代冷啟動 | **異** | 同量級，不是同一條件下的 A/B |

**記分窗的選擇與理由（紅方 2 建議「訓練用原鏈的 T−L、評估用 T−5000」；本稿的判斷不同，說明如下）：** 本稿把**訓練 fitness 的記分窗也定為 T−5000**，與評估一致。理由：(i) 訓練窗 T−L（C2 下 L=2412，窗 17588）比評估窗 T−5000 多包含最後 2412–5000 步；`score_window` 的設計目的是排除尾端截斷偏誤（`env.py:81`），攻擊者若在訓練時利用尾端 2412–5000 步的截斷效應，評估窗（T−5000）會把這些好處扣掉，等於替新鏈設了一個對它不利的訓練／評估錯配，使「沒有更強」的結論偏向保守而降低資訊量；(ii) 本實驗要回答的是「C2 一側的搜尋夠不夠強」，應給新鏈最公平的機會，讓它優化被評估的量；(iii) E1 v2 的適應鏈本身就是訓練窗＝評估窗＝T−5000，兩者因此同尺。代價：偏離原鏈的訓練窗（已列為「異」）。**已定：訓練 fitness 記分窗為 T−5000**（理由即上述 (i)–(iii)：與評估窗一致，讓新攻擊者發揮最大，使「沒有更強」的結論不因訓練目標錯配而偏弱）；凍結後不得改。

**驗證 seeds 數與頻率寫死：** n=128、每 50 代、8 點、T=1e5、記分窗 T−5000；任何一項改變都需新登記。

### 2.4 攻擊鏈設計

| 項目 | 值 |
|---|---|
| 鏈數 | 3（run_id 0,1,2；optimizer seed＝94000+run；訓練池排列種子＝94100+run） |
| 防守設定 | C2：p=0.1、tol＝6.175461735115074、κ_ref＝0.4663589529547838、L＝2412.0（sel `choice.cfg`），M3C、r=0.5 |
| 初始化／優化器 | θ=0；sep-CMA-ES，λ=256，σ0=0.3，mirrored sampling |
| 每代評估 | n_S=256 個訓練 seeds，取自 EA1_TRAIN 排列後的區塊：`block(g)＝[perm[(g·256+i) mod 1000] for i in 0..255]`，再映到 `82000000+perm[…]`；T_train=20000；記分窗 T−5000；fitness＝個人效用均值／T_score |
| 總代數 | 416，不早停、不延長 |
| 驗證 | 每 50 代，EA1_SEL（128 seeds）、T=1e5、記分窗 T−5000，驗證 optimizer mean；共 8 點；每點存 theta、G_SEL、raw 路徑；取最大者為該鏈 checkpoint（平手取早） |
| A_fresh | 3 個 checkpoint 中，於 `select` 階段同一次呼叫重評的 G_SEL（C2）最大者（平手取小編號） |
| 飽和旗標（只描述） | 沿用 `saturation()`：gmax＝400、val_every＝50，「g≤300 的最佳」對「g∈{350,400} 的最佳」，增益＜0.001 標飽和；不觸發延長，不改判準 |

**SEL 上必做的評估（紅方 9；全部在 astarc 上、凍結選模之前）：**

**可運行的前提（寫死）：** 3 條鏈都須跑滿 416 代且無 NaN，才能進 `select`；任何一條無法完成（NaN、反覆失敗、無法續跑等），就停下來問使用者，**不縮鏈、不改判準**。

1. 3 條鏈各自 8 個驗證點在 C2 下的 G_SEL（鏈內即時算）。**鏈內即時算的值只用來選該鏈自己的 checkpoint**，不用於跨攻擊者比較。
2. **重評（`select` 階段，同一次 `simulate` 呼叫，C2、EA1_SEL 128 seeds）：** 策略順序固定為 `[誠實, a*, chain0 checkpoint, chain1 checkpoint, chain2 checkpoint]`（誠實為 policy 0）。**A_fresh 的選取、A_C2′ 的選取（§2.6 第 4 項）與 a\* 的 G_SEL，一律使用這一次重評所得的 G_SEL**；A_fresh 與 A_C2′ 不得使用鏈內即時值。這才是同機同批配對。
3. a\* 與 3 個原設定 frozen_run0–2 在**原設定**下於 EA1_SEL 的 G（為「原設定同型 Δ」的選取用，§2.6 第 1 項；同樣一次呼叫、順序 `[誠實, a*, frozen_run0, frozen_run1, frozen_run2]`）。
4. 描述性：12 個既有條目與 3 條新鏈 checkpoint 在 C2／原設定下的 SEL G（共 15 個攻擊者 × 2 設定），不參與判定。

**訓練 seeds 池的選擇：** 用全新 EA1_TRAIN，理由：(1) 沿用 E1 v2「不再重用 0–999 作新階段環境 seeds」原則；(2) 不需靠「這些 seeds 是否已用來訓練過 a\*」論證無洩漏。不沿用 0–999（已定，§10）。

### 2.5 評估方式（同機）

- 路徑：`scripts/e1_pilot.py:simulate`（graph、float64、`ad_h=inf`、記分窗 T−5000）；test 用 `simulate_chunked`，chunk＝64。test 與 testcheck 為 strict（無 graph fallback、無自動重試）。
- **testcheck（紅方 8）：用 EA1_SEL 的前 64 個 seeds（82100000–82100063），絕不用 EA1_TEST。** 以 test 完全相同的路徑（`simulate_chunked`、chunk 64、graph、與 test 相同的策略組合共 4 組）評估，並與 `sim_eager`（不分塊、不用 graph）比對，容差無，snap 三欄位逐元素完全相同；**只在 astarc 本機內比對**。結果寫 `testcheck.json`（含 manifest 與 freeze_record 雜湊、路徑識別、機器識別）。
- test 與 testcheck 一律單 stream、預設 stream、依序執行（與 E1 v2 的已驗證路徑相同）；**不使用 §3 的多 lane**。
- 配對：同 seeds、同記分窗；誠實策略為每次評估的 policy 0。
- **EA1_TEST 上的評估（各一次，經 `once` ledger）：**
  1. `main_C2`：C2 下 [誠實, a\*, A_fresh]（主判準）。
  2. `main_orig`：原設定下 [誠實, a\*, A_fresh]（轉移，描述性）。
  3. `sec_C2`／`sec_orig`：C2 與原設定下 [誠實, 12 個 E1 v2 條目, 3 條新鏈各自的 SEL 最佳 checkpoint]，僅描述、未校正多重比較。12 個條目＝frozen_run0–2、orig_adapt0–2、C1_adapt0–2、C2_adapt0–2（重算自 `secondary.json`／sel `attackers`），其中 9 個是不同參數（e1v2結果 §5）。
- 主判準不受 2、3 的結果影響。
- 耗時：待 smoke 量測（§5）。

### 2.6 次要結果（只描述，不參與判定）

1. **原設定同型 Δ（基準；紅方 6；只讀不判）。** Δ_orig,k(s) ＝ G_orig(frozen_run k; s) − G_orig(a\*; s)，k∈{0,1,2}，s 為 EA1_TEST，原設定下、astarc 同機配對；另以 EA1_SEL 上 G_orig 最大者（平手取小編號）為「F_orig」，報 Δ_orig,F 的 mean 與單側 95% LB／UB（與 A_fresh 的選取規則同型）。frozen_run0–2 是原鏈在第 350 代的 checkpoint，即「冷啟動、λ=256、n_S=256、416 代設定下訓練出的攻擊者」。
   **先驗預期 Δ_orig<0（登記時寫下）：** a\* 本身是 frozen_run0 再加 60 代適應，並在 12 個條目中依 AS 取最大，結構上比 frozen_run 多了訓練與選擇；E1 v2 test 上的既有數字也是負的（G_orig：frozen_run0／1／2＝0.03432／0.03326／0.03337，a\*＝0.03476；差約 −0.0004 到 −0.0015，重算自 `test/orig_secondary_raw.json`）。**因此在 C2 一側出現「持平或更弱」（Δ≤0）不能當作獨立證據來說 C2 一側的搜尋不弱**——即使 C2 一側的搜尋與原設定一側同樣有力，也預期看到 Δ<0。這個基準只用來讓讀者看「C2 一側的 Δ 與原設定一側的 Δ 是否同一型態」，不做統計比較。
2. **機器漂移檢查（紅方 1；描述性）。** 在 astarc 上用 a\*（theta 取自 sel `astar.orig.theta`，theta_sha256 須等於 `88ed926a…`）重評 E1 v2 **已用過**的 AS（72300000–72300127，128 seeds，T=1e5，記分窗 T−5000），原設定與 C2 各一次，與 sel `G_AS_astar` 比較（原設定 0.03646720854981316、C2 0.033836472647898505）；並與 sel `transfer_matrix_AS` 指向的 E1 v2 raw（`evaluations/f99ad…`、`evaluations/aef52…`）做逐 seed 比較。報：平均差、逐 seed 最大絕對差、逐位相同的 seed 數。**無通過／失敗門檻、不影響任何判定**；寫死的用法：若不是逐位相同，摘要與結果文件必須明講「astarc 與學校的數字不逐位相同，差為 …」，且所有跨機比較（第 3、5、6 項）一律標為「跨機、僅供參考」。AS 是 E1 v2 已用的 seeds，用於此檢查不構成新的 test 暴露。
3. **各鏈 checkpoint 逐一對 a\* 的 Δ（紅方 3；描述性）。** 3 條鏈各自的 SEL 最佳 checkpoint 在 C2 下對 a\* 的 Δ（mean 與單側 95% 區間），來自 `sec_C2`。**不參與判定；3 個區間未校正多重比較；不得用來取代 A_fresh 的結論。**
4. **更新的降幅 D′（描述性）：** D′＝G_orig(a\*) − G_C2(A_C2′)，A_C2′ 為 a\* 與 A_fresh 中 **EA1_SEL 上 G_C2（§2.4 第 2 項同一次重評的值）較大者**（規則事先寫死；平手取 a\*）。A_C2′ 是在 SEL 上選取的，test 上 D′≤D 不保證成立。報 mean 與單側 95% LB／UB，與 δ＝0.005 比較；**不重新套用 E1 v2 決策樹、不改 E1 v2 的判定**；D′ 是**描述性**，主表仍列 D，不與主判準並列。
5. **a\* 的 D 第二樣本（E-A2 的一部分）：** D_new＝G_orig(a\*)−G_C2(a\*)（EA1_TEST，astarc），與 E1 v2 的 0.004236 [0.003883, 0.004589]、AS 的 0.002631 比較；**跨機，僅供參考**（見第 2 項）。
6. 12 個既有條目在 EA1_TEST 的 G 與 E1 v2 test 的差（跨機，僅供參考）。
7. 誠實服務：C2 與原設定的誠實列（policy 0）在新 seeds 上的 ΔR、Δα 與 UB（描述性）。
8. 三條新鏈各自的 SEL 軌跡、訓練曲線、飽和旗標、最佳代數、各鏈 checkpoint 的 test G_C2。

### 2.7 三種結果的預先寫法

下列句子是登記時寫好的模板；數字屆時代入，不改結構。主詞一律是「SEL 選出的 A_fresh」。

**（一）更強（LB>ε）。**
- 對主張的影響：E1 v2 的主判定（a\* 上 D＝0.004236，排除 0.005）**不被推翻**：它是預先登記、對固定 a\* 的結果，仍然成立。改變的是「a\* 就是 C2 下最強攻擊」這個理解，以及降幅的外推：真實降幅（對最強已知攻擊者）比 0.0042 更小。
- 方向：C2 一側更強預期讓 D′ 更小（A_C2′ 在 SEL 上選取，test 上不保證），「效果不足」更成立，不是被修正成「有效」。
- **可能的解釋不止一個，不可寫成單一原因（紅方 4）。** 至少有：
  1. **E1 v2 的 C2 適應鏈搜尋偏弱**（預算約為原鏈的 1/55、warm start 自與 a\* 同源的起點、σ0=0.15 較小）；
  2. **C2 是對已知攻擊者池調出來的，可能是防守對已知池過配**：C2 的 Round 2 訓練池是 36 手寫＋frozen_run0–2＋C1 適應攻擊者 3 條（§1.2），a\* 本身不在池中但與池高度相關；E1 v2 的適應鏈又都從 frozen 起點出發，離這個池不遠。從頭訓練的攻擊鏈不在這個池的鄰域，若它在 C2 下較強，可能反映的是 C2 對池外攻擊者的弱點，而不是「適應鏈的預算不夠」；
  3. 從頭訓練本身探索到另一類策略（與搜尋預算無關）；
  4. 3 取 1 的選擇偏誤（已由全新 test 吸收，但不能估計「更多條鏈會不會更強」）。
  **本實驗無法區分 1 與 2（以及 3）。** 結果文件只能寫「存在比 a\* 更強的 C2 攻擊者」，並列出上述解釋；不得寫「E1 v2 的 C2 側搜尋偏弱」作為結論。
- 怎麼寫：「E1 v2 的攻擊集合對 C2 不完整：SEL 選出的從頭訓練攻擊者 A_fresh 在 C2 下比 a\* 高 Δ（單側 95% 下界 LB）。以更新後的最強已知攻擊者計，C2 的降幅為 D′（[LB′, UB′]），低於 E1 v2 報告的 0.004236。E1 v2 的降幅是對『固定的轉移攻擊者 a\*』而言。原因不能由本實驗判定（候選：適應鏈預算偏小；C2 對已知對手池過配；從頭訓練探索到不同策略）。結論『C2 不構成修補』不變，且更強。」
- E1 v2 結論的修正：e1v2結果 §6.1「若攻擊端偏弱，真實降幅可能比測到的更小」由推論升級為「已量測到」；§7「不能宣稱最強攻擊」維持，並新增一條可宣稱：「存在比 a\* 更強的 C2 攻擊者」。論文摘要與主表的 C2 降幅**仍列 D**；D′ 另行標為「描述性」，不與主判準並列。
- 後續：原設定一側也從頭訓練不在本登記範圍（§2.8）；若要做須另寫新登記。

**（二）沒有更強（UB<ε）。**
- (a) 更弱（UB<0）：A_fresh 在 C2 下不如 a\*。**先驗預期本來就是 Δ<0（§2.6 第 1 項：a\* 多了 60 代適應與 12 取 1 的選擇），所以這個結果本身不是獨立證據。** 怎麼寫：「SEL 選出的從頭訓練攻擊者 A_fresh 在 C2 下比 a\* 低 |Δ|（單側 95% 上界 UB<0）；原設定一側的同型比較（frozen 冷啟動鏈 vs a\*）為 Δ_orig（描述性，§2.6 第 1 項），兩側型態 [相同／不同]。這個結果不能解讀為『C2 一側的搜尋不弱』的獨立支持，因為預期本來就是 Δ<0；它只表示沒有發現比 a\* 更強的從頭訓練攻擊者。」
- (b) 小幅提升但未達 ε（LB>0 且 UB<ε）：「A_fresh 顯著高於 a\* 但排除高出 ε 以上；以 ε 為準不構成『更強』。」寫法同 (c)，並如實報 Δ 與區間。
- (c) 持平（0≤UB<ε 且 LB≤0）：「排除了 A_fresh 高出 a\* 達 0.001 以上；不排除小於 0.001 的差。」
- 對 E1 v2：§6.1／§6.2 的限制由「本實驗無法區分」改為「以另一種搜尋設定檢查過，沒有發現更強」。**仍不能寫「a\* 是最強攻擊」**（§8）。

**（三）證據不足（跨過 ε）。**
- 對主張：不改 E1 v2 的任何數字；限制段原樣保留，並加一句「額外的從頭訓練對照無法在 ±ε 內分辨」。
- 怎麼寫：「SEL 選出的從頭訓練攻擊者 A_fresh 在 C2 下相對 a\* 的差為 Δ（單側 95% 區間 [LB, UB]），跨過預先登記的門檻 0.001，因此既不能宣稱更強，也不能排除更強。」
- 不事後加鏈、加 seeds 或改 ε 來「解決」；若要追加，須另寫新登記（新 seeds）。

### 2.8 原設定一側的從頭訓練：不在本登記範圍

**已定：原設定一側不重訓。** 原設定一側已有同型資料可作基準（§2.6 第 1 項的 frozen_run0–2：冷啟動、λ=256、n_S=256、416 代），只讀不判。對原設定也從頭訓練（對稱設計）不在本登記範圍；若要做，須另寫新登記、使用新 seeds。殘餘不對稱的誠實揭露見 §8 限制 3。

---

## 第三部分：執行方式（把 GPU 與 CPU 用得好；紅方之外的新需求）

### 3.1 先講結論

- **GPU：** 3 條鏈可以在**一個程序**裡用 3 個執行緒、各自一條 CUDA stream、各自一個 `GraphSim` 並行；這套機制 `arbitration/rl/multitrain.py` 已經實作並有測試（speedup §8.2–8.6），**不需要修改 `arbitration/rl` 的任何檔案**，所以不影響 E1 v2 manifest 的 `protected_sha256`。astarc 規則是「同一時間只跑一個 GPU 程式」；一個程序裡 3 個 lane 仍是一個 GPU 程式，符合規則。
- **預期增益有限，且在 astarc 上沒量過：** 本機 RTX 5070 Ti 上量到的是單 cell graph 後 GPU util 已 99%，並行 2 lane +18%、3 lane +25%（speedup §8.3）。astarc 的卡不同，我推論它同樣是 GPU 飽和（Quadro RTX 4000 的 fp64 吞吐低，**推論，未量測**），3 lane 的增益可能小於 25%。所以是否採用，由 smoke 量到的吞吐決定（§3.3），不是預先假設。
- **CPU：** 20 核**不是瓶頸，不應為了「用滿」而製造負載**。這個負載的 CPU 端工作很輕（見 §3.5），多的核心沒有合法的方法讓同一個模擬跑得更快。能合法使用多餘核心的是同時段的**純 CPU 工作**（測試、稽核、報告乾跑）。

### 3.2 現況查到的事實（程式位置）

| 事實 | 位置 |
|---|---|
| 每代 λ+1＝257 條 policy × 256 seeds＝65792 個 batch 元素，由**一次** `env.simulate` 呼叫、同一組 kernel 處理；候選之間已經是資料平行 | `train.evaluate_generation`（一次 `sim_fn(rows, seeds)`）；`env.simulate`；E1 v2 的 `sim`→`v1.simulate`。**所以「把每代 λ 個候選 rollout 再平行化」沒有可做的空間**：它們本來就在同一個 batch 內；再切小只會降低每個 kernel 的佔用率（推論） |
| 一個 `GraphSim`（含其 CUDA graph 與靜態緩衝區）屬於一條 stream／一個執行緒 | `graphsim.py` 的 `GraphSim` docstring（"One instance per CUDA stream / thread"） |
| 多 lane：每 lane＝一個執行緒＋一條 CUDA stream＋自己的 `GraphSim`；建圖用 `_BUILD_LOCK` 序列化（多執行緒同時 capture 不合法）；`capture_error_mode="thread_local"` | `multitrain.py:Lane`、`run_concurrent`；`graphsim.py:_BUILD_LOCK` |
| 每個 lane 的隨機串流獨立：環境端為每個 seed 自建 `torch.Generator`（`env._make_gens`），CMA-ES 的 numpy Generator 各鏈自有；`graph` 不把 RNG 帶進圖內（RNG block 在圖外 eager 抽取並 `copy_` 進靜態緩衝區） | `env.py:48–56`；`cmaes.py:20`；`graphsim.py` 模組 docstring |
| 既有的並行逐位相同測試（本機，torch 2.12、RTX 5070 Ti）：3 個 cell 同時，part 與 checkpoint 與各自單獨 eager 逐位相同；正式大小的 `d2_graph_check.py` 亦同 | `tests/direction2/test_d2_multi.py`（`test_concurrent_cells_equal_each_cell_alone_eager`）；speedup §8.2、§8.4 |
| **這些測試只在本機 torch 2.12 驗證過；astarc（torch 2.6.0+cu126、sm_75）上沒有跑過** | speedup §8.7 第 1、3 點（原文即說 2.6 上沒跑過、不同驅動行為沒驗證） |
| E1 v2 的 `attack_chain` 是單鏈、用全域 `GRAPH`（預設 stream）；`main` 設 `torch.set_num_threads(1)` | `scripts/e1_v2.py:593`、`:1045` |
| `GraphSim.run` 內用 `torch.cuda.stream(self.stream)`，但 `Lane` 的 stream 需要整個 lane 執行緒的工作都放在該 stream 的 context 內才完整（`multitrain.run_segment` 即如此） | `graphsim.py:run`；`multitrain.py:run_segment` |

### 3.3 方案與採用規則

**方案：** `scripts/ea1.py` 的 `train` 階段開 3 個執行緒（run 0、1、2），每個執行緒：建立 `Lane(graph=True, K=32)`（import 自 `multitrain`，不改它）；整個執行緒體放在 `torch.cuda.stream(lane.stream)` 內；跑自己那條鏈的 416 代與 8 次驗證；每代存 checkpoint。主執行緒 join；單一 lane 失敗以 `_guard` 風格捕捉，不拖垮其他 lane，但整個階段以非 0 結束。`test`、`testcheck`、`select` 不使用多 lane（§2.5）。

**共享可變狀態只有三處，全部加鎖：** (a) `_BUILD_LOCK`（`graphsim` 內建）；(b) `status.json`／stamp 寫入；(c) git commit／push（`publish`）。每條鏈的 optimizer、perm、檔案、state 完全獨立；沒有共享的 RNG；不使用 `torch.manual_seed` 全域種子（環境端不依賴）。

**採用規則（smoke 在正式凍結前量吞吐比；正式 manifest 凍結後重跑逐位相同驗收，train 前寫進 `run_config.json` 並 commit；寫死如下，§7 步驟 4–6）：** smoke 量 (i) 單 lane 跑 5 代的每代秒數 t1；(ii) 3 lane 同時各跑 5 代的每代秒數 t3（每 lane）。總吞吐比＝3·t1 / t3。**若比 ≥ 1.10 且 3 lane 的逐位相同測試（§3.4）通過且 GPU 記憶體峰值 ≤ 6 GB，則採 `lanes=3`；否則 `lanes=1`（3 條鏈依序跑）。** 這是純執行層的選擇：因為 §3.4 保證結果逐位相同，lane 數不影響任何統計結果，也不需要在設計凍結前決定；但門檻 1.10（吞吐比 ≥1.10 且逐位相同）已定、寫死，不得因為看到數字再調。

**三鏈跑滿規則（寫死）：** 3 條鏈都須跑滿 416 代且無 NaN 才能進 `select`；任何一條無法完成就停下來問使用者，不縮鏈、不改判準（與 §2.4 同）。

**其他執行規則：**
- **GPU 共用：** 開跑前查 `nvidia-smi --query-compute-apps`；有別人的程式就排後面，不停別人的。跑的途中若他人開了 GPU 程式，依 astarc 規則讓出：各 lane 在**世代邊界**檢查，發現他人程序則完成當代、存 checkpoint 後以續跑碼退出，由你決定何時重啟（不硬搶 VRAM）。
- **記憶體：** 每個 graph lane 在本機約 0.53 GB（speedup §8.3，astarc 未量），3 lane 約 1.6 GB，加驗證 graph；astarc 可用約 7.5 GB。smoke 量 `torch.cuda.max_memory_reserved` 與 `nvidia-smi`。
- **續跑：** 每條鏈每代存 checkpoint（E1 v2 同）。VM 被關機或 lane 被中斷後，3 條鏈可能停在不同代數，各自從自己的 checkpoint 續跑；結果不受影響（§3.4 第 4 項測試）。
- **取回與 commit：** astarc 上定期 `rsync` 取回；commit／push 在本機做，不在 astarc 與本機兩邊推同一分支（10/15–10/16 事故教訓）。**已定：結果以 `rsync` 取回本機後在本機 commit**，astarc 不 push。

### 3.4 平行化不得改變任何隨機串流或結果：逐位相同測試（寫死，開跑前必須通過）

**定義：** 「結果」＝每代的 CMA-ES 候選矩陣 X、每代 U 矩陣與 fitness 向量、optimizer 的 `state_dict()`（mean、sigma、pc、ps、C、gen、numpy RNG state）、每次驗證的 G_SEL。比較用 `np.array_equal`／整數與 float 的逐位比對（不是 allclose）。

**CPU 測試（無 GPU，用 `capture=False` 的 `GraphSim` 機制模擬，小尺寸 λ、n_S、T）：**
1. 3 條鏈各自依序跑前 N 代（N≥20）vs 3 個執行緒同時跑前 N 代：全部逐位相同。
2. 執行緒啟動順序、sleep 注入造成的交錯不同：結果逐位相同（重複 ≥5 種交錯）。
3. 一條鏈在中途丟例外，另外兩條的結果不變且逐位等於各自單獨跑。
4. 平行中斷後以單 lane 續跑 vs 一次跑完：逐位相同。
5. 各鏈的訓練 seeds 區塊序列與 optimizer 抽樣序列，在平行與依序兩種執行下相同。

**GPU 驗收（astarc，真 capture；正式大小 λ=256、n_S=256、T_train=20000、C2、EA1_TRAIN 前三代；加一次正式大小驗證 128 seeds × T=1e5，使用 EA1_SEL）：**
1. **A**：每條鏈單獨、eager、一個 lane 外的預設 stream，跑 N=3 代＋1 次驗證。
2. **B**：每條鏈單獨、graph、單一 lane，同上。
3. **C**：3 條鏈同時、graph、3 個 lane，同上。
4. 要求 **A＝B＝C 逐位相同**（上述「結果」全部欄位）。A 對 B 同時是 §2.5 的同機 graph=eager 檢查；B 對 C 就是「單鏈依序 vs 平行」。N=3 是下限，若 smoke 時間允許取 N=5（與採用規則的 5 代量測合併，同一次執行）。
5. 任一項不逐位相同 → `lanes=1`，且在 `run_config.json` 記錄失敗原因；若連 A＝B 都不成立 → graph 路徑不可用，回報給你，不自行改成 eager（改變執行路徑需另行決定，且 testcheck 的前提就不成立）。

**訓練後檢查（`lanes=3` 時必做，`select` 之前；item 8）：** 抽一條鏈（run 0），取其第 200 代的 checkpoint，在**單 lane** 上重算該代 fitness，與平行訓練時記錄的該代結果（候選矩陣 X、fitness 向量）逐位比對（`np.array_equal`）。重算所需的第 199 代結束時 optimizer 狀態，由訓練階段在每條鏈第 199 代另存快照提供（只多存一份，不改計算）。**不同則停下來問使用者**，不自行改判。結果寫 `post_train_check.json`。

測試結果寫 `parallel_check.json`，`train` 階段啟動時 `require_parallel_check` 檢查它存在、通過、雜湊與目前 manifest 與程式一致（比照 E1 v2 的 `require_testcheck`）；且只在 `lanes>1` 時要求 C 項。

### 3.5 CPU 端（20 核）怎麼用

**CPU 端實際做什麼（讀程式）：** 每代 CMA-ES `ask/tell`（55 維 sep，numpy，推論為毫秒級以下，`cmaes.py`）；建 257 個 `MLPPolicy`；`np.unique`；`SimState` 初始化與 768 個 per-seed `torch.Generator`；每 1024 步一個 RNG block 的抽取（256 seeds × 4 次 `randn`/`rand`，約 1000 個小 kernel 的 Python 派發）；JSON checkpoint 寫入。graph 模式下，步進本身由 `graph.replay()` 每 32 步一次，CPU 幾乎不吃。**單 lane 時 CPU 約 1 核；3 lane 約 3 核＋主執行緒（推論，由 smoke 的 `top -H`／`pidstat` 實測）**。

**可以合法用多餘核心的：**
1. 3 個 lane 執行緒各占一核（天然使用）。多執行緒受 GIL 限制：RNG block 抽取與 policy 建構是 Python 派發，持有 GIL，這可能是 3 lane 增益只有 +25% 的原因之一（推論）。
2. **同時段的純 CPU 工作**，與 GPU 工作並行，不碰 GPU、不碰 `torch.cuda`：`tests/direction2/test_ea1.py` 的 CPU 測試（可用 `pytest -q -n 4`）、E1 v2 回歸（`tests/direction2/test_e1_v2.py`，CPU）、seed 稽核重跑、`report` 在合成資料上的乾跑、檢定力表重算。以 `nice -n 10` 啟動、總用核數 ≤ 8，避免擠壓共用機上其他帳號（`cxz` 的 tmux、`s12360137` 的 docker 模擬器，astarc.md §1）。
3. 結束後的 `report`（CPU）。

**刻意不做（理由）：**
- **多程序各跑一條鏈**：違反「同一時間只跑一個 GPU 程式」；3 個 CUDA context 只能時間分割，預期沒有增益（推論）。
- **在 CPU 上跑 rollout 分擔 GPU**：CPU 與 GPU 數值路徑不同，配對失效；且慢很多。
- **把 RNG block 抽取移到 helper 執行緒預抽**：需要改 `env.py`／`graphsim.py`（protected），且須重新證明逐位相同。
- **把 BLAS／OpenMP 執行緒開多**：numpy 的小矩陣運算（`w @ Y` 等）多執行緒時加總順序可能改變位元（推論，未測），破壞「與依序執行逐位相同」的前提。固定 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`、`torch.set_num_threads(1)`（E1 v2 同），並記入 manifest。
- **為了「用滿」製造 CPU 負載**：沒有任何統計或速度上的好處，只會影響共用機上其他人。

**若你仍要求「用滿」的最強形式：** 唯一可能顯著提高 GPU 利用率的是增加 lane 數（>3），但這需要證明逐位相同且增益遞減（speedup §8.6 預期遞減，未量）；本設計只登記 lanes∈{1,3}。要試 >3 需另議。

---

## 4. 攻擊者數量、預算、seeds 總表

| 項目 | 值 |
|---|---|
| 新鏈數 | 3（C2 防守）；原設定一側不重訓（已定，§2.8）。3 條都須跑滿 416 代且無 NaN |
| 每鏈 | λ=256、n_S=256、σ0=0.3、416 代、T_train=20000、冷啟動 θ=0、訓練 fitness 記分窗 T−5000 |
| 驗證 | EA1_SEL（82100000–82100127），每 50 代（8 點），T=1e5，記分窗 T−5000 |
| 訓練 seeds | EA1_TRAIN（82000000–82000999），排列種子 94100+run；optimizer seed 94000+run |
| 一次性 test | EA1_TEST（82400000–82401023），n=1024，ledger 各評估一次 |
| 評估 | astarc、graph、chunk 64、記分窗 T−5000、逐 seed 同機配對、單側 95% t 界（df=1023） |
| 凍結的比較對象 | a\*＝`orig_adapt0`（sel `astar.orig`，theta_id 開頭 `88ed926a…`）；C2 cfg 同 sel |
| 執行環境 | astarc；torch 2.6.0+cu126、numpy 2.2.6、scipy 1.15.3；manifest 記錄並每階段檢查 |

---

## 5. 算力估計（紅方 1：待 smoke 量測）

**目前沒有 astarc 的任何量測。** 學校機的數字（28.2 s/代、3.37 h/條）是另一張卡（RTX8000-12Q vGPU）的紀錄，**不得用來估 astarc**。下表留待 smoke 後填入；填入的是量到的數字，不是估計。

**smoke 量測方法（在正式 manifest 凍結**之前**執行，用 smoke 模式，輸出在獨立目錄；用 EA1_TRAIN／EA1_SEL 前幾個 seeds，永不碰 EA1_TEST；量完把數字摘要寫進實作說明後**清除 smoke 輸出**，才進入正式 manifest 凍結，§7）：**
1. **單 lane，正式大小**（λ=256、n_S=256、T_train=20000、C2、graph）：冷啟動鏈跑 **5 代**。記錄每代 wall 秒數；**第 1 代含建圖，另列，不計入平均**；t_gen1＝第 2–5 代平均；t_step_ms＝t_gen1/20000。
2. **3 lane 同時**，各跑 5 代：t_gen3（每 lane 第 2–5 代平均）。
3. **一次正式大小驗證**（EA1_SEL 128 seeds × 2 策略 × T=1e5，單 lane）：t_val1；與 3 lane 同時時的 t_val3（若可量）。
4. **test 型評估各一個**（chunk 64、graph，用 EA1_SEL 的 seeds，不用 test）：1 個攻擊者與 12 個攻擊者各一次，外推到 1024 seeds 的 t_test。
5. 同時記錄：`nvidia-smi dmon -s u` 的 GPU 利用率、`max_memory_reserved`、`top -H` 的 CPU 使用核數。

| 量測 | 單位 | astarc 值 |
|---|---|---|
| t_gen1（單 lane，訓練） | s/代 | 待量測 |
| t_gen3（3 lane，每 lane） | s/代 | 待量測 |
| 吞吐比 3·t_gen1/t_gen3 | — | 待量測（採用規則 §3.3：≥1.10 才採 3 lane） |
| t_val1／t_val3 | s | 待量測 |
| GPU 利用率（單／3 lane） | % | 待量測 |
| GPU 記憶體峰值（3 lane） | GB | 待量測 |
| CPU 使用核數（單／3 lane） | 核 | 待量測 |
| t_test（主 2 個評估／次要 2 個評估） | s | 待量測 |

**總時數公式（量測後填）：** 單鏈 ≈ 416·t_gen + 8·t_val。`lanes=1`：3 鏈依序＝3·(416·t_gen1 + 8·t_val1)；`lanes=3`：≈ 416·t_gen3 + 8·t_val3（3 鏈同時）。再加 SEL 重評（§2.4 的 15×2＋a\* 重評）、AS 漂移檢查（a\* 於 AS 兩個設定）、testcheck、test（四組評估）、緩衝 20%。

**寫死：量測後總時數若與任何預期不同，不影響任何判準、預算、seeds、ε。** 設計預算（λ、n_S、416 代、驗證頻率）已凍結，不隨耗時調整；耗時只影響「何時開跑、是否分次啟動、是否採 3 lane」。若量測結果顯示總時數長到你不能接受，是在 smoke 後、正式凍結前由**你**決定整個實驗跑或不跑，而不是縮減預算。

**共用機的影響（推論，沒有量測）：** astarc 是共用機，他人 GPU 程式的出現會使我們讓出（§3.3），可能拉長牆鐘時間；可用空間（`df -h /`）扣掉用量後須剩 ≥10G，開跑前查；VM 可能被學長關機，需 rsync 取回 checkpoint。單次軟上限沿用 E1 v2 的 18 h 與續跑碼 75。

---

## 6. 程式修改清單與測試計畫

**新增（不改任何既有檔案）：**

| 檔案 | 內容 |
|---|---|
| `docs/direction2/design/ea1.md` | 本設計（定稿後的正式版，commit 後算登記） |
| `scripts/ea1.py` | 階段：`audit`／`freeze`／`smoke`／`train`／`select`／`testcheck`／`test`／`report`；params、manifest（含 seeds 三塊、預算雜湊、輸入與受保護檔雜湊、環境欄位）、冷啟動鏈、perm 區塊函式、EA1_SEL 驗證、lane 驅動、選模凍結（`selection_frozen.json`＋`freeze_record.json`）、`once` ledger、決策樹、report |
| `scripts/ea1_launch.sh` | 改寫自 `scripts/e1_v2_launch.sh`：先跑 CPU 測試、GPU 占用檢查（astarc 規則）、環境釘選檢查、exit 75 自動續跑、test／testcheck 不重試 |
| `tests/direction2/test_ea1.py` | 見下 |
| `results/direction2/ea1/` | 輸出（manifest、seed_audit、chains、run_config、parallel_check、selection_frozen、freeze_record、testcheck、test/ledger、result、secondary、drift） |

**不改：** §1.4 列出的所有檔案（含 `arbitration/rl/*.py`）。E-A1 的 manifest 另存自己的 `input_sha256`，並把 E1 v2 的 `selection_frozen.json`／`attacker/orig_run0.json` 的雜湊納入。

**測試計畫（CPU 合成資料為主，不開 test seeds；比照 `tests/direction2/test_e1_v2.py`）：**

| 項目 | 驗證 |
|---|---|
| seed 稽核 | 三塊展開後 8608／與既有（含 E1 v2 五塊）12064／交集 0；故意撞 72300000、41000063、`+3·10^6` 必失敗；塊內無重複 |
| 階段與 seeds 錯配 | 訓練只能取 EA1_TRAIN、驗證與 testcheck 只能取 EA1_SEL、test 只能在 `freeze_record` 已 commit 後開；其他必拋錯。**testcheck 用的 seeds 必須是 EA1_SEL 前 64 個，取到 EA1_TEST 任一 seed 必失敗** |
| perm 區塊函式（紅方 7） | n_S∈[1,1000]、跨邊界無重複、可重現；**排列由 `random.Random(94100+run)` 產生，且與 `Random(9000+run)` 的排列不同（逐項比對至少一個 run）**；用 0–999 池與 `train.seed_block` 同規則逐項比對「索引規則」；optimizer seed＝94000+run 由 manifest 讀出 |
| 冷啟動鏈 | 起點 mean＝0；λ=256／σ0=0.3／416／驗證每 50／n=128／記分窗 T−5000 全部來自 manifest；續跑時預算或防守設定不符則拒絕 |
| 決策樹 | 合成 Δ：LB 恰等於 ε、UB 恰等於 ε、UB 恰等於 0、LB 恰等於 0、各節點（含「小幅提升」）各一例、σ=0；與 §2.2 檢定力表重現（nct） |
| 選模規則 | 平手取小編號；A_fresh 取 EA1_SEL 上 G_C2 最大者；A_C2′ 規則（平手取 a\*）；a\* theta 雜湊必須等於 `88ed926a…` |
| 原設定同型 Δ | F_orig 選取規則（EA1_SEL 上 G_orig 最大者，平手取小編號）；Δ_orig 計算與 A_fresh 路徑對稱 |
| ledger | started 無 raw 則停；completed 不可重跑；竄改 raw 失敗 |
| manifest／環境 | 改任一預算欄位後雜湊變、不符則拒絕；GPU 型號或 torch／numpy／scipy 版本與 manifest 不符則 GPU 階段拒絕開跑 |
| **平行逐位相同（CPU，§3.4）** | 5 項（依序 vs 執行緒並行、交錯注入、單 lane 失敗隔離、中斷續跑、seeds 與 RNG 序列相同） |
| 訓練後檢查（item 8） | 第 200 代 checkpoint 單 lane 重算與平行結果逐位比對；故意改一位元必失敗；不同時停下來問使用者 |
| 小型 GPU 驗收（astarc，train/SEL seeds，不碰 test） | §3.4 的 A＝B＝C 逐位相同；graph 無 fallback；t_step 與利用率校準 |
| E1 v2 不受影響 | 開工前後各對 §1.4 列出的檔案算 sha256 並對照 man `protected_sha256`；各跑一次 `tests/direction2/test_e1_v2.py`（CPU）與 `scripts/e1_v2.py report`（CPU，應仍 `matches result.json: True`；在 astarc 或本機皆可，記錄當時 numpy 版本） |

---

## 7. 凍結與 commit 順序（比照 E1 v2）

1. **設計 commit**：`docs/direction2/design/ea1.md`（本文定稿）。你確認後才 commit。
2. **實作 commit**：`scripts/ea1.py`、`scripts/ea1_launch.sh`、`tests/direction2/test_ea1.py`、實作說明（偏離揭露，仿 `e1-v2-impl-notes.md`）。CPU 測試通過、E1 v2 不受影響檢查通過後。
3. **astarc 環境建立與檢查**：建 venv（先過磁碟門檻：扣掉用量後須剩 ≥10G、GPU 無他人程式）、裝 torch 2.6.0+cu126、numpy 2.2.6、scipy 1.15.3；確認 `torch.cuda.get_device_capability()` 印出 `(7, 5)`；跑 CPU 測試。
4. **smoke 與平行驗收（正式 manifest 凍結之前）**（§5 量測方法、§3.4 GPU 驗收；smoke 模式、獨立輸出目錄，只用 EA1_TRAIN／EA1_SEL 的前幾個 seeds，永不開 test）：記下吞吐比與逐位相同結論，數字摘要寫進實作說明；**然後清除 smoke 輸出**。若總時數不能接受，在此停下來由你決定。
5. **manifest 凍結（S0）**：在 astarc 上 `ea1.py freeze` 寫 `results/direction2/ea1/manifest.json`＋`seed_audit.json`（三塊 seeds、預算、ε、δ、輸入與受保護檔雜湊、a\* 與 C2 cfg 來源雜湊、**環境欄位**）；`rsync` 取回本機 commit。此前不得碰任何新 seeds 的正式用途。
6. **正式平行驗收與 `lanes` 決定**：以正式 manifest 重跑 §3.4 GPU 驗收（EA1_TRAIN 前三代＋EA1_SEL 驗證），寫 `parallel_check.json`；依步驟 4 的吞吐比（門檻 ≥1.10）與逐位相同結果，寫 `run_config.json`，commit。
7. **訓練**：3 條鏈（`train`），每個 publish 點 `rsync` 取回並在本機 commit；不從別處推同一分支。3 條都須跑滿 416 代且無 NaN；任何一條無法完成就停下來問使用者。`lanes=3` 時，訓練後做 §3.4 的訓練後檢查（第 200 代）。
8. **選模凍結**：`select`：在 EA1_SEL 上選每鏈 checkpoint；同一次呼叫重評 `[誠實, a*, 3 個 checkpoint]`，據此選 A_fresh、A_C2′（§2.4）；F_orig、a\* 的原設定 SEL 重評；**機器漂移檢查（a\* 於 AS）也在此階段做**；寫 `selection_frozen.json`、`freeze_record.json`；commit。無論結果，test 照登記執行（沒有「不去 test」的分支，因為主判準不依賴 AS 式改善檢查）。
9. **testcheck**：EA1_SEL 前 64 seeds，graph＝eager 逐位相同（astarc 本機內）。
10. **test 一次**：4 組評估各一次，ledger；主判定 commit；次要結果 commit。
11. **report**（CPU）：從 raw 重算決策樹，在 astarc 與本機各一次，記錄 numpy／scipy 版本與差異（E1 v2 曾觀察到 numpy 版本不同使 t 值末位不同）。
12. **結果文件**：`docs/direction2/results/e1-v2.md` 頂端只加一行指向本實驗結果文件 `docs/direction2/results/ea1.md` 的註記（不改該檔任何數字）。注意：`docs/direction2/design/e1-v2.md` 在 E1 v2 manifest 的 `input_sha256` 內，**不可修改**；被修改的只有結果文件 `results/e1-v2.md`（不在 input／protected 內），並須在修改後確認 E1 v2 的 `load_manifest` 檢查仍通過。

---

## 8. 限制與不能宣稱的事

**限制：**
1. 只改了初始化與訓練設定，沒有改優化器或策略類別。新舊攻擊者都是 25 特徵、55 參數的公開資訊 MLP＋sep-CMA-ES；若兩者收斂到同一個局部最優附近，本實驗無法分辨「真的沒有更強攻擊」與「這個攻擊者類別的上限」（另一個實驗，不在本次範圍）。
2. 3 條鏈、1 組設定；鏈間 G 的散佈在原設定下約 0.001（§2.6 第 1 項），新鏈同量級；3 取 1 的選擇偏誤由全新 test 吸收，但無法估計「更多鏈會不會更強」。
3. 新鏈與 a\* 的預算同量級但不是同一條件下的 A/B（§2.3 最後一列）；偏離表已列出的差異（訓練窗、訓練池、驗證 n、機器）都可能影響結果，方向不一定可知。殘餘不對稱：(1) 主實驗三條鏈在 val 2000–2031（n=32）選模，新鏈在 EA1_SEL（n=128）選，選擇雜訊較小，對新鏈略有利；(2) 主實驗的中間 checkpoint 沒有存 theta，無法在 EA1_SEL 上重新選模；(3) 訓練窗、訓練池、機器皆不同。
4. 原設定一側未重訓（已定）：D′ 的原設定側仍是 a\*，D′ 是描述性更新；「原設定同型 Δ」只是基準。
5. 範圍同 E1 v2：M3C、r=0.5、p=0.1、三旋鈕的單一候選 C2、單輪。
6. 所有主判準數字來自 astarc 一台機器；跨機數字（D_new、12 條目、機器漂移檢查）只作描述，且 astarc 與學校不保證逐位相同。
7. 本實驗標為**探索後的確認性對照**：登記判準與門檻在看到新 seeds 前寫死，但 ε 的選擇受到 E1 v2 已觀察到的數字影響（已看過 12 個條目在 test 的 G 與 D）；論文須揭露。
8. 「simulate 只有三個衍生串流」來自讀程式與 grep（§1.3），不是執行期追蹤。
9. 平行化的逐位相同保證只在 astarc 上由 §3.4 測試證明（若採 3 lane）；沒有證明對其他機器成立。
10. 子標籤（更弱、小幅提升）未做多重比較校正（§2.2）。

**不能宣稱：**
- 「a\* 是 C2 下最強攻擊」或「C2 下已無更強攻擊」（無論哪個結果）。
- 「warm-start 不對稱造成的偏差已被消除」：只比較了一種從頭訓練設定。
- **「本實驗驗證了 C2 結論的穩健性」**：C2 一側不可能翻轉 E1 v2 的判定（§0）；它只是完整性聲明。
- 「同配方」「與原鏈相同的訓練」：必須用偏離表的說法（同 λ／n_S／σ0／代數，目標窗與驗證不同）。
- 把「持平或更弱」當作 C2 一側搜尋不弱的獨立證據（先驗預期就是 Δ<0，§2.6 第 1 項）。
- 把「3 條鏈」或「從頭訓練」泛指為檢定對象：主詞只能是 SEL 選出的 A_fresh。
- 「C2 無效」或「防守旋鈕空間內不存在修補」（範圍是 C2 一個候選）。
- 把次要結果（轉移、12 條目、D′、D_new、各鏈 Δ、機器漂移）當成判準，或與登記判準並列為「結果」。
- 以結果為由事後調整 ε、δ、鏈數、seeds 數、記分窗、lane 數的門檻；任何延伸須新登記、新 seeds。
- Nash、收斂、均衡；跨機制、跨 r、跨 p；真實機器人。
- 本實驗已解決 AS→test 差距（2.36 SE）：D_new 只提供第二個樣本（且跨機），原因仍未定。

---

## 9. 與 options.md 的差異（需要同步修正）

- E-A1 的鏈數「對原設定與 C2 各 3 條、共 6 條、240 代」→ 本稿「僅 C2 3 條、416 代、λ=256／n=256」，原設定一側用主實驗既有三條作基準；算力不再估計為「6–8 h」，改為 astarc 待 smoke 量測。
- 「隨機初始化」→「冷啟動 θ=0」。
- 執行機器由「學校機」→「astarc Quadro RTX 4000」；所有學校機、katago、學校規則相關說法刪除。
- 新 seeds：已預跑，EA1_TRAIN／SEL／TEST 三塊，交集 0（§1.3）。

## 10. 已定項目（定案值；凍結後不得改）

| # | 項目 | 定案值 |
|---|---|---|
| 1 | 執行機器 | astarc（Quadro RTX 4000 8GB, sm_75） |
| 2 | 門檻 ε | 0.001 |
| 3 | 原設定一側 | 不重訓（§2.8） |
| 4 | 訓練 seeds | 全新 EA1_TRAIN（82000000–82000999） |
| 5 | n_test | 1024 |
| 6 | 訓練 fitness 記分窗 | T−5000。理由：與評估窗一致，讓新攻擊者發揮最大，使「沒有更強」的結論不因訓練目標錯配而偏弱 |
| 7 | lanes 採用門檻 | 3 lane 吞吐比 ≥1.10 且逐位相同才採 3 lane，否則 1 lane；只登記 1 或 3 |
| 8 | 結果取回 | `rsync` 回本機後在本機 commit |
| 9 | astarc 磁碟 | 扣掉用量後須剩 ≥10G，否則不開跑 |
| 10 | 3 條鏈跑滿規則 | 416 代且無 NaN 才能進 select；無法完成就停下來問使用者，不縮鏈、不改判準 |
