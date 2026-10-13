# 盲設計漏洞集 2（M3C 機制變異，4 個）

說明：以下四個變異都只改 `frontier.py` 中 M3C 分支（`mech == "M3C"`）及其周邊 hook，彼此對應的機制環節不同，不重複：

| 編號 | 名稱 | 機制環節 | 利用所需的時序組合 |
|---|---|---|---|
| V1 | RestedAuditDiscount | 稽核抽樣機率（`o = win_eff & (Ua < p_el)`） | 連敗 G 輪之後的第一個勝出時刻 |
| V2 | SuspensionParole | 懲罰（`susp` / 暫停期的釋放規則） | 暫停期間持續「棄權式」回報，再回到說謊 |
| V3 | SlopeShrink | 殘差計算（`rh` → `resid`） | 只在自身型別 z 為正且高時，按 z 比例灌水 |
| V4 | SelfCalibratingSlope | 殘差基準（逐 agent 線上校準 `rh`）＋ flag 時重置 | 說謊幅度須沿著緩坡上升，且被暫停後重來 |

共同符號沿用 frontier.py：`t` 為輪次，`tau` 為稽核延遲，`idx` 為得標者，`win_eff` 為實際得標，`elig = t > susp`，`zv/zw` 為回報與實現值的稽核 z 分數，`resid = (zw - rh*zv)/sdh`，`Sn = max(0, S - resid - kref)`，`Sn > tol` 即 flag，flag 後 `S := 0`、`susp := t + L`。所有變異都不新增亂數抽取，也不改動 `Ua / Rn / Rx / E` 的消耗順序。

---

## V1

1. `名稱`：RestedAuditDiscount（歇夠了的得標者少被稽核）

2. `變異定義`：改動位置為「稽核指派」那一行（`o = win_eff & (Ua[s][sidx] < p_el)`）。新增狀態 `lose_run[B,K]`（浮點，初值 0）：agent k 在「具備資格（elig）的輪次中」連續沒得標的輪數。常數 `G = 5`（設計常數，非旋鈕）。

```
# 每輪稽核指派前
rested = lose_run.gather(1, idx[:, None])[:, 0] >= G        # 這個得標者此前已連敗 >= G 輪
p_eff  = p_el * torch.where(rested, 1.0 - s, 1.0)            # s 為旋鈕
o      = win_eff & (Ua[s_][sidx] < p_eff)                    # 同一個 Ua，不多抽
# 指派之後更新狀態
won   = one_hot(idx, K).bool() & win_eff[:, None]
lose_run = torch.where(won, 0, lose_run + elig.to(F64))      # 暫停中的輪次不累計（避免剛釋放就算「歇夠了」）
lose_run = torch.where(flag_this_round_agent_mask, 0, lose_run)  # 被 flag 的 agent 歸零
```
其餘（pend_a / pend_v / CUSUM 更新）不變。

3. `旋鈕`：`s`（稽核折扣）。恆等值 `s = 0`。s 越大越強（rested 狀態的稽核機率 p·(1−s)，s=1 時完全不被稽核）。範圍 [0,1]，連續。恆等性：`s=0` 時 `p_eff = p_el * 1.0` 與 `p_el` 逐位元相同，比較用的 `Ua` 抽取不變，`lose_run` 只被寫入、不被其他分支讀取。

4. `利用策略`（只用自己的得標紀錄與型別）：
```
n = 自己連續「具備資格卻沒得標」的輪數（每輪由「自己有沒有得標」更新；被暫停時不計，被暫停後歸零）
if eligible and n >= 5:
    回報一個中等幅度的灌水值（讓自己這輪剛好贏，例如 z+dz 使 v 略高於預期對手，不要灌到 1）
else:
    誠實回報
```
得標後 n 歸零，所以每次連敗 5 輪才灌水一次。

5. `預期得利`（推論）：誠實時約每輪有 (2/3)^5 ≈ 13% 的時間處於「歇夠了」狀態（假設三家對稱），灌水頻率上限約每輪 0.1 次，每次把一次落選變成得標，得利約正比於灌水頻率 × 自己的型別值。偵測側：被稽核的灌水每次使 S 增加約 d'−kref，而誠實得標被稽核時 S 平均下降約 kref，S 淨漂移 ≈ λ_lie·p(1−s)·(d'−kref) − λ_honest·p·kref（λ 為每輪頻率）。此漂移由正轉負的門檻在 s* = 1 − λ_honest·kref/(λ_lie·(d'−kref))，所以推論得利對 s 呈門檻式：s < s* 時被 flag 的成本吃掉得利（淨得利約 0 或負），s > s* 後得利近似隨 s 線性上升至 s=1 飽和；s=0 時得利為零（本來就是原機制）。

   `簡單策略為何不易利用`：固定灌水與少量持續灌水幾乎每輪得標，lose_run 幾乎永遠是 0，永遠拿不到折扣；只在決策時灌水、只在差一點就贏時灌水，會在得標頻繁的狀態下灌水，同樣拿不到折扣（連敗 5 輪的狀態只佔約一成，且與「差一點贏」不重合）；集中爆發式灌水期間幾乎持續得標，一樣不在 rested 狀態。要吃到折扣，必須把「灌水」精準放在連敗 G 輪之後的那一次，需要讀自己的得標史。

---

## V2

1. `名稱`：SuspensionParole（暫停期內「棄權」可提早假釋）

2. `變異定義`：改動位置為每輪 `elig = t > susp` 與 `pen += ~elig` 之後。新增狀態 `calm[B,K]`（浮點，初值 0），常數 `EPS_REP = 0.02`（設計常數）。

```
sus  = ~elig                                               # 目前被暫停的 agent
calm = torch.where(sus & (v <= EPS_REP), calm + 1.0, torch.zeros_like(calm))
parole = sus & (calm >= Q)                                 # Q 為旋鈕
susp = torch.where(parole, torch.full_like(susp, float(t)), susp)   # 下一輪起恢復資格 (t+1 > t)
calm = torch.where(parole, torch.zeros_like(calm), calm)
```
v 是機制本來就收到的回報（`vm` 之前的 v），暫停中的 agent 也有回報。放在稽核解析（`cs = t % M`）之前也無妨：解析用的 `elig_ag = t > susp` 在這輪仍為 False。

3. `旋鈕`：`Q`（假釋所需的連續棄權輪數）。恆等值 `Q = +inf`（`calm >= inf` 恆為 False，`susp` 與其他狀態從不被改寫）。Q 越小越強；整數，合理範圍 1 至 L（Q ≥ L 時沒有實際效果，因為自然釋放先發生；注意預設 `L=INF` 時任何有限 Q 都使暫停可被縮短）。恆等性：`Q=inf` 時所有 `torch.where` 都選回原值，且無亂數抽取。

4. `利用策略`（只用自己是否被暫停與自己的回報）：
```
if suspended:           # 自己看得到
    回報 v = 0           # 反正被暫停也得不到分配，棄權不花成本
else:
    照任一種灌水策略回報（例如固定 dz）
```
被 flag 後只需 Q 輪「棄權」就恢復資格，有效暫停長度由 L 縮為 min(L, Q)。

5. `預期得利`（推論）：每個 flag 週期被縮短的暫停長度為 (L−Q)+ 輪，得利約為 (L−Q)+ × 每輪灌水利益，且每個週期都重複；因此得利對 Q 單調遞減，Q ≥ L 時為零，Q=1 時最強（暫停幾乎失效，懲罰力道降到 1 輪）。Q=inf 時為零。附帶：灌水程度可以更大（反正被抓代價只剩 Q 輪），所以總得利可能超過線性估計。

   `簡單策略為何不易利用`：策略集裡的灌水函數不依賴「是否被暫停」，暫停期間仍持續回報高值（固定灌水）或回報誠實值（只在決策時灌水），而誠實值 u ≤ 0.02 的機率很小，所以不會觸發 calm 計數；集中爆發式灌水的 off 期間誠實回報也不是 0。必須在暫停狀態下刻意回報近 0，才能吃到假釋。

---

## V3

1. `名稱`：SlopeShrink（殘差基準的斜率被保守縮小）

2. `變異定義`：改動位置為 M3C 解析區，`rh = torch.where(onl, ..., r_as)` 之後、`sdh = ...` 之前。

```
rh  = rh * (1.0 - s)                         # s 為旋鈕；其餘公式照舊
sdh = sqrt((1 - rh**2).clamp(min=1e-18))
resid = (zw - rh*zv) / sdh
```
`sxy/sxx` 的累計與 online 的 rh 估計仍用未縮小的值（只縮小稽核使用的斜率）。

3. `旋鈕`：`s`（斜率縮小比例）。恆等值 `s = 0`（`rh*1.0` 逐位元不變）。s 越大漏洞越強，範圍 [0, 0.6]，連續。恆等性：`s=0` 時 `rh`、`sdh`、`resid` 逐位元不變，沒有新分支、無新亂數。

4. `利用策略`（只用自己的型別與回報）：
```
z = 自己型別對應的稽核 z 分數（誠實回報時會得到的 zv）
if z > 0:
    回報使稽核 z 分數等於 z / (1 - s) 的值        # 即按 z 比例放大，delta = s*z/(1-s)
else:
    誠實回報
```

5. `預期得利`（推論）：在 `rh=rho_true` 的假設下，對 `zv = z + δ` 的回報，`E[resid | z] = (s·rho·z − (1−s)·rho·δ)/sdh'`；取 `δ = s·z/(1−s)` 時平均殘差為 0，S 漂移為 −kref（完全不被偵測），而且還可以再多灌一點（讓平均殘差到 −kref·sdh' 以內）。得利約正比 δ ≈ s·z（小 s 時線性），受 v ≤ 1 的飽和限制；s=0 時 δ=0，得利為零。副作用（推論）：誠實的低 z 得標者殘差平均為正負對稱偏移，假陽性會隨 s 上升。

   `簡單策略為何不易利用`：固定灌水、少量持續灌水用的是與 z 無關的常數 δ，在 z ≤ 0 的輪次殘差平均 −(1−s)·rho·δ + s·rho·z 更負，偵測力反而比原機制強；只在決策時灌水與只在差一點就贏時灌水也沒有按 z 的正負與大小調整幅度；爆發式灌水沒有型別條件。只有「只在 z>0 時、且幅度與 z 成比例」才會被隱藏。

---

## V4

1. `名稱`：SelfCalibratingSlope（逐 agent 線上校準斜率，flag 時重置）

2. `變異定義`：改動位置為 M3C 解析區，在 `resid` 計算與 `cus` 更新之間。新增狀態 `rh_a[B,K]`（初值 `r_as[:, None]`）與旋鈕欄位 `eta`（每個 element 一個純量，預設 0）。

```
use_cal = eta > 0
rh_c  = torch.where(use_cal, rh_a.gather(1, agc[:, None])[:, 0], rh)    # 逐 agent 的校準斜率
sdh   = sqrt((1 - rh_c**2).clamp(min=1e-18))
resid = (zw - rh_c*zv) / sdh                                           # 其餘 CUSUM 流程不變
# 在 el_（flag）算出之後：
err    = zw - rh_c*zv
new_rh = (rh_a_agent + eta * zv * err / (1 + zv*zv)).clamp(0, 0.995)    # 正規化 LMS
new_rh = torch.where(el_, r_as, new_rh)                                 # flag 時重置為先驗
rh_a   = torch.where(ohag.bool() & (upd & use_cal)[:, None], new_rh[:, None].expand(B, K), rh_a)
```
`rh_a_agent = rh_a.gather(1, agc[:, None])[:, 0]`。

3. `旋鈕`：`eta`（校準步長）。恆等值 `eta = 0`（`use_cal` 為 False，`rh_c` 取原本的 `rh`，`rh_a` 從不被更新）。eta 越大越強，範圍 [0, 0.3]，連續。恆等性：`eta=0` 時 `rh_c = rh` 逐位元相同，`rh_a` 不被讀取也不被寫入，無新亂數。

4. `利用策略`（只用自己的型別、回報、是否被暫停）：
```
t_rel = 最近一次「被暫停後恢復」的輪次（沒被暫停過就是 0）
dz_t  = min(dz_max, kappa * (t - t_rel))        # 灌水幅度沿緩坡上升
回報 ndtr(z + dz_t)                              # 被暫停（flag）後，t_rel 重設，從 0 重新爬
```
kappa 要小到讓校準追得上（kappa 約與 eta 成正比），dz_max 約取 4 個 z 單位（讓回報趨近 1）。

5. `預期得利`（推論）：穩態下若 dz 固定，NLMS 收斂到 `rh ≈ rho/(1+dz²)`，平均殘差 `≈ −rh·dz/sdh(rh)`，S 漂移 `= −resid − kref`；在 rho=0.9、kref=0.5 時，dz ≳ 4 漂移轉為負，完全不被 flag（近似，忽略 v≤1 的截斷與得標選擇偏誤）。爬坡需時約 dz_max/kappa ∝ 1/eta；得利約等於（總時間 − 爬坡時間）× 每輪利益，故對 eta 單調上升並飽和，eta 小到爬坡超過整個 T 時得利約為 0；eta=0 時得利為零。

   `簡單策略為何不易利用`：固定灌水與集中爆發式的幅度一開始就很大，校準來不及追上，S 在第一兩次稽核就越過 tol 而被 flag，且 flag 時 `rh_a` 被重置，所以永遠累積不起校準；少量持續灌水的幅度小，本來就在 kref 以內，沒有額外增益；只在決策時灌水與只在差一點就贏時灌水的灌水頻率低，`rh_a` 學不動。必須讓幅度「緩慢爬升」且被 flag 後歸零重來，才能讓基準被帶偏。
