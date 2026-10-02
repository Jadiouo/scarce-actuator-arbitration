# 決策紀錄（commander）

目標：依 A–F 規劃把專案深化成作品級研究。分工：commander 只規劃／派工／審查，subagent 一律 sonnet。

## 已定案
- 2026-10-01 `oracle` 改名 `greedy_true`（它是短視貪婪，不是最優）。
- 2026-10-01 白方：確定性線性衰減＋固定速率 → 私有資訊前提為空（可由距上次服務時間推出 h）。採納：加 `damage="shock"`（隱藏 Poisson 衝擊）、持續性回報雜訊（ar1/bias）。
- 2026-10-01 C（價格＝最優 index 的移動項）降格為「可證偽的數值檢驗」：比對 DP shadow price 與模擬 λ*。文獻無解析 λ*（Naor 1969 是同一類外部性通行費）。
- 2026-10-01 Whittle 不做（移動成本耦合 arms，分解不成立；文獻無此結構的 index）。A 改為 N=3 精確 DP + rollout。
- 2026-10-01 D、F 保留但縮小（白方建議砍；使用者目標要求 A–F 完成）：D＝N scaling + 2 actuators；F＝網頁 demo + 論文式寫作；不做 ROS/Gazebo。
- 2026-10-01 紅方（攻擊 E）成立項：
  - priced 的增益主要是抑制 thrash；對 no-thrash 基準只在過載時 +~0.025，低於容量無效益 → README 定價段落要重寫。
  - bug：callers 為空時丟掉承諾（每次執行 91.7 次）；`none_random` rng 未配對（要直接修）；`strategic` ≡ nearest(θ=0.05)。
  - 不看回報的 learned 派遣端已勝過 greedy_true 與 honest；成本感知 index 勝 greedy 但仍輸 round-robin。
  - 所有引用核實正確。

- 2026-10-01 第 2 階段（v2 模型）結果採納：距離感知後，回報相對 learned 信念無正價值；priced 對 no-thrash 基準 ≈ 0；只有距離感知策略在過載時勝 `none`。舊「資訊無價值」主要源自忽略距離。
- 2026-10-01 研究主軸改為：用精確 DP 畫出「資訊在哪個區域有價值」（VoI 對 shock_frac、spread、SERVICE/移動比），並量化 `none` 的 optimality gap。

- 2026-10-02 A 階段（N=3 精確 DP）採納（待紅方）：none 低負載 gap 0.01–0.02（接近最優），過載／高異質性／長服務才遠離；VoI 幾乎只隨 shock_frac 上升（0.5→~0.01）。honest 的 gap（0.042）> 全部 VoI（0.011）→ 回報應融合進 age 信念而非取代；B 階段先做 fused_index。

- 2026-10-02 使用者指示：所有實驗一律用 GPU（本機 RTX 5070 Ti，torch cu130），不做額外測試；只保留一次性 GPU vs CPU 逐 seed 數值一致檢查。B 階段改為：CPU 只寫參考程式碼不跑，完整 game_sweep 與 dp_sweep（K=160、N=4）在 GPU 上跑。不用雲端 GPU（需付費）。

- 2026-10-02 紅方（攻擊 A）成立：(1) dp.py 的 J* 只是「不可改道」類別最優，shock 下每步重選目標 +0.012（8/8 seeds）→ 改做逐步決策 MDP（位置×health，GPU）作為真上界；(2) gap 偏斜，改報中位數／分布、丟暫態、用同 seed 模擬 dp 比較；(3) K=60 使 VoI 高估 5–15%，做 K 外推；(4) N=3 結論不可推到 N=8，gap 未隨 N 縮小 → 主張限定 N=3–4。B 階段初步：fused_index 回收約 58% VoI（CPU 煙測，待 GPU 正式跑）。

- 2026-10-02 GPU 版（arbitration/gpu/）完成：sim 與 DP 跟 CPU 逐 seed 一致（~1e-16）。
- 2026-10-02 B 階段（GPU，results/game.json）初步主結果：天真融合的策略性損失 0.078（N=3）／0.13（N=8），約為 VoI 的 8–13 倍；事後稽核（扣除估計偏差）損失 ≈0 且保留資訊好處。待紅方：策略空間只有常數 b，「挑時機灌水」可能繞過到達時稽核；N=8 稽核類收斂率 3–6%。

- 2026-10-02 紅方（攻擊 B）成立：到達時稽核擋不住挑時機灌水（N=8 均衡損失 0.011，單方增益 ≫ eps）；派遣時稽核可擋住（≈0）。主張改為「稽核時機決定成敗」。補：擴大策略空間（常數／時機／距離／τ 依賴）、eps 敏感度、holdout seeds、winner's curse 量測；主數字改報絕對差，不用 VoI 比值。

- 2026-10-02 白方中期檢查採納：預先登記（docs/preregistration.md）；主數字用絕對值＋CI 不用倍數；加尾部指標；策略族加共謀、反向時機、對閾值自適應；C 改為一張圖檢驗「定價＝抑制 thrash」並和策略性損失比大小；D 改為 N∈{3,4,8,16} 趨勢，2 actuators 只做 exploratory。
- 推論（待驗證）：在不可改道＋fused 只用當下 claim 時，派遣時稽核＝稽核所有被採用的 claim。

- 2026-10-02 B v2 中期（N=3 r=.004 shock .5 holdout）：H3 成立（audit_index 全策略損失 0.024*，依距離灌水最有效）；H4 損失 ≈0.003（disp）／≈0（disp+penalty）但 max_gain>2eps → 依預登記判為「未獲確認」，不事後改規則；split-sample 列 exploratory。懲罰誤觸發率 35–40%（限制）。網格縮減：shock 0.25/0.75 用縮減策略集（偏離預登記，需標記）；B 跳過 N=4。

- 2026-10-02 逐步 DP 完成（results/dp_step.json）：VoI_step（shock .5，N=3）= 0.012/0.019/0.023/0.018（r=.002–.016），約 2× semi；途中改道本身值 ~0.01；N=4 同量級。H1：r=.004 點估計 0.019，貼近預登記門檻 0.02，未被否證。age-only 下改道無價值（重載小 cap 數值確認，其他為推論）。
- 2026-10-02 派 C+D（GPU，exploratory）：C＝priced vs no-thrash 基準＋DP 隱含距離匯率 vs λ*；D＝N∈{3,4,8,16} 趨勢＋2 actuators 簡化版。

- 2026-10-02 B v2 最終（results/game.json "game_v2"）：H2 成立（fused 均衡 −0.24 N=3／−0.40 N=8）；H3 成立（audit_index 損失 0.024／0.042）；H4 依預登記未獲確認（disp 損失 0.003–0.006，約為到達稽核的 1/7–1/10，N=8 CI>0 且幾乎不收斂）。主張改為「派遣時稽核把策略性損失降一個數量級，但未消除」。偏離預登記：策略集縮減、部分格點未跑（N8 r.004/.008、N8 f.25/.75）、調參只在一格。
- 2026-10-02 派 F 網頁 demo（docs/demo/，不放具體數字）。

- 2026-10-02 demo 完成（docs/demo/）：情境③改依距離>10；所有情境改用中位數附近的代表性 seed（原 seed 12 誇大團隊差 ~18×，已撤換），並加註單次執行雜訊大。互動經 headless Chrome 驗證，console error 0。

- 2026-10-02 C+D 完成。修正先前結論：λ* 調參（seeds 0–15 選、holdout 評估）後 priced 比 no-thrash 高 0.015–0.09（v2），舊「≈0」來自未調參 λ；但 priced 仍輸距離感知 index，且與 DP 決策一致率 0.51 → λ·d 是粗糙距離過濾，C 假設（最優策略的去中心化實作）被否證。D：N≥8 round-robin 勝所有資訊型啟發式；VoI 代理只有 DP 的 15–45%；fused 損失隨 N 增大（0.165→0.41）；2 actuators 無主效應。主張清單見 docs/claims.md。

- 2026-10-02 紅方總攻擊全部採納（docs/claims.md 已改寫為最終版）：H1 不主張（與門檻無法區分、service=1 被否證、估計器偏差 ~0.002）；H4 依預登記否證並撤回（N=3 CI 下界亦 >0），只主張「降約一個數量級但未消除」；max_gain 樣本內被 winner's curse 墊高需揭露；fig11 用舊數字（critical，須重畫）；各損失數字需標 cell；C7 的 0.51 一致率為假象。

- 2026-10-02 最終紅方：小修後可交付（15 個關鍵數字皆吻合）。已改：H1 在 service=1 改為「無法區分」；預登記改稱 pre-specified internal protocol，並揭露寫於探索之後；派遣時稽核加上「已測策略族＋有限搜尋、比值偏樂觀、5/10 CI 含 0」。A–F 完成。

## 待決
- commit 與 tag（需使用者決定）；GitHub Pages 開啟；公開前隱私稽核。
- v2 模型下回報是否有正價值（第 2 階段結果）。
- 論文主張（thesis）在 A/B 結果出來後定。
