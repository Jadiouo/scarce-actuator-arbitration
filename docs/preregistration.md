# 預先登記（正式實驗）

寫於 2026-10-02，在最終確認實驗（final runs）開跑之前。此前的實驗都是 exploratory。在這份文件之後才新增的分析，一律標示 **exploratory**。

## 主張與否證條件
- **H1（資訊價值小）**：N=3、shock 模型下，VoI = J*_step(full) − J*(age) 的絕對值，以 K 外推後報告，附 bootstrap 95% CI。
  - 主張的形式：「在 r=0.004、shock_frac=0.5 時，VoI < 0.02」。
  - 否證：CI 下界 ≥ 0.02。
- **H2（天真信任回報的代價大）**：fused_index 在均衡下的團隊 health 低於 learned_index，以 paired 95% CI 判定。
  - 否證：CI 包含 0。
- **H3（到達時稽核失效）**：在擴大後的策略族下，audit_index 的均衡損失 > 0，以 paired CI 判定。
  - 否證：N=3 與 N=8 的 CI 都包含 0。
- **H4（派遣時稽核在已測策略族內有效）**：audit_disp 的均衡損失 CI 包含 0，且 max_gain ≤ 2·eps。
  - 否證：任一 N 的損失 CI 下界 > 0，或 max_gain > 2·eps。
  - 結果若被否證，就撤回 H4 這條主張。

## 指標
- **主要指標**：長期平均 team mean health。
- **次要指標**：死亡 robot 比例（frac_dead_robot_steps）、每步最小 health 的 p05。
- 主要和次要指標都報告，不論方向。

## 網格（整張報告，不挑點）
- shock_frac ∈ {0.25, 0.5, 0.75}
- N ∈ {3, 4, 8}（DP 只做 N ≤ 4）
- 負載 r ∈ {0.002, 0.004, 0.008}
- spread = 0.6
- σ = 0.15

## 程序
- seeds 數量：模擬 32 個，DP 區域圖 16 個。
- penalty 參數調整用 seeds 0–15，評估用 holdout seeds 1000–1031。
- 丟掉前 500 步暫態。
- gap 同時報中位數、平均值 ± t 95% CI，以及每 seed 分布。
- 均衡同時報收斂率、cycle 率、max_gain，以及 eps ∈ {0.001, 0.002, 0.005} 的敏感度。
- 用語規則：
  - 「均衡」只用在收斂的情況。未收斂的情況稱為「近似」。
  - 「擋得住」一律寫成「在已測策略族內」。
