# Direction 2 凍結紀錄（REQ-S1-13、REQ-S1-14、T-42）

- 凍結 commit：`a72d3d731a450fe4724f0fa118943e25c39b7577`（分支 direction2-freeze，本機，未 push；本檔為其後的單獨 commit）
- 時間：2026-10-07T14:18:04+08:00
- 規格：docs/direction2-spec.md 版本 v1.3，sha256 `b0963760b23f92a7ac3438f7b6c428bdcaf6920063c3650690c338cee4eb2d2d`
- 漏洞套件：results/direction2/vuln_suite.json sha256 `1ec74ae331e59a3d6692055d5939e3eb59373eb1d4929a9b214cdde9014fada0`
  （vuln_suite_v12.json：`f338eddc255a48005b308db33e4e7a5e068a588f2979f1be080b5afc9a6b5e89`）
- 實測：results/direction2_meas.json sha256 `edb88a5aefd6a51642643eb237f3d3f3a01f9a494f652d9179652a137a170046`
- 凍結項目（S1-13）：
  - 特徵表：規格 §4（以 spec sha256 與 arbitration/rl/obs.py sha256 `27e84accc1824da355975a8c6c21224e340ac1d5b4e22476d1a2674d65b2099e` 涵蓋）
  - k = 8（arbitration/rl/obs_reconstruct.py `_K = 8`）
  - P_MAX = 64（arbitration/rl/policy.py）；P_DIAG = 300
  - 預算規則：規格 §5／§5.5（λ=256、T_train=20000、n_S=64、300 世代；arbitration/rl/budget.py sha256 `bf195bab453a03f20a3a8473eaacc12f4bc25bed07485822aac01fd9282765a0`）
- 程式：arbitration/rl/*.py 與 scripts/run_d2_*.py 合併 sha256 `2bf0ca7a1a6c05cdc600524863097fa3577f7e7059f2dac40f253861aa71fce4`
- 測試：tests/direction2/*.py 合併 sha256 `67e4d897154b77959f1c153bae8d5375fb7d090a32edbe3c1ea5b56587c77677`；CPU 測試（-m "not gpu"）98 passed、31 deselected
- 注意：規格 S1-14 原訂另立 docs/direction2-freeze.md（含 MDE_D,plan、G_s、暖冷啟動登記等）；本檔為 pre-training 的最小憑證，上述尚未決定項目須於 S1 訓練前補登。
