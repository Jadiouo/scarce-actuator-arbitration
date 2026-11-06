# E-A1 設計檔勘誤與分支歷史重寫（2026-11-06）

`docs/direction2/design/e-a1.md` 原本有一行提到共用機上另外兩個帳號的名字。這些是第三方資訊，不該公開，所以把那段改成「其他帳號的程式」，並重寫了 `direction2-freeze` 從該檔第一次出現起的歷史。

**這是唯一的改動。** 新舊 commit 逐一比對，差異都只有這個檔的這一行；判準、門檻、seeds、選取規則、程式、結果都沒有變。

| 項目 | 值 |
|---|---|
| 設計檔 sha256（重寫前，manifest 的 `design_sha256`） | `ea5e90772fcdfc35f98c9c57ee1169b305430ba57fbc607be6131083c7ab7787` |
| 設計檔 sha256（重寫後） | `9ff931cfcd29061c9a640f4e31303b209aa94b445185984cb8b1a3d2fc2d12c6` |
| manifest 的 `code_git_sha`（重寫前的 commit） | `bebc08c`，對應新 commit 見下表 |

## 新舊 commit 對照

文件和紀錄裡提到的舊 commit，用這張表換算。

| 舊 | 新 | 主旨 |
|---|---|---|
| `c66305d` | `3b7deeb` | E-A1: pre-registered design (attacker-strength control on C2) |
| `bebc08c` | `8f4c972` | E-A1: implementation, tests, smoke summary (astarc) |
| `fe2e0fa` | `60a7d7f` | E-A1: S0 manifest freeze (astarc) |
| `7a29f3b` | `87205b7` | E-A1: paracheck (lanes=1; graph == eager on astarc) |
| `441b677` | `83c4cdf` | Worklog: framing A1 and E-A1 training in progress (10/20-10/24) |
| `d018b8e` | `5267ae3` | E-A1: chains and selection freeze (A_fresh = run0 gen300) |
| `a7b3f0c` | `d3d8224` | E-A1: testcheck passed |
| `c6425d0` | `3b6ad3c` | E-A1: primary result (not stronger, flat) |
| `9ee2156` | `599f9cb` | E-A1: secondary results (descriptive) |
| `2a8cae6` | `4855798` | E-A1: results document, post-hoc recheck script, report checks |

## 對重現的影響

- `results/direction2/ea1/manifest.json` 沒有改，所以仍記錄重寫前的雜湊和 commit。
- `scripts/ea1.py report` 會檢查這兩項，重寫後不再通過。
- 從原始資料重算請用 `python3 scripts/ea1_recheck.py`：它只讀 `results/`，不依賴這兩項。
