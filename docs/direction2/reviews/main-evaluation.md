# 正式主實驗驗證紀錄（2026-10-13）

正式評估與全部數值驗算在學校機器完成，程序 exit 0。主要結果為 J3；結果解讀及 S1／test 暴露限制見[主實驗報告](../results/main.md)。本紀錄驗證計算與交付，不把統計複現延伸成機制安全或最佳回應的證明。

| 檢查 | 結果 |
|---|---|
| 正式評估前 CPU 專用測試 | 23 passed、2 deselected；合成資料的選取、配對 CI、MDE 凍結與 ledger 續跑 |
| GPU graph／eager 一致性 | 1 passed、6 deselected；只使用訓練 seeds 的短批次 |
| 方向二完整 CPU 回歸 | 245 passed、1 skipped、78 deselected；209.53 秒 |
| 獨立數字稽核 | 112 項 CI／MDE 與原始逐 seed 效用一致；另核對選取、校驗碼、k 與 J3 |
| test ledger | 原有 S1 32 筆逐項未變；主實驗新增 10 筆，合計 42 |
| 凍結文件 | `direction2-spec.md`、`direction2-freeze.md` 內容與雜湊未變；漏洞交付格式搬移後亦保持原始雜湊 |
| 文件連結 | README、部署手冊及全部 docs Markdown 的相對連結目標均存在 |

評估數值程式於 `710367e` 提交後才開始 validation。每個階段保存原始資料並備份；全部主實驗結果於 `af53ce4` 組裝完成。完整 CPU 回歸使用包含文件整理的 `fa2d78a`；後續只有報告生成與文字補充，沒有修改模擬器、訓練、評估數值程式或測試資料。

唯一 skipped 的 CPU 測試為 `test_eager_path_unchanged_by_refactor_cpu`：其既有 golden digest 指定原始 PyTorch 版本，與學校的 2.6.0+cu126 不同。沒有修改 skip 規則，也沒有略過正式評估的 GPU 一致性檢查。

第一次擴大 CPU 回歸有 7 項環境失敗：6 項因 `--test-mode` 要求假 repo 位於被測 repo 外，另 1 項因 shell 子程序用了缺少 NumPy 的系統 Python。重跑使用暫時的 `.claude/worktrees/main-eval-validation`，測試資料仍放在專案內的 `.scratch/`、位於該 worktree 外；PATH 指向既有 venv。所有保護規則和測試斷言均保留。驗證 worktree 已移除。

重建數字報告只需 `.venv/bin/python scripts/report_d2_main.py`，使用已保存快照，在 CPU 執行，不重新評估政策。完整 CPU 指令為 `python -m pytest tests/direction2 -m 'not gpu' -q --basetemp=<專案內、被測 worktree 外的目錄>`；使用既有 venv，並將 TMPDIR 留在專案內。

機器可讀的 commit、環境版本與 SHA-256：[main_validation.json](../../../results/direction2/main_validation.json)。逐項數字：[number-audit.md](../results/number-audit.md)。
