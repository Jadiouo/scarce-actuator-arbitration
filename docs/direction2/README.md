# 方向二：學習型對手的正式評估

問題是：在固定的 M3C 稽核機制下，只看公開資訊的小型神經網路，能否取得手寫策略族沒有找到的額外利益？r=0.5 是主要設定；r=0.9 依原定規格只作探索性描述。

建議閱讀順序：

先看[本階段結論與收尾](results/conclusion.md)：獨立環境 holdout 的確認結果、哪些問題已完成，以及下一個研究問題的邊界。[完整 holdout 結果](results/independent-holdout.md)與[執行前協定](design/independent-holdout.md)保留這次額外確認的判準與稽核。

1. [正式主實驗結果](results/main.md)：六個 run 的 test、配對 95% CI、凍結 MDE 與複現判定。
2. [主實驗後診斷](results/postmain-diagnostics.md)：截斷偏差、固定策略的特徵消融與公開報告行為；屬 validation 上的事後分析。
3. [數字稽核](results/number-audit.md)：從原始逐 seed 效用重算；[評估協定](design/main-evaluation.md)說明事前選取和判定規則。
4. [S1 標定結果與限制](results/s1.md)：原登記閘門通過，但事後 off-knob 診斷顯示 D2／O5 有基礎優勢混淆，不能無條件宣稱找回族外漏洞 3/3。
5. [10/13 訓練完成查核](history/main-status-2026-10-13.md)：保存正式 test 前的狀態與驗證結果。
6. [凍結規格](../direction2-spec.md)與[凍結登記](../direction2-freeze.md)：研究範圍和判定規則的來源。

| 文件類型 | 內容 |
|---|---|
| 設計 | [測試對照表](design/test-map.md)、[S1 評估實作](design/s1-evaluation.md)、[加速計畫](design/speedup.md) |
| 漏洞設計 | [盲設計 prompt](design/blind-designer-prompt.md)、[有效設計](design/blind-vulns.md)、[交付格式](design/vuln-format.md) |
| 審查 | [正式主實驗驗證](reviews/main-evaluation.md)、[S1 評估實作審查](reviews/s1-evaluation.md)、[S1 結果審查](reviews/s1-results.md)、[加速審查](reviews/speedup.md) |
| 歷史 | [研究與部署紀錄](history/direction2-notes.md) |
| 封存 | [受汙染、不計入統計的第一份盲設計](archive/blind-vulns-contaminated.md) |

原始產物位於 [results/direction2/](../../results/direction2/)。正式主實驗逐 seed 原始資料和選取紀錄位於 `results/direction2/main_eval/`，`main.json` 是彙整入口。機制參數、策略權重和 test seeds 不因整理文件而改動。
