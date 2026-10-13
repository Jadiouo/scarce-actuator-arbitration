# E1 第一批：防守調參與一次攻擊再適應（2026-10-13）

使用者授權：先把可長跑的訓練部署到學校，等用量 reset 後再整理研究。這批是探索性 pilot，產生候選、checkpoint、訓練曲線與 validation 交叉評估；不改既有研究結論、不做正式 test，也不重用先前 independent holdout 做選擇。

## 問題與固定預算

1. 保留 M3C 規則、r=0.5、float64、公開資訊 55 參數攻擊模型。防守方用 full CMA-ES 調 h∈[4,8]、κ_ref∈[0.35,0.65]、L∈[1000,5000]；p 固定 0.1，ad_h 固定無限大。讓不同防守方使用相同稽核頻率，這批不搜 p 或 M4。
2. 兩個防守訓練目標：HW（36 個公開手寫策略）；pooled（同一手寫族＋三個已凍結主實驗模型）。各做兩個 optimizer seeds，共四條訓練；每條 λ=8、12 代。從原設定 (h=6,κ=0.5,L=2500) 起始；每代使用相同 16 個 train seeds 作候選間 CRN，代間按既有 seed_block 輪換。
3. 每個候選都量測完整 frozen pool，但 HW 訓練與選取只看手寫欄。先滿足同 seeds 下的誠實約束：R/T 不比原設定增加超過 0.002、平均不合格比例不增加超過 0.01；可行候選再最小化 max(0,max_policy mean_seed G)。不可行候選排在所有可行候選之後，按標準化約束違反量排序。
4. 每三代取該代 training fitness 最佳者做 validation，並保留原設定為 fallback。每個目標從兩條 run 的可行 validation 候選中選最小 worst G；若沒有較好的可行者就選原設定，明確記錄沒有改善，仍保存失敗候選。
5. 原設定與選出的兩個防守方，分別做兩條攻擊再適應：從原 run0／run1 的權重 warm start，sep-CMA-ES、σ=0.15、λ=64、n_S=32、120 代，每 20 代驗證 optimizer mean，含初始模型比較，validation 選 checkpoint。總計六條，預算相同。這測 warm-start 適應，不宣稱從零訓練成功率。
6. 最後把六個再適應攻擊 checkpoint 和三個原攻擊 checkpoint，交叉放到三種防守設定量測；也保留手寫族交叉表。只做一次 defender→attacker 最佳回應，不做循環演化或宣稱收斂。

## 尺與否證條件

- train seeds 仍為 0–999；validation 改用 61000000–61000031，與舊 splits、41000000 系列 holdout 及其 generator offsets 分離。所有程式入口拒絕混入舊 test／holdout。新 validation 可供選模，因此最後交叉表不是新的確認性 test。
- T_train=20000；T_val=100000。不同 L 使用**共同記分窗**：T_score=T−5000，即 train 15000／val 95000。以相同觀測長度比較，避免 L 改變分母與尾端範圍；因此不直接與舊 97500 記分窗的絕對值混表。
- G 必須減掉「同防守設定」的 honest 效用；誠實服務約束則與「原設定、同 seeds、同記分窗」配對比較。honest 每位 agent 的不合格比例、regret 與完整效用快照均保留。
- 重啟／續跑必須核對 manifest、程式與原模型雜湊；候選 cache 綁定設定、政策、seeds、T 與記分窗。每個候選完成即存檔；每代保存 CMA 狀態與 RNG；每代及攻擊每 20 代備份 GitHub。
- 沒有可行且改善的 validation 候選，記為這批防守搜尋未改善。若再適應／交叉攻擊讓原本的 G 降幅縮到 <0.005，記為修補效果薄弱的探索性訊號；正式的配對 CI、效應門檻與獨立測試需在後續分析時另行凍結，這批不提前宣稱修補成功。
- 不以此套件聲稱 S1 已重新標定。改機制後的 planted-exploit 檢測力未檢驗。

## 開工前設計檢查

依原 §18.3 做攻防兩個角度檢查；由同一執行者完成，並非獨立紅白隊審查。

| 攻擊角度指出的問題 | 這批的處理／剩餘限制 |
|---|---|
| 把所有人停權也能壓低攻擊收益 | 誠實 regret 與不合格比例雙約束，報告逐 agent 結果；pilot 約束按均值，不能宣稱確認性非劣性 |
| 增加稽核成本得到表面改善 | p 固定 0.1 |
| 只對舊模型修補，對手適應後失效 | 自動排入六條公平預算的攻擊再適應，以及跨防守交叉表 |
| 變 L 也改變記分分母 | 所有候選 T_score 固定為 T−5000 |
| 看完 holdout 再選防守 | 使用另外的 validation；舊 test、holdout 不進訓練或選取 |
| 單次交替不等於共同演化收斂 | 明確只稱第一批 pilot；不宣稱 Nash、最強攻擊或收斂 |
| 不同攻擊 warm start 造成比較不公平 | 三個防守設定使用相同兩個初始權重、optimizer seeds、train blocks 與預算 |

## 長跑與交接

學校單 GPU、tmux `e1-pilot`；不使用本機 GPU。預估 8–14 小時，短測後把實測估計寫入 `results/direction2/e1_pilot/preflight.json`。完成既定工作即停止；單次啟動軟上限 18 小時，在下一個保存點停止，外層 19 小時只作卡死保險。遇技術錯誤最多自動重試三次，從已保存的 candidate／optimizer state 繼續；不跳過失敗工作。

入口：`bash scripts/e1_pilot_launch.sh`。查看 `results/direction2/e1_pilot/status.json`、`selection.json`、`defender/`、`attacker/`、`cross_validation.json`。日誌在 `results/direction2/runlogs/e1_pilot_20261013.log`；manifest 與每階段結果自動備份到 `direction2-freeze`。重置用量後先讀 status 和本文件，接著分析交叉表；這次不安排正式 holdout。
