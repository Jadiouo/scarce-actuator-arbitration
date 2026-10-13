# 主實驗正式評估協定（2026-10-13，test 前登記）

使用者要求完成學校機器的主實驗 test 評估、配對信賴區間與複現判定。本次補齊訓練結束後尚缺的評估入口，不改動凍結的規格、策略、機制、預算、seeds 或判定規則。

1. 輸入固定為 run_plan.json 中 main_M3C_r0.5 與 main_M3C_r0.9，各 run 0、1、2。每個 run 按既有驗證紀錄選最佳 checkpoint，平手取較早者。主 run 在 validation 選，平手取較小 run_id，test 後不更換。
2. T=100000；T_score=T−L=97500；與訓練驗證相同的 float64 模擬器與 public 觀察，沒有尾端強制誠實。學校採單程序、MLP 批次 CUDA Graph；手寫策略批次沿用 eager。正式 val 逐一重算均值並對照訓練記錄，誠實批次的 util、regret、pen 也互相核對。
3. validation 評估 S_HW 36 個公開策略；主基準不含 H0。另外選含 H0 的 37 個候選、含 oracle 的 44 個候選。後者含四個 edge_oracle 與四個 quota_savemin_oracle；依規格，四個 quota 策略在 M3C 等價誠實空操作，明確映射為 honest adapter，沒有混入主要 S_HW。
4. 在 validation 完整選定 checkpoint、主 run、手寫策略，並寫下每個 run 的 MDE_G、MDE_D、MDE_D_plus_h0、各被選手寫策略的 MDE。全部 MDE 使用原有 metrics.mde。選取紀錄與 raw batch 都保存並附校驗；test 不重新計算 MDE。兩個 cell 的 validation 都先做完，再進 test。
5. test seeds 僅經 final_eval.evaluate 提供；同 cell/policy 只評一次。每個 cell 的三個 MLP 一次批次評估，共用同 seeds 的 honest 對照；手寫評估只包含 validation 選中的策略。先原子寫 raw snapshot，再登錄 ledger，崩潰後只從已存 raw 重建；部分批次完成也可續接。快取檢查 seeds、政策 SHA、設定、程式 SHA；不接受過期快取。final_eval 的 ledger 寫入改成原子替換，數值規則未變。
6. G_s=(U_policy−U_honest)/T_score；D_s=G_RL,s−G_HW,s。95% CI 是逐 seed 配對後 mean±t(31,0.975)·SD/sqrt(32)。主值達到 D_lo>0 且 D≥凍結 MDE_D 才通過 D 規則。複現 k 按規格只計三個 run 中 D_lo>0 的數量；不是把 96 個值當獨立樣本，也不要求另外兩個 run 各自通過 MDE。
7. 正式 J1–J8 判定只用 r=0.5、validation 選定主 run、S1 原登記二值閘門與 k。含 H0 判定附帶列出，不影響主判定。r=0.9 只作探索性、未經 S1 標定的描述。
8. 同時保存 honest regret／誤罰率、攻擊下 regret、每位 agent 的暫停比例；每個 run 與手寫基準均報告。輸出 main_eval/*__val.json、*__test.json、各 raw batch、final_eval_ledger.json 與 main.json。

已知揭露：r=0.5 run0 的權重與完整訓練曲線等同 S1 D2 的兩種大小，且該策略在 S1 已看過 test；主實驗 run0 不能當成相對於 S1 的新獨立證據。本次照原登記跑三個 run，不改 seed、不重選較有利的候選。S1 原規則判 3/3 通過，但事後 off-knob 的漏洞利用證據只有 2/3；D2 與 O5 的 detected 混入基礎優勢。正式 J 判定與這項限制會一起報告。

驗證方式：先在學校執行合成資料 CPU 測試，獨立驗算配對 CI、validation-only 選取、MDE 凍結、H0、ledger 續跑與快取拒絕；再以訓練 seeds 做短 GPU graph/eager 一致性檢查。CPU 合成測試用假的效用函式驗證 seed 路由；驗收不在正式 test seeds 上執行真實政策或模擬器。正式 test 入口及數值程式於 validation 前提交。

學校執行：`PYTHON=.venv/bin/python bash scripts/d2_main_eval_launch.sh`，放進 tmux；日誌與暫存都在專案內。每個 val/test 階段完成就提交並 push 到 direction2-freeze，不 force push；推送失敗時停在已保存的階段。
