# designer_1 受汙染證據

來源：docs 派工單與 subagent transcript（a7396e44a26243ddf.output）抽出的工具呼叫，見 tool_calls.jsonl（共 5 筆：Bash 1、Read 2、Write 1、SubagentHandback 1）。

## 實際讀取的範圍

| 檔案 | 讀取 | 評估 |
|---|---|---|
| arbitration/gpu/frontier.py | Read offset=55 limit=175（另有一次 grep） | 允許 |
| docs/direction2-spec.md | Read offset=295 limit=22，回傳行 295–316（當時版本的行號，含 8.1 標題到 REQ-S1-07 的 O4 項） | 超出允許範圍 |
| arbitration/rl/、tests/、docs/direction2-notes.md、規格 §4、§5 | 無讀取紀錄 | 合規 |

## 是否讀到含 k=8 的段落：是

回傳內容第 315 行（REQ-S1-07 的 B2）原文含「公開版歷史窗 k=8、EMA 時間尺度約 100 輪」，即被派工單禁止的特徵細節（k=8、EMA、歷史窗）。經逐一檢查 tool_result，只有該次規格 Read 的回傳含 k=8 / EMA / 歷史窗。

## 同時讀到的其他不該知道的內容

- 盲區漏洞清單與名稱：B1、B2、O4、O5（第 299 行），以及 B1、B2、O4 的描述（第 314–316 行）。
- 自動算子 O1、O2、O3、O6 的具體定義（第 309–312 行）。
- 策略評估相關資訊：手寫策略集 S_HW（32 個）、「只使用公開版資訊」的限制（第 302–303 行）。

## 結論

規格 §8.1 與 §8.2 在當時混在同一段 8.x 內，要求「只讀 §8.1」的 Read 範圍一路延伸到 §8.2 的盲區項目。designer_1 受汙染：已確認讀到 k=8、EMA、歷史窗以及盲區與其他算子的內容。其產出 docs/direction2-blind-vulns.md 的開頭自述亦承認看過 O1/O2/O3/O6/O4 定義並刻意避開，會偏離盲設計。需以 docs/direction2-vuln-format.md 重做。
