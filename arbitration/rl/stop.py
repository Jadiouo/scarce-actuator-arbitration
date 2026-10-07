"""REQ-STOP-*: outcome table J1-J8, wording and the REQ-STOP-06 disclosure list (spec v1.3)."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_COMMON_LIMIT = ("機制參數沒有針對 RL 重新調參，結論僅限此固定配置（M3C：p=0.1、h=6、L=2500、kref=0.5、r=0.5／0.9）；"
                 "觀察資訊集為公開版，特徵為手工充分統計量的近似，遺漏的特徵會漏掉依賴更長歷史或絕對時間的漏洞。")
_FEATURE_LIMIT = "結論僅限此特徵類別（公開資訊、手工特徵 k=8）。"

_LABELS = {
    "J1": "(iii-c) 候選漏洞，未經檢測力背書",
    "J2": "(iii) 學不起來／標定失敗",
    "J3": "(i) 找到（完全複現）",
    "J4": "(i-b) 找到（部分複現 2/3）",
    "J5": "(iv) 未複現（只有 1/3）",
    "J6": "(v) 顯著但小於 MDE_D（描述性）",
    "J7": "(ii) 在此策略類別、預算、MDE 下未找到",
    "J8": "(vi) 優化失敗類：學習型對手顯著弱於最佳手寫",
}
_INCOMPLETE_LABEL = "未完成（S1 已通過，主實驗尚未完成）"


@dataclass
class Outcome:
    """row: 'J1'..'J8'; label: spec result label; incomplete: True iff S1 passed but D does not exist (note N1)."""
    row: str
    label: str
    incomplete: bool = False


def classify(S1: bool, D: Optional[float], D_lo: Optional[float], D_hi: Optional[float], MDE_D: Optional[float],
             k: int) -> Outcome:
    """REQ-STOP-03: exhaustive, mutually exclusive J1..J8.  D=None (main experiment unrun) = 'D not significant'."""
    exists = D is not None and D_lo is not None and D_hi is not None
    if not exists:
        if S1:                                              # N1: J7 row, flagged incomplete
            return Outcome("J7", _INCOMPLETE_LABEL, True)
        return Outcome("J2", _LABELS["J2"], False)
    if MDE_D is None:
        raise ValueError("MDE_D is required when D exists")
    big = D_lo > 0 and D >= MDE_D                           # D rule
    if not S1:
        return Outcome("J1", _LABELS["J1"]) if big else Outcome("J2", _LABELS["J2"])
    if big:
        row = "J3" if k >= 3 else "J4" if k == 2 else "J5"
        return Outcome(row, _LABELS[row])
    if D_lo > 0:                                            # significant but 0 < D < MDE_D
        return Outcome("J6", _LABELS["J6"])
    if D_hi < 0:
        return Outcome("J8", _LABELS["J8"])
    return Outcome("J7", _LABELS["J7"])                     # D_lo <= 0 <= D_hi


def _n(numbers: Dict[str, Any], key: str, fmt: str = "{:.4f}") -> str:
    """Numbers are filled by the caller (REQ-STOP-05); a missing one is marked, never invented."""
    v = numbers.get(key)
    if v is None:
        return f"[缺:{key}]"
    return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _ci(numbers: Dict[str, Any], lo: str, hi: str) -> str:
    return f"[{_n(numbers, lo)}, {_n(numbers, hi)}]"


def s1_description(numbers: Dict[str, Any]) -> str:
    """REQ-STOP-04 (v1.3): the S1 description sentence.  Numbers (filled by the caller): x (of the 6 in-class), y (of the 3 out-of-family, s=0.02),
    y_s001 (optional, s=0.01), budget, mde (or MDE_D), blind (the O5 result text).  No binomial inference, no detection-rate wording."""
    numbers = dict(numbers)
    if numbers.get("mde") is None and numbers.get("MDE_D") is not None:
        numbers["mde"] = numbers["MDE_D"]
    y2 = numbers.get("y_s001")
    s = (f"6 個類內漏洞中 RL 找回 {_n(numbers, 'x', '{:d}')} 個；其中族外 3 個找回 {_n(numbers, 'y', '{:d}')} 個"
         f"（s=0.02；" + (f"族外在 s=0.01 找回 {_n(numbers, 'y_s001', '{:d}')}/3；" if y2 is not None else "") +
         f"固定預算：{_n(numbers, 'budget')}，各 1 個 run；MDE_D={_n(numbers, 'mde')}）。"
         "6 個類內漏洞＝族外 3 個（D1、D2、O1）＋族內 3 個（O2、O3、D4）；族內漏洞手寫細網格（S_HW 36 個）已能取得大部分得利，只作描述，不計入通過判定。"
         "族內已知的 D3 單獨報告、不計入。"
         f"對單一盲區範例（O5）的描述：{_n(numbers, 'blind')}（預期不被找到）；本套件沒有 B1／B2（絕對時間、長記憶）的證據，"
         "不得由 O5 的結果推論對這類盲區的偵測能力。此為對本套件的描述，不外推到其他漏洞，也不推論其他漏洞被找回的比例。")
    return s


# REQ-STOP-06: the disclosure list.  Each item = (key, phrases that must appear, builder of the text).  Numbers come from the JSON files (REQ-STOP-05).
_DISCLOSURES = (
    ("d1_hybrid", ("D1", "混合"),
     lambda n: "【D1 混合參考策略】D1 的參考策略是「字面策略＋固定灌水 dz0」的混合，混合是為了湊足登錄大小（commander 已接受，meta 已標示）。"),
    ("honest_drift", ("honest_drift", "漂移"),
     lambda n: f"【honest_drift 漂移】漏洞環境中誠實（θ=0）的得利與基礎環境不同（逐項漂移量：{_n(n, 'honest_drift')}）；"
               "此漂移不影響同一環境內配對的 G 與 Δ，但須逐項記錄。"),
    ("d4_nonmonotone", ("D4", "不單調"),
     lambda n: "【D4 knob 不單調】D4 的 Δ 對 knob 不單調，校準以掃描完成（knob_monotone=false）。"),
    ("survivor", ("倖存者偏誤",),
     lambda n: f"【倖存者偏誤】套件只保留 Δ>0 且可校準的漏洞：D3 的 Δ 為負而單獨報告、O4（最高約 {_n(n, 'o4_max')}）與 O6（最高約 {_n(n, 'o6_max')}）"
               "校準不到 0.01 而移出、B1／B2 未實作；因此套件不代表所有可能的漏洞。"),
    ("designer1", ("designer_1", "受汙染", "不計入"),
     lambda n: "【designer_1 受汙染】designer_1 的材料含特徵細節，不計入任何統計（REQ-S1-23）。"),
    ("blind_o5", ("只有一個盲區範例", "O5", "B1", "B2"),
     lambda n: "【盲區】本套件只有一個盲區範例（O5），沒有 B1／B2（絕對時間、長記憶）的證據；盲區結果只能寫成對單一盲區範例（O5）的描述。"),
    ("fine_grid", ("細網格", "36", "保守"),
     lambda n: "【細網格】手寫基準 S_HW 擴為 36 個（REQ-HW-06）使手寫族更強，「RL 對手寫最佳」的比較對 RL 較為保守；"
               "族內三者（O2、O3、D4）因細網格已被手寫族涵蓋而只描述。"),
    ("val2", ("val2", "不是測試集"),
     lambda n: "【val2】knob 以 val∪val2 合併選擇（贏家詛咒的對策）；val2（3000–3031）不是測試集，不出現在任何測試結果中。"),
    ("nS", ("565", "n_S", "涵蓋"),
     lambda n: "【n_S 涵蓋】n_S=256 只涵蓋三種 σ 的中位數最壞情況；單一隨機對的最壞情況需 n_S≈565，目前無法涵蓋（REQ-OPT-13）。"),
    ("burst", ("burst", "時鐘", "保守"),
     lambda n: "【burst 時鐘】手寫的 burst 策略使用代理人自己的回合時鐘，而 RL 的觀察不含時間，因此 RL 對手寫策略的比較對 RL 較為保守（REQ-HW-05）。"),
)


def disclosure_text(numbers: Dict[str, Any]) -> str:
    """REQ-STOP-06: the full disclosure list as text; numbers from the dict (honest_drift, o4_max, o6_max); an absent number is marked '[缺:key]'."""
    return "【揭露清單】" + "".join(f(numbers) for _, _, f in _DISCLOSURES)


def missing_disclosures(text: str) -> List[str]:
    """REQ-STOP-06: keys of the disclosure items whose required phrases are not all present in `text`."""
    return [k for k, phrases, _ in _DISCLOSURES if not all(p in text for p in phrases)]


def unfilled_numbers(text: str) -> List[str]:
    """REQ-STOP-05: names of numbers that the text still marks as absent ('[缺:key]')."""
    seen: List[str] = []
    for m in re.finditer(r"\[缺:([A-Za-z0-9_]+)\]", text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def wording(outcome: Outcome, numbers: Dict[str, Any]) -> str:
    """REQ-STOP-04/06: sec 9.3 wording for the row (J3-J8 carry the S1 description and the full disclosure list); numbers filled by the caller (REQ-STOP-05)."""
    if outcome.incomplete:
        return "S1 已通過，但主實驗（r=0.5 的 3 個 run）尚未完成，無法判定 D。結果標為「未完成」。"
    row = outcome.row
    D = f"D={_n(numbers, 'D')}（CI {_ci(numbers, 'D_lo', 'D_hi')}"
    desc = s1_description(numbers)
    disc = disclosure_text(numbers)
    if row in ("J3", "J4"):
        kk = 3 if row == "J3" else 2
        tail = "此為此固定配置、此策略類別與預算下的可利用度下界。" if row == "J3" else "部分複現；另 1 個 run 的 D 的 CI 下界 ≤0。"
        return (f"在 M3C（h=6、L=2500、p=0.1、kref=0.5、r=0.5）上，學習型公開資訊對手（MLP ≤64 參數）在測試 seeds 上的得利 "
                f"G_RL={_n(numbers, 'G_RL')}（CI {_ci(numbers, 'G_RL_lo', 'G_RL_hi')}）超過最佳手寫無資訊策略 "
                f"G_手寫={_n(numbers, 'G_HW')}（CI {_ci(numbers, 'G_HW_lo', 'G_HW_hi')}），差值 {D}，≥MDE_D={_n(numbers, 'MDE_D')}），"
                f"3 個 run 中 {kk} 個複現（{kk}/3）。{tail}{_FEATURE_LIMIT}{desc}{_COMMON_LIMIT}{disc}")
    if row == "J5":
        return (f"僅 1 個 run（共 3 個，1/3）達到 D 規則（{D}）；未複現，不構成找到。只稱單一 run 的候選結果。{desc}{disc}")
    if row == "J6":
        return (f"{D}）統計上顯著，但小於 MDE_D={_n(numbers, 'MDE_D')}（本流程在 80% 檢定力下可確認的最小大小）；"
                f"僅作描述，不作為找到或機制可利用的主張。{desc}{disc}")
    if row == "J7":
        return (f"在此策略類別（公開資訊、MLP ≤64 參數、特徵 k=8）、預算（固定預算，見下）與 MDE_D={_n(numbers, 'MDE_D')} 下，"
                f"未找到優於手寫策略的得利（{D}）。{_FEATURE_LIMIT}{desc}此結果不構成機制安全的證明。{disc}")
    if row == "J8":
        return (f"{D}，上界 <0）：學習型對手顯著弱於最佳手寫策略，歸為優化失敗，不是機制較強的證據。{desc}{disc}")
    if row == "J1":
        return (f"{D}，≥MDE_D={_n(numbers, 'MDE_D')}）但 S1 未通過（s=0.02：族外 {_n(numbers, 'y', '{:d}')}/3，門檻 3/3；"
                f"錨點 NAIVE {_n(numbers, 'naive')}、M4 {_n(numbers, 'm4')}），為候選漏洞，未經檢測力背書，須人工查證。")
    if row == "J2":
        return (f"此訓練流程的檢測力標定（S1）未達標準（s=0.02：族外 {_n(numbers, 'y', '{:d}')}/3，門檻 3/3；"
                f"錨點 NAIVE {_n(numbers, 'naive')}、M4 {_n(numbers, 'm4')}），因此無法對 M3C 的可利用度作出任何下界或否定結論。")
    raise ValueError(row)


def w1_stop(opt_infeasible_r05: bool, mde_d_plan: float, pilot_failed: bool) -> bool:
    """REQ-STOP-02: True if any W1 stop condition holds ((a) infeasible, (b) mde_d_plan>0.02, (c) pilot failed)."""
    return bool(opt_infeasible_r05) or mde_d_plan > 0.02 or bool(pilot_failed)
