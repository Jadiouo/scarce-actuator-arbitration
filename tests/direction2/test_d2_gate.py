"""T-45..T-47: statistics, S1 gate, outcome classifier (spec s14.6)."""
import itertools
import os

import pytest
from scipy import stats

from conftest import ROOT, need
from arbitration.rl import metrics, s1, stop


def test_T45_clopper_pearson_lower_bound():
    """T-45 (REQ-S1-16, v1.3): one-sided 95% CP lower bounds; (3,3)=0.36840315, (2,3)=0.13535036, (6,6)=0.60696223, (5,6)=0.41819659 plus the older
    (6,8)... values; equals beta.ppf(0.05,x,n-x+1); P(Bin(n,LB)>=x)=0.05; monotone in x; LB appears only in descriptive table fields, never in the
    result wording (and never as an inference about the detection rate of other vulnerabilities)."""
    for (x, n), v in {(3, 3): 0.36840315, (2, 3): 0.13535036, (6, 6): 0.60696223, (5, 6): 0.41819659,
                      (6, 8): 0.40031061, (7, 8): 0.52932059, (8, 8): 0.68765602, (5, 8): 0.28924082, (10, 10): 0.74113445}.items():
        assert abs(metrics.cp_lower(x, n) - v) < 1e-7, (x, n)
        assert abs(metrics.cp_lower(x, n) - stats.beta.ppf(0.05, x, n - x + 1)) < 1e-12
        assert abs(stats.binom.sf(x - 1, n, metrics.cp_lower(x, n)) - 0.05) <= 1e-9
    assert abs(metrics.cp_lower(3, 3) - 0.05 ** (1 / 3)) < 1e-12 and abs(metrics.cp_lower(6, 6) - 0.05 ** (1 / 6)) < 1e-12
    assert metrics.cp_lower(0, 8) == 0.0 and abs(metrics.cp_lower(8, 8) - 0.05 ** (1 / 8)) < 1e-12
    for n in (3, 6, 8):
        lbs = [metrics.cp_lower(x, n) for x in range(n + 1)]
        assert all(a < b for a, b in zip(lbs[1:], lbs[2:])) and lbs[0] == 0.0, n
    assert metrics.cp_lower(0, 3) == 0.0 and metrics.cp_lower(0, 6) == 0.0
    assert abs(metrics.cp_lower(1, 8) - stats.beta.ppf(0.05, 1, 8)) < 1e-12 and abs(metrics.cp_lower(1, 8) - 0.0063913) < 1e-6
    # descriptive tables: out-of-family n=3 (both sizes) and class n=6 (s=0.02)
    rows = {r["size"]: r for r in s1.summary_table({0.02: 3, 0.01: 2}, n=3)}
    assert {"size", "x", "n", "p_hat", "LB"} <= set(rows[0.02])
    assert rows[0.02]["x"] == 3 and rows[0.02]["n"] == 3 and abs(rows[0.02]["p_hat"] - 1.0) < 1e-15 and abs(rows[0.02]["LB"] - 0.36840315) < 1e-7
    assert rows[0.01]["x"] == 2 and abs(rows[0.01]["p_hat"] - 2 / 3) < 1e-15 and abs(rows[0.01]["LB"] - 0.13535036) < 1e-7
    r6 = {r["size"]: r for r in s1.summary_table({0.02: 5}, n=6)}[0.02]
    assert r6["n"] == 6 and abs(r6["p_hat"] - 5 / 6) < 1e-15 and abs(r6["LB"] - 0.41819659) < 1e-7
    assert {r["size"]: r for r in s1.summary_table({0.02: 0}, n=3)}[0.02]["LB"] == 0.0
    assert s1.N_OUT_OF_FAMILY == 3 and s1.N_IN_CLASS == 6
    txt = stop.s1_description(dict(x=5, y=2, y_s001=3, budget="T_train=20000, lam=256, n_S=256, G_gens=416", mde=0.0058, blind="未找到"))
    for banned in ("檢測率", "Clopper", "CP 95% 下界", "LB", "0.37", "0.3684"):
        assert banned not in txt, banned


def test_T46_s1_gate_rule():
    """T-46 (REQ-S1-10/15/17, STOP-01, v1.3): S1 passes iff both (counted) anchors pass AND all 3 out-of-family vulnerabilities (D1, D2, O1) are detected at
    s=0.02 (3/3); 2/3 fails; an unfinished run (None) = undetected / anchor not passed (denominator stays 3); an informational M4 anchor is ignored; the
    in-family trio (O2, O3, D4), the blind O5, D3, designer_1 and s=0.01 results can take ANY value without changing the verdict; descriptive sizes
    never enter; the schedule runs s=0.02 first (9 runs: 3 + 3 + 1 + 2)."""
    det = lambda y: [True] * y + [False] * (3 - y)
    assert s1.s1_gate(True, True, False, det(3)) is True
    for y in (0, 1, 2):
        assert s1.s1_gate(True, True, False, det(y)) is False, y
    assert s1.s1_gate(False, True, False, det(3)) is False and s1.s1_gate(True, False, False, det(3)) is False
    assert s1.s1_gate(None, True, False, det(3)) is False and s1.s1_gate(True, None, False, det(3)) is False
    assert s1.s1_gate(True, None, True, det(3)) is True and s1.s1_gate(True, False, True, det(3)) is True       # informational M4 ignored
    assert s1.s1_gate(False, None, True, det(3)) is False and s1.s1_gate(True, None, True, det(2)) is False
    assert s1.s1_gate(True, True, False, [True, True, None]) is False and s1.s1_gate(True, True, False, [None, None, None]) is False
    for bad in (det(3) + [True] * 3, [True] * 6, [True] * 8, [True, True], []):                 # only the 3 out-of-family flags are accepted
        with pytest.raises(ValueError):
            s1.s1_gate(True, True, False, bad)
    # by id: everything that is not out-of-family may take any value (True/False/None) without changing the verdict
    base_ok = dict(D1=True, D2=True, O1=True)
    base_bad = dict(D1=True, D2=True, O1=False)
    others = ["O2", "O3", "D4", "O5", "D3", "designer_1"]
    for vals in itertools.product((True, False, None), repeat=len(others)):
        extra = dict(zip(others, vals))
        assert s1.s1_gate_by_id(True, True, False, dict(base_ok, **extra)) is True, extra
        assert s1.s1_gate_by_id(True, True, False, dict(base_bad, **extra)) is False, extra
    assert s1.s1_gate_by_id(True, True, False, dict(D1=True, D2=True)) is False                  # a missing out-of-family run counts as undetected
    assert s1.s1_gate_by_id(True, True, False, {}) is False
    assert s1.s1_gate_by_id(True, None, True, base_ok) is True and s1.s1_gate_by_id(True, None, False, base_ok) is False
    with pytest.raises(ValueError):
        s1.s1_gate_by_id(True, True, False, dict(base_ok, O4=True))                              # O4 / O6 / B1 / B2 are not in the suite
    with pytest.raises(ValueError):
        s1.s1_gate_by_id(True, True, False, dict(base_ok, B1=True))
    sch = s1.schedule([0.01, 0.02])
    assert len(sch) == 12 and all(sz == 0.02 for _, sz in sch[:9]) and len({i for i, sz in sch[:9]}) == 9


def _rows_exp(S1, D, lo, hi, mde, k):
    """Independent restatement of the 8 mutually exclusive conditions of s9.2."""
    conds = {
        "J1": (not S1) and lo > 0 and D >= mde,
        "J2": (not S1) and not (lo > 0 and D >= mde),
        "J3": S1 and lo > 0 and D >= mde and k == 3,
        "J4": S1 and lo > 0 and D >= mde and k == 2,
        "J5": S1 and lo > 0 and D >= mde and k == 1,
        "J6": S1 and lo > 0 and 0 < D < mde,
        "J7": S1 and lo <= 0 <= hi,
        "J8": S1 and hi < 0,
    }
    hit = [j for j, c in conds.items() if c]
    assert len(hit) == 1, (S1, D, lo, hi, mde, k, hit)
    return hit[0]


CI_CASES = {"pos": (0.03, 0.01, 0.05), "zero_lo": (0.01, 0.0, 0.02), "mid": (0.005, -0.01, 0.02), "zero_hi": (-0.01, -0.02, 0.0),
            "neg": (-0.03, -0.05, -0.01), "pos_tiny": (0.002, 0.0005, 0.0035)}


def test_T47_outcome_classifier():
    """T-47 (REQ-STOP-01..04, v1.3): full grid S1 x CI position x (D>=MDE) x k gives exactly one of J1..J8 (exhaustive, exclusive; boundaries D_lo=0,
    D=MDE, D_hi=0; signature unchanged: S1 is binary); D=None -> 'not significant' (J7 flagged incomplete if S1 passed, else J2); wording contains the
    required phrases ('6 個類內漏洞中 RL 找回 x 個；其中族外 3 個找回 y 個', 'single blind example (O5)', 'no B1/B2', J5 not replicated, J6 smaller than MDE_D,
    J8 optimisation failure, J1 candidate, J1/J2 'out-of-family y/3, threshold 3/3') and no binomial-inference phrases or old thresholds."""
    import inspect
    assert list(inspect.signature(stop.classify).parameters) == ["S1", "D", "D_lo", "D_hi", "MDE_D", "k"]
    texts = {}
    nums = lambda D, lo, hi, mde, k: dict(D=D, D_lo=lo, D_hi=hi, MDE_D=mde, k=k, x=4, y=2, y_s001=3, naive="通過", m4="通過",
                                           budget="T_train=20000, lam=256, n_S=256, G_gens=416", blind="未找到")
    for S1, (name, (D, lo, hi)), mde, k in itertools.product((True, False), CI_CASES.items(), (0.01, 0.03, 0.002), (1, 2, 3)):
        exp = _rows_exp(S1, D, lo, hi, mde, k)
        out = stop.classify(S1, D, lo, hi, mde, k)
        assert out.row == exp and out.incomplete is False, (S1, name, mde, k)
        texts.setdefault(exp, stop.wording(out, nums(D, lo, hi, mde, k)))
    assert set(texts) == {"J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"}
    for row, t in texts.items():
        for banned in ("檢測率", "Clopper", "CP 95% 下界", "6/8", "≥5/6", "5/6"):
            assert banned not in t, (row, banned)
    for row in ("J3", "J4", "J5", "J6", "J7", "J8"):
        t = texts[row]
        assert "對本套件的描述" in t and "固定預算" in t, row
        assert "6 個類內漏洞中 RL 找回 4 個；其中族外 3 個找回 2 個" in t, row
        assert "對單一盲區範例（O5）的描述" in t and "沒有 B1／B2" in t, row
    assert "僅限此固定配置" in texts["J3"] and "僅限此固定配置" in texts["J4"]
    assert "未複現" in texts["J5"] and "小於 MDE_D" in texts["J6"] and "優化失敗" in texts["J8"] and "候選漏洞" in texts["J1"]
    assert "不構成機制安全的證明" in texts["J7"] and "未找到" in texts["J7"]
    for row in ("J1", "J2"):
        assert "族外 2/3，門檻 3/3" in texts[row], row
    j7 = stop.classify(True, None, None, None, None, 0)
    assert j7.row == "J7" and j7.incomplete is True
    w = stop.wording(j7, {})
    assert "未完成" in w and "未找到" not in w
    j2 = stop.classify(False, None, None, None, None, 0)
    assert j2.row == "J2" and j2.incomplete is False
    assert stop.classify(True, 0.005, -0.01, 0.02, 0.01, 1).incomplete is False


def _limit_ok(t):
    """REQ-OBS-09 / s16: the feature-class limitation is stated (own phrase, or the s9.3 common-limitation sentence, or the J7 'features k=8')."""
    return ("僅限此特徵類別" in t) or ("手工充分統計量的近似" in t and "遺漏的特徵" in t) or ("特徵 k=8" in t)


def test_OBS09_limitation_wording():
    """OBS-09 (REQ-OBS-09, s9.3, s16): the wording of every 'found' / 'not found' conclusion states that the conclusion is limited to the feature
    class (hand-made features k=8, public information): J3, J4 carry the common limitation sentence ('特徵為手工充分統計量的近似，遺漏的特徵會漏掉依賴
    更長歷史或絕對時間的漏洞'), J7 names the feature class ('特徵 k=8' or '僅限此特徵類別') and never claims absence outright; the N1
    'incomplete' text makes no 'not found' claim.  (That EVERY external text carries the note cannot be checked automatically.)"""
    num = lambda D, lo, hi, mde, k: dict(D=D, D_lo=lo, D_hi=hi, MDE_D=mde, k=k)
    cases = {
        "J3": (stop.classify(True, 0.03, 0.01, 0.05, 0.02, 3), num(0.03, 0.01, 0.05, 0.02, 3)),
        "J4": (stop.classify(True, 0.03, 0.01, 0.05, 0.02, 2), num(0.03, 0.01, 0.05, 0.02, 2)),
        "J7": (stop.classify(True, 0.002, -0.01, 0.014, 0.01, 1), num(0.002, -0.01, 0.014, 0.01, 1)),
    }
    for row, (out, nums) in cases.items():
        assert out.row == row
        t = stop.wording(out, nums)
        assert _limit_ok(t), f"{row}: no feature-class limitation in the wording"
    t3 = stop.wording(*[cases["J3"][0], cases["J3"][1]])
    assert "手工充分統計量的近似" in t3 and "遺漏的特徵" in t3 and "絕對時間" in t3
    t7 = stop.wording(cases["J7"][0], cases["J7"][1])
    assert ("特徵 k=8" in t7 or "僅限此特徵類別" in t7) and "不構成機制安全的證明" in t7 and "未找到" in t7
    for banned in ("機制安全。", "不存在漏洞", "沒有漏洞", "安全無虞"):
        assert banned not in t7
    inc = stop.wording(stop.classify(True, None, None, None, None, 0), {})
    assert "未完成" in inc and "未找到" not in inc
    assert "對本套件的描述" in stop.s1_description(dict(x=5, y=3, y_s001=2, budget="T_train=20000, lam=256, n_S=256, G_gens=416", mde=0.0058,
                                                        blind="未找到"))


def test_OBJ06_expectation_registered_before_work():
    """OBJ-06 (REQ-OBJ-06): the prior expectation (most likely (ii) or (iii), together > 80%; (i) < 15%) is on file.  The spec names no separate
    registry file, so the registered places are checked: docs/direction2-spec.md (REQ-OBJ-06) and the planning log docs/direction2-notes.md
    ('期望（推論）'); both must exist and state the same numbers.  (Telling the user is a process step and cannot be tested.)"""
    spec = os.path.join(ROOT, "docs", "direction2-spec.md")
    notes = os.path.join(ROOT, "docs", "direction2-notes.md")
    assert os.path.exists(spec) and os.path.exists(notes)
    spec_txt, notes_txt = open(spec, encoding="utf-8").read(), open(notes, encoding="utf-8").read()
    i = spec_txt.index("**REQ-OBJ-06")                   # the definition (the bare id also occurs in the revision log)
    seg = spec_txt[i:i + 400]
    assert "(ii)" in seg and "(iii)" in seg and ">80%" in seg.replace(" ", "") and "<15%" in seg.replace(" ", "")
    j = notes_txt.index("期望（推論）")
    nseg = notes_txt[j:j + 300].replace(" ", "")
    assert "(ii)" in nseg and "(iii)" in nseg and ">80%" in nseg and "<15%" in nseg


def test_T69_v12_disclosures_in_conclusion_wording_OPT13_HW05_S123():
    """T-69 (REQ-OPT-13, REQ-HW-05, REQ-S1-23, s9.3 common-limitation sentence v1.2): the wording of every conclusion that reports D (J3, J4, J7)
    carries the three v1.2 disclosures: (a) the n_S coverage limit: the single-pair worst case needs n_S about 565 and is not covered (text names
    '565'); (b) the burst handwritten strategies use the agent's own round clock while RL's observation has no time, so the RL-vs-handwritten
    comparison is conservative for RL (text names 'burst', a clock/time word and 保守); (c) designer_1 is contaminated, not counted in S1, reported
    only (text names 'designer_1' and 受汙染). """
    num = lambda D, lo, hi, mde, k: dict(D=D, D_lo=lo, D_hi=hi, MDE_D=mde, k=k)
    cases = [("J3", stop.classify(True, 0.03, 0.01, 0.05, 0.02, 3), num(0.03, 0.01, 0.05, 0.02, 3)),
             ("J4", stop.classify(True, 0.03, 0.01, 0.05, 0.02, 2), num(0.03, 0.01, 0.05, 0.02, 2)),
             ("J7", stop.classify(True, 0.002, -0.01, 0.014, 0.01, 1), num(0.002, -0.01, 0.014, 0.01, 1))]
    for row, out, nums in cases:
        assert out.row == row
        t = stop.wording(out, nums)
        assert "565" in t and ("n_S" in t) and ("涵蓋" in t), f"{row}: n_S coverage disclosure (REQ-OPT-13) missing"
        assert "burst" in t and ("時鐘" in t or "時間" in t) and "保守" in t, f"{row}: burst own-clock disclosure (REQ-HW-05) missing"
        assert "designer_1" in t and "受汙染" in t and "不計入" in t, f"{row}: designer_1 contamination disclosure (REQ-S1-23) missing"


# ====================================================================================================================== spec v1.3 additions
_DISCLOSURE_PHRASES = {          # independent reference copy of REQ-STOP-06 (the implementation's own list is not imported here)
    "d1_hybrid": ["D1", "混合"],
    "honest_drift": ["honest_drift", "漂移"],
    "d4_nonmonotone": ["D4", "不單調"],
    "survivor": ["倖存者偏誤"],
    "designer1": ["designer_1", "受汙染", "不計入"],
    "blind_o5": ["只有一個盲區範例", "O5", "B1", "B2"],
    "fine_grid": ["細網格", "36", "保守"],
    "val2": ["val2", "不是測試集"],
    "nS": ["565", "n_S", "涵蓋"],
    "burst": ["burst", "時鐘", "保守"],
}
_J_CASES = {       # J3..J8: (classify args)
    "J3": (True, 0.03, 0.01, 0.05, 0.02, 3), "J4": (True, 0.03, 0.01, 0.05, 0.02, 2), "J5": (True, 0.03, 0.01, 0.05, 0.02, 1),
    "J6": (True, 0.01, 0.002, 0.02, 0.02, 1), "J7": (True, 0.002, -0.01, 0.014, 0.01, 1), "J8": (True, -0.03, -0.05, -0.01, 0.01, 1),
}
_DISC_NUMBERS = dict(honest_drift="D1:+0.0123/D2:-0.0045/D4:+0.0067", o4_max=0.0027, o6_max=0.0036, nS_worst=565,
                     x=4, y=3, y_s001=2, budget="T_train=20000, lam=256, n_S=256, G_gens=416", blind="未找到",
                     G_RL=0.05, G_RL_lo=0.04, G_RL_hi=0.06, G_HW=0.02, G_HW_lo=0.01, G_HW_hi=0.03)


def _jtext(row, numbers=None):
    S1, D, lo, hi, mde, k = _J_CASES[row]
    out = stop.classify(S1, D, lo, hi, mde, k)
    assert out.row == row
    nums = dict(_DISC_NUMBERS if numbers is None else numbers, D=D, D_lo=lo, D_hi=hi, MDE_D=mde, k=k)
    return stop.wording(out, nums)


def test_T75_disclosure_list_v13():
    """T-75 (REQ-STOP-06, STOP-05, OPT-13, HW-05, S1-23; v1.3): the result text of J3..J8 carries ALL ten disclosure items: D1 hybrid reference policy,
    honest_drift, D4 knob not monotone, survivor bias (D3 / O4 / O6 / B1 / B2 removed or missing), designer_1 contaminated and not counted, only one blind
    example (O5) and no B1/B2 evidence, fine grid of 36 makes the comparison conservative for RL, val2 is not the test set, n_S about 565 not covered,
    burst own clock.  The reference phrases are an independent copy in this test; `stop.missing_disclosures(text)` must list a key as soon as its
    phrase is deleted from an otherwise complete text; numbers come from the numbers dict (JSON) and an absent number is marked, never invented."""
    miss = need(stop, "missing_disclosures", "REQ-STOP-06")
    unfilled = need(stop, "unfilled_numbers", "REQ-STOP-05")
    for row in ("J3", "J4", "J5", "J6", "J7", "J8"):
        t = _jtext(row)
        for key, phrases in _DISCLOSURE_PHRASES.items():
            for ph in phrases:
                assert ph in t, f"{row}: disclosure '{key}' lacks '{ph}'"
        assert miss(t) == [], (row, miss(t))
        assert unfilled(t) == [], (row, unfilled(t))
        # numbers come from the dict: the drift summary string and the O4 / O6 maxima appear verbatim
        assert "D1:+0.0123/D2:-0.0045/D4:+0.0067" in t and "0.0027" in t and "0.0036" in t, row
        # every key breaks as soon as its phrase is removed from the complete text
        for key, phrases in _DISCLOSURE_PHRASES.items():
            for ph in phrases:
                cut = t.replace(ph, "□")
                assert key in miss(cut), (row, key, ph)
    # an absent number is marked (and reported by unfilled_numbers), never invented
    t = _jtext("J3", numbers=dict(x=4, y=3, budget="b", mde=0.0058, blind="未找到"))
    assert "[缺:honest_drift]" in t and "[缺:o4_max]" in t and "[缺:o6_max]" in t
    assert set(unfilled(t)) >= {"honest_drift", "o4_max", "o6_max"}
    # the incomplete N1 text makes no 'not found' claim and carries no disclosure obligation
    assert "未找到" not in stop.wording(stop.classify(True, None, None, None, None, 0), {})
    # J1 / J2 (S1 failed): the S1 wording is still free of binomial inference
    for row, args in (("J1", (False, 0.03, 0.01, 0.05, 0.02, 3)), ("J2", (False, 0.0, -0.01, 0.01, 0.02, 1))):
        out = stop.classify(*args)
        assert out.row == row
        txt = stop.wording(out, dict(D=args[1], D_lo=args[2], D_hi=args[3], MDE_D=args[4], k=3, x=4, y=2, naive="通過", m4="通過"))
        assert "門檻 3/3" in txt and "6/8" not in txt and "檢測率" not in txt
