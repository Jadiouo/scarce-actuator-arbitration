#!/usr/bin/env python3
"""CPU-only independent number audit and report for completed formal main evaluation.

Uses stored snapshots only. Recomputes CIs with scipy (not arbitration.rl.metrics),
checks every reported paired contrast and the frozen validation MDEs, and writes
the result report, number audit, and a standalone confidence-interval figure.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import stop


def read(p):
    return json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ci(a):
    a = np.asarray(a, float)
    assert a.shape == (32,) and np.isfinite(a).all()
    half = stats.t.ppf(.975, 31) * a.std(ddof=1) / math.sqrt(32)
    return dict(mean=float(a.mean()), lo=float(a.mean() - half), hi=float(a.mean() + half), half=float(half))


def gains(raw):
    u = np.asarray(raw["snap"]["util_b"], float)[:, :, 1]
    return (u - u[0]) / raw["T_score"]


def fmt(x):
    return f"{x['mean']:+.5f} [{x['lo']:+.5f}, {x['hi']:+.5f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    data = root / "results/direction2"
    doc = read(data / "main.json")
    tc = stats.t.ppf(.975, 31)
    kap = optimize.brentq(lambda d: stats.nct.sf(tc, 31, d) + stats.nct.cdf(-tc, 31, d) - .8, 1, 6, xtol=1e-12)
    audited = []

    def check_ci(got, values, source):
        expected = ci(values)
        for key in expected:
            assert math.isclose(got[key], expected[key], rel_tol=1e-11, abs_tol=1e-13), (source, key)
        audited.append(dict(source=source, reported=got, recomputed=expected, consistent=True))

    def check_system(got, raw, row, source):
        ts = raw["T_score"]
        pen = np.asarray(raw["snap"]["pen"])[row] / ts
        check_ci(got["regret_per_round"], np.asarray(raw["snap"]["reg_b"])[row] / ts, source + ".regret_per_round")
        check_ci(got["mean_ineligible_fraction"], pen.mean(-1), source + ".mean_ineligible_fraction")
        for j in range(3):
            check_ci(got["ineligible_fraction_by_agent"][j], pen[:, j], source + f".ineligible_fraction_by_agent[{j}]")

    for ix, cell in enumerate(doc["cells"]):
        name = cell["cell"]
        folder = data / "main_eval"
        val = read(folder / f"{name}__val.json")
        raw = read(folder / f"{name}__test_raw.json")
        rl = read(folder / f"{name}__test_rl_raw.json")
        hand = read(folder / f"{name}__test_hw_raw.json")
        vrl = read(folder / f"{name}__val_rl_raw.json")
        vhw = read(folder / f"{name}__val_hw_raw.json")
        gh = dict(zip(hand["contract"]["policies"], gains(hand)))
        vg = dict(zip(vhw["contract"]["policies"], gains(vhw)))
        assert cell["selection_sha256"] == val["selection_sha256"] == raw["selection_sha256"]
        assert digest({k: v for k, v in val.items() if k != "selection_sha256"}) == val["selection_sha256"]
        assert digest({k: v for k, v in raw.items() if k != "raw_sha256"}) == raw["raw_sha256"] == cell["raw_sha256"]
        for batch in (rl, hand, vrl, vhw):
            assert batch["contract"]["source_sha256"] == val["source_sha256"] == cell["source_sha256"]
        names = vhw["contract"]["policies"]
        public = [n for n in names if n != "honest" and "oracle" not in n]
        assert len(public) == 36
        for role, candidates in (("public", public), ("plus_h0", public + ["honest"]),
                                 ("including_oracle", [n for n in names if n != "honest"])):
            assert val["selected_hw"][role] == max(candidates, key=lambda n: vg[n].mean())
        for name, value in val["hw_validation_means"].items():
            assert math.isclose(value, vg[name].mean(), rel_tol=1e-11, abs_tol=1e-13)
        assert rl["contract"]["seeds"] == hand["contract"]["seeds"] == raw["seeds"]
        assert raw["T_score"] == cell["T_score"] == 97500
        assert np.array_equal(gains(rl)[1:], np.asarray(raw["rl_gains"]))
        for n, v in raw["hw_gains"].items():
            assert np.array_equal(gh[n], np.asarray(v))
        assert cell["selected_run"] == max(val["checkpoints"], key=lambda r: (r["G_val_train"], -r["run_id"]))["run_id"]
        assert {r["run_id"] for r in cell["runs"]} == {0, 1, 2}
        d_arrays, dh_arrays = [], []
        for j, run in enumerate(cell["runs"]):
            base = f"cells[{ix}].runs[{j}]"
            g = gains(rl)[j + 1]
            d = g - gh[val["selected_hw"]["public"]]
            dh = g - gh[val["selected_hw"]["plus_h0"]]
            check_ci(run["G_RL"], g, base + ".G_RL")
            check_ci(run["D"], d, base + ".D")
            check_ci(run["D_plus_h0"], dh, base + ".D_plus_h0")
            check_system(run["system"], rl, j + 1, base + ".system")
            val_g = gains(vrl)[j + 1]
            for field, values in [("MDE_G", val_g), ("MDE_D", val_g - vg[val["selected_hw"]["public"]]),
                                  ("MDE_D_plus_h0", val_g - vg[val["selected_hw"]["plus_h0"]])]:
                expected = float(kap * values.std(ddof=1) / np.sqrt(32))
                assert math.isclose(run[field], expected, rel_tol=1e-11, abs_tol=1e-13)
                audited.append(dict(source=base + "." + field, reported=run[field], recomputed=expected, consistent=True))
            assert run["found_D"] == bool(ci(d)["lo"] > 0 and ci(d)["mean"] >= run["MDE_D"])
            d_arrays.append(d)
            dh_arrays.append(dh)
        assert cell["replication_k"] == sum(ci(d)["lo"] > 0 for d in d_arrays)
        assert cell["replication_k_plus_h0"] == sum(ci(d)["lo"] > 0 for d in dh_arrays)
        check_system(cell["honest_system"], rl, 0, f"cells[{ix}].honest_system")
        for role, baseline in cell["G_HW"].items():
            check_ci(baseline, gh[baseline["name"]], f"cells[{ix}].G_HW.{role}")
        for n, system in cell["hw_system"].items():
            j = hand["contract"]["policies"].index(n)
            check_system(system, hand, j, f"cells[{ix}].hw_system[{n}]")

    primary = next(c for c in doc["cells"] if c["r"] == .5)
    chosen = next(r for r in primary["runs"] if r["run_id"] == primary["selected_run"])
    d = chosen["D"]
    expected_row = ("J1" if d["lo"] > 0 and d["mean"] >= chosen["MDE_D"] else "J2") if not doc["S1_gate"]["S1_pass"] else (
        ("J3" if primary["replication_k"] == 3 else "J4" if primary["replication_k"] == 2 else "J5")
        if d["lo"] > 0 and d["mean"] >= chosen["MDE_D"] else
        "J6" if d["lo"] > 0 else "J8" if d["hi"] < 0 else "J7")
    assert doc["verdict"]["row"] == expected_row
    audit = dict(schema="direction2-main-number-audit/1", verified=True, audited_metrics=len(audited),
                 source_sha256=hashlib.sha256((data / "main.json").read_bytes()).hexdigest(),
                 method="independent scipy t interval and noncentral-t MDE; paired differences rebuilt from utility snapshots",
                 assertions=audited, verdict=expected_row)
    (data / "main_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    report_dir = root / "docs/direction2/results"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = ["# 方向二主實驗：正式 test、配對信賴區間與複現判定", "",
               f"依原登記規則，主要設定 r=0.5 判為 **{expected_row}：{doc['verdict']['label']}**。"
               f"validation 選定 run{primary['selected_run']}；D={fmt(d)}，MDE_D={chosen['MDE_D']:.5f}，"
               f"三個 run 中 {primary['replication_k']}/3 的配對 D 信賴區間下界大於 0。", "",
               "此判定必須連同 S1 限制閱讀：原登記閘門通過，但事後 off-knob 診斷只有 2/3 族外漏洞有利用證據；"
               "D2、O5 的 detected 混入基礎優勢。因此結果支持此固定機制與策略類別下的數值比較，"
               "不能把 S1 解讀成對任意漏洞的可靠檢測力保證。", "",
               "兩個 cell 各三個 run 全部報告；每個 checkpoint 與主 run 都在 validation 選定。test 使用原定 32 個 seeds，"
               "T=100000、記分到 T−L=97500。G 是相對同 seed 誠實報告的每輪個人效用增量；D 是 RL 與 validation 選定手寫基準的逐 seed 差。"
               "下列 95% CI 使用 paired t(31)，沒有把三個 run 合併成 96 個獨立樣本。", "",
               "| r | run | 選定世代 | G_RL [95% CI] | D [95% CI] | MDE_D | D 規則 |",
               "|---|---:|---:|---|---|---:|---|"]
    for cell in doc["cells"]:
        for r in cell["runs"]:
            tag = "（主值）" if r["run_id"] == cell["selected_run"] else ""
            summary.append(f"| {cell['r']} | {r['run_id']}{tag} | {r['checkpoint']['gen']} | {fmt(r['G_RL'])} | {fmt(r['D'])} | {r['MDE_D']:.5f} | {'通過' if r['found_D'] else '未通過'} |")
    summary += ["", "| r | validation 選定手寫基準 | G_HW [95% CI] | 含 H0 選定者 | 含 oracle 選定者及 G |", "|---|---|---|---|---|"]
    for c in doc["cells"]:
        h, o = c["G_HW"]["public"], c["G_HW"]["including_oracle"]
        summary.append(f"| {c['r']} | {h['name']} | {fmt(h)} | {c['G_HW']['plus_h0']['name']} | {o['name']}：{fmt(o)} |")
    summary += ["", f"含 H0 的附帶判定為 {doc['verdict_plus_h0']['row']}；不改變主要判定。"
                "複現數依規格計 D_lo>0，與每個 run 是否同時跨過自己的 MDE 是兩件事。"
                "r=0.9 是探索性、未經 S1 標定，而且事前量測顯示既定訓練預算下功效不足，不用來決定主要 J 列。", "",
                "這些正差值表示：只看公開資訊的學習策略，在固定 M3C 上找到了手寫候選沒有捕捉到的額外私利。"
                "表中的 oracle 策略有額外資訊，不屬公平的主要對照；其更高的 G 也提醒我們，當前搜尋沒有證明已找到最強攻擊。", "",
                "![正式測試配對差值](../../../figures/direction2-main-test.png)", "",
                "系統層面的代價如下。honest α 是三位 agent 平均的不合格比例；攻擊下另外列出仍誠實的 agent 0、2 被暫停的比例。這些數字均來自同一 T_score 快照。", "",
                "| r | 策略 | R/T [95% CI] | honest α 或 agent 0、2 暫停比例 |", "|---|---|---|---|"]
    for c in doc["cells"]:
        h = c["honest_system"]
        summary.append(f"| {c['r']} | 全體誠實 | {fmt(h['regret_per_round'])} | α={fmt(h['mean_ineligible_fraction'])} |")
        hw_name = c["selected_hw"]["public"]
        s = c["hw_system"][hw_name]
        summary.append(f"| {c['r']} | 手寫 {hw_name} | {fmt(s['regret_per_round'])} | agent0={s['ineligible_fraction_by_agent'][0]['mean']:.5f}；agent2={s['ineligible_fraction_by_agent'][2]['mean']:.5f} |")
        for r in c["runs"]:
            s = r["system"]
            summary.append(f"| {c['r']} | RL run{r['run_id']} | {fmt(s['regret_per_round'])} | agent0={s['ineligible_fraction_by_agent'][0]['mean']:.5f}；agent2={s['ineligible_fraction_by_agent'][2]['mean']:.5f} |")
    honest_regret = primary["honest_system"]["regret_per_round"]["mean"]
    attack_regret = chosen["system"]["regret_per_round"]["mean"]
    summary += ["", f"以主要設定為例，誠實時平均 R/T={honest_regret:.5f}，run{primary['selected_run']} 攻擊下為 {attack_regret:.5f}。"
                "攻擊者的個人增益伴隨較高的系統 regret，不能解讀成整體系統效能提升。這裡列出均值比較，未另作 regret 差值的顯著性判定。", "",
                "**先前 test 暴露與證據獨立性。** r=0.5 run0 與 S1 的 D2 兩格權重、訓練曲線完全相同。"
                "S1 已在同一組 test seeds 評估過該策略；因此本次 main run0 不能算相對於 S1 的新增獨立證據。"
                "其餘兩個 run 是原定的不同訓練 seed。此次保留全部原定 run、checkpoint、手寫候選與 test seeds，"
                "沒有依 test 分數更換主 run。完整政策雜湊對照在 main.json 的 prior_identical_policies。", "",
                "數值只針對固定 M3C 參數與公開資訊 MLP（25 特徵、55 參數）；機制沒有針對 RL 重新調參。"
                "信賴區間反映給定政策下的環境 seed 變異，不涵蓋重新訓練的所有不確定性。這不是最佳回應的證明，"
                "也不直接代表真實機器人系統。", ""]
    reg = read(data / "vuln_suite.json")
    s1 = read(data / "s1.json")
    run_plan = read(data / "run_plan.json")
    budget = next(c for c in run_plan["cells"] if c["name"] == "s1_D1_s0.02")
    in_class = next(s for s in s1["summary"] if s["scope"] == "in_class" and s["size"] == .02)
    o5 = next(r for r in s1["runs"] if r["id"] == "O5")
    mdes = [r["MDE_D"] for r in s1["runs"] if r.get("category") == "out_of_family"]
    s1_desc = stop.s1_description(dict(
        x=in_class["x"], y=s1["gate"]["y"], y_s001=s1["gate"]["y_s001"],
        budget=f"T_train={budget['T_train']}、λ={budget['lam']}、n_S={budget['n_S']}、G_gens={budget['G_gens']}",
        mde=f"族外六格 {min(mdes):.5f} 至 {max(mdes):.5f}",
        blind=f"原規則 detected={o5['detected']}，D={fmt(o5['D'])}；但 off-knob 診斷沒有利用證據，屬基礎優勢混淆"))
    assert not stop.unfilled_numbers(s1_desc)
    summary += ["依凍結規格附上 S1 描述（原 detected 定義）：" + s1_desc, "",
                "上述「找回」是原規則的計數；事後利用證據僅族外 2/3。D2 為偽陽性，O5 亦未證實利用，"
                "不可把此計數當成對未知漏洞的檢測率。S1 的 O1 與 NAIVE／M4 錨點也曾在正式 S1 入口前的煙霧測試暴露 test seeds，"
                "完整時間線保留於 [S1 報告](s1.md)。", "",
                "固定 M3C 配置為 p=0.1、h=6、L=2500、kref=0.5。公開特徵是手工充分統計量的近似，"
                "未涵蓋的長歷史或絕對時間模式仍可能形成漏掉的漏洞。", ""]
    disclosures = stop.disclosure_text(dict(honest_drift=reg["honest_drift_summary"], o4_max=reg["o4_max"], o6_max=reg["o6_max"]))
    assert not stop.unfilled_numbers(disclosures) and not stop.missing_disclosures(disclosures)
    summary += [disclosures, "", "所有評估在學校機器以原有 float64 模擬器執行；MLP 使用已驗證一致的 CUDA Graph，手寫策略使用 eager。"
                "原始逐 seed 快照、validation 選取與凍結 MDE、test ledger 都有保存。"
                f"本報告另以獨立 scipy 計算核對 {len(audited)} 項 CI／MDE，全部一致。", "",
                "資料：[main.json](../../../results/direction2/main.json)、[逐 seed 與選取紀錄](../../../results/direction2/main_eval/)、"
                "[機器可讀稽核](../../../results/direction2/main_audit.json)、[數字來源表](number-audit.md)。"
                "重建：`python3 scripts/report_d2_main.py`（CPU，不重新評估政策）。",
                "相關：[正式評估協定](../design/main-evaluation.md)、[S1 結果](s1.md)、[凍結規格](../../direction2-spec.md)。"]
    # Each CI/MDE has a deterministic JSON source path and a recomputation check.
    audit_md = ["# 方向二主實驗數字稽核", "", f"共 {len(audited)} 項 CI／MDE 從原始 snapshot 重算，全部一致。判定另以獨立條件式核對為 {expected_row}。",
                "", "原始檔：results/direction2/main.json；以下 JSON 路徑均相對其根節點。逐 seed 原始資料位於 results/direction2/main_eval/。", "",
                "| JSON 路徑 | 數字 | 重算一致 |", "|---|---|---|"]
    for row in audited:
        value = fmt(row["reported"]) if isinstance(row["reported"], dict) else f"{row['reported']:.8f}"
        audit_md.append(f"| `{row['source']}` | {value} | 是 |")
    (report_dir / "main.md").write_text("\n".join(summary) + "\n")
    (report_dir / "number-audit.md").write_text("\n".join(audit_md) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".scratch/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = ["#2563eb", "#d97706", "#059669"]
    for ax, cell in zip(axes, doc["cells"]):
        for y, r in enumerate(cell["runs"]):
            d = r["D"]
            ax.errorbar(d["mean"], y, xerr=d["half"], fmt="o", capsize=5, color=colors[y], linewidth=2)
            ax.plot(r["MDE_D"], y, marker="|", ms=16, mew=2, color="#be123c")
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set_yticks(range(3), [f"run {k}" + (" (selected)" if k == cell["selected_run"] else "") for k in range(3)])
        ax.invert_yaxis()
        ax.set_ylim(2.6, -.6)
        ax.set_title(f"r={cell['r']}" + (" (primary)" if cell["r"] == .5 else " (exploratory)"))
        ax.set_xlabel("Paired gain difference: learned - handwritten")
        ax.set_xlim(-.001, .021)
        ax.set_xticks([0, .005, .01, .015, .02])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Formal test: paired 95% confidence intervals", fontweight="bold")
    fig.text(.5, .015, "Bars: paired t(31) CI on 32 environment seeds. Red ticks: frozen validation MDE.\nS1 calibration is confounded; run0 at r=0.5 was previously evaluated in S1.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .1, 1, .95])
    (root / "figures").mkdir(exist_ok=True)
    fig.savefig(root / "figures/direction2-main-test.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(dict(audited_metrics=len(audited), verdict=expected_row, primary_D=chosen["D"],
                          replication_k=primary["replication_k"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
