"""Re-estimate the Direction-2 GPU hours (spec 5.5 formula  G_gens*T_train*t_step + floor(G_gens/50)*t_val) with the MEASURED step time of the
vulnerable environments (results/direction2/measure_vuln_timing.json, D4 and D2; lam=256, n_S=256, T=2e4).

Rules: cells with a vulnerability operator use their own measured step time (D4, D2) or, for the operators not measured (D1, O1, O2, O3, O5), the
LARGER of the two measured ones (an assumption, disclosed); plain M3C / M4 cells keep the v2 step time.  The 7 diagnostic cells (wired=false) are not
counted; the NAIVE anchor reuses the pilot (0 h); the s=0.01 cells are counted only for sizes in G_s.  Output: results/direction2/hours_estimate.json.

Usage: python3 scripts/estimate_d2_hours.py [--g-s 0.01 0.02]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
VAL_EVERY = 50


def run_s(G, t_step_ms, t_val_s, T_train=20000):
    return G * T_train * t_step_ms / 1000.0 + (G // VAL_EVERY) * t_val_s


def estimate(plan, timing, v2, G_s, reused=("s1_NAIVE_anchor",)):
    base = v2["run_time_lam256_T2e4_300gen"]
    plain = dict(mean=base["nS256_mean"]["t_step_ms"], p95=base["nS256_p95"]["t_step_ms"], t_val=v2["validation"]["t_val_s"])
    meas = timing["cells"]
    by_op = {c["operator"]: dict(mean=c["t_step_ms_mean"], p95=c["t_step_ms_p95"], t_val=c["t_val_s"]) for c in meas.values()}
    worst = dict(mean=max(x["mean"] for x in by_op.values()), p95=max(x["p95"] for x in by_op.values()), t_val=max(x["t_val"] for x in by_op.values()))
    rows, tot = [], {}
    for c in plan["cells"]:
        if not c.get("wired", True):
            continue                                                  # the 7 diagnostic cells
        name, n = c["name"], len(c["run_ids"])
        if name in reused:
            rows.append(dict(cell=name, group=c["group"], runs=0, mean_h=0.0, p95_h=0.0, note="result reused from the pilot (no retraining)"))
            continue
        if c["group"] == "s1" and c.get("size") == 0.01 and 0.01 not in G_s:
            rows.append(dict(cell=name, group=c["group"], runs=0, mean_h=0.0, p95_h=0.0, note="size 0.01 not in G_s: not run"))
            continue
        t = (by_op.get(c["operator"], worst) if c.get("operator") else plain)
        src = "measured" if c.get("operator") in by_op else ("assumed=max(D4,D2)" if c.get("operator") else "v2 plain")
        mean = run_s(c["G_gens"], t["mean"], t["t_val"]) * n / 3600.0
        p95 = run_s(c["G_gens"], t["p95"], t["t_val"]) * n / 3600.0
        rows.append(dict(cell=name, group=c["group"], runs=n, t_step_ms_mean=t["mean"], t_step_ms_p95=t["p95"], t_val_s=t["t_val"], t_source=src,
                         mean_h=mean, p95_h=p95))
    for g in ("pilot", "s1", "main"):
        sel = [r for r in rows if r["group"] == g]
        tot[g] = dict(runs=sum(r["runs"] for r in sel), mean_h=sum(r["mean_h"] for r in sel), p95_h=sum(r["p95_h"] for r in sel))
    tot["pilot_cold_only"] = next(dict(mean_h=r["mean_h"], p95_h=r["p95_h"]) for r in rows if r["cell"] == "pilot_naive_cold")
    tot["s1_plus_main"] = dict(runs=tot["s1"]["runs"] + tot["main"]["runs"], mean_h=tot["s1"]["mean_h"] + tot["main"]["mean_h"],
                               p95_h=tot["s1"]["p95_h"] + tot["main"]["p95_h"])
    tot["all_incl_pilot_cold_excl_warm_pilot"] = dict(mean_h=tot["s1_plus_main"]["mean_h"] + tot["pilot_cold_only"]["mean_h"],
                                                      p95_h=tot["s1_plus_main"]["p95_h"] + tot["pilot_cold_only"]["p95_h"])
    return dict(G_s=list(G_s), rows=rows, totals=tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-s", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--out", default=os.path.join("results", "direction2", "hours_estimate.json"))
    a = ap.parse_args()
    ld = lambda p: json.load(open(os.path.join(ROOT, p)))
    doc = estimate(ld("results/direction2/run_plan.json"), ld("results/direction2/measure_vuln_timing.json"),
                   ld("results/direction2/measure_v2_summary.json"), a.g_s)
    doc["inputs"] = dict(plan="results/direction2/run_plan.json", timing="results/direction2/measure_vuln_timing.json",
                         v2="results/direction2/measure_v2_summary.json",
                         rules="diag cells (wired=false) excluded; s1_NAIVE_anchor reused from pilot_naive_cold (0 h); s=0.01 cells only if in G_s; "
                               "operators other than D4/D2 use max(D4, D2) (not measured)")
    json.dump(doc, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(json.dumps(doc["totals"], indent=1))


if __name__ == "__main__":
    main()
