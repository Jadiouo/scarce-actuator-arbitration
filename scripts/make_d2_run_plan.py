"""Generate results/direction2/run_plan.json: the frozen Direction-2 training plan (one entry per cell).

Everything here comes from the FINAL values of spec v1.3 section 5.5 (constants below, each with its JSON key path) and from
results/direction2/vuln_suite.json (knobs).  Nothing is read from a measurement file at training time: the training entry point
(`python -m arbitration.rl.train --plan ... --cell ...`) takes the parameters from the generated file.

The script also cross-checks `arbitration/rl/budget.py` (decide_params) with the MEAS v2 inputs; a mismatch is reported in the plan
(`budget_crosscheck`), budget.py is never modified.  Output is deterministic (no timestamps): the sha256 of the file is registered
in results/direction2/FREEZE.md and docs/direction2-freeze.md.

Usage: python3 scripts/make_d2_run_plan.py [--out results/direction2/run_plan.json]
"""
import argparse
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arbitration.rl import ablation, budget, env, s1  # noqa: E402

# ----------------------------------------------------------------------------------------------- spec v1.3 section 5.5 (final)
LAM, T_TRAIN, N_S = 256, 20000, 256                       # meas.json: rules.S1.choice.{lam,T_train}; v2.json: nS_need["0.5"].nS_from_median
G_GENS = {0.5: 416, 0.9: 499}                              # OPT-10: floor(W_run / (T_train * t_step)), t_step = 1.0812 ms; W_run 9000 / 10800
W_RUN_S = {"S1": 9000.0, "main": 10800.0}                  # meas.json: rules.S1.W_run_s, rules.main.W_run_s
T_STEP_REG_MS = 1.0811940001076437                         # meas.json: rules.S1.choice.t_step_ms (pre-registered input of the G_gens formula)
VAL_EVERY = 50                                             # REQ-OPT-06 (v1.2)
PATIENCE = None                                            # the spec defines no early-stop patience: checkpoint = best G_val (REQ-OPT-06)
SIGMA0 = 0.3                                               # REQ-OPT-01
T_VAL = 100000                                             # REQ-OPT-06: validation at T=1e5
OBS_VERSION = "public"                                     # REQ-OPT-12: explicit, never the default
F_PUBLIC, HIDDEN = 25, 2
VAL_SEEDS = [2000, 2031]
TRAIN_POOL = [0, 999]
WARM = dict(method="least-squares regression of the MLP output onto the best S_HW handwritten strategy on train seeds (train.warm_start_theta)",
            warm_T=4096, warm_seeds=16, ridge=1e-4, hidden=HIDDEN, F=F_PUBLIC, source="arbitration/rl/train.py: main() defaults, fit_lstsq(ridge=1e-4)")
V2 = "results/direction2/measure_v2_summary.json"
V1 = "results/direction2_meas.json"


def sha256_file(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def run_time_s(G, t_step_ms, t_val_s):
    """spec 5.5: G*T_train*t_step + floor(G/50)*t_val (start-up overhead not included)."""
    return G * T_TRAIN * t_step_ms / 1000.0 + (G // VAL_EVERY) * t_val_s


def budget_crosscheck(v1, v2):
    """decide_params with MEAS v2 worst-case sd_CRN (and, for the record, with MEAS v1) against the frozen values."""
    Ts = (20000, 40000, 100000)
    summ = v2["m2_summary"]
    worst = {r: max(m["sd_crn_median"] for m in summ if m["r"] == r) for r in (0.5, 0.9)}   # T = 2e4 only (v2 measured T=2e4)
    sd_v2 = {(r, T): worst[r] for r in (0.5, 0.9) for T in Ts}
    sd_v1 = {(m["r"], m["T"]): m["sd_crn_median"] for m in v1["m2_summary"]}
    t_step = lambda B: T_STEP_REG_MS
    B_max = v1["B_max"]
    out = {"inputs": dict(B_max=B_max, t_step_ms=T_STEP_REG_MS, W_run_S1_s=W_RUN_S["S1"], W_run_main_s=W_RUN_S["main"],
                          sd_crn_worst_sigma_median_v2={str(k): v for k, v in worst.items()},
                          note="v2 measured sd_CRN at T=2e4 only; it is used for every T_train candidate (same convention as tests T-24b)")}
    for tag, sd in (("v2", sd_v2), ("v1", sd_v1)):
        for stage, W in (("S1", W_RUN_S["S1"]), ("main", W_RUN_S["main"])):
            res = budget.decide_params(t_step, B_max, W, sd)
            out[f"{tag}_{stage}"] = {k: res[k] for k in ("feasible", "lam", "T_train", "n_S", "G_gens", "r09_underpowered")}
    frozen = dict(S1=(LAM, T_TRAIN, N_S, G_GENS[0.5]), main_r09=(LAM, T_TRAIN, N_S, G_GENS[0.9]))
    got_s1 = out["v2_S1"]
    got_main = out["v2_main"]
    out["frozen"] = dict(lam=LAM, T_train=T_TRAIN, n_S=N_S, G_gens_r05=G_GENS[0.5], G_gens_r09=G_GENS[0.9])
    out["v2_matches_frozen"] = (
        (got_s1["lam"], got_s1["T_train"], got_s1["n_S"], got_s1["G_gens"]) == frozen["S1"]
        and (got_main["lam"], got_main["T_train"], got_main["n_S"], got_main["G_gens"]) == frozen["main_r09"])
    out["v1_would_give_n_S"] = out["v1_S1"]["n_S"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/direction2/run_plan.json")
    a = ap.parse_args()
    v1, v2 = load(V1), load(V2)
    suite = load("results/direction2/vuln_suite.json")
    entries = suite["entries"]

    # consistency of the constants with the JSON they cite (a failure means this script is out of sync with the spec; nothing is guessed)
    assert v2["nS_need"]["0.5"]["nS_from_median"] == N_S
    assert abs(v1["rules"]["S1"]["choice"]["t_step_ms"] - T_STEP_REG_MS) < 1e-15
    assert v1["rules"]["S1"]["choice"]["lam"] == LAM and v1["rules"]["S1"]["choice"]["T_train"] == T_TRAIN
    assert v1["rules"]["S1"]["W_run_s"] == W_RUN_S["S1"] and v1["rules"]["main"]["W_run_s"] == W_RUN_S["main"]
    for r, W in ((0.5, W_RUN_S["S1"]), (0.9, W_RUN_S["main"])):
        assert min(1000, int(W // (T_TRAIN * T_STEP_REG_MS / 1000.0))) == G_GENS[r]
    t_mean = v2["run_time_lam256_T2e4_300gen"]["nS256_mean"]["t_step_ms"]
    t_p95 = v2["run_time_lam256_T2e4_300gen"]["nS256_p95"]["t_step_ms"]
    t_val = v2["validation"]["t_val_s"]

    def est(G):
        return dict(mean_s=run_time_s(G, t_mean, t_val), p95_s=run_time_s(G, t_p95, t_val), t_step_mean_ms=t_mean, t_step_p95_ms=t_p95,
                    t_val_s=t_val, n_validations=G // VAL_EVERY,
                    basis="spec 5.5: G_gens*T_train*t_step + floor(G_gens/50)*t_val (v2.json run_time_lam256_T2e4_300gen.nS256_{mean,p95}.t_step_ms; validation.t_val_s); start-up not included")

    def entry_of(vid, size):
        c = [(i, e) for i, e in enumerate(entries) if e["id"] == vid and (e.get("size") == size)]
        assert len(c) == 1, (vid, size, c)
        return c[0]

    cells = []

    def add(name, group, mechanism, r, run_ids, *, variant=None, vuln_id=None, size=None, G=None, start=None, extra=None, conditional=None,
            order=None, notes=None):
        G = G if G is not None else G_GENS[0.5]
        per = est(G)
        op = knob = category = suite_idx = None
        if vuln_id is not None:
            suite_idx, e = entry_of(vuln_id, size)
            op, knob, category = e["operator"], e["knob"], e["category"]
        cell = dict(
            name=name, group=group, order=order, mechanism=mechanism, variant=variant, vuln_id=vuln_id, operator=op, knob=knob,
            knob_source=(f"results/direction2/vuln_suite.json: entries[{suite_idx}].knob" if suite_idx is not None else None),
            size=size, category=category, r=r,
            lam=LAM, T_train=T_TRAIN, T_val=T_VAL, n_S=N_S, G_gens=G, val_every=VAL_EVERY, patience=PATIENCE, sigma0=SIGMA0,
            start=start or dict(policy="by_pilot", allowed=["cold", "warm"],
                                rule="REQ-OPT-07: one route for all of S1 and main, chosen by the NAIVE pilot (REQ-S1-20); needs results/direction2/pilot_decision.json"),
            obs_version=OBS_VERSION, F=F_PUBLIC, hidden=HIDDEN,
            run_ids=list(run_ids), algo_seeds=[9000 + k for k in run_ids],
            train_seeds=dict(pool=TRAIN_POOL, schedule="REQ-OPT-03: block [g*n_S,(g+1)*n_S) mod 1000 of perm_k, perm_k = Random(9000+run_id).shuffle(range(1000))"),
            val_seeds=VAL_SEEDS,
            est_per_run=per, n_runs=len(run_ids),
            est_total=dict(mean_s=per["mean_s"] * len(run_ids), p95_s=per["p95_s"] * len(run_ids)),
            conditional=conditional, wired=True, notes=notes)
        if extra:
            cell.update(extra)
        cells.append(cell)

    cold = dict(policy="cold", allowed=["cold"], rule="REQ-S1-20: the pilot starts cold (theta=0)")
    warm = dict(policy="warm", allowed=["warm"], rule="REQ-S1-20: runs only if the cold pilot is below 0.8*G*_ref (validation); same budget",
                warm_start=WARM)
    pilot_eval = dict(pilot_eval=dict(
        seeds="train and validation seeds only (REQ-S1-20; s1.require_pilot_seeds); no test seeds",
        threshold="validation G >= 0.8 * G*_ref (G*_ref re-measured on validation seeds, expected about 0.28)",
        always_report_1=dict(policy="v1 == 1 (adapter b=1.0, clip(u+1)=1), public-information realizable", baseline_fn="arbitration.rl.s1.naive_pilot_baseline",
                             compare_with="best handwritten strategy on NAIVE chosen on validation (REQ-HW-02; reference dz=2.0, 0.2793)",
                             Ts=[T_TRAIN, T_VAL], seeds=VAL_SEEDS, paired=True, same_T_same_seeds=True, report_T_change=True)))
    add("pilot_naive_cold", "pilot", "M3C", 0.5, [0], variant="NAIVE (p=0)", start=cold, extra=pilot_eval, order=0,
        notes="REQ-S1-20 learnability pilot; NAIVE = M3C with p=0 (env.anchor_config('NAIVE'))")
    add("pilot_naive_warm", "pilot", "M3C", 0.5, [0], variant="NAIVE (p=0)", start=warm, extra=pilot_eval, order=1,
        conditional="only if pilot_naive_cold fails the 0.8*G*_ref threshold (REQ-S1-20)", notes="registered warm route (REQ-OPT-07)")

    G_s = [0.01, 0.02]                                     # planning estimate (MDE_est(r=0.5)=0.0058); the gate value comes from M-3 (REQ-S1-22)
    for k, (vid, size) in enumerate(s1.schedule(G_s)):
        order = 2 + k
        if vid == "NAIVE":
            add("s1_NAIVE_anchor", "s1", "M3C", 0.5, [0], variant="NAIVE (p=0)", vuln_id="NAIVE", size=None, order=order,
                notes="REQ-S1-08 anchor: G rule, not the increment rule")
        elif vid == "M4rw0":
            add("s1_M4rw0_anchor", "s1", "M4", 0.99, [0], variant="M4 eps=0.2 L=20 rw=0", vuln_id="M4rw0", order=order,
                extra=dict(mech_cfg=dict(env.anchor_config("M4"))),
                notes="REQ-S1-09 anchor; the only S1 run not at r=0.5; informational if W1 G*_ref CI lower bound <= 0 (suite: m4_anchor_status=counted)")
        else:
            cond = "only if 0.01 is in G_s (G_s = {s in {0.01,0.02}: s >= MDE_D,plan}, REQ-S1-22)" if size == 0.01 else None
            add(f"s1_{vid}_s{size}", "s1", "M3C", 0.5, [0], vuln_id=vid, size=size, order=order, conditional=cond)

    for r in (0.5, 0.9):
        add(f"main_M3C_r{r}", "main", "M3C", r, [0, 1, 2], G=G_GENS[r], order=100 + int(r * 10),
            notes=("r=0.9 is underpowered at n_S<=256 (REQ-OPT-11): trained and reported descriptively only" if r == 0.9 else None))

    grp = ablation.groups()
    for j, g in enumerate(grp):
        add(f"diag_ablate_g{j + 1}", "diag", "M3C", 0.5, [0], order=200 + j,
            extra=dict(wired=False, diag=dict(req="REQ-S1-19(a)", drop_feature_indices_0based=g, F_full=F_PUBLIC)),
            notes="leave-one-group-out feature ablation; not blocking (decision 5); not yet wired in the training entry (T-55)")
    add("diag_history64", "diag", "M3C", 0.5, [0], order=206,
        extra=dict(wired=False, diag=dict(req="REQ-S1-19(b)", F=ablation.history64_dim(), P_DIAG=300, note="64-step history; diagnostic observation F=134")),
        notes="not blocking; not yet wired in the training entry (T-55)")

    tot_mean = sum(c["est_total"]["mean_s"] for c in cells)
    tot_p95 = sum(c["est_total"]["p95_s"] for c in cells)
    n_runs = sum(c["n_runs"] for c in cells)
    plan = dict(
        schema="direction2-run-plan/1",
        spec=dict(file="docs/direction2-spec.md", version="v1.3", sha256=sha256_file("docs/direction2-spec.md"),
                  section="5.5 (final values), 8.4 REQ-S1-10, 8.4 REQ-S1-20/22/25, 5.4 REQ-OPT-07, REQ-OPT-12"),
        vuln_suite=dict(file="results/direction2/vuln_suite.json", sha256=sha256_file("results/direction2/vuln_suite.json")),
        budget_file=dict(file="arbitration/rl/budget.py", sha256=sha256_file("arbitration/rl/budget.py"),
                         role="NOT consulted at training time: parameters come from this file; budget.py was only cross-checked (budget_crosscheck)"),
        frozen_budget=dict(
            lam=LAM, T_train=T_TRAIN, n_S=N_S, G_gens={"0.5": G_GENS[0.5], "0.9": G_GENS[0.9]}, val_every=VAL_EVERY, patience=PATIENCE,
            sigma0=SIGMA0, T_val=T_VAL, obs_version=OBS_VERSION,
            json_key_paths=dict(
                lam="results/direction2_meas.json: rules.S1.choice.lam (= rules.main.choice.lam)",
                T_train="results/direction2_meas.json: rules.S1.choice.T_train",
                n_S="results/direction2/measure_v2_summary.json: nS_need[\"0.5\"].nS_from_median (=256)",
                G_gens="results/direction2_meas.json: rules.S1.W_run_s=9000, rules.main.W_run_s=10800, rules.S1.choice.t_step_ms=1.0812 -> floor(W/(T_train*t_step)) = 416 / 499",
                t_step="results/direction2/measure_v2_summary.json: run_time_lam256_T2e4_300gen.nS256_mean.t_step_ms / nS256_p95.t_step_ms",
                t_val="results/direction2/measure_v2_summary.json: validation.t_val_s")),
        budget_crosscheck=budget_crosscheck(v1, v2),
        G_s_plan=dict(value=G_s, status="planning estimate; the gate value comes from M-3 MDE_D,plan (REQ-S1-22); s=0.01 cells are conditional"),
        start_registration=dict(cold=dict(theta0="zeros", sigma0=SIGMA0), warm=WARM, chosen="not yet decided: NAIVE pilot (REQ-S1-20) decides, recorded in results/direction2/pilot_decision.json"),
        totals=dict(n_cells=len(cells), n_runs=n_runs, mean_h=tot_mean / 3600.0, p95_h=tot_p95 / 3600.0,
                    note="all cells incl. the conditional warm pilot and s=0.01 cells; spec 5.5 table: about 94.8 h mean / 132.4 h p95"),
        cells=cells)
    out = os.path.join(ROOT, a.out)
    with open(out, "w") as f:
        json.dump(plan, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
    print(f"{len(cells)} cells, {n_runs} runs, mean {tot_mean / 3600:.2f} h, p95 {tot_p95 / 3600:.2f} h; "
          f"budget v2 match: {plan['budget_crosscheck']['v2_matches_frozen']}; sha256 {sha256_file(a.out)}")


if __name__ == "__main__":
    main()
