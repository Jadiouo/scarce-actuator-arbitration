"""Direction 2 S1 vulnerability suite: designer_2 mutations (D1..D4), calibration and registry (spec s8, REQ-S1-01..07).

GPU parts must be run through gpujob (one GPU), each part < 25 min.  CPU parts run directly.
  python scripts/run_d2_vulns.py check                          -> results/direction2/vuln_checks.json            (CPU, small)
  python scripts/run_d2_vulns.py sweep OP [--seeds train|val]   -> results/direction2/vuln_parts/sweep_OP_<seeds>_<n>.json   (GPU)
  python scripts/run_d2_vulns.py val OP KNOB SIZE ...           -> validation measurement of one (vuln, knob)              (GPU)
  python scripts/run_d2_vulns.py anchors                        -> NAIVE / M4 anchors G*_ref on validation seeds          (GPU)
  python scripts/run_d2_vulns.py merge                          -> results/direction2/vuln_suite.json                     (CPU)
Calibration uses training seeds 0-63 only; the registry values come from the validation seeds 2000-2031 (never test seeds).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arbitration.rl import env, hw, policy, vulns  # noqa: E402
from arbitration.rl.metrics import mean_ci  # noqa: E402

OUT = os.path.join(ROOT, "results", "direction2")
PARTS = os.path.join(OUT, "vuln_parts")
TRAIN = list(range(64))
VAL = list(range(2000, 2032))
F64 = torch.float64
INF = float("inf")


# ----------------------------------------------------------------------------------------------------------------- CPU checks
def _eq(a, b, fields):
    return all(a["final"][f].equal(b["final"][f]) for f in fields)


FIELDS = ["reg_b", "util_b", "flags", "pen", "audits", "susp", "nraid"]
D_OFF = {"D1": 0.0, "D2": INF, "D3": 0.0, "D4": 0.0}
D_ON = {"D1": 0.5, "D2": 3.0, "D3": 0.4, "D4": 0.2}
D_TARGET = {"D1": "audit_prob", "D2": "release", "D3": "resid", "D4": "resid"}


def run_checks(out_path=None):
    res = dict(identity={}, effect={}, hook_level={}, rules_sandbox={}, per_element_knob={})
    pols = [policy.AdapterPolicy(d) for d in ({}, dict(dz=0.3), dict(b=0.1), dict(b=1.0), dict(dz=1.0))]
    rules = [vulns.RulePolicy("d2_parole", dict(dz=1.0)), vulns.RulePolicy("d2_rested", dict(dz=1.0)),
             vulns.RulePolicy("d2_slope", dict(s=0.4)), vulns.RulePolicy("d2_climb", dict(kappa=0.01))]
    seeds = [0, 1, 5000]
    for cfg_name, cfg, T in (("L300", dict(p=0.1, tol=6.0, L=300.0, kref=0.5), 3000), ("L2500", dict(p=0.1, tol=6.0, L=2500.0, kref=0.5), 6000)):
        base = env.simulate("M3C", T, cfg, pols, seeds, r=0.5, dev="cpu")
        for op in vulns.DESIGNER_OPERATOR_IDS:
            v = vulns.simulate_vuln(op, D_OFF[op], "M3C", T, cfg, pols, seeds, r=0.5, dev="cpu")
            res["identity"][f"{op}/{cfg_name}"] = bool(_eq(v, base, FIELDS))
            # a knob sequence (one entry per policy) with identity values must also be bitwise identical
            v = vulns.simulate_vuln(op, [D_OFF[op]] * len(pols), "M3C", T, cfg, pols, seeds, r=0.5, dev="cpu")
            res["per_element_knob"][f"{op}/{cfg_name}"] = bool(_eq(v, base, FIELDS))
        # rule policies at the identity knob equal the same rule run without any mutation (operator None through the same code path)
        allp = pols + rules
        ref = vulns.simulate_vuln(None, 0.0, "M3C", T, cfg, allp, seeds, r=0.5, dev="cpu")
        for op in vulns.DESIGNER_OPERATOR_IDS:
            v = vulns.simulate_vuln(op, D_OFF[op], "M3C", T, cfg, allp, seeds, r=0.5, dev="cpu")
            res["identity"][f"{op}/{cfg_name}/with_rules"] = bool(_eq(v, ref, FIELDS))
        for op in vulns.DESIGNER_OPERATOR_IDS:
            v = vulns.simulate_vuln(op, D_ON[op], "M3C", T, cfg, allp, seeds, r=0.5, dev="cpu")
            res["effect"][f"{op}/{cfg_name}"] = [f for f in FIELDS if not v["final"][f].equal(ref["final"][f])]
    # ---- hook-level isolation: only the target hook differs from the base mechanism, and it equals the designer formula
    cfg = dict(p=0.5, tol=6.0, L=100.0, kref=0.5)
    hp = [policy.AdapterPolicy(dict(b=1.0)), policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=0.3))] + rules
    for op in vulns.DESIGNER_OPERATOR_IDS:
        knob = D_ON[op]
        out = vulns.simulate_vuln(op, knob, "M3C", 2500, cfg, hp, [0, 1, 2, 3], r=0.5, dev="cpu", hook_log=True)
        log = out["hook_log"]
        rec = {}
        for h, base_fn in vulns.BASE_HOOKS.items():
            r_ = log[h]
            m = r_["mask"].bool()
            inp = {k: v for k, v in r_.items() if k not in ("mask", "out")}
            if h == D_TARGET[op]:
                exp = (vulns._d1_audit_prob(inp, knob) if op == "D1" else vulns._d3_resid(inp, knob) if op == "D3" else vulns._d4_resid(inp, knob))
                base = base_fn(inp)
                rec[h] = dict(target=True, equals_formula=bool(torch.equal(r_["out"][m], exp[m])), effective=bool((r_["out"][m] != base[m]).any()), calls=int(m.sum()))
            else:
                rec[h] = dict(target=False, equals_base=bool(torch.equal(r_["out"][m], base_fn(inp).to(r_["out"].dtype)[m])), calls=int(m.sum()))
        if op == "D2":
            r_ = log["release"]
            inp = {k: v for k, v in r_.items() if k not in ("mask", "out")}
            sus = inp["sus1"] > 0.5
            calm_new = vulns._calm_next(sus, inp["v1"], inp["calm1"])
            exp = torch.where(sus & (calm_new >= knob), inp["t"], inp["susp1"])
            rec["release"] = dict(target=True, equals_formula=bool(torch.equal(r_["out"], exp)), effective=bool((r_["out"] != inp["susp1"]).any()))
        else:
            assert "release" not in log
        res["hook_level"][op] = rec
    # ---- public-history rules: in-loop twin == sandbox rule on recorded runs
    cfg = dict(p=0.5, tol=6.0, L=60.0, kref=0.5)
    specs = [dict(kind="rule", name="d2_rested", params=dict(dz=1.0)), dict(kind="rule", name="d2_parole", params=dict(dz=1.0)),
             dict(kind="rule", name="d2_slope", params=dict(s=0.4, gain=1.0)), dict(kind="rule", name="d2_climb", params=dict(kappa=0.02, dz_max=4.0)),
             dict(kind="rule", name="streak_lie", params=dict(dz=1.0, m=2.0)), dict(kind="rule", name="late_lie", params=dict(dz=2.0, t0=100.0)),
             dict(kind="rule", name="launder", params=dict(dz=2.0, W=40.0, Z=20.0)),
             dict(kind="rule", name="d2_rested", params=dict(dz=2.0, dz0=0.3)), dict(kind="rule", name="d2_slope", params=dict(s=0.4, gain=1.5, dz0=0.3))]
    for op in ("D1", "D2", "D3", "D4", None):
        knob = D_ON[op] if op else 0.0
        out = vulns.simulate_vuln(op, knob, "M3C", 700, cfg, [vulns.build_ref_policy(s) for s in specs], [0, 1], r=0.5, dev="cpu", public_log=True)
        pl = out["public_log"].numpy()
        for i, sp in enumerate(specs):
            worst, susp_rounds = 0.0, 0
            for sd in range(pl.shape[1]):
                h = pl[i, sd]
                rep = vulns.run_ref_policy_sandbox(dict(ref_policy=sp, r=0.5), h)
                worst = max(worst, float(np.abs(rep - h[:, 1]).max()))
                susp_rounds += int(((np.arange(1, h.shape[0] + 1)) <= h[:, 3]).sum())
            res["rules_sandbox"][f"{sp['name']}{'+dz0' if 'dz0' in sp['params'] else ''}/{op}"] = dict(max_abs_diff=worst, suspended_rounds=susp_rounds)
    res["all_identity_ok"] = all(res["identity"].values()) and all(res["per_element_knob"].values())
    res["all_hook_level_ok"] = all(v.get("equals_formula", v.get("equals_base", False)) for d in res["hook_level"].values() for v in d.values())
    res["all_rules_ok"] = all(v["max_abs_diff"] < 1e-9 for v in res["rules_sandbox"].values())
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(out_path or os.path.join(OUT, "vuln_checks.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


# ----------------------------------------------------------------------------------------------------------------- candidate reference policies
ADAPT_GRID = [dict(kind="adapter", scols=d) for d in (
    [dict(dz=x) for x in (0.4, 0.6, 0.7, 0.8, 1.2, 1.4, 1.6, 1.8, 2.5, 3.0, 4.0)] + [dict(dzc=x) for x in (0.75, 1.25, 2.0, 3.0)])]
OPS = ("O1", "O2", "O3", "O6", "D1", "D2", "D3", "D4", "O4", "O5")
KNOB_COARSE = {
    "O1": [0.2, 0.4, 0.6, 0.8, 0.9, 1.0], "O2": [0.02, 0.05, 0.1, 0.2, 0.4, 0.6], "O3": [8.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0],
    "O6": [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99], "O4": [0.2, 0.4, 0.6, 0.8, 0.95], "O5": [50, 35, 25, 20, 15, 10, 5, 2],
    "D1": [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0], "D2": [2500, 1000, 300, 100, 30, 10, 3, 1], "D3": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "D4": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]}


DENSE_ADAPT = ([dict(kind="adapter", scols=dict(dz=round(0.3 + 0.1 * i, 2))) for i in range(38)] +
               [dict(kind="adapter", scols=dict(dzc=round(0.4 + 0.2 * i, 2))) for i in range(14)])


def cands_for(op, knob, mode="coarse"):
    """Candidate reference policies for (operator, knob); public information only, except the blind operators (O4, O5), which may use time."""
    R = lambda name, **p: dict(kind="rule", name=name, params=p)
    if mode == "hybrid" and op == "D1":
        return [R("d2_rested", dz=d, G=5.0, dz0=d0) for d in (1.5, 2.0, 3.0) for d0 in (0.2, 0.3, 0.4)]
    if mode == "hybrid2" and op == "D1":
        return [R("d2_rested", dz=d, G=5.0, dz0=d0) for d in (2.0, 2.5, 3.0, 4.0) for d0 in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)]
    if mode == "hybrid" and op == "D3":
        return [R("d2_slope", s=float(knob), gain=g, dz0=d0) for g in (1.0, 1.5, 2.0, 3.0) for d0 in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)]
    if mode == "dense2" and op == "O5":
        return [R("launder", dz=d, W=w, Z=float(knob)) for d in (0.5, 0.7, 1.0) for w in (70.0, 100.0, 150.0, 200.0, 300.0)]
    if mode == "dense" and op in ("O2", "O3", "O6"):
        return list(DENSE_ADAPT)
    if mode == "dense" and op == "O4":
        return [R("late_lie", dz=d, t0=t0) for d in (1.0, 2.0, 3.0) for t0 in (500.0, 1000.0, 2000.0)] + list(DENSE_ADAPT)
    if mode == "dense" and op == "O1":
        return [R("streak_lie", dz=d, m=3.0) for d in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)]
    if mode == "dense" and op == "D2":
        return [R("d2_parole", dz=d) for d in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)]
    if mode == "dense" and op == "D4":
        return [R("d2_climb", kappa=c * float(knob), dz_max=m) for c in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1) for m in (2.0, 3.0, 4.0)]
    if mode == "dense" and op == "O5":
        return [R("launder", dz=d, W=w, Z=float(knob)) for d in (0.7, 1.0, 1.5) for w in (30.0, 40.0, 50.0, 70.0, 100.0)] + list(ADAPT_GRID[:6])
    if op in ("O2", "O3", "O6"):
        return list(ADAPT_GRID)
    if op == "O1":
        return [R("streak_lie", dz=d, m=3.0) for d in (0.5, 1.0, 2.0, 3.0)]
    if op == "D1":
        return [R("d2_rested", dz=d, G=5.0) for d in (0.5, 1.0, 1.5, 2.0, 3.0)]
    if op == "D2":
        return [R("d2_parole", dz=d) for d in (0.5, 1.0, 2.0, 3.0)]
    if op == "D3":
        return [R("d2_slope", s=float(knob), gain=g) for g in (1.0, 1.25, 1.5, 2.0)]
    if op == "D4":
        return [R("d2_climb", kappa=c * float(knob), dz_max=m) for c in (0.005, 0.01, 0.02, 0.05) for m in (3.0, 4.0)]
    if op == "O4":
        return [R("late_lie", dz=d, t0=t0) for d in (1.0, 2.0, 3.0) for t0 in (500.0, 1000.0, 2000.0, 3000.0)] + list(ADAPT_GRID[:6])
    if op == "O5":
        return [R("launder", dz=d, W=w, Z=float(knob)) for d in (1.0, 2.0, 3.0) for w in (50.0, 100.0, 200.0, 400.0)] + list(ADAPT_GRID[:6])
    raise ValueError(op)


def _tolist(a):
    return np.round(np.asarray(a, dtype=float), 8).tolist()


def run_sweep(op, knobs, seeds_name, tag, cands_json=None, T=100000, max_elems=26000, mode="coarse"):
    seeds = TRAIN if seeds_name == "train" else (TRAIN + VAL if seeds_name == "both" else VAL)
    if seeds_name != "both" and os.path.exists(os.path.join(PARTS, f"{op}_{seeds_name}_{tag}.json")):
        print("part exists, skipped (resume):", f"{op}_{seeds_name}_{tag}.json")
        return
    fixed = json.load(open(cands_json)) if cands_json else None       # {str(knob): [spec, ...]}
    blocks = [dict(knob=float(k) if not (isinstance(k, float) and math.isinf(k)) else INF,
                   cands=(fixed[str(k)] if fixed else cands_for(op, k, mode))) for k in knobs]
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()
    per = (1 + max(len(b["cands"]) for b in blocks) + len(hw.build_s_hw())) * len(seeds)
    nb = max(1, max_elems // per)
    res = []
    for i in range(0, len(blocks), nb):
        chunk = blocks[i:i + nb]
        r = vulns.measure_blocks(op, chunk, seeds, r=0.5, T=T, dev="cuda")
        for b, x in zip(chunk, r):
            res.append(dict(knob=b["knob"], cands=b["cands"], G_cands=[_tolist(g) for g in x["G_cands"]], G_hw=[_tolist(g) for g in x["G_hw"]],
                            hw_names=x["hw_names"]))
        print(f"{op} {seeds_name}: {len(res)}/{len(blocks)} knobs  {time.time() - t0:.0f}s", flush=True)
    if seeds_name == "both":                        # one simulation, two part files (training / validation seeds are separate columns)
        for nm, sl, sd in (("train", slice(0, 64), TRAIN), ("val", slice(64, 96), VAL)):
            rr = [dict(r_, G_cands=[g[sl] for g in r_["G_cands"]], G_hw=[g[sl] for g in r_["G_hw"]]) for r_ in res]
            doc = dict(op=op, seeds=nm, seed_list=[sd[0], sd[-1]], T=T, r=0.5, results=rr, wall_s=time.time() - t0)
            name = os.path.join(PARTS, f"{op}_{nm}_{tag}.json")
            json.dump(doc, open(name, "w"))
            print("wrote", name)
        return
    doc = dict(op=op, seeds=seeds_name, seed_list=[seeds[0], seeds[-1]], T=T, r=0.5, results=res, wall_s=time.time() - t0)
    name = os.path.join(PARTS, f"{op}_{seeds_name}_{tag}.json")
    json.dump(doc, open(name, "w"))
    print("wrote", name)


def run_anchor_naive():
    """REQ-S1-08: NAIVE anchor (M3C, p=0, r=0.5), public constant lie dz=2.0, paired vs honest, validation seeds."""
    a = env.anchor_config("NAIVE")
    cfg = dict(p=a["p"], tol=a["tol"], L=a["L"], kref=a["kref"], ad_h=INF)
    t0 = time.time()
    out = env.simulate("M3C", 100000, cfg, [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=2.0))], VAL, r=a["r"], dev="cuda")
    U = out["snap"]["util_b"][:, :, 1].cpu().numpy().astype(float)
    G = (U[1] - U[0]) / out["T_score"]
    ci = mean_ci(G)
    doc = dict(spec="REQ-S1-08", mech="M3C", p=0.0, r=0.5, strategy="dz=2.0", T=100000, T_score=out["T_score"], seeds="val 2000-2031",
               G_star_ref=dict(mean=ci["mean"], lo=ci["lo"], hi=ci["hi"], half=ci["half"], sd=float(G.std(ddof=1))), json_ref=0.2793,
               json_ref_note="quota.json main[4].Gmax (reference only)", G_star_s=_tolist(G), wall_s=time.time() - t0)
    os.makedirs(PARTS, exist_ok=True)
    json.dump(doc, open(os.path.join(OUT, "measure_naive.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "G_star_s"}, indent=1))


def summarize(doc):
    """per knob: best candidate (largest mean G*), best handwritten, Delta (paired) with CI"""
    rows = []
    for r in doc["results"]:
        gc, gh = np.array(r["G_cands"]), np.array(r["G_hw"])
        ci = int(np.argmax(gc.mean(1)))
        hi = int(np.argmax(gh.mean(1)))
        d = mean_ci(gc[ci] - gh[hi])
        rows.append(dict(knob=r["knob"], cand=ci, spec=r["cands"][ci], G_star=float(gc[ci].mean()), hw_name=r["hw_names"][hi], G_hw=float(gh[hi].mean()),
                         delta=d["mean"], lo=d["lo"], hi=d["hi"]))
    return rows


def _train_rows(op, tags=None):
    rows = []
    for fn in sorted(os.listdir(PARTS)):
        if fn.startswith(op + "_train_") and not fn.endswith("identity.json") and not fn.split("_")[2].startswith("plan"):
            if tags and fn.split("_")[2][:-5] not in tags:
                continue
            rows += summarize(json.load(open(os.path.join(PARTS, fn))))
    return [r for r in rows if not math.isinf(r["knob"])][::-1]      # later (denser) parts win ties


def _rescale(op, spec, row_knob, knob):
    sp = json.loads(json.dumps(spec))
    if sp["kind"] == "rule":
        if op == "D3":
            sp["params"]["s"] = float(knob)
        elif op == "D4":
            sp["params"]["kappa"] = sp["params"]["kappa"] * knob / row_knob
        elif op == "O5":
            sp["params"]["Z"] = float(knob)
    return sp


def make_plan(op, knobs, tag="plan", from_tags=None):
    """Reference policy for each planned knob = the best candidate (on TRAINING seeds) of the nearest already-swept knob, rescaled to the knob."""
    rows = _train_rows(op, from_tags)
    plan = {}
    for k in knobs:
        k = int(k) if op in ("D2", "O5") else float(k)
        near = min(rows, key=lambda r: abs(r["knob"] - k))
        plan[str(k)] = [_rescale(op, near["spec"], near["knob"], k)]
    path = os.path.join(PARTS, f"plan_{op}_{tag}.json")
    json.dump(plan, open(path, "w"), indent=0)
    print("wrote", path, json.dumps(plan)[:300])


def run_analyze(path):
    doc = json.load(open(path))
    for row in summarize(doc):
        print(f"knob={row['knob']:<8g} Delta={row['delta']:+.5f} [{row['lo']:+.5f},{row['hi']:+.5f}]  G*={row['G_star']:.5f} HW={row['G_hw']:.5f} ({row['hw_name']})  ref={json.dumps(row['spec'])}")


# ----------------------------------------------------------------------------------------------------------------- merge -> vuln_suite.json
KIND = {"O1": "auto", "O2": "auto", "O3": "auto", "O6": "auto", "D1": "designer", "D2": "designer", "D3": "designer", "D4": "designer",
        "O4": "blind", "O5": "blind"}
SIZES_RUN = (0.01, 0.02)
D_NAMES = {"D1": "V1_RestedAuditDiscount", "D2": "V2_SuspensionParole", "D3": "V3_SlopeShrink", "D4": "V4_SelfCalibratingSlope"}


def _load(op, nm, tag):
    f = os.path.join(PARTS, f"{op}_{nm}_{tag}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def _row(doc, i):
    r = doc["results"][i]
    gc, gh = np.array(r["G_cands"]), np.array(r["G_hw"])
    hb = int(np.argmax(gh.mean(1)))
    return dict(knob=r["knob"], spec=r["cands"][0], G_star=mean_ci(gc[0]), G_hw_name=r["hw_names"][hb], G_hw_mean=float(gh[hb].mean()),
                delta=mean_ci(gc[0] - gh[hb]))


def run_merge(plans):
    """plans: {op: [tag, ...]} plan parts to look into (each has *_train_* and *_val_* files with identical knobs).  For every (op, size) the
    chosen knob = the planned knob whose VALIDATION delta lies in [0.8s, 1.2s] with CI lower bound > 0 and is closest to s."""
    entries, uncal = [], []
    checks = json.load(open(os.path.join(OUT, "vuln_checks.json")))
    for op, tags in plans.items():
        for s in SIZES_RUN:
            if KIND[op] == "blind" and s != 0.02:
                continue
            cand = []
            for tag in tags:
                dv, dt = _load(op, "val", tag), _load(op, "train", tag)
                if dv is None or dt is None:
                    continue
                for i in range(len(dv["results"])):
                    v, t = _row(dv, i), _row(dt, i)
                    cand.append((v, t, tag))
            ok = [(v, t, tag) for v, t, tag in cand if 0.8 * s <= v["delta"]["mean"] <= 1.2 * s and v["delta"]["lo"] > 0]
            if not ok:
                best = max(cand, key=lambda c: -abs(c[0]["delta"]["mean"] - s)) if cand else None
                uncal.append(dict(id=op, kind=KIND[op], size=s, source={"auto": "auto", "designer": "designer_2", "blind": "blind"}[KIND[op]], status="uncalibratable",
                                  nearest=None if best is None else dict(knob=best[0]["knob"], delta_val=best[0]["delta"], delta_train64=best[1]["delta"]["mean"])))
                continue
            v, t, tag = min(ok, key=lambda c: abs(c[0]["delta"]["mean"] - s))
            e = dict(id=op, kind=KIND[op], operator=op, knob=(int(v["knob"]) if op in ("D2", "O5") else v["knob"]), size=s, blind=KIND[op] == "blind", r=0.5, T=100000,
                     ref_policy=v["spec"], G_star_val=v["G_star"], G_HW_best_val=dict(name=v["G_hw_name"], mean=v["G_hw_mean"]),
                     delta_val=v["delta"], delta_train64=t["delta"]["mean"], delta_train64_ci=t["delta"], calibration_part=tag,
                     n_candidate_knobs_checked=len(cand))
            e["source"] = {"auto": "auto", "designer": "designer_2", "blind": "blind"}[KIND[op]]
            if op in D_NAMES:
                e["designer_name"] = D_NAMES[op]
            entries.append(e)
    return entries, uncal


def _max_train_delta(op, tags=None):
    rows = _train_rows(op, tags)
    if not rows:
        return None
    b = max(rows, key=lambda r: r["delta"])
    return dict(knob=b["knob"], delta_train64=b["delta"], lo=b["lo"], hi=b["hi"], ref_policy=b["spec"])


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def do_merge():
    plans = {op: ["plan_v1"] for op in ("D1", "D2", "D4", "O1", "O2", "O3", "O5")}
    entries, uncal = run_merge(plans)
    for op in ("D3", "O6", "O4"):
        for s in SIZES_RUN:
            if KIND[op] == "blind" and s != 0.02:
                continue
            uncal.append(dict(id=op, kind=KIND[op], size=s, source={"auto": "auto", "designer": "designer_2", "blind": "blind"}[KIND[op]], status="uncalibratable", reason="no tested knob reaches the band [0.8s,1.2s] (training seeds)",
                              best_train=_max_train_delta(op)))
    for op in ("B1", "B2"):
        uncal.append(dict(id=op, kind="blind", size=0.02, status="not_implemented",
                          reason="spec 8.2 gives only an example (absolute-time window / 600-round memory) and no designer definition was supplied"))
    for u in uncal:
        if u["id"] == "D1" or u["id"] == "D3":
            u["literal_best_train"] = _max_train_delta(u["id"], ["coarse"])
        if u["id"] == "D3":
            u["hybrid_best_train"] = _max_train_delta("D3", ["hybrid"])
    exact = {}
    for fn in sorted(os.listdir(PARTS)) if os.path.isdir(PARTS) else []:
        if fn.startswith("verify_") and fn.endswith(".json"):
            for r_ in json.load(open(os.path.join(PARTS, fn))):
                exact[(r_["id"], r_["size"])] = r_["exact"]
    for e in entries:
        x = exact.get((e["id"], e["size"]))
        if x is not None:
            e.update(G_star_val=x["G_star_val"], G_HW_best_val=x["G_HW_best_val"], delta_val=x["delta_val"], exact_from_recompute=True)
    naive = json.load(open(os.path.join(OUT, "measure_naive.json")))
    m4 = json.load(open(os.path.join(OUT, "measure_m4.json")))
    entries += [
        dict(id="NAIVE", kind="anchor", source="anchor", operator=None, knob=None, size=None, blind=False, r=0.5, T=100000,
             ref_policy=dict(kind="adapter", scols=dict(dz=2.0)), G_star_ref=naive["G_star_ref"], measured_in="results/direction2/measure_naive.json",
             definition="M3C, p=0, r=0.5 (REQ-S1-08); G rule, not the increment definition"),
        dict(id="M4rw0", kind="anchor", source="anchor", operator=None, knob=None, size=None, blind=False, r=0.99, T=100000,
             ref_policy=dict(kind="adapter", scols=dict(b=0.3)), G_star_ref=m4["G_star_ref"], measured_in="results/direction2/measure_m4.json",
             m4_anchor_status=m4["m4_anchor_status"], definition="M4, eps=0.2, L=20, rw=0, r=0.99 (REQ-S1-09)")]
    doc = dict(
        schema="REQ-S1-03: entries[] = (vulnerability, size) with id, kind, operator, knob, size, blind, ref_policy, G_star_val, G_HW_best_val, "
               "delta_val, delta_train64; anchors have G_star_ref instead; uncalibratable[] lists planned (vuln, size) pairs that could not be calibrated",
        git_sha=_git_sha(), generated=time.strftime("%Y-%m-%d"), sizes_run=list(SIZES_RUN), size_definition="Delta = G*_V - G_HW_best_V (paired, validation seeds 2000-2031)",
        entries=entries, uncalibratable=uncal,
        excluded=dict(designer_1=dict(reason="contaminated blind design (docs/direction2-blind-vulns.md); not read, not implemented, not counted")),
        checks=dict(file="results/direction2/vuln_checks.json", all_identity_ok=checks_flag("all_identity_ok"), all_hook_level_ok=checks_flag("all_hook_level_ok"),
                    all_rules_ok=checks_flag("all_rules_ok")),
        meta=META)
    path = os.path.join(OUT, "vuln_suite.json")
    json.dump(doc, open(path, "w"), indent=1)
    print("wrote", path, "entries", len(entries), "uncalibratable", len(uncal))
    for e in entries:
        if e["kind"] != "anchor":
            print(f"{e['id']:3s} s={e['size']} knob={e['knob']} dval={e['delta_val']['mean']:+.5f} [{e['delta_val']['lo']:+.5f},{e['delta_val']['hi']:+.5f}] "
                  f"G*={e['G_star_val']['mean']:.5f} [{e['G_star_val']['lo']:.5f},{e['G_star_val']['hi']:.5f}] HW={e['G_HW_best_val']['name']} {e['G_HW_best_val']['mean']:.5f} train={e['delta_train64']:+.5f}")
    for u in uncal:
        print("UNCAL", u["id"], u["size"], u["status"])


def run_verify(i0, i1):
    """T-39 logic (v1.3) on registry entries [i0, i1): recompute Delta_fine / Delta_coarse on val U val2 (64 seeds, 36-name S_HW) from per-seed gains,
    t(63) CI; the basis delta (fine for out_of_family/blind, coarse for in_family) must lie in [0.8s, 1.2s] with CI lower bound > 0."""
    from scipy import stats
    reg = [e for e in vulns.load_registry() if e["kind"] != "anchor" and e.get("category") not in ("anchor", "contaminated_excluded")][i0:i1]
    names_fine = [n for n, _ in hw.build_s_hw()]
    names_coarse = [n for n, _ in hw.build_s_hw_coarse()]
    tcrit = float(stats.t.ppf(0.975, 63))
    out = []

    def best_delta(g_star, by, names):
        means = {n: float(np.mean(by[n])) for n in names}
        best = max(names, key=lambda n: (means[n], -names.index(n)))
        d_s = g_star - np.asarray(by[best], dtype=float)
        m = float(d_s.mean())
        half = tcrit * float(d_s.std(ddof=1)) / math.sqrt(len(d_s))
        return best, m, m - half, m + half, means[best]

    for e in reg:
        res = vulns.recompute_delta(e)
        by = res["G_HW_s_by_name"]
        g_star = np.asarray(res["G_star_s"], dtype=float)
        bf, mf, lof, hif, hwf = best_delta(g_star, by, names_fine)
        bc, mc, loc, hic, _ = best_delta(g_star, by, names_coarse)
        cat = e.get("category")
        basis = "coarse" if cat == "in_family" else "fine"
        m, lo = (mc, loc) if basis == "coarse" else (mf, lof)
        s = e.get("size")
        rec = dict(id=e["id"], size=s, knob=e["knob"], category=cat, basis=basis, best_fine=bf, best_coarse=bc, delta_fine=mf, delta_coarse=mc, lo_basis=lo,
                   in_band=bool(s is not None and 0.8 * s <= m <= 1.2 * s), lo_pos=bool(lo > 0),
                   exact=dict(G_star_val=mean_ci(g_star), G_HW_best_val=dict(name=bf, mean=hwf), delta_fine=mean_ci(g_star - np.asarray(by[bf], dtype=float)),
                              delta_coarse=mean_ci(g_star - np.asarray(by[bc], dtype=float))))
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "exact"}), flush=True)
    json.dump(out, open(os.path.join(PARTS, f"verify_{i0}_{i1}.json"), "w"), indent=1)
    print("wrote verify part")


# ----------------------------------------------------------------------------------------------------------------- v1.3 recalibration (val U val2)
PARTS13 = os.path.join(OUT, "vuln_parts_v13")
CALIB = list(vulns.CALIB_SEEDS)                      # val 2000-2031 then val2 3000-3031 (REQ-S1-28); val2 is for calibration/selection only
CAT13 = {"D1": "out_of_family", "D2": "out_of_family", "O1": "out_of_family", "O2": "in_family", "O3": "in_family", "D4": "in_family",
         "D3": "in_family_known", "O5": "blind"}
SRC13 = {"D1": "designer_2", "D2": "designer_2", "D3": "designer_2", "D4": "designer_2", "O1": "auto", "O2": "auto", "O3": "auto", "O5": "blind"}
KIND13 = {"D1": "designer", "D2": "designer", "D3": "designer", "D4": "designer", "O1": "auto", "O2": "auto", "O3": "auto", "O5": "blind"}
BASIS13 = {"out_of_family": "fine", "blind": "fine", "in_family": "coarse"}


def _knob_val(op, k):
    return int(k) if op in ("D2", "O5") else float(k)


def run_recal(op, knobs, plan_path, tag):
    """One part file per call: the plan's reference policy of each knob measured on the 64 calibration seeds against honest + the 36 S_HW (one simulation).
    Resumable: an existing part is never recomputed."""
    os.makedirs(PARTS13, exist_ok=True)
    name = os.path.join(PARTS13, f"{op}_calib_{tag}.json")
    if os.path.exists(name):
        print("part exists, skipped (resume):", name)
        return
    plan = json.load(open(plan_path))
    blocks = [dict(knob=_knob_val(op, k), cands=plan[str(_knob_val(op, k))]) for k in knobs]
    t0 = time.time()
    res = vulns.measure_blocks(op, blocks, CALIB, r=0.5, T=100000, dev="cuda")
    out = []
    for b, x in zip(blocks, res):
        out.append(dict(knob=b["knob"], ref_policy=b["cands"][0], G_star=_tolist(x["G_cands"][0]), G_hw=[_tolist(g) for g in x["G_hw"]], hw_names=x["hw_names"]))
    doc = dict(op=op, seeds="val+val2", seed_list=CALIB, T=100000, r=0.5, results=out, wall_s=time.time() - t0)
    json.dump(doc, open(name, "w"))
    print("wrote", name, f"{time.time() - t0:.0f}s", flush=True)


def _best_deltas(g_star, g_hw, names):
    """(fine, coarse) per-seed paired increments against the best handwritten strategy (largest mean, ties -> earlier) of the 36-list / the 32 coarse names."""
    coarse = {n for n, _ in hw.build_s_hw_coarse()}
    gh = np.asarray(g_hw, dtype=float)
    bf = int(np.argmax(gh.mean(1)))
    ic = [i for i, n in enumerate(names) if n in coarse]
    bc = ic[int(np.argmax(gh[ic].mean(1)))]
    g = np.asarray(g_star, dtype=float)
    return dict(fine=g - gh[bf], coarse=g - gh[bc], best_fine=names[bf], best_coarse=names[bc], bf=bf, bc=bc)


def _calib_rows(op):
    rows = {}
    for fn in sorted(os.listdir(PARTS13)) if os.path.isdir(PARTS13) else []:
        if fn.startswith(op + "_calib_") and fn.endswith(".json"):
            for r in json.load(open(os.path.join(PARTS13, fn)))["results"]:
                rows[float(r["knob"])] = dict(r, part=fn)
    return rows


def select13(op, size):
    """Knob selection for (op, size) over all measured knobs of the op with the official v1.3 rule (s1.calibrate_knob_v13): basis delta in [0.8s, 1.2s]
    and paired t(63) CI lower bound > 0; closest to s wins.  Returns the s1 result plus the table of every knob measured."""
    from arbitration.rl import s1
    rows = _calib_rows(op)
    cat = CAT13[op]
    grid = sorted(rows)
    cache = {k: _best_deltas(rows[k]["G_star"], rows[k]["G_hw"], rows[k]["hw_names"]) for k in grid}
    res = s1.calibrate_knob_v13(op, size, cat, lambda k, sd: {"fine": cache[k]["fine"], "coarse": cache[k]["coarse"]}, grid, CALIB)
    table = []
    for k in grid:
        f, c = mean_ci(cache[k]["fine"]), mean_ci(cache[k]["coarse"])
        table.append(dict(knob=k, delta_fine=f["mean"], fine_lo=f["lo"], delta_coarse=c["mean"], coarse_lo=c["lo"], best_fine=cache[k]["best_fine"],
                          best_coarse=cache[k]["best_coarse"], part=rows[k]["part"]))
    return res, table, rows, cache


def run_finalize(items, tag):
    """GPU: for every chosen (op, knob) measure (a) the training-seed delta_train64 (train 0-63, the reference policy of the entry) and (b) the honest
    drift (util of the honest policy in the vulnerable environment minus the base environment, paired per seed, / T_score) on the 64 calibration seeds."""
    os.makedirs(PARTS13, exist_ok=True)
    name = os.path.join(PARTS13, f"finalize_{tag}.json")
    if os.path.exists(name):
        print("part exists, skipped (resume):", name)
        return
    t0 = time.time()
    out = []
    honest = [policy.AdapterPolicy({})]
    base = vulns.simulate_vuln(None, 0.0, "M3C", 100000, env.m3c_config(0.5), honest, CALIB, r=0.5, dev="cuda")
    ub = base["snap"]["util_b"][:, :, 1].cpu().numpy().astype(float)[0]
    for op, knob, spec in items:
        knob = _knob_val(op, knob)
        v = vulns.simulate_vuln(op, knob, "M3C", 100000, env.m3c_config(0.5), honest, CALIB, r=0.5, dev="cuda")
        uv = v["snap"]["util_b"][:, :, 1].cpu().numpy().astype(float)[0]
        drift = (uv - ub) / v["T_score"]
        r = vulns.measure_blocks(op, [dict(knob=knob, cands=[spec])], TRAIN, r=0.5, T=100000, dev="cuda")[0]
        d = _best_deltas(r["G_cands"][0], r["G_hw"], r["hw_names"])
        out.append(dict(op=op, knob=knob, honest_drift=_tolist(drift), delta_train64_fine=mean_ci(d["fine"]), delta_train64_coarse=mean_ci(d["coarse"]),
                        best_fine_train=d["best_fine"]))
        print(op, knob, "drift", mean_ci(drift)["mean"], "train fine", mean_ci(d["fine"])["mean"], "coarse", mean_ci(d["coarse"])["mean"], flush=True)
    json.dump(dict(items=out, wall_s=time.time() - t0), open(name, "w"))
    print("wrote", name, f"{time.time() - t0:.0f}s")


def run_anchors13():
    """NAIVE and M4 anchors re-measured on val U val2 (64 seeds, paired vs honest); the val-only values stay in measure_naive.json / measure_m4.json."""
    from arbitration.rl import s1
    os.makedirs(PARTS13, exist_ok=True)
    name = os.path.join(PARTS13, "anchors.json")
    if os.path.exists(name):
        print("part exists, skipped (resume):", name)
        return
    t0 = time.time()
    a = env.anchor_config("NAIVE")
    cfg = dict(p=a["p"], tol=a["tol"], L=a["L"], kref=a["kref"], ad_h=INF)
    o = env.simulate("M3C", 100000, cfg, [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(dz=2.0))], CALIB, r=a["r"], dev="cuda")
    U = o["snap"]["util_b"][:, :, 1].cpu().numpy().astype(float)
    gn = (U[1] - U[0]) / o["T_score"]
    m = s1.anchor_definitions()["M4"]
    o = env.simulate("M4", 100000, dict(eps=m["eps"], L=float(m["L"]), rw=m["rw"]), [policy.AdapterPolicy({}), policy.AdapterPolicy(dict(b=0.3))], CALIB, r=m["r"], dev="cuda")
    U = o["snap"]["util_b"][:, :, 1].cpu().numpy().astype(float)
    gm = (U[1] - U[0]) / o["T_score"]
    doc = dict(seeds="val+val2", NAIVE=dict(G_star_ref=mean_ci(gn), sd=float(gn.std(ddof=1)), G_star_s=_tolist(gn)),
               M4=dict(G_star_ref=mean_ci(gm), sd=float(gm.std(ddof=1)), G_star_s=_tolist(gm), m4_anchor_status=s1.m4_anchor_status(mean_ci(gm)["lo"])), wall_s=time.time() - t0)
    json.dump(doc, open(name, "w"))
    print(json.dumps({k: (v if k == "seeds" else {kk: vv for kk, vv in v.items() if kk != "G_star_s"}) for k, v in doc.items() if k != "wall_s"}, indent=1))


def run_verify13(items, tag):
    """GPU: `vulns.recompute_delta` (the T-39 computation: 64 seeds, 36-name S_HW, one entry per simulation) for every chosen entry.  These are the values
    the registry stores (identical to what T-39 recomputes).  items: [{'op','knob','spec'}]"""
    os.makedirs(PARTS13, exist_ok=True)
    name = os.path.join(PARTS13, f"verify_{tag}.json")
    if os.path.exists(name):
        print("part exists, skipped (resume):", name)
        return
    t0 = time.time()
    out = []
    for it in items:
        e = dict(operator=it["op"], knob=_knob_val(it["op"], it["knob"]), ref_policy=it["spec"], r=0.5, T=100000)
        res = vulns.recompute_delta(e)
        names = list(res["G_HW_s_by_name"])
        by = np.array([res["G_HW_s_by_name"][n] for n in names])
        d = _best_deltas(res["G_star_s"], by, names)
        out.append(dict(op=it["op"], knob=e["knob"], size=it.get("size"), G_star_s=[float(x) for x in res["G_star_s"]], best_fine=d["best_fine"], best_coarse=d["best_coarse"],
                        G_HW_fine_mean=float(by[d["bf"]].mean()), G_HW_coarse_mean=float(by[d["bc"]].mean()),
                        delta_fine_s=[float(x) for x in d["fine"]], delta_coarse_s=[float(x) for x in d["coarse"]]))
        print(it["op"], it.get("size"), e["knob"], "fine", float(d["fine"].mean()), "coarse", float(d["coarse"].mean()), d["best_fine"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(dict(items=out, wall_s=time.time() - t0), open(name, "w"))
    print("wrote", name)


# D3 (descriptive, not sized; REQ-S1-01): the strongest reference found on TRAINING seeds (hybrid d2_slope, max train delta), measured once on val U val2
D3_DESCRIPTIVE = (0.3, dict(kind="rule", name="d2_slope", params=dict(s=0.3, gain=1.0, dz0=0.5)))


def _load_parts13(prefix):
    out = []
    for fn in sorted(os.listdir(PARTS13)):
        if fn.startswith(prefix) and fn.endswith(".json"):
            out += json.load(open(os.path.join(PARTS13, fn)))["items"]
    return out


META13_DISCLOSURES = [
    "D1 uses a HYBRID reference policy (the literal designer strategy plus a constant lie dz0=0.2 in all other rounds); the constant lie was added to reach the registered sizes (to make the size), not part of the designer's text; the literal strategy alone never reaches 0.01 (v1.2 registry, uncalibratable[].literal_best_train).",
    "honest_drift: the honest policy's utility in the vulnerable environment minus the base environment (paired, per round); the honest benchmark drifts with the knob but the paired G is unaffected; recorded per entry.",
    "D4's knob response is not monotone (Delta rises, peaks and falls again as eta grows; knob_monotone is false for D4 and O3); the scan therefore covers the whole measured grid and the knob closest to the target among acceptors is chosen.",
    "Survivor bias: the suite keeps only vulnerabilities whose Delta can be calibrated above zero; D3 (Delta <= 0, in_family_known), O4 and O6 (cannot reach 0.01) were removed or left unsized and B1, B2 were never implemented; the suite is NOT a random sample of vulnerabilities.",
    "designer_1 is contaminated (it read the specification before designing); it is excluded, not read, not implemented, not counted.",
    "Blind-zone evidence is ONE example (O5); there is no B1/B2 (absolute-time, long-memory) evidence; statements are limited to a description of the single blind example O5.",
    "The fine handwritten grid (36 strategies: dz in {2.5, 3, 4} and b=1 added) makes the handwritten family stronger, so the comparison 'RL vs handwritten best' is conservative for RL.",
    "val2 (seeds 3000-3031) is used only for calibration and selection together with val (2000-2031); it is never a test set (REQ-SEED-08); the red team had already used these seeds for reconnaissance (gpujob #77-#79).",
    "n_S=256 does not cover the single-pair worst case (n_S about 565, REQ-OPT-13); this is NOT covered.",
    "The handwritten 'burst' strategies use the agent's own clock (REQ-HW-05) and the O5 reference policy `launder` uses the agent's own clock (REQ-S1-27 scope: blind zone); RL observations contain no time.",
    "in_family vulnerabilities (O2, O3, D4) are sized by Delta_coarse (best of the 32 coarse-grid strategies), not Delta_fine; their Delta_fine can be <= 0 (that is what in_family means); they are described, not counted in the pass criterion.",
    "Knob selection is done on the same 64 seeds on which the registry reports Delta (selection on the calibration seeds: the reported Delta of the chosen knob is in-sample); reference-policy parameters were chosen on TRAINING seeds 0-63 only; val2 was added against the winner's curse but does not remove it.",
    "D4's reference policy was changed in v1.3 (final) from the absolute-round `_LClimb` to the age since recovery max(since-L, 0) with `since` rebuilt from the public susp history; its reference parameters (kappa, dz_max) were re-selected on TRAINING seeds (D4_train_d4a/d4b parts) and the knob re-calibrated.",
    "D3 is measured descriptively once (knob 0.3, hybrid slope-shrink reference chosen on training seeds): Delta_fine and Delta_coarse are negative and not sized; no RL is run on D3.",
    "Honest-benchmark note: the 36-strategy best-handwritten choice is made on the same seeds as the reference policy; ties go to the earlier strategy.",
]


def do_merge13():
    from arbitration.rl import s1
    ver = {(v["op"], v["size"]): v for v in _load_parts13("verify_")}
    fin = {(f["op"], float(f["knob"])): f for f in _load_parts13("finalize_")}
    v12 = json.load(open(os.path.join(OUT, "vuln_suite_v12.json")))
    entries, uncal, calib_log = [], [], {}
    for op in ("D1", "D2", "O1", "O2", "O3", "D4", "O5"):
        for size in SIZES_RUN:
            if CAT13[op] == "blind" and size != 0.02:
                continue
            res, table, rows, cache = select13(op, size)
            calib_log[f"{op}/{size}"] = dict(status=res["status"], knob=res.get("knob"), n_knobs_measured=len(table), knob_monotone=res["knob_monotone"], table=table)
            if res["status"] != "ok":
                best = max(table, key=lambda t: -abs(t["delta_coarse" if CAT13[op] == "in_family" else "delta_fine"] - size))
                uncal.append(dict(id=op, category=CAT13[op], size=size, status="uncalibratable", n_knobs_measured=len(table), nearest=best))
                continue
            knob = _knob_val(op, res["knob"])
            v, f = ver[(op, size)], fin[(op, float(knob))]
            cat = CAT13[op]
            basis = BASIS13[cat]
            g = np.asarray(v["G_star_s"])
            df, dc = np.asarray(v["delta_fine_s"]), np.asarray(v["delta_coarse_s"])
            db = df if basis == "fine" else dc
            gs, ghm = mean_ci(g), v["G_HW_fine_mean"]
            dfi, dco = mean_ci(df), mean_ci(dc)
            dval = dict(dfi if basis == "fine" else dco)
            e = dict(id=op, kind=KIND13[op], category=cat, source=SRC13[op], operator=op, knob=knob, size=size, blind=(cat == "blind"), size_basis=basis, r=0.5, T=100000,
                     ref_policy=rows[float(knob)]["ref_policy"], G_star_val=gs, G_HW_best_val=dict(name=v["best_fine"], mean=ghm), G_HW_best_coarse_val=dict(name=v["best_coarse"], mean=v["G_HW_coarse_mean"]),
                     ratio_HW_over_Gstar=ghm / gs["mean"], delta_fine=dfi, delta_coarse=dco, delta_val=dval, delta_val_only=float(db[:32].mean()), delta_val2_only=float(db[32:].mean()),
                     calib_seeds="val+val2", honest_drift=mean_ci(np.asarray(f["honest_drift"])), knob_monotone=res["knob_monotone"],
                     delta_train64=f["delta_train64_" + basis]["mean"], delta_train64_ci=f["delta_train64_" + basis], n_knobs_measured=len(table),
                     calibration_part=rows[float(knob)]["part"], in_band=bool(0.8 * size <= dval["mean"] <= 1.2 * size), ci_lo_positive=bool(dval["lo"] > 0))
            if op in D_NAMES:
                e["designer_name"] = D_NAMES[op]
            entries.append(e)
    # D3: descriptive, never sized
    v, f = ver[("D3", None)], fin[("D3", float(D3_DESCRIPTIVE[0]))]
    g = np.asarray(v["G_star_s"])
    gs, ghm = mean_ci(g), v["G_HW_fine_mean"]
    entries.append(dict(id="D3", kind="designer", category="in_family_known", source="designer_2", operator="D3", knob=D3_DESCRIPTIVE[0], size=None, blind=False, size_basis=None, r=0.5, T=100000,
                        ref_policy=D3_DESCRIPTIVE[1], G_star_val=gs, G_HW_best_val=dict(name=v["best_fine"], mean=ghm), G_HW_best_coarse_val=dict(name=v["best_coarse"], mean=v["G_HW_coarse_mean"]),
                        ratio_HW_over_Gstar=ghm / gs["mean"], delta_fine=mean_ci(np.asarray(v["delta_fine_s"])), delta_coarse=mean_ci(np.asarray(v["delta_coarse_s"])), delta_val=None,
                        calib_seeds="val+val2", honest_drift=mean_ci(np.asarray(f["honest_drift"])), knob_monotone=None, delta_train64=f["delta_train64_fine"]["mean"],
                        designer_name=D_NAMES["D3"], descriptive_only=True))
    an = json.load(open(os.path.join(PARTS13, "anchors.json")))
    naive, m4 = json.load(open(os.path.join(OUT, "measure_naive.json"))), json.load(open(os.path.join(OUT, "measure_m4.json")))
    entries += [
        dict(id="NAIVE", kind="anchor", category="anchor", source="anchor", operator=None, knob=None, size=None, blind=False, r=0.5, T=100000, ref_policy=dict(kind="adapter", scols=dict(dz=2.0)),
             G_star_ref=an["NAIVE"]["G_star_ref"], G_star_ref_val_only=naive["G_star_ref"], calib_seeds="val+val2", measured_in="results/direction2/vuln_parts_v13/anchors.json",
             definition="M3C, p=0, r=0.5 (REQ-S1-08); G rule, not the increment definition"),
        dict(id="M4rw0", kind="anchor", category="anchor", source="anchor", operator=None, knob=None, size=None, blind=False, r=0.99, T=100000, ref_policy=dict(kind="adapter", scols=dict(b=0.3)),
             G_star_ref=an["M4"]["G_star_ref"], G_star_ref_val_only=m4["G_star_ref"], m4_anchor_status=an["M4"]["m4_anchor_status"], calib_seeds="val+val2",
             measured_in="results/direction2/vuln_parts_v13/anchors.json", definition="M4, eps=0.2, L=20, rw=0, r=0.99 (REQ-S1-09)"),
        dict(id="designer_1", kind="contaminated", category="contaminated_excluded", source="designer_1", operator=None, knob=None, size=None, blind=False,
             status="excluded: contaminated blind design (it read the specification); not read, not implemented, not counted")]
    removed = [dict(id=u["id"], size=u["size"], status=u["status"] if u["status"] != "uncalibratable" else "uncalibratable (v1.2: cannot reach 0.01)", best_train=u.get("best_train"), reason=u.get("reason"))
               for u in v12["uncalibratable"] if u["id"] in ("O4", "O6", "B1", "B2")]
    chk = json.load(open(os.path.join(PARTS13, "vuln_checks_v13.json")))
    ident = json.load(open(os.path.join(PARTS13, "identity_T72.json"))) if os.path.exists(os.path.join(PARTS13, "identity_T72.json")) else None
    o4 = [u for u in v12["uncalibratable"] if u["id"] == "O4"][0]["best_train"]["delta_train64"]
    o6 = [u for u in v12["uncalibratable"] if u["id"] == "O6"][0]["best_train"]["delta_train64"]
    doc = dict(
        schema="REQ-S1-03 (v1.3): entries[] = (vulnerability, size) with id, kind, category, source, operator, knob, size, blind, size_basis, ref_policy, G_star_val, G_HW_best_val (36-list), "
               "ratio_HW_over_Gstar, delta_fine, delta_coarse, delta_val (= basis delta), delta_val_only, delta_val2_only, calib_seeds, honest_drift, knob_monotone, delta_train64; anchors carry G_star_ref",
        version="v1.3", git_sha=_git_sha(), generated=time.strftime("%Y-%m-%d"), sizes_run=list(SIZES_RUN),
        size_definition="Delta = G*_V - G_HW_best_V, paired per seed, on val U val2 (64 seeds); out_of_family and blind: best of 36 (fine); in_family: best of 32 (coarse)",
        calib_seeds=dict(name="val+val2", n_seeds=64, val=[2000, 2031], val2=[3000, 3031], order="val then val2", t_df=63),
        acceptance="basis Delta in [0.8 s, 1.2 s] and paired t(63) 95% CI lower bound > 0",
        entries=entries, uncalibratable=uncal, removed_from_suite=removed, calibration_log=calib_log,
        identity=dict(checks_file="results/direction2/vuln_parts_v13/vuln_checks_v13.json", all_identity_ok=chk["all_identity_ok"], all_hook_level_ok=chk["all_hook_level_ok"],
                      all_rules_ok=chk["all_rules_ok"], long_horizon_T1e5_O_series=ident),
        o4_max=o4, o6_max=o6, honest_drift_summary="/".join(f"{e['id']}:{e['honest_drift']['mean']:+.4f}" for e in entries if e["id"] in ("D1", "D2", "D4") and e["size"] == 0.02),
        meta=dict(disclosures=META13_DISCLOSURES, designer_mapping=META["designer_mapping"], interpretations=META["interpretations"],
                  calibration="v1.3: 64 seeds val 2000-2031 + val2 3000-3031 (REQ-S1-28), T=100000, r=0.5, M3C {p=0.1,tol=6,L=2500,kref=0.5}; reference-policy parameters chosen on TRAINING seeds 0-63; knob chosen on the 64 calibration seeds",
                  handwritten="S_HW = 36 strategies of hw.build_s_hw() (fine); the 32-strategy list is the coarse list (in_family sizing)",
                  files=dict(parts="results/direction2/vuln_parts_v13/", previous="results/direction2/vuln_suite_v12.json")))
    path = os.path.join(OUT, "vuln_suite.json")
    json.dump(doc, open(path, "w"), indent=1)
    print("wrote", path, "entries", len(entries), "uncalibratable", [(u["id"], u["size"]) for u in uncal])
    for e in entries:
        if e.get("size") is not None or e["id"] == "D3":
            print(f"{e['id']:3s} {e['category']:15s} s={e['size']} knob={e['knob']} fine={e['delta_fine']['mean']:+.5f} [{e['delta_fine']['lo']:+.5f},{e['delta_fine']['hi']:+.5f}] "
                  f"coarse={e['delta_coarse']['mean']:+.5f} [{e['delta_coarse']['lo']:+.5f},{e['delta_coarse']['hi']:+.5f}] G*={e['G_star_val']['mean']:.5f} HW={e['G_HW_best_val']['name']}")


def checks_flag(k):
    try:
        return json.load(open(os.path.join(OUT, "vuln_checks.json")))[k]
    except Exception:
        return None


META = dict(
    designer_mapping="D1..D4 = V1..V4 of docs/direction2-blind-vulns-2.md (designer_2): V1 RestedAuditDiscount, V2 SuspensionParole, V3 SlopeShrink, V4 SelfCalibratingSlope",
    interpretations=[
        "env.py uses the true slope r_true as the audit slope (frontier's r_as branch). V3 therefore shrinks r_true: rh = r_true*(1-s); V4's rh_a starts at r_true (the 'prior' r_as).",
        "V4 (conservative reading of 'online calibration'): rh_a[B,K] is per agent, initial value r_true, NLMS step eta exactly as written (zv*err/(1+zv^2), clamp [0,0.995]), updated on every eligible audit of that agent, reset to r_true when that agent is flagged; honest agents are calibrated too (as written). eta is the knob (eta=0: rh_c=rh, rh_a never read or written).",
        "V1: lose_run is kept for all K agents and the audit discount applies to whoever wins the round (the mutation is on the mechanism's audit assignment, not on agent 1 only); suspended rounds do not count; a flag zeroes the flagged agent's counter; G=5 fixed.",
        "V2: calm/parole kept for all K agents; the release hook runs right after pen += ~elig and before the audit settlement; Q integer, Q=+inf is the identity; Q>=L has no effect (natural release first).",
        "Hooks: designer mutations act on the 5 existing hooks (V1 audit_prob, V3/V4 resid) plus one new hook 'release' (V2, logged only when present); they carry an extension object for state and set VulnOperator.full=True (hook output applies to every agent). O1..O6 code paths are untouched.",
        "Reference policies are 'rule' policies: a public-history function (PUBLIC_RULES, runs in the sandbox) plus an in-loop vectorised twin (LOOP_RULES) used by the simulator; scripts/run_d2_vulns.py check verifies they agree to 1e-9 on recorded runs. The sandbox now keeps the own suspension end susp1 of the current round (public at the round start, obs: susp_minus_t); v1 and won1 of the current round stay hidden.",
        "Free parameters of the reference policies (lie sizes dz, kappa/eta ratio, dz_max, W) were chosen on TRAINING seeds 0-63 only (grid search, the winner per knob); validation seeds were used to measure and to choose among 4 neighbouring knobs the one whose validation delta lies in [0.8s,1.2s] with CI lower bound > 0.",
        "Size = Delta = G*_V - G_HW_best_V per REQ-S1-02. Because Delta is measured against the best of the 32 coarse-grid handwritten strategies, Delta(knob) is not always monotone (e.g. O3, O2 near 0.25 where the HW grid point dz=2.0 is optimal, D4 where Delta FALLS as eta grows because HW strategies also learn to exploit larger eta).",
        "Only the sizes s in {0.01, 0.02} were calibrated (task instruction; spec 8.4 gate set G_s); 0.003 and 0.005 are not in the registry, blind vulnerabilities only at 0.02. T-39 (>=48 entries) and T-42 (4 sizes) therefore cannot pass on this registry by construction.",
        "D1 and D3: the literal designer strategy (honest except one lie after G losses / lie only if z>0) never reaches the target sizes (see uncalibratable[].literal_best_train). D1 uses the hybrid 'literal strategy + constant lie dz0 in all other rounds' (rule d2_rested with dz0>0); D3 remained uncalibratable even with a hybrid (dz0 + z-proportional lie).",
    ],
    calibration="training seeds 0-63 (sweeps in results/direction2/vuln_parts/*_train_*.json), validation 2000-2031, T=100000, r=0.5, M3C {p=0.1,tol=6,L=2500,kref=0.5}",
    handwritten="S_HW = 32 strategies of hw.build_s_hw(), best per environment on the same seeds",
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("op", nargs="?")
    ap.add_argument("--knobs", default=None)
    ap.add_argument("--seeds", default="train")
    ap.add_argument("--tag", default="coarse")
    ap.add_argument("--cands-json", default=None)
    ap.add_argument("--T", type=int, default=100000)
    ap.add_argument("--mode", default="coarse")
    ap.add_argument("--from-tags", default=None)
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--items", default=None)
    a = ap.parse_args()
    if a.parts_dir:
        PARTS = os.path.join(OUT, a.parts_dir)
    if a.cmd == "check":
        run_checks(os.path.join(PARTS13, "vuln_checks_v13.json") if a.parts_dir else None)
    elif a.cmd == "sweep":
        ks = [float(x) for x in a.knobs.split(",")] if a.knobs else KNOB_COARSE[a.op]
        ks = [int(k) if (a.op in ("D2", "O5") and not math.isinf(k)) else k for k in ks]
        run_sweep(a.op, ks, a.seeds, a.tag, a.cands_json, a.T, mode=a.mode)
    elif a.cmd == "anchors":
        run_anchor_naive()
    elif a.cmd == "mkplan":
        make_plan(a.op, [float(x) for x in a.knobs.split(",")], a.tag, a.from_tags.split(",") if a.from_tags else None)
    elif a.cmd == "verify":
        run_verify(int(a.knobs.split(",")[0]), int(a.knobs.split(",")[1]))
    elif a.cmd == "merge":
        do_merge()
    elif a.cmd == "analyze":
        run_analyze(a.op)
    elif a.cmd == "recal":
        run_recal(a.op, [float(x) for x in a.knobs.split(",")], a.plan, a.tag)
    elif a.cmd == "finalize":
        run_finalize([(i["op"], i["knob"], i["spec"]) for i in json.load(open(a.items))], a.tag)
    elif a.cmd == "merge13":
        do_merge13()
    elif a.cmd == "verify13":
        run_verify13(json.load(open(a.items)), a.tag)
    elif a.cmd == "anchors13":
        run_anchors13()
