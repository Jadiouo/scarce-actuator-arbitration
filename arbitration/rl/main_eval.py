"""Frozen-plan main evaluation: validation selection, paired test CIs, and replication.

Protocol: docs/direction2/design/main-evaluation.md. No test seed constants live here;
the only entry to that split is final_eval.evaluate. Raw batches are saved atomically
before ledger registration, so interrupted reporting never needs another simulation.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import final_eval, hw, metrics, plan, policy, s1_eval, seeds, stop, vulns

SCHEMA = "direction2-main-eval/1"
EVAL_DIR = "results/direction2/main_eval"
MAIN_JSON = "results/direction2/main.json"
SNAP_FIELDS = ("util_b", "reg_b", "pen")


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path(root, cell, stage):
    return os.path.join(root, EVAL_DIR, f"{cell}__{stage}.json")


def code_hash():
    """Fingerprint code separately from the git commit (result-only commits are OK)."""
    root = Path(plan.REPO_ROOT)
    paths = sorted(list((root / "arbitration/rl").glob("*.py")) + list((root / "arbitration/gpu").glob("*.py"))
                   + [root / "scripts/run_quota.py"])
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def checkpoints(root, cell, settings):
    if cell["group"] != "main" or cell["run_ids"] != [0, 1, 2] or settings["obs_version"] != "public":
        raise ValueError("main evaluation requires the three registered public-observation runs")
    return [s1_eval.load_best_checkpoint(cell["name"], k, cell["G_gens"], root, settings["F"], settings["hidden"])
            for k in cell["run_ids"]]


def checkpoint_meta(cks):
    return [{k: c[k] for k in ("cell", "run_id", "gen", "G_val_train", "theta_sha256", "gens_done")} for c in cks]


def candidate_strategies():
    """36 public candidates + eight historical oracle names, in registered order.

    The four quota_savemin_oracle entries are honest no-ops under M3C (spec HW-01),
    and are represented by the honest adapter rather than unsupported quota columns.
    They are included only in the auxiliary 'including oracle' comparison.
    """
    public = hw.build_s_hw()
    oracle = [(n, {} if n.startswith("quota_") else dict(d)) for n, d in hw.all_strategies() if "oracle" in n]
    if len(public) != 36 or len(oracle) != 8:
        raise ValueError("registered handwritten candidate composition changed")
    return public, oracle


def make_runner(settings, dev="cuda", graph=None):
    def run(pols, seed_list, use_graph=False):
        out = vulns.simulate_vuln(settings["operator"], settings["knob"], settings["mech"], settings["T_val"], settings["cfg"],
                                  pols, seed_list, r=settings["r"], dev=dev, graph=graph if use_graph else None)
        snap = {k: out["snap"][k].detach().cpu().numpy() for k in SNAP_FIELDS}
        return dict(T_score=int(out["T_score"]), snap=snap)
    return run


def _validate_batch(raw, contract):
    if raw.get("contract") != contract:
        raise ValueError("cached batch contract mismatch: seeds, settings, policies, or code changed")
    n, ns = len(contract["policies"]), len(contract["seeds"])
    t, l = contract["settings"]["T_val"], int(contract["settings"]["cfg"]["L"])
    expected_ts = t - l if t > l else t
    if raw["T_score"] != expected_ts:
        raise ValueError("wrong scoring window in evaluation batch")
    for key, shape in (("util_b", (n, ns, 3)), ("reg_b", (n, ns)), ("pen", (n, ns, 3))):
        a = np.asarray(raw["snap"][key], dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid {key} snapshot: expected finite {shape}")
    return raw


def _batch(root, cell, stage, runner, settings, pols, ids, seed_list, use_graph, source_hash):
    contract = dict(schema=SCHEMA, cell=cell, stage=stage, seeds=[int(x) for x in seed_list],
                    settings={k: settings[k] for k in ("mech", "cfg", "r", "T_val", "obs_version", "operator", "knob")},
                    policies=ids, source_sha256=source_hash)
    dest = path(root, cell, stage)
    if os.path.exists(dest):
        return _validate_batch(read(dest), contract)
    t0 = time.time()
    out = runner(pols, list(seed_list), use_graph)
    raw = dict(contract=contract, T_score=int(out["T_score"]), snap={k: np.asarray(out["snap"][k], float).tolist() for k in SNAP_FIELDS},
               wall_s=time.time() - t0)
    _validate_batch(raw, contract)
    s1_eval._write_json(dest, raw)
    print(f"{cell} {stage}: saved {len(pols)} policies x {len(seed_list)} seeds in {raw['wall_s']:.1f}s", flush=True)
    return raw


def _rl_batch(root, cell, stage, runner, settings, cks, seed_list, graph, source_hash):
    pols = [policy.MLPPolicy(F=settings["F"], hidden=settings["hidden"], obs_version="public")]
    pols += [policy.MLPPolicy(F=settings["F"], hidden=settings["hidden"], theta=c["theta"], obs_version="public") for c in cks]
    ids = ["honest"] + [dict(run_id=c["run_id"], theta_sha256=c["theta_sha256"]) for c in cks]
    return _batch(root, cell, stage, runner, settings, pols, ids, seed_list, graph, source_hash)


def _hw_batch(root, cell, stage, runner, settings, strategies, seed_list, source_hash):
    names = ["honest"] + [n for n, _ in strategies]
    pols = [policy.AdapterPolicy({})] + [policy.AdapterPolicy(d) for _, d in strategies]
    raw = _batch(root, cell, stage, runner, settings, pols, names, seed_list, False, source_hash)
    return raw, names


def gains(raw):
    U = np.asarray(raw["snap"]["util_b"])[:, :, 1]
    return (U - U[0]) / raw["T_score"]


def _check_honest(rl, hand):
    if rl["T_score"] != hand["T_score"]:
        raise ValueError("RL and handwritten scoring windows differ")
    diffs = {}
    for k in SNAP_FIELDS:
        diffs[k] = s1_eval._check_honest(np.asarray(rl["snap"][k])[0], np.asarray(hand["snap"][k])[0])
    return diffs


def _select(g, names):
    return max(names, key=lambda n: float(np.mean(g[n])))   # first name wins ties


def validate_validation(val, cell, cks):
    if val.get("schema") != SCHEMA or val.get("stage") != "val" or val.get("cell") != cell["name"]:
        raise ValueError("validation record has wrong cell or schema")
    if val["checkpoints"] != checkpoint_meta(cks) or val["source_sha256"] != code_hash():
        raise ValueError("validation record does not match checkpoints or evaluator code")
    if digest({k: v for k, v in val.items() if k != "selection_sha256"}) != val["selection_sha256"]:
        raise ValueError("frozen validation selection/MDE record was modified")
    for r in val["runs"]:
        if any(not np.isfinite(r[k]) or r[k] < 0 for k in ("mde_frozen", "mde_d_frozen", "mde_d_plus_h0")):
            raise ValueError("invalid frozen MDE")
    return val


def validation_stage(root, cell, settings, cks, runner, use_graph=False, check_consistency=True):
    dest = path(root, cell["name"], "val")
    if os.path.exists(dest):
        return validate_validation(read(dest), cell, cks)
    vs = list(seeds.splits()["val"])
    seeds.require_split(vs, "val")
    code = code_hash()
    rl = _rl_batch(root, cell["name"], "val_rl_raw", runner, settings, cks, vs, use_graph, code)
    public, oracle = candidate_strategies()
    hand, names = _hw_batch(root, cell["name"], "val_hw_raw", runner, settings, public + oracle, vs, code)
    honest_diff = _check_honest(rl, hand)
    gr = gains(rl)
    gh = dict(zip(names, gains(hand)))
    pub_names = [n for n, _ in public]
    selected = dict(public=_select(gh, pub_names), plus_h0=_select(gh, pub_names + ["honest"]),
                    including_oracle=_select(gh, pub_names + [n for n, _ in oracle]))
    rows = []
    for i, c in enumerate(cks):
        g = gr[i + 1]
        delta = abs(float(g.mean()) - c["G_val_train"])
        if check_consistency and delta > s1_eval.TRAIN_VAL_RTOL * max(1e-3, abs(c["G_val_train"])):
            raise ValueError(f"run{c['run_id']}: recomputed validation mean differs from training: {delta}")
        d, dh = g - gh[selected["public"]], g - gh[selected["plus_h0"]]
        rows.append(dict(run_id=c["run_id"], G_val=metrics.mean_ci(g), D_val=metrics.mean_ci(d),
                         mde_frozen=metrics.mde(g, vs), mde_d_frozen=metrics.mde(d, vs), mde_d_plus_h0=metrics.mde(dh, vs),
                         G_val_recompute_absdiff=delta))
    chosen_run = max(cks, key=lambda c: (c["G_val_train"], -c["run_id"]))["run_id"]
    val = dict(schema=SCHEMA, stage="val", cell=cell["name"], r=cell["r"], checkpoints=checkpoint_meta(cks),
               selected_run=chosen_run, selected_hw=selected, hw_validation_means={n: float(g.mean()) for n, g in gh.items()},
               hw_mde_frozen={n: metrics.mde(gh[n], vs) for n in dict.fromkeys(selected.values())},
               runs=rows, T=settings["T_val"], T_score=rl["T_score"], source_sha256=code,
               honest_batches_maxdiff=honest_diff, git_sha=s1_eval._git_sha(root))
    val["selection_sha256"] = digest(val)
    s1_eval._write_json(dest, val)                         # written and frozen before any test entry
    return val


def system_metrics(raw, row):
    ts = raw["T_score"]
    reg = np.asarray(raw["snap"]["reg_b"])[row] / ts
    pen = np.asarray(raw["snap"]["pen"])[row] / ts
    return dict(regret_per_round=metrics.mean_ci(reg), mean_ineligible_fraction=metrics.mean_ci(pen.mean(-1)),
                ineligible_fraction_by_agent=[metrics.mean_ci(pen[:, j]) for j in range(3)])


def test_stage(root, cell, settings, cks, runner, val, use_graph=False, freeze_path=None, ledger_path=None):
    validate_validation(val, cell, cks)
    dest = path(root, cell["name"], "test")
    if os.path.exists(dest):
        result = read(dest)
        if result["selection_sha256"] != val["selection_sha256"]:
            raise ValueError("test result refers to another validation selection")
        return result
    raw_path = path(root, cell["name"], "test_raw")
    public, oracle = candidate_strategies()
    table = dict(public + oracle)
    wanted = list(dict.fromkeys(val["selected_hw"].values()))
    strategies = [(n, table[n]) for n in wanted if n != "honest"]
    ledger_path = ledger_path or os.path.join(root, "results/direction2/final_eval_ledger.json")
    freeze_path = freeze_path or os.path.join(root, "docs/direction2-freeze.md")
    cache = {}

    def raw_for(seed_list):
        if "raw" in cache:
            return cache["raw"]
        if os.path.exists(raw_path):
            raw = read(raw_path)
            if raw["selection_sha256"] != val["selection_sha256"] or raw["seeds"] != list(seed_list):
                raise ValueError("cached test raw data has different seeds or frozen selection")
        else:
            rl = _rl_batch(root, cell["name"], "test_rl_raw", runner, settings, cks, seed_list, use_graph, val["source_sha256"])
            hand, names = _hw_batch(root, cell["name"], "test_hw_raw", runner, settings, strategies, seed_list, val["source_sha256"])
            raw = dict(cell=cell["name"], seeds=[int(s) for s in seed_list], T_score=rl["T_score"],
                       selection_sha256=val["selection_sha256"], rl_gains=gains(rl)[1:].tolist(),
                       hw_gains=dict(zip(names, gains(hand).tolist())), honest_batches_maxdiff=_check_honest(rl, hand),
                       honest_system=system_metrics(rl, 0), rl_system=[system_metrics(rl, i + 1) for i in range(3)],
                       hw_system={n: system_metrics(hand, i) for i, n in enumerate(names)})
            raw["raw_sha256"] = digest(raw)
            s1_eval._write_json(raw_path, raw)             # durable before any ledger write
        if digest({k: v for k, v in raw.items() if k != "raw_sha256"}) != raw["raw_sha256"]:
            raise ValueError("cached test raw data checksum mismatch")
        for gs in raw["rl_gains"] + list(raw["hw_gains"].values()):
            if np.asarray(gs).shape != (32,) or not np.isfinite(gs).all():
                raise ValueError("invalid saved per-seed test gains")
        cache["raw"] = raw
        return raw

    def evaluator(index, baseline):
        def ev(seed_list):
            raw = raw_for(seed_list)
            if index is None:
                return dict(G_s=raw["hw_gains"][baseline])
            g = np.asarray(raw["rl_gains"][index])
            return dict(G_s=g, D_s=g - np.asarray(raw["hw_gains"][baseline]))
        return ev

    def evaluate_once(record, ev):
        if final_eval.already_evaluated(cell["name"], record["policy_id"], ledger_path):
            if not os.path.exists(raw_path):
                raise RuntimeError("ledger entry exists but durable raw data is missing; refusing another evaluation")
            old = read(raw_path)
            final_eval.require_test_seeds(old["seeds"])
            result = ev(old["seeds"])
            return metrics.final_eval(record, result["G_s"], result.get("D_s"))
        return final_eval.evaluate(record, ev, freeze_path=freeze_path, ledger_path=ledger_path)

    rows = []
    for i, c in enumerate(cks):
        v = val["runs"][i]
        pid = f"run{c['run_id']}-g{c['gen']}-{c['theta_sha256'][:16]}"
        record = dict(cell=cell["name"], policy_id=pid, mde_frozen=v["mde_frozen"], mde_d_frozen=v["mde_d_frozen"])
        out = evaluate_once(record, evaluator(i, val["selected_hw"]["public"]))
        raw = cache["raw"]
        gs = np.asarray(raw["rl_gains"][i])
        dh = gs - np.asarray(raw["hw_gains"][val["selected_hw"]["plus_h0"]])
        ch = metrics.mean_ci(dh)
        rows.append(dict(run_id=c["run_id"], checkpoint=checkpoint_meta([c])[0], G_RL=out["G"], D=out["D"],
                         MDE_G=v["mde_frozen"], MDE_D=v["mde_d_frozen"], found_G=out["found_G"], found_D=out["found_D"],
                         D_plus_h0=ch, MDE_D_plus_h0=v["mde_d_plus_h0"],
                         found_D_plus_h0=metrics.found_rule(ch["mean"], ch["lo"], v["mde_d_plus_h0"]),
                         system=raw["rl_system"][i]))
    for name in wanted:
        # Baseline ledger thresholds are their OWN validation G MDEs; not the RL MDE.
        record = dict(cell=cell["name"], policy_id="hw-" + name, mde_frozen=val["hw_mde_frozen"][name])
        evaluate_once(record, evaluator(None, name))
    raw = cache["raw"]
    ds = [np.asarray(g) - np.asarray(raw["hw_gains"][val["selected_hw"]["public"]]) for g in raw["rl_gains"]]
    dhs = [np.asarray(g) - np.asarray(raw["hw_gains"][val["selected_hw"]["plus_h0"]]) for g in raw["rl_gains"]]
    result = dict(schema=SCHEMA, stage="test", cell=cell["name"], r=cell["r"], selected_run=val["selected_run"],
                  selected_hw=val["selected_hw"], runs=rows, replication_k=metrics.replication_count(ds),
                  replication_k_plus_h0=metrics.replication_count(dhs),
                  G_HW={key: dict(name=name, **metrics.mean_ci(raw["hw_gains"][name])) for key, name in val["selected_hw"].items()},
                  honest_system=raw["honest_system"], hw_system=raw["hw_system"],
                  T=val["T"], T_score=val["T_score"], seed_range=[min(raw["seeds"]), max(raw["seeds"])], n_seeds=len(raw["seeds"]),
                  selection_sha256=val["selection_sha256"], raw_sha256=raw["raw_sha256"], source_sha256=val["source_sha256"],
                  honest_batches_maxdiff=raw["honest_batches_maxdiff"], git_sha=s1_eval._git_sha(root))
    s1_eval._write_json(dest, result)
    return result


def assemble(root, run_plan):
    cells = []
    for pc in [c for c in run_plan["cells"] if c["group"] == "main"]:
        p = path(root, pc["name"], "test")
        if not os.path.exists(p):
            raise RuntimeError(f"formal evaluation incomplete: {pc['name']}")
        cell = read(p)
        cell["exploratory"] = pc["r"] != .5
        cells.append(cell)
    s1 = read(os.path.join(root, "results/direction2/s1.json"))
    primary = next(c for c in cells if c["r"] == .5)
    chosen = next(r for r in primary["runs"] if r["run_id"] == primary["selected_run"])
    gate = bool(s1["gate"]["S1_pass"])
    d = chosen["D"]
    outcome = stop.classify(gate, d["mean"], d["lo"], d["hi"], chosen["MDE_D"], primary["replication_k"])
    dh = chosen["D_plus_h0"]
    aux = stop.classify(gate, dh["mean"], dh["lo"], dh["hi"], chosen["MDE_D_plus_h0"], primary["replication_k_plus_h0"])
    existing = []
    for p in sorted(Path(root, "results/direction2/s1_eval").glob("*__val.json")):
        old = read(p)
        for cell in cells:
            for run in cell["runs"]:
                if old["checkpoint"]["theta_sha256"] == run["checkpoint"]["theta_sha256"]:
                    existing.append(dict(main_cell=cell["cell"], run_id=run["run_id"], previous_cell=old["cell"],
                                         theta_sha256=old["checkpoint"]["theta_sha256"],
                                         previous_test_record_exists=os.path.exists(str(p).replace("__val.json", "__test.json"))))
    doc = dict(schema="direction2-main/1", spec="docs/direction2-spec.md v1.3", cells=cells, S1_gate=s1["gate"],
               verdict=dict(row=outcome.row, label=outcome.label, incomplete=outcome.incomplete, selected_run=primary["selected_run"],
                            replication_k=primary["replication_k"], replication_definition="count of runs with paired D CI lower bound > 0"),
               verdict_plus_h0=dict(row=aux.row, label=aux.label), prior_identical_policies=existing,
               limitations=["S1 passed its registered rule, but the post-hoc off-knob diagnosis supports exploitation in only 2/3 out-of-family cases.",
                            "D2 and O5 detection is confounded by baseline advantage; S1 calibration cannot support an unqualified detection-power claim.",
                            "The main r=0.5 run0 policy was previously evaluated in S1; it is not new evidence independent of S1.",
                            "r=0.9 is exploratory, underpowered at the registered training budget, and not calibrated by S1.",
                            "CIs cover environment-seed variation conditional on the chosen policy; the three runs share the same evaluation seeds.",
                            "The fixed public-observation policy class and budget yield an exploitability lower bound, not an optimum or a safety proof."],
               git_sha=s1_eval._git_sha(root))
    s1_eval._write_json(os.path.join(root, MAIN_JSON), doc)
    return doc
