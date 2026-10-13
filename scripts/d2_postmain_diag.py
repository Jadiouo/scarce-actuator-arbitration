#!/usr/bin/env python3
"""Post-main validation-only diagnostics; protocol docs/direction2/design/postmain-diagnostics.md.

No training, checkpoint selection or formal test/ledger writes. Original simulation
code is untouched. Resume requires identical protocol, code, seeds and weights.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import ablation, env, graphsim, hw, main_eval, obs_reconstruct, plan, policy, seeds
from scripts.d2_main_eval import backup

OUT = ROOT / "results/direction2/postmain"
PROTOCOL = ROOT / "docs/direction2/design/postmain-diagnostics.md"
GROUP_NAMES = ["current_type", "suspension", "recent_wins", "recent_reports", "residual", "slow_ema", "only_current_type"]
GROUPS = ablation.groups() + [list(range(2, 25))]
FIELDS = ("util_b", "reg_b", "pen", "flags", "audits")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def interval(x, comparisons=1):
    a = np.asarray(x, float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all():
        raise ValueError("interval requires a finite per-seed vector")
    half = stats.t.ppf(1 - .05 / (2 * comparisons), len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return dict(mean=float(a.mean()), lo=float(a.mean() - half), hi=float(a.mean() + half), half=float(half))


def masked(p, group):
    w, b, v, c = p.blocks()
    w = w.copy()  # blocks() exposes views; never mutate the frozen original policy.
    w[group] = 0
    return policy.MLPPolicy.from_blocks(w, b, v, c, obs_version="public")


def pol_id(p):
    return dict(theta_sha256=hashlib.sha256(p.theta.astype("<f8").tobytes()).hexdigest()) if isinstance(p, policy.MLPPolicy) else dict(scols=p.scols)


def contract(stage, settings, pols, names, sd, T):
    seeds.require_split(sd, "val")
    return dict(schema="postmain-diagnostics/1", stage=stage, protocol_sha256=sha(PROTOCOL),
                diagnostic_script_sha256=sha(__file__), simulator_sha256=main_eval.code_hash(),
                settings=settings, names=names, policies=[pol_id(p) for p in pols], seeds=list(sd), T=T)


def snapshot(out):
    return {k: out["snap"][k].detach().cpu().numpy().tolist() for k in FIELDS}


def cached(path, con):
    if not Path(path).exists():
        return None
    doc = read(path)
    if doc["contract"] != con or doc["sha256"] != digest({k: v for k, v in doc.items() if k != "sha256"}):
        raise ValueError("diagnostic cache contract/checksum mismatch")
    return doc


def save_raw(path, con, out, wall, **extra):
    doc = dict(contract=con, T_score=int(out["T_score"]), snap=snapshot(out), wall_s=wall, **extra)
    doc["sha256"] = digest(doc)
    write(path, doc)
    return doc


def check_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.allclose(a, b, rtol=1e-9, atol=0):
        raise AssertionError("normal/honest snapshot mismatch")
    return float(np.max(np.abs(a - b)))


def load_cell(pc):
    st = plan.cell_to_settings(pc)
    cks = main_eval.checkpoints(str(ROOT), pc, st)
    val = read(main_eval.path(str(ROOT), pc["name"], "val"))
    main_eval.validate_validation(val, pc, cks)
    pols = [policy.MLPPolicy(F=25, hidden=2, theta=c["theta"], obs_version="public") for c in cks]
    rl = read(main_eval.path(str(ROOT), pc["name"], "val_rl_raw"))
    hand = read(main_eval.path(str(ROOT), pc["name"], "val_hw_raw"))
    return st, cks, val, pols, rl, hand


def tail(pc):
    st, cks, val, pols, rl, hand = load_cell(pc)
    selected = val["selected_run"]
    name = val["selected_hw"]["public"]
    j = hand["contract"]["policies"].index(name)
    chosen = [policy.AdapterPolicy({}), pols[selected], policy.AdapterPolicy(dict(hw.build_s_hw())[name])]
    names = ["honest", f"run{selected}", name]
    sd = list(seeds.splits()["val"])
    con = contract("tail_honest", st, chosen, names, sd, st["T_val"])
    dest = OUT / f"{pc['name']}__tail_raw.json"
    raw = cached(dest, con)
    if raw is None:
        start = time.time()
        out = env.simulate("M3C", st["T_val"], st["cfg"], chosen, sd, r=pc["r"], dev="cuda", diagnostic_variant="tail_honest")
        raw = save_raw(dest, con, out, time.time() - start)
    for key in ("util_b", "reg_b", "pen"):
        check_equal(raw["snap"][key][0], rl["snap"][key][0])
        check_equal(hand["snap"][key][0], rl["snap"][key][0])
    ts = raw["T_score"]
    assert ts == rl["T_score"] == hand["T_score"] == 97500
    normals = [main_eval.gains(rl)[selected + 1], main_eval.gains(hand)[j]]
    variant = main_eval.gains(raw)[1:]
    mdes = [val["runs"][selected]["mde_frozen"], val["hw_mde_frozen"][name]]
    rows = []
    for i, (normal, var, mde) in enumerate(zip(normals, variant, mdes)):
        d = interval(normal - var)
        rows.append(dict(policy=names[i + 1], r=pc["r"], cell=pc["name"], seeds=sd, split="validation",
                         T=st["T_val"], L=st["cfg"]["L"], T_score=ts, G_normal=float(normal.mean()), G_variant=float(var.mean()),
                         delta=d["mean"], ci=[d["lo"], d["hi"]], MDE_G_frozen=mde,
                         flag_lower_rule=bool(d["lo"] > 0 and d["lo"] >= mde),
                         flag_mean_rule=bool(d["lo"] > 0 and d["mean"] >= mde),
                         normal_per_seed=normal.tolist(), variant_per_seed=var.tolist(), delta_per_seed=(normal-var).tolist(),
                         raw_source=str(dest.relative_to(ROOT)), selection_sha256=val["selection_sha256"]))
    dnormal, dvariant = normals[0] - normals[1], variant[0] - variant[1]
    result = dict(rows=rows, D_normal=interval(dnormal), D_variant=interval(dvariant),
                  delta_D=interval(dnormal-dvariant), delta_D_per_seed=(dnormal-dvariant).tolist())
    write(OUT / f"{pc['name']}__tail.json", result)
    print(f"{pc['name']} tail: " + "; ".join(f"{x['policy']} delta={x['delta']:+.6f} CI={x['ci']}" for x in rows), flush=True)


def ablate(pc):
    st, cks, val, pols, rl, hand = load_cell(pc)
    sd = list(seeds.splits()["val"])
    batch, names, indices = [policy.MLPPolicy(obs_version="public")], ["honest"], []
    for i, p in enumerate(pols):
        start = len(batch)
        batch += [p] + [masked(p, g) for g in GROUPS]
        names += [f"run{i}/original"] + [f"run{i}/{n}" for n in GROUP_NAMES]
        indices.append(start)
    con = contract("ablation", st, batch, names, sd, st["T_val"])
    dest = OUT / "ablation_raw.json"
    raw = cached(dest, con)
    if raw is None:
        start = time.time()
        out = env.simulate("M3C", st["T_val"], st["cfg"], batch, sd, r=pc["r"], dev="cuda", graph=graphsim.from_env("cuda", force=True))
        raw = save_raw(dest, con, out, time.time()-start)
    check_equal(raw["snap"]["util_b"][0], rl["snap"]["util_b"][0])
    gs = main_eval.gains(raw)
    baseline = main_eval.gains(hand)[hand["contract"]["policies"].index(val["selected_hw"]["public"])]
    rows = []
    for i, start in enumerate(indices):
        for key in ("util_b", "reg_b", "pen"):
            check_equal(raw["snap"][key][start], rl["snap"][key][i+1])
        for k, name in enumerate(GROUP_NAMES):
            g = gs[start+k+1]
            drop = gs[start]-g
            adj = interval(drop, comparisons=21)
            rows.append(dict(run=i, group=name, indices=GROUPS[k], G=interval(g), D=interval(g-baseline),
                             drop=interval(drop), drop_simultaneous_95=adj,
                             material_dependency=bool(adj["lo"] > 0 and drop.mean() >= .005),
                             G_per_seed=g.tolist(), D_per_seed=(g-baseline).tolist(), drop_per_seed=drop.tolist()))
    write(ROOT / "results/direction2/postmain_ablation.json", dict(schema="postmain-ablation/1", split="validation", seeds=sd,
          r=pc["r"], T=st["T_val"], T_score=raw["T_score"], rows=rows, family_size=21, effect_threshold=.005,
          baseline=val["selected_hw"]["public"], original_G=[interval(gs[i]) for i in indices], raw_sha256=raw["sha256"]))
    print("ablation complete: " + ", ".join(f"run{i}: " + "/".join(x["group"] for x in rows if x["run"]==i and x["material_dependency"]) for i in range(3)), flush=True)


def behavior_summary(arrays, names, burn, stop):
    u, v, won, susp = [arrays[k][..., burn:stop] for k in ("u1", "v1", "won1", "susp1")]
    prev = np.concatenate([np.zeros_like(arrays["won1"][..., :1]), arrays["won1"][..., :-1]], -1)[..., burn:stop]
    elig = np.arange(burn+1, stop+1)[None, None, :] > susp
    rows = []
    for i, name in enumerate(names):
        groups = {"eligible": elig[i], "suspended": ~elig[i], "eligible_prev_win": elig[i] & prev[i], "eligible_prev_loss": elig[i] & ~prev[i]}
        groups.update({f"u_bin_{b}": elig[i] & (u[i] >= b/10) & (u[i] < (b+1)/10) for b in range(10)})
        for group, mask in groups.items():
            by_seed = []
            for j in range(u.shape[1]):
                m = mask[j]
                n = int(m.sum())
                d = (v-u)[i,j,m]
                by_seed.append(dict(count=n, mean_delta=float(d.mean()) if n else None,
                                    inflate_rate=float((d>1e-6).mean()) if n else None,
                                    underreport_rate=float((d < -1e-6).mean()) if n else None,
                                    win_rate=float(won[i,j,m].mean()) if n else None))
            d = (v-u)[i][mask]
            rows.append(dict(policy=name, group=group, count=int(mask.sum()), mean_delta=float(d.mean()) if len(d) else None,
                             inflate_rate=float((d>1e-6).mean()) if len(d) else None,
                             underreport_rate=float((d < -1e-6).mean()) if len(d) else None, by_seed=by_seed))
    return rows


def behavior(pc):
    import torch
    st, cks, val, pols, rl, hand = load_cell(pc)
    sd, T = list(seeds.splits()["val"])[:8], 20000
    hname = val["selected_hw"]["public"]
    names = ["honest", "run0", "run1", "run2", hname]
    batch = [policy.AdapterPolicy({})] + pols + [policy.AdapterPolicy(dict(hw.build_s_hw())[hname])]
    con = contract("behavior", st, batch, names, sd, T)
    dest = OUT / "behavior_raw.json"
    trace = OUT / "behavior_public_trace.npz"
    raw = cached(dest, con)
    if raw is None:
        start = time.time()
        out = env.simulate("M3C", T, st["cfg"], batch, sd, r=pc["r"], dev="cuda", diagnostic=True)
        public = env.extract_public_log(out)
        # Rebuild the complete public observation stream on the same device;
        # independent implementation, private simulator fields never enter it.
        rebuilt = obs_reconstruct.reconstruct(public)
        if not torch.equal(rebuilt, out["obs"]):
            raise AssertionError("public observation reconstruction is not exact")
        arrays = {k: v.detach().cpu().numpy() for k, v in public.items() if k != "consts"}
        tmp = trace.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, trace)
        raw = save_raw(dest, con, out, time.time()-start, consts=public["consts"], trace_sha256=sha(trace), reconstruction_bitwise_equal=True)
    if sha(trace) != raw["trace_sha256"]:
        raise ValueError("behavior trace checksum mismatch")
    with np.load(trace, allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    rows = behavior_summary(arrays, names, 2500, 17500)
    write(ROOT / "results/direction2/postmain_behavior.json", dict(schema="postmain-behavior/1", split="validation", seeds=sd,
          T=T, summary_rounds=[2501,17500], rows=rows, reconstruction_bitwise_equal=raw["reconstruction_bitwise_equal"],
          trace_sha256=raw["trace_sha256"], trace=str(trace.relative_to(ROOT))))
    print(f"behavior complete: {len(names)} policies x {len(sd)} seeds, full public trace {trace.stat().st_size/1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["run", "tail", "ablation", "behavior"])
    ap.add_argument("--backup", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".scratch").mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(1)
    with (ROOT / ".scratch/postmain.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan.verify_repo_state(str(ROOT))
        for p in [str(Path(__file__).relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT))]:
            if not plan.is_committed_and_clean(str(ROOT), p):
                raise RuntimeError(f"diagnostic entry/protocol must be committed: {p}")
        protected = [ROOT / "results/direction2/main.json", ROOT / "results/direction2/final_eval_ledger.json"]
        original = {str(p):sha(p) for p in protected}
        pl = plan.load_plan(str(ROOT / plan.DEFAULT_PLAN), str(ROOT / plan.DEFAULT_FREEZE))
        pcs = [c for c in pl["cells"] if c["group"] == "main"]
        for pc in pcs:
            if a.stage in ("run", "tail"):
                tail(pc)
                if a.backup:
                    backup([str(p.relative_to(ROOT)) for p in OUT.glob(pc["name"]+"__tail*.json")], f"Direction 2: validation tail diagnostic {pc['name']}")
        if a.stage in ("run", "tail"):
            write(ROOT / "results/direction2/trunc_bias.json", [r for pc in pcs for r in read(OUT/f"{pc['name']}__tail.json")["rows"]])
        primary = next(c for c in pcs if c["r"]==.5)
        if a.stage in ("run", "ablation"):
            ablate(primary)
        if a.stage in ("run", "behavior"):
            behavior(primary)
        assert original == {str(p):sha(p) for p in protected}, "formal result or ledger changed"
        if a.backup:
            paths = [str(p.relative_to(ROOT)) for p in OUT.glob("*") if p.suffix in (".json", ".npz")]
            paths += [str(p.relative_to(ROOT)) for p in (ROOT / "results/direction2").glob("postmain_*.json")]
            if (ROOT / "results/direction2/trunc_bias.json").exists():
                paths.append("results/direction2/trunc_bias.json")
            backup(paths, "Direction 2: post-main ablations and public behavior diagnostics")


if __name__ == "__main__":
    main()
