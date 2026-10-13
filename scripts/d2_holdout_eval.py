#!/usr/bin/env python3
"""Additional independent-environment holdout; never alters the original test gateway."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arbitration.rl import env, graphsim, hw, main_eval, plan, policy, seeds
from scripts.d2_main_eval import backup

BASE = "d0fe4164f3f35e0b9bcf5ae2604489c10195f2c1"
OUT = ROOT / "results/direction2/holdout"
PROTOCOL = ROOT / "docs/direction2/design/independent-holdout.md"
MANIFEST = OUT / "manifest.json"
VAL = ROOT / "results/direction2/main_eval/main_M3C_r0.5__val.json"
OFFSETS = (0, 10**6, 2*10**6)
CORE_HASH = "5c5ac62595a7aa24e595885adb12f65bcba505eabc2932ab6643ffc39e6c1ee2"
ENTRY_FILES = ["scripts/d2_holdout_eval.py", "scripts/d2_holdout_launch.sh", "scripts/d2_main_eval.py",
               "tests/direction2/test_d2_holdout.py", "docs/direction2/design/independent-holdout.md"]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def encode_settings(settings):
    def sentinel(name):
        if name not in ("Infinity", "-Infinity"):
            raise ValueError("NaN in configuration")
        return {"float_sentinel": name}
    return json.loads(json.dumps(settings), parse_constant=sentinel)


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def interval(x, family=1):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite per-seed vector")
    mean = float(x.mean())
    half = float(stats.t.ppf(1-.05/(2*family), len(x)-1)*x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=mean, lo=mean-half, hi=mean+half, half=half)


def verdict(ci, delta):
    if ci["lo"] > delta:
        return "confirmed_material_advantage"
    if ci["lo"] > 0:
        return "confirmed_positive_only"
    if ci["hi"] < delta:
        return "material_advantage_ruled_out"
    return "inconclusive"


def require_disjoint(new, historical):
    if len(set(new)) != len(new) or any(type(x) is not int or x < 0 for x in new):
        raise ValueError("invalid or duplicate seeds")
    streams = [s+o for s in new for o in OFFSETS]
    if len(set(streams)) != len(streams):
        raise ValueError("new generator streams collide")
    # Historical engines may have used a fourth stream. Treat every saved seed
    # value conservatively as a possible base, including optimizer/test seed fields.
    old = {s+o for s in historical for o in (*OFFSETS, 3*10**6)}
    if set(streams) & old:
        raise ValueError("holdout generator stream overlaps history")


def seed_values(obj):
    values = set()
    def collect(x):
        if type(x) is int:
            values.add(x)
        elif isinstance(x, list):
            for y in x:
                collect(y)
        elif isinstance(x, dict):
            for y in x.values():
                collect(y)
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if "seed" in k.lower():
                    collect(v)
                    if "range" in k.lower() and isinstance(v, list) and len(v)==2 and all(type(a) is int for a in v):
                        a,b=v
                        if 0 <= b-a <= 100000:
                            values.update(range(a,b+1))
                        elif b >= a:
                            raise ValueError("seed range too broad for automatic audit")
                else:
                    walk(v)
        elif isinstance(x,list):
            for y in x:
                if isinstance(y,(list,dict)):
                    walk(y)
    walk(obj)
    return values


def seed_audit(new):
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True).splitlines()
    historical = {int(s) for split in seeds.splits().values() for s in split}
    sources = {}
    for name in files:
        p = ROOT/name
        if p.suffix not in (".json", ".npz"):
            continue
        # Audit exactly the saved baseline, not a silently edited working copy.
        original = subprocess.check_output(["git", "show", BASE+":"+name], cwd=ROOT)
        if p.read_bytes() != original:
            raise ValueError("historical artifact changed: "+name)
        sources[name] = hashlib.sha256(original).hexdigest()
        if p.suffix == ".json":
            historical.update(seed_values(json.loads(original)))
        else:
            with np.load(p, allow_pickle=False) as z:
                if "state" in z.files:
                    historical.update(seed_values(json.loads(str(z["state"]))))
    require_disjoint(new, historical)
    return dict(schema="holdout-seed-audit/1", baseline_commit=BASE, source_sha256=sources,
                files_scanned=len(sources), historical_seed_field_values=sorted(historical),
                new_base_seeds=new, offsets=list(OFFSETS), no_generator_overlap=True,
                limitation="Saved project records only; cannot certify unrecorded external executions.")


def context():
    pl = plan.load_plan(str(ROOT/plan.DEFAULT_PLAN), str(ROOT/plan.DEFAULT_FREEZE))
    pc = next(c for c in pl["cells"] if c["group"]=="main" and c["r"]==.5)
    st = plan.cell_to_settings(pc)
    cks = main_eval.checkpoints(str(ROOT), pc, st)
    val = main_eval.validate_validation(read(VAL), pc, cks)
    if val["selected_run"]!=0 or val["selected_hw"]["public"]!="dz=0.3" or main_eval.code_hash()!=CORE_HASH:
        raise ValueError("original policies, selection or numerical code changed")
    return st, cks, val


def freeze():
    if MANIFEST.exists():
        raise RuntimeError("manifest already exists; do not replace the registered holdout")
    st, cks, val = context()
    new = list(range(41_000_000, 41_000_000+64))
    audit = seed_audit(new)
    write(OUT/"seed_audit.json", audit)
    protected = [str(p.relative_to(ROOT)) for folder in ("arbitration/rl", "arbitration/gpu", "results/direction2/main_eval", "results/direction2/ckpt")
                 for p in (ROOT/folder).glob("*") if p.is_file() and p.suffix in (".py", ".json", ".npz")]
    protected += ["scripts/run_quota.py", "results/direction2/main.json", "results/direction2/final_eval_ledger.json",
                  "docs/direction2-spec.md", "docs/direction2-freeze.md"]
    sd = val["runs"][0]["D_val"]["half"]*math.sqrt(32)/stats.t.ppf(.975,31)
    half = stats.t.ppf(.975,63)*sd/8
    m = dict(schema="independent-holdout/1", baseline_commit=BASE, cell=val["cell"], selected_run=0,
             baseline="dz=0.3", handwritten=dict(hw.build_s_hw())["dz=0.3"], checkpoints=main_eval.checkpoint_meta(cks),
             seeds=new, generator_offsets=list(OFFSETS), mech="M3C", r=.5, T=100000, T_score=97500,
             cfg=encode_settings(st["cfg"]), delta=.005, family_size=3, observation="public", dtype="float64",
             selection_sha256=val["selection_sha256"], source_sha256=CORE_HASH,
             inputs_sha256={p:sha(ROOT/p) for p in ENTRY_FILES}, seed_audit_sha256=sha(OUT/"seed_audit.json"),
             protected_sha256={p:sha(ROOT/p) for p in sorted(protected)},
             planning=dict(validation_sd_D=float(sd), expected_half_width_95=float(half),
                           conditional_power=float(stats.nct.sf(stats.t.ppf(.975,63),63,(val["runs"][0]["D_val"]["mean"]-.005)/(sd/8)))),
             stop_rule="Fixed 64 seeds, one primary contrast, no adaptation or sample extension.")
    # Infinity is encoded only in metadata; runner retains the original runtime cfg.
    m["sha256"] = digest(m)
    write(MANIFEST,m)
    print("Manifest frozen; must commit it and its seed audit before run.", flush=True)


def validate_manifest():
    m = read(MANIFEST)
    if m["sha256"] != digest({k:v for k,v in m.items() if k!="sha256"}):
        raise ValueError("manifest checksum changed")
    for name in [*m["inputs_sha256"], str(MANIFEST.relative_to(ROOT)), str((OUT/"seed_audit.json").relative_to(ROOT))]:
        if not plan.is_committed_and_clean(str(ROOT),name):
            raise ValueError("holdout input is not committed and clean: "+name)
    for name,h in {**m["inputs_sha256"], **m["protected_sha256"]}.items():
        if sha(ROOT/name)!=h:
            raise ValueError("frozen input changed: "+name)
    if sha(OUT/"seed_audit.json")!=m["seed_audit_sha256"]:
        raise ValueError("seed audit changed")
    require_disjoint(m["seeds"],read(OUT/"seed_audit.json")["historical_seed_field_values"])
    if len(m["seeds"])!=64 or m["selected_run"]!=0 or m["delta"]!=.005:
        raise ValueError("registered design changed")
    context()
    return m


def validate_raw(raw, contract):
    if raw["contract"]!=contract or raw["sha256"]!=digest({k:v for k,v in raw.items() if k!="sha256"}):
        raise ValueError("raw cache contract or checksum mismatch")
    p,n=contract["n_policies"],len(contract["seeds"])
    if raw["T_score"]!=contract["T_score"]:
        raise ValueError("wrong scoring window")
    for key,shape in (("util_b",(p,n,3)),("reg_b",(p,n)),("pen",(p,n,3))):
        a=np.asarray(raw["snap"][key],float)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError("invalid snapshot: "+key)
    return raw


def once(folder, name, contract, runner):
    folder=Path(folder)
    ledger_path=folder/"ledger.json"
    ledger=read(ledger_path) if ledger_path.exists() else {}
    raw_path=folder/(name+"_raw.json")
    if name in ledger and ledger[name]["contract_sha256"]!=digest(contract):
        raise ValueError("ledger contract mismatch")
    if raw_path.exists():
        raw=validate_raw(read(raw_path),contract)
        if name not in ledger:
            raise ValueError("raw without a registered attempt")
        if ledger[name]["status"]=="completed" and ledger[name]["raw_sha256"]!=sha(raw_path):
            raise ValueError("ledger/raw hash mismatch")
    else:
        if name in ledger:
            raise RuntimeError("attempt already started but raw is missing; investigate, never silently repeat")
        ledger[name]=dict(status="started", contract_sha256=digest(contract), started_unix=time.time())
        write(ledger_path,ledger)
        t=time.time()
        result=runner()
        raw=dict(contract=contract,T_score=int(result["T_score"]),snap={k:np.asarray(result["snap"][k],float).tolist() for k in main_eval.SNAP_FIELDS},wall_s=time.time()-t)
        raw["sha256"]=digest(raw)
        validate_raw(raw,contract)
        write(raw_path,raw)
    ledger[name].update(status="completed",raw_sha256=sha(raw_path))
    write(ledger_path,ledger)
    return raw


def summarize(m, rl, hand):
    main_eval._check_honest(rl,hand)
    gr,gh=main_eval.gains(rl)[1:],main_eval.gains(hand)[1]
    rows=[]
    for i,g in enumerate(gr):
        d=g-gh
        rows.append(dict(run=i,G=interval(g),D=interval(d),D_simultaneous_95=interval(d,3),
                         G_per_seed=g.tolist(),D_per_seed=d.tolist(),system=main_eval.system_metrics(rl,i+1)))
    primary=rows[m["selected_run"]]["D"]
    return dict(schema="independent-holdout-result/1",manifest_sha256=m["sha256"],n_seeds=len(m["seeds"]),
                selected_run=m["selected_run"],delta=m["delta"],verdict=verdict(primary,m["delta"]),
                positive_advantage_ruled_out=bool(primary["hi"]<=0),runs=rows,G_HW=interval(gh),
                G_HW_per_seed=gh.tolist(),consistent_runs=sum(x["D_simultaneous_95"]["lo"]>0 for x in rows),
                honest_system=main_eval.system_metrics(rl,0),handwritten_system=main_eval.system_metrics(hand,1),
                raw_sha256=dict(rl=rl["sha256"],hw=hand["sha256"]))


def run(do_backup=False):
    m=validate_manifest()
    st,cks,_=context()
    import torch
    torch.set_num_threads(1)
    graph=graphsim.from_env("cuda",force=True)
    runner=main_eval.make_runner(st,"cuda",graph)
    results={}
    for name in ("rl","hw"):
        pols=([policy.MLPPolicy(obs_version="public")]+[policy.MLPPolicy(theta=c["theta"],obs_version="public") for c in cks]
              if name=="rl" else [policy.AdapterPolicy({}),policy.AdapterPolicy(m["handwritten"])])
        con=dict(manifest_sha256=m["sha256"],stage=name,seeds=m["seeds"],n_policies=len(pols),T_score=m["T_score"])
        results[name]=once(OUT,name,con,lambda:runner(pols,m["seeds"],name=="rl"))
        print(f"Saved {name} batch ({results[name]['wall_s']:.1f}s); no interim significance decision.",flush=True)
        if do_backup:
            backup([str((OUT/(name+"_raw.json")).relative_to(ROOT)),str((OUT/"ledger.json").relative_to(ROOT))],"Direction 2: independent holdout "+name+" raw batch")
    result=summarize(m,results["rl"],results["hw"])
    validate_manifest()
    write(OUT/"result.json",result)
    if do_backup:
        backup([str((OUT/"result.json").relative_to(ROOT))],"Direction 2: fixed-sample independent holdout verdict")
    print(json.dumps({k:result[k] for k in ("verdict","consistent_runs","n_seeds")}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["freeze","run"])
    ap.add_argument("--backup",action="store_true")
    a=ap.parse_args()
    os.chdir(ROOT)
    (ROOT/".scratch").mkdir(exist_ok=True)
    with (ROOT/".scratch/holdout.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=="freeze":freeze()
        else:run(a.backup)


if __name__=="__main__":
    main()
