#!/usr/bin/env python3
"""Direction 2 S1 queue monitor (read-only, CPU only).

Reads results/direction2/s1_queue_manifest.json, `gpujob ls` and the part files; prints per cell the completed segments, running /
failed / timed-out jobs, the estimated remaining time, and for every failed job its log tail plus the command to resubmit it.
It never submits, kills or edits anything: resubmission commands are printed for you to run by hand (in segment order).

Usage:  python3 scripts/d2_s1_status.py [--manifest PATH] [--tail N] [--timeout-s 1500]
Exit code: 0 = no failed/timed-out/inconsistent job, 1 = something needs attention.
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOCK = "/tmp/gpujob-%d.socket" % os.getuid()
_LS = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*)$")


def tsp(*args):
    env = dict(os.environ, TS_SOCKET=SOCK)
    try:
        return subprocess.run(["tsp", *args], capture_output=True, text=True, env=env, timeout=20).stdout
    except Exception:
        return ""


def parse_ls():
    """{id: dict(state, elevel, times, cmd)} from `gpujob ls` (tsp -l)."""
    out = {}
    for line in tsp("-l").splitlines()[1:]:
        m = _LS.match(line)
        if not m:
            continue
        jid, state, rest = int(m.group(1)), m.group(2), m.group(3)
        el = tm = None
        if state == "finished":
            f = rest.split(None, 3)          # output, e-level, times, command
            el = int(f[1]) if len(f) > 1 and f[1].lstrip("-").isdigit() else None
            tm = f[2] if len(f) > 2 else None
        out[jid] = dict(state=state, elevel=el, times=tm, line=line)
    return out


def part_status(j):
    p = os.path.join(REPO, "results", "direction2", "parts", "%s__run%d__g%04d-%04d.json" % (j["cell"], j["run"], j["g0"], j["g1"]))
    if not os.path.exists(p):
        return "missing", None
    try:
        d = json.load(open(p))
    except Exception:
        return "unreadable", None
    return d.get("status", "?"), d


def tail_of(jid, n):
    path = tsp("-o", str(jid)).strip()
    if not path or not os.path.exists(path):
        return ["(no output file)"]
    try:
        return open(path, errors="replace").read().splitlines()[-n:]
    except OSError as e:
        return ["(cannot read %s: %s)" % (path, e)]


def hms(s):
    return "%dh%02dm" % (s // 3600, s % 3600 // 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(REPO, "results", "direction2", "s1_queue_manifest.json"))
    ap.add_argument("--tail", type=int, default=8, help="log lines shown for a failed job")
    ap.add_argument("--timeout-s", type=float, default=1500.0)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    ls = parse_ls()
    if not ls:
        print("WARNING: `gpujob ls` returned nothing (queue cleared or tsp unavailable); relying on part files only.")
    cells = {}
    rem_mean = rem_p95 = 0.0
    bad = []
    running = []
    for j in man["jobs"]:
        q = ls.get(j["job_id"])
        if q and j["cell"] not in q["line"]:
            q = None
            print("WARNING: job #%d in gpujob ls is not %s (manifest/queue mismatch)" % (j["job_id"], j["cell"]))
        pst, rec = part_status(j)
        c = cells.setdefault(j["cell"], dict(total=0, done=0, run=0, queued=0, fail=0, jobs=[]))
        c["total"] += 1
        state = q["state"] if q else "unknown"
        wall = None
        if pst == "done":
            c["done"] += 1
            wall = rec.get("wall_s")
            if wall is not None and wall > a.timeout_s:
                bad.append((j, "timeout", "part wall_s=%.0f s > %.0f s (overrun)" % (wall, a.timeout_s)))
            continue
        rem_mean += j["est_mean_s"]
        rem_p95 += j["est_p95_s"]
        if state == "running":
            c["run"] += 1
            running.append(j)
        elif state == "finished":                     # finished but the part is not done: it failed (or ended partial)
            c["fail"] += 1
            tm = q["times"]
            real = float(tm.split("/")[0]) if tm and re.match(r"^[\d.]+/", tm) else 0.0
            why = "exit=%s, part=%s" % (q["elevel"], pst)
            bad.append((j, "timeout" if real > a.timeout_s else "failed", why + (", real=%.0f s" % real if real else "")))
        elif state in ("queued", "allocating"):
            c["queued"] += 1
        else:                                          # not in the queue and no done part (cleared / killed / lost)
            c["fail"] += 1
            bad.append((j, "lost", "not in gpujob ls (state=%s) and part=%s" % (state, pst)))
    print("S1 queue status  (manifest %s, jobs #%d-#%d)" % (os.path.basename(a.manifest), man["job_id_range"][0], man["job_id_range"][1]))
    print("%-18s %9s %4s %4s %6s %5s" % ("cell", "done/seg", "run", "que", "fail", "state"))
    for name in man["cells"]:
        c = cells.get(name)
        if not c:
            continue
        s = "DONE" if c["done"] == c["total"] else ("FAIL" if c["fail"] else ("run" if c["run"] else "wait"))
        print("%-18s %4d/%-4d %4d %4d %6d %5s" % (name, c["done"], c["total"], c["run"], c["queued"], c["fail"], s))
    nd = sum(c["done"] for c in cells.values())
    print("\nsegments done %d/%d" % (nd, len(man["jobs"])))
    for j in running:
        print("running: #%d %s g%d-%d" % (j["job_id"], j["cell"], j["g0"], j["g1"]))
    print("estimated remaining: mean %s, p95 %s  (manifest total mean %s / p95 %s; running job counted in full)" % (
        hms(rem_mean), hms(rem_p95), hms(man["total_est"]["mean_s"]), hms(man["total_est"]["p95_s"])))
    for j, kind, why in bad:
        print("\n[%s] #%d %s run%d g%d-%d : %s" % (kind.upper(), j["job_id"], j["cell"], j["run"], j["g0"], j["g1"], why))
        if j["job_id"] in ls:
            for line in tail_of(j["job_id"], a.tail):
                print("    | " + line)
        later = [x for x in man["jobs"] if x["cell"] == j["cell"] and x["run"] == j["run"] and x["g0"] >= j["g0"] and part_status(x)[0] != "done"]
        print("  resubmit by hand (in this order; later segments need this one's checkpoint, and those that already failed on a missing checkpoint need rerunning too):")
        for x in later:
            print("    gpujob %s" % x["command"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
