"""REQ-ENV-20: truncation-bias measurement (diagnostic only; never imported by training/selection/classification)."""
import json
import os
from typing import Any, Dict, Sequence

import numpy as np

from . import env
from .metrics import mean_ci
from .policy import AdapterPolicy


def measure(policy: Any, r: float, T: int, L: float, seeds: Sequence[int], out_path: str, dev: str = "cuda",
            name: str = "policy") -> Dict[str, Any]:
    """REQ-ENV-20: G_normal vs G_variant (variant forces v1=u1 for t>T-2L); writes JSON row
    {policy,r,G_normal,G_variant,delta,ci}; delta = G_normal - G_variant (paired, t31 CI as [lo,hi])."""
    cfg = dict(env.m3c_config(r))
    cfg["L"] = float(L)
    pols = [AdapterPolicy({}), policy]
    normal = env.simulate("M3C", T, cfg, pols, seeds, r=r, dev=dev)
    variant = env.simulate("M3C", T, cfg, pols, seeds, r=r, dev=dev, diagnostic_variant="tail_honest")
    Ts = normal["T_score"]

    def gain(o):
        u = o["snap"]["util_b"][:, :, 1].detach().cpu().numpy().astype(float)
        return (u[1] - u[0]) / Ts

    g_n, g_v = gain(normal), gain(variant)
    d = mean_ci(g_n - g_v, "t31")
    row = {"policy": name, "r": float(r), "T": int(T), "L": float(L), "T_score": int(Ts), "n_seeds": len(list(seeds)),
           "G_normal": float(g_n.mean()), "G_variant": float(g_v.mean()), "delta": float(d["mean"]), "ci": [d["lo"], d["hi"]]}
    rows = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                old = json.load(f)
            rows = old if isinstance(old, list) else [old]
        except (ValueError, OSError):
            rows = []
    rows = [x for x in rows if not (x.get("policy") == name and x.get("r") == row["r"])] + [row]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, out_path)
    return row
