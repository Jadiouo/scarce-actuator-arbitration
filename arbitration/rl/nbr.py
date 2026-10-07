"""REQ-NBR-*: neighbouring-parameter transfer (no retraining)."""
import json
import os
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .metrics import difference_ci, mean_ci

LIMITATION = ("機制參數沒有針對 RL 重新調參（h=6、L=2500、p=0.1、kref=0.5 為方向一選定值），結論僅限此固定配置。"
              "鄰近參數遷移為描述性結果（策略未重新訓練），不構成對參數空間的任何主張。")


def neighbor_configs(base: Dict[str, float]) -> List[Dict[str, float]]:
    """REQ-NBR-01: exactly h=5, h=7, L=1250 (all other parameters unchanged)."""
    out = []
    for key, val in (("tol", 5.0), ("tol", 7.0), ("L", 1250.0)):
        c = dict(base)
        c[key] = val
        out.append(c)
    return out


def neighbor_t_score(T: int, L: float) -> int:
    """REQ-NBR-01: T - L."""
    return int(T - L)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def evaluate_neighbors(theta: Any, hw_name: str, seeds: Sequence[int], sim_fn: Callable[..., Dict[str, Any]], out_path: str,
                       T: int = 100000) -> Dict[str, Any]:
    """REQ-NBR-01/02: for each neighbour call sim_fn(config, who, seeds, T, T_score, variant=None) -> {'G_s','rt','alpha'}
    for who in ('rl','hw'); write out_path JSON including the REQ-NBR-02 limitation sentence; policy params untouched."""
    base = dict(p=0.1, tol=6.0, L=2500.0, kref=0.5)
    rows = []
    for cfg in neighbor_configs(base):
        Ts = neighbor_t_score(T, cfg["L"])
        res = {who: sim_fn(dict(cfg), who, list(seeds), T, Ts, variant=None) for who in ("rl", "hw")}
        g_rl, g_hw = np.asarray(res["rl"]["G_s"], float), np.asarray(res["hw"]["G_s"], float)
        rows.append({"config": cfg, "T": int(T), "T_score": Ts,
                     "rl": {"G": mean_ci(g_rl), "rt": res["rl"]["rt"], "alpha": res["rl"]["alpha"]},
                     "hw": {"name": hw_name, "G": mean_ci(g_hw), "rt": res["hw"]["rt"], "alpha": res["hw"]["alpha"]},
                     "D": difference_ci(g_rl, g_hw)})
    doc = {"limitation": LIMITATION, "hw_name": hw_name, "neighbors": rows, "retrained": False}
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(doc), f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    return doc
