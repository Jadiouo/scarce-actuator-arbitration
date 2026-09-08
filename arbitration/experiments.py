"""Sweeps behind the three figures. `python -m arbitration.experiments` runs all."""
import json, os
import numpy as np
from scipy.stats import spearmanr
from .model import run, POLICIES

SEEDS, STEPS, N, SPREAD = 24, 1500, 8, 0.6
RATES = [0.0005, 0.001, 0.002, 0.004, 0.007, 0.012, 0.020, 0.040]
LAMS = [0.0, 0.002, 0.004, 0.006, 0.010, 0.016, 0.025]
ARMS = ["none", "none_random", "oracle", "nearest", "honest", "strategic", "priced"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _mean(policy, **kw):
    rs = [run(policy, n=N, steps=STEPS, seed=s, spread=SPREAD, preempt=True, **kw)
          for s in range(SEEDS)]
    keys = ("mean_health", "min_health", "frac_below_02", "frac_any_dead",
            "precision", "n_services", "retargets_per_service")
    return {k: float(np.nanmean([getattr(r, k) for r in rs])) for k in keys}


def load_sweep():
    """Figures 1 and 2: every policy across two decades of degradation rate."""
    return {a: {str(r): _mean(a, rate=r) for r in RATES} for a in ARMS}


def price_sweep():
    """Figure 3: the request price, below capacity and when overloaded."""
    return {str(r): {str(l): _mean("priced", rate=r, lam=l) for l in LAMS}
            for r in (0.001, 0.004)}


def precision_outcome_rho(sweep, arms=("none", "oracle", "nearest",
                                       "honest", "strategic", "priced")):
    """Rank correlation between dispatch precision and team outcome, per load."""
    out = []
    for r in RATES:
        p = [sweep[a][str(r)]["precision"] for a in arms]
        u = [sweep[a][str(r)]["mean_health"] for a in arms]
        out.append({"rate": r,
                    "load": sweep["none"][str(r)]["frac_any_dead"],
                    "rho": float(spearmanr(p, u)[0])})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sweep = load_sweep()
    price = price_sweep()
    rho = precision_outcome_rho(sweep)
    json.dump({"load_sweep": sweep, "price_sweep": price, "rho": rho,
               "config": {"seeds": SEEDS, "steps": STEPS, "n": N,
                          "spread": SPREAD, "rates": RATES, "lams": LAMS}},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"{'rate':>8} {'load':>6} {'rho':>7}")
    for row in rho:
        print(f"{row['rate']:8.4f} {row['load']:6.2f} {row['rho']:+7.3f}")
    print(f"\nwritten to {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
