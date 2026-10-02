import json
import numpy as np
from arbitration import experiments as ex


def test_ci_and_paired():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    c = ex.ci95(x)
    assert c["mean"] == 2.5 and c["lo"] < 2.5 < c["hi"] and c["n"] == 4
    p = ex.paired(x + 1, x)
    assert abs(p["mean"] - 1.0) < 1e-12 and p["lo"] == p["hi"] and p["significant"]


def test_quick_end_to_end(tmp_path):
    out = tmp_path / "q.json"
    res = ex.main(["--quick", "--out", str(out)])
    d = json.loads(out.read_text())
    assert set(d) >= {"load_sweep", "paired", "rho", "lambda_sweep", "sigma_sweep", "hyst_sweep", "v2_sweep"}
    assert "oracle" not in d["load_sweep"] and "greedy_true" in d["load_sweep"]
    cell = d["load_sweep"]["honest"][str(ex.QUICK["rates"][0])]["mean_health"]
    assert len(cell["per_seed"]) == ex.QUICK["seeds"] and cell["lo"] <= cell["mean"] <= cell["hi"]
    for r, x in d["lambda_sweep"].items():
        assert x["lam_star"] in x["lams"] and len(x["lam_star_ci"]) == 2
    v = d["v2_sweep"]
    assert set(v["configs"]) == {"legacy", "v2"} and "rollout_true" in v["arms"]
    c = v["configs"]["v2"]["cells"]["learned|nopre"][str(ex.QUICK["rates"][0])]["mean_health"]
    assert len(c["per_seed"]) == ex.QUICK["seeds"]
    # quick run is deterministic
    assert ex.main(["--quick", "--out", str(tmp_path / "q2.json")])["rho"] == res["rho"]
