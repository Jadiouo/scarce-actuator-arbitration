"""T-26..T-35, T-62: seeds, handwritten baseline, metrics (spec s14.4)."""
import ast
import json
import os

import numpy as np
import pytest
from scipy import stats

from conftest import ROOT, load_json, std_samples
from arbitration.rl import hw, metrics, numaudit, s1, seeds, train


def test_T26_seed_splits_disjoint():
    """T-26 (REQ-SEED-01/08, v1.3): sizes 1000/32/32/32 (train/val/val2/test), ranges 0-999 / 2000-2031 / 3000-3031 (val2) / 5000-5031; the generator-seed
    sets {s, s+1e6, s+2e6} of the FOUR splits are pairwise disjoint; replay seeds 1000-1031 belong to no split and are rejected by every guard;
    `require_split` accepts 'train','val','val2','test' and raises ValueError for every seed outside the claimed split (including val2 vs the other three)."""
    sp = seeds.splits()
    assert (len(sp["train"]), len(sp["val"]), len(sp["val2"]), len(sp["test"])) == (1000, 32, 32, 32)
    assert list(sp["train"]) == list(range(0, 1000)) and list(sp["val"]) == list(range(2000, 2032))
    assert list(sp["val2"]) == list(range(3000, 3032))
    assert list(sp["test"]) == list(range(5000, 5032)) and list(sp["replay"]) == list(range(1000, 1032))
    names = ["train", "val", "val2", "test"]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not seeds.generator_seed_set(names[i]) & seeds.generator_seed_set(names[j]), (names[i], names[j])
    assert len(seeds.generator_seed_set("train")) == 3000 and len(seeds.generator_seed_set("val2")) == 96
    assert not set(sp["replay"]) & (set(sp["train"]) | set(sp["val"]) | set(sp["val2"]) | set(sp["test"]))
    for sp_name in names:
        with pytest.raises(ValueError):
            seeds.require_split(list(sp["replay"]), sp_name)
        seeds.require_split(list(sp[sp_name])[:3], sp_name)
        seeds.require_split(list(sp[sp_name]), sp_name)
    # every ordered pair of DIFFERENT splits must raise (any combination), also for a single seed and for mixed lists
    for claimed in names:
        for actual in names:
            if claimed == actual:
                continue
            with pytest.raises(ValueError):
                seeds.require_split(list(sp[actual]), claimed)
            with pytest.raises(ValueError):
                seeds.require_split([sp[actual][0]], claimed)
            with pytest.raises(ValueError):
                seeds.require_split([sp[claimed][0], sp[actual][0]], claimed)         # mixed list
    # boundaries just outside each split
    for sp_name, (lo, hi) in {"train": (0, 999), "val": (2000, 2031), "val2": (3000, 3031), "test": (5000, 5031)}.items():
        seeds.require_split([lo, hi], sp_name)
        for bad in (lo - 1, hi + 1):
            with pytest.raises(ValueError):
                seeds.require_split([bad], sp_name)
    with pytest.raises(ValueError):
        seeds.require_split([3000], "val3")                       # unknown split name


def _probe_hits(tmp_path, name, src):
    f = tmp_path / name
    f.write_text(src)
    return seeds.scan_forbidden_references([str(f)])


def test_T27_test_seeds_unreachable_from_training(tmp_path):
    """T-27 (REQ-SEED-02/03): AST scan (train, cmaes, budget, hw, s1, vulns) finds no test-seed constant, no literal 5000-5031 and no aliased import
    of one; the scanner itself is validated with probes (plain import, aliased import, module attribute, literal 5000..5031); training rejects
    seeds >= 1000."""
    rl_dir = os.path.join(ROOT, "arbitration", "rl")
    mods = [os.path.join(rl_dir, f"{m}.py") for m in ("train", "cmaes", "budget", "hw", "s1", "vulns")]
    assert all(os.path.exists(m) for m in mods)
    assert seeds.scan_forbidden_references(mods) == []
    # v1.3 (REQ-SEED-06): EVERY arbitration/rl module except final_eval.py is scanned (not only the six above)
    every = sorted(os.path.join(rl_dir, f) for f in os.listdir(rl_dir) if f.endswith(".py") and f != "final_eval.py")
    assert set(mods) <= set(every) and len(every) >= 20
    assert seeds.scan_forbidden_references(every) == []
    must_flag = {
        "plain.py": "from x import TEST_SEEDS\n",
        "alias.py": "from arbitration.rl.seeds import TEST_SEEDS as X\nprint(X)\n",
        "alias_mod.py": "import arbitration.rl.seeds as sd\nprint(sd.TEST_SEEDS)\n",
        "attr.py": "from arbitration.rl import seeds\nprint(seeds.TEST_SEEDS)\n",
        "lit_range.py": "ev = list(range(5000, 5032))\n",
        "lit_list.py": "ev = [5000, 5001, 5002]\n",
        "lit_one.py": "ev = 5031\n",
        "lit_def.py": "def f(s=5016):\n    return s\n",
    }
    for name, src in must_flag.items():
        assert len(_probe_hits(tmp_path, name, src)) >= 1, f"scanner missed: {name}"
    benign = "a = 4999\nb = 5032\nc = 2000\nd = 100000\ne = '5000'\n# 5000 in a comment\ndef f(seeds):\n    return seeds\n"
    assert _probe_hits(tmp_path, "benign.py", benign) == []
    hits = _probe_hits(tmp_path, "format.py", "from x import TEST_SEEDS\n")
    assert hits and hits[0].startswith(str(tmp_path / "format.py") + ":") and hits[0].count(":") >= 2          # 'path:line:name'
    train.require_train_seeds([0, 999])
    for bad in ([2000], [5000], [1000], [-1], [0, 1000], [3000], [3031], [0, 3000]):             # v1.3: val2 seeds are rejected too
        with pytest.raises(ValueError):
            train.require_train_seeds(bad)
    # v1.3 (REQ-SEED-08): the test-evaluation entry refuses val2 seeds (a ValueError naming the split, NOT NotImplementedError: final_eval is implemented)
    from arbitration.rl import final_eval
    with pytest.raises(ValueError, match="val2"):
        final_eval.evaluate_on_test(dict(seeds=list(range(3000, 3032))))


def _fe_record(**kw):
    return dict(dict(seeds=list(range(5000, 5032)), cell="c0", policy_id="p0", mde_frozen=0.006), **kw)


def _fe_eval(seeds_):
    from conftest import std_samples
    return dict(G_s=std_samples(0.05, 0.01, n=len(seeds_)), D_s=std_samples(0.02, 0.01, n=len(seeds_)))


def test_T27b_final_eval_accepts_only_test_seeds(tmp_path):
    """T-27b (REQ-SEED-02/03/06/08, v1.3 final): `final_eval.evaluate_on_test` is IMPLEMENTED (it does not raise NotImplementedError) and accepts exactly the 32
    test seeds (5000-5031).  train, val, val2, replay seeds, mixed lists, a partial list and duplicates raise ValueError (never NotImplementedError; val2 and
    val are named in the message).  It also needs the signed freeze file, evaluates each (cell, policy) once (ledger), uses the FROZEN MDE (never recomputed)
    and applies metrics.final_eval."""
    from arbitration.rl import final_eval
    fz, led = tmp_path / "freeze.md", tmp_path / "ledger.json"
    fz.write_text("frozen")
    kw = dict(freeze_path=str(fz), ledger_path=str(led))
    # refusals: ValueError, never NotImplementedError, and nothing is evaluated or written
    called = []
    spy = lambda s_: called.append(s_) or _fe_eval(s_)
    for name, bad in (("val2", range(3000, 3032)), ("val", range(2000, 2032)), ("train", range(0, 32)), ("replay", range(1000, 1032)),
                      ("mixed", list(range(5000, 5031)) + [3000]), ("mixed", list(range(5000, 5031)) + [2000]), ("test partial", [5000, 5001]),
                      ("test duplicate", list(range(5000, 5031)) + [5000]), ("test outside", list(range(5001, 5033))), ("none", [])):
        with pytest.raises(ValueError):
            final_eval.evaluate_on_test(_fe_record(seeds=list(bad)), spy, **kw)
    with pytest.raises(ValueError, match="val2"):
        final_eval.evaluate_on_test(_fe_record(seeds=list(range(3000, 3032))), spy, **kw)
    with pytest.raises(ValueError, match="val"):
        final_eval.evaluate_on_test(_fe_record(seeds=list(range(2000, 2032))), spy, **kw)
    with pytest.raises(ValueError):
        final_eval.evaluate_on_test(dict(cell="c0", policy_id="p0", mde_frozen=0.006), spy, **kw)         # no seeds
    assert called == [] and not led.exists(), "a refused request must not evaluate anything or write the ledger"
    # required keys: frozen MDE is mandatory
    for k in ("cell", "policy_id", "mde_frozen"):
        rec = _fe_record()
        rec.pop(k)
        with pytest.raises(ValueError):
            final_eval.evaluate_on_test(rec, spy, **kw)
    # freeze file must exist
    with pytest.raises(FileNotFoundError):
        final_eval.evaluate_on_test(_fe_record(), spy, freeze_path=str(tmp_path / "nofreeze.md"), ledger_path=str(led))
    assert called == []
    # accepted: implemented, returns metrics.final_eval with the frozen MDE
    out = final_eval.evaluate_on_test(_fe_record(), spy, **kw)
    assert called == [list(range(5000, 5032))] and out["mde_used"] == 0.006 and out["found_G"] is True and out["found_D"] is True
    assert out["n_seeds"] == 32 and out["seed_range"] == [5000, 5031] and out["cell"] == "c0" and out["policy_id"] == "p0"
    assert abs(out["G"]["mean"] - 0.05) < 1e-12
    # once per (cell, policy): a second request raises RuntimeError and does not evaluate again; another cell or policy is allowed
    with pytest.raises(RuntimeError):
        final_eval.evaluate_on_test(_fe_record(), spy, **kw)
    assert len(called) == 1
    final_eval.evaluate_on_test(_fe_record(cell="c1"), spy, **kw)
    final_eval.evaluate_on_test(_fe_record(policy_id="p1"), spy, **kw)
    assert set(json.loads(led.read_text())) == {"c0|p0", "c1|p0", "c0|p1"}
    # the frozen MDE decides detection (not recomputed): the same gains with a larger frozen MDE are not found
    big = final_eval.evaluate_on_test(_fe_record(cell="c2", mde_frozen=0.5), spy, **kw)
    assert big["found_G"] is False and big["mde_used"] == 0.5
    # a malformed evaluator result is rejected and leaves no ledger entry
    with pytest.raises(ValueError):
        final_eval.evaluate_on_test(_fe_record(cell="c3"), lambda s_: dict(G_s=[0.1] * 5), **kw)
    assert "c3|p0" not in json.loads(led.read_text())
    # the module itself is the only one that may reference test seeds; it must not hard-code the val2 range either (T-70)
    assert final_eval.require_test_seeds(range(5000, 5032)) == list(range(5000, 5032))


def test_T28_hw_set_definition():
    """T-28 (REQ-HW-01/03/04/06, v1.3): S_HW has 36 strategies (WITHOUT H0) = the 45 of strategies() minus 13 (H0, 8 oracle, 4 quota_compress_uninf) plus the
    4 fine-grid strategies dz=2.5, dz=3.0, dz=4.0, b=1.0; none named oracle/quota_; group sizes b=6, dz=15, dzcrit=3, burst=3, edge thr=3, edge dz=2,
    stealth=4; with H0 it is 37; the old 32-list stays available as the coarse list; realizability table covers all 36 (public); run_quota.py unmodified."""
    import subprocess
    from scripts.run_quota import strategies
    full = [n for n, _ in strategies()]
    assert len(full) == 45 and [n for n, _ in hw.all_strategies()] == full
    s_hw = hw.build_s_hw()
    names = [n for n, _ in s_hw]
    fine = ["dz=2.5", "dz=3.0", "dz=4.0", "b=1.0"]
    assert len(names) == 36 and len(set(names)) == 36 and "honest" not in names
    assert not any("oracle" in n or n.startswith("quota_") for n in names)
    removed = [n for n in full if n not in names]
    assert len(removed) == 13
    assert set(removed) == {"honest"} | {n for n in full if "oracle" in n} | {n for n in full if n.startswith("quota_compress_uninf")}
    assert set(names) == (set(full) - set(removed)) | set(fine) and all(n not in full for n in fine)
    # spec order inside the groups (ties are broken by this order, REQ-HW-02)
    assert [n for n in names if n.startswith("b=")] == ["b=0.05", "b=0.1", "b=0.2", "b=0.3", "b=0.5", "b=1.0"]
    dz = [n for n in names if n.startswith("dz=")]
    assert dz == ["dz=0.01", "dz=0.02", "dz=0.03", "dz=0.05", "dz=0.07", "dz=0.1", "dz=0.15", "dz=0.2", "dz=0.3", "dz=0.5", "dz=1.0", "dz=2.0",
                  "dz=2.5", "dz=3.0", "dz=4.0"]
    groups = dict(b=lambda n: n.startswith("b="), dz=lambda n: n.startswith("dz="), dzcrit=lambda n: n.startswith("dzcrit="),
                  burst=lambda n: n.startswith("burst"), edge_thr=lambda n: n.startswith("edge_uninf thr"),
                  edge_dz=lambda n: n.startswith("edge_uninf dz"), stealth=lambda n: n.startswith("stealth"))
    sizes = {g: sum(f(n) for n in names) for g, f in groups.items()}
    assert sizes == dict(b=6, dz=15, dzcrit=3, burst=3, edge_thr=3, edge_dz=2, stealth=4), sizes
    assert sum(sizes.values()) == 36
    d = dict(s_hw)
    assert d["dz=2.5"] == dict(dz=2.5) and d["dz=3.0"] == dict(dz=3.0) and d["dz=4.0"] == dict(dz=4.0) and d["b=1.0"] == dict(b=1.0)
    # the 37-version and the legacy 32-list
    with_h0 = [n for n, _ in hw.build_s_hw_with_h0()]
    assert len(with_h0) == 37 and "honest" in with_h0 and [n for n in with_h0 if n != "honest"] == names
    coarse = [n for n, _ in hw.build_s_hw_coarse()]
    assert len(coarse) == 32 and coarse == [n for n in names if n not in fine]
    tab = hw.public_realizability_table()
    assert set(tab) == set(names) and all(isinstance(v["public"], bool) and v["info"] for v in tab.values())
    assert all(tab[n]["public"] is True for n in fine) and all("own" in tab[n]["info"] for n in fine)
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "scripts/run_quota.py"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", "scripts/run_quota.py must not be modified (REQ-HW-01)"


def test_T29_hw_selection_val_only_untruncated():
    """T-29 (REQ-HW-02, SEED-05, MET-02/04, v1.3): selection by validation mean only over the 36 candidates (ties -> earlier); all-negative -> least
    negative with negative test G_HW; D can be negative (untruncated); the +H0 variant (37 incl. honest) is reported separately and is not an input
    of classify; a candidate list that is not the 36 is refused."""
    names = [f"s{i}" for i in range(36)]
    rng = np.random.default_rng(0)
    val = {n: rng.normal(0.01, 0.002, 32) for n in names}
    test = {n: rng.normal(0.0, 0.002, 32) for n in names}
    val["s5"] = np.full(32, 0.5)
    val["s9"] = np.full(32, 0.5)
    test["s5"] = np.full(32, -0.4) + std_samples(0, 0.001)          # selected by validation, yet the worst on test
    test["s9"] = np.full(32, 0.3)                                    # tied on validation, better on test: must NOT be preferred (earlier wins)
    test["s20"] = np.full(32, 9.9)                                   # best by far on test: a selection that peeks at test picks this
    test["s33"] = np.full(32, 9.9)                                   # a fine-grid slot (the last 4 names are the new ones): same
    res = hw.select_handwritten(val, test, names)
    assert res["selected"] == "s5" and res["selected"] != "s20" and res["selected"] != "s9"
    assert np.allclose(res["G_hw_s"], test["s5"]) and abs(res["G_hw"] - np.mean(test["s5"])) < 1e-15 and res["G_hw"] < 0
    flipped = hw.select_handwritten(dict(val, s9=np.full(32, 0.6)), test, names)           # val decides, not test: s9 now strictly best on val
    assert flipped["selected"] == "s9"
    late = hw.select_handwritten(dict(val, s35=np.full(32, 0.7)), test, names)             # the LAST candidate can be selected (no truncation to 32)
    assert late["selected"] == "s35"
    neg_val = {n: np.full(32, -0.01 - 0.001 * i) for i, n in enumerate(names)}
    neg_test = {n: np.full(32, -0.02 - 0.001 * i) for i, n in enumerate(names)}
    r2 = hw.select_handwritten(neg_val, neg_test, names)
    assert r2["selected"] == "s0" and r2["G_hw"] < 0
    assert r2["plus_h0"]["selected"] == "honest" and r2["plus_h0"]["G_hw"] == 0.0
    assert res["plus_h0"]["selected"] == "s5"                       # honest (0) does not beat s5 on validation
    # the +H0 candidate set has 37 members: honest enters at the END of the list (ties are won by the 36)
    tie_val = {n: np.zeros(32) for n in names}
    tie_test = {n: np.full(32, -0.01) for n in names}
    rt = hw.select_handwritten(tie_val, tie_test, names)
    assert rt["selected"] == "s0" and rt["plus_h0"]["selected"] == "s0"
    for bad_names in (names[:32], names[:35], names + ["extra"]):
        with pytest.raises(ValueError):
            hw.select_handwritten({n: val.get(n, np.zeros(32)) for n in bad_names}, {n: test.get(n, np.zeros(32)) for n in bad_names}, bad_names)
    D = metrics.difference_ci(np.full(32, -0.05) + std_samples(0, 0.01), np.full(32, 0.01))
    assert D["mean"] < 0 and D["hi"] < 0
    import inspect
    from arbitration.rl import stop
    assert "plus_h0" not in inspect.signature(stop.classify).parameters


def test_T30_paired_gain_and_ci():
    """T-30 (REQ-MET-01/02/03): G_s paired by seed, no clipping; CI uses t(31)=2.0395 (replay: 1.96); equals hand computation."""
    rng = np.random.default_rng(3)
    Up, Uh = rng.normal(0.3, 0.05, 32) * 1000, rng.normal(0.2, 0.05, 32) * 1000
    G = np.asarray(metrics.paired_gain(Up, Uh, 1000))
    assert np.allclose(G, (Up - Uh) / 1000, rtol=0, atol=1e-15)
    neg = np.asarray(metrics.paired_gain(Uh, Up, 1000))
    assert np.allclose(neg, -G, rtol=0, atol=1e-15)
    m, sd = G.mean(), G.std(ddof=1)
    c = metrics.mean_ci(G, "t31")
    assert abs(c["mean"] - m) < 1e-15 and abs(c["half"] - 2.0395 * sd / np.sqrt(32)) <= 1e-4 * c["half"]
    assert abs(c["lo"] - (m - c["half"])) < 1e-15 and abs(c["hi"] - (m + c["half"])) < 1e-15
    z = metrics.mean_ci(G, "z196")
    assert abs(z["half"] - 1.96 * sd / np.sqrt(32)) < 1e-15


def test_T31_difference_ci_paired():
    """T-31 (REQ-MET-04): D_s = G_RL,s - G_HW,s per seed with t31 CI; matches hand computation; 3 runs listed with the
    validation-selected run as main value."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.03, 0.01, 32), rng.normal(0.02, 0.01, 32)
    d = metrics.difference_ci(a, b)
    D = a - b
    assert abs(d["mean"] - D.mean()) < 1e-15
    assert abs(d["half"] - 2.0395 * D.std(ddof=1) / np.sqrt(32)) <= 1e-4 * d["half"]
    assert abs(d["lo"] - (D.mean() - d["half"])) < 1e-12 and abs(d["hi"] - (D.mean() + d["half"])) < 1e-12
    runs = [rng.normal(0.03, 0.01, 32) for _ in range(3)]
    ks = metrics.replication_count([r - b for r in runs])
    assert ks == sum(metrics.mean_ci(r - b)["lo"] > 0 for r in runs)


def test_T32_mde_formula():
    """T-32 (REQ-MET-06): kappa=2.8922 (nct root of P(|T31(d)|>2.0395)=0.80, 1e-4); MDE=kappa*SD/sqrt(32) for G_s and D_s; Monte Carlo
    power at that effect = 0.80+-0.02; MDE from validation seeds only; test evaluation reads mde_frozen."""
    k = metrics.kappa()
    tc = stats.t.ppf(0.975, 31)
    p = lambda d: stats.nct.sf(tc, 31, d) + stats.nct.cdf(-tc, 31, d)
    assert abs(k - 2.8922) <= 1e-4 and abs(p(k) - 0.80) < 5e-4
    val = list(range(2000, 2032))
    for x in (std_samples(0.02, 0.01), std_samples(0.0, 0.03)):
        assert abs(metrics.mde(x, val) - k * np.std(x, ddof=1) / np.sqrt(32)) < 1e-15
    with pytest.raises(ValueError):
        metrics.mde(std_samples(0.02, 0.01), list(range(5000, 5032)))
    rng = np.random.default_rng(0)
    sigma = 0.01
    mu = k * sigma / np.sqrt(32)
    x = rng.normal(mu, sigma, (20000, 32))
    t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(32))
    assert abs((np.abs(t) > tc).mean() - 0.80) <= 0.02
    rec = dict(mde_frozen=0.5)
    r = metrics.final_eval(rec, std_samples(0.03, 0.005))
    assert r["mde_used"] == 0.5 and r["found_G"] is False
    r = metrics.final_eval(dict(mde_frozen=0.001), std_samples(0.03, 0.005))
    assert r["mde_used"] == 0.001 and r["found_G"] is True
    with pytest.raises(KeyError):
        metrics.final_eval({}, std_samples(0.03, 0.005))


def test_T33_found_rule_truth_table():
    """T-33 (REQ-MET-07/08): found = (lo>0 and value>=MDE) over the four combinations (boundary value==MDE counts); replication count k and labels."""
    for value, lo, mde, exp in ((0.03, 0.01, 0.02, True), (0.03, 0.01, 0.04, False), (0.03, -0.01, 0.02, False),
                                (0.03, -0.01, 0.04, False), (0.02, 0.01, 0.02, True), (0.03, 0.0, 0.02, False)):
        assert metrics.found_rule(value, lo, mde) is exp, (value, lo, mde)
    sig, non = std_samples(0.05, 0.01), std_samples(0.0, 0.02)
    assert metrics.replication_count([sig, sig, sig]) == 3 and metrics.replication_count([sig, non, sig]) == 2
    assert metrics.replication_count([non, sig, non]) == 1
    assert (metrics.replication_label(3), metrics.replication_label(2), metrics.replication_label(1)) == ("完全複現", "部分複現", "未複現")


def test_T34_rt_and_false_punish_definitions():
    """T-34 (REQ-MET-05): R/T = reg_b/T_score; alpha = mean over the 3 agents of pen / T_score; per-agent suspension fraction of agents 0 and 2."""
    reg = np.array([10.0, 20.0, 30.0])
    pen = np.array([[3.0, 6.0, 9.0], [0.0, 0.0, 0.0], [30.0, 0.0, 60.0]])
    rt, al = metrics.rt_alpha(reg, pen, 100)
    assert np.allclose(rt, reg / 100) and np.allclose(al, pen.mean(1) / 100)
    f = np.asarray(metrics.agent_susp_fraction(pen, 100))
    assert np.allclose(f, pen[:, [0, 2]] / 100)


def test_T35_number_audit_registry():
    """T-35 (REQ-MET-09, STOP-05): every reported number has a JSON path; registry value must equal the JSON value; missing path/mismatch flagged."""
    q = load_json("quota.json")["main"][12]["Gmax_uninformed"][0]
    good = dict(name="G_uninf_r05", value=q, file="results/quota.json", path="main[12].Gmax_uninformed[0]")
    assert numaudit.check_registry([good], root=ROOT) == []
    probs = numaudit.check_registry([dict(good, value=q + 1e-6)], root=ROOT)
    assert len(probs) == 1 and probs[0]["problem"] == "mismatch"
    probs = numaudit.check_registry([dict(good, path="main[12].nope[0]")], root=ROOT)
    assert len(probs) == 1 and probs[0]["problem"] == "missing_path"


def test_T62_s1_gate_sizes_and_run_count():
    """T-62 (REQ-S1-10/22, MET-06, STOP-02, v1.3): G_s = {s in {0.01,0.02} : s >= MDE_D,plan}; 0.02 not in G_s => STOP-02;
    N = 3|G_s| + 3 + 1 + 2 (out-of-family x G_s, in-family x 1, blind x 1, anchors 2): |G_s|=2 -> 12, |G_s|=1 -> 9; descriptive sizes (< MDE) are not scheduled;
    MDE_est=0.0058 -> G_s={0.01,0.02}, N=12.  The schedule agrees with run_count term by term, s=0.02 first (9 runs), D3/designer_1/O4/O6/B1/B2 never."""
    g = s1.gate_sizes(0.0058)
    assert g["G_s"] == [0.01, 0.02] and g["stop02"] is False and s1.run_count(g["G_s"]) == 12
    assert s1.gate_sizes(0.015)["G_s"] == [0.02] and s1.run_count([0.02]) == 9
    g4 = s1.gate_sizes(0.0025)                                   # 0.003 / 0.005 no longer exist as registry sizes (REQ-S1-02 v1.3)
    assert g4["G_s"] == [0.01, 0.02] and s1.run_count(g4["G_s"]) == 12
    g0 = s1.gate_sizes(0.021)
    assert g0["G_s"] == [] and g0["stop02"] is True
    assert s1.gate_sizes(0.02)["G_s"] == [0.02]
    assert s1.gate_sizes(0.0100001)["G_s"] == [0.02] and s1.gate_sizes(0.01)["G_s"] == [0.01, 0.02]      # boundary: s >= MDE
    assert s1.SIZES == (0.01, 0.02)
    sch = s1.schedule([0.01, 0.02])
    assert len(sch) == s1.run_count([0.01, 0.02]) == 12 and len(set(sch)) == 12
    assert all(sz == 0.02 for _, sz in sch[:9]) and all(sz == 0.01 for _, sz in sch[9:])
    cat = lambda i: s1.CATEGORY_OF[i]
    first, rest = sch[:9], sch[9:]
    assert sorted(i for i, _ in first if cat(i) == "out_of_family") == ["D1", "D2", "O1"]
    assert sorted(i for i, _ in first if cat(i) == "in_family") == ["D4", "O2", "O3"]
    assert [i for i, _ in first if cat(i) == "blind"] == ["O5"]
    assert sorted(i for i, _ in first if cat(i) == "anchor") == sorted(["NAIVE", "M4rw0"])
    assert sorted(i for i, _ in rest) == ["D1", "D2", "O1"]                                     # s=0.01: out-of-family only
    names = {i for i, _ in sch}
    assert not names & {"D3", "designer_1", "O4", "O6", "B1", "B2"}
    assert len(s1.schedule([0.02])) == 9 == s1.run_count([0.02])
    with pytest.raises(ValueError):
        s1.schedule([0.01])                                     # 0.02 not in the gate set: STOP-02, no schedule


def test_T68_burst_uses_own_clock_and_is_public_HW05():
    """T-68 (REQ-HW-05, v1.2): the 3 `burst` strategies of S_HW run on the agent's OWN round counter and are classed public (REQ-HW-04):
    (a) exactly the 3 burst names are in S_HW, all with public=True and an info text that names the own clock ('clock'); no other strategy's
    info mentions a clock (the time information is a property of burst only).  (b) behaviour: the rounds where the burst report differs from the
    honest u1 are exactly {t : t mod (W+H) < W} - a fixed function of the own round index, identical for different seeds (it does not depend
    on z, on the other agents or on suspension).  An implementation that keyed the burst on something else (e.g. the state of the mechanism or an
    absolute clock shared with the others) would give a different on-set.  (c) RL's observation has no time (T-14) - so the handwritten family has
    strictly more time information than the RL observation; the disclosure of that asymmetry is checked in T-69."""
    import torch
    from conftest import strategy_dict
    from arbitration.rl import env, policy
    names = [n for n, _ in hw.build_s_hw() if n.startswith("burst")]
    assert names == ["burst dz1 W10 H90", "burst dz1 W30 H170", "burst dz2 W10 H490"]
    tab = hw.public_realizability_table()
    for n in names:
        assert tab[n]["public"] is True and "clock" in tab[n]["info"].lower(), n
    assert not any("clock" in v["info"].lower() for n, v in tab.items() if not n.startswith("burst"))
    cfg = dict(env.m3c_config(0.5), L=2500.0)
    for n in names:
        d = strategy_dict(n)
        W, H = int(d["W"]), int(d["H"])
        T = min(2 * (W + H) + 20, 520)
        out = env.simulate("M3C", T, cfg, [policy.AdapterPolicy(d)], [0, 1, 2], r=0.5, dev="cpu", T_score=T, diagnostic=True)
        lg = out["log"]
        on = ((lg["v1"] - lg["u1"]).abs() > 1e-12)[0]
        exp = torch.tensor([(t % (W + H)) < W for t in range(T)])
        for s_ in range(3):
            got = on[s_]
            # the deviation can only be invisible where clip(u1+dz*..) == u1 (never for dz>0 here); the on-set must equal the clock pattern exactly
            assert torch.equal(got, exp), f"{n}: seed {s_} on-set differs from the own-clock pattern"
