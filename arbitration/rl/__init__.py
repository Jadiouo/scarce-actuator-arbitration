"""Direction 2 (learned adversary vs M3C): implementation skeleton.

Every public function/class below is a signature plus docstring; bodies raise NotImplementedError("REQ-xxx").
Specification: docs/direction2-spec.md (v1.0-rev1+).  Acceptance tests: tests/direction2/.
Module -> REQ block:  env (ENV) | obs, obs_reconstruct (OBS) | policy (ACT) | cmaes (OPT-01/04) | train (OPT-02/03/06, SEED-03)
budget (OPT-08..11, GPU-08) | seeds (SEED) | hw (HW) | metrics (MET) | numaudit (MET-09) | vulns (S1-01..07)
s1 (S1-08..22) | stop (STOP) | m4 (M4) | meas (MEAS) | gpu_rules (GPU-01/06/07) | parts (GPU-03/04) | nbr (NBR)
| ablation (S1-19) | trunc (ENV-20).
"""
