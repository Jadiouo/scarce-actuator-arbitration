"""CPU reference for the features that model.run does not have: per-explorer inflation RULES
(constant / only while the actuator is undecided / only while it is far / only while the explorer's tau is
low / only while it is committed), the dispatch-time audit (audit_disp_index, audit_disp_penalty_index),
a burn-in for the health metrics, and the 5th percentile of the per-step minimum health.

`run_ref` is model.run with a handful of patches applied to its source text (so everything else is the
same code), used by the tests to check sim_torch seed by seed."""
import inspect
import math
import numpy as np
from .. import model as _m

_src = inspect.getsource(_m.run)


def _patch(src):
    def rep(a, b):
        nonlocal src
        assert src.count(a) == 1, (src.count(a), a)
        src = src.replace(a, b)
    rep("inflate=None, stream=0, **opts):", "inflate=None, stream=0, imode=None, ipar=None, disp_offset=0.0, burn=0, **opts):")
    rep('''        if infl is not None:
            claim = claim + infl
''', '''        if infl is not None:
            if imode is None:
                claim = claim + infl
            else:
                dd_ = ring_distance(posts, pos)
                on = np.where(np.asarray(imode) == 0, True, np.where(np.asarray(imode) == 1, target < 0,
                    np.where(np.asarray(imode) == 2, dd_ > np.asarray(ipar), np.where(np.asarray(imode) == 3,
                    tau < np.asarray(ipar), target >= 0))))
                eff = infl * on.astype(float)
                if (np.asarray(imode) == 5).any():
                    sa = auditor.s if auditor is not None else np.zeros(n)
                    ka = auditor.k if auditor is not None else np.zeros(n)
                    cap = np.maximum(np.minimum(np.asarray(ipar), cfg.audit_z * max(float(sigma), 1e-9) * np.sqrt(ka + 1.0) - sa - 0.01), 0.0)
                    eff = np.where(np.asarray(imode) == 5, cap, eff)
                claim = claim + eff
''')
    rep('''auditor = (_Auditor(n, sigma, cfg.audit_n0, cfg.audit_z,
                        cfg.penalty_T if policy == "audit_penalty_index" else 0)
               if policy in ("audit_index", "audit_penalty_index") else None)''',
        '''auditor = (_Auditor(n, sigma, cfg.audit_n0, cfg.audit_z,
                        cfg.penalty_T if policy in PEN else 0)
               if policy in AUD else None)
    disp_claim, disp_t = np.zeros(n), np.zeros(n)''')
    src = src.replace('policy == "audit_penalty_index"', 'policy in PEN')
    rep('''        hacc += h
        acc += (h.mean(), h.min(), (h < 0.2).mean(), float((h <= 0).any()),
                (h <= 0).mean())''', '''        if t >= burn:
            hacc += h
            acc += (h.mean(), h.min(), (h < 0.2).mean(), float((h <= 0).any()),
                    (h <= 0).mean())
            mins.append(h.min())''')
    rep("    acc = np.zeros(5)\n", "    acc = np.zeros(5)\n    mins = _MINS\n")
    rep('''        if target != prev_target and target >= 0:
            d_num, d_den = tau[target], max(tau.max(), 1e-9)''', '''        if target != prev_target and target >= 0:
            d_num, d_den = tau[target], max(tau.max(), 1e-9)
            disp_claim[target], disp_t[target] = claim[target], t''')
    rep('''            if uses_learner:
                learner.observe(target, h[target], t)
            if auditor is not None:
                auditor.observe(target, claim[target], tau[target], t)''', '''            if policy in DISP:
                rh_ = learner.rhat()[target]
                samp = (disp_claim[target] - (tau[target] - rh_ * (t - disp_t[target]))) - disp_offset
            if uses_learner:
                learner.observe(target, h[target], t)
            if auditor is not None:
                if policy in DISP:
                    auditor.observe(target, samp, 0.0, t)
                else:
                    auditor.observe(target, claim[target], tau[target], t)''')
    rep("    m, mn, b, dead, dead_r = acc / steps", "    m, mn, b, dead, dead_r = acc / max(steps - burn, 1)")
    rep("dead_r, hacc / steps)", "dead_r, hacc / max(steps - burn, 1))")
    rep("def run(", "def run_ref(")
    return src


GAME = _m.GAME_POLICIES + ("audit_disp_index", "audit_disp_penalty_index")
PEN = ("audit_penalty_index", "audit_disp_penalty_index")
DISP = ("audit_disp_index", "audit_disp_penalty_index")
AUD = ("audit_index", "audit_penalty_index") + DISP
_ns = dict(vars(_m))
_ns.update(_MINS=[], GAME_POLICIES=GAME, PEN=PEN, DISP=DISP, AUD=AUD)
exec(compile(_patch(_src), "<run_ref>", "exec"), _ns)
_run_ref = _ns["run_ref"]


def run_ref(policy, **kw):
    """Returns (Result, p05 of the per-step min health after burn-in)."""
    _ns["_MINS"] = mins = []
    r = _run_ref(policy, **kw)
    return r, (float(np.percentile(mins, 5)) if mins else float("nan"))
