/* Simplified abstract model of "when to audit a free claim".
   K=3 agents, one item per round, agent 1 may misreport, the others are honest.
   Types: Gaussian AR(1) in z-space, marginals U[0.2,1], U[0,1], cdf x^2.
   Verification arrives tau rounds late; the control is r = rho^tau.
   Pure JS, no dependencies; works in the browser (window.Sim) and in node. */
(function (root) {
  'use strict';
  const K = 3, CMIN = 0.2, LIAR = 1, KREF = 0.5;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) { let u = 0; while (u <= 1e-12) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }

  function erfc(x) { // Numerical Recipes, |err| < 1.2e-7
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
  function PhiInv(p) { // Acklam
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425; let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const clamp = (x, lo, hi) => x < lo ? lo : x > hi ? hi : x;
  const F = (k, x) => clamp(k === 0 ? (x - CMIN) / (1 - CMIN) : k === 1 ? x : x * x, 0, 1);
  const Finv = (k, q) => k === 0 ? CMIN + (1 - CMIN) * q : k === 1 ? q : Math.sqrt(q);
  const zOf = (k, v) => PhiInv(clamp(F(k, v), 1e-6, 1 - 1e-6));

  const MECHS = {
    NAIVE: { name: 'NAIVE', long: 'Believe the reports', help: 'Give the item to whoever claims the most. No checking at all.' },
    M2: { name: 'M2', long: 'Audit on arrival', help: 'Sometimes check the winner when the server arrives, comparing the claim made then with what is seen then. A caught agent is suspended for a while.' },
    M3C: { name: 'M3C', long: 'Audit at dispatch + CUSUM', help: 'Sometimes check the winner’s claim from the moment of the decision against the truth seen tau rounds later, accumulate the surprises (CUSUM) and suspend only when they pile up. Suspension is forgiven after a while.' },
    QM: { name: 'QM', long: 'Marginal quota', help: 'No checking. Each agent may use each claim level only as often as its true type would, per block of rounds; extra claims are rewritten to the nearest level still allowed.' },
    M4: { name: 'M4', long: 'Low-probability raid', help: 'Rarely, throw away a round and look at the truth right now for one agent, who may be the winner. A false claim found then gets the agent suspended. A raid costs one round.' }
  };
  const STRATS = {
    honest: { name: 'Honest', help: 'Reports the true urgency.' },
    const: { name: 'Constant inflation', help: 'Adds b to its claim every round.' },
    timing: { name: 'Timing inflation', help: 'Inflates when the decision is made, tells the truth when asked to confirm at arrival.' },
    trickle: { name: 'Trickle', help: 'A small shift of the claim, sized to stay just under what the audit notices.' },
    edge: { name: 'Uninformed edge', help: 'Only when it would barely lose, bids a learned threshold. Uses only its own type and history, not the others’ claims.' }
  };

  const P = { p: 0.15, h: 5, L: 40, eps: 0.05, rw: 1 / 3, QW: 100, nb: 5, L2: 20, L4: 250, T: 800, tauMax: 6, band: 0.15, eta: 0.03 };

  function makeWorld(seed, T) {
    const N = T + P.tauMax + 2;
    const r1 = mulberry32(seed >>> 0), r2 = mulberry32((seed + 1000003) >>> 0), r3 = mulberry32((seed + 2000003) >>> 0), r4 = mulberry32((seed + 3000007) >>> 0);
    const w = { seed, T, N, z0: [gauss(r1), gauss(r1), gauss(r1)], E: [], Ua: [], Rn: [], Rx: [] };
    for (let t = 0; t <= N; t++) {
      w.E.push([gauss(r1), gauss(r1), gauss(r1)]);
      w.Ua.push(r2());
      w.Rn.push([r3(), r3(), r3(), r3(), r3()]);
      w.Rx.push(r4());
    }
    return w;
  }
  function types(world, rho) {
    const sq = Math.sqrt(1 - rho * rho), z = [world.z0.slice()], u = [];
    for (let t = 1; t <= world.N; t++) {
      const zp = z[t - 1], zn = [0, 0, 0];
      for (let k = 0; k < K; k++) zn[k] = rho * zp[k] + sq * world.E[t][k];
      z[t] = zn;
    }
    for (let t = 0; t <= world.N; t++) u[t] = [0, 1, 2].map(k => Finv(k, Phi(z[t][k])));
    return { z, u };
  }

  /* cfg: {mech, strat, b, rho, tau, T}. Returns per-round records (index 1..T). */
  function simulate(world, cfg) {
    const T = cfg.T || world.T, tau = cfg.tau, rho = cfg.rho, b = cfg.b;
    const r = Math.pow(rho, tau), sd = Math.sqrt(Math.max(1 - r * r, 1e-12)), crit = KREF * sd / Math.max(r, 1e-9);
    const { z, u } = types(world, rho);
    const mech = cfg.mech, strat = cfg.strat;
    const susp = [0, 0, 0], cus = [0, 0, 0];
    const pend = {};          // time -> {ag, pv}
    let th = 0.5;
    const nb = P.nb, Qcap0 = []; for (let i = 0; i < nb; i++) Qcap0.push(Math.floor((i + 1) * P.QW / nb) - Math.floor(i * P.QW / nb));
    let cap = [Qcap0.slice(), Qcap0.slice(), Qcap0.slice()];
    const rec = { T, r, u: [], v: [], vt: [], idx: [], wasted: [], raid: [], audit: [], susp: [], liarSusp: [], gain: [0], util1: [0], regret: [0], nAudit: 0, nRaid: 0, flagsLiar: 0, flagsHonest: 0, rewrites: 0, honestRewrites: 0 };
    let u1c = 0, regc = 0;
    for (let t = 1; t <= T; t++) {
      const ut = u[t], elig = [0, 1, 2].map(k => t > susp[k]);
      // ---- liar's claim
      let v1 = ut[LIAR], vv = ut[LIAR], lied = false;
      if (strat === 'const') { v1 = Math.min(1, ut[LIAR] + b); lied = true; vv = v1; }
      else if (strat === 'timing') { v1 = Math.min(1, ut[LIAR] + b); lied = true; vv = ut[LIAR]; }
      else if (strat === 'trickle') { v1 = Math.min(1, Phi(z[t][LIAR] + crit)); lied = true; vv = v1; }
      else if (strat === 'edge') {
        if (ut[LIAR] < th && ut[LIAR] >= th - P.band) { v1 = Math.min(1, th); lied = true; }
        vv = v1;
      }
      let v = [ut[0], v1, ut[2]];
      let idx, win = true, rd = false, tg = -1, rewritten = [false, false, false];
      if (mech === 'QM') {
        const rep = [0, 0, 0];
        for (let k = 0; k < K; k++) {
          const q = k === LIAR ? v[k] : F(k, ut[k]);
          const d = Math.min(nb - 1, Math.floor(q * nb));
          let best = -1, bc = 1e9;
          for (let c = 0; c < nb; c++) if (cap[k][c] > 0) { const co = Math.abs(c - d) + (c > d ? 1e-3 : 0); if (co < bc) { bc = co; best = c; } }
          if (best < 0) best = d;
          const honestBin = Math.min(nb - 1, Math.floor(F(k, ut[k]) * nb));
          if (best !== honestBin) { rewritten[k] = true; if (k !== LIAR) rec.honestRewrites++; else rec.rewrites++; }
          cap[k][best]--; rep[k] = Finv(k, (best + 0.5) / nb);
        }
        v = rep; idx = 0; for (let k = 1; k < K; k++) if (v[k] > v[idx]) idx = k;
        if (t % P.QW === 0) cap = [Qcap0.slice(), Qcap0.slice(), Qcap0.slice()];
      } else {
        idx = -1; let mx = -1;
        for (let k = 0; k < K; k++) if (elig[k] && v[k] > mx) { mx = v[k]; idx = k; }
        win = mx >= 0; if (!win) idx = -1;
        if (mech === 'M4' && win) {
          const Rn = world.Rn[t]; let tgo = -1, best = -1;
          for (let k = 0; k < K; k++) if (elig[k] && k !== idx && Rn[2 + k] > best) { best = Rn[2 + k]; tgo = k; }
          const pickW = world.Rx[t] < P.rw;
          tg = pickW ? idx : tgo;
          rd = Rn[0] < P.eps && (pickW || tgo >= 0);
          if (rd) {
            rec.nRaid++;
            if (v[tg] - ut[tg] > 1e-9) { susp[tg] = t + P.L4; if (tg === LIAR) rec.flagsLiar++; else rec.flagsHonest++; }
          } else tg = -1;
        }
      }
      const winEff = win && !rd;
      let ag, uw;
      if (rd) { ag = -1; uw = 0; }
      else if (winEff) { ag = idx; uw = ut[idx]; }
      else { ag = Math.min(K - 1, Math.floor(world.Rn[t][1] * K)); uw = ut[ag]; }
      const umax = Math.max(ut[0], ut[1], ut[2]);
      regc += umax - uw; if (ag === LIAR) u1c += uw;
      if (strat === 'edge' && lied && elig[LIAR] && win) th += P.eta * ((idx !== LIAR ? 1 : 0) - 0.5);
      // ---- schedule and settle audits
      if (winEff && (mech === 'M2' || mech === 'M3C') && world.Ua[t] < P.p) { pend[t + tau] = { ag: idx, pv: v[idx] }; rec.nAudit++; }
      let auditEv = null;
      const pa = pend[t];
      if (pa) {
        const k = pa.ag, e = t > susp[k];
        let flag = false, detail = null;
        if (mech === 'M2') {
          const claimNow = k === LIAR ? vv : ut[k];
          flag = e && claimNow - ut[k] > 1e-9; detail = { claim: claimNow, truth: ut[k] };
        } else {
          const zv = zOf(k, pa.pv), zw = zOf(k, ut[k]);
          const resid = (zw - r * zv) / sd;
          const Sn = Math.max(0, cus[k] - resid - KREF);
          if (e) { flag = Sn > P.h; cus[k] = flag ? 0 : Sn; }
          detail = { resid, S: Sn };
        }
        if (flag) { susp[k] = t + (mech === 'M2' ? P.L2 : P.L); if (k === LIAR) rec.flagsLiar++; else rec.flagsHonest++; }
        auditEv = { ag: k, flag, ...detail };
        delete pend[t];
      }
      rec.u[t] = ut.slice(); rec.v[t] = v.slice(); rec.vt[t] = vv; rec.idx[t] = rd ? -1 : ag; rec.wasted[t] = rd;
      rec.raid[t] = rd ? { tg, flag: v[tg] - ut[tg] > 1e-9 } : null; rec.audit[t] = auditEv;
      rec.susp[t] = [0, 1, 2].map(k => t <= susp[k]); rec.liarSusp[t] = t <= susp[LIAR];
      rec.util1[t] = u1c; rec.regret[t] = regc; (rec.rw = rec.rw || [])[t] = rewritten;
    }
    return rec;
  }

  /* Full paired run: strategy run + honest counterfactual in the same world. */
  function run(world, cfg) {
    const s = simulate(world, cfg);
    const h = cfg.strat === 'honest' ? s : simulate(world, Object.assign({}, cfg, { strat: 'honest' }));
    const T = s.T, gain = [0], gR = [0];
    for (let t = 1; t <= T; t++) { gain[t] = (s.util1[t] - h.util1[t]) / t; gR[t] = s.regret[t] / t; }
    s.gain = gain; s.rtS = gR; s.rtH = h.regret.map((x, t) => t ? x / t : 0); s.honest = h;
    return s;
  }
  const api = { K, MECHS, STRATS, P, makeWorld, simulate, run, mulberry32, Phi, PhiInv };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Sim = api;
})(typeof window !== 'undefined' ? window : globalThis);
