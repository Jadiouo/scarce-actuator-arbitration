/* Scarce-actuator arbitration: interactive demo.
 * A simplified JavaScript port of the "v2" model of arbitration/model.py
 * (shock damage, AR(1) report noise, keep_commitment, no preemption).
 * Part 1: simulation core (also runs under node). Part 2: UI (browser only). */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ constants
  const RING = 100, SPEED = 1, SERVICE = 5, SIGMA = 0.15, THETA = 0.30;
  const N = 8, SPREAD = 0.6;
  const SHOCK_RATE = 0.01, AR1_PHI = 0.9, AUDIT_N0 = 1.0, FUSE_G = 60, FUSE_K = 40;
    const POLICIES = {
    none:            'none (round-robin)',
    learned_index:   'learned_index (ignores reports)',
    honest_index:    'honest_index',
    fused_index:     'fused_index',
    audit_index:     'audit_index (audit on arrival)',
    audit_disp_index:'audit_disp_index (audit at dispatch)'
  };
  const POLICY_HELP = {
    none: 'Blind sweep around the ring in spatial order. Ignores every report.',
    learned_index: 'Learns each explorer’s decay rate from what it sees on arrival, then picks by expected urgency per unit of actuator time. Reports are ignored.',
    honest_index: 'Believes claims at face value: among explorers whose claim exceeds a threshold, serves the best claim per unit of travel + service time.',
    fused_index: 'Bayesian fusion of the claim with an age-based prior, then the same distance-aware index. No check on whether claims are honest.',
    audit_index: 'fused_index plus an audit on arrival: the true state is seen then and compared with the claim; each explorer’s estimated bias is deducted.',
    audit_disp_index: 'Same audit, but it compares the claim that was made at the moment of dispatch with what turned out to be true.'
  };
  const INFL_MODES = { constant: 'constant', timing: 'timing (when actuator decides)', distance: 'distance (when actuator is far)' };

  // ------------------------------------------------------------------ PRNG
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeGauss(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0; while (u < 1e-12) u = rand();
      const v = rand(), m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
  }

  // ------------------------------------------------------------------ helpers
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  function ringDist(a, b) { const d = Math.abs(a - b); return Math.min(d, RING - d); }
  const LF = [0];
  function logFact(n) { for (let i = LF.length; i <= n; i++) LF[i] = LF[i - 1] + Math.log(i); return LF[n]; }

  // pmf of tau = min(1, S_age) on grid j/G (as model.age_prior, shock damage)
  function agePrior(age, r, f, p, G) {
    const pmf = new Float64Array(G + 1);
    age = Math.max(0, Math.floor(age));
    if (f === 0) { pmf[Math.min(G, Math.round(Math.min(1, r * age) * G))] = 1; return pmf; }
    const base = (1 - f) * r * age, m = f * r / p;
    if (age * p > 25 || base >= 1) { pmf[G] = 1; return pmf; }
    const pk = new Float64Array(FUSE_K + 1);
    let tot = 0;
    for (let k = 0; k <= FUSE_K; k++) {
      if (k > age) { pk[k] = 0; continue; }
      pk[k] = Math.exp(logFact(age) - logFact(k) - logFact(age - k) + k * Math.log(p) + (age - k) * Math.log(1 - p));
      tot += pk[k];
    }
    for (let k = 0; k <= FUSE_K; k++) pk[k] /= tot;
    const F = new Float64Array(G);
    for (let j = 0; j < G; j++) {
      const c = (j + 0.5) / G;
      let v = c >= base ? pk[0] : 0;
      const y = Math.max(c - base, 0) / m;
      const e = Math.exp(-y);
      let term = e, cum = e;             // cum = e^-y * sum_{i<k} y^i/i!
      for (let k = 1; k <= FUSE_K; k++) {
        v += pk[k] * (1 - cum);
        term *= y / k; cum += term;
      }
      F[j] = Math.min(v, 1);
    }
    for (let j = G - 2; j >= 0; j--) F[j] = Math.min(F[j], F[j + 1]);   // monotone
    let prev = 0;
    for (let j = 0; j < G; j++) { pmf[j] = Math.max(F[j] - prev, 0); prev = F[j]; }
    pmf[G] = Math.max(1 - prev, 0);
    return pmf;
  }
  function fusedTau(claim, age, r, sd, f) {
    if (f === 0) return Math.min(1, r * Math.max(age, 0));
    const G = FUSE_G, pmf = agePrior(age, r, f, SHOCK_RATE, G);
    let wsum = 0, gsum = 0, lmax = -Infinity;
    const lw = new Float64Array(G + 1);
    for (let j = 0; j <= G; j++) {
      const z = (claim - j / G) / Math.max(sd, 1e-9);
      lw[j] = Math.log(pmf[j] + 1e-300) - 0.5 * z * z;
      if (lw[j] > lmax) lmax = lw[j];
    }
    for (let j = 0; j <= G; j++) { const w = Math.exp(lw[j] - lmax); wsum += w; gsum += w * j / G; }
    return gsum / wsum;
  }
  const indexOf = (tau, rh, d) => Math.min(1, tau + rh * d / SPEED) / (d / SPEED + SERVICE);

  // ------------------------------------------------------------------ simulation
  /* cfg: { policy, seed, rate, shockFrac, nInfl, inflMode, b }.
   * Instance (posts, rates, start), shocks, report noise and inflater choice come from
   * policy-independent RNG streams, so two Sims with the same cfg minus policy see the same world. */
  class Sim {
    constructor(cfg) {
      this.cfg = cfg;
      const seed = cfg.seed | 0;
      const rng = mulberry32(seed), g0 = makeGauss(rng);
      this.srng = mulberry32(seed ^ 0x9E3779B1);
      this.nrng = mulberry32(seed ^ 0x85EBCA6B);
      this.ngauss = makeGauss(this.nrng);
      const posts = [];
      for (let i = 0; i < N; i++) posts.push(rng() * RING);
      posts.sort((a, b) => a - b);
      this.posts = posts;
      this.rates = posts.map(() => cfg.rate * Math.exp(SPREAD * g0()));
      this.pos = rng() * RING;
      const perm = [...Array(N).keys()];
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      this.infl = new Array(N).fill(false);
      for (let i = 0; i < Math.min(cfg.nInfl, N); i++) this.infl[perm[i]] = true;
      const f = cfg.shockFrac;
      this.baseDecay = this.rates.map(r => r * (1 - f));
      this.shockSize = this.rates.map(r => f * r / SHOCK_RATE);
      this.e = posts.map(() => SIGMA * this.ngauss());          // AR(1) error, stationary start
      this.arSd = SIGMA * Math.sqrt(1 - AR1_PHI * AR1_PHI);
      this.h = new Array(N).fill(1);
      this.claim = new Array(N).fill(0);
      this.tau = new Array(N).fill(0);
      this.target = -1; this.serving = 0; this.rr = 0; this.t = 0;
      this.nServ = 0;
      // learner
      this.drop = new Array(N).fill(0); this.el = new Array(N).fill(0); this.tReset = new Array(N).fill(0);
      // auditor
      this.aS = new Array(N).fill(0); this.aK = new Array(N).fill(0);
      this.aDc = new Array(N).fill(0); this.aDt = new Array(N).fill(0);
      // metrics
      this.hacc = new Array(N).fill(0); this.sumTeam = 0; this.deadSteps = 0;
      this.win = []; this.winSum = 0; this.WINDOW = 300;
      this.hist = [];
      this.isAudit = cfg.policy === 'audit_index' || cfg.policy === 'audit_disp_index';
      this.usesLearner = ['learned_index', 'fused_index', 'audit_index', 'audit_disp_index'].includes(cfg.policy);
    }
    rhat() {
      const seen = this.el.map(x => x > 0);
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) if (seen[i]) { s += this.drop[i] / this.el[i]; c++; }
      const prior = c ? s / c : 0.005;
      return this.el.map((x, i) => seen[i] ? this.drop[i] / x : prior);
    }
    bias(i) { return this.aS[i] / (this.aK[i] + AUDIT_N0); }
    step() {
      const t = this.t, cfg = this.cfg, h = this.h, tau = this.tau, claim = this.claim;
      let sumH = 0, dead = 0;
      for (let i = 0; i < N; i++) {
        const hit = (this.srng() < SHOCK_RATE ? 1 : 0) * (-Math.log(1 - this.srng())) * this.shockSize[i];
        h[i] = clamp(h[i] - this.baseDecay[i] - hit, 0, 1);
        tau[i] = 1 - h[i];
        this.hacc[i] += h[i]; sumH += h[i]; if (h[i] <= 0) dead++;
        this.e[i] = AR1_PHI * this.e[i] + this.arSd * this.ngauss();
      }
      const team = sumH / N;
      this.sumTeam += team; this.deadSteps += dead / N;
      this.win.push(team); this.winSum += team;
      if (this.win.length > this.WINDOW) this.winSum -= this.win.shift();
      // claims (+ inflation rule)
      const undecided = this.target < 0;
      for (let i = 0; i < N; i++) {
        let eff = 0;
        if (this.infl[i]) {
          if (cfg.inflMode === 'constant') eff = cfg.b;
          else if (cfg.inflMode === 'timing') eff = undecided ? cfg.b : 0;
          else eff = ringDist(this.posts[i], this.pos) > (cfg.distThr || 10) ? cfg.b : 0;
        }
        claim[i] = tau[i] + this.e[i] + eff;
      }
      this.t++;
      if (this.t % 10 === 0) this.hist.push({ t: this.t, cum: this.sumTeam / this.t, roll: this.winSum / this.win.length });
      if (this.serving > 0) {               // a service in progress cannot be cut
        this.serving--;
        if (this.serving === 0) { h[this.target] = 1; this.tReset[this.target] = t; this.target = -1; }
        return;
      }
      const prevTarget = this.target;
      if (this.target < 0) {
        if (cfg.policy === 'none') { this.target = this.rr % N; this.rr++; }
        else {
          const d = this.posts.map(p => ringDist(p, this.pos));
          let best = -1, bs = -Infinity;
          const consider = (i, s) => { if (s > bs) { bs = s; best = i; } };
          if (cfg.policy === 'honest_index') {
            for (let i = 0; i < N; i++) if (claim[i] > THETA) consider(i, claim[i] / (d[i] / SPEED + SERVICE));
          } else {
            const rh = this.rhat(), f = cfg.shockFrac;
            for (let i = 0; i < N; i++) {
              let tp;
              if (cfg.policy === 'learned_index') tp = Math.min(1, rh[i] * (t - this.tReset[i]));
              else {
                let cl = claim[i], sd = SIGMA;
                if (this.isAudit) { cl -= this.bias(i); sd = SIGMA * Math.sqrt(1 + 1 / (this.aK[i] + AUDIT_N0)); }
                tp = fusedTau(cl, t - this.tReset[i], rh[i], sd, f);
              }
              consider(i, indexOf(tp, rh[i], d[i]));
            }
          }
          if (best < 0) return;            // nobody calls: actuator idles
          this.target = best;
        }
      }
      const tg = this.target;
      if (cfg.policy === 'audit_disp_index' && tg !== prevTarget) { this.aDc[tg] = claim[tg]; this.aDt[tg] = t; }
      // move one step
      let dd = this.posts[tg] - this.pos;
      if (dd > RING / 2) dd -= RING; if (dd < -RING / 2) dd += RING;
      let arrived;
      if (Math.abs(dd) <= SPEED) { this.pos = this.posts[tg]; arrived = true; }
      else { this.pos = (((this.pos + Math.sign(dd) * SPEED) % RING) + RING) % RING; arrived = false; }
      if (arrived) {
        this.serving = SERVICE; this.nServ++;
        if (this.isAudit) {
          if (cfg.policy === 'audit_index') this.aS[tg] += claim[tg] - tau[tg];
          else {
            const rh = this.rhat()[tg], dt = t - this.aDt[tg];
            this.aS[tg] += this.aDc[tg] - (tau[tg] - rh * dt) - (cfg.dispOffset || 0);
          }
          this.aK[tg]++;
        }
        if (this.usesLearner) {
          const el = t - this.tReset[tg];
          if (el > 0) { this.drop[tg] += 1 - h[tg]; this.el[tg] += el; }
        }
      }
    }
    // ---- metrics
    get cumMean() { return this.t ? this.sumTeam / this.t : 1; }
    get rollMean() { return this.win.length ? this.winSum / this.win.length : 1; }
    get deadFrac() { return this.t ? this.deadSteps / this.t : 0; }
    groupHealth(inflGroup) {
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) if (this.infl[i] === inflGroup) { s += this.hacc[i]; c++; }
      return c && this.t ? s / c / this.t : NaN;
    }
    groupBias(inflGroup) {
      if (!this.isAudit) return NaN;
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) if (this.infl[i] === inflGroup) { s += this.bias(i); c++; }
      return c ? s / c : NaN;
    }
  }

  // Selection-bias offset of the dispatch audit (as the Python runs): mean audit sample of truthful audit_disp_index runs.
  const calCache = {};
  function calibrateOffset(rate, shockFrac) {
    const key = rate.toFixed(4) + '|' + shockFrac.toFixed(3);
    if (calCache[key] !== undefined) return calCache[key];
    let s = 0, c = 0;
    for (let seed = 9001; seed < 9007; seed++) {
      const m = new Sim({ policy: 'audit_disp_index', seed, rate, shockFrac, nInfl: 0, inflMode: 'constant', b: 0, dispOffset: 0 });
      for (let i = 0; i < 2500; i++) m.step();
      for (let i = 0; i < N; i++) { s += m.aS[i]; c += m.aK[i]; }
    }
    return (calCache[key] = c ? s / c : 0);
  }

  const api = { calibrateOffset, Sim, mulberry32, POLICIES, POLICY_HELP, INFL_MODES, RING, N, SERVICE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ArbCore = api;
  if (typeof document === 'undefined') return;

  // ================================================================== UI
  const $ = id => document.getElementById(id);
  const fmt = (x, d = 3) => (x === undefined || x === null || Number.isNaN(x)) ? '—' : x.toFixed(d);
  const state = { speed: 1, paused: false, compare: false };
  let sims = [], panels = [], acc = 0, last = 0, colors = {};
  const SPEEDS = { 1: 30, 10: 300, 100: 3000 };

  const PRESETS = {
    distance: {
      title: 'Is information worth anything?',
      text: 'Both dispatchers weigh distance. One believes honest reports, the other ignores them and learns decay rates from what it sees on arrival. Watch whether the reports buy any extra health.',
      set: { compare: true, a: 'honest_index', b: 'learned_index', nInfl: 0, mode: 'constant', bsz: 0.4, rate: 0.008, shock: 0.5, seed: 5 }
    },
    naive: {
      title: 'The cost of believing',
      text: 'Three explorers always add a constant to their claim. A dispatcher that fuses claims without checking them keeps being drawn to the liars, while the one that ignores reports is unaffected. Compare the honest explorers’ health with the inflaters’.',
      set: { compare: true, a: 'fused_index', b: 'learned_index', nInfl: 3, mode: 'constant', bsz: 0.4, rate: 0.006, shock: 0.5, seed: 19 }
    },
    audit: {
      title: 'When you audit matters',
      text: 'Two explorers inflate their claim only while the actuator is far away (more than 10 units), so by the time it arrives their claims look honest. An audit on arrival sees little bias; an audit at dispatch compares against the claim that actually won the dispatch. Compare the inflaters\u2019 gain over the honest explorers and the estimated bias. A single run is noisy; across many seeds the inflaters\u2019 advantage under arrival audits is consistent, while the team-level difference is small \u2014 see the paper for averages.',
      set: { compare: true, a: 'audit_index', b: 'audit_disp_index', nInfl: 2, mode: 'distance', dthr: 10, bsz: 0.5, rate: 0.006, shock: 0.5, seed: 22 }
    }
  };

  function readColors() {
    const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
    colors = { bg: g('--panel'), fg: g('--fg'), mute: g('--muted'), line: g('--line'), a: g('--a'), b: g('--b'),
               liar: g('--liar'), act: g('--actuator'), claim: g('--claim') };
  }
  function healthColor(h) { return 'hsl(' + Math.round(8 + 192 * h) + ',62%,' + (46 - 6 * (1 - h)) + '%)'; }

  function params() {
    return { seed: parseInt($('seed').value, 10) || 0, rate: +$('rate').value, shockFrac: +$('shock').value,
             nInfl: +$('ninfl').value, inflMode: $('mode').value, b: +$('bsz').value, distThr: +$('dthr').value };
  }
  function buildSims() {
    const p = params();
    p.dispOffset = calibrateOffset(p.rate, p.shockFrac);
    sims = [new Sim(Object.assign({ policy: $('polA').value }, p))];
    if (state.compare) sims.push(new Sim(Object.assign({ policy: $('polB').value }, p)));
    buildPanels(); updateLabels(); acc = 0;
  }
  function buildPanels() {
    const host = $('arenas'); host.innerHTML = ''; panels = [];
    host.className = 'arenas' + (state.compare ? ' two' : '');
    sims.forEach((s, k) => {
      const el = document.createElement('section'); el.className = 'panel';
      const tag = state.compare ? (k ? 'B' : 'A') : '';
      el.innerHTML = '<h3>' + (tag ? '<span class="tag t' + tag + '">' + tag + '</span>' : '') + POLICIES[s.cfg.policy] + '</h3>' +
        '<canvas></canvas><dl class="metrics">' +
        '<div><dt>Team health, recent</dt><dd data-m="roll"></dd></div>' +
        '<div><dt>Team health, cumulative</dt><dd data-m="cum"></dd></div>' +
        '<div><dt>Time at zero health</dt><dd data-m="dead"></dd></div>' +
        '<div><dt>Honest explorers’ health</dt><dd data-m="hon"></dd></div>' +
        '<div><dt>Inflaters’ health</dt><dd data-m="inf"></dd></div>' +
        '<div><dt>Est. bias b̂ (inflaters / honest)</dt><dd data-m="bh"></dd></div></dl>';
      host.appendChild(el);
      const cv = el.querySelector('canvas');
      panels.push({ el, cv, ctx: cv.getContext('2d'), m: Object.fromEntries([...el.querySelectorAll('[data-m]')].map(x => [x.dataset.m, x])), w: 0 });
    });
    sizeCanvases();
  }
  function sizeCanvases() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    panels.forEach(p => {
      const w = Math.max(220, Math.floor(p.cv.parentElement.clientWidth - 2));
      p.w = w; p.cv.style.width = w + 'px'; p.cv.style.height = w + 'px';
      p.cv.width = Math.round(w * dpr); p.cv.height = Math.round(w * dpr); p.dpr = dpr;
    });
    const c = $('chart'), cw = Math.max(260, c.parentElement.clientWidth - 2), ch = cw < 480 ? 180 : 220;
    c.style.width = cw + 'px'; c.style.height = ch + 'px';
    c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); c.dpr = dpr; c.cw = cw; c.ch = ch;
  }

  // ---- drawing
  function drawArena(p, s) {
    const ctx = p.ctx, W = p.w;
    ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
    ctx.clearRect(0, 0, W, W);
    const cx = W / 2, cy = W / 2, R = W * 0.33, nr = W * 0.032;
    const ang = x => x / RING * 2 * Math.PI - Math.PI / 2;
    const P = (x, r) => [cx + r * Math.cos(ang(x)), cy + r * Math.sin(ang(x))];
    ctx.lineWidth = 1.5; ctx.strokeStyle = colors.line; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    const [ax, ay] = P(s.pos, R);
    // target line
    if (s.target >= 0) {
      const [tx, ty] = P(s.posts[s.target], R);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = colors.act; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
    }
    for (let i = 0; i < N; i++) {
      const th = ang(s.posts[i]), ux = Math.cos(th), uy = Math.sin(th), vx = -uy, vy = ux;
      const [x, y] = P(s.posts[i], R), h = s.h[i], rad = nr * (0.55 + 0.6 * h);
      // bars: true urgency (left), claim (right)
      const L = W * 0.12, b0 = nr * 1.45, off = W * 0.011, bw = W * 0.012;
      const bar = (o, len, col, hollow) => {
        const x0 = x + ux * b0 + vx * o, y0 = y + uy * b0 + vy * o;
        ctx.lineWidth = bw; ctx.lineCap = 'butt'; ctx.strokeStyle = col;
        if (hollow) { ctx.globalAlpha = 0.9; }
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + ux * len, y0 + uy * len); ctx.stroke(); ctx.globalAlpha = 1;
      };
      bar(-off, clamp(s.tau[i], 0, 1) * L, colors.mute);
      bar(off, clamp(s.claim[i], 0, 1.3) * L, s.infl[i] ? colors.liar : colors.claim);
      // node
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fillStyle = healthColor(h); ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = h <= 0 ? colors.fg : colors.bg; ctx.stroke();
      if (s.infl[i]) {
        ctx.lineWidth = 2.2; ctx.strokeStyle = colors.liar; ctx.beginPath(); ctx.arc(x, y, nr * 1.28, 0, 7); ctx.stroke();
        const bxp = x - ux * nr * 1.9, byp = y - uy * nr * 1.9;      // badge inside the ring
        ctx.fillStyle = colors.liar; ctx.beginPath(); ctx.arc(bxp, byp, nr * 0.62, 0, 7); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = 'bold ' + Math.round(nr * 0.95) + 'px system-ui,sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('!', bxp, byp + 0.5);
      }
      if (i === s.target && s.serving > 0) {      // service progress
        ctx.lineWidth = 3; ctx.strokeStyle = colors.act; ctx.beginPath();
        ctx.arc(x, y, nr * 1.28, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * (1 - s.serving / SERVICE)); ctx.stroke();
      }
    }
    // actuator
    ctx.save(); ctx.translate(ax, ay); ctx.rotate(Math.PI / 4);
    const a = nr * 0.95; ctx.fillStyle = colors.act; ctx.strokeStyle = colors.bg; ctx.lineWidth = 2;
    ctx.fillRect(-a, -a, 2 * a, 2 * a); ctx.strokeRect(-a, -a, 2 * a, 2 * a); ctx.restore();
    // centre readout
    ctx.fillStyle = colors.mute; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = Math.round(W * 0.034) + 'px system-ui,sans-serif';
    ctx.fillText('step ' + s.t, cx, cy - W * 0.02);
    ctx.fillText(s.serving > 0 ? 'servicing' : (s.target >= 0 ? 'travelling' : 'idle'), cx, cy + W * 0.025);
  }

  function drawChart() {
    const c = $('chart'), ctx = c.getContext('2d'), W = c.cw, H = c.ch;
    ctx.setTransform(c.dpr, 0, 0, c.dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const m = { l: 40, r: 10, t: 14, b: 22 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    let lo = 1, hi = 0, tmax = 10;
    sims.forEach(s => s.hist.forEach(q => { lo = Math.min(lo, q.cum, q.roll); hi = Math.max(hi, q.cum, q.roll); tmax = Math.max(tmax, q.t); }));
    if (hi <= lo) { lo = 0; hi = 1; }
    lo = Math.max(0, Math.floor((lo - 0.03) * 20) / 20); hi = Math.min(1, Math.ceil((hi + 0.03) * 20) / 20);
    const X = t => m.l + t / tmax * pw, Y = v => m.t + (1 - (v - lo) / (hi - lo)) * ph;
    ctx.font = '11px system-ui,sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 1; ctx.strokeStyle = colors.line; ctx.fillStyle = colors.mute;
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(W - m.r, y); ctx.stroke(); ctx.fillText(v.toFixed(2), m.l - 6, y); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let k = 0; k <= 4; k++) ctx.fillText(Math.round(tmax * k / 4), m.l + pw * k / 4, H - m.b + 5);
    sims.forEach((s, k) => {
      const col = k ? colors.b : colors.a;
      const line = (key, w, alpha, dash) => {
        ctx.beginPath(); ctx.lineWidth = w; ctx.strokeStyle = col; ctx.globalAlpha = alpha; ctx.setLineDash(dash);
        s.hist.forEach((q, j) => { j ? ctx.lineTo(X(q.t), Y(q[key])) : ctx.moveTo(X(q.t), Y(q[key])); });
        ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
      };
      line('roll', 1.2, 0.35, []); line('cum', 2.4, 1, k ? [6, 3] : []);
    });
  }

  function updateMetrics() {
    sims.forEach((s, i) => {
      const m = panels[i].m;
      m.roll.textContent = fmt(s.rollMean); m.cum.textContent = fmt(s.cumMean);
      m.dead.textContent = (s.deadFrac * 100).toFixed(1) + '%';
      m.hon.textContent = fmt(s.groupHealth(false)); m.inf.textContent = s.cfg.nInfl ? fmt(s.groupHealth(true)) : '—';
      m.bh.textContent = s.isAudit ? (s.cfg.nInfl ? fmt(s.groupBias(true), 2) : '—') + ' / ' + fmt(s.groupBias(false), 2) : 'n/a (no audit)';
    });
    const d = $('delta');
    if (sims.length === 2) {
      const v = sims[0].cumMean - sims[1].cumMean;
      d.hidden = false;
      $('deltaV').textContent = (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
      $('deltaV').className = v >= 0 ? 'pos' : 'neg';
      const gain = x => { const g = x.groupHealth(true) - x.groupHealth(false); return Number.isNaN(g) ? '\u2014' : (g >= 0 ? '+' : '\u2212') + Math.abs(g).toFixed(3); };
      $('gainV').textContent = sims[0].cfg.nInfl ? 'A ' + gain(sims[0]) + ' | B ' + gain(sims[1]) : '\u2014 (no inflaters)';
    } else d.hidden = true;
    $('legendAB').hidden = sims.length !== 2;
  }

  function frame(ts) {
    const dt = Math.min(0.1, (ts - last) / 1000); last = ts;
    if (!state.paused) {
      acc += dt * SPEEDS[state.speed];
      const n = Math.min(Math.floor(acc), 600); acc -= Math.floor(acc);
      for (let k = 0; k < n; k++) sims.forEach(s => s.step());
    }
    panels.forEach((p, i) => drawArena(p, sims[i]));
    drawChart(); updateMetrics();
    requestAnimationFrame(frame);
  }

  // ---- controls
  function updateLabels() {
    $('rateV').textContent = (+$('rate').value).toFixed(3);
    $('shockV').textContent = (+$('shock').value).toFixed(2);
    $('nV').textContent = $('ninfl').value; $('bV').textContent = (+$('bsz').value).toFixed(2);
    $('polBwrap').hidden = !state.compare;
    $('helpA').textContent = (state.compare ? 'A: ' : '') + POLICY_HELP[$('polA').value];
    $('helpB').textContent = state.compare ? 'B: ' + POLICY_HELP[$('polB').value] : '';
    $('helpB').hidden = !state.compare;
    $('cmp').setAttribute('aria-pressed', state.compare);
    $('cmp').textContent = state.compare ? 'Side-by-side: on' : 'Side-by-side: off';
    const none = +$('ninfl').value === 0;
    $('mode').disabled = none; $('dthr').disabled = none || $('mode').value !== 'distance'; $('bsz').disabled = none;
    $('seedLbl').textContent = 'Same seed = same world, shocks and noise for every policy.';
    document.querySelectorAll('[data-speed]').forEach(b => b.setAttribute('aria-pressed', +b.dataset.speed === state.speed));
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
    $('pause').setAttribute('aria-pressed', state.paused);
  }
  // reserve space for variable-length captions so the layout does not jump
  function reserve(elm, texts, html) {
    const probe = elm.cloneNode(false); probe.removeAttribute('id'); probe.removeAttribute('hidden');
    probe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none;min-height:0;height:auto;display:block;width:' + elm.getBoundingClientRect().width + 'px';
    elm.parentElement.appendChild(probe);
    let hmax = 0;
    for (const tx of texts) { if (html) probe.innerHTML = tx; else probe.textContent = tx; hmax = Math.max(hmax, probe.getBoundingClientRect().height); }
    probe.remove(); elm.style.minHeight = Math.ceil(hmax) + 'px';
  }
  function reserveAll() {
    const hs = Object.values(POLICY_HELP);
    reserve($('helpA'), hs.map(x => 'A: ' + x)); reserve($('helpB'), hs.map(x => 'B: ' + x));
    reserve($('presetText'), Object.values(PRESETS).map(P => '<strong>' + P.title + '.</strong> ' + P.text), true);
  }
  function applyPreset(key) {
    const P = PRESETS[key], s = P.set;
    state.compare = s.compare; $('polA').value = s.a; $('polB').value = s.b; $('ninfl').value = s.nInfl;
    $('mode').value = s.mode; if (s.seed) $('seed').value = s.seed; $('dthr').value = s.dthr || 10; $('bsz').value = s.bsz; $('rate').value = s.rate; $('shock').value = s.shock;
    $('presetText').innerHTML = '<strong>' + P.title + '.</strong> ' + P.text;
    document.querySelectorAll('[data-preset]').forEach(b => b.setAttribute('aria-pressed', b.dataset.preset === key));
    buildSims();
  }
  function fastForward(n) {
    for (let k = 0; k < n; k++) sims.forEach(s => s.step());
  }
  function init() {
    for (const [k, v] of Object.entries(POLICIES)) {
      $('polA').add(new Option(v, k)); $('polB').add(new Option(v, k));
    }
    for (const [k, v] of Object.entries(INFL_MODES)) $('mode').add(new Option(v, k));
    ['polA', 'polB', 'rate', 'shock', 'ninfl', 'mode', 'bsz', 'seed', 'dthr'].forEach(id => $(id).addEventListener('input', () => {
      document.querySelectorAll('[data-preset]').forEach(b => b.setAttribute('aria-pressed', false));
      buildSims();
    }));
    $('cmp').onclick = () => { state.compare = !state.compare; buildSims(); };
    $('reset').onclick = buildSims;
    $('newseed').onclick = () => { $('seed').value = Math.floor(Math.random() * 9000) + 1; buildSims(); };
    $('pause').onclick = () => { state.paused = !state.paused; updateLabels(); };
    $('ff').onclick = () => fastForward(2000);
    document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => { state.speed = +b.dataset.speed; updateLabels(); });
    document.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => applyPreset(b.dataset.preset));
    window.addEventListener('resize', () => { sizeCanvases(); reserveAll(); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);
    readColors();
    $('polA').value = 'audit_index'; $('polB').value = 'audit_disp_index';
    applyPreset('audit'); reserveAll();
    window.addEventListener('load', reserveAll);
    requestAnimationFrame(t => { last = t; frame(t); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})(typeof window !== 'undefined' ? window : globalThis);
