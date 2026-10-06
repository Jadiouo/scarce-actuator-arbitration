(function () {
  'use strict';
  const S = window.Sim, T = S.P.T, $ = id => document.getElementById(id);
  const PRESETS = {
    naive: { seed: 25, A: 'NAIVE', B: 'M3C', strat: 'const', b: 0.3, rho: 0.9, tau: 1,
      text: 'A server that just believes claims rewards an agent that always adds a fixed amount: it takes items it does not need, and the others lose them. With the same lie, an audit at dispatch (B) makes the lie backfire. Check the gain curve and the regret curve.' },
    arrival: { seed: 2, A: 'M2', B: 'NAIVE', strat: 'timing', b: 0.3, rho: 0.9, tau: 1,
      text: 'An audit on arrival compares the claim made at arrival with the truth at arrival. A timing liar inflates when the decision is made and tells the truth when asked to confirm, so it is never caught. The two curves coincide: the audit sees nothing and changes nothing.' },
    dispatch: { seed: 25, A: 'M3C', B: 'M2', strat: 'timing', b: 0.3, rho: 0.9, tau: 1,
      text: 'The same timing liar against an audit that keeps the claim made at the moment of decision and checks it against the truth a little later. Surprises accumulate (CUSUM) and the liar gets suspended. In this model the lie now backfires. Try the trickle or uninformed-edge strategies, and lower r, to see where it gets harder.' },
    raid: { seed: 15, A: 'M4', B: 'M3C', strat: 'trickle', b: 0.3, rho: 0.9, tau: 1,
      text: 'Raids rarely throw a round away to look at the truth right now, and they can target the winner, so a lie has no delay to hide behind. The price is the wasted rounds, visible as higher honest-only R/T. The strategies here are a small hand-written set, so this is not proof that no strategy can win.' }
  };
  const el = { A: null, B: null };
  const st = { cur: 0, speed: 40, playing: false, acc: 0, last: 0, cmp: true, run: { A: null, B: null }, preset: null };
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- setup controls
  for (const id of ['mechA', 'mechB']) for (const [k, m] of Object.entries(S.MECHS)) $(id).add(new Option(`${m.name}: ${m.long}`, k));
  for (const [k, m] of Object.entries(S.STRATS)) $('strat').add(new Option(m.name, k));
  const cfgOf = key => ({ mech: $(key === 'A' ? 'mechA' : 'mechB').value, strat: $('strat').value, b: +$('bsz').value, rho: +$('rho').value, tau: +$('tau').value, T });

  function makePanel(key) {
    const d = document.createElement('section'); d.className = 'panel';
    d.innerHTML = `<h3><span class="tag t${key}">${key}</span><span class="ptitle"></span></h3>
      <canvas class="now" role="img" aria-label="Claims versus truth this round"></canvas>
      <canvas class="tl" role="img" aria-label="Timeline of allocations, audits and suspensions"></canvas>
      <p class="status" aria-live="off"></p>
      <dl class="metrics"></dl>`;
    return { root: d, title: d.querySelector('.ptitle'), now: d.querySelector('.now'), tl: d.querySelector('.tl'), status: d.querySelector('.status'), metrics: d.querySelector('.metrics') };
  }

  function recompute() {
    const world = S.makeWorld(+$('seed').value || 0, T);
    st.run.A = S.run(world, cfgOf('A'));
    st.run.B = st.cmp ? S.run(world, cfgOf('B')) : null;
    $('rV').textContent = Math.pow(+$('rho').value, +$('tau').value).toFixed(3);
    $('rhoV').textContent = (+$('rho').value).toFixed(2); $('tauV').textContent = $('tau').value; $('bV').textContent = (+$('bsz').value).toFixed(2);
    const s = $('strat').value; $('bwrap').hidden = !(s === 'const' || s === 'timing');
    $('helpS').textContent = S.STRATS[s].help;
    $('helpA').textContent = 'A: ' + S.MECHS[$('mechA').value].help;
    $('helpB').textContent = st.cmp ? 'B: ' + S.MECHS[$('mechB').value].help : '';
    $('mechBwrap').hidden = !st.cmp;
    const pr = st.preset && PRESETS[st.preset];
    $('seedLbl').textContent = pr && +$('seed').value === pr.seed ? `Seed ${pr.seed} is the seed closest to the median outcome among seeds 0 to 29 for this scenario (chosen by rule, not for effect size).` : 'Same seed, same world: reload with the same settings to reproduce.';
    layout(); draw();
  }

  function layout() {
    const ar = $('arenas');
    ar.className = 'arenas' + (st.cmp ? ' two' : '');
    ar.textContent = '';
    el.A = makePanel('A'); ar.appendChild(el.A.root);
    if (st.cmp) { el.B = makePanel('B'); ar.appendChild(el.B.root); } else el.B = null;
  }

  // ---------- drawing helpers
  function setup(c, h) {
    const w = Math.max(200, Math.floor(c.parentElement.clientWidth)), dpr = window.devicePixelRatio || 1;
    c.style.height = h + 'px'; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.font = '12px system-ui,sans-serif'; return { g, w, h };
  }
  function hatch(g, x, y, w, h, col) {
    g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); g.strokeStyle = col; g.globalAlpha = .55; g.lineWidth = 1.5;
    for (let i = -h; i < w; i += 6) { g.beginPath(); g.moveTo(x + i, y + h); g.lineTo(x + i + h, y); g.stroke(); }
    g.restore();
  }
  function diamond(g, x, y, r, fill) { g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath(); g.fillStyle = fill; g.fill(); }
  function ring(g, x, y, r, bad) {
    g.beginPath(); g.arc(x, y, r, 0, 7); g.lineWidth = 2;
    if (bad) { g.fillStyle = css('--warn'); g.fill(); } else { g.strokeStyle = css('--pos'); g.stroke(); }
  }

  function drawNow(p, rec, t) {
    const { g, w, h } = setup(p.now, 150), fg = css('--fg'), mut = css('--muted');
    if (t < 1) { g.fillStyle = mut; g.fillText('Press play or step to start.', 10, 24); return; }
    const colW = w / 3, ph = 96, y0 = 14, u = rec.u[t], v = rec.v[t];
    for (let k = 0; k < 3; k++) {
      const x = k * colW, bw = Math.min(30, colW * 0.17), cx = x + colW / 2;
      g.fillStyle = css('--grid'); g.fillRect(x + 6, y0, colW - 12, ph);
      g.fillStyle = css('--truth'); g.fillRect(cx - bw - 3, y0 + ph * (1 - u[k]), bw, ph * u[k]);
      g.fillStyle = k === 1 ? css('--liar') : css('--claim'); g.fillRect(cx + 3, y0 + ph * (1 - v[k]), bw, ph * v[k]);
      if (rec.susp[t][k]) hatch(g, x + 6, y0, colW - 12, ph, css('--warn'));
      if (rec.idx[t] === k) { g.strokeStyle = fg; g.lineWidth = 2.5; g.strokeRect(x + 6, y0, colW - 12, ph); }
      g.fillStyle = k === 1 ? css('--liar') : mut; g.textAlign = 'center';
      g.fillText(k === 1 ? 'Agent 1 (may lie)' : 'Agent ' + k, cx, y0 + ph + 17);
      if (rec.susp[t][k]) { g.fillStyle = css('--warn'); g.fillText('suspended', cx, y0 + ph + 32); }
      if (rec.raid[t] && rec.raid[t].tg === k) diamond(g, x + 16, y0 + 12, 7, rec.raid[t].flag ? css('--warn') : fg);
      const a = rec.audit[t]; if (a && a.ag === k) ring(g, x + colW - 16, y0 + 12, 6, a.flag);
    }
    g.textAlign = 'left';
  }

  function drawTL(p, rec, t) {
    const { g, w, h } = setup(p.tl, 150), mut = css('--muted'), WIN = 80, lab = 56, cw = (w - lab - 4) / WIN;
    const rh = 20, top = 4;
    g.fillStyle = mut; g.textAlign = 'left';
    ['A0', 'A1', 'A2'].forEach((s, k) => { g.fillStyle = k === 1 ? css('--liar') : mut; g.fillText(s, 2, top + k * (rh + 3) + 14); });
    g.fillStyle = mut; g.fillText('A1 claim', 0, top + 3 * (rh + 3) + 24); g.fillText('vs truth', 0, top + 3 * (rh + 3) + 38);
    const t0 = Math.max(1, t - WIN + 1);
    for (let k = 0; k < 3; k++) { g.fillStyle = css('--grid'); g.fillRect(lab, top + k * (rh + 3), WIN * cw, rh); }
    const sy = top + 3 * (rh + 3) + 6, sh = h - sy - 4;
    g.fillStyle = css('--grid'); g.fillRect(lab, sy, WIN * cw, sh);
    let lastTruth = null;
    for (let i = t0; i <= t; i++) {
      const x = lab + (i - t0 + (t < WIN ? 0 : 0)) * cw;
      const ys = k => top + k * (rh + 3);
      for (let k = 0; k < 3; k++) {
        if (rec.susp[i][k]) hatch(g, x, ys(k), Math.max(cw, 1), rh, css('--warn'));
      }
      if (rec.wasted[i]) { g.fillStyle = css('--truth'); g.globalAlpha = .6; g.fillRect(x, top, Math.max(cw, 1), 3 * (rh + 3) - 3); g.globalAlpha = 1; }
      else if (rec.idx[i] >= 0) { g.fillStyle = rec.idx[i] === 1 ? css('--liar') : css('--claim'); g.fillRect(x + .5, ys(rec.idx[i]) + 2, Math.max(cw - 1, 1), rh - 4); }
      const a = rec.audit[i]; if (a) ring(g, x + cw / 2, ys(a.ag) + rh / 2, Math.min(6, cw + 1), a.flag);
      const rd = rec.raid[i]; if (rd) diamond(g, x + cw / 2, ys(rd.tg) + rh / 2, 6, rd.flag ? css('--warn') : css('--fg'));
    }
    // strip: truth vs claim of agent 1
    g.lineWidth = 1.5;
    for (const [col, getter] of [[css('--truth'), i => rec.u[i][1]], [css('--liar'), i => rec.v[i][1]]]) {
      g.strokeStyle = col; g.beginPath();
      for (let i = t0; i <= t; i++) { const x = lab + (i - t0) * cw + cw / 2, y = sy + sh * (1 - getter(i)); i === t0 ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.stroke();
    }
    g.fillStyle = mut; g.textAlign = 'right'; g.fillText(`rounds ${t0} to ${t}`, w - 2, h - 6); g.textAlign = 'left';
  }

  function lineChart(canvas, series, ymin, ymax, t) {
    const { g, w, h } = setup(canvas, Math.min(260, Math.max(190, window.innerWidth * 0.3)));
    const L = 44, R = 10, Tp = 8, B = 22, pw = w - L - R, ph = h - Tp - B, mut = css('--muted');
    const X = i => L + pw * i / T, Y = v => Tp + ph * (1 - (v - ymin) / (ymax - ymin));
    g.strokeStyle = css('--grid'); g.fillStyle = mut; g.lineWidth = 1; g.textAlign = 'right';
    for (let i = 0; i <= 4; i++) { const v = ymin + (ymax - ymin) * i / 4, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(v.toFixed(2), L - 5, y + 4); }
    if (ymin < 0 && ymax > 0) { g.strokeStyle = css('--fg'); g.globalAlpha = .5; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - R, Y(0)); g.stroke(); g.globalAlpha = 1; }
    g.textAlign = 'center'; [0, T / 2, T].forEach(v => g.fillText(String(v), X(v), h - 6));
    for (const s of series) {
      g.strokeStyle = s.col; g.lineWidth = 2; g.setLineDash(s.dash ? [6, 4] : []); g.beginPath();
      for (let i = 1; i <= t; i++) { const x = X(i), y = Y(s.d[i]); i === 1 ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.stroke();
    }
    g.setLineDash([]); g.strokeStyle = mut; g.beginPath(); g.moveTo(X(t), Tp); g.lineTo(X(t), Tp + ph); g.stroke();
    g.textAlign = 'left';
  }
  function range(arrs, pad, floor0) {
    let lo = Infinity, hi = -Infinity; for (const a of arrs) for (let i = 1; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
    if (floor0) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (hi - lo < 0.05) { hi += 0.03; lo -= 0.03; }
    return [lo - (hi - lo) * pad, hi + (hi - lo) * pad];
  }

  function stats(rec, t) {
    let au = 0, rd = 0, hf = 0, ls = 0, rwh = 0;
    for (let i = 1; i <= t; i++) {
      const a = rec.audit[i]; if (a) { au++; if (a.flag && a.ag !== 1) hf++; }
      const r = rec.raid[i]; if (r) { rd++; if (r.flag && r.tg !== 1) hf++; }
      if (rec.liarSusp[i]) ls++;
      const w = rec.rw && rec.rw[i]; if (w) rwh += (w[0] ? 1 : 0) + (w[2] ? 1 : 0);
    }
    return { au, rd, hf, ls, rwh };
  }
  const fmt = (x, d = 3) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);

  function statusText(rec, t, mech) {
    if (t < 1) return '';
    const i = rec.idx[t], u = rec.u[t], v = rec.v[t], parts = [];
    if (rec.wasted[t]) parts.push(`Round ${t}: raid, the round is thrown away.`);
    else parts.push(`Round ${t}: item to agent ${i} (claim ${v[i].toFixed(2)}, truth ${u[i].toFixed(2)}).`);
    const a = rec.audit[t];
    if (a) parts.push(`Audit settled for agent ${a.ag}: ${a.flag ? 'mismatch, suspended' : 'passed'}.`);
    const r = rec.raid[t]; if (r) parts.push(`Raid looked at agent ${r.tg}: ${r.flag ? 'false claim, suspended' : 'claim was true'}.`);
    if (rec.liarSusp[t]) parts.push('Agent 1 is suspended.');
    return parts.join(' ');
  }

  function metricHTML(rec, t, mech) {
    const s = stats(rec, t), g = rec.gain[t] || 0;
    const rows = [
      ['Liar gain G (vs honest)', t ? `<span style="color:${g >= 0 ? css('--warn') : css('--pos')}">${fmt(g)}</span>` : '–'],
      ['Honest-only R/T (price of mechanism)', t ? rec.rtH[t].toFixed(4) : '–'],
      ['R/T with this liar', t ? rec.rtS[t].toFixed(4) : '–'],
      ['Audits settled / raids', mech === 'M4' ? `${s.rd} raids` : (mech === 'M2' || mech === 'M3C') ? `${s.au} audits` : 'none'],
      ['Rounds agent 1 suspended', (mech === 'NAIVE' || mech === 'QM') ? 'n/a' : t ? `${(100 * s.ls / t).toFixed(0)}%` : '–'],
      ['Honest agents wrongly punished', (mech === 'NAIVE' || mech === 'QM') ? 'n/a' : String(s.hf)]
    ];
    if (mech === 'QM') rows[3] = ['Honest claims rewritten', t ? `${(100 * s.rwh / (2 * t)).toFixed(0)}%` : '–'];
    return rows.map(r => `<div><dt>${r[0]}</dt><dd>${r[1]}</dd></div>`).join('');
  }

  function draw() {
    const t = st.cur; $('tV').textContent = `${t} / ${T}`; $('scrub').value = t;
    const keys = st.cmp ? ['A', 'B'] : ['A'];
    for (const k of keys) {
      const p = el[k], rec = st.run[k], m = $(k === 'A' ? 'mechA' : 'mechB').value;
      p.title.textContent = `${S.MECHS[m].name}: ${S.MECHS[m].long}`;
      drawNow(p, rec, t); drawTL(p, rec, t);
      p.status.textContent = statusText(rec, t, m);
      p.metrics.innerHTML = metricHTML(rec, t, m);
    }
    const sa = st.run.A, sb = st.run.B;
    const arrG = [sa.gain].concat(sb ? [sb.gain] : []), [g0, g1] = range(arrG, 0.1, true);
    lineChart($('chartG'), [{ d: sa.gain, col: css('--a') }].concat(sb ? [{ d: sb.gain, col: css('--b'), dash: true }] : []), g0, g1, t);
    const arrR = [sa.rtS].concat(sb ? [sb.rtS] : []), [r0, r1] = range(arrR, 0.1, true);
    lineChart($('chartR'), [{ d: sa.rtS, col: css('--a') }].concat(sb ? [{ d: sb.rtS, col: css('--b'), dash: true }] : []), Math.max(0, r0), r1, t);
    const d = $('delta');
    d.innerHTML = t ? `A: <span class="${sa.gain[t] >= 0 ? 'pos' : 'neg'}">${fmt(sa.gain[t])}</span>` + (sb ? ` &nbsp; B: <span class="${sb.gain[t] >= 0 ? 'pos' : 'neg'}">${fmt(sb.gain[t])}</span>` : '') : '';
    d.querySelectorAll('.pos').forEach(e => e.style.color = css('--warn')); d.querySelectorAll('.neg').forEach(e => e.style.color = css('--pos'));
  }

  // ---------- playback
  function tick(ts) {
    if (!st.playing) return;
    if (!st.last) st.last = ts; st.acc += (ts - st.last) / 1000 * st.speed; st.last = ts;
    const n = Math.floor(st.acc);
    if (n > 0) { st.acc -= n; st.cur = Math.min(T, st.cur + n); draw(); if (st.cur >= T) setPlay(false); }
    if (st.playing) requestAnimationFrame(tick);
  }
  function setPlay(on) {
    st.playing = on; st.last = 0; st.acc = 0; $('pause').textContent = on ? 'Pause' : 'Play'; $('pause').setAttribute('aria-pressed', on ? 'false' : 'false');
    if (on) { if (st.cur >= T) st.cur = 0; requestAnimationFrame(tick); }
  }

  function applyPreset(name) {
    const p = PRESETS[name]; st.preset = name;
    $('mechA').value = p.A; $('mechB').value = p.B; $('strat').value = p.strat; $('bsz').value = p.b; $('rho').value = p.rho; $('tau').value = p.tau; $('seed').value = p.seed;
    st.cmp = true; $('cmp').setAttribute('aria-pressed', 'true'); $('cmp').textContent = 'Side-by-side: on';
    document.querySelectorAll('[data-preset]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.preset === name)));
    $('presetText').innerHTML = `${p.text}<small>Representative seed ${p.seed}: the closest to the median outcome among seeds 0 to 29 under these settings.</small>`;
    st.cur = 0; recompute();
    if (reduceMotion) { st.cur = T; draw(); } else setPlay(true);
  }
  function manual() { st.preset = null; document.querySelectorAll('[data-preset]').forEach(b => b.setAttribute('aria-pressed', 'false')); $('presetText').textContent = ''; recompute(); }

  document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => applyPreset(b.dataset.preset)));
  for (const id of ['mechA', 'mechB', 'strat', 'bsz', 'rho', 'tau', 'seed']) $(id).addEventListener('input', manual);
  $('cmp').addEventListener('click', () => { st.cmp = !st.cmp; $('cmp').setAttribute('aria-pressed', String(st.cmp)); $('cmp').textContent = 'Side-by-side: ' + (st.cmp ? 'on' : 'off'); manual(); });
  $('newseed').addEventListener('click', () => { $('seed').value = Math.floor(Math.random() * 100000); manual(); });
  $('pause').addEventListener('click', () => setPlay(!st.playing));
  $('step').addEventListener('click', () => { setPlay(false); st.cur = Math.min(T, st.cur + 1); draw(); });
  $('restart').addEventListener('click', () => { st.cur = 0; draw(); setPlay(true); });
  $('end').addEventListener('click', () => { setPlay(false); st.cur = T; draw(); });
  $('scrub').addEventListener('input', e => { setPlay(false); st.cur = +e.target.value; draw(); });
  document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => {
    st.speed = +b.dataset.speed; document.querySelectorAll('[data-speed]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 80); });
  if (window.matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  window.__demo = { st, applyPreset, draw };
  applyPreset('naive');
})();
