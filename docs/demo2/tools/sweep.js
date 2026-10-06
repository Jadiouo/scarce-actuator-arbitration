const S = require('../sim.js');
const med = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const base = { b: 0.3, rho: 0.9, tau: 1, T: 800 };
const grid = [];
for (const m of Object.keys(S.MECHS)) for (const st of ['const', 'timing', 'trickle', 'edge']) grid.push([m, st]);
if (process.argv[2] === 'grid') {
  for (const [m, st] of grid) {
    const g = [], r = [];
    for (let s = 0; s < 30; s++) { const x = S.run(S.makeWorld(s, 800), { ...base, mech: m, strat: st }); g.push(x.gain[800]); r.push(x.rtH[800]); }
    console.log(m.padEnd(6), st.padEnd(8), 'G med', med(g).toFixed(3), 'honestR/T', med(r).toFixed(4));
  }
}
const SC = require('./scenarios.json');
const out = {};
for (const [name, sc] of Object.entries(SC)) {
  const rows = [];
  for (let s = 0; s < 30; s++) {
    const A = S.run(S.makeWorld(s, 800), { ...sc.cfg, mech: sc.A }), B = S.run(S.makeWorld(s, 800), { ...sc.cfg, mech: sc.B });
    rows.push({ s, gA: A.gain[800], gB: B.gain[800], d: A.gain[800] - B.gain[800], rA: A.rtH[800], rB: B.rtH[800] });
  }
  const key = sc.metric === 'gA' ? (x => x.gA) : (x => x.d);
  const md = med(rows.map(key));
  const pick = rows.reduce((a, b) => Math.abs(key(b) - md) < Math.abs(key(a) - md) ? b : a);
  out[name] = { seed: pick.s, metric: sc.metric || 'diff', medMetric: md, medDiff: med(rows.map(x => x.d)), medGA: med(rows.map(x => x.gA)), medGB: med(rows.map(x => x.gB)), picked: pick,
    min: Math.min(...rows.map(x => x.d)), max: Math.max(...rows.map(x => x.d)) };
  console.log(name, JSON.stringify(out[name], (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
}
require('fs').writeFileSync(__dirname + '/scenario_seeds.json', JSON.stringify(out, null, 1));
