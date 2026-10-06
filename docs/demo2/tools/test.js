const S = require('../sim.js');
let bad = 0;
const chk = (c, m) => { if (!c) { bad++; console.log('FAIL', m); } };
const fin = r => [r.gain, r.rtS, r.rtH, r.util1, r.regret].every(a => a.every(Number.isFinite));
for (const m of Object.keys(S.MECHS)) for (const st of Object.keys(S.STRATS)) for (const [rho, tau] of [[0.5, 1], [0.9, 1], [0.99, 5], [0.9, 3]]) {
  const w = S.makeWorld(7, 800), cfg = { mech: m, strat: st, b: 0.3, rho, tau, T: 800 };
  const a = S.run(w, cfg), c = S.run(S.makeWorld(7, 800), cfg);
  chk(fin(a), `NaN ${m} ${st} ${rho} ${tau}`);
  chk(JSON.stringify([a.gain, a.regret, a.idx]) === JSON.stringify([c.gain, c.regret, c.idx]), `repro ${m} ${st}`);
  if (st === 'honest') chk(Math.abs(a.gain[800]) < 1e-12, `honest gain nonzero ${m}`);
}
const a = S.run(S.makeWorld(1, 800), { mech: 'NAIVE', strat: 'const', b: .3, rho: .9, tau: 1, T: 800 });
const b2 = S.run(S.makeWorld(2, 800), { mech: 'NAIVE', strat: 'const', b: .3, rho: .9, tau: 1, T: 800 });
chk(a.gain[800] !== b2.gain[800], 'seeds differ');
console.log(bad ? bad + ' failures' : 'all ok');
process.exit(bad ? 1 : 0);
