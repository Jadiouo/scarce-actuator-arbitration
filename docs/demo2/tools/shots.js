// Click every control in headless Chrome, fail on any console error, save screenshots.
const puppeteer = require('puppeteer-core');
const path = require('path');
(async () => {
  const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox'] });
  const out = path.join(__dirname, '..', 'screenshots'); require('fs').mkdirSync(out, { recursive: true });
  const errs = [];
  for (const [label, scheme, vp] of [['light-desktop', 'light', { width: 1280, height: 900 }], ['dark-desktop', 'dark', { width: 1280, height: 900 }], ['light-mobile', 'light', { width: 375, height: 800, isMobile: true, deviceScaleFactor: 2 }], ['dark-mobile', 'dark', { width: 375, height: 800, isMobile: true, deviceScaleFactor: 2 }]]) {
    const page = await browser.newPage(); await page.setViewport(vp);
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
    page.on('console', m => { if (['error', 'warning'].includes(m.type())) errs.push(label + ': ' + m.text()); });
    page.on('pageerror', e => errs.push(label + ' pageerror: ' + e.message));
    await page.goto('file://' + path.join(__dirname, '..', 'index.html'));
    await new Promise(r => setTimeout(r, 400));
    const clicks = []; 
    for (const p of ['naive', 'arrival', 'dispatch', 'raid']) {
      await page.click(`[data-preset=${p}]`); await page.click('#end');
      if (label === 'light-desktop' || label === 'dark-mobile' || label === 'light-mobile' || label === 'dark-desktop') await page.screenshot({ path: path.join(out, `${label}-${p}.png`), fullPage: true });
      clicks.push(p);
      const sw = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      if (sw > 0) errs.push(`${label} ${p}: horizontal overflow ${sw}px`);
    }
    // every other button / selector option
    for (const id of ['pause', 'step', 'restart', 'end', 'cmp', 'cmp', 'newseed']) await page.click('#' + id);
    for (const b of await page.$$('[data-speed]')) await b.click();
    for (const id of ['mechA', 'mechB', 'strat']) {
      const opts = await page.$$eval(`#${id} option`, o => o.map(x => x.value));
      for (const v of opts) await page.select('#' + id, v);
    }
    for (const [id, v] of [['rho', '0.5'], ['rho', '0.99'], ['tau', '5'], ['bsz', '0.6']]) await page.$eval('#' + id, (e, v) => { e.value = v; e.dispatchEvent(new Event('input', { bubbles: true })); }, v);
    await page.$eval('#scrub', e => { e.value = 400; e.dispatchEvent(new Event('input', { bubbles: true })); });
    const txt = await page.evaluate(() => document.body.innerText);
    if (/NaN|undefined|Infinity/.test(txt)) errs.push(label + ': bad text in page');
    const sw = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (sw > 0) errs.push(`${label} after controls: horizontal overflow ${sw}px`);
    await page.close();
  }
  await browser.close();
  console.log(errs.length ? 'PROBLEMS:\n' + errs.join('\n') : 'no console errors, no overflow, no NaN');
  process.exit(errs.length ? 1 : 0);
})();
