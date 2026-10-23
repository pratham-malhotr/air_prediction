/* Optional browser QA: real HTTP requests to the saved Python models. */
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {spawn} = require('node:child_process');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'artifacts', 'frontend_review');

function playwright() {
  try { return require('playwright'); } catch {}
  const cache = path.join(os.homedir(), '.npm', '_npx');
  if (fs.existsSync(cache)) {
    for (const entry of fs.readdirSync(cache)) {
      const candidate = path.join(cache, entry, 'node_modules', 'playwright');
      if (fs.existsSync(path.join(candidate, 'package.json'))) return require(candidate);
    }
  }
  throw new Error('The optional browser check needs Playwright and Chromium. Install with npm install --no-save playwright, then npx playwright install chromium.');
}

async function main() {
  const {chromium} = playwright();
  fs.mkdirSync(output, {recursive: true});
  const server = spawn('bash', [path.join(root, 'scripts/start_frontend.sh'), '--no-open', '--port', '8016'], {cwd: root});
  let browser;
  let startup = '';
  const errors = [];
  try {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('The model service did not become ready.\n' + startup)), 40000);
      server.stdout.on('data', chunk => { startup += chunk.toString(); if (startup.includes('Aircast is ready:')) { clearTimeout(timeout); resolve(); } });
      server.stderr.on('data', chunk => { startup += chunk.toString(); });
      server.once('error', error => { clearTimeout(timeout); reject(error); });
      server.once('exit', code => { if (code) { clearTimeout(timeout); reject(new Error(startup)); } });
    });
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}, deviceScaleFactor: 1});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8016', {waitUntil: 'networkidle'});
    await page.locator('#connection-text').filter({hasText: 'Models connected'}).waitFor();
    await page.locator('#forecast-button:not([disabled])').waitFor();
    assert(Number(await page.locator('#forecast-value').innerText()) > 0);
    await page.click('.chart-ranges button[data-hours="24"]');
    assert.equal(await page.locator('.chart-ranges button[data-hours="24"]').getAttribute('aria-pressed'), 'true');
    await page.click('.chart-ranges button[data-hours="168"]');
    await page.locator('#history-chart').hover({position: {x: 100, y: 100}});
    assert(await page.locator('#chart-tooltip').isVisible());
    await page.click('.chart-ranges button[data-hours="72"]');
    await page.screenshot({path: path.join(output, 'desktop-forecast.png'), fullPage: true});

    await page.fill('#pm25', '65');
    const manualResponse = page.waitForResponse(response => response.url().endsWith('/api/forecast') && response.request().method() === 'POST');
    await page.click('#forecast-button');
    const manual = await (await manualResponse).json();
    assert.equal(manual.latest, 65);
    assert.equal(manual.selected_model, 'xgboost');
    assert.equal(new Date(manual.target_time) - new Date(manual.origin_time), 3600000);
    await page.locator('#forecast-button:not([disabled])').waitFor();
    await page.selectOption('#model', 'ridge');
    const ridgeResponse = page.waitForResponse(response => response.url().endsWith('/api/forecast'));
    await page.click('#forecast-button');
    assert.equal((await (await ridgeResponse).json()).selected_model, 'ridge');
    await page.locator('#forecast-button:not([disabled])').waitFor();
    await page.fill('#pm25', '-1');
    await page.click('#forecast-button');
    assert.equal(await page.locator('#pm25').getAttribute('aria-invalid'), 'true');
    assert(await page.locator('#global-error').isVisible());
    await page.fill('#pm25', '65');

    await page.click('.nav-item[data-view="performance"]');
    assert.equal(await page.locator('#metrics-body tr').count(), 4);
    assert((await page.locator('#metrics-body').innerText()).includes('21.088'));
    assert((await page.locator('#metrics-body').innerText()).includes('21.063'));
    await page.screenshot({path: path.join(output, 'desktop-performance.png'), fullPage: true});
    await page.click('.nav-item[data-view="about"]');
    assert(await page.locator('#about-title').isVisible());
    await page.click('.nav-item[data-view="forecast"]');
    await page.click('#tab-history');
    const uploadPrediction = page.waitForResponse(response => response.url().endsWith('/api/forecast'));
    await page.setInputFiles('#csv-file', path.join(root, 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'));
    const uploaded = await (await uploadPrediction).json();
    assert.equal(uploaded.mode, 'history');
    assert.equal(uploaded.target_time, '2015-01-01T00:00:00+08:00');
    await page.locator('#download-forecast:not([disabled])').waitFor();
    const downloadEvent = page.waitForEvent('download');
    await page.click('#download-forecast');
    const download = await downloadEvent;
    await download.saveAs(path.join(output, 'forecast-export.csv'));
    assert(fs.readFileSync(path.join(output, 'forecast-export.csv'), 'utf8').includes('pm25_ug_m3'));
    await page.screenshot({path: path.join(output, 'desktop-upload.png'), fullPage: true});

    await page.click('#reset-sample');
    await page.locator('#forecast-button:not([disabled])').waitFor();
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(output, 'mobile-forecast.png'), fullPage: true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.click('#menu-button');
    await page.click('.nav-item[data-view="performance"]');
    assert(await page.locator('#performance-title').isVisible());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(output, 'mobile-performance.png'), fullPage: true});
    const report = await page.request.get('http://127.0.0.1:8016/report');
    assert.equal(report.status(), 200);
    assert(report.headers()['content-type'].includes('application/pdf'));
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser_checks.json'), JSON.stringify({status: 'passed', checks: ['real model forecast', 'chart ranges and tooltips', 'manual changes', 'model selection', 'invalid input', 'measured metrics', 'CSV upload', 'CSV export', 'mobile layout', 'mobile navigation', 'report link', 'no JavaScript errors'], screenshots: ['desktop-forecast.png', 'desktop-performance.png', 'desktop-upload.png', 'mobile-forecast.png', 'mobile-performance.png']}, null, 2) + '\n');
    console.log('Browser checks passed. Screenshots saved to ' + output);
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }
}

main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
