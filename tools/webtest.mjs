// End-to-end test of the web page: drives a headless Chromium-family browser
// over the DevTools pipe (no npm packages) through photo → outline → locate,
// at phone size, saving a screenshot per step and printing the results table
// plus any JS errors. Exits non-zero on failure or a page error.
//
// usage: node tools/webtest.mjs <url> <photo> <outPrefix> [sensitivity]
//   e.g. node tools/webtest.mjs http://localhost:8080/ samples/C5.jpg /tmp/c5
// ARGUS_BROWSER selects the binary (default: brave-browser; chromium and
// google-chrome work too). A throwaway profile is used and removed afterwards.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const [url, photo, out, sens] = process.argv.slice(2);
const profile = mkdtempSync(join(tmpdir(), 'argus-webtest-'));
const browser = spawn(process.env.ARGUS_BROWSER || 'brave-browser', [
    '--headless=new', '--remote-debugging-pipe', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-gpu', 'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });

const toBrowser = browser.stdio[3], fromBrowser = browser.stdio[4];
let nextId = 1, buf = '';
const waiting = new Map();
// If the browser can't start or dies, fail pending commands instead of crashing on EPIPE
const browserGone = (why) => {
    for (const done of waiting.values()) done({ error: { message: `browser ${why}` } });
    waiting.clear();
};
browser.on('error', (err) => browserGone(`failed to start: ${err.message}`));
browser.on('exit', (code) => browserGone(`exited (code ${code})`));
toBrowser.on('error', () => browserGone('closed the DevTools pipe'));
const events = [];
fromBrowser.on('data', (chunk) => {
    buf += chunk.toString();
    let i;
    while ((i = buf.indexOf('\0')) >= 0) {
        const msg = JSON.parse(buf.slice(0, i));
        buf = buf.slice(i + 1);
        if (msg.id && waiting.has(msg.id)) { waiting.get(msg.id)(msg); waiting.delete(msg.id); }
        else if (msg.method) events.push(msg);
    }
});
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
    const id = nextId++;
    waiting.set(id, (m) => (m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result)));
    toBrowser.write(JSON.stringify({ id, method, params, sessionId }) + '\0');
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let exitCode = 0;

try {
    const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
    const { sessionId: s } = await send('Target.attachToTarget', { targetId, flatten: true });
    const page = (m, p) => send(m, p, s);
    const evaluate = async (expr) => (await page('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result.value;
    const shot = async (name) => {
        const { data } = await page('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
        writeFileSync(`${out}_${name}.png`, Buffer.from(data, 'base64'));
    };

    await page('Runtime.enable');
    await page('Page.enable');
    await page('Emulation.setDeviceMetricsOverride', { width: 412, height: 915, deviceScaleFactor: 1, mobile: true });
    await page('Page.navigate', { url });
    for (let i = 0; i < 50 && !(await evaluate('document.querySelectorAll("#matchers input").length')); i++) await sleep(200);
    await shot('1_start');

    const { root } = await page('DOM.getDocument');
    const { nodeId } = await page('DOM.querySelector', { nodeId: root.nodeId, selector: '#gallery' });
    await page('DOM.setFileInputFiles', { nodeId, files: [resolve(photo)] });
    for (let i = 0; i < 50 && (await evaluate('document.getElementById("locate").disabled')); i++) await sleep(200);
    if (sens) {
        await evaluate(`{const el=document.getElementById('sensitivity'); el.value=${sens}; el.dispatchEvent(new Event('input'));}`);
        await sleep(1000);
    }
    console.log('outline:', await evaluate('document.getElementById("outline-status").textContent'));
    await shot('2_preview');

    const t0 = Date.now();
    await evaluate('document.getElementById("locate").click()');
    for (let i = 0; i < 150 && (await evaluate('document.getElementById("results").hidden')); i++) await sleep(200);
    for (let i = 0; i < 20 && !(await evaluate('document.getElementById("overview-img").complete')); i++) await sleep(200);
    await sleep(800);   // smooth scroll + thumbnail
    console.log(`results after ${Date.now() - t0} ms:`);
    console.log(await evaluate('[...document.querySelectorAll("#result-rows tr")].map(r => [...r.cells].map(c => c.textContent.trim()).join(" | ")).join("\\n")'));
    console.log('page width overflow:', await evaluate('document.documentElement.scrollWidth - document.documentElement.clientWidth'));
    await shot('3_results');

    for (const e of events) {
        if (e.method === 'Runtime.exceptionThrown') {
            exitCode = 1;
            console.log('EXCEPTION:', e.params.exceptionDetails.exception?.description || e.params.exceptionDetails.text);
        }
        if (e.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(e.params.type))
            console.log(`console.${e.params.type}:`, e.params.args.map((a) => a.value ?? a.description).join(' '));
    }
} catch (err) {
    console.log('FAILED:', err.message);
    exitCode = 1;
} finally {
    browser.kill();
    rmSync(profile, { recursive: true, force: true });
    process.exit(exitCode);
}
