// Minimal DevTools-protocol driver for a headless Chromium-family browser over
// --remote-debugging-pipe (no npm packages; Node 20 has no WebSocket client).
// ARGUS_BROWSER selects the binary (default: brave-browser; chromium and
// google-chrome work too). Not BROWSER: desktops often set that to something else.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Starts the browser with a throwaway profile and opens one blank tab.
// Returns { page(method, params), evaluate(expr), screenshot(path), events, close() }.
export async function openPage() {
    const profile = mkdtempSync(join(tmpdir(), 'argus-browser-'));
    const browser = spawn(process.env.ARGUS_BROWSER || 'brave-browser', [
        '--headless=new', '--remote-debugging-pipe', `--user-data-dir=${profile}`,
        '--no-first-run', '--no-default-browser-check', '--disable-gpu', 'about:blank',
    ], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });

    const toBrowser = browser.stdio[3], fromBrowser = browser.stdio[4];
    let nextId = 1, buf = '';
    const waiting = new Map();
    const events = [];
    // If the browser can't start or dies, fail pending commands instead of crashing on EPIPE
    const browserGone = (why) => {
        for (const done of waiting.values()) done({ error: { message: `browser ${why}` } });
        waiting.clear();
    };
    browser.on('error', (err) => browserGone(`failed to start: ${err.message}`));
    browser.on('exit', (code) => browserGone(`exited (code ${code})`));
    toBrowser.on('error', () => browserGone('closed the DevTools pipe'));
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
    const close = () => {
        browser.kill();
        rmSync(profile, { recursive: true, force: true });
    };

    let tab;
    try {
        const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
        const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
        const page = (m, p) => send(m, p, sessionId);
        await page('Runtime.enable');
        await page('Page.enable');
        tab = {
            page,
            events,
            close,
            evaluate: async (expr) => {
                const r = await page('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
                if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
                return r.result.value;
            },
            screenshot: async (path) => {
                const { data } = await page('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
                writeFileSync(path, Buffer.from(data, 'base64'));
            },
        };
    } catch (err) {
        close();
        throw err;
    }
    return tab;
}

// Exceptions and console errors/warnings the page produced, as text lines
export function pageProblems(events) {
    const lines = [];
    for (const e of events) {
        if (e.method === 'Runtime.exceptionThrown')
            lines.push(`EXCEPTION: ${e.params.exceptionDetails.exception?.description || e.params.exceptionDetails.text}`);
        if (e.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(e.params.type))
            lines.push(`console.${e.params.type}: ${e.params.args.map((a) => a.value ?? a.description).join(' ')}`);
    }
    return lines;
}
