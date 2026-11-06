// End-to-end test of the web page: drives a headless browser (see cdp.mjs)
// through photo → outline → locate at phone size, saving a screenshot per step
// and printing the results table plus any JS errors. Needs a running server.
// Exits non-zero on failure or if the page logged an error or warning.
//
// usage: node tools/webtest.mjs <url> <photo> <outPrefix> [sensitivity]
//   e.g. node tools/webtest.mjs http://localhost:8080/ samples/C5.jpg /tmp/c5
import { resolve } from 'node:path';
import { openPage, pageProblems, sleep } from './cdp.mjs';

const [url, photo, out, sens] = process.argv.slice(2);
let exitCode = 0;
let tab;

try {
    tab = await openPage();
    const { page, evaluate } = tab;
    const shot = (name) => tab.screenshot(`${out}_${name}.png`);

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

    const problems = pageProblems(tab.events);
    for (const line of problems) console.log(line);
    if (problems.length) exitCode = 1;
} catch (err) {
    console.log('FAILED:', err.message);
    exitCode = 1;
} finally {
    if (tab) tab.close();
    process.exit(exitCode);
}
