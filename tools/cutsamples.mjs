// Cuts out real piece photos exactly as the phone page does: runs web/cutout.js
// and web/photo.js in a headless browser (see cdp.mjs) on every image in a
// folder. Writes <outDir>/<name>.png (the upload the server would receive) and
// <outDir>/outline/<name>.jpg (the outline preview), and prints one line per
// photo. No Argus server needed: web/ is served by this script.
//
// usage: node tools/cutsamples.mjs [samplesDir] [outDir] [sensitivity]
//   defaults: samples  samples/cut  1
import { createServer } from 'node:http';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPage, pageProblems, sleep } from './cdp.mjs';

const [samplesDir = 'samples', outDir = join(samplesDir, 'cut'), sens = '1'] = process.argv.slice(2);
const webDir = fileURLToPath(new URL('../web/', import.meta.url));

// The harness exposes cut(base64, sensitivity) using the page's own modules
const HARNESS = `<!doctype html><meta charset="utf-8"><script type="module">
import { findPiece } from '/cutout.js';
import { cutoutPng, decodePhoto, drawOutline, workCopy } from '/photo.js';
const toBase64 = (blob) => new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result.slice(r.result.indexOf(',') + 1));
    r.readAsDataURL(blob);
});
window.cut = async (b64, sensitivity) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const photo = await decodePhoto(new Blob([bytes]));
    const work = workCopy(photo);
    const t0 = performance.now();
    const found = findPiece(work.rgba, work.w, work.h, sensitivity);
    const ms = Math.round(performance.now() - t0);
    const preview = document.createElement('canvas');
    drawOutline(preview, work, found);
    const outline = preview.toDataURL('image/jpeg', 0.85);
    return {
        ms,
        size: [photo.width, photo.height],
        coverage: found ? found.coverage : 0,
        touchesBorder: found ? found.touchesBorder : false,
        outline: outline.slice(outline.indexOf(',') + 1),
        cutout: found ? await toBase64(await cutoutPng(photo, work, found)) : null,
    };
};
window.ready = true;
</script>`;

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer((req, res) => {
    let status = 200, body = HARNESS, type = 'text/html';
    if (req.url !== '/') {
        try {
            body = readFileSync(join(webDir, basename(req.url)));
            type = TYPES[extname(req.url)] || 'application/octet-stream';
        } catch {
            status = 404; body = 'not found';
        }
    }
    res.writeHead(status, { 'Content-Type': type });
    res.end(body);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));

let exitCode = 0;
let tab;
try {
    const photos = readdirSync(samplesDir).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();
    mkdirSync(join(outDir, 'outline'), { recursive: true });

    tab = await openPage();
    await tab.page('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
    for (let i = 0; i < 50 && !(await tab.evaluate('window.ready === true')); i++) await sleep(100);

    for (const file of photos) {
        const name = basename(file, extname(file));
        const b64 = readFileSync(join(samplesDir, file)).toString('base64');
        const r = await tab.evaluate(`cut(${JSON.stringify(b64)}, ${Number(sens)})`);
        writeFileSync(join(outDir, 'outline', `${name}.jpg`), Buffer.from(r.outline, 'base64'));
        if (r.cutout) writeFileSync(join(outDir, `${name}.png`), Buffer.from(r.cutout, 'base64'));
        console.log(`${name.padEnd(8)} ${String(r.ms).padStart(4)} ms  coverage ${(r.coverage * 100).toFixed(1).padStart(5)}%` +
                    `${r.touchesBorder ? '  touches border' : ''}${r.cutout ? '' : '  NO PIECE'}`);
    }
    const problems = pageProblems(tab.events);
    for (const line of problems) console.log(line);
    if (problems.length) exitCode = 1;
} catch (err) {
    console.log('FAILED:', err.message);
    exitCode = 1;
} finally {
    if (tab) tab.close();
    server.close();
    process.exit(exitCode);
}
