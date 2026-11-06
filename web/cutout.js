// Finds a puzzle piece photographed on a plain, contrasting surface.
//
// Pure functions on RGBA pixel arrays (no DOM), so the same code can cut out
// sample photos offline. Steps:
//   1. Model the background from a frame of border pixels, as a plane per
//      channel (a + b·x + c·y) so a lighting gradient across the table is fine.
//   2. Score every pixel by its colour distance to that model; luminance
//      counts half, so soft shadows around the piece weigh less than colour.
//   3. Threshold with Otsu (scaled by `sensitivity`), then close and open the
//      mask to bridge gaps and drop specks.
//   4. Keep the largest blob that does not touch the image border, and fill
//      its holes.

const BORDER_FRACTION = 0.04;   // width of the frame sampled as background
const MORPH_FRACTION  = 0.008;  // morphology radius, relative to the longest side
const LUMA_WEIGHT     = 0.5;    // weight of brightness vs. colour difference

// Least-squares plane v = a + b·x + c·y through (x, y, v) samples
function fitPlane(xs, ys, vs) {
    const n = xs.length;
    let sx = 0, sy = 0, sv = 0, sxx = 0, syy = 0, sxy = 0, sxv = 0, syv = 0;
    for (let i = 0; i < n; i++) {
        const x = xs[i], y = ys[i], v = vs[i];
        sx += x; sy += y; sv += v;
        sxx += x * x; syy += y * y; sxy += x * y;
        sxv += x * v; syv += y * v;
    }
    // Solve the 3×3 normal equations by Cramer's rule
    const m = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]];
    const r = [sv, sxv, syv];
    const det3 = (a) =>
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    const d = det3(m);
    let plane = [sv / n, 0, 0];   // flat fallback for a degenerate fit
    if (Math.abs(d) > 1e-9) {
        plane = [0, 1, 2].map((k) => det3(m.map((row, i) => row.map((v, j) => (j === k ? r[i] : v)))) / d);
    }
    return plane;
}

// Colour distance of each pixel to the background model, as a Float32Array
function backgroundDistance(rgba, w, h) {
    const border = Math.max(2, Math.round(Math.max(w, h) * BORDER_FRACTION));
    const xs = [], ys = [], ch = [[], [], []];
    for (let y = 0; y < h; y += 2) {
        for (let x = 0; x < w; x += 2) {
            if (x < border || y < border || x >= w - border || y >= h - border) {
                const i = (y * w + x) * 4;
                xs.push(x / w); ys.push(y / h);
                ch[0].push(rgba[i]); ch[1].push(rgba[i + 1]); ch[2].push(rgba[i + 2]);
            }
        }
    }
    const planes = ch.map((vs) => fitPlane(xs, ys, vs));

    const dist = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const p = y * w + x, i = p * 4;
            const d = [0, 1, 2].map((c) => rgba[i + c] - (planes[c][0] + planes[c][1] * x / w + planes[c][2] * y / h));
            const luma = (d[0] + d[1] + d[2]) / 3;
            const cr = d[0] - luma, cg = d[1] - luma, cb = d[2] - luma;
            dist[p] = Math.sqrt(LUMA_WEIGHT * 3 * luma * luma + cr * cr + cg * cg + cb * cb);
        }
    }
    return dist;
}

// Otsu threshold over values in [0, max]
function otsu(values) {
    const BINS = 256;
    let max = 0;
    for (const v of values) max = Math.max(max, v);
    const hist = new Float64Array(BINS);
    const scale = max > 0 ? (BINS - 1) / max : 0;
    for (const v of values) hist[Math.round(v * scale)]++;

    const total = values.length;
    let sumAll = 0;
    for (let i = 0; i < BINS; i++) sumAll += i * hist[i];
    let wB = 0, sumB = 0, best = 0, bestVar = -1;
    for (let i = 0; i < BINS; i++) {
        wB += hist[i];
        const wF = total - wB;
        if (wB > 0 && wF > 0) {
            sumB += i * hist[i];
            const mB = sumB / wB, mF = (sumAll - sumB) / wF;
            const between = wB * wF * (mB - mF) * (mB - mF);
            if (between > bestVar) { bestVar = between; best = i; }
        } else {
            sumB += i * hist[i];
        }
    }
    return scale > 0 ? best / scale : 0;
}

// Square dilation (grow = true) or erosion of a 0/1 mask, radius r, via a summed-area table
function morph(mask, w, h, r, grow) {
    const W = w + 1;
    const sat = new Int32Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
        let rowSum = 0;
        for (let x = 0; x < w; x++) {
            rowSum += mask[y * w + x];
            sat[(y + 1) * W + x + 1] = sat[y * W + x + 1] + rowSum;
        }
    }
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
        for (let x = 0; x < w; x++) {
            const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
            const sum = sat[y1 * W + x1] - sat[y0 * W + x1] - sat[y1 * W + x0] + sat[y0 * W + x0];
            const area = (y1 - y0) * (x1 - x0);
            out[y * w + x] = grow ? (sum > 0 ? 1 : 0) : (sum === area ? 1 : 0);
        }
    }
    return out;
}

// Labels 4-connected blobs of a 0/1 mask; returns { labels, blobs: [{ size, touchesBorder }] }
function components(mask, w, h) {
    const labels = new Int32Array(w * h).fill(-1);
    const blobs = [];
    const stack = new Int32Array(w * h);
    for (let start = 0; start < w * h; start++) {
        if (mask[start] && labels[start] < 0) {
            const id = blobs.length;
            let top = 0, size = 0, touchesBorder = false;
            stack[top++] = start;
            labels[start] = id;
            while (top > 0) {
                const p = stack[--top];
                const x = p % w, y = (p - x) / w;
                size++;
                if (x === 0 || y === 0 || x === w - 1 || y === h - 1) touchesBorder = true;
                const nbrs = [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1];
                for (const q of nbrs) {
                    if (q >= 0 && mask[q] && labels[q] < 0) { labels[q] = id; stack[top++] = q; }
                }
            }
            blobs.push({ size, touchesBorder });
        }
    }
    return { labels, blobs };
}

// Marks everything the border can't reach without crossing the blob (blob + holes)
function fillHoles(blob, w, h) {
    const outside = new Uint8Array(w * h);
    const stack = [];
    const seed = (p) => { if (!blob[p] && !outside[p]) { outside[p] = 1; stack.push(p); } };
    for (let x = 0; x < w; x++) { seed(x); seed((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { seed(y * w); seed(y * w + w - 1); }
    while (stack.length) {
        const p = stack.pop();
        const x = p % w;
        if (x > 0) seed(p - 1);
        if (x < w - 1) seed(p + 1);
        if (p >= w) seed(p - w);
        if (p < w * (h - 1)) seed(p + w);
    }
    const filled = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) filled[p] = outside[p] ? 0 : 1;
    return filled;
}

// Finds the piece in an RGBA image (w × h). `sensitivity` scales the threshold:
// above 1 is stricter (tighter outline), below 1 takes in more.
// Returns { mask: Uint8Array 0/1, bbox: {x, y, w, h}, coverage, touchesBorder },
// or null when nothing stands out from the background.
export function findPiece(rgba, w, h, sensitivity = 1) {
    const dist = backgroundDistance(rgba, w, h);
    const threshold = otsu(dist) * sensitivity;

    let mask = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) mask[p] = dist[p] > threshold ? 1 : 0;

    const r = Math.max(1, Math.round(Math.max(w, h) * MORPH_FRACTION));
    mask = morph(morph(mask, w, h, r, true), w, h, r, false);   // close
    mask = morph(morph(mask, w, h, r, false), w, h, r, true);   // open

    const { labels, blobs } = components(mask, w, h);
    // Prefer the largest blob clear of the border; fall back to the largest overall
    let best = -1;
    for (const pass of [true, false]) {
        for (let i = 0; i < blobs.length; i++) {
            if ((!pass || !blobs[i].touchesBorder) && (best < 0 || blobs[i].size > blobs[best].size)) best = i;
        }
        if (best >= 0) break;
    }

    let result = null;
    if (best >= 0) {
        const blob = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++) blob[p] = labels[p] === best ? 1 : 0;
        const filled = fillHoles(blob, w, h);

        let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (filled[y * w + x]) {
                    area++;
                    if (x < x0) x0 = x;
                    if (x > x1) x1 = x;
                    if (y < y0) y0 = y;
                    if (y > y1) y1 = y;
                }
            }
        }
        result = {
            mask: filled,
            bbox: { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 },
            coverage: area / (w * h),
            touchesBorder: blobs[best].touchesBorder,
        };
    }
    return result;
}
