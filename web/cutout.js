// Finds a puzzle piece photographed on a plain surface, including low-contrast
// ones (sepia pieces on brown cardboard or wood grain).
//
// Pure functions on RGBA pixel arrays (no DOM), so the same code can cut out
// sample photos offline. Steps:
//   1. Model the background from a frame of border pixels: a plane per channel
//      (a + b·x + c·y, for lighting gradients) plus the covariance of what the
//      planes leave over (the table's own texture), refitted on the best 80% so
//      a stray object at the edge doesn't skew it.
//   2. Score every pixel by its Mahalanobis distance to that model, i.e. in
//      units of the table's own variation, and smooth the scores slightly.
//   3. Threshold with Otsu (scaled by `sensitivity`), close and open the mask,
//      and take the largest blob clear of the image border as the seed.
//   4. Grow the seed through pixels passing a weaker threshold (dark parts of
//      the print close to the table's colour), ignoring thin strands and
//      dropping growth that leaks out to the photo edge.
//   5. Close and open again, keep the main blob, and fill its holes.

const BORDER_FRACTION = 0.04;   // width of the frame sampled as background
const MORPH_FRACTION  = 0.008;  // morphology radius, relative to the longest side
const NOISE_FLOOR     = 2;      // minimum per-channel spread assumed for the table (camera noise)
const INLIER_FRACTION = 0.8;    // share of border samples kept when refitting the background
const SMOOTH_FRACTION = 0.004;  // radius of the distance smoothing, relative to the longest side
const MIN_DISTANCE    = 3;      // never threshold below this many standard deviations
const DIST_CAP        = 40;     // distances above this don't move the Otsu threshold
const WEAK_RATIO      = 0.4;    // weaker threshold (× the main one) for growing the piece
const LEAK_EDGE_FRACTION = 0.1; // growth touching the photo edge for longer than this is table
const LEAK_REACH_FRACTION = 0.05; // ...as is edge-touching growth reaching this far past the seed

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

// Inverse of a symmetric 3×3 matrix (adjugate / determinant)
function invert3(m) {
    const [a, b, c] = m[0], [, e, f] = m[1], [, , i] = m[2];
    const A = e * i - f * f, B = c * f - b * i, C = b * f - c * e;
    const det = a * A + b * B + c * C;
    return [[A / det, B / det, C / det],
            [B / det, (a * i - c * c) / det, (b * c - a * f) / det],
            [C / det, (b * c - a * f) / det, (a * e - b * b) / det]];
}

// Background model from a frame of border pixels: a plane per channel (lighting
// gradient) plus the covariance of what the planes don't explain (the table's
// own texture). Refitted on the best-fitting samples so a stray object at the
// edge doesn't skew it. Returns each pixel's Mahalanobis distance to the model.
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

    let keep = xs.map((_, k) => k);
    let planes, inv;
    for (let pass = 0; pass < 3; pass++) {
        planes = ch.map((vs) => fitPlane(keep.map((k) => xs[k]), keep.map((k) => ys[k]), keep.map((k) => vs[k])));
        const resid = (k) => [0, 1, 2].map((c) => ch[c][k] - (planes[c][0] + planes[c][1] * xs[k] + planes[c][2] * ys[k]));
        const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (const k of keep) {
            const r = resid(k);
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a][b] += r[a] * r[b] / keep.length;
        }
        for (let a = 0; a < 3; a++) cov[a][a] += NOISE_FLOOR * NOISE_FLOOR;
        inv = invert3(cov);
        // Keep the best-fitting samples of the whole frame for the next pass
        const d2 = xs.map((_, k) => {
            const r = resid(k);
            let q = 0;
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) q += r[a] * inv[a][b] * r[b];
            return q;
        });
        const cut = [...d2].sort((p, q) => p - q)[Math.floor(d2.length * INLIER_FRACTION)];
        keep = xs.map((_, k) => k).filter((k) => d2[k] <= cut);
    }

    const dist = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const p = y * w + x, i = p * 4;
            const r = [0, 1, 2].map((c) => rgba[i + c] - (planes[c][0] + planes[c][1] * x / w + planes[c][2] * y / h));
            let q = 0;
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) q += r[a] * inv[a][b] * r[b];
            dist[p] = q;   // squared for now; averaged before the square root
        }
    }
    return dist;
}

// Mean over a (2r+1)² window, via a summed-area table
function boxMean(values, w, h, r) {
    const W = w + 1;
    const sat = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
        let rowSum = 0;
        for (let x = 0; x < w; x++) {
            rowSum += values[y * w + x];
            sat[(y + 1) * W + x + 1] = sat[y * W + x + 1] + rowSum;
        }
    }
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
        for (let x = 0; x < w; x++) {
            const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
            out[y * w + x] = (sat[y1 * W + x1] - sat[y0 * W + x1] - sat[y1 * W + x0] + sat[y0 * W + x0]) /
                             ((y1 - y0) * (x1 - x0));
        }
    }
    return out;
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

// Labels 4-connected blobs of a 0/1 mask;
// returns { labels, blobs: [{ size, touchesBorder, borderPixels, box: {x0, y0, x1, y1} }] }
function components(mask, w, h) {
    const labels = new Int32Array(w * h).fill(-1);
    const blobs = [];
    const stack = new Int32Array(w * h);
    for (let start = 0; start < w * h; start++) {
        if (mask[start] && labels[start] < 0) {
            const id = blobs.length;
            let top = 0, size = 0, borderPixels = 0;
            const box = { x0: w, y0: h, x1: -1, y1: -1 };
            stack[top++] = start;
            labels[start] = id;
            while (top > 0) {
                const p = stack[--top];
                const x = p % w, y = (p - x) / w;
                size++;
                if (x === 0 || y === 0 || x === w - 1 || y === h - 1) borderPixels++;
                box.x0 = Math.min(box.x0, x); box.x1 = Math.max(box.x1, x);
                box.y0 = Math.min(box.y0, y); box.y1 = Math.max(box.y1, y);
                const nbrs = [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1];
                for (const q of nbrs) {
                    if (q >= 0 && mask[q] && labels[q] < 0) { labels[q] = id; stack[top++] = q; }
                }
            }
            blobs.push({ size, touchesBorder: borderPixels > 0, borderPixels, box });
        }
    }
    return { labels, blobs };
}

// Grows a 0/1 seed through the 0/1 `allowed` pixels it touches (4-connected)
function grow(seed, allowed, w, h) {
    const out = Uint8Array.from(seed);
    const stack = [];
    for (let p = 0; p < w * h; p++) if (seed[p]) stack.push(p);
    while (stack.length) {
        const p = stack.pop();
        const x = p % w;
        const nbrs = [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, p >= w ? p - w : -1, p < w * (h - 1) ? p + w : -1];
        for (const q of nbrs) {
            if (q >= 0 && allowed[q] && !out[q]) { out[q] = 1; stack.push(q); }
        }
    }
    return out;
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
    const smooth = Math.max(1, Math.round(Math.max(w, h) * SMOOTH_FRACTION));
    const dist = boxMean(backgroundDistance(rgba, w, h), w, h, smooth).map((q) => Math.min(DIST_CAP, Math.sqrt(q)));
    const threshold = Math.max(MIN_DISTANCE, otsu(dist)) * sensitivity;

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
        // Grow the piece into connected pixels that pass a weaker threshold: parts
        // of the print close to the table's colour, which the main threshold drops
        let blob = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++) blob[p] = labels[p] === best ? 1 : 0;
        const weakThreshold = Math.max(MIN_DISTANCE, threshold * WEAK_RATIO);
        const weak = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++) weak[p] = dist[p] > weakThreshold ? 1 : 0;
        const grown = grow(blob, weak, w, h);
        // Growth that reaches the photo edge and either runs along it or spreads
        // well past the seed has leaked into the table: drop it. (A piece filling
        // the frame may graze the edge with a knob; that stays.)
        const leakEdge = Math.round(Math.max(w, h) * LEAK_EDGE_FRACTION);
        const reach = Math.round(Math.max(w, h) * LEAK_REACH_FRACTION);
        const seedBox = blobs[best].box;
        const isLeak = (b) => b.borderPixels > leakEdge || (b.touchesBorder &&
            (b.box.x0 < seedBox.x0 - reach || b.box.x1 > seedBox.x1 + reach ||
             b.box.y0 < seedBox.y0 - reach || b.box.y1 > seedBox.y1 + reach));
        // Thin strands (wood grain, scratches) are never piece parts: open them away first
        let extra = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++) extra[p] = grown[p] && !blob[p] ? 1 : 0;
        extra = morph(morph(extra, w, h, r, false), w, h, r, true);
        const leaks = components(extra, w, h);
        for (let p = 0; p < w * h; p++) {
            if (extra[p] && !isLeak(leaks.blobs[leaks.labels[p]])) blob[p] = 1;
        }
        blob = morph(morph(blob, w, h, r, true), w, h, r, false);   // close
        blob = morph(morph(blob, w, h, r, false), w, h, r, true);   // open
        // Keep only the main blob: specks the growth picked up may have come loose
        const parts = components(blob, w, h);
        let main = 0;
        for (let i = 1; i < parts.blobs.length; i++) if (parts.blobs[i].size > parts.blobs[main].size) main = i;
        for (let p = 0; p < w * h; p++) blob[p] = parts.labels[p] === main ? 1 : 0;
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
