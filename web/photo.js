// Canvas side of the cut-out: decode the photo, make the small working copy
// that cutout.js analyses, draw the outline preview, and build the upload PNG.
// Shared by the page and tools/cutsamples.mjs so both cut pieces identically.

export const WORK_MAX_DIM   = 800;    // outline detection runs at this size
export const UPLOAD_MAX_DIM = 1024;   // longest side of the uploaded cut-out
const CROP_MARGIN           = 0.03;   // least border around the piece, relative to the photo
const EDIT_MARGIN           = 0.2;    // border around the piece for hand edits, relative to the piece

// Decodes an image file/blob upright (EXIF orientation applied)
export async function decodePhoto(blob) {
    let bitmap;
    try {
        bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch {
        bitmap = await createImageBitmap(blob);
    }
    return bitmap;
}

// Downscaled copy for detection: { canvas, rgba, w, h }
export function workCopy(photo) {
    const scale = Math.min(1, WORK_MAX_DIM / Math.max(photo.width, photo.height));
    const w = Math.round(photo.width * scale), h = Math.round(photo.height * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(photo, 0, 0, w, h);
    return { canvas, rgba: ctx.getImageData(0, 0, w, h).data, w, h };
}

// Draws the working copy onto `canvas` with the background dimmed and the outline traced
export function drawOutline(canvas, work, found) {
    const { canvas: src, w, h } = work;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(src, 0, 0);
    const mask = found && found.mask;
    if (mask) {
        const img = ctx.getImageData(0, 0, w, h);
        const px = img.data;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const p = y * w + x, i = p * 4;
                const edge = mask[p] && (x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
                    !mask[p - 1] || !mask[p + 1] || !mask[p - w] || !mask[p + w]);
                if (edge) {
                    px[i] = 255; px[i + 1] = 214; px[i + 2] = 10;
                } else if (!mask[p]) {
                    px[i] *= 0.35; px[i + 1] *= 0.35; px[i + 2] *= 0.35;
                }
            }
        }
        ctx.putImageData(img, 0, 0);
    }
}

// Editable outline over the photo region around the piece, at the scale its
// upload will have (UPLOAD_MAX_DIM px over the piece plus CROP_MARGIN): { x, y, w, h } is the region in photo pixels, { mw, mh }
// the mask size and `alpha` the mask (1 = piece). Starts from the detected
// outline, scaled up with smoothing and re-thresholded for a clean edge; with
// nothing detected it covers the whole photo, empty, ready to be painted.
export function editMask(photo, work, found) {
    const { w, h } = work;
    const k = photo.width / w;
    let x0 = 0, y0 = 0, x1 = w, y1 = h, uploadSide = Math.max(w, h);
    if (found) {
        const { bbox } = found;
        const cropMargin = Math.round(Math.max(w, h) * CROP_MARGIN);
        uploadSide = Math.min(w, bbox.x + bbox.w + cropMargin) - Math.max(0, bbox.x - cropMargin);
        uploadSide = Math.max(uploadSide, Math.min(h, bbox.y + bbox.h + cropMargin) - Math.max(0, bbox.y - cropMargin));
        const margin = Math.max(cropMargin, Math.round(Math.max(bbox.w, bbox.h) * EDIT_MARGIN));
        x0 = Math.max(0, bbox.x - margin); y0 = Math.max(0, bbox.y - margin);
        x1 = Math.min(w, bbox.x + bbox.w + margin); y1 = Math.min(h, bbox.y + bbox.h + margin);
    }
    const region = { x: x0 * k, y: y0 * k, w: (x1 - x0) * k, h: (y1 - y0) * k };
    const scale = Math.min(1, UPLOAD_MAX_DIM / (uploadSide * k));
    const mw = Math.max(1, Math.round(region.w * scale)), mh = Math.max(1, Math.round(region.h * scale));
    const alpha = new Uint8Array(mw * mh);

    if (found) {
        const { mask } = found;
        const small = document.createElement('canvas');
        small.width = x1 - x0; small.height = y1 - y0;
        const sctx = small.getContext('2d');
        const simg = sctx.createImageData(small.width, small.height);
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                simg.data[((y - y0) * small.width + (x - x0)) * 4 + 3] = mask[y * w + x] ? 255 : 0;
            }
        }
        sctx.putImageData(simg, 0, 0);
        const big = document.createElement('canvas');
        big.width = mw; big.height = mh;
        const bctx = big.getContext('2d', { willReadFrequently: true });
        bctx.imageSmoothingQuality = 'high';
        bctx.drawImage(small, 0, 0, mw, mh);
        const up = bctx.getImageData(0, 0, mw, mh).data;
        for (let p = 0; p < alpha.length; p++) alpha[p] = up[p * 4 + 3] >= 128 ? 1 : 0;
    }
    return { ...region, mw, mh, alpha };
}

// The photo region of an editMask() at mask resolution, as ImageData
export function editPixels(photo, edit) {
    const c = document.createElement('canvas');
    c.width = edit.mw; c.height = edit.mh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(photo, edit.x, edit.y, edit.w, edit.h, 0, 0, edit.mw, edit.mh);
    return ctx.getImageData(0, 0, edit.mw, edit.mh);
}

// The outline of an editMask() and a CROP_MARGIN border, cut from the
// full-resolution photo at up to UPLOAD_MAX_DIM px with alpha = outline, as a
// PNG Blob (null when the outline is empty)
export function cutoutPng(photo, edit) {
    const { mw, mh, alpha } = edit;
    let bx0 = mw, by0 = mh, bx1 = -1, by1 = -1;
    for (let y = 0; y < mh; y++) {
        for (let x = 0; x < mw; x++) {
            if (alpha[y * mw + x]) {
                if (x < bx0) bx0 = x;
                if (x > bx1) bx1 = x;
                if (y < by0) by0 = y;
                if (y > by1) by1 = y;
            }
        }
    }
    let result = Promise.resolve(null);
    if (bx1 >= 0) {
        const k = edit.w / mw;   // photo pixels per mask pixel
        const margin = Math.round(Math.max(photo.width, photo.height) * CROP_MARGIN / k);
        const x0 = Math.max(0, bx0 - margin), y0 = Math.max(0, by0 - margin);
        const x1 = Math.min(mw, bx1 + 1 + margin), y1 = Math.min(mh, by1 + 1 + margin);
        const srcW = (x1 - x0) * k, srcH = (y1 - y0) * k;
        const scale = Math.min(1, UPLOAD_MAX_DIM / Math.max(srcW, srcH));
        const outW = Math.max(1, Math.round(srcW * scale)), outH = Math.max(1, Math.round(srcH * scale));

        // Mask as alpha, scaled with smoothing and re-thresholded for a clean edge
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = x1 - x0; maskCanvas.height = y1 - y0;
        const mctx = maskCanvas.getContext('2d');
        const mimg = mctx.createImageData(maskCanvas.width, maskCanvas.height);
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                mimg.data[((y - y0) * maskCanvas.width + (x - x0)) * 4 + 3] = alpha[y * mw + x] ? 255 : 0;
            }
        }
        mctx.putImageData(mimg, 0, 0);

        const out = document.createElement('canvas');
        out.width = outW; out.height = outH;
        const octx = out.getContext('2d', { willReadFrequently: true });
        octx.imageSmoothingQuality = 'high';
        octx.drawImage(maskCanvas, 0, 0, outW, outH);
        const a = octx.getImageData(0, 0, outW, outH).data;
        octx.clearRect(0, 0, outW, outH);
        octx.drawImage(photo, edit.x + x0 * k, edit.y + y0 * k, srcW, srcH, 0, 0, outW, outH);
        const img = octx.getImageData(0, 0, outW, outH);
        for (let i = 3; i < img.data.length; i += 4) img.data[i] = a[i] >= 128 ? 255 : 0;
        octx.putImageData(img, 0, 0);
        result = new Promise((resolve) => out.toBlob(resolve, 'image/png'));
    }
    return result;
}
