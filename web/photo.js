// Canvas side of the cut-out: decode the photo, make the small working copy
// that cutout.js analyses, draw the outline preview, and build the upload PNG.
// Shared by the page and tools/cutsamples.mjs so both cut pieces identically.

export const WORK_MAX_DIM   = 800;    // outline detection runs at this size
export const UPLOAD_MAX_DIM = 1024;   // longest side of the uploaded cut-out
const CROP_MARGIN           = 0.03;   // extra border around the piece, relative to the photo

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

// Full-resolution crop of the piece with alpha = outline, as a PNG Blob
export function cutoutPng(photo, work, found) {
    const { bbox, mask } = found;
    const { w, h } = work;
    const k = photo.width / w;
    const margin = Math.round(Math.max(w, h) * CROP_MARGIN);
    const x0 = Math.max(0, bbox.x - margin), y0 = Math.max(0, bbox.y - margin);
    const x1 = Math.min(w, bbox.x + bbox.w + margin), y1 = Math.min(h, bbox.y + bbox.h + margin);

    const srcW = (x1 - x0) * k, srcH = (y1 - y0) * k;
    const scale = Math.min(1, UPLOAD_MAX_DIM / Math.max(srcW, srcH));
    const outW = Math.max(1, Math.round(srcW * scale)), outH = Math.max(1, Math.round(srcH * scale));

    // Low-res mask as alpha, scaled up with smoothing and re-thresholded for a clean edge
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = x1 - x0; maskCanvas.height = y1 - y0;
    const mctx = maskCanvas.getContext('2d');
    const mimg = mctx.createImageData(maskCanvas.width, maskCanvas.height);
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            mimg.data[((y - y0) * maskCanvas.width + (x - x0)) * 4 + 3] = mask[y * w + x] ? 255 : 0;
        }
    }
    mctx.putImageData(mimg, 0, 0);

    const out = document.createElement('canvas');
    out.width = outW; out.height = outH;
    const octx = out.getContext('2d', { willReadFrequently: true });
    octx.imageSmoothingQuality = 'high';
    octx.drawImage(maskCanvas, 0, 0, outW, outH);
    const alpha = octx.getImageData(0, 0, outW, outH).data;
    octx.clearRect(0, 0, outW, outH);
    octx.drawImage(photo, x0 * k, y0 * k, srcW, srcH, 0, 0, outW, outH);
    const img = octx.getImageData(0, 0, outW, outH);
    for (let i = 3; i < img.data.length; i += 4) img.data[i] = alpha[i] >= 128 ? 255 : 0;
    octx.putImageData(img, 0, 0);

    return new Promise((resolve) => out.toBlob(resolve, 'image/png'));
}
