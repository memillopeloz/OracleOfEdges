// Hand refinement of the piece outline: draws an editMask() region of the photo
// with the background dimmed and the outline traced, and paints into its mask
// with a round brush (add or erase) under one finger or the mouse.
import { editPixels } from './photo.js';

const DIM        = 0.35;                 // brightness outside the outline
const EDGE_RGB   = [255, 214, 10];       // outline colour
const UNDO_DEPTH = 12;   // a snapshot is ~1.5–2 MB

export class OutlineEditor {
    // In: canvas to draw into; onChange() is called after every finished stroke
    constructor(canvas, onChange) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onChange = onChange;
        this.edit = null;          // editMask() being refined
        this.base = null;          // photo pixels of the region (ImageData)
        this.view = null;          // what is shown (ImageData)
        this.tool = null;          // 'add' | 'erase' | null (viewing only)
        this.radius = 0.02;        // brush radius, relative to the region's longest side
        this.undo = [];
        this.stroke = null;        // { id, last: [x, y] } while painting
        this.dirty = null;         // [x0, y0, x1, y1] waiting to be redrawn
        this.cursor = null;        // [x, y] brush outline position while painting

        canvas.addEventListener('pointerdown', (e) => this.down(e));
        canvas.addEventListener('pointermove', (e) => this.move(e));
        for (const type of ['pointerup', 'pointercancel']) canvas.addEventListener(type, (e) => this.up(e));
    }

    // Shows a new region, forgetting earlier edits
    load(photo, edit) {
        this.edit = edit;
        this.base = editPixels(photo, edit);
        this.view = new ImageData(new Uint8ClampedArray(this.base.data), edit.mw, edit.mh);
        this.canvas.width = edit.mw;
        this.canvas.height = edit.mh;
        this.undo = [];
        this.stroke = null;
        this.cursor = null;
        this.redraw([0, 0, edit.mw, edit.mh]);
    }

    setTool(tool) {
        this.tool = tool;
        this.canvas.classList.toggle('painting', Boolean(tool));
    }

    canUndo() { return this.undo.length > 0; }

    undoLast() {
        if (this.undo.length) {
            this.edit.alpha.set(this.undo.pop());
            this.redraw([0, 0, this.edit.mw, this.edit.mh]);
            this.onChange();
        }
    }

    // Pixels inside the outline
    area() {
        let n = 0;
        if (this.edit) for (const a of this.edit.alpha) n += a;
        return n;
    }

    // ── Painting ────────────────────────────────────────────────────────────

    // Pointer position in mask pixels (the canvas is letterboxed by object-fit: contain)
    toMask(e) {
        const r = this.canvas.getBoundingClientRect();
        const s = Math.min(r.width / this.canvas.width, r.height / this.canvas.height);
        const ox = (r.width - this.canvas.width * s) / 2, oy = (r.height - this.canvas.height * s) / 2;
        return [(e.clientX - r.left - ox) / s, (e.clientY - r.top - oy) / s];
    }

    brushPx() {
        return Math.max(1, this.radius * Math.max(this.edit.mw, this.edit.mh));
    }

    down(e) {
        if (this.tool && this.edit && !this.stroke && e.isPrimary) {
            e.preventDefault();
            this.canvas.setPointerCapture(e.pointerId);
            this.undo.push(this.edit.alpha.slice());
            if (this.undo.length > UNDO_DEPTH) this.undo.shift();
            const p = this.toMask(e);
            this.stroke = { id: e.pointerId, last: p };
            this.paint(p, p);
        }
    }

    move(e) {
        if (this.stroke && e.pointerId === this.stroke.id) {
            e.preventDefault();
            const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
            for (const ce of events.length ? events : [e]) {
                const p = this.toMask(ce);
                this.paint(this.stroke.last, p);
                this.stroke.last = p;
            }
        }
    }

    up(e) {
        if (this.stroke && e.pointerId === this.stroke.id) {
            this.stroke = null;
            const r = Math.ceil(this.brushPx()) + 3;
            if (this.cursor) this.queue([this.cursor[0] - r, this.cursor[1] - r, this.cursor[0] + r, this.cursor[1] + r]);
            this.cursor = null;
            this.onChange();
        }
    }

    // Sweeps the brush disc from a to b
    paint(a, b) {
        const { mw, mh, alpha } = this.edit;
        const r = this.brushPx(), r2 = r * r;
        const value = this.tool === 'add' ? 1 : 0;
        const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - r)), x1 = Math.min(mw - 1, Math.ceil(Math.max(a[0], b[0]) + r));
        const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - r)), y1 = Math.min(mh - 1, Math.ceil(Math.max(a[1], b[1]) + r));
        const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                // squared distance from (x, y) to the segment a–b
                const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / len2)) : 0;
                const ex = x - a[0] - t * dx, ey = y - a[1] - t * dy;
                if (ex * ex + ey * ey <= r2) alpha[y * mw + x] = value;
            }
        }
        const old = this.cursor;
        this.cursor = b;
        const pad = Math.ceil(r) + 3;
        this.queue([x0 - 3, y0 - 3, x1 + 4, y1 + 4]);
        if (old) this.queue([old[0] - pad, old[1] - pad, old[0] + pad, old[1] + pad]);
        this.queue([b[0] - pad, b[1] - pad, b[0] + pad, b[1] + pad]);
    }

    // ── Drawing ─────────────────────────────────────────────────────────────

    queue(rect) {
        const d = this.dirty;
        this.dirty = d ? [Math.min(d[0], rect[0]), Math.min(d[1], rect[1]), Math.max(d[2], rect[2]), Math.max(d[3], rect[3])] : rect;
        if (!d) requestAnimationFrame(() => { const r = this.dirty; this.dirty = null; this.redraw(r); });
    }

    // Recomputes the shown pixels inside rect [x0, y0, x1, y1)
    redraw(rect) {
        const { mw, mh, alpha } = this.edit;
        const x0 = Math.max(0, Math.floor(rect[0])), y0 = Math.max(0, Math.floor(rect[1]));
        const x1 = Math.min(mw, Math.ceil(rect[2])), y1 = Math.min(mh, Math.ceil(rect[3]));
        if (x1 > x0 && y1 > y0) {
            const t = Math.max(1, Math.round(Math.max(mw, mh) / 400));   // outline thickness
            const src = this.base.data, dst = this.view.data;
            const outside = (x, y) => x < 0 || y < 0 || x >= mw || y >= mh || !alpha[y * mw + x];
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const p = y * mw + x, i = p * 4;
                    let edge = false;
                    if (alpha[p]) {
                        for (let d = 1; d <= t && !edge; d++) {
                            edge = outside(x - d, y) || outside(x + d, y) || outside(x, y - d) || outside(x, y + d);
                        }
                    }
                    if (edge) {
                        dst[i] = EDGE_RGB[0]; dst[i + 1] = EDGE_RGB[1]; dst[i + 2] = EDGE_RGB[2];
                    } else {
                        const k = alpha[p] ? 1 : DIM;
                        dst[i] = src[i] * k; dst[i + 1] = src[i + 1] * k; dst[i + 2] = src[i + 2] * k;
                    }
                    dst[i + 3] = 255;
                }
            }
            this.ctx.putImageData(this.view, 0, 0, x0, y0, x1 - x0, y1 - y0);
        }
        if (this.cursor) {
            this.ctx.beginPath();
            this.ctx.arc(this.cursor[0], this.cursor[1], this.brushPx(), 0, 2 * Math.PI);
            this.ctx.lineWidth = Math.max(1, Math.round(Math.max(mw, mh) / 500));
            this.ctx.strokeStyle = this.tool === 'add' ? 'rgba(255,255,255,0.9)' : 'rgba(255,80,80,0.9)';
            this.ctx.stroke();
        }
    }
}
