// Argus client: photo → outline preview → cut-out upload → results table.
import { findPiece } from './cutout.js';
import { OutlineEditor } from './editor.js';
import { cutoutPng, decodePhoto, editMask, workCopy } from './photo.js';

const MATCHER_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#14b8a6'];
const STORAGE_KEY    = 'argus.matchers';
const ZOOM_MARGIN    = 0.12;  // border around a zoomed cell, as a fraction of the cell
const SPOT_RADIUS    = 0.06;  // overview spot radius, in cells
const MIN_EDITED_AREA = 400;   // mask pixels a hand-painted outline needs before it can be located

const $ = (id) => document.getElementById(id);

const state = {
    info: null,          // /api/info
    photo: null,         // full-resolution ImageBitmap
    work: null,          // { canvas, rgba, w, h } downscaled copy used for detection
    found: null,         // findPiece() result
    editor: null,        // OutlineEditor over the preview canvas
    edited: false,       // outline touched up by hand since the last detection
    cutout: null,        // PNG Blob uploaded last (shown with the results)
    results: [],         // successful results of the last locate, for the overview and zoom
    zoomCell: null,      // { row, col } shown zoomed, or null
    busy: false,
};

// ── Matchers ────────────────────────────────────────────────────────────────

function colorOf(name) {
    const i = state.info.matchers.findIndex((m) => m.name === name);
    return MATCHER_COLORS[Math.max(0, i) % MATCHER_COLORS.length];
}

function loadSelection() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { /* none saved */ }
    return Array.isArray(saved) ? saved : null;
}

function saveSelection() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedMatchers())); } catch { /* ignore */ }
}

function selectedMatchers() {
    return [...document.querySelectorAll('#matchers input:checked')].map((el) => el.value);
}

function renderMatchers() {
    const saved = loadSelection();
    const box = $('matchers');
    box.replaceChildren();
    for (const m of state.info.matchers) {
        const label = document.createElement('label');
        label.className = 'chip';
        label.style.setProperty('--chip', colorOf(m.name));
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = m.name;
        input.disabled = !m.available;
        input.checked = m.available && (saved ? saved.includes(m.name) : true);
        input.addEventListener('change', () => { saveSelection(); updateLocateButton(); });
        const dot = document.createElement('span');
        dot.className = 'dot';
        label.append(input, dot, document.createTextNode(m.name));
        if (!m.available) label.title = 'Not available on this server';
        box.append(label);
    }
}

// ── Photo and outline ───────────────────────────────────────────────────────

async function loadPhoto(file) {
    state.photo = await decodePhoto(file);
    state.work = workCopy(state.photo);
    $('sensitivity').value = 1;
    $('results').hidden = true;
    $('preview').hidden = false;
    detect();
    $('preview').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Automatic outline at the current sensitivity, replacing any hand edits
function detect() {
    const { rgba, w, h } = state.work;
    state.found = findPiece(rgba, w, h, Number($('sensitivity').value));
    state.edited = false;
    state.editor.load(state.photo, editMask(state.photo, state.work, state.found));
    outlineChanged();
}

function outlineChanged() {
    $('undo').disabled = !state.editor.canUndo();
    describeOutline();
    updateLocateButton();
}

function outlineUsable() {
    const f = state.found;
    const automatic = Boolean(f) && f.coverage > 0.002 && f.coverage < 0.8;
    return state.edited ? state.editor.area() > MIN_EDITED_AREA : automatic;
}

function describeOutline() {
    const f = state.found;
    let text = 'Outline found — check it hugs the piece (touch it up if needed), then locate.';
    let level = 'ok';
    if (state.edited) {
        const ok = outlineUsable();
        text = ok ? 'Outline touched up by hand.' : 'Paint over the piece with Add to outline it.';
        level = ok ? 'ok' : 'bad';
    } else if (!f || f.coverage <= 0.002) {
        text = 'No piece found. Paint over it with Add, or retake on a plain surface that contrasts with it.';
        level = 'bad';
        text = 'No piece found. Use a plain surface that contrasts with the piece.';
        level = 'bad';
    } else if (f.coverage >= 0.8) {
        text = 'The background could not be told apart. Try a plainer surface or a tighter outline.';
        level = 'bad';
    } else if (f.touchesBorder) {
        text = 'The outline touches the photo edge. Centre the piece with some space around it.';
        level = 'warn';
    } else if (f.coverage < 0.01) {
        text = 'The piece is small in the photo — moving closer gives better results.';
        level = 'warn';
    }
    const el = $('outline-status');
    el.textContent = text;
    el.dataset.level = level;
}

// ── Locate ──────────────────────────────────────────────────────────────────

function updateLocateButton() {
    const ready = Boolean(state.photo) && outlineUsable() && selectedMatchers().length > 0 && !state.busy;
    $('locate').disabled = !ready;
    $('locate').textContent = state.busy ? 'Locating…' : 'Locate piece';
    $('locate').classList.toggle('loading', state.busy);
}

async function locate() {
    state.busy = true;
    updateLocateButton();
    try {
        state.cutout = await cutoutPng(state.photo, state.editor.edit);
        const form = new FormData();
        form.append('piece', state.cutout, 'piece.png');
        for (const name of selectedMatchers()) form.append('matcher', name);
        const res = await fetch('/api/locate', { method: 'POST', body: form });
        const body = await res.json();
        if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
        renderResults(body.results);
    } catch (err) {
        showError(err.message);
    } finally {
        state.busy = false;
        updateLocateButton();
    }
}

function showError(message) {
    $('results').hidden = false;
    state.results = [];
    closeZoom();
    $('result-rows').replaceChildren();
    $('overview-marks').replaceChildren();
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.className = 'error';
    cell.textContent = message;
    row.append(cell);
    $('result-rows').append(row);
}

function renderResults(results) {
    $('results').hidden = false;
    $('piece-thumb').src = URL.createObjectURL(state.cutout);

    const rows = results.map((r) => {
        const tr = document.createElement('tr');
        const method = document.createElement('td');
        method.innerHTML = '<span class="dot"></span>';
        method.firstChild.style.background = colorOf(r.matcher);
        method.append(r.matcher);
        tr.append(method);

        if (r.error) {
            const td = document.createElement('td');
            td.colSpan = 2;
            td.className = 'error';
            td.textContent = r.error;
            tr.append(td);
        } else {
            const coord = document.createElement('td');
            coord.className = 'coord';
            coord.textContent = r.coord;
            const conf = document.createElement('td');
            const pct = Math.round(Math.max(0, Math.min(1, r.confidence)) * 100);
            conf.innerHTML = `<div class="bar"><span style="width:${pct}%"></span></div><small>${pct}%</small>`;
            conf.querySelector('.bar span').style.background = colorOf(r.matcher);
            tr.append(coord, conf);
        }
        const time = document.createElement('td');
        time.className = 'num';
        time.textContent = `${r.timeMs} ms`;
        tr.append(time);
        return tr;
    });
    $('result-rows').replaceChildren(...rows);
    state.results = results.filter((r) => !r.error && r.row >= 0);
    state.zoomCell = null;
    $('zoom').hidden = true;
    drawOverview();
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Grid over the artwork, with each matcher's cell outlined in its colour, a
// spot where it puts the piece, and the zoomed cell (if any) dashed
function drawOverview() {
    const results = state.results;
    const { rows, cols, rowLabels, colLabels, artworkWidth, artworkHeight } = state.info;
    const NS = 'http://www.w3.org/2000/svg';
    const svg = $('overview-marks');
    svg.setAttribute('viewBox', `0 0 ${cols} ${rows}`);
    const parts = [];
    for (let c = 1; c < cols; c++) parts.push(`<line x1="${c}" y1="0" x2="${c}" y2="${rows}" class="grid"/>`);
    for (let r = 1; r < rows; r++) parts.push(`<line x1="0" y1="${r}" x2="${cols}" y2="${r}" class="grid"/>`);
    svg.innerHTML = parts.join('');

    // Matchers that agree on a cell are nested so every outline stays visible
    const perCell = new Map();
    for (const r of results) {
        const key = `${r.row},${r.col}`;
        const depth = perCell.get(key) || 0;
        perCell.set(key, depth + 1);
        const inset = 0.04 + depth * 0.07;
        const rect = document.createElementNS(NS, 'rect');
        rect.setAttribute('x', r.col + inset);
        rect.setAttribute('y', r.row + inset);
        rect.setAttribute('width', Math.max(0.05, 1 - 2 * inset));
        rect.setAttribute('height', Math.max(0.05, 1 - 2 * inset));
        rect.setAttribute('class', 'mark');
        rect.style.stroke = colorOf(r.matcher);
        rect.style.fill = colorOf(r.matcher);
        svg.append(rect);
    }
    // Cells aren't square on screen: the ellipse's radii make the spot round
    const aspect = (artworkWidth / cols) / (artworkHeight / rows);
    for (const r of results) {
        const c = outlineCentre(r.outline);
        if (c) {
            const spot = document.createElementNS(NS, 'ellipse');
            spot.setAttribute('cx', c[0] * cols / artworkWidth);
            spot.setAttribute('cy', c[1] * rows / artworkHeight);
            spot.setAttribute('rx', SPOT_RADIUS);
            spot.setAttribute('ry', SPOT_RADIUS * aspect);
            spot.setAttribute('class', 'spot');
            spot.style.fill = colorOf(r.matcher);
            svg.append(spot);
        }
    }
    if (state.zoomCell) {
        const rect = document.createElementNS(NS, 'rect');
        rect.setAttribute('x', state.zoomCell.col);
        rect.setAttribute('y', state.zoomCell.row);
        rect.setAttribute('width', 1);
        rect.setAttribute('height', 1);
        rect.setAttribute('class', 'chosen');
        svg.append(rect);
    }
    $('overview-cols').replaceChildren(...colLabels.map((l) => Object.assign(document.createElement('span'), { textContent: l })));
    $('overview-rows').replaceChildren(...rowLabels.map((l) => Object.assign(document.createElement('span'), { textContent: l })));
}

// ── Zoom ────────────────────────────────────────────────────────────────────

// Centre of an outline's bounding box, or null for an empty outline
function outlineCentre(outline) {
    let centre = null;
    if (outline && outline.length) {
        const xs = outline.map((p) => p[0]), ys = outline.map((p) => p[1]);
        centre = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
    }
    return centre;
}

// Artwork rectangle of a cell, as GridConfig::cellRect slices it
function cellRect(row, col) {
    const { rows, cols, artworkWidth: W, artworkHeight: H } = state.info;
    const x0 = Math.floor(col * W / cols), x1 = Math.floor((col + 1) * W / cols);
    const y0 = Math.floor(row * H / rows), y1 = Math.floor((row + 1) * H / rows);
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// Shows a cell of the artwork at full resolution with every matcher's outline
function openZoom(row, col) {
    const { rowLabels, colLabels, artworkWidth: W, artworkHeight: H } = state.info;
    const cell = cellRect(row, col);
    const m = Math.round(Math.max(cell.w, cell.h) * ZOOM_MARGIN);
    const x0 = Math.max(0, cell.x - m), y0 = Math.max(0, cell.y - m);
    const region = { x: x0, y: y0, w: Math.min(W, cell.x + cell.w + m) - x0, h: Math.min(H, cell.y + cell.h + m) - y0 };
    state.zoomCell = { row, col };
    drawOverview();

    $('zoom').hidden = false;
    $('zoom-title').textContent = `Cell ${rowLabels[row]}${colLabels[col]}`;
    $('zoom-thumb').src = $('piece-thumb').src;
    const frame = document.querySelector('.zoom-frame');
    frame.classList.add('loading');
    const img = $('zoom-img');
    img.onload = () => frame.classList.remove('loading');
    img.src = `/api/artwork/region?x=${region.x}&y=${region.y}&w=${region.w}&h=${region.h}`;

    const NS = 'http://www.w3.org/2000/svg';
    const svg = $('zoom-marks');
    svg.setAttribute('viewBox', `${region.x} ${region.y} ${region.w} ${region.h}`);
    const cellBox = document.createElementNS(NS, 'rect');
    for (const [k, v] of Object.entries({ x: cell.x, y: cell.y, width: cell.w, height: cell.h })) cellBox.setAttribute(k, v);
    cellBox.setAttribute('class', 'chosen');
    const marks = [cellBox];
    const legend = [];
    for (const r of state.results) {
        const c = outlineCentre(r.outline);
        const inView = c && c[0] >= region.x && c[0] < region.x + region.w && c[1] >= region.y && c[1] < region.y + region.h;
        if (inView) {
            const poly = document.createElementNS(NS, 'polygon');
            poly.setAttribute('points', r.outline.map((p) => p.join(',')).join(' '));
            poly.setAttribute('class', 'outline');
            poly.style.stroke = colorOf(r.matcher);
            poly.style.fill = colorOf(r.matcher);
            marks.push(poly);
        }
        const item = document.createElement('span');
        item.innerHTML = '<span class="dot"></span>';
        item.firstChild.style.background = colorOf(r.matcher);
        item.append(inView ? r.matcher : `${r.matcher} (${r.coord})`);
        if (!inView) item.style.opacity = 0.6;
        legend.push(item);
    }
    svg.replaceChildren(...marks);
    $('zoom-legend').replaceChildren(...legend);
    $('zoom').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeZoom() {
    state.zoomCell = null;
    $('zoom').hidden = true;
    if (state.info) drawOverview();
}

// Tap on the overview → zoom into that cell
function overviewTap(e) {
    const { rows, cols } = state.info;
    const r = e.currentTarget.getBoundingClientRect();
    const col = Math.floor((e.clientX - r.left) / r.width * cols);
    const row = Math.floor((e.clientY - r.top) / r.height * rows);
    if (row >= 0 && row < rows && col >= 0 && col < cols) openZoom(row, col);
}

// ── Wiring ──────────────────────────────────────────────────────────────────

function pickPhoto(input) {
    const file = input.files && input.files[0];
    if (file) loadPhoto(file).catch((err) => alert(`Could not read that image: ${err.message}`));
    input.value = '';
}

async function init() {
    for (const id of ['camera', 'gallery']) $(id).addEventListener('change', (e) => pickPhoto(e.target));
    $('retake').addEventListener('click', () => $('camera').click());
    $('locate').addEventListener('click', locate);
    document.querySelector('.overview .frame').addEventListener('click', overviewTap);
    $('zoom-close').addEventListener('click', closeZoom);

    state.editor = new OutlineEditor($('preview-canvas'), () => { state.edited = true; outlineChanged(); });
    for (const button of document.querySelectorAll('.tool')) {
        button.addEventListener('click', () => {
            const on = button.getAttribute('aria-pressed') !== 'true';
            for (const b of document.querySelectorAll('.tool')) b.setAttribute('aria-pressed', String(on && b === button));
            state.editor.setTool(on ? button.dataset.tool : null);
        });
    }
    $('undo').addEventListener('click', () => state.editor.undoLast());
    $('reset').addEventListener('click', () => { if (state.work) detect(); });
    $('brush').addEventListener('input', () => { state.editor.radius = Number($('brush').value); });

    let pending = false;
    $('sensitivity').addEventListener('input', () => {
        if (!pending) {
            pending = true;
            requestAnimationFrame(() => { pending = false; detect(); });
        }
    });

    try {
        const res = await fetch('/api/info');
        state.info = await res.json();
        renderMatchers();
        $('overview-img').src = '/api/artwork';
    } catch {
        $('matchers').textContent = 'Could not reach the Argus server.';
    }
}

init();
