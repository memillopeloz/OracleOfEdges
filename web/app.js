// Argus client: photo → outline preview → cut-out upload → results table.
import { findPiece } from './cutout.js';
import { cutoutPng, decodePhoto, drawOutline, workCopy } from './photo.js';

const MATCHER_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#14b8a6'];
const STORAGE_KEY    = 'argus.matchers';

const $ = (id) => document.getElementById(id);

const state = {
    info: null,          // /api/info
    photo: null,         // full-resolution ImageBitmap
    work: null,          // { canvas, rgba, w, h } downscaled copy used for detection
    found: null,         // findPiece() result
    cutout: null,        // PNG Blob ready to upload
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

function detect() {
    const { rgba, w, h } = state.work;
    const found = findPiece(rgba, w, h, Number($('sensitivity').value));
    state.found = found;
    state.cutout = null;
    drawOutline($('preview-canvas'), state.work, found);
    describeOutline();
    updateLocateButton();
    if (outlineUsable()) {
        cutoutPng(state.photo, state.work, found).then((blob) => {
            if (state.found === found) {   // ignore if the outline changed meanwhile
                state.cutout = blob;
                updateLocateButton();
            }
        });
    }
}

function outlineUsable() {
    const f = state.found;
    return Boolean(f) && f.coverage > 0.002 && f.coverage < 0.8;
}

function describeOutline() {
    const f = state.found;
    let text = 'Outline found — check it hugs the piece, then locate.';
    let level = 'ok';
    if (!f || f.coverage <= 0.002) {
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
    const ready = Boolean(state.cutout) && selectedMatchers().length > 0 && !state.busy;
    $('locate').disabled = !ready;
    $('locate').textContent = state.busy ? 'Locating…' : 'Locate piece';
    $('locate').classList.toggle('loading', state.busy);
}

async function locate() {
    const form = new FormData();
    form.append('piece', state.cutout, 'piece.png');
    for (const name of selectedMatchers()) form.append('matcher', name);

    state.busy = true;
    updateLocateButton();
    try {
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
    drawOverview(results.filter((r) => !r.error && r.row >= 0));
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Grid over the artwork, with each matcher's cell outlined in its colour
function drawOverview(results) {
    const { rows, cols, rowLabels, colLabels } = state.info;
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
    $('overview-cols').replaceChildren(...colLabels.map((l) => Object.assign(document.createElement('span'), { textContent: l })));
    $('overview-rows').replaceChildren(...rowLabels.map((l) => Object.assign(document.createElement('span'), { textContent: l })));
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
