// Temporary admin page: save real sample photos to the server's samples/ folder,
// named by cell, with the same outline preview as the main page as a check.
import { findPiece } from './cutout.js';
import { decodePhoto, drawOutline, workCopy } from './photo.js';

const $ = (id) => document.getElementById(id);

const state = {
    info: null,       // /api/info
    samples: {},      // label → [file names]
    cell: null,       // selected label
    target: '',       // '' = add as new, else the file to replace
    file: null,       // the original photo File, uploaded as is
    busy: false,
};

async function loadSamples() {
    const res = await fetch('/api/admin/samples');
    if (!res.ok) throw new Error('admin off');
    state.samples = (await res.json()).cells;
}

function renderCells() {
    const { rowLabels, colLabels } = state.info;
    const grid = $('cells');
    grid.style.setProperty('--cols', colLabels.length);
    grid.replaceChildren(...rowLabels.flatMap((r) => colLabels.map((c) => {
        const label = r + c;
        const count = (state.samples[label] || []).length;
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'cell';
        b.classList.toggle('selected', label === state.cell);
        b.classList.toggle('empty', count === 0);
        b.innerHTML = `<span>${label}</span><small>${count || ''}</small>`;
        b.addEventListener('click', () => selectCell(label));
        return b;
    })));
}

function selectCell(label) {
    state.cell = label;
    state.target = '';
    renderCells();
    renderTargets();
    updateSave();
}

// "New photo" plus one choice per existing file of the cell, to replace it
function renderTargets() {
    const files = state.samples[state.cell] || [];
    $('existing').hidden = !state.cell;
    const choices = [['', 'New photo'], ...files.map((f) => [f, `Replace ${f}`])];
    $('targets').replaceChildren(...choices.map(([value, text]) => {
        const label = document.createElement('label');
        label.className = 'chip';
        label.style.setProperty('--chip', value ? '#ef4444' : '#10b981');
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'target';
        input.value = value;
        input.checked = value === state.target;
        input.addEventListener('change', () => { state.target = value; updateSave(); });
        const dot = document.createElement('span');
        dot.className = 'dot';
        label.append(input, dot, document.createTextNode(text));
        return label;
    }));
}

function updateSave() {
    const ready = Boolean(state.cell && state.file) && !state.busy;
    $('save').disabled = !ready;
    $('save').textContent = state.busy ? 'Saving…'
        : state.cell ? (state.target ? `Replace ${state.target}` : `Save as new ${state.cell} photo`)
        : 'Pick a cell first';
}

async function loadPhoto(file) {
    const photo = await decodePhoto(file);
    const work = workCopy(photo);
    const found = findPiece(work.rgba, work.w, work.h);
    drawOutline($('preview-canvas'), work, found);
    const ok = found && found.coverage > 0.002 && found.coverage < 0.8;
    const el = $('outline-status');
    el.textContent = ok
        ? `Outline found (${(found.coverage * 100).toFixed(0)}% of the photo). If it misses the piece, retake it.`
        : 'No clear outline. Try a different surface or angle; you can still save it.';
    el.dataset.level = ok && !found.touchesBorder ? 'ok' : 'warn';
    state.file = file;
    $('saved').hidden = true;
    $('preview').hidden = false;
    updateSave();
}

async function save() {
    const form = new FormData();
    form.append('photo', state.file, state.file.name || 'photo.jpg');
    form.append('cell', state.cell);
    if (state.target) form.append('replace', state.target);

    state.busy = true;
    updateSave();
    try {
        const res = await fetch('/api/admin/sample', { method: 'POST', body: form });
        const body = await res.json();
        if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
        $('saved').textContent = `Saved ${body.saved}.`;
        $('saved').dataset.level = 'ok';
        state.file = null;
        state.target = '';
        $('preview').hidden = true;
        await loadSamples();
        renderCells();
        renderTargets();
    } catch (err) {
        $('saved').textContent = `Not saved: ${err.message}`;
        $('saved').dataset.level = 'bad';
    } finally {
        $('saved').hidden = false;
        state.busy = false;
        updateSave();
    }
}

async function init() {
    for (const id of ['camera', 'gallery']) {
        $(id).addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) loadPhoto(file).catch((err) => alert(`Could not read that image: ${err.message}`));
            e.target.value = '';
        });
    }
    $('save').addEventListener('click', save);
    try {
        state.info = await (await fetch('/api/info')).json();
        await loadSamples();
        renderCells();
        updateSave();
    } catch {
        $('off').hidden = false;
        $('pick').hidden = true;
        $('shoot').hidden = true;
    }
}

init();
