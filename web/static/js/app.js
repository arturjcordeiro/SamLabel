/**
 * app.js — SAM Labeler main application
 *
 * Imports:
 *   - SAMEngine  from ./sam.js
 *   - Renderer   from ./renderer.js
 *   - exporter   from ./exporter.js
 */

import { SAMEngine }                              from './sam.js';
import { Renderer, nextColor, maskToBbox, maskToPolygon, countPixels } from './renderer.js';
import { exportZip }                              from './exporter.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const fileInput     = $('file-input');
const dropZone      = $('drop-zone');
const modelSetup    = $('model-setup');
const mainCanvas    = $('main-canvas');
const overlayCanvas = $('overlay-canvas');
const thumbList     = $('thumb-list');
const annList       = $('ann-list');
const annCount      = $('ann-count');
const loadingEl     = $('loading');
const loadingMsg    = $('loading-msg');
const samDot        = $('sam-dot');
const samStatusEl   = $('sam-status');
const annStatusEl   = $('ann-status');
const imgCounter    = $('img-counter');
const progressBar   = $('progress-bar');
const coordsEl      = $('coords');
const labelInput    = $('label-input');
const zoomLabel     = $('zoom-label');

const ctx  = mainCanvas.getContext('2d');
const renderer = new Renderer(overlayCanvas);

// ── Global state ──────────────────────────────────────────────────────────────
const state = {
  images:      [],       // [{file, name, url}]
  currentIdx:  -1,
  annotations: {},       // imageName → [ann, …]
  selected:    null,

  tool:        'click',  // 'click' | 'bbox' | 'neg'
  zoom:        1,

  pendingPts:  [],       // [{x, y, label}]
  bboxDraw:    null,     // {x1,y1,x2,y2} live rubber-band
  isDragging:  false,

  naturalW:    0,
  naturalH:    0,
};

const sam = new SAMEngine();

// ── Bootstrap: fetch server config ───────────────────────────────────────────
async function boot() {
  let encoderUrl = null;
  let decoderUrl = null;

  try {
    const cfg = await fetch('/config.json').then(r => r.json());

    if (cfg.encoder_available && cfg.decoder_available) {
      encoderUrl = cfg.encoder_url;
      decoderUrl = cfg.decoder_url;

      // Update model-setup UI to show found models
      $('enc-status').textContent = `✓ ${cfg.encoder_path.split('/').pop()} (${cfg.encoder_size_mb} MB)`;
      $('enc-status').className   = 'mstatus ok';
      $('dec-status').textContent = `✓ ${cfg.decoder_path.split('/').pop()} (${cfg.decoder_size_mb} MB)`;
      $('dec-status').className   = 'mstatus ok';
    }
  } catch (_) {
    // No server config (opened directly) — user must pick files manually
  }

  if (encoderUrl && decoderUrl) {
    modelSetup.classList.add('hidden');
    loadSAM(encoderUrl, decoderUrl);
  } else {
    // Show model-setup overlay so user can browse for local ONNX files
    modelSetup.classList.remove('hidden');
  }
}

// ── SAM loading ───────────────────────────────────────────────────────────────
async function loadSAM(encoderUrl, decoderUrl) {
  setSamStatus('yellow', 'Loading models…');
  setLoading(true, 'Loading ONNX Runtime…');

  try {
    await sam.load(encoderUrl, decoderUrl, msg => setLoading(true, msg));
    setSamStatus('green', 'SAM ready');
    toast('SAM models loaded ✓');
    modelSetup.classList.add('hidden');
    setLoading(false);

    // Encode current image if already loaded
    if (state.currentIdx >= 0) encodeCurrentImage();
  } catch (e) {
    setSamStatus('red', 'Load failed: ' + e.message);
    setLoading(false);
    toast('⚠ Model load failed — check console');
    console.error('[SAM load]', e);
  }
}

// Model-setup: manual file browse
$('enc-browse-btn').onclick = () => $('enc-file').click();
$('dec-browse-btn').onclick = () => $('dec-file').click();

$('enc-file').onchange = e => {
  if (!e.target.files[0]) return;
  window._encFile = e.target.files[0];
  $('enc-status').textContent = `✓ ${e.target.files[0].name}`;
  $('enc-status').className   = 'mstatus ok';
  checkManualModels();
};
$('dec-file').onchange = e => {
  if (!e.target.files[0]) return;
  window._decFile = e.target.files[0];
  $('dec-status').textContent = `✓ ${e.target.files[0].name}`;
  $('dec-status').className   = 'mstatus ok';
  checkManualModels();
};

function checkManualModels() {
  if (window._encFile && window._decFile) {
    const encUrl = URL.createObjectURL(window._encFile);
    const decUrl = URL.createObjectURL(window._decFile);
    loadSAM(encUrl, decUrl);
  }
}

// ── Image loading ─────────────────────────────────────────────────────────────
const VALID_EXT = /\.(jpe?g|png|bmp|webp|tiff?)$/i;

$('btn-open').onclick   = () => fileInput.click();
$('dz-browse').onclick  = () => fileInput.click();
fileInput.onchange      = e => loadFiles([...e.target.files]);

const canvasArea = $('canvas-area');
['dragover','drop'].forEach(ev => canvasArea.addEventListener(ev, e => e.preventDefault()));
canvasArea.addEventListener('drop', e => {
  const files = [...e.dataTransfer.files].filter(f => VALID_EXT.test(f.name));
  if (files.length) loadFiles(files);
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  const files = [...e.dataTransfer.files].filter(f => VALID_EXT.test(f.name));
  if (files.length) loadFiles(files);
});

function loadFiles(files) {
  const imgs = files.filter(f => VALID_EXT.test(f.name) || f.type.startsWith('image/'));
  if (!imgs.length) { toast('No supported images found'); return; }
  imgs.sort((a, b) => a.name.localeCompare(b.name));

  state.images.forEach(i => URL.revokeObjectURL(i.url));
  state.images      = imgs.map(f => ({ file: f, name: f.name, url: URL.createObjectURL(f) }));
  state.annotations = {};
  state.currentIdx  = -1;
  sam.embedding     = null;

  dropZone.classList.add('hidden');
  $('btn-export').disabled = false;
  buildThumbnails();
  navigateTo(0);
}

// ── Thumbnails ────────────────────────────────────────────────────────────────
function buildThumbnails() {
  thumbList.innerHTML = '';
  state.images.forEach((img, i) => {
    const div  = document.createElement('div');
    div.className = 'thumb-item';
    div.dataset.idx = i;

    const im  = document.createElement('img');
    im.className = 'thumb-img'; im.src = img.url; im.alt = img.name;
    im.loading = 'lazy';

    const nm  = document.createElement('span');
    nm.className = 'thumb-name'; nm.textContent = img.name;

    const badge = document.createElement('div');
    badge.className = 'thumb-badge';

    div.appendChild(im); div.appendChild(nm); div.appendChild(badge);
    div.onclick = () => navigateTo(i);
    thumbList.appendChild(div);
    img._el = div;
    img._badge = badge;
  });
}

function updateThumbBadge(idx) {
  const img  = state.images[idx];
  if (!img) return;
  const anns = state.annotations[img.name] || [];
  img._el?.classList.toggle('labeled', anns.length > 0);
}

// ── Navigation ────────────────────────────────────────────────────────────────
$('btn-prev').onclick = () => navigateTo(Math.max(0, state.currentIdx - 1));
$('btn-next').onclick = () => navigateTo(Math.min(state.images.length - 1, state.currentIdx + 1));

async function navigateTo(idx) {
  if (idx < 0 || idx >= state.images.length) return;

  const prev = state.images[state.currentIdx];
  if (prev) prev._el?.classList.remove('active');

  state.currentIdx = idx;
  state.selected   = null;
  state.pendingPts = [];
  state.bboxDraw   = null;

  sam.embedding = null;  // invalidate embedding for new image

  const img  = state.images[idx];
  img._el?.classList.add('active');
  img._el?.scrollIntoView({ block: 'nearest' });

  imgCounter.textContent = `${idx + 1} / ${state.images.length}`;
  progressBar.style.width = `${((idx + 1) / state.images.length) * 100}%`;

  await drawImageOnCanvas(img.url);
  fitZoom();
  renderer.draw(currentAnns(), state.selected, [], null);
  refreshAnnPanel();

  if (sam.isReady) encodeCurrentImage();
}

async function drawImageOnCanvas(url) {
  const bm = await createImageBitmap(await (await fetch(url)).blob());
  state.naturalW = bm.width;
  state.naturalH = bm.height;

  mainCanvas.width    = bm.width;
  mainCanvas.height   = bm.height;
  overlayCanvas.width = bm.width;
  overlayCanvas.height= bm.height;

  ctx.drawImage(bm, 0, 0);
  bm.close();

  annStatusEl.textContent = `${state.images[state.currentIdx].name}  ${bm.width}×${bm.height}`;
}

// ── SAM encoding ──────────────────────────────────────────────────────────────
let _encodeController = null;

async function encodeCurrentImage() {
  if (!sam.isReady || state.currentIdx < 0) return;

  setSamStatus('yellow', 'Encoding image…');
  setLoading(true, 'Running SAM encoder…');

  try {
    const bm = await createImageBitmap(
      await (await fetch(state.images[state.currentIdx].url)).blob()
    );
    await sam.encode(bm);
    bm.close();
    setSamStatus('green', 'SAM ready · click to segment');
  } catch (e) {
    setSamStatus('red', 'Encoding failed');
    console.error('[SAM encode]', e);
    toast('⚠ Encoding failed — ' + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Canvas interaction ────────────────────────────────────────────────────────
function canvasXY(e) {
  const rect = overlayCanvas.getBoundingClientRect();
  return {
    x: Math.round((e.clientX - rect.left) / state.zoom),
    y: Math.round((e.clientY - rect.top)  / state.zoom),
  };
}

overlayCanvas.addEventListener('mousemove', e => {
  const { x, y } = canvasXY(e);
  coordsEl.textContent = `x: ${x.toString().padStart(4)}  y: ${y.toString().padStart(4)}`;

  if (state.tool === 'bbox' && state.isDragging && state.bboxDraw) {
    state.bboxDraw.x2 = x;
    state.bboxDraw.y2 = y;
    renderer.draw(currentAnns(), state.selected, state.pendingPts, state.bboxDraw);
  }
});

overlayCanvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const { x, y } = canvasXY(e);
  if (state.tool === 'bbox') {
    state.isDragging = true;
    state.bboxDraw   = { x1: x, y1: y, x2: x, y2: y };
  }
});

overlayCanvas.addEventListener('mouseup', async e => {
  if (e.button !== 0) return;
  const { x, y } = canvasXY(e);

  if (state.currentIdx < 0) return;

  if (!sam.isReady) { toast('SAM not loaded'); return; }

  if (state.tool === 'bbox') {
    state.isDragging = false;
    if (!state.bboxDraw) return;
    const { x1, y1 } = state.bboxDraw;
    const x2 = x, y2 = y;
    state.bboxDraw = null;
    if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) return;

    if (!sam.embedding) { toast('Still encoding — please wait'); return; }
    setLoading(true, 'Running SAM decoder…');
    try {
      const result = await sam.decodeBox(
        Math.min(x1, x2), Math.min(y1, y2),
        Math.max(x1, x2), Math.max(y1, y2)
      );
      commitAnnotation(result, 'bbox');
    } catch(err) { console.error(err); toast('Decode failed: ' + err.message); }
    setLoading(false);
    return;
  }

  // click or neg point
  const label = state.tool === 'neg' ? 0 : 1;
  state.pendingPts.push({ x, y, label });

  if (!sam.embedding) { toast('Still encoding — please wait'); return; }

  setLoading(true, 'Running SAM decoder…');
  try {
    const result = await sam.decodePoints(state.pendingPts);
    // Auto-commit: each click produces one annotation
    commitAnnotation(result, 'click');
    state.pendingPts = [];
  } catch(err) { console.error(err); toast('Decode failed: ' + err.message); }
  setLoading(false);
});

overlayCanvas.addEventListener('mouseleave', () => {
  coordsEl.textContent = 'x: —    y: —';
});

// ── Commit annotation ─────────────────────────────────────────────────────────
function commitAnnotation(result, type) {
  const imgName = state.images[state.currentIdx].name;
  if (!state.annotations[imgName]) state.annotations[imgName] = [];

  const bbox    = maskToBbox(result.mask, result.width, result.height);
  const polygon = maskToPolygon(result.mask, result.width, result.height);
  const pixels  = countPixels(result.mask);

  const ann = {
    id:      Date.now() + Math.random(),
    type,
    mask:    result.mask,
    maskW:   result.width,
    maskH:   result.height,
    bbox,
    polygon,
    pixels,
    label:   labelInput.value.trim() || 'object',
    color:   nextColor(),
    score:   result.score,
  };

  state.annotations[imgName].push(ann);
  state.selected = ann;

  updateThumbBadge(state.currentIdx);
  refreshAnnPanel();
  renderer.draw(currentAnns(), state.selected, [], null);
}

// ── Annotation panel ──────────────────────────────────────────────────────────
function currentAnns() {
  const name = state.images[state.currentIdx]?.name;
  return name ? (state.annotations[name] || []) : [];
}

function refreshAnnPanel() {
  const anns = currentAnns();
  annCount.textContent = anns.length;
  annList.innerHTML = '';

  anns.forEach((ann, i) => {
    const div = document.createElement('div');
    div.className = 'ann-item' + (state.selected?.id === ann.id ? ' sel' : '');

    const dot = document.createElement('div');
    dot.className = 'ann-dot'; dot.style.background = ann.color;

    const info = document.createElement('div');
    info.className = 'ann-info';

    const t = document.createElement('div');
    t.className = 'ann-type';
    t.textContent = `#${i+1} ${ann.label}`;

    const m = document.createElement('div');
    m.className = 'ann-meta';
    const [x1,y1,x2,y2] = ann.bbox;
    m.textContent = `${ann.type} · ${x2-x1}×${y2-y1}px · iou ${ann.score?.toFixed(2) ?? '—'}`;

    info.appendChild(t); info.appendChild(m);

    const del = document.createElement('span');
    del.className = 'ann-del'; del.textContent = '✕';
    del.onclick = ev => { ev.stopPropagation(); deleteAnn(ann.id); };

    div.appendChild(dot); div.appendChild(info); div.appendChild(del);
    div.onclick = () => {
      state.selected = ann;
      refreshAnnPanel();
      renderer.draw(currentAnns(), state.selected, [], null);
    };
    annList.appendChild(div);
  });

  $('btn-del-ann').disabled = !state.selected;
}

function deleteAnn(id) {
  const name = state.images[state.currentIdx]?.name;
  if (!name) return;
  state.annotations[name] = (state.annotations[name] || []).filter(a => a.id !== id);
  if (state.selected?.id === id) state.selected = null;
  updateThumbBadge(state.currentIdx);
  refreshAnnPanel();
  renderer.draw(currentAnns(), state.selected, [], null);
}

// ── Tool buttons ──────────────────────────────────────────────────────────────
['click', 'bbox', 'neg'].forEach(t => {
  $(`tool-${t}`).onclick = () => {
    state.tool       = t;
    state.pendingPts = [];
    state.bboxDraw   = null;
    state.isDragging = false;
    renderer.draw(currentAnns(), state.selected, [], null);
    ['click', 'bbox', 'neg'].forEach(x =>
      $(`tool-${x}`).classList.toggle('active', x === t)
    );
  };
});

$('btn-clear').onclick = () => {
  const name = state.images[state.currentIdx]?.name;
  if (!name) return;
  state.annotations[name] = [];
  state.selected = null; state.pendingPts = [];
  updateThumbBadge(state.currentIdx);
  refreshAnnPanel();
  renderer.draw([], null, [], null);
};

$('btn-del-ann').onclick = () => {
  if (state.selected) deleteAnn(state.selected.id);
};

// ── Zoom ──────────────────────────────────────────────────────────────────────
$('btn-zoom-in').onclick  = () => setZoom(state.zoom * 1.25);
$('btn-zoom-out').onclick = () => setZoom(state.zoom / 1.25);
$('btn-zoom-fit').onclick = fitZoom;

function setZoom(z) {
  z = Math.max(0.05, Math.min(10, z));
  state.zoom = z;
  const cw = Math.round(mainCanvas.width  * z);
  const ch = Math.round(mainCanvas.height * z);
  const cc = $('canvas-container');
  cc.style.width  = cw + 'px';
  cc.style.height = ch + 'px';
  mainCanvas.style.width     = cw + 'px';
  mainCanvas.style.height    = ch + 'px';
  overlayCanvas.style.width  = cw + 'px';
  overlayCanvas.style.height = ch + 'px';
  zoomLabel.textContent = Math.round(z * 100) + '%';
}

function fitZoom() {
  if (!mainCanvas.width) return;
  const wrap = $('canvas-wrap');
  const zw   = (wrap.clientWidth  - 48) / mainCanvas.width;
  const zh   = (wrap.clientHeight - 48) / mainCanvas.height;
  setZoom(Math.min(zw, zh, 1));
}

// ── Export ────────────────────────────────────────────────────────────────────
$('btn-export').onclick = async () => {
  const fmt = document.querySelector('input[name="fmt"]:checked').value;
  toast('Building archive…');

  const result = await exportZip(
    fmt,
    state.images,
    state.annotations,
    msg => setLoading(true, msg)
  );
  setLoading(false);

  if (!result) { toast('No annotations to export'); return; }

  const a = document.createElement('a');
  a.href = URL.createObjectURL(result.blob);
  a.download = `sam_labels_${Date.now()}.zip`;
  a.click();
  toast(`Exported ${result.exported} image(s) ✓`);
};

// Radio row styling
document.querySelectorAll('input[name="fmt"]').forEach(r => {
  r.addEventListener('change', () => {
    document.querySelectorAll('.radio-row').forEach(row => row.classList.remove('checked'));
    r.closest('.radio-row').classList.add('checked');
  });
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key;
  if      (k === 'ArrowLeft'  || k === 'a' || k === 'A') navigateTo(Math.max(0, state.currentIdx - 1));
  else if (k === 'ArrowRight' || k === 'd' || k === 'D') navigateTo(Math.min(state.images.length - 1, state.currentIdx + 1));
  else if (k === 'r' || k === 'R') $('btn-clear').click();
  else if (k === 'Delete')         $('btn-del-ann').click();
  else if (k === '+' || k === '=') setZoom(state.zoom * 1.2);
  else if (k === '-')              setZoom(state.zoom / 1.2);
  else if (k === 'f' || k === 'F') fitZoom();
  else if (k === 's' || k === 'S') $('btn-export').click();
  else if (k === '1') $('tool-click').click();
  else if (k === '2') $('tool-bbox').click();
  else if (k === '3') $('tool-neg').click();
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function setSamStatus(color, msg) {
  samDot.className = `s-dot ${color}`;
  samStatusEl.textContent = msg;
}

function setLoading(on, msg = '') {
  loadingEl.classList.toggle('hidden', !on);
  if (msg) loadingMsg.textContent = msg;
}

let _toastTimer;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

// ── Start ─────────────────────────────────────────────────────────────────────
boot();
