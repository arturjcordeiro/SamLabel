/**
 * renderer.js — Canvas drawing helpers for SAM Labeler
 *
 * Handles all overlay rendering: masks, bounding boxes, point markers,
 * hover previews, and selection outlines.
 */

const ANN_COLORS = [
  '#4fffb0', '#ff6b6b', '#ffd93d', '#4a9eff',
  '#c77dff', '#ff9f43', '#00cfe8', '#ff5c8d',
  '#a8e063', '#fd79a8', '#55efc4', '#fdcb6e',
];
let _colorIdx = 0;
export function nextColor() { return ANN_COLORS[_colorIdx++ % ANN_COLORS.length]; }

export class Renderer {
  /**
   * @param {HTMLCanvasElement} overlay - the transparent canvas drawn on top
   */
  constructor(overlay) {
    this.canvas = overlay;
    this.ctx    = overlay.getContext('2d');
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Redraw all annotations + pending state.
   *
   * @param {Array}  annotations   full list for current image
   * @param {Object|null} selected currently selected annotation
   * @param {Array}  pendingPts    [{x,y,label}] not yet committed
   * @param {Object|null} bboxDraw live bbox being drawn {x1,y1,x2,y2}
   */
  draw(annotations, selected, pendingPts, bboxDraw) {
    this.clear();

    // Committed annotations
    for (const ann of annotations) {
      this._drawAnnotation(ann, selected?.id === ann.id);
    }

    // Live bbox rubber-band
    if (bboxDraw) {
      const { x1, y1, x2, y2 } = bboxDraw;
      const ctx = this.ctx;
      ctx.save();
      ctx.strokeStyle = '#ffd93d';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.restore();
    }

    // Pending point markers
    for (const p of pendingPts) {
      this._drawPoint(p.x, p.y, p.label === 1 ? '#4fffb0' : '#ff6b6b');
    }
  }

  _drawAnnotation(ann, isSelected) {
    const ctx = this.ctx;
    const W   = this.canvas.width;
    const H   = this.canvas.height;
    const col = ann.color;

    // ── Mask fill ──────────────────────────────────────────────────────────
    if (ann.mask && ann.maskW && ann.maskH) {
      const offC   = document.createElement('canvas');
      offC.width   = ann.maskW;
      offC.height  = ann.maskH;
      const offCtx = offC.getContext('2d');
      const imgD   = offCtx.createImageData(ann.maskW, ann.maskH);

      const r = parseInt(col.slice(1, 3), 16);
      const g = parseInt(col.slice(3, 5), 16);
      const b = parseInt(col.slice(5, 7), 16);
      const a = isSelected ? 170 : 110;

      for (let i = 0; i < ann.mask.length; i++) {
        if (ann.mask[i] > 0) {
          imgD.data[i * 4]     = r;
          imgD.data[i * 4 + 1] = g;
          imgD.data[i * 4 + 2] = b;
          imgD.data[i * 4 + 3] = a;
        }
      }
      offCtx.putImageData(imgD, 0, 0);
      ctx.drawImage(offC, 0, 0, ann.maskW, ann.maskH, 0, 0, W, H);
    }

    // ── Bounding box ───────────────────────────────────────────────────────
    const [x1, y1, x2, y2] = ann.bbox;
    if (x2 > x1 && y2 > y1) {
      const mW = ann.maskW || W;
      const mH = ann.maskH || H;
      const sx = W / mW;
      const sy = H / mH;

      ctx.save();
      ctx.strokeStyle = col;
      ctx.lineWidth   = isSelected ? 2 : 1.5;
      if (!isSelected) ctx.setLineDash([4, 2]);
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

      // Label chip
      ctx.font      = 'bold 11px JetBrains Mono, monospace';
      const label   = ann.label || 'object';
      const tw      = ctx.measureText(label).width;
      const chipX   = x1 * sx;
      const chipY   = y1 * sy;
      ctx.fillStyle = col;
      ctx.fillRect(chipX, chipY - 17, tw + 10, 17);
      ctx.fillStyle = '#0a0b0d';
      ctx.fillText(label, chipX + 5, chipY - 4);

      ctx.restore();

      // Selection dashed outline
      if (isSelected) {
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x1 * sx - 3, y1 * sy - 3, (x2 - x1) * sx + 6, (y2 - y1) * sy + 6);
        ctx.restore();
      }
    }
  }

  _drawPoint(x, y, color) {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle   = 'rgba(0,0,0,0.5)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1.5;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

// ── Polygon extraction (border pixel tracing) ────────────────────────────────

export function maskToPolygon(mask, mW, mH, maxPts = 96) {
  const border = [];

  for (let r = 0; r < mH; r++) {
    for (let c = 0; c < mW; c++) {
      if (mask[r * mW + c] === 0) continue;
      const top   = r > 0     ? mask[(r - 1) * mW + c]     : 0;
      const bot   = r < mH-1  ? mask[(r + 1) * mW + c]     : 0;
      const left  = c > 0     ? mask[r * mW + (c - 1)]     : 0;
      const right = c < mW-1  ? mask[r * mW + (c + 1)]     : 0;
      if (!top || !bot || !left || !right) border.push([c, r]);
    }
  }

  if (border.length <= maxPts) return border;
  const step = border.length / maxPts;
  return Array.from({ length: maxPts }, (_, i) => border[Math.round(i * step)]);
}

export function maskToBbox(mask, mW, mH) {
  let x1 = mW, y1 = mH, x2 = 0, y2 = 0;
  for (let r = 0; r < mH; r++) {
    for (let c = 0; c < mW; c++) {
      if (mask[r * mW + c] > 0) {
        if (c < x1) x1 = c;
        if (c > x2) x2 = c;
        if (r < y1) y1 = r;
        if (r > y2) y2 = r;
      }
    }
  }
  if (x2 < x1) return [0, 0, 0, 0];
  return [x1, y1, x2, y2];
}

export function countPixels(mask) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i] > 0) n++;
  return n;
}
