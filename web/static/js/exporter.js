/**
 * exporter.js — Export annotations as PNG masks and/or JSON
 *
 * Supports:
 *   - "mask"  → per-annotation uint8 PNG (0/255) + merged PNG
 *   - "json"  → COCO-like JSON with bbox, polygon, area
 *   - "both"  → both formats
 *
 * Output is packed into a ZIP using JSZip (loaded lazily from CDN).
 */

async function ensureJSZip() {
  if (typeof JSZip !== 'undefined') return;
  await new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
    s.onload = res; s.onerror = () => rej(new Error('Failed to load JSZip'));
    document.head.appendChild(s);
  });
}

/**
 * Build a binary mask PNG for a single annotation.
 * Output size matches the original image (W × H).
 * Mask values: 0=background, 255=foreground.
 */
export async function buildSingleMaskPNG(imgUrl, ann) {
  const { naturalW, naturalH } = await getImageSize(imgUrl);
  const offC = new OffscreenCanvas(naturalW, naturalH);
  const oc   = offC.getContext('2d');
  const imgD = oc.createImageData(naturalW, naturalH);

  const mW = ann.maskW, mH = ann.maskH;
  const sx = naturalW / mW,  sy = naturalH / mH;

  for (let r = 0; r < mH; r++) {
    for (let c = 0; c < mW; c++) {
      if (ann.mask[r * mW + c] > 0) {
        // Use bilinear-ish: fill a small block
        const px = Math.round(c * sx);
        const py = Math.round(r * sy);
        if (px < naturalW && py < naturalH) {
          const idx = (py * naturalW + px) * 4;
          imgD.data[idx]     = 255;
          imgD.data[idx + 1] = 255;
          imgD.data[idx + 2] = 255;
          imgD.data[idx + 3] = 255;
        }
      }
    }
  }

  oc.putImageData(imgD, 0, 0);
  const blob = await offC.convertToBlob({ type: 'image/png' });
  return blob;
}

/**
 * Build a merged mask PNG (all annotations combined, same uint8 0/255).
 */
export async function buildMergedMaskPNG(imgUrl, anns) {
  const { naturalW, naturalH } = await getImageSize(imgUrl);
  const offC = new OffscreenCanvas(naturalW, naturalH);
  const oc   = offC.getContext('2d');
  const imgD = oc.createImageData(naturalW, naturalH);

  for (const ann of anns) {
    const mW = ann.maskW, mH = ann.maskH;
    const sx = naturalW / mW,  sy = naturalH / mH;
    for (let r = 0; r < mH; r++) {
      for (let c = 0; c < mW; c++) {
        if (ann.mask[r * mW + c] > 0) {
          const px = Math.round(c * sx);
          const py = Math.round(r * sy);
          if (px < naturalW && py < naturalH) {
            const idx = (py * naturalW + px) * 4;
            imgD.data[idx] = imgD.data[idx+1] = imgD.data[idx+2] = imgD.data[idx+3] = 255;
          }
        }
      }
    }
  }

  oc.putImageData(imgD, 0, 0);
  return offC.convertToBlob({ type: 'image/png' });
}

/**
 * Build COCO-style JSON for one image.
 */
export function buildJSON(imgName, naturalW, naturalH, anns) {
  return {
    image:       imgName,
    width:       naturalW,
    height:      naturalH,
    num_annotations: anns.length,
    annotations: anns.map((ann, i) => ({
      id:      i + 1,
      label:   ann.label,
      type:    ann.type,
      bbox: {
        x1:     ann.bbox[0],
        y1:     ann.bbox[1],
        x2:     ann.bbox[2],
        y2:     ann.bbox[3],
        width:  ann.bbox[2] - ann.bbox[0],
        height: ann.bbox[3] - ann.bbox[1],
      },
      // COCO-format segmentation polygon (flat [x,y,x,y,...])
      segmentation: ann.polygon.flat(),
      area_pixels:  ann.pixels,
      color:        ann.color,
    })),
  };
}

/**
 * Export all labeled images to a ZIP archive and trigger download.
 *
 * @param {string}  fmt        'mask' | 'json' | 'both'
 * @param {Array}   images     [{name, url}]
 * @param {Object}  annotations { imageName → [ann, …] }
 * @param {Function} onProgress (msg) => void
 */
export async function exportZip(fmt, images, annotations, onProgress) {
  onProgress?.('Preparing ZIP…');
  await ensureJSZip();

  const zip      = new JSZip();
  let exported   = 0;
  const summary  = { exported_at: new Date().toISOString(), format: fmt, images: [] };

  for (const img of images) {
    const anns = annotations[img.name] || [];
    if (!anns.length) continue;

    const stem = img.name.replace(/\.[^.]+$/, '');
    onProgress?.(`Exporting ${img.name}…`);

    const { naturalW, naturalH } = await getImageSize(img.url);

    if (fmt === 'mask' || fmt === 'both') {
      // Merged mask (all annotations)
      const merged = await buildMergedMaskPNG(img.url, anns);
      zip.file(`masks/${stem}_mask.png`, merged);

      // Per-annotation masks
      for (let i = 0; i < anns.length; i++) {
        const single = await buildSingleMaskPNG(img.url, anns[i]);
        zip.file(`masks/per_annotation/${stem}_ann${i+1}_${anns[i].label}.png`, single);
      }
    }

    if (fmt === 'json' || fmt === 'both') {
      const json = buildJSON(img.name, naturalW, naturalH, anns);
      zip.file(`json/${stem}.json`, JSON.stringify(json, null, 2));
    }

    summary.images.push({ name: img.name, annotations: anns.length, width: naturalW, height: naturalH });
    exported++;
  }

  if (!exported) return null;

  zip.file('summary.json', JSON.stringify(summary, null, 2));
  onProgress?.('Compressing…');
  const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
  return { blob, exported };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const _sizeCache = new Map();

async function getImageSize(url) {
  if (_sizeCache.has(url)) return _sizeCache.get(url);
  const bm  = await createImageBitmap(await (await fetch(url)).blob());
  const res = { naturalW: bm.width, naturalH: bm.height };
  bm.close();
  _sizeCache.set(url, res);
  return res;
}
