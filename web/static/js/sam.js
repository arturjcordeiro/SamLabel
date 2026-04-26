/**
 * sam.js — SAM ONNX inference engine
 *
 * Handles:
 *  - Loading encoder + decoder ONNX sessions
 *  - Image preprocessing (correct RGB normalisation, letterbox resize)
 *  - Running the encoder to get an image embedding
 *  - Running the decoder with point or box prompts
 *  - Introspecting model input/output names at runtime to handle
 *    any SAM variant (vit_b / vit_l / vit_h, quantised or full)
 */

const SAM_SIZE = 1024;   // SAM expects 1024×1024 input

// ImageNet channel means & stds used by SAM
const PIXEL_MEAN = [123.675, 116.28,  103.53];
const PIXEL_STD  = [58.395,  57.12,   57.375];

export class SAMEngine {
  constructor() {
    this.encoder   = null;
    this.decoder   = null;
    this.embedding = null;   // { tensor, origW, origH }

    // Discovered I/O names (populated after model load)
    this._encIn   = null;
    this._encOut  = null;
    this._decIn   = {};
    this._decOut  = {};
  }

  // ── Load models ──────────────────────────────────────────────────────────

  async load(encoderUrl, decoderUrl, onProgress) {
    onProgress?.('Loading ONNX Runtime…');
    await this._ensureORT();

    ort.env.wasm.wasmPaths = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/';
    // Use wasm backend; set numThreads=1 to avoid SharedArrayBuffer requirement
    const opts = { executionProviders: ['wasm'], sessionOptions: { executionMode: 'sequential' } };

    onProgress?.(`Loading encoder…`);
    this.encoder = await ort.InferenceSession.create(encoderUrl, opts);
    this._discoverEncoderIO();

    onProgress?.(`Loading decoder…`);
    this.decoder = await ort.InferenceSession.create(decoderUrl, opts);
    this._discoverDecoderIO();

    onProgress?.(null);
  }

  _discoverEncoderIO() {
    // Encoder has exactly 1 input (the image tensor) and 1 output (embedding)
    this._encIn  = this.encoder.inputNames[0];
    this._encOut = this.encoder.outputNames[0];
    console.log(`[SAM] Encoder  in="${this._encIn}"  out="${this._encOut}"`);
  }

  _discoverDecoderIO() {
    // Map logical names → actual tensor names in this model
    const inp = this.decoder.inputNames;
    const out = this.decoder.outputNames;
    console.log('[SAM] Decoder inputs:', inp);
    console.log('[SAM] Decoder outputs:', out);

    const findIn  = (...candidates) => inp.find(n => candidates.some(c => n.includes(c)));
    const findOut = (...candidates) => out.find(n => candidates.some(c => n.includes(c)));

    this._decIn = {
      embedding:   findIn('image_embed', 'embedding', 'image_encodings'),
      pointCoords: findIn('point_coords'),
      pointLabels: findIn('point_labels'),
      maskInput:   findIn('mask_input', 'mask_tokens'),
      hasMask:     findIn('has_mask_input', 'has_mask'),
      origSize:    findIn('orig_im_size', 'orig_size', 'image_size'),
    };

    this._decOut = {
      masks:  findOut('masks', 'low_res_masks'),
      iou:    findOut('iou_predictions', 'iou'),
    };

    console.log('[SAM] Decoder input map:', this._decIn);
    console.log('[SAM] Decoder output map:', this._decOut);

    // Validate
    for (const [k, v] of Object.entries(this._decIn)) {
      if (!v) console.warn(`[SAM] ⚠ Could not map decoder input: ${k}`);
    }
  }

  // ── Image encoding ────────────────────────────────────────────────────────

  /**
   * Encode an ImageBitmap or HTMLImageElement.
   * Stores the embedding internally; call getEmbedding() to retrieve it.
   */
  async encode(imageSource) {
    if (!this.encoder) throw new Error('Encoder not loaded');

    const { tensor, origW, origH, padX, padY, scaleX, scaleY } =
      await this._preprocessImage(imageSource);

    const feeds = { [this._encIn]: tensor };
    const results = await this.encoder.run(feeds);
    const embTensor = results[this._encOut];

    this.embedding = { tensor: embTensor, origW, origH, padX, padY, scaleX, scaleY };
    return this.embedding;
  }

  /**
   * Resize image to SAM_SIZE × SAM_SIZE with letterboxing,
   * normalise with ImageNet mean/std, return CHW Float32 tensor.
   *
   * Also returns the scaling factors needed to map image coords → SAM coords.
   */
  async _preprocessImage(source) {
    // Draw into SAM_SIZE × SAM_SIZE offscreen canvas (letterboxed)
    const offC = new OffscreenCanvas(SAM_SIZE, SAM_SIZE);
    const oc   = offC.getContext('2d');

    // Determine natural size
    let origW, origH;
    if (source instanceof HTMLImageElement) {
      origW = source.naturalWidth;
      origH = source.naturalHeight;
    } else if (source instanceof ImageBitmap) {
      origW = source.width;
      origH = source.height;
    } else {
      origW = source.width;
      origH = source.height;
    }

    // Scale keeping aspect ratio
    const scale  = Math.min(SAM_SIZE / origW, SAM_SIZE / origH);
    const scaledW = Math.round(origW * scale);
    const scaledH = Math.round(origH * scale);
    const padX   = Math.floor((SAM_SIZE - scaledW) / 2);
    const padY   = Math.floor((SAM_SIZE - scaledH) / 2);

    // Black background (already 0)
    oc.drawImage(source, padX, padY, scaledW, scaledH);
    const imageData = oc.getImageData(0, 0, SAM_SIZE, SAM_SIZE);
    const px = imageData.data;

    // Build Float32 CHW tensor [1, 3, SAM_SIZE, SAM_SIZE]
    const N = SAM_SIZE * SAM_SIZE;
    const tensor = new Float32Array(3 * N);

    for (let i = 0; i < N; i++) {
      // px layout: RGBA interleaved
      tensor[i]         = (px[i * 4]     - PIXEL_MEAN[0]) / PIXEL_STD[0]; // R
      tensor[N + i]     = (px[i * 4 + 1] - PIXEL_MEAN[1]) / PIXEL_STD[1]; // G
      tensor[2 * N + i] = (px[i * 4 + 2] - PIXEL_MEAN[2]) / PIXEL_STD[2]; // B
    }

    const t = new ort.Tensor('float32', tensor, [SAM_SIZE, SAM_SIZE, 3]);

    return {
      tensor: t,
      origW,  origH,
      padX,   padY,
      scaleX: scaledW / origW,
      scaleY: scaledH / origH,
    };
  }

  // ── Map image coordinates → SAM 1024 space ───────────────────────────────

  _toSAMCoords(x, y) {
    const { padX, padY, scaleX, scaleY } = this.embedding;
    return [
      x * scaleX + padX,
      y * scaleY + padY,
    ];
  }

  // ── Decode: point prompt ──────────────────────────────────────────────────

  /**
   * @param {Array<{x,y,label}>} points  label: 1=fg, 0=bg
   * @returns {{ mask: Uint8Array, width, height, score }}
   */
  async decodePoints(points) {
    this._requireEmbedding();

    const { origW, origH } = this.embedding;
    const n = points.length;

    // SAM decoder requires at least 1 padding point with label -1
    const totalPts = n + 1;
    const coords   = new Float32Array(totalPts * 2);
    const labels   = new Float32Array(totalPts);

    for (let i = 0; i < n; i++) {
      const [sx, sy]  = this._toSAMCoords(points[i].x, points[i].y);
      coords[i * 2]   = sx;
      coords[i * 2+1] = sy;
      labels[i]       = points[i].label;
    }
    // padding dummy point
    coords[n * 2]   = 0;
    coords[n * 2+1] = 0;
    labels[n]       = -1;

    return this._runDecoder(coords, labels, totalPts, origW, origH);
  }

  /**
   * @param {number} x1,y1,x2,y2  box in image coordinates
   */
  async decodeBox(x1, y1, x2, y2) {
    this._requireEmbedding();
    const { origW, origH } = this.embedding;

    const [sx1, sy1] = this._toSAMCoords(x1, y1);
    const [sx2, sy2] = this._toSAMCoords(x2, y2);

    // SAM box prompt: two corner points with labels 2 (top-left) and 3 (bottom-right)
    const coords = new Float32Array([sx1, sy1, sx2, sy2]);
    const labels = new Float32Array([2, 3]);

    return this._runDecoder(coords, labels, 2, origW, origH);
  }

  async _runDecoder(coords, labels, numPts, origW, origH) {
    const di = this._decIn;
    const mask256 = new Float32Array(1 * 1 * 256 * 256);   // empty mask input

    const feeds = {
      [di.embedding]:   this.embedding.tensor,
      [di.pointCoords]: new ort.Tensor('float32', coords,  [1, numPts, 2]),
      [di.pointLabels]: new ort.Tensor('float32', labels,  [1, numPts]),
      [di.maskInput]:   new ort.Tensor('float32', mask256, [1, 1, 256, 256]),
      [di.hasMask]:     new ort.Tensor('float32', new Float32Array([0]), [1]),
      [di.origSize]:    new ort.Tensor('float32', new Float32Array([origH, origW]), [2]),
    };

    // Remove undefined keys (some models omit certain optional inputs)
    for (const k of Object.keys(feeds)) {
      if (k === 'undefined' || !feeds[k]) delete feeds[k];
    }

    const results = await this.decoder.run(feeds);

    const iouData  = results[this._decOut.iou].data;
    const maskData = results[this._decOut.masks];

    // Pick the mask with the highest IoU score
    const bestIdx = iouData.indexOf(Math.max(...iouData));
    const mW      = maskData.dims[maskData.dims.length - 1];
    const mH      = maskData.dims[maskData.dims.length - 2];
    const offset  = bestIdx * mH * mW;
    const rawMask = maskData.data;

    // Threshold at 0 → binary uint8 (0 or 255)
    const binary = new Uint8Array(mH * mW);
    for (let i = 0; i < mH * mW; i++) {
      binary[i] = rawMask[offset + i] > 0 ? 255 : 0;
    }

    return {
      mask:   binary,
      width:  mW,
      height: mH,
      score:  iouData[bestIdx],
    };
  }

  _requireEmbedding() {
    if (!this.embedding) throw new Error('No image embedding — call encode() first');
    if (!this.decoder)   throw new Error('Decoder not loaded');
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  async _ensureORT() {
    if (typeof ort !== 'undefined') return;
    await new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/ort.min.js';
      s.onload = res; s.onerror = () => rej(new Error('Failed to load onnxruntime-web'));
      document.head.appendChild(s);
    });
  }

  get isReady() {
    return !!(this.encoder && this.decoder);
  }
}
