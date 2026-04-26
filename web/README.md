# SAM Labeler

Browser-based dataset labeling tool powered by Segment Anything Model (SAM).
Runs fully locally — no cloud, no data leaves your machine.

```
sam_labeler/
├── index.html            ← main UI
├── server.py             ← local server (auto-opens browser)
├── models/
│   ├── encoder.onnx      ← SAM image encoder  (place here)
│   └── decoder.onnx      ← SAM mask decoder   (place here)
└── static/
    ├── css/app.css
    └── js/
        ├── app.js        ← UI orchestration
        ├── sam.js        ← ONNX inference engine
        ├── renderer.js   ← canvas drawing
        └── exporter.js   ← PNG mask + JSON export
```

---

## 1. Get SAM ONNX models

Download the quantised ViT-B models (~26 MB total) from HuggingFace:

```
https://huggingface.co/datasets/Xenova/sam-vit-base/tree/main/onnx
```

Files needed:
- `encoder_model_quantized.onnx`  → rename to `encoder.onnx`
- `decoder_model_quantized.onnx`  → rename to `decoder.onnx`

Place both in the `models/` folder.

**Custom model paths** (ViT-L, ViT-H, or non-standard paths):
```bash
python server.py --encoder /path/to/sam_vit_h_encoder.onnx \
                 --decoder /path/to/sam_vit_h_decoder.onnx
```

---

## 2. Run the server

```bash
python server.py
```

The browser opens automatically at `http://localhost:5000`.

Options:
```
--port 8080          use a different port
--no-browser         don't auto-open browser
--encoder <path>     custom encoder path
--decoder <path>     custom decoder path
```

---

## 3. Labeling workflow

1. **Open images** — drag a folder onto the canvas, or click "Open Folder"
2. **Select a tool**:
   - **Click** — left-click any object → SAM segments it instantly
   - **BBox** — drag a box around an object → tighter segmentation
   - **Neg** — click areas to *exclude* from the mask
3. **Set a class label** in the right panel before clicking
4. **Navigate** with `A` / `D` or the Prev/Next buttons
5. **Export** with `S` or the Export button

---

## Export formats

| Format | Contents |
|--------|----------|
| **PNG Mask** | `masks/<stem>_mask.png` — merged uint8 (0/255)<br>`masks/per_annotation/<stem>_ann1_<label>.png` — one per annotation |
| **JSON** | `json/<stem>.json` — bbox, polygon (border points), area, label, IoU score |
| **Both** | PNG masks + JSON |

All files are packed into a `.zip` archive.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1` | Click tool |
| `2` | BBox tool |
| `3` | Neg point tool |
| `A` / `←` | Previous image |
| `D` / `→` | Next image |
| `R` | Clear all annotations for current image |
| `Del` | Delete selected annotation |
| `F` | Fit image to window |
| `+` / `-` | Zoom in / out |
| `S` | Export all to ZIP |

---

## Requirements

- Python 3.8+ (stdlib only — no pip install needed for the server)
- Modern browser (Chrome/Edge/Firefox) with WebAssembly support
