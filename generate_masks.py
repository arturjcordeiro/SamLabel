"""
SAM Click-to-Segment Tool  (multi-mask edition)
================================================
Click on any object to segment it with SAM. Supports single images and
folder iteration — navigate through every image in a directory.

Multiple masks per image are saved inside a per-image subfolder.

Controls
--------
  Left-click              Segment the object under the cursor (NEW mask)
  Shift + Left-click      ADD clicked region to the current mask
  Ctrl  + Left-click      REMOVE clicked region from the current mask

  S                       Save the current mask
  N                       New empty mask (start a fresh one)
  Tab / PageDown          Next saved mask
  PageUp                  Previous saved mask
  X                       Delete the current saved mask

  D / Right               Next image     (folder mode only)
  A / Left                Previous image (folder mode only)
  Q / ESC                 Quit

Usage
-----
  # Single image
  python sam_click_segment.py --image path/to/image.jpg --checkpoint sam_vit_h_4b8939.pth

  # Folder of images
  python sam_click_segment.py --folder path/to/images/ --checkpoint sam_vit_h_4b8939.pth

  # Faster model
  python sam_click_segment.py --folder path/to/images/ --checkpoint sam_vit_b_01ec64.pth --model-type vit_b

SAM checkpoint download (pick one):
  ViT-H (best):    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ViT-L:           https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  ViT-B (fastest): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Requirements
------------
  pip install segment-anything opencv-python numpy torch torchvision
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Distinct BGR colours for each saved mask slot (cycles)
MASK_COLOURS = [
    (0,  120, 255),   # orange
    (255, 60,  60),   # blue-ish red
    (60, 220,  60),   # green
    (220, 60, 220),   # magenta
    (60, 220, 220),   # yellow-ish
    (255, 180,  0),   # sky blue
    (0,  180, 255),   # warm yellow
]

ACTIVE_MASK_COLOUR = (0, 255, 200)  # teal — mask being edited
POINT_ADD_COLOUR   = (0, 255, 80)   # green  — add-mode click
POINT_REMOVE_COLOUR= (0, 60, 255)   # red    — remove-mode click
POINT_NEW_COLOUR   = (0, 200, 255)  # yellow — new-mask click
ALPHA              = 0.45
WINDOW_NAME        = "SAM Click to Segment"
WIN_MAX_W          = 1280
WIN_MAX_H          = 800


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM interactive click-to-segment tool")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  help="Path to a single input image")
    src.add_argument("--folder", help="Path to a folder of images to iterate through")

    parser.add_argument("--checkpoint", required=True,
                        help="Path to SAM model checkpoint (.pth)")
    parser.add_argument("--model-type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model variant (default: vit_h)")
    parser.add_argument("--output-dir", default="masks",
                        help="Directory to save masks (default: ./masks)")
    parser.add_argument("--device", default=None,
                        help="Compute device: cuda / cpu (auto-detected if omitted)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image list helpers
# ---------------------------------------------------------------------------

def collect_images(folder: str) -> list:
    p = Path(folder)
    if not p.is_dir():
        sys.exit(f"[ERROR] Folder not found: {folder}")
    images = sorted(
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        sys.exit(f"[ERROR] No supported images found in: {folder}")
    return images


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def blend_mask(image: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = colour
    return cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)


def draw_point(image: np.ndarray, x: int, y: int, colour, mode_char="") -> np.ndarray:
    img = image.copy()
    cv2.circle(img, (x, y), 8, (0, 0, 0), -1)
    cv2.circle(img, (x, y), 6, colour,    -1)
    if mode_char:
        cv2.putText(img, mode_char, (x + 9, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return img


def put_text_shadowed(img, text, pos, scale, colour, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour,    thickness,     cv2.LINE_AA)


def draw_hud(image: np.ndarray, lines: list) -> np.ndarray:
    """Stamp (text, colour) lines at the bottom."""
    img = image.copy()
    h = img.shape[0]
    for i, (text, colour) in enumerate(reversed(lines)):
        y = h - 10 - i * 26
        put_text_shadowed(img, text, (10, y), 0.55, colour)
    return img


def draw_mask_strip(image: np.ndarray, saved_masks: list, active_idx: int,
                    strip_h: int = 70) -> np.ndarray:
    """
    Render a thumbnail strip of all saved masks at the top of *image*.
    The active one is highlighted with a bright border.
    """
    h, w = image.shape[:2]
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)

    n = len(saved_masks)
    if n == 0:
        put_text_shadowed(strip, "No masks saved yet.  Press S to save.",
                          (10, strip_h // 2 + 6), 0.50, (160, 160, 160))
        return np.vstack([strip, image])

    thumb_w = min(80, (w - 10) // n - 4)
    thumb_h = strip_h - 10

    for i, (mask, colour) in enumerate(saved_masks):
        x0 = 5 + i * (thumb_w + 4)
        # build mini composite
        thumb_bg  = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        # scale mask down
        small_mask = cv2.resize(mask.astype(np.uint8),
                                (thumb_w, thumb_h),
                                interpolation=cv2.INTER_NEAREST)
        thumb_bg[small_mask > 0] = colour
        thumb_bg = cv2.addWeighted(thumb_bg, 0.7,
                                   np.full_like(thumb_bg, 40), 0.3, 0)

        # border
        border_col = (255, 255, 0) if i == active_idx else (80, 80, 80)
        cv2.rectangle(thumb_bg, (0, 0), (thumb_w - 1, thumb_h - 1),
                      border_col, 2 if i == active_idx else 1)

        # index label
        put_text_shadowed(thumb_bg, str(i + 1), (4, thumb_h - 4), 0.45,
                          (255, 255, 255))

        strip[5:5 + thumb_h, x0:x0 + thumb_w] = thumb_bg

    # hint on right side
    put_text_shadowed(strip,
                      "Tab/PgDn=next  PgUp=prev  X=delete  N=new",
                      (5, strip_h - 4), 0.40, (140, 140, 140))

    return np.vstack([strip, image])


# ---------------------------------------------------------------------------
# Per-image interactive session
# ---------------------------------------------------------------------------

class ImageSession:
    """
    All mutable state for one image's editing session.

    saved_masks  : list of (bool ndarray H×W, BGR colour)
    active_mask  : current mask being built (bool ndarray H×W), or None
    active_idx   : which saved mask is "selected" for viewing (-1 = none)
    points       : list of (x, y, label, colour, char) for the active mask
    """
    def __init__(self):
        self.saved_masks = []   # [(mask_array, colour), ...]
        self.active_mask = None # current unsaved working mask
        self.active_idx  = -1  # index of highlighted saved mask
        self.points      = []   # clicks on the working mask


def predict_from_points(predictor, points):
    """Run SAM inference from accumulated (x, y, label) points."""
    if not points:
        return None
    coords  = np.array([[p[0], p[1]] for p in points])
    labels  = np.array([p[2]         for p in points])
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx], scores[best_idx]


def compose_display(bgr_image, session, strip_h=70):
    """Build the full display frame from scratch."""
    vis = bgr_image.copy()

    # 1) Draw all saved masks (dimmed)
    for i, (mask, colour) in enumerate(session.saved_masks):
        dimmed = tuple(int(c * 0.55) for c in colour)
        overlay = vis.copy()
        overlay[mask > 0] = dimmed
        vis = cv2.addWeighted(overlay, ALPHA * 0.8, vis, 1 - ALPHA * 0.8, 0)

    # 2) Draw active mask on top (bright)
    if session.active_mask is not None:
        overlay = vis.copy()
        overlay[session.active_mask > 0] = ACTIVE_MASK_COLOUR
        vis = cv2.addWeighted(overlay, ALPHA, vis, 1 - ALPHA, 0)

    # 3) Draw click points
    for (px, py, lbl, col, ch) in session.points:
        vis = draw_point(vis, px, py, col, ch)

    return vis


def run_image(predictor, bgr_image, image_path, out_dir,
              image_index, image_total, folder_mode):
    """
    Interactive loop for one image.
    Returns one of: "next" | "prev" | "quit"
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(f"[INFO] Encoding [{image_index + 1}/{image_total}] {image_path.name} ...")
    predictor.set_image(rgb_image)
    print("[INFO] Ready.")

    nav_hint  = "A/←=Prev  D/→=Next  |  " if folder_mode else ""
    progress  = f"[{image_index + 1}/{image_total}]  {image_path.name}"
    session   = ImageSession()
    strip_h   = 70

    # ── helpers ─────────────────────────────────────────────────────────────

    def next_colour():
        return MASK_COLOURS[len(session.saved_masks) % len(MASK_COLOURS)]

    def hud_lines(status_text):
        return [
            (f"{nav_hint}S=Save  N=New  Tab/PgDn=NextMask  PgUp=PrevMask  X=Delete  Q=Quit",
             (160, 160, 160)),
            (progress, (220, 220, 100)),
            (status_text, (255, 255, 255)),
        ]

    def refresh(status="Click: new  Shift+Click: add  Ctrl+Click: remove"):
        vis = compose_display(bgr_image, session)
        vis = draw_hud(vis, hud_lines(status))
        combined = draw_mask_strip(vis, session.saved_masks,
                                   session.active_idx, strip_h)
        cv2.imshow(WINDOW_NAME, combined)

    def save_active_mask():
        """Save current active mask into session.saved_masks."""
        if session.active_mask is None or not session.active_mask.any():
            print("[WARN] Nothing to save.")
            return
        colour  = next_colour()
        session.saved_masks.append((session.active_mask.copy(), colour))
        session.active_idx = len(session.saved_masks) - 1
        # persist to disk
        flush_mask_to_disk(len(session.saved_masks) - 1)
        print(f"[SAVE] Mask {len(session.saved_masks)} saved.")
        # clear working state
        session.active_mask = None
        session.points      = []

    def flush_mask_to_disk(slot_idx):
        mask, _ = session.saved_masks[slot_idx]
        img_subdir = out_dir / image_path.stem
        img_subdir.mkdir(parents=True, exist_ok=True)
        outname = img_subdir / f"mask_{slot_idx + 1:03d}.png"
        binary  = mask.astype(np.uint8) * 255
        cv2.imwrite(str(outname), binary)
        print(f"[DISK] {outname}  pixels={int(mask.sum())}")

    def delete_active_saved():
        if not session.saved_masks:
            print("[WARN] No saved masks to delete.")
            return
        idx = max(session.active_idx, 0)
        session.saved_masks.pop(idx)
        # also delete from disk
        img_subdir = out_dir / image_path.stem
        # re-write remaining masks with correct indices
        # first remove all existing numbered masks
        for f in img_subdir.glob("mask_*.png"):
            f.unlink()
        for i, (m, _) in enumerate(session.saved_masks):
            outname = img_subdir / f"mask_{i + 1:03d}.png"
            cv2.imwrite(str(outname), m.astype(np.uint8) * 255)
        session.active_idx = max(0, idx - 1) if session.saved_masks else -1
        print(f"[DEL] Mask {idx + 1} deleted. {len(session.saved_masks)} remaining.")

    # ── mouse callback ───────────────────────────────────────────────────────

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Offset y because of the strip at the top
        y_img = y - strip_h
        if y_img < 0:
            return  # click was in the thumbnail strip

        shift_held = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        ctrl_held  = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        if shift_held:
            # ADD to existing mask
            label = 1
            mode  = "add"
            col   = POINT_ADD_COLOUR
            ch    = "+"
        elif ctrl_held:
            # REMOVE from existing mask
            label = 0
            mode  = "remove"
            col   = POINT_REMOVE_COLOUR
            ch    = "−"
        else:
            # NEW mask — clear working state first
            session.active_mask = None
            session.points      = []
            label = 1
            mode  = "new"
            col   = POINT_NEW_COLOUR
            ch    = ""

        session.points.append((x_img := x, y_img, label, col, ch))
        print(f"[INFO] {mode.upper()} click ({x_img}, {y_img}) ...")

        result = predict_from_points(predictor, session.points)
        if result is None:
            return
        mask, score = result
        session.active_mask = mask

        refresh(f"[{mode.upper()}] Score:{score:.3f}  Pixels:{int(mask.sum())}  "
                f"| S=Save  N=New mask")

    # ── initial display ──────────────────────────────────────────────────────

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,
                     min(bgr_image.shape[1], WIN_MAX_W),
                     min(bgr_image.shape[0] + strip_h, WIN_MAX_H))
    refresh("Click an object to segment it")
    cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ── keyboard event loop ──────────────────────────────────────────────────
    # IMPORTANT: do NOT mask with & 0xFF — that destroys special key codes.
    # cv2.waitKey returns -1 when no key is pressed, otherwise a 32-bit int.
    # Special keys on Linux (X11): Delete=65535, PageUp=65365, PageDown=65366
    #                               Left=65361, Right=65363, Tab=9, BackSpace=65288
    # On Windows they differ; we cover both by checking multiple values.

    while True:
        raw = cv2.waitKey(20)
        if raw == -1:
            # no key — only check window-close
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            continue

        key8  = raw & 0xFF          # lower 8 bits  — reliable for ASCII keys
        key32 = raw & 0xFFFF        # lower 16 bits — covers most special keys

        # Uncomment the next line temporarily if a key isn't working:
        # print(f"[DEBUG] raw={raw}  key8={key8}  key32={key32}  char={chr(key8) if 32<=key8<127 else '?'}")

        # ---- Save current working mask ----
        if key8 in (ord('s'), ord('S')):
            save_active_mask()
            refresh()

        # ---- New empty mask ----
        elif key8 in (ord('n'), ord('N')):
            session.active_mask = None
            session.points      = []
            session.active_idx  = -1
            print("[INFO] New mask started.")
            refresh("New mask – click to segment")

        # ---- Cycle forward through saved masks ----
        # Tab=9, PageDown=65366 (Linux), 34 (Windows)
        elif key8 == 9 or key32 in (65366, 34):
            if session.saved_masks:
                session.active_idx = (session.active_idx + 1) % len(session.saved_masks)
                print(f"[INFO] Viewing mask {session.active_idx + 1}/{len(session.saved_masks)}")
                refresh(f"Mask {session.active_idx + 1} of {len(session.saved_masks)}")

        # ---- Cycle backward through saved masks ----
        # PageUp=65365 (Linux), 33 (Windows)
        elif key32 in (65365, 33):
            if session.saved_masks:
                session.active_idx = (session.active_idx - 1) % len(session.saved_masks)
                print(f"[INFO] Viewing mask {session.active_idx + 1}/{len(session.saved_masks)}")
                refresh(f"Mask {session.active_idx + 1} of {len(session.saved_masks)}")

        # ---- Delete active saved mask ----
        # X key (safe, unambiguous), plus Delete=65535/65439 (Linux), BackSpace=65288/8
        elif key8 in (ord('x'), ord('X')) or key32 in (65535, 65439, 65288, 8):
            delete_active_saved()
            refresh()

        # ---- Next image ----
        # D key, or Right arrow=65363 (Linux), 83 (Windows numpad)
        elif folder_mode and (key8 in (ord('d'), ord('D')) or key32 in (65363, 83)):
            return "next"

        # ---- Previous image ----
        # A key, or Left arrow=65361 (Linux), 81 (Windows numpad)
        elif folder_mode and (key8 in (ord('a'), ord('A')) or key32 in (65361, 81)):
            return "prev"

        # ---- Quit ----
        elif key8 in (ord('q'), ord('Q'), 27):
            return "quit"

        # ---- Window closed ----
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        sys.exit("[ERROR] Run:  pip install segment-anything")

    print(f"[INFO] Loading SAM ({args.model_type}) ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("[INFO] Model ready.")

    if args.folder:
        image_paths = collect_images(args.folder)
        folder_mode = True
    else:
        p = Path(args.image)
        if not p.is_file():
            sys.exit(f"[ERROR] Image not found: {args.image}")
        image_paths = [p]
        folder_mode = False

    total   = len(image_paths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {total} image(s) queued.  Output → {out_dir}")

    # Loading placeholder
    placeholder = np.zeros((220, 520, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Loading model ...", (90, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIN_MAX_W, WIN_MAX_H)
    cv2.imshow(WINDOW_NAME, placeholder)
    cv2.waitKey(1)

    idx = 0
    while 0 <= idx < total:
        image_path = image_paths[idx]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[WARN] Cannot read {image_path.name}, skipping.")
            idx += 1
            continue

        action = run_image(
            predictor   = predictor,
            bgr_image   = bgr,
            image_path  = image_path,
            out_dir     = out_dir,
            image_index = idx,
            image_total = total,
            folder_mode = folder_mode,
        )

        if   action == "next": idx = min(idx + 1, total - 1)
        elif action == "prev": idx = max(idx - 1, 0)
        elif action == "quit": break

    print("[INFO] Done.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
