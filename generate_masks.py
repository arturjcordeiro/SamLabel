"""
SAM Click-to-Segment Tool
=========================
Click on any object to segment it with SAM. Supports single images and
folder iteration — navigate through every image in a directory.

Controls
--------
  Left-click      Segment the object under the cursor
  S               Save current mask as binary uint8 PNG (0 / 255)
  R               Reset / clear the current mask
  D / Right       Next image     (folder mode only)
  A / Left        Previous image (folder mode only)
  Q / ESC         Quit

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
MASK_COLOR       = (0, 120, 255)   # BGR – orange tint for mask overlay
POINT_COLOR      = (0, 255, 80)    # BGR – green dot for click
ALPHA            = 0.45            # mask overlay blend strength
WINDOW_NAME      = "SAM Click to Segment"
WIN_MAX_W        = 1280
WIN_MAX_H        = 800


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
    """Return sorted list of image paths inside *folder*."""
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

def blend_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = MASK_COLOR
    return cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)


def draw_point(image: np.ndarray, x: int, y: int) -> np.ndarray:
    img = image.copy()
    cv2.circle(img, (x, y), 6, (0, 0, 0),   -1)
    cv2.circle(img, (x, y), 4, POINT_COLOR,  -1)
    return img


def draw_hud(image: np.ndarray, lines: list) -> np.ndarray:
    """Stamp (text, colour) lines anchored to the bottom of *image*."""
    img = image.copy()
    h   = img.shape[0]
    for i, (text, colour) in enumerate(reversed(lines)):
        y = h - 10 - i * 28
        # dark shadow for legibility on any background
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour,     1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Per-image interactive session
# ---------------------------------------------------------------------------

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

    nav_hint = "A/Left=Prev  D/Right=Next  |  " if folder_mode else ""
    progress = f"[{image_index + 1}/{image_total}]  {image_path.name}"

    state = {
        "mask":       None,
        "display":    None,
        "mask_count": 0,
    }

    def base_hud(top_line):
        return draw_hud(bgr_image, [
            (f"{nav_hint}S=Save  R=Reset  Q=Quit", (180, 180, 180)),
            (progress,                               (220, 220, 100)),
            (top_line,                               (255, 255, 255)),
        ])

    # ── mouse callback ───────────────────────────────────────────────────────
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"[INFO] Click ({x}, {y}) ...")

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        best_idx  = int(np.argmax(scores))
        best_mask = masks[best_idx]
        state["mask"] = best_mask

        vis = blend_mask(bgr_image, best_mask)
        vis = draw_point(vis, x, y)
        vis = draw_hud(vis, [
            (f"{nav_hint}S=Save  R=Reset  Q=Quit", (180, 180, 180)),
            (progress,                               (220, 220, 100)),
            (f"Score: {scores[best_idx]:.3f}   Pixels: {int(best_mask.sum())}",
             (255, 255, 255)),
        ])

        state["display"] = vis
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME,
                     min(bgr_image.shape[1], WIN_MAX_W),
                     min(bgr_image.shape[0], WIN_MAX_H))
        cv2.imshow(WINDOW_NAME, vis)

    # ── show initial frame ───────────────────────────────────────────────────
    init_frame       = base_hud("Click an object to segment it")
    state["display"] = init_frame

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIN_MAX_W, WIN_MAX_H)
    cv2.imshow(WINDOW_NAME, init_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ── event loop ───────────────────────────────────────────────────────────
    while True:
        key = cv2.waitKey(20) & 0xFF

        # Save ----------------------------------------------------------------
        if key in (ord('s'), ord('S')):
            if state["mask"] is None:
                print("[WARN] No mask yet – click an object first.")
            else:
                state["mask_count"] += 1
                outname     = out_dir / f"{image_path.stem}.png"
                binary_mask = (state["mask"].astype(np.uint8)) * 255
                cv2.imwrite(str(outname), binary_mask)
                print(f"[SAVE] {outname}  "
                      f"shape={binary_mask.shape}  "
                      f"unique={np.unique(binary_mask).tolist()}")

                confirm = state["display"].copy()
                cv2.putText(confirm, f"Saved: {outname.name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 100), 2, cv2.LINE_AA)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME,
                     min(bgr_image.shape[1], WIN_MAX_W),
                     min(bgr_image.shape[0], WIN_MAX_H))
                cv2.imshow(WINDOW_NAME, confirm)
                cv2.waitKey(800)
                cv2.imshow(WINDOW_NAME, state["display"])

        # Reset ---------------------------------------------------------------
        elif key in (ord('r'), ord('R')):
            state["mask"]    = None
            state["display"] = init_frame
            cv2.imshow(WINDOW_NAME, init_frame)
            print("[INFO] Reset.")

        # Next image ----------------------------------------------------------
        elif folder_mode and key in (ord('d'), ord('D'), 83):  # 83 = Right arrow
            return "next"

        # Previous image ------------------------------------------------------
        elif folder_mode and key in (ord('a'), ord('A'), 81):  # 81 = Left arrow
            return "prev"

        # Quit ----------------------------------------------------------------
        elif key in (ord('q'), ord('Q'), 27):
            return "quit"

        # Window closed with x ------------------------------------------------
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    # ── device ───────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── load SAM ─────────────────────────────────────────────────────────────
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        sys.exit("[ERROR] Run:  pip install segment-anything")

    print(f"[INFO] Loading SAM ({args.model_type}) ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("[INFO] Model ready.")

    # ── build image list ─────────────────────────────────────────────────────
    if args.folder:
        image_paths = collect_images(args.folder)
        folder_mode = True
    else:
        p = Path(args.image)
        if not p.is_file():
            sys.exit(f"[ERROR] Image not found: {args.image}")
        image_paths = [p]
        folder_mode = False

    total = len(image_paths)
    print(f"[INFO] {total} image(s) queued.")

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── create window once (loading placeholder) ─────────────────────────────
    placeholder = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Loading model ...", (100, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIN_MAX_W, WIN_MAX_H)
    cv2.imshow(WINDOW_NAME, placeholder)
    cv2.waitKey(1)

    # ── iterate images ───────────────────────────────────────────────────────
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

        if action == "next":
            idx = min(idx + 1, total - 1)
        elif action == "prev":
            idx = max(idx - 1, 0)
        elif action == "quit":
            break

    print("[INFO] Done.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
