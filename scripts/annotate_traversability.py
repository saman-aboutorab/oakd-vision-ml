"""P3 traversability annotation tool.

Loads RGB frames saved by collect_traversability.py, overlays an 8×6 grid,
and lets you paint labels by clicking/dragging cells. Labels are saved as
JSON alongside each frame.

Labels and keyboard shortcuts:
    F — free      (green)   Nav2 cost 0
    C — caution   (yellow)  Nav2 cost 100
    O — obstacle  (red)     Nav2 cost 254
    U — unknown   (grey)    Nav2 cost 128

Navigation:
    ENTER / RIGHT — save current labels and go to next frame
    BACKSPACE / LEFT — go back one frame (labels already saved are kept)
    S — skip frame (no label file written, revisit later)
    Q — quit and save progress

Painting:
    Click a cell — paint it with the active label
    Click + drag — paint multiple cells in one stroke
    D — toggle depth overlay (colourmap blended on top of RGB)

Workflow tip: most frames are mostly floor → default is 'free'.
Just paint the exceptions (obstacles, caution zones, unknown background).

Usage:
    python scripts/annotate_traversability.py
    python scripts/annotate_traversability.py --start 50   # resume from frame 50
    python scripts/annotate_traversability.py --only-unlabeled  # skip already labeled
"""

import argparse
import json
import os
from pathlib import Path

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("dataset/traversability/raw")

GRID_COLS = 8
GRID_ROWS = 6

LABELS = ["free", "caution", "obstacle", "unknown"]
DEFAULT_LABEL = "free"

LABEL_COLOURS = {
    "free":     (0,   200,  0),    # green
    "caution":  (0,   200, 255),   # yellow
    "obstacle": (0,    40, 220),   # red
    "unknown":  (130, 130, 130),   # grey
}
LABEL_KEYS = {
    ord("f"): "free",
    ord("c"): "caution",
    ord("o"): "obstacle",
    ord("u"): "unknown",
}

CELL_ALPHA = 0.35          # label overlay transparency
DISPLAY_W  = 960           # display width (height scaled to maintain aspect ratio)
DEPTH_ALPHA = 0.45         # depth colourmap blend strength
DEPTH_MAX_MM = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(raw_dir: Path) -> list[Path]:
    """Return sorted list of RGB frame paths."""
    return sorted(raw_dir.glob("*_rgb.jpg"))


def load_labels(frame_path: Path) -> list[list[str]]:
    """Load existing labels for a frame, or return default grid (all free)."""
    label_path = label_path_for(frame_path)
    if label_path.exists():
        data = json.loads(label_path.read_text())
        return data["labels"]
    return [[DEFAULT_LABEL] * GRID_COLS for _ in range(GRID_ROWS)]


def save_labels(frame_path: Path, labels: list[list[str]]):
    label_path = label_path_for(frame_path)
    stem = frame_path.stem.replace("_rgb", "")
    data = {
        "frame": int(stem),
        "grid": [GRID_COLS, GRID_ROWS],
        "labels": labels,
    }
    label_path.write_text(json.dumps(data, indent=2))


def label_path_for(frame_path: Path) -> Path:
    stem = frame_path.stem.replace("_rgb", "")
    return frame_path.parent / f"{stem}_labels.json"


def is_labeled(frame_path: Path) -> bool:
    return label_path_for(frame_path).exists()


def load_depth_coloured(frame_path: Path) -> np.ndarray | None:
    stem = frame_path.stem.replace("_rgb", "")
    depth_path = frame_path.parent / f"{stem}_depth.npy"
    if not depth_path.exists():
        return None
    depth_mm = np.load(str(depth_path))
    clipped = np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.float32)
    norm = (clipped / DEPTH_MAX_MM * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    coloured[depth_mm == 0] = (20, 20, 20)
    return coloured


def pixel_to_cell(x: int, y: int, img_h: int, img_w: int):
    """Convert pixel coordinates to (row, col) grid cell. Returns None if out of bounds."""
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS
    col = int(x / cell_w)
    row = int(y / cell_h)
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return row, col
    return None


def draw_grid(img: np.ndarray, labels: list[list[str]], active_label: str,
              show_depth: bool, depth_coloured: np.ndarray | None) -> np.ndarray:
    h, w = img.shape[:2]
    display = img.copy()

    # Blend depth overlay if enabled
    if show_depth and depth_coloured is not None:
        depth_resized = cv2.resize(depth_coloured, (w, h))
        display = cv2.addWeighted(display, 1 - DEPTH_ALPHA, depth_resized, DEPTH_ALPHA, 0)

    # Paint cell colour overlays
    overlay = display.copy()
    cell_w = w / GRID_COLS
    cell_h = h / GRID_ROWS
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            label = labels[row][col]
            colour = LABEL_COLOURS[label]
            x1 = int(col * cell_w)
            y1 = int(row * cell_h)
            x2 = int((col + 1) * cell_w)
            y2 = int((row + 1) * cell_h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    display = cv2.addWeighted(display, 1 - CELL_ALPHA, overlay, CELL_ALPHA, 0)

    # Draw grid lines
    for col in range(1, GRID_COLS):
        x = int(col * cell_w)
        cv2.line(display, (x, 0), (x, h), (200, 200, 200), 1)
    for row in range(1, GRID_ROWS):
        y = int(row * cell_h)
        cv2.line(display, (0, y), (w, y), (200, 200, 200), 1)

    # Draw label text in each cell
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            label = labels[row][col]
            colour = LABEL_COLOURS[label]
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            initial = label[0].upper()
            cv2.putText(display, initial, (cx - 5, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    # HUD bar
    hud_h = 44
    hud = np.zeros((hud_h, w, 3), dtype=np.uint8)
    hud[:] = (25, 25, 25)

    # Active label indicator
    act_col = LABEL_COLOURS[active_label]
    cv2.rectangle(hud, (4, 4), (24, 40), act_col, -1)
    cv2.putText(hud, f"[{active_label[0].upper()}] {active_label}",
                (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, act_col, 1)

    # Key hints
    hints = "F=free  C=caution  O=obstacle  U=unknown  |  D=depth  |  ENTER=save+next  BACK=prev  S=skip  Q=quit"
    cv2.putText(hud, hints, (200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    return np.vstack([display, hud])


def count_labels(frames: list[Path]) -> tuple[int, int]:
    labeled = sum(1 for f in frames if is_labeled(f))
    return labeled, len(frames)


# ---------------------------------------------------------------------------
# Main annotator loop
# ---------------------------------------------------------------------------

def run(start_idx: int, only_unlabeled: bool):
    frames = load_frames(RAW_DIR)
    if not frames:
        print(f"No frames found in {RAW_DIR.resolve()}")
        print("Run collect_traversability.py first.")
        return

    labeled, total = count_labels(frames)
    print(f"\nP3 Annotator")
    print(f"Frames total  : {total}")
    print(f"Already labeled: {labeled} / {total}")
    print(f"Remaining     : {total - labeled}")
    print(f"Controls      : F/C/O/U=label  click/drag=paint  ENTER=save+next  BACK=prev  S=skip  D=depth  Q=quit\n")

    if only_unlabeled:
        frames = [f for f in frames if not is_labeled(f)]
        print(f"--only-unlabeled: showing {len(frames)} unlabeled frames\n")

    if not frames:
        print("All frames are already labeled!")
        return

    idx = min(start_idx, len(frames) - 1)

    win = "P3 Annotator — F/C/O/U + click to label — ENTER=next  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    active_label = DEFAULT_LABEL
    show_depth = False
    mouse_down = False
    labels = load_labels(frames[idx])
    depth_coloured = load_depth_coloured(frames[idx])

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_down, labels
        # Ignore clicks in HUD area (bottom 44px)
        img_h = param["img_h"]
        if y >= img_h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
        if event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
        if mouse_down and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            cell = pixel_to_cell(x, y, img_h, param["img_w"])
            if cell:
                row, col = cell
                labels[row][col] = active_label

    while 0 <= idx < len(frames):
        frame_path = frames[idx]
        bgr_raw = cv2.imread(str(frame_path))
        if bgr_raw is None:
            idx += 1
            continue

        # Scale to display width
        h_raw, w_raw = bgr_raw.shape[:2]
        display_h = int(h_raw * DISPLAY_W / w_raw)
        bgr = cv2.resize(bgr_raw, (DISPLAY_W, display_h))

        # Scale depth to match
        if depth_coloured is not None:
            depth_vis = cv2.resize(depth_coloured, (DISPLAY_W, display_h))
        else:
            depth_vis = None

        img_param = {"img_h": display_h, "img_w": DISPLAY_W}
        cv2.setMouseCallback(win, on_mouse, img_param)

        labeled_count, total_count = count_labels(frames if not only_unlabeled else
                                                   load_frames(RAW_DIR))
        print(f"  Frame {idx+1}/{len(frames)} — {frame_path.name}  "
              f"[labeled {labeled_count}/{total_count}]")

        redraw = True
        while True:
            if redraw:
                canvas = draw_grid(bgr, labels, active_label, show_depth, depth_vis)
                # Title bar with frame info
                title_bar = np.zeros((28, DISPLAY_W, 3), dtype=np.uint8)
                title_bar[:] = (40, 40, 40)
                stem = frame_path.stem.replace("_rgb", "")
                title_text = (f"Frame {idx+1}/{len(frames)}  |  {frame_path.name}  "
                              f"|  labeled={labeled_count}/{total_count}")
                cv2.putText(title_bar, title_text, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                canvas = np.vstack([title_bar, canvas])
                cv2.imshow(win, canvas)
                cv2.resizeWindow(win, DISPLAY_W, canvas.shape[0])
                redraw = False

            key = cv2.waitKey(30) & 0xFF

            if key in LABEL_KEYS:
                active_label = LABEL_KEYS[key]
                redraw = True

            elif key == ord("d"):
                show_depth = not show_depth
                redraw = True

            elif key == ord("s"):          # skip
                print(f"    Skipped.")
                idx += 1
                labels = load_labels(frames[idx]) if idx < len(frames) else None
                depth_coloured = load_depth_coloured(frames[idx]) if idx < len(frames) else None
                break

            elif key in (13, 83):          # ENTER or RIGHT arrow
                save_labels(frame_path, labels)
                print(f"    Saved → {label_path_for(frame_path).name}")
                idx += 1
                if idx < len(frames):
                    labels = load_labels(frames[idx])
                    depth_coloured = load_depth_coloured(frames[idx])
                break

            elif key in (8, 81):           # BACKSPACE or LEFT arrow
                idx = max(0, idx - 1)
                labels = load_labels(frames[idx])
                depth_coloured = load_depth_coloured(frames[idx])
                break

            elif key == ord("q"):
                print(f"\nQuitting. Progress saved.")
                cv2.destroyAllWindows()
                labeled, total = count_labels(load_frames(RAW_DIR))
                print(f"Labeled so far: {labeled} / {total}\n")
                return

            # Repaint when mouse is dragging
            if mouse_down:
                redraw = True

    cv2.destroyAllWindows()
    labeled, total = count_labels(load_frames(RAW_DIR))
    print(f"\nAll frames processed. Labeled: {labeled} / {total}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start",          type=int,  default=0,
                        help="Frame index to start from")
    parser.add_argument("--only-unlabeled", action="store_true",
                        help="Skip frames that already have a label file")
    args = parser.parse_args()
    run(args.start, args.only_unlabeled)
