"""P3 traversability annotation tool.

Loads RGB frames saved by collect_traversability.py, overlays an 8×6 grid,
and lets you navigate and paint labels using the keyboard.

Labels:
    F — free      (green)   Nav2 cost 0
    C — caution   (yellow)  Nav2 cost 100
    O — obstacle  (red)     Nav2 cost 254
    U — unknown   (grey)    Nav2 cost 128

Cursor movement:
    Arrow keys          — move cursor one cell
    Shift + Arrow       — extend selection (paint a rectangle)

Painting:
    F / C / O / U       — paint selected cell(s) with that label, advance right
    R                   — paint entire current ROW with active label
    A                   — paint ALL cells with active label (all-free / all-unknown frames)
    Z                   — undo last paint action

Frame navigation:
    ENTER               — save labels and go to next frame
    BACKSPACE           — go back one frame
    S                   — skip frame (no label saved)
    Q                   — quit

Display:
    D                   — toggle depth colourmap overlay

Workflow tip: defaults all cells to 'free'. For a typical indoor frame:
  1. Press U, then R on rows 0–2 (top rows = unknown background)
  2. Press O/C on specific cells near obstacles
  3. Press ENTER — done in ~10 seconds per frame.

Usage:
    python scripts/annotate_traversability.py
    python scripts/annotate_traversability.py --start 50
    python scripts/annotate_traversability.py --only-unlabeled
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
    "free":     (0,   200,   0),
    "caution":  (0,   200, 255),
    "obstacle": (0,    40, 220),
    "unknown":  (130, 130, 130),
}
LABEL_KEYS = {
    ord("f"): "free",
    ord("c"): "caution",
    ord("o"): "obstacle",
    ord("u"): "unknown",
}

CELL_ALPHA  = 0.32
DISPLAY_W   = 960
DEPTH_ALPHA = 0.45
DEPTH_MAX_MM = 4000

# OpenCV special key codes
KEY_UP    = 82
KEY_DOWN  = 84
KEY_LEFT  = 81
KEY_RIGHT = 83
KEY_ENTER = 13
KEY_BACK  = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*_rgb.jpg"))


def load_labels(frame_path: Path) -> list[list[str]]:
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


def count_labels(frames: list[Path]) -> tuple[int, int]:
    labeled = sum(1 for f in frames if is_labeled(f))
    return labeled, len(frames)


def copy_labels(labels: list[list[str]]) -> list[list[str]]:
    return [row[:] for row in labels]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_frame(bgr: np.ndarray, labels: list[list[str]], active_label: str,
               cursor: tuple[int, int], sel_start: tuple[int, int] | None,
               show_depth: bool, depth_vis: np.ndarray | None,
               frame_info: str, labeled_count: int, total_count: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    display = bgr.copy()

    if show_depth and depth_vis is not None:
        display = cv2.addWeighted(display, 1 - DEPTH_ALPHA, depth_vis, DEPTH_ALPHA, 0)

    # Cell colour overlay
    overlay = display.copy()
    cell_w = w / GRID_COLS
    cell_h = h / GRID_ROWS
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            colour = LABEL_COLOURS[labels[row][col]]
            x1, y1 = int(col * cell_w), int(row * cell_h)
            x2, y2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    display = cv2.addWeighted(display, 1 - CELL_ALPHA, overlay, CELL_ALPHA, 0)

    # Grid lines
    for col in range(1, GRID_COLS):
        cv2.line(display, (int(col * cell_w), 0), (int(col * cell_w), h), (200, 200, 200), 1)
    for row in range(1, GRID_ROWS):
        cv2.line(display, (0, int(row * cell_h)), (w, int(row * cell_h)), (200, 200, 200), 1)

    # Label initials
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            colour = LABEL_COLOURS[labels[row][col]]
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            cv2.putText(display, labels[row][col][0].upper(),
                        (cx - 5, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    # Selection highlight
    if sel_start is not None:
        r0, c0 = min(cursor[0], sel_start[0]), min(cursor[1], sel_start[1])
        r1, c1 = max(cursor[0], sel_start[0]), max(cursor[1], sel_start[1])
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                x1, y1 = int(col * cell_w), int(row * cell_h)
                x2, y2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Cursor box
    cr, cc = cursor
    cx1, cy1 = int(cc * cell_w), int(cr * cell_h)
    cx2, cy2 = int((cc + 1) * cell_w), int((cr + 1) * cell_h)
    cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (255, 255, 255), 3)
    cv2.rectangle(display, (cx1 + 2, cy1 + 2), (cx2 - 2, cy2 - 2), (0, 0, 0), 1)

    # Title bar
    title_bar = np.zeros((28, w, 3), dtype=np.uint8)
    title_bar[:] = (40, 40, 40)
    cv2.putText(title_bar, f"{frame_info}  |  labeled={labeled_count}/{total_count}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # HUD bar
    hud = np.zeros((44, w, 3), dtype=np.uint8)
    hud[:] = (25, 25, 25)
    act_col = LABEL_COLOURS[active_label]
    cv2.rectangle(hud, (4, 6), (22, 38), act_col, -1)
    cv2.putText(hud, f"[{active_label[0].upper()}] {active_label}",
                (28, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, act_col, 1)
    hints = "Arrows=move  F/C/O/U=paint  R=row  A=all  Z=undo  D=depth  ENTER=save+next  BACK=prev  S=skip  DEL=delete  Q=quit"
    cv2.putText(hud, hints, (200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    return np.vstack([title_bar, display, hud])


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
    print(f"Frames total   : {total}")
    print(f"Already labeled: {labeled} / {total}")
    print(f"Remaining      : {total - labeled}")
    print(f"Controls       : Arrows=move  F/C/O/U=paint  R=row  A=all  Z=undo  ENTER=save+next  Q=quit\n")

    if only_unlabeled:
        frames = [f for f in frames if not is_labeled(f)]
        print(f"--only-unlabeled: {len(frames)} unlabeled frames\n")

    if not frames:
        print("All frames already labeled!")
        return

    idx = min(start_idx, len(frames) - 1)

    win = "P3 Annotator"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)

    def load_frame_data(i):
        fp = frames[i]
        raw = cv2.imread(str(fp))
        h_raw, w_raw = raw.shape[:2]
        display_h = int(h_raw * DISPLAY_W / w_raw)
        bgr = cv2.resize(raw, (DISPLAY_W, display_h))
        depth_col = load_depth_coloured(fp)
        depth_vis = cv2.resize(depth_col, (DISPLAY_W, display_h)) if depth_col is not None else None
        lbls = load_labels(fp)
        return fp, bgr, depth_vis, lbls

    frame_path, bgr, depth_vis, labels = load_frame_data(idx)
    undo_stack: list[list[list[str]]] = []

    cursor = (GRID_ROWS - 1, 0)   # start at bottom-left (closest floor)
    sel_start = None
    active_label = DEFAULT_LABEL
    show_depth = False
    all_frames = load_frames(RAW_DIR)

    while 0 <= idx < len(frames):
        labeled_count, total_count = count_labels(all_frames)
        frame_info = f"Frame {idx+1}/{len(frames)}  |  {frame_path.name}"
        print(f"  {frame_info}  [labeled {labeled_count}/{total_count}]")

        while True:
            canvas = draw_frame(bgr, labels, active_label, cursor, sel_start,
                                show_depth, depth_vis, frame_info,
                                labeled_count, total_count)
            cv2.imshow(win, canvas)
            cv2.resizeWindow(win, DISPLAY_W, canvas.shape[0])

            key = cv2.waitKey(0) & 0xFF   # wait indefinitely — no busy loop needed

            cr, cc = cursor

            # --- Movement ---
            if key == KEY_UP:
                if sel_start is None:
                    cursor = (max(0, cr - 1), cc)
                else:
                    cursor = (max(0, cr - 1), cc)
            elif key == KEY_DOWN:
                if sel_start is None:
                    cursor = (min(GRID_ROWS - 1, cr + 1), cc)
                else:
                    cursor = (min(GRID_ROWS - 1, cr + 1), cc)
            elif key == KEY_LEFT:
                cursor = (cr, max(0, cc - 1))
                if sel_start is None:
                    pass
            elif key == KEY_RIGHT and key not in LABEL_KEYS.values():
                cursor = (cr, min(GRID_COLS - 1, cc + 1))

            # Shift held = extend selection (OpenCV reports shift via flags but
            # we detect it by checking if sel_start should be set)
            # Simple approach: hold nothing to move, press label key to paint+move

            # --- Label painting ---
            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                active_label = label
                undo_stack.append(copy_labels(labels))
                if sel_start is not None:
                    # Paint selection rectangle
                    r0 = min(cr, sel_start[0]); r1 = max(cr, sel_start[0])
                    c0 = min(cc, sel_start[1]); c1 = max(cc, sel_start[1])
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            labels[r][c] = label
                    sel_start = None
                else:
                    labels[cr][cc] = label
                    # Advance cursor right (wrap to next row)
                    if cc < GRID_COLS - 1:
                        cursor = (cr, cc + 1)
                    elif cr < GRID_ROWS - 1:
                        cursor = (cr + 1, 0)

            # --- Row paint ---
            elif key == ord("r"):
                undo_stack.append(copy_labels(labels))
                for c in range(GRID_COLS):
                    labels[cr][c] = active_label
                sel_start = None
                # Move cursor down to next row
                cursor = (min(GRID_ROWS - 1, cr + 1), cc)

            # --- All cells ---
            elif key == ord("a"):
                undo_stack.append(copy_labels(labels))
                for r in range(GRID_ROWS):
                    for c in range(GRID_COLS):
                        labels[r][c] = active_label
                sel_start = None

            # --- Undo ---
            elif key == ord("z"):
                if undo_stack:
                    labels = undo_stack.pop()
                    print("    Undo.")

            # --- Selection mode toggle (X key) ---
            elif key == ord("x"):
                if sel_start is None:
                    sel_start = cursor
                    print(f"    Selection started at {cursor} — move cursor then press label key")
                else:
                    sel_start = None

            # --- Depth toggle ---
            elif key == ord("d"):
                show_depth = not show_depth

            # --- Skip ---
            elif key == ord("s"):
                print("    Skipped.")
                idx += 1
                if idx < len(frames):
                    frame_path, bgr, depth_vis, labels = load_frame_data(idx)
                    undo_stack.clear()
                    cursor = (GRID_ROWS - 1, 0)
                    sel_start = None
                break

            # --- Delete frame (duplicate / unusable) ---
            elif key == 255:   # DEL key
                stem = frame_path.stem.replace("_rgb", "")
                to_delete = [
                    frame_path,
                    frame_path.parent / f"{stem}_depth.npy",
                    frame_path.parent / f"{stem}_meta.json",
                    frame_path.parent / f"{stem}_labels.json",
                ]
                for p in to_delete:
                    if p.exists():
                        p.unlink()
                print(f"    DELETED {frame_path.name} and sidecar files.")
                frames.pop(idx)
                all_frames = load_frames(RAW_DIR)
                if idx >= len(frames):
                    idx = len(frames) - 1
                if frames:
                    frame_path, bgr, depth_vis, labels = load_frame_data(idx)
                    undo_stack.clear()
                    cursor = (GRID_ROWS - 1, 0)
                    sel_start = None
                break

            # --- Save + next ---
            elif key == KEY_ENTER:
                save_labels(frame_path, labels)
                print(f"    Saved → {label_path_for(frame_path).name}")
                idx += 1
                if idx < len(frames):
                    frame_path, bgr, depth_vis, labels = load_frame_data(idx)
                    undo_stack.clear()
                    cursor = (GRID_ROWS - 1, 0)
                    sel_start = None
                break

            # --- Previous ---
            elif key == KEY_BACK:
                idx = max(0, idx - 1)
                frame_path, bgr, depth_vis, labels = load_frame_data(idx)
                undo_stack.clear()
                cursor = (GRID_ROWS - 1, 0)
                sel_start = None
                break

            # --- Quit ---
            elif key == ord("q"):
                print(f"\nQuitting.")
                cv2.destroyAllWindows()
                labeled, total = count_labels(all_frames)
                print(f"Labeled so far: {labeled} / {total}\n")
                return

    cv2.destroyAllWindows()
    labeled, total = count_labels(all_frames)
    print(f"\nAll done. Labeled: {labeled} / {total}")


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
