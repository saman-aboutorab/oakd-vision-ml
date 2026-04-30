"""P3 Traversability CNN — live camera demo.

Streams RGB + aligned depth from the OAK-D, runs TraversabilityNet on every
frame, and overlays a coloured 8×6 grid showing the traversability prediction
for each cell of the scene.

Colour coding:
    Green  (F) — free       — safe to drive on
    Yellow (C) — caution    — possible hazard, slow down
    Red    (O) — obstacle   — blocked, must stop/detour
    Grey   (U) — unknown    — depth missing or ambiguous

Usage:
    python scripts/live_traversability.py
    python scripts/live_traversability.py --checkpoint runs/fusion/attention/best.pt
    python scripts/live_traversability.py --checkpoint runs/fusion/concat/best.pt --no-depth

Controls (while window is open):
    Q / ESC   — quit
    S         — save current annotated frame as PNG
    D         — toggle depth panel (side-by-side vs RGB only)
    C         — toggle confidence labels on/off
"""

import argparse
import os
import time
from pathlib import Path

os.environ["QT_LOGGING_RULES"]  = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"]  = "ERROR"
os.environ["QT_QPA_FONTDIR"]    = "/usr/share/fonts"

import cv2
import numpy as np
import depthai as dai

from oakd_vision.fusion.inference import TraversabilityPredictor, CLASS_COLOURS_BGR
from oakd_vision.fusion.traversability_dataset import INT_TO_LABEL

SAVE_DIR = Path("runs/fusion/live_captures")


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    fps: float,
    saved: int,
    show_depth: bool,
    show_conf: bool,
    strategy: str,
):
    h, w = frame.shape[:2]
    bar_h = 28
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(frame,
                f"P3 Traversability | strategy={strategy} | "
                f"fps={fps:.1f} | saved={saved} | "
                f"[Q=quit  S=save  D=depth  C=conf]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def depth_to_colour(depth_mm: np.ndarray, max_mm: int = 4000) -> np.ndarray:
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm    = (clipped / max_mm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    coloured[depth_mm == 0] = (30, 30, 30)
    return coloured


# ---------------------------------------------------------------------------
# OAK-D pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera)
    cam.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)

    mono_left  = pipeline.create(dai.node.Camera)
    mono_right = pipeline.create(dai.node.Camera)
    mono_left.build(dai.CameraBoardSocket.CAM_B)
    mono_right.build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 480)

    mono_left.requestOutput((640, 480)).link(stereo.left)
    mono_right.requestOutput((640, 480)).link(stereo.right)

    rgb_q   = rgb_out.createOutputQueue(maxSize=2, blocking=False)
    depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

    return pipeline, rgb_q, depth_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    predictor = TraversabilityPredictor(
        checkpoint      = args.checkpoint,
        config          = args.config,
        fusion_strategy = args.strategy,
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting OAK-D pipeline…")
    pipeline, rgb_q, depth_q = build_pipeline()

    # Infer strategy name for HUD from checkpoint dir (e.g. "gated_f3")
    ckpt = Path(args.checkpoint)
    strategy_name = ckpt.parent.name

    win = "P3 Traversability"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 480)

    show_depth   = not args.no_depth
    show_conf    = True
    saved        = 0
    fps          = 0.0
    t_prev       = time.perf_counter()
    last_rgb     = None
    last_depth_mm = None
    last_display  = None

    with pipeline:
        print("Pipeline started. Press Q to quit.\n")

        while True:
            rgb_frame   = rgb_q.tryGet()
            depth_frame = depth_q.tryGet()

            if rgb_frame is not None:
                last_rgb = rgb_frame.getCvFrame()      # BGR uint8 (640×480)
            if depth_frame is not None:
                last_depth_mm = depth_frame.getFrame() # uint16 mm (640×480)

            if last_rgb is not None and last_depth_mm is not None:
                t_now = time.perf_counter()
                fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now

                # --- Inference ---
                grid_labels, grid_probs = predictor.predict(last_rgb, last_depth_mm)

                # --- Overlay on RGB ---
                annotated = predictor.draw_overlay(
                    last_rgb, grid_labels, grid_probs,
                    show_confidence=show_conf,
                )
                annotated = predictor.draw_legend(annotated)

                # --- Compose display ---
                if show_depth:
                    depth_vis = depth_to_colour(last_depth_mm)
                    display = np.hstack([annotated, depth_vis])
                else:
                    display = annotated

                draw_hud(display, fps, saved, show_depth, show_conf, strategy_name)
                last_display = display
                cv2.imshow(win, display)

            elif last_rgb is not None:
                # Show raw RGB while waiting for first depth frame
                waiting = last_rgb.copy()
                cv2.putText(waiting, "Waiting for depth...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow(win, waiting)

            elif last_display is not None:
                cv2.imshow(win, last_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                if last_display is not None:
                    fname = SAVE_DIR / f"live_{int(time.time())}.png"
                    cv2.imwrite(str(fname), last_display)
                    saved += 1
                    print(f"  Saved → {fname}")
            elif key in (ord('d'), ord('D')):
                show_depth = not show_depth
                cv2.resizeWindow(win, 1280 if show_depth else 640, 480)
            elif key in (ord('c'), ord('C')):
                show_conf = not show_conf

    cv2.destroyAllWindows()
    print(f"\nDone. {saved} frames saved to {SAVE_DIR.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint", default="runs/fusion/gated_f3/best.pt",
        help="Path to trained best.pt checkpoint",
    )
    parser.add_argument(
        "--config", default="training/configs/fusion_config.yaml",
        help="Path to fusion_config.yaml",
    )
    parser.add_argument(
        "--strategy", default=None,
        choices=["concat", "attention", "gated"],
        help="Override fusion strategy (default: inferred from config)",
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Show RGB-only display (depth still used by model)",
    )
    args = parser.parse_args()
    main(args)
