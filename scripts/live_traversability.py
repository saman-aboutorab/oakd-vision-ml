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
    python scripts/live_traversability.py --checkpoint runs/fusion/attention_f3/best.pt
    python scripts/live_traversability.py --no-depth

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

from oakd_vision.fusion.inference import TraversabilityPredictor
from oakd_vision.fusion.traversability_dataset import INT_TO_LABEL

SAVE_DIR = Path("runs/fusion/live_captures")


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def draw_hud(frame, fps, saved, show_depth, show_conf, strategy):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(frame,
                f"P3 Traversability | {strategy} | fps={fps:.1f} | saved={saved} | "
                f"[Q=quit  S=save  D=depth  C=conf]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def depth_to_colour(depth_mm, max_mm=4000):
    clipped  = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm     = (clipped / max_mm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    coloured[depth_mm == 0] = (30, 30, 30)
    return coloured


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
    strategy_name = Path(args.checkpoint).parent.name

    win = "P3 Traversability"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 480)

    show_depth    = not args.no_depth
    show_conf     = True
    saved         = 0
    fps           = 0.0
    t_prev        = time.perf_counter()
    last_rgb      = None
    last_depth_mm = None
    last_display  = None

    print("\nStarting OAK-D pipeline...")

    # Use the same pattern as collect_traversability.py (confirmed working):
    # all node + queue creation happens inside the with block.
    with dai.Pipeline(dai.Device()) as pipeline:
        device    = pipeline.getDefaultDevice()
        usb_speed = device.getUsbSpeed()
        print(f"USB speed: {usb_speed.name}")

        use_depth = usb_speed in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS)
        if not use_depth:
            print("WARNING: USB 2.0 — depth disabled. Model will run on zeroed depth.")

        # RGB
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((640, 480), fps=25)
        rgb_q   = rgb_out.createOutputQueue(maxSize=2, blocking=False)

        # Stereo depth (USB 3.0 only)
        depth_q = None
        if use_depth:
            cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setLeftRightCheck(True)
            stereo.setOutputSize(640, 480)
            cam_left.requestFullResolutionOutput().link(stereo.left)
            cam_right.requestFullResolutionOutput().link(stereo.right)
            depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()
        print("Pipeline started. Press Q to quit.\n")

        while pipeline.isRunning():
            # Grab latest RGB
            rgb_msg = rgb_q.tryGet()
            if rgb_msg is not None:
                last_rgb = rgb_msg.getCvFrame()

            # Grab latest depth
            if depth_q is not None:
                depth_msg = depth_q.tryGet()
                if depth_msg is not None:
                    last_depth_mm = depth_msg.getFrame()
            elif last_rgb is not None and last_depth_mm is None:
                # USB 2.0 fallback: zeroed depth so model still runs
                last_depth_mm = np.zeros((480, 640), dtype=np.uint16)

            # Display
            if last_rgb is not None and last_depth_mm is not None:
                t_now  = time.perf_counter()
                fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now

                grid_labels, grid_probs = predictor.predict(last_rgb, last_depth_mm)
                annotated = predictor.draw_overlay(last_rgb, grid_labels, grid_probs,
                                                   show_confidence=show_conf)
                annotated = predictor.draw_legend(annotated)

                if show_depth and use_depth:
                    display = np.hstack([annotated, depth_to_colour(last_depth_mm)])
                else:
                    display = annotated

                draw_hud(display, fps, saved, show_depth, show_conf, strategy_name)
                last_display = display
                cv2.imshow(win, display)

            elif last_rgb is not None:
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
                cv2.resizeWindow(win, 1280 if (show_depth and use_depth) else 640, 480)
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
        help="Override fusion strategy (default: inferred from checkpoint dir name)",
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Show RGB-only display (depth still used by model)",
    )
    args = parser.parse_args()
    main(args)
