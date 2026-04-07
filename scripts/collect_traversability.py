"""P3 traversability data collection script.

Streams RGB + aligned stereo depth from the OAK-D Lite (USB 3.0 required)
and saves synchronized frame pairs to disk for later annotation.

Each saved frame produces two files:
  dataset/traversability/raw/NNNN_rgb.jpg    — 640×480 BGR image
  dataset/traversability/raw/NNNN_depth.npy  — 640×480 uint16 depth map (mm)

A JSON sidecar is also written per frame:
  dataset/traversability/raw/NNNN_meta.json  — timestamp, run, usb_speed

Usage:
    python scripts/collect_traversability.py --run floor
    python scripts/collect_traversability.py --run obstacles
    python scripts/collect_traversability.py --run clutter
    python scripts/collect_traversability.py --run lighting

Controls (while window is open):
    SPACE   — save current frame manually
    A       — toggle auto-save (saves one frame every --interval seconds)
    Q       — quit

Tip: USB 3.0 (blue port) is required — depth needs the bandwidth.
     Check the terminal for "USB speed: SUPER" at startup.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np
import depthai as dai

SAVE_DIR = Path("dataset/traversability/raw")

# Depth colourmap parameters for display only (not saved)
DEPTH_MAX_MM = 5000    # clip depth display at 5m


def depth_to_colour(depth_mm: np.ndarray) -> np.ndarray:
    """Convert uint16 depth map (mm) to a coloured image for display."""
    clipped = np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.float32)
    norm = (clipped / DEPTH_MAX_MM * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    # Black out zero-depth pixels (no measurement)
    coloured[depth_mm == 0] = (30, 30, 30)
    return coloured


def draw_hud(display: np.ndarray, saved: int, auto: bool, run: str, interval: float):
    h, w = display.shape[:2]
    # Status bar background
    cv2.rectangle(display, (0, h - 38), (w, h), (20, 20, 20), -1)
    auto_str = f"AUTO:{interval:.1f}s" if auto else "AUTO:off"
    auto_col = (0, 220, 0) if auto else (120, 120, 120)
    cv2.putText(display, f"run={run}  saved={saved}  {auto_str}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, auto_col, 1)
    cv2.putText(display, "SPACE=save  A=auto  Q=quit",
                (w - 280, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def save_frame(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    run: str,
    usb_speed: str,
    counter: int,
) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{counter:05d}"

    # RGB — save as JPEG (good quality, small size)
    rgb_path = SAVE_DIR / f"{stem}_rgb.jpg"
    cv2.imwrite(str(rgb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Depth — save as numpy array (lossless uint16, millimetres)
    depth_path = SAVE_DIR / f"{stem}_depth.npy"
    np.save(str(depth_path), depth_mm)

    # Metadata sidecar
    meta = {
        "frame": counter,
        "run": run,
        "usb_speed": usb_speed,
        "timestamp": time.time(),
        "rgb_file": rgb_path.name,
        "depth_file": depth_path.name,
        "rgb_shape": list(bgr.shape),
        "depth_shape": list(depth_mm.shape),
        "depth_unit": "mm",
    }
    meta_path = SAVE_DIR / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return rgb_path


def count_existing() -> int:
    """Count already-saved frames so we continue numbering correctly."""
    if not SAVE_DIR.exists():
        return 0
    return len(list(SAVE_DIR.glob("*_rgb.jpg")))


def run_collector(run_name: str, interval: float):
    saved = count_existing()
    print(f"\nP3 Data Collector — run='{run_name}'")
    print(f"Save directory : {SAVE_DIR.resolve()}")
    print(f"Already saved  : {saved} frames")
    print(f"Auto-save rate : {interval}s (toggle with A)")
    print("Controls       : SPACE=save  A=toggle-auto  Q=quit\n")

    with dai.Pipeline(dai.Device()) as pipeline:
        device = pipeline.getDefaultDevice()
        usb_speed = device.getUsbSpeed()
        usb_name = usb_speed.name
        print(f"USB speed: {usb_name}")

        if usb_speed not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
            print("WARNING: USB 2.0 detected — depth disabled.")
            print("         Plug into the blue USB 3.0 port for depth data.\n")
            use_depth = False
        else:
            use_depth = True
            print("USB 3.0 — depth enabled.\n")

        # RGB camera
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((640, 480), fps=25)
        rgb_q = rgb_out.createOutputQueue(maxSize=2, blocking=False)

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
        print("Pipeline started.\n")

        win = "P3 Data Collector — SPACE=save  A=auto  Q=quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 480)   # side-by-side RGB + depth

        last_depth_mm = np.zeros((480, 640), dtype=np.uint16)
        last_display = np.zeros((480, 1280, 3), dtype=np.uint8)
        auto_save = False
        last_auto_t = time.perf_counter()

        while pipeline.isRunning():
            rgb_msg = rgb_q.tryGet()

            if rgb_msg is None:
                cv2.imshow(win, last_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            bgr = rgb_msg.getCvFrame()

            if depth_q is not None:
                depth_msg = depth_q.tryGet()
                if depth_msg is not None:
                    last_depth_mm = depth_msg.getFrame()

            # Build side-by-side display: RGB | depth colourmap
            depth_vis = depth_to_colour(last_depth_mm)
            side_by_side = np.concatenate([bgr, depth_vis], axis=1)   # [480, 1280, 3]

            # Auto-save trigger
            now = time.perf_counter()
            do_save = False
            if auto_save and (now - last_auto_t) >= interval:
                do_save = True
                last_auto_t = now

            if do_save:
                if use_depth:
                    path = save_frame(bgr, last_depth_mm, run_name, usb_name, saved)
                    saved += 1
                    print(f"  AUTO SAVED [{saved:05d}]  {path.name}")
                else:
                    print("  Skipped — no depth data (USB 2.0)")

                # Flash green border
                cv2.rectangle(side_by_side, (2, 2), (1277, 477), (0, 220, 0), 4)

            draw_hud(side_by_side, saved, auto_save, run_name, interval)
            last_display = side_by_side
            cv2.imshow(win, side_by_side)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if use_depth:
                    path = save_frame(bgr, last_depth_mm, run_name, usb_name, saved)
                    saved += 1
                    print(f"  SAVED [{saved:05d}]  {path.name}")
                    # Flash
                    flash = side_by_side.copy()
                    cv2.rectangle(flash, (2, 2), (1277, 477), (0, 220, 0), 6)
                    cv2.imshow(win, flash)
                    cv2.waitKey(80)
                else:
                    print("  Cannot save — no depth (USB 2.0)")
            elif key == ord("a"):
                auto_save = not auto_save
                last_auto_t = time.perf_counter()
                state = "ON" if auto_save else "OFF"
                print(f"  Auto-save {state}  (every {interval}s)")

    cv2.destroyAllWindows()
    print(f"\nDone. {saved} total frames in {SAVE_DIR.resolve()}")
    print("Next step: python scripts/annotate_traversability.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--run",
        default="floor",
        choices=["floor", "obstacles", "clutter", "lighting"],
        help="Which capture run — floor | obstacles | clutter | lighting",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Auto-save interval in seconds when auto-save is ON (press A)",
    )
    args = parser.parse_args()
    run_collector(args.run, args.interval)
