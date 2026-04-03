"""Interactive dataset capture script for a single object class.

Connects to the OAK-D Lite, streams RGB + aligned stereo depth, and saves
frame pairs (JPG + depth NPY) on SPACE press.

Run once per class:
    python scripts/capture_dataset.py shoe
    python scripts/capture_dataset.py mug

Saved to:
    dataset/raw/<class_name>/<class_name>_0000.jpg
    dataset/raw/<class_name>/<class_name>_0000_depth.npy   (if depth available)

USB speed requirements:
    USB 3.0 (SS port, blue or labelled SS) — full mode: RGB + aligned depth
    USB 2.0                                 — fallback: RGB-only (still fine for YOLO training)

The script auto-detects USB speed and falls back gracefully.

--- depthai v3 API notes ---
- `dai.Pipeline()` opens the device internally; do NOT pass `dai.Device()`
- ColorCamera / MonoCamera are deprecated but still work in v3
- Output queues via `.createOutputQueue()` on node outputs — no XLinkOut node
- StereoDepth preset renamed: HIGH_DENSITY → DENSITY, HIGH_ACCURACY → ACCURACY
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
import depthai as dai

# --- Config ---
CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "object"
SAVE_DIR = Path(f"dataset/raw/{CLASS_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

counter = len(list(SAVE_DIR.glob("*.jpg")))


def check_usb_speed(pipeline) -> str:
    """Return USB speed name, e.g. 'SUPER' (USB 3) or 'HIGH' (USB 2)."""
    try:
        device = pipeline.getDefaultDevice()
        return device.getUsbSpeed().name
    except Exception:
        return "UNKNOWN"


def build_pipeline_full():
    """RGB + aligned stereo depth — requires USB 3.0."""
    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    rgb_q   = cam.preview.createOutputQueue(maxSize=4, blocking=False)
    depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    return p, rgb_q, depth_q


def build_pipeline_rgb_only():
    """RGB-only — works on USB 2.0. No depth saved, still fine for YOLO training."""
    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(15)  # lower FPS reduces USB bandwidth

    rgb_q = cam.preview.createOutputQueue(maxSize=4, blocking=False)
    return p, rgb_q, None


# --- Try full pipeline first, fall back to RGB-only ---
print(f"Class: '{CLASS_NAME}'  |  {counter} frames already saved in {SAVE_DIR}")
print("Detecting USB speed...")

try:
    pipeline, rgb_q, depth_q = build_pipeline_full()
    pipeline.start()
    usb = check_usb_speed(pipeline)
    if usb not in ("SUPER", "SUPER_PLUS", "SUPER_PLUS_PLUS"):
        print(f"USB speed: {usb} — not USB 3.0, restarting in RGB-only mode")
        pipeline.stop()
        pipeline, rgb_q, depth_q = build_pipeline_rgb_only()
        pipeline.start()
        mode = "RGB-only (USB 2.0 fallback — depth not saved)"
    else:
        mode = f"Full RGB + depth  (USB {usb})"
except Exception as e:
    print(f"Full pipeline failed ({e}), falling back to RGB-only")
    pipeline, rgb_q, depth_q = build_pipeline_rgb_only()
    pipeline.start()
    mode = "RGB-only (fallback)"

print(f"Mode: {mode}")
print("SPACE = save  |  Q = quit")
print("Vary: near (0.3-0.7m), mid (0.7-1.5m), far (1.5-2.5m), angles, lighting\n")

# Show a placeholder window immediately so it's visible before first frame arrives
placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.putText(placeholder, "Starting camera...", (160, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imshow("OAK-D Capture", placeholder)
cv2.waitKey(1)

# --- Capture loop ---
while pipeline.isRunning():
    rgb_msg = rgb_q.tryGet()   # non-blocking: returns None immediately if no frame yet
    if rgb_msg is None:
        cv2.waitKey(1)
        continue

    bgr = rgb_msg.getCvFrame()

    depth_mm = None
    if depth_q is not None:
        depth_msg = depth_q.tryGet()
        if depth_msg is not None:
            depth_mm = depth_msg.getFrame()

    display = bgr.copy()
    cv2.putText(display, f"[{mode[:30]}]", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(display, "Near 0.3-0.7m", (8, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    cv2.putText(display, "Mid  0.7-1.5m", (8, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(display, "Far  1.5-2.5m", (8, 99),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1)
    cv2.putText(display, "SPACE=save  Q=quit", (8, 625),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.imshow("OAK-D Capture", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
        cv2.imwrite(str(img_path), bgr)
        if depth_mm is not None:
            dep_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}_depth.npy"
            np.save(str(dep_path), depth_mm)
        print(f"Saved {img_path.name}" + (" + depth" if depth_mm is not None else " (no depth)"))
        counter += 1
    elif key == ord("q"):
        pipeline.stop()
        break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
