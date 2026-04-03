"""Interactive dataset capture script for a single object class.

Connects to the OAK-D Lite, streams RGB + aligned stereo depth, and saves
frame pairs (JPG + depth NPY) on SPACE press.

Run once per class:
    python scripts/capture_dataset.py shoe
    python scripts/capture_dataset.py mug

Saved to:
    dataset/raw/<class_name>/<class_name>_0000.jpg
    dataset/raw/<class_name>/<class_name>_0000_depth.npy

Requires USB 3.0 (SS/blue port). Run `lsusb` and confirm bcdUSB 3.x.

--- depthai v3 API ---
- `dai.Pipeline(dai.Device())` opens the device then wraps it in a pipeline
- Use new `Camera` node with `.build(socket)` + `.requestOutput(capability)`
- Output queues via `.createOutputQueue()` directly on node outputs
- Stereo fed via `cam.requestFullResolutionOutput().link(stereo.left/right)`
- StereoDepth preset renamed: HIGH_DENSITY → DENSITY
"""

import sys
import warnings
from pathlib import Path

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"  # suppress Qt font warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
import depthai as dai

# --- Config ---
CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "object"
SAVE_DIR = Path(f"dataset/raw/{CLASS_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
counter = len(list(SAVE_DIR.glob("*.jpg")))

print(f"Class: '{CLASS_NAME}'  |  {counter} frames already saved in {SAVE_DIR}")
print("Connecting to OAK-D...")

# --- Pipeline (v3: pass Device to Pipeline so we can call .build() on Camera nodes) ---
with dai.Pipeline(dai.Device()) as pipeline:

    device = pipeline.getDefaultDevice()
    print(f"USB speed: {device.getUsbSpeed().name}")

    # RGB camera — v3 Camera node
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((640, 640), fps=25)

    # Mono cameras for stereo
    cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # StereoDepth aligned to RGB
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setOutputSize(640, 640)  # must match RGB output size and be a multiple of 16
    cam_left.requestFullResolutionOutput().link(stereo.left)
    cam_right.requestFullResolutionOutput().link(stereo.right)

    # Output queues
    rgb_q   = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    print("Pipeline started — camera window opening...")
    print("SPACE = save  |  Q = quit")
    print("Vary: near (0.3-0.7m), mid (0.7-1.5m), far (1.5-2.5m), angles, lighting\n")

    # Show placeholder — remind user to click the window before pressing SPACE
    placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Starting camera...", (110, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(placeholder, "CLICK THIS WINDOW FIRST", (80, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    cv2.putText(placeholder, "then SPACE to save, Q to quit", (70, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow("OAK-D Capture", placeholder)
    cv2.waitKey(1)

    last_depth_mm = None  # cache last valid depth so display doesn't blink

    while pipeline.isRunning():
        rgb_msg   = rgb_q.tryGet()
        depth_msg = depth_q.tryGet()

        if rgb_msg is None:
            cv2.waitKey(30)
            continue

        bgr = rgb_msg.getCvFrame()
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)  # fix camera orientation

        if depth_msg is not None:
            raw = depth_msg.getFrame()
            last_depth_mm = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        depth_mm = last_depth_mm

        display = bgr.copy()
        cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Near 0.3-0.7m", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(display, "Mid  0.7-1.5m", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(display, "Far  1.5-2.5m", (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1)
        cx, cy = bgr.shape[1] // 2, bgr.shape[0] // 2
        depth_status = f"depth: {int(depth_mm[cy, cx])}mm" if depth_mm is not None else "depth: --"
        cv2.putText(display, depth_status, (8, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(display, "CLICK WINDOW THEN: SPACE=save  Q=quit", (8, 630),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.imshow("OAK-D Capture", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(" ") or key == 32:
            img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
            cv2.imwrite(str(img_path), bgr)
            saved_msg = f"  SAVED [{counter:04d}]  {img_path.name}"
            if depth_mm is not None:
                dep_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}_depth.npy"
                np.save(str(dep_path), depth_mm)
                saved_msg += " + depth"
            print(saved_msg)
            counter += 1
            # Flash green border on screen so you know it saved without looking at terminal
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1]-1, flash.shape[0]-1), (0, 255, 0), 12)
            cv2.putText(flash, "SAVED!", (flash.shape[1]//2 - 60, flash.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("OAK-D Capture", flash)
            cv2.waitKey(200)  # show flash for 200ms
        elif key == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
