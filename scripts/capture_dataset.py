"""Interactive dataset capture script for a single object class.

Connects to the OAK-D Lite, streams RGB + aligned stereo depth, and saves
frame pairs (JPG + depth NPY) on SPACE press.

Run once per class:
    python scripts/capture_dataset.py shoe
    python scripts/capture_dataset.py mug

Saved to:
    dataset/raw/<class_name>/<class_name>_0000.jpg
    dataset/raw/<class_name>/<class_name>_0000_depth.npy

After capturing all classes, run dataset_builder.py to split + format for YOLO.

--- depthai v3 API notes ---
- `dai.Pipeline()` opens the device internally; do NOT pass `dai.Device()` to it
- ColorCamera / MonoCamera are deprecated but still work in v3
- Output queues via `.createOutputQueue()` on node outputs — no XLinkOut node
- StereoDepth preset renamed: HIGH_DENSITY → DENSITY, HIGH_ACCURACY → ACCURACY
- setDepthAlign and setLeftRightCheck still work unchanged
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

counter = len(list(SAVE_DIR.glob("*.jpg")))  # resume numbering from last session

print(f"Class: '{CLASS_NAME}'  |  {counter} frames already saved in {SAVE_DIR}")
print("SPACE = save  |  Q = quit")
print("Vary: near (0.3-0.7m), mid (0.7-1.5m), far (1.5-2.5m), angles, lighting\n")

# --- Pipeline (v3: no device arg, Pipeline opens device internally) ---
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)  # v3: was HIGH_DENSITY
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# v3: output queues directly on node outputs — no XLinkOut node
rgb_q   = cam_rgb.preview.createOutputQueue(maxSize=4, blocking=False)
depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

# --- Capture loop ---
pipeline.start()

while pipeline.isRunning():
    rgb_msg   = rgb_q.get()
    depth_msg = depth_q.get()

    if rgb_msg is None or depth_msg is None:
        continue

    bgr      = rgb_msg.getCvFrame()
    depth_mm = depth_msg.getFrame()   # uint16, millimetres

    display = bgr.copy()
    cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Near 0.3-0.7m",  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(display, "Mid  0.7-1.5m",  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(display, "Far  1.5-2.5m",  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
    cv2.putText(display, "SPACE=save  Q=quit", (10, 625),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.imshow("OAK-D Capture", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
        dep_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}_depth.npy"
        cv2.imwrite(str(img_path), bgr)
        np.save(str(dep_path), depth_mm)
        print(f"Saved {img_path.name}")
        counter += 1
    elif key == ord("q"):
        pipeline.stop()
        break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
