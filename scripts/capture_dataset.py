"""Interactive dataset capture script for a single object class.

Connects to the OAK-D Lite, streams RGB + aligned stereo depth, and saves
frame pairs (JPG + depth NPY) on SPACE press.

Run once per class:
    python scripts/capture_dataset.py shoe
    python scripts/capture_dataset.py mug
    ...

Saved to:
    dataset/raw/<class_name>/<class_name>_0000.jpg
    dataset/raw/<class_name>/<class_name>_0000_depth.npy

After capturing all classes, run dataset_builder.py to split + format for YOLO.
"""

import sys
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# --- Config ---
CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "object"
SAVE_DIR = Path(f"dataset/raw/{CLASS_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Pipeline ---
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # RGB sensor

mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth pixels to the RGB camera frame so they correspond 1:1
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
cam_rgb.preview.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)

# --- Capture loop ---
counter = len(list(SAVE_DIR.glob("*.jpg")))  # resume numbering from last session

with dai.Device(pipeline) as device:
    rgb_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)
    print(f"Capturing '{CLASS_NAME}'  ({counter} frames already saved)")
    print("SPACE = save  |  Q = quit")
    print("Aim for variety: near (0.3-0.7m), medium (0.7-1.5m), far (1.5-2.5m)")
    print("Vary angle, lighting, and background between shots.\n")

    while True:
        rgb_frame = rgb_q.get().getCvFrame()
        depth_frame = depth_q.get().getFrame()  # uint16, millimetres

        display = rgb_frame.copy()
        cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, "Near 0.3-0.7m",  (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(display, "Mid  0.7-1.5m",  (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(display, "Far  1.5-2.5m",  (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        cv2.putText(display, "SPACE=save  Q=quit", (10, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("OAK-D Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
            dep_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}_depth.npy"
            cv2.imwrite(str(img_path), rgb_frame)
            np.save(str(dep_path), depth_frame)
            print(f"Saved {img_path.name}")
            counter += 1
        elif key == ord("q"):
            break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
