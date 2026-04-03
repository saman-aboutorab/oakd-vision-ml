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

    # Show placeholder immediately so window is visible before first frame
    placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Starting camera...", (140, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("OAK-D Capture", placeholder)
    cv2.waitKey(1)

    while pipeline.isRunning():
        rgb_msg   = rgb_q.tryGet()
        depth_msg = depth_q.tryGet()

        if rgb_msg is None:
            cv2.waitKey(1)
            continue

        bgr      = rgb_msg.getCvFrame()
        depth_mm = depth_msg.getFrame() if depth_msg is not None else None

        display = bgr.copy()
        cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Near 0.3-0.7m", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(display, "Mid  0.7-1.5m", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(display, "Far  1.5-2.5m", (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1)
        depth_status = f"depth: {int(depth_mm[320,320])}mm" if depth_mm is not None else "depth: --"
        cv2.putText(display, depth_status, (8, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(display, "SPACE=save  Q=quit", (8, 625),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.imshow("OAK-D Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
            cv2.imwrite(str(img_path), bgr)
            saved_msg = f"Saved {img_path.name}"
            if depth_mm is not None:
                dep_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}_depth.npy"
                np.save(str(dep_path), depth_mm)
                saved_msg += " + depth"
            print(saved_msg)
            counter += 1
        elif key == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
