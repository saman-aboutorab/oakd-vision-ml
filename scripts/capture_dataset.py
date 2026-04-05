"""Interactive dataset capture script for a single object class.

Connects to the OAK-D Lite, streams RGB only, and saves JPGs on SPACE press.
Depth is NOT captured here — it is not needed for YOLO training (model trains
on JPGs only). Depth is used at inference time on the robot.

Run once per class:
    python scripts/capture_dataset.py shoe
    python scripts/capture_dataset.py mug

Saved to:
    dataset/raw/<class_name>/<class_name>_0000.jpg

Works on USB 2.0 and USB 3.0 (RGB-only is well within USB 2.0 bandwidth).

--- depthai v3 API ---
- `dai.Pipeline(dai.Device())` opens the device then wraps it in a pipeline
- Use new `Camera` node with `.build(socket)` + `.requestOutput(capability)`
- Output queues via `.createOutputQueue()` directly on node outputs
"""

import sys
import warnings
from pathlib import Path

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

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

with dai.Pipeline(dai.Device()) as pipeline:

    device = pipeline.getDefaultDevice()
    print(f"USB speed: {device.getUsbSpeed().name}")

    # RGB camera only — no depth needed for YOLO training
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((640, 640), fps=25)
    rgb_q = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    print("Pipeline started — camera window opening...")
    print("SPACE = save  |  Q = quit")
    print("Vary: near (0.3-0.7m), mid (0.7-1.5m), far (1.5-2.5m), angles, lighting\n")

    # Show placeholder
    placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Starting camera...", (110, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(placeholder, "CLICK THIS WINDOW FIRST", (80, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    cv2.putText(placeholder, "then SPACE to save, Q to quit", (70, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow("OAK-D Capture", placeholder)
    cv2.waitKey(1)

    while pipeline.isRunning():
        rgb_msg = rgb_q.tryGet()

        if rgb_msg is None:
            cv2.waitKey(30)
            continue

        bgr = rgb_msg.getCvFrame()

        display = bgr.copy()
        cv2.putText(display, f"Class: {CLASS_NAME}  saved: {counter}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Near 0.3-0.7m", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(display, "Mid  0.7-1.5m", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(display, "Far  1.5-2.5m", (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1)
        cv2.putText(display, "CLICK WINDOW THEN: SPACE=save  Q=quit", (8, 630),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.imshow("OAK-D Capture", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(" ") or key == 32:
            img_path = SAVE_DIR / f"{CLASS_NAME}_{counter:04d}.jpg"
            cv2.imwrite(str(img_path), bgr)
            print(f"  SAVED [{counter:04d}]  {img_path.name}")
            counter += 1
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1]-1, flash.shape[0]-1), (0, 255, 0), 12)
            cv2.putText(flash, "SAVED!", (flash.shape[1]//2 - 60, flash.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("OAK-D Capture", flash)
            cv2.waitKey(200)
        elif key == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
print(f"\nDone. {counter} total frames for '{CLASS_NAME}' in {SAVE_DIR}")
