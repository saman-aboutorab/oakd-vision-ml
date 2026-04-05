"""P2 ReID data collection script.

Runs the P1 detector on a live OAK-D feed and auto-saves crops of detected
objects into identity folders for ReID training.

Usage — one run per identity instance:
    python scripts/collect_reid.py shoe 1      # → dataset/reid/shoe_001/
    python scripts/collect_reid.py shoe 2      # → dataset/reid/shoe_002/
    python scripts/collect_reid.py mug 1       # → dataset/reid/mug_001/
    python scripts/collect_reid.py person_feet 1

Instructions:
    - Place ONE object instance in front of the camera
    - Slowly move it: rotate it, change distance, change angle
    - Script auto-saves a crop every ~1.5 seconds when detected
    - Aim for ~35 crops per identity (takes about 1 minute)
    - Press Q to stop and move to the next identity

Works on USB 2.0 (RGB only — no depth needed for ReID crops).
"""

import argparse
import os
import sys
import time

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np
from pathlib import Path

import depthai as dai
from oakd_vision.detector import YOLODetector

# Crop is saved every this many frames (at 25fps → ~1.5s between saves)
SAVE_EVERY_N_FRAMES = 38
# Padding around detection box (fraction of box size)
PADDING = 0.10
# Crop resize for ReID model
CROP_SIZE = (128, 128)
# Minimum confidence to accept a detection
MIN_CONF = 0.45


def add_padding(x1, y1, x2, y2, pad, img_h, img_w):
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(img_w, int(x2 + bw * pad))
    y2 = min(img_h, int(y2 + bh * pad))
    return x1, y1, x2, y2


def run(class_name: str, instance_id: int, model_path: str):
    save_dir = Path(f"dataset/reid/{class_name}_{instance_id:03d}")
    save_dir.mkdir(parents=True, exist_ok=True)
    counter = len(list(save_dir.glob("*.jpg")))

    print(f"\nIdentity: {class_name}_{instance_id:03d}")
    print(f"Saving to: {save_dir}")
    print(f"Already saved: {counter} crops")
    print(f"Target: 35 crops  |  auto-saves every ~1.5s when '{class_name}' detected")
    print("Q = quit and move to next identity\n")

    detector = YOLODetector(model_path=model_path, mode="pytorch", conf_threshold=MIN_CONF)

    with dai.Pipeline(dai.Device()) as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"USB speed: {device.getUsbSpeed().name}")

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((640, 640), fps=25)
        rgb_q = rgb_out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        win = f"ReID Collect — {class_name}_{instance_id:03d}  |  Q=quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 640)

        placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting camera...", (160, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, placeholder)
        cv2.waitKey(1)

        frame_count = 0
        last_display = placeholder

        while pipeline.isRunning():
            rgb_msg = rgb_q.tryGet()
            if rgb_msg is None:
                cv2.imshow(win, last_display)
                cv2.waitKey(1)
                continue

            bgr = rgb_msg.getCvFrame()
            frame_count += 1

            detections = detector.detect(bgr)
            # Only care about the target class
            target_dets = [d for d in detections if d.class_name == class_name]

            display = bgr.copy()
            h, w = bgr.shape[:2]

            saved_this_frame = False
            for det in target_dets:
                x1, y1, x2, y2 = det.bbox
                x1p, y1p, x2p, y2p = add_padding(x1, y1, x2, y2, PADDING, h, w)

                # Draw detection box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{det.confidence:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Auto-save every N frames, only the highest-confidence detection
                if frame_count % SAVE_EVERY_N_FRAMES == 0 and not saved_this_frame:
                    crop = bgr[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, CROP_SIZE)
                        path = save_dir / f"{class_name}_{instance_id:03d}_{counter:04d}.jpg"
                        cv2.imwrite(str(path), crop_resized)
                        counter += 1
                        saved_this_frame = True
                        print(f"  SAVED [{counter-1:04d}]  {path.name}")
                        # Flash crop in corner so user sees it saved
                        display[8:8+64, 8:8+64] = cv2.resize(crop_resized, (64, 64))
                        cv2.rectangle(display, (6, 6), (74, 74), (0, 255, 0), 2)

            # HUD
            progress = min(counter, 35)
            bar_w = int((progress / 35) * 200)
            cv2.rectangle(display, (8, 610), (208, 628), (60, 60, 60), -1)
            cv2.rectangle(display, (8, 610), (8 + bar_w, 628), (0, 220, 0), -1)
            cv2.putText(display, f"{class_name}_{instance_id:03d}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display, f"saved: {counter}/35", (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0) if counter >= 35 else (200, 200, 200), 1)
            if not target_dets:
                cv2.putText(display, f"looking for '{class_name}'...", (8, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
            if counter >= 35:
                cv2.putText(display, "TARGET REACHED — press Q", (100, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_display = display
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    detector.close()
    print(f"\nDone. {counter} crops saved to {save_dir}")
    print(f"Next: python scripts/collect_reid.py {class_name} {instance_id + 1}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("class_name", help="Object class (must match P1 classes)")
    parser.add_argument("instance_id", type=int, help="Instance number, e.g. 1, 2, 3...")
    parser.add_argument("--model", default="runs/detect/runs/train/p1_v1/weights/best.pt")
    args = parser.parse_args()

    if args.instance_id < 1:
        print("instance_id must be >= 1")
        sys.exit(1)

    run(args.class_name, args.instance_id, args.model)
