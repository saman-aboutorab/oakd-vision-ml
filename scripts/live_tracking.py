"""P2 live tracking demo: OAK-D → detect → ReID track → persistent IDs.

Each detected object gets a persistent integer ID that survives brief occlusions.
The ID colour stays consistent across frames so you can visually follow objects.

Architecture:
  OAK-D RGB  →  YOLOv8n detector (GPU)  →  ReID model (GPU)  →  MOTTracker
                                                                    ↓
                                             persistent bounding boxes + IDs

Usage:
    python scripts/live_tracking.py
    python scripts/live_tracking.py \\
        --detector runs/detect/runs/train/p1_v1/weights/best.pt \\
        --reid     runs/reid/best.pt \\
        --conf     0.35

Press Q to quit.
"""

import argparse
import os
import time

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np
import torch
import yaml

import depthai as dai
from oakd_vision.detector import YOLODetector
from oakd_vision.tracker.reid_model import ReIDNet
from oakd_vision.tracker.mot_tracker import MOTTracker, Detection, extract_crop


# ---------------------------------------------------------------------------
# Colour palette — one colour per track ID (cycles every 20 IDs)
# ---------------------------------------------------------------------------

_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 255,  51), ( 51, 255,  51),
    ( 51, 255, 255), ( 51,  51, 255), (255,  51, 255), (  0, 128, 255),
    (255, 128,   0), (128,   0, 255), (  0, 255, 128), (255,   0, 128),
    (128, 255,   0), (  0, 128, 128), (128,   0, 128), (128, 128,   0),
    (200,  80,  80), ( 80, 200,  80), ( 80,  80, 200), (180, 180,  60),
]


def id_colour(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tracks(frame: np.ndarray, tracks) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox.astype(int)
        colour = id_colour(t.track_id)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Label background + text
        label = f"ID:{t.track_id}  {t.class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(detector_path: str, reid_path: str, reid_config: str, conf: float):
    # --- Load config ---
    with open(reid_config) as f:
        cfg = yaml.safe_load(f)
    crop_size = tuple(cfg["data"]["crop_size"])
    embedding_dim = cfg["model"]["embedding_dim"]
    dropout = cfg["model"]["dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- Load detector ---
    detector = YOLODetector(model_path=detector_path, mode="pytorch", conf_threshold=conf)
    print(f"Detector loaded. Classes: {detector.class_names}")

    # --- Load ReID model ---
    reid_model = ReIDNet(embedding_dim=embedding_dim, dropout=dropout).to(device)
    reid_model.load_state_dict(torch.load(reid_path, map_location=device))
    reid_model.eval()
    print(f"ReID model loaded from {reid_path}")

    # --- Build tracker ---
    tracker = MOTTracker(
        reid_model=reid_model,
        device=device,
        n_init=3,        # confirm after 3 consecutive matches
        max_age=30,      # keep track alive for 30 missed frames (~1.2s at 25fps)
        iou_gate=0.7,
        reid_weight=0.5,
        crop_size=crop_size,
    )

    print("Press Q to quit.\n")

    with dai.Pipeline(dai.Device()) as pipeline:
        device_hw = pipeline.getDefaultDevice()
        print(f"USB speed: {device_hw.getUsbSpeed().name}")

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((640, 640), fps=25)
        rgb_q = rgb_out.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()
        print("Pipeline started.\n")

        win = "P2 Live Tracking  —  Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 640)
        placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting camera...", (160, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, placeholder)
        cv2.waitKey(1)

        last_display = placeholder
        fps_counter = 0
        fps = 0.0
        t_start = time.perf_counter()

        while pipeline.isRunning():
            rgb_msg = rgb_q.tryGet()

            if rgb_msg is None:
                cv2.imshow(win, last_display)
                cv2.waitKey(1)
                continue

            bgr = rgb_msg.getCvFrame()

            # --- Detect ---
            raw_dets = detector.detect(bgr)

            # Convert to Detection objects for tracker
            detections = []
            crops = []
            for d in raw_dets:
                bbox = np.array(d.bbox, dtype=float)   # (x1, y1, x2, y2)
                detections.append(Detection(
                    bbox=bbox,
                    conf=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                ))
                crops.append(extract_crop(bgr, bbox, padding=0.10))

            # --- Track ---
            confirmed_tracks = tracker.update(detections, crops)

            # --- Visualise ---
            display = draw_tracks(bgr, confirmed_tracks)
            last_display = display

            # FPS
            fps_counter += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                t_start = time.perf_counter()

            cv2.putText(display, f"FPS: {fps:.1f}  Tracks: {len(confirmed_tracks)}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    detector.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="runs/detect/runs/train/p1_v1/weights/best.pt")
    parser.add_argument("--reid",     default="runs/reid/best.pt")
    parser.add_argument("--config",   default="training/configs/reid_config.yaml")
    parser.add_argument("--conf",     type=float, default=0.35)
    args = parser.parse_args()
    run(args.detector, args.reid, args.config, args.conf)
