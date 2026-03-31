"""P1 live demo: OAK-D → detect → 3D localize in real time.

Streams RGB + depth from the OAK-D Lite, runs YOLODetector, fuses detections
with stereo depth, and overlays bounding boxes + 3D positions on the display.

Supports all three backends:
    python scripts/live_demo.py --model models/best.pt   --mode pytorch
    python scripts/live_demo.py --model models/best.onnx --mode onnx
    python scripts/live_demo.py --model models/best.blob --mode vpu

Press Q to quit. FPS is shown in the top-left corner.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import depthai as dai
from oakd_vision.detector import YOLODetector, DepthFusion
from oakd_vision.utils.camera import CameraIntrinsics


def load_class_names(data_yaml: str) -> list:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])


def build_pipeline() -> dai.Pipeline:
    """Minimal pipeline: RGB + aligned depth only (no on-device NN)."""
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    return pipeline


def run_demo(model_path: str, mode: str, data_yaml: str, conf: float):
    class_names = load_class_names(data_yaml) if data_yaml else []

    detector = YOLODetector(
        model_path=model_path,
        mode=mode,
        class_names=class_names,
        conf_threshold=conf,
    )

    intrinsics = CameraIntrinsics.oak_d_lite_480p()  # 640x640 preview ≈ 480p intrinsics
    fuser = DepthFusion(intrinsics)

    print(f"Model: {model_path}  mode={mode}  classes={len(class_names)}")
    print("Press Q to quit.\n")

    with dai.Device(build_pipeline()) as device:
        rgb_q = device.getOutputQueue("rgb", maxSize=1, blocking=False)
        dep_q = device.getOutputQueue("depth", maxSize=1, blocking=False)

        # Try to load calibrated intrinsics from device EEPROM
        try:
            calib = device.readCalibration()
            intrinsics = CameraIntrinsics.from_depthai_calibration(calib, 640, 640)
            print("Loaded intrinsics from device calibration.")
        except Exception:
            print("Using default intrinsics (device calibration unavailable).")

        fuser = DepthFusion(intrinsics)

        fps_counter = 0
        fps = 0.0
        t_start = time.perf_counter()

        while True:
            rgb_msg = rgb_q.tryGet()
            dep_msg = dep_q.tryGet()

            if rgb_msg is None or dep_msg is None:
                continue

            bgr = rgb_msg.getCvFrame()
            depth_mm = dep_msg.getFrame()

            detections = detector.detect(bgr)
            detections_3d = fuser.fuse(detections, depth_mm)
            display = DepthFusion.overlay(bgr, detections_3d)

            # FPS
            fps_counter += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                t_start = time.perf_counter()

            cv2.putText(display, f"FPS: {fps:.1f}  mode={mode}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow("P1 Live Demo — Q to quit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 live detection + 3D fusion demo")
    parser.add_argument("--model", required=True, help=".pt / .onnx / .blob model path")
    parser.add_argument("--mode", default="pytorch", choices=["pytorch", "onnx", "vpu"])
    parser.add_argument("--data", default="training/configs/yolov8n_custom.yaml",
                        help="dataset.yaml (for class names)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    run_demo(args.model, args.mode, args.data, args.conf)
