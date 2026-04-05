"""P1 live demo: OAK-D → detect → 3D localize in real time.

Streams RGB + aligned stereo depth from the OAK-D Lite, runs YOLOv8n on the
laptop GPU, fuses detections with depth to get 3D positions, and overlays
bounding boxes + distance labels on the display.

Usage:
    python scripts/live_demo.py --model runs/detect/runs/train/p1_v1/weights/best.pt

Press Q to quit.

Requires USB 3.0 (blue port) — RGB + depth together need the bandwidth.
"""

import argparse
import os
import time

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np

import depthai as dai
from oakd_vision.detector import YOLODetector, DepthFusion
from oakd_vision.utils.camera import CameraIntrinsics


def run_demo(model_path: str, conf: float):
    detector = YOLODetector(
        model_path=model_path,
        mode="pytorch",
        conf_threshold=conf,
    )
    print(f"Model loaded. Classes: {detector.class_names}")
    print("Press Q to quit.\n")

    with dai.Pipeline(dai.Device()) as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"USB speed: {device.getUsbSpeed().name}")

        # RGB camera
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((640, 640), fps=25)
        rgb_q = rgb_out.createOutputQueue(maxSize=2, blocking=False)

        # Stereo depth — only on USB 3.0 (SUPER)
        usb_speed = device.getUsbSpeed()
        use_depth = usb_speed == dai.UsbSpeed.SUPER or usb_speed == dai.UsbSpeed.SUPER_PLUS
        depth_q = None
        if use_depth:
            cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setLeftRightCheck(True)
            stereo.setOutputSize(640, 640)
            cam_left.requestFullResolutionOutput().link(stereo.left)
            cam_right.requestFullResolutionOutput().link(stereo.right)
            depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
            print("Depth enabled — 3D positions will be shown.")
        else:
            print("USB 2.0 detected — depth disabled, showing 2D boxes only.")

        # Load calibrated intrinsics from device
        try:
            calib = device.readCalibration()
            intrinsics = CameraIntrinsics.from_depthai_calibration(calib, 640, 640)
            print("Loaded intrinsics from device calibration.")
        except Exception:
            intrinsics = CameraIntrinsics.oak_d_lite_480p()
            print("Using default intrinsics.")

        fuser = DepthFusion(intrinsics)

        pipeline.start()
        print("Pipeline started.\n")

        # Open window at full size immediately
        win = "P1 Live Demo  —  Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 640)
        placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting camera...", (160, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(placeholder, "Q to quit", (240, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.imshow(win, placeholder)
        cv2.waitKey(1)

        last_depth_mm = None
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

            if depth_q is not None:
                depth_msg = depth_q.tryGet()
                if depth_msg is not None:
                    last_depth_mm = depth_msg.getFrame()

            # Run YOLO on laptop GPU
            detections = detector.detect(bgr)

            # Fuse with depth (or zeros if USB 2.0 / depth not yet arrived)
            depth_map = last_depth_mm if last_depth_mm is not None else np.zeros((640, 640), dtype=np.uint16)
            detections_3d = fuser.fuse(detections, depth_map)
            display = DepthFusion.overlay(bgr, detections_3d)
            last_display = display

            # FPS counter
            fps_counter += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                t_start = time.perf_counter()

            cv2.putText(display, f"FPS: {fps:.1f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    detector.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/detect/runs/train/p1_v1/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()
    run_demo(args.model, args.conf)
