"""OAK-D Lite dataset capture: synchronized RGB + stereo depth.

Builds a DepthAI pipeline that streams colour frames and depth maps aligned
to the RGB sensor. Press SPACE to save a frame pair, 'q' to quit.

Saved files per frame:
    <output_dir>/<prefix>_<NNNN>.jpg   — BGR colour image (uint8)
    <output_dir>/<prefix>_<NNNN>.npy   — depth map (uint16, millimetres)

Usage:
    python -m oakd_vision.capture.oakd_capture \\
        --output data/raw \\
        --prefix frame \\
        --preview
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import depthai as dai

    DAI_AVAILABLE = True
except ImportError:
    DAI_AVAILABLE = False


class OAKDCapture:
    """Thin wrapper around a DepthAI pipeline for RGB + depth capture.

    The depth stream is aligned to the RGB sensor plane and resized to match
    the colour frame resolution, so every pixel has a direct depth value.

    Usage::

        with OAKDCapture() as cam:
            bgr, depth_mm = cam.get_frame()
    """

    def __init__(
        self,
        color_resolution: str = "1080p",
        depth_preset: str = "HIGH_ACCURACY",
    ):
        """
        Args:
            color_resolution: "1080p" (1920×1080) or "480p" (640×480)
            depth_preset: StereoDepth quality preset — "HIGH_ACCURACY" (denser
                disparity, slower) or "HIGH_DENSITY" (more pixels, more noise)
        """
        if not DAI_AVAILABLE:
            raise ImportError(
                "depthai is not installed. Install it with:\n"
                "  pip install 'oakd_vision[depthai]'"
            )

        self._color_resolution = color_resolution
        self._depth_preset = depth_preset
        self._device = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> "dai.Pipeline":
        pipeline = dai.Pipeline()

        # --- Nodes ---
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_depth = pipeline.create(dai.node.XLinkOut)

        # --- RGB camera ---
        res_map = {
            "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            "480p": dai.ColorCameraProperties.SensorResolution.THE_480_P,
        }
        cam_rgb.setResolution(res_map[self._color_resolution])
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # --- Mono cameras for stereo ---
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        # --- StereoDepth ---
        # Align depth to the RGB lens so every colour pixel has a depth value.
        preset = getattr(dai.node.StereoDepth.PresetMode, self._depth_preset)
        stereo.setDefaultProfilePreset(preset)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)   # reduces occlusion artifacts
        stereo.setSubpixel(False)        # subpixel off = faster, uint16 output

        # --- XLink outputs ---
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")

        # --- Connections ---
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        cam_rgb.video.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        return pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Connect to the device and start streaming."""
        self._device = dai.Device(self._build_pipeline())

    def stop(self):
        """Close the device connection."""
        if self._device is not None:
            self._device.close()
            self._device = None

    def get_frame(self) -> tuple:
        """Return the latest (bgr, depth_mm) frame pair, or (None, None).

        Returns:
            bgr: HxWx3 uint8 BGR colour frame
            depth_mm: HxW uint16 depth map in millimetres (0 = invalid pixel)
        """
        rgb_q = self._device.getOutputQueue("rgb", maxSize=1, blocking=False)
        dep_q = self._device.getOutputQueue("depth", maxSize=1, blocking=False)

        rgb_msg = rgb_q.tryGet()
        dep_msg = dep_q.tryGet()

        if rgb_msg is None or dep_msg is None:
            return None, None

        return rgb_msg.getCvFrame(), dep_msg.getFrame()

    def get_intrinsics(self):
        """Read the factory calibration from the device at runtime.

        Returns:
            CameraIntrinsics loaded from the device EEPROM.
        """
        from oakd_vision.utils.camera import CameraIntrinsics

        calib = self._device.readCalibration()
        w, h = (1920, 1080) if self._color_resolution == "1080p" else (640, 480)
        return CameraIntrinsics.from_depthai_calibration(calib, w, h)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ------------------------------------------------------------------
# CLI — interactive capture loop
# ------------------------------------------------------------------

def _capture_loop(output_dir: str, prefix: str, preview: bool):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    with OAKDCapture() as cam:
        print(f"Ready. SPACE = save  |  q = quit  →  {out}")

        while True:
            bgr, depth_mm = cam.get_frame()
            if bgr is None:
                time.sleep(0.01)
                continue

            if preview:
                depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imshow("OAK-D (SPACE=save, q=quit)", np.hstack([bgr, depth_vis]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                stem = f"{prefix}_{saved:04d}"
                cv2.imwrite(str(out / f"{stem}.jpg"), bgr)
                np.save(str(out / f"{stem}.npy"), depth_mm)
                saved += 1
                print(f"Saved {stem}  ({saved} total)")

    cv2.destroyAllWindows()
    print(f"\nDone. {saved} frame pairs saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D Lite dataset capture")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--prefix", default="frame", help="Filename prefix")
    parser.add_argument("--preview", action="store_true", help="Show live RGB+depth window")
    args = parser.parse_args()
    _capture_loop(args.output, args.prefix, args.preview)
