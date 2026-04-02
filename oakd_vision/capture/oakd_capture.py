"""OAK-D Lite capture: RGB + stereo depth aligned to RGB.

Rewritten for depthai v3 API. Key v3 changes from v2:
- `dai.Pipeline()` opens the device internally; do NOT pass `dai.Device()` to it
- ColorCamera / MonoCamera still work (deprecated but functional in v3)
- Output queues via `node_output.createOutputQueue()` — no XLinkOut node needed
- StereoDepth preset renamed: HIGH_DENSITY → DENSITY, HIGH_ACCURACY → ACCURACY
- `pipeline.isRunning()` replaces the old device context manager loop pattern

Saves image/depth pairs as:
    <output_dir>/<prefix>_<NNNN>.jpg   — colour frame (BGR)
    <output_dir>/<prefix>_<NNNN>.npy   — depth map (uint16, millimetres)

Usage:
    python -m oakd_vision.capture.oakd_capture \\
        --output data/raw \\
        --prefix frame \\
        --preview
"""

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import depthai as dai
    DAI_AVAILABLE = True
except ImportError:
    DAI_AVAILABLE = False


class OAKDCapture:
    """DepthAI v3 pipeline wrapper for RGB + aligned stereo depth capture."""

    def __init__(
        self,
        rgb_size: tuple = (640, 640),
        fps: int = 30,
        depth_preset: str = "DENSITY",
    ):
        """
        Args:
            rgb_size: (width, height) for the RGB output stream.
            fps: target frame rate.
            depth_preset: StereoDepth preset — "DENSITY" or "ACCURACY".
                          (v3 renamed HIGH_DENSITY→DENSITY, HIGH_ACCURACY→ACCURACY)
        """
        if not DAI_AVAILABLE:
            raise ImportError("depthai is not installed. Run: pip install depthai")

        self.rgb_size = rgb_size
        self.fps = fps
        self.depth_preset = depth_preset
        self._pipeline = None
        self._rgb_q = None
        self._depth_q = None

    def start(self):
        """Build and start the pipeline."""
        self._pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = self._pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(*self.rgb_size)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        # Mono cameras for stereo
        mono_left  = self._pipeline.create(dai.node.MonoCamera)
        mono_right = self._pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        # StereoDepth — aligned to RGB so depth[y][x] == rgb[y][x]
        stereo = self._pipeline.create(dai.node.StereoDepth)
        preset = getattr(dai.node.StereoDepth.PresetMode, self.depth_preset)
        stereo.setDefaultProfilePreset(preset)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Output queues (v3: attach directly to node outputs, no XLinkOut)
        self._rgb_q   = cam_rgb.preview.createOutputQueue(maxSize=1, blocking=False)
        self._depth_q = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        self._pipeline.start()

    def stop(self):
        """Stop the pipeline."""
        if self._pipeline and self._pipeline.isRunning():
            self._pipeline.stop()
        self._pipeline = None

    def get_frame(self) -> tuple:
        """Return (bgr_frame, depth_mm) or (None, None) if not ready.

        Returns:
            bgr_frame : HxWx3 uint8 BGR image
            depth_mm  : HxW uint16 depth in millimetres (aligned to RGB)
        """
        rgb_msg   = self._rgb_q.get()
        depth_msg = self._depth_q.get()
        if rgb_msg is None or depth_msg is None:
            return None, None
        return rgb_msg.getCvFrame(), depth_msg.getFrame()

    def is_running(self) -> bool:
        return self._pipeline is not None and self._pipeline.isRunning()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def capture_dataset(output_dir: str, prefix: str = "frame", preview: bool = False):
    """Interactive capture loop. SPACE to save a frame pair, Q to quit."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    with OAKDCapture() as cam:
        print(f"Ready — saving to {out}")
        print("SPACE = save frame  |  Q = quit")

        while cam.is_running():
            bgr, depth_mm = cam.get_frame()
            if bgr is None:
                time.sleep(0.005)
                continue

            if preview:
                depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
                depth_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
                display = np.hstack([bgr, cv2.resize(depth_color, (bgr.shape[1], bgr.shape[0]))])
                cv2.imshow("OAK-D Capture (SPACE=save, Q=quit)", display)
            else:
                cv2.imshow("OAK-D Capture (SPACE=save, Q=quit)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                name = f"{prefix}_{saved:04d}"
                cv2.imwrite(str(out / f"{name}.jpg"), bgr)
                np.save(str(out / f"{name}.npy"), depth_mm)
                saved += 1
                print(f"Saved {name}  ({saved} total)")

    cv2.destroyAllWindows()
    print(f"Done. {saved} frames saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D Lite dataset capture")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--prefix",  default="frame", help="Filename prefix")
    parser.add_argument("--preview", action="store_true", help="Show depth side-by-side")
    args = parser.parse_args()
    capture_dataset(args.output, args.prefix, args.preview)
