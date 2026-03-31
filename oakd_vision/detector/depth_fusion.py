"""Fuses 2D YOLO detections with OAK-D stereo depth to produce 3D positions.

For each detection, we sample the depth map at the bounding box center region
(see utils/depth.py for why we crop to the inner 50%), then back-project the
bbox center pixel through the camera intrinsics to get an [X, Y, Z] position
in the camera frame.

Camera frame convention (OpenCV / OAK-D):
    Z = optical axis (forward, away from lens)
    X = right
    Y = down

Usage::

    from oakd_vision.utils.camera import CameraIntrinsics
    from oakd_vision.detector.depth_fusion import DepthFusion

    intrinsics = CameraIntrinsics.oak_d_lite_1080p()
    fuser = DepthFusion(intrinsics)

    detections_3d = fuser.fuse(detections, depth_mm_frame)
    for d in detections_3d:
        if d.position_3d is not None:
            x, y, z = d.position_3d
            print(f"{d.class_name}: {z:.2f}m at ({x:.2f}, {y:.2f}, {z:.2f})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from oakd_vision.utils.camera import CameraIntrinsics
from oakd_vision.utils.depth import get_depth_for_bbox
from oakd_vision.detector.yolo_inference import Detection


@dataclass
class Detection3D:
    """2D detection extended with a 3D camera-frame position."""

    bbox: tuple           # (x1, y1, x2, y2) pixels
    class_id: int
    class_name: str
    confidence: float
    depth_m: float        # raw depth estimate at bbox center (NaN if unavailable)
    position_3d: Optional[np.ndarray]  # [X, Y, Z] metres in camera frame


class DepthFusion:
    """Lifts 2D detections to 3D by querying a stereo depth map."""

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def fuse(self, detections: list, depth_map: np.ndarray) -> list:
        """Add 3D positions to a list of 2D detections.

        Args:
            detections: list of Detection from YOLODetector.detect()
            depth_map: HxW uint16 stereo depth map in millimetres

        Returns:
            List of Detection3D, preserving the original detection order.
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            depth_m = get_depth_for_bbox(depth_map, x1, y1, x2, y2)

            if np.isnan(depth_m):
                pos3d = None
            else:
                u = (x1 + x2) / 2.0  # bbox center column
                v = (y1 + y2) / 2.0  # bbox center row
                pos3d = self.intrinsics.pixel_to_3d(u, v, depth_m)

            results.append(Detection3D(
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                depth_m=depth_m,
                position_3d=pos3d,
            ))

        return results

    @staticmethod
    def overlay(frame: np.ndarray, detections_3d: list) -> np.ndarray:
        """Draw detection boxes and 3D labels onto a copy of frame.

        Args:
            frame: HxWx3 uint8 BGR image
            detections_3d: list of Detection3D

        Returns:
            Annotated copy of frame.
        """
        out = frame.copy()
        for d in detections_3d:
            x1, y1, x2, y2 = d.bbox
            cv_color = (0, 200, 0)
            cv2_text = f"{d.class_name} {d.confidence:.2f}"

            if d.position_3d is not None:
                x, y, z = d.position_3d
                cv2_text += f"  {z:.2f}m ({x:+.2f},{y:+.2f})"

            import cv2
            cv2.rectangle(out, (x1, y1), (x2, y2), cv_color, 2)
            cv2.putText(out, cv2_text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv_color, 1)

        return out
