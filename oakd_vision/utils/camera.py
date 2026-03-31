"""OAK-D Lite camera intrinsics and 3D back-projection utilities.

The OAK-D Lite ships with a factory calibration. These constants are
approximate defaults; at runtime, DepthAI can load the calibrated values
directly from the device (see OAKDCapture.get_intrinsics()).
"""

import numpy as np

# Approximate factory intrinsics for the OAK-D Lite RGB camera (IMX378)
# at 1920x1080. cx/cy are the principal point (image centre).
_OAK_D_LITE_1080P = {"fx": 1076.3, "fy": 1076.3, "cx": 960.0, "cy": 540.0}

# Same camera scaled to 640x480 (multiply by 640/1920 = 1/3)
_OAK_D_LITE_480P = {"fx": 480.1, "fy": 480.1, "cx": 320.0, "cy": 240.0}


class CameraIntrinsics:
    """Pinhole camera intrinsics + 3D back-projection.

    Attributes:
        fx, fy: focal lengths in pixels
        cx, cy: principal point in pixels
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @classmethod
    def oak_d_lite_1080p(cls) -> "CameraIntrinsics":
        return cls(**_OAK_D_LITE_1080P)

    @classmethod
    def oak_d_lite_480p(cls) -> "CameraIntrinsics":
        return cls(**_OAK_D_LITE_480P)

    @classmethod
    def from_depthai_calibration(cls, calib, width: int, height: int) -> "CameraIntrinsics":
        """Load intrinsics directly from a connected OAK-D device.

        Args:
            calib: result of device.readCalibration()
            width, height: output resolution of the RGB stream
        """
        import depthai as dai

        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        # M is a 3x3 list-of-lists: [[fx,0,cx],[0,fy,cy],[0,0,1]]
        return cls(fx=M[0][0], fy=M[1][1], cx=M[0][2], cy=M[1][2])

    def pixel_to_3d(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Back-project a pixel at known depth to a 3D camera-frame point.

        The camera frame follows the standard OpenCV convention:
            Z  = optical axis (forward)
            X  = right
            Y  = down

        Args:
            u: pixel column (x-coordinate)
            v: pixel row (y-coordinate)
            depth_m: depth along the optical axis in metres

        Returns:
            np.ndarray shape (3,): [X, Y, Z] in metres
        """
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float32)
