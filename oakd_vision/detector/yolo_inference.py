"""Unified YOLOv8 inference API for three execution backends.

    mode="pytorch"  — Ultralytics YOLO, runs on CPU/GPU. Best for development.
    mode="onnx"     — ONNX Runtime, runs on CPU. Good for testing export quality.
    mode="vpu"      — DepthAI NeuralNetwork, runs on OAK-D Myriad X. ~25 FPS.

All three backends expose the same interface::

    detector = YOLODetector("models/best.pt",  mode="pytorch", class_names=[...])
    detector = YOLODetector("models/best.onnx", mode="onnx",   class_names=[...])
    detector = YOLODetector("models/best.blob", mode="vpu",    class_names=[...])

    detections = detector.detect(bgr_frame)   # list[Detection]
    for d in detections:
        print(d.class_name, d.confidence, d.bbox)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import depthai as dai

    DAI_AVAILABLE = True
except ImportError:
    DAI_AVAILABLE = False


@dataclass
class Detection:
    """Single object detection in pixel coordinates."""

    bbox: tuple          # (x1, y1, x2, y2) in pixels
    class_id: int
    class_name: str
    confidence: float


class YOLODetector:
    """Unified YOLOv8 detector supporting pytorch, onnx, and vpu backends."""

    def __init__(
        self,
        model_path: str,
        mode: str = "pytorch",
        class_names: Optional[list] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: tuple = (640, 640),
    ):
        """
        Args:
            model_path: path to .pt, .onnx, or .blob model file
            mode: "pytorch" | "onnx" | "vpu"
            class_names: list of class name strings. Required for onnx/vpu modes.
            conf_threshold: minimum confidence to keep a detection
            iou_threshold: NMS IoU threshold (onnx/vpu modes)
            input_size: (width, height) model expects (onnx/vpu modes)
        """
        self.model_path = Path(model_path)
        self.mode = mode
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self._model = None      # pytorch / onnxruntime session
        self._device = None     # depthai device (vpu mode)
        self._in_queue = None   # depthai input queue (vpu mode)
        self._out_queue = None  # depthai output queue (vpu mode)

        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self.mode == "pytorch":
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            # Extract class names from model if not provided
            if not self.class_names:
                self.class_names = list(self._model.names.values())

        elif self.mode == "onnx":
            import onnxruntime as ort
            self._model = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_name = self._model.get_inputs()[0].name

        elif self.mode == "vpu":
            if not DAI_AVAILABLE:
                raise ImportError("depthai is required for vpu mode")
            self._load_vpu()

        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Choose: pytorch | onnx | vpu")

    def _load_vpu(self):
        """Build a DepthAI pipeline with XLinkIn → NeuralNetwork → XLinkOut."""
        pipeline = dai.Pipeline()

        nn = pipeline.create(dai.node.NeuralNetwork)
        xin = pipeline.create(dai.node.XLinkIn)
        xout = pipeline.create(dai.node.XLinkOut)

        nn.setBlobPath(str(self.model_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        xin.setStreamName("input")
        xout.setStreamName("output")

        xin.out.link(nn.input)
        nn.out.link(xout.input)

        self._device = dai.Device(pipeline)
        self._in_queue = self._device.getInputQueue("input", maxSize=1, blocking=False)
        self._out_queue = self._device.getOutputQueue("output", maxSize=1, blocking=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on a single BGR frame.

        Args:
            frame: HxWx3 uint8 BGR image (from OpenCV or OAK-D)

        Returns:
            List of Detection objects, sorted by confidence descending.
        """
        if self.mode == "pytorch":
            return self._detect_pytorch(frame)
        elif self.mode == "onnx":
            return self._detect_onnx(frame)
        elif self.mode == "vpu":
            return self._detect_vpu(frame)

    def _detect_pytorch(self, frame: np.ndarray) -> list:
        results = self._model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            detections.append(Detection((x1, y1, x2, y2), cls_id, name, conf))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _detect_onnx(self, frame: np.ndarray) -> list:
        # Preprocess: BGR→RGB, resize, HWC→CHW, normalize to [0,1]
        h_orig, w_orig = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]

        raw = self._model.run(None, {self._onnx_input_name: img})[0]  # [1, 4+nc, 8400]
        return self._postprocess_onnx(raw, w_orig, h_orig)

    def _postprocess_onnx(self, raw: np.ndarray, w_orig: int, h_orig: int) -> list:
        """Decode YOLOv8 ONNX output [1, 4+nc, 8400] into Detection list."""
        preds = raw[0].T  # [8400, 4+nc]
        nc = preds.shape[1] - 4

        boxes_cxcywh = preds[:, :4]       # normalized cx,cy,w,h
        class_scores = preds[:, 4:]       # [8400, nc]

        # Max class score and id per anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = confidences >= self.conf_threshold
        if not mask.any():
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h (in model input pixels) → x1,y1,x2,y2 (original pixels)
        iw, ih = self.input_size
        scale_x = w_orig / iw
        scale_y = h_orig / ih

        cx = boxes_cxcywh[:, 0] * scale_x
        cy = boxes_cxcywh[:, 1] * scale_y
        bw = boxes_cxcywh[:, 2] * scale_x
        bh = boxes_cxcywh[:, 3] * scale_y

        x1 = (cx - bw / 2).astype(int)
        y1 = (cy - bh / 2).astype(int)
        x2 = (cx + bw / 2).astype(int)
        y2 = (cy + bh / 2).astype(int)

        # NMS via OpenCV
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        detections = []
        for i in (indices.flatten() if len(indices) else []):
            cls_id = int(class_ids[i])
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            detections.append(Detection(
                (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                cls_id,
                name,
                float(confidences[i]),
            ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _detect_vpu(self, frame: np.ndarray) -> list:
        """Send frame to Myriad X, receive raw output, decode."""
        iw, ih = self.input_size
        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float16)  # blobs expect FP16
        img = np.transpose(img, (2, 0, 1)).flatten()

        buf = dai.Buffer()
        buf.setData(img.tobytes())
        self._in_queue.send(buf)

        result = self._out_queue.get()
        raw = np.array(result.getFirstLayerFp16()).reshape(1, -1, 8400)
        return self._postprocess_onnx(raw.astype(np.float32), frame.shape[1], frame.shape[0])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        if self._device is not None:
            self._device.close()
            self._device = None

    def __del__(self):
        self.close()
