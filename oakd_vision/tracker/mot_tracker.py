"""MOTTracker — Multi-Object Tracker combining Kalman filter + Hungarian algorithm.

Architecture overview
---------------------
Each detected object is assigned a *Track*. Every track maintains:
  1. A **Kalman filter** — predicts where the object will be in the next frame
     even when the detector misses it.
  2. A **ReID embedding** — a 128-dim fingerprint from the ReID model used to
     re-identify the same object after an occlusion.

Assignment pipeline (each frame)
---------------------------------
  1. Predict next position for all active tracks (Kalman predict step).
  2. Compute a cost matrix [Tracks × Detections]:
       cost = 0.5 × IoU-distance + 0.5 × ReID-cosine-distance
     IoU-distance alone handles fast motion; ReID alone handles occlusion;
     combining both is more robust than either alone.
  3. Gate: set cost = ∞ for pairs where IoU-distance > 0.7 (no spatial overlap).
  4. Run Hungarian algorithm (scipy) to find the globally optimal assignment.
  5. Update matched tracks with the new detection (Kalman update + embedding).
  6. Unmatched detections → new tentative tracks.
  7. Unmatched tracks → increment miss counter; delete after max_age frames.

Track lifecycle
---------------
  tentative  →  (n_init consecutive matches)  →  confirmed
  confirmed  →  (max_age consecutive misses)  →  deleted

Only confirmed tracks are returned to the caller.

Kalman state vector
--------------------
  [cx, cy, w, h, vx, vy, vw, vh]  (constant-velocity model)
  Measurement: [cx, cy, w, h]

Usage
-----
  tracker = MOTTracker(reid_model, device)
  for frame in stream:
      detections = detector.detect(frame)
      crops = [crop_bbox(frame, d.bbox) for d in detections]
      tracks = tracker.update(detections, crops)
      for t in tracks:
          draw_box(frame, t.bbox, label=f"ID:{t.track_id}  {t.class_name}")
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from oakd_vision.tracker.reid_model import ReIDNet
from oakd_vision.tracker.triplet_dataset import build_transforms


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, numpy only — no extra deps)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Kalman filter tracking a single bounding box.

    State:   [cx, cy, w, h, vx, vy, vw, vh]
    Measure: [cx, cy, w, h]
    """

    _shared_F = np.eye(8)              # state transition — set once below
    _shared_H = np.zeros((4, 8))      # measurement matrix
    _shared_Q = np.eye(8)             # process noise
    _shared_R = np.eye(4)             # measurement noise

    @classmethod
    def _init_matrices(cls):
        # Constant-velocity: position += velocity * dt (dt=1 frame)
        for i in range(4):
            cls._shared_F[i, i + 4] = 1.0
        # H selects [cx, cy, w, h] from state
        for i in range(4):
            cls._shared_H[i, i] = 1.0
        # Process noise — velocity components get more uncertainty
        cls._shared_Q[4:, 4:] *= 10.0
        # Measurement noise — slightly larger for w, h
        cls._shared_R[2, 2] *= 4.0
        cls._shared_R[3, 3] *= 4.0

    def __init__(self, bbox: np.ndarray):
        """
        Args:
            bbox: [x1, y1, x2, y2]
        """
        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(8) * 10.0       # covariance — large uncertainty at init
        self.P[4:, 4:] *= 100.0         # velocity completely unknown at start

    def predict(self) -> np.ndarray:
        """Advance state by one time step. Returns predicted [cx, cy, w, h]."""
        F = self._shared_F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._shared_Q
        # Clip negative w/h (can drift slightly below zero)
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.x[:4].copy()

    def update(self, bbox: np.ndarray):
        """Incorporate a new measurement [x1, y1, x2, y2]."""
        z = np.array(_xyxy_to_cxcywh(bbox), dtype=float)
        H, R = self._shared_H, self._shared_R
        y = z - H @ self.x                             # innovation
        S = H @ self.P @ H.T + R                       # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)            # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        """Current predicted position as [x1, y1, x2, y2]."""
        return _cxcywh_to_xyxy(self.x[:4])


# Initialise shared matrices once at import time
KalmanBoxTracker._init_matrices()


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

_id_counter = itertools.count(1)


@dataclass
class Track:
    """A single tracked object."""
    track_id: int
    class_id: int
    class_name: str
    kalman: KalmanBoxTracker
    embedding: Optional[np.ndarray] = None      # latest ReID embedding (unit vector)

    hits: int = 1                               # total times matched
    age: int = 1                                # total frames alive
    misses: int = 0                             # consecutive unmatched frames
    state: str = "tentative"                    # tentative | confirmed | deleted

    @property
    def bbox(self) -> np.ndarray:
        return self.kalman.bbox

    def predict(self):
        self.kalman.predict()
        self.age += 1

    def update(self, bbox: np.ndarray, embedding: Optional[np.ndarray], n_init: int):
        self.kalman.update(bbox)
        if embedding is not None:
            self.embedding = embedding
        self.hits += 1
        self.misses = 0
        if self.state == "tentative" and self.hits >= n_init:
            self.state = "confirmed"

    def mark_missed(self, max_age: int):
        self.misses += 1
        if self.misses >= max_age:
            self.state = "deleted"


# ---------------------------------------------------------------------------
# Detection container (mirrors what YOLODetector returns)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Minimal detection struct consumed by the tracker."""
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixels
    conf: float
    class_id: int
    class_name: str


# ---------------------------------------------------------------------------
# MOTTracker
# ---------------------------------------------------------------------------

class MOTTracker:
    """Kalman + Hungarian + ReID multi-object tracker.

    Args:
        reid_model:   Trained ReIDNet (already on device, eval mode).
        device:       torch.device for running ReID inference.
        n_init:       Frames a track must match before being confirmed (default 3).
        max_age:      Consecutive missed frames before a track is deleted (default 30).
        iou_gate:     IoU-distance above this threshold → pair is infeasible (default 0.7).
        reid_weight:  Weight for ReID in cost = reid_w*reid + (1-reid_w)*iou (default 0.5).
        crop_size:    H×W fed to ReID model (must match training, default 128×128).
    """

    def __init__(
        self,
        reid_model: ReIDNet,
        device: torch.device,
        n_init: int = 3,
        max_age: int = 30,
        iou_gate: float = 0.7,
        reid_weight: float = 0.5,
        crop_size: Tuple[int, int] = (128, 128),
    ):
        self.reid_model = reid_model
        self.reid_model.eval()
        self.device = device
        self.n_init = n_init
        self.max_age = max_age
        self.iou_gate = iou_gate
        self.reid_weight = reid_weight
        self._transform = build_transforms(crop_size, augment=False)

        self._tracks: List[Track] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        crops: List[np.ndarray],           # BGR crops aligned with detections
    ) -> List[Track]:
        """Run one tracker step. Returns all *confirmed* tracks.

        Args:
            detections: list of Detection objects for the current frame.
            crops:      list of BGR numpy arrays (H×W×3), one per detection.
                        Pass an empty array [] if a crop couldn't be extracted.

        Returns:
            Confirmed tracks with updated bboxes and IDs.
        """
        # 1. Predict new positions for existing tracks
        for t in self._tracks:
            t.predict()

        # 2. Compute ReID embeddings for detections (batch inference)
        det_embeddings = self._embed_crops(crops)

        # 3. Match tracks ↔ detections
        matched, unmatched_tracks, unmatched_dets = self._match(
            self._tracks, detections, det_embeddings
        )

        # 4. Update matched tracks
        for t_idx, d_idx in matched:
            self._tracks[t_idx].update(
                detections[d_idx].bbox,
                det_embeddings[d_idx],
                self.n_init,
            )

        # 5. Mark unmatched tracks as missed
        for t_idx in unmatched_tracks:
            self._tracks[t_idx].mark_missed(self.max_age)

        # 6. Create new tentative tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = Track(
                track_id=next(_id_counter),
                class_id=det.class_id,
                class_name=det.class_name,
                kalman=KalmanBoxTracker(det.bbox),
                embedding=det_embeddings[d_idx],
            )
            self._tracks.append(new_track)

        # 7. Remove deleted tracks
        self._tracks = [t for t in self._tracks if t.state != "deleted"]

        return [t for t in self._tracks if t.state == "confirmed"]

    def reset(self):
        """Clear all tracks (e.g. between scenes)."""
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_crops(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Run ReID model on all crops in one batch. Returns list of unit-norm numpy vectors."""
        if not crops:
            return []

        tensors = []
        valid_idx = []
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                tensors.append(None)
                continue
            # BGR → RGB PIL-like tensor
            rgb = crop[:, :, ::-1].copy()   # BGR → RGB
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb)
            tensors.append(self._transform(pil))
            valid_idx.append(i)

        result: List[Optional[np.ndarray]] = [None] * len(crops)
        if not valid_idx:
            return result

        batch = torch.stack([tensors[i] for i in valid_idx]).to(self.device)
        with torch.no_grad():
            embs = self.reid_model(batch)   # [N, D], already L2-normalized
        embs_np = embs.cpu().numpy()

        for out_i, crop_i in enumerate(valid_idx):
            result[crop_i] = embs_np[out_i]

        return result

    def _match(
        self,
        tracks: List[Track],
        detections: List[Detection],
        det_embeddings: List[Optional[np.ndarray]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian assignment. Returns (matched pairs, unmatched track ids, unmatched det ids)."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build cost matrix [T × D]
        cost = self._cost_matrix(tracks, detections, det_embeddings)

        # Assign using scipy Hungarian (minimizes cost)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_tracks, unmatched_dets = [], [], []

        assigned_tracks = set()
        assigned_dets = set()
        for t_idx, d_idx in zip(row_ind, col_ind):
            if cost[t_idx, d_idx] >= 1e9:   # gated out
                continue
            matched.append((t_idx, d_idx))
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in assigned_tracks]
        unmatched_dets   = [i for i in range(len(detections)) if i not in assigned_dets]

        return matched, unmatched_tracks, unmatched_dets

    def _cost_matrix(
        self,
        tracks: List[Track],
        detections: List[Detection],
        det_embeddings: List[Optional[np.ndarray]],
    ) -> np.ndarray:
        """Build [T × D] cost matrix: combined IoU + ReID distance."""
        T, D = len(tracks), len(detections)
        cost = np.full((T, D), fill_value=1e9, dtype=float)

        track_bboxes = np.array([t.bbox for t in tracks])           # [T, 4]
        det_bboxes   = np.array([d.bbox for d in detections])       # [D, 4]
        iou_dist = 1.0 - _batch_iou(track_bboxes, det_bboxes)      # [T, D]

        # Gate: spatial distance alone
        feasible = iou_dist < self.iou_gate

        for t_idx in range(T):
            for d_idx in range(D):
                if not feasible[t_idx, d_idx]:
                    continue  # leave at 1e9

                iou_cost = iou_dist[t_idx, d_idx]

                # ReID distance (cosine, embeddings already L2-normalized → dot product)
                t_emb = tracks[t_idx].embedding
                d_emb = det_embeddings[d_idx]
                if t_emb is not None and d_emb is not None:
                    cosine_sim = float(np.dot(t_emb, d_emb))
                    reid_cost = (1.0 - cosine_sim) / 2.0   # map [-1,1] → [1,0]
                    cost[t_idx, d_idx] = (
                        (1.0 - self.reid_weight) * iou_cost
                        + self.reid_weight * reid_cost
                    )
                else:
                    # No embedding yet — fall back to IoU only
                    cost[t_idx, d_idx] = iou_cost

        return cost


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _xyxy_to_cxcywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between every pair in boxes_a [M,4] and boxes_b [N,4].

    Returns [M, N] IoU matrix.
    """
    # Expand dims for broadcasting
    a = boxes_a[:, np.newaxis, :]   # [M, 1, 4]
    b = boxes_b[np.newaxis, :, :]   # [1, N, 4]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])   # [M]
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])   # [N]

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    return inter_area / np.maximum(union, 1e-6)


def extract_crop(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """Crop a detection from a BGR frame with optional padding.

    Args:
        frame:   full BGR frame [H, W, 3]
        bbox:    [x1, y1, x2, y2] in pixels
        padding: fractional padding added to each side

    Returns:
        BGR crop (may be empty if bbox is invalid)
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw)
    y2 = min(H, y2 + ph)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]
