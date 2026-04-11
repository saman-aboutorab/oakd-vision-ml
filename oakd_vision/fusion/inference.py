"""P3 Traversability CNN — single-frame inference.

TraversabilityPredictor wraps TraversabilityNet and exposes a simple API:
    predictor.predict(bgr_frame, depth_mm) -> grid of class labels + probabilities

This is the building block for live_traversability.py.

Example:
    from oakd_vision.fusion.inference import TraversabilityPredictor

    predictor = TraversabilityPredictor("runs/fusion/concat/best.pt")
    grid_labels, grid_probs = predictor.predict(bgr, depth_mm)
    # grid_labels: (grid_rows, grid_cols) int array
    # grid_probs:  (grid_rows, grid_cols, num_classes) float array
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from oakd_vision.fusion.fusion_model import TraversabilityNet
from oakd_vision.fusion.traversability_dataset import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ---------------------------------------------------------------------------
# Colour palette for the overlay (BGR for OpenCV)
# free=green, caution=yellow, obstacle=red, unknown=grey
# ---------------------------------------------------------------------------
CLASS_COLOURS_BGR = {
    0: (50,  200, 50),    # free     — green
    1: (0,   200, 230),   # caution  — yellow
    2: (50,  50,  220),   # obstacle — red
    3: (140, 140, 140),   # unknown  — grey
}
CLASS_ALPHA = 0.40        # transparency of the coloured overlay


class TraversabilityPredictor:
    """Load a trained checkpoint and run per-patch inference on a full frame.

    Args:
        checkpoint:      Path to best.pt (or last.pt).
        config:          Path to fusion_config.yaml.  If None, uses defaults.
        fusion_strategy: Override the strategy name (needed if config is absent).
        device:          "cuda" / "cpu" / "auto" (default).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path | None = None,
        fusion_strategy: str | None = None,
        device: str = "auto",
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        # --- Config ---
        if config is not None:
            with open(config) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {
                "data":  {"patch_size": [64, 64], "grid_cols": 8, "grid_rows": 6,
                          "depth_max_mm": 4000},
                "model": {"embedding_dim": 256, "num_classes": 4,
                          "dropout": 0.3, "fusion_strategy": "concat"},
            }

        self.patch_size  = tuple(cfg["data"]["patch_size"])    # (H, W)
        self.grid_cols   = cfg["data"]["grid_cols"]
        self.grid_rows   = cfg["data"]["grid_rows"]
        self.depth_max   = cfg["data"]["depth_max_mm"]

        strategy = fusion_strategy or cfg["model"]["fusion_strategy"]

        # --- Device ---
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- Model ---
        self.model = TraversabilityNet(
            embedding_dim   = cfg["model"]["embedding_dim"],
            num_classes     = cfg["model"]["num_classes"],
            fusion_strategy = strategy,
            dropout         = cfg["model"]["dropout"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # --- Transforms ---
        ph, pw = self.patch_size
        self._rgb_tf = transforms.Compose([
            transforms.Resize((ph, pw)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        print(f"TraversabilityPredictor ready — strategy={strategy} "
              f"device={self.device} checkpoint={checkpoint}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, bgr: np.ndarray, depth_mm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on one full frame.

        Args:
            bgr:       Full camera frame, shape (H, W, 3), uint8, BGR.
            depth_mm:  Aligned depth frame, shape (H, W), uint16, mm.

        Returns:
            grid_labels: (grid_rows, grid_cols) int32 — class index per cell.
            grid_probs:  (grid_rows, grid_cols, num_classes) float32 — softmax probs.
        """
        img_h, img_w = bgr.shape[:2]
        ph, pw = self.patch_size

        rgb_batch   = []
        depth_batch = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                y1 = int(row * img_h / self.grid_rows)
                y2 = int((row + 1) * img_h / self.grid_rows)
                x1 = int(col * img_w / self.grid_cols)
                x2 = int((col + 1) * img_w / self.grid_cols)

                # RGB patch
                rgb_crop = bgr[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)
                pil_rgb  = Image.fromarray(rgb_crop)
                rgb_batch.append(self._rgb_tf(pil_rgb))

                # Depth patch
                d_crop   = depth_mm[y1:y2, x1:x2].astype(np.float32)
                d_resized = cv2.resize(d_crop, (pw, ph), interpolation=cv2.INTER_NEAREST)
                d_norm   = np.clip(d_resized, 0, self.depth_max) / self.depth_max
                depth_batch.append(
                    torch.from_numpy(d_norm).unsqueeze(0).float()
                )

        rgb_t   = torch.stack(rgb_batch).to(self.device)     # [N, 3, H, W]
        depth_t = torch.stack(depth_batch).to(self.device)   # [N, 1, H, W]

        logits = self.model(rgb_t, depth_t)                   # [N, num_classes]
        probs  = torch.softmax(logits, dim=1).cpu().numpy()   # [N, num_classes]
        preds  = probs.argmax(axis=1)                         # [N]

        N = self.grid_rows * self.grid_cols
        grid_labels = preds.reshape(self.grid_rows, self.grid_cols).astype(np.int32)
        grid_probs  = probs.reshape(self.grid_rows, self.grid_cols, NUM_CLASSES)

        return grid_labels, grid_probs

    # ------------------------------------------------------------------
    # Overlay helper
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        bgr: np.ndarray,
        grid_labels: np.ndarray,
        grid_probs: np.ndarray,
        alpha: float = CLASS_ALPHA,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw coloured grid overlay on the frame.

        Args:
            bgr:              Original BGR frame.
            grid_labels:      (grid_rows, grid_cols) int — class per cell.
            grid_probs:       (grid_rows, grid_cols, num_classes) float — softmax probs.
            alpha:            Overlay transparency (0 = invisible, 1 = opaque).
            show_confidence:  Print confidence % inside each cell.

        Returns:
            Annotated BGR frame (same shape as input).
        """
        out    = bgr.copy()
        overlay = bgr.copy()
        h, w   = bgr.shape[:2]

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                y1 = int(row * h / self.grid_rows)
                y2 = int((row + 1) * h / self.grid_rows)
                x1 = int(col * w / self.grid_cols)
                x2 = int((col + 1) * w / self.grid_cols)

                cls   = int(grid_labels[row, col])
                conf  = float(grid_probs[row, col, cls])
                colour = CLASS_COLOURS_BGR[cls]

                cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)

                if show_confidence:
                    label_char = INT_TO_LABEL[cls][0].upper()  # F/C/O/U
                    text = f"{label_char} {conf:.0%}"
                    cv2.putText(out, text,
                                (x1 + 4, y1 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                                colour, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        return out

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Draw a small legend in the bottom-left corner."""
        out = frame.copy()
        y0  = frame.shape[0] - 10
        entries = [
            (0, "free"),
            (1, "caution"),
            (2, "obstacle"),
            (3, "unknown"),
        ]
        for cls, name in reversed(entries):
            colour = CLASS_COLOURS_BGR[cls]
            cv2.rectangle(out, (8, y0 - 12), (20, y0), colour, -1)
            cv2.putText(out, name, (24, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y0 -= 18
        return out
