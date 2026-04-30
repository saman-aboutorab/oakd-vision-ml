"""P3 Traversability CNN — FastAPI REST endpoint.

Accepts a paired RGB + depth image upload, runs the trained gated fusion model,
and returns:
  - JSON: per-cell class labels and confidence scores
  - PNG:  annotated overlay image (downloadable)

This lets anyone test the model without owning an OAK-D camera.

Usage (dev):
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /predict          — upload rgb + depth files → JSON grid
    POST /predict/overlay  — upload rgb + depth files → PNG image
    GET  /health           — liveness check
    GET  /docs             — auto-generated Swagger UI
"""

import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from oakd_vision.fusion.inference import TraversabilityPredictor
from oakd_vision.fusion.traversability_dataset import INT_TO_LABEL, NUM_CLASSES

# ---------------------------------------------------------------------------
# App + model setup
# ---------------------------------------------------------------------------

CHECKPOINT = Path(os.getenv("CHECKPOINT", "runs/fusion/gated_f3/best.pt"))
CONFIG     = Path(os.getenv("CONFIG",     "training/configs/fusion_config.yaml"))

app = FastAPI(
    title       = "Traversability CNN API",
    description = (
        "RGB-D traversability prediction using a dual-branch CNN "
        "(ResNet18 RGB + custom depth branch, gated fusion). "
        "Upload a 640×480 RGB image and matching depth map → "
        "get per-patch traversability labels."
    ),
    version     = "1.0.0",
)

# Load model once at startup
_predictor: TraversabilityPredictor | None = None

@app.on_event("startup")
def load_model():
    global _predictor
    if not CHECKPOINT.exists():
        print(f"WARNING: checkpoint not found at {CHECKPOINT}. /predict will fail.")
        return
    _predictor = TraversabilityPredictor(checkpoint=CHECKPOINT, config=CONFIG)
    print(f"Model ready — checkpoint={CHECKPOINT}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class CellResult(BaseModel):
    row:        int
    col:        int
    label:      str
    confidence: float
    probs:      dict[str, float]

class PredictResponse(BaseModel):
    grid_rows:  int
    grid_cols:  int
    cells:      list[CellResult]
    summary:    dict[str, int]   # label → count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode RGB image. Send a valid JPEG or PNG.")
    return img  # BGR uint8


def decode_depth(data: bytes) -> np.ndarray:
    """Accept either a .npy file (uint16 mm) or a 16-bit PNG."""
    if data[:6] == b'\x93NUMPY':
        # NumPy .npy format
        arr = np.load(io.BytesIO(data))
    else:
        # 16-bit PNG
        buf = np.frombuffer(data, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH)
    if arr is None or arr.ndim != 2:
        raise HTTPException(400, "Could not decode depth. Send a .npy (uint16) or 16-bit PNG.")
    return arr.astype(np.uint16)


def build_response(grid_labels, grid_probs) -> PredictResponse:
    rows, cols = grid_labels.shape
    cells = []
    summary: dict[str, int] = {v: 0 for v in INT_TO_LABEL.values()}

    for r in range(rows):
        for c in range(cols):
            cls  = int(grid_labels[r, c])
            lbl  = INT_TO_LABEL[cls]
            conf = float(grid_probs[r, c, cls])
            summary[lbl] += 1
            cells.append(CellResult(
                row        = r,
                col        = c,
                label      = lbl,
                confidence = round(conf, 4),
                probs      = {
                    INT_TO_LABEL[i]: round(float(grid_probs[r, c, i]), 4)
                    for i in range(NUM_CLASSES)
                },
            ))

    return PredictResponse(grid_rows=rows, grid_cols=cols, cells=cells, summary=summary)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictResponse, summary="Predict traversability grid (JSON)")
async def predict(
    rgb:   UploadFile = File(..., description="RGB image — JPEG or PNG, 640×480"),
    depth: UploadFile = File(..., description="Depth map — .npy uint16 mm, or 16-bit PNG"),
):
    """Upload a paired RGB + depth frame and get back a JSON traversability grid.

    Each cell in the 8×6 grid has a label (free/caution/obstacle/unknown)
    and per-class confidence scores.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded — check CHECKPOINT path.")

    bgr      = decode_rgb(await rgb.read())
    depth_mm = decode_depth(await depth.read())

    if bgr.shape[:2] != depth_mm.shape[:2]:
        raise HTTPException(
            400,
            f"RGB shape {bgr.shape[:2]} and depth shape {depth_mm.shape} must match."
        )

    grid_labels, grid_probs = _predictor.predict(bgr, depth_mm)
    return build_response(grid_labels, grid_probs)


@app.post("/predict/overlay", summary="Predict traversability grid (PNG overlay)")
async def predict_overlay(
    rgb:   UploadFile = File(..., description="RGB image — JPEG or PNG, 640×480"),
    depth: UploadFile = File(..., description="Depth map — .npy uint16 mm, or 16-bit PNG"),
):
    """Same as /predict but returns the annotated frame as a PNG image.

    Useful for quick visual inspection — paste the URL into a browser or curl it.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded — check CHECKPOINT path.")

    bgr      = decode_rgb(await rgb.read())
    depth_mm = decode_depth(await depth.read())

    grid_labels, grid_probs = _predictor.predict(bgr, depth_mm)
    annotated = _predictor.draw_overlay(bgr, grid_labels, grid_probs, show_confidence=True)
    annotated = _predictor.draw_legend(annotated)

    _, buf = cv2.imencode(".png", annotated)
    return Response(content=buf.tobytes(), media_type="image/png")
