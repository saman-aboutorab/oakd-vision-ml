"""Utilities for extracting reliable depth values from OAK-D stereo depth maps.

OAK-D stereo output is a uint16 frame where each pixel is depth in millimetres.
Invalid/occluded pixels are 0. Background objects can pollute the edges of a
bounding box, so we crop to the inner 50% of the bbox before computing depth.
"""

import numpy as np


def get_depth_for_bbox(
    depth_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_valid_mm: int = 300,
    max_valid_mm: int = 10_000,
    percentile: float = 50.0,
) -> float:
    """Estimate the depth (metres) of an object within a bounding box.

    Uses the inner 50% of the bbox area to reduce contamination from
    background pixels that often appear at detection boundaries.

    Args:
        depth_map: HxW uint16 array, values in millimetres (0 = invalid)
        x1, y1, x2, y2: bounding box pixel corners
        min_valid_mm: pixels below this value are discarded (too close)
        max_valid_mm: pixels above this value are discarded (too far / noise)
        percentile: percentile to compute on valid pixels (50 = median)

    Returns:
        Depth in metres, or float('nan') if no valid pixels are found.
    """
    w, h = x2 - x1, y2 - y1

    # Inner 50% crop: shrink each side by 25%
    cx1 = x1 + w // 4
    cx2 = x2 - w // 4
    cy1 = y1 + h // 4
    cy2 = y2 - h // 4

    region = depth_map[cy1:cy2, cx1:cx2]
    valid = region[(region >= min_valid_mm) & (region <= max_valid_mm)]

    if valid.size == 0:
        return float("nan")

    return float(np.percentile(valid, percentile)) / 1000.0  # mm → m
