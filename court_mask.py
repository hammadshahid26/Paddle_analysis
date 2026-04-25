# """
# court_mask.py
# -------------
# Automatic court ROI polygon via blue-color segmentation.

# Padel courts have a uniform saturated blue playing surface, which
# makes HSV-based segmentation extremely reliable. We build the mask
# once from the median of N sample frames (so players don't occlude
# the floor), find the largest connected region, and convert it to a
# polygon that both person and ball detectors can use.

# No manual clicking. No per-frame compute. Fast, robust, and it
# replaces the make_polygon.py utility entirely.
# """
# from typing import Tuple, Optional

# import cv2
# import numpy as np


# # HSV ranges for "padel blue" playing surface.
# # H: 100-130 (blue hues), S: >= 80 (saturated), V: >= 80 (bright enough)
# DEFAULT_HSV_LOW  = np.array([95,  70,  60], dtype=np.uint8)
# DEFAULT_HSV_HIGH = np.array([130, 255, 255], dtype=np.uint8)


# def build_court_polygon(
#     frames: list,
#     n_samples: int = 15,
#     hsv_low: np.ndarray = DEFAULT_HSV_LOW,
#     hsv_high: np.ndarray = DEFAULT_HSV_HIGH,
#     dilate_px: int = 80,
#     approx_epsilon: float = 0.01,
# ) -> Tuple[Optional[np.ndarray], np.ndarray]:
#     """
#     Build a court ROI polygon from sample frames.

#     Parameters
#     ----------
#     frames        : list of BGR frames (from cv2.VideoCapture)
#     n_samples     : how many evenly-spaced frames to median together
#     hsv_low/high  : HSV thresholds for blue court floor
#     dilate_px     : morphological dilation in pixels — expands ROI
#                     to include the run-up area just outside the lines
#     approx_epsilon: polygon simplification factor
#                     (higher = simpler polygon, fewer vertices)

#     Returns
#     -------
#     polygon : (N, 2) np.ndarray of vertex coords, or None if failed
#     mask    : 2D uint8 mask of the court region (same size as frame)
#     """
#     if len(frames) == 0:
#         return None, np.zeros((720, 1280), dtype=np.uint8)

#     # Sample N frames evenly from the whole clip
#     idxs = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
#     samples = [frames[i] for i in idxs]

#     # Pixel-wise median => floor stays blue, players (transient) get washed out
#     stack = np.stack(samples, axis=0)
#     median_bgr = np.median(stack, axis=0).astype(np.uint8)

#     hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, hsv_low, hsv_high)

#     # Clean up: close small gaps, then dilate for the run-up area
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

#     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

#     # Largest connected component = court floor
#     n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     if n_cc <= 1:
#         return None, mask
#     # skip background (label 0); pick biggest other component
#     best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
#     court_mask = (labels == best).astype(np.uint8) * 255

#     # Dilate to include just-outside-the-lines zone (players' feet can land there)
#     if dilate_px > 0:
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
#         )
#         court_mask = cv2.dilate(court_mask, kernel)

#     # Contour -> convex hull -> simplified polygon
#     contours, _ = cv2.findContours(
#         court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     if not contours:
#         return None, court_mask
#     cnt = max(contours, key=cv2.contourArea)

#     # Convex hull first — keeps shape simple and monotonic
#     hull = cv2.convexHull(cnt)
#     peri = cv2.arcLength(hull, closed=True)
#     approx = cv2.approxPolyDP(hull, approx_epsilon * peri, closed=True)
#     polygon = approx.reshape(-1, 2)

#     return polygon, court_mask




"""
court_mask.py
-------------
Automatic court ROI polygon via blue-color segmentation.
Plus: detect_court_lines() for finding net + service lines via white-pixel
detection along vertical scanlines (camera-agnostic zone calibration).
"""
from typing import Tuple, Optional, Dict

import cv2
import numpy as np


# HSV ranges for "padel blue" playing surface
DEFAULT_HSV_LOW  = np.array([95,  70,  60], dtype=np.uint8)
DEFAULT_HSV_HIGH = np.array([130, 255, 255], dtype=np.uint8)


def build_court_polygon(
    frames: list,
    n_samples: int = 15,
    hsv_low: np.ndarray = DEFAULT_HSV_LOW,
    hsv_high: np.ndarray = DEFAULT_HSV_HIGH,
    dilate_px: int = 100,
    approx_epsilon: float = 0.01,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if len(frames) == 0:
        return None, np.zeros((720, 1280), dtype=np.uint8)

    idxs = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    samples = [frames[i] for i in idxs]
    stack = np.stack(samples, axis=0)
    median_bgr = np.median(stack, axis=0).astype(np.uint8)

    hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_cc <= 1:
        return None, mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    court_mask = (labels == best).astype(np.uint8) * 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        court_mask = cv2.dilate(court_mask, kernel)

    contours, _ = cv2.findContours(
        court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, court_mask
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, closed=True)
    approx = cv2.approxPolyDP(hull, approx_epsilon * peri, closed=True)
    polygon = approx.reshape(-1, 2)

    return polygon, court_mask


def _build_median_frame(frames: list, n_samples: int = 15) -> np.ndarray:
    """Same median construction as polygon — players average out."""
    idxs = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    samples = [frames[i] for i in idxs]
    stack = np.stack(samples, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def detect_court_lines(
    frames: list,
    floor_polygon: np.ndarray,
    n_samples: int = 15,
) -> Optional[Dict[str, float]]:
    """
    Detect horizontal court lines (net + service lines) by scanning for
    white pixels within the court polygon on the median frame.

    The median frame removes players, so what's left is just the court.
    White pixels concentrate along the painted lines (and the net).

    Returns dict with:
        net_y           : Y of the net (the strongest horizontal feature)
        far_service_y   : Y of the far service line
        near_service_y  : Y of the near service line, if visible
        far_back_y      : Y of the far baseline (top of polygon)
        near_back_y     : Y of the near baseline (bottom of polygon)

    Returns None if detection fails (caller falls back to ratio method).
    """
    if floor_polygon is None or len(frames) == 0:
        return None

    median = _build_median_frame(frames, n_samples)
    H, W = median.shape[:2]

    bp = np.asarray(floor_polygon, dtype=np.int32)
    poly_top = int(bp[:, 1].min())
    poly_bot = int(bp[:, 1].max())

    # Mask: only consider pixels INSIDE the court polygon
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [bp], 255)

    # Convert to grayscale, find white pixels (high V in HSV)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    # White pixels: low saturation + high value
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_mask = ((sat < 60) & (val > 160)).astype(np.uint8) * 255
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=poly_mask)

    # For each Y row in the polygon range, count white pixels
    row_white_counts = np.zeros(poly_bot - poly_top + 1, dtype=np.int32)
    for i, y in enumerate(range(poly_top, poly_bot + 1)):
        row_white_counts[i] = int(np.sum(white_mask[y, :] > 0))

    if row_white_counts.max() < 5:
        # not enough white pixels detected anywhere — fail
        return None

    # Smooth the row-counts to merge adjacent bright rows
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(row_white_counts, kernel, mode='same')

    # Find peaks: local maxima above some threshold
    threshold = max(8, smoothed.max() * 0.25)
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if smoothed[i] < threshold:
            continue
        # local max within +/- 2
        if (smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i-2]
            and smoothed[i] >= smoothed[i+1] and smoothed[i] >= smoothed[i+2]):
            peaks.append((poly_top + i, float(smoothed[i])))

    if len(peaks) < 2:
        return None

    # Sort peaks by strength (white-pixel count), descending
    peaks_by_strength = sorted(peaks, key=lambda p: -p[1])

    # The NET is typically the strongest horizontal feature in the upper
    # portion of the polygon (it's a continuous white band, while service
    # lines are thinner). Pick the strongest peak in the top 40% of the
    # polygon as the net.
    poly_depth = poly_bot - poly_top
    net_candidates = [p for p in peaks_by_strength
                      if (p[0] - poly_top) <= 0.4 * poly_depth]
    if not net_candidates:
        return None
    net_y = float(net_candidates[0][0])

    # Service lines: peaks below the net
    below_net = [p for p in peaks_by_strength if p[0] > net_y + 15]
    if not below_net:
        return None

    # Near service line: pick the strongest peak between the net and the
    # bottom of the polygon, ideally around the middle of the near half
    near_half_depth = poly_bot - net_y
    # Pick the peak with strongest signal whose position is in the
    # 25-75% range of the near half (avoids the very bottom edge)
    valid_near = [p for p in below_net
                  if 0.25 * near_half_depth < (p[0] - net_y) < 0.85 * near_half_depth]
    near_service_y = None
    if valid_near:
        near_service_y = float(sorted(valid_near, key=lambda p: -p[1])[0][0])

    # Far service line: peaks ABOVE the net (between far baseline and net)
    above_net = [p for p in peaks if p[0] < net_y - 5]
    far_half_depth = net_y - poly_top
    valid_far = [p for p in above_net
                 if 0.25 * far_half_depth < (net_y - p[0]) < 0.85 * far_half_depth]
    far_service_y = None
    if valid_far:
        far_service_y = float(sorted(valid_far, key=lambda p: -p[1])[0][0])

    return {
        'net_y'         : net_y,
        'far_service_y' : far_service_y,
        'near_service_y': near_service_y,
        'far_back_y'    : float(poly_top),
        'near_back_y'   : float(poly_bot),
    }