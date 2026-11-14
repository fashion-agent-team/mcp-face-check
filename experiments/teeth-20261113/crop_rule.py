"""The v2 mouth crop rule, shared by training-data prep and inference.

Given the inner-lip contour (the mouth opening) in image pixels:
  1. take its axis-aligned bounding box (x0, y0, x1, y1)
  2. side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN), MARGIN = 0.18
  3. square of that side centred on the box centre
  4. resample to OUT x OUT (384) with bilinear; outside the image: reflect

The app / server should do the same with ML Kit's lip contours: the inner
lip is UPPER_LIP_BOTTOM + LOWER_LIP_TOP. The model then resizes 384 -> 224/320
itself (see the model's .json), so any OUT >= the model size works.
"""
import cv2
import numpy as np

MARGIN = 0.18
OUT = 384


def crop_box(inner_pts, margin=MARGIN):
    """(left, top, side) of the square crop for these inner-lip points."""
    p = np.asarray(inner_pts, np.float32)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    side = float(max(x1 - x0, y1 - y0) * (1 + 2 * margin))
    return float((x0 + x1) / 2 - side / 2), float((y0 + y1) / 2 - side / 2), side


def crop(img, inner_pts, margin=MARGIN, out=OUT):
    """Crop + resize. Returns the crop and the 2x3 affine image -> crop."""
    left, top, side = crop_box(inner_pts, margin)
    s = out / side
    m = np.array([[s, 0, -left * s], [0, s, -top * s]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    res = cv2.warpAffine(img, m, (out, out), flags=interp, borderMode=cv2.BORDER_REFLECT)
    return res, m


def apply(m, pts):
    p = np.asarray(pts, np.float32)
    return p @ m[:, :2].T + m[:, 2]
