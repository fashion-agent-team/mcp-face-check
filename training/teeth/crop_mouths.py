"""Crop mouths with visible teeth from FFHQ faces.

For each PNG in the source folder, find lip landmarks with MediaPipe FaceMesh,
keep faces whose mouth is open with bright, low-saturation pixels (teeth)
between the inner lips, and save a 384x384 mouth crop plus the inner-lip
polygon (in crop coordinates) as JSON.

Re-runnable: every processed id (kept or rejected) is logged to
processed.jsonl with its metrics, and is skipped next time. Use --recheck to
re-apply new thresholds to logged metrics without re-running detection
(only works for ids whose crop was saved, i.e. that passed the opening check).

Usage:
    .venv/bin/python crop_mouths.py [--src DIR] [--open 0.12] [--teeth 0.25]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
DEFAULT_SRC = HOME / "Downloads" / "ffhq-subset"
OUT_DIR = Path(__file__).resolve().parent
MOUTH_DIR = OUT_DIR / "mouths"
LOG = OUT_DIR / "processed.jsonl"
MODEL = OUT_DIR / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

CROP = 384
MARGIN = 0.25
# FaceMesh lip contours, in order around the polygon.
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
         409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
         415, 310, 311, 312, 13, 82, 81, 80, 191]
UPPER_INNER, LOWER_INNER = 13, 14
LEFT_CORNER, RIGHT_CORNER = 61, 291


class Detector:
    """FaceMesh via the legacy solutions API, or the Tasks API as fallback."""

    def __init__(self):
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)
            self.kind = "solutions"
        else:
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            if not MODEL.exists():
                import urllib.request
                urllib.request.urlretrieve(MODEL_URL, MODEL)
            opts = vision.FaceLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
                num_faces=1)
            self.mesh = vision.FaceLandmarker.create_from_options(opts)
            self.mp = mp
            self.kind = "tasks"

    def landmarks(self, rgb):
        """Return an (N, 2) array of pixel coordinates, or None."""
        h, w = rgb.shape[:2]
        if self.kind == "solutions":
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0].landmark
        else:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.mesh.detect(img)
            if not res.face_landmarks:
                return None
            lms = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)


def teeth_share(bgr, inner, outer):
    """Share of pixels inside the inner lips that look like teeth.

    Compared against this face's own lips (the ring between outer and inner
    contours), so shade and warm lighting don't matter: teeth are less red
    (lower LAB a*) and at least about as light as the lips, while the dark
    mouth gap and the tongue are not.
    """
    h, w = bgr.shape[:2]
    m_in = np.zeros((h, w), np.uint8)
    m_out = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m_in, [inner.astype(np.int32)], 255)
    cv2.fillPoly(m_out, [outer.astype(np.int32)], 255)
    inside, lips = m_in > 0, (m_out > 0) & (m_in == 0)
    n = int(inside.sum())
    if n == 0 or lips.sum() == 0:
        return 0.0, n
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_l, lip_a = np.median(lab[..., 0][lips]), np.median(lab[..., 1][lips])
    l, a = lab[..., 0][inside], lab[..., 1][inside]
    teeth = (a < lip_a - 6) & (l > lip_l * 0.9)
    return float(teeth.mean()), n


def square_crop(bgr, outer):
    """Square box around the outer lips with MARGIN on each side."""
    x0, y0 = outer.min(0)
    x1, y1 = outer.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = cx - side / 2, cy - side / 2
    scale = CROP / side
    # Affine warp handles boxes that run past the image edge.
    m = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], np.float32)
    crop = cv2.warpAffine(bgr, m, (CROP, CROP), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REFLECT)
    return crop, m


def png_complete(path):
    try:
        with open(path, "rb") as f:
            f.seek(-8, os.SEEK_END)
            return f.read(8)[:4] == b"IEND"
    except OSError:
        return False


def load_log():
    done = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--open", type=float, default=0.10,
                    help="min inner-lip opening height / mouth width")
    ap.add_argument("--teeth", type=float, default=0.20,
                    help="min share of bright low-saturation pixels in the opening")
    ap.add_argument("--recheck", action="store_true",
                    help="re-apply thresholds to logged metrics, delete crops that now fail")
    ap.add_argument("--limit", type=int, default=0, help="process at most N new files")
    args = ap.parse_args()
    MOUTH_DIR.mkdir(parents=True, exist_ok=True)
    done = load_log()

    if args.recheck:
        kept = 0
        for rec in done.values():
            ok = (rec.get("open_ratio", 0) >= args.open and
                  rec.get("teeth_share", 0) >= args.teeth)
            if not ok:
                for ext in (".png", ".json"):
                    (MOUTH_DIR / (rec["id"] + ext)).unlink(missing_ok=True)
            kept += ok and (MOUTH_DIR / (rec["id"] + ".png")).exists()
        print(f"recheck: {kept} kept of {len(done)} logged")
        return

    files = sorted(p for p in args.src.glob("*.png") if p.stem not in done)
    if args.limit:
        files = files[:args.limit]
    det = Detector()
    stats = {"kept": 0, "no_face": 0, "closed": 0, "no_teeth": 0, "bad_file": 0}
    with LOG.open("a") as log:
        for i, path in enumerate(files, 1):
            if not png_complete(path):
                stats["bad_file"] += 1  # still downloading; retry next run
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                stats["bad_file"] += 1
                continue
            rec = {"id": path.stem}
            pts = det.landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if pts is None:
                rec["status"] = "no_face"
            else:
                width = np.linalg.norm(pts[LEFT_CORNER] - pts[RIGHT_CORNER])
                opening = np.linalg.norm(pts[UPPER_INNER] - pts[LOWER_INNER])
                rec["open_ratio"] = round(float(opening / width), 4)
                inner = pts[INNER]
                share, area = teeth_share(bgr, inner, pts[OUTER])
                rec["teeth_share"] = round(share, 4)
                if rec["open_ratio"] < args.open:
                    rec["status"] = "closed"
                elif share < args.teeth:
                    rec["status"] = "no_teeth"
                else:
                    rec["status"] = "kept"
                    crop, m = square_crop(bgr, pts[OUTER])
                    to_crop = lambda p: (p @ m[:, :2].T + m[:, 2]).round(2).tolist()
                    cv2.imwrite(str(MOUTH_DIR / f"{path.stem}.png"), crop)
                    (MOUTH_DIR / f"{path.stem}.json").write_text(json.dumps({
                        "id": path.stem, "size": CROP,
                        "inner_lip": to_crop(inner),
                        "outer_lip": to_crop(pts[OUTER]),
                        "open_ratio": rec["open_ratio"],
                        "teeth_share": rec["teeth_share"],
                    }))
            stats[rec["status"]] += 1
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if i % 100 == 0:
                print(f"{i}/{len(files)} {stats}", flush=True)

    total_kept = sum(1 for r in load_log().values() if r["status"] == "kept")
    print(f"this run: {sum(stats.values())} files {stats}")
    print(f"total kept so far: {total_kept} of {len(load_log())} processed")


if __name__ == "__main__":
    sys.exit(main())
