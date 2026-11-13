"""Re-crop the kept faces with the v2 rule (crop_rule.py) from the originals.

Uses the same selection as v1: every id still in mouths/ (passed
crop_mouths.py and flag_crops.py), so v1 and v2 cover the same faces.
Landmarks are detected again on the 1024px original, and the crop is taken
from it directly (no upscaling of the v1 crops).

Output: mouths_tight/<id>.png (384x384) + <id>.json (inner/outer lip in crop
coordinates, same keys as mouths/), and tight_sheet.jpg to eyeball.

Usage:
    .venv/bin/python crop_tight.py [--src ~/Downloads/ffhq-subset]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import crop_mouths as cm
import crop_rule

OUT_DIR = Path(__file__).resolve().parent
SRC_SEL = OUT_DIR / "mouths"
DST = OUT_DIR / "mouths_tight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=cm.DEFAULT_SRC)
    args = ap.parse_args()
    DST.mkdir(exist_ok=True)
    ids = sorted(p.stem for p in SRC_SEL.glob("*.json"))
    todo = [i for i in ids if not (DST / f"{i}.json").exists()]
    det = cm.Detector()
    failed = []
    for k, mid in enumerate(todo, 1):
        bgr = cv2.imread(str(args.src / f"{mid}.png"))
        pts = det.landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if bgr is not None else None
        if pts is None:
            failed.append(mid)
            continue
        out, m = crop_rule.crop(bgr, pts[cm.INNER])
        old = json.loads((SRC_SEL / f"{mid}.json").read_text())
        cv2.imwrite(str(DST / f"{mid}.png"), out)
        (DST / f"{mid}.json").write_text(json.dumps({
            "id": mid, "size": crop_rule.OUT, "crop_rule": "inner_lip_bbox",
            "margin": crop_rule.MARGIN,
            "inner_lip": crop_rule.apply(m, pts[cm.INNER]).round(2).tolist(),
            "outer_lip": crop_rule.apply(m, pts[cm.OUTER]).round(2).tolist(),
            "open_ratio": old["open_ratio"], "teeth_share": old["teeth_share"],
        }))
        if k % 200 == 0:
            print(f"{k}/{len(todo)}", flush=True)
    print(f"cropped {len(todo) - len(failed)} of {len(todo)} new, "
          f"total {len(list(DST.glob('*.json')))}; no face now: {failed}")

    # Sheet of 36 to eyeball the framing.
    ids = sorted(p.stem for p in DST.glob("*.json"))
    rng = np.random.default_rng(0)
    pick = rng.choice(ids, min(36, len(ids)), replace=False)
    t = 192
    sheet = np.full((6 * t, 6 * t, 3), 40, np.uint8)
    for k, mid in enumerate(pick):
        img = cv2.imread(str(DST / f"{mid}.png"))
        meta = json.loads((DST / f"{mid}.json").read_text())
        cv2.polylines(img, [np.array(meta["inner_lip"]).round().astype(np.int32)],
                      True, (0, 255, 0), 2)
        tile = cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, mid, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        r, c = divmod(k, 6)
        sheet[r * t:(r + 1) * t, c * t:(c + 1) * t] = tile
    cv2.imwrite(str(OUT_DIR / "tight_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
