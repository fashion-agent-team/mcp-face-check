"""Flag mouth crops that would make bad "clean teeth" examples.

Two checks inside the inner-lip polygon, both relative to this face's own
lips (ring between outer and inner contours) and skin (crop corners):
  - tongue: a large share of light, lip-red pixels in a wide-open mouth
  - occluder: a large share of skin-coloured pixels (finger, hand) or of
    very dark pixels with few teeth (microphone, object in front)

Flagged crops are MOVED (png + json) to mouths_flagged/, ids listed with
--keep-real go to real_food/ instead. Writes flagged_sheet.jpg and
flags.jsonl with the features of every crop.

Usage:
    .venv/bin/python flag_crops.py [--dry-run] [--keep-real 00449]
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
MOUTH_DIR = OUT_DIR / "mouths"
FLAG_DIR = OUT_DIR / "mouths_flagged"
REAL_DIR = OUT_DIR / "real_food"

TONGUE_MAX = 0.30
TONGUE_OPEN = 0.33  # gums/lower lip in a smile also read as "tongue"
SKIN_MAX = 0.30
DARK_MAX = 0.55


def features(bgr, meta):
    size = bgr.shape[0]
    inner = np.array(meta["inner_lip"], np.int32)
    outer = np.array(meta["outer_lip"], np.int32)
    m_in = np.zeros((size, size), np.uint8)
    m_out = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m_in, [inner], 255)
    cv2.fillPoly(m_out, [outer], 255)
    inside, lips = m_in > 0, (m_out > 0) & (m_in == 0)
    # Skin: the crop's lower corners, below the lip line and beside the chin.
    skin = np.zeros((size, size), bool)
    c = size // 8
    skin[-2 * c:-c, c:2 * c] = True
    skin[-2 * c:-c, -2 * c:-c] = True
    skin &= m_out == 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    lip_l, lip_a = np.median(L[lips]), np.median(A[lips])
    sk = np.median(lab[skin], axis=0)
    l, a, b = L[inside], A[inside], B[inside]
    tongue = (a >= lip_a - 4) & (l > lip_l * 0.95)
    skin_px = np.sqrt((l - sk[0]) ** 2 * 0.25 + (a - sk[1]) ** 2 + (b - sk[2]) ** 2) < 7
    dark = l < 0.45 * lip_l
    return {
        "tongue": round(float(tongue.mean()), 3),
        "skin": round(float(skin_px.mean()), 3),
        "dark": round(float(dark.mean()), 3),
    }


def reasons(f, meta):
    r = []
    if f["tongue"] > TONGUE_MAX and meta["open_ratio"] > TONGUE_OPEN:
        r.append("tongue")
    if f["skin"] > SKIN_MAX:
        r.append("skin")
    if f["dark"] > DARK_MAX and meta["teeth_share"] < 0.4:
        r.append("dark")
    return r


def sheet(items, path, grid=6, tile=192):
    rows = max(1, (len(items[:grid * grid]) + grid - 1) // grid)
    out = np.full((rows * tile, grid * tile, 3), 40, np.uint8)
    for k, (img, meta, label) in enumerate(items[:grid * grid]):
        img = img.copy()
        poly = np.array(meta["inner_lip"], np.float32).round().astype(np.int32)
        cv2.polylines(img, [poly], True, (0, 0, 255), 2)
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, grid)
        out[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def move(mid, dest):
    dest.mkdir(exist_ok=True)
    for ext in (".png", ".json"):
        shutil.move(str(MOUTH_DIR / (mid + ext)), str(dest / (mid + ext)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-real", nargs="*", default=["00449"],
                    help="ids with real food: moved to real_food/")
    args = ap.parse_args()

    flagged, log = [], []
    for jp in sorted(MOUTH_DIR.glob("*.json")):
        meta = json.loads(jp.read_text())
        img = cv2.imread(str(jp.with_suffix(".png")))
        f = features(img, meta)
        r = reasons(f, meta)
        log.append({"id": meta["id"], **f, "flags": r})
        if r or meta["id"] in args.keep_real:
            tag = "real_food" if meta["id"] in args.keep_real else "+".join(r)
            label = f"{meta['id']} {tag} t{f['tongue']:.2f} s{f['skin']:.2f} d{f['dark']:.2f}"
            flagged.append((meta["id"], tag, img, meta, label))

    (OUT_DIR / "flags.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    sheet([(i, m, l) for _, _, i, m, l in flagged], OUT_DIR / "flagged_sheet.jpg")
    counts = {}
    for _, tag, *_ in flagged:
        counts[tag] = counts.get(tag, 0) + 1
    print(f"checked {len(log)}, flagged {len(flagged)}: {counts}")
    if not args.dry_run:
        for mid, tag, *_ in flagged:
            move(mid, REAL_DIR if tag == "real_food" else FLAG_DIR)
        print(f"moved; clean left: {len(list(MOUTH_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
