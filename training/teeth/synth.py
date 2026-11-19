"""Make a balanced clean / food-stuck dataset from the mouth crops.

For every crop in --src (default mouths_tight/) this writes --variants clean
images (label 0) and as many images with 1-3 procedurally drawn food bits on the
teeth (label 1). Clean copies of a face differ only in augmentation. All go through
the same random augmentation, so the classifier can't tell them apart by
brightness, blur or JPEG artefacts. Faces are split 80/20 by source id, so no
face is in both train and val (same face split as v1).

Output:
    <out>/{train,val}/{clean,food}/<id>_<k>.jpg
    <out>/labels.json   one record per image, with food bit boxes
    <out>/meta.json     crop rule, for train.py / eval

Usage:
    .venv/bin/python synth.py [--seed 0] [--val 0.2] [--variants 2] [--src mouths_tight] [--out dataset_v2]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# colours: BGR palette, one picked per bit and jittered
# size: blob base diameter as a share of mouth width; elong: length/width
# stretch (a strand of size 0.07, elong 8 is about 0.2 x 0.025 mouth widths)
# alpha: opacity range (rice is translucent); on_teeth: share that must
# overlap teeth (strands and big leaves may trail over the gap)
FOOD = {
    "laver": dict(colors=[(38, 52, 30), (30, 40, 28), (45, 62, 42)],
                  size=(0.04, 0.08), elong=(1.8, 3.5), shape="flake"),
    # v3: wider red range (bright orange-red flakes to dark chili skin), bigger and
    # longer bits, since v2 missed a real flake wedged between the front teeth.
    "red_pepper": dict(colors=[(25, 55, 175), (20, 70, 190), (30, 45, 150), (35, 90, 205),
                               (20, 60, 215), (30, 40, 120), (40, 100, 225), (25, 35, 185)],
                       size=(0.03, 0.08), elong=(1.0, 3.0), shape="flake"),
    "sesame": dict(colors=[(140, 180, 210), (120, 160, 195), (160, 195, 220)],
                   size=(0.035, 0.05), elong=(1.6, 2.0), shape="smooth"),
    "herb": dict(colors=[(40, 95, 55), (50, 120, 70), (35, 80, 60), (60, 110, 90)],
                 size=(0.03, 0.07), elong=(1.2, 2.5), shape="flake"),
    "black_pepper": dict(colors=[(25, 25, 28), (35, 32, 30), (45, 40, 38)],
                         size=(0.028, 0.04), elong=(1.0, 1.3), shape="flake"),
    "noodle": dict(colors=[(150, 205, 225), (170, 215, 230), (120, 190, 215)],
                   size=(0.06, 0.08), elong=(6.0, 10.0), shape="strand", on_teeth=0.4),
    "rice": dict(colors=[(205, 218, 226), (215, 225, 230), (195, 212, 222)],
                 size=(0.045, 0.065), elong=(1.6, 2.2), shape="smooth", alpha=(0.55, 0.75)),
    "leaf": dict(colors=[(45, 110, 60), (30, 80, 45), (60, 130, 80), (40, 90, 75)],
                 size=(0.08, 0.12), elong=(1.3, 2.0), shape="flake", on_teeth=0.5),
}
PALE = {"sesame", "noodle", "rice"}
# v3: red bits are drawn three times as often (they were the weakest type in v2).
KIND_WEIGHT = {k: (3.0 if k == "red_pepper" else 1.0) for k in FOOD}
# v3: lipstick colours (BGR) painted on the lips of clean and food images alike,
# so "red near the teeth" alone does not mean food.
LIPSTICK = [(70, 90, 215), (60, 70, 190), (80, 110, 235), (90, 80, 200), (50, 50, 160),
            (120, 120, 220), (60, 100, 230)]  # need a shadow rim to show on white teeth
TIGHT_MARGIN = 0.10  # optional re-framing around the outer lips (--margin)


def masks(img, meta):
    """Inner-lip mask and a teeth mask (lighter, less red than the lips)."""
    size = img.shape[0]
    m_in = np.zeros((size, size), np.uint8)
    m_out = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m_in, [np.array(meta["inner_lip"]).round().astype(np.int32)], 255)
    cv2.fillPoly(m_out, [np.array(meta["outer_lip"]).round().astype(np.int32)], 255)
    lips = (m_out > 0) & (m_in == 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_l, lip_a = np.median(lab[..., 0][lips]), np.median(lab[..., 1][lips])
    teeth = ((lab[..., 1] < lip_a - 6) & (lab[..., 0] > lip_l * 0.9) & (m_in > 0))
    teeth = cv2.morphologyEx(teeth.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    return m_in, teeth


def candidate_points(teeth, inner):
    """Points on the teeth, gaps between teeth and gum line weighted up."""
    ys, xs = np.nonzero(teeth)
    if len(xs) == 0:
        return None, None
    # Edges of the teeth mask inside the mouth: gaps and the gum/lip line.
    edge = cv2.morphologyEx(teeth, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    edge = (edge > 0) & (cv2.erode(inner, np.ones((5, 5), np.uint8)) > 0)
    ey, ex = np.nonzero(edge)
    return np.stack([xs, ys], 1), (np.stack([ex, ey], 1) if len(ex) else None)


def blob(rng, d, elong, jag):
    """Soft irregular blob alpha in a square patch, about d pixels across."""
    ss = 4  # supersample for soft anti-aliased edges
    n = int(d * elong ** 0.5 * (1 + jag) + 6)  # long axis fits after rotation
    big = np.zeros((n * ss, n * ss), np.uint8)
    k = rng.integers(7, 14)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    r = d / 2 * (1 + rng.uniform(-jag, jag, k))
    x = r * np.cos(ang) * elong ** 0.5
    y = r * np.sin(ang) / elong ** 0.5
    rot = rng.uniform(0, np.pi)
    xr, yr = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    pts = np.stack([(xr + n / 2) * ss, (yr + n / 2) * ss], 1).round().astype(np.int32)
    cv2.fillPoly(big, [pts], 255)
    a = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return cv2.GaussianBlur(a, (0, 0), 0.9)


def strand(rng, d, elong):
    """Soft curved strand (noodle): a thick quadratic Bezier, d*sqrt(elong) long."""
    ss = 4
    length, width = d * elong ** 0.5, max(1.5, d / elong ** 0.5)
    n = int(length * 1.15 + width + 6)
    big = np.zeros((n * ss, n * ss), np.uint8)
    rot = rng.uniform(0, np.pi)
    u = np.array([np.cos(rot), np.sin(rot)])
    v = np.array([-u[1], u[0]])
    c = np.array([n / 2, n / 2])
    p0, p2 = c - u * length / 2, c + u * length / 2
    p1 = c + v * rng.uniform(-0.35, 0.35) * length  # bend
    t = np.linspace(0, 1, 40)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    cv2.polylines(big, [(curve * ss).round().astype(np.int32)], False, 255,
                  thickness=max(1, int(round(width * ss))), lineType=cv2.LINE_AA)
    a = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return cv2.GaussianBlur(a, (0, 0), 0.8)


def paste_bit(img, kind, mouth_w, teeth, inner, pts, edge_pts, rng):
    spec = FOOD[kind]
    d = max(3.0, rng.uniform(*spec["size"]) * mouth_w)
    elong = rng.uniform(*spec["elong"])
    if spec["shape"] == "strand":
        alpha = strand(rng, d, elong)
    else:
        alpha = blob(rng, d, elong, 0.15 if spec["shape"] == "smooth" else 0.45)
    n = alpha.shape[0]
    if edge_pts is not None and rng.random() < 0.7:
        cx, cy = edge_pts[rng.integers(len(edge_pts))]
    else:
        cx, cy = pts[rng.integers(len(pts))]
    x0, y0 = int(cx - n // 2), int(cy - n // 2)
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + n > W or y0 + n > H:
        return None
    region = img[y0:y0 + n, x0:x0 + n].astype(np.float32)
    # Keep the bit inside the mouth opening (soft edge on the lip line).
    keep = cv2.GaussianBlur(inner[y0:y0 + n, x0:x0 + n].astype(np.float32) / 255,
                            (0, 0), 1.0)
    alpha = alpha * keep * rng.uniform(*spec.get("alpha", (0.75, 0.92)))
    if (alpha > 0.3).sum() < 4:
        return None
    # Most of the bit must sit on teeth, not on the lips or the dark gap.
    near_teeth = cv2.dilate(teeth[y0:y0 + n, x0:x0 + n], np.ones((5, 5), np.uint8)) > 0
    if (alpha * near_teeth).sum() < spec.get("on_teeth", 0.6) * alpha.sum():
        return None
    # Match local lighting: scale the food colour by the teeth brightness here.
    local = region[teeth[y0:y0 + n, x0:x0 + n] > 0]
    shade = np.clip((local.mean() if len(local) else region.mean()) / 215.0, 0.35, 1.15)
    color = spec["colors"][rng.integers(len(spec["colors"]))]
    jitter = rng.uniform(0.85, 1.15) * rng.uniform(0.92, 1.08, 3)  # value + hue drift
    base = np.clip(np.array(color, np.float32) * jitter * shade, 0, 255)
    # Mottled texture: fine noise plus a few broader blotches.
    sd = 0.12 if spec["shape"] == "flake" else 0.06
    tex = 1 + rng.normal(0, sd, (n, n)) + cv2.resize(
        rng.normal(0, sd, (4, 4)), (n, n), interpolation=cv2.INTER_CUBIC)
    tex = cv2.GaussianBlur(tex.astype(np.float32), (0, 0), 0.7)[..., None]
    # Shading from a random light direction so it is not flat colour.
    g = rng.uniform(0.08, 0.2)
    th = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / max(n - 1, 1) - 0.5
    light = (1 - 2 * g * (xx * np.cos(th) + yy * np.sin(th)))[..., None]
    food = np.clip(base[None, None, :] * tex * light, 0, 255)
    if kind in PALE:  # pale food on white teeth needs a faint shadow rim
        rim = np.clip(cv2.GaussianBlur(alpha, (0, 0), 1.2) - alpha, 0, 1)[..., None]
        region = region * (1 - 0.5 * rim)
    a = alpha[..., None]
    out = region * (1 - a) + food * a
    out = cv2.GaussianBlur(out, (0, 0), 0.4) * a + out * (1 - a)
    img[y0:y0 + n, x0:x0 + n] = np.clip(out, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(alpha > 0.3)
    return [int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max()), int(y0 + ys.max())]


def add_food(img, meta, rng):
    inner, teeth = masks(img, meta)
    pts, edge_pts = candidate_points(teeth, inner)
    if pts is None:
        return None, []
    outer = np.array(meta["outer_lip"])
    mouth_w = float(np.linalg.norm(outer[0] - outer[10]))
    img = img.copy()
    bits = []
    for _ in range(int(rng.integers(1, 4))):
        kinds = list(FOOD)
        w = np.array([KIND_WEIGHT[k] for k in kinds])
        kind = kinds[rng.choice(len(kinds), p=w / w.sum())]
        for _try in range(6):
            box = paste_bit(img, kind, mouth_w, teeth, inner, pts, edge_pts, rng)
            if box:
                bits.append({"type": kind, "bbox": box})
                break
    return (img if bits else None), bits


def lipstick(img, meta, rng):
    """v3: tint the lips (outer minus inner lip) towards a lipstick colour."""
    size = img.shape[0]
    m_in = np.zeros((size, size), np.uint8)
    m_out = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m_in, [np.array(meta["inner_lip"]).round().astype(np.int32)], 255)
    cv2.fillPoly(m_out, [np.array(meta["outer_lip"]).round().astype(np.int32)], 255)
    lips = ((m_out > 0) & (m_in == 0)).astype(np.float32)
    lips = cv2.GaussianBlur(lips, (0, 0), 2.0)[..., None] * rng.uniform(0.35, 0.7)
    f = img.astype(np.float32)
    shade = f.mean(2, keepdims=True) / 150.0
    color = np.array(LIPSTICK[rng.integers(len(LIPSTICK))], np.float32) * shade
    return np.clip(f * (1 - lips) + color * lips, 0, 255).astype(np.uint8)


def degrade(img, rng):
    """v3: low resolution (small or far mouth, video stills), beauty smoothing, sharpening."""
    size = img.shape[0]
    if rng.random() < 0.4:
        k = rng.uniform(0.3, 0.7)
        small = cv2.resize(img, (max(8, int(size * k)),) * 2, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.25:
        img = cv2.bilateralFilter(img, 7, rng.uniform(20, 50), 5)
    if rng.random() < 0.2:
        blur = cv2.GaussianBlur(img, (0, 0), 1.5)
        img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
    return img


def augment(img, boxes, rng):
    """Same random augmentation for both labels; boxes follow the rotation."""
    size = img.shape[0]
    img = degrade(img, rng)
    f = img.astype(np.float32) * rng.uniform(0.8, 1.2)
    t = rng.uniform(-0.08, 0.08)  # colour temperature: warm up red, cool blue
    f[..., 2] *= 1 + t
    f[..., 0] *= 1 - t
    img = np.clip(f, 0, 255).astype(np.uint8)
    angle = rng.uniform(-8, 8)
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(1, 4), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, move_boxes(boxes, m, size), int(rng.integers(60, 96))


def tight_crop(img, outer, margin=TIGHT_MARGIN):
    """Square crop around the outer lips with `margin` each side, same size out.

    Returns the crop and the 2x3 affine from input to crop coordinates.
    eval_onnx.py uses this too, so real photos get the same framing.
    """
    size = img.shape[0]
    outer = np.asarray(outer, np.float32)
    x0, y0 = outer.min(0)
    x1, y1 = outer.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    left, top = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
    s = size / side
    m = np.array([[s, 0, -left * s], [0, s, -top * s]], np.float32)
    crop = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    return crop, m


def move_boxes(boxes, m, size):
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        c = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], np.float32) @ m.T
        lo, hi = c.min(0).clip(0, size - 1), c.max(0).clip(0, size - 1)
        out.append({**b, "bbox": [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]})
    return out


def val_ids(ids, share, seed):
    """Same face split as v1: shuffle the sorted ids, first `share` go to val."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return set(order[:round(len(order) * share)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--variants", type=int, default=2,
                    help="food images per face (and as many clean copies)")
    ap.add_argument("--margin", type=float, default=None,
                    help="re-frame around the outer lips with this margin (off by default)")
    ap.add_argument("--src", type=Path, default=OUT_DIR / "mouths_tight")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "dataset_v3")
    args = ap.parse_args()
    src_dir, data_dir = args.src, args.out

    ids = sorted(p.stem for p in src_dir.glob("*.json"))
    in_val = val_ids(ids, args.val, args.seed)

    if data_dir.exists():
        shutil.rmtree(data_dir)
    for split in ("train", "val"):
        for label in ("clean", "food"):
            (data_dir / split / label).mkdir(parents=True)

    records, skipped = [], []
    for i, mid in enumerate(ids, 1):
        meta = json.loads((src_dir / f"{mid}.json").read_text())
        img = cv2.imread(str(src_dir / f"{mid}.png"))
        rng = np.random.default_rng([args.seed, int(mid)])
        foods = [add_food(img, meta, rng) for _ in range(args.variants)]
        foods = [f for f in foods if f[0] is not None]
        if not foods:
            skipped.append(mid)  # no usable teeth pixels; keeps classes balanced
            continue
        split = "val" if mid in in_val else "train"
        pairs = [("clean", img, [])] * len(foods) + [("food", f, b) for f, b in foods]
        for k, (label, src, boxes) in enumerate(pairs):
            if args.margin is not None:
                src, m = tight_crop(src, meta["outer_lip"], args.margin)
                boxes = move_boxes(boxes, m, src.shape[0])
            if rng.random() < 0.35:
                src = lipstick(src, meta, rng)
            out, boxes, q = augment(src, boxes, rng)
            rel = Path(split) / label / f"{mid}_{k % len(foods)}.jpg"
            cv2.imwrite(str(data_dir / rel), out, [cv2.IMWRITE_JPEG_QUALITY, q])
            records.append({"file": str(rel), "split": split, "label": int(label == "food"),
                            "source_id": mid, "jpeg_quality": q, "bits": boxes})
        if i % 200 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    (data_dir / "labels.json").write_text(json.dumps(records, indent=1))
    # train.py copies this into the model config so eval crops the same way.
    crop = json.loads((src_dir / f"{ids[0]}.json").read_text())
    (data_dir / "meta.json").write_text(json.dumps(
        {"source": src_dir.name, "crop_rule": crop.get("crop_rule", "outer_lip_bbox"),
         "crop_margin": crop.get("margin", 0.25), "reframe_margin": args.margin,
         "faces": len(ids) - len(skipped), "variants": args.variants,
         "val_share": args.val}, indent=1))
    counts = {}
    for r in records:
        k = f"{r['split']}/{'food' if r['label'] else 'clean'}"
        counts[k] = counts.get(k, 0) + 1
    kinds = {}
    for r in records:
        for b in r["bits"]:
            kinds[b["type"]] = kinds.get(b["type"], 0) + 1
    print(f"images: {counts}, skipped (no teeth pixels): {len(skipped)}")
    print(f"food bits: {kinds}")


if __name__ == "__main__":
    main()
