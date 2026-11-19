"""Run the exported ONNX classifier on real photos.

The model was trained on mouth crops; which crop is in the model's .json
("crop"): v1 = outer-lip box + 25%, v2 = inner-lip box + 18% (crop_rule.py).

Inputs can be:
  - face photos, with --crop-faces: the mouth is found with MediaPipe
    (crop_mouths.py; needs mediapipe + opencv, as on the Mac) and cropped with
    the model's rule. Images where no face is found are reported and skipped.
  - mouth crops that already follow the model's rule: used as they are.
  - v1 crops with their .json next to them (mouths/, real_food/): for a v2
    model they are re-cropped with crop_rule.py from the stored inner lip.

Usage:
    python eval_onnx.py runs/mnv3s_224/teeth_mobilenetv3.onnx real_food/
    python eval_onnx.py MODEL.onnx ~/Pictures/real_teeth --crop-faces --csv out.csv
    python eval_onnx.py MODEL.onnx photos/ --label food   # prints accuracy too

Needs: onnxruntime (or onnxruntime-gpu on Jetson), numpy, pillow.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def gather(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            out.append(p)
    return out


def preprocess(img, cfg):
    s = cfg["size"]
    img = img.convert("RGB").resize((s, s), Image.BILINEAR)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return x.transpose(2, 0, 1)[None]


def v2_rule(cfg):
    return cfg.get("crop", {}).get("crop_rule") == "inner_lip_bbox"


class MouthCropper:
    """Face photo -> 384x384 mouth crop with the model's crop rule."""

    def __init__(self, cfg):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cv2
        import crop_mouths
        import crop_rule
        self.cv2, self.cm, self.rule, self.v2 = cv2, crop_mouths, crop_rule, v2_rule(cfg)
        self.det = crop_mouths.Detector()

    def __call__(self, path):
        bgr = self.cv2.imread(str(path))
        if bgr is None:
            return None
        pts = self.det.landmarks(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
        if pts is None:
            return None
        if self.v2:
            crop, _ = self.rule.crop(bgr, pts[self.cm.INNER])
        else:
            crop, _ = self.cm.square_crop(bgr, pts[self.cm.OUTER])
        return Image.fromarray(self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2RGB))


def load_crop(path, cfg):
    """A ready crop; v1 crops with a .json get re-cropped for a v2 model."""
    meta_file = path.with_suffix(".json")
    if v2_rule(cfg) and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("crop_rule") != "inner_lip_bbox" and "inner_lip" in meta:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            try:
                import cv2
                import crop_rule
            except ImportError:
                raise SystemExit(f"{path}: v1 crop needs opencv to re-crop for this v2 "
                                 "model; pass a v2 crop (e.g. real_food_v2/) or install opencv")
            crop, _ = crop_rule.crop(cv2.imread(str(path)), meta["inner_lip"])
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return Image.open(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--crop-faces", action="store_true",
                    help="inputs are face photos; crop the mouth first")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--label", choices=["clean", "food"],
                    help="true label of all inputs, to print accuracy")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    import onnxruntime as ort
    cfg = json.loads(args.model.with_suffix(".json").read_text())
    thr = args.threshold if args.threshold is not None else cfg["threshold"]
    providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                             "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(args.model), providers=providers)
    cropper = MouthCropper(cfg) if args.crop_faces else None

    rows, skipped = [], []
    for path in gather(args.inputs):
        img = cropper(path) if cropper else load_crop(path, cfg)
        if img is None:
            skipped.append(str(path))
            continue
        logit = float(sess.run(["logits"], {"image": preprocess(img, cfg)})[0][0, 0])
        prob = 1 / (1 + np.exp(-logit))
        rows.append({"file": str(path), "p_food": round(prob, 4),
                     "pred": "food" if prob >= thr else "clean"})
        print(f"{rows[-1]['pred']:5s} p_food={prob:.3f}  {path}")

    print(f"\n{len(rows)} images, {sum(r['pred'] == 'food' for r in rows)} predicted food "
          f"(threshold {thr}, providers {sess.get_providers()[0]})")
    if skipped:
        print(f"no face found in {len(skipped)}: {', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}")
    if args.label and rows:
        acc = np.mean([r["pred"] == args.label for r in rows])
        print(f"accuracy vs label '{args.label}': {acc:.3f}")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "p_food", "pred"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
