"""Evaluate teeth models on a folder of face photos with inner_lip .json next to each image
(food/ and clean/ subfolders), 6-view TTA like the server. Prints JSON with per-image p.
Usage: python eval_gen.py GEN_DIR MODEL.onnx [MODEL2.onnx ...]"""
import sys, json, glob, os, numpy as np, cv2, onnxruntime as ort
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def crop(img, pts, margin, out=384):
    p = np.asarray(pts, np.float32); x0, y0 = p.min(0); x1, y1 = p.max(0)
    side = float(max(x1 - x0, y1 - y0) * (1 + 2 * margin)); l = (x0 + x1) / 2 - side / 2; t = (y0 + y1) / 2 - side / 2
    s = out / side; m = np.array([[s, 0, -l * s], [0, s, -t * s]], np.float32)
    return cv2.warpAffine(img, m, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def prep(rgb):
    return (((cv2.resize(rgb, (384, 384)).astype(np.float32) / 255) - MEAN) / STD).transpose(2, 0, 1)
gen = sys.argv[1]; items = []
for lab in ("food", "clean", "bases"):
    for f in sorted(glob.glob(os.path.join(gen, lab, "*.png")) + glob.glob(os.path.join(gen, lab, "*.jpg"))):
        js = os.path.splitext(f)[0] + ".json"
        if not os.path.exists(js): continue
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); inner = json.load(open(js))["inner_lip"]
        views = []
        for k in (1.25, 1.36, 1.5):
            c = crop(rgb, inner, (k - 1) / 2); views += [prep(c), prep(c[:, ::-1].copy())]
        items.append((lab, os.path.basename(f), np.stack(views)))
out = {}
for mp in sys.argv[2:]:
    s = ora = ort.InferenceSession(mp)
    out[os.path.basename(mp)] = {f"{lab}/{n}": round(float((1 / (1 + np.exp(-s.run(None, {"image": v})[0][:, 0]))).mean()), 4) for lab, n, v in items}
print(json.dumps(out))
