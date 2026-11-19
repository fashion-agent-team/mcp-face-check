import os
"""Evaluate a teeth model (torch .pt or ONNX) on the fixed test sets.
A) synth v2 val (1,048, one view, thr 0.5)  B) synth v4 val (one view)
C) clean real val faces (unaltered mouths_tight val crops, 6-view TTA, 0.7/0.3)
D) real food photos (crops listed in real/*.png with inner box json), 6-view TTA."""
import sys, json, glob, numpy as np, torch, cv2
from pathlib import Path
sys.path.insert(0, os.environ.get("TEETH_WORK", os.path.expanduser("~/teeth-work")))
import crop_rule
from torchvision import models
import torch.nn as nn
W = Path(os.environ.get("TEETH_WORK", os.path.expanduser("~/teeth-work")))
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
torch.set_num_threads(4)

def load(path):
    if str(path).endswith(".onnx"):
        import onnxruntime as ort
        s = ort.InferenceSession(str(path))
        return lambda x: s.run(None, {"image": x})[0][:, 0]
    m = models.mobilenet_v3_small(); m.classifier[3] = nn.Linear(1024, 1)
    m.load_state_dict(torch.load(path, map_location="cpu")["model"]); m.eval()
    def f(x):
        with torch.no_grad(): return m(torch.from_numpy(x)).numpy()[:, 0]
    return f

def prep(rgb, size=384):
    im = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return (((im.astype(np.float32) / 255) - MEAN) / STD).transpose(2, 0, 1)

def run(f, xs, bs=32):
    out = []
    for i in range(0, len(xs), bs): out.append(f(np.stack(xs[i:i + bs])))
    return 1 / (1 + np.exp(-np.concatenate(out)))

def views(rgb, inner):
    """6 views like the server: margin 1.25/1.36/1.5 x flip, from crop + inner-lip pts."""
    vs = []
    for k in (1.25, 1.36, 1.5):
        c, _ = crop_rule.crop(rgb, inner, margin=(k - 1) / 2)
        vs += [prep(c), prep(c[:, ::-1].copy())]
    return vs

def synth(f, ds):
    recs = [r for r in json.load(open(W / ds / "labels.json")) if r["split"] == "val"]
    xs = [prep(cv2.cvtColor(cv2.imread(str(W / ds / r["file"])), cv2.COLOR_BGR2RGB)) for r in recs]
    p = run(f, xs); y = np.array([r["label"] for r in recs]); pr = p >= 0.5
    out = {"n": len(y), "acc": float((pr == y).mean()), "recall": float(pr[y == 1].mean()),
           "fpr": float(pr[y == 0].mean())}
    for t in ("red_pepper", "sesame", "rice", "laver", "black_pepper", "noodle", "herb", "leaf"):
        idx = [i for i, r in enumerate(recs) if r["label"] and any(b["type"] == t for b in r.get("bits", []))]
        if idx: out[t] = float(pr[idx].mean())
    return out

def tta_set(f, items):
    ps = []
    for rgb, inner in items:
        ps.append(float(run(f, views(rgb, inner)).mean()))
    return np.array(ps)

_clean = None
def clean_items():
    global _clean
    if _clean is None:
        recs = json.load(open(W / "dataset_v2" / "labels.json"))
        ids = sorted({Path(r["file"]).name.split("_")[0] for r in recs if r["split"] == "val"})
        _clean = []
        for i in ids:
            m = json.load(open(W / "mouths_tight" / f"{i}.json"))
            _clean.append((cv2.cvtColor(cv2.imread(str(W / "mouths_tight" / f"{i}.png")), cv2.COLOR_BGR2RGB), m["inner_lip"]))
    return _clean

def real_items(folder):
    items, names = [], []
    for p in sorted(glob.glob(str(W / folder / "*.png")) + sorted(glob.glob(str(W / folder / "*.jpg")))):
        m = json.load(open(Path(p).with_suffix(".json")))
        items.append((cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), m["inner_lip"])); names.append(Path(p).stem)
    return items, names

def evaluate(path):
    f = load(path); res = {"model": str(path)}
    res["synth_v2"] = synth(f, "dataset_v2")
    if (W / "dataset_v4/labels.json").exists(): res["synth_v4"] = synth(f, "dataset_v4")
    pc = tta_set(f, clean_items())
    res["clean_real"] = {"n": len(pc), "false_alarm": float((pc >= 0.7).mean()), "unsure": float(((pc > 0.3) & (pc < 0.7)).mean())}
    for folder, lab in (("real_test/food", 1), ("real_test/clean", 0)):
        items, names = real_items(folder)
        if items:
            p = tta_set(f, items)
            res[folder] = {"n": len(p), "p": dict(zip(names, np.round(p, 3).tolist())),
                           "found" if lab else "false_alarm": float((p >= 0.7).mean()),
                           "unsure": float(((p > 0.3) & (p < 0.7)).mean())}
    return res

if __name__ == "__main__":
    for p in sys.argv[1:]: print(json.dumps(evaluate(p), ensure_ascii=False, indent=1))
