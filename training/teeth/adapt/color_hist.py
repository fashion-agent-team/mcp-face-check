import os
"""Learned colour-only baseline: Lab colour histogram of the middle of the crop -> logistic regression.
No shapes, no positions: only "which colours are in the mouth area and how much"."""
import sys, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, os.environ.get("TEETH_WORK", os.path.expanduser("~/teeth-work"))); import crop_rule
W = Path(os.environ.get("TEETH_WORK", os.path.expanduser("~/teeth-work")))
def feat(bgr):
    n = 192; im = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    m = int(n * 0.14); im = im[m:n - m, m:n - m]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # normalise brightness by the bright (teeth) level so exposure matters less
    L = lab[:, 0] / max(np.percentile(lab[:, 0], 90), 1) * 200
    h, _ = np.histogramdd(np.stack([np.clip(L, 0, 255), lab[:, 1], lab[:, 2]], 1), bins=(8, 12, 12),
                          range=((0, 256), (80, 200), (80, 200)))
    h = h.ravel() / len(lab)
    return np.sqrt(h)
def load(ds, split, limit=None):
    recs = [r for r in json.load(open(W / ds / "labels.json")) if r["split"] == split]
    if limit: recs = recs[::max(1, len(recs) // limit)]
    X = np.stack([feat(cv2.imread(str(W / ds / r["file"]))) for r in recs]); y = np.array([r["label"] for r in recs])
    return X, y, recs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
if __name__ == "__main__":
    Xtr, ytr, _ = load("dataset_v4", "train")
    Xte, yte, rte = load("dataset_v2", "val")
    recs = json.load(open(W / "dataset_v2/labels.json")); ids = sorted({r["source_id"] for r in recs if r["split"] == "val"})
    Xc = []
    for i in ids:
        m = json.load(open(W / f"mouths_tight/{i}.json")); c, _ = crop_rule.crop(cv2.imread(str(W / f"mouths_tight/{i}.png")), m["inner_lip"]); Xc.append(feat(c))
    Xc = np.stack(Xc)
    Xr, nr = [], []
    for p in sorted((W / "real_test/food").glob("*.png")):
        m = json.load(open(p.with_suffix(".json"))); c, _ = crop_rule.crop(cv2.imread(str(p)), m["inner_lip"]); Xr.append(feat(c)); nr.append(p.stem)
    Xr = np.stack(Xr)
    out = {}
    for name, clf in (("logreg", LogisticRegression(C=3, max_iter=3000)), ("gbm", HistGradientBoostingClassifier(max_iter=300))):
        clf.fit(Xtr, ytr)
        pt, pc, pr = clf.predict_proba(Xte)[:, 1], clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xr)[:, 1]
        res = {"synth_acc@0.5": float(((pt >= .5) == yte).mean())}
        for fa in (0.015, 0.05):
            t = float(np.quantile(pc, 1 - fa))
            r = {"thr": round(t, 3), "clean_fa": float((pc > t).mean()), "synth_recall": float((pt[yte == 1] > t).mean())}
            for ty in ("red_pepper", "laver", "black_pepper", "herb", "leaf", "sesame", "rice", "noodle"):
                idx = [k for k, x in enumerate(rte) if x["label"] and any(b["type"] == ty for b in x["bits"])]
                r[ty] = round(float((pt[idx] > t).mean()), 3)
            r["real"] = dict(zip(nr, [(round(float(v), 3), bool(v > t)) for v in pr])); res[f"fa{fa}"] = r
        out[name] = res
    import pickle; pickle.dump(clf, open(W / "color_gbm.pkl", "wb"))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(W / "color_hist_result.json", "w"), ensure_ascii=False, indent=1)
