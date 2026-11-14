"""Threshold-free comparison: AUC on synth v2 val, recall at matched false-alarm, clean real at matched threshold."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/user/scripts")
import evalset as E
from sklearn.metrics import roc_auc_score
W = E.W
recs = [r for r in json.load(open(W / "dataset_v2/labels.json")) if r["split"] == "val"]
X = [E.prep(cv2.cvtColor(cv2.imread(str(W / "dataset_v2" / r["file"])), cv2.COLOR_BGR2RGB)) for r in recs]
y = np.array([r["label"] for r in recs])
res = {}
for path in sys.argv[1:]:
    f = E.load(path); p = E.run(f, X)
    pc = E.tta_set(f, E.clean_items())
    o = {"auc": round(roc_auc_score(y, p), 4)}
    for fa in (0.02, 0.05):
        t = np.quantile(p[y == 0], 1 - fa); o[f"recall@synthFA{fa}"] = round(float((p[y == 1] > t).mean()), 3)
    # threshold so that clean-real TTA false alarm = 1.5% (4/262)
    t = np.sort(pc)[-5] + 1e-6
    o["thr_for_cleanFA1.5%"] = round(float(t), 3)
    o["synth_recall@that_thr"] = round(float((p[y == 1] >= t).mean()), 3)
    res[path.split("/")[-2]] = o; print(path.split("/")[-2], o, flush=True)
    np.save(W / f"probs_{path.split('/')[-2]}.npy", p)
