import os
import sys, json, pickle, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalset as E, color_hist as CH
W = E.W
gbm = pickle.load(open(W / "color_gbm.pkl", "rb"))
recs = [r for r in json.load(open(W / "dataset_v2/labels.json")) if r["split"] == "val"]
y = np.array([r["label"] for r in recs])
p = np.load(W / "probs_v4_384.npy")
cs = gbm.predict_proba(np.stack([CH.feat(cv2.imread(str(W / "dataset_v2" / r["file"]))) for r in recs]))[:, 1]
f = E.load(W / "runs/v4_384/teeth_mobilenetv3.onnx")
items = E.clean_items(); pc = E.tta_set(f, items)
import crop_rule
cc = gbm.predict_proba(np.stack([CH.feat(cv2.cvtColor(crop_rule.crop(rgb, inner)[0], cv2.COLOR_RGB2BGR)) for rgb, inner in items]))[:, 1]
def rule(pm, col, tc):
    return (pm >= 0.7) | ((pm > 0.3) & (pm < 0.7) & (col >= tc))
print("v4 alone: synth recall", (p[y == 1] >= .7).mean().round(3), "synth clean FA", (p[y == 0] >= .7).mean().round(3), "real clean FA", (pc >= .7).mean().round(3), "unsure synth food", ((p[y==1]>.3)&(p[y==1]<.7)).mean().round(3))
for tc in (0.5, 0.7, 0.8, 0.9, 0.95):
    print(tc, "synth recall", rule(p[y == 1], cs[y == 1], tc).mean().round(3), "synth clean FA", rule(p[y == 0], cs[y == 0], tc).mean().round(3), "real clean FA", rule(pc, cc, tc).mean().round(3))
