"""features.csv → 항목마다 로지스틱 회귀를 학습해 판단 기준값(확률 0.5가 되는 측정값)을 정한다.

서버는 측정만 하고, "얼마나 다르면 비대칭인가"는 사람이 라벨 붙인 셀카에서 배운 값으로 판단한다.
결과는 out/symmetry-thresholds.yml. --apply면 서버 리소스(src/main/resources/symmetry-thresholds.yml)에도 쓴다.
두 라벨(고름/다름)이 각각 MIN_PER_CLASS장보다 적은 항목은 학습하지 않고 지금 기준값을 둔다.

실행: python3 train.py data/features.csv [--apply]
"""
import csv
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

MIN_PER_CLASS = 5
# 측정 항목 → application.yml 키(face.symmetry.*)
KEYS = {"browHeight": "brow-height", "browTail": "brow-tail", "browLength": "brow-length",
        "browThickness": "brow-thickness", "lipCorner": "lip-corner", "lipHalfWidth": "lip-half-width",
        "cheekColor": "cheek-redness", "lipColor": "lip-color"}
# 윗입술·아랫입술은 서버에서 기준값 하나(lip-line)를 같이 씀 → 둘을 합쳐 학습
SHARED = {"lipLine": ["upperLipLine", "lowerLipLine"]}


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fit(x, y):
    """측정값 하나로 로지스틱 회귀. 기준값 = 확률 0.5가 되는 x, 교차 검증 정확도."""
    X = np.array(x).reshape(-1, 1)
    model = LogisticRegression(class_weight="balanced").fit(X, y)
    w, b = model.coef_[0][0], model.intercept_[0]
    if w <= 0:
        return None, None  # 값이 클수록 비대칭이어야 하는데 반대면 데이터가 이상함
    folds = min(5, min(np.bincount(y)))
    acc = cross_val_score(LogisticRegression(class_weight="balanced"), X, y,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean() if folds >= 2 else float("nan")
    return -b / w, acc


def main(path, apply=False):
    rows = load(path)
    groups = {name: [name] for name in KEYS} | SHARED
    learned = {}
    print(f"{'항목':<14}{'고름':>5}{'다름':>5}{'기준값':>9}{'정확도':>8}")
    for group, names in groups.items():
        x, y = [], []
        for r in rows:
            for n in names:
                if r.get(n, "") != "":
                    x.append(float(r[n]))
                    y.append(int(r[n + "_label"]))
        y = np.array(y, dtype=int)
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        if neg < MIN_PER_CLASS or pos < MIN_PER_CLASS:
            print(f"{group:<14}{neg:>5}{pos:>5}{'(데이터 부족)':>14}")
            continue
        threshold, acc = fit(x, y)
        if threshold is None:
            print(f"{group:<14}{neg:>5}{pos:>5}{'(학습 실패)':>14}")
            continue
        key = KEYS.get(group, "lip-line")
        learned[key] = round(float(threshold), 3)
        print(f"{group:<14}{neg:>5}{pos:>5}{threshold:>9.3f}{acc:>8.2f}")
    if not learned:
        print("학습한 기준값이 없어요.")
        return
    text = "# training/train.py가 라벨 붙인 셀카로 학습한 기준값. 손으로 고치지 말고 다시 학습하세요.\n" \
           f"# 데이터: {path} ({len(rows)}장)\nface:\n  symmetry:\n" + \
           "".join(f"    {k}: {v}\n" for k, v in sorted(learned.items()))
    out = pathlib.Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / "symmetry-thresholds.yml").write_text(text)
    print("저장:", out / "symmetry-thresholds.yml")
    if apply:
        res = pathlib.Path(__file__).parent.parent / "src/main/resources/symmetry-thresholds.yml"
        res.write_text(text)
        print("서버에 적용:", res)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "data/features.csv", apply="--apply" in sys.argv)
