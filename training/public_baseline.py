"""공개 얼굴 데이터셋(CelebA)으로 비대칭 기준값을 정한다. 팀 셀카 라벨이 생기기 전까지 쓰는 기준.

생각:
  - CelebA에는 사진마다 'Heavy_Makeup', 'Wearing_Lipstick' 같은 속성 라벨이 있다(사람이 붙임).
    화장한 얼굴 사진은 대부분 남에게 보이려고 고르게 한 화장이다 → 이 사진들의 측정값 분포가 "고른 화장"의 범위.
  - 기준값 = 그 분포의 상위 PCT 백분위. "화장한 공개 사진 100장 중 95장보다 더 다르면 말해 준다"는 뜻.
    타고난 얼굴 비대칭, 윤곽점 오차, 사진 화질 흔들림이 모두 이 범위에 들어간다.
  - 같은 사진에 한쪽만 다르게 그려 넣어(눈썹 올림, 한쪽 블러셔, 한쪽 립 옅게, 입술 선 넘김) 그 기준으로 몇 %를
    잡는지 잰다(검출률). 그려 넣는 방법은 make_demo_data.py의 Editor와 같다.

사진은 CelebA 원본(img_celeba, 크기 제각각)에서 얼굴 상자(list_bbox_celeba.txt)를 넉넉히 잘라 셀카처럼 600x800으로 맞춘다.
얼굴 상자가 MIN_FACE px보다 작은 사진은 키우면 흐려져서 뺀다. 정렬된 작은 사진(img_align_celeba, 178x218)은 --aligned로
쓸 수 있지만 입술 쪽 측정이 흔들려서 기준이 느슨해진다(2026-11-09 비교).
CelebA는 비상업 연구용이다. 사진은 레포에 올리지 않고 결과 숫자만 남긴다.

실행: 서버를 띄운 뒤
  python3 public_baseline.py <CelebA 폴더(Anno/, Img/img_celeba/)> [--n 3000] [--pct 95] [--apply]
"""
import argparse
import base64
import json
import pathlib

import cv2
import numpy as np

from make_demo_data import EDITS, MODEL, Editor, Landmarker, fetch
from mcp import Mcp
from train import KEYS

CHECKS = ["browHeight", "browTail", "browLength", "browThickness", "lipCorner", "lipHalfWidth",
          "upperLipLine", "lowerLipLine", "cheekColor", "lipColor"]
GROUPS = {name: [name] for name in KEYS} | {"lipLine": ["upperLipLine", "lowerLipLine"]}
# 화장한 여성 사진, 눈썹·얼굴이 가려지지 않은 것
WANT = {"Heavy_Makeup": 1, "Wearing_Lipstick": 1, "Male": -1, "Blurry": -1, "Eyeglasses": -1,
        "Wearing_Hat": -1, "Bangs": -1}
MAX_YAW = 10
MIN_FACE = 180  # 얼굴 상자 너비(px, 원본)
OUT_W, OUT_H = 600, 800


def attrs(root):
    lines = (root / "Anno/list_attr_celeba.txt").read_text().split("\n")
    names = lines[1].split()
    for line in lines[2:]:
        p = line.split()
        if p:
            yield p[0], dict(zip(names, map(int, p[1:])))


def boxes(root):
    lines = (root / "Anno/list_bbox_celeba.txt").read_text().split("\n")
    return {p[0]: tuple(map(int, p[1:5])) for p in (line.split() for line in lines[2:]) if len(p) == 5}


def selfie(img, box):
    """얼굴 상자를 셀카 구도(얼굴이 화면 너비의 절반쯤)로 잘라 OUT_W x OUT_H로."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h * 0.5
    cw = w * 2.0
    ch = cw * OUT_H / OUT_W
    M = np.float32([[OUT_W / cw, 0, -(cx - cw / 2) * OUT_W / cw], [0, OUT_H / ch, -(cy - ch / 2) * OUT_H / ch]])
    return cv2.warpAffine(img, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def jpeg(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()


def measure(mcp, img, lm):
    r = mcp.call("face_makeup_check", {"imageBase64": jpeg(img), "landmarks": lm})
    sym = r.get("symmetry")
    return {c["name"]: c["value"] for c in sym["checks"]} if sym else None


def main(a):
    root = pathlib.Path(a.celeba)
    imgdir = root / ("Img/img_align_celeba" if a.aligned else "Img/img_celeba")
    box = {} if a.aligned else boxes(root)
    files = [f for f, at in attrs(root) if all(at[k] == v for k, v in WANT.items()) and (imgdir / f).exists()
             and (a.aligned or box[f][2] >= MIN_FACE)][:a.n]
    detect = Landmarker(fetch(MODEL, pathlib.Path(__file__).parent / ".cache/face_landmarker.task"))
    mcp = Mcp()
    rng = np.random.default_rng(0)
    plain, edited, skipped = [], [], 0
    for i, f in enumerate(files):
        img = cv2.imread(str(imgdir / f))
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if a.aligned else selfie(img, box[f])
        found = detect(img)
        if not found or abs(found[0]["yaw"]) > MAX_YAW:
            skipped += 1
            continue
        lm, pts = found
        v = measure(mcp, img, lm)
        if v is None:
            skipped += 1
            continue
        plain.append(v)
        # 같은 사진에 한쪽만 다르게 그려 넣기
        name, fn, (lo, hi), _ = EDITS[i % len(EDITS)]
        size = float(rng.uniform(lo, hi))
        out = getattr(Editor(img, pts), fn)(img, size)
        found2 = detect(out)
        if found2:
            v2 = measure(mcp, out, found2[0])
            if v2 is not None:
                edited.append({"edit": name, "size": size, "values": v2})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    print(f"사진 {len(files)}장 중 측정 {len(plain)}장, 건너뜀 {skipped}장(옆얼굴·윤곽점 없음·구도)")

    thresholds = {}
    print(f"\n{'항목':<14}{'중앙값':>8}{'p90':>8}{'p95':>8}{'p99':>8}  → 기준값(p{a.pct:g})")
    for group, names in GROUPS.items():
        x = np.array([abs(float(p[n])) for p in plain for n in names if n in p])
        if len(x) < 30:
            print(f"{group:<14}(측정 {len(x)}개, 부족)")
            continue
        q = np.percentile(x, [50, 90, 95, 99])
        thresholds[group] = float(np.percentile(x, a.pct))
        print(f"{group:<14}" + "".join(f"{v:>8.3f}" for v in q) + f"  → {thresholds[group]:.3f}")

    print("\n그려 넣은 차이를 그 기준으로 잡는 비율(크기 구간별)")
    report = {}
    for name, _, (lo, hi), _ in EDITS:
        group = "lipLine" if name in GROUPS["lipLine"] else name
        rows = [e for e in edited if e["edit"] == name and name in e["values"]]
        if group not in thresholds or not rows:
            continue
        edges = np.linspace(lo, hi, 6)
        line = []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            sel = [abs(e["values"][name]) > thresholds[group] for e in rows if b0 <= e["size"] < b1]
            line.append((b0, b1, len(sel), float(np.mean(sel)) if sel else float("nan")))
        report[name] = line
        print(f"  {name}: " + ", ".join(f"{b0:.2f}~{b1:.2f}: {r:.0%}({n})" for b0, b1, n, r in line))

    out = pathlib.Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / "public_baseline.json").write_text(json.dumps(
        {"photos": len(files), "measured": len(plain), "pct": a.pct, "thresholds": thresholds, "detection": report},
        ensure_ascii=False, indent=1))
    text = (f"# training/public_baseline.py: 공개 데이터셋 CelebA의 화장한 얼굴 사진 {len(plain)}장에서 잰 값의 "
            f"{a.pct:g}번째 백분위(100장 중 {a.pct:g}장이 이 값 아래).\n# 팀 셀카에 라벨을 붙여 train.py로 다시 학습하면 그 값으로 바꾸세요.\n"
            "face:\n  symmetry:\n" +
            "".join(f"    {KEYS.get(g, 'lip-line')}: {round(v, 3)}\n" for g, v in sorted(thresholds.items())))
    (out / "symmetry-thresholds.yml").write_text(text)
    print("\n저장:", out / "symmetry-thresholds.yml")
    if a.apply:
        res = pathlib.Path(__file__).parent.parent / "src/main/resources/symmetry-thresholds.yml"
        res.write_text(text)
        print("서버에 적용:", res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("celeba")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--aligned", action="store_true", help="정렬된 작은 사진(img_align_celeba)을 2배 키워 씀")
    p.add_argument("--pct", type=float, default=95)
    p.add_argument("--apply", action="store_true")
    main(p.parse_args())
