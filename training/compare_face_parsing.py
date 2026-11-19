"""얼굴 분할 모델 비교(docs/face-parsing-models.md).

1) 속도: python3 compare_face_parsing.py speed a.onnx b.onnx ...
   서버와 같은 입력(1x3x512x512)으로 onnxruntime CPU 4스레드, 15번 돌린 가운데 값.
2) 검출률: 모델마다 서버를 --face.parsing.model=file:<onnx>로 띄우고 public_baseline.py(--apply 없이)를 돌려
   out/public_baseline.json을 모델 이름별로 복사해 둔 뒤
   python3 compare_face_parsing.py detect 그전=old.json r34=new.json ...
   항목마다 그려 넣지 않은 사진의 95번째 백분위를 그 모델의 기준으로 잡고(오탐 5%로 맞춤) 크기 구간별 검출률을 냄.
   두 모델을 같은 사진끼리 짝지어 "한쪽만 잡은 수"로 부호 검정(McNemar)도 함.
"""
import json
import sys
import time
from fractions import Fraction
from math import comb

import numpy as np

EDITS = {"browHeight": (0, .2), "browLength": (0, .4), "browThickness": (0, .08), "upperLipLine": (0, .1)}
GROUP = {"upperLipLine": ["upperLipLine", "lowerLipLine"]}  # 서버는 윗·아랫입술 선을 한 기준으로 묶음


def speed(paths):
    import os
    import onnxruntime as ort
    x = np.random.default_rng(0).standard_normal((1, 3, 512, 512)).astype(np.float32)
    for p in paths:
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        s = ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
        n = s.get_inputs()[0].name
        for _ in range(3):
            s.run(None, {n: x})
        ts = []
        for _ in range(15):
            t = time.time()
            s.run(None, {n: x})
            ts.append(time.time() - t)
        print(f"{os.path.basename(p):<34} {os.path.getsize(p) / 1e6:6.1f}MB  {np.median(ts) * 1000:5.0f}ms")


def hits(d, edit):
    keys = GROUP.get(edit, [edit])
    th = np.percentile([abs(p[k]) for p in d["plain"] for k in keys if k in p], 95)
    return th, {e["file"]: (e["size"], abs(e["values"].get(edit, 0)) > th) for e in d["edited"] if e["edit"] == edit}


def sign_test(b, c):
    b, c = int(b), int(c)  # numpy 정수면 2 ** n이 넘침
    n, k = b + c, min(b, c)
    return min(1.0, float(Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2 ** n))) if n else 1.0


def detect(args):
    runs = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in args]
    # 사진 수가 다른 실행(--n을 줄여 잰 느린 모델)이 섞이면 모두 가진 사진만 남겨 같은 조건으로 비교
    plain = set.intersection(*({p["_file"] for p in d["plain"]} for _, d in runs))
    edited = set.intersection(*({e["file"] for e in d["edited"]} for _, d in runs))
    runs = [(n, {"plain": [p for p in d["plain"] if p["_file"] in plain],
                 "edited": [e for e in d["edited"] if e["file"] in edited]}) for n, d in runs]
    print(f"같이 잰 사진: 그리지 않은 {len(plain)}장, 그려 넣은 {len(edited)}장")
    base_name, base = runs[0]
    for edit, (lo, hi) in EDITS.items():
        edges = np.linspace(lo, hi, 6)
        for name, d in runs:
            th, h = hits(d, edit)
            cells = [np.mean([hit for s, hit in h.values() if b0 <= s < b1]) for b0, b1 in zip(edges[:-1], edges[1:])]
            print(f"{edit:<14} {name:<8} 기준 {th:6.3f}  " + " ".join(f"{c:4.0%}" for c in cells) + f"  평균 {np.mean(cells):4.0%}")
    print(f"\n{base_name} 대비 같은 사진에서 한쪽만 잡은 수(4항목 합)")
    for name, d in runs[1:]:
        b = c = 0
        for edit in EDITS:
            _, a = hits(base, edit)
            _, o = hits(d, edit)
            b += sum(o[f][1] and not a[f][1] for f in a if f in o)
            c += sum(a[f][1] and not o[f][1] for f in a if f in o)
        print(f"  {name}: 새로 잡음 {b}, 놓침 {c}, p={sign_test(b, c):.3f}")


if __name__ == "__main__":
    {"speed": speed, "detect": detect}[sys.argv[1]](sys.argv[2:])
