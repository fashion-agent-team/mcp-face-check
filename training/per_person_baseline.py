"""같은 사람의 다른 사진(예전 셀카)을 기준으로 삼으면 비대칭을 더 잘 잡는지 CelebA로 시험한다.

CelebA는 한 사람(유명인) 사진이 여러 장 있다(identity_CelebA.txt). 사람마다 화장한 정면 사진을 여러 장 골라
1) 모든 사진을 서버로 재고(부호 있는 값: 왼쪽이 크면 +),
2) 사진마다 한쪽만 다르게 그려 넣은 사진을 하나 더 잰다(public_baseline.py와 같은 그려 넣기).
비교하는 두 방식:
- 지금 방식: |값| > (모든 사람 사진의 |값| 95번째 백분위)
- 본인 기준: |값 - 같은 사람의 다른 사진 K장 값의 가운데 값| > (그 차이의 95번째 백분위)
두 방식 다 "그려 넣지 않은 사진의 95%는 말하지 않음"에 맞춘 뒤 검출률을 비교한다.

실행: 서버 띄우기 → python3 per_person_baseline.py <CelebA 폴더> <persons.json> [--k 3 5 9]
  persons.json = {"사람 번호": ["000001.jpg", ...]} (사진은 fetch_celeba.py로 받은 img_celeba 안에 있어야 함)
CelebA는 비상업 연구용입니다. 사진은 레포에 올리지 않습니다.
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from make_demo_data import EDITS, MODEL, Editor, Landmarker, fetch
from mcp import Mcp
from public_baseline import MAX_YAW, boxes, measure, selfie


def collect(root, persons, out):
    """사람마다 원래 사진 값과 그려 넣은 사진 값을 잰다. out에 이미 있으면 이어서."""
    done = json.loads(out.read_text()) if out.exists() else {}
    box = boxes(root)
    detect = Landmarker(fetch(MODEL, pathlib.Path(__file__).parent / ".cache/face_landmarker.task"))
    mcp = Mcp()
    rng = np.random.default_rng(0)
    i = 0
    for pid, files in persons.items():
        if pid in done:
            i += len(files)
            continue
        rows = []
        for f in files:
            i += 1
            img = selfie(cv2.imread(str(root / "Img/img_celeba" / f)), box[f])
            found = detect(img)
            if not found or abs(found[0]["yaw"]) > MAX_YAW:
                continue
            v = measure(mcp, img, found[0])
            if v is None:
                continue
            row = {"file": f, "plain": v}
            name, fn, (lo, hi), _ = EDITS[i % len(EDITS)]
            size = float(rng.uniform(lo, hi))
            out_img = getattr(Editor(img, found[1]), fn)(img, size)
            found2 = detect(out_img)
            if found2:
                v2 = measure(mcp, out_img, found2[0])
                if v2 is not None:
                    row["edit"] = {"name": name, "size": size, "values": v2}
            rows.append(row)
        done[pid] = rows
        out.write_text(json.dumps(done))
        print(f"{i} 장, 사람 {len(done)}/{len(persons)}", flush=True)
    return done


def evaluate(data, ks, rng):
    names = sorted({n for rows in data.values() for r in rows for n in r["plain"]})
    print(f"\n{'항목':<14}{'방식':<10}{'기준':>8}  그려 넣은 차이 크기별 검출률(작음→큼)")
    for name in names:
        group = [e for e in EDITS if e[0] == name]
        # 지금 방식
        plain = [r["plain"][name] for rows in data.values() for r in rows if name in r["plain"]]
        th = np.percentile(np.abs(plain), 95)
        res = {"지금": (th, [(r["edit"]["size"], abs(r["edit"]["values"][name]) > th)
                             for rows in data.values() for r in rows
                             if r.get("edit", {}).get("name") == name and name in r["edit"]["values"]])}
        for k in ks:
            resid, hits = [], []
            for rows in data.values():
                vals = [(j, r["plain"][name]) for j, r in enumerate(rows) if name in r["plain"]]
                for j, r in enumerate(rows):
                    others = [v for jj, v in vals if jj != j]
                    if len(others) < k:
                        continue
                    base = float(np.median(rng.choice(others, k, replace=False)))
                    if name in r["plain"]:
                        resid.append(r["plain"][name] - base)
                    e = r.get("edit")
                    if e and e["name"] == name and name in e["values"]:
                        hits.append((e["size"], e["values"][name] - base))
            if len(resid) < 30:
                continue
            thk = np.percentile(np.abs(resid), 95)
            res[f"본인 {k}장"] = (thk, [(s, abs(d) > thk) for s, d in hits])
        for label, (t, hits) in res.items():
            line = ""
            if group:
                lo, hi = group[0][2]
                edges = np.linspace(lo, hi, 6)
                cells = []
                for b0, b1 in zip(edges[:-1], edges[1:]):
                    sel = [h for s, h in hits if b0 <= s < b1]
                    cells.append(f"{np.mean(sel):4.0%}({len(sel)})" if sel else "   -")
                line = " ".join(cells)
            print(f"{name:<14}{label:<10}{t:8.3f}  {line}")


def main(a):
    root = pathlib.Path(a.celeba)
    persons = json.loads(pathlib.Path(a.persons).read_text())
    out = pathlib.Path(__file__).parent / "out/per_person.json"
    out.parent.mkdir(exist_ok=True)
    data = collect(root, persons, out) if not a.eval_only else json.loads(out.read_text())
    print(f"사람 {len(data)}명, 측정 사진 {sum(map(len, data.values()))}장")
    evaluate(data, a.k, np.random.default_rng(1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("celeba")
    p.add_argument("persons")
    p.add_argument("--k", type=int, nargs="+", default=[3, 5, 9])
    p.add_argument("--eval-only", action="store_true")
    main(p.parse_args())
