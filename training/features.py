"""라벨 붙인 셀카 → 서버(face_makeup_check)로 항목별 측정값을 뽑아 features.csv로.

입력 폴더 구조 (data/ 아래, 레포에 올리지 않음):
  labels.csv           file,issues    (issues: 비대칭인 항목을 ;로. 고르면 비움. 예: browHeight;cheekColor)
  <file>.jpg           앱이 보낸 것과 같은 셀카(셀카 방향)
  <file>.json          그 장면의 ML Kit 윤곽점 (mcp-tools.md의 landmarks)

실행: 서버를 띄운 뒤  python3 features.py data
"""
import base64
import csv
import json
import pathlib
import sys

from mcp import Mcp

CHECKS = ["browHeight", "browTail", "browLength", "browThickness", "lipCorner", "lipHalfWidth",
          "upperLipLine", "lowerLipLine", "cheekColor", "lipColor"]


def main(folder):
    root = pathlib.Path(folder)
    mcp = Mcp()
    rows = []
    with open(root / "labels.csv", newline="") as f:
        for lab in csv.DictReader(f):
            img = root / lab["file"]
            lm = json.loads(img.with_suffix(".json").read_text())
            r = mcp.call("face_makeup_check", {"imageBase64": base64.b64encode(img.read_bytes()).decode(),
                                               "landmarks": lm})
            sym = r.get("symmetry")
            if not sym:
                print("건너뜀(구도·밝기):", lab["file"], r.get("framing") or r.get("lighting") or r.get("error"))
                continue
            values = {c["name"]: c["value"] for c in sym["checks"]}
            issues = {s.strip() for s in (lab.get("issues") or "").split(";") if s.strip()}
            row = {"file": lab["file"]}
            for name in CHECKS:
                row[name] = values.get(name, "")
                row[name + "_label"] = int(name in issues)
            rows.append(row)
    out = root / "features.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}장 → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
