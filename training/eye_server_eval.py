"""[눈화장] 서버로 직접 재는 평가: FFHQ 정면 사진에 한쪽 섀도·아이라인을 그려 넣어 face_makeup_check에 보내고,
서버가 건너뛴 사진(안경, 한쪽 그늘)은 빼고 잘못 알림·검출률을 낸다. 서버의 건너뛰기까지 포함한 실제 값.
사진은 FFHQ(비상업) → 레포에 올리지 않음. 학습(eye_ffhq_data.py)은 정렬 600번째 뒤만 쓰므로 앞 600장은 평가용.

실행: 서버(8094)를 띄운 뒤  python eye_server_eval.py <FFHQ 사진 폴더> [기준값 ...]
"""
import glob, sys
import cv2, numpy as np
_argv = sys.argv; sys.argv = ["x"]
from eye_makeup_data import eye_shadow, Landmarker, fetch, MODEL, HERE
from cheek_color_data import jpg
from public_baseline import jpeg
from mcp import Mcp
sys.argv = _argv

KEYS = ["shadow.3", "shadow.5", "shadow.7", "liner.5", "liner.8"]
det = Landmarker(fetch(MODEL, HERE / ".cache/face_landmarker.task")); m = Mcp()


def ask(img):
    """서버 점수(사진 왼쪽 눈이 진하면 +). 건너뛰었으면 nan, 다시 찍기면 None"""
    fd = det(img)
    if not fd: return None
    r = m.call("face_makeup_check", {"imageBase64": jpeg(img), "landmarks": fd[0]})
    sym = r.get("symmetry")
    if not sym: return None
    c = {x["name"]: x for x in sym.get("checks", [])}.get("eyeMakeup")
    return float("nan") if not c else c["value"] * (-1 if c["side"] == "right" else 1)


rows = []
for i, f in enumerate(sorted(glob.glob(sys.argv[1] + "/*"))[:600]):
    img = cv2.resize(cv2.imread(f), (600, 600)); fd = det(img)
    if not fd or abs(fd[0]["yaw"]) > 10: continue
    rng = np.random.default_rng(i); side = bool(rng.integers(2)); sg = 1 if side else -1
    v = [ask(jpg(img))]
    for k in KEYS:
        kind, a = k.split("."); v.append(ask(jpg(eye_shadow(img, fd[1], side, float("0." + a), rng, liner=(kind == "liner")))))
    rows.append([np.nan if x is None else x * (sg if j else 1) for j, x in enumerate(v)] + [v[0] is None])
    if len(rows) % 50 == 0: print(len(rows), flush=True)
R = np.array(rows, float); retake = R[:, 6] > 0; ok = np.isfinite(R[:, 0])
print(f"사진 {len(R)}장: 잼 {ok.sum()}, 건너뜀 {(~ok & ~retake).sum()}, 다시 찍기 {retake.sum()}")
plain = np.abs(R[ok, 0]); p95 = float(np.percentile(plain, 95))
for th in [p95] + [float(x) for x in sys.argv[2:]]:
    d = [np.mean(R[ok & np.isfinite(R[:, k + 1]), k + 1] > th) for k in range(5)]
    print(f"기준 {th:.3f} 잘못 알림 {np.mean(plain > th):.1%} | " + " ".join(f"{k} {x:.0%}" for k, x in zip(KEYS, d)) + f" | 전체 {np.mean(d):.1%}")
