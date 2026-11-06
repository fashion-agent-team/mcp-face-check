"""학습 과정이 끝까지 도는지 보는 데모 데이터. 진짜 학습에는 쓰지 마세요(라벨이 사람 판단이 아님).

공개 사진(MediaPipe 테스트용 인물 사진) 한 장에 눈썹 높이·블러셔·립스틱·입술 선 차이를 크기를 바꿔 가며 넣고,
일정 크기 이상이면 '다름' 라벨을 붙인다. 각도·밝기·크기도 조금씩 흔든다. 윤곽점은 앱 ML Kit 대신 MediaPipe로 뽑는다.
실행: python3 make_demo_data.py data_demo 80
"""
import csv
import json
import pathlib
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

PHOTO = "https://storage.googleapis.com/mediapipe-assets/business-person.png"
MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
# MediaPipe 478점 → ML Kit FaceContour 이름(점 개수도 맞춤). MediaPipe의 right = 셀카 사진 왼쪽
IDX = {
    "face": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "leftEyebrowTop": [107, 66, 105, 63, 70], "leftEyebrowBottom": [55, 65, 52, 53, 46],
    "rightEyebrowTop": [336, 296, 334, 293, 300], "rightEyebrowBottom": [285, 295, 282, 283, 276],
    "leftEye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "rightEye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "upperLipTop": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "upperLipBottom": [191, 80, 81, 82, 13, 312, 311, 310, 415],
    "lowerLipTop": [95, 88, 178, 87, 14, 317, 402, 318, 324],
    "lowerLipBottom": [146, 91, 181, 84, 17, 314, 405, 321, 375],
    "noseBridge": [168, 4], "noseBottom": [98, 2, 327], "leftCheek": [205], "rightCheek": [425],
}


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


class Landmarker:
    def __init__(self, model):
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)), output_facial_transformation_matrixes=True))

    def __call__(self, bgr):
        r = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not r.face_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in r.face_landmarks[0]])
        m = np.array(r.facial_transformation_matrixes[0])[:3, :3]
        yaw = float(np.degrees(np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2]))))
        out = {k: [[round(float(pts[i, 0]), 5), round(float(pts[i, 1]), 5)] for i in v] for k, v in IDX.items()}
        out["yaw"] = round(yaw, 1)
        return out, pts


class Editor:
    """기본 얼굴의 윤곽점 위치를 기준으로 화장 차이를 그려 넣는다."""

    def __init__(self, base, pts):
        self.base = base
        self.H, self.W = base.shape[:2]
        self.px = pts * [self.W, self.H]
        self.iod = np.linalg.norm(self.px[IDX["leftEye"]].mean(0) - self.px[IDX["rightEye"]].mean(0))
        c = (self.px[205] + [0, -0.3 * self.iod]).astype(int)
        self.skin = np.median(base[c[1] - 5:c[1] + 5, c[0] - 5:c[0] + 5].reshape(-1, 3), axis=0)

    def mask(self, idx, dilate=0):
        m = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(m, [self.px[idx].astype(np.int32)], 255)
        return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)) if dilate else m

    def raise_left_brow(self, img, d):
        m = self.mask(IDX["leftEyebrowTop"] + IDX["leftEyebrowBottom"][::-1], int(0.06 * self.iod))
        out = cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)
        dy = int(round(d * self.iod))
        ms, shifted = np.roll(m, -dy, axis=0), np.roll(img, -dy, axis=0)
        out[ms > 0] = shifted[ms > 0]
        return out

    def blush_left(self, img, alpha):
        c = self.px[205]
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        r = 0.2 * self.iod * 0.7
        a = alpha * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r ** 2))[..., None]
        return (img * (1 - a) + np.array([105, 85, 215]) * a).clip(0, 255).astype(np.uint8)

    def lighter_right_lip(self, img, k):
        m = cv2.bitwise_and(self.mask(IDX["upperLipTop"] + IDX["lowerLipBottom"][::-1]),
                            cv2.bitwise_not(self.mask(IDX["upperLipBottom"] + IDX["lowerLipTop"][::-1])))
        m[:, :int(self.px[0][0])] = 0
        m = cv2.GaussianBlur(m, (5, 5), 0)[..., None] / 255.0 * k
        return (img * (1 - m) + self.skin * m).clip(0, 255).astype(np.uint8)

    def overdraw_left_upper_lip(self, img, d):
        up = self.px[IDX["upperLipTop"]]
        sel = up[up[:, 0] <= self.px[0][0]]
        poly = np.vstack([sel, (sel + [0, -d * self.iod])[::-1]]).astype(np.int32)
        m = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        lip = np.median(self.base[self.mask(IDX["upperLipTop"] + IDX["upperLipBottom"][::-1]) > 0], axis=0)
        m = cv2.GaussianBlur(m, (3, 3), 0)[..., None] / 255.0
        return (img * (1 - m) + lip * m).clip(0, 255).astype(np.uint8)


# (항목, 그리는 함수, 크기 범위, 이 크기부터 '다름' 라벨)
EDITS = [("browHeight", "raise_left_brow", (0.0, 0.2), 0.08),
         ("cheekColor", "blush_left", (0.0, 0.5), 0.2),
         ("lipColor", "lighter_right_lip", (0.0, 0.7), 0.3),
         ("upperLipLine", "overdraw_left_upper_lip", (0.0, 0.1), 0.04)]


def jitter(img, rng):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
    out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return (out * rng.uniform(0.85, 1.1)).clip(0, 255).astype(np.uint8)


def main(folder, n):
    root = pathlib.Path(folder)
    cache = pathlib.Path(__file__).parent / ".cache"
    src = cv2.imread(str(fetch(PHOTO, cache / "business-person.png")))
    detect = Landmarker(fetch(MODEL, cache / "face_landmarker.task"))
    base = cv2.resize(src[90:560, 280:700], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, pts = detect(base)
    ed = Editor(base, pts)
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        name, fn, (lo, hi), visible = EDITS[i % len(EDITS)]
        size = rng.uniform(lo, hi)
        img = jitter(getattr(ed, fn)(base, size), rng)
        found = detect(img)
        if not found:
            continue
        file = f"{i:03d}.jpg"
        cv2.imwrite(str(root / file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (root / file).with_suffix(".json").write_text(json.dumps(found[0]))
        rows.append({"file": file, "issues": name if size >= visible else ""})
    with open(root / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "issues"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}장 → {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_demo", int(sys.argv[2]) if len(sys.argv) > 2 else 80)
