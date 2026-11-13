"""눈화장 모델 학습 데이터: 평가(public_baseline.py)에 쓰지 않는 CelebA 사진의 한쪽 눈에 아이섀도·아이라인을 더 바르고(양쪽에 바르고
한쪽만 더 바른 것 포함), 서버(OnnxPairModel.Region.EYE)와 같은 범위로 자른 눈 두 조각과 정답(진하기 차이, 사진 왼쪽이면 +)을 저장.

눈화장만 보게 섞은 것: 갈색·분홍·보라·회색·금색·남색 섀도, 높이·번짐·바깥 꼬리 길이, 아이라인(굵기·날개), 얼굴 반쪽이 밝거나
따뜻한 조명(정답 0), 볼 블러셔·입술 색만 바꾼 사진(정답 0), 사진 전체 흐림·노이즈.
CelebA는 비상업 연구용: 사진·만든 파일은 올리지 마세요.

사용: python eye_makeup_data.py <CelebA 폴더> [사진 수=30000] [REP=2] [묶음 번호]  → out/eye_makeup[_번호]_train.npz"""
import sys, pathlib, numpy as np, cv2, multiprocessing as mp
from make_demo_data import MODEL, Editor, Landmarker, fetch, IDX
from public_baseline import attrs, boxes, selfie, MAX_YAW
from cheek_color_data import light, blush, jpg
C = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
HERE = pathlib.Path(__file__).parent
PW, PH = 120, 96   # OnnxPairModel.Region.EYE와 같아야 함
X0, X1, Y0, Y1 = -1.05, -0.05, -0.55, 0.3   # 바깥 → 가운데, 눈 위 → 눈 아래

# BGR. 곱하기 블렌드로 피부 결을 남김
SHADES = np.array([[60, 75, 110], [45, 60, 95], [130, 110, 190], [140, 90, 150], [110, 105, 110],
                   [80, 80, 85], [90, 150, 190], [110, 80, 60], [70, 95, 140], [40, 40, 50]], float)
LINER = np.array([[25, 25, 30], [30, 40, 60], [15, 15, 15]], float)


def eye_pts(pts, W, H, photo_left):
    a, b = pts[IDX["leftEye"]] * [W, H], pts[IDX["rightEye"]] * [W, H]
    ab = [pts[IDX["leftEyebrowBottom"]] * [W, H], pts[IDX["rightEyebrowBottom"]] * [W, H]]
    if (a.mean(0)[0] < b.mean(0)[0]) == photo_left: return a, ab[0], np.linalg.norm(a.mean(0) - b.mean(0))
    return b, ab[1], np.linalg.norm(a.mean(0) - b.mean(0))


def eye_shadow(img, pts, photo_left, alpha, rng, liner=None):
    """사진 왼쪽(photo_left) 또는 오른쪽 눈 위에 섀도(liner=False) 또는 아이라인(True)"""
    H, W = img.shape[:2]
    eye, brow, iod = eye_pts(pts, W, H, photo_left)
    c = eye.mean(0)
    up = eye[eye[:, 1] <= c[1] + 0.02 * iod]; up = up[np.argsort(up[:, 0])]
    b = brow[np.argsort(brow[:, 0])]; by = np.interp(up[:, 0], b[:, 0], b[:, 1])
    outer_right = c[0] > (pts[1][0] * W)   # 바깥 꼬리 방향: 사진 오른쪽 눈이면 오른쪽
    liner = bool(rng.random() < 0.3) if liner is None else liner
    k = rng.uniform(.06, .18) if liner else rng.uniform(.3, .85)
    top = up.copy(); top[:, 1] = up[:, 1] + (by - up[:, 1]) * k
    poly = [up, top[::-1]]
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [np.round(np.vstack(poly)).astype(np.int32)], 1.0)
    if rng.random() < 0.5:   # 바깥 꼬리(날개)
        o = up[-1] if outer_right else up[0]; d = 1 if outer_right else -1
        L = rng.uniform(.05, .2) * iod; t = rng.uniform(.02, .06) * iod if liner else rng.uniform(.06, .14) * iod
        tri = np.array([o + [0, -t], o + [d * L, -L * rng.uniform(.3, .8)], o + [0, t * .3]])
        cv2.fillPoly(m, [np.round(tri).astype(np.int32)], 1.0)
    if not liner and rng.random() < 0.25:   # 아래 눈꺼풀에도 살짝
        lo = eye[eye[:, 1] > c[1]]; lo = lo[np.argsort(lo[:, 0])]
        bot = lo.copy(); bot[:, 1] += rng.uniform(.03, .08) * iod
        m2 = np.zeros_like(m); cv2.fillPoly(m2, [np.round(np.vstack([lo, bot[::-1]])).astype(np.int32)], 1.0); m = np.maximum(m, m2 * .6)
    m = cv2.GaussianBlur(m, (0, 0), iod * (rng.uniform(.008, .02) if liner else rng.uniform(.02, .06))) * alpha
    col = (LINER if liner else SHADES)[rng.integers(len(LINER if liner else SHADES))] * rng.uniform(.9, 1.1)
    f = img.astype(np.float32)
    mix = f * col / 255 if rng.random() < 0.8 else np.broadcast_to(col, f.shape)   # 곱하기 또는 덮기(반짝이·밝은 색)
    out = f * (1 - m[..., None]) + mix * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def patches(img, pts):
    H, W = img.shape[:2]
    a, b = pts[IDX["leftEye"]].mean(0) * [W, H], pts[IDX["rightEye"]].mean(0) * [W, H]
    l, r = (a, b) if a[0] < b[0] else (b, a)
    d = r - l; iod = np.hypot(*d); c, s = d / iod; R = np.array([[c, -s], [s, c]])
    out = []
    for sgn in (-1, 1):
        xs = np.linspace(X0, X1, PW) * (1 if sgn < 0 else -1); ys = np.linspace(Y0, Y1, PH)
        gx, gy = np.meshgrid(xs, ys)
        p = (l + r) / 2 + (np.stack([gx, gy], -1) @ R.T) * iod - 0.5
        out.append(cv2.remap(img, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def edit(img, pts, rng):
    """정답: 사진 왼쪽 눈이 더 진하면 +"""
    E = Editor(img, pts); u = rng.random(); y = 0.0; ed = img
    if u < 0.6:
        side = bool(rng.integers(2)); a = float(rng.uniform(0.1, 0.85))
        ed = eye_shadow(img, pts, side, a, rng); y = a if side else -a
        if rng.random() < .3:   # 양쪽에 다 바르고 한쪽만 더(같은 색은 아니어도 됨, 정답은 차이)
            extra = float(rng.uniform(0, .5)) * a
            ed = eye_shadow(ed, pts, not side, extra, rng); y = y - extra if side else y + extra
    elif u < 0.7:   # 양쪽 같게(정답 0)
        a = float(rng.uniform(.1, .7)); s = int(rng.integers(1 << 30))
        ed = eye_shadow(eye_shadow(img, pts, True, a, np.random.default_rng(s)), pts, False, a, np.random.default_rng(s))
    elif u < 0.8:
        ed = blush(E, img, float(rng.uniform(.1, .5)), rng)
    elif u < 0.85:
        ed = E.lighter_right_lip(img, float(rng.uniform(0, .7)))
    if rng.random() < 0.35: ed = light(E, ed, rng)
    if rng.random() < 0.15: ed = cv2.GaussianBlur(ed, (0, 0), rng.uniform(.5, 1.5))
    if rng.random() < 0.15: ed = np.clip(ed + rng.normal(0, rng.uniform(2, 6), ed.shape), 0, 255).astype(np.uint8)
    return ed, y


det = None
REDETECT = False   # 편집한 사진에서 윤곽점을 다시 찾으면 2배 느림. 눈화장은 윤곽점을 거의 안 움직여서 원본 윤곽점을 씀
SEED = 0
def work(args):
    global det
    k, f, box = args
    if det is None: det = Landmarker(fetch(MODEL, HERE / '.cache/face_landmarker.task'))
    rng = np.random.default_rng(k + SEED)
    raw = cv2.imread(str(C / 'Img/img_celeba' / f))
    if raw is None: return None   # 받다 끊긴 파일
    img = selfie(raw, box)
    if rng.integers(2): img = cv2.flip(img, 1)
    found = det(img)
    if not found or abs(found[0]["yaw"]) > MAX_YAW: return None
    ed, y = edit(img, found[1], rng)
    ed = jpg(ed)
    if REDETECT:
        f2 = det(ed)
        if not f2: return None
        found = f2
    a, b = patches(ed, found[1])
    return f, y, a, b


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30000; REP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    SEED = int(sys.argv[4]) * 10_000_000 if len(sys.argv) > 4 else 0; part = f'_{sys.argv[4]}' if len(sys.argv) > 4 else ''
    from public_baseline import WANT
    evalset = {f for f, at in attrs(C) if all(at[k] == v for k, v in WANT.items())}
    box = boxes(C)
    files = [f for f, at in attrs(C) if f not in evalset and at['Eyeglasses'] == -1 and at['Blurry'] == -1
             and (C / 'Img/img_celeba' / f).exists() and box[f][2] >= 120][:n]
    print(len(files), flush=True)
    res = []
    with mp.Pool(4) as p:
        for i, r in enumerate(p.imap(work, [(k, f, box[f]) for k, f in enumerate(files * REP)], chunksize=8)):
            if r: res.append(r)
            if i % 2000 == 0: print(i, len(res), flush=True)
    (HERE / 'out').mkdir(exist_ok=True); np.savez_compressed(HERE / f'out/eye_makeup{part}_train.npz', a=np.stack([r[2] for r in res]), b=np.stack([r[3] for r in res]), y=np.array([r[1] for r in res], np.float32))
    print('done', len(res))
