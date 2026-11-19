"""눈썹 두께 모델 학습 데이터: 평가(public_baseline.py)에 쓰지 않는 CelebA 사진에 한쪽 눈썹을 아래로 더 두껍게 그리고,
서버(OnnxPairModel.Region.BROW)와 같은 범위로 자른 눈썹 두 조각과 정답(더 그린 두께 IOD, 사진 왼쪽이면 +)을 저장.

두께만 보게 섞은 것: 그리는 방식 2가지(아래로 늘림, 아래 선 따라 굵은 선)와 색·흐림·진하기, 눈썹 근처 어두운 점(정답 0),
높이·길이만 바꾼 눈썹(정답 0). 확인용으로 "위로 늘림"은 학습에 넣지 않음(평가 사진에서 72% 잡음).
CelebA는 비상업 연구용: 사진·만든 파일은 올리지 마세요.

사용: python brow_thickness_data.py <CelebA 폴더> [사진 수=100000] [REP=4]  → out/brow_thickness_train.npz"""
import sys, pathlib, json, numpy as np, cv2, multiprocessing as mp
from make_demo_data import MODEL, Editor, Landmarker, fetch, IDX
from public_baseline import attrs, boxes, selfie, WANT, MIN_FACE, MAX_YAW
C = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
HERE = pathlib.Path(__file__).parent
PW, PH = 128, 80   # 눈썹 조각: x -1.45~-0.05, y -0.95~-0.07 (IOD). OnnxBrowLength와 같아야 함
X0, X1, Y0, Y1 = -1.45, -0.05, -0.95, -0.07

def eyes(lm, w, h):
    le = np.array(lm["leftEye"]) * [w, h]; re = np.array(lm["rightEye"]) * [w, h]
    a, b = le.mean(0), re.mean(0)
    return (b, a) if a[0] > b[0] else (a, b)

def patches(img, lm):
    """서버와 같은 회전·크기 기준(두 눈 가운데, IOD)으로 왼쪽 눈썹 조각과 좌우 뒤집은 오른쪽 눈썹 조각."""
    h, w = img.shape[:2]; l, r = eyes(lm, w, h)
    d = r - l; iod = np.hypot(*d); c, s = d / iod; R = np.array([[c, -s], [s, c]])
    out = []
    for sgn in (-1, 1):
        xs = np.linspace(X0, X1, PW) * (1 if sgn < 0 else -1)
        ys = np.linspace(Y0, Y1, PH)
        gx, gy = np.meshgrid(xs, ys)
        p = (l + r) / 2 + (np.stack([gx, gy], -1) @ R.T) * iod
        out.append(cv2.remap(img, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out

def jpg(img):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)

def tail_mask(E, d):
    top, tail_left, m = E._brow()
    x0, x1 = top[:, 0].min(), top[:, 0].max(); cut = x0 + d * (x1 - x0) if tail_left else x1 - d * (x1 - x0)
    part = m.copy(); xx = np.arange(E.W); part[:, (xx > cut) if tail_left else (xx < cut)] = 0
    return cv2.dilate(part, np.ones((3, 3), np.uint8))

def shorten(E, img, d, how):
    part = tail_mask(E, d)
    if how == 0: return cv2.inpaint(img, part, 5, cv2.INPAINT_TELEA)
    if how == 1: return cv2.inpaint(img, part, 7, cv2.INPAINT_NS)
    # 주변을 크게 흐린 색으로 덮기
    k = int(E.iod * 0.25) | 1
    bg = cv2.medianBlur(cv2.inpaint(img, part, 3, cv2.INPAINT_TELEA), k)
    a = cv2.GaussianBlur(part, (5, 5), 0)[..., None] / 255.0
    return (img * (1 - a) + bg * a).clip(0, 255).astype(np.uint8)

def thicken(E, img, d, rng, how):
    """왼쪽 눈썹을 아래로 d IOD 더 그림. how 0: 평가와 같은 방식(아래로 늘림), 1: 아래 선을 따라 굵은 선"""
    _, _, m = E._brow()
    k = max(1, int(round(d * E.iod)))
    if how == 0:
        grown = cv2.dilate(m, np.ones((k + 1, 1), np.uint8), anchor=(0, k))
    else:
        bot = E.px[IDX["leftEyebrowBottom"]].astype(np.int32)
        grown = m.copy(); cv2.polylines(grown, [bot + [0, k // 2]], False, 255, k + 1)
    add = cv2.bitwise_and(grown, cv2.bitwise_not(m))
    dark = np.percentile(img[m > 0].reshape(-1, 3).sum(1), rng.uniform(15, 45))
    brow = np.median(img[m > 0][img[m > 0].sum(1) <= dark], axis=0) * rng.uniform(0.85, 1.1)
    kk = int(rng.choice([3, 5]))
    a = cv2.GaussianBlur(add, (kk, kk), 0)[..., None] / 255.0 * rng.uniform(0.6, 1.0)
    return (img * (1 - a) + brow * a).clip(0, 255).astype(np.uint8)

def dark_blob(E, img, rng):
    side = 'left' if rng.random() < .5 else 'right'
    top = E.px[IDX[side + 'EyebrowTop']]
    c = top[rng.integers(len(top))] + [rng.normal(0, .15) * E.iod, rng.choice([-1, 1]) * rng.uniform(.18, .35) * E.iod]
    m = np.zeros((E.H, E.W), np.uint8)
    cv2.ellipse(m, tuple(int(v) for v in c), (int(rng.uniform(.05, .15) * E.iod) + 1, int(rng.uniform(.015, .04) * E.iod) + 1), float(rng.uniform(-30, 30)), 0, 360, 255, -1)
    _, _, bm = E._brow()
    col = np.median(img[bm > 0], axis=0) * rng.uniform(0.8, 1.1)
    a = cv2.GaussianBlur(m, (5, 5), 0)[..., None] / 255.0 * rng.uniform(.4, .9)
    return (img * (1 - a) + col * a).clip(0, 255).astype(np.uint8)

def distract(E, img, rng):
    side = 'left' if rng.random() < .5 else 'right'
    top = E.px[IDX[side + 'EyebrowTop']]
    c = top[rng.integers(len(top))] + [rng.normal(0, .15) * E.iod, -rng.uniform(.15, .45) * E.iod]
    m = np.zeros((E.H, E.W), np.uint8)
    ax = (int(rng.uniform(.06, .2) * E.iod), int(rng.uniform(.03, .08) * E.iod))
    cv2.ellipse(m, tuple(int(v) for v in c), ax, float(rng.uniform(-30, 30)), 0, 360, 255, -1)
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA if rng.random() < .5 else cv2.INPAINT_NS)

det = None
def work(args):
    global det
    k, f, box = args
    if det is None: det = Landmarker(fetch(MODEL, HERE / '.cache/face_landmarker.task'))
    rng = np.random.default_rng(k)
    img = selfie(cv2.imread(str(C / 'Img/img_celeba' / f)), box)
    flip = bool(rng.integers(2))
    if flip: img = cv2.flip(img, 1)
    found = det(img)
    if not found or abs(found[0]["yaw"]) > MAX_YAW: return None
    lm, pts = found
    E = Editor(img, pts); u = rng.random(); size = 0.0; ed = img
    if u < 0.55:
        size = float(rng.uniform(0, 0.08)); ed = thicken(E, img, size, rng, int(rng.integers(2)))
    elif u < 0.7:
        ed = E.raise_left_brow(img, float(rng.uniform(0, 0.2)) * (1 if rng.random() < .5 else -1))
    elif u < 0.85:
        ed = shorten(E, img, float(rng.uniform(0, 0.4)), int(rng.integers(3)))
    if rng.random() < 0.4:   # 눈썹 근처 어두운 점(정답 0): 어두운 덩어리가 아니라 눈썹 두께를 보게
        ed = dark_blob(E, ed, rng)
    if flip: ed = cv2.flip(ed, 1); f2 = det(ed)
    else: f2 = det(ed)
    if not f2: return None
    a, b = patches(jpg(ed), f2[0])
    return f, size * (-1 if flip else 1), a, b

if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000; REP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    evalset = {f for f, at in attrs(C) if all(at[k] == v for k, v in WANT.items())}
    box = boxes(C)
    files = [f for f, at in attrs(C) if f not in evalset and at['Eyeglasses'] == -1 and at['Blurry'] == -1
             and (C / 'Img/img_celeba' / f).exists() and box[f][2] >= 150][:n]
    print(len(files), flush=True)
    res = []
    with mp.Pool(4) as p:
        for i, r in enumerate(p.imap(work, [(k, f, box[f]) for k, f in enumerate(files * REP)], chunksize=8)):
            if r: res.append(r)
            if i % 500 == 0: print(i, len(res), flush=True)
    (HERE / 'out').mkdir(exist_ok=True); np.savez_compressed(HERE / 'out/brow_thickness_train.npz', a=np.stack([r[2] for r in res]), b=np.stack([r[3] for r in res]), y=np.array([r[1] for r in res]),
                        f=np.array([r[0] for r in res]))
    print('done', len(res))
