"""볼 터치 모델 학습 데이터: 평가(public_baseline.py)에 쓰지 않는 CelebA 사진의 한쪽 볼에 블러셔를 더 바르고(양쪽에 바르고
한쪽만 더 바른 것 포함), 서버(OnnxPairModel.Region.CHEEK)와 같은 범위로 자른 볼 두 조각과 정답(진하기 차이, 사진 왼쪽이면 +)을 저장.

볼 터치만 보게 섞은 것: 분홍·코랄·피치 6가지 색, 크기·위치, 얼굴 반쪽이 따뜻하거나 밝은 조명(정답 0), 볼이 아닌 곳의 붉은 점(정답 0),
입술 색만 바꾼 사진(정답 0). 확인: 학습에 없던 베리색 비스듬한 타원 98%, 붉지 않은 브론즈는 10%.
CelebA는 비상업 연구용: 사진·만든 파일은 올리지 마세요.

사용: python cheek_color_data.py <CelebA 폴더> [사진 수=100000] [REP=4]  → out/cheek_color_train.npz"""
import sys, pathlib, numpy as np, cv2, multiprocessing as mp
from make_demo_data import MODEL, Editor, Landmarker, fetch, IDX
from public_baseline import attrs, boxes, selfie, WANT, MAX_YAW
C = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
HERE = pathlib.Path(__file__).parent
PW, PH = 112, 112   # OnnxPairModel.Region.CHEEK와 같아야 함
X0, X1, Y0, Y1 = -1.15, -0.2, 0.15, 1.05   # 바깥 → 가운데

def eyes(lm, w, h):
    le = np.array(lm["leftEye"]) * [w, h]; re = np.array(lm["rightEye"]) * [w, h]
    a, b = le.mean(0), re.mean(0)
    return (b, a) if a[0] > b[0] else (a, b)

def patches(img, lm):
    h, w = img.shape[:2]; l, r = eyes(lm, w, h)
    d = r - l; iod = np.hypot(*d); c, s = d / iod; R = np.array([[c, -s], [s, c]])
    out = []
    for sgn in (-1, 1):
        xs = np.linspace(X0, X1, PW) * (1 if sgn < 0 else -1); ys = np.linspace(Y0, Y1, PH)
        gx, gy = np.meshgrid(xs, ys)
        p = (l + r) / 2 + (np.stack([gx, gy], -1) @ R.T) * iod - 0.5
        out.append(cv2.remap(img, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out

def mirror_pts(pts):
    """좌우로 뒤집은 사진의 윤곽점(0~1 좌표). 번호는 그대로라 왼쪽 볼(205)이 뒤집힌 사진에선 오른쪽에 있음"""
    p = pts.copy(); p[:, 0] = 1 - p[:, 0]; return p

def jpg(img):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)

COLORS = np.array([[105, 85, 215], [120, 110, 230], [90, 100, 220], [130, 120, 210], [100, 80, 190], [110, 130, 235]], float)  # BGR 분홍·코랄·피치

def blush(E, img, alpha, rng, center=None, col=None):
    c = E.px[205] + rng.normal(0, .05, 2) * E.iod if center is None else center
    yy, xx = np.mgrid[0:E.H, 0:E.W]
    rx = rng.uniform(.10, .2) * E.iod; ry = rx * rng.uniform(.6, 1.1)
    a = alpha * np.exp(-((xx - c[0]) ** 2 / (2 * rx ** 2) + (yy - c[1]) ** 2 / (2 * ry ** 2)))[..., None]
    col = COLORS[rng.integers(len(COLORS))] * rng.uniform(.92, 1.05) if col is None else col
    return (img * (1 - a) + col * a).clip(0, 255).astype(np.uint8)

def light(E, img, rng):
    """얼굴 한쪽이 따뜻하거나 밝은 조명(정답 0): 볼 한 곳이 아니라 얼굴 반쪽 전체가 바뀜"""
    xx = np.arange(E.W)[None, :, None]; cx = E.px[1][0]
    g = 1 / (1 + np.exp(-(xx - cx) / (rng.uniform(.2, .6) * E.iod))) * rng.choice([-1, 1])
    tint = np.array([rng.uniform(-12, 0), rng.uniform(-6, 6), rng.uniform(0, 18)]) * rng.uniform(.3, 1)
    return (img + g * tint + g * rng.uniform(-15, 15)).clip(0, 255).astype(np.uint8)

def red_spot(E, img, rng):
    """볼이 아닌 곳(이마·턱·코 옆)에 붉은 점(정답 0)"""
    where = [E.px[10], E.px[152], E.px[1]][rng.integers(3)]
    c = where + rng.normal(0, .1, 2) * E.iod
    return blush(E, img, float(rng.uniform(.1, .4)), rng, center=c)

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
        size = float(rng.uniform(0, 0.5)); ed = blush(E, img, size, rng)
        if rng.random() < .3:   # 양쪽에 다 바르고 한쪽만 더(정답은 차이)
            extra = float(rng.uniform(0, .3)); ed = cv2.flip(blush(Editor(cv2.flip(ed, 1), mirror_pts(pts)), cv2.flip(ed, 1), extra, rng), 1); size -= extra
    elif u < 0.7:
        ed = E.lighter_right_lip(img, float(rng.uniform(0, .7)))
    if rng.random() < 0.35: ed = light(E, ed, rng)
    if rng.random() < 0.3: ed = red_spot(E, ed, rng)
    if flip: ed = cv2.flip(ed, 1)
    f2 = det(ed)
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
    (HERE / 'out').mkdir(exist_ok=True); np.savez_compressed(HERE / 'out/cheek_color_train.npz', a=np.stack([r[2] for r in res]), b=np.stack([r[3] for r in res]), y=np.array([r[1] for r in res]))
    print('done', len(res))
