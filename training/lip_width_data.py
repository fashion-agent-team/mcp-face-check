"""입술 너비 모델 학습 데이터: 평가(public_baseline.py)에 쓰지 않는 CelebA 사진에 한쪽 입꼬리 밖으로 립을 더 그리고,
서버(OnnxPairModel.Region.LIP_CORNER)와 같은 범위로 자른 입꼬리 두 조각과 정답(더 그린 길이 IOD, 사진 왼쪽이면 +)을 저장.

입꼬리 길이만 보게 섞은 것: 그리는 모양 2가지(삼각형, 둥근 모양)와 색·흐림·진하기, 입꼬리가 아닌 곳의 립 색 점(정답 0),
입술 색·윗입술 선만 바꾼 사진(정답 0). 확인용으로 "선 두 줄" 모양은 학습에 넣지 않음(평가 사진에서 87% 잡음).
CelebA는 비상업 연구용: 사진·만든 파일은 올리지 마세요.

사용: python lip_width_data.py <CelebA 폴더> [사진 수=100000] [REP=4]  → out/lip_width_train.npz"""
import sys, pathlib, numpy as np, cv2, multiprocessing as mp
from make_demo_data import MODEL, Editor, Landmarker, fetch, IDX
from public_baseline import attrs, boxes, selfie, WANT, MAX_YAW
C = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
HERE = pathlib.Path(__file__).parent
PW, PH = 128, 96   # OnnxPairModel.Region.LIP_CORNER와 같아야 함
X0, X1, Y0, Y1 = -0.85, -0.05, 0.75, 1.35   # 바깥 끝 → 가운데

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

def jpg(img):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)

def lip_color(E, rng):
    lipm = E.mask(IDX["upperLipTop"] + IDX["lowerLipBottom"][::-1])
    inner = E.mask(IDX["upperLipBottom"] + IDX["lowerLipTop"][::-1])
    px = E.base[cv2.bitwise_and(lipm, cv2.bitwise_not(inner)) > 0]
    q = rng.uniform(25, 75)
    return np.percentile(px, q, axis=0) * rng.uniform(0.9, 1.08), lipm

def extend(E, img, d, rng, how):
    """입꼬리(61) 밖으로 d IOD만큼 립을 더 그림. how: 0 삼각형(평가와 같은 모양), 1 둥근 모양, 2 굵은 선 두 줄"""
    c = E.px[61]; out = np.sign(c[0] - E.px[0][0]); tip = c + [out * d * E.iod, rng.normal(0, 0.01) * E.iod]
    m = np.zeros((E.H, E.W), np.uint8)
    if how == 0:
        cv2.fillPoly(m, [np.array([E.px[40], tip, E.px[91]]).astype(np.int32)], 255)
    elif how == 1:
        mid = (E.px[40] + E.px[91]) / 2; ctr = (mid + tip) / 2
        ax = (max(1, int(np.linalg.norm(tip - mid) / 2 + 1)), max(1, int(abs(E.px[91][1] - E.px[40][1]) / 2 * rng.uniform(.5, 1))))
        cv2.ellipse(m, tuple(int(v) for v in ctr), ax, 0, 0, 360, 255, -1)
    else:
        t = max(1, int(rng.uniform(.015, .035) * E.iod))
        cv2.line(m, tuple(E.px[40].astype(int)), tuple(tip.astype(int)), 255, t)
        cv2.line(m, tuple(E.px[91].astype(int)), tuple(tip.astype(int)), 255, t)
    col, lipm = lip_color(E, rng)
    m[lipm > 0] = 0
    k = int(rng.choice([3, 5]))
    a = cv2.GaussianBlur(m, (k, k), 0)[..., None] / 255.0 * rng.uniform(0.7, 1.0)
    return (img * (1 - a) + col * a).clip(0, 255).astype(np.uint8)

def blob(E, img, rng):
    """입꼬리가 아닌 곳(입술 위·아래 피부)에 립 색 점(정답 0): 색 덩어리가 아니라 입꼬리 길이를 보게"""
    c = E.px[61] if rng.random() < .5 else E.px[291]
    p = c + np.array([rng.normal(0, .1), rng.choice([-1, 1]) * rng.uniform(.12, .25)]) * E.iod
    m = np.zeros((E.H, E.W), np.uint8)
    cv2.ellipse(m, tuple(int(v) for v in p), (int(rng.uniform(.03, .08) * E.iod) + 1, int(rng.uniform(.015, .03) * E.iod) + 1), float(rng.uniform(-40, 40)), 0, 360, 255, -1)
    col, _ = lip_color(E, rng)
    a = cv2.GaussianBlur(m, (3, 3), 0)[..., None] / 255.0 * rng.uniform(.5, .9)
    return (img * (1 - a) + col * a).clip(0, 255).astype(np.uint8)

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
        size = float(rng.uniform(0, 0.15)); ed = extend(E, img, size, rng, int(rng.integers(2)))  # how 2(선)는 확인용으로 남김
    elif u < 0.7:
        ed = E.lighter_right_lip(img, float(rng.uniform(0, .7)))
    elif u < 0.85:
        ed = E.overdraw_left_upper_lip(img, float(rng.uniform(0, .1)))
    if rng.random() < 0.4: ed = blob(E, ed, rng)
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
    (HERE / 'out').mkdir(exist_ok=True); np.savez_compressed(HERE / 'out/lip_width_train.npz', a=np.stack([r[2] for r in res]), b=np.stack([r[3] for r in res]), y=np.array([r[1] for r in res]))
    print('done', len(res))
