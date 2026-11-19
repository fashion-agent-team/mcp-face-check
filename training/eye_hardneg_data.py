"""실사진에서 잘못 알림을 낸 경우를 흉내 낸 정답 그대로 편집: 한쪽 눈 위로 머리카락, 강한 옆 그림자, 한쪽 색 비침.
정답(화장 차이)은 그대로 두고 그 위에 덧씌움. 사용(비상업 사진, 올리지 마세요):
python eye_hardneg_data.py ffhq <FFHQ 사진 폴더> <REP=25> eye_hn_ffhq | celeba <CelebA 폴더> <사진수=14000> eye_hn_celeba <묶음=5>"""
import sys, glob, pathlib, numpy as np, cv2, multiprocessing as mp
_argv = sys.argv; sys.argv = ['x']
import eye_makeup_data as E
from cheek_color_data import jpg
sys.argv = _argv
HAIR = np.array([[20, 20, 25], [30, 40, 60], [50, 70, 100], [90, 130, 170], [140, 180, 210], [200, 200, 200], [60, 60, 140]], float)
_orig = E.edit
def eyec(pts, W, H, left):
    a, b = pts[E.IDX["leftEye"]].mean(0) * [W, H], pts[E.IDX["rightEye"]].mean(0) * [W, H]
    l, r = (a, b) if a[0] < b[0] else (b, a)
    return (l if left else r), np.hypot(*(r - l))
def hair(img, pts, rng):
    H, W = img.shape[:2]; c, iod = eyec(pts, W, H, bool(rng.integers(2)))
    out = img.astype(np.float32); col = HAIR[rng.integers(len(HAIR))] * rng.uniform(.8, 1.2)
    m = np.zeros((H, W), np.float32); n = int(rng.integers(3, 40))
    x0 = c[0] + rng.uniform(-.5, .5) * iod; ang = rng.uniform(-.6, .6)
    for _ in range(n):
        xs = x0 + rng.normal(0, .15) * iod; t = np.linspace(0, 1, 20)
        y = c[1] - .9 * iod + t * iod * rng.uniform(.6, 1.5)
        x = xs + np.sin(ang) * (y - c[1] + .9 * iod) + rng.normal(0, .02) * iod * np.cumsum(rng.normal(0, 1, 20)) / 5
        cv2.polylines(m, [np.round(np.stack([x, y], 1)).astype(np.int32)], False, float(rng.uniform(.4, 1)), max(1, int(iod * rng.uniform(.005, .03))))
    m = cv2.GaussianBlur(m, (0, 0), max(.5, iod * .004))[..., None]
    return np.clip(out * (1 - m) + col * m, 0, 255).astype(np.uint8)
def side_shade(img, pts, rng):
    H, W = img.shape[:2]; c, iod = eyec(pts, W, H, True); r = eyec(pts, W, H, False)[0]; cx = (c[0] + r[0]) / 2
    xx = np.arange(W)[None, :, None]
    g = 1 / (1 + np.exp(-(xx - cx - rng.normal(0, .3) * iod) / (rng.uniform(.05, .5) * iod)))
    if rng.integers(2): g = 1 - g
    f = 1 - g * rng.uniform(.15, .55)
    tint = 1 + g * (rng.uniform(-.15, .15, 3) if rng.random() < .5 else 0)
    return np.clip(img * f * tint, 0, 255).astype(np.uint8)
def edit(img, pts, rng):
    ed, y = _orig(img, pts, rng)
    u = rng.random()
    if u < .25: ed = hair(ed, pts, rng)
    elif u < .5: ed = side_shade(ed, pts, rng)
    return ed, y
E.edit = edit
FILES = sorted(glob.glob(str(pathlib.Path(sys.argv[2]) / '*')))[600:] if len(sys.argv) > 2 and sys.argv[1] == 'ffhq' else []
det = None
def work_ffhq(k):
    global det
    if det is None: det = E.Landmarker(E.fetch(E.MODEL, E.HERE / '.cache/face_landmarker.task'))
    rng = np.random.default_rng(77_000_000 + k); raw = cv2.imread(FILES[k % len(FILES)])
    if raw is None: return None
    sz = int(rng.integers(256, 720)); img = cv2.resize(raw, (sz, sz), interpolation=cv2.INTER_AREA)
    if rng.integers(2): img = cv2.flip(img, 1)
    found = det(img)
    if not found or abs(found[0]["yaw"]) > 15: return None
    ed, y = edit(img, found[1], rng); a, b = E.patches(jpg(ed), found[1]); return None, y, a, b
if __name__ == '__main__':
    mode, n, name = sys.argv[1], int(sys.argv[3]), sys.argv[4]
    if mode == 'ffhq': jobs, fn = list(range(len(FILES) * n)), work_ffhq
    else:
        E.C = pathlib.Path(sys.argv[2]); E.SEED = int(sys.argv[5]) * 10_000_000
        from public_baseline import WANT, attrs, boxes
        ev = {f for f, at in attrs(E.C) if all(at[k] == v for k, v in WANT.items())}; box = boxes(E.C)
        files = [f for f, at in attrs(E.C) if f not in ev and at['Eyeglasses'] == -1 and at['Blurry'] == -1 and (E.C / 'Img/img_celeba' / f).exists() and box[f][2] >= 120][:n]
        jobs, fn = [(k, f, box[f]) for k, f in enumerate(files)], E.work
    res = []
    with mp.Pool(4) as p:
        for i, r in enumerate(p.imap(fn, jobs, chunksize=8)):
            if r: res.append(r)
            if i % 2000 == 0: print(i, len(res), flush=True)
    np.savez_compressed(E.HERE / f'out/{name}.npz', a=np.stack([r[2] for r in res]), b=np.stack([r[3] for r in res]), y=np.array([r[1] for r in res], np.float32))
    print('done', len(res))
