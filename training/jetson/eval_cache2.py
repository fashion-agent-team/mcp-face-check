"""평가 묶음: FFHQ 정렬 앞 600장(학습에 안 씀) 중 정면(|yaw|≤10)에 한쪽 눈화장(섀도 .3/.5/.7, 라인 .5/.8)을 그리고
서버와 같은 범위로 자른 눈 조각을 7가지로 살짝 옮겨 자른 것(TTA)까지 저장. 서버가 건너뛰는 사진(안경 2% 넘음, 눈 밑 밝기
차이 15 넘음)을 거르려고 얼굴 분할(서버와 같은 BiSeNet r34)로 안경 비율도 같이 저장.
사용: python3 eval_cache.py <FFHQ 사진 폴더>  → out/pack/eval_p.npy [N,6,7,2,96,120,3] BGR, eval_meta.npz(sg, glasses, light)"""
import sys, glob, pathlib, numpy as np, cv2, onnxruntime as ort
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
_argv = sys.argv; sys.argv = ['x']
from eye_makeup_data import eye_shadow, Landmarker, fetch, MODEL, IDX, PW, PH, X0, X1, Y0, Y1
from cheek_color_data import jpg
sys.argv = _argv
OUT = HERE / 'out/eval2'; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'eval2_meta.npz').exists(): print('skip eval_cache'); raise SystemExit
det = Landmarker(fetch(MODEL, HERE / '.cache/face_landmarker.task'))
JIT = [(0, 0, 1), (.03, 0, 1), (-.03, 0, 1), (0, .03, 1), (0, -.03, 1), (0, 0, 1.06), (0, 0, .94)]
KEYS = ['plain', 'shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']


def frame(pts, W, H):
    a, b = pts[IDX["leftEye"]].mean(0) * [W, H], pts[IDX["rightEye"]].mean(0) * [W, H]
    l, r = (a, b) if a[0] < b[0] else (b, a)
    d = r - l; iod = np.hypot(*d); c, s = d / iod
    return (l + r) / 2, iod, np.array([[c, -s], [s, c]])


def patches(img, pts, dx, dy, sc):
    """서버 OnnxPairModel.Region.EYE와 같게: 두 조각 모두 바깥쪽이 왼쪽"""
    H, W = img.shape[:2]; m, iod, R = frame(pts, W, H); out = []
    for sgn in (-1, 1):
        xs = (np.linspace(X0, X1, PW) * sc + dx) * (1 if sgn < 0 else -1); ys = np.linspace(Y0, Y1, PH) * sc + dy
        gx, gy = np.meshgrid(xs, ys); p = m + (np.stack([gx, gy], -1) @ R.T) * iod - 0.5
        out.append(cv2.remap(img, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def allp(img):
    f = det(img)
    if not f: return None
    return np.stack([np.stack(patches(img, f[1], *j)) for j in JIT]), f[1]  # [J,2,H,W,3]


# 얼굴 분할(서버 FaceParser와 같은 구도·정규화)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
ps = ort.InferenceSession(str(HERE.parent / 'src/main/resources/models/face_parsing_bisenet_r34_fp16.onnx'), so)
PIN = ps.get_inputs()[0]; SPAN, CY, SZ = 4.2, 0.6, 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def glasses(img, pts):
    H, W = img.shape[:2]; m, iod, R = frame(pts, W, H)
    g = (np.arange(SZ) + 0.5) * SPAN / SZ; gx, gy = np.meshgrid(-SPAN / 2 + g, CY - SPAN / 2 + g)
    p = m + (np.stack([gx, gy], -1) @ R.T) * iod - 0.5
    x = cv2.remap(img, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = ((x[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    if 'float16' in PIN.type: x = x.astype(np.float16)
    lab = np.asarray(ps.run(None, {PIN.name: x})[0]).reshape(SZ, SZ); cell = SPAN / SZ; fr = []
    for xa, xb in ((X0, X1), (-X1, -X0)):  # 눈 조각 범위(양쪽)
        i0, i1 = int((xa + SPAN / 2) / cell), int((xb + SPAN / 2) / cell)
        j0, j1 = int((Y0 - (CY - SPAN / 2)) / cell), int((Y1 - (CY - SPAN / 2)) / cell)
        fr.append(float(np.mean(lab[j0:j1, i0:i1] == 6)))   # 6 = 안경
    return max(fr)


def light(pl):
    """눈 밑 피부 밝기(L*) 좌우 차이: 조각 [82:,24:96]"""
    L = [cv2.cvtColor(pl[s], cv2.COLOR_BGR2LAB)[82:, 24:96, 0].astype(np.float32).mean() * 100 / 255 for s in range(2)]
    return abs(L[0] - L[1])


files = sorted(glob.glob(str(pathlib.Path(sys.argv[1]) / '*')))[:600]
P, SG, GL, LD = [], [], [], []
for i, f in enumerate(files):
    img = cv2.imread(f)
    if img is None: continue
    img = cv2.resize(img, (600, 600)); fd = det(img)
    if not fd or abs(fd[0]['yaw']) > 10: continue
    rng = np.random.default_rng(i); side = bool(rng.integers(2))
    first = allp(jpg(img))
    if first is None: continue
    vs = [first[0]]
    for k in KEYS[1:]:
        kind, a = k.split('.'); a = float('0.' + a)
        r = allp(jpg(eye_shadow(img, fd[1], side, a, rng, liner=(kind == 'liner'))))
        vs.append(None if r is None else r[0])
    if any(v is None for v in vs): continue
    P.append(np.stack(vs)); SG.append(1 if side else -1)
    GL.append(glasses(jpg(img), first[1])); LD.append(light(first[0][0]))
    if i % 50 == 0: print(i, len(P), flush=True)
np.save(OUT / 'eval2_p.npy', np.stack(P))
np.savez(OUT / 'eval2_meta.npz', sg=np.array(SG), glasses=np.array(GL), light=np.array(LD))
print('done', len(P), 'gated', int(np.sum((np.array(GL) <= 0.02) & (np.array(LD) <= 15))))
