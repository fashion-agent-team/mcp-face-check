"""평가에 안 쓰는 FFHQ(정렬 600번째 뒤, 앞 600장은 평가용) 사진으로 눈화장 학습 쌍 만들기 → out/eye_makeup_ffhq_train.npz
FFHQ는 비상업: 사진·만든 파일은 올리지 마세요. 사용: python eye_ffhq_data.py <FFHQ 사진 폴더> [REP=40]"""
import sys, glob, pathlib, numpy as np, cv2, multiprocessing as mp
_argv = sys.argv; sys.argv = ['x']
import eye_makeup_data as E
from cheek_color_data import jpg
sys.argv = _argv
FILES = sorted(glob.glob(str(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.') / '*')))[600:]
det = None
def work(k):
    global det
    if det is None: det = E.Landmarker(E.fetch(E.MODEL, E.HERE / '.cache/face_landmarker.task'))
    rng = np.random.default_rng(99_000_000 + k)
    raw = cv2.imread(FILES[k % len(FILES)])
    if raw is None: return None
    sz = int(rng.integers(256, 720)); img = cv2.resize(raw, (sz, sz), interpolation=cv2.INTER_AREA)
    if rng.integers(2): img = cv2.flip(img, 1)
    found = det(img)
    if not found or abs(found[0]["yaw"]) > 15: return None
    ed, y = E.edit(img, found[1], rng)
    a, b = E.patches(jpg(ed), found[1])
    return y, a, b
if __name__ == '__main__':
    REP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    res = []
    with mp.Pool(4) as p:
        for i, r in enumerate(p.imap(work, range(len(FILES) * REP), chunksize=8)):
            if r: res.append(r)
            if i % 2000 == 0: print(i, len(res), flush=True)
    np.savez_compressed(E.HERE / 'out/eye_makeup_ffhq_train.npz', a=np.stack([r[1] for r in res]), b=np.stack([r[2] for r in res]), y=np.array([r[0] for r in res], np.float32))
    print('done', len(res))
