"""CelebA-HQ 중 원본 CelebA 번호 < 100000 사진으로 눈화장 학습 쌍(eye_makeup_data.edit 그대로).
별도 평가 세트(원본 번호 >= 100000)와 절대 겹치지 않게 번호로 거르고, 평가 목록과도 한 번 더 대조.
사용: python chq_train_data.py <CelebAMask-HQ 폴더> <평가 id 목록> <시작> <개수> <REP> <묶음이름> <training 폴더>"""
import sys, pathlib, numpy as np, cv2, multiprocessing as mp
D, held, st, cnt, REP, NAME, TR = sys.argv[1:8]
st, cnt, REP = int(st), int(cnt), int(REP)
sys.path.insert(0, TR)
_argv = sys.argv; sys.argv = ['x']
import eye_makeup_data as E
from cheek_color_data import jpg
sys.argv = _argv
D = pathlib.Path(D)
heldout = {int(x) for x in open(held).read().split()}
ok = []
for line in (D / 'CelebA-HQ-to-CelebA-mapping.txt').read_text().split('\n')[1:]:
    p = line.split()
    if len(p) >= 2 and int(p[1]) < 100000 and int(p[0]) not in heldout:
        ok.append(int(p[0]))
ok = sorted(ok)
assert not (set(ok) & heldout)
FILES = [str(D / 'CelebA-HQ-img' / f'{i}.jpg') for i in ok][st:st + cnt]
det = None


def work(k):
    global det
    if det is None: det = E.Landmarker(E.fetch(E.MODEL, E.HERE / '.cache/face_landmarker.task'))
    rng = np.random.default_rng(81_000_000 + st * 100 + k)
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
    mp.set_start_method('fork')
    print('eligible', len(ok), 'this part', len(FILES), flush=True)
    res = []
    with mp.Pool(5) as p:
        for i, r in enumerate(p.imap(work, range(len(FILES) * REP), chunksize=8)):
            if r: res.append(r)
            if i % 4000 == 0: print(i, len(res), flush=True)
    out = E.HERE / f'out/{NAME}.npz'; out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, a=np.stack([r[1] for r in res]), b=np.stack([r[2] for r in res]), y=np.array([r[0] for r in res], np.float32))
    print('done', len(res), out, flush=True)
