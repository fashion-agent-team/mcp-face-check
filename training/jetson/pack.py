"""out/eye_*.npz(압축, 통째로 읽어야 함)를 out/pack/{a,b,y}.npy로 이어 붙임 → 젯슨에서 mmap으로 조금씩 읽음(메모리 8GB 공유라 통째로 못 올림)"""
import pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent
NAMES = ['eye_makeup_1_train', 'eye_makeup_2_train', 'eye_makeup_ffhq_train', 'eye_hn_ffhq', 'eye_hn_celeba']
OUT = HERE / 'out/pack'; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'y.npy').exists(): print('skip pack'); raise SystemExit
n = 0; shape = None
for k in NAMES:
    with np.load(HERE / f'out/{k}.npz') as z: y = z['y']; n += len(y)
    print(k, len(y), flush=True)
A = np.lib.format.open_memmap(OUT / 'a.tmp.npy', 'w+', np.uint8, (n, 96, 120, 3))
B = np.lib.format.open_memmap(OUT / 'b.tmp.npy', 'w+', np.uint8, (n, 96, 120, 3))
Y = np.zeros(n, np.float32); i = 0
for k in NAMES:
    with np.load(HERE / f'out/{k}.npz') as z:
        a = z['a']; m = len(a); A[i:i + m] = a; del a
        B[i:i + m] = z['b']; Y[i:i + m] = z['y']; i += m
A.flush(); B.flush(); del A, B
(OUT / 'a.tmp.npy').rename(OUT / 'a.npy'); (OUT / 'b.tmp.npy').rename(OUT / 'b.npy'); np.save(OUT / 'y.npy', Y)
print('packed', n)
