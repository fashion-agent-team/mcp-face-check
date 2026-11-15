"""out/jetson/*/R.npy로 하나씩·여러 개 평균(앙상블) 성적을 비교 → out/jetson/best.txt (서버 모델 server와 비교). 사용: python3 ens.py [pack 폴더]"""
import sys, json, pathlib, itertools, numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent; J = HERE / 'out/jetson'
M = np.load(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / 'out/pack') / 'eval_meta.npz'); SG = M['sg']; g = (M['glasses'] <= 0.02) & (M['light'] <= 15)
K = ['shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']
def score(R):
    Rm = R.mean(2); th = np.percentile(np.abs(Rm[g, 0]), 95)
    d = [float(np.mean(Rm[g, k + 1] * SG[g] > th)) for k in range(5)]
    return round(float(np.mean(d)), 3), [round(x, 2) for x in d]
Rs = {p.parent.name: np.load(p) for p in sorted(J.glob('*/R.npy'))}
Rs = {k: R / (np.std(R[:, 0]) + 1e-9) for k, R in Rs.items()}  # 모델마다 점수 크기가 달라서 실사진 퍼짐으로 맞춤
single = sorted(((score(R), k) for k, R in Rs.items()), reverse=True)
lines = [f'{k:<14} all {s[0]:.3f}  ' + ' '.join(f'{x:.2f}' for x in s[1]) for s, k in single]
top = [k for _, k in single if k != 'server'][:6]; ens = []
for r in (2, 3):
    for c in itertools.combinations(top, r): ens.append((score(np.mean([Rs[k] for k in c], 0)), '+'.join(c)))
ens.sort(reverse=True)
lines += ['--- 앙상블(상위 5)'] + [f'{k:<30} all {s[0]:.3f}  ' + ' '.join(f'{x:.2f}' for x in s[1]) for s, k in ens[:5]]
txt = '\n'.join(lines); (J / 'best.txt').write_text(txt + '\n'); print(txt, flush=True)
