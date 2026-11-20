"""[눈화장] 끝난 r18 판들을 두 평가(FFHQ eval_p, 별도 CelebA-HQ eval2)로 재고, 가중치 평균(모델 수프)을 시험.
- 판마다 out/jetson/<이름>/eval2.json(별도 세트 점수)을 채움(이미 있으면 건너뜀)
- 수프: FFHQ 점수 상위 K개 균일 평균, greedy(FFHQ 순서로 넣어 FFHQ가 안 떨어지면 유지), BatchNorm 다시 맞추기(학습 데이터로 forward)
고르기는 FFHQ로만, 별도 세트는 확인용(고르기 부풀림 방지). 결과: out/jetson/soup.jsonl, 화면.
사용: python soup_eval.py [backfill|soup] [--k 5 10] [--bn 200]"""
import sys, json, glob, pathlib, argparse, numpy as np, torch, torch.nn as nn
HERE = pathlib.Path(__file__).resolve().parent.parent
J = HERE / 'out/jetson'
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['backfill', 'soup']); ap.add_argument('--k', type=int, nargs='*', default=[5, 10])
ap.add_argument('--bn', type=int, default=200); ap.add_argument('--pack', default=str(HERE / 'out/pack'))
args = ap.parse_args()
src = open(pathlib.Path(__file__).with_name('train_eye.py')).read()
ns = {'nn': nn, 'torch': torch, 'np': np}
exec(src[src.index('class Net(nn.Module):'):src.index('\n\nclass TVBack')], ns)
NetR18 = ns['NetR18']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t(x): return torch.from_numpy(np.ascontiguousarray(x)).to(dev).permute(0, 3, 1, 2).float() / 255.0


SETS = {'ffhq': (pathlib.Path(args.pack) / 'eval_p.npy', pathlib.Path(args.pack) / 'eval_meta.npz'),
        'chq': (HERE / 'out/eval2/eval2_p.npy', HERE / 'out/eval2/eval2_meta.npz')}
CACHE = {k: (np.load(p, mmap_mode='r'), np.load(m)) for k, (p, m) in SETS.items() if p.exists()}


def report(R, M):
    SG = M['sg']; g = (M['glasses'] <= 0.02) & (M['light'] <= 15); Rm = R.mean(2)
    th = np.percentile(np.abs(Rm[g, 0]), 95); K = ['shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']
    out = {'n': int(g.sum()), 'p95': round(float(th), 3)}
    for k, n in enumerate(K): out[n] = round(float(np.mean(Rm[g, k + 1] * SG[g] > th)), 3)
    out['all'] = round(float(np.mean([out[n] for n in K])), 3)
    return out


def score(net, which):
    P, M = CACHE[which]; R = np.zeros(P.shape[:3], np.float32); net.eval()
    with torch.no_grad():
        for i in range(len(P)):
            x = np.asarray(P[i]).reshape(-1, 2, *P.shape[4:])
            R[i] = net(t(x[:, 0]), t(x[:, 1])).float().cpu().numpy().reshape(P.shape[1:3])
    return report(R, M)


def runs():
    out = []
    for e in sorted(J.glob('*/eval.json')):
        r = json.load(open(e))
        if r.get('arch') == 'r18' and float(r.get('up', 1.0)) == 1.0 and (e.parent / 'eye.pt').exists():
            out.append((e.parent.name, r['all']))
    return sorted(out, key=lambda x: -x[1])


def load(name):
    n = NetR18(False, 18, 1.0).to(dev); n.load_state_dict(torch.load(J / name / 'eye.pt', map_location=dev)); return n


def avg(names):
    sds = [torch.load(J / n / 'eye.pt', map_location='cpu') for n in names]
    out = {}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point: out[k] = sum(s[k] for s in sds) / len(sds)
        else: out[k] = sds[0][k]
    n = NetR18(False, 18, 1.0).to(dev); n.load_state_dict(out); return n


def bn_reset(net, steps):
    """BatchNorm 통계를 학습 데이터로 다시 잼(가중치 평균 뒤 통계가 안 맞는 문제)"""
    A = np.load(HERE / 'out/pack/a.npy', mmap_mode='r'); B = np.load(HERE / 'out/pack/b.npy', mmap_mode='r')
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None
    net.train(); rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(steps):
            j = np.sort(rng.choice(len(A), 64, replace=False)); net(t(A[j]), t(B[j]))
    net.eval(); return net


if args.mode == 'backfill':
    for name, ff in runs():
        f = J / name / 'eval2.json'
        if f.exists() or 'chq' not in CACHE: continue
        r = score(load(name), 'chq'); r.update(name=name, ffhq_all=ff); json.dump(r, open(f, 'w'))
        print(name, 'ffhq', ff, 'chq', r['all'], flush=True)
else:
    R = runs(); log = open(J / 'soup.jsonl', 'a')
    def rec(tag, net):
        a = score(net, 'ffhq'); b = score(net, 'chq') if 'chq' in CACHE else {}
        line = dict(tag=tag, ffhq=a, chq=b); log.write(json.dumps(line) + '\n'); log.flush()
        print(f"{tag:40s} ffhq {a['all']:.3f}  chq {b.get('all', float('nan')):.3f}", flush=True); return a['all']
    best = R[0][0]; rec(f'single {best}', load(best))
    for k in args.k:
        names = [n for n, _ in R[:k]]
        rec(f'uniform top{k}', avg(names)); rec(f'uniform top{k} +bn', bn_reset(avg(names), args.bn))
    soup = [R[0][0]]; cur = rec('greedy start', load(soup[0]))
    for n, _ in R[1:40]:
        a = score(avg(soup + [n]), 'ffhq')['all']
        if a >= cur: soup.append(n); cur = a; print('  + keep', n, a, flush=True)
    rec(f'greedy ({len(soup)})', avg(soup)); rec(f'greedy ({len(soup)}) +bn', bn_reset(avg(soup), args.bn))
    json.dump(soup, open(J / 'greedy_soup.json', 'w'))
