"""[젯슨 GPU] 눈화장 좌우 비교 모델 학습 + 서버식 평가. train_eye_model.py(넓은 망 4판 = 지금 서버 모델)와 같은 구조·증강.
- 데이터: out/pack/{a,b,y}.npy를 mmap으로 조금씩 읽음(make_data.sh로 맥에서 만들어 복사)
- 에폭마다 out/jetson/<이름>/ck.pt 저장 → 끊기면 같은 명령으로 이어서
- 메모리 감시: 젯슨 남은 메모리(MemAvailable)가 --min-free GB 아래거나, 다른 스레드가 ~/gpu-train.lock을 만들면 종료 코드 75(서버·다른 학습 먼저)
- 끝나면 eye_<이름>.onnx, 평가(eval.json: 안경·조명으로 거른 사진에서 실사진 p95 기준 검출률), R.npy(앙상블 평가용)
사용: python3 train_eye.py --name s13 [--seed 13] [--epochs 16] [--width 1.0] [--lr 3e-3] [--batch 64] [--arch small|r18]
      python3 train_eye.py --eval-onnx ../../src/main/resources/models/eye_makeup.onnx --name server  (지금 모델 평가만)"""
import argparse, json, os, pathlib, sys, threading, queue, time, numpy as np, torch, torch.nn as nn
HERE = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--seed', type=int, default=13)
ap.add_argument('--epochs', type=int, default=16); ap.add_argument('--width', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--min-free', type=float, default=1.0); ap.add_argument('--eval-onnx')
ap.add_argument('--pack', default=str(HERE / 'out/pack'))
ap.add_argument('--arch', default='small', choices=['small', 'r18', 'r34', 'r50'])
ap.add_argument('--up', type=float, default=1.0)   # 미리 학습된 몸통 앞에서 조각을 키움(96x120은 ImageNet 몸통엔 작아 마지막 특징이 3x4뿐). 2면 계산 약 4배   # r18 = ImageNet으로 미리 학습된 ResNet18 몸통(합성→실사진 일반화 기대)
args = ap.parse_args()
PACK = pathlib.Path(args.pack); OUT = HERE / 'out/jetson' / args.name; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.backends.cudnn.benchmark = True


LOCK = pathlib.Path(os.environ.get('GPU_LOCK', os.path.expanduser('~/gpu-train.lock')))


def other_job():
    """다른 스레드가 젯슨 GPU를 쓰겠다고 잠금 파일을 만들었으면 True(우리 것은 'face-check'로 시작)"""
    try: return not LOCK.read_text().startswith('face-check')
    except OSError: return False


def mem_free_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'): return int(line.split()[1]) / 1048576
    return 99.0


class Net(nn.Module):
    def __init__(s, w=1.0):
        super().__init__()
        c = [max(8, int(round(k * w))) for k in (24, 48, 96, 128)]
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(3, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3]), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(2 * c[3], 64), nn.ReLU(), nn.Linear(64, 1))
    def pair(s, ea, eb): return (s.h(torch.cat([ea, eb], 1)) - s.h(torch.cat([eb, ea], 1))).squeeze(1)
    def forward(s, a, b): return s.pair(s.f(a), s.f(b))


class Basic(nn.Module):
    """torchvision ResNet BasicBlock과 같은 이름(공식 ImageNet 가중치를 그대로 읽으려고)"""
    def __init__(s, i, o, st):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.bn1 = nn.BatchNorm2d(o); s.relu = nn.ReLU(inplace=True)
        s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.bn2 = nn.BatchNorm2d(o)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if st != 1 or i != o else None
    def forward(s, x):
        y = s.bn2(s.conv2(s.relu(s.bn1(s.conv1(x)))))
        return s.relu(y + (x if s.downsample is None else s.downsample(x)))


class Bottle(nn.Module):
    """torchvision ResNet Bottleneck과 같은 이름"""
    def __init__(s, i, w, st):
        super().__init__(); o = w * 4
        s.conv1 = nn.Conv2d(i, w, 1, bias=False); s.bn1 = nn.BatchNorm2d(w)
        s.conv2 = nn.Conv2d(w, w, 3, st, 1, bias=False); s.bn2 = nn.BatchNorm2d(w)
        s.conv3 = nn.Conv2d(w, o, 1, bias=False); s.bn3 = nn.BatchNorm2d(o); s.relu = nn.ReLU(inplace=True)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if st != 1 or i != o else None
    def forward(s, x):
        y = s.relu(s.bn1(s.conv1(x))); y = s.relu(s.bn2(s.conv2(y))); y = s.bn3(s.conv3(y))
        return s.relu(y + (x if s.downsample is None else s.downsample(x)))


class R18(nn.Module):
    """입력 BGR 0~1 → RGB·ImageNet 정규화 → ResNet18(또는 34) → 512"""
    URLS = {18: 'https://download.pytorch.org/models/resnet18-f37072fd.pth', 34: 'https://download.pytorch.org/models/resnet34-b627a593.pth',
            50: 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'}
    def __init__(s, pretrained=True, depth=18, up=1.0):
        super().__init__()
        n = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3]}[depth]; s.up = up
        if depth == 50:
            def layer(i, w, st, k): return nn.Sequential(Bottle(i, w, st), *[Bottle(w * 4, w, 1) for _ in range(k - 1)])
            s.layer1 = layer(64, 64, 1, n[0]); s.layer2 = layer(256, 128, 2, n[1]); s.layer3 = layer(512, 256, 2, n[2]); s.layer4 = layer(1024, 512, 2, n[3])
        else:
            def layer(i, o, st, k): return nn.Sequential(Basic(i, o, st), *[Basic(o, o, 1) for _ in range(k - 1)])
            s.layer1 = layer(64, 64, 1, n[0]); s.layer2 = layer(64, 128, 2, n[1]); s.layer3 = layer(128, 256, 2, n[2]); s.layer4 = layer(256, 512, 2, n[3])
        s.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); s.bn1 = nn.BatchNorm2d(64); s.relu = nn.ReLU(inplace=True); s.maxpool = nn.MaxPool2d(3, 2, 1)
        s.out_dim = 2048 if depth == 50 else 512
        s.URL = s.URLS[depth]
        s.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); s.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained:
            sd = torch.hub.load_state_dict_from_url(s.URL, map_location='cpu', progress=False)
            miss = s.load_state_dict({k: v for k, v in sd.items() if not k.startswith('fc.')}, strict=False)
            assert miss.missing_keys == ['mean', 'std'] and not miss.unexpected_keys, miss
    def forward(s, x):
        x = (x[:, [2, 1, 0]] - s.mean) / s.std
        if s.up != 1.0: x = nn.functional.interpolate(x, scale_factor=s.up, mode='bilinear', align_corners=False)
        x = s.maxpool(s.relu(s.bn1(s.conv1(x))))
        x = s.layer4(s.layer3(s.layer2(s.layer1(x))))
        return x.mean((2, 3))


class NetR18(Net):
    def __init__(s, pretrained=True, depth=18, up=1.0):
        nn.Module.__init__(s)
        s.f = R18(pretrained, depth, up)
        s.h = nn.Sequential(nn.Linear(2 * s.f.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))


class Wrap(nn.Module):
    """서버 입력: [2,3,H,W] RGB 0~1 (0=사진 왼쪽 조각, 1=오른쪽 조각을 뒤집은 것, 둘 다 바깥쪽이 왼쪽). 출력: [1] 0번 − 1번"""
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x):
        e = s.n.f(x[:, [2, 1, 0]])
        return s.n.pair(e[0:1], e[1:2]).reshape(1)


def t(x):  # uint8 BGR [N,H,W,3] → GPU float [N,3,H,W] (학습 데이터와 서버 둘 다 BGR 순서로 망에 들어감)
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


def evaluate(score_fn, save=True):
    """평가 묶음 [N,6,7,2,H,W,3] → R[N,6,7]; 서버처럼 안경 ≤0.02·밝기 차이 ≤15만 남겨 실사진 p95 기준 검출률"""
    P = np.load(PACK / 'eval_p.npy', mmap_mode='r'); M = np.load(PACK / 'eval_meta.npz')
    R = np.zeros(P.shape[:3], np.float32)
    for i in range(len(P)):
        x = np.asarray(P[i]).reshape(-1, 2, *P.shape[4:])  # [42,2,H,W,3]
        R[i] = score_fn(x).reshape(P.shape[1:3])
    if save: np.save(OUT / 'R.npy', R)
    return report(R, M)


def report(R, M):
    SG = M['sg']; g = (M['glasses'] <= 0.02) & (M['light'] <= 15); Rm = R.mean(2)
    th = np.percentile(np.abs(Rm[g, 0]), 95); K = ['shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']
    out = {'name': args.name, 'n': int(g.sum()), 'p95': round(float(th), 3)}
    for k, n in enumerate(K): out[n] = round(float(np.mean(Rm[g, k + 1] * SG[g] > th)), 3)
    out['all'] = round(float(np.mean([out[n] for n in K])), 3)
    return out


if args.eval_onnx:  # 기존 onnx 평가만(서버 모델 기준점)
    import onnxruntime as ort
    m = ort.InferenceSession(args.eval_onnx, providers=['CPUExecutionProvider'])
    def sf(x):
        xr = (x[..., ::-1].astype(np.float32) / 255).transpose(0, 1, 4, 2, 3)
        return np.array([m.run(None, {'patches': np.ascontiguousarray(xr[j])})[0][0] for j in range(len(xr))])
    res = evaluate(sf); json.dump(res, open(OUT / 'eval.json', 'w')); print(json.dumps(res)); sys.exit(0)

A = np.load(PACK / 'a.npy', mmap_mode='r'); B = np.load(PACK / 'b.npy', mmap_mode='r'); Y = np.load(PACK / 'y.npy')
n = len(Y); idx = np.random.default_rng(0).permutation(n); va = np.sort(idx[: n // 10]); tr = idx[n // 10:]
CK_EXISTS = (OUT / 'ck.pt').exists()
print(args.name, args.arch, 'pairs', n, 'device', dev, 'free GB', round(mem_free_gb(), 2), flush=True)
net = (NetR18(not CK_EXISTS, int(args.arch[1:]), args.up) if args.arch != 'small' else Net(args.width)).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
steps = len(tr) // args.batch + 1
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps)
scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
CK = OUT / 'ck.pt'; start = 0
if CK.exists():
    c = torch.load(CK, weights_only=False, map_location=dev)
    net.load_state_dict(c['net']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched']); scaler.load_state_dict(c['scaler'])
    np.random.set_state(c['np']); torch.set_rng_state(c['torch'].cpu().to(torch.uint8)); start = c['ep'] + 1
    print('resume from ep', start, flush=True)


def aug(x):  # 한쪽씩 따로 밝기·대비 흔들기(좌우 조명 차이에 강해짐), 작은 이동
    k = len(x)
    x = x * (0.7 + 0.6 * torch.rand(k, 1, 1, 1, device=dev)) + 0.1 * (torch.rand(k, 1, 1, 1, device=dev) - 0.5)
    dx, dy = np.random.randint(-6, 7), np.random.randint(-4, 5)
    return torch.roll(x, (dy, dx), (2, 3)).clamp(0, 1)


def batches(order):
    """mmap에서 미리 읽어 두는 스레드(디스크 읽기와 GPU 계산을 겹침)"""
    q = queue.Queue(maxsize=6)
    def run():
        for i in range(0, len(order), args.batch):
            j = np.sort(order[i:i + args.batch]); q.put((A[j], B[j], Y[j]))
        q.put(None)
    threading.Thread(target=run, daemon=True).start()
    while (b := q.get()) is not None: yield b


def save(ep):
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                    np=np.random.get_state(), torch=torch.get_rng_state(), ep=ep), str(CK) + '.tmp')
    os.replace(str(CK) + '.tmp', CK)


log = open(OUT / 'train.log', 'a')
for ep in range(start, args.epochs):
    net.train(); tot = 0; t0 = time.time()
    order = np.random.permutation(tr)
    for s, (a, b, y) in enumerate(batches(order)):
        if s % 200 == 0 and (mem_free_gb() < args.min_free or other_job()):
            print(f'메모리 부족(남은 {mem_free_gb():.2f}GB) 또는 다른 학습({LOCK}) → 멈춤. ep{ep}는 처음부터 다시', flush=True)
            sys.exit(75)
        a, b, y = t(a), t(b), torch.from_numpy(y).to(dev)
        if np.random.rand() < .5: a, b, y = b, a, -y   # 좌우 바꿔도 같은 규칙
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            p = net(aug(a), aug(b))
        loss = nn.functional.smooth_l1_loss(p.float(), y, beta=0.05)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(y)
    net.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(va), 256):
            j = va[i:i + 256]; ps.append(net(t(A[j]), t(B[j])).float().cpu())
    p = torch.cat(ps).numpy(); yv = Y[va]; th = np.percentile(np.abs(p[yv == 0]), 95); e = yv != 0
    msg = f"ep{ep} loss {tot/len(tr):.4f} val corr {np.corrcoef(p, yv)[0,1]:.3f} detect@p95 {np.mean(np.abs(p[e]) > th):.0%} {time.time()-t0:.0f}s free {mem_free_gb():.1f}GB"
    def sf_ep(x):
        with torch.no_grad(): return net(t(x[:, 0]), t(x[:, 1])).float().cpu().numpy()
    if (PACK / 'eval_p.npy').exists():  # 실사진 점수도 에폭마다 기록(고르는 데 쓰지 말고 몇 에폭이 맞는지 보는 용도)
        r = evaluate(sf_ep, save=False); msg += f" | real all {r['all']:.3f} (" + ' '.join(f"{r[k]:.2f}" for k in ['shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']) + ')'
    print(msg, flush=True); log.write(msg + '\n'); log.flush(); save(ep)

net.eval(); torch.save(net.state_dict(), OUT / 'eye.pt')
w = Wrap(net.float().cpu()).eval()
torch.onnx.export(w, torch.rand(2, 3, 96, 120), str(OUT / f'eye_{args.name}.onnx'), input_names=['patches'], output_names=['score'], opset_version=17, dynamo=False)
net.to(dev)
def sf(x):
    with torch.no_grad(): return net(t(x[:, 0]), t(x[:, 1])).float().cpu().numpy()
res = evaluate(sf); res.update(seed=args.seed, epochs=args.epochs, width=args.width, lr=args.lr, arch=args.arch, up=args.up)
json.dump(res, open(OUT / 'eval.json', 'w')); print('EVAL', json.dumps(res), flush=True)
with open(HERE / 'out/jetson/results.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
