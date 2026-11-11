"""좌우 비교 모델 학습(눈썹 길이, 입술 너비): 좌우 조각을 같은 작은 CNN에 넣어 점수를 내고, 왼쪽 − 오른쪽이 정답(부호 있음)이 되게 학습.
같은 CNN을 양쪽에 쓰고 빼므로 좌우를 바꾸면 부호만 바뀜(타고난 비대칭을 학습하지 않음). CPU 4코어 약 25분.
끝나면 src/main/resources/models/<이름>.onnx로 내보냄. 기준값은 서버를 켜고 public_baseline.py로 다시 재서 brow-length-model·lip-half-width-model에.

사용: python train_pair_model.py brow_length|brow_thickness|lip_width [에폭=15]  (먼저 brow_length_data.py 또는 lip_width_data.py)"""
import sys, json, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(4)
import pathlib
HERE = pathlib.Path(__file__).parent
NAME = sys.argv[1]
d = np.load(HERE / f'out/{NAME}_train.npz'); A, B, Y = d['a'], d['b'], d['y'].astype(np.float32)
n = len(Y); idx = np.random.default_rng(0).permutation(n); va = idx[: n // 10]; tr = idx[n // 10:]
def t(x): return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 96), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1))
    def forward(s, a, b): return (s.f(a) - s.f(b)).squeeze(1)
net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 15
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * (len(tr) // 64 + 1))
def aug(x):
    # 밝기·대비 흔들기, 작은 이동
    x = x * (0.7 + 0.6 * torch.rand(len(x), 1, 1, 1)) + 0.1 * (torch.rand(len(x), 1, 1, 1) - 0.5)
    dx, dy = np.random.randint(-6, 7), np.random.randint(-4, 5)
    return torch.roll(x, (dy, dx), (2, 3)).clamp(0, 1)
for ep in range(EP):
    net.train(); rng = np.random.permutation(tr); tot = 0
    for i in range(0, len(rng), 64):
        j = rng[i:i + 64]; a, b, y = t(A[j]), t(B[j]), torch.from_numpy(Y[j])
        if np.random.rand() < .5: a, b, y = b, a, -y   # 좌우 바꿔도 같은 규칙
        p = net(aug(a), aug(b)); loss = nn.functional.smooth_l1_loss(p, y, beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(j)
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(t(A[va[i:i+256]]), t(B[va[i:i+256]])) for i in range(0, len(va), 256)]).numpy()
    yv = Y[va]; pl = np.abs(p[yv == 0]); th = np.percentile(pl, 95); e = yv != 0
    print(f"ep{ep} loss {tot/len(tr):.4f} val corr {np.corrcoef(p, yv)[0,1]:.3f} detect@p95 {np.mean(np.abs(p[e]) > th):.0%}", flush=True)
# 검증은 같은 사진의 다른 편집이 섞여 있어서 낙관적임. 진짜 평가는 public_baseline.py(평가 사진은 학습에 안 씀)

net.eval()
class Wrap(nn.Module):
    """입력: [2,3,H,W] RGB 0~1 (0=사진 왼쪽 조각, 1=오른쪽 조각을 좌우로 뒤집은 것, 둘 다 바깥쪽이 왼쪽). 출력: [1] 0번 − 1번"""
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x):
        x = x[:, [2, 1, 0]]
        e = s.n.f(x)
        return (e[0:1] - e[1:2]).reshape(1)
w = Wrap(net).eval()
x = torch.rand(2, 3, A.shape[1], A.shape[2])
torch.onnx.export(w, x, str(HERE.parent / f'src/main/resources/models/{NAME}.onnx'), input_names=['patches'], output_names=['score'], opset_version=17, dynamo=False)
