"""얼굴 분할 모델(BiSeNet ResNet34)을 서버가 실제로 넣는 구도로 미세조정한다.

왜: 공개 모델은 CelebAMask-HQ 원본 구도(1024→512 축소, 448 무작위 자르기)로 학습했다. 서버(OnnxFaceParser)는
두 눈을 기준으로 돌려 맞춘 512x512(눈 사이의 4.2배 정사각형, 가운데는 눈보다 0.6 아래)를 넣는다.
그래서 학습 사진도 같은 방식으로 잘라 구도를 맞추고, 가장 약한 눈썹(2·3)과 입술(12·13)에 손실 가중치를 더 준다.
눈 위치는 정답 분할(눈 4·5)의 무게중심으로 잡는다(ML Kit 눈 윤곽의 가운데와 비슷함).

데이터: CelebAMask-HQ(비상업 연구용, 사진은 저장소에 넣지 않음). 아래 둘 중 하나의 폴더 구조.
  <root>/CelebA-HQ-img/*.jpg + <root>/CelebAMask-HQ-mask-anno/<0..14>/<id5>_<부위>.png  (원본 배포 형태)
  <root>/CelebA-HQ-img/*.jpg + <root>/mask/<id>.png                                   (이미 합친 분류 지도)
  <root>/CelebA-HQ-to-CelebA-mapping.txt 가 있으면 --exclude 목록(CelebA 파일 이름)의 사진은 학습·검증에서 뺀다.
  검출률 평가(public_baseline.py)에 쓰는 CelebA 사진과 겹치지 않게 하려는 것.

실행:
  python3 finetune_face_parsing.py <root> --weights resnet34.pt --exclude eval_files.txt --out ft.onnx
  → ft.onnx(서버 입출력, fp32), ft_fp16.onnx(서버용), ft.pt(가중치), 검증 IoU 로그
필요: pip install torch numpy opencv-python-headless onnx onnxconverter-common
"""
import argparse
import pathlib
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from face_parsing_net import BiSeNet, Labels

N = 512
SPAN, CENTER_Y = 4.2, 0.6  # OnnxFaceParser와 같게
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255
# 서버 분류 번호 순서(그전 모델과 같음). 원본 부위 그림을 이 순서로 칠하고, 뒤에 칠한 것이 앞을 덮는다.
PARTS = ["skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth",
         "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat"]
FLIP = {2: 3, 3: 2, 4: 5, 5: 4, 7: 8, 8: 7}
NAMES = {"brow": [2, 3], "eye": [4, 5], "lip": [12, 13], "skin": [1], "nose": [10], "hair": [17]}


def read_mask(root, idx):
    p = root / "mask" / f"{idx}.png"
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    m = np.zeros((512, 512), np.uint8)
    folder = root / "CelebAMask-HQ-mask-anno" / str(idx // 2000)
    for c, part in enumerate(PARTS, 1):
        q = folder / f"{idx:05d}_{part}.png"
        if q.exists():
            m[cv2.imread(str(q), cv2.IMREAD_GRAYSCALE) > 127] = c
    return m


def eyes(mask):
    """정답 분할에서 두 눈의 무게중심(마스크 좌표). 눈이 안 보이면 None."""
    pts = []
    for c in (4, 5):
        ys, xs = np.nonzero(mask == c)
        if len(xs) < 15:
            return None
        pts.append((xs.mean() + 0.5, ys.mean() + 0.5))
    return sorted(pts)


def crop(img, mask, e, jitter):
    """서버와 같은 얼굴 좌표 구도로 512x512를 자른다. jitter면 크기·기울기·위치를 조금 흔든다."""
    (x1, y1), (x2, y2) = e
    s = img.shape[0] / mask.shape[0]  # 사진은 1024, 지도는 512
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    iod = np.hypot(x2 - x1, y2 - y1)
    ang = np.arctan2(y2 - y1, x2 - x1)
    if jitter:
        iod *= random.uniform(0.92, 1.08)
        ang += np.radians(random.uniform(-6, 6))
        mx += random.uniform(-0.06, 0.06) * iod
        my += random.uniform(-0.06, 0.06) * iod
    cell = SPAN / N
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * iod
    # 출력 칸 (i, j)의 가운데 = 얼굴 좌표 (x0 + (i+.5)cell, y0 + (j+.5)cell) → 지도 좌표(픽셀 모서리 기준) m + R·그것
    A = R * cell
    t = np.array([mx, my]) + R @ np.array([-SPAN / 2 + cell / 2, CENTER_Y - SPAN / 2 + cell / 2])

    def cv(scale):  # cv2 WARP_INVERSE_MAP은 정수 좌표가 픽셀 가운데라 0.5를 뺀다
        return np.hstack([A * scale, (t * scale - 0.5)[:, None]])

    im = cv2.warpAffine(img, cv(s), (N, N), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    lb = cv2.warpAffine(mask, cv(1.0), (N, N), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
    return im, lb


class Faces(Dataset):
    def __init__(s, root, ids, train):
        s.root, s.ids, s.train = root, ids, train

    def __len__(s):
        return len(s.ids)

    def __getitem__(s, k):
        idx = s.ids[k]
        img = cv2.cvtColor(cv2.imread(str(s.root / "CelebA-HQ-img" / f"{idx}.jpg")), cv2.COLOR_BGR2RGB)
        mask = read_mask(s.root, idx)
        e = eyes(mask)
        if e is None:  # 목록 만들 때 걸렀지만 혹시 모르니
            return s[(k + 1) % len(s)]
        im, lb = crop(img, mask, e, s.train)
        im = im.astype(np.float32) / 255
        if s.train:
            if random.random() < 0.5:
                im = im[:, ::-1].copy()
                lb = lb[:, ::-1].copy()
                sw = lb.copy()
                for a, b in FLIP.items():
                    sw[lb == a] = b
                lb = sw
            im = im * random.uniform(0.75, 1.25) + random.uniform(-0.08, 0.08)  # 대비·밝기
            g = im.mean(2, keepdims=True)
            im = np.clip(g + (im - g) * random.uniform(0.7, 1.3), 0, 1)  # 채도
            if random.random() < 0.5:  # 휴대폰 JPEG
                q = random.randint(55, 95)
                im = cv2.imdecode(cv2.imencode(".jpg", (im * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                                  cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        x = ((im - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(lb.astype(np.int64))


def ohem(logits, target, weight, thresh=0.7, frac=1 / 16):
    loss = F.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE, reduction="none").view(-1)
    loss = loss[target.view(-1) != IGNORE]
    if loss.numel() == 0:
        return logits.sum() * 0
    n_min = max(1, int(loss.numel() * frac))
    loss, _ = loss.sort(descending=True)
    th = -np.log(thresh)
    return loss[loss > th].mean() if loss[min(n_min, loss.numel() - 1)] > th else loss[:n_min].mean()


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    inter = torch.zeros(19)
    union = torch.zeros(19)
    for x, y in loader:
        p = net(x.to(dev))[0].argmax(1).cpu()
        ok = y != IGNORE
        for c in range(19):
            a, b = (p == c) & ok, (y == c) & ok
            inter[c] += (a & b).sum()
            union[c] += (a | b).sum()
    net.train()
    return {k: float(inter[v].sum() / union[v].sum().clamp(min=1)) for k, v in NAMES.items()}


def ids_of(root, exclude):
    ids = sorted(int(p.stem) for p in (root / "CelebA-HQ-img").glob("*.jpg"))
    mp = root / "CelebA-HQ-to-CelebA-mapping.txt"
    if exclude:
        if not mp.exists():
            raise SystemExit(f"{mp}이 없어 평가 사진과 겹치는 것을 뺄 수 없어요")
        ex = set(pathlib.Path(exclude).read_text().split())
        orig = {int(p[0]): p[2] for p in (l.split() for l in mp.read_text().split("\n")[1:]) if len(p) == 3}
        before = len(ids)
        ids = [i for i in ids if orig.get(i) not in ex]
        print(f"평가 사진과 겹치는 {before - len(ids)}장을 뺐어요")
    keep = [i for i in ids if eyes(read_mask(root, i)) is not None]
    print(f"학습에 쓸 사진 {len(keep)}장(두 눈이 보이는 것)")
    return keep


def export(net, out):
    import onnx
    from onnxconverter_common import float16
    net = net.cpu().eval()
    torch.onnx.export(Labels(net), torch.zeros(1, 3, N, N), out, input_names=["image"], output_names=["labels"],
                      opset_version=17, dynamo=False)
    fp16 = out.with_name(out.stem + "_fp16.onnx")
    onnx.save(float16.convert_float_to_float16(onnx.load(out), keep_io_types=True), fp16)
    print("저장:", out, fp16)


def main(a):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    root = pathlib.Path(a.root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ids = ids_of(root, a.exclude)
    random.Random(0).shuffle(ids)
    val, train = ids[:a.val], ids[a.val:]
    tl = DataLoader(Faces(root, train, True), a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(Faces(root, val, False), 8, num_workers=a.workers)

    net = BiSeNet(19, "resnet34")
    net.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    net.to(dev).train()
    w = torch.ones(19)
    w[[2, 3, 12, 13]] = a.part_weight
    w = w.to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / a.iters) ** 0.9)

    base = evaluate(net, vl, dev)
    print("미세조정 전 검증 IoU:", {k: round(v, 4) for k, v in base.items()}, flush=True)
    best, it, t0 = base["brow"] + base["lip"], 0, time.time()
    out = pathlib.Path(a.out)
    torch.save(net.state_dict(), out.with_suffix(".pt"))
    while it < a.iters:
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            o, o16, o32 = net(x)
            loss = ohem(o, y, w) + ohem(o16, y, w) + ohem(o32, y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % 50 == 0:
                print(f"{it}/{a.iters} 손실 {loss.item():.4f} {(time.time() - t0) / it:.2f}s/번", flush=True)
            if it % a.eval_every == 0 or it == a.iters:
                r = evaluate(net, vl, dev)
                score = r["brow"] + r["lip"]
                print(f"{it}번 검증 IoU:", {k: round(v, 4) for k, v in r.items()}, "(최고)" if score > best else "", flush=True)
                if score > best:
                    best = score
                    torch.save(net.state_dict(), out.with_suffix(".pt"))
            if it >= a.iters:
                break
    net.load_state_dict(torch.load(out.with_suffix(".pt"), map_location="cpu", weights_only=True))
    export(net, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--weights", required=True)
    p.add_argument("--exclude")
    p.add_argument("--out", default="ft.onnx")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--part-weight", type=float, default=2.0)
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    main(p.parse_args())
