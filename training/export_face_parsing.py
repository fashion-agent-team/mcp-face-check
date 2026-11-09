"""얼굴 분할 모델(BiSeNet, CelebAMask-HQ 19분류)을 서버용 ONNX(fp16)로 만든다.
→ src/main/resources/models/face_parsing_bisenet_fp16.onnx (서버 OnnxFaceParser가 읽음)

가중치: zllrunning/face-parsing.PyTorch의 79999_iter.pth(MIT 코드, CelebAMask-HQ로 학습 → 비상업 연구용).
원래 배포처(Google Drive)가 막힌 곳에서도 받을 수 있게 공개 Docker Hub 이미지(eygsoft/face-parsing)의 레이어에서 꺼낸다.
모델 구조는 원래 코드와 같게 여기서 새로 썼고, 가중치는 weights_only=True로만 읽는다(피클 코드 실행 없음).

필요: pip install torch onnx onnxconverter-common onnxruntime
실행: python3 export_face_parsing.py [--weights 79999_iter.pth]
"""
import argparse
import io
import json
import pathlib
import subprocess
import tarfile
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE = "eygsoft/face-parsing"
WEIGHTS_SHA256 = "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567"
OUT = pathlib.Path(__file__).parent.parent / "src/main/resources/models/face_parsing_bisenet_fp16.onnx"

class Block(nn.Module):
    def __init__(s, i, o, st=1):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.bn1 = nn.BatchNorm2d(o)
        s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.bn2 = nn.BatchNorm2d(o)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if i != o or st != 1 else None
    def forward(s, x):
        r = s.bn2(s.conv2(F.relu(s.bn1(s.conv1(x)))))
        return F.relu((s.downsample(x) if s.downsample is not None else x) + r)

class Resnet18(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); s.bn1 = nn.BatchNorm2d(64); s.maxpool = nn.MaxPool2d(3, 2, 1)
        L = lambda i, o, st: nn.Sequential(Block(i, o, st), Block(o, o))
        s.layer1, s.layer2, s.layer3, s.layer4 = L(64, 64, 1), L(64, 128, 2), L(128, 256, 2), L(256, 512, 2)
    def forward(s, x):
        x = s.layer1(s.maxpool(F.relu(s.bn1(s.conv1(x)))))
        f8 = s.layer2(x); f16 = s.layer3(f8); return f8, f16, s.layer4(f16)

class CBR(nn.Module):
    def __init__(s, i, o, k=3, st=1, p=1):
        super().__init__(); s.conv = nn.Conv2d(i, o, k, st, p, bias=False); s.bn = nn.BatchNorm2d(o)
    def forward(s, x): return F.relu(s.bn(s.conv(x)))

class ARM(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.conv = CBR(i, o); s.conv_atten = nn.Conv2d(o, o, 1, bias=False); s.bn_atten = nn.BatchNorm2d(o)
    def forward(s, x):
        f = s.conv(x); a = torch.sigmoid(s.bn_atten(s.conv_atten(f.mean((2, 3), keepdim=True)))); return f * a

class ContextPath(nn.Module):
    def __init__(s):
        super().__init__()
        s.resnet = Resnet18(); s.arm16 = ARM(256, 128); s.arm32 = ARM(512, 128)
        s.conv_head32 = CBR(128, 128); s.conv_head16 = CBR(128, 128); s.conv_avg = CBR(512, 128, 1, 1, 0)
    def forward(s, x):
        f8, f16, f32 = s.resnet(x)
        avg = s.conv_avg(f32.mean((2, 3), keepdim=True))
        up32 = s.conv_head32(F.interpolate(s.arm32(f32) + avg, f16.shape[2:], mode="nearest"))
        up16 = s.conv_head16(F.interpolate(s.arm16(f16) + up32, f8.shape[2:], mode="nearest"))
        return f8, up16

class FFM(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.convblk = CBR(i, o, 1, 1, 0)
        s.conv1 = nn.Conv2d(o, o // 4, 1, bias=False); s.conv2 = nn.Conv2d(o // 4, o, 1, bias=False)
    def forward(s, a, b):
        f = s.convblk(torch.cat([a, b], 1))
        at = torch.sigmoid(s.conv2(F.relu(s.conv1(f.mean((2, 3), keepdim=True))))); return f * at + f

class Out(nn.Module):
    def __init__(s, i, m, n):
        super().__init__(); s.conv = CBR(i, m); s.conv_out = nn.Conv2d(m, n, 1, bias=False)
    def forward(s, x): return s.conv_out(s.conv(x))

class BiSeNet(nn.Module):
    def __init__(s, n=19):
        super().__init__()
        s.cp = ContextPath(); s.ffm = FFM(256, 256); s.conv_out = Out(256, 256, n)
        s.conv_out16 = Out(128, 64, n); s.conv_out32 = Out(128, 64, n)  # 학습용 보조 출력(가중치 맞추려고만 둠)
    def forward(s, x):
        f8, f16 = s.cp(x)
        o = s.conv_out(s.ffm(f8, f16))
        return F.interpolate(o, x.shape[2:], mode="bilinear", align_corners=True).argmax(1).to(torch.uint8)


def fetch_weights(path):
    """Docker Hub 이미지의 app 레이어(약 49MB)에서 79999_iter.pth만 꺼냄."""
    tok = json.load(urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{IMAGE}:pull"))["token"]
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{IMAGE}/manifests/latest", headers={
        "Authorization": f"Bearer {tok}", "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    layers = json.load(urllib.request.urlopen(req))["layers"]
    for layer in sorted(layers, key=lambda l: l["size"]):
        if not 40e6 < layer["size"] < 60e6:
            continue
        blob = subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {tok}",
                               f"https://registry-1.docker.io/v2/{IMAGE}/blobs/{layer['digest']}"],
                              capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for m in tar:
                if m.name.endswith("79999_iter.pth"):
                    path.write_bytes(tar.extractfile(m).read())
                    return path
    raise SystemExit("가중치를 찾지 못했어요")


def main(a):
    import hashlib
    import onnx
    from onnxconverter_common import float16
    w = pathlib.Path(a.weights)
    if not w.exists():
        fetch_weights(w)
    if hashlib.sha256(w.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise SystemExit("가중치 파일이 예상과 달라요(sha256)")
    net = BiSeNet()
    missing, unexpected = net.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    net.eval()
    tmp = OUT.with_suffix(".fp32.onnx")
    torch.onnx.export(net, torch.randn(1, 3, 512, 512), tmp, input_names=["image"], output_names=["labels"],
                      opset_version=17, dynamo=False)
    onnx.save(float16.convert_float_to_float16(onnx.load(tmp), keep_io_types=True), OUT)
    tmp.unlink()
    print("저장:", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(pathlib.Path(__file__).parent / ".cache/79999_iter.pth"))
    main(p.parse_args())
