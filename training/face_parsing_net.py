"""BiSeNet 얼굴 분할 모델 구조(ResNet18/34 백본). finetune_face_parsing.py가 씀.

yakhyo/face-parsing(MIT)이 공개한 resnet34.pt 가중치를 그대로 읽을 수 있게 층 이름과 계산 순서를 그 코드와 같게 여기서 새로 썼다.
가중치는 weights_only=True로만 읽는다(피클 코드 실행 없음). ImageNet 가중치를 따로 받지 않는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(s, i, o, st=1):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, st, 1, bias=False)
        s.bn1 = nn.BatchNorm2d(o)
        s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        s.bn2 = nn.BatchNorm2d(o)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if i != o or st != 1 else None

    def forward(s, x):
        r = s.bn2(s.conv2(F.relu(s.bn1(s.conv1(x)))))
        return F.relu((s.downsample(x) if s.downsample is not None else x) + r)


class ResNet(nn.Module):
    def __init__(s, layers):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        s.bn1 = nn.BatchNorm2d(64)
        s.maxpool = nn.MaxPool2d(3, 2, 1)

        def make(i, o, n, st):
            return nn.Sequential(BasicBlock(i, o, st), *[BasicBlock(o, o) for _ in range(n - 1)])

        s.layer1 = make(64, 64, layers[0], 1)
        s.layer2 = make(64, 128, layers[1], 2)
        s.layer3 = make(128, 256, layers[2], 2)
        s.layer4 = make(256, 512, layers[3], 2)
        s.fc = nn.Linear(512, 1000)  # 쓰지 않음(가중치 파일에 있어서 이름만 맞춤)

    def forward(s, x):
        x = s.layer1(s.maxpool(F.relu(s.bn1(s.conv1(x)))))
        f8 = s.layer2(x)
        f16 = s.layer3(f8)
        return f8, f16, s.layer4(f16)


class ConvBNReLU(nn.Module):
    def __init__(s, i, o, k=3):
        super().__init__()
        s.conv = nn.Conv2d(i, o, k, 1, k // 2, bias=False)
        s.norm = nn.BatchNorm2d(o)

    def forward(s, x):
        return F.relu(s.norm(s.conv(x)))


class ARM(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.conv_block = ConvBNReLU(i, o)
        s.attention = nn.Sequential(nn.Conv2d(o, o, 1, bias=False), nn.BatchNorm2d(o), nn.Sigmoid())

    def forward(s, x):
        f = s.conv_block(x)
        return f * s.attention(f.mean((2, 3), keepdim=True))


class ContextPath(nn.Module):
    def __init__(s, layers):
        super().__init__()
        s.backbone = ResNet(layers)
        s.arm16 = ARM(256, 128)
        s.arm32 = ARM(512, 128)
        s.conv_head32 = ConvBNReLU(128, 128)
        s.conv_head16 = ConvBNReLU(128, 128)
        s.conv_avg = ConvBNReLU(512, 128, 1)

    def forward(s, x):
        f8, f16, f32 = s.backbone(x)
        avg = s.conv_avg(f32.mean((2, 3), keepdim=True))
        up32 = s.conv_head32(F.interpolate(s.arm32(f32) + avg, f16.shape[2:], mode="nearest"))
        up16 = s.conv_head16(F.interpolate(s.arm16(f16) + up32, f8.shape[2:], mode="nearest"))
        return f8, up16, up32


class FFM(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.conv_block = ConvBNReLU(i, o, 1)
        s.conv1 = nn.Conv2d(o, o // 4, 1, bias=False)
        s.conv2 = nn.Conv2d(o // 4, o, 1, bias=False)

    def forward(s, a, b):
        f = s.conv_block(torch.cat([a, b], 1))
        return f * torch.sigmoid(s.conv2(F.relu(s.conv1(f.mean((2, 3), keepdim=True))))) + f


class Head(nn.Module):
    def __init__(s, i, m, n):
        super().__init__()
        s.conv_block = ConvBNReLU(i, m)
        s.conv = nn.Conv2d(m, n, 1, bias=False)

    def forward(s, x):
        return s.conv(s.conv_block(x))


class BiSeNet(nn.Module):
    """학습 때는 (본 출력, 보조 출력 2개) 로짓, eval()에서도 같음. 서버용은 Labels로 감싸서 내보낸다."""

    def __init__(s, n=19, backbone="resnet34"):
        super().__init__()
        s.fpn = ContextPath({"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}[backbone])
        s.ffm = FFM(256, 256)
        s.conv_out = Head(256, 256, n)
        s.conv_out16 = Head(128, 64, n)
        s.conv_out32 = Head(128, 64, n)

    def forward(s, x):
        f8, up16, up32 = s.fpn(x)
        size = x.shape[2:]
        up = lambda t: F.interpolate(t, size, mode="bilinear", align_corners=True)
        return up(s.conv_out(s.ffm(f8, up16))), up(s.conv_out16(up16)), up(s.conv_out32(up32))


class Labels(nn.Module):
    """서버 계약: 정규화된 1x3x512x512 → 1x512x512 uint8 분류 번호."""

    def __init__(s, net):
        super().__init__()
        s.net = net

    def forward(s, x):
        return s.net(x)[0].argmax(1).to(torch.uint8)
