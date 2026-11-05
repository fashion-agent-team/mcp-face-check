package com.fashionagent.facecheck.makeup;

import java.util.List;

/**
 * 두 눈 중심으로 정한 얼굴 좌표. 두 눈 가운데가 (0, 0), 두 눈을 잇는 선이 x축(고개 기울임을 돌려 맞춤),
 * 두 눈 사이 거리(IOD)가 1입니다. 왼쪽 눈은 (-0.5, 0), 오른쪽 눈은 (0.5, 0), y는 아래로 커집니다.
 * 얼굴 가운데 선은 x = 0이고, 좌우 비교는 오른쪽 값을 x → -x로 뒤집어 왼쪽과 맞춰 봅니다.
 */
final class FaceFrame {

    final double ox;
    final double oy;
    final double cos;
    final double sin;
    /** 픽셀 단위 두 눈 사이 거리 */
    final double iod;

    private FaceFrame(Pt leftEye, Pt rightEye) {
        this.ox = (leftEye.x() + rightEye.x()) / 2;
        this.oy = (leftEye.y() + rightEye.y()) / 2;
        double dx = rightEye.x() - leftEye.x();
        double dy = rightEye.y() - leftEye.y();
        this.iod = Math.hypot(dx, dy);
        this.cos = dx / iod;
        this.sin = dy / iod;
    }

    static FaceFrame of(Pt leftEye, Pt rightEye) {
        return new FaceFrame(leftEye, rightEye);
    }

    /** 사진에서 고개가 기운 각도(도). 오른쪽 눈이 더 아래면 +. */
    double rollDegrees() {
        return Math.toDegrees(Math.atan2(sin, cos));
    }

    Pt local(Pt px) {
        double dx = px.x() - ox;
        double dy = px.y() - oy;
        return new Pt((dx * cos + dy * sin) / iod, (-dx * sin + dy * cos) / iod);
    }

    List<Pt> local(List<Pt> px) {
        return px.stream().map(this::local).toList();
    }

    Pt pixel(Pt local) {
        double rx = local.x() * iod;
        double ry = local.y() * iod;
        return new Pt(ox + rx * cos - ry * sin, oy + rx * sin + ry * cos);
    }
}
