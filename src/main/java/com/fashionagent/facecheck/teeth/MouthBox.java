package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.FaceCheckException;
import java.util.Map;

/**
 * 앱이 ML Kit 입술 윤곽 UPPER_LIP_BOTTOM + LOWER_LIP_TOP 점으로 구한 입 안쪽 박스(사진 너비·높이를 1로 본 비율).
 * 셀카 방향으로 뒤집은 뒤 좌표입니다. h ÷ w는 입을 벌린 정도입니다.
 */
public record MouthBox(double x, double y, double w, double h) {

    public static MouthBox from(Map<String, Double> m) {
        if (m == null) {
            return null;
        }
        Double x = m.get("x");
        Double y = m.get("y");
        Double w = m.get("w");
        Double h = m.get("h");
        if (x == null || y == null || w == null || h == null) {
            throw FaceCheckException.badRequest("mouthBox에는 x, y, w, h가 모두 있어야 해요.");
        }
        if (w <= 0 || h < 0 || w > 1.5 || h > 1.5 || x < -0.5 || y < -0.5 || x > 1.5 || y > 1.5) {
            throw FaceCheckException.badRequest("mouthBox는 사진 크기를 1로 본 비율로 보내 주세요.");
        }
        return new MouthBox(x, y, w, h);
    }

    public double centerX() {
        return x + w / 2;
    }

    public double centerY() {
        return y + h / 2;
    }

    /** 입 안쪽 높이 ÷ 입 너비. */
    public double openRatio() {
        return h / w;
    }
}
