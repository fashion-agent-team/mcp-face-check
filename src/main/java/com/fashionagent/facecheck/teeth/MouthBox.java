package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.FaceCheckException;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/** 앱이 ML Kit 입술 윤곽으로 구한 입 영역. 사진 크기를 1로 본 비율, 셀카 방향. */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MouthBox(double x, double y, double w, double h) {

    void validate() {
        boolean ok = Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(w) && Double.isFinite(h)
                && w > 0 && h > 0 && w <= 1.5 && h <= 1.5 && x > -1 && y > -1 && x < 1 && y < 1;
        if (!ok) {
            throw FaceCheckException.badRequest("mouthBox는 사진 크기를 1로 본 비율이어야 해요.");
        }
    }
}
