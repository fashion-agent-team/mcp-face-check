package com.fashionagent.facecheck.makeup;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;
import org.springframework.ai.tool.annotation.ToolParam;

/**
 * 앱이 보내는 ML Kit 얼굴 윤곽점(FaceContour). 이름은 ML Kit 상수를 camelCase로 바꾼 것입니다
 * (LEFT_EYEBROW_TOP → leftEyebrowTop). 점은 [x, y]이고 사진 너비·높이를 1로 본 비율,
 * 셀카 화면 방향으로 좌우를 뒤집은 뒤의 좌표입니다.
 *
 * <p>ML Kit의 left/right가 사진 기준인지 얼굴 주인 기준인지는 버전마다 헷갈리므로, 서버는 이름을 믿지 않고
 * 두 눈의 x 위치로 왼쪽(사진 왼쪽 = 셀카 화면에서 사용자 왼쪽)을 다시 정합니다.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FaceLandmarks(
        @ToolParam(required = false, description = "얼굴 외곽 (FACE, 36점)") List<List<Double>> face,
        @ToolParam(required = false, description = "LEFT_EYEBROW_TOP (5점)") List<List<Double>> leftEyebrowTop,
        @ToolParam(required = false, description = "LEFT_EYEBROW_BOTTOM (5점)") List<List<Double>> leftEyebrowBottom,
        @ToolParam(required = false, description = "RIGHT_EYEBROW_TOP (5점)") List<List<Double>> rightEyebrowTop,
        @ToolParam(required = false, description = "RIGHT_EYEBROW_BOTTOM (5점)") List<List<Double>> rightEyebrowBottom,
        @ToolParam(required = false, description = "LEFT_EYE (16점)") List<List<Double>> leftEye,
        @ToolParam(required = false, description = "RIGHT_EYE (16점)") List<List<Double>> rightEye,
        @ToolParam(required = false, description = "UPPER_LIP_TOP (11점)") List<List<Double>> upperLipTop,
        @ToolParam(required = false, description = "UPPER_LIP_BOTTOM (9점)") List<List<Double>> upperLipBottom,
        @ToolParam(required = false, description = "LOWER_LIP_TOP (9점)") List<List<Double>> lowerLipTop,
        @ToolParam(required = false, description = "LOWER_LIP_BOTTOM (9점)") List<List<Double>> lowerLipBottom,
        @ToolParam(required = false, description = "NOSE_BRIDGE (2점)") List<List<Double>> noseBridge,
        @ToolParam(required = false, description = "NOSE_BOTTOM (3점)") List<List<Double>> noseBottom,
        @ToolParam(required = false, description = "LEFT_CHEEK (1점)") List<List<Double>> leftCheek,
        @ToolParam(required = false, description = "RIGHT_CHEEK (1점)") List<List<Double>> rightCheek,
        @ToolParam(required = false, description = "ML Kit headEulerAngleY (고개 좌우 돌림, 도)") Double yaw,
        @ToolParam(required = false, description = "ML Kit headEulerAngleZ (고개 기울임, 도)") Double roll) {

    /** 대칭 계산에 꼭 필요한 윤곽이 다 있는지. 없으면 사진만으로 봅니다. */
    public boolean complete() {
        return enough(leftEye) && enough(rightEye) && enough(leftEyebrowTop) && enough(rightEyebrowTop)
                && enough(upperLipTop) && enough(lowerLipBottom);
    }

    static boolean enough(List<List<Double>> points) {
        return points != null && points.size() >= 3;
    }
}
