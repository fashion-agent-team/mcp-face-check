package com.fashionagent.facecheck;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;
import org.springframework.util.unit.DataSize;

/**
 * application.yml의 face.*. 대칭 기준값은 처음 정한 값이고 실제 사진으로 맞춰 갑니다.
 * 길이는 두 눈 중심 사이 거리(IOD)를 1로 본 값, ratio는 두 값 중 큰 쪽에 대한 차이 비율입니다.
 */
@ConfigurationProperties("face")
public record FaceCheckProperties(
        @DefaultValue("8MB") DataSize maxImageBytes,
        @DefaultValue Symmetry symmetry,
        @DefaultValue Teeth teeth) {

    /** teeth_check. 확률 기준은 첫 값이고 실제 사진으로 평가한 뒤 맞춥니다. */
    public record Teeth(
            /** ONNX 모델 위치 (Spring Resource 경로). */
            @DefaultValue("classpath:models/teeth_mobilenetv3.onnx") String model,
            /** 모델 입력 크기(정사각형 한 변). 모델 .json의 size. */
            @DefaultValue("320") int inputSize,
            /** 이 확률 이상이면 "낀 것 있음". */
            @DefaultValue("0.7") double yesThreshold,
            /** 이 확률 이하면 "없음". 그 사이는 애매함(null). */
            @DefaultValue("0.3") double noThreshold,
            /** 입 안쪽 높이 ÷ 입 너비가 이보다 작으면 "입을 더 벌려 주세요" (학습 데이터가 0.10 이상). */
            @DefaultValue("0.10") double minOpenRatio,
            /** 입 너비가 사진 너비의 이 비율보다 작으면 "더 가까이". */
            @DefaultValue("0.15") double minMouthWidth,
            /** 입 박스가 사진 밖으로 이 비율(박스 크기 대비) 넘게 나가면 잘린 것으로 봄. */
            @DefaultValue("0.1") double maxOutside,
            /** 입 크롭 평균 밝기(L*)가 이보다 낮으면 lighting=dark. */
            @DefaultValue("25") double darkLightness) {
    }

    public record Symmetry(
            /** 눈썹 높이(눈 중심 ~ 눈썹) 좌우 차이, IOD. */
            @DefaultValue("0.07") double browHeight,
            /** 눈썹 꼬리(바깥 끝) 높이 차이, IOD. */
            @DefaultValue("0.09") double browTail,
            /** 눈썹 길이 차이 비율. */
            @DefaultValue("0.15") double browLength,
            /** 눈썹 길이 모델(brow_length.onnx) 점수 기준. 모델이 있을 때 browLength 대신 씀. */
            @DefaultValue("0.045") double browLengthModel,
            /** 눈썹 조각 선명도(라플라시안 ÷ 대비)가 이보다 낮으면 흐린 사진으로 보고 길이 모델을 쓰지 않음(skipped에 browLength). */
            @DefaultValue("0.053") double browSharpness,
            /** 눈썹 두께 차이 비율 (위·아래 윤곽이 다 있을 때). */
            @DefaultValue("0.25") double browThickness,
            /** 눈썹 두께 모델(brow_thickness.onnx) 점수 기준. 모델이 있을 때 browThickness 대신 씀. */
            @DefaultValue("0.010") double browThicknessModel,
            /** 입꼬리 높이 차이, IOD. */
            @DefaultValue("0.05") double lipCorner,
            /** 입술 반쪽 너비: 사진 입꼬리가 윤곽점 입꼬리보다 더 나간 길이의 좌우 차이, IOD. */
            @DefaultValue("0.10") double lipHalfWidth,
            /** 입술 너비 모델(lip_width.onnx) 점수 기준. 모델이 있을 때 lipHalfWidth 대신 씀. */
            @DefaultValue("0.015") double lipHalfWidthModel,
            /** 윗입술·아랫입술 선 높이 차이 비율. */
            @DefaultValue("0.25") double lipLine,
            /** 볼 붉은 정도(CIELAB a*) 차이. */
            @DefaultValue("6") double cheekRedness,
            /** 볼 터치 모델(cheek_color.onnx) 점수 기준. 모델이 있을 때 cheekRedness 대신 씀. */
            @DefaultValue("0.080") double cheekColorModel,
            /** 입술 색 차이 (CIELAB ΔE76). */
            @DefaultValue("10") double lipColor,
            /** 눈화장(아이섀도·아이라인): 눈꺼풀이 같은 쪽 눈 밑 피부보다 어두운 정도(L*)의 좌우 차이. */
            @DefaultValue("15") double eyeMakeup,
            /** 눈화장 모델(eye_makeup.onnx) 점수 기준. 모델이 있을 때 eyeMakeup 대신 씀. */
            @DefaultValue("0.1") double eyeMakeupModel,
            /** 좌우 밝기(L*) 차이가 이보다 크면 한쪽이 그늘진 것으로 보고 색 비교를 건너뜀. */
            @DefaultValue("15") double unevenLight,
            /** 얼굴 평균 밝기(L*)가 이보다 낮으면 lighting=dark. */
            @DefaultValue("28") double darkLightness,
            /** 고개 좌우 돌림(yaw) 허용 각도. */
            @DefaultValue("12") double maxYaw,
            /** 고개 기울임(roll) 허용 각도. 이 안에서는 돌려 맞춰서 비교. */
            @DefaultValue("25") double maxRoll,
            /** 얼굴 윤곽이 가운데에서 한쪽으로 쏠린 비율 (yaw 각도가 없을 때 정면인지 보는 값). */
            @DefaultValue("0.15") double maxFaceSkew,
            /** 한 번에 읽어 줄 문장 수. */
            @DefaultValue("3") int maxNotes) {
    }
}
