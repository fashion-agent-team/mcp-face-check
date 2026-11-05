package com.fashionagent.facecheck;

import java.time.Duration;
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
        @DefaultValue OpenAi openai,
        @DefaultValue Symmetry symmetry) {

    public record OpenAi(
            @DefaultValue("") String apiKey,
            @DefaultValue("gpt-5.4-mini") String model,
            @DefaultValue("https://api.openai.com") String baseUrl,
            @DefaultValue("30s") Duration timeout) {

        public boolean enabled() {
            return apiKey != null && !apiKey.isBlank();
        }
    }

    public record Symmetry(
            /** 눈썹 높이(눈 중심 ~ 눈썹) 좌우 차이, IOD. */
            @DefaultValue("0.07") double browHeight,
            /** 눈썹 꼬리(바깥 끝) 높이 차이, IOD. */
            @DefaultValue("0.09") double browTail,
            /** 눈썹 길이 차이 비율. */
            @DefaultValue("0.15") double browLength,
            /** 눈썹 두께 차이 비율 (위·아래 윤곽이 다 있을 때). */
            @DefaultValue("0.25") double browThickness,
            /** 입꼬리 높이 차이, IOD. */
            @DefaultValue("0.05") double lipCorner,
            /** 입술 반쪽 너비(가운데 ~ 입꼬리) 차이 비율. */
            @DefaultValue("0.12") double lipHalfWidth,
            /** 윗입술·아랫입술 선 높이 차이 비율. */
            @DefaultValue("0.25") double lipLine,
            /** 볼 붉은 정도(CIELAB a*) 차이. */
            @DefaultValue("6") double cheekRedness,
            /** 입술 색 차이 (CIELAB ΔE76). */
            @DefaultValue("10") double lipColor,
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
