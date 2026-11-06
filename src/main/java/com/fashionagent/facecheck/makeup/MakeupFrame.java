package com.fashionagent.facecheck.makeup;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.springframework.ai.tool.annotation.ToolParam;

/** 얼굴이 잘 잡혀 멈춘 1초 동안 앱이 더 보내는 장면 하나. */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MakeupFrame(
        @ToolParam(description = "셀카 방향 JPEG base64") String imageBase64,
        @ToolParam(required = false, description = "그 장면의 ML Kit 윤곽점") FaceLandmarks landmarks) {
}
