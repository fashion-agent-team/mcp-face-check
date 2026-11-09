package com.fashionagent.facecheck.makeup;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ResourceLoader;

@Configuration
class FaceParsingConfig {

    /** face.parsing.model: 얼굴 분할 ONNX 위치. 비우거나 파일이 없으면 눈썹을 밝기로 나눠 잼. */
    @Bean(destroyMethod = "close")
    OnnxFaceParser faceParser(ResourceLoader loader,
            @Value("${face.parsing.model:classpath:models/face_parsing_bisenet_fp16.onnx}") String model) {
        return new OnnxFaceParser(model.isBlank() ? null : loader.getResource(model));
    }
}
