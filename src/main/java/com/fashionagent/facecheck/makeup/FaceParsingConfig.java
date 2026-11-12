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
            @Value("${face.parsing.model:classpath:models/face_parsing_bisenet_r34_fp16.onnx}") String model) {
        return new OnnxFaceParser(model.isBlank() ? null : loader.getResource(model));
    }

    /** 좌우 비교 모델(눈썹 길이·두께, 입술 너비, 볼 터치, 눈화장). 위치를 비우거나 파일이 없으면 그 항목은 그전 방식으로 잼. */
    @Bean(destroyMethod = "close")
    PairModels pairModels(ResourceLoader loader,
            @Value("${face.brow-length.model:classpath:models/brow_length.onnx}") String browLength,
            @Value("${face.brow-thickness.model:classpath:models/brow_thickness.onnx}") String browThickness,
            @Value("${face.lip-width.model:classpath:models/lip_width.onnx}") String lipWidth,
            @Value("${face.cheek-color.model:classpath:models/cheek_color.onnx}") String cheekColor,
            @Value("${face.eye-makeup.model:classpath:models/eye_makeup.onnx}") String eyeMakeup) {
        return new PairModels(
                new OnnxPairModel(browLength.isBlank() ? null : loader.getResource(browLength), OnnxPairModel.Region.BROW),
                new OnnxPairModel(browThickness.isBlank() ? null : loader.getResource(browThickness), OnnxPairModel.Region.BROW),
                new OnnxPairModel(lipWidth.isBlank() ? null : loader.getResource(lipWidth), OnnxPairModel.Region.LIP_CORNER),
                new OnnxPairModel(cheekColor.isBlank() ? null : loader.getResource(cheekColor), OnnxPairModel.Region.CHEEK),
                new OnnxPairModel(eyeMakeup.isBlank() ? null : loader.getResource(eyeMakeup), OnnxPairModel.Region.EYE));
    }
}
