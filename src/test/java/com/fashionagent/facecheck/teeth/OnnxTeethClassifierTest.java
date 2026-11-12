package com.fashionagent.facecheck.teeth;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

class OnnxTeethClassifierTest {

    @Test
    @DisplayName("ONNX Runtime으로 모델을 돌려 sigmoid(logit)을 돌려줌 (시험용 모델: logit = 4 × 평균)")
    void runsModel() throws Exception {
        try (var c = new OnnxTeethClassifier(new ClassPathResource("models/tiny_mean.onnx"), 32)) {
            assertThat(c.ready()).isTrue();
            float[] x = new float[3 * 32 * 32];
            Arrays.fill(x, 0.5f);
            assertThat(c.foodProbability(x)).isCloseTo(1 / (1 + Math.exp(-2.0)), org.assertj.core.data.Offset.offset(1e-5));
            Arrays.fill(x, -1f);
            assertThat(c.foodProbability(x)).isCloseTo(1 / (1 + Math.exp(4.0)), org.assertj.core.data.Offset.offset(1e-5));
        }
    }

    @Test
    @DisplayName("모델 파일이 없으면 ready=false (서버는 뜨고 teeth_check만 not_ready)")
    void missing() throws Exception {
        try (var c = new OnnxTeethClassifier(new ClassPathResource("models/none.onnx"), 320)) {
            assertThat(c.ready()).isFalse();
        }
    }

    @Test
    @DisplayName("실제 치아 모델(models/teeth_mobilenetv3.onnx)이 읽히고 입 크롭 한 장에 0~1 확률을 냄")
    void realModel() throws Exception {
        var res = new ClassPathResource("models/teeth_mobilenetv3.onnx");
        org.junit.jupiter.api.Assumptions.assumeTrue(res.exists(), "모델 파일 없음");
        try (var c = new OnnxTeethClassifier(res, 384)) {
            assertThat(c.ready()).isTrue();
            var img = new java.awt.image.BufferedImage(640, 480, java.awt.image.BufferedImage.TYPE_INT_RGB);
            var g = img.createGraphics();
            g.setColor(new java.awt.Color(150, 60, 70));
            g.fillRect(0, 0, 640, 480);
            g.setColor(new java.awt.Color(235, 230, 215));
            g.fillRect(220, 220, 200, 40);
            g.dispose();
            double p = c.foodProbability(MouthCrop.toTensor(MouthCrop.crop(img, new MouthBox(220 / 640.0, 220 / 480.0, 200 / 640.0, 40 / 480.0)), 384));
            assertThat(p).isBetween(0.0, 1.0);
        }
    }
}
