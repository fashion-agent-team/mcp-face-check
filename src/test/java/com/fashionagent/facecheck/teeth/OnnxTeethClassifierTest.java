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
}
