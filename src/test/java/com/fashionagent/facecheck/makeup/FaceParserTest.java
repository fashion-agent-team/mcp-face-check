package com.fashionagent.facecheck.makeup;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.facecheck.SyntheticFace;
import java.awt.image.BufferedImage;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

/** 분할 모델이 고른 눈썹 칸으로 재기 */
class FaceParserTest {

    static final int N = 210; // 칸 하나 = 0.02 IOD
    static final double X0 = -2.1;
    static final double Y0 = -1.5;

    /** 얼굴 좌표의 직사각형 눈썹(x는 안쪽→바깥, 위 y, 아래 y)을 칸에 그림 */
    static FaceLabels grid(double[] left, double[] right) {
        byte[] b = new byte[N * N];
        FaceLabels lab = new FaceLabels(b, N, X0, Y0, N * 0.02);
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                double x = lab.x(i);
                double y = lab.y(j);
                if (in(-x, y, left)) {
                    b[j * N + i] = FaceLabels.L_BROW;
                } else if (in(x, y, right)) {
                    b[j * N + i] = FaceLabels.R_BROW;
                } else if (y < -0.9) {
                    b[j * N + i] = FaceLabels.HAIR;
                }
            }
        }
        return lab;
    }

    static boolean in(double x, double y, double[] r) {
        return x >= r[0] && x <= r[1] && y >= r[2] && y <= r[3];
    }

    @Test
    @DisplayName("좌우 같은 눈썹은 높이·길이·두께가 같고, 머리카락 칸은 세지 않음")
    void same() {
        FaceLabels lab = grid(new double[] {0.15, 0.95, -0.45, -0.33}, new double[] {0.15, 0.95, -0.45, -0.33});
        MakeupPixels.Brow l = MakeupPixels.brow(lab, -1, 0.45);
        MakeupPixels.Brow r = MakeupPixels.brow(lab, 1, 0.45);
        assertThat(l.center()).isCloseTo(r.center(), org.assertj.core.data.Offset.offset(0.011));
        assertThat(l.center()).isCloseTo(0.39, org.assertj.core.data.Offset.offset(0.02));
        assertThat(l.length()).isCloseTo(r.length(), org.assertj.core.data.Offset.offset(0.1)); // 경계가 칸 경계와 겹쳐 좌우 한두 칸 차이까지
        assertThat(l.thickness()).isCloseTo(0.12, org.assertj.core.data.Offset.offset(0.021));
    }

    @Test
    @DisplayName("왼쪽만 0.06 올리고 두껍게 그리면 높이·두께 차이로 나옴")
    void raisedAndThicker() {
        FaceLabels lab = grid(new double[] {0.15, 0.95, -0.55, -0.37}, new double[] {0.15, 0.95, -0.45, -0.33});
        MakeupPixels.Brow l = MakeupPixels.brow(lab, -1, 0.45);
        MakeupPixels.Brow r = MakeupPixels.brow(lab, 1, 0.45);
        assertThat(l.center() - r.center()).isCloseTo(0.07, org.assertj.core.data.Offset.offset(0.02));
        assertThat(l.thickness() / r.thickness()).isGreaterThan(1.25);
    }

    @Test
    @DisplayName("눈썹 칸이 거의 없으면 null(밝기로 재는 방법으로 넘어감)")
    void tooFew() {
        FaceLabels lab = grid(new double[] {0.15, 0.2, -0.42, -0.4}, new double[] {0.15, 0.95, -0.45, -0.33});
        assertThat(MakeupPixels.brow(lab, -1, 0.45)).isNull();
    }

    @Test
    @DisplayName("레포에 든 ONNX 모델을 읽고 512x512로 나눔")
    void onnxModel() throws Exception {
        try (OnnxFaceParser p = new OnnxFaceParser(new ClassPathResource("models/face_parsing_bisenet_fp16.onnx"))) {
            assertThat(p.ready()).isTrue();
            var face = new SyntheticFace();
            BufferedImage img = face.image();
            var lm = MakeupAnalyzerTest.JSON.convertValue(face.landmarks(), FaceLandmarks.class);
            List<Pt> le = Pt.pixels(lm.leftEye(), img.getWidth(), img.getHeight());
            List<Pt> re = Pt.pixels(lm.rightEye(), img.getWidth(), img.getHeight());
            FaceLabels lab = p.parse(img, FaceFrame.of(Pt.mean(le), Pt.mean(re)));
            assertThat(lab.size()).isEqualTo(512);
            for (byte b : lab.labels()) {
                assertThat(b & 0xff).isLessThan(19);
            }
        }
    }

    @Test
    @DisplayName("모델이 없으면 ready=false, 분석은 지금처럼 밝기로")
    void missingModel() throws Exception {
        try (OnnxFaceParser p = new OnnxFaceParser(new ClassPathResource("models/none.onnx"))) {
            assertThat(p.ready()).isFalse();
            assertThat(p.parse(new SyntheticFace().image(), FaceFrame.of(new Pt(0, 0), new Pt(10, 0)))).isNull();
        }
    }
}
