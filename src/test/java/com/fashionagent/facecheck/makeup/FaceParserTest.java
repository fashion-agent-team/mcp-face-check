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
                } else if (Math.abs(x) < 0.15 && y > 0.1 && y < 0.6) {
                    b[j * N + i] = FaceLabels.NOSE;
                } else if (y < -0.9) {
                    b[j * N + i] = FaceLabels.HAIR;
                } else if (Math.abs(x) < 1.2) {
                    b[j * N + i] = 1; // 피부
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
        assertThat(l.thickness()).isCloseTo(0.12, org.assertj.core.data.Offset.offset(0.03)); // 칸 경계로 한 줄 더 들어갈 수 있음
        // 길이 = 눈썹 길이 ÷ 얼굴 반쪽 너비(코 가운데 ~ 옆선 1.2)
        assertThat(l.length()).isCloseTo(0.8 / 1.2, org.assertj.core.data.Offset.offset(0.05));
    }

    @Test
    @DisplayName("한쪽 눈썹 꼬리를 짧게 그리면 길이 차이로 나옴")
    void shorter() {
        FaceLabels lab = grid(new double[] {0.15, 0.75, -0.45, -0.33}, new double[] {0.15, 0.95, -0.45, -0.33});
        double l = MakeupPixels.brow(lab, -1, 0.45).length();
        double r = MakeupPixels.brow(lab, 1, 0.45).length();
        assertThat(1 - l / r).isCloseTo(0.25, org.assertj.core.data.Offset.offset(0.06));
    }

    @Test
    @DisplayName("코를 못 찾으면 null(밝기로 재는 방법으로 넘어감)")
    void noNose() {
        FaceLabels lab = grid(new double[] {0.15, 0.95, -0.45, -0.33}, new double[] {0.15, 0.95, -0.45, -0.33});
        byte[] b = lab.labels();
        for (int k = 0; k < b.length; k++) {
            if (b[k] == FaceLabels.NOSE) {
                b[k] = 1;
            }
        }
        assertThat(MakeupPixels.brow(lab, -1, 0.45)).isNull();
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
    @DisplayName("윗입술 왼쪽을 위로 넘겨 그리면 분할 칸 두께가 그만큼 두꺼워짐, 윤곽점 두께는 그대로")
    void upperLipOverdrawn() {
        FaceLabels lab = grid(new double[] {0.15, 0.95, -0.45, -0.33}, new double[] {0.15, 0.95, -0.45, -0.33});
        byte[] b = lab.labels();
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                double x = lab.x(i);
                double y = lab.y(j);
                double top = x < 0 ? 1.15 : 1.2; // 왼쪽(화면 x<0)만 0.05 위로
                if (Math.abs(x) < 0.5 && y > top && y < 1.3) {
                    b[j * N + i] = FaceLabels.U_LIP;
                } else if (Math.abs(x) < 0.5 && y > 1.3 && y < 1.42) {
                    b[j * N + i] = FaceLabels.L_LIP;
                }
            }
        }
        double[] up = MakeupPixels.lipThickness(lab, FaceLabels.U_LIP, 0.5, 0.5, 1.0, 1.6);
        assertThat(up[0]).isCloseTo(0.15, org.assertj.core.data.Offset.offset(0.025));
        assertThat(up[1]).isCloseTo(0.10, org.assertj.core.data.Offset.offset(0.025));
        double[] down = MakeupPixels.lipThickness(lab, FaceLabels.L_LIP, 0.5, 0.5, 1.0, 1.6);
        assertThat(down[0]).isCloseTo(down[1], org.assertj.core.data.Offset.offset(0.021));
        // 윤곽점(넘겨 그린 선을 안 따라감): 좌우 같은 두께
        List<Pt> lipTop = List.of(new Pt(-0.5, 1.2), new Pt(0, 1.2), new Pt(0.5, 1.2));
        List<Pt> lipMid = List.of(new Pt(0.5, 1.3), new Pt(-0.5, 1.3), new Pt(0, 1.3));
        double[] gap = MakeupPixels.lineGap(lipTop, lipMid, 0.5, 0.5);
        assertThat(gap[0]).isCloseTo(0.1, org.assertj.core.data.Offset.offset(1e-9));
        assertThat(gap[1]).isCloseTo(0.1, org.assertj.core.data.Offset.offset(1e-9));
    }

    @Test
    @DisplayName("입술 칸이 한쪽에 없으면 null(입술 선은 재지 않음)")
    void noLip() {
        FaceLabels lab = grid(new double[] {0.15, 0.95, -0.45, -0.33}, new double[] {0.15, 0.95, -0.45, -0.33});
        assertThat(MakeupPixels.lipThickness(lab, FaceLabels.U_LIP, 0.5, 0.5, 1.0, 1.6)).isNull();
    }

    @Test
    @DisplayName("뒤집어 나눈 결과의 눈썹 칸을 더하면 한쪽에서만 본 덧그린 부분까지 두께에 들어감")
    void browsUnion() {
        double[] brow = {0.15, 0.95, -0.45, -0.33};
        double[] thicker = {0.15, 0.95, -0.45, -0.27};
        FaceLabels a = grid(brow, brow);
        FaceLabels b = grid(thicker, brow); // 다른 쪽 결과는 왼쪽 아래를 더 눈썹으로 봄
        FaceLabels u = a.withBrowsFrom(b);
        assertThat(MakeupPixels.brow(u, -1, 0.45).thickness())
                .isGreaterThan(MakeupPixels.brow(a, -1, 0.45).thickness() + 0.04);
        assertThat(MakeupPixels.brow(u, 1, 0.45).thickness())
                .isCloseTo(MakeupPixels.brow(a, 1, 0.45).thickness(), org.assertj.core.data.Offset.offset(1e-9));
        assertThat(a.withBrowsFrom(null)).isSameAs(a);
    }

    @Test
    @DisplayName("레포에 든 ONNX 모델을 읽고 512x512로 나눔")
    void onnxModel() throws Exception {
        try (OnnxFaceParser p = new OnnxFaceParser(new ClassPathResource("models/face_parsing_bisenet_r34_fp16.onnx"))) {
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
            // 뒤집어 나눈 결과도 원래 방향으로 돌아와야 함: 대부분의 칸이 같은 분류
            FaceLabels mir = p.parseMirrored(img, FaceFrame.of(Pt.mean(le), Pt.mean(re)));
            assertThat(mir.size()).isEqualTo(512);
            int same = 0;
            for (int k = 0; k < mir.labels().length; k++) {
                same += mir.labels()[k] == lab.labels()[k] ? 1 : 0;
            }
            assertThat(same).isGreaterThan(mir.labels().length * 8 / 10); // 합성 얼굴에서 약 90%
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
