package com.fashionagent.facecheck.image;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.fashionagent.facecheck.FaceCheckException;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

/** 사진 base64 풀기(입력 검사)와 Lab 색 계산. 화장·치아 판단이 모두 이 값을 쓴다. */
class ImagesTest {

    private static final long MAX = 4L * 1024 * 1024;

    private static String jpeg(int w, int h, Color color) throws Exception {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, w, h);
        g.dispose();
        var out = new ByteArrayOutputStream();
        ImageIO.write(img, "jpg", out);
        return Base64.getEncoder().encodeToString(out.toByteArray());
    }

    private static String codeOf(Runnable r) {
        try {
            r.run();
        } catch (FaceCheckException e) {
            return e.code();
        }
        throw new AssertionError("FaceCheckException이 나야 함");
    }

    @Test
    @DisplayName("Given 64×48 JPEG When 풀면 Then 같은 크기의 이미지")
    void decodesJpeg() throws Exception {
        BufferedImage img = Images.decode(jpeg(64, 48, Color.RED), MAX);

        assertThat(img.getWidth()).isEqualTo(64);
        assertThat(img.getHeight()).isEqualTo(48);
    }

    @Test
    @DisplayName("Given data:image/jpeg;base64, 머리말과 앞뒤 공백·줄바꿈 When 풀면 Then 머리말을 떼고 읽는다")
    void stripsDataUrlAndWhitespace() throws Exception {
        String b64 = jpeg(40, 40, Color.BLUE);
        String wrapped = "  data:image/jpeg;base64," + b64.substring(0, 20) + "\n" + b64.substring(20) + "\n ";

        assertThat(Images.decode(wrapped, MAX).getWidth()).isEqualTo(40);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {"   ", "\n"})
    @DisplayName("Given 사진이 없음 When 풀면 Then no_image")
    void noImage(String b64) {
        assertThat(codeOf(() -> Images.decode(b64, MAX))).isEqualTo("no_image");
    }

    @Test
    @DisplayName("Given maxBytes보다 큰 base64 When 풀면 Then 디코딩 전에 bad_image(너무 커요)")
    void tooLarge() throws Exception {
        String b64 = jpeg(64, 64, Color.GREEN);

        assertThatThrownBy(() -> Images.decode(b64, 100)).isInstanceOf(FaceCheckException.class)
                .hasMessageContaining("너무 커요");
    }

    @Test
    @DisplayName("Given 사진이 아닌 바이트 When 풀면 Then bad_image")
    void notAnImage() {
        String text = Base64.getEncoder().encodeToString("hello, not a jpeg".getBytes());

        assertThat(codeOf(() -> Images.decode(text, MAX))).isEqualTo("bad_image");
    }

    @Test
    @DisplayName("Given base64가 중간에 끊김 When 풀면 Then bad_image")
    void brokenBase64() {
        assertThat(codeOf(() -> Images.decode("QUJD\nR", MAX))).isEqualTo("bad_image");
    }

    @Test
    @DisplayName("Given 32px보다 작은 사진(31×100) When 풀면 Then bad_image, 32×32는 읽는다")
    void tooSmall() throws Exception {
        String small = jpeg(31, 100, Color.WHITE);
        String ok = jpeg(32, 32, Color.WHITE);

        assertThat(codeOf(() -> Images.decode(small, MAX))).isEqualTo("bad_image");
        assertThat(Images.decode(ok, MAX).getWidth()).isEqualTo(32);
    }

    @Test
    @DisplayName("Given 몇 KB인데 1만×1만 픽셀인 PNG When 풀면 Then 다 풀지 않고 건너뛰며 읽어 2500px로 받는다")
    void hugeDimensionsAreSubsampled() throws Exception {
        // given: 흑백 1비트라 압축하면 수십 KB
        BufferedImage huge = new BufferedImage(10_000, 10_000, BufferedImage.TYPE_BYTE_BINARY);
        var out = new ByteArrayOutputStream();
        ImageIO.write(huge, "png", out);
        String b64 = Base64.getEncoder().encodeToString(out.toByteArray());
        assertThat(b64.length()).isLessThan(200_000);

        // when
        BufferedImage img = Images.decode(b64, MAX);

        // then: 10000 / 2048 = 4칸씩 건너뜀
        assertThat(img.getWidth()).isEqualTo(2500);
        assertThat(img.getHeight()).isEqualTo(2500);
    }

    @Test
    @DisplayName("Given 폰 사진 크기(4032×3024) When 풀면 Then 줄이지 않고 그대로 읽는다(판정이 바뀌지 않음)")
    void phoneSizeIsNotSubsampled() throws Exception {
        BufferedImage img = Images.decode(jpeg(4032, 3024, Color.GRAY), MAX);

        assertThat(img.getWidth()).isEqualTo(4032);
        assertThat(img.getHeight()).isEqualTo(3024);
    }

    @Test
    @DisplayName("Lab: 흰색은 L≈100·a,b≈0, 검정은 L=0, 빨강은 a가 크게 +")
    void labKnownColors() {
        double[] white = Images.lab(0xFFFFFF);
        double[] black = Images.lab(0x000000);
        double[] red = Images.lab(0xFF0000);

        assertThat(white[0]).isCloseTo(100.0, within(0.01));
        assertThat(white[1]).isCloseTo(0.0, within(0.01));
        assertThat(white[2]).isCloseTo(0.0, within(0.01));
        assertThat(black[0]).isCloseTo(0.0, within(0.01));
        // sRGB 빨강의 Lab 기준값 ≈ (53.24, 80.09, 67.20)
        assertThat(red[0]).isCloseTo(53.24, within(0.05));
        assertThat(red[1]).isCloseTo(80.09, within(0.05));
        assertThat(red[2]).isCloseTo(67.20, within(0.05));
    }

    @Test
    @DisplayName("Lab: 회색은 밝을수록 L이 커지고 a·b는 0 근처")
    void labGrayMonotonic() {
        double prev = -1;
        for (int v = 0; v <= 255; v += 15) {
            double[] lab = Images.lab((v << 16) | (v << 8) | v);
            assertThat(lab[0]).isGreaterThan(prev);
            assertThat(Math.abs(lab[1])).isLessThan(0.01);
            assertThat(Math.abs(lab[2])).isLessThan(0.01);
            prev = lab[0];
        }
    }

    @Test
    @DisplayName("LabMean: 비었으면 NaN·0개, 두 색 평균과 ΔE76 거리")
    void labMean() {
        var empty = new Images.LabMean();
        assertThat(empty.count()).isZero();
        assertThat(empty.l()).isNaN();

        var m = new Images.LabMean();
        m.add(new double[] {10, 20, 30});
        m.add(new double[] {30, 40, 50});
        assertThat(m.count()).isEqualTo(2);
        assertThat(m.l()).isEqualTo(20);
        assertThat(m.a()).isEqualTo(30);
        assertThat(m.b()).isEqualTo(40);

        var o = new Images.LabMean();
        o.add(new double[] {23, 34, 40});
        assertThat(m.distance(o)).isCloseTo(5.0, within(1e-9));
        assertThat(m.distance(m)).isZero();
    }
}
