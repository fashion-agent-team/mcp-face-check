package com.fashionagent.facecheck.teeth;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.util.unit.DataSize;

public class TeethCheckServiceTest {

    public static final FaceCheckProperties.Teeth T = new FaceCheckProperties.Teeth(
            "classpath:models/teeth_mobilenetv3.onnx", 320, 0.7, 0.3, 0.10, 0.15, 0.1, 25);

    static final Map<String, Double> OPEN = Map.of("x", 0.3, "y", 0.4, "w", 0.4, "h", 0.12);

    /** 정해 둔 확률을 돌려주는 가짜 모델. 받은 입력 크기도 기록. */
    static final class Fixed implements TeethClassifier {
        final double p;
        final boolean ready;
        int lastLength;
        int calls;

        Fixed(double p, boolean ready) {
            this.p = p;
            this.ready = ready;
        }

        @Override
        public boolean ready() {
            return ready;
        }

        @Override
        public int inputSize() {
            return 320;
        }

        @Override
        public double foodProbability(float[] chw) {
            lastLength = chw.length;
            calls++;
            return p;
        }
    }

    static TeethCheckService service(TeethClassifier c) {
        return new TeethCheckService(new FaceCheckProperties(DataSize.ofMegabytes(8), null, T), c);
    }

    static String jpeg(Color c) {
        BufferedImage img = new BufferedImage(480, 640, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, 480, 640);
        g.setColor(Color.WHITE);
        g.fillRect(160, 270, 160, 60);
        g.dispose();
        try {
            var out = new ByteArrayOutputStream();
            ImageIO.write(img, "jpg", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    static final String PHOTO = jpeg(new Color(200, 120, 120));

    @Test
    @DisplayName("확률 0.7 이상 → 낀 것 있음, 0.3 이하 → 없음, 그 사이 → null(애매함)")
    void thresholds() {
        var yes = service(new Fixed(0.82, true)).check(PHOTO, OPEN);
        assertThat(yes.framing()).isNull();
        assertThat(yes.teeth().foreignObject()).isTrue();
        assertThat(yes.teeth().confidence()).isEqualTo(0.82);
        assertThat(yes.teeth().note()).contains("낀 것 같아요");

        var no = service(new Fixed(0.1, true)).check(PHOTO, OPEN);
        assertThat(no.teeth().foreignObject()).isFalse();
        assertThat(no.teeth().note()).contains("없어 보여요");

        var unsure = service(new Fixed(0.5, true)).check(PHOTO, OPEN);
        assertThat(unsure.teeth().foreignObject()).isNull();
        assertThat(unsure.teeth().note()).contains("확실하지 않아요");
    }

    @Test
    @DisplayName("모델에는 3×320×320 텐서가 6번 들어감 (mouthBox 없으면 사진 가운데)")
    void tensorShape() {
        var c = new Fixed(0.1, true);
        service(c).check(PHOTO, null);
        assertThat(c.lastLength).isEqualTo(3 * 320 * 320);
        // 여백 3가지 × 원래·좌우 뒤집기 = 6번 보고 평균
        assertThat(c.calls).isEqualTo(6);
    }

    @Test
    @DisplayName("입을 덜 벌림·너무 작음·화면 밖 → framing 안내만, 모델은 부르지 않음")
    void framing() {
        var c = new Fixed(0.9, true);
        var closed = service(c).check(PHOTO, Map.of("x", 0.3, "y", 0.4, "w", 0.4, "h", 0.02));
        assertThat(closed.framing()).contains("입을 조금 더 벌리고");
        assertThat(closed.teeth()).isNull();
        assertThat(c.lastLength).isZero();

        assertThat(service(c).check(PHOTO, Map.of("x", 0.45, "y", 0.4, "w", 0.08, "h", 0.03)).framing())
                .contains("가까이");
        assertThat(service(c).check(PHOTO, Map.of("x", -0.2, "y", 0.4, "w", 0.4, "h", 0.12)).framing())
                .contains("왼쪽으로 잘렸어요");
        assertThat(service(c).check(PHOTO, Map.of("x", 0.3, "y", 0.9, "w", 0.4, "h", 0.2)).framing())
                .contains("아래쪽으로 잘렸어요");
    }

    @Test
    @DisplayName("어두우면 lighting=dark, 판정하지 않음")
    void dark() {
        var c = new Fixed(0.9, true);
        BufferedImage img = new BufferedImage(480, 640, BufferedImage.TYPE_INT_RGB);
        String black;
        try {
            var out = new ByteArrayOutputStream();
            ImageIO.write(img, "jpg", out);
            black = Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        var r = service(c).check(black, OPEN);
        assertThat(r.lighting()).isEqualTo("dark");
        assertThat(r.teeth().foreignObject()).isNull();
        assertThat(c.lastLength).isZero();
    }

    @Test
    @DisplayName("모델 파일이 없으면 not_ready, mouthBox가 이상하면 bad_request")
    void errors() {
        assertThatThrownBy(() -> service(new Fixed(0.5, false)).check(PHOTO, OPEN))
                .isInstanceOf(FaceCheckException.class).extracting("code").isEqualTo("not_ready");
        assertThatThrownBy(() -> service(new Fixed(0.5, true)).check(PHOTO, Map.of("x", 120.0, "y", 300.0, "w", 200.0, "h", 50.0)))
                .isInstanceOf(FaceCheckException.class).extracting("code").isEqualTo("bad_request");
        assertThatThrownBy(() -> service(new Fixed(0.5, true)).check(PHOTO, Map.of("x", 0.3)))
                .isInstanceOf(FaceCheckException.class).extracting("code").isEqualTo("bad_request");
    }
}
