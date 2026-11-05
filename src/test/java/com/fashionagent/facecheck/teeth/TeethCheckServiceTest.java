package com.fashionagent.facecheck.teeth;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.SyntheticFace;
import com.fashionagent.facecheck.teeth.TeethClassifier.TeethFinding;
import java.awt.image.BufferedImage;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.support.StaticListableBeanFactory;
import org.springframework.util.unit.DataSize;

class TeethCheckServiceTest {

    static final FaceCheckProperties PROPS = new FaceCheckProperties(DataSize.ofMegabytes(8),
            new FaceCheckProperties.OpenAi("", "m", "u", Duration.ofSeconds(1)), null);

    /** 받은 사진을 기억하는 가짜 분류기 (정지원 님 모델 자리) */
    static class Recording implements TeethClassifier {
        final List<BufferedImage> seen = new ArrayList<>();
        final List<Boolean> cropped = new ArrayList<>();

        @Override
        public TeethFinding classify(BufferedImage mouth, boolean wasCropped) {
            seen.add(mouth);
            cropped.add(wasCropped);
            return new TeethFinding(true, 0.82, "앞니 사이에 뭔가 낀 것 같아요.");
        }
    }

    static TeethCheckService service(TeethClassifier c) {
        var beans = new StaticListableBeanFactory();
        if (c != null) {
            beans.addBean("teeth", c);
        }
        return new TeethCheckService(PROPS, beans.getBeanProvider(TeethClassifier.class));
    }

    @Test
    @DisplayName("분류기가 아직 없으면 not_ready")
    void notReady() {
        assertThatThrownBy(() -> service(null).check(new SyntheticFace().jpegBase64(), null))
                .isInstanceOf(FaceCheckException.class).extracting("code").isEqualTo("not_ready");
    }

    @Test
    @DisplayName("mouthBox에 여유 15%를 두고 잘라서 분류기에 넘김")
    void crops() {
        var c = new Recording();
        var f = new SyntheticFace();
        TeethResult r = service(c).check(f.jpegBase64(), new MouthBox(0.3, 0.5, 0.4, 0.2));
        assertThat(r.framing()).isNull();
        assertThat(r.lighting()).isEqualTo("ok");
        assertThat(r.source()).isEqualTo("model");
        assertThat(r.teeth().foreignObject()).isTrue();
        assertThat(c.cropped).containsExactly(true);
        // 600x800 사진: 너비 0.4*1.3*600 = 312, 높이 0.2*1.3*800 = 208
        assertThat(c.seen.getFirst().getWidth()).isBetween(311, 313);
        assertThat(c.seen.getFirst().getHeight()).isBetween(207, 209);
    }

    @Test
    @DisplayName("mouthBox가 없으면 사진 전체")
    void noBox() {
        var c = new Recording();
        service(c).check(new SyntheticFace().jpegBase64(), null);
        assertThat(c.cropped).containsExactly(false);
        assertThat(c.seen.getFirst().getWidth()).isEqualTo(600);
    }

    @Test
    @DisplayName("입이 잘리거나 작거나 덜 벌렸으면 framing, 분류기는 부르지 않음")
    void framing() {
        var c = new Recording();
        var s = service(c);
        String img = new SyntheticFace().jpegBase64();
        assertThat(s.check(img, new MouthBox(0.7, 0.5, 0.4, 0.2)).framing()).contains("다 보이지 않아요");
        assertThat(s.check(img, new MouthBox(0.45, 0.5, 0.1, 0.05)).framing()).contains("가까이");
        assertThat(s.check(img, new MouthBox(0.3, 0.5, 0.4, 0.03)).framing()).contains("벌려");
        assertThat(c.seen).isEmpty();
    }

    @Test
    @DisplayName("어두우면 dark")
    void dark() {
        var f = new SyntheticFace();
        f.brightness = 0.15;
        TeethResult r = service(new Recording()).check(f.jpegBase64(), new MouthBox(0.3, 0.5, 0.4, 0.2));
        assertThat(r.lighting()).isEqualTo("dark");
        assertThat(r.teeth()).isNull();
    }

    @Test
    @DisplayName("mouthBox 값이 비율이 아니면 bad_request")
    void badBox() {
        assertThatThrownBy(() -> service(new Recording()).check(new SyntheticFace().jpegBase64(), new MouthBox(120, 300, 200, 80)))
                .isInstanceOf(FaceCheckException.class).extracting("code").isEqualTo("bad_request");
    }
}
