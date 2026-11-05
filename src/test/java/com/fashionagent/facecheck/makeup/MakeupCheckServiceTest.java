package com.fashionagent.facecheck.makeup;

import static com.fashionagent.facecheck.SyntheticFace.shift;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.SyntheticFace;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.util.unit.DataSize;

/** 실시간으로 비추다 멈춘 1초 동안 보낸 여러 장을 중간값으로 판단 (2026-11-05 결정) */
class MakeupCheckServiceTest {

    final MakeupCheckService service = new MakeupCheckService(
            new FaceCheckProperties(DataSize.ofMegabytes(8), MakeupAnalyzerTest.T));

    static MakeupFrame frame(SyntheticFace f) {
        return new MakeupFrame(f.jpegBase64(), MakeupAnalyzerTest.JSON.convertValue(f.landmarks(), FaceLandmarks.class));
    }

    static SyntheticFace raised(double dy) {
        var f = new SyntheticFace();
        f.edits.put("leftEyebrowTop", shift(0, -dy));
        f.edits.put("leftEyebrowBottom", shift(0, -dy));
        return f;
    }

    MakeupResult run(SyntheticFace... faces) {
        MakeupFrame first = frame(faces[0]);
        List<MakeupFrame> more = java.util.Arrays.stream(faces).skip(1).map(MakeupCheckServiceTest::frame).toList();
        return service.check(first.imageBase64(), first.landmarks(), more);
    }

    @Test
    @DisplayName("한 장만 눈썹이 튀어도(순간 표정) 중간값은 고르다")
    void oneOutlier() {
        MakeupResult r = run(new SyntheticFace(), raised(0.15), new SyntheticFace());
        assertThat(r.symmetry().ok()).isTrue();
        assertThat(r.symmetry().frames()).isEqualTo(3);
    }

    @Test
    @DisplayName("대부분 장면에서 차이가 있으면 잡음")
    void majority() {
        MakeupResult r = run(raised(0.12), raised(0.13), new SyntheticFace());
        assertThat(r.symmetry().notes()).containsExactly("왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.");
    }

    @Test
    @DisplayName("절반 넘게 구도가 안 맞으면 그 안내, 한 장만 안 맞으면 나머지로 판단")
    void framingMajority() {
        var cut = new SyntheticFace();
        cut.centerY = 70;
        assertThat(run(cut, cut, new SyntheticFace()).framing()).contains("위로 올려");
        MakeupResult ok = run(cut, new SyntheticFace(), new SyntheticFace());
        assertThat(ok.framing()).isNull();
        assertThat(ok.symmetry().frames()).isEqualTo(2);
    }

    @Test
    @DisplayName("한 번에 5장까지")
    void tooMany() {
        var f = new SyntheticFace();
        assertThatThrownBy(() -> run(f, f, f, f, f, f)).isInstanceOf(FaceCheckException.class)
                .extracting("code").isEqualTo("bad_request");
    }
}
