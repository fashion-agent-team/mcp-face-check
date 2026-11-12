package com.fashionagent.facecheck.makeup;

import static com.fashionagent.facecheck.SyntheticFace.scaleX;
import static com.fashionagent.facecheck.SyntheticFace.shift;
import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.SyntheticFace;
import com.fashionagent.facecheck.makeup.MakeupResult.Check;
import java.awt.Color;
import java.util.List;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import tools.jackson.databind.json.JsonMapper;

class MakeupAnalyzerTest {

    static final FaceCheckProperties.Symmetry T = new FaceCheckProperties.Symmetry(
            0.07, 0.09, 0.15, 0.046, 0.25, 0.009, 0.05, 0.12, 0.014, 0.25, 6, 0.081, 10, 15, 28, 12, 25, 0.15, 3);
    static final JsonMapper JSON = JsonMapper.builder().build();

    final MakeupAnalyzer analyzer = new MakeupAnalyzer(T);

    MakeupResult run(SyntheticFace f) {
        FaceLandmarks lm = JSON.convertValue(f.landmarks(), FaceLandmarks.class);
        assertThat(lm.complete()).isTrue();
        return analyzer.analyze(lm, f.image());
    }

    @ParameterizedTest(name = "기울기 {0}도")
    @ValueSource(doubles = {0, 12, -18})
    @DisplayName("좌우가 같은 얼굴은 ok, 모든 항목을 계산함 (고개가 기울어도 돌려 맞춤)")
    void symmetric(double roll) {
        var f = new SyntheticFace();
        f.rollDegrees = roll;
        MakeupResult r = run(f);
        assertThat(r.framing()).isNull();
        assertThat(r.lighting()).isEqualTo("ok");
        assertThat(r.symmetry().notes()).isEmpty();
        assertThat(r.symmetry().ok()).isTrue();
        assertThat(r.symmetry().skipped()).isEmpty();
        assertThat(r.symmetry().checks()).extracting(Check::name).containsExactly(
                "browHeight", "browTail", "browLength", "browThickness", "lipCorner", "lipHalfWidth",
                "upperLipLine", "lowerLipLine", "cheekColor", "lipColor");
    }

    @Test
    @DisplayName("왼쪽 눈썹을 높게 그리면 '왼쪽 눈썹이 오른쪽보다 조금 높게'")
    void browHigher() {
        var f = new SyntheticFace();
        f.edits.put("leftEyebrowTop", shift(0, -0.12));
        f.edits.put("leftEyebrowBottom", shift(0, -0.12));
        MakeupResult r = run(f);
        assertThat(r.symmetry().ok()).isFalse();
        assertThat(r.symmetry().notes()).containsExactly("왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.");
    }

    @Test
    @DisplayName("윤곽점은 그대로인데 사진에서만 눈썹을 높게 그려도 픽셀로 잡음")
    void browDrawnOnlyInImage() {
        var f = new SyntheticFace();
        f.drawEdits.put("leftEyebrowTop", shift(0, -0.12));
        f.drawEdits.put("leftEyebrowBottom", shift(0, -0.12));
        MakeupResult r = run(f);
        assertThat(r.symmetry().notes()).containsExactly("왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.");
        assertThat(check(r, "browHeight").basis()).isEqualTo("pixels");
    }

    @Test
    @DisplayName("윤곽점은 그대로인데 사진에서만 윗입술 선을 넘게 그려도 픽셀로 잡음")
    void lipDrawnOnlyInImage() {
        var f = new SyntheticFace();
        f.drawEdits.put("upperLipTop", pts -> pts.stream()
                .map(p -> p[0] < 0 ? new double[] {p[0], 1.0 + (p[1] - 1.0) * 1.8} : p).toList());
        MakeupResult r = run(f);
        assertThat(r.symmetry().notes()).contains("윗입술 선이 왼쪽이 더 도톰하게 그려졌어요.");
        assertThat(check(r, "upperLipLine").basis()).isEqualTo("pixels");
    }

    @Test
    @DisplayName("누드 립처럼 입술이 피부색과 비슷하면 입술 모양은 윤곽점으로 잼")
    void nudeLipsFallBack() {
        var f = new SyntheticFace();
        f.lipLeft = new Color(222, 178, 152);
        f.lipRight = new Color(222, 178, 152);
        MakeupResult r = run(f);
        assertThat(r.symmetry().ok()).isTrue();
        assertThat(check(r, "lipCorner").basis()).isEqualTo("landmarks");
        assertThat(check(r, "browHeight").basis()).isEqualTo("pixels");
    }

    static Check check(MakeupResult r, String name) {
        return r.symmetry().checks().stream().filter(c -> c.name().equals(name)).findFirst().orElseThrow();
    }

    @Test
    @DisplayName("ML Kit 이름의 left/right가 반대로 와도 사진 x 위치로 왼쪽을 정함")
    void namesSwapped() {
        var f = new SyntheticFace();
        f.swapNames = true;
        f.edits.put("leftEyebrowTop", shift(0, -0.12));
        f.edits.put("leftEyebrowBottom", shift(0, -0.12));
        assertThat(run(f).symmetry().notes()).containsExactly("왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.");
    }

    @Test
    @DisplayName("오른쪽 눈썹 꼬리를 짧게 그리면 '오른쪽 눈썹이 왼쪽보다 조금 짧게'")
    void browShorter() {
        var f = new SyntheticFace();
        f.edits.put("rightEyebrowTop", scaleX(0.15, 0.7));
        f.edits.put("rightEyebrowBottom", scaleX(0.15, 0.7));
        assertThat(run(f).symmetry().notes()).contains("오른쪽 눈썹이 왼쪽보다 조금 짧게 그려졌어요.");
    }

    @Test
    @DisplayName("오른쪽 눈썹 꼬리만 올리면 꼬리 문장")
    void browTail() {
        var f = new SyntheticFace();
        UnaryOperator<List<double[]>> lift = pts -> pts.stream()
                .map(p -> new double[] {p[0], p[1] - 0.25 * Math.max(0, p[0] - 0.5) / 0.5}).toList();
        f.edits.put("rightEyebrowTop", lift);
        f.edits.put("rightEyebrowBottom", lift);
        assertThat(run(f).symmetry().notes()).contains("오른쪽 눈썹 꼬리가 왼쪽보다 올라가 있어요.");
    }

    @Test
    @DisplayName("왼쪽 눈썹을 두껍게 그리면 두께 문장")
    void browThicker() {
        var f = new SyntheticFace();
        f.edits.put("leftEyebrowBottom", shift(0, 0.06));
        assertThat(run(f).symmetry().notes()).contains("왼쪽 눈썹이 오른쪽보다 두껍게 그려졌어요.");
    }

    @Test
    @DisplayName("왼쪽 입꼬리가 올라가면 '입술 왼쪽 끝이 오른쪽 끝보다 올라가'")
    void lipCorner() {
        var f = new SyntheticFace();
        UnaryOperator<List<double[]>> lift = pts -> pts.stream()
                .map(p -> new double[] {p[0], p[1] - 0.12 * Math.max(0, -p[0]) / 0.45}).toList();
        for (String c : List.of("upperLipTop", "upperLipBottom", "lowerLipTop", "lowerLipBottom")) {
            f.edits.put(c, lift);
        }
        assertThat(run(f).symmetry().notes()).contains("입술 왼쪽 끝이 오른쪽 끝보다 올라가 있어요.");
    }

    @Test
    @DisplayName("윗입술 선을 왼쪽만 크게 그리면 윗입술 문장")
    void upperLipOverdrawn() {
        var f = new SyntheticFace();
        f.edits.put("upperLipTop", pts -> pts.stream()
                .map(p -> p[0] < 0 ? new double[] {p[0], 1.0 + (p[1] - 1.0) * 1.6} : p).toList());
        assertThat(run(f).symmetry().notes()).contains("윗입술 선이 왼쪽이 더 도톰하게 그려졌어요.");
    }

    @Test
    @DisplayName("입술 선을 오른쪽으로 길게 그리면 너비 문장 (윤곽점 입꼬리는 그대로, 그린 선만 바뀜)")
    void lipWider() {
        var f = new SyntheticFace();
        for (String c : List.of("upperLipTop", "upperLipBottom", "lowerLipTop", "lowerLipBottom")) {
            f.drawEdits.put(c, pts -> pts.stream().map(p -> p[0] > 0 ? new double[] {p[0] * 1.4, p[1]} : p).toList());
        }
        assertThat(run(f).symmetry().notes()).contains("입술 선이 오른쪽으로 더 길게 그려졌어요.");
    }

    @Test
    @DisplayName("왼쪽 볼에만 블러셔 → '왼쪽 볼이 오른쪽보다 더 붉어요'")
    void blushOneSide() {
        var f = new SyntheticFace();
        f.blushLeft = new Color(232, 150, 140);
        MakeupResult r = run(f);
        assertThat(r.symmetry().notes())
                .containsExactly("왼쪽 볼이 오른쪽보다 더 붉어요. 블러셔가 한쪽에 더 발린 것 같아요.");
        Check cheek = check(r, "cheekColor");
        assertThat(cheek.side()).isEqualTo("left");
        assertThat(cheek.value()).isGreaterThan(6);
    }

    @Test
    @DisplayName("양쪽 볼에 블러셔가 같으면 ok")
    void blushBoth() {
        var f = new SyntheticFace();
        f.blushLeft = new Color(232, 150, 140);
        f.blushRight = new Color(232, 150, 140);
        assertThat(run(f).symmetry().ok()).isTrue();
    }

    @Test
    @DisplayName("오른쪽 입술 립스틱이 덜 발리면 '입술 색이 오른쪽이 왼쪽보다 연해요'")
    void lipColor() {
        var f = new SyntheticFace();
        f.lipRight = new Color(214, 140, 140);
        assertThat(run(f).symmetry().notes()).containsExactly("입술 색이 오른쪽이 왼쪽보다 연해요.");
    }

    @Test
    @DisplayName("한쪽이 조금 어두워도(빛이 옆에서) 립스틱이 같으면 입술 색은 고름: 옆 피부와 견주어 비교")
    void lipColorSideLight() {
        var f = new SyntheticFace();
        f.shadeRight = 0.12;
        MakeupResult r = run(f);
        assertThat(r.symmetry().skipped()).isEmpty();
        var lip = r.symmetry().checks().stream().filter(c -> c.name().equals("lipColor")).findFirst().orElseThrow();
        assertThat(Math.abs(lip.value())).isLessThan(6); // 입술끼리만 비교하면 7.4(옛 방식)
        assertThat(lip.ok()).isTrue();
    }

    @Test
    @DisplayName("한쪽이 그늘지면 색 비교는 건너뛰고(skipped) 모양만 봄")
    void unevenLight() {
        var f = new SyntheticFace();
        f.shadeRight = 0.35;
        f.lipRight = new Color(214, 140, 140);
        MakeupResult r = run(f);
        assertThat(r.lighting()).isEqualTo("ok");
        assertThat(r.symmetry().skipped()).containsExactly("cheekColor", "lipColor");
        assertThat(r.symmetry().ok()).isTrue();
    }

    @Test
    @DisplayName("문장은 차이가 큰 순서로 최대 3개")
    void maxNotes() {
        var f = new SyntheticFace();
        f.edits.put("leftEyebrowTop", shift(0, -0.2));
        f.edits.put("leftEyebrowBottom", shift(0, -0.2));
        f.edits.put("rightEyebrowTop", scaleX(0.15, 0.6));
        f.edits.put("rightEyebrowBottom", scaleX(0.15, 0.6));
        f.blushLeft = new Color(235, 140, 130);
        f.lipRight = new Color(214, 140, 140);
        MakeupResult r = run(f);
        assertThat(r.symmetry().notes()).hasSize(3);
        // 기준값 대비 차이가 큰 순서로 고름
        assertThat(r.symmetry().notes()).first().asString().startsWith("왼쪽 볼이");
        assertThat(r.symmetry().notes()).contains("왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.");
        assertThat(r.symmetry().checks().stream().filter(c -> !c.ok()).count()).isGreaterThan(3);
    }

    @Test
    @DisplayName("얼굴 위쪽이 잘리면 framing, symmetry는 없음")
    void cutTop() {
        var f = new SyntheticFace();
        f.centerY = 70;
        MakeupResult r = run(f);
        assertThat(r.framing()).isEqualTo("얼굴 위쪽이 화면 밖으로 나갔어요. 폰을 조금 위로 올려 주세요.");
        assertThat(r.symmetry()).isNull();
    }

    @Test
    @DisplayName("셀카 화면 왼쪽이 잘리면 폰을 왼쪽으로")
    void cutLeft() {
        var f = new SyntheticFace();
        f.centerX = 100;
        assertThat(run(f).framing()).isEqualTo("얼굴 왼쪽이 화면 밖으로 나갔어요. 폰을 왼쪽으로 조금 옮겨 주세요.");
    }

    @Test
    @DisplayName("얼굴이 작으면 가까이")
    void tooSmall() {
        var f = new SyntheticFace();
        f.iod = 60;
        assertThat(run(f).framing()).contains("가까이");
    }

    @Test
    @DisplayName("고개를 옆으로 돌리면(yaw 20도) 정면 안내")
    void turned() {
        var f = new SyntheticFace();
        f.yaw = 20.0;
        assertThat(run(f).framing()).contains("옆으로 돌아가");
    }

    @Test
    @DisplayName("yaw가 없어도 얼굴 외곽이 한쪽으로 쏠리면 정면 안내")
    void turnedByContour() {
        var f = new SyntheticFace();
        f.edits.put("face", pts -> pts.stream().map(p -> p[0] > 0 ? new double[] {p[0] * 0.6, p[1]} : p).toList());
        assertThat(run(f).framing()).contains("옆으로 돌아가");
    }

    @Test
    @DisplayName("고개가 많이 기울면(30도) 기울기 안내")
    void tilted() {
        var f = new SyntheticFace();
        f.rollDegrees = 30;
        assertThat(run(f).framing()).contains("기울었어요");
    }

    @Test
    @DisplayName("어두우면 lighting dark")
    void dark() {
        var f = new SyntheticFace();
        f.brightness = 0.2;
        MakeupResult r = run(f);
        assertThat(r.lighting()).isEqualTo("dark");
        assertThat(r.framing()).isNull();
        assertThat(r.symmetry()).isNull();
    }
}
