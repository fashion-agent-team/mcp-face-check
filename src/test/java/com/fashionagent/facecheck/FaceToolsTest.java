package com.fashionagent.facecheck;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;
import static org.mockito.BDDMockito.then;
import static org.mockito.Mockito.mock;

import com.fashionagent.facecheck.makeup.MakeupCheckService;
import com.fashionagent.facecheck.makeup.MakeupResult;
import com.fashionagent.facecheck.teeth.TeethCheckService;
import com.fashionagent.facecheck.teeth.TeethResult;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * MCP 도구 계층 테스트. REST 서버의 컨트롤러 테스트(@WebMvcTest)와 같은 자리입니다.
 * 화장·치아 판단 서비스는 Mock으로 바꾸고(정상 동작한다고 가정), 도구가 맞는 서비스에 값을 넘기는지와
 * 돌려주는 결과·오류 코드만 봅니다. 진짜 모델로 끝까지 보는 건 McpServerTest입니다.
 *
 * <p>MCP에는 HTTP 상태 코드가 없어서, 실패는 {"error": {"code", "message"}}로 돌려줍니다
 * (bad_request ≈ 400, upstream_failed ≈ 502).
 */
class FaceToolsTest {

    private MakeupCheckService makeup;
    private TeethCheckService teeth;
    private FaceTools tools;

    @BeforeEach
    void setUp() {
        makeup = mock(MakeupCheckService.class);
        teeth = mock(TeethCheckService.class);
        tools = new FaceTools(makeup, teeth);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> errorOf(Object result) {
        assertThat(result).isInstanceOf(Map.class);
        return (Map<String, Object>) ((Map<String, Object>) result).get("error");
    }

    @Test
    @DisplayName("teeth_check Given 입 사진과 입 박스 When 부르면 Then 치아 서비스 결과를 그대로 돌려주고 화장 서비스는 부르지 않는다")
    void teethOk() {
        // given
        Map<String, Double> box = Map.of("x", 0.3, "y", 0.5, "w", 0.4, "h", 0.2);
        TeethResult found = new TeethResult(null, "ok", new TeethResult.Teeth(false, 0.92, "이 사이에 낀 건 없어요."));
        given(teeth.check("jpegBase64", box)).willReturn(found);

        // when
        Object result = tools.teethCheck("jpegBase64", box);

        // then
        assertThat(result).isEqualTo(found);
        then(makeup).shouldHaveNoInteractions();
    }

    @Test
    @DisplayName("face_makeup_check Given 셀카 When 부르면 Then 화장 서비스 결과를 그대로 돌려주고 치아 서비스는 부르지 않는다")
    void makeupOk() {
        // given
        MakeupResult found = new MakeupResult("face_too_small", null, null);
        given(makeup.check("jpegBase64", null, null)).willReturn(found);

        // when
        Object result = tools.faceMakeupCheck("jpegBase64", null, null);

        // then
        assertThat(result).isEqualTo(found);
        then(teeth).shouldHaveNoInteractions();
    }

    @Test
    @DisplayName("Given 사진이 base64가 아님 When teeth_check를 부르면 Then 예외를 던지지 않고 bad_request를 돌려준다")
    void teethBadRequest() {
        // given
        given(teeth.check("not-an-image", null)).willThrow(FaceCheckException.badRequest("사진을 읽지 못했어요."));

        // when
        Object result = tools.teethCheck("not-an-image", null);

        // then
        assertThat(errorOf(result)).containsEntry("code", "bad_request")
                .containsEntry("message", "사진을 읽지 못했어요.");
    }

    @Test
    @DisplayName("Given 모델에서 예상 못 한 오류 When face_makeup_check를 부르면 Then upstream_failed와 쉬운 문장을 돌려준다")
    void unexpectedFailure() {
        // given
        given(makeup.check("jpegBase64", null, null)).willThrow(new IllegalStateException("onnx session closed"));

        // when
        Object result = tools.faceMakeupCheck("jpegBase64", null, null);

        // then: 안의 오류 문장(onnx session closed)은 밖으로 내보내지 않는다
        assertThat(errorOf(result)).containsEntry("code", "upstream_failed")
                .containsEntry("message", "지금은 확인하지 못했어요.");
    }
}
