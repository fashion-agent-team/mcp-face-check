package com.fashionagent.facecheck;

import com.fashionagent.facecheck.makeup.FaceLandmarks;
import com.fashionagent.facecheck.makeup.MakeupCheckService;
import com.fashionagent.facecheck.makeup.MakeupFrame;
import com.fashionagent.facecheck.teeth.TeethCheckService;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * MCP 도구 2개 (앱 레포 docs/contracts/mcp-tools.md). agent가 사용자에게 "얼굴을 볼까요, 이를 확인할까요?"라고
 * 묻고 답에 맞는 도구 하나를 직접 부릅니다(GPT가 고르지 않음). 실패해도 예외를 던지지 않고
 * {"error": {"code", "message"}}를 돌려줍니다. 사진은 판단에만 쓰고 남기지 않습니다.
 */
@Component
public class FaceTools {

    private static final Logger log = LoggerFactory.getLogger(FaceTools.class);

    private final MakeupCheckService makeup;
    private final TeethCheckService teeth;

    public FaceTools(MakeupCheckService makeup, TeethCheckService teeth) {
        this.makeup = makeup;
        this.teeth = teeth;
    }

    @Tool(name = "face_makeup_check", description = """
            얼굴 전체가 나온 셀카 한 장과 앱의 ML Kit 얼굴 윤곽점으로 화장 대칭(눈썹 높이·길이·두께, 입술 선, 볼·입술 색 좌우)을 \
            확인합니다. 앱이 얼굴을 실시간으로 비추다 잘 잡혀 멈춘 1초 동안 3~5장을 보내면(첫 장 + frames) 항목별 중간값으로 판단합니다. \
            사용자가 "얼굴 확인해 줘"에서 얼굴(화장)을 고르면 부르세요. 왼쪽·오른쪽은 셀카 화면 기준(사용자 왼쪽)입니다. \
            framing이 있으면 그 안내를 읽고 다시 찍고, lighting이 dark면 밝은 곳으로 안내합니다. 아니면 symmetry.notes를 읽어 줍니다.""")
    public Object faceMakeupCheck(
            @ToolParam(description = "셀카 방향으로 좌우를 맞춘 JPEG 사진 base64. agent가 넣습니다.") String imageBase64,
            @ToolParam(required = false, description = "앱의 ML Kit 얼굴 윤곽점. 없으면 no_landmarks(다시 찍기).") FaceLandmarks landmarks,
            @ToolParam(required = false, description = "같은 1초 동안 더 찍은 장면(최대 4개). agent가 넣습니다.") List<MakeupFrame> frames) {
        return run("face_makeup_check", () -> makeup.check(imageBase64, landmarks, frames));
    }

    @Tool(name = "teeth_check", description = """
            입 가까이에서 찍은 사진 한 장으로 이 사이에 낀 것을 확인합니다(우리가 학습한 모델을 이 서버에서 돌림, 사진은 외부로 \
            보내지 않음). 사용자가 "얼굴 확인해 줘"에서 이를 고르면 부르세요. framing이 있으면 그 안내를 읽고 다시 찍고, \
            아니면 teeth.note를 읽어 줍니다. teeth.foreignObject가 null(애매함)이면 note를 읽은 뒤 "가족에게 사진 보내서 \
            물어볼까요?"라고 묻고, 사용자가 원할 때만 helper_ask로 넘깁니다.""")
    public Object teethCheck(
            @ToolParam(description = "셀카 방향으로 좌우를 맞춘 JPEG 사진 base64. agent가 넣습니다.") String imageBase64,
            @ToolParam(required = false, description = "앱이 ML Kit 입술 윤곽 UPPER_LIP_BOTTOM + LOWER_LIP_TOP 점으로 구한 입 안쪽 박스(비율) {x, y, w, h}. 없으면 사진 가운데를 봅니다.") Map<String, Double> mouthBox) {
        return run("teeth_check", () -> teeth.check(imageBase64, mouthBox));
    }

    private Object run(String tool, Supplier<Object> body) {
        try {
            return body.get();
        } catch (FaceCheckException e) {
            return error(e.code(), e.getMessage());
        } catch (RuntimeException e) {
            log.warn("{} 실패: {}", tool, e.toString());
            return error("upstream_failed", "지금은 확인하지 못했어요.");
        }
    }

    static Map<String, Object> error(String code, String message) {
        return Map.of("error", Map.of("code", code, "message", message));
    }
}
