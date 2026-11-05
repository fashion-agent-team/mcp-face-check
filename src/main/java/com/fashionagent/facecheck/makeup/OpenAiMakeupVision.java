package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * 윤곽점이 없을 때만 쓰는 폴백: OpenAI Chat Completions에 사진 한 장을 보내고 JSON으로 받습니다.
 * 판단하지 말고 보이는 차이만 말하게 합니다. gpt-5 계열은 temperature를 바꿀 수 없어 보내지 않습니다.
 */
public class OpenAiMakeupVision implements MakeupVision {

    private static final Logger log = LoggerFactory.getLogger(OpenAiMakeupVision.class);

    static final String PROMPT = """
            시각장애인이 셀카 화면 방향(거울처럼 좌우를 뒤집은)으로 찍은 얼굴 사진입니다. 사진 왼쪽이 사용자 왼쪽입니다.
            화장의 좌우 대칭만 봅니다: 눈썹 높이·길이·두께·꼬리, 입술 선(입꼬리, 윗입술·아랫입술 두께), 볼 블러셔 색, 입술 색.
            보이는 차이만 짧게 말하고 꾸미지 마세요. 피부·얼굴 생김새·외모 평가는 하지 않습니다.
            확실하지 않으면 차이가 없다고 하세요.
            framing: 얼굴이 잘렸거나 너무 작거나 옆으로 돌았으면 어떻게 폰을 옮길지 한국어 존댓말 한 문장, 괜찮으면 null.
            lighting: 너무 어두워 색을 볼 수 없으면 dark, 아니면 ok.
            notes: 좌우가 다른 점을 한국어 존댓말 한 문장씩 최대 3개(예: "왼쪽 눈썹이 오른쪽보다 조금 짧게 그려졌어요."). 고르면 빈 배열.
            """;

    private static final Map<String, Object> SCHEMA = Map.of(
            "type", "object",
            "additionalProperties", false,
            "required", List.of("framing", "lighting", "notes"),
            "properties", Map.of(
                    "framing", Map.of("type", List.of("string", "null")),
                    "lighting", Map.of("type", "string", "enum", List.of("ok", "dark")),
                    "notes", Map.of("type", "array", "items", Map.of("type", "string"))));

    private final RestClient http;
    private final String model;
    private final JsonMapper json;

    public OpenAiMakeupVision(RestClient http, FaceCheckProperties.OpenAi props, JsonMapper json) {
        this.http = http;
        this.model = props.model();
        this.json = json;
    }

    @Override
    public MakeupResult look(String imageBase64) {
        String b64 = imageBase64.strip();
        String url = b64.startsWith("data:") ? b64 : "data:image/jpeg;base64," + b64;
        Map<String, Object> body = Map.of(
                "model", model,
                "reasoning_effort", "low",
                "messages", List.of(
                        Map.of("role", "system", "content", PROMPT),
                        Map.of("role", "user", "content", List.of(
                                Map.of("type", "text", "text", "이 얼굴의 화장 좌우 대칭을 봐 주세요."),
                                Map.of("type", "image_url", "image_url", Map.of("url", url, "detail", "high"))))),
                "response_format", Map.of("type", "json_schema",
                        "json_schema", Map.of("name", "makeup_symmetry", "strict", true, "schema", SCHEMA)));
        JsonNode res;
        try {
            res = http.post().uri("/v1/chat/completions").contentType(MediaType.APPLICATION_JSON)
                    .body(body).retrieve().body(JsonNode.class);
        } catch (RestClientException e) {
            log.warn("OpenAI 호출 실패: {}", e.getMessage());
            throw new FaceCheckException("upstream_failed", "지금은 얼굴을 확인하지 못했어요.");
        }
        String content = res == null ? "" : res.path("choices").path(0).path("message").path("content").asString("");
        JsonNode out;
        try {
            out = json.readTree(content);
        } catch (RuntimeException e) {
            log.warn("OpenAI 응답을 읽지 못함");
            throw new FaceCheckException("upstream_failed", "지금은 얼굴을 확인하지 못했어요.");
        }
        String framing = out.path("framing").isString() ? out.path("framing").asString() : null;
        if (framing != null && framing.isBlank()) {
            framing = null;
        }
        String lighting = "dark".equals(out.path("lighting").asString("")) ? "dark" : "ok";
        if (framing != null || "dark".equals(lighting)) {
            return new MakeupResult(framing, lighting, null, MakeupResult.GPT);
        }
        List<String> notes = new ArrayList<>();
        out.path("notes").valueStream().map(n -> n.asString("").strip()).filter(s -> !s.isEmpty()).limit(3)
                .forEach(notes::add);
        return new MakeupResult(null, lighting, new MakeupResult.Symmetry(notes.isEmpty(), notes, List.of(), List.of()),
                MakeupResult.GPT);
    }
}
