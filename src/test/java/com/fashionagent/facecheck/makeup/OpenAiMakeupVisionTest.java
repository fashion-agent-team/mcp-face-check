package com.fashionagent.facecheck.makeup;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.header;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.jsonPath;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import java.time.Duration;
import java.util.Map;
import org.hamcrest.Matchers;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.json.JsonMapper;

class OpenAiMakeupVisionTest {

    static final JsonMapper JSON = JsonMapper.builder().build();

    MockRestServiceServer server;
    OpenAiMakeupVision vision;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("https://api.openai.test")
                .defaultHeader("Authorization", "Bearer test-key");
        server = MockRestServiceServer.bindTo(builder).build();
        vision = new OpenAiMakeupVision(builder.build(),
                new FaceCheckProperties.OpenAi("test-key", "gpt-test", "https://api.openai.test", Duration.ofSeconds(5)), JSON);
    }

    private static String reply(Map<String, Object> content) {
        return JSON.writeValueAsString(Map.of("choices",
                java.util.List.of(Map.of("message", Map.of("content", JSON.writeValueAsString(content))))));
    }

    @Test
    @DisplayName("사진을 data URL로 보내고 JSON 스키마로 받음, temperature는 보내지 않음, source=gpt")
    void notes() {
        server.expect(requestTo("https://api.openai.test/v1/chat/completions"))
                .andExpect(header("Authorization", "Bearer test-key"))
                .andExpect(jsonPath("$.model").value("gpt-test"))
                .andExpect(jsonPath("$.temperature").doesNotExist())
                .andExpect(jsonPath("$.response_format.json_schema.strict").value(true))
                .andExpect(jsonPath("$.messages[1].content[1].image_url.url").value(Matchers.startsWith("data:image/jpeg;base64,/9j/")))
                .andRespond(withSuccess(reply(Map.of("framing", "", "lighting", "ok",
                        "notes", java.util.List.of("왼쪽 눈썹이 오른쪽보다 조금 짧게 그려졌어요.", " "))), MediaType.APPLICATION_JSON));
        MakeupResult r = vision.look("/9j/AAAA");
        assertThat(r.source()).isEqualTo("gpt");
        assertThat(r.framing()).isNull();
        assertThat(r.symmetry().ok()).isFalse();
        assertThat(r.symmetry().notes()).containsExactly("왼쪽 눈썹이 오른쪽보다 조금 짧게 그려졌어요.");
        server.verify();
    }

    @Test
    @DisplayName("GPT가 구도 문제를 말하면 framing만")
    void framing() {
        server.expect(requestTo("https://api.openai.test/v1/chat/completions"))
                .andRespond(withSuccess(reply(Map.of("framing", "얼굴이 잘렸어요. 폰을 조금 멀리 들어 주세요.",
                        "lighting", "ok", "notes", java.util.List.of())), MediaType.APPLICATION_JSON));
        MakeupResult r = vision.look("data:image/jpeg;base64,/9j/AAAA");
        assertThat(r.framing()).contains("멀리");
        assertThat(r.symmetry()).isNull();
    }

    @Test
    @DisplayName("OpenAI가 실패하면 upstream_failed")
    void failure() {
        server.expect(requestTo("https://api.openai.test/v1/chat/completions")).andRespond(withServerError());
        assertThatThrownBy(() -> vision.look("/9j/AAAA")).isInstanceOf(FaceCheckException.class)
                .extracting("code").isEqualTo("upstream_failed");
    }
}
