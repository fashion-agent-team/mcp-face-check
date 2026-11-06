package com.fashionagent.facecheck;

import static org.assertj.core.api.Assertions.assertThat;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import io.modelcontextprotocol.spec.McpSchema;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.web.client.RestClient;

/** 실제로 서버를 띄우고 MCP 클라이언트로 /mcp에 붙어 본다(agent가 붙는 방식과 같음). */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {})
class McpServerTest {

    @LocalServerPort
    int port;

    McpSyncClient client;

    @BeforeEach
    void setUp() {
        var transport = HttpClientStreamableHttpTransport.builder("http://localhost:" + port).endpoint("/mcp").build();
        client = McpClient.sync(transport).build();
        client.initialize();
    }

    @AfterEach
    void close() {
        client.closeGracefully();
    }

    @Test
    @DisplayName("tools list에 face_makeup_check, teeth_check, imageBase64만 필수")
    @SuppressWarnings("unchecked")
    void toolsList() {
        var tools = client.listTools().tools();
        assertThat(tools).extracting(McpSchema.Tool::name).containsExactlyInAnyOrder("face_makeup_check", "teeth_check");
        var makeup = tools.stream().filter(t -> t.name().equals("face_makeup_check")).findFirst().orElseThrow();
        assertThat((List<String>) makeup.inputSchema().get("required")).containsExactly("imageBase64");
        assertThat(makeup.inputSchema().get("properties").toString()).contains("landmarks", "leftEyebrowTop", "upperLipTop");
    }

    @Test
    @DisplayName("윤곽점 + 사진 → 대칭 결과 JSON (agent가 읽는 framing·lighting·symmetry·source)")
    void makeup() {
        var f = new SyntheticFace();
        f.edits.put("leftEyebrowTop", SyntheticFace.shift(0, -0.12));
        f.edits.put("leftEyebrowBottom", SyntheticFace.shift(0, -0.12));
        String out = text(call("face_makeup_check", Map.of("imageBase64", f.jpegBase64(), "landmarks", f.landmarks())));
        assertThat(out).contains("\"framing\":null", "\"lighting\":\"ok\"", "\"ok\":false",
                "왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요.", "\"source\":\"model\"");
    }

    @Test
    @DisplayName("멈춘 1초 동안의 여러 장(frames)을 같이 보내면 중간값으로 판단")
    void frames() {
        var f = new SyntheticFace();
        Map<String, Object> frame = Map.of("imageBase64", f.jpegBase64(), "landmarks", f.landmarks());
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", f.jpegBase64(), "landmarks", f.landmarks(),
                "frames", List.of(frame, frame))))).contains("\"ok\":true", "\"frames\":3");
    }

    @Test
    @DisplayName("PNG도 읽음, 좌우 같으면 ok")
    void png() {
        var f = new SyntheticFace();
        String png = Base64.getEncoder().encodeToString(f.bytes("png"));
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", png, "landmarks", f.landmarks()))))
                .contains("\"ok\":true", "\"notes\":[]");
    }

    @Test
    @DisplayName("윤곽점이 없거나 모자라면 no_landmarks (GPT로 대신 보지 않음)")
    void noLandmarks() {
        var f = new SyntheticFace();
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", f.jpegBase64()))))
                .contains("\"code\":\"no_landmarks\"");
        Map<String, Object> partial = new HashMap<>(f.landmarks());
        partial.remove("upperLipTop");
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", f.jpegBase64(), "landmarks", partial))))
                .contains("\"code\":\"no_landmarks\"");
    }

    @Test
    @DisplayName("잘못된 입력은 error JSON")
    void errors() {
        var f = new SyntheticFace();
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", "")))).contains("\"code\":\"no_image\"");
        assertThat(text(call("face_makeup_check", Map.of("imageBase64",
                Base64.getEncoder().encodeToString("not an image".getBytes()), "landmarks", f.landmarks()))))
                .contains("\"code\":\"bad_image\"");
        Map<String, Object> pixels = new HashMap<>(f.landmarks());
        pixels.put("leftEye", List.of(List.of(240.0, 360.0), List.of(250.0, 355.0), List.of(260.0, 360.0)));
        assertThat(text(call("face_makeup_check", Map.of("imageBase64", f.jpegBase64(), "landmarks", pixels))))
                .contains("\"code\":\"bad_request\"");
    }

    @Test
    @DisplayName("teeth_check는 분류 모델이 들어오기 전까지 not_ready")
    void teethNotReady() {
        assertThat(text(call("teeth_check", Map.of("imageBase64", new SyntheticFace().jpegBase64(),
                "mouthBox", Map.of("x", 0.3, "y", 0.55, "w", 0.4, "h", 0.2)))))
                .contains("\"code\":\"not_ready\"");
    }

    @Test
    @DisplayName("actuator health는 UP")
    void health() {
        assertThat(RestClient.create("http://localhost:" + port).get().uri("/actuator/health").retrieve().body(String.class))
                .contains("UP");
    }

    private McpSchema.CallToolResult call(String tool, Map<String, Object> args) {
        return client.callTool(new McpSchema.CallToolRequest(tool, args));
    }

    private static String text(McpSchema.CallToolResult result) {
        return ((McpSchema.TextContent) result.content().getFirst()).text();
    }
}
