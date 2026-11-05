package com.fashionagent.facecheck;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.facecheck.makeup.MakeupVision;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/** OPENAI_API_KEY가 있으면 GPT 폴백 빈이 생긴다(실제 호출은 안 함). */
@SpringBootTest(properties = "face.openai.api-key=test-only-not-a-key")
class OpenAiConfigTest {

    @Autowired
    ObjectProvider<MakeupVision> vision;

    @Test
    @DisplayName("키가 있으면 MakeupVision 빈이 있음")
    void created() {
        assertThat(vision.getIfAvailable()).isNotNull();
    }
}
