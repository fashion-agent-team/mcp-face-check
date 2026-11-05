package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.http.client.ClientHttpRequestFactoryBuilder;
import org.springframework.boot.http.client.HttpClientSettings;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.json.JsonMapper;

/** OPENAI_API_KEY가 있을 때만 GPT 폴백을 만듭니다. 키는 환경 변수로만 받습니다. */
@Configuration(proxyBeanMethods = false)
class MakeupVisionConfig {

    @Bean
    @ConditionalOnExpression("!'${face.openai.api-key:}'.isBlank()")
    MakeupVision openAiMakeupVision(FaceCheckProperties props, RestClient.Builder builder, JsonMapper json) {
        FaceCheckProperties.OpenAi o = props.openai();
        var factory = ClientHttpRequestFactoryBuilder.detect()
                .build(HttpClientSettings.defaults().withTimeouts(o.timeout(), o.timeout()));
        RestClient http = builder.clone().baseUrl(o.baseUrl()).requestFactory(factory)
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + o.apiKey()).build();
        return new OpenAiMakeupVision(http, o, json);
    }
}
