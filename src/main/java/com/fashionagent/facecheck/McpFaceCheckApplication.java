package com.fashionagent.facecheck;

import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
@ConfigurationPropertiesScan
public class McpFaceCheckApplication {

    public static void main(String[] args) {
        SpringApplication.run(McpFaceCheckApplication.class, args);
    }

    /** {@link FaceTools}의 @Tool 메서드를 MCP 도구로 내놓는다. */
    @Bean
    ToolCallbackProvider faceToolCallbacks(FaceTools faceTools) {
        return MethodToolCallbackProvider.builder().toolObjects(faceTools).build();
    }
}
