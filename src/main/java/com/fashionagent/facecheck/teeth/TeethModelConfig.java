package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.FaceCheckProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ResourceLoader;

@Configuration
public class TeethModelConfig {

    @Bean(destroyMethod = "close")
    OnnxTeethClassifier teethClassifier(FaceCheckProperties props, ResourceLoader loader) {
        FaceCheckProperties.Teeth t = props.teeth();
        return new OnnxTeethClassifier(loader.getResource(t.model()), t.inputSize());
    }
}
