package com.fashionagent.facecheck.teeth;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * 우리가 학습한 치아 이물질 분류 모델(MobileNetV3-Small, 합성 데이터 학습)을 ONNX Runtime으로 이 서버 안에서 돌립니다.
 * 사진을 외부 AI로 보내지 않습니다. 입력 "image" float32 [1, 3, S, S], 출력 "logits" float32 [1, 1].
 * 모델 파일이 없으면 {@link #ready()}가 false이고 teeth_check는 not_ready를 돌려줍니다.
 */
public class OnnxTeethClassifier implements TeethClassifier, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxTeethClassifier.class);

    private final int size;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public OnnxTeethClassifier(Resource model, int size) {
        this.size = size;
        OrtSession s = null;
        OrtEnvironment e = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                byte[] bytes = is.readAllBytes();
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(bytes, new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                log.info("치아 모델을 읽었어요: {} ({} bytes, 입력 {}px)", model.getDescription(), bytes.length, size);
            } catch (IOException | OrtException ex) {
                log.warn("치아 모델을 읽지 못했어요: {}", ex.toString());
                s = null;
            }
        } else {
            log.warn("치아 모델 파일이 없어요({}). teeth_check는 not_ready를 돌려줍니다.", model == null ? "-" : model.getDescription());
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
    }

    @Override
    public boolean ready() {
        return session != null;
    }

    @Override
    public int inputSize() {
        return size;
    }

    @Override
    public double foodProbability(float[] chw) {
        if (session == null) {
            throw new IllegalStateException("model not loaded");
        }
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, size, size});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            float[][] logits = (float[][]) r.get(0).getValue();
            return 1.0 / (1.0 + Math.exp(-logits[0][0]));
        } catch (OrtException e) {
            throw new IllegalStateException("model run failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
