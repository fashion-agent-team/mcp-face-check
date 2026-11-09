package com.fashionagent.facecheck.makeup;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * BiSeNet 얼굴 분할(CelebAMask-HQ 19분류, ONNX). 만드는 법은 training/export_face_parsing.py.
 * 입력: 얼굴 좌표로 돌려 맞춘 512x512 RGB(ImageNet 평균·표준편차로 정규화). 출력: 칸마다 분류 번호(uint8).
 */
final class OnnxFaceParser implements FaceParser, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxFaceParser.class);
    static final int SIZE = 512;
    /** 자르는 범위(얼굴 좌표): 두 눈 사이의 4.2배 정사각형, 가운데는 눈보다 0.6 아래. CelebA-HQ 구도와 비슷함 */
    static final double SPAN = 4.2;
    static final double CENTER_Y = 0.6;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    OnnxFaceParser(Resource model) {
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                byte[] bytes = is.readAllBytes();
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(bytes, new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                log.info("얼굴 분할 모델을 읽었어요: {} ({} bytes)", model.getDescription(), bytes.length);
            } catch (IOException | OrtException ex) {
                log.warn("얼굴 분할 모델을 읽지 못했어요: {}. 눈썹은 밝기로 나눠 잽니다.", ex.toString());
                s = null;
            }
        } else {
            log.warn("얼굴 분할 모델 파일이 없어요({}). 눈썹은 밝기로 나눠 잽니다.", model == null ? "-" : model.getDescription());
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
    public FaceLabels parse(BufferedImage img, FaceFrame f) {
        if (session == null) {
            return null;
        }
        double x0 = -SPAN / 2;
        double y0 = CENTER_Y - SPAN / 2;
        float[] chw = input(img, f, x0, y0);
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, SIZE, SIZE});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            byte[] labels = new byte[SIZE * SIZE];
            ((OnnxTensor) r.get(0)).getByteBuffer().get(labels);
            return new FaceLabels(labels, SIZE, x0, y0, SPAN);
        } catch (OrtException ex) {
            log.warn("얼굴 분할 실패: {}", ex.toString());
            return null;
        }
    }

    /** 얼굴 좌표 격자마다 사진에서 쌍선형 보간으로 색을 가져옴(고개 기울임도 돌려 맞춰짐) */
    static float[] input(BufferedImage img, FaceFrame f, double x0, double y0) {
        float[] chw = new float[3 * SIZE * SIZE];
        double cell = SPAN / SIZE;
        int w = img.getWidth();
        int h = img.getHeight();
        for (int j = 0; j < SIZE; j++) {
            for (int i = 0; i < SIZE; i++) {
                Pt p = f.pixel(new Pt(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell));
                double px = Math.min(Math.max(p.x() - 0.5, 0), w - 1.001);
                double py = Math.min(Math.max(p.y() - 0.5, 0), h - 1.001);
                int ix = (int) px;
                int iy = (int) py;
                double fx = px - ix;
                double fy = py - iy;
                int c00 = img.getRGB(ix, iy);
                int c10 = img.getRGB(Math.min(ix + 1, w - 1), iy);
                int c01 = img.getRGB(ix, Math.min(iy + 1, h - 1));
                int c11 = img.getRGB(Math.min(ix + 1, w - 1), Math.min(iy + 1, h - 1));
                for (int c = 0; c < 3; c++) {
                    int sh = 16 - 8 * c; // R, G, B
                    double v = ((c00 >> sh & 0xff) * (1 - fx) + (c10 >> sh & 0xff) * fx) * (1 - fy)
                            + ((c01 >> sh & 0xff) * (1 - fx) + (c11 >> sh & 0xff) * fx) * fy;
                    chw[c * SIZE * SIZE + j * SIZE + i] = (float) ((v / 255.0 - MEAN[c]) / STD[c]);
                }
            }
        }
        return chw;
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
