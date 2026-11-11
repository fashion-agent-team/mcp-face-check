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
 * 좌우 조각 비교 모델(ONNX, 약 0.8MB). 입력: [2,3,H,W] RGB 0~1. 0번은 사진 왼쪽 조각, 1번은 오른쪽 조각을 좌우로 뒤집은 것
 * (둘 다 바깥쪽이 열 0). 조각 범위는 얼굴 좌표(두 눈 사이 = 1)로 정함. 출력: [1] 점수(0번 − 1번).
 */
final class OnnxPairModel implements PairModel, AutoCloseable {

    /** 조각 크기(px)와 얼굴 좌표 범위. x는 가운데 쪽(xIn)~바깥(xOut) 절댓값, y는 위~아래. 학습 때와 같아야 함 */
    record Region(int w, int h, double xIn, double xOut, double yTop, double yBot) {
        /** 눈썹: 바깥 1.45 ~ 가운데 0.05, 눈 위 0.95 ~ 0.07 */
        static final Region BROW = new Region(128, 80, 0.05, 1.45, -0.95, -0.07);
        /** 입꼬리: 바깥 0.85 ~ 가운데 0.05, 눈 아래 0.75 ~ 1.35 */
        static final Region LIP_CORNER = new Region(128, 96, 0.05, 0.85, 0.75, 1.35);
    }

    private static final Logger log = LoggerFactory.getLogger(OnnxPairModel.class);

    private final Region region;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    OnnxPairModel(Resource model, Region region) {
        this.region = region;
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(is.readAllBytes(), new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                log.info("좌우 비교 모델을 읽었어요: {}", model.getDescription());
            } catch (IOException | OrtException ex) {
                log.warn("좌우 비교 모델을 읽지 못했어요: {}. 그 항목은 그전 방식으로 잽니다.", ex.toString());
                s = null;
            }
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
    }

    @Override
    public Double score(BufferedImage img, FaceFrame f) {
        if (session == null) {
            return null;
        }
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(patches(img, f, region)), new long[] {2, 3, region.h(), region.w()});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            float[] out = new float[1];
            ((OnnxTensor) r.get(0)).getFloatBuffer().get(out);
            return Double.isFinite(out[0]) ? (double) out[0] : null;
        } catch (OrtException ex) {
            log.warn("좌우 비교 모델 실패: {}", ex.toString());
            return null;
        }
    }

    /** 양쪽 조각을 쌍선형 보간으로 잘라 [2,3,H,W]로. 바깥쪽이 열 0. */
    static float[] patches(BufferedImage img, FaceFrame f, Region rg) {
        int W = rg.w();
        int H = rg.h();
        float[] out = new float[2 * 3 * H * W];
        int w = img.getWidth();
        int h = img.getHeight();
        for (int k = 0; k < 2; k++) {
            double sgn = k == 0 ? -1 : 1;
            for (int j = 0; j < H; j++) {
                double y = rg.yTop() + j * (rg.yBot() - rg.yTop()) / (H - 1);
                for (int i = 0; i < W; i++) {
                    double x = sgn * (rg.xOut() + i * (rg.xIn() - rg.xOut()) / (W - 1));
                    Pt p = f.pixel(new Pt(x, y));
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
                        int sh = 16 - 8 * c;
                        double v = ((c00 >> sh & 0xff) * (1 - fx) + (c10 >> sh & 0xff) * fx) * (1 - fy)
                                + ((c01 >> sh & 0xff) * (1 - fx) + (c11 >> sh & 0xff) * fx) * fy;
                        out[((k * 3 + c) * H + j) * W + i] = (float) (v / 255.0);
                    }
                }
            }
        }
        return out;
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
