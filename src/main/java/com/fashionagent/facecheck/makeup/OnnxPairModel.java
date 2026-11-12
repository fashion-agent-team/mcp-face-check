package com.fashionagent.facecheck.makeup;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * 좌우 조각 비교 모델(ONNX, 약 0.8MB). 입력: [2,3,H,W] RGB 0~1. 0번은 사진 왼쪽 조각, 1번은 오른쪽 조각을 좌우로 뒤집은 것
 * (둘 다 바깥쪽이 열 0). 조각 범위는 얼굴 좌표(두 눈 사이 = 1)로 정함. 출력: [1] 점수(0번 − 1번).
 * 모델을 여러 개 주거나 자르는 범위를 조금씩 옮겨(jitters) 여러 번 재면 점수를 평균함(윤곽점 흔들림·모델 하나의 버릇이 줄어듦).
 */
final class OnnxPairModel implements PairModel, AutoCloseable {

    /** 조각 크기(px)와 얼굴 좌표 범위. x는 가운데 쪽(xIn)~바깥(xOut) 절댓값, y는 위~아래. 학습 때와 같아야 함 */
    record Region(int w, int h, double xIn, double xOut, double yTop, double yBot) {
        /** 눈썹: 바깥 1.45 ~ 가운데 0.05, 눈 위 0.95 ~ 0.07 */
        static final Region BROW = new Region(128, 80, 0.05, 1.45, -0.95, -0.07);
        /** 입꼬리: 바깥 0.85 ~ 가운데 0.05, 눈 아래 0.75 ~ 1.35 */
        static final Region LIP_CORNER = new Region(128, 96, 0.05, 0.85, 0.75, 1.35);
        /** 볼: 바깥 1.15 ~ 가운데 0.2, 눈 아래 0.15 ~ 1.05 */
        static final Region CHEEK = new Region(112, 112, 0.2, 1.15, 0.15, 1.05);
        /** 눈(눈썹 아래~눈 밑): 바깥 1.05 ~ 가운데 0.05, 눈 위 0.55 ~ 눈 아래 0.3 */
        static final Region EYE = new Region(120, 96, 0.05, 1.05, -0.55, 0.3);
    }

    /** 자르는 범위 옮기기: dx(가운데 쪽 +), dy(아래 +)는 두 눈 사이 = 1 기준, scale은 범위 크기 배율 */
    record Jitter(double dx, double dy, double scale) {
        static final List<Jitter> NONE = List.of(new Jitter(0, 0, 1));
        /** 눈화장 평가(FFHQ)에서 쓴 7가지. training/eye_makeup_data.py의 patches와 같은 규칙 */
        static final List<Jitter> SEVEN = List.of(new Jitter(0, 0, 1), new Jitter(.03, 0, 1), new Jitter(-.03, 0, 1),
                new Jitter(0, .03, 1), new Jitter(0, -.03, 1), new Jitter(0, 0, 1.06), new Jitter(0, 0, .94));
    }

    private static final Logger log = LoggerFactory.getLogger(OnnxPairModel.class);

    private final Region region;
    private final List<Jitter> jitters;
    private final OrtEnvironment env;
    private final List<OrtSession> sessions;

    OnnxPairModel(Resource model, Region region) {
        this(model == null ? List.of() : List.of(model), region, Jitter.NONE);
    }

    /** 모델 하나라도 못 읽으면 그 모델만 빼고 씀. 하나도 없으면 score는 null(그전 방식으로 잼) */
    OnnxPairModel(List<Resource> models, Region region, List<Jitter> jitters) {
        this.region = region;
        this.jitters = jitters.isEmpty() ? Jitter.NONE : List.copyOf(jitters);
        OrtEnvironment e = null;
        List<OrtSession> ss = new ArrayList<>();
        for (Resource model : models) {
            if (model == null || !model.exists()) {
                continue;
            }
            try (InputStream is = model.getInputStream()) {
                e = OrtEnvironment.getEnvironment();
                ss.add(e.createSession(is.readAllBytes(), new OrtSession.SessionOptions()));
                log.info("좌우 비교 모델을 읽었어요: {}", model.getDescription());
            } catch (IOException | OrtException ex) {
                log.warn("좌우 비교 모델을 읽지 못했어요: {}. 그 모델은 빼고 잽니다.", ex.toString());
            }
        }
        this.env = e;
        this.sessions = List.copyOf(ss);
    }

    @Override
    public Double score(BufferedImage img, FaceFrame f) {
        if (sessions.isEmpty()) {
            return null;
        }
        double sum = 0;
        int n = 0;
        for (Jitter j : jitters) {
            float[] x = patches(img, f, region, j);
            for (OrtSession s : sessions) {
                try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[] {2, 3, region.h(), region.w()});
                        OrtSession.Result r = s.run(Map.of(s.getInputNames().iterator().next(), t))) {
                    float[] out = new float[1];
                    ((OnnxTensor) r.get(0)).getFloatBuffer().get(out);
                    if (!Double.isFinite(out[0])) {
                        return null;
                    }
                    sum += out[0];
                    n++;
                } catch (OrtException ex) {
                    log.warn("좌우 비교 모델 실패: {}", ex.toString());
                    return null;
                }
            }
        }
        return sum / n;
    }

    static float[] patches(BufferedImage img, FaceFrame f, Region rg) {
        return patches(img, f, rg, Jitter.NONE.get(0));
    }

    /** 양쪽 조각을 쌍선형 보간으로 잘라 [2,3,H,W]로. 바깥쪽이 열 0. */
    /** 양쪽 조각을 쌍선형 보간으로 잘라 [2,3,H,W]로. 바깥쪽이 열 0. */
    static float[] patches(BufferedImage img, FaceFrame f, Region rg, Jitter jt) {
        int W = rg.w();
        int H = rg.h();
        float[] out = new float[2 * 3 * H * W];
        int w = img.getWidth();
        int h = img.getHeight();
        for (int k = 0; k < 2; k++) {
            double sgn = k == 0 ? -1 : 1;
            for (int j = 0; j < H; j++) {
                double y = (rg.yTop() + j * (rg.yBot() - rg.yTop()) / (H - 1)) * jt.scale() + jt.dy();
                for (int i = 0; i < W; i++) {
                    double x = sgn * ((rg.xOut() + i * (rg.xIn() - rg.xOut()) / (W - 1)) * jt.scale() - jt.dx());
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

    /**
     * 양쪽 조각의 선명도: 회색조 라플라시안(3×3, 가장자리는 반사) 절댓값 평균 ÷ 회색조 표준편차. 대비로 나눠서 밝기·화장 진하기와
     * 상관없이 "윤곽이 얼마나 날카로운지"만 봄. 흐린 사진(초점 나감, 흔들림)일수록 작음.
     */
    static double sharpness(float[] patches, Region rg) {
        int W = rg.w();
        int H = rg.h();
        int n = W * H;
        double lapSum = 0;
        double sum = 0;
        double sq = 0;
        for (int k = 0; k < 2; k++) {
            float[] g = new float[n];
            for (int i = 0; i < n; i++) {
                int o = k * 3 * n + i;
                g[i] = (float) (0.299 * patches[o] + 0.587 * patches[o + n] + 0.114 * patches[o + 2 * n]);
                sum += g[i];
                sq += (double) g[i] * g[i];
            }
            for (int j = 0; j < H; j++) {
                int up = (j == 0 ? 1 : j - 1) * W;
                int dn = (j == H - 1 ? H - 2 : j + 1) * W;
                for (int i = 0; i < W; i++) {
                    int l = i == 0 ? 1 : i - 1;
                    int r = i == W - 1 ? W - 2 : i + 1;
                    int row = j * W;
                    lapSum += Math.abs(g[up + i] + g[dn + i] + g[row + l] + g[row + r] - 4 * g[row + i]);
                }
            }
        }
        double mean = sum / (2.0 * n);
        double std = Math.sqrt(Math.max(sq / (2.0 * n) - mean * mean, 0));
        return lapSum / (2.0 * n) / (std + 1e-3);
    }

    @Override
    public void close() throws OrtException {
        for (OrtSession s : sessions) {
            s.close();
        }
    }
}
