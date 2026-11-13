package com.fashionagent.facecheck.teeth;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * 학습 때와 같은 입 자르기(맥북 teeth-synth/crop_rule.py와 같은 규칙).
 * <ol>
 *   <li>입 안쪽 박스 중심에서, 긴 변 × 1.36(각 변 18% 여백) 정사각형</li>
 *   <li>사진 밖은 거울(reflect) 패딩</li>
 *   <li>384×384로 줄임(모델 입력도 384라 더 줄이지 않음)</li>
 *   <li>RGB, 0~1, ImageNet mean/std 정규화, NCHW float32</li>
 * </ol>
 */
public final class MouthCrop {

    public static final double MARGIN = 1.36;
    public static final int CROP_SIZE = 384;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private MouthCrop() {
    }

    /** 같은 입을 조금씩 다르게 본 여백 배율. 모델 확률이 입 박스 위치에 따라 흔들려서 평균을 냅니다. */
    public static final double[] TTA_MARGINS = {1.25, MARGIN, 1.5};

    /** 사진에서 입 정사각형을 잘라 CROP_SIZE로 만든다. box가 없으면 사진 가운데(짧은 변의 80%)를 쓴다. */
    public static BufferedImage crop(BufferedImage img, MouthBox box) {
        return crop(img, box, MARGIN);
    }

    /** {@link #crop(BufferedImage, MouthBox)}과 같고 여백 배율만 다르게(box가 없으면 가운데 크기를 같은 비율로). */
    public static BufferedImage crop(BufferedImage img, MouthBox box, double margin) {
        int w = img.getWidth();
        int h = img.getHeight();
        double cx;
        double cy;
        double side;
        if (box != null) {
            cx = box.centerX() * w;
            cy = box.centerY() * h;
            side = Math.max(box.w() * w, box.h() * h) * margin;
        } else {
            cx = w / 2.0;
            cy = h / 2.0;
            side = Math.min(w, h) * 0.8 * margin / MARGIN;
        }
        int s = Math.max(8, (int) Math.round(side));
        int left = (int) Math.round(cx - side / 2);
        int top = (int) Math.round(cy - side / 2);
        BufferedImage sq = new BufferedImage(s, s, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < s; y++) {
            int sy = reflect(top + y, h);
            for (int x = 0; x < s; x++) {
                sq.setRGB(x, y, img.getRGB(reflect(left + x, w), sy));
            }
        }
        return resize(sq, CROP_SIZE);
    }

    /** 줄일 때는 넓이 평균(area), 키울 때는 bilinear. */
    static BufferedImage resize(BufferedImage src, int size) {
        if (src.getWidth() == size && src.getHeight() == size) {
            return src;
        }
        BufferedImage out = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            if (src.getWidth() > size) {
                Image scaled = src.getScaledInstance(size, size, Image.SCALE_AREA_AVERAGING);
                g.drawImage(scaled, 0, 0, null);
            } else {
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(src, 0, 0, size, size, null);
            }
        } finally {
            g.dispose();
        }
        return out;
    }

    /** CROP_SIZE 크롭 → 모델 입력 크기로 줄이고 정규화한 NCHW 배열. */
    public static float[] toTensor(BufferedImage crop, int size) {
        BufferedImage img = resize(crop, size);
        float[] out = new float[3 * size * size];
        int plane = size * size;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int rgb = img.getRGB(x, y);
                int i = y * size + x;
                out[i] = (((rgb >> 16) & 0xFF) / 255f - MEAN[0]) / STD[0];
                out[plane + i] = (((rgb >> 8) & 0xFF) / 255f - MEAN[1]) / STD[1];
                out[2 * plane + i] = ((rgb & 0xFF) / 255f - MEAN[2]) / STD[2];
            }
        }
        return out;
    }

    /** 좌우로 뒤집은 사진. */
    public static BufferedImage mirror(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                out.setRGB(w - 1 - x, y, src.getRGB(x, y));
            }
        }
        return out;
    }

    /** numpy/OpenCV BORDER_REFLECT_101과 같은 거울 좌표. */
    static int reflect(int i, int n) {
        if (n == 1) {
            return 0;
        }
        int period = 2 * (n - 1);
        int m = Math.floorMod(i, period);
        return m < n ? m : period - m;
    }
}
