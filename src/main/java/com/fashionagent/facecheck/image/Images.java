package com.fashionagent.facecheck.image;

import com.fashionagent.facecheck.FaceCheckException;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Iterator;

import javax.imageio.ImageIO;
import javax.imageio.ImageReadParam;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

/** 사진 base64 풀기와 색 계산. 사진은 메모리에서만 쓰고 저장하지 않습니다. */
public final class Images {

    private Images() {
    }

    public static BufferedImage decode(String imageBase64, long maxBytes) {
        if (imageBase64 == null || imageBase64.isBlank()) {
            throw new FaceCheckException("no_image", "사진이 없어요.");
        }
        String b64 = imageBase64.strip();
        int comma = b64.indexOf(',');
        if (b64.startsWith("data:") && comma > 0) {
            b64 = b64.substring(comma + 1);
        }
        if (b64.length() / 4L * 3 > maxBytes) {
            throw new FaceCheckException("bad_image", "사진이 너무 커요.");
        }
        byte[] bytes;
        try {
            bytes = Base64.getMimeDecoder().decode(b64);
        } catch (IllegalArgumentException e) {
            throw new FaceCheckException("bad_image", "사진을 읽지 못했어요.");
        }
        try {
            BufferedImage img = read(bytes, DECODE_MAX_SIDE);
            if (img == null || img.getWidth() < 32 || img.getHeight() < 32) {
                throw new FaceCheckException("bad_image", "사진을 읽지 못했어요.");
            }
            return img;
        } catch (IOException | RuntimeException e) {
            if (e instanceof FaceCheckException fe) {
                throw fe;
            }
            throw new FaceCheckException("bad_image", "사진을 읽지 못했어요.");
        }
    }

    /**
     * 사진을 풀 때 긴 변 기준. 이보다 2배 이상 큰 사진은 건너뛰며 읽습니다(서브샘플링). 몇 KB짜리 PNG가 4만×4만 픽셀이면
     * 그대로 풀 때 수 GB가 들어 서버가 멈추는 것을 막습니다. 폰 사진(4032px까지)은 그대로 읽어 판정이 바뀌지 않습니다.
     */
    static final int DECODE_MAX_SIDE = 2048;

    /** 크기를 먼저 읽고, 긴 변이 maxSide의 2배 이상이면 건너뛰며 읽습니다. 읽을 수 없으면 null. */
    static BufferedImage read(byte[] bytes, int maxSide) throws IOException {
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(bytes))) {
            if (in == null) {
                return null;
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(in);
            if (!readers.hasNext()) {
                return null;
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(in, true, true);
                int longSide = Math.max(reader.getWidth(0), reader.getHeight(0));
                ImageReadParam param = reader.getDefaultReadParam();
                int step = Math.max(1, longSide / maxSide);
                if (step > 1) {
                    param.setSourceSubsampling(step, step, 0, 0);
                }
                return reader.read(0, param);
            } finally {
                reader.dispose();
            }
        }
    }

    /** sRGB → CIELAB (D65). 돌려주는 값: {L, a, b}. */
    public static double[] lab(int rgb) {
        double r = linear(((rgb >> 16) & 0xFF) / 255.0);
        double g = linear(((rgb >> 8) & 0xFF) / 255.0);
        double b = linear((rgb & 0xFF) / 255.0);
        double x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
        double y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
        double z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
        double fx = f(x);
        double fy = f(y);
        double fz = f(z);
        return new double[] {116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)};
    }

    private static double linear(double c) {
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    }

    private static double f(double t) {
        return t > 216.0 / 24389 ? Math.cbrt(t) : (24389.0 / 27 * t + 16) / 116;
    }

    /** 여러 픽셀의 Lab 평균을 모은다. */
    public static final class LabMean {
        private double l;
        private double a;
        private double b;
        private int n;

        public void add(double[] lab) {
            l += lab[0];
            a += lab[1];
            b += lab[2];
            n++;
        }

        public int count() {
            return n;
        }

        public double l() {
            return n == 0 ? Double.NaN : l / n;
        }

        public double a() {
            return n == 0 ? Double.NaN : a / n;
        }

        public double b() {
            return n == 0 ? Double.NaN : b / n;
        }

        /** ΔE76 */
        public double distance(LabMean o) {
            return Math.sqrt(Math.pow(l() - o.l(), 2) + Math.pow(a() - o.a(), 2) + Math.pow(b() - o.b(), 2));
        }
    }
}
