package com.fashionagent.facecheck;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import javax.imageio.ImageIO;

/**
 * 테스트용 가짜 얼굴: ML Kit 윤곽점 모양(이름·점 개수)과 그 윤곽대로 칠한 사진을 같이 만든다.
 * 좌표는 두 눈 사이 거리 = 1인 얼굴 좌표로 그린 뒤 사진 위치·크기·기울기로 옮긴다.
 */
public final class SyntheticFace {

    public int width = 600;
    public int height = 800;
    public double centerX = 300;
    public double centerY = 360;
    public double iod = 130;
    public double rollDegrees = 0;
    public Color background = new Color(120, 120, 125);
    public Color skin = new Color(226, 184, 156);
    public Color lipLeft = new Color(188, 62, 74);
    public Color lipRight = new Color(188, 62, 74);
    public Color blushLeft = null;
    public Color blushRight = null;
    /** 사진 오른쪽 절반을 이만큼 어둡게(0~1). 한쪽 그늘 */
    public double shadeRight = 0;
    /** 전체 밝기 배율 */
    public double brightness = 1;
    public Double yaw = null;
    /** 윤곽 이름 → 얼굴 좌표 점 바꾸기 */
    public final Map<String, UnaryOperator<List<double[]>>> edits = new LinkedHashMap<>();
    /** 사진에만 적용하는 바꾸기(윤곽점은 그대로). 윤곽점이 그린 화장을 못 따라가는 실제 상황 */
    public final Map<String, UnaryOperator<List<double[]>>> drawEdits = new LinkedHashMap<>();
    /** true면 ML Kit 이름의 left/right를 바꿔서 보낸다(이름이 반대로 와도 서버가 x로 맞추는지) */
    public boolean swapNames = false;

    public Map<String, List<double[]>> localContours() {
        Map<String, List<double[]>> c = new LinkedHashMap<>();
        c.put("face", ellipse(0, 0.4, 1.2, 1.4, 36));
        c.put("leftEye", ellipse(-0.5, 0, 0.17, 0.07, 16));
        c.put("rightEye", ellipse(0.5, 0, 0.17, 0.07, 16));
        c.put("leftEyebrowTop", brow(-1, -0.40));
        c.put("leftEyebrowBottom", brow(-1, -0.31));
        c.put("rightEyebrowTop", brow(1, -0.40));
        c.put("rightEyebrowBottom", brow(1, -0.31));
        c.put("upperLipTop", lipLine(11, -0.13, true));
        c.put("upperLipBottom", lipLine(9, -0.015, false));
        c.put("lowerLipTop", lipLine(9, 0.015, false));
        c.put("lowerLipBottom", lipLine(9, 0.16, true));
        c.put("noseBridge", List.of(new double[] {0, -0.15}, new double[] {0, 0.45}));
        c.put("noseBottom", List.of(new double[] {-0.15, 0.66}, new double[] {0, 0.70}, new double[] {0.15, 0.66}));
        c.put("leftCheek", List.of(new double[] {-0.55, 0.55}));
        c.put("rightCheek", List.of(new double[] {0.55, 0.55}));
        edits.forEach((name, fn) -> c.put(name, fn.apply(c.get(name))));
        return c;
    }

    /** 앱이 보낼 landmarks 맵(비율 좌표) */
    public Map<String, Object> landmarks() {
        Map<String, Object> out = new LinkedHashMap<>();
        localContours().forEach((name, pts) -> {
            String key = name;
            if (swapNames) {
                key = name.startsWith("left") ? "right" + name.substring(4)
                        : name.startsWith("right") ? "left" + name.substring(5) : name;
            }
            List<List<Double>> list = new ArrayList<>();
            for (double[] p : pts) {
                double[] px = toPixel(p);
                list.add(List.of(px[0] / width, px[1] / height));
            }
            out.put(key, list);
        });
        if (yaw != null) {
            out.put("yaw", yaw);
        }
        return out;
    }

    public BufferedImage image() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(background);
        g.fillRect(0, 0, width, height);
        Map<String, List<double[]>> c = localContours();
        drawEdits.forEach((name, fn) -> c.put(name, fn.apply(c.get(name))));
        fill(g, skin, c.get("face"));
        if (blushLeft != null) {
            fill(g, blushLeft, ellipse(-0.55, 0.55, 0.2, 0.2, 24));
        }
        if (blushRight != null) {
            fill(g, blushRight, ellipse(0.55, 0.55, 0.2, 0.2, 24));
        }
        Color brow = new Color(70, 50, 40);
        fill(g, brow, closed(c.get("leftEyebrowTop"), c.get("leftEyebrowBottom")));
        fill(g, brow, closed(c.get("rightEyebrowTop"), c.get("rightEyebrowBottom")));
        fill(g, Color.WHITE, c.get("leftEye"));
        fill(g, Color.WHITE, c.get("rightEye"));
        // 입술: 가운데 선으로 나눠 왼쪽·오른쪽 색을 따로 칠함
        List<double[]> lips = closed(c.get("upperLipTop"), c.get("lowerLipBottom"));
        java.awt.Shape clip = g.getClip();
        double[] mid = toPixel(new double[] {0, 1.0});
        g.setClip(0, 0, (int) Math.round(mid[0]), height);
        fill(g, lipLeft, lips);
        g.setClip((int) Math.round(mid[0]), 0, width, height);
        fill(g, lipRight, lips);
        g.setClip(clip);
        g.dispose();
        if (shadeRight > 0 || brightness != 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double k = brightness * (x >= centerX ? 1 - shadeRight : 1);
                    int rgb = img.getRGB(x, y);
                    int r = (int) Math.min(255, ((rgb >> 16) & 0xFF) * k);
                    int gg = (int) Math.min(255, ((rgb >> 8) & 0xFF) * k);
                    int b = (int) Math.min(255, (rgb & 0xFF) * k);
                    img.setRGB(x, y, (r << 16) | (gg << 8) | b);
                }
            }
        }
        return img;
    }

    public String jpegBase64() {
        return Base64.getEncoder().encodeToString(bytes("jpg"));
    }

    public byte[] bytes(String format) {
        try (var out = new ByteArrayOutputStream()) {
            ImageIO.write(image(), format, out);
            return out.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    // ---------- 점 만들기·바꾸기 ----------

    /** 눈썹 5점: 안쪽(가운데 가까이) → 바깥쪽 */
    static List<double[]> brow(int side, double y) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            double u = i / 4.0;
            double x = side * (0.15 + 0.7 * u);
            out.add(new double[] {x, y - 0.06 * Math.sin(Math.PI * u)});
        }
        return out;
    }

    /** 입술 선: 입꼬리(±0.45, 1.0)에서 가운데로 dy만큼 부풀림 */
    static List<double[]> lipLine(int n, double dy, boolean includeCorners) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double u = includeCorners ? i / (double) (n - 1) : (i + 1) / (double) (n + 1);
            double x = -0.45 + 0.9 * u;
            double bulge = 1 - Math.pow(x / 0.45, 2);
            out.add(new double[] {x, 1.0 + dy * bulge});
        }
        return out;
    }

    static List<double[]> ellipse(double cx, double cy, double rx, double ry, int n) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * i / n;
            out.add(new double[] {cx + rx * Math.cos(a), cy + ry * Math.sin(a)});
        }
        return out;
    }

    static List<double[]> closed(List<double[]> top, List<double[]> bottom) {
        List<double[]> out = new ArrayList<>(top.stream().sorted((a, b) -> Double.compare(a[0], b[0])).toList());
        out.addAll(bottom.stream().sorted((a, b) -> Double.compare(b[0], a[0])).toList());
        return out;
    }

    /** 점마다 (dx, dy)만큼 옮기기 */
    public static UnaryOperator<List<double[]>> shift(double dx, double dy) {
        return pts -> pts.stream().map(p -> new double[] {p[0] + dx, p[1] + dy}).toList();
    }

    /** x 방향으로 기준점(ax)에서 k배 늘이기·줄이기 */
    public static UnaryOperator<List<double[]>> scaleX(double ax, double k) {
        return pts -> pts.stream().map(p -> new double[] {ax + (p[0] - ax) * k, p[1]}).toList();
    }

    double[] toPixel(double[] p) {
        double a = Math.toRadians(rollDegrees);
        double x = p[0] * iod;
        double y = p[1] * iod;
        return new double[] {centerX + x * Math.cos(a) - y * Math.sin(a), centerY + x * Math.sin(a) + y * Math.cos(a)};
    }

    private void fill(Graphics2D g, Color color, List<double[]> pts) {
        Path2D path = new Path2D.Double();
        for (int i = 0; i < pts.size(); i++) {
            double[] px = toPixel(pts.get(i));
            if (i == 0) {
                path.moveTo(px[0], px[1]);
            } else {
                path.lineTo(px[0], px[1]);
            }
        }
        path.closePath();
        g.setColor(color);
        g.fill(path);
    }
}
