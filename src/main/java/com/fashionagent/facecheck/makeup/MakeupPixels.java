package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.image.Images;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 실제로 그려진 눈썹·입술 모양을 픽셀로 잽니다. 윤곽점(ML Kit·MediaPipe)은 얼굴 뼈대를 따라가서 그린 화장선을 거의 반영하지
 * 않으므로(비교 측정: 눈썹을 IOD 0.12 올려 그려도 윤곽점은 0.05만 움직임, 입술 선을 넘게 그린 건 0), 윤곽점은 어디를 볼지만
 * 정하고 모양은 색으로 나눈 픽셀에서 잽니다. 나누기가 안 되면(눈썹이 아주 옅음, 누드 립 등) null을 돌려주고 윤곽점으로 잽니다.
 */
final class MakeupPixels {

    /** 눈썹 픽셀: 주변 피부보다 L*가 이만큼 어두운 것 */
    static final double BROW_DARKER = 12;
    /** 윤곽점 눈썹 안쪽 끝·바깥 끝에서 이만큼만 더 봄(관자놀이 머리카락, 미간 그림자를 뺌) */
    static final double BROW_X_MARGIN_IN = 0.08;
    static final double BROW_X_MARGIN_OUT = 0.10;
    /** 입술과 주변 피부의 a*(붉은 정도) 차이가 이보다 작으면 입술 선을 색으로 나눌 수 없음 */
    static final double LIP_MIN_CONTRAST = 4;

    private MakeupPixels() {
    }

    /** 눈썹 한쪽의 모양. 길이는 같은 쪽 눈 너비로 나눈 값(고개가 조금 돌아도 원근 차이가 상쇄됨). */
    record Brow(double center, double tail, double length, double thickness) {
    }

    /** 입술 모양. corner = 오른쪽 입꼬리 y - 왼쪽 입꼬리 y (+면 왼쪽이 더 높음). */
    record Lips(double halfLeft, double halfRight, double corner, double[] up, double[] down, Double upRatio,
            Double downRatio) {
        Lips(double halfLeft, double halfRight, double corner, double[] up, double[] down) {
            this(halfLeft, halfRight, corner, up, down, null, null);
        }
    }

    /**
     * @param sgn       -1 왼쪽, 1 오른쪽
     * @param browLocal 그쪽 눈썹 윤곽점(위+아래, 얼굴 좌표)
     * @param browPoly  그쪽 눈썹 다각형(위 윤곽 + 아래 윤곽 거꾸로). 아래 윤곽이 없으면 위 윤곽을 아래로 0.08 내린 띠
     * @param face      얼굴 외곽(얼굴 좌표), 없으면 빈 목록
     * @param eyeWidth  같은 쪽 눈 너비(얼굴 좌표)
     */
    static Brow brow(BufferedImage img, FaceFrame f, int sgn, List<Pt> browLocal, List<Pt> browPoly, List<Pt> face,
            double eyeWidth) {
        double cy = browLocal.stream().mapToDouble(Pt::y).average().orElse(-0.35);
        double x0 = sgn < 0 ? -1.15 : 0.05;
        double x1 = sgn < 0 ? -0.05 : 1.15;
        double y0 = cy - 0.3;
        double y1 = Math.min(cy + 0.22, -0.12);
        List<Pt> faceIn = shrink(face, 0.94);
        Region r = Region.of(img, f, x0, y0, x1, y1, p -> faceIn.size() < 8 || inside(faceIn, p));
        if (r.count < 50) {
            return null;
        }
        // 피부 밝기는 가로 0.1 칸마다 따로 구함: 한쪽에서 빛이 들어와 띠 안에서 밝기가 변해도 견딤
        int bins = 12;
        List<List<Double>> perBin = new ArrayList<>();
        for (int b = 0; b < bins; b++) {
            perBin.add(new ArrayList<>());
        }
        for (int i = 0; i < r.in.length; i++) {
            if (r.in[i]) {
                perBin.get(bin(r.lx[i], x0, bins)).add(r.lab[i * 3]);
            }
        }
        // 중앙값 대신 75% 값: 띠 한 칸을 눈썹이 절반 가까이 덮어도 피부 밝기가 나옴
        double all = upper(r.values(0));
        double[] skinL = new double[bins];
        for (int b = 0; b < bins; b++) {
            skinL[b] = perBin.get(b).size() >= 20 ? upper(perBin.get(b)) : all;
        }
        boolean[] mask = new boolean[r.w * r.h];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = r.in[i] && r.lab[i * 3] < skinL[bin(r.lx[i], x0, bins)] - BROW_DARKER;
        }
        mask = open(mask, r.w, r.h);
        // 가로는 윤곽점 눈썹 양 끝에서 조금만 넘게 봄: 관자놀이 머리카락, 미간 그림자를 뺌
        double lo = browLocal.stream().mapToDouble(p -> p.x() * sgn).min().orElse(0) - BROW_X_MARGIN_IN;
        double hi = browLocal.stream().mapToDouble(p -> p.x() * sgn).max().orElse(9) + BROW_X_MARGIN_OUT;
        for (int i = 0; i < mask.length; i++) {
            double x = r.lx[i] * sgn;
            if (mask[i] && (x < lo || x > hi)) {
                mask[i] = false;
            }
        }
        // 윤곽점이 그린 눈썹을 덜 따라가므로(높이 0.12를 0.05만 따라감) 겹침 대신
        // "크고 윤곽점 눈썹 가운데에 가까운, 가로로 긴" 덩어리를 고름. 머리카락은 크지만 멀거나 세로로 서 있고, 지운 자국은 작음
        Pt target = Pt.mean(browLocal);
        int[] comp = browComponent(mask, r, target);
        if (comp.length < 30) {
            return null;
        }
        double[] xs = new double[comp.length];
        double[] ys = new double[comp.length];
        for (int k = 0; k < comp.length; k++) {
            xs[k] = r.lx[comp[k]] * sgn;
            ys[k] = r.ly[comp[k]];
        }
        double inner = percentile(xs, 2);
        double outer = percentile(xs, 98);
        if (outer - inner < 0.15 || eyeWidth <= 0) {
            return null;
        }
        double tailSum = 0;
        int tailN = 0;
        double ySum = 0;
        for (int k = 0; k < xs.length; k++) {
            ySum += ys[k];
            if (xs[k] > outer - 0.2 * (outer - inner)) {
                tailSum += ys[k];
                tailN++;
            }
        }
        double area = comp.length / (f.iod * f.iod / (r.step * r.step)); // 얼굴 좌표 넓이
        return new Brow(-ySum / xs.length, -tailSum / Math.max(1, tailN), (outer - inner) / eyeWidth,
                area / (outer - inner));
    }

    /** @param outer 입술 바깥 다각형(얼굴 좌표) */
    static Lips lips(BufferedImage img, FaceFrame f, List<Pt> outer, List<Pt> innerTop, List<Pt> innerBottom) {
        double pad = 0.125;
        double x0 = outer.stream().mapToDouble(Pt::x).min().orElse(0) - pad;
        double x1 = outer.stream().mapToDouble(Pt::x).max().orElse(0) + pad;
        double y0 = outer.stream().mapToDouble(Pt::y).min().orElse(0) - pad;
        double y1 = outer.stream().mapToDouble(Pt::y).max().orElse(0) + pad;
        Region r = Region.of(img, f, x0, y0, x1, y1, p -> inside(outer, p) || distance(outer, p) < pad);
        List<Double> lipA = new ArrayList<>();
        List<Double> skinA = new ArrayList<>();
        boolean[] in = new boolean[r.w * r.h];
        for (int i = 0; i < in.length; i++) {
            if (!r.in[i]) {
                continue;
            }
            in[i] = inside(outer, new Pt(r.lx[i], r.ly[i]));
            (in[i] ? lipA : skinA).add(r.lab[i * 3 + 1]);
        }
        if (lipA.size() < 30 || skinA.size() < 30) {
            return null;
        }
        double la = median(lipA);
        double sa = median(skinA);
        if (la - sa < LIP_MIN_CONTRAST) {
            return null;
        }
        double cut = (la + sa) / 2;
        boolean[] mask = new boolean[r.w * r.h];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = r.in[i] && r.lab[i * 3 + 1] > cut;
        }
        mask = open(mask, r.w, r.h);
        int[] comp = largestOverlapping(mask, r, outer);
        if (comp.length < 50) {
            return null;
        }
        double[] xs = new double[comp.length];
        double[] ys = new double[comp.length];
        for (int k = 0; k < comp.length; k++) {
            xs[k] = r.lx[comp[k]];
            ys[k] = r.ly[comp[k]];
        }
        double cl = percentile(xs, 1);
        double cr = percentile(xs, 99);
        if (-cl <= 0.05 || cr <= 0.05) {
            return null;
        }
        double ycl = meanWhere(xs, ys, x -> x < cl + 0.05 * (cr - cl));
        double ycr = meanWhere(xs, ys, x -> x > cr - 0.05 * (cr - cl));
        double[] up = new double[2];
        double[] down = new double[2];
        for (int side = 0; side < 2; side++) {
            double half = side == 0 ? -cl : cr;
            double sgn = side == 0 ? -1 : 1;
            double upSum = 0;
            double downSum = 0;
            int n = 0;
            for (double u = 0.15; u <= 0.86; u += 0.1) {
                double x = sgn * u * half;
                double base = ycl + (x - cl) / (cr - cl) * (ycr - ycl);
                double top = Double.MAX_VALUE;
                double bottom = -Double.MAX_VALUE;
                int c = 0;
                for (int k = 0; k < xs.length; k++) {
                    if (Math.abs(xs[k] - x) < 0.02) {
                        top = Math.min(top, ys[k]);
                        bottom = Math.max(bottom, ys[k]);
                        c++;
                    }
                }
                if (c >= 3) {
                    upSum += base - top;
                    downSum += bottom - base;
                    n++;
                }
            }
            if (n < 4) {
                return null;
            }
            up[side] = upSum / n;
            down[side] = downSum / n;
        }
        Double upRatio = innerTop.size() >= 3 ? robustRatio(xs, ys, cl, cr, innerTop, -1) : null;
        Double downRatio = innerBottom.size() >= 3 ? robustRatio(xs, ys, cl, cr, innerBottom, 1) : null;
        return new Lips(-cl, cr, ycr - ycl, up, down, upRatio, downRatio);
    }

    /**
     * 입술 선 좌우 비교: 입술 픽셀의 위(아래) 끝에서 윤곽점 안쪽 선(윗입술 아래 선, 아랫입술 위 선)까지 두께를,
     * 가운데 선에서 같은 거리(u)의 왼쪽·오른쪽끼리 비율로 내고 그 가운데 값을 씁니다. 입꼬리 선에서 재던 방식보다
     * 입을 조금 벌리거나 입꼬리를 잘못 잡아도 덜 흔들리고, 열마다 끝은 10% 값이라 튀는 픽셀에도 덜 흔들립니다
     * (CelebA 비교: 윗입술 선을 넘겨 그린 것 검출 23% → 60%, training/public-baseline.md). 못 재면 null.
     */
    private static Double robustRatio(double[] xs, double[] ys, double cl, double cr, List<Pt> inner, int dir) {
        List<Pt> line = inner.stream().sorted(java.util.Comparator.comparingDouble(Pt::x)).toList();
        List<Double> ratios = new ArrayList<>();
        for (double u = 0.15; u <= 0.76; u += 0.05) {
            double[] th = new double[2];
            boolean ok = true;
            for (int side = 0; side < 2 && ok; side++) {
                double x = (side == 0 ? -1 : 1) * u * (side == 0 ? -cl : cr);
                if (x < line.getFirst().x() || x > line.getLast().x()) {
                    ok = false;
                    break;
                }
                List<Double> col = new ArrayList<>();
                for (int k = 0; k < xs.length; k++) {
                    if (Math.abs(xs[k] - x) < 0.015) {
                        col.add(ys[k]);
                    }
                }
                if (col.size() < 4) {
                    ok = false;
                    break;
                }
                double[] c = col.stream().mapToDouble(Double::doubleValue).toArray();
                double edge = percentile(c, dir < 0 ? 10 : 90);
                double base = lineY(line, x);
                th[side] = dir < 0 ? base - edge : edge - base;
                if (th[side] <= 0.01) {
                    ok = false;
                }
            }
            if (ok) {
                ratios.add((th[0] - th[1]) / Math.max(th[0], th[1]));
            }
        }
        return ratios.size() < 4 ? null : median(ratios);
    }

    private static double lineY(List<Pt> line, double x) {
        for (int i = 1; i < line.size(); i++) {
            Pt a = line.get(i - 1);
            Pt b = line.get(i);
            if (x <= b.x()) {
                return b.x() == a.x() ? b.y() : a.y() + (x - a.x()) / (b.x() - a.x()) * (b.y() - a.y());
            }
        }
        return line.getLast().y();
    }

    private static int bin(double x, double x0, int bins) {
        return (int) Math.max(0, Math.min(bins - 1, Math.floor((x - x0) / 0.1)));
    }

    // ---------- 영역·마스크 ----------

    /** 얼굴 좌표 사각형을 덮는 픽셀 격자(큰 사진은 건너뛰며 봄). 각 칸의 얼굴 좌표와 Lab을 미리 계산. */
    private static final class Region {
        int w;
        int h;
        int step;
        int count;
        boolean[] in;
        double[] lx;
        double[] ly;
        double[] lab;

        static Region of(BufferedImage img, FaceFrame f, double x0, double y0, double x1, double y1,
                java.util.function.Predicate<Pt> keep) {
            List<Pt> corners = List.of(f.pixel(new Pt(x0, y0)), f.pixel(new Pt(x1, y0)), f.pixel(new Pt(x0, y1)),
                    f.pixel(new Pt(x1, y1)));
            int px0 = (int) Math.max(0, corners.stream().mapToDouble(Pt::x).min().orElse(0));
            int px1 = (int) Math.min(img.getWidth() - 1, corners.stream().mapToDouble(Pt::x).max().orElse(0));
            int py0 = (int) Math.max(0, corners.stream().mapToDouble(Pt::y).min().orElse(0));
            int py1 = (int) Math.min(img.getHeight() - 1, corners.stream().mapToDouble(Pt::y).max().orElse(0));
            Region r = new Region();
            // 두 눈 사이가 150픽셀쯤 되게 건너뜀: 큰 사진도 계산량이 같음
            r.step = Math.max(1, (int) Math.floor(f.iod / 150));
            r.w = Math.max(0, (px1 - px0) / r.step + 1);
            r.h = Math.max(0, (py1 - py0) / r.step + 1);
            int n = r.w * r.h;
            r.in = new boolean[n];
            r.lx = new double[n];
            r.ly = new double[n];
            r.lab = new double[n * 3];
            for (int j = 0; j < r.h; j++) {
                for (int i = 0; i < r.w; i++) {
                    int x = px0 + i * r.step;
                    int y = py0 + j * r.step;
                    int k = j * r.w + i;
                    Pt l = f.local(new Pt(x + 0.5, y + 0.5));
                    r.lx[k] = l.x();
                    r.ly[k] = l.y();
                    if (l.x() < x0 || l.x() > x1 || l.y() < y0 || l.y() > y1 || !keep.test(l)) {
                        continue;
                    }
                    double[] c = Images.lab(img.getRGB(x, y));
                    System.arraycopy(c, 0, r.lab, k * 3, 3);
                    r.in[k] = true;
                    r.count++;
                }
            }
            return r;
        }

        List<Double> values(int channel) {
            List<Double> out = new ArrayList<>();
            for (int i = 0; i < in.length; i++) {
                if (in[i]) {
                    out.add(lab[i * 3 + channel]);
                }
            }
            return out;
        }
    }

    /** 3x3 열기(깎고 다시 불림): 점 같은 잡음 제거 */
    static boolean[] open(boolean[] m, int w, int h) {
        return morph(morph(m, w, h, true), w, h, false);
    }

    private static boolean[] morph(boolean[] m, int w, int h, boolean erode) {
        boolean[] out = new boolean[m.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean v = erode;
                for (int dy = -1; dy <= 1 && v == erode; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = x + dx;
                        int yy = y + dy;
                        boolean p = xx >= 0 && yy >= 0 && xx < w && yy < h && m[yy * w + xx];
                        if (erode ? !p : p) {
                            v = !erode;
                            break;
                        }
                    }
                }
                out[y * w + x] = v;
            }
        }
        return out;
    }


    /** 이어진 덩어리(4방향)마다 칸 목록을 만든다. */
    private static List<int[]> components(boolean[] mask, Region r) {
        boolean[] seen = new boolean[mask.length];
        List<int[]> out = new ArrayList<>();
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        for (int s = 0; s < mask.length; s++) {
            if (!mask[s] || seen[s]) {
                continue;
            }
            List<Integer> cells = new ArrayList<>();
            seen[s] = true;
            queue.add(s);
            while (!queue.isEmpty()) {
                int c = queue.poll();
                cells.add(c);
                int x = c % r.w;
                int y = c / r.w;
                int[][] nb = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
                for (int[] p : nb) {
                    if (p[0] >= 0 && p[1] >= 0 && p[0] < r.w && p[1] < r.h) {
                        int q = p[1] * r.w + p[0];
                        if (mask[q] && !seen[q]) {
                            seen[q] = true;
                            queue.add(q);
                        }
                    }
                }
            }
            out.add(cells.stream().mapToInt(Integer::intValue).toArray());
        }
        return out;
    }

    /** 점수가 가장 높은 덩어리. 없으면 빈 배열. */
    /**
     * 눈썹 덩어리 고르기. 점수 = 크기 × 윤곽점 눈썹 가운데와 가까움 × 가로로 긴 정도. 관자놀이 머리카락처럼 세로로 선 덩어리는
     * 크고 가까워도 눈썹이 아님. 고른 덩어리와 같은 높이에서 가로로 이어지는 덩어리(가운데가 옅어 끊긴 눈썹)는 합침.
     */
    private static int[] browComponent(boolean[] mask, Region r, Pt target) {
        List<int[]> comps = components(mask, r);
        double[][] st = new double[comps.size()][];
        int bestI = -1;
        double bestScore = 0;
        for (int i = 0; i < comps.size(); i++) {
            int[] c = comps.get(i);
            double cx = 0, cy = 0, x0 = 9, x1 = -9, y0 = 9, y1 = -9;
            for (int k : c) {
                double x = r.lx[k], y = r.ly[k];
                cx += x;
                cy += y;
                x0 = Math.min(x0, x);
                x1 = Math.max(x1, x);
                y0 = Math.min(y0, y);
                y1 = Math.max(y1, y);
            }
            cx /= c.length;
            cy /= c.length;
            double w = x1 - x0 + 0.01;
            double h = y1 - y0 + 0.01;
            double shape = Math.min(1, (w / h) / 2.5);
            st[i] = new double[] {cx, cy, x0, x1, w / h};
            double score = c.length * Math.exp(-Math.pow(Math.hypot(cx - target.x(), cy - target.y()) / 0.15, 2))
                    * shape * shape;
            if (score > bestScore) {
                bestScore = score;
                bestI = i;
            }
        }
        if (bestI < 0) {
            return new int[0];
        }
        double[] b = st[bestI];
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < comps.size(); i++) {
            double[] o = st[i];
            boolean join = i == bestI || (comps.get(i).length >= 8 && o[4] >= 1.5 && Math.abs(o[1] - b[1]) < 0.06
                    && Math.max(o[2] - b[3], b[2] - o[3]) < 0.08);
            if (join) {
                for (int k : comps.get(i)) {
                    out.add(k);
                }
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** 윤곽점 다각형과 가장 많이 겹치는 덩어리. */
    private static int[] largestOverlapping(boolean[] mask, Region r, List<Pt> poly) {
        int[] best = new int[0];
        int bestOverlap = 0;
        for (int[] c : components(mask, r)) {
            int overlap = 0;
            for (int k : c) {
                if (inside(poly, new Pt(r.lx[k], r.ly[k]))) {
                    overlap++;
                }
            }
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                best = c;
            }
        }
        return best;
    }

    // ---------- 도형·통계 ----------

    static boolean inside(List<Pt> poly, Pt p) {
        boolean in = false;
        for (int i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
            Pt a = poly.get(i);
            Pt b = poly.get(j);
            if ((a.y() > p.y()) != (b.y() > p.y())
                    && p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x()) {
                in = !in;
            }
        }
        return in;
    }

    static double distance(List<Pt> poly, Pt p) {
        double best = Double.MAX_VALUE;
        for (int i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
            Pt a = poly.get(j);
            Pt b = poly.get(i);
            double dx = b.x() - a.x();
            double dy = b.y() - a.y();
            double len = dx * dx + dy * dy;
            double t = len == 0 ? 0 : Math.max(0, Math.min(1, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len));
            best = Math.min(best, Math.hypot(p.x() - a.x() - t * dx, p.y() - a.y() - t * dy));
        }
        return best;
    }

    private static List<Pt> shrink(List<Pt> poly, double k) {
        if (poly.isEmpty()) {
            return poly;
        }
        Pt c = Pt.mean(poly);
        return poly.stream().map(p -> new Pt(c.x() + (p.x() - c.x()) * k, c.y() + (p.y() - c.y()) * k)).toList();
    }

    private static double median(List<Double> v) {
        double[] a = v.stream().mapToDouble(Double::doubleValue).toArray();
        Arrays.sort(a);
        return a.length == 0 ? 0 : a[a.length / 2];
    }

    private static double upper(List<Double> v) {
        return percentile(v.stream().mapToDouble(Double::doubleValue).toArray(), 75);
    }

    private static double percentile(double[] v, double p) {
        double[] a = v.clone();
        Arrays.sort(a);
        return a[(int) Math.min(a.length - 1, Math.max(0, Math.round(p / 100 * (a.length - 1))))];
    }

    private static double meanWhere(double[] xs, double[] ys, java.util.function.DoublePredicate keep) {
        double s = 0;
        int n = 0;
        for (int k = 0; k < xs.length; k++) {
            if (keep.test(xs[k])) {
                s += ys[k];
                n++;
            }
        }
        return n == 0 ? 0 : s / n;
    }
}
