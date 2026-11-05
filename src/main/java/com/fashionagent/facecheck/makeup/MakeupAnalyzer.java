package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.image.Images;
import com.fashionagent.facecheck.image.Images.LabMean;
import com.fashionagent.facecheck.makeup.MakeupResult.Check;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * ML Kit 윤곽점 + 사진으로 화장 대칭을 계산합니다(모델 없이 규칙). 순서:
 * 좌우 정하기 → 구도(잘림·크기·정면) → 밝기 → 눈썹·입술 위치 → 볼·입술 색.
 * 왼쪽·오른쪽은 받은 사진 기준(셀카 화면 = 사용자 왼쪽)입니다.
 */
final class MakeupAnalyzer {

    static final String LEFT = "left";
    static final String RIGHT = "right";
    static final String PIXELS = "pixels";
    static final String LANDMARKS = "landmarks";
    static final String COLOR = "color";

    private final FaceCheckProperties.Symmetry t;

    MakeupAnalyzer(FaceCheckProperties.Symmetry thresholds) {
        this.t = thresholds;
    }

    MakeupResult analyze(FaceLandmarks lm, BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        Sides s = Sides.of(lm, w, h);
        FaceFrame frame = FaceFrame.of(Pt.mean(s.leftEye), Pt.mean(s.rightEye));

        String framing = framing(lm, s, frame, w, h);
        if (framing != null) {
            return MakeupResult.retake(framing, "ok");
        }
        if (faceLightness(s, frame, img) < t.darkLightness()) {
            return MakeupResult.retake(null, "dark");
        }

        List<Check> checks = new ArrayList<>();
        List<String> notes = new ArrayList<>();
        List<Note> found = new ArrayList<>();
        brows(s, frame, img, checks, found);
        lips(s, frame, img, checks, found);
        List<String> skipped = colors(s, frame, img, checks, found);

        found.sort(Comparator.comparingDouble(Note::severity).reversed());
        found.stream().limit(t.maxNotes()).forEach(n -> notes.add(n.text()));
        return new MakeupResult(null, "ok", new MakeupResult.Symmetry(notes.isEmpty(), notes, checks, skipped),
                MakeupResult.MODEL);
    }

    // ---------- 구도 ----------

    private String framing(FaceLandmarks lm, Sides s, FaceFrame frame, int w, int h) {
        // 얼굴 상자: FACE 윤곽이 있으면 그것, 없으면 눈썹·입술에서 어림(이마·턱까지 넓힘)
        List<Pt> box = new ArrayList<>();
        List<Pt> faceLocal = frame.local(s.face);
        if (s.face.size() >= 8) {
            box.addAll(s.face);
        } else {
            double top = min(frame.local(s.leftBrowTop), Pt::y) - 0.7;
            double bottom = max(frame.local(s.lowerLipBottom), Pt::y) + 0.55;
            for (double[] c : new double[][] {{-1.05, top}, {1.05, top}, {-1.05, bottom}, {1.05, bottom}}) {
                box.add(frame.pixel(new Pt(c[0], c[1])));
            }
        }
        double minX = min(box, Pt::x) / w;
        double maxX = max(box, Pt::x) / w;
        double minY = min(box, Pt::y) / h;
        double maxY = max(box, Pt::y) / h;
        double edge = 0.005;
        if (minY < edge) {
            return "얼굴 위쪽이 화면 밖으로 나갔어요. 폰을 조금 위로 올려 주세요.";
        }
        if (maxY > 1 - edge) {
            return "턱이 화면 밖으로 나갔어요. 폰을 조금 아래로 내려 주세요.";
        }
        // 셀카 화면(거울) 기준: 사진 왼쪽이 잘렸으면 폰을 사용자 왼쪽으로 옮기면 들어온다
        if (minX < edge) {
            return "얼굴 왼쪽이 화면 밖으로 나갔어요. 폰을 왼쪽으로 조금 옮겨 주세요.";
        }
        if (maxX > 1 - edge) {
            return "얼굴 오른쪽이 화면 밖으로 나갔어요. 폰을 오른쪽으로 조금 옮겨 주세요.";
        }
        double faceWidth = maxX - minX;
        if (faceWidth < 0.3 || frame.iod < 40) {
            return "얼굴이 작게 보여요. 폰을 얼굴에 조금 더 가까이 대 주세요.";
        }
        if (lm.yaw() != null ? Math.abs(lm.yaw()) > t.maxYaw() : skewed(faceLocal, s, frame)) {
            return "얼굴이 옆으로 돌아가 있어요. 폰을 코 앞에 두고 똑바로 봐 주세요.";
        }
        double roll = lm.roll() != null ? lm.roll() : frame.rollDegrees();
        if (Math.abs(roll) > t.maxRoll()) {
            return "고개가 한쪽으로 기울었어요. 고개를 똑바로 해 주세요.";
        }
        return null;
    }

    /** yaw 각도가 없을 때: 얼굴 외곽이나 코끝이 가운데 선에서 한쪽으로 쏠렸는지. */
    private boolean skewed(List<Pt> faceLocal, Sides s, FaceFrame frame) {
        if (faceLocal.size() >= 8) {
            double left = -min(faceLocal, Pt::x);
            double right = max(faceLocal, Pt::x);
            return left <= 0 || right <= 0 || Math.abs(left - right) / (left + right) > t.maxFaceSkew();
        }
        if (!s.noseBottom.isEmpty()) {
            return Math.abs(Pt.mean(frame.local(s.noseBottom)).x()) > 0.12;
        }
        return false;
    }

    private double faceLightness(Sides s, FaceFrame frame, BufferedImage img) {
        // 두 눈 아래 ~ 입 위, 볼까지 넓게: 머리카락·배경이 덜 섞이는 곳
        LabMean m = new LabMean();
        sampleRect(img, frame, -0.9, 0.2, 0.9, 0.9, m, 0, 100);
        return m.count() == 0 ? 0 : m.l();
    }

    // ---------- 눈썹 ----------

    private void brows(Sides s, FaceFrame f, BufferedImage img, List<Check> checks, List<Note> found) {
        List<Pt> lt = f.local(s.leftBrowTop);
        List<Pt> rt = f.local(s.rightBrowTop);
        List<Pt> lb = f.local(s.leftBrowBottom);
        List<Pt> rb = f.local(s.rightBrowBottom);
        boolean both = lb.size() >= 3 && rb.size() >= 3;
        double eyeL = width(f.local(s.leftEye));
        double eyeR = width(f.local(s.rightEye));
        List<Pt> face = f.local(s.face);

        // 먼저 사진에서 그려진 눈썹을 나눠 잼. 한쪽이라도 안 되면 둘 다 윤곽점으로
        MakeupPixels.Brow pl = MakeupPixels.brow(img, f, -1, concat(lt, lb), browPoly(lt, lb), face, eyeL);
        MakeupPixels.Brow pr = pl == null ? null : MakeupPixels.brow(img, f, 1, concat(rt, rb), browPoly(rt, rb), face, eyeR);
        MakeupPixels.Brow bl = pl != null && pr != null ? pl : landmarkBrow(lt, lb, both, -1, eyeL);
        MakeupPixels.Brow br = pl != null && pr != null ? pr : landmarkBrow(rt, rb, both, 1, eyeR);
        String basis = pl != null && pr != null ? PIXELS : LANDMARKS;

        Check height = check("browHeight", bl.center() - br.center(), t.browHeight(), basis, checks);
        if (!height.ok()) {
            found.add(new Note(height, side(height) + " 눈썹이 " + other(height) + "보다 조금 높게 그려졌어요."));
        }
        // 꼬리: 높이에서 이미 말했으면 겹쳐 말하지 않음
        Check tail = check("browTail", bl.tail() - br.tail(), t.browTail(), basis, checks);
        if (!tail.ok() && height.ok()) {
            found.add(new Note(tail, side(tail) + " 눈썹 꼬리가 " + other(tail) + "보다 올라가 있어요."));
        }
        Check length = check("browLength", ratio(bl.length(), br.length()), t.browLength(), basis, checks);
        if (!length.ok()) {
            String shorter = LEFT.equals(length.side()) ? RIGHT : LEFT;
            found.add(new Note(length, ko(shorter) + " 눈썹이 " + ko(opposite(shorter)) + "보다 조금 짧게 그려졌어요."));
        }
        if (bl.thickness() > 0 && br.thickness() > 0) {
            Check thick = check("browThickness", ratio(bl.thickness(), br.thickness()), t.browThickness(), basis, checks);
            if (!thick.ok()) {
                found.add(new Note(thick, side(thick) + " 눈썹이 " + other(thick) + "보다 두껍게 그려졌어요."));
            }
        }
    }

    /** 윤곽점만으로 잰 눈썹(사진에서 나누지 못했을 때). 위·아래 윤곽이 다 있으면 가운데 선과 두께까지. */
    private static MakeupPixels.Brow landmarkBrow(List<Pt> top, List<Pt> bottom, boolean both, int sgn, double eyeWidth) {
        double center = -meanY(both ? concat(top, bottom) : top);
        Pt tail = sgn < 0 ? top.stream().min(Comparator.comparingDouble(Pt::x)).orElseThrow()
                : top.stream().max(Comparator.comparingDouble(Pt::x)).orElseThrow();
        double length = (max(top, Pt::x) - min(top, Pt::x)) / (eyeWidth > 0 ? eyeWidth : 1);
        double thickness = both ? meanY(bottom) - meanY(top) : 0;
        return new MakeupPixels.Brow(center, -tail.y(), length, thickness);
    }

    private static List<Pt> browPoly(List<Pt> top, List<Pt> bottom) {
        List<Pt> low = bottom.size() >= 3 ? bottom : top.stream().map(p -> new Pt(p.x(), p.y() + 0.08)).toList();
        return MakeupAnalyzer.polygon(top, low);
    }

    private static double width(List<Pt> pts) {
        return pts.isEmpty() ? 0 : max(pts, Pt::x) - min(pts, Pt::x);
    }

    // ---------- 입술 ----------

    private void lips(Sides s, FaceFrame f, BufferedImage img, List<Check> checks, List<Note> found) {
        List<Pt> top = f.local(s.upperLipTop);
        List<Pt> bottom = f.local(s.lowerLipBottom);
        MakeupPixels.Lips px = MakeupPixels.lips(img, f, polygon(top, bottom));
        MakeupPixels.Lips lm = px != null ? px : landmarkLips(s, f, top, bottom);
        if (lm == null) {
            return; // 입이 가운데 선 한쪽에만 있으면(돌린 얼굴) 비교하지 않음
        }
        String basis = px != null ? PIXELS : LANDMARKS;

        Check corner = check("lipCorner", lm.corner(), t.lipCorner(), basis, checks);
        if (!corner.ok()) {
            found.add(new Note(corner, "입술 " + side(corner) + " 끝이 " + other(corner) + " 끝보다 올라가 있어요."));
        }
        Check width = check("lipHalfWidth", ratio(lm.halfLeft(), lm.halfRight()), t.lipHalfWidth(), basis, checks);
        if (!width.ok()) {
            found.add(new Note(width, "입술 선이 " + side(width) + "으로 더 길게 그려졌어요."));
        }
        if (lm.up() != null && lm.up()[0] > 0.02 && lm.up()[1] > 0.02) {
            Check c = check("upperLipLine", ratio(lm.up()[0], lm.up()[1]), t.lipLine(), basis, checks);
            if (!c.ok()) {
                found.add(new Note(c, "윗입술 선이 " + side(c) + "이 더 도톰하게 그려졌어요."));
            }
        }
        if (lm.down() != null && lm.down()[0] > 0.02 && lm.down()[1] > 0.02) {
            Check c = check("lowerLipLine", ratio(lm.down()[0], lm.down()[1]), t.lipLine(), basis, checks);
            if (!c.ok()) {
                found.add(new Note(c, "아랫입술 선이 " + side(c) + "이 더 도톰하게 그려졌어요."));
            }
        }
    }

    /** 윤곽점만으로 잰 입술(누드 립처럼 색으로 나눌 수 없을 때). */
    private static MakeupPixels.Lips landmarkLips(Sides s, FaceFrame f, List<Pt> top, List<Pt> bottom) {
        List<Pt> all = concat(concat(top, bottom), concat(f.local(s.upperLipBottom), f.local(s.lowerLipTop)));
        Pt cl = all.stream().min(Comparator.comparingDouble(Pt::x)).orElseThrow();
        Pt cr = all.stream().max(Comparator.comparingDouble(Pt::x)).orElseThrow();
        if (-cl.x() <= 0.05 || cr.x() <= 0.05) {
            return null;
        }
        return new MakeupPixels.Lips(-cl.x(), cr.x(), cr.y() - cl.y(), lineHeights(top, cl, cr, -1),
                lineHeights(bottom, cl, cr, 1));
    }

    /** 입꼬리 선에서 윤곽까지 높이의 좌·우 평균. dir -1이면 위쪽(윗입술), 1이면 아래쪽. */
    private static double[] lineHeights(List<Pt> contour, Pt cl, Pt cr, int dir) {
        double[] out = new double[2];
        for (int side = 0; side < 2; side++) {
            double half = side == 0 ? -cl.x() : cr.x();
            List<double[]> uv = new ArrayList<>();
            for (Pt p : contour) {
                boolean onSide = side == 0 ? p.x() < 0 : p.x() > 0;
                if (!onSide) {
                    continue;
                }
                double base = cl.y() + (p.x() - cl.x()) / (cr.x() - cl.x()) * (cr.y() - cl.y());
                uv.add(new double[] {Math.abs(p.x()) / half, dir * (p.y() - base)});
            }
            if (uv.size() < 2) {
                return null;
            }
            uv.sort(Comparator.comparingDouble(a -> a[0]));
            double sum = 0;
            int n = 0;
            for (double u = 0.15; u <= 0.86; u += 0.1) {
                sum += interpolate(uv, u);
                n++;
            }
            out[side] = sum / n;
        }
        return out;
    }

    private static double interpolate(List<double[]> uv, double u) {
        if (u <= uv.getFirst()[0]) {
            return uv.getFirst()[1];
        }
        for (int i = 1; i < uv.size(); i++) {
            double[] a = uv.get(i - 1);
            double[] b = uv.get(i);
            if (u <= b[0]) {
                return b[0] == a[0] ? b[1] : a[1] + (u - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
            }
        }
        return uv.getLast()[1];
    }

    // ---------- 색 ----------

    /** 볼·입술 색을 좌우로 비교합니다. 한쪽이 그늘지면 둘 다 건너뛰고 skipped에 남깁니다. */
    private List<String> colors(Sides s, FaceFrame f, BufferedImage img, List<Check> checks, List<Note> found) {
        Pt cheekL = s.leftCheek.isEmpty() ? new Pt(-0.55, 0.55) : f.local(s.leftCheek.getFirst());
        Pt cheekR = s.rightCheek.isEmpty() ? new Pt(0.55, 0.55) : f.local(s.rightCheek.getFirst());
        LabMean ml = new LabMean();
        LabMean mr = new LabMean();
        sampleDisc(img, f, cheekL, 0.14, ml);
        sampleDisc(img, f, cheekR, 0.14, mr);
        if (ml.count() < 20 || mr.count() < 20) {
            return List.of("cheekColor", "lipColor");
        }
        if (Math.abs(ml.l() - mr.l()) > t.unevenLight()) {
            return List.of("cheekColor", "lipColor");
        }
        Check cheek = check("cheekColor", ml.a() - mr.a(), t.cheekRedness(), COLOR, checks);
        if (!cheek.ok()) {
            found.add(new Note(cheek, side(cheek) + " 볼이 " + other(cheek) + "보다 더 붉어요. 블러셔가 한쪽에 더 발린 것 같아요."));
        }

        LabMean ll = new LabMean();
        LabMean lr = new LabMean();
        sampleLips(img, f, s, ll, lr);
        if (ll.count() < 20 || lr.count() < 20) {
            return List.of("lipColor");
        }
        double de = ll.distance(lr);
        // 더 연한(밝은) 쪽을 side로
        String lighter = ll.l() >= lr.l() ? LEFT : RIGHT;
        boolean ok = de <= t.lipColor();
        Check lip = new Check("lipColor", lighter, round(de), t.lipColor(), ok, COLOR);
        checks.add(lip);
        if (!ok) {
            found.add(new Note(lip, "입술 색이 " + ko(lighter) + "이 " + ko(opposite(lighter)) + "보다 연해요."));
        }
        return List.of();
    }

    private static void sampleDisc(BufferedImage img, FaceFrame f, Pt centerLocal, double radius, LabMean out) {
        Pt c = f.pixel(centerLocal);
        double r = radius * f.iod;
        int x0 = (int) Math.max(0, Math.floor(c.x() - r));
        int x1 = (int) Math.min(img.getWidth() - 1, Math.ceil(c.x() + r));
        int y0 = (int) Math.max(0, Math.floor(c.y() - r));
        int y1 = (int) Math.min(img.getHeight() - 1, Math.ceil(c.y() + r));
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                if (Math.hypot(x + 0.5 - c.x(), y + 0.5 - c.y()) <= r) {
                    addIf(out, Images.lab(img.getRGB(x, y)), 15, 97);
                }
            }
        }
    }

    private static void sampleRect(BufferedImage img, FaceFrame f, double lx0, double ly0, double lx1, double ly1,
            LabMean out, double minL, double maxL) {
        List<Pt> corners = List.of(f.pixel(new Pt(lx0, ly0)), f.pixel(new Pt(lx1, ly0)),
                f.pixel(new Pt(lx0, ly1)), f.pixel(new Pt(lx1, ly1)));
        int x0 = clamp(min(corners, Pt::x), img.getWidth());
        int x1 = clamp(max(corners, Pt::x), img.getWidth());
        int y0 = clamp(min(corners, Pt::y), img.getHeight());
        int y1 = clamp(max(corners, Pt::y), img.getHeight());
        int step = Math.max(1, (int) (f.iod / 40));
        for (int y = y0; y <= y1; y += step) {
            for (int x = x0; x <= x1; x += step) {
                Pt l = f.local(new Pt(x + 0.5, y + 0.5));
                if (l.x() >= lx0 && l.x() <= lx1 && l.y() >= ly0 && l.y() <= ly1) {
                    addIf(out, Images.lab(img.getRGB(x, y)), minL, maxL);
                }
            }
        }
    }

    /** 입술 바깥 윤곽 안(벌린 입 안쪽은 뺌)을 가운데 선 기준으로 좌우에 나눠 담습니다. */
    private static void sampleLips(BufferedImage img, FaceFrame f, Sides s, LabMean left, LabMean right) {
        List<Pt> outer = polygon(f.local(s.upperLipTop), f.local(s.lowerLipBottom));
        List<Pt> inner = s.upperLipBottom.size() >= 3 && s.lowerLipTop.size() >= 3
                ? polygon(f.local(s.upperLipBottom), f.local(s.lowerLipTop)) : List.of();
        List<Pt> px = outer.stream().map(f::pixel).toList();
        int x0 = clamp(min(px, Pt::x), img.getWidth());
        int x1 = clamp(max(px, Pt::x), img.getWidth());
        int y0 = clamp(min(px, Pt::y), img.getHeight());
        int y1 = clamp(max(px, Pt::y), img.getHeight());
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                Pt l = f.local(new Pt(x + 0.5, y + 0.5));
                if (Math.abs(l.x()) < 0.04 || !inside(outer, l) || (!inner.isEmpty() && inside(inner, l))) {
                    continue;
                }
                addIf(l.x() < 0 ? left : right, Images.lab(img.getRGB(x, y)), 10, 97);
            }
        }
    }

    /** 위 윤곽(왼→오) + 아래 윤곽(오→왼)으로 닫힌 다각형. */
    static List<Pt> polygon(List<Pt> upper, List<Pt> lower) {
        List<Pt> out = new ArrayList<>(upper.stream().sorted(Comparator.comparingDouble(Pt::x)).toList());
        out.addAll(lower.stream().sorted(Comparator.comparingDouble(Pt::x).reversed()).toList());
        return out;
    }

    private static boolean inside(List<Pt> poly, Pt p) {
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

    private static void addIf(LabMean m, double[] lab, double minL, double maxL) {
        // 그림자·머리카락(너무 어두움)과 번들거림(너무 밝음)은 뺌
        if (lab[0] >= minL && lab[0] <= maxL) {
            m.add(lab);
        }
    }

    // ---------- 공통 ----------

    /** diff > 0이면 left가 더 큼. */
    private static Check check(String name, double diff, double limit, String basis, List<Check> checks) {
        double value = round(Math.abs(diff));
        String side = value == 0 ? null : diff > 0 ? LEFT : RIGHT;
        Check c = new Check(name, side, value, limit, Math.abs(diff) <= limit, basis);
        checks.add(c);
        return c;
    }

    /** (left - right) / 큰 쪽. 부호는 check()에서 큰 쪽을 정하는 데 씀. */
    private static double ratio(double left, double right) {
        double big = Math.max(Math.abs(left), Math.abs(right));
        return big == 0 ? 0 : (left - right) / big;
    }

    private static String side(Check c) {
        return ko(c.side());
    }

    private static String other(Check c) {
        return ko(opposite(c.side()));
    }

    static String ko(String side) {
        return LEFT.equals(side) ? "왼쪽" : "오른쪽";
    }

    private static String opposite(String side) {
        return LEFT.equals(side) ? RIGHT : LEFT;
    }

    private static double round(double v) {
        return Math.round(v * 1000) / 1000.0;
    }

    private static double meanY(List<Pt> pts) {
        return pts.stream().mapToDouble(Pt::y).average().orElse(0);
    }

    private static double min(List<Pt> pts, java.util.function.ToDoubleFunction<Pt> fn) {
        return pts.stream().mapToDouble(fn).min().orElse(0);
    }

    private static double max(List<Pt> pts, java.util.function.ToDoubleFunction<Pt> fn) {
        return pts.stream().mapToDouble(fn).max().orElse(0);
    }

    private static int clamp(double v, int size) {
        return (int) Math.max(0, Math.min(size - 1, Math.round(v)));
    }

    private static List<Pt> concat(List<Pt> a, List<Pt> b) {
        List<Pt> out = new ArrayList<>(a);
        out.addAll(b);
        return out;
    }

    private record Note(Check check, String text) {
        double severity() {
            return check.limit() == 0 ? check.value() : check.value() / check.limit();
        }
    }

    /** 픽셀 좌표로 바꾸고, 두 눈의 x 위치로 왼쪽·오른쪽을 다시 정한 윤곽들. */
    private record Sides(List<Pt> face, List<Pt> leftEye, List<Pt> rightEye, List<Pt> leftBrowTop,
            List<Pt> leftBrowBottom, List<Pt> rightBrowTop, List<Pt> rightBrowBottom, List<Pt> upperLipTop,
            List<Pt> upperLipBottom, List<Pt> lowerLipTop, List<Pt> lowerLipBottom, List<Pt> noseBottom,
            List<Pt> leftCheek, List<Pt> rightCheek) {

        static Sides of(FaceLandmarks lm, int w, int h) {
            List<Pt> le = Pt.pixels(lm.leftEye(), w, h);
            List<Pt> re = Pt.pixels(lm.rightEye(), w, h);
            boolean swap = Pt.mean(le).x() > Pt.mean(re).x();
            List<Pt> lbt = Pt.pixels(lm.leftEyebrowTop(), w, h);
            List<Pt> lbb = Pt.pixels(lm.leftEyebrowBottom(), w, h);
            List<Pt> rbt = Pt.pixels(lm.rightEyebrowTop(), w, h);
            List<Pt> rbb = Pt.pixels(lm.rightEyebrowBottom(), w, h);
            List<Pt> lc = Pt.pixels(lm.leftCheek(), w, h);
            List<Pt> rc = Pt.pixels(lm.rightCheek(), w, h);
            return new Sides(Pt.pixels(lm.face(), w, h),
                    swap ? re : le, swap ? le : re,
                    swap ? rbt : lbt, swap ? rbb : lbb, swap ? lbt : rbt, swap ? lbb : rbb,
                    Pt.pixels(lm.upperLipTop(), w, h), Pt.pixels(lm.upperLipBottom(), w, h),
                    Pt.pixels(lm.lowerLipTop(), w, h), Pt.pixels(lm.lowerLipBottom(), w, h),
                    Pt.pixels(lm.noseBottom(), w, h),
                    swap ? rc : lc, swap ? lc : rc);
        }
    }
}
