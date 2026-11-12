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
    private final FaceParser parser;
    private final PairModels models;

    MakeupAnalyzer(FaceCheckProperties.Symmetry thresholds) {
        this(thresholds, FaceParser.NONE);
    }

    MakeupAnalyzer(FaceCheckProperties.Symmetry thresholds, FaceParser parser) {
        this(thresholds, parser, PairModels.NONE);
    }

    MakeupAnalyzer(FaceCheckProperties.Symmetry thresholds, FaceParser parser, PairModels models) {
        this.t = thresholds;
        this.parser = parser == null ? FaceParser.NONE : parser;
        this.models = models == null ? PairModels.NONE : models;
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
        FaceLabels labels = parser.ready() ? parser.parse(img, frame) : null;
        // 두께 모델이 있으면 뒤집어 한 번 더 나누기(두께에만 씀, 약 0.1초)는 하지 않음
        Double thickModel = models.browThickness().score(img, frame);
        FaceLabels mirrored = labels != null && thickModel == null ? parser.parseMirrored(img, frame) : null;
        boolean lengthSkipped = brows(s, frame, img, labels, mirrored, thickModel, checks);
        lips(s, frame, img, labels, checks);
        List<String> skipped = new ArrayList<>(colors(s, frame, img, checks));
        skipped.addAll(eyes(s, frame, img, checks));
        if (lengthSkipped) {
            skipped.add("browLength");
        }
        return result(checks, List.copyOf(skipped), t.maxNotes());
    }

    /** 잰 항목들로 결과를 만듭니다: 기준을 넘은 항목을 (차이 ÷ 기준) 큰 순서로 최대 maxNotes 문장. */
    static MakeupResult result(List<Check> checks, List<String> skipped, int maxNotes) {
        boolean heightOk = checks.stream().noneMatch(c -> c.name().equals("browHeight") && !c.ok());
        List<String> notes = checks.stream()
                .filter(c -> !c.ok() && c.side() != null)
                .filter(c -> !(c.name().equals("browTail") && !heightOk)) // 높이에서 말했으면 꼬리는 겹쳐 말하지 않음
                // 길이 모델은 한쪽 눈썹을 올려 그린 것도 길이 차이로 볼 때가 있어서(CelebA 21%) 높이에서 말했으면 길이는 말하지 않음
                .filter(c -> !(c.name().equals("browLength") && !heightOk))
                .sorted(Comparator.comparingDouble((Check c) -> c.limit() == 0 ? c.value() : c.value() / c.limit()).reversed())
                .limit(maxNotes)
                .map(MakeupAnalyzer::note)
                .toList();
        return new MakeupResult(null, "ok", new MakeupResult.Symmetry(notes.isEmpty(), notes, checks, skipped));
    }

    /** 항목 하나를 읽어 줄 문장. side는 더 높은·긴·두꺼운·붉은·연한 쪽. */
    static String note(Check c) {
        String a = ko(c.side());
        String b = ko(opposite(c.side()));
        return switch (c.name()) {
            case "browHeight" -> a + " 눈썹이 " + b + "보다 조금 높게 그려졌어요.";
            case "browTail" -> a + " 눈썹 꼬리가 " + b + "보다 올라가 있어요.";
            case "browLength" -> b + " 눈썹이 " + a + "보다 조금 짧게 그려졌어요.";
            case "browThickness" -> a + " 눈썹이 " + b + "보다 두껍게 그려졌어요.";
            case "lipCorner" -> "입술 " + a + " 끝이 " + b + " 끝보다 올라가 있어요.";
            case "lipHalfWidth" -> "입술 선이 " + a + "으로 더 길게 그려졌어요.";
            case "upperLipLine" -> "윗입술 선이 " + a + "이 더 도톰하게 그려졌어요.";
            case "lowerLipLine" -> "아랫입술 선이 " + a + "이 더 도톰하게 그려졌어요.";
            case "cheekColor" -> a + " 볼이 " + b + "보다 더 붉어요. 블러셔가 한쪽에 더 발린 것 같아요.";
            case "lipColor" -> "입술 색이 " + a + "이 " + b + "보다 연해요.";
            case "eyeMakeup" -> a + " 눈화장이 " + b + "보다 진해요. 아이섀도나 아이라인이 한쪽에 더 발린 것 같아요.";
            default -> a + "과 " + b + "이 조금 달라 보여요.";
        };
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

    /** 눈썹 높이·꼬리·길이·두께. 사진이 흐려서 길이를 건너뛰었으면 true. */
    private boolean brows(Sides s, FaceFrame f, BufferedImage img, FaceLabels labels, FaceLabels mirrored,
            Double thickModel, List<Check> checks) {
        List<Pt> lt = f.local(s.leftBrowTop);
        List<Pt> rt = f.local(s.rightBrowTop);
        List<Pt> lb = f.local(s.leftBrowBottom);
        List<Pt> rb = f.local(s.rightBrowBottom);
        boolean both = lb.size() >= 3 && rb.size() >= 3;
        double eyeL = width(f.local(s.leftEye));
        double eyeR = width(f.local(s.rightEye));
        List<Pt> face = f.local(s.face);

        // 1) 분할 모델이 있으면 모델이 고른 눈썹 칸으로, 2) 없거나 못 찾으면 밝기로 나눠서, 3) 그것도 안 되면 윤곽점으로.
        // 한쪽이라도 안 되면 다음 방법으로 둘 다 잼(좌우를 같은 방법으로 비교)
        MakeupPixels.Brow pl = labels == null ? null : MakeupPixels.brow(labels, -1, eyeL);
        MakeupPixels.Brow pr = pl == null ? null : MakeupPixels.brow(labels, 1, eyeR);
        boolean seg = pl != null && pr != null;
        if (!seg) {
            pl = MakeupPixels.brow(img, f, -1, concat(lt, lb), browPoly(lt, lb), face, eyeL);
            pr = pl == null ? null : MakeupPixels.brow(img, f, 1, concat(rt, rb), browPoly(rt, rb), face, eyeR);
        }
        MakeupPixels.Brow bl = pl != null && pr != null ? pl : landmarkBrow(lt, lb, both, -1, eyeL);
        MakeupPixels.Brow br = pl != null && pr != null ? pr : landmarkBrow(rt, rb, both, 1, eyeR);
        String basis = pl != null && pr != null ? PIXELS : LANDMARKS;

        Check height = check("browHeight", bl.center() - br.center(), t.browHeight(), basis, checks);
        // 꼬리: 높이에서 이미 말했으면 겹쳐 말하지 않음
        Check tail = check("browTail", bl.tail() - br.tail(), t.browTail(), basis, checks);
        // 길이: 눈썹 길이 모델이 있으면 그 점수로(왼쪽이 짧으면 +라서 부호를 바꿈: 값의 부호는 긴 쪽). CelebA: 한쪽 꼬리를 지운 것
        // 검출 22% → 84%(같은 2,832장, 같은 사진 짝 새로 잡음 263 / 놓침 3). 모델이 없거나 실패하면 분할·밝기·윤곽점으로 잰 길이 비
        Double model = models.browLength().score(img, f);
        // 흐린 사진(초점 나감·흔들림)에선 길이 모델이 꼬리가 흐릿한 쪽을 짧다고 볼 때가 많아서(CelebA 손대지 않은 사진 300장을
        // 흐리게 하면 잘못 알림 4% → 16%) 눈썹 조각이 기준(선명한 사진의 하위 5%)보다 흐리면 길이는 재지 않고 skipped에 남김.
        // 흐린 사진 잘못 알림 16% → 5%. 여러 장을 보내면 선명한 장의 값만 합침
        boolean blurry = model != null && OnnxPairModel.sharpness(OnnxPairModel.patches(img, f, OnnxPairModel.Region.BROW),
                OnnxPairModel.Region.BROW) < t.browSharpness();
        if (model != null && !blurry) {
            check("browLength", -model, t.browLengthModel(), PIXELS, checks);
        } else if (model == null) {
            check("browLength", ratio(bl.length(), br.length()), t.browLength(), basis, checks);
        }
        // 두께는 뒤집어 나눈 결과의 눈썹 칸을 더해서 잼: 모델이 덧그린 아래쪽을 한쪽에서만 눈썹으로 보는 일이 많아서
        // (CelebA: 한쪽을 두껍게 그린 것 검출 45% → 53%, 같은 사진 짝 새로 잡음 45 / 놓침 8). 높이·길이는 원래 결과가 더 나음
        double thickL = bl.thickness();
        double thickR = br.thickness();
        if (seg && mirrored != null) {
            FaceLabels union = labels.withBrowsFrom(mirrored);
            MakeupPixels.Brow ul = MakeupPixels.brow(union, -1, eyeL);
            MakeupPixels.Brow ur = ul == null ? null : MakeupPixels.brow(union, 1, eyeR);
            if (ul != null && ur != null) {
                thickL = ul.thickness();
                thickR = ur.thickness();
            }
        }
        // 두께 모델이 있으면 그 점수로(사진 왼쪽 눈썹을 더 두껍게 그렸으면 +). CelebA 2,832장: 한쪽을 두껍게 그린 것 검출 51% → 92%(같은 사진 짝 +178 / −6)
        if (thickModel != null) {
            check("browThickness", thickModel, t.browThicknessModel(), PIXELS, checks);
        } else if (thickL > 0 && thickR > 0) {
            check("browThickness", ratio(thickL, thickR), t.browThickness(), basis, checks);
        }
        return blurry;
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

    private void lips(Sides s, FaceFrame f, BufferedImage img, FaceLabels labels, List<Check> checks) {
        List<Pt> top = f.local(s.upperLipTop);
        List<Pt> bottom = f.local(s.lowerLipBottom);
        MakeupPixels.Lips px = MakeupPixels.lips(img, f, polygon(top, bottom), f.local(s.upperLipBottom),
                f.local(s.lowerLipTop));
        MakeupPixels.Lips lm = px != null ? px : landmarkLips(s, f, top, bottom);
        if (lm == null) {
            return; // 입이 가운데 선 한쪽에만 있으면(돌린 얼굴) 비교하지 않음
        }
        String basis = px != null ? PIXELS : LANDMARKS;

        Check corner = check("lipCorner", lm.corner(), t.lipCorner(), basis, checks);
        // 입술 너비: 사진에서 나눈 입꼬리가 윤곽점 입꼬리보다 얼마나 더 나갔는지(IOD)를 좌우 비교. 윤곽점은 그린 선을 거의 안 따라가고
        // 웃음·고개 돌림 같은 원래 비대칭은 같이 따라가서, 빼면 "더 그린 만큼"만 남음
        // (CelebA: 좌우 흔들림 0.16 → 0.05, 한쪽 입꼬리를 밖으로 그린 것 검출 20% → 41%). 색으로 못 나누면(누드 립) 재지 않음
        // 입술 너비 모델이 있으면 그 점수로(사진 왼쪽 입꼬리를 더 길게 그렸으면 +). CelebA: 한쪽 입꼬리를 밖으로 그린 것
        // 검출 20% → 82%(같은 2,832장, 같은 사진 짝 새로 잡음 252 / 놓침 3). 없거나 실패하면 아래 색·윤곽점 방식
        Double width = models.lipWidth().score(img, f);
        MakeupPixels.Lips bone = px != null ? landmarkLips(s, f, top, bottom) : null;
        if (width != null) {
            check("lipHalfWidth", width, t.lipHalfWidthModel(), PIXELS, checks);
        } else if (bone != null) {
            check("lipHalfWidth", (px.halfLeft() - bone.halfLeft()) - (px.halfRight() - bone.halfRight()),
                    t.lipHalfWidth(), basis, checks);
        }
        // 입술 선: 분할 모델이 나눈 윗입술·아랫입술 칸의 두께를 좌우 비교하고, 같은 자리의 윤곽점 두께 비를 빼서
        // 타고난 비대칭을 지움(윤곽점은 넘겨 그린 선을 거의 안 따라감). CelebA: 윗입술 좌우 흔들림 0.45 → 0.18,
        // 한쪽 윗입술을 0.06~0.1 IOD 넘겨 그린 것 검출 60% → 85%. 분할 모델이 없을 때만 아래 밝기·윤곽점 방식으로
        if (labels != null) {
            lipLines(s, f, labels, top, bottom, checks);
            return;
        }
        Double up = lm.upRatio() != null ? lm.upRatio()
                : lm.up() != null && lm.up()[0] > 0.02 && lm.up()[1] > 0.02 ? ratio(lm.up()[0], lm.up()[1]) : null;
        Double down = lm.downRatio() != null ? lm.downRatio()
                : lm.down() != null && lm.down()[0] > 0.02 && lm.down()[1] > 0.02 ? ratio(lm.down()[0], lm.down()[1])
                : null;
        if (up != null) {
            check("upperLipLine", up, t.lipLine(), basis, checks);
        }
        if (down != null) {
            check("lowerLipLine", down, t.lipLine(), basis, checks);
        }
    }

    /** 분할 모델로 잰 입술 선 두께 비 − 윤곽점 두께 비. 입술 칸이 거의 없으면(입을 가림·누드 립 일부) 재지 않음 */
    private void lipLines(Sides s, FaceFrame f, FaceLabels labels, List<Pt> top, List<Pt> bottom, List<Check> checks) {
        MakeupPixels.Lips shape = landmarkLips(s, f, top, bottom);
        if (shape == null) {
            return;
        }
        double yTop = min(top, Pt::y) - 0.15;
        double yBot = max(bottom, Pt::y) + 0.15;
        List<Pt> upperBottom = f.local(s.upperLipBottom);
        List<Pt> lowerTop = f.local(s.lowerLipTop);
        lipLine("upperLipLine", MakeupPixels.lipThickness(labels, FaceLabels.U_LIP, shape.halfLeft(),
                shape.halfRight(), yTop, yBot), MakeupPixels.lineGap(top, upperBottom, shape.halfLeft(),
                shape.halfRight()), checks);
        lipLine("lowerLipLine", MakeupPixels.lipThickness(labels, FaceLabels.L_LIP, shape.halfLeft(),
                shape.halfRight(), yTop, yBot), MakeupPixels.lineGap(lowerTop, bottom, shape.halfLeft(),
                shape.halfRight()), checks);
    }

    private void lipLine(String name, double[] seg, double[] bone, List<Check> checks) {
        if (seg != null && bone != null) {
            check(name, ratio(seg[0], seg[1]) - ratio(bone[0], bone[1]), t.lipLine(), PIXELS, checks);
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
    private List<String> colors(Sides s, FaceFrame f, BufferedImage img, List<Check> checks) {
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
        // 볼 터치 모델이 있으면 그 점수로(사진 왼쪽 볼에 더 발랐으면 +). 평균 붉은 정도(a*)는 볼 크기·위치가 조금만 달라도 흔들려서
        // 옅게 바른 것을 놓쳤음. CelebA 2,832장: 58% → 77%, 같은 사진 짝 +74 / −7(분홍·코랄·베리 계열. 브론즈처럼 붉지 않은 색은 거의 못 잡음)
        Double cheekModel = models.cheekColor().score(img, f);
        if (cheekModel != null) {
            check("cheekColor", cheekModel, t.cheekColorModel(), COLOR, checks);
        } else {
            check("cheekColor", ml.a() - mr.a(), t.cheekRedness(), COLOR, checks);
        }

        LabMean ll = new LabMean();
        LabMean lr = new LabMean();
        sampleLips(img, f, s, ll, lr);
        if (ll.count() < 20 || lr.count() < 20) {
            return List.of("lipColor");
        }
        // 입술 색을 같은 쪽 입술 옆 피부와 견준 차이(L*, a*, b*)를 좌우로 비교: 한쪽에서 빛이 들어와 밝기가 달라도
        // 피부도 같이 밝아지므로 지워짐 (CelebA 비교: 한쪽 립을 옅게 바른 것 검출 49% → 82%)
        LabMean sl = new LabMean();
        LabMean sr = new LabMean();
        sampleLipSkin(img, f, s, sl, sr);
        if (sl.count() < 20 || sr.count() < 20) {
            return List.of("lipColor");
        }
        double dL = (ll.l() - sl.l()) - (lr.l() - sr.l());
        double da = (ll.a() - sl.a()) - (lr.a() - sr.a());
        double db = (ll.b() - sl.b()) - (lr.b() - sr.b());
        double de = Math.sqrt(dL * dL + da * da + db * db);
        // 피부와 색 차이(a*, b*)가 작은 쪽이 더 연한 쪽
        boolean leftPaler = Math.hypot(ll.a() - sl.a(), ll.b() - sl.b()) < Math.hypot(lr.a() - sr.a(), lr.b() - sr.b());
        check("lipColor", leftPaler ? de : -de, t.lipColor(), COLOR, checks);
        return List.of();
    }

    // ---------- 눈화장 ----------

    /**
     * 눈꺼풀(눈 윗선에서 눈썹 아랫선 쪽으로 45%)이 같은 쪽 눈 밑 피부보다 얼마나 어두운지(L*)를 좌우로 비교합니다.
     * 아이섀도·아이라인을 한쪽에 더 바르면 그쪽 눈꺼풀이 어두워집니다. 같은 쪽 피부와 견주므로 한쪽에서 빛이 들어와도
     * 어느 정도 지워지고, 눈 밑 피부 밝기가 좌우로 [unevenLight]보다 다르면 건너뜁니다.
     * 2026-11-12 지원님 반반 화장 사진에서 눈화장 차이를 못 잡아서 추가. 기준은 FFHQ 실사진 95번째 백분위.
     */
    private List<String> eyes(Sides s, FaceFrame f, BufferedImage img, List<Check> checks) {
        LabMean[] lid = {new LabMean(), new LabMean()};
        LabMean[] skin = {new LabMean(), new LabMean()};
        List<List<Pt>> eyes = List.of(s.leftEye, s.rightEye);
        List<List<Pt>> brows = List.of(s.leftBrowBottom, s.rightBrowBottom);
        for (int i = 0; i < 2; i++) {
            List<Pt> eye = f.local(eyes.get(i));
            List<Pt> brow = f.local(brows.get(i));
            if (eye.size() < 6 || brow.size() < 2) {
                return List.of("eyeMakeup");
            }
            double cy = meanY(eye);
            List<Pt> upper = eye.stream().filter(p -> p.y() <= cy).sorted(Comparator.comparingDouble(Pt::x)).toList();
            List<Pt> lower = eye.stream().filter(p -> p.y() > cy).sorted(Comparator.comparingDouble(Pt::x)).toList();
            if (upper.size() < 3 || lower.size() < 3) {
                return List.of("eyeMakeup");
            }
            List<double[]> bx = brow.stream().sorted(Comparator.comparingDouble(Pt::x))
                    .map(p -> new double[] {p.x(), p.y()}).toList();
            List<Pt> lidTop = upper.stream().map(p -> new Pt(p.x(), p.y() + (interpolate(bx, p.x()) - p.y()) * 0.45)).toList();
            samplePoly(img, f, concatReversed(upper, lidTop), lid[i], 5, 97);
            List<Pt> skinTop = lower.stream().map(p -> new Pt(p.x(), p.y() + 0.22)).toList();
            List<Pt> skinBottom = lower.stream().map(p -> new Pt(p.x(), p.y() + 0.40)).toList();
            samplePoly(img, f, concatReversed(skinTop, skinBottom), skin[i], 15, 97);
        }
        for (int i = 0; i < 2; i++) {
            if (lid[i].count() < 15 || skin[i].count() < 15) {
                return List.of("eyeMakeup");
            }
        }
        if (Math.abs(skin[0].l() - skin[1].l()) > t.unevenLight()) {
            return List.of("eyeMakeup");
        }
        // + 이면 왼쪽 눈꺼풀이 더 어두움(더 진하게 바름)
        double diff = (skin[0].l() - lid[0].l()) - (skin[1].l() - lid[1].l());
        check("eyeMakeup", diff, t.eyeMakeup(), COLOR, checks);
        return List.of();
    }

    /** a(왼→오) + b를 거꾸로(오→왼) 이은 닫힌 다각형. */
    private static List<Pt> concatReversed(List<Pt> a, List<Pt> b) {
        List<Pt> out = new ArrayList<>(a);
        List<Pt> rev = new ArrayList<>(b);
        java.util.Collections.reverse(rev);
        out.addAll(rev);
        return out;
    }

    /** 얼굴 좌표 다각형 안의 픽셀을 담습니다. */
    private static void samplePoly(BufferedImage img, FaceFrame f, List<Pt> polyLocal, LabMean out, double minL, double maxL) {
        List<Pt> px = polyLocal.stream().map(f::pixel).toList();
        int x0 = clamp(min(px, Pt::x), img.getWidth());
        int x1 = clamp(max(px, Pt::x), img.getWidth());
        int y0 = clamp(min(px, Pt::y), img.getHeight());
        int y1 = clamp(max(px, Pt::y), img.getHeight());
        int step = Math.max(1, (int) (f.iod / 80));
        for (int y = y0; y <= y1; y += step) {
            for (int x = x0; x <= x1; x += step) {
                if (inside(polyLocal, f.local(new Pt(x + 0.5, y + 0.5)))) {
                    addIf(out, Images.lab(img.getRGB(x, y)), minL, maxL);
                }
            }
        }
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

    /** 입술 바로 바깥 피부(입술 윤곽에서 0.03~0.10 떨어진 띠)를 좌·우로. 입술 색을 그쪽 피부와 견주어 조명 차이를 지움. */
    private static void sampleLipSkin(BufferedImage img, FaceFrame f, Sides s, LabMean left, LabMean right) {
        List<Pt> outer = polygon(f.local(s.upperLipTop), f.local(s.lowerLipBottom));
        double pad = 0.10;
        double x0 = outer.stream().mapToDouble(Pt::x).min().orElse(0) - pad;
        double x1 = outer.stream().mapToDouble(Pt::x).max().orElse(0) + pad;
        double y0 = outer.stream().mapToDouble(Pt::y).min().orElse(0) - pad;
        double y1 = outer.stream().mapToDouble(Pt::y).max().orElse(0) + pad;
        List<Pt> corners = List.of(f.pixel(new Pt(x0, y0)), f.pixel(new Pt(x1, y0)), f.pixel(new Pt(x0, y1)),
                f.pixel(new Pt(x1, y1)));
        int px0 = clamp(min(corners, Pt::x), img.getWidth());
        int px1 = clamp(max(corners, Pt::x), img.getWidth());
        int py0 = clamp(min(corners, Pt::y), img.getHeight());
        int py1 = clamp(max(corners, Pt::y), img.getHeight());
        int step = Math.max(1, (int) (f.iod / 60));
        for (int y = py0; y <= py1; y += step) {
            for (int x = px0; x <= px1; x += step) {
                Pt l = f.local(new Pt(x + 0.5, y + 0.5));
                if (Math.abs(l.x()) < 0.04 || inside(outer, l)) {
                    continue;
                }
                double d = MakeupPixels.distance(outer, l);
                if (d >= 0.03 && d <= pad) {
                    addIf(l.x() < 0 ? left : right, Images.lab(img.getRGB(x, y)), 15, 97);
                }
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
    static Check check(String name, double diff, double limit, String basis, List<Check> checks) {
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

    static String ko(String side) {
        return LEFT.equals(side) ? "왼쪽" : "오른쪽";
    }

    static String opposite(String side) {
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
