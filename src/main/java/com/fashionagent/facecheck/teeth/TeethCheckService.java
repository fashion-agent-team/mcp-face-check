package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.image.Images;
import com.fashionagent.facecheck.image.Images.LabMean;
import java.awt.image.BufferedImage;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Service;

/**
 * teeth_check의 공통 부분: 사진 풀기, mouthBox 확인(잘림·크기), 입 부분 자르기, 밝기. 분류는 {@link TeethClassifier}.
 */
@Service
public class TeethCheckService {

    static final double MARGIN = 0.15;
    static final double DARK_LIGHTNESS = 28;

    private final FaceCheckProperties props;
    private final ObjectProvider<TeethClassifier> classifier;

    public TeethCheckService(FaceCheckProperties props, ObjectProvider<TeethClassifier> classifier) {
        this.props = props;
        this.classifier = classifier;
    }

    public TeethResult check(String imageBase64, MouthBox box) {
        TeethClassifier c = classifier.getIfAvailable();
        if (c == null) {
            throw new FaceCheckException("not_ready", "이 확인은 아직 준비 중이에요.");
        }
        BufferedImage img = Images.decode(imageBase64, props.maxImageBytes().toBytes());
        BufferedImage mouth = img;
        if (box != null) {
            box.validate();
            String framing = framing(box);
            if (framing != null) {
                return new TeethResult(framing, "ok", null, c.source());
            }
            mouth = crop(img, box);
        }
        if (lightness(mouth) < DARK_LIGHTNESS) {
            return new TeethResult(null, "dark", null, c.source());
        }
        return new TeethResult(null, "ok", c.classify(mouth, box != null), c.source());
    }

    static String framing(MouthBox b) {
        double edge = 0.005;
        if (b.y() < edge || b.x() < edge || b.x() + b.w() > 1 - edge || b.y() + b.h() > 1 - edge) {
            return "입이 다 보이지 않아요. 폰을 입 정면에 대 주세요.";
        }
        if (b.w() < 0.2) {
            return "입이 작게 보여요. 폰을 입에 조금 더 가까이 대 주세요.";
        }
        if (b.h() < 0.06 || b.h() / b.w() < 0.15) {
            return "입이 다 보이지 않아요. 입을 조금 더 벌려 주세요.";
        }
        return null;
    }

    static BufferedImage crop(BufferedImage img, MouthBox b) {
        int w = img.getWidth();
        int h = img.getHeight();
        int x0 = (int) Math.max(0, Math.floor((b.x() - b.w() * MARGIN) * w));
        int y0 = (int) Math.max(0, Math.floor((b.y() - b.h() * MARGIN) * h));
        int x1 = (int) Math.min(w, Math.ceil((b.x() + b.w() * (1 + MARGIN)) * w));
        int y1 = (int) Math.min(h, Math.ceil((b.y() + b.h() * (1 + MARGIN)) * h));
        return img.getSubimage(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    }

    private static double lightness(BufferedImage img) {
        LabMean m = new LabMean();
        int step = Math.max(1, Math.max(img.getWidth(), img.getHeight()) / 80);
        for (int y = 0; y < img.getHeight(); y += step) {
            for (int x = 0; x < img.getWidth(); x += step) {
                m.add(Images.lab(img.getRGB(x, y)));
            }
        }
        return m.l();
    }
}
