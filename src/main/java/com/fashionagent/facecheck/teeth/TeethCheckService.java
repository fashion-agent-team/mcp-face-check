package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.image.Images;
import java.awt.image.BufferedImage;
import java.util.Map;
import org.springframework.stereotype.Service;

/**
 * teeth_check: 입 근접 사진 → 이 사이에 낀 것. 우리가 학습한 모델(ONNX)을 이 서버에서 돌리고, 사진은 외부 AI로
 * 보내지 않습니다(개인정보, 2026-11-06 결정). 확률이 애매하면 지어내지 않고 foreignObject=null로 돌려줍니다.
 */
@Service
public class TeethCheckService {

    private final FaceCheckProperties props;
    private final TeethClassifier classifier;

    public TeethCheckService(FaceCheckProperties props, TeethClassifier classifier) {
        this.props = props;
        this.classifier = classifier;
    }

    public TeethResult check(String imageBase64, Map<String, Double> mouthBox) {
        if (!classifier.ready()) {
            throw new FaceCheckException("not_ready", "이 확인은 아직 준비 중이에요.");
        }
        FaceCheckProperties.Teeth t = props.teeth();
        BufferedImage img = Images.decode(imageBase64, props.maxImageBytes().toBytes());
        MouthBox box = MouthBox.from(mouthBox);

        String framing = framing(box, t);
        if (framing != null) {
            return TeethResult.retake(framing);
        }
        BufferedImage crop = MouthCrop.crop(img, box);
        if (meanLightness(crop) < t.darkLightness()) {
            return new TeethResult(null, "dark",
                    new TeethResult.Teeth(null, null, "어두워서 이가 잘 보이지 않아요. 밝은 곳에서 다시 해 주세요."));
        }
        double p = classifier.foodProbability(MouthCrop.toTensor(crop, classifier.inputSize()));
        double conf = Math.round(p * 100) / 100.0;
        if (p >= t.yesThreshold()) {
            return new TeethResult(null, "ok", new TeethResult.Teeth(true, conf, "이 사이에 뭔가 낀 것 같아요."));
        }
        if (p <= t.noThreshold()) {
            return new TeethResult(null, "ok", new TeethResult.Teeth(false, conf, "이 사이에 낀 건 없어 보여요."));
        }
        return new TeethResult(null, "ok", new TeethResult.Teeth(null, conf, "잘 보이지 않아서 확실하지 않아요."));
    }

    /** 입이 사진 밖으로 나갔거나, 너무 작거나, 덜 벌렸으면 다시 찍기 안내. mouthBox가 없으면 확인하지 못함. */
    static String framing(MouthBox box, FaceCheckProperties.Teeth t) {
        if (box == null) {
            return null;
        }
        double out = t.maxOutside();
        if (box.x() < -out * box.w()) {
            return "입이 화면 왼쪽으로 잘렸어요. 입이 화면 가운데 오게 해 주세요.";
        }
        if (box.x() + box.w() > 1 + out * box.w()) {
            return "입이 화면 오른쪽으로 잘렸어요. 입이 화면 가운데 오게 해 주세요.";
        }
        if (box.y() < -out * box.h()) {
            return "입이 화면 위쪽으로 잘렸어요. 폰을 조금 위로 올려 주세요.";
        }
        if (box.y() + box.h() > 1 + out * box.h()) {
            return "입이 화면 아래쪽으로 잘렸어요. 폰을 조금 아래로 내려 주세요.";
        }
        if (box.w() < t.minMouthWidth()) {
            return "입이 너무 작게 보여요. 폰을 입에 조금 더 가까이 대 주세요.";
        }
        if (box.openRatio() < t.minOpenRatio()) {
            return "이가 잘 안 보여요. 입을 조금 더 벌리고 '이~' 해 주세요.";
        }
        return null;
    }

    static double meanLightness(BufferedImage img) {
        Images.LabMean m = new Images.LabMean();
        int step = Math.max(1, img.getWidth() / 64);
        for (int y = 0; y < img.getHeight(); y += step) {
            for (int x = 0; x < img.getWidth(); x += step) {
                m.add(Images.lab(img.getRGB(x, y)));
            }
        }
        return m.l();
    }
}
