package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.image.Images;
import java.awt.image.BufferedImage;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Service;

/** face_makeup_check: 윤곽점이 다 있으면 규칙으로 계산(source=model), 없으면 GPT 비전(source=gpt). */
@Service
public class MakeupCheckService {

    private final FaceCheckProperties props;
    private final MakeupAnalyzer analyzer;
    private final ObjectProvider<MakeupVision> vision;

    public MakeupCheckService(FaceCheckProperties props, ObjectProvider<MakeupVision> vision) {
        this.props = props;
        this.analyzer = new MakeupAnalyzer(props.symmetry());
        this.vision = vision;
    }

    public MakeupResult check(String imageBase64, FaceLandmarks landmarks) {
        BufferedImage img = Images.decode(imageBase64, props.maxImageBytes().toBytes());
        if (landmarks != null && landmarks.complete()) {
            return analyzer.analyze(landmarks, img);
        }
        MakeupVision v = vision.getIfAvailable();
        if (v == null) {
            throw new FaceCheckException("no_landmarks", "얼굴 윤곽을 받지 못해서 확인하지 못했어요. 얼굴을 다시 비춰 주세요.");
        }
        return v.look(imageBase64);
    }
}
