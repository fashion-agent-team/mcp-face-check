package com.fashionagent.facecheck.makeup;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.facecheck.SyntheticFace;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

class BrowLengthModelTest {

    @Test
    @DisplayName("눈썹 길이 모델: 사진을 좌우로 뒤집으면 점수 부호만 바뀜(같은 모델을 양쪽에 써서 좌우 버릇이 없음)")
    void mirroredFlipsSign() throws Exception {
        try (OnnxBrowLength m = new OnnxBrowLength(new ClassPathResource("models/brow_length.onnx"))) {
            var face = new SyntheticFace();
            BufferedImage img = face.image();
            var lm = MakeupAnalyzerTest.JSON.convertValue(face.landmarks(), FaceLandmarks.class);
            int w = img.getWidth();
            List<Pt> le = Pt.pixels(lm.leftEye(), w, img.getHeight());
            List<Pt> re = Pt.pixels(lm.rightEye(), w, img.getHeight());
            Double s = m.score(img, FaceFrame.of(Pt.mean(le), Pt.mean(re)));
            assertThat(s).isNotNull();

            AffineTransform flip = new AffineTransform(-1, 0, 0, 1, w, 0);
            BufferedImage mir = new AffineTransformOp(flip, AffineTransformOp.TYPE_NEAREST_NEIGHBOR).filter(img, null);
            Pt l = Pt.mean(re);
            Pt r = Pt.mean(le);
            Double sm = m.score(mir, FaceFrame.of(new Pt(w - r.x(), r.y()), new Pt(w - l.x(), l.y())));
            assertThat(sm).isNotNull();
            assertThat(s + sm).isBetween(-0.01, 0.01);
        }
    }

    @Test
    @DisplayName("눈썹 길이 모델 파일이 없으면 null, 분석은 분할 결과로 길이를 잼")
    void missingModel() throws Exception {
        try (OnnxBrowLength m = new OnnxBrowLength(new ClassPathResource("models/none.onnx"))) {
            assertThat(m.score(new SyntheticFace().image(), FaceFrame.of(new Pt(0, 0), new Pt(10, 0)))).isNull();
        }
        assertThat(BrowLengthModel.NONE.score(null, null)).isNull();
    }
}
