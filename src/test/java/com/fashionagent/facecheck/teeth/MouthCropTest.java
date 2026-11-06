package com.fashionagent.facecheck.teeth;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class MouthCropTest {

    @Test
    @DisplayName("거울 패딩은 OpenCV BORDER_REFLECT_101과 같음")
    void reflect() {
        assertThat(MouthCrop.reflect(-1, 5)).isEqualTo(1);
        assertThat(MouthCrop.reflect(-2, 5)).isEqualTo(2);
        assertThat(MouthCrop.reflect(5, 5)).isEqualTo(3);
        assertThat(MouthCrop.reflect(3, 5)).isEqualTo(3);
    }

    @Test
    @DisplayName("입 박스 긴 변 × 1.36 정사각형 → 384, 정규화한 텐서는 3×S×S")
    void cropAndTensor() {
        BufferedImage img = new BufferedImage(1000, 800, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 800; y++) {
            for (int x = 0; x < 1000; x++) {
                img.setRGB(x, y, x >= 400 && x < 600 && y >= 380 && y < 420 ? 0xFFFFFF : 0x000000);
            }
        }
        // 입 안쪽 박스 = 흰 사각형(200×40px). 정사각형 한 변 = 200 × 1.36 = 272px
        var crop = MouthCrop.crop(img, new MouthBox(0.4, 0.475, 0.2, 0.05));
        assertThat(crop.getWidth()).isEqualTo(384);
        // 가운데는 흰색, 위쪽 끝은 검정
        assertThat(crop.getRGB(192, 192) & 0xFF).isGreaterThan(200);
        assertThat(crop.getRGB(192, 5) & 0xFF).isLessThan(30);
        // 흰 띠 너비 = 384 / 1.36 ≈ 282px
        int white = 0;
        for (int x = 0; x < 384; x++) {
            if ((crop.getRGB(x, 192) & 0xFF) > 128) {
                white++;
            }
        }
        assertThat(white).isBetween(276, 288);

        float[] t = MouthCrop.toTensor(crop, 320);
        assertThat(t).hasSize(3 * 320 * 320);
        // 흰색 R 채널 = (1 - 0.485) / 0.229
        assertThat(t[160 * 320 + 160]).isCloseTo(2.2489f, org.assertj.core.data.Offset.offset(0.01f));
    }

    @Test
    @DisplayName("사진 밖으로 나간 부분은 거울 패딩으로 채움(검정 테두리 없음)")
    void outside() {
        BufferedImage img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 200; y++) {
            for (int x = 0; x < 200; x++) {
                img.setRGB(x, y, 0x808080);
            }
        }
        var crop = MouthCrop.crop(img, new MouthBox(-0.2, 0.4, 0.5, 0.2));
        assertThat(crop.getRGB(0, 192) & 0xFF).isBetween(120, 136);
    }
}
