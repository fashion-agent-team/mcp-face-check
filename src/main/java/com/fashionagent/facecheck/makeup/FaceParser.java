package com.fashionagent.facecheck.makeup;

import java.awt.image.BufferedImage;

/**
 * 얼굴 분할 모델: 사진을 눈썹·머리카락·입술 등으로 픽셀마다 나눕니다. 눈썹 모양을 "주변보다 어두운 픽셀"로 재면
 * 머리카락·그늘이 섞여서 흔들렸는데(CelebA 300명 비교: 흔들림의 80% 이상이 사진마다 다름), 분할 모델은 눈썹만 골라 줍니다.
 */
interface FaceParser {

    /** 모델이 없거나 읽지 못했으면 false. 그때는 지금처럼 밝기로 나눠 잽니다. */
    boolean ready();

    /** 얼굴 좌표(FaceFrame) 기준 격자로 나눈 결과. 실패하면 null. */
    FaceLabels parse(BufferedImage img, FaceFrame f);

    /**
     * 사진을 좌우로 뒤집어 나눈 뒤 다시 뒤집어 돌려준 결과(왼쪽·오른쪽 분류 번호도 바꿔 맞춤). 모델이 왼쪽·오른쪽 눈썹을 조금 다르게
     * 그리는 버릇을 덜어 내려고 씀. 지원하지 않거나 실패하면 null.
     */
    default FaceLabels parseMirrored(BufferedImage img, FaceFrame f) {
        return null;
    }

    FaceParser NONE = new FaceParser() {
        @Override
        public boolean ready() {
            return false;
        }

        @Override
        public FaceLabels parse(BufferedImage img, FaceFrame f) {
            return null;
        }
    };
}
