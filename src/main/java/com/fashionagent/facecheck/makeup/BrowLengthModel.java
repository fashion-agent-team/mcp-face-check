package com.fashionagent.facecheck.makeup;

import java.awt.image.BufferedImage;

/**
 * 눈썹 길이 차이를 직접 판단하는 작은 학습 모델. 분할 모델로 눈썹 끝을 찾아 길이를 재면 지운 꼬리를 여전히 눈썹으로 보는 일이 많아
 * (CelebA: 꼬리를 0.25~0.4 지워도 잰 길이는 15%만 줄어듦) 한쪽을 짧게 그린 것을 23%만 잡았음. 그래서 양쪽 눈썹 조각을 같은 모델에
 * 넣어 "왼쪽이 얼마나 짧은지"를 바로 맞추게 학습함(training/train_brow_length.py).
 */
interface BrowLengthModel {

    /** 사진 왼쪽 눈썹이 짧을수록 +, 오른쪽이 짧을수록 −(대략 지운 비율). 모델이 없거나 실패하면 null. */
    Double score(BufferedImage img, FaceFrame f);

    BrowLengthModel NONE = (img, f) -> null;
}
