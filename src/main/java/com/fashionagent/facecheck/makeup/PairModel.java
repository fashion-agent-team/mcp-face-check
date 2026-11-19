package com.fashionagent.facecheck.makeup;

import java.awt.image.BufferedImage;

/**
 * 얼굴 왼쪽·오른쪽 같은 자리 조각을 같은 작은 모델에 넣어 "어느 쪽이 얼마나 다른지"를 바로 맞추는 학습 모델.
 * 재는 방식으로는 한계가 있던 눈썹 길이(분할 모델이 지운 꼬리를 여전히 눈썹으로 봄)와 입술 너비에 씀.
 * 같은 모델을 양쪽에 쓰고 빼서 좌우를 바꾸면 부호만 바뀜(training/train_pair_model.py).
 */
interface PairModel {

    /** 사진 왼쪽 쪽으로 + (어느 쪽이 "더"인지는 모델마다 다름). 모델이 없거나 실패하면 null. */
    Double score(BufferedImage img, FaceFrame f);

    PairModel NONE = (img, f) -> null;
}
