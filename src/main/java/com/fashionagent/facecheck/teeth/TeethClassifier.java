package com.fashionagent.facecheck.teeth;

/** 입 크롭 한 장 → "이 사이에 뭔가 낀" 확률(0~1). */
public interface TeethClassifier {

    /** 모델 파일이 있어 판정할 수 있는지. */
    boolean ready();

    /** 입력 크기(정사각형 한 변, 픽셀). */
    int inputSize();

    /**
     * @param chw {@link MouthCrop#toTensor}가 만든 NCHW(N=1) float 배열
     * @return 음식이 낀 확률 (sigmoid(logit))
     */
    double foodProbability(float[] chw);
}
