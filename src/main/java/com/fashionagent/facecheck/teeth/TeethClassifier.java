package com.fashionagent.facecheck.teeth;

import java.awt.image.BufferedImage;

/**
 * 이에 낀 것 분류 (정지원 님 담당: MobileNet 파인튜닝 → ONNX). 이 인터페이스를 구현한 빈(@Component)을 하나
 * 넣으면 teeth_check가 그걸 씁니다. 없으면 teeth_check는 not_ready 오류를 돌려줍니다.
 *
 * <p>입력 사진은 {@link TeethCheckService}가 준비합니다: 앱이 보낸 mouthBox(입 영역)에 여유를 15% 두고 잘라낸 것,
 * mouthBox가 없으면 사진 전체(이때 입 찾기는 구현이 맡음). 셀카 방향이라 사진 왼쪽이 사용자 왼쪽입니다.
 * 구도·밝기 확인은 이미 끝난 사진만 들어옵니다.
 */
public interface TeethClassifier {

    TeethFinding classify(BufferedImage mouth, boolean cropped);

    /** teeth_check 결과의 source 값. 우리 모델이면 model, GPT 폴백이면 gpt. */
    default String source() {
        return "model";
    }

    /**
     * @param foreignObject 이에 뭔가 낀 것 같으면 true
     * @param confidence    0~1
     * @param note          읽어 줄 한국어 한 문장(예: "앞니 사이에 뭔가 낀 것 같아요."). 없으면 null
     */
    record TeethFinding(boolean foreignObject, double confidence, String note) {
    }
}
