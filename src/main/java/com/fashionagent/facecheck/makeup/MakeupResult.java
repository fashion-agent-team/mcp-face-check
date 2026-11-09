package com.fashionagent.facecheck.makeup;

import java.util.List;

/**
 * face_makeup_check 결과 (앱 레포 docs/contracts/mcp-tools.md). framing이 있거나 lighting이 dark면
 * agent가 그 안내만 하고 다시 찍으므로 symmetry는 null입니다.
 */
public record MakeupResult(String framing, String lighting, Symmetry symmetry) {

    static MakeupResult retake(String framing, String lighting) {
        return new MakeupResult(framing, lighting, null);
    }

    /**
     * @param ok      좌우가 고르면 true
     * @param notes   읽어 줄 한국어 문장(차이가 큰 순서, 최대 face.symmetry.max-notes개)
     * @param checks  계산한 항목 전부(training/에서 기준값을 학습할 때 쓰는 값)
     * @param skipped 조명 때문에 건너뛴 항목(cheekColor, lipColor)
     * @param frames  판단에 쓴 장면 수(여러 장을 보내면 항목별 중간값)
     */
    public record Symmetry(boolean ok, List<String> notes, List<Check> checks, List<String> skipped, int frames) {

        public Symmetry(boolean ok, List<String> notes, List<Check> checks, List<String> skipped) {
            this(ok, notes, checks, skipped, 1);
        }
    }

    /**
     * 비교 항목 하나.
     *
     * @param name  browHeight, browTail, browLength, browThickness, lipCorner, lipHalfWidth, upperLipLine,
     *              lowerLipLine, cheekColor, lipColor
     * @param side  더 높은·긴·두꺼운·붉은·연한 쪽(left | right). 차이가 없으면 null
     * @param value 좌우 차이(항목마다 단위가 다름, FaceCheckProperties.Symmetry 참고)
     * @param limit 이 값을 넘으면 notes에 넣음
     * @param basis 무엇으로 쟀는지: pixels(사진에서 그려진 모양), landmarks(윤곽점, 픽셀로 나누지 못했을 때), color
     */
    public record Check(String name, String side, double value, double limit, boolean ok, String basis) {
    }
}
