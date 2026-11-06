package com.fashionagent.facecheck.teeth;

/**
 * teeth_check 결과 (앱 레포 docs/contracts/mcp-tools.md). framing이 있으면 그것만 읽고 다시 찍습니다.
 * teeth.foreignObject: true(낀 것 있음) / false(없음) / null(애매함 → agent가 가족에게 물어볼지 제안).
 */
public record TeethResult(String framing, String lighting, Teeth teeth) {

    public record Teeth(Boolean foreignObject, Double confidence, String note) {
    }

    static TeethResult retake(String framing) {
        return new TeethResult(framing, "ok", null);
    }
}
