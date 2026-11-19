package com.fashionagent.facecheck;

/** 도구 결과 {"error": {"code", "message"}}로 바뀌는 오류. message는 사용자에게 읽어 줄 수 있는 한국어. */
public class FaceCheckException extends RuntimeException {

    private final String code;

    public FaceCheckException(String code, String message) {
        super(message);
        this.code = code;
    }

    public String code() {
        return code;
    }

    public static FaceCheckException badRequest(String message) {
        return new FaceCheckException("bad_request", message);
    }
}
