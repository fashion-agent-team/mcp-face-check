package com.fashionagent.facecheck.makeup;

/**
 * 얼굴 좌표에 맞춘 정사각 격자의 분류 결과. 격자 한 칸 (i, j)의 얼굴 좌표는 {@link #x(int)}, {@link #y(int)}.
 * 분류 번호는 CelebAMask-HQ 순서(BiSeNet 학습 데이터)입니다.
 */
record FaceLabels(byte[] labels, int size, double x0, double y0, double span) {

    static final int L_BROW = 2;
    static final int R_BROW = 3;
    static final int U_LIP = 12;
    static final int L_LIP = 13;
    static final int HAIR = 17;

    /** 격자 한 칸의 한 변(얼굴 좌표, IOD 단위) */
    double cell() {
        return span / size;
    }

    double x(int i) {
        return x0 + (i + 0.5) * cell();
    }

    double y(int j) {
        return y0 + (j + 0.5) * cell();
    }

    int at(int i, int j) {
        return labels[j * size + i] & 0xff;
    }

    boolean brow(int i, int j) {
        int c = at(i, j);
        return c == L_BROW || c == R_BROW;
    }
}
