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

    static final int NOSE = 10;

    /** 얼굴 안쪽으로 보는 분류: 피부, 눈썹, 눈, 코, 입 안, 입술 (귀·목·머리카락 제외) */
    boolean face(int i, int j) {
        int c = at(i, j);
        return c == 1 || c == L_BROW || c == R_BROW || c == 4 || c == 5 || c == NOSE || c == 11 || c == U_LIP
                || c == L_LIP;
    }

    /** 코 칸의 가로 가운데 값(얼굴 좌표). 고개가 돌면 코가 눈 가운데에서 비켜나므로 얼굴 가운데 선으로 씀. 못 찾으면 NaN */
    double noseX() {
        java.util.List<Double> xs = new java.util.ArrayList<>();
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                if (at(i, j) == NOSE) {
                    xs.add(x(i));
                }
            }
        }
        if (xs.size() < 50) {
            return Double.NaN;
        }
        java.util.Collections.sort(xs);
        return xs.get(xs.size() / 2);
    }

    /** 높이 y 근처(±0.06) 띠에서 그쪽(sgn) 얼굴 바깥 끝 |x|. 못 찾으면 NaN */
    double faceEdge(int sgn, double y) {
        java.util.List<Double> xs = new java.util.ArrayList<>();
        for (int j = 0; j < size; j++) {
            if (Math.abs(y(j) - y) >= 0.06) {
                continue;
            }
            for (int i = 0; i < size; i++) {
                if (x(i) * sgn > 0 && face(i, j)) {
                    xs.add(Math.abs(x(i)));
                }
            }
        }
        if (xs.size() <= 20) {
            return Double.NaN;
        }
        java.util.Collections.sort(xs);
        return xs.get((int) Math.min(xs.size() - 1, Math.round(0.99 * (xs.size() - 1))));
    }

    boolean brow(int i, int j) {
        int c = at(i, j);
        return c == L_BROW || c == R_BROW;
    }
}
