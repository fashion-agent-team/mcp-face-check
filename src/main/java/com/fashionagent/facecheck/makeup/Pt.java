package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckException;
import java.util.ArrayList;
import java.util.List;

/** 점 하나. 쓰는 곳에 따라 비율 좌표, 픽셀 좌표, 얼굴 좌표(FaceFrame) 중 하나입니다. */
record Pt(double x, double y) {

    /** [[x, y], ...] 비율 좌표 → 픽셀 좌표. null이면 빈 목록. */
    static List<Pt> pixels(List<List<Double>> raw, int width, int height) {
        List<Pt> out = new ArrayList<>();
        if (raw == null) {
            return out;
        }
        for (List<Double> p : raw) {
            if (p == null || p.size() < 2 || p.get(0) == null || p.get(1) == null) {
                throw FaceCheckException.badRequest("윤곽점 모양이 맞지 않아요.");
            }
            double x = p.get(0);
            double y = p.get(1);
            if (!Double.isFinite(x) || !Double.isFinite(y) || x < -0.5 || x > 1.5 || y < -0.5 || y > 1.5) {
                throw FaceCheckException.badRequest("윤곽점은 사진 크기를 1로 본 비율이어야 해요.");
            }
            out.add(new Pt(x * width, y * height));
        }
        return out;
    }

    static Pt mean(List<Pt> pts) {
        double x = 0;
        double y = 0;
        for (Pt p : pts) {
            x += p.x;
            y += p.y;
        }
        return new Pt(x / pts.size(), y / pts.size());
    }
}
