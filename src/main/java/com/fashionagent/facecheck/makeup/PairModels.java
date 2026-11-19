package com.fashionagent.facecheck.makeup;

/** 화장 대칭에 쓰는 좌우 비교 모델 묶음. */
record PairModels(PairModel browLength, PairModel browThickness, PairModel lipWidth, PairModel cheekColor, PairModel eyeMakeup) implements AutoCloseable {

    static final PairModels NONE = new PairModels(PairModel.NONE, PairModel.NONE, PairModel.NONE, PairModel.NONE, PairModel.NONE);

    PairModels {
        browLength = browLength == null ? PairModel.NONE : browLength;
        browThickness = browThickness == null ? PairModel.NONE : browThickness;
        lipWidth = lipWidth == null ? PairModel.NONE : lipWidth;
        cheekColor = cheekColor == null ? PairModel.NONE : cheekColor;
        eyeMakeup = eyeMakeup == null ? PairModel.NONE : eyeMakeup;
    }

    @Override
    public void close() throws Exception {
        for (PairModel m : new PairModel[] {browLength, browThickness, lipWidth, cheekColor, eyeMakeup}) {
            if (m instanceof AutoCloseable c) {
                c.close();
            }
        }
    }
}
