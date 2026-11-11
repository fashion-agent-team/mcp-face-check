package com.fashionagent.facecheck.makeup;

/** 화장 대칭에 쓰는 좌우 비교 모델 묶음. */
record PairModels(PairModel browLength, PairModel lipWidth) implements AutoCloseable {

    static final PairModels NONE = new PairModels(PairModel.NONE, PairModel.NONE);

    PairModels {
        browLength = browLength == null ? PairModel.NONE : browLength;
        lipWidth = lipWidth == null ? PairModel.NONE : lipWidth;
    }

    @Override
    public void close() throws Exception {
        for (PairModel m : new PairModel[] {browLength, lipWidth}) {
            if (m instanceof AutoCloseable c) {
                c.close();
            }
        }
    }
}
