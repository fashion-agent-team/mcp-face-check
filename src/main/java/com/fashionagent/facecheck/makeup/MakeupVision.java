package com.fashionagent.facecheck.makeup;

/** 윤곽점이 없을 때 사진만으로 화장 대칭을 보는 방법(지금은 GPT 비전). */
public interface MakeupVision {

    MakeupResult look(String imageBase64);
}
