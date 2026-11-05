# 대칭 판단 기준값 학습

서버(`face_makeup_check`)는 **재기만** 합니다: 눈썹 높이 차이, 볼 붉은 정도 차이 같은 값. "얼마나 다르면 비대칭이라고 말할까"는
사람이 라벨 붙인 셀카로 학습합니다. 항목마다 측정값 하나로 로지스틱 회귀를 학습하고, 확률 0.5가 되는 값을 기준값으로 씁니다.
측정값을 서버에서 받아 오므로 학습과 실제 판단이 같은 코드로 잽니다.

## 데이터 모으기

`data/`(레포에 올리지 않음):

```
data/labels.csv     file,issues
                    001.jpg,                      ← 고름
                    002.jpg,browHeight;cheekColor ← 다름(항목을 ;로)
data/001.jpg        앱이 보내는 것과 같은 셀카(셀카 방향, 얼굴을 비추다 멈춘 순간)
data/001.json       그 장면의 ML Kit 윤곽점(mcp-tools.md의 landmarks)
```

- 항목 이름: `browHeight browTail browLength browThickness lipCorner lipHalfWidth upperLipLine lowerLipLine cheekColor lipColor`
- 라벨은 화장을 보는 사람(팀원·가족)이 사진을 보고 붙입니다. 같은 사람이 일부러 한쪽만 다르게 그린 사진과 고르게 그린 사진을 섞으면 좋습니다.
- 항목마다 고름·다름이 각각 5장 이상 있어야 그 항목을 학습합니다(적으면 지금 기준값 유지). 신뢰할 만하려면 항목마다 수십 장.
- 실시간 영상에서 멈춘 장면을 여러 장 뽑을 수 있어서, 사람 한 명이 화장 몇 가지로 꽤 많은 장면을 만들 수 있습니다.

## 실행

```bash
pip install -r requirements.txt
(cd .. && ./gradlew bootRun) &                 # 서버 띄우기 (localhost:8094)
python3 features.py data                       # → data/features.csv (서버가 잰 값 + 라벨)
python3 train.py data/features.csv             # → out/symmetry-thresholds.yml, 항목별 교차 검증 정확도
python3 train.py data/features.csv --apply     # 서버 리소스에도 써서 다음 빌드부터 적용
```

## 데모 데이터 (과정 확인용)

`make_demo_data.py`는 공개 인물 사진 한 장에 화장 차이를 크기를 바꿔 넣고 일정 크기 이상이면 '다름'으로 라벨을 붙입니다.
윤곽점은 ML Kit 대신 MediaPipe로 뽑습니다. **라벨이 사람 판단이 아니고 얼굴도 한 명이라 실제 기준값으로 쓰면 안 됩니다.**

```bash
python3 make_demo_data.py data_demo 120 && python3 features.py data_demo && python3 train.py data_demo/features.csv
```

2026-11-05 데모 결과(120장): browHeight 기준 0.090(정확도 0.97), cheekColor 7.85(0.98), lipColor 14.3(0.99), 입술 선 0.451(0.96).
입술 선 기준이 처음 값(0.25)보다 높게 나온 건 사진 속 사람이 원래 한쪽으로 웃는 얼굴이라 고른 사진도 0.37쯤 나오기 때문입니다.
