# 개발 방법

## 필요한 것

- Java 21 (Gradle은 `./gradlew`가 받아 옵니다)
- DB·키 없이 돌아갑니다. GPT는 쓰지 않습니다.

## 실행

```bash
./gradlew bootRun     # http://localhost:8094, MCP는 /mcp (Streamable HTTP)
./gradlew test        # 가짜 얼굴(사진 + 윤곽점)로 규칙 확인 + 실제 /mcp 접속
docker build -t ghcr.io/fashion-agent-team/mcp-face-check:dev .
python3 training/train.py training/data/features.csv --apply   # 기준값 다시 학습
```

## 여러 장 (실시간으로 비추기)

앱은 얼굴을 비추는 동안 ML Kit로 구도를 안내하고, 잘 잡혀 1초 멈추면 그동안 3~5장을 보냅니다(첫 장은 `imageBase64`·`landmarks`, 나머지는 `frames`).
서버는 장마다 잰 뒤 항목별로 왼쪽 +, 오른쪽 −로 바꾼 값의 **중간값**으로 다시 판단합니다(`symmetry.frames` = 쓴 장 수).
절반 넘는 장이 구도·밝기 문제면 그 안내를 돌려주고, 한두 장만 문제면 나머지로 판단합니다. 순간 표정이나 흔들림에 덜 속습니다.

## landmarks (앱이 보내는 것)

ML Kit Face Detection의 **윤곽(contour) 모드** 결과를 상수 이름 camelCase로 보냅니다. 점은 `[x, y]`, 사진 너비·높이를 1로 본 비율,
셀카 화면 방향으로 뒤집은 뒤의 좌표입니다.

| 키 | ML Kit | 필수 |
| --- | --- | --- |
| `leftEye`, `rightEye` | LEFT_EYE, RIGHT_EYE | ✔ |
| `leftEyebrowTop`, `rightEyebrowTop` | LEFT/RIGHT_EYEBROW_TOP | ✔ |
| `leftEyebrowBottom`, `rightEyebrowBottom` | LEFT/RIGHT_EYEBROW_BOTTOM | 두께 비교 |
| `upperLipTop`, `lowerLipBottom` | UPPER_LIP_TOP, LOWER_LIP_BOTTOM | ✔ |
| `upperLipBottom`, `lowerLipTop` | UPPER_LIP_BOTTOM, LOWER_LIP_TOP | 벌린 입 안쪽을 색 비교에서 뺌 |
| `face` | FACE | 잘림·정면 확인(없으면 어림) |
| `noseBridge`, `noseBottom` | NOSE_BRIDGE, NOSE_BOTTOM | 정면 확인 |
| `leftCheek`, `rightCheek` | LEFT/RIGHT_CHEEK (1점) | 볼 색 위치(없으면 어림) |
| `yaw`, `roll` | headEulerAngleY, headEulerAngleZ | 정면·기울기 확인 |

필수가 하나라도 없으면 `no_landmarks`(다시 찍기)입니다. ML Kit의 left/right가 사진 기준인지 얼굴 주인 기준인지 헷갈리기 쉬워서,
**서버는 이름을 믿지 않고 두 눈의 x 위치로 왼쪽을 다시 정합니다.**

## 대칭 규칙

1. **얼굴 좌표**: 두 눈 중심의 가운데를 (0, 0), 두 눈을 잇는 선을 가로축(고개 기울기를 돌려 맞춤), 두 눈 사이 거리(IOD)를 1로 둡니다.
   얼굴 가운데 선은 x = 0이고, 오른쪽 값을 뒤집어 왼쪽과 비교합니다. 사진 크기·거리·기울기와 상관없이 같은 기준이 됩니다.
2. **구도**(하나라도 걸리면 `framing`만 돌려주고 끝): 얼굴 상자가 사진 밖으로 나감 → 그 방향으로 폰 옮기기, 얼굴 너비 < 사진 30% → 가까이,
   `|yaw|` > 12° 또는 얼굴 외곽이 한쪽으로 15% 넘게 쏠림 → 정면 보기, 기울기 > 25° → 고개 똑바로.
3. **밝기**: 눈 아래~입 위 평균 L* < 28 → `lighting: "dark"`.
4. **모양**(IOD 단위, ratio는 큰 쪽 대비). 윤곽점은 얼굴 뼈대를 따라가서 **그린 눈썹·입술 선을 거의 따라가지 않습니다**
   (실제 사진 비교: 눈썹을 0.12 올려 그려도 윤곽점은 0.05만 움직였고, 입술 선을 넘게 그린 건 0). 그래서 윤곽점으로 어디를 볼지만 정하고,
   모양은 사진 픽셀에서 잽니다(`MakeupPixels`, `checks[].basis = "pixels"`).
   - 눈썹: 얼굴 분할 모델(BiSeNet, `OnnxFaceParser`)이 눈썹이라고 한 픽셀로 잽니다. 두께는 가로 위치마다 잰 세로 두께의 가운데 값.
     모델이 없거나 못 찾으면: 눈썹 위 띠에서 주변 피부보다 L*가 12 이상 어두운 픽셀을 윤곽점 눈썹 가로 범위 안에서 보고,
     덩어리 중 "크고, 윤곽점 눈썹 가운데에 가깝고, 가로로 긴" 것.
     길이는 분할 모델일 때 같은 높이의 얼굴 반쪽 너비(코 가운데 ~ 옆선)로, 밝기로 잴 때는 같은 쪽 눈 너비로 나눠 고개가 조금 돌아도 원근 차이를 줄입니다.
   - 입술: 입술 다각형 안·밖의 a*(붉은 정도) 가운데 값으로 나눔. 입술과 피부 a* 차이가 4보다 작으면(누드 립) 나누지 못하므로 윤곽점으로 잽니다(`basis = "landmarks"`).

   | 항목 | 계산 | 처음 값 → 지금 기준(CelebA) | 읽는 문장 |
   | --- | --- | --- | --- |
   | `browHeight` | 눈 선 ~ 눈썹 가운데 선 높이 차 | 0.07 → 0.042 | 왼쪽 눈썹이 오른쪽보다 조금 높게 그려졌어요. |
   | `browTail` | 눈썹 바깥 끝 높이 차 (높이에서 말했으면 생략) | 0.09 → 0.063 | 오른쪽 눈썹 꼬리가 왼쪽보다 올라가 있어요. |
   | `browLength` | 눈썹 가로 길이 비율 차 | 15% → 22% | 오른쪽 눈썹이 왼쪽보다 조금 짧게 그려졌어요. |
   | `browThickness` | 눈썹 위·아래 윤곽 사이 비율 차 | 25% → 18% | 왼쪽 눈썹이 오른쪽보다 두껍게 그려졌어요. |
   | `lipCorner` | 두 입꼬리 높이 차 | 0.05 → 0.085 | 입술 왼쪽 끝이 오른쪽 끝보다 올라가 있어요. |
   | `lipHalfWidth` | 사진 입꼬리가 윤곽점 입꼬리보다 더 나간 길이의 좌우 차(IOD). 색으로 못 나누면 재지 않음 | 0.12 → 0.103 | 입술 선이 오른쪽으로 더 길게 그려졌어요. |
   | `upperLipLine`, `lowerLipLine` | 입술 픽셀 끝에서 윤곽점 안쪽 선(윗입술 아래 선, 아랫입술 위 선)까지 두께를 가운데에서 같은 거리끼리 좌우 비율로 내고 가운데 값(픽셀로 못 재면 입꼬리 선 기준 윤곽점) | 25% → 40% | 윗입술 선이 왼쪽이 더 도톰하게 그려졌어요. |

5. **색**(CIELAB, 그림자 L* < 15·번들거림 > 97 픽셀은 뺌)
   - 두 볼(반지름 0.14 원)의 L* 차가 15를 넘으면 한쪽이 그늘진 것으로 보고 색 비교를 건너뜁니다(`skipped`).
   - `cheekColor`: 볼 a*(붉은 정도) 차 > 6(지금 5.7) → "왼쪽 볼이 오른쪽보다 더 붉어요. 블러셔가 한쪽에 더 발린 것 같아요."
   - `lipColor`: 입술 안(벌린 입 안쪽 제외)을 가운데 선으로 나누고, 쪽마다 입술 바로 바깥 피부(윤곽에서 0.03~0.10)와의 색 차이(L*, a*, b*)를 구해 그 차이끼리 ΔE > 10(지금 7.2). 한쪽에서 빛이 들어와도 피부가 같이 밝아져 지워집니다. 피부와 색 차이가 작은 쪽을 "입술 색이 오른쪽이 왼쪽보다 연해요."
6. 기준을 넘은 항목을 (차이 ÷ 기준) 큰 순서로 최대 3문장 `notes`에 넣습니다. `checks`에는 모든 항목의 값이 남아 기준값을 맞출 때 씁니다.

기준값은 **라벨 붙인 셀카로 학습**해서 정합니다([training/README.md](../training/README.md)): 서버가 잰 값 → 항목마다 로지스틱 회귀 → 확률 0.5가 되는 값을
`src/main/resources/symmetry-thresholds.yml`로 저장, 서버가 시작할 때 읽습니다. 그 파일이 없으면 아래 처음 값을 씁니다.
**지금 기준(2026-11-09)은 공개 데이터셋 CelebA로 정했습니다**([training/public-baseline.md](../training/public-baseline.md)): 화장한 정면 얼굴 사진 2,832장을 서버로 재서 항목마다 95번째 백분위를 기준으로 씁니다. "잘 화장한 공개 사진 100장 중 95장보다 더 다르면 말한다"는 뜻입니다. 팀 셀카에 라벨이 생기면 `train.py`로 다시 학습해 바꿉니다.
처음 값은 손으로 정한 값입니다. 실제 사진(웃는 얼굴) 한 장에서도 원래 비대칭(입꼬리 한쪽 웃음, 머리카락에 가린 눈썹 꼬리)이 기준을 넘었습니다. 실제 셀카로 맞춰 가며 `application.yml`의 `face.symmetry.*`로 바꿉니다(이름은 `FaceCheckProperties` 참고).
사람 얼굴은 원래 조금 비대칭이고 표정(한쪽 웃음, 눈썹 올림)도 차이로 잡히므로, 앱 안내에서 "무표정으로 정면"을 말해 주는 게 좋습니다.

## 치아 (teeth_check)

우리가 학습한 분류 모델 `src/main/resources/models/teeth_mobilenetv3.onnx`(MobileNetV3-Small, 입력 320)를 ONNX Runtime Java로 이 서버 안에서 돌립니다.
모델 파일이 없으면 서버는 뜨고 `teeth_check`만 `not_ready`를 돌려줍니다. 위치는 `face.teeth.model`로 바꿀 수 있습니다.

1. **구도**(`mouthBox`가 있을 때, 하나라도 걸리면 `framing`만 돌려주고 끝): 박스가 사진 밖으로 박스 크기의 10% 넘게 나감 → 그 방향 안내,
   입 너비 < 사진 너비 15% → 가까이, 입 안쪽 높이 ÷ 너비 < 0.10 → "입을 더 벌리고 '이~'" (학습 데이터가 0.10 이상이라 그보다 덜 벌린 사진은 판단하지 않음).
2. **자르기**(학습과 같은 규칙, `teeth/MouthCrop`): `mouthBox`(ML Kit `UPPER_LIP_BOTTOM` + `LOWER_LIP_TOP` 점의 박스) 중심에서 긴 변 × 1.36 정사각형,
   사진 밖은 거울 패딩 → 384 → 320. RGB, 0~1, ImageNet mean/std, NCHW float32. `mouthBox`가 없으면 사진 가운데(짧은 변의 80%)를 봅니다.
3. **밝기**: 크롭 평균 L* < 25 → `lighting: "dark"`, 판단하지 않음.
4. **판단**: p = sigmoid(logit). p ≥ 0.7 → `foreignObject: true`, p ≤ 0.3 → `false`, 그 사이 → `null`(애매함, agent가 가족에게 물어볼지 제안).
   기준은 첫 값이고 실제 사진으로 평가한 뒤 `face.teeth.yes-threshold`, `no-threshold`로 맞춥니다.

모델 성능(합성 데이터 검증 1,048장): 정확도 87.2%, "낀 것 있음"이라고 했을 때 맞음 0.96, 실제로 낀 것을 찾음 0.78. 크고 어두운 조각(김, 채소)은 잘 찾고
작은 조각(깨, 고춧가루) 하나만 낀 경우는 자주 놓칩니다. 실제 폰 사진 평가는 아직입니다.
학습 데이터는 FFHQ 얼굴(NVIDIA, CC BY-NC-SA 4.0)에서 입을 잘라 음식 조각을 합성한 것이라 레포에 올리지 않습니다(모델 파일에는 사진이 들어 있지 않음).

## 코드 구성

| 파일 | 하는 일 |
| --- | --- |
| `FaceTools` | MCP 도구 `face_makeup_check`, `teeth_check` (`@Tool`). 실패하면 `{"error": {"code", "message"}}` |
| `makeup/MakeupAnalyzer` | 위 대칭 규칙 |
| `makeup/OnnxFaceParser` | 얼굴 분할 모델(BiSeNet ONNX)로 눈썹 픽셀 고르기. 만드는 법 `training/export_face_parsing.py` |
| `makeup/MakeupPixels` | 사진에서 그려진 눈썹·입술 모양 재기(나누지 못하면 null → 윤곽점) |
| `makeup/FaceFrame` | 두 눈 기준 얼굴 좌표 |
| `makeup/MakeupCheckService` | 여러 장 중간값으로 합치기 |
| `training/` | 기준값 학습(Python): `features.py` → `train.py` |
| `teeth/TeethCheckService` | 치아: 구도·밝기 확인 → 모델 → 확률 기준 |
| `teeth/MouthCrop` | 학습과 같은 입 자르기·정규화 |
| `teeth/OnnxTeethClassifier` | ONNX Runtime으로 모델 실행 |
| `image/Images` | base64 풀기(JPEG·PNG, 8MB), sRGB → CIELAB |

## 오류 코드

`no_image`, `bad_image`, `bad_request`(윤곽점이 비율이 아님, 6장 이상), `no_landmarks`(윤곽점이 없거나 모자람 → 다시 찍기),
`not_ready`(teeth_check 모델 파일이 없음), `upstream_failed`.
