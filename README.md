# mcp-face-check

**얼굴 확인 MCP 서버** · 패션 에이전트(시각장애인용 코디 도우미)의 한 부분입니다.

"얼굴 확인해 줘" → agent가 "얼굴 화장을 볼까요, 이를 확인할까요?"라고 묻고, 답에 맞는 도구 하나를 부릅니다.
입력·출력 약속은 앱 레포 [`docs/contracts/mcp-tools.md`](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md#mcp-face-check)에 있습니다.

## 도구

| 도구 | 담당 | 하는 일 | 상태 |
| --- | --- | --- | --- |
| `face_makeup_check(imageBase64, landmarks, frames)` | 박창근 | 앱 ML Kit 얼굴 윤곽점(위치) + 사진 픽셀(그려진 모양·색) → 좌우 비교(눈썹 높이·꼬리·길이·두께, 입꼬리·입술 선, 볼·입술 색). 기준값은 라벨 셀카로 학습 | 동작 |
| `teeth_check(imageBase64, mouthBox)` | 정지원 | 입 안쪽 박스로 잘라 우리가 학습한 분류 모델(MobileNetV3-Small, ONNX)을 서버에서 돌림 → 이 사이에 낀 것 있음/없음/애매함 | 동작 (합성 데이터로 학습, 실제 사진 평가 전) |

- 왼쪽·오른쪽은 **셀카 화면 기준**(사진 왼쪽 = 사용자 왼쪽)입니다. 앱이 사진과 윤곽점을 셀카 방향으로 뒤집어 보냅니다.
- 앱은 얼굴을 실시간으로 비추며 구도를 안내하고, 잘 잡혀 멈춘 1초 동안 3~5장을 보냅니다. 서버는 장마다 잰 값의 중간값으로 판단합니다.
- GPT 같은 외부 AI를 쓰지 않습니다. 얼굴·치아 사진은 개인정보라 이 서버 밖으로 나가지 않습니다. 얼굴 점은 이미 학습된 모델(ML Kit)이 찾고, "얼마나 다르면 비대칭인가"는 [training/](training/README.md)에서 라벨 붙인 셀카로 학습합니다.
- 사진은 판단에만 쓰고 저장하지 않습니다.

개발 방법, 대칭 규칙과 기준값은 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)를 보세요.

## 전체 구조

```
application(fashion-agent) ─REST→ fashion-agent-gateway ─→ fashion-agent-agent ─MCP→ mcp-weather / mcp-schedule / mcp-clothes-state / mcp-face-check / mcp-closet / mcp-helper
```

## 브랜치

GitFlow를 따릅니다. `develop`에서 `feature/…` 브랜치를 따서 PR로 합칩니다. 규칙은 앱 레포의 `docs/GITFLOW.md`를 봅니다.

## 비밀 값

이 서버는 비밀 값이 필요 없습니다. 학습용 셀카(training/data)는 레포에 올리지 않습니다.
