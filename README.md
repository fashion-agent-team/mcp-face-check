# mcp-face-check

**얼굴 확인 MCP 서버** · 패션 에이전트(시각장애인용 코디 도우미)의 한 부분입니다.

"얼굴 확인해 줘" → agent가 "얼굴 화장을 볼까요, 이를 확인할까요?"라고 묻고, 답에 맞는 도구 하나를 부릅니다.
입력·출력 약속은 앱 레포 [`docs/contracts/mcp-tools.md`](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md#mcp-face-check)에 있습니다.

## 도구

| 도구 | 담당 | 하는 일 | 상태 |
| --- | --- | --- | --- |
| `face_makeup_check(imageBase64, landmarks)` | 박창근 | 앱 ML Kit 얼굴 윤곽점 → 대칭 규칙(눈썹 높이·꼬리·길이·두께, 입꼬리·입술 선, 볼·입술 색 좌우) | 동작 |
| `teeth_check(imageBase64, mouthBox)` | 정지원 | 입 영역을 잘라 이 사이 이물질 분류(MobileNet 파인튜닝, ONNX) | 자리만 있음(`not_ready`) |

- 왼쪽·오른쪽은 **셀카 화면 기준**(사진 왼쪽 = 사용자 왼쪽)입니다. 앱이 사진과 윤곽점을 셀카 방향으로 뒤집어 보냅니다.
- 사진은 판단에만 쓰고 저장하지 않습니다.
- 윤곽점이 없을 때만 GPT 비전(OpenAI)으로 봅니다(`source: "gpt"`). `OPENAI_API_KEY`가 없으면 `no_landmarks` 오류를 돌려줍니다.

## 치아 분류 붙이는 곳 (정지원 님)

`teeth/TeethClassifier` 인터페이스를 구현한 `@Component`를 하나 넣으면 `teeth_check`가 그걸 씁니다.
입 영역 자르기(mouthBox + 여유 15%), 구도(잘림·크기·벌림)·밝기 확인, 결과 JSON 모양은 `TeethCheckService`가 이미 합니다.
분류기는 잘린 입 사진을 받아 `TeethFinding(foreignObject, confidence, note)`만 돌려주면 됩니다.

개발 방법, 대칭 규칙과 기준값은 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)를 보세요.

## 전체 구조

```
application(fashion-agent) ─REST→ fashion-agent-gateway ─→ fashion-agent-agent ─MCP→ mcp-weather / mcp-schedule / mcp-clothes-state / mcp-face-check / mcp-closet / mcp-helper
```

## 브랜치

GitFlow를 따릅니다. `develop`에서 `feature/…` 브랜치를 따서 PR로 합칩니다. 규칙은 앱 레포의 `docs/GITFLOW.md`를 봅니다.

## 비밀 값

OpenAI 키는 레포에 넣지 않습니다. 배포 환경의 비밀 값(.env, 커밋 금지)으로만 둡니다.
