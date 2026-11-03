# mcp-face-check

**face_check MCP 서버** · 패션 에이전트(시각장애인용 코디 도우미)의 한 부분입니다.

얼굴 사진을 찍으면 화장 대칭 여부, 이에 낀 음식물 같은 것을 확인합니다.

## 도구 (안)

- `check_face(image)`: 화장 대칭, 이에 낀 것, 머리 정돈 상태와 위치

## 지금 상태

없음. 앱은 "준비 중"이라고 답합니다. 사진 분석 방식은 `mcp-clothes-state`(Gemini 비전 + 프롬프트)를 그대로 본떠 만들면 됩니다.

## 할 일

- [ ] MCP 서버 만들기
- [ ] Gemini 비전 프롬프트 작성 (판단 말고 보이는 사실만)
- [ ] 앱에 얼굴 촬영 안내(전면 카메라, 거리) 추가

## 전체 구조

```
application(fashion-agent) ─REST→ fashion-agent-gateway ─→ fashion-agent-agent ─MCP→ mcp-weather / mcp-schedule / mcp-clothes-state / mcp-face-check / mcp-closet
                                                                                              ↓
                                                                                              DB (계정·회원정보·일정·옷장)
```

## 브랜치

GitFlow를 따릅니다. `develop`에서 `feature/…` 브랜치를 따서 PR로 합칩니다. 규칙은 앱 레포의 `docs/GITFLOW.md`를 봅니다.

## 비밀 값

API 키와 DB 접속 정보는 레포에 넣지 않습니다. 배포 환경의 비밀 값(Secret)으로만 둡니다.
