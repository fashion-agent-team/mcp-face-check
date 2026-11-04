# mcp-face-check

**face-check MCP 서버 · 얼굴 확인 (개별 기능)** · 패션 에이전트(시각장애인용 음성 AI 옷 추천)의 한 부분입니다.

담당: (미정)

## 한눈에

- 포트: 8094, 엔드포인트: `http://mcp-face-check:8094/mcp` (Streamable HTTP)
- 기술: Spring Boot, Spring AI MCP Server, 대칭 규칙(앱 ML Kit 윤곽점), 치아 이물질 분류 모델(MobileNet 파인튜닝 → ONNX Runtime Java)
- 상태 확인: `GET /actuator/health`

## MCP 도구

| 도구 | 하는 일 |
| --- | --- |
| `face_check` | 사진(+앱 ML Kit 윤곽점) → 구도, 조명, 화장 대칭, 이에 낀 것 |

입출력 JSON은 [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)를 그대로 따릅니다.

## 이 레포가 지킬 것

- **개별 기능**입니다("얼굴 확인해 줘"). 옷 추천 흐름에서는 부르지 않습니다.
- 대칭은 학습 없이 좌우 윤곽점 위치와 볼·입술 색(LAB)을 비교해 계산합니다. 학습은 치아 분류 하나만.
- 사진은 저장하지 않습니다.
- 앱은 카메라를 켜 두고 ML Kit로 얼굴 구도를 실시간 안내하다가, 잘 들어오면 한 장을 자동 전송합니다.

## 할 일

- [ ] 뼈대 + `/mcp` + Dockerfile
- [ ] 대칭 규칙 (윤곽점 입력)
- [ ] 치아 사진 모으기·라벨링, 분류 모델 학습(`training/`), ONNX 실행
- [ ] (모델 전) Gemini 폴백

## 공통 약속

모든 레포가 지키는 약속은 앱 레포에 있습니다. **코드보다 약속 문서를 먼저 고칩니다.**

- [서비스 간 약속](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/README.md): 공통 규칙, 포트, 인증(`X-User-Id`), 오류 모양, DB 스키마, 옷 추천 흐름
- [MCP 도구 입출력](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/mcp-tools.md)
- [앱 ↔ 게이트웨이 API (OpenAPI)](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/contracts/openapi.yaml)
- [아키텍처와 기술 스택](https://github.com/fashion-agent-team/fashion-agent/blob/develop/docs/ARCHITECTURE.md)

## 버전

| 항목 | 버전 |
| --- | --- |
| Java | 21 |
| Spring Boot | 4.0.9 |
| Spring Cloud | 2025.1.4 |
| Spring AI (BOM) | 2.0.2 |

## 브랜치

GitFlow를 따릅니다. `develop`에서 `feature/…` 브랜치를 따서 PR로 합칩니다. 규칙은 앱 레포의 `docs/GITFLOW.md`를 봅니다.

## 비밀 값

API 키와 DB 비밀번호는 레포에 넣지 않습니다. 로컬은 `.env`(커밋 금지), 배포는 서버 환경 변수로 둡니다. 목록은 앱 레포 `docs/infra/.env.example`에 있습니다.
