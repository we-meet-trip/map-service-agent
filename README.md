# map-service-agent

MAP 서비스의 추천 엔진. FastAPI + LangGraph + Gemini 2.5 Flash.

## 역할

- LangGraph 노드 그래프 구성·실행 (입력 파싱 → 날씨 조회 → 장소 검색 → 보강 → 점수·랭킹 → 추천 이유 생성)
- Gemini 2.5 Flash 직접 호출 (요청당 LLM ≤ 3회)
- hub의 결정적 룰 엔진을 통과한 후보 풀 안에서 LLM이 자율 선택
- 자체 데이터 소유 없음 — POI · 날씨 · 경로는 hub의 REST API로 조회
- stateless — 일정 단위 정책은 user-BFF가 강제, agent는 요청마다 독립 실행
- LangGraph 체크포인트는 PostgreSQL `langgraph` schema에 SDK가 자동 영속화

## 폴더 구조

```
map-service-agent/
├── Dockerfile                    Python 3.12-slim multi-stage
├── requirements.txt              핵심 의존성 (fastapi · langgraph · google-genai 등)
└── app/
    ├── __init__.py
    ├── main.py                   FastAPI 진입점 + /health
    ├── graph/agent_graph.py      LangGraph StateGraph 정의
    ├── nodes/agent_nodes.py      그래프 노드 함수
    ├── clients/agent_clients.py  hub REST 클라이언트 · Gemini 클라이언트
    └── schemas/agent_schemas.py  요청/응답 Pydantic 모델
```

## 실행 (단독 빌드 — 통합 실행은 map-service-infra 사용 권장)

```bash
docker build -t map-service-agent:dev .
docker run --rm -p 8000:8000 --env-file ../map-service-infra/.env map-service-agent:dev
curl http://127.0.0.1:8000/health
```

## 의존성

- Python 3.12
- PostgreSQL (LangGraph 체크포인터)
- Redis (token-bucket · Streams)
- hub 서비스 (HTTP)
- Gemini API 키

## License

MIT — see [LICENSE](LICENSE).
