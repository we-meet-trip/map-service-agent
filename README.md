# map-service-agent

MAP 서비스의 추천 엔진. FastAPI + LangGraph + Gemini 2.5 Flash.

## 역할

- LangGraph 노드 그래프 구성·실행 (입력 파싱 → 날씨 조회 → 장소 검색 → 보강 → 점수·랭킹 → 추천 이유 생성)
- Gemini 2.5 Flash 직접 호출 (추천 요청당 LLM ≤ 3회, 블로그 요약은 별도 파이프라인)
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
    ├── graph/agent_graph.py      추천 파이프라인 StateGraph 정의
    ├── graph/summary_graph.py    블로그 요약 파이프라인 StateGraph 정의
    ├── nodes/agent_nodes.py      추천 파이프라인 노드 함수
    ├── nodes/summary_nodes.py    요약 파이프라인 노드 함수
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


## 출시 품질 계약

- 학습은 HOLD이다. `TRAINING_CAPTURE_ENABLED=false`가 기본이며 학습 전용 JSON 생성과
  Streams 발행을 함께 차단한다. 완료 결과/진행 이벤트와 내용 없는 LLM 토큰 사용량 계측은 유지한다.
- 실측 출처 후보가 없거나 선택 결과가 부족하면 실패한다. LLM 장소/좌표 창작 fallback은 없다.
- `stage=route`는 입력 장소 배열 순서를 보존하며 `optimize=true`를 명시한 경우만 최적화한다.
  선택 장소는 Hub 검색으로 실제 출처를 확인하고 원본 좌표를 사용한다. 확인할 수 없으면 실패한다.
- 수동 일정의 시간 초과는 장소를 몰래 삭제하지 않고 실패한다. 자동 일정을 줄인 뒤에도
  시간이 초과하면 성공/trimmed로 발행하지 않는다.
- 봉인에서 꺼낸 장소도 전체 AgentRequest 검증을 다시 받아 목록 1~10개 상한을 우회할 수 없다.
- `python -m pytest -q`는 외부 API 대역으로 회귀를 검사한다. `eval.run`은 저장된 완료 결과만
  평가하며 학습 수집을 요구하지 않는다. 새 검증을 통과하는 실제 평가 결과 재수집은 별도 단계다.
