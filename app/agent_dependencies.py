"""싱글톤 의존성 컨테이너.

lifespan 에서 인스턴스를 만들어 모듈 전역 슬롯에 보관하고,
LangGraph 노드들이 본 모듈을 통해 접근한다.

설계 의도:
  - LangGraph 노드 함수는 동기/비동기 일반 함수이고, FastAPI 의 의존성
    주입(Depends) 을 거치지 않는다. 그래서 노드들이 외부 클라이언트에
    접근할 통로가 필요한데, 본 모듈은 모듈 전역 슬롯(`_hub_client`,
    `_streams`, `_gemini`) + setter/getter 로 그 역할을 수행한다.
  - 노드 함수들은 함수 본문 안에서 본 모듈을 import 하고 getter 를 호출한다
    (순환 import 방지 + 초기화 순서 보장).
  - `reset_all()` 은 lifespan 종료 시 슬롯을 None 으로 비워, 단위 테스트나
    재기동 시 stale 상태를 막는다.
"""
from __future__ import annotations

from app.clients.agent_clients import (
    GeminiClient,
    HubClient,
    StreamsPublisher,
)

# ─── 모듈 전역 슬롯 ─────────────────────────────────────────────
# lifespan 이 setter 로 채우고, 노드들이 getter 로 읽는다.
# 초기값 None 은 "아직 주입되지 않음" 을 의미한다.
_hub_client: HubClient | None = None
_streams: StreamsPublisher | None = None
_status_streams: StreamsPublisher | None = None
_gemini: GeminiClient | None = None


def set_hub_client(client: HubClient) -> None:
    """HubClient 인스턴스를 전역 슬롯에 주입.

    호출처: `app/main.py` lifespan(부팅 단계).
    """
    global _hub_client
    _hub_client = client


def get_hub_client() -> HubClient:
    """전역 슬롯에서 HubClient 를 꺼낸다.

    슬롯이 비어 있으면 `RuntimeError("HubClient is not initialized")`.
    호출처: `app/nodes/agent_nodes.py` 의 `fetch_weather` 노드.
    """
    if _hub_client is None:
        raise RuntimeError("HubClient is not initialized")
    return _hub_client


def set_streams_publisher(publisher: StreamsPublisher) -> None:
    """StreamsPublisher 인스턴스를 전역 슬롯에 주입.

    호출처: `app/main.py` lifespan(부팅 단계).
    """
    global _streams
    _streams = publisher


def get_streams_publisher() -> StreamsPublisher:
    """전역 슬롯에서 StreamsPublisher 를 꺼낸다.

    슬롯이 비어 있으면 `RuntimeError("StreamsPublisher is not initialized")`.
    호출처:
      - `app/nodes/agent_nodes.py` 의 `publish_done` 노드.
      - `app/main.py` 의 `_publish_failure`.
    """
    if _streams is None:
        raise RuntimeError("StreamsPublisher is not initialized")
    return _streams


def set_status_publisher(publisher: StreamsPublisher) -> None:
    """status 스트림(agent:jobs:status) 발행자를 전역 슬롯에 주입.

    호출처: `app/main.py` lifespan(부팅 단계).
    """
    global _status_streams
    _status_streams = publisher


def get_status_publisher() -> StreamsPublisher:
    """전역 슬롯에서 status 발행자를 꺼낸다.

    슬롯이 비어 있으면 `RuntimeError("StatusPublisher is not initialized")`.
    호출처: `app/nodes/agent_nodes.py` 의 `_emit_stage` — 예외는 호출측이
    삼키므로(진행 이벤트는 best-effort), 미주입 환경(단위 테스트)에서도
    잡 실행이 깨지지 않는다.
    """
    if _status_streams is None:
        raise RuntimeError("StatusPublisher is not initialized")
    return _status_streams


def set_gemini_client(client: GeminiClient) -> None:
    """GeminiClient 인스턴스를 전역 슬롯에 주입.

    호출처: `app/main.py` lifespan(부팅 단계).
    """
    global _gemini
    _gemini = client


def get_gemini_client() -> GeminiClient:
    """전역 슬롯에서 GeminiClient 를 꺼낸다.

    슬롯이 비어 있으면 `RuntimeError("GeminiClient is not initialized")`.
    호출처: `app/nodes/agent_nodes.py` 의
      `recommend_places`, `recommend_route` 노드.
    """
    if _gemini is None:
        raise RuntimeError("GeminiClient is not initialized")
    return _gemini


def reset_all() -> None:
    """모든 전역 슬롯을 None 으로 되돌린다.

    호출처: `app/main.py` lifespan 의 종료 finally 블록.
    """
    global _hub_client, _streams, _status_streams, _gemini
    _hub_client = None
    _streams = None
    _status_streams = None
    _gemini = None
