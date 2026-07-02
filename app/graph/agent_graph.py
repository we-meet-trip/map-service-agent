"""LangGraph StateGraph 빌더.

`build_graph()` 는 `app/nodes/agent_nodes.py` 의 추천 파이프라인 노드를
연결한 **컴파일된 StateGraph** 를 반환한다. `app/main.py` lifespan 이 1회
호출해 `app.state.graph` 에 저장하고, `_run_job` 이
`graph.ainvoke({"job_id": ..., "request": ...})` 로 실행한다.

파이프라인 (PoC 축약 — SoT §6.1 C1 흐름의 골격. hub /v1/places·/v1/rules/*
호출과 score_and_rank/llm_reason 세분 노드는 운영단계 확장 대상):
  parse_input -> fetch_weather -> search_places -> recommend_places
  -> recommend_route -> build_payload -> publish_done

에러 라우팅 (`_route_after`):
  parse_input / fetch_weather / recommend_places 중 어느 노드든
  `state["error"]` 를 설정하면 후속 노드를 건너뛰고 `build_payload` 로
  단축 분기한다. recommend_route 이후는 성공/실패 모두 `build_payload` 로
  수렴하므로, `publish_done` 은 항상 호출되어 성공/실패 한쪽
  `JobDonePayload` 를 Redis Streams 에 게시한다.
  (각 노드도 `state["error"]` 를 보고 자체 no-op 하므로, 단축 분기는
  불필요한 LLM/HTTP 호출을 줄이는 최적화다.)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.agent_nodes import (
    AgentState,
    build_payload,
    fetch_weather,
    parse_input,
    publish_done,
    recommend_places,
    recommend_route,
    search_places,
)


def _route_after(next_node: str):
    """`state["error"]` 가 있으면 build_payload 로, 없으면 next_node 로 분기."""

    def _router(state: AgentState) -> str:
        return "build_payload" if state.get("error") else next_node

    return _router


def build_graph():
    """추천 파이프라인 노드를 연결한 컴파일된 StateGraph 인스턴스를 반환한다."""
    graph = StateGraph(AgentState)
    graph.add_node("parse_input", parse_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("search_places", search_places)
    graph.add_node("recommend_places", recommend_places)
    graph.add_node("recommend_route", recommend_route)
    graph.add_node("build_payload", build_payload)
    graph.add_node("publish_done", publish_done)

    graph.add_edge(START, "parse_input")
    graph.add_conditional_edges(
        "parse_input",
        _route_after("fetch_weather"),
        {"fetch_weather": "fetch_weather", "build_payload": "build_payload"},
    )
    graph.add_conditional_edges(
        "fetch_weather",
        _route_after("search_places"),
        {
            "search_places": "search_places",
            "build_payload": "build_payload",
        },
    )
    # search_places 는 실패해도 error 를 세우지 않고 저하 표시로 폴백하므로
    # 항상 recommend_places 로 진행한다.
    graph.add_edge("search_places", "recommend_places")
    graph.add_conditional_edges(
        "recommend_places",
        _route_after("recommend_route"),
        {
            "recommend_route": "recommend_route",
            "build_payload": "build_payload",
        },
    )
    graph.add_edge("recommend_route", "build_payload")
    graph.add_edge("build_payload", "publish_done")
    graph.add_edge("publish_done", END)
    return graph.compile()
