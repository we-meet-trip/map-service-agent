"""LangGraph StateGraph 빌더.

`build_graph()` 는 `app/nodes/agent_nodes.py` 의 추천 파이프라인 노드를
연결한 **컴파일된 StateGraph** 를 반환한다. `app/main.py` lifespan 이 1회
호출해 `app.state.graph` 에 저장하고, `_run_job` 이
`graph.ainvoke({"job_id": ..., "request": ...}, config=...)` 로 실행한다.

파이프라인 (SoT §6.1 C1 흐름):
  parse_input -> fetch_weather -> search_places
  -> rules_filter -> score_and_rank      (hub /v1/rules/* 대기 — no-op)
  -> recommend_places -> recommend_route
  -> llm_reason                          (정상 경로 한정, 실패 시 degrade)
  -> build_payload -> publish_done

에러 라우팅 (`_route_after`):
  parse_input / fetch_weather / recommend_places / recommend_route 중
  어느 노드든 `state["error"]` 를 설정하면 후속 노드를 건너뛰고
  `build_payload` 로 단축 분기한다. llm_reason 은 error 를 만들지 않는
  enhancement 노드(실패 시 degrade)이므로, 그 뒤는 단순 엣지로
  `build_payload` 에 수렴한다. `publish_done` 은 항상 호출되어
  성공/실패 한쪽 `JobDonePayload` 를 Redis Streams 에 게시한다.
  (각 노드도 `state["error"]` 를 보고 자체 no-op 하므로, 단축 분기는
  불필요한 LLM/HTTP 호출을 줄이는 최적화다.)

체크포인터 (SoT B5):
  `build_graph(checkpointer=...)` 로 주입한다. None 이면 체크포인트 없이
  컴파일한다(단위 테스트·CHECKPOINT_ENABLED=false). 주입 시 `_run_job`
  은 `config={"configurable": {"thread_id": job_id}}` 로 invoke 해야 한다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.agent_nodes import (
    AgentState,
    build_payload,
    fetch_weather,
    llm_reason,
    parse_input,
    publish_done,
    recommend_places,
    recommend_route,
    rules_filter,
    score_and_rank,
    search_places,
)


def _route_after(next_node: str):
    """`state["error"]` 가 있으면 build_payload 로, 없으면 next_node 로 분기."""

    def _router(state: AgentState) -> str:
        return "build_payload" if state.get("error") else next_node

    return _router


def build_graph(checkpointer=None):
    """추천 파이프라인 노드를 연결한 컴파일된 StateGraph 인스턴스를 반환한다.

    checkpointer: LangGraph 체크포인터(예: AsyncPostgresSaver). None 이면
                  체크포인트 없이 컴파일한다.
    """
    graph = StateGraph(AgentState)
    graph.add_node("parse_input", parse_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("search_places", search_places)
    graph.add_node("rules_filter", rules_filter)
    graph.add_node("score_and_rank", score_and_rank)
    graph.add_node("recommend_places", recommend_places)
    graph.add_node("recommend_route", recommend_route)
    graph.add_node("llm_reason", llm_reason)
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
    # search_places 는 실패해도 error 를 세우지 않고 저하 표시로 폴백하고,
    # rules_filter / score_and_rank 는 no-op 자리 노드라 error 를 만들지
    # 않으므로 단순 엣지로 recommend_places 까지 직진한다.
    graph.add_edge("search_places", "rules_filter")
    graph.add_edge("rules_filter", "score_and_rank")
    graph.add_edge("score_and_rank", "recommend_places")
    graph.add_conditional_edges(
        "recommend_places",
        _route_after("recommend_route"),
        {
            "recommend_route": "recommend_route",
            "build_payload": "build_payload",
        },
    )
    # recommend_route 성공 시에만 llm_reason(이유·옷차림) 을 수행한다 —
    # 실패 경로에서 LLM 호출 1회를 아끼고 곧장 실패 페이로드로 간다.
    graph.add_conditional_edges(
        "recommend_route",
        _route_after("llm_reason"),
        {
            "llm_reason": "llm_reason",
            "build_payload": "build_payload",
        },
    )
    graph.add_edge("llm_reason", "build_payload")
    graph.add_edge("build_payload", "publish_done")
    graph.add_edge("publish_done", END)
    return graph.compile(checkpointer=checkpointer)
