"""LangGraph StateGraph 빌더.

`build_graph()` 는 `app/nodes/agent_nodes.py` 의 추천 파이프라인 노드를
연결한 **컴파일된 StateGraph** 를 반환한다. `app/main.py` lifespan 이 1회
호출해 `app.state.graph` 에 저장하고, `_run_job` 이
`graph.ainvoke({"job_id": ..., "request": ...}, config=...)` 로 실행한다.

파이프라인:
  parse_input -> fetch_weather -> plan_strategy -> search_places
  -> rules_filter -> score_and_rank      (hub /v1/rules/*)
  -> recommend_places -> recommend_route
  -> build_timeline -> fit_time_budget   (hub /v1/rules/estimate/dwell)
  -> llm_reason                          (초기 추천·재탐색 정상 경로만)
  -> build_payload -> publish_done

시간축:
  build_timeline 이 체류시간과 방문 시각을 계산하고, 하루 활동 시간을
  넘긴 일차가 있으면 fit_time_budget 이 결정적으로 줄인다. 두 노드 모두
  LLM 을 쓰지 않으며 충족할 수 없는 시간표는 실패로 표시한다.

블로그 후기 요약은 이 그래프에 없다. 요약은 일정에 딸려 나오는 값이 아니라
장소를 눌렀을 때 필요한 정보이므로 별도 파이프라인
(`app/graph/summary_graph.py`)이 맡는다.

stage="route" 분기:
  사용자가 방문지를 이미 골라 온 요청은 탐색·선정 구간이 통째로 불필요하다.
  fetch_weather 다음에서 갈라져 load_given_places 로 장소를 세운 뒤 곧장
  recommend_route 에 합류한다. 시간표 계산 후 생성 설명을 건너뛰고
  build_payload 로 간다. 수동 경로와 명시적 optimize의 LLM 호출은 0회다.

LLM 호출:
  초기 추천·재탐색은 선정 1 + 이유 1 = 2회다. 방문 순서와 구간은 recommend_route 가 좌표로
  계산하므로 모델을 부르지 않는다. 요청당 상한 3 에서 1회가 남아, 앞 단계가
  형식을 한 번 어겨도 교정 재시도로 복구할 수 있다.

에러 라우팅 (`_route_after`):
  parse_input / fetch_weather / recommend_places / recommend_route 중
  어느 노드든 `state["error"]` 를 설정하면 후속 노드를 건너뛰고
  `build_payload` 로 단축 분기한다. llm_reason 은 error 를 만들지 않는
  enhancement 노드(실패 시 degrade)이므로, 그 뒤는 단순 엣지로
  `build_payload` 에 수렴한다. `publish_done` 은 항상 호출되어 성공/실패
  한쪽 `JobDonePayload` 를 Redis Streams 에 게시한다.
  (각 노드도 `state["error"]` 를 보고 자체 no-op 하므로, 단축 분기는
  불필요한 LLM/HTTP 호출을 줄이는 최적화다.)

체크포인터:
  `build_graph(checkpointer=...)` 로 주입한다. None 이면 체크포인트 없이
  컴파일한다(단위 테스트·CHECKPOINT_ENABLED=false). 주입 시 `_run_job`
  은 `config={"configurable": {"thread_id": job_id}}` 로 invoke 해야 한다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.agent_nodes import (
    AgentState,
    build_payload,
    build_timeline,
    fetch_weather,
    fit_time_budget,
    llm_reason,
    load_given_places,
    parse_input,
    plan_strategy,
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


def _route_after_weather(state: AgentState) -> str:
    """날씨 다음 갈림길 — 장소를 찾을지, 받은 장소를 쓸지 고른다."""
    if state.get("error"):
        return "build_payload"
    if state["request"].stage == "route":
        return "load_given_places"
    return "plan_strategy"


def _route_after_time_budget(state: AgentState) -> str:
    """Selected-place routes need geometry and a timetable, not generated recommendations."""
    if state.get("error") or state["request"].stage == "route":
        return "build_payload"
    return "llm_reason"


def build_graph(checkpointer=None):
    """추천 파이프라인 노드를 연결한 컴파일된 StateGraph 인스턴스를 반환한다.

    checkpointer: LangGraph 체크포인터(예: AsyncPostgresSaver). None 이면
                  체크포인트 없이 컴파일한다.
    """
    graph = StateGraph(AgentState)
    graph.add_node("parse_input", parse_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("plan_strategy", plan_strategy)
    graph.add_node("load_given_places", load_given_places)
    graph.add_node("search_places", search_places)
    graph.add_node("rules_filter", rules_filter)
    graph.add_node("score_and_rank", score_and_rank)
    graph.add_node("recommend_places", recommend_places)
    graph.add_node("recommend_route", recommend_route)
    graph.add_node("build_timeline", build_timeline)
    graph.add_node("fit_time_budget", fit_time_budget)
    graph.add_node("llm_reason", llm_reason)
    graph.add_node("build_payload", build_payload)
    graph.add_node("publish_done", publish_done)

    graph.add_edge(START, "parse_input")
    graph.add_conditional_edges(
        "parse_input",
        _route_after("fetch_weather"),
        {"fetch_weather": "fetch_weather", "build_payload": "build_payload"},
    )
    # 사용자가 방문지를 골라 온 요청(stage=route)은 탐색·선정 구간을 건너뛴다.
    graph.add_conditional_edges(
        "fetch_weather",
        _route_after_weather,
        {
            "plan_strategy": "plan_strategy",
            "load_given_places": "load_given_places",
            "build_payload": "build_payload",
        },
    )
    # 전략은 계산으로 세우므로 실패할 일이 없다 — 단순 엣지로 탐색에 잇는다.
    graph.add_edge("plan_strategy", "search_places")
    # 장소가 이미 정해졌으므로 곧장 동선 단계로 합류한다.
    graph.add_conditional_edges(
        "load_given_places",
        _route_after("recommend_route"),
        {
            "recommend_route": "recommend_route",
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
    # recommend_route 성공 시에만 시간축을 세운다 — 실패 경로에서 hub 호출과
    # LLM 호출을 아끼고 곧장 실패 페이로드로 간다.
    graph.add_conditional_edges(
        "recommend_route",
        _route_after("build_timeline"),
        {
            "build_timeline": "build_timeline",
            "build_payload": "build_payload",
        },
    )
    # Time overflow is checked before choosing the optional explanation branch.
    graph.add_edge("build_timeline", "fit_time_budget")
    # Manual routing and explicit optimization are deterministic. Do not send
    # selected places or trip context to Gemini merely to explain that route.
    graph.add_conditional_edges(
        "fit_time_budget", _route_after_time_budget,
        {"llm_reason": "llm_reason", "build_payload": "build_payload"},
    )
    # llm_reason 은 실패해도 error 를 세우지 않고 저하 표시만 남기므로
    # 단순 엣지로 합류 지점까지 직진한다.
    graph.add_edge("llm_reason", "build_payload")
    graph.add_edge("build_payload", "publish_done")
    graph.add_edge("publish_done", END)
    return graph.compile(checkpointer=checkpointer)
