"""요약 파이프라인 StateGraph 빌더.

`build_summary_graph()` 는 `app/nodes/summary_nodes.py` 의 노드를 연결한
**컴파일된 StateGraph** 를 반환한다. 추천 파이프라인과는 별개의 그래프이며,
장소별 블로그 후기 두 줄 요약만 담당한다.

파이프라인:
  prepare_places -> summarize -> collect_bullets

근거 분기 (`_route_after_prepare`):
  쓸 만한 후기를 가진 장소가 하나도 없으면 요약할 것이 없다. 이때는
  모델을 부르지 않고 곧장 끝낸다 — 호출 예산을 지키는 지점이다.

체크포인터를 받지 않는다:
  요약은 한 요청 안에서 시작하고 끝나 이어받을 것이 없다. 체크포인터를
  붙이면 회수할 잡 없는 기록이 장소를 누를 때마다 쌓인다.

모듈 수준에서 1회 컴파일해 재사용한다:
  주입할 것이 없어 앱 기동 훅을 기다릴 이유가 없고, 요약 경로가 기동
  순서와 무관하게 동작한다. 컴파일은 외부 자원을 건드리지 않는 순수
  연산이라 임포트 시점에 수행해도 안전하다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.summary_nodes import (
    SummaryState,
    collect_bullets,
    prepare_places,
    summarize,
)


def _route_after_prepare(state: SummaryState) -> str:
    """뷰가 하나도 없으면 모델을 부르지 않고 끝낸다."""
    return "summarize" if state.get("views") else END


def build_summary_graph():
    """요약 파이프라인 노드를 연결한 컴파일된 StateGraph 를 반환한다."""
    graph = StateGraph(SummaryState)
    graph.add_node("prepare_places", prepare_places)
    graph.add_node("summarize", summarize)
    graph.add_node("collect_bullets", collect_bullets)

    graph.add_edge(START, "prepare_places")
    graph.add_conditional_edges(
        "prepare_places",
        _route_after_prepare,
        {"summarize": "summarize", END: END},
    )
    graph.add_edge("summarize", "collect_bullets")
    graph.add_edge("collect_bullets", END)
    return graph.compile()


# 프로세스 1회 컴파일 후 재사용한다.
SUMMARY_GRAPH = build_summary_graph()
