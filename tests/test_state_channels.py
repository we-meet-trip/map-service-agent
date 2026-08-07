"""AgentState 선언과 컴파일된 그래프 채널의 일치 검증.

왜 필요한가: LangGraph 는 AgentState TypedDict 에 선언된 키로만 채널을
만든다. 노드가 선언되지 않은 키를 state 에 써도 다음 노드로 전달되지 않고
조용히 사라진다(예외도 경고도 없다). 실제로 `reviews` 가 이 방식으로
recommend_places → llm_reason 구간에서 소실되어, 이유 생성 프롬프트의
review_snippets 가 항상 빈 배열이 된 결함이 있었다.

노드 단위 테스트는 이 결함을 잡지 못한다. 노드를 직접 호출하면 같은 dict 를
in-place 로 다루므로 미선언 키도 그대로 보이기 때문이다. 그래서 컴파일된
그래프의 채널 집합을 직접 확인한다.
"""
from __future__ import annotations

from app.graph.agent_graph import build_graph
from app.nodes.agent_nodes import AgentState


def _declared_keys() -> set[str]:
    return set(AgentState.__annotations__.keys())


def _channel_keys() -> set[str]:
    """컴파일된 그래프의 채널 중 LangGraph 내부 채널을 제외한 것."""
    compiled = build_graph(checkpointer=None)
    return {
        name
        for name in compiled.channels
        if not name.startswith("__") and not name.startswith("branch:")
    }


def test_every_declared_field_becomes_a_channel():
    """AgentState 에 선언한 필드는 전부 그래프 채널이 되어야 한다."""
    missing = _declared_keys() - _channel_keys()
    assert not missing, f"채널로 만들어지지 않은 선언 필드: {sorted(missing)}"


def test_reviews_and_scores_are_channels():
    """노드 경계를 넘어야 하는 두 키가 채널에 존재하는지 명시 검증.

    reviews 는 recommend_places 가 쓰고 llm_reason 이 읽으므로 채널이 없으면
    기능이 조용히 죽는다. scores 는 score_and_rank 가 기록하는 관측용 키다.
    """
    channels = _channel_keys()
    assert "reviews" in channels
    assert "scores" in channels


def test_state_written_keys_are_declared():
    """노드가 state 에 쓰는 키가 전부 AgentState 에 선언돼 있는지 확인.

    agent_nodes.py 원문에서 `state["..."] = ` 패턴을 뽑아 선언 집합과 대조한다.
    새 상태 키를 추가하면서 TypedDict 갱신을 잊는 회귀를 여기서 막는다.
    """
    import inspect
    import re

    from app.nodes import agent_nodes

    src = inspect.getsource(agent_nodes)
    written = set(re.findall(r'state\[\s*"([a-z_]+)"\s*\]\s*=', src))
    undeclared = written - _declared_keys()
    assert not undeclared, (
        f"state 에 쓰지만 AgentState 에 선언되지 않은 키: {sorted(undeclared)}"
    )
