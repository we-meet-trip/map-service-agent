# LangGraph StateGraph 정의를 본 모듈에 구현한다.
# 추천 파이프라인 노드를 연결하여 단일 그래프 인스턴스로 빌드한다:
#   parse_input -> fetch_weather -> search_places -> enrich
#   -> rules_filter -> score_and_rank -> llm_reason


class AgentGraph:
    """추천 파이프라인 노드를 연결한 LangGraph StateGraph 빌더."""
    pass
