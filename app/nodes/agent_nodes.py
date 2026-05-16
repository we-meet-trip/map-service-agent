# LangGraph 노드 함수를 본 모듈에 정의한다.
# 노드 1개당 1 함수로 구현하며, 추천 파이프라인 단계는 다음과 같다:
#   parse_input -> fetch_weather -> search_places -> enrich
#   -> rules_filter -> score_and_rank -> llm_reason


def parse_input(state):
    """추천 요청 입력을 파싱하여 후속 노드에서 사용할 상태로 변환한다."""
    return state
