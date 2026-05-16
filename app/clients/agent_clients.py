# 외부 시스템 호출 클라이언트 모듈.
# hub 게이트웨이와 Gemini LLM 호출을 캡슐화하는 클라이언트를 정의한다.


class HubClient:
    """장소·날씨·경로·룰 데이터를 hub로부터 조회하는 HTTP 클라이언트.

    호출 단위로 timeout·retry·circuit-breaker를 적용하고, 응답을 agent 내부
    스키마로 변환하여 노드 함수에 전달한다.
    """
    pass


class GeminiClient:
    """Gemini 모델 호출을 캡슐화하는 클라이언트.

    프롬프트 구성·응답 파싱·토큰 사용량 제어(rate limit)를 담당한다.
    """
    pass
