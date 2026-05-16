# 요청·응답 Pydantic 모델을 본 모듈에 정의한다.

from pydantic import BaseModel


class AgentRequest(BaseModel):
    """추천 요청 페이로드 모델.

    schedule_id(일정 식별자), stage(파이프라인 단계 지시),
    exclude(재탐색 시 제외 대상) 필드를 포함한다.
    """
    pass
