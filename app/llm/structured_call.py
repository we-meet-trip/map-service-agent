"""요청당 LLM 호출 예산 + 교정 재시도 헬퍼.

요청 1건이 쓸 수 있는 Gemini 호출 수를 단일 지점에서 강제한다. 상한값은
호출자가 `max_calls` 로 넘기며(설정 `GEMINI_MAX_CALLS_PER_REQUEST`),
**교정 재시도를 포함한 모든 호출이 `state["llm_calls_used"]` 예산 1을
소비**한다.

교정 재시도(self-repair):
  기존 노드들의 "동일 프롬프트 단순 재시도" 를 대체한다. 구조화 응답이
  스키마 검증에 실패하면(`StructuredOutputError`), 실패 원인과 무효
  출력 일부를 `<error_feedback>` 태그로 원 프롬프트에 덧붙여 1회
  재호출한다. 오류 내용이 곧 모델에 대한 지시가 되므로 단순 재시도보다
  수렴 확률이 높다.

호출 경로:
  agent_nodes._select_places / _invent_places / recommend_route /
  llm_reason → call_structured → GeminiClient.generate_structured.
"""
from __future__ import annotations

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from app.clients.agent_clients import StructuredOutputError
from app.security.sanitize import sanitize_text

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 교정 프롬프트에 인용하는 무효 출력/오류 메시지의 절단 길이.
# 프롬프트 비대와 (모델 출력 유래 텍스트의) 재주입 표면을 함께 제한한다.
_FEEDBACK_TRUNCATE = 500


class LLMBudgetExceeded(RuntimeError):
    """요청당 LLM 호출 예산(GEMINI_MAX_CALLS_PER_REQUEST) 소진.

    호출자가 노드 성격에 따라 실패(state["error"]) 또는
    degrade(llm_reason 생략)로 처리한다.
    """

    def __init__(self) -> None:
        super().__init__("llm call budget exceeded for this request")


def _escape_feedback(text: str) -> str:
    """피드백 인용문을 새니타이즈해 재주입 표면을 제거한다.

    무효 출력은 모델 유래 텍스트다 — 1차 응답에 심긴 비가시 문자·태그가
    교정 프롬프트로 되돌아가는 2차 주입을 막기 위해, 입력측과 동일한
    `sanitize_text`(제어·비가시 문자 제거 + 꺾쇠 이스케이프 + 절단)를
    그대로 재사용한다.
    """
    return sanitize_text(text, _FEEDBACK_TRUNCATE)


def _consume_budget(state: dict, max_calls: int) -> None:
    """예산 1을 소비한다. 소진 상태면 LLMBudgetExceeded."""
    used = state.get("llm_calls_used", 0)
    if used >= max_calls:
        raise LLMBudgetExceeded()
    state["llm_calls_used"] = used + 1


async def call_structured(
    state: dict,
    prompt: str,
    response_schema: Type[T],
    *,
    system_instruction: str | None = None,
    max_calls: int,
) -> T:
    """예산 관리 하에 구조화 호출을 수행하고, 검증 실패 시 1회 교정한다.

    state: AgentState. `llm_calls_used` 키로 예산을 추적한다.
    prompt: 데이터 태그로 격리된 user 콘텐츠 프롬프트.
    response_schema: 응답 Pydantic 모델.
    system_instruction: 불변 규칙 시스템 지시(사용자 문자열 혼입 금지).
    max_calls: 요청당 호출 예산(설정 GEMINI_MAX_CALLS_PER_REQUEST).

    실패:
      - LLMBudgetExceeded: 예산 소진(호출 자체를 시작 못함).
      - StructuredOutputError: 교정 재시도까지 실패(또는 재시도 예산 없음).
      - 그 외(타임아웃/GeminiQuotaError 등): 재시도 없이 그대로 전파.

    토큰 사용량은 state["llm_usage"] 에 쌓인다. 자체 모델로 옮길 때 필요한
    컴퓨트를 이 값으로 가늠하는데, 지금은 어디에도 남지 않아 짐작밖에 할 수
    없다. 재시도분도 함께 센다 — 실제로 쓴 양이 그것이기 때문이다.

    호출처: app/nodes/agent_nodes.py 의 LLM 노드들.
    """
    from app.agent_dependencies import get_gemini_client

    gemini = get_gemini_client()

    _consume_budget(state, max_calls)
    # 이번 호출이 쓴 토큰을 담을 자리. 호출마다 새로 만들어 넘기므로 동시에
    # 도는 다른 잡의 수치와 섞이지 않는다.
    usage: list[dict] = []
    try:
        return await _call(gemini, state, usage, prompt, response_schema,
                           system_instruction, max_calls)
    finally:
        # 성공이든 실패든 쓴 만큼은 센다. 실패한 호출도 토큰을 쓰므로,
        # 성공분만 세면 실제 소비량을 낮춰 보게 된다.
        if usage:
            state["llm_usage"] = (state.get("llm_usage") or []) + usage


async def _call(gemini, state, usage, prompt, response_schema,
                system_instruction, max_calls):
    """실제 호출과 교정 재시도. 토큰 계량은 바깥 finally 가 맡는다."""
    try:
        return await gemini.generate_structured(
            prompt, response_schema, system_instruction=system_instruction,
            usage_sink=usage,
        )
    except StructuredOutputError as e:
        logger.warning(
            "structured output invalid (%s), corrective retry considered",
            type(e.cause).__name__,
        )
        # 출력이 잘려서 실패한 것이라면 재시도하지 않는다. 같은 프롬프트는
        # 같은 길이의 답을 만들어 또 잘리므로, 재시도가 예산 1 과 프롬프트
        # 전체를 확정적으로 버리는 셈이다. 이 경우 길이를 줄이는 것이
        # 해법이지 다시 묻는 것이 해법이 아니다.
        if e.truncated:
            logger.warning("output truncated — skipping corrective retry")
            raise
        # 잔여 예산이 없으면 교정 재시도 없이 **원 예외**를 전파한다
        # (LLMBudgetExceeded 로 바꿔치지 않는다 — 실패 원인 보존).
        if state.get("llm_calls_used", 0) >= max_calls:
            raise
        state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1
        corrective = (
            f"{prompt}"
            "<error_feedback>\n"
            "직전 응답이 JSON 스키마 검증에 실패했습니다. 아래 오류를\n"
            "교정하여, 스키마에 정확히 부합하는 JSON 만 다시 출력하십시오.\n"
            f"오류: {_escape_feedback(str(e.cause))}\n"
            f"무효 출력(일부): {_escape_feedback(e.raw_text)}\n"
            "</error_feedback>\n"
        )
        return await gemini.generate_structured(
            corrective, response_schema, system_instruction=system_instruction,
            usage_sink=usage,
        )
