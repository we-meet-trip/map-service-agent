"""call_structured(예산 + 교정 재시도)의 계약 단위 테스트.

핵심 계약: 교정 재시도를 포함한 모든 호출이 예산을 소비하며,
어떤 경우에도 요청당 호출 수가 max_calls 를 넘지 않는다.
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app import agent_dependencies as deps
from app.clients.agent_clients import StructuredOutputError
from app.llm.structured_call import LLMBudgetExceeded, call_structured


class _Echo(BaseModel):
    """테스트용 응답 스키마."""
    value: int


class _FlakyGemini:
    """앞 N회는 StructuredOutputError, 이후 성공하는 대역."""

    def __init__(self, fail_times: int, result=None, error: Exception | None = None):
        self.fail_times = fail_times
        self.result = result
        self.error = error
        self.prompts: list[str] = []

    async def generate_structured(self, prompt, schema, *, system_instruction=None):
        self.prompts.append(prompt)
        if len(self.prompts) <= self.fail_times:
            if self.error is not None:
                raise self.error
            raise StructuredOutputError(
                "invalid", raw_text='{"bad": "<oops>"}',
                cause=ValueError("value is not a valid integer"),
            )
        return self.result


def _run(coro):
    return asyncio.run(coro)


def test_success_consumes_one_budget() -> None:
    """정상 1회 호출 = 예산 1 소비."""
    fake = _FlakyGemini(fail_times=0, result=_Echo(value=1))
    deps.set_gemini_client(fake)
    try:
        state: dict = {}
        out = _run(call_structured(state, "p", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert out.value == 1
    assert state["llm_calls_used"] == 1


def test_corrective_retry_feeds_back_error() -> None:
    """검증 실패 시 오류·무효 출력을 피드백한 교정 프롬프트로 1회 재시도."""
    fake = _FlakyGemini(fail_times=1, result=_Echo(value=2))
    deps.set_gemini_client(fake)
    try:
        state: dict = {}
        out = _run(call_structured(state, "orig-prompt", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert out.value == 2
    assert state["llm_calls_used"] == 2  # 재시도도 예산 소비
    corrective = fake.prompts[1]
    assert corrective.startswith("orig-prompt")
    assert "<error_feedback>" in corrective
    assert "value is not a valid integer" in corrective
    # 무효 출력의 태그 문자는 이스케이프되어 인용된다(펜스 브레이크아웃 방지).
    assert "&lt;oops&gt;" in corrective
    assert "<oops>" not in corrective


def test_budget_exhausted_before_call() -> None:
    """예산 소진 상태면 호출 없이 LLMBudgetExceeded."""
    fake = _FlakyGemini(fail_times=0, result=_Echo(value=3))
    deps.set_gemini_client(fake)
    try:
        state = {"llm_calls_used": 3}
        with pytest.raises(LLMBudgetExceeded):
            _run(call_structured(state, "p", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert fake.prompts == []  # 호출 자체가 없어야 한다
    assert state["llm_calls_used"] == 3


def test_no_retry_budget_propagates_original_error() -> None:
    """잔여 예산이 없으면 교정 재시도 없이 원 예외 전파."""
    fake = _FlakyGemini(fail_times=2, result=_Echo(value=4))
    deps.set_gemini_client(fake)
    try:
        state = {"llm_calls_used": 2}  # 남은 예산 1 — 재시도 몫 없음
        with pytest.raises(StructuredOutputError):
            _run(call_structured(state, "p", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert len(fake.prompts) == 1  # 교정 재시도 미수행
    assert state["llm_calls_used"] == 3


def test_non_schema_error_no_retry() -> None:
    """타임아웃 등 비-스키마 오류는 재시도 없이 그대로 전파."""
    fake = _FlakyGemini(fail_times=1, error=asyncio.TimeoutError())
    deps.set_gemini_client(fake)
    try:
        state: dict = {}
        with pytest.raises(asyncio.TimeoutError):
            _run(call_structured(state, "p", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert len(fake.prompts) == 1
    assert state["llm_calls_used"] == 1


class _TruncatedGemini:
    """출력이 잘려 실패했다고 알리는 대역."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None
    ):
        self.prompts.append(prompt)
        raise StructuredOutputError(
            "invalid",
            raw_text='{"value": 1',
            cause=ValueError("EOF while parsing"),
            truncated=True,
        )


def test_truncated_output_skips_corrective_retry() -> None:
    """잘려서 실패한 응답은 다시 묻지 않는다.

    같은 프롬프트는 같은 길이의 답을 만들어 또 잘린다. 재시도해 봐야
    예산 1 과 프롬프트 전체를 확정적으로 버릴 뿐이라, 예산이 남아
    있어도 건너뛴다.
    """
    gemini = _TruncatedGemini()
    deps.set_gemini_client(gemini)
    state: dict = {}
    try:
        with pytest.raises(StructuredOutputError):
            asyncio.run(
                call_structured(state, "p", _Echo, max_calls=3)
            )
    finally:
        deps.reset_all()
    assert len(gemini.prompts) == 1, "재시도가 나가면 안 된다"
    assert state["llm_calls_used"] == 1


def test_non_truncated_failure_still_retries() -> None:
    """잘린 것이 아니면 기존대로 한 번 교정한다."""
    gemini = _FlakyGemini(fail_times=1, result=_Echo(value=7))
    deps.set_gemini_client(gemini)
    state: dict = {}
    try:
        out = asyncio.run(call_structured(state, "p", _Echo, max_calls=3))
    finally:
        deps.reset_all()
    assert out.value == 7
    assert len(gemini.prompts) == 2
