"""GeminiClient 구조화 호출의 config 전달·실패 변환 단위 테스트.

실제 google-genai SDK 대신 가짜 모듈을 sys.modules 에 주입해
(1) GenerateContentConfig 로 전달되는 system_instruction/temperature/
max_output_tokens/response_schema 값, (2) 검증 실패 시
StructuredOutputError 로의 변환(원문 보존), (3) limiter.acquire() 선행
호출을 검증한다.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest
from pydantic import BaseModel


class _Echo(BaseModel):
    """테스트용 응답 스키마."""
    value: int


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAioModels:
    """generate_content 호출 인자를 기록하고 준비된 응답을 돌려주는 대역."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return _FakeResponse(self.response_text)


def _install_fake_genai(monkeypatch, response_text: str) -> _FakeAioModels:
    """sys.modules 에 가짜 google/google.genai 모듈 트리를 주입한다."""
    fake_models = _FakeAioModels(response_text)

    class _FakeClient:
        # 신규 GeminiClient 는 재시도 설정 시 http_options 를 넘긴다.
        # 기록해 두어 테스트가 retry_options/timeout 전달을 단언한다.
        def __init__(self, api_key: str, http_options=None) -> None:
            self.aio = types.SimpleNamespace(models=fake_models)
            self.http_options = http_options

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_budget) -> None:
            self.thinking_budget = thinking_budget

    class _FakeHttpRetryOptions:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeHttpOptions:
        def __init__(self, *, timeout=None, retry_options=None) -> None:
            self.timeout = timeout
            self.retry_options = retry_options

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _FakeClient
    genai_types_mod = types.ModuleType("google.genai.types")
    genai_types_mod.GenerateContentConfig = _FakeGenerateContentConfig
    genai_types_mod.ThinkingConfig = _FakeThinkingConfig
    genai_types_mod.HttpRetryOptions = _FakeHttpRetryOptions
    genai_types_mod.HttpOptions = _FakeHttpOptions
    genai_mod.types = genai_types_mod
    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types_mod)
    return fake_models


class _RecordingLimiter:
    """acquire() 호출 여부만 기록하는 limiter 대역."""

    def __init__(self) -> None:
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1


def test_config_passthrough(monkeypatch) -> None:
    """temperature/max_output_tokens/system_instruction 이 config 로 전달된다."""
    fake_models = _install_fake_genai(monkeypatch, '{"value": 7}')
    from app.clients.agent_clients import GeminiClient

    limiter = _RecordingLimiter()
    client = GeminiClient(
        api_key="k",
        model="m",
        timeout=5.0,
        temperature=0.0,
        max_output_tokens=1234,
        thinking_budget=0,
        limiter=limiter,
    )
    result = asyncio.run(
        client.generate_structured(
            "prompt-body", _Echo, system_instruction="rules"
        )
    )
    assert result.value == 7
    assert limiter.acquired == 1
    cfg = fake_models.calls[0]["config"].kwargs
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_schema"] is _Echo
    assert cfg["system_instruction"] == "rules"
    assert cfg["temperature"] == 0.0
    assert cfg["max_output_tokens"] == 1234
    # thinking 을 끄는 ThinkingConfig(budget=0)가 config 로 전달된다.
    assert cfg["thinking_config"].thinking_budget == 0
    assert fake_models.calls[0]["contents"] == "prompt-body"


def test_negative_thinking_budget_omits_config(monkeypatch) -> None:
    """thinking_budget<0 이면 ThinkingConfig 를 설정하지 않는다(모델 기본)."""
    fake_models = _install_fake_genai(monkeypatch, '{"value": 1}')
    from app.clients.agent_clients import GeminiClient

    client = GeminiClient(
        api_key="k", model="m", timeout=5.0, thinking_budget=-1
    )
    asyncio.run(client.generate_structured("p", _Echo))
    assert "thinking_config" not in fake_models.calls[0]["config"].kwargs


def test_retry_config_passthrough(monkeypatch) -> None:
    """재시도 설정이 http_options.retry_options 로 전달되고 429 는 제외된다."""
    _install_fake_genai(monkeypatch, '{"value": 1}')
    from app.clients.agent_clients import GeminiClient

    client = GeminiClient(
        api_key="k", model="m", timeout=5.0,
        retry_attempts=3, retry_initial_delay=1.0, retry_max_delay=8.0,
        retry_status_codes="503,500", http_timeout_seconds=30.0,
    )
    http = client._client.http_options
    assert http is not None
    assert http.timeout == 30000  # ms
    ro = http.retry_options
    assert ro.kwargs["attempts"] == 3
    assert ro.kwargs["http_status_codes"] == [503, 500]  # 429 미포함
    assert ro.kwargs["initial_delay"] == 1.0
    assert ro.kwargs["max_delay"] == 8.0


def test_retry_custom_status_codes(monkeypatch) -> None:
    """쉼표 문자열 파싱 — 502/504 확장 가능, 비숫자 토큰은 무시."""
    _install_fake_genai(monkeypatch, '{"value": 1}')
    from app.clients.agent_clients import GeminiClient

    client = GeminiClient(
        api_key="k", model="m", retry_attempts=2,
        retry_status_codes="503, 500, 502, 504, x",
        http_timeout_seconds=0.0,
    )
    ro = client._client.http_options.retry_options
    assert ro.kwargs["http_status_codes"] == [503, 500, 502, 504]
    # per-attempt timeout 0 → timeout 미설정(retry_options 만 있으므로 http_options 존재)
    assert client._client.http_options.timeout is None


def test_retry_disabled_paths(monkeypatch) -> None:
    """attempts<=1 또는 빈 코드셋이면 retry_options None; 둘 다 없으면 http_options 생략."""
    _install_fake_genai(monkeypatch, '{"value": 1}')
    from app.clients.agent_clients import GeminiClient

    # attempts=1 → 재시도 없음. per-attempt timeout 만 설정 → http_options 존재(timeout만)
    c1 = GeminiClient(api_key="k", model="m", retry_attempts=1,
                      http_timeout_seconds=20.0)
    assert c1._client.http_options.retry_options is None
    assert c1._client.http_options.timeout == 20000

    # 빈 코드셋 → 재시도 없음(빈 리스트를 SDK 로 넘기지 않음)
    c2 = GeminiClient(api_key="k", model="m", retry_attempts=3,
                      retry_status_codes="", http_timeout_seconds=0.0)
    assert c2._client.http_options is None  # 재시도·타임아웃 모두 없음 → 기존 경로

    # 재시도·타임아웃 모두 비활성 → http_options 아예 생략(오늘 경로 등가)
    c3 = GeminiClient(api_key="k", model="m", retry_attempts=1,
                      http_timeout_seconds=0.0)
    assert c3._client.http_options is None


def test_validation_failure_raises_structured_error(monkeypatch) -> None:
    """스키마 위반 응답이 StructuredOutputError(원문 보존)로 변환된다."""
    _install_fake_genai(monkeypatch, '{"wrong": true}')
    from app.clients.agent_clients import GeminiClient, StructuredOutputError

    client = GeminiClient(api_key="k", model="m", timeout=5.0)
    with pytest.raises(StructuredOutputError) as exc_info:
        asyncio.run(client.generate_structured("p", _Echo))
    err = exc_info.value
    assert err.raw_text == '{"wrong": true}'
    assert isinstance(err, ValueError)  # 기존 except ValueError 분기 호환
    assert err.cause is not None
