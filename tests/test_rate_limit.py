"""GeminiRateLimiter.acquire 의 4개 시나리오 단위 테스트.

실제 Redis 대신 limiter._script 를 스크립트된 대역으로 교체하고,
app.llm.rate_limit._sleep 을 계측용 페이크로 바꿔 대기 시간을 검증한다.
(Lua 스크립트 자체의 Redis 통합 동작은 단계 15 의 docker compose 수동
검증 항목이다.)
"""
from __future__ import annotations

import asyncio

import pytest

from app.llm import rate_limit
from app.llm.rate_limit import GeminiQuotaError, GeminiRateLimiter


def _limiter() -> GeminiRateLimiter:
    """접속하지 않는 limiter 인스턴스(from_url 은 lazy connect)."""
    return GeminiRateLimiter(
        "redis://test-host:6379", db=3, capacity=10,
        refill_per_min=10, daily_cap=200,
    )


class _ScriptedEval:
    """미리 정한 결과 시퀀스를 차례로 돌려주는 _script 대역."""

    def __init__(self, results: list[list]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def __call__(self, keys, args):
        self.calls.append({"keys": keys, "args": args})
        return self._results.pop(0)


def _patch_sleep(monkeypatch) -> list[float]:
    """_sleep 을 무대기 페이크로 바꾸고 요청된 대기 시간 목록을 돌려준다."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(rate_limit, "_sleep", fake_sleep)
    return slept


def test_immediate_ok(monkeypatch) -> None:
    """토큰 여유 시 대기 없이 1회에 통과한다."""
    limiter = _limiter()
    script = _ScriptedEval([["ok", 0]])
    limiter._script = script
    slept = _patch_sleep(monkeypatch)

    asyncio.run(limiter.acquire())
    assert len(script.calls) == 1
    assert slept == []
    # 키 규약: [일일 키(gemini:daily:YYYYMMDD), 토큰 키]
    assert script.calls[0]["keys"][0].startswith("gemini:daily:")
    assert script.calls[0]["keys"][1] == "gemini:tokens"


def test_wait_then_ok(monkeypatch) -> None:
    """버킷 고갈 시 권고 시간만큼 대기 후 재시도해 통과한다."""
    limiter = _limiter()
    limiter._script = _ScriptedEval([["wait", 500], ["ok", 0]])
    slept = _patch_sleep(monkeypatch)

    asyncio.run(limiter.acquire())
    assert slept == [0.5]


def test_rpm_exceeded_after_60s(monkeypatch) -> None:
    """누적 대기가 60초를 넘으면 gemini_rpm_exceeded."""
    limiter = _limiter()
    # 30초 대기 권고가 반복 → 30 + 30 소진 후 3번째에서 상한 초과
    limiter._script = _ScriptedEval(
        [["wait", 30000], ["wait", 30000], ["wait", 30000]]
    )
    slept = _patch_sleep(monkeypatch)

    with pytest.raises(GeminiQuotaError) as exc_info:
        asyncio.run(limiter.acquire())
    assert str(exc_info.value) == "gemini_rpm_exceeded"
    assert exc_info.value.code == "gemini_rpm_exceeded"
    assert slept == [30.0, 30.0]  # 3번째 대기는 상한 초과로 미수행


def test_rpd_exceeded_immediately(monkeypatch) -> None:
    """일일 cap 도달 시 대기 없이 즉시 gemini_rpd_exceeded."""
    limiter = _limiter()
    limiter._script = _ScriptedEval([["rpd", 0]])
    slept = _patch_sleep(monkeypatch)

    with pytest.raises(GeminiQuotaError) as exc_info:
        asyncio.run(limiter.acquire())
    assert str(exc_info.value) == "gemini_rpd_exceeded"
    assert slept == []


def test_kst_midnight_ttl() -> None:
    """일일 키 TTL 이 KST 자정까지 남은 초로 계산된다."""
    from datetime import datetime, timedelta, timezone

    kst = timezone(timedelta(hours=9))
    now = datetime(2026, 7, 5, 23, 59, 0, tzinfo=kst)
    assert GeminiRateLimiter._seconds_until_kst_midnight(now) == 60
    noon = datetime(2026, 7, 5, 12, 0, 0, tzinfo=kst)
    assert GeminiRateLimiter._seconds_until_kst_midnight(noon) == 12 * 3600
