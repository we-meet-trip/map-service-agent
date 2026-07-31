"""AgentSettings 환경 변수 배선 단위 테스트.

.env.example 에 문서화된 변수명이 실제 Settings 필드로 로드되는지
(과거 GEMINI_RPM_LIMIT 등이 extra="ignore" 로 조용히 무시되던 결함의
회귀 방지)와 기본값을 검증한다. `_env_file=None` 으로 작업 디렉토리의
.env 간섭을 차단한다.
"""
from __future__ import annotations

from app.agent_settings import AgentSettings


def test_defaults() -> None:
    """환경 변수 미주입 시 기본값 — SoT 결정값과 일치해야 한다."""
    s = AgentSettings(_env_file=None)
    assert s.GEMINI_RPM_LIMIT == 10
    assert s.GEMINI_RPD_LIMIT == 250
    assert s.GEMINI_RPD_CAP == 200
    # 장소 선정 1 + 동선 1 + 추천이유 1 + 리뷰요약 1
    assert s.GEMINI_MAX_CALLS_PER_REQUEST == 4
    assert s.GEMINI_TEMPERATURE == 0.2
    assert s.GEMINI_MAX_OUTPUT_TOKENS == 8192
    assert s.GEMINI_THINKING_BUDGET == 0
    assert s.GEMINI_RETRY_ATTEMPTS == 3
    assert s.GEMINI_RETRY_INITIAL_DELAY == 1.0
    assert s.GEMINI_RETRY_MAX_DELAY == 8.0
    assert s.GEMINI_RETRY_STATUS_CODES == "503,500"
    assert s.GEMINI_HTTP_TIMEOUT_SECONDS == 30.0
    assert s.GEMINI_TIMEOUT_SECONDS == 100.0
    assert s.REDIS_DB_STREAMS == 2
    assert s.REDIS_DB_RATELIMIT == 3
    assert s.STATUS_STREAM_NAME == "agent:jobs:status"
    assert s.STATUS_STREAM_MAXLEN == 2000
    assert s.CHECKPOINT_ENABLED is True
    assert s.LANGGRAPH_SCHEMA == "langgraph"
    assert s.JOB_TIMEOUT_SECONDS == 300.0
    assert s.SHUTDOWN_GRACE_SECONDS == 310.0
    assert s.HUB_TIMEOUT_SECONDS == 5.0
    assert s.AUTH_ENFORCED is False
    assert s.REVIEWS_ENRICH_ENABLED is True
    assert s.REVIEWS_MAX_PLACES == 3
    assert s.REVIEWS_DISPLAY == 5
    assert s.REVIEWS_FETCH_CAP_PER_JOB == 10
    assert s.SUMMARY_ENABLED is True
    # 추천 장소 상한(7곳)과 같아야 뒷순번 장소도 요약을 받는다.
    assert s.SUMMARY_MAX_PLACES == 7
    assert s.RULES_ENABLED is True
    assert s.LOG_LEVEL == "INFO"


def test_env_injection(monkeypatch) -> None:
    """환경 변수 주입 시 각 필드가 실제로 로드되는지 확인."""
    monkeypatch.setenv("GEMINI_RPM_LIMIT", "5")
    monkeypatch.setenv("GEMINI_RPD_CAP", "100")
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_REQUEST", "2")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.0")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("REDIS_DB_RATELIMIT", "5")
    monkeypatch.setenv("STATUS_STREAM_NAME", "agent:jobs:status-test")
    monkeypatch.setenv("CHECKPOINT_ENABLED", "false")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw-test")
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("AUTH_ENFORCED", "true")
    monkeypatch.setenv("REVIEWS_ENRICH_ENABLED", "false")
    monkeypatch.setenv("REVIEWS_MAX_PLACES", "5")
    monkeypatch.setenv("RULES_ENABLED", "false")
    s = AgentSettings(_env_file=None)
    assert s.GEMINI_RPM_LIMIT == 5
    assert s.GEMINI_RPD_CAP == 100
    assert s.GEMINI_MAX_CALLS_PER_REQUEST == 2
    assert s.GEMINI_TEMPERATURE == 0.0
    assert s.GEMINI_MAX_OUTPUT_TOKENS == 1024
    assert s.REDIS_DB_RATELIMIT == 5
    assert s.STATUS_STREAM_NAME == "agent:jobs:status-test"
    assert s.CHECKPOINT_ENABLED is False
    assert s.POSTGRES_PASSWORD.get_secret_value() == "pw-test"
    assert s.JOB_TIMEOUT_SECONDS == 120.0
    assert s.AUTH_ENFORCED is True
    assert s.REVIEWS_ENRICH_ENABLED is False
    assert s.REVIEWS_MAX_PLACES == 5
    assert s.RULES_ENABLED is False
