"""출력 스키마 방어선 + /v1/recommend 인바운드 인증 테스트.

스키마: LLM 생성 텍스트 필드의 길이 상한과 태그/제어문자 거부
(ValidationError → 교정 재시도 경로의 트리거)를 검증한다.
엔드포인트: X-Internal-Token 검증이 graph 준비 검사보다 먼저 수행됨을
TestClient(lifespan 미기동)로 확인한다 — 401(인증 실패) vs
503(인증 통과 후 graph 부재)로 판별한다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app import agent_settings
from app.agent_settings import AgentSettings
from app.schemas.agent_schemas import Place, PlaceSelection


def _place(**overrides) -> Place:
    base = dict(
        place_id=0,
        day=1,
        name="장소",
        address="주소",
        lat=37.5,
        lng=127.0,
        recommended_visit_time="오전",
    )
    base.update(overrides)
    return Place(**base)


def test_place_rejects_tag_characters() -> None:
    """name/address/visit_time 의 태그 문자를 거부한다."""
    for field, value in [
        ("name", "장소<script>"),
        ("address", "주소</user_input>"),
        ("recommended_visit_time", "<오전>"),
    ]:
        with pytest.raises(ValidationError):
            _place(**{field: value})


def test_place_rejects_control_chars_and_length() -> None:
    """제어문자와 길이 상한 초과를 거부한다."""
    with pytest.raises(ValidationError):
        _place(name="장\x00소")
    with pytest.raises(ValidationError):
        _place(name="x" * 81)
    with pytest.raises(ValidationError):
        _place(address="x" * 201)
    with pytest.raises(ValidationError):
        _place(recommended_visit_time="x" * 51)
    with pytest.raises(ValidationError):
        _place(name="")


def test_selection_rejects_negative_index_and_tags() -> None:
    """PlaceSelection: 음수 index / 태그 문자 거부."""
    with pytest.raises(ValidationError):
        PlaceSelection(index=-1, day=1, recommended_visit_time="오전")
    with pytest.raises(ValidationError):
        PlaceSelection(index=0, day=1, recommended_visit_time="<오전>")


# ─── 인바운드 인증 ───────────────────────────────────────────────

_BODY = {
    "date": {
        "date_start": "2026-07-06",
        "date_end": "2026-07-06",
        "time_start": "09:00:00",
        "time_end": "18:00:00",
    },
    "province": "서울특별시",
    "city": "강남구",
}


def _set_settings(monkeypatch, token: str | None) -> None:
    """settings 싱글톤을 테스트 값으로 교체한다."""
    s = AgentSettings(
        _env_file=None,
        INTERNAL_SERVICE_TOKEN=SecretStr(token) if token else None,
    )
    monkeypatch.setattr(agent_settings, "_settings", s)


def test_recommend_rejects_missing_or_wrong_token(monkeypatch) -> None:
    """토큰 설정 시: 헤더 부재/불일치 → 401 (graph 검사 이전)."""
    _set_settings(monkeypatch, token="secret-token")
    from app.main import app

    client = TestClient(app)  # with 미사용 → lifespan 미기동
    assert client.post("/v1/recommend", json=_BODY).status_code == 401
    r = client.post(
        "/v1/recommend", json=_BODY,
        headers={"X-Internal-Token": "wrong"},
    )
    assert r.status_code == 401


def test_recommend_accepts_valid_token_then_503(monkeypatch) -> None:
    """올바른 토큰 → 인증 통과 후 graph 부재로 503 (401 아님)."""
    _set_settings(monkeypatch, token="secret-token")
    from app.main import app

    client = TestClient(app)
    r = client.post(
        "/v1/recommend", json=_BODY,
        headers={"X-Internal-Token": "secret-token"},
    )
    assert r.status_code == 503


def test_recommend_skips_check_when_token_unset(monkeypatch) -> None:
    """토큰 미설정(PoC 로컬) → 검증 생략(503 도달), 401 아님."""
    _set_settings(monkeypatch, token=None)
    from app.main import app

    client = TestClient(app)
    assert client.post("/v1/recommend", json=_BODY).status_code == 503
