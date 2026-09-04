"""다른 서비스로 보낼 좌표를 감싸는 쪽 검사.

여는 쪽은 hub(파이썬)와 BFF(자바)다. 형식이 어긋나면 배포한 뒤 좌표가 오가는
모든 길이 한꺼번에 막히는데, 화면에는 사유 없는 실패로만 보인다.
"""
from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from app.agent_settings import get_settings
from app.crypto import location_seal

KEY_B64 = "bWFwLXdpcmUtdGVzdC1rZXktMDAwMDAwMDAwMDAwMCE="
KEY = base64.b64decode(KEY_B64)


@pytest.fixture(autouse=True)
def _wire_key():
    settings = get_settings()
    before = settings.LOCATION_WIRE_KEY
    settings.LOCATION_WIRE_KEY = SecretStr(KEY_B64)
    yield
    settings.LOCATION_WIRE_KEY = before


def _open(token: str) -> dict:
    """받는 쪽이 하는 그대로 연다."""
    _, iv, ct = token.split(".", 2)

    def b64u(text: str) -> bytes:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))

    return json.loads(AESGCM(KEY).decrypt(b64u(iv), b64u(ct), b"map|loc|v1"))


def test_sealed_value_hides_coordinates():
    token = location_seal.seal({"origin": {"lat": 35.1587, "lng": 129.1604}})

    assert token.startswith("v1.")
    assert "35.1587" not in token and "129.1604" not in token


def test_round_trip_keeps_values():
    token = location_seal.seal({"origin": {"lat": 35.1587, "lng": 129.1604}})

    assert _open(token)["origin"]["lat"] == 35.1587


def test_adds_issue_time():
    # 만든 시각이 없으면 한 번 지나간 봉투가 영원히 유효해서, 주워 둔 것을
    # 나중에 그대로 다시 보낼 수 있다.
    assert "iat" in _open(location_seal.seal({"a": 1}))


def test_never_repeats():
    payload = {"origin": {"lat": 35.1587, "lng": 129.1604}}

    assert location_seal.seal(payload) != location_seal.seal(payload)


def test_url_safe_characters_only():
    # 요청 줄에 그대로 실려야 하는데, 더하기와 빗금은 주소에서 다른 뜻으로 읽힌다.
    token = location_seal.seal({"a": 1})

    assert all(c.isalnum() or c in "._-" for c in token)


def test_disabled_without_key():
    settings = get_settings()
    settings.LOCATION_WIRE_KEY = SecretStr("")

    assert not location_seal.enabled()


def test_refuses_short_key():
    settings = get_settings()
    settings.LOCATION_WIRE_KEY = SecretStr(base64.b64encode(b"0" * 24).decode())

    with pytest.raises(location_seal.SealError):
        location_seal.seal({"a": 1})


def test_opened_places_become_schema_objects():
    """봉투에서 꺼낸 장소는 스키마로 세워져야 한다.

    검사 없이 dict 인 채로 넘기면 장소를 읽는 첫 노드가
    "'dict' object has no attribute 'day'" 로 터진다 — 화면에는 사유 없는
    502 로만 보이고, 좌표 범위 검사도 그때는 이미 지나간 뒤다.
    """
    from datetime import date, time

    from app.main import _resolve_places
    from app.schemas.agent_schemas import (
        AgentRequest,
        DateRange,
        SelectedPlace,
    )

    req = AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="강원특별자치도",
        city="속초시",
        stage="route",
        loc=location_seal.seal({
            "places": [
                {"name": "속초해변", "address": "강원 속초시",
                 "lat": 38.19, "lng": 128.60, "day": 1},
                {"name": "영금정", "address": "강원 속초시",
                 "lat": 38.21, "lng": 128.60, "day": 1},
            ]
        }),
    )

    out = _resolve_places(req)

    assert all(isinstance(p, SelectedPlace) for p in out.places)
    assert out.places[0].day == 1
    assert out.loc is None


def test_opened_places_out_of_range_are_refused():
    """봉투 안이라도 국내 밖 좌표는 받지 않는다."""
    from datetime import date, time

    from fastapi import HTTPException

    from app.main import _resolve_places
    from app.schemas.agent_schemas import AgentRequest, DateRange

    req = AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="강원특별자치도",
        city="속초시",
        stage="route",
        loc=location_seal.seal({
            "places": [
                {"name": "해외", "address": "", "lat": 10.0, "lng": 127.0,
                 "day": 1},
            ]
        }),
    )

    with pytest.raises(HTTPException) as err:
        _resolve_places(req)
    assert err.value.status_code == 400
