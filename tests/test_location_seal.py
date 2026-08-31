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
