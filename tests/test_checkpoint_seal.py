"""체크포인트 봉인 검사.

핵심은 DB 에 남는 바이트에 좌표가 보이지 않는 것이므로, 겉모양이 아니라
저장 바이트를 직접 보고 확인한다. 대조군(봉하지 않은 serializer)에는 좌표
필드명이 그대로 보인다는 것까지 함께 단언해, 부재 단언이 실제로 무언가를
걸러내고 있음을 증명한다.
"""
from __future__ import annotations

import base64

import pytest
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import SecretStr

from app.agent_settings import AgentSettings
from app.crypto.checkpoint_seal import (
    CheckpointCipher,
    CheckpointSealError,
    build_cipher,
    parse_keys,
)

# 무의미한 고정값 시험 열쇠. 실제 배포 열쇠와 무관하다.
KEY1 = b"\x11" * 32
KEY2 = b"\x22" * 32
KEY1_B64 = base64.b64encode(KEY1).decode()
KEY2_B64 = base64.b64encode(KEY2).decode()
STATE = {"lat": 37.5, "lng": 127.0, "query": "성수동 카페"}


def _serde(keys: dict[str, bytes], active: str) -> EncryptedSerializer:
    return EncryptedSerializer(cipher=CheckpointCipher(keys, active))


def test_sealed_bytes_hide_coordinates():
    typ, data = _serde({"k1": KEY1}, "k1").dumps_typed(STATE)

    assert typ.endswith("+aesgcm:k1")
    assert b"lat" not in data and b"lng" not in data and b"37.5" not in data


def test_round_trip_restores_state():
    serde = _serde({"k1": KEY1}, "k1")

    assert serde.loads_typed(serde.dumps_typed(STATE)) == STATE


def test_plain_serializer_exposes_coordinates():
    # 대조군: 봉하지 않으면 좌표 필드명이 저장 바이트에 그대로 보인다.
    _, data = JsonPlusSerializer().dumps_typed(STATE)

    assert b"lat" in data and b"lng" in data


def test_wrong_key_fails_to_open():
    sealed = _serde({"k1": KEY1}, "k1").dumps_typed(STATE)

    with pytest.raises(CheckpointSealError):
        _serde({"k1": KEY2}, "k1").loads_typed(sealed)


def test_kid_rotation_reads_old_records():
    # k1 시절 기록을, k2 로 갈아탄 뒤에도(k1 을 목록에 남겨) 읽을 수 있다.
    old = _serde({"k1": KEY1}, "k1").dumps_typed(STATE)
    rotated = _serde({"k1": KEY1, "k2": KEY2}, "k2")

    assert rotated.loads_typed(old) == STATE
    assert rotated.dumps_typed(STATE)[0].endswith("+aesgcm:k2")


def test_unknown_kid_refused():
    sealed = _serde({"k1": KEY1}, "k1").dumps_typed(STATE)

    with pytest.raises(CheckpointSealError):
        _serde({"k2": KEY2}, "k2").loads_typed(sealed)


def test_legacy_plaintext_record_passes_through():
    # 봉하기 전에 쌓인 기록은 typ 에 + 표식이 없다. 그대로 읽혀야 배포하는
    # 순간 옛 대화가 전부 깨지는 일이 없다.
    plain = JsonPlusSerializer().dumps_typed(STATE)

    assert _serde({"k1": KEY1}, "k1").loads_typed(plain) == STATE


def test_parse_keys_reads_multiple_entries():
    assert parse_keys(f"k1:{KEY1_B64}, k2:{KEY2_B64}") == {
        "k1": KEY1,
        "k2": KEY2,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",  # 빈 목록
        "k1",  # 구분자 없음
        f"k1:{KEY1_B64},k1:{KEY2_B64}",  # kid 중복
        "k1:!!!!",  # base64 아님
        "k1:" + base64.b64encode(b"short").decode(),  # 32바이트 아님
    ],
)
def test_parse_keys_refuses_bad_entries(raw):
    with pytest.raises(CheckpointSealError):
        parse_keys(raw)


def _settings(keys: str, active: str, wire: str = "") -> AgentSettings:
    s = AgentSettings(_env_file=None)
    s.CHECKPOINT_ENC_KEYS = SecretStr(keys)
    s.CHECKPOINT_ENC_ACTIVE_KID = active
    s.LOCATION_WIRE_KEY = SecretStr(wire)
    return s


def test_build_cipher_requires_keys():
    with pytest.raises(CheckpointSealError):
        build_cipher(_settings("", "k1"))


def test_build_cipher_requires_active_kid_in_map():
    with pytest.raises(CheckpointSealError):
        build_cipher(_settings(f"k1:{KEY1_B64}", "nope"))


def test_build_cipher_rejects_wire_key_reuse():
    # 좌표 전송 열쇠를 그대로 돌려쓰면 부팅을 막는다(용도 분리).
    with pytest.raises(CheckpointSealError):
        build_cipher(_settings(f"k1:{KEY1_B64}", "k1", wire=KEY1_B64))


def test_build_cipher_accepts_distinct_keys():
    cipher = build_cipher(_settings(f"k1:{KEY1_B64}", "k1", wire=KEY2_B64))
    name, sealed = cipher.encrypt(b"x")

    assert name == "aesgcm:k1"
    assert cipher.decrypt(name, sealed) == b"x"
