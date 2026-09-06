"""다른 서비스로 보낼 좌표를 봉투에 담는다.

좌표는 그동안 요청 본문과 스트림에 값 그대로 실려 나갔다. 컨테이너 사이라
바깥에서는 못 보지만, 같은 망에 붙은 다른 컨테이너와 중간에 남는 기록에는
그대로 드러난다.

봉투를 만들 수 있다는 것 자체가 자격이 된다. 열쇠를 가진 서비스만 만들 수
있고 열쇠를 가진 서비스만 열 수 있으므로, 좌표를 쓰려면 서비스 인증을 먼저
거쳐야 한다는 조건이 형식으로 강제된다.

형식:
    v1.<난수>.<암호문+검증표>       두 조각 모두 URL 안전 base64(채움문자 없음)

여는 쪽은 hub(app/crypto/location_seal.py)와 BFF(LocationSeal)다. 세 곳이
같은 형식을 쓰며, 한 곳만 바꾸면 그 순간부터 좌표가 오가는 모든 길이
막히므로 형식을 바꿀 때는 반드시 함께 바꾼다.

여기서도 연다. BFF 가 추천을 맡길 때 고른 장소의 좌표를 함께 보내는데, 그
자리가 유일하게 감싸지지 않은 채 남아 있었다.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.agent_settings import get_settings

VERSION = "v1"
AAD = b"map|loc|v1"
KEY_BYTES = 32

# 봉투를 만든 지 이만큼 지나면 열지 않는다. 한 번 지나간 봉투가 영원히
# 유효하면 주워 둔 것을 나중에 그대로 다시 보낼 수 있다. hub 와 같은 값을 쓴다.
MAX_AGE_SECONDS = 300


class SealError(Exception):
    """감쌀 수 없다. 열쇠 설정이 잘못된 경우다."""


def enabled() -> bool:
    """열쇠가 있으면 감싸서 보낸다."""
    return bool(get_settings().LOCATION_WIRE_KEY.get_secret_value())


def _key() -> bytes:
    raw = get_settings().LOCATION_WIRE_KEY.get_secret_value()
    if not raw:
        raise SealError("wire key not configured")
    try:
        key = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001 - 어떤 형태든 설정 오류로 접는다
        raise SealError("wire key is not base64") from exc
    if len(key) != KEY_BYTES:
        raise SealError("wire key must decode to 32 bytes")
    return key


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def seal(payload: dict) -> str:
    """값 묶음을 봉투에 담는다.

    만든 시각을 여기서 한 번만 붙인다. 호출부마다 붙이게 하면 어느 한 곳에서
    빠지고, 그러면 그 봉투는 영원히 유효해져 주워 둔 것을 나중에 그대로 다시
    보낼 수 있다.
    """
    body = dict(payload)
    body["iat"] = int(time.time())
    iv = os.urandom(12)
    sealed = AESGCM(_key()).encrypt(iv, json.dumps(body).encode(), AAD)
    return f"{VERSION}.{_b64url(iv)}.{_b64url(sealed)}"


def _b64url_decode(text: str) -> bytes:
    # 채움문자를 떼고 보내므로 길이를 4의 배수로 되돌린 뒤 푼다.
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def is_sealed(value: str | None) -> bool:
    """봉투 모양인지만 본다. 열어 보지는 않는다."""
    return bool(value) and value.startswith(VERSION + ".") and value.count(".") == 2


def open_seal(token: str) -> dict:
    """봉투를 열어 안에 담긴 값을 돌려준다.

    열쇠가 다르거나, 내용이 손대어졌거나, 너무 오래된 봉투면 SealError 를
    올린다. 세 경우를 구분하지 않는 것은 밖에서 그 차이를 보고 무엇을 바꿔
    가며 시도할 수 있기 때문이다.
    """
    if not is_sealed(token):
        raise SealError("not a sealed value")
    _, iv_part, ct_part = token.split(".", 2)
    try:
        plain = AESGCM(_key()).decrypt(
            _b64url_decode(iv_part), _b64url_decode(ct_part), AAD
        )
    except Exception as exc:
        raise SealError("cannot open sealed value") from exc

    try:
        payload = json.loads(plain)
    except json.JSONDecodeError as exc:
        raise SealError("sealed value is not readable") from exc
    if not isinstance(payload, dict):
        raise SealError("sealed value is not readable")

    issued = payload.get("iat")
    if type(issued) not in (int, float) or not math.isfinite(issued):
        raise SealError("sealed value has no issue time")
    age = time.time() - issued
    # 앞뒤로 흔들리는 시계를 감안해 미래 쪽도 조금 허용한다. 서버끼리 시계가
    # 몇 초 어긋나는 것만으로 요청이 통째로 막히면 안 된다.
    if age > MAX_AGE_SECONDS or age < -60:
        raise SealError("sealed value expired")
    return payload
