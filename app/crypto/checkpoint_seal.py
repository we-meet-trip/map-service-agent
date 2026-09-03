"""langgraph 체크포인트를 저장 전에 봉한다.

체크포인터는 대화 상태를 langgraph 스키마에 그대로 쓰는데, 그 상태에는
사용자가 고른 좌표가 담긴다. 덤프나 백업을 얻은 쪽이 좌표를 바로 읽을 수
있었으므로, 저장 직전에 감싸 DB 에는 암호문만 남긴다.

열쇠는 location_seal 과 같은 32바이트 base64 인데, 여기서는 여러 개를 kid
로 구분해 둔다. 기록마다 어떤 kid 로 봉했는지가 "aesgcm:<kid>" 로 함께
남고 읽을 때 그 kid 의 열쇠를 골라 열므로, 열쇠를 새로 바꿔도 옛 kid 를
목록에 남겨 두면 옛 기록을 계속 읽을 수 있다. 봉하기 전에 쌓인 평문 기록은
표식이 없어서 바깥 serializer 가 그대로 통과시킨다.
"""
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.agent_settings import AgentSettings

CIPHER_PREFIX = "aesgcm:"
AAD = b"map|ckpt|v1"
KEY_BYTES = 32
IV_BYTES = 12


class CheckpointSealError(Exception):
    """봉할 수 없거나 열 수 없다. 열쇠 설정이 잘못된 경우가 대부분이다."""


def parse_keys(raw: str) -> dict[str, bytes]:
    """"kid:base64" 쉼표 목록을 열쇠 맵으로 푼다."""
    keys: dict[str, bytes] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        kid, sep, encoded = entry.partition(":")
        kid = kid.strip()
        if not sep or not kid:
            raise CheckpointSealError("checkpoint key entry must be kid:base64")
        if kid in keys:
            raise CheckpointSealError(f"duplicate checkpoint kid {kid!r}")
        try:
            key = base64.b64decode(encoded.strip(), validate=True)
        except Exception as exc:  # noqa: BLE001 - 어떤 형태든 설정 오류로 접는다
            raise CheckpointSealError(
                f"checkpoint key {kid!r} is not base64"
            ) from exc
        if len(key) != KEY_BYTES:
            raise CheckpointSealError(
                f"checkpoint key {kid!r} must decode to 32 bytes"
            )
        keys[kid] = key
    if not keys:
        raise CheckpointSealError("no checkpoint keys configured")
    return keys


class CheckpointCipher:
    """EncryptedSerializer 에 꽂는 AESGCM cipher. kid 로 열쇠를 고른다."""

    def __init__(self, keys: dict[str, bytes], active_kid: str) -> None:
        if active_kid not in keys:
            raise CheckpointSealError(f"active kid {active_kid!r} not in key map")
        self._keys = keys
        self._active_kid = active_kid

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        # 난수 IV 를 매번 새로 뽑아 암호문 앞에 붙여 둔다. 같은 상태를 두 번
        # 저장해도 남는 바이트가 매번 달라진다.
        iv = os.urandom(IV_BYTES)
        key = self._keys[self._active_kid]
        return (
            CIPHER_PREFIX + self._active_kid,
            iv + AESGCM(key).encrypt(iv, plaintext, AAD),
        )

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if not ciphername.startswith(CIPHER_PREFIX):
            raise CheckpointSealError(f"unsupported cipher {ciphername!r}")
        key = self._keys.get(ciphername[len(CIPHER_PREFIX):])
        if key is None:
            raise CheckpointSealError("unknown checkpoint kid")
        try:
            return AESGCM(key).decrypt(
                ciphertext[:IV_BYTES], ciphertext[IV_BYTES:], AAD
            )
        except Exception as exc:
            # 열쇠가 다르거나 내용이 손대어진 경우다. 어느 쪽인지 구분해
            # 주지 않는다 — 밖에서 차이를 보고 바꿔 가며 시도할 수 있기 때문.
            raise CheckpointSealError("cannot open checkpoint record") from exc


def build_cipher(settings: AgentSettings) -> CheckpointCipher:
    """설정에서 열쇠를 읽어 cipher 를 만든다. 잘못돼 있으면 예외로 부팅을 막는다."""
    raw = settings.CHECKPOINT_ENC_KEYS.get_secret_value()
    if not raw:
        raise CheckpointSealError("CHECKPOINT_ENC_KEYS not configured")
    keys = parse_keys(raw)
    # 좌표 전송 열쇠를 그대로 돌려쓰는 것은 거부한다. 하나가 새면 전송로와
    # 저장소가 한꺼번에 열리므로 용도마다 다른 열쇠를 강제한다.
    wire = settings.LOCATION_WIRE_KEY.get_secret_value()
    if wire:
        try:
            wire_key = base64.b64decode(wire)
        except Exception:  # noqa: BLE001 - 못 푸는 값이면 겹칠 일도 없다
            wire_key = None
        if wire_key in keys.values():
            raise CheckpointSealError(
                "checkpoint keys must differ from LOCATION_WIRE_KEY"
            )
    return CheckpointCipher(keys, settings.CHECKPOINT_ENC_ACTIVE_KID)
