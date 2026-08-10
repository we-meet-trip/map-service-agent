"""응답 스키마가 매 호출 입력에 얹는 무게를 감시한다.

`response_schema` 로 넘긴 모델은 그 JSON 스키마가 프롬프트 입력의 일부로
함께 나간다. 그리고 pydantic 은 **클래스 docstring 을 스키마의
description 으로 싣는다.** 즉 설계 배경을 docstring 에 적으면 그 글자
수만큼을 호출마다 다시 지불한다.

그래서 응답 스키마로 쓰는 모델은 docstring 을 한 줄로 두고 배경은 주석에
남긴다. 주석은 스키마에 실리지 않는다.

여기서 보는 것은 "설명문이 스키마를 지배하지 않는가" 이지 정확한 바이트
수가 아니다. 필드가 늘어 스키마가 커지는 것은 정상이므로 총량 상한은
느슨하게 두고, 설명문 비중만 조인다.
"""
from __future__ import annotations

import json

from app.schemas.agent_schemas import (
    BulletsEnvelope,
    InventedPlaces,
    PlacesSelection,
    ReasonEnvelope,
)

# 모델에게 실제로 넘기는 응답 스키마 전부.
RESPONSE_SCHEMAS = [
    InventedPlaces,
    PlacesSelection,
    ReasonEnvelope,
    BulletsEnvelope,
]

# 설명문이 스키마에서 차지해도 되는 최대 비중.
_DESCRIPTION_SHARE_MAX = 0.25
# 한 줄 요약을 넘어선 docstring 을 걸러 내는 길이(자).
_DOCSTRING_LEN_MAX = 40


def _description_chars(schema: dict) -> int:
    """스키마 전체(중첩 정의 포함)의 description 글자 수."""
    total = len(schema.get("description", ""))
    for nested in (schema.get("$defs") or {}).values():
        total += len(nested.get("description", ""))
    return total


def test_descriptions_do_not_dominate_response_schemas() -> None:
    """설명문이 스키마의 4분의 1을 넘지 않는다."""
    for model in RESPONSE_SCHEMAS:
        schema = model.model_json_schema()
        total = len(json.dumps(schema, ensure_ascii=False))
        share = _description_chars(schema) / total
        assert share <= _DESCRIPTION_SHARE_MAX, (
            f"{model.__name__} 의 스키마가 설명문으로 채워져 있다"
            f"({share:.0%}). 배경 설명은 docstring 이 아니라 주석에 둔다."
        )


def test_response_schema_docstrings_stay_one_line() -> None:
    """응답 스키마와 그 중첩 모델의 docstring 은 한 줄로 둔다."""
    seen: set[type] = set()

    def walk(model) -> None:
        if model in seen:
            return
        seen.add(model)
        doc = (model.__doc__ or "").strip()
        assert "\n" not in doc and len(doc) <= _DOCSTRING_LEN_MAX, (
            f"{model.__name__} 의 docstring 이 길다. 그대로 모델에게 실려"
            f" 나가므로 배경은 클래스 위 주석으로 옮긴다."
        )
        for field in model.model_fields.values():
            for arg in getattr(field.annotation, "__args__", ()) or ():
                if hasattr(arg, "model_fields"):
                    walk(arg)

    for m in RESPONSE_SCHEMAS:
        walk(m)
    # 중첩 모델까지 실제로 훑었는지 확인한다 — 루프가 조용히 비면
    # 이 테스트가 아무것도 지키지 않는다.
    assert len(seen) > len(RESPONSE_SCHEMAS)
