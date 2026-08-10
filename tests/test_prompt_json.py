"""프롬프트 JSON 직렬화 헬퍼.

검증 계약:
  - 한글을 이스케이프하지 않는다.
  - 콤마·콜론 뒤에 공백을 넣지 않는다.
  - 빈 값(None / "" / 빈 컨테이너)은 키째로 뺀다.
  - 0 과 False 는 빼지 않는다.
  - date/time 처럼 직렬화되지 않는 값이 섞여도 깨지지 않는다.
  - 원본을 건드리지 않는다.
"""
from __future__ import annotations

import json
from datetime import date, time

from app.llm.prompt_json import dump_prompt_json, prune_empty


def test_korean_is_not_escaped() -> None:
    """한글이 \\u 로 부풀지 않는다."""
    out = dump_prompt_json({"city": "강릉시"})
    assert out == '{"city":"강릉시"}'


def test_separators_have_no_spaces() -> None:
    """콤마·콜론 뒤 공백을 넣지 않는다."""
    out = dump_prompt_json({"a": 1, "b": [1, 2]})
    assert " " not in out
    assert out == '{"a":1,"b":[1,2]}'


def test_empty_values_are_dropped() -> None:
    """빈 값은 키째로 사라진다."""
    out = json.loads(
        dump_prompt_json(
            {
                "name": "장소",
                "address": "",
                "category": None,
                "review_snippets": [],
                "meta": {},
            }
        )
    )
    assert out == {"name": "장소"}


def test_zero_and_false_survive() -> None:
    """0 과 False 는 값이 있는 것이다 — 빈 값과 구분한다.

    강수확률 0 을 지우면 "비 안 옴" 과 "예보 없음" 이 같은 모양이 된다.
    """
    out = json.loads(
        dump_prompt_json(
            {"precipitation_prob": 0, "indoor_ratio": 0.0, "flag": False}
        )
    )
    assert out == {
        "precipitation_prob": 0,
        "indoor_ratio": 0.0,
        "flag": False,
    }


def test_nested_empty_collapses_outward() -> None:
    """안쪽이 전부 비면 바깥 키도 함께 사라진다."""
    out = json.loads(dump_prompt_json({"a": {"b": None, "c": ""}, "d": 1}))
    assert out == {"d": 1}


def test_empty_items_are_dropped_from_lists() -> None:
    """리스트 안의 빈 원소도 걷어 낸다."""
    out = json.loads(dump_prompt_json({"xs": [{"a": 1}, {}, None, ""]}))
    assert out == {"xs": [{"a": 1}]}


def test_unserializable_values_become_strings() -> None:
    """date/time 이 섞여도 프롬프트 조립이 깨지지 않는다."""
    out = json.loads(
        dump_prompt_json({"d": date(2026, 7, 6), "t": time(9, 0)})
    )
    assert out == {"d": "2026-07-06", "t": "09:00:00"}


def test_source_object_is_not_mutated() -> None:
    """원본은 그대로 둔다 — 뷰만 다듬는 불변식을 지킨다."""
    src = {"a": None, "b": [None, 1]}
    prune_empty(src)
    assert src == {"a": None, "b": [None, 1]}
