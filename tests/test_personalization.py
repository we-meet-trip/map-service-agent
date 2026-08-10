"""테마 친화 가점.

검증 계약(가장 중요한 것부터):
  - 테마가 없거나 스위치가 꺼져 있으면 **도입 전과 완전히 같다.**
  - 가점은 상한(_AFFINITY_MAX)을 절대 넘지 않는다.
  - 목록 앞의 테마가 뒤의 테마를 이긴다(명시 선택이 프로필을 이긴다).
  - RULES_ENABLED=false 면 함께 꺼진다.
  - LLM 호출 수는 변하지 않는다.
"""
from __future__ import annotations

import asyncio
from datetime import date, time

from app import agent_dependencies as deps
from app.agent_settings import get_settings
from app.nodes.agent_nodes import (
    _AFFINITY_MAX,
    _affinity,
    score_and_rank,
)
from app.schemas.agent_schemas import AgentRequest, DateRange


def _request(theme=None) -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="강원특별자치도",
        city="강릉시",
        theme=theme,
    )


def _cand(idx: int, *, group=None, category="") -> dict:
    return {
        "content_id": f"c{idx}",
        "name": f"장소{idx}",
        "lat": 37.8,
        "lng": 128.9,
        "category_group_code": group,
        "category": category,
    }


class _FakeHub:
    """base_score 를 그대로 점수로 돌려주는 룰 엔진 대역.

    hub 는 base_score 에 실내 보너스만 얹으므로, 보너스 조건을 만들지
    않으면 여기서 나온 값이 곧 agent 가 넣은 base_score 다.
    """

    def __init__(self) -> None:
        self.pois: list[dict] = []

    async def score_indoor_bonus(self, pois, day_pop_max):
        self.pois = pois
        return {
            "scored": [
                {"content_id": p["content_id"], "score": p["base_score"]}
                for p in pois
            ]
        }


def _run(theme, candidates) -> _FakeHub:
    hub = _FakeHub()
    deps.set_hub_client(hub)
    try:
        asyncio.run(
            score_and_rank(
                {
                    "job_id": "j",
                    "request": _request(theme),
                    "candidates": candidates,
                    "grounded": True,
                }
            )
        )
    finally:
        deps.reset_all()
    return hub


def _base_scores(hub: _FakeHub) -> list[float]:
    return [p["base_score"] for p in hub.pois]


# ─── 무변화 계약 ──────────────────────────────────────────────────

def test_no_theme_means_no_change() -> None:
    """테마가 없으면 base_score 가 기존 rank-decay 그대로다."""
    cands = [_cand(0, group="FD6"), _cand(1, group="CE7")]
    assert _base_scores(_run(None, cands)) == [1.0, 0.99]


def test_switch_off_means_no_change(monkeypatch) -> None:
    """스위치를 끄면 테마가 있어도 도입 전과 같다."""
    settings = get_settings()
    monkeypatch.setattr(settings, "PERSONALIZATION_ENABLED", False)
    cands = [_cand(0, group="FD6"), _cand(1, group="CE7")]
    assert _base_scores(_run(["food"], cands)) == [1.0, 0.99]


def test_rules_switch_off_disables_affinity_too(monkeypatch) -> None:
    """룰을 끄면 노드 자체가 통과하므로 가점도 함께 사라진다."""
    settings = get_settings()
    monkeypatch.setattr(settings, "RULES_ENABLED", False)
    hub = _run(["food"], [_cand(0, group="FD6")])
    assert hub.pois == [], "룰이 꺼졌는데 hub 를 불렀다"


def test_unknown_theme_scores_nothing() -> None:
    """매핑에 없는 테마는 가점을 만들지 않는다."""
    assert _affinity(_cand(0, group="FD6"), ["산책"]) == 0.0


# ─── 가점 규칙 ────────────────────────────────────────────────────

def test_matching_candidate_gets_bounded_bonus() -> None:
    """맞는 후보만 가점을 받고, 상한을 넘지 않는다."""
    scores = _base_scores(
        _run(["food"], [_cand(0, group="CE7"), _cand(1, group="FD6")])
    )
    assert scores[0] == 1.0                      # 카페 — 걸리지 않음
    assert scores[1] == 0.99 + _AFFINITY_MAX     # 음식점 — 가점


def test_category_text_matches_when_group_code_is_absent() -> None:
    """그룹 코드가 없는 두루누비 코스는 분류 텍스트로 걸린다."""
    course = _cand(0, group=None, category="걷기길")
    assert _affinity(course, ["nature"]) == _AFFINITY_MAX


def test_earlier_theme_outranks_later_one() -> None:
    """앞 테마가 뒤 테마를 이긴다 — 명시 선택이 프로필을 이긴다.

    호출 측이 사용자가 직접 고른 테마를 앞에 두므로, 순서를 점수로 옮기면
    그것만으로 우선순위가 선다.
    """
    first = _affinity(_cand(0, group="FD6"), ["food", "cafe"])
    second = _affinity(_cand(1, group="CE7"), ["food", "cafe"])
    assert first == _AFFINITY_MAX
    assert 0 < second < first


def test_multiple_matches_take_the_max_not_the_sum() -> None:
    """여러 테마에 걸려도 합하지 않는다.

    합하면 두루 걸치는 무난한 장소가 한 테마에 정확히 맞는 장소를 이긴다.
    """
    both = _cand(0, group="AT4", category="관광,명소 > 공원")
    got = _affinity(both, ["photo", "nature", "activity"])
    assert got == _AFFINITY_MAX


def test_bonus_never_exceeds_indoor_rule_weight() -> None:
    """취향 가점이 hub 실내 가점(0.15)보다 작아야 한다.

    날씨·이동수단 같은 물리적 제약이 취향을 항상 이겨야 하기 때문이다.
    """
    assert _AFFINITY_MAX < 0.15


def test_missing_category_fields_are_safe() -> None:
    """분류 정보가 없는 후보에서도 예외 없이 0 을 돌려준다."""
    assert _affinity({}, ["food"]) == 0.0
    assert _affinity(_cand(0), ["food"]) == 0.0
    assert _affinity(_cand(0, group="FD6"), [None, 123]) == 0.0
