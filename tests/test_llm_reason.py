"""llm_reason 노드 + build_payload 병합 + publish_done clothing 테스트."""
from __future__ import annotations

import asyncio
import json
from datetime import date, time

from app import agent_dependencies as deps
from app.agent_settings import get_settings
from app.nodes.agent_nodes import build_payload, llm_reason, publish_done
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    Place,
    PlaceReason,
    ReasonEnvelope,
)


def _request() -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
    )


def _places(n: int = 2) -> list[Place]:
    return [
        Place(
            place_id=i,
            day=1,
            name=f"장소{i}",
            address="주소",
            lat=37.5,
            lng=127.0,
            recommended_visit_time="오전",
        )
        for i in range(n)
    ]


def _envelope(ids: list[int], clothing: str = "가벼운 겉옷") -> ReasonEnvelope:
    return ReasonEnvelope(
        reasons=[
            PlaceReason(place_id=i, reason=f"이유{i}") for i in ids
        ],
        clothing=clothing,
    )


class _SeqGemini:
    """호출 순서대로 준비된 결과(또는 예외)를 돌려주는 대역."""

    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls = 0

    async def generate_structured(self, prompt, schema, *, system_instruction=None):
        self.calls += 1
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _state(**over) -> dict:
    st = {"job_id": "j", "request": _request(), "places": _places()}
    st.update(over)
    return st


def test_normal_full_coverage() -> None:
    """전체 커버 시 reasons/clothing 이 채워지고 예산 1 소비."""
    deps.set_gemini_client(_SeqGemini([_envelope([0, 1])]))
    try:
        out = asyncio.run(llm_reason(_state()))
    finally:
        deps.reset_all()
    assert out["reasons"] == {0: "이유0", 1: "이유1"}
    assert out["clothing"] == "가벼운 겉옷"
    assert out["llm_calls_used"] == 1
    assert "degraded_reason" not in out


def test_budget_exhausted_degrades() -> None:
    """예산 소진 시 호출 없이 degrade(잡 실패 아님).

    소진 기준값은 설정에서 읽는다 — 예산 상한이 바뀌어도 "이미 다 썼다"
    는 조건이 그대로 유지되게 한다.
    """
    gemini = _SeqGemini([_envelope([0, 1])])
    deps.set_gemini_client(gemini)
    max_calls = get_settings().GEMINI_MAX_CALLS_PER_REQUEST
    try:
        out = asyncio.run(llm_reason(_state(llm_calls_used=max_calls)))
    finally:
        deps.reset_all()
    assert gemini.calls == 0
    assert out["degraded_reason"] == "llm_budget_exhausted"
    assert out.get("error") is None
    assert "reasons" not in out


def test_partial_coverage_retries_with_feedback() -> None:
    """커버리지 미달 시 누락 id 를 피드백해 1회 보완 호출한다.

    주의: 예산 3의 실파이프라인에서는 본 노드 진입이 3회째 소비라
    보완 호출이 발생하지 않는다(기본 결과 = 부분 유지 + degrade,
    아래 test_partial_coverage_without_budget 가 그 경로).
    본 테스트는 예산 상한 상향 시의 보완 메커니즘 자체를 검증한다.
    """
    gemini = _SeqGemini([_envelope([0]), _envelope([0, 1], "우산 필수")])
    deps.set_gemini_client(gemini)
    try:
        out = asyncio.run(llm_reason(_state()))
    finally:
        deps.reset_all()
    assert gemini.calls == 2
    assert out["reasons"] == {0: "이유0", 1: "이유1"}
    assert out["clothing"] == "우산 필수"
    assert "degraded_reason" not in out


def test_partial_coverage_without_budget_keeps_partial() -> None:
    """보완 예산이 없으면 부분 결과 유지 + degrade 표기."""
    gemini = _SeqGemini([_envelope([0])])
    deps.set_gemini_client(gemini)
    try:
        # 이미 2회 소비 → 이번 호출로 3회 = 예산 소진, 보완 불가
        out = asyncio.run(llm_reason(_state(llm_calls_used=2)))
    finally:
        deps.reset_all()
    assert gemini.calls == 1
    assert out["reasons"] == {0: "이유0"}
    assert out["degraded_reason"] == "reason_coverage_partial"
    assert out.get("error") is None


def test_out_of_range_ids_are_dropped() -> None:
    """범위 밖 place_id 는 폐기된다(모델 창작 id 차단)."""
    gemini = _SeqGemini(
        [_envelope([0, 1, 99]), _envelope([0, 1, 99])]
    )
    deps.set_gemini_client(gemini)
    try:
        out = asyncio.run(llm_reason(_state()))
    finally:
        deps.reset_all()
    assert set(out["reasons"]) == {0, 1}


def test_error_state_noop() -> None:
    """에러 상태에선 no-op(호출 없음)."""
    gemini = _SeqGemini([_envelope([0, 1])])
    deps.set_gemini_client(gemini)
    try:
        out = asyncio.run(llm_reason(_state(error="boom")))
    finally:
        deps.reset_all()
    assert gemini.calls == 0
    assert "reasons" not in out


def test_exception_degrades_not_fails() -> None:
    """타임아웃 등 예외도 degrade — error 를 세우지 않는다."""
    gemini = _SeqGemini([asyncio.TimeoutError()])
    deps.set_gemini_client(gemini)
    try:
        out = asyncio.run(llm_reason(_state()))
    finally:
        deps.reset_all()
    assert out.get("error") is None
    assert out["degraded_reason"].startswith("llm_reason_failed:")


def test_build_payload_merges_reasons() -> None:
    """build_payload 가 reasons 를 Place.reason 으로 병합한다."""
    st = _state(reasons={0: "이유0"}, clothing="겉옷")
    out = asyncio.run(build_payload(st))
    assert out["places"][0].reason == "이유0"
    assert out["places"][1].reason is None


class _CapturePublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def publish(self, job_id, status, payload_json):
        self.payloads.append(json.loads(payload_json))
        return "1-0"


def test_publish_done_includes_reason_and_clothing() -> None:
    """publish_done 페이로드에 reason·clothing 이 직렬화된다."""
    pub = _CapturePublisher()
    deps.set_streams_publisher(pub)
    try:
        st = _state(
            reasons={0: "이유0", 1: "이유1"},
            clothing="겉옷",
            visit_order=[0, 1],
            legs=[],
        )
        st = asyncio.run(build_payload(st))
        # legs 길이 검증은 recommend_route 소관 — 여기선 직렬화만 본다.
        st["legs"] = []
        st["visit_order"] = [0, 1]
        asyncio.run(publish_done(st))
    finally:
        deps.reset_all()
    payload = pub.payloads[0]
    assert payload["clothing"] == "겉옷"
    assert payload["places"][0]["reason"] == "이유0"
    assert payload["status"] == "done"
