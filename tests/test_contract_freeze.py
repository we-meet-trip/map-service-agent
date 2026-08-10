"""외부에 노출되는 요청·응답 형태를 고정한다.

이 테스트가 지키는 것은 **모양**이지 값이 아니다. 필드 이름(와이어 이름
기준 — alias 가 있으면 alias), 타입, 필수 여부 세 가지만 본다. 값은 모델
응답이라 실행할 때마다 달라져 고정할 수 없다.

왜 필요한가:
  내부 구현을 고치다 보면 응답 모델에 필드 하나를 얹는 일이 쉽게 벌어진다.
  받는 쪽(BFF·client)은 모르는 필드를 무시하도록 되어 있어 그 순간에는
  아무도 실패하지 않고, 한참 뒤에 계약이 어긋난 채로 굳은 것을 발견한다.
  여기서 미리 막는다.

고칠 때:
  계약을 **의도적으로** 바꾸는 경우에만 골든을 함께 고친다. 테스트가
  빨간불이면 먼저 "이 필드가 정말 나가야 하는가" 를 묻고, 답이 예일 때만
  골든을 갱신한다. 골든을 먼저 맞추고 나중에 생각하는 순서는 이 테스트를
  무의미하게 만든다.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, time
from typing import Literal, Union, get_args, get_origin

from app import agent_dependencies as deps
from app.nodes.agent_nodes import publish_done
from app.schemas import agent_schemas as S


def _type_repr(tp) -> str:
    """타입 힌트를 골든에 적기 좋은 짧은 문자열로 바꾼다.

    Optional/Literal/Annotated 를 사람이 읽는 형태로 접는다. Annotated 는
    붙어 있는 제약(min_length 등)을 떼고 바탕 타입만 남긴다 — 여기서 보는
    것은 타입이지 검증 규칙이 아니다.
    """
    if tp is type(None):
        return "None"
    origin = get_origin(tp)
    if origin is None:
        return getattr(tp, "__name__", str(tp))
    args = get_args(tp)
    if origin is Literal:
        return "Literal[" + ",".join(str(a) for a in args) + "]"
    if getattr(origin, "__name__", "") == "Annotated":
        return _type_repr(args[0])
    reps = [_type_repr(a) for a in args]
    if origin is Union:
        if len(reps) == 2 and "None" in reps:
            base = next(r for r in reps if r != "None")
            return f"Optional[{base}]"
        return "Union[" + ",".join(reps) + "]"
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{','.join(reps)}]"


def _shape(model) -> dict[str, str]:
    """모델의 와이어 형태를 `필드 -> "타입|req|opt"` 로 만든다."""
    out: dict[str, str] = {}
    for name, field in model.model_fields.items():
        wire = field.alias or name
        req = "req" if field.is_required() else "opt"
        out[wire] = f"{_type_repr(field.annotation)}|{req}"
    return out


# 지금 밖으로 나가고 있는 형태. 아래 값은 손으로 적은 것이 아니라 현재
# 모델에서 뽑아 박은 것이다.
GOLDEN: dict[str, dict[str, str]] = {
    # POST /v1/recommend 요청 본문
    "AgentRequest": {
        "date": "DateRange|req",
        "budget": "Optional[int]|opt",
        "theme": "Optional[list[str]]|opt",
        "mobility": "Optional[Mobility]|opt",
        "province": "str|req",
        "city": "str|req",
        "schedule_id": "Optional[str]|opt",
        "stage": "Literal[init,mode1,route]|opt",
        "exclude": "Optional[list[str]]|opt",
        "places": "Optional[list[SelectedPlace]]|opt",
    },
    # AgentRequest.date 중첩
    "DateRange": {
        "date_start": "date|req",
        "date_end": "date|req",
        "time_start": "time|req",
        "time_end": "time|req",
    },
    # AgentRequest.places[] 중첩
    "SelectedPlace": {
        "name": "str|req",
        "address": "str|opt",
        "lat": "float|req",
        "lng": "float|req",
        "day": "int|opt",
        "content_id": "Optional[str]|opt",
        "category": "Optional[str]|opt",
    },
    # POST /v1/recommend 202 응답
    "AgentJobAccepted": {
        "job_id": "str|req",
        "status": "Literal[in_progress]|opt",
        "retry_after_seconds": "int|opt",
    },
    # agent:jobs:done 스트림 메시지 payload
    "JobDonePayload": {
        "job_id": "str|req",
        "status": "Literal[done,failed]|req",
        "places": "Optional[list[Place]]|opt",
        "visit_order": "Optional[list[int]]|opt",
        "legs": "Optional[list[Leg]]|opt",
        "clothing": "Optional[str]|opt",
        "timeline_status": "Optional[Literal[ok,trimmed,unverified]]|opt",
        "warnings": "Optional[list[str]]|opt",
        "error": "Optional[str]|opt",
    },
    # JobDonePayload.places[] 중첩
    "Place": {
        "place_id": "int|req",
        "day": "int|req",
        "name": "str|req",
        "address": "str|req",
        "lat": "float|req",
        "lng": "float|req",
        "recommended_visit_time": "str|req",
        "content_id": "Optional[str]|opt",
        "source": "Optional[str]|opt",
        "category": "Optional[str]|opt",
        "category_group_code": "Optional[str]|opt",
        "phone": "Optional[str]|opt",
        "place_url": "Optional[str]|opt",
        "crs_dstnc_km": "Optional[float]|opt",
        "crs_total_min": "Optional[int]|opt",
        "crs_level": "Optional[int]|opt",
        "brd_div": "Optional[str]|opt",
        "gpx_url": "Optional[str]|opt",
        "route_idx": "Optional[str]|opt",
        "grounded": "bool|opt",
        "stay_minutes": "Optional[int]|opt",
        "visit_start": "Optional[str]|opt",
        "visit_end": "Optional[str]|opt",
        "stay_source": "Optional[str]|opt",
        "reason": "Optional[str]|opt",
        "bullets": "Optional[list[str]]|opt",
    },
    # JobDonePayload.legs[] 중첩 (from/to 는 alias)
    "Leg": {
        "from": "int|req",
        "to": "int|req",
        "mode": "Mobility|req",
        "estimated_distance_km": "float|req",
        "estimated_duration_min": "int|req",
    },
    # POST /v1/reviews/summary 요청
    "ReviewsSummaryRequest": {
        "place_name": "str|req",
        "category": "Optional[str]|opt",
        "reviews": "list[ReviewSnippet]|req",
    },
    # POST /v1/reviews/summary 응답
    "ReviewsSummaryResponse": {
        "bullets": "list[str]|opt",
    },
    # POST /v1/reviews/summary/batch 요청
    "ReviewsSummaryBatchRequest": {
        "places": "list[SummaryPlaceRequest]|req",
    },
    # POST /v1/reviews/summary/batch 응답
    "ReviewsSummaryBatchResponse": {
        "results": "list[PlaceSummaryResult]|opt",
    },
    # 배치 요청 places[] 중첩
    "SummaryPlaceRequest": {
        "place_name": "str|req",
        "category": "Optional[str]|opt",
        "reviews": "list[ReviewSnippet]|req",
    },
    # 요약 요청 reviews[] 중첩
    "ReviewSnippet": {
        "title": "str|opt",
        "description": "str|req",
    },
    # 배치 응답 results[] 중첩
    "PlaceSummaryResult": {
        "index": "int|req",
        "bullets": "list[str]|req",
    },
}


def test_schema_shapes_are_frozen() -> None:
    """밖으로 나가는 모델의 필드·타입·필수 여부가 골든과 같다."""
    for name, expected in GOLDEN.items():
        actual = _shape(getattr(S, name))
        assert actual == expected, (
            f"{name} 의 계약이 달라졌다. 의도한 변경이면 GOLDEN 을 함께 "
            f"고쳐라.\n  기대: {expected}\n  실제: {actual}"
        )


class _CapturePublisher:
    """발행된 payload 를 파싱해 모아 두는 대역."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def publish(self, job_id, status, payload_json):
        self.payloads.append(json.loads(payload_json))
        return "1-0"


def _request() -> S.AgentRequest:
    return S.AgentRequest(
        date=S.DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
        mobility=S.Mobility.walk,
    )


def _place(place_id: int) -> S.Place:
    return S.Place(
        place_id=place_id,
        day=1,
        name=f"장소{place_id}",
        address="주소",
        lat=37.5,
        lng=127.0,
        recommended_visit_time="오전",
    )


def _publish(state: dict) -> dict:
    """publish_done 을 돌려 실제로 나간 payload 를 돌려준다."""
    pub = _CapturePublisher()
    deps.set_streams_publisher(pub)
    try:
        asyncio.run(publish_done(state))
    finally:
        deps.reset_all()
    return pub.payloads[0]


def test_done_payload_wire_keys_are_frozen() -> None:
    """성공 payload 가 실제로 내보내는 키 집합을 고정한다.

    모델 선언이 아니라 `publish_done` 이 만든 결과를 본다. 선언에는 있는데
    노드가 채우지 않는 필드, 반대로 노드가 얹은 필드까지 함께 잡힌다.
    """
    payload = _publish(
        {
            "job_id": "job-1",
            "request": _request(),
            "places": [_place(0), _place(1)],
            "visit_order": [0, 1],
            "legs": [
                S.Leg(
                    **{
                        "from": 0,
                        "to": 1,
                        "mode": S.Mobility.walk,
                        "estimated_distance_km": 1.0,
                        "estimated_duration_min": 15,
                    }
                )
            ],
            "clothing": "겉옷",
            "timeline_status": "ok",
            "warnings": ["주의"],
        }
    )
    assert set(payload) == set(GOLDEN["JobDonePayload"])
    assert set(payload["places"][0]) == set(GOLDEN["Place"])
    assert set(payload["legs"][0]) == set(GOLDEN["Leg"])
    assert payload["status"] == "done"


def test_failed_payload_wire_keys_are_frozen() -> None:
    """실패 payload 도 같은 키 집합으로 나간다.

    실패라고 필드를 빼면 받는 쪽이 두 가지 모양을 다뤄야 한다. 성공과 같은
    모양을 유지하고 값만 비운다.
    """
    payload = _publish(
        {
            "job_id": "job-2",
            "request": _request(),
            "error": "recommend_places failed",
        }
    )
    assert set(payload) == set(GOLDEN["JobDonePayload"])
    assert payload["status"] == "failed"
    assert payload["places"] is None
