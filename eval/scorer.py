"""추천 결과를 사람 손 없이 채점한다.

왜 필요한가:
  모델을 바꾸든 랭킹을 학습시키든, 나아졌는지 나빠졌는지 말하려면 같은 잣대로
  잰 숫자가 있어야 한다. 지금은 그 잣대가 없어서 "좋아 보인다" 밖에 말할 수
  없다.

왜 LLM 으로 채점하지 않는가:
  이 잣대를 쓰는 목적 하나가 "외부 모델 의존을 줄였는데 품질이 유지되는가" 다.
  채점하는 쪽이 외부 모델이면 그 질문에 답할 수 없다 — 재는 자와 재는 대상이
  같은 것에 묶인다. 그래서 여기에는 망을 타는 코드가 없고, 같은 입력이면 늘
  같은 점수가 나온다.

무엇을 재는가 (다섯 축):
  schema      결과가 약속된 형태인가
  grounded    실측 후보로 골랐는가, 아니면 지어냈는가
  rules       이동수단 반경을 넘긴 장소가 있는가
  route       방문 순서가 장소 집합과 맞아떨어지는가
  timeline    방문 시각이 하루 안에서 앞뒤가 맞는가

주의: 뒤의 세 축은 모델보다 검색·결정론적 노드의 영향을 크게 받는다. 모델을
바꿔 비교할 때는 그 점을 감안해 읽어야 하며, 세 축이 함께 움직이면 모델이
아니라 그 앞단이 바뀐 것이다.
"""
from __future__ import annotations

from typing import Any


def _straight_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 좌표 사이 대권거리(km).

    agent 의 동선 계산과 같은 식을 쓴다. 채점이 본편과 다른 자로 재면
    본편은 통과인데 채점만 실패하는 일이 생긴다.
    """
    from app.nodes.agent_nodes import _straight_km as impl
    return impl(lat1, lng1, lat2, lng2)


def _minutes(hhmm: str | None) -> int | None:
    """"HH:MM" → 자정 기준 분. 형태가 아니면 None."""
    if not hhmm or len(hhmm) != 5 or hhmm[2] != ":":
        return None
    try:
        return int(hhmm[:2]) * 60 + int(hhmm[3:])
    except ValueError:
        return None


def score_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """약속된 형태인가. 여기서 떨어지면 나머지 축은 잴 것이 없다."""
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "not an object"}
    if payload.get("status") != "done":
        return {"ok": False, "reason": f"status={payload.get('status')}"}
    places = payload.get("places")
    if not isinstance(places, list) or not places:
        return {"ok": False, "reason": "places empty"}
    for p in places:
        for key in ("day", "name", "lat", "lng"):
            if p.get(key) is None:
                return {"ok": False, "reason": f"place missing {key}"}
    return {"ok": True, "place_count": len(places)}


def score_grounded(training: dict[str, Any] | None) -> dict[str, Any]:
    """실측 후보로 골랐는가.

    창작으로 내려간 비율이 오르면 존재하지 않는 장소가 나올 여지가 커진다.
    모델을 바꿀 때 가장 먼저 나빠지는 축이라 따로 센다.
    """
    if not training:
        return {"known": False}
    return {
        "known": True,
        "grounded": bool(training.get("grounded")),
        "path": training.get("path"),
        "invented": training.get("path") in ("invent", "select_then_invent"),
    }


def score_rules(payload: dict[str, Any], max_radius_km: float | None) -> dict[str, Any]:
    """이동수단 반경을 넘긴 장소를 센다.

    기준점은 그날 장소들의 중심이다. agent 가 출발지를 문자열로만 알고 있어
    같은 근사를 쓴다(승인된 근사).

    max_radius_km 가 없으면(자동차·대중교통) 잴 것이 없다.
    """
    if max_radius_km is None:
        return {"applicable": False}
    places = payload.get("places") or []
    by_day: dict[int, list[dict]] = {}
    for p in places:
        by_day.setdefault(int(p.get("day", 1)), []).append(p)

    violations = []
    for day, group in sorted(by_day.items()):
        clat = sum(float(p["lat"]) for p in group) / len(group)
        clng = sum(float(p["lng"]) for p in group) / len(group)
        for p in group:
            d = _straight_km(clat, clng, float(p["lat"]), float(p["lng"]))
            if d > max_radius_km:
                violations.append({"day": day, "name": p.get("name"), "km": round(d, 2)})
    return {
        "applicable": True,
        "max_radius_km": max_radius_km,
        "violations": len(violations),
        "detail": violations[:5],
    }


def score_route(payload: dict[str, Any]) -> dict[str, Any]:
    """방문 순서가 장소 집합과 맞아떨어지는가.

    빠뜨리거나 중복되면 화면이 그리는 동선과 목록이 어긋난다. 순서가 아예
    없는 경우와 구분해서 센다 — 없는 것과 틀린 것은 다른 문제다.
    """
    places = payload.get("places") or []
    order = payload.get("visit_order")
    ids = [p.get("place_id") for p in places]
    if order is None:
        return {"present": False, "place_count": len(places)}
    problems = []
    if len(order) != len(ids):
        problems.append(f"길이 불일치 order={len(order)} places={len(ids)}")
    if len(set(order)) != len(order):
        problems.append("중복 방문")
    unknown = [i for i in order if i not in ids]
    if unknown:
        problems.append(f"목록에 없는 place_id {unknown[:3]}")
    return {"present": True, "ok": not problems, "problems": problems}


def score_timeline(payload: dict[str, Any],
                   time_start: str | None,
                   time_end: str | None) -> dict[str, Any]:
    """방문 시각이 하루 안에서 앞뒤가 맞는가.

    같은 날 안에서 시작 시각이 뒤로만 가는지, 활동 시간대를 벗어나지 않는지를
    본다. 시각이 비어 있는 결과(경로 전용 등)는 잴 것이 없다고 표시한다.
    """
    places = payload.get("places") or []
    timed = [p for p in places if p.get("visit_start")]
    if not timed:
        return {"applicable": False}

    lo = _minutes(time_start)
    hi = _minutes(time_end)
    problems = []
    by_day: dict[int, list[dict]] = {}
    for p in timed:
        by_day.setdefault(int(p.get("day", 1)), []).append(p)

    for day, group in sorted(by_day.items()):
        mins = [_minutes(p.get("visit_start")) for p in group]
        if any(m is None for m in mins):
            problems.append(f"{day}일차 시각 형식 오류")
            continue
        if mins != sorted(mins):
            problems.append(f"{day}일차 시각 역행")
        if lo is not None and min(mins) < lo:
            problems.append(f"{day}일차 시작 시각 이전")
        if hi is not None and max(mins) > hi:
            problems.append(f"{day}일차 종료 시각 이후")
    return {"applicable": True, "ok": not problems, "problems": problems}


def score_case(case: dict[str, Any],
               payload: dict[str, Any],
               training: dict[str, Any] | None = None) -> dict[str, Any]:
    """한 건을 다섯 축으로 채점한다."""
    expect = case.get("expect") or {}
    schema = score_schema(payload)
    result = {
        "case_id": case.get("id"),
        "schema": schema,
        "grounded": score_grounded(training),
    }
    if not schema["ok"]:
        # 형태가 깨졌으면 나머지는 재도 의미가 없다. 0 점이 아니라
        # "재지 못함" 으로 남긴다 — 둘을 섞으면 평균이 거짓말을 한다.
        result["rules"] = {"applicable": False}
        result["route"] = {"present": False}
        result["timeline"] = {"applicable": False}
        return result
    result["rules"] = score_rules(payload, expect.get("max_radius_km"))
    result["route"] = score_route(payload)
    result["timeline"] = score_timeline(
        payload, expect.get("time_start"), expect.get("time_end"))
    return result


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """전체를 한 줄로 줄인다. 기준선과 견주는 값이 이것이다."""
    total = len(scored)
    if total == 0:
        return {"cases": 0}
    schema_ok = sum(1 for s in scored if s["schema"]["ok"])
    grounded_known = [s for s in scored if s["grounded"].get("known")]
    invented = sum(1 for s in grounded_known if s["grounded"].get("invented"))
    rule_applicable = [s for s in scored if s["rules"].get("applicable")]
    rule_clean = sum(1 for s in rule_applicable if s["rules"]["violations"] == 0)
    route_present = [s for s in scored if s["route"].get("present")]
    route_ok = sum(1 for s in route_present if s["route"]["ok"])
    tl_applicable = [s for s in scored if s["timeline"].get("applicable")]
    tl_ok = sum(1 for s in tl_applicable if s["timeline"]["ok"])

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "cases": total,
        "schema_ok_ratio": ratio(schema_ok, total),
        "invented_ratio": ratio(invented, len(grounded_known)),
        "rule_clean_ratio": ratio(rule_clean, len(rule_applicable)),
        "route_ok_ratio": ratio(route_ok, len(route_present)),
        "timeline_ok_ratio": ratio(tl_ok, len(tl_applicable)),
    }
