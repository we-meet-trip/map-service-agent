"""평가 채점기 테스트.

채점기가 틀리면 모델 교체 판단이 통째로 틀어진다. 통과해야 할 것을 떨어뜨리는
것보다, 떨어져야 할 것을 통과시키는 쪽이 위험하다 — 나빠진 것을 못 보고
넘기게 되기 때문이다. 그래서 "잡아내는가" 를 중심으로 본다.
"""
from __future__ import annotations

from eval.run import compare
from eval.scorer import (
    score_case,
    score_route,
    score_rules,
    score_schema,
    score_timeline,
    summarize,
)


def _place(pid: int, day: int, lat: float, lng: float, start: str | None = None) -> dict:
    p = {"place_id": pid, "day": day, "name": f"장소{pid}",
         "address": "주소", "lat": lat, "lng": lng,
         "grounded": True, "source": "kakao", "content_id": f"kakao:{pid}"}
    if start:
        p["visit_start"] = start
        p["visit_end"] = start
    return p


def _payload(places: list[dict], order: list[int] | None = None) -> dict:
    body = {"status": "done", "places": places}
    if order is not None:
        body["visit_order"] = order
        body["legs"] = [{"from": a, "to": b, "estimated_duration_min": 0}
                        for a, b in zip(order, order[1:])]
    return body


class TestSchema:
    def test_정상_페이로드는_통과한다(self):
        assert score_schema(_payload([_place(0, 1, 37.5, 127.0)]))["ok"]

    def test_실패한_잡은_떨어진다(self):
        assert not score_schema({"status": "failed", "places": []})["ok"]

    def test_장소가_비면_떨어진다(self):
        assert not score_schema(_payload([]))["ok"]

    def test_좌표가_없으면_떨어진다(self):
        broken = {"status": "done", "places": [{"day": 1, "name": "x", "lat": 37.5}]}
        assert not score_schema(broken)["ok"]


class TestRules:
    def test_반경_안이면_위반이_없다(self):
        places = [_place(0, 1, 37.500, 127.000), _place(1, 1, 37.505, 127.005)]
        assert score_rules(_payload(places), 3.0)["violations"] == 0

    def test_반경을_넘기면_잡아낸다(self):
        # 위도 1도는 약 111km. 0.1도면 약 11km 로 도보 반경을 크게 넘는다.
        places = [_place(0, 1, 37.500, 127.000), _place(1, 1, 37.600, 127.000)]
        result = score_rules(_payload(places), 3.0)
        assert result["violations"] > 0

    def test_날짜가_다르면_따로_잰다(self):
        # 서울과 부산이 각각 다른 날이면 그날 안에서는 가까우므로 위반이 없다.
        places = [_place(0, 1, 37.50, 127.00), _place(1, 1, 37.51, 127.01),
                  _place(2, 2, 35.15, 129.05), _place(3, 2, 35.16, 129.06)]
        assert score_rules(_payload(places), 3.0)["violations"] == 0

    def test_반경이_없는_이동수단은_재지_않는다(self):
        places = [_place(0, 1, 37.5, 127.0), _place(1, 1, 35.1, 129.0)]
        assert score_rules(_payload(places), None)["applicable"] is False


class TestRoute:
    def test_순서가_맞으면_통과한다(self):
        places = [_place(0, 1, 37.5, 127.0), _place(1, 1, 37.51, 127.01)]
        assert score_route(_payload(places, [0, 1]))["ok"]

    def test_중복_방문을_잡아낸다(self):
        places = [_place(0, 1, 37.5, 127.0), _place(1, 1, 37.51, 127.01)]
        assert not score_route(_payload(places, [0, 0]))["ok"]

    def test_목록에_없는_번호를_잡아낸다(self):
        places = [_place(0, 1, 37.5, 127.0)]
        assert not score_route(_payload(places, [9]))["ok"]

    def test_순서가_아예_없으면_구분해서_남긴다(self):
        # 없는 것과 틀린 것은 다른 문제다. 섞으면 통과율이 거짓말을 한다.
        places = [_place(0, 1, 37.5, 127.0)]
        assert score_route(_payload(places))["present"] is False


class TestTimeline:
    def test_시각이_앞뒤로_맞으면_통과한다(self):
        places = [_place(0, 1, 37.5, 127.0, "10:00"),
                  _place(1, 1, 37.51, 127.01, "13:00")]
        assert score_timeline(_payload(places), "09:00", "20:00")["ok"]

    def test_역행하는_시각을_잡아낸다(self):
        places = [_place(0, 1, 37.5, 127.0, "15:00"),
                  _place(1, 1, 37.51, 127.01, "11:00")]
        assert not score_timeline(_payload(places), "09:00", "20:00")["ok"]

    def test_활동_시간대를_벗어나면_잡아낸다(self):
        places = [_place(0, 1, 37.5, 127.0, "07:00")]
        assert not score_timeline(_payload(places), "09:00", "20:00")["ok"]

    def test_시각이_없으면_재지_않는다(self):
        places = [_place(0, 1, 37.5, 127.0)]
        assert score_timeline(_payload(places), "09:00", "20:00")["applicable"] is False


class TestCaseAndSummary:
    def test_형태가_깨지면_나머지는_재지_않음으로_남긴다(self):
        # 0 점으로 적으면 평균이 "형태는 멀쩡한데 규칙만 어긴 결과" 와
        # 구분되지 않는다.
        case = {"id": "c1", "expect": {"max_radius_km": 3.0}}
        scored = score_case(case, {"status": "failed", "places": []})
        assert scored["schema"]["ok"] is False
        assert scored["rules"]["applicable"] is False

    def test_창작_경로가_grounded_축에_드러난다(self):
        case = {"id": "c1", "expect": {}}
        scored = score_case(case, _payload([_place(0, 1, 37.5, 127.0)]),
                            {"path": "invent", "grounded": False})
        assert scored["grounded"]["invented"] is True

    def test_요약이_축별_비율을_낸다(self):
        case = {"id": "c1", "expect": {"max_radius_km": 3.0}}
        good = score_case(case, _payload([_place(0, 1, 37.5, 127.0)], [0]),
                          {"path": "select", "grounded": True})
        bad = score_case(case, {"status": "failed", "places": []},
                         {"path": "invent", "grounded": False})
        s = summarize([good, bad])
        assert s["cases"] == 2
        assert s["schema_ok_ratio"] == 0.5
        assert s["invented_ratio"] == 0.5


class TestRegressionCompare:
    def test_높을수록_좋은_축은_떨어질_때_잡는다(self):
        assert compare({"schema_ok_ratio": 0.7}, {"schema_ok_ratio": 0.9}, 0.05)

    def test_낮을수록_좋은_축은_오를_때_잡는다(self):
        # 창작 비율은 오르는 것이 나빠지는 것이다. 방향을 반대로 보면
        # 품질이 무너지는데 통과로 읽힌다.
        assert compare({"invented_ratio": 0.4}, {"invented_ratio": 0.1}, 0.05)

    def test_허용치_안의_흔들림은_넘어간다(self):
        assert not compare({"schema_ok_ratio": 0.97}, {"schema_ok_ratio": 1.0}, 0.05)

    def test_이번에_재지_못한_축을_잡아낸다(self):
        # 재지 못한 것을 조용히 넘기면, 축이 사라진 것이 통과로 보인다.
        assert compare({}, {"rule_clean_ratio": 1.0}, 0.05)


class TestTimelineUsesVisitOrder:
    """시각을 방문 순서대로 견주는지.

    결과에 담긴 배열 순서는 들르는 순서가 아니다. 그대로 읽으면 앞뒤가 멀쩡한
    일정도 역행으로 잡히는데, 실제로 그래서 여섯 건이 전부 빨간 불이었다.
    """

    def _payload(self, visit_order, times):
        return {
            "status": "done",
            "places": [
                {
                    "place_id": i,
                    "day": 1,
                    "name": f"곳{i}",
                    "address": "주소",
                    "lat": 37.5,
                    "lng": 127.0,
                    "recommended_visit_time": t,
                    "visit_start": t,
                    "visit_end": t,
                }
                for i, t in enumerate(times)
            ],
            "visit_order": visit_order,
            "legs": [],
        }

    def test_배열이_뒤섞여도_방문_순서가_맞으면_통과(self):
        from app.nodes import agent_nodes  # noqa: F401  (import 경로 확인용)
        from eval.scorer import score_timeline

        # 배열로 읽으면 10:00 → 15:56 → 11:15 로 역행이지만,
        # 실제로 들르는 순서는 10:00 → 11:15 → 15:56 이다.
        payload = self._payload([0, 2, 1], ["10:00", "15:56", "11:15"])

        result = score_timeline(payload, "10:00", "20:00")

        assert result["applicable"] is True
        assert result["ok"] is True, result["problems"]

    def test_방문_순서로도_역행이면_잡는다(self):
        from eval.scorer import score_timeline

        payload = self._payload([0, 1, 2], ["10:00", "15:56", "11:15"])

        result = score_timeline(payload, "10:00", "20:00")

        assert result["ok"] is False
        assert any("역행" in p for p in result["problems"])

    def test_방문_순서가_없으면_배열_순서를_쓴다(self):
        # 옛 결과에는 순서가 없을 수 있다. 그때도 재기는 해야 한다.
        from eval.scorer import score_timeline

        payload = self._payload([], ["10:00", "11:15", "12:00"])
        payload.pop("visit_order")

        result = score_timeline(payload, "10:00", "20:00")

        assert result["ok"] is True

    def test_활동_시간대_밖은_여전히_잡는다(self):
        from eval.scorer import score_timeline

        payload = self._payload([0, 1], ["10:00", "23:30"])

        result = score_timeline(payload, "10:00", "20:00")

        assert result["ok"] is False
        assert any("종료" in p for p in result["problems"])
