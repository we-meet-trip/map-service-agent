"""학습 신호(training) 계약 테스트.

결과와 함께 "무엇을 보고 무엇을 골랐는지" 를 남기는 부분이다. 나중에 랭킹을
학습시키려면 이 기록이 있어야 하는데, 지금은 후보 목록이 그래프 상태 안에서만
살다 사라진다.

여기서 확인하는 것은 세 가지다.
  - 경로(select / invent / select_then_invent / route)가 실제 실행과 맞는가
  - 후보·점수 키가 없는 경로에서도 터지지 않는가
  - 결과 본문(payload)에 후보가 섞여 들어가지 않는가

마지막 항목이 중요한 이유: payload 는 BFF 가 초안으로 저장해 클라이언트 응답
본문으로 그대로 내보낸다. 후보 수십 개가 거기 들어가면 앱이 매번 그만큼을 더
받는다.
"""
from __future__ import annotations

import json

from app.nodes.agent_nodes import TRAINING_SCHEMA_VERSION, _training_signal
from app.schemas.agent_schemas import Place


def _place(place_id: int, content_id: str | None) -> Place:
    return Place(
        place_id=place_id,
        day=1,
        name=f"장소{place_id}",
        address="주소",
        lat=37.5,
        lng=127.0,
        recommended_visit_time="10:00",
        content_id=content_id,
    )


def _candidates(n: int) -> list[dict]:
    return [
        {
            "content_id": f"c{i}",
            "name": f"후보{i}",
            "category": "CE7",
            "lat": 37.5 + i * 0.001,
            "lng": 127.0 + i * 0.001,
        }
        for i in range(n)
    ]


class TestSelectPath:
    def test_후보와_선택결과가_함께_남는다(self):
        state = {
            "job_id": "j1",
            "selection_path": "select",
            "grounded": True,
            "candidates": _candidates(3),
            "places": [_place(0, "c1")],
            "scores": {"c0": 0.9, "c1": 0.8, "c2": 0.7, "c3": 0.6},
        }
        signal = json.loads(_training_signal(state))

        assert signal["schema_version"] == TRAINING_SCHEMA_VERSION
        assert signal["path"] == "select"
        assert signal["grounded"] is True
        assert signal["candidate_count"] == 3
        assert signal["chosen_content_ids"] == ["c1"]

    def test_후보에_노출_순위가_붙는다(self):
        # 순위가 없으면 "고르지 않은 후보" 를 그대로 부정 표본으로 쓰게 되고,
        # 위에 있어서 뽑히기 쉬웠던 효과가 선호로 둔갑한다.
        state = {
            "job_id": "j1",
            "selection_path": "select",
            "candidates": _candidates(3),
            "places": [],
        }
        signal = json.loads(_training_signal(state))
        assert [c["rank"] for c in signal["candidates"]] == [0, 1, 2]

    def test_점수는_자를_기준_이전임을_이름으로_밝힌다(self):
        # 점수는 후보를 자르기 전 집합으로 만들어져 candidates 와 개수가 다르다.
        state = {
            "job_id": "j1",
            "selection_path": "select",
            "candidates": _candidates(2),
            "places": [],
            "scores": {"c0": 0.9, "c1": 0.8, "c2": 0.7},
        }
        signal = json.loads(_training_signal(state))
        assert "scores_pre_cap" in signal
        assert "scores" not in signal
        assert len(signal["scores_pre_cap"]) == 3
        assert signal["candidate_count"] == 2


class TestOtherPaths:
    def test_창작_경로는_후보가_비어도_남는다(self):
        state = {"job_id": "j1", "selection_path": "invent", "places": [_place(0, None)]}
        signal = json.loads(_training_signal(state))
        assert signal["path"] == "invent"
        assert signal["candidate_count"] == 0
        assert signal["candidates"] == []

    def test_선정에서_창작으로_넘어간_것이_구분된다(self):
        # degraded_reason 은 뒤 단계가 덮으므로 그것으로는 구분할 수 없다.
        state = {
            "job_id": "j1",
            "selection_path": "select_then_invent",
            "candidates": _candidates(2),
            "places": [],
            "degraded_reason": "llm_budget_exhausted",
        }
        signal = json.loads(_training_signal(state))
        assert signal["path"] == "select_then_invent"

    def test_사용자가_직접_고른_경로도_구분된다(self):
        state = {"job_id": "j1", "selection_path": "route", "places": [_place(0, "c9")]}
        signal = json.loads(_training_signal(state))
        assert signal["path"] == "route"
        assert signal["candidate_count"] == 0


class TestMissingKeys:
    def test_후보_점수_장소_키가_없어도_터지지_않는다(self):
        # route 나 저하 경로에서는 이 키들이 아예 없다. 여기서 KeyError 가 나면
        # 결과 발행 자체가 막힌다.
        signal = json.loads(_training_signal({"job_id": "j1", "selection_path": "route"}))
        assert signal["candidate_count"] == 0
        assert signal["chosen_content_ids"] == []
        assert "scores_pre_cap" not in signal

    def test_경로를_모르면_남기지_않는다(self):
        # 경로가 없으면 후보 목록의 의미가 정해지지 않아 기록해도 쓸 수 없다.
        assert _training_signal({"job_id": "j1", "candidates": _candidates(3)}) is None

    def test_실패한_잡은_남기지_않는다(self):
        state = {"job_id": "j1", "selection_path": "select", "error": "boom"}
        assert _training_signal(state) is None


class TestTokenUsage:
    def test_토큰_수가_합계와_함께_남는다(self):
        # 호출 횟수만으로는 필요한 컴퓨트를 알 수 없다. 후보를 수십 개 싣는
        # 요청과 이유만 쓰는 요청이 크게 다르기 때문이다.
        state = {
            "job_id": "j1",
            "selection_path": "select",
            "places": [],
            "llm_usage": [
                {"prompt": 4000, "output": 1200, "total": 5200},
                {"prompt": 900, "output": 2100, "total": 3000},
            ],
        }
        signal = json.loads(_training_signal(state))
        assert signal["llm_tokens"] == {
            "calls": 2, "prompt": 4900, "output": 3300, "total": 8200,
        }
        assert len(signal["llm_usage"]) == 2

    def test_토큰_수가_없으면_생략한다(self):
        # 옛 판이나 계량을 못 받은 응답에서는 이 값이 없다.
        state = {"job_id": "j1", "selection_path": "route", "places": []}
        signal = json.loads(_training_signal(state))
        assert "llm_tokens" not in signal

    def test_일부_값이_비어도_더하기가_깨지지_않는다(self):
        state = {
            "job_id": "j1", "selection_path": "select", "places": [],
            "llm_usage": [{"prompt": None, "output": 100, "total": None}],
        }
        signal = json.loads(_training_signal(state))
        assert signal["llm_tokens"]["output"] == 100
        assert signal["llm_tokens"]["prompt"] == 0


class TestPayloadSeparation:
    def test_후보가_결과_본문에_섞이지_않는다(self):
        import asyncio

        from app import agent_dependencies as deps
        from app.nodes.agent_nodes import publish_done

        captured: dict = {}

        class _Pub:
            async def publish(self, job_id, status, payload_json, training_json=None):
                captured["payload"] = payload_json
                captured["training"] = training_json
                return "1-1"

        deps.set_streams_publisher(_Pub())
        try:
            state = {
                "job_id": "j1",
                "selection_path": "select",
                "grounded": True,
                "candidates": _candidates(40),
                "places": [_place(0, "c1")],
                "visit_order": [0],
                "legs": [],
            }
            asyncio.run(publish_done(state))
        finally:
            deps.reset_all()

        assert "후보0" not in captured["payload"]
        assert "candidates" not in captured["payload"]
        # 같은 정보는 별도 필드에만 실린다.
        assert "후보0" in captured["training"]
