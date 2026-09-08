"""세어 둔 성향 통계와 그것을 랭킹에 얹는 규칙.

여기서 지키는 것은 셋이다.

  - 자료가 모자란 칸은 아무 것도 하지 않는다. 한두 번의 우연을 취향으로
    굳히면 통계가 아니라 그 사람 한 명을 되풀이하는 것이 된다.
  - 가점을 합하지 않는다. 합하면 상한이 무너져 "날씨가 취향을 이긴다" 는
    규약이 깨진다.
  - 통계를 못 읽어도 추천은 그대로 돈다. 곁들이는 값이 본래 일을 막으면 안 된다.
"""
from __future__ import annotations

import json

import pytest

from app.nodes.agent_nodes import (
    _AFFINITY_MAX,
    _segment_gain,
)
from eval.segment_stats import build, segment_key


def _row(age, gender, candidates, eligible=True):
    return {
        "l1_eligible": eligible,
        "user_segment": {"age_band": age, "gender": gender},
        "candidates": candidates,
    }


def _cand(cid, saved=False):
    return {"content_id": cid, "saved": saved}


class TestBuild:
    def test_문턱을_넘은_곳만_남는다(self):
        # 두 번 보인 곳은 버리고 세 번 보인 곳만 남긴다.
        rows = [_row("20s", "m", [_cand("a", True), _cand("b", True)]),
                _row("20s", "m", [_cand("a", True), _cand("b")]),
                _row("20s", "m", [_cand("a")])]

        stats = build(rows, min_support=3)

        places = stats["segments"]["20s|m"]["places"]
        assert "a" in places
        assert "b" not in places
        assert stats["dropped_below_support"] == 1

    def test_저장_비율을_센다(self):
        rows = [_row("20s", "m", [_cand("a", True)]) for _ in range(3)]
        rows.append(_row("20s", "m", [_cand("a")]))

        stats = build(rows, min_support=3)

        entry = stats["segments"]["20s|m"]["places"]["a"]
        assert entry["shown"] == 4
        assert entry["saved"] == 3
        assert entry["rate"] == 0.75

    def test_학습에_못쓰는_세션은_세지_않는다(self):
        # 직접 고른 동선이나 이을 정답이 없는 것을 섞으면 비율이 뜻을 잃는다.
        rows = [_row("20s", "m", [_cand("a", True)], eligible=False)
                for _ in range(5)]

        stats = build(rows, min_support=1)

        assert stats["sessions_used"] == 0
        assert stats["segments"] == {}

    def test_모르는_칸은_모른다고_적는다(self):
        # 빼 버리면 읽는 쪽이 "없음" 과 "안 물어봄" 을 구분하지 못한다.
        assert segment_key({}) == "unknown|unknown"
        assert segment_key({"age_band": "30s"}) == "30s|unknown"

    def test_자료가_없으면_빈_통계다(self):
        stats = build([], min_support=3)
        assert stats["segments"] == {}
        assert stats["sessions_used"] == 0


class TestGain:
    @pytest.mark.parametrize("rate", [None, True, False, [], {}, "invalid",
                                      float("nan"), float("inf"), -float("inf"),
                                      "NaN", "Infinity", -0.01, 1.01, 10 ** 400])
    def test_invalid_rate_adds_no_score(self, rate):
        assert _segment_gain({"content_id": "a"}, {"places": {"a": {"rate": rate}}}) == 0.0

    @pytest.mark.parametrize("rate", [0, 0.25, 1, "0.75"])
    def test_valid_rate_keeps_existing_score(self, rate):
        assert _segment_gain({"content_id": "a"}, {"places": {"a": {"rate": rate}}}) == _AFFINITY_MAX * float(rate)

    @pytest.mark.parametrize("segment", [[], "bad", 1, {"places": []},
                                          {"places": {"a": []}}, {"places": {"a": "bad"}}])
    def test_invalid_nested_shape_adds_no_score(self, segment):
        assert _segment_gain({"content_id": "a"}, segment) == 0.0

    def test_통계에_있는_곳만_가점을_받는다(self):
        segment = {"places": {"a": {"shown": 5, "saved": 5, "rate": 1.0}}}

        assert _segment_gain({"content_id": "a"}, segment) == _AFFINITY_MAX
        assert _segment_gain({"content_id": "b"}, segment) == 0.0

    def test_가점이_상한을_넘지_않는다(self):
        # 이 상한이 실내 가점(0.15)보다 작아야 날씨가 취향을 이긴다.
        segment = {"places": {"a": {"shown": 9, "saved": 9, "rate": 1.0}}}

        gain = _segment_gain({"content_id": "a"}, segment)

        assert gain <= _AFFINITY_MAX
        assert _AFFINITY_MAX < 0.15

    def test_통계가_없으면_가점이_없다(self):
        assert _segment_gain({"content_id": "a"}, None) == 0.0
        assert _segment_gain({"content_id": "a"}, {}) == 0.0

    def test_식별자가_없는_후보는_건너뛴다(self):
        segment = {"places": {"a": {"rate": 1.0}}}
        assert _segment_gain({}, segment) == 0.0


class TestCombination:
    def test_두_가점을_합하지_않는다(self):
        # 합하면 상한이 무너진다. 큰 것 하나만 쓴다.
        from app.nodes.agent_nodes import _affinity

        cand = {"content_id": "a", "category_group_code": "CE7", "category": "카페"}
        segment = {"places": {"a": {"shown": 5, "saved": 5, "rate": 1.0}}}

        theme_gain = _affinity(cand, ["cafe"])
        seg_gain = _segment_gain(cand, segment)
        combined = max(theme_gain, seg_gain)

        assert theme_gain > 0
        assert seg_gain > 0
        assert combined <= _AFFINITY_MAX
        assert theme_gain + seg_gain > _AFFINITY_MAX  # 합했다면 넘었을 것이다


class TestLoading:
    @pytest.fixture(autouse=True)
    def isolated_cache(self, monkeypatch):
        from app.nodes import agent_nodes
        monkeypatch.setattr(agent_nodes, "_SEGMENT_STATS", None)
        monkeypatch.setattr(agent_nodes, "_SEGMENT_STATS_LOADED", False)

    @pytest.mark.parametrize("stats", [None, [], [1], "bad", 1,
        {"segments": []}, {"segments": None}, {"segments": {"unknown|unknown": []}},
        {"segments": {"unknown|unknown": {"places": []}}},
        {"segments": {"unknown|unknown": {"places": {"a": []}}}}])
    def test_malformed_artifact_is_cached_as_empty(self, tmp_path, monkeypatch, stats):
        from types import SimpleNamespace
        from app.nodes import agent_nodes
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(stats))
        monkeypatch.setattr(agent_nodes, "get_settings", lambda: SimpleNamespace(SEGMENT_STATS_PATH=str(path)))
        assert agent_nodes._segment_stats() == {}
        assert agent_nodes._segment_for(None, None) is None
        assert agent_nodes._segment_stats() == {}

    def test_valid_artifact_is_unchanged_and_loaded_once(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from app.nodes import agent_nodes
        stats = build([_row("unknown", "unknown", [_cand("a", True)])] * 3, min_support=3)
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(stats))
        monkeypatch.setattr(agent_nodes, "get_settings", lambda: SimpleNamespace(SEGMENT_STATS_PATH=str(path)))
        assert agent_nodes._segment_stats() == stats
        assert _segment_gain({"content_id": "a"}, agent_nodes._segment_for(None, None)) == _AFFINITY_MAX
        path.write_text("[]")
        assert agent_nodes._segment_stats() == stats

    @pytest.mark.parametrize("raw", [b"{broken", b"\xff", b"[" * 2000 + b"0" + b"]" * 2000])
    def test_unreadable_or_excessively_nested_json_is_ignored(self, tmp_path, monkeypatch, raw):
        from types import SimpleNamespace
        from app.nodes import agent_nodes
        path = tmp_path / "stats.json"
        path.write_bytes(raw)
        monkeypatch.setattr(agent_nodes, "get_settings", lambda: SimpleNamespace(SEGMENT_STATS_PATH=str(path)))
        assert agent_nodes._segment_stats() == {}
        assert agent_nodes._segment_for(None, None) is None

    @pytest.mark.parametrize("stats", [[], {"segments": {"unknown|unknown": {
        "places": {"a": {"rate": float("nan")}, "b": {"rate": 100}}}}}])
    def test_bad_artifact_does_not_break_or_reorder_recommendation(self, tmp_path, monkeypatch, stats):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from app import agent_dependencies
        from app.nodes import agent_nodes

        path = tmp_path / "stats.json"
        path.write_text(json.dumps(stats))
        monkeypatch.setattr(agent_nodes, "get_settings", lambda: SimpleNamespace(
            SEGMENT_STATS_PATH=str(path), RULES_ENABLED=True, PERSONALIZATION_ENABLED=True))
        monkeypatch.setattr(agent_nodes, "_emit_stage", AsyncMock())

        async def score(pois, day_pop_max):
            assert [p["base_score"] for p in pois] == [1.0, 0.99]
            return {"scored": [{"content_id": p["content_id"], "score": p["base_score"]} for p in pois]}

        monkeypatch.setattr(agent_dependencies, "get_hub_client", lambda: SimpleNamespace(score_indoor_bonus=score))
        state = {"request": SimpleNamespace(theme=[]), "grounded": True,
                 "candidates": [{"content_id": "a"}, {"content_id": "b"}]}
        result = asyncio.run(agent_nodes.score_and_rank(state))
        assert not result.get("error")
        assert [c["content_id"] for c in result["candidates"]] == ["a", "b"]
