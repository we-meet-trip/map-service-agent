import asyncio
import json

import httpx
import pytest

from app import agent_dependencies as deps
from app.errors import exception_failure
from app.main import _run_job
from app.nodes.agent_nodes import recommend_places, search_places
from app.schemas.agent_schemas import PlaceSelection, PlacesSelection
from tests.test_graph_routing import _invoke, _FakeHub as GraphHub, _SeqGemini
from tests.test_grounding import _request, _candidate, _FakeHub, _FakeGemini


@pytest.mark.parametrize("selections", [[], [PlaceSelection(index=99, day=1, recommended_visit_time="오전")],
                                        [PlaceSelection(index=0, day=99, recommended_visit_time="오전")]])
def test_invalid_selection_is_not_a_search_shortage(selections):
    deps.set_gemini_client(_FakeGemini(PlacesSelection(selections=selections)))
    try:
        result = asyncio.run(recommend_places({"job_id": "j", "request": _request(), "candidates": [_candidate(0)]}))
    finally:
        deps.reset_all()
    assert result["code"] == "selection_invalid"
    assert result["retryable"] is False
    assert result["llm_calls_used"] == 1


def test_search_error_differs_from_empty_success():
    request = _request()
    deps.set_hub_client(_FakeHub(exc=httpx.ConnectError("private-url")))
    try:
        result = asyncio.run(search_places({"job_id": "j", "request": request}))
        result = asyncio.run(recommend_places(result))
    finally:
        deps.reset_all()
    assert result["code"] == "upstream_unavailable"
    assert result["retryable"] is True
    assert result["request"] == request


def test_compiled_graph_keeps_failure_fields():
    _, publisher = _invoke(_request(), GraphHub([]), _SeqGemini([]))
    payload = publisher.payloads[0]
    assert payload["status"] == "failed"
    assert payload["code"] == "no_matching_places"
    assert payload["retryable"] is False


@pytest.mark.parametrize("exc,code,retryable", [(RuntimeError("private-token"), "generation_failed", False),
                                               (asyncio.TimeoutError(), "generation_timeout", True),
                                               (asyncio.CancelledError(), "upstream_unavailable", True)])
def test_worker_terminal_contract(exc, code, retryable):
    class Graph:
        async def ainvoke(self, *args, **kwargs):
            raise exc
    class Publisher:
        payload = None
        async def publish(self, **kwargs):
            self.payload = json.loads(kwargs["payload_json"])
    publisher = Publisher()
    deps.set_streams_publisher(publisher)
    try:
        try:
            asyncio.run(_run_job(Graph(), "j", _request(), 1))
        except asyncio.CancelledError:
            pass
    finally:
        deps.reset_all()
    assert publisher.payload["code"] == code
    assert publisher.payload["retryable"] is retryable
    assert "private-token" not in json.dumps(publisher.payload)


def test_provider_denial_does_not_promise_retry():
    response = httpx.Response(503, request=httpx.Request("GET", "https://internal.invalid"),
                             json={"detail": {"code": "upstream_unavailable", "retryable": False}})
    result = exception_failure(httpx.HTTPStatusError("private-body", request=response.request, response=response))
    assert result == {"error": "upstream_unavailable", "code": "upstream_unavailable", "retryable": False}
