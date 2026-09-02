"""준비 확인(/health/ready) 테스트.

살아 있음(/health)과 준비됨(/health/ready)을 갈라 두는 것이 요점이다.
저장소가 끊겼을 때 준비 확인만 503 이 되고 살아 있음은 계속 200 이어야
한다. 둘이 같아지면 저장소가 끊긴 상태가 정상으로 보이거나, 반대로 잠깐의
저장소 장애가 과정을 다시 띄우는 신호가 된다.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import app.main as main


class _FakePublisher:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    async def ping(self) -> bool:
        return self._alive


class _FakePool:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    @asynccontextmanager
    async def connection(self):
        if not self._alive:
            raise RuntimeError("표에 닿지 못했다")

        class _C:
            async def execute(self, *_a, **_k):
                return None

        yield _C()


def _get(monkeypatch, *, streams: bool, pool):
    monkeypatch.setattr(main, "get_streams_publisher", lambda: _FakePublisher(streams))
    main.app.state.checkpoint_pool = pool
    c = TestClient(main.app)
    # lifespan 을 태우지 않고 라우트만 부른다. 준비 확인 자체가 검사 대상이다.
    return c.get("/health/ready"), c.get("/health")


def test_둘_다_답하면_준비됨(monkeypatch):
    ready, _ = _get(monkeypatch, streams=True, pool=_FakePool(True))
    assert ready.status_code == 200
    assert ready.json()["deps"] == {"streams": "ok", "checkpoint": "ok"}


def test_발행자가_끊기면_준비되지_않음(monkeypatch):
    ready, live = _get(monkeypatch, streams=False, pool=_FakePool(True))
    assert ready.status_code == 503
    assert ready.json()["deps"]["streams"] == "down"
    # 살아 있음은 그대로 200 이어야 한다. 이 둘이 같아지면 구분이 사라진다.
    assert live.status_code == 200


def test_표가_끊기면_준비되지_않음(monkeypatch):
    ready, live = _get(monkeypatch, streams=True, pool=_FakePool(False))
    assert ready.status_code == 503
    assert ready.json()["deps"]["checkpoint"] == "down"
    assert live.status_code == 200


def test_이어붙이기를_꺼_둔_것은_끊긴_것과_다르다(monkeypatch):
    ready, _ = _get(monkeypatch, streams=True, pool=None)
    assert ready.status_code == 200
    assert ready.json()["deps"]["checkpoint"] == "disabled"
