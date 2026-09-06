"""Keep the monitoring contract intact when upgrading FastAPI/Starlette."""
from fastapi.testclient import TestClient

from app.main import app


def test_http_metrics_survive_framework_security_upgrade():
    # No lifespan: this exercises ASGI instrumentation without DB/model/API calls.
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/unknown-metrics-probe").status_code == 404
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total{" in response.text
    assert "http_request_duration_seconds_bucket{" in response.text
    assert 'handler="/health"' in response.text
    assert "/unknown-metrics-probe" not in response.text
