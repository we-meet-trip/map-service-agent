"""Fixed recommendation failure codes; provider text never enters result payloads."""
import asyncio

import httpx

from app.clients.agent_clients import StructuredOutputError
from app.llm.rate_limit import GeminiQuotaError
from app.llm.structured_call import LLMBudgetExceeded


def failure(code: str = "generation_failed", retryable: bool = False) -> dict:
    return {"error": code, "code": code, "retryable": retryable}


def exception_failure(exc: BaseException) -> dict:
    if isinstance(exc, GeminiQuotaError):
        return failure("quota_exceeded")
    if isinstance(exc, (StructuredOutputError, LLMBudgetExceeded)):
        return failure("selection_invalid")
    if isinstance(exc, asyncio.CancelledError):
        return failure("upstream_unavailable", True)
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return failure("upstream_unavailable", True)
    if isinstance(exc, httpx.HTTPStatusError):
        # Hub explicitly distinguishes provider outage from a successful empty search.
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, dict) and detail.get("code") == "upstream_unavailable":
            return failure("upstream_unavailable", detail.get("retryable") is True)
        if exc.response.status_code == 429:
            return failure("quota_exceeded")
        return failure("upstream_unavailable", exc.response.status_code >= 500)
    if isinstance(exc, httpx.RequestError):
        return failure("upstream_unavailable", True)
    return failure()
