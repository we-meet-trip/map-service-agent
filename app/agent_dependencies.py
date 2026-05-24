from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.agent_settings import get_settings

if TYPE_CHECKING:
    from app.clients.agent_clients import GeminiClient


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    from app.clients.agent_clients import GeminiClient

    settings = get_settings()
    return GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )
