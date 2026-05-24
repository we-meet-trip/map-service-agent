from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from google import genai as _genai_module  # noqa: F401


class HubClient:
    """장소·날씨·경로·룰 데이터를 hub로부터 조회하는 HTTP 클라이언트."""
    pass


class GeminiClient:
    """Gemini 모델 호출을 캡슐화하는 클라이언트.

    프롬프트 구성·응답 파싱·토큰 사용량 제어(rate limit)를 담당한다.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        client: "_genai_module.Client | None" = None,
    ) -> None:
        from google import genai

        self._model = model
        self._client = (
            client
            if client is not None
            else genai.Client(api_key=api_key.get_secret_value())
        )

    async def generate_text(self, prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""
