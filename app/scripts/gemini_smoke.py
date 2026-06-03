import asyncio

from app.agent_settings import get_settings
from app.clients.agent_clients import GeminiClient


async def _main() -> None:
    # 단독 실행 스크립트라 lifespan 의 DI 컨테이너(_gemini 슬롯)가 초기화되지
    # 않으므로 get_gemini_client() 대신 GeminiClient 를 직접 만든다
    # (app/main.py lifespan 의 생성 패턴과 동일).
    settings = get_settings()
    if not settings.GEMINI_API_KEY.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is required")
    client = GeminiClient(
        api_key=settings.GEMINI_API_KEY.get_secret_value(),
        model=settings.GEMINI_MODEL,
        timeout=settings.GEMINI_TIMEOUT_SECONDS,
    )
    response = await client.generate_text("ping")
    print(f"OK model={settings.GEMINI_MODEL} response_chars={len(response)}")


if __name__ == "__main__":
    asyncio.run(_main())
