import asyncio

from app.agent_dependencies import get_gemini_client
from app.agent_settings import get_settings


async def _main() -> None:
    settings = get_settings()
    client = get_gemini_client()
    response = await client.generate_text("ping")
    print(f"OK model={settings.GEMINI_MODEL} response_chars={len(response)}")


if __name__ == "__main__":
    asyncio.run(_main())
