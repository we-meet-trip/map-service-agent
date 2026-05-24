import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent_dependencies import get_gemini_client
from app.agent_settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.GEMINI_API_KEY.get_secret_value():
        logger.warning(
            "GEMINI_API_KEY is empty - gemini_smoke and llm calls will fail"
        )
    yield
    get_settings.cache_clear()
    get_gemini_client.cache_clear()


app = FastAPI(
    title="map-service-agent",
    version="0.0.1-poc",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent"}
