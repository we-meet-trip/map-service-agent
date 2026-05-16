from fastapi import FastAPI

app = FastAPI(title="map-service-agent", version="0.0.1-poc")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent"}
