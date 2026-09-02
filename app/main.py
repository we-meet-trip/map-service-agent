"""agent-service FastAPI 엔트리포인트.

본 모듈은 다음을 담당한다.

  1) `lifespan`: 앱 기동·종료 훅. `get_settings()` 로 환경설정을 읽어
     HubClient / StreamsPublisher / GeminiClient 를 만들고
     `agent_dependencies` 모듈의 setter 로 전역 슬롯에 주입한다.
     이후 `build_graph()` 로 LangGraph 인스턴스를 만들어
     `app.state.graph` 에 저장하고, 백그라운드 작업 집합
     `app.state.bg_tasks` 와 `app.state.job_timeout_seconds` 도 초기화한다.
     종료 시점에는 진행 중 백그라운드 작업이 `SHUTDOWN_GRACE_SECONDS`
     안에 끝나길 기다리고, 못 끝나면 cancel 한 뒤 외부 클라이언트를
     `aclose()` 하고 의존성 슬롯을 `reset_all()` 한다.

  2) `/health` GET: 헬스 체크.
  3) `/v1/recommend` POST: 추천 잡 비동기 등록. 새 `job_id` 를 만들고
     `_run_job` 을 `asyncio.create_task` 로 띄워 `app.state.bg_tasks` 에
     추적하며, 클라이언트에게는 즉시 202 `AgentJobAccepted` 응답.

  4) `_run_job`: LangGraph 의 `ainvoke` 를 `asyncio.wait_for` 로
     `JOB_TIMEOUT_SECONDS` 안에 강제 종료하고, 타임아웃/취소/예외
     모든 경우에 `_publish_failure` 로 실패 페이로드를 발행한다.
  5) `_publish_failure`: `agent_dependencies.get_streams_publisher()` 로
     StreamsPublisher 를 얻어 status="failed" `JobDonePayload` 를 게시한다.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.agent_dependencies import (
    get_streams_publisher,
    reset_all,
    set_gemini_client,
    set_hub_client,
    set_status_publisher,
    set_streams_publisher,
)
from app.crypto import location_seal
from app.agent_settings import get_settings
from app.clients.agent_clients import (
    GeminiClient,
    HubClient,
    StreamsPublisher,
)
from app.graph.agent_graph import build_graph
from app.graph.summary_graph import SUMMARY_GRAPH
from app.llm.rate_limit import GeminiQuotaError, GeminiRateLimiter
from app.llm.structured_call import LLMBudgetExceeded
from app.schemas.agent_schemas import (
    AgentJobAccepted,
    AgentRequest,
    JobDonePayload,
    PlaceSummaryResult,
    ReviewsSummaryBatchRequest,
    ReviewsSummaryBatchResponse,
    ReviewsSummaryRequest,
    ReviewsSummaryResponse,
    SummaryPlaceRequest,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """루트 로거에 stdout 핸들러를 붙인다(핸들러가 없을 때만).

    uvicorn 의 기본 로깅 설정은 `uvicorn*` 로거만 구성하고 루트 로거는
    건드리지 않는다. 그래서 이 함수 없이는 `app.*` 로거로 남긴 기록이
    출력 대상을 못 찾아 전량 유실된다.

    이미 핸들러가 있으면(uvicorn `--log-config`, 테스트 하니스, 상위
    호스트가 설정한 경우) 그 설정을 존중하고 아무것도 하지 않는다 —
    핸들러를 덧붙이면 같은 로그가 두 줄씩 출력된다.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan — 시작 시 외부 클라이언트 및 그래프 준비, 종료 시 정리.

    동작:
      - `_configure_logging(LOG_LEVEL)` 로 루트 로거를 가장 먼저 세운다.
      - `get_settings()` 로 환경설정을 읽고, `GEMINI_API_KEY` 가 비어 있으면
        `RuntimeError("GEMINI_API_KEY is required")` 로 즉시 실패.
      - `INTERNAL_SERVICE_TOKEN` 이 설정되어 있으면 그 SecretStr 평문 값을
        HubClient 의 `X-Internal-Token` 헤더로 전달한다(없으면 None).
      - `AUTH_ENFORCED=true` 인데 토큰이 비어 있으면
        `RuntimeError("AUTH_ENFORCED=true requires INTERNAL_SERVICE_TOKEN")`
        로 부팅 중단(fail-fast). false(기본)면 토큰 미설정을 허용한다.
      - HubClient, StreamsPublisher, GeminiClient 를 만들어 모듈 전역 슬롯
        (`set_hub_client` / `set_streams_publisher` / `set_gemini_client`) 에
        주입한다. LangGraph 노드들은 함수 내부 import 와 getter 로 본 슬롯을
        참조한다(`app/agent_dependencies.py`).
      - `build_graph()` 로 컴파일된 LangGraph 를 `app.state.graph` 에 저장.
      - `app.state.bg_tasks` 는 `_run_job` 태스크들을 추적하는 집합.
      - `app.state.job_timeout_seconds` 는 잡 1건당 허용 시간(`_run_job`에서 사용).

    종료(`finally`):
      - 진행 중 백그라운드 태스크들을 `SHUTDOWN_GRACE_SECONDS` 동안 기다리고,
        시간 초과 시 미완료 태스크에 `cancel()` 을 보낸 뒤 다시 회수.
      - HubClient/StreamsPublisher/GeminiClient 의 `aclose()` 호출(균일 정리).
      - `reset_all()` 로 모든 전역 슬롯을 None 으로 되돌린다.
    """
    settings = get_settings()
    # 로깅은 다른 어떤 초기화보다 먼저 — 아래 fail-fast 게이트가 걸릴 때도
    # 원인이 stdout 에 남아야 한다.
    _configure_logging(settings.LOG_LEVEL)
    if not settings.GEMINI_API_KEY.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is required")

    # INTERNAL_SERVICE_TOKEN 은 SecretStr | None — 평문을 1회만 꺼내
    # 아래 부팅 게이트와 HubClient 헤더에 공유한다.
    token = (
        settings.INTERNAL_SERVICE_TOKEN.get_secret_value()
        if settings.INTERNAL_SERVICE_TOKEN
        else None
    )
    # AUTH_ENFORCED=true 배포에서는 토큰 부재 시 부팅을 중단한다
    # (fail-fast — GEMINI_API_KEY 부재와 동일 패턴). false(기본)면
    # `_verify_internal_token` 이 토큰 미설정 시 경고 로그와 함께 검증을
    # 생략하는 기존 동작을 유지한다.
    if settings.AUTH_ENFORCED and not token:
        raise RuntimeError(
            "AUTH_ENFORCED=true requires INTERNAL_SERVICE_TOKEN"
        )
    hub_client = HubClient(
        base_url=settings.HUB_BASE_URL,
        token=token,
        timeout=settings.HUB_TIMEOUT_SECONDS,
    )
    streams = StreamsPublisher(
        redis_url=settings.REDIS_URL,
        db=settings.REDIS_DB_STREAMS,
        stream=settings.STREAM_NAME,
        maxlen=settings.STREAM_MAXLEN,
    )
    # 진행 이벤트(agent:jobs:status) 전용 발행자. done 스트림과
    # 동일 DB 에 둔다. 소비자는 후속 주차(user-BFF progress) — 발행만 한다.
    status_streams = StreamsPublisher(
        redis_url=settings.REDIS_URL,
        db=settings.REDIS_DB_STREAMS,
        stream=settings.STATUS_STREAM_NAME,
        maxlen=settings.STATUS_STREAM_MAXLEN,
    )
    # Gemini 호출 한도 집행기. GeminiClient 가 generate
    # 직전 acquire() 로 소비하며, RPM/RPD 초과 시 GeminiQuotaError 로
    # 잡이 실패 페이로드(gemini_rpm_exceeded / gemini_rpd_exceeded)를 낸다.
    limiter = GeminiRateLimiter(
        redis_url=settings.REDIS_URL,
        db=settings.REDIS_DB_RATELIMIT,
        capacity=settings.GEMINI_RPM_LIMIT,
        refill_per_min=settings.GEMINI_RPM_LIMIT,
        daily_cap=settings.GEMINI_RPD_CAP,
    )
    gemini = GeminiClient(
        api_key=settings.GEMINI_API_KEY.get_secret_value(),
        model=settings.GEMINI_MODEL,
        timeout=settings.GEMINI_TIMEOUT_SECONDS,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        thinking_budget=settings.GEMINI_THINKING_BUDGET,
        retry_attempts=settings.GEMINI_RETRY_ATTEMPTS,
        retry_initial_delay=settings.GEMINI_RETRY_INITIAL_DELAY,
        retry_max_delay=settings.GEMINI_RETRY_MAX_DELAY,
        retry_status_codes=settings.GEMINI_RETRY_STATUS_CODES,
        http_timeout_seconds=settings.GEMINI_HTTP_TIMEOUT_SECONDS,
        limiter=limiter,
    )
    set_hub_client(hub_client)
    set_streams_publisher(streams)
    set_status_publisher(status_streams)
    set_gemini_client(gemini)

    # LangGraph Postgres 체크포인터 (schema=langgraph).
    # CHECKPOINT_ENABLED=true 인데 연결/스키마 준비가 실패하면 부팅을
    # 중단한다(fail-fast — GEMINI_API_KEY 부재와 동일 패턴, 사용자 결정
    # 2026-07-05). 관련 모듈은 지연 import 한다 — 로컬 단위 테스트
    # 환경에는 psycopg/checkpoint-postgres 가 없을 수 있고, 테스트는
    # checkpointer=None 경로만 사용한다.
    checkpointer = None
    checkpoint_pool = None
    if settings.CHECKPOINT_ENABLED:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        schema = settings.LANGGRAPH_SCHEMA
        if not schema.isidentifier():
            raise RuntimeError(
                f"invalid LANGGRAPH_SCHEMA identifier: {schema!r}"
            )
        conninfo = (
            f"postgresql://{settings.POSTGRES_USER}:"
            f"{settings.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
            f"/{settings.POSTGRES_DB}"
        )
        try:
            # search_path 를 langgraph 스키마로 고정해 setup() 이 public
            # 에 테이블을 만들지 않도록 한다(AsyncPostgresSaver 는 스키마
            # 인자를 직접 받지 않는다 — conninfo options 로 지정).
            # 빌려줄 때마다 연결이 살아 있는지 먼저 확인한다.
            #
            # 확인을 걸지 않으면 풀은 이미 끊긴 연결도 그대로 내준다. 끊긴 줄은
            # 그 연결로 첫 질의를 던져 실패한 뒤에야 알게 되고, 그때는 이미 잡
            # 하나가 죽은 뒤다. 풀은 실패한 연결을 버리고 새로 하나 채우므로,
            # 끊긴 연결 수만큼 잡이 연달아 죽고 나서야 저절로 멎는다.
            #
            # 데이터베이스가 한 번 재기동하면 풀에 담긴 연결이 전부 이 상태가
            # 된다. 확인을 걸어 두면 빌려주는 쪽에서 조용히 새 연결로 바꿔
            # 주므로 잡은 한 건도 잃지 않는다. 대신 대여마다 왕복이 한 번 더
            # 드는데, 이 풀은 추천 요청당 몇 번만 쓰여서 그 값이 싸다.
            checkpoint_pool = AsyncConnectionPool(
                conninfo,
                open=False,
                check=AsyncConnectionPool.check_connection,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "options": f"-c search_path={schema}",
                },
            )
            await checkpoint_pool.open()
            async with checkpoint_pool.connection() as conn:
                await conn.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {schema}"
                )
            checkpointer = AsyncPostgresSaver(checkpoint_pool)
            await checkpointer.setup()
            logger.info(
                "agent: langgraph checkpointer ready schema=%s", schema
            )
        except Exception as e:
            # fail-fast 경로에서도 이미 생성한 클라이언트들을 대칭적으로
            # 정리한다 — finally 블록은 yield 이전 raise 로 도달 불가.
            if checkpoint_pool is not None:
                await checkpoint_pool.close()
            await hub_client.aclose()
            await streams.aclose()
            await status_streams.aclose()
            await gemini.aclose()
            await limiter.aclose()
            reset_all()
            raise RuntimeError(
                "langgraph checkpointer init failed (set "
                "CHECKPOINT_ENABLED=false to boot without it)"
            ) from e

    app.state.checkpoint_pool = checkpoint_pool
    app.state.graph = build_graph(checkpointer=checkpointer)
    app.state.bg_tasks: set[asyncio.Task] = set()
    app.state.job_timeout_seconds = settings.JOB_TIMEOUT_SECONDS
    logger.info(
        "agent: lifespan initialized model=%s job_timeout=%.1fs",
        settings.GEMINI_MODEL, settings.JOB_TIMEOUT_SECONDS,
    )
    try:
        yield
    finally:
        pending = list(app.state.bg_tasks)
        if pending:
            logger.info(
                "agent: waiting for %d background task(s) to finish",
                len(pending),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=settings.SHUTDOWN_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "agent: shutdown grace exceeded, cancelling %d task(s)",
                    sum(1 for t in pending if not t.done()),
                )
                for t in pending:
                    if not t.done():
                        t.cancel()
                # cancel 후 회수에도 상한을 둔다. CancelledError 를
                # 삼키는 태스크가 있어도 종료가 무한정 매달리지 않도록
                # SHUTDOWN_GRACE_SECONDS 로 한 번 더 bound 한다.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=settings.SHUTDOWN_GRACE_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "agent: cancelled task(s) did not settle within "
                        "grace; abandoning"
                    )
        await hub_client.aclose()
        await streams.aclose()
        await status_streams.aclose()
        await gemini.aclose()
        await limiter.aclose()
        if checkpoint_pool is not None:
            await checkpoint_pool.close()
        reset_all()
        logger.info("agent: lifespan disposed")


app = FastAPI(
    title="map-service-agent",
    version="0.0.1-poc",
    lifespan=lifespan,
)

# Prometheus 계측 → GET /metrics (인증 없음, map-net 내부 Prometheus 스크레이프).
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


def _verify_internal_token(provided: str | None) -> None:
    """B2 인바운드 요청의 `X-Internal-Token` 을 검증한다.

    - `INTERNAL_SERVICE_TOKEN` 이 설정된 경우: 헤더 부재 또는 불일치 시
      401. 비교는 `hmac.compare_digest` 로 상수 시간 수행.
    - 미설정(PoC 로컬 등): 검증을 생략하되 경고 로그를 남긴다 — 잡 1건이
      Gemini 유료 호출을 유발하므로 미인증 노출은 비용 유발 DoS 표면이다.

    호출처: `recommend` 엔드포인트 진입부.
    """
    settings = get_settings()
    expected = (
        settings.INTERNAL_SERVICE_TOKEN.get_secret_value()
        if settings.INTERNAL_SERVICE_TOKEN
        else ""
    )
    if not expected:
        logger.warning(
            "INTERNAL_SERVICE_TOKEN not set - /v1/recommend accepts "
            "unauthenticated requests"
        )
        return
    # bytes 비교: str 비교는 비-ASCII 입력(공격자 제어 헤더)에서
    # compare_digest 가 TypeError 를 던져 401 대신 500 이 된다.
    if provided is None or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid internal token")


@app.get("/health")
async def health() -> dict[str, str]:
    """헬스 체크 엔드포인트.

    반환: ``{"status": "ok", "service": "agent"}``.
    Dockerfile 의 HEALTHCHECK 가 본 엔드포인트의 200 응답으로 컨테이너
    상태를 판정한다. 외부 시스템(hub/redis/gemini) 의 가용성은 검사하지
    않으며, 프로세스가 살아 있고 ASGI 가 응답 가능한지만 본다.
    """
    return {"status": "ok", "service": "agent"}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    """바깥 저장소까지 닿는지 본다.

    위의 /health 는 과정이 살아 있다는 것만 알린다. 저장소가 끊겨도 그 쪽은
    계속 200 을 돌려주므로, 일을 받을 수 없는 상태가 정상으로 보인다. 여기서는
    실제로 한 번씩 물어보고, 하나라도 답하지 않으면 503 으로 알린다.

    이어 붙이기를 꺼 두면 표를 아예 쓰지 않는다. 그때는 없는 의존을 있다고
    적지 않는다. hub 는 묻지 않는다. 옆 서비스가 잠깐 내려간 것 때문에 이 쪽까지
    못 쓰는 것으로 표시되면 장애 하나가 둘로 늘어난다.
    """
    deps: dict[str, str] = {}

    try:
        deps["streams"] = "ok" if await get_streams_publisher().ping() else "down"
    except RuntimeError:
        deps["streams"] = "down"

    pool = getattr(app.state, "checkpoint_pool", None)
    if pool is None:
        deps["checkpoint"] = "disabled"
    else:
        try:
            async with pool.connection() as conn:
                await conn.execute("select 1")
            deps["checkpoint"] = "ok"
        except Exception as e:
            logger.warning("readiness: checkpoint store unreachable err=%s", e)
            deps["checkpoint"] = "down"

    ready = "down" not in deps.values()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "degraded", "service": "agent", "deps": deps},
    )


async def _publish_failure(job_id: str, error: str) -> None:
    """잡 실패 페이로드를 Redis Streams 에 게시.

    인자:
      job_id: 실패한 잡의 식별자.
      error: 실패 사유 텍스트. `JobDonePayload.error` 로 직렬화된다.

    동작:
      `agent_dependencies.get_streams_publisher()` 로 StreamsPublisher 를
      가져와 status="failed" `JobDonePayload` 를 만들고
      `model_dump_json(by_alias=True)` 로 직렬화한 뒤 publish.
      게시 자체가 실패해도 잡을 죽이지 않기 위해 모든 예외를 잡아
      `logger.exception` 으로만 기록한다.

    호출처: `_run_job` 의 타임아웃/취소/예외 경로.
    """
    try:
        publisher = get_streams_publisher()
        payload = JobDonePayload(
            job_id=job_id, status="failed", error=error
        )
        await publisher.publish(
            job_id=job_id,
            status=payload.status,
            payload_json=payload.model_dump_json(by_alias=True),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "agent publish failed for job_id=%s", job_id
        )


async def _run_job(
    graph, job_id: str, req: AgentRequest, timeout_seconds: float
) -> None:
    """백그라운드 잡 실행기 — LangGraph 1회 호출을 시간 제한과 함께 수행.

    인자:
      graph: `build_graph()` 가 반환한 컴파일된 LangGraph 인스턴스.
      job_id: 추천 잡 식별자(`/v1/recommend` 가 생성한 UUID4 문자열).
      req: 요청 본문 `AgentRequest`.
      timeout_seconds: 잡 1건당 허용 시간(`lifespan` 에서 settings.JOB_TIMEOUT_SECONDS).

    동작:
      `graph.ainvoke({"job_id": job_id, "request": req},
      config={"configurable": {"thread_id": job_id}})` 를
      `asyncio.wait_for(...)` 로 감싸 강제 종료한다. thread_id 는
      체크포인터의 스레드 식별자다(체크포인터 없이 컴파일된 그래프에도
      무해하다).

    오류 처리(모두 `_publish_failure` 로 실패 페이로드 발행):
      - `asyncio.TimeoutError`: "job timeout after N seconds".
      - `asyncio.CancelledError`: "job cancelled on shutdown" 게시 후 재-raise.
      - 그 외 `Exception`: `str(e)` 그대로 실패 사유로 기록.

    호출처: `/v1/recommend` 가 `asyncio.create_task(_run_job(...))` 로 호출.
    """
    try:
        await asyncio.wait_for(
            graph.ainvoke(
                {"job_id": job_id, "request": req},
                config={"configurable": {"thread_id": job_id}},
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "agent job timeout job_id=%s after %.1fs",
            job_id, timeout_seconds,
        )
        await _publish_failure(
            job_id,
            f"job timeout after {timeout_seconds:.0f} seconds",
        )
    except asyncio.CancelledError:
        logger.warning("agent job cancelled job_id=%s", job_id)
        await _publish_failure(job_id, "job cancelled on shutdown")
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("agent job failed job_id=%s", job_id)
        await _publish_failure(job_id, str(e))


def _resolve_places(req: AgentRequest) -> AgentRequest:
    """고른 장소를 봉투에서 꺼낸다.

    열쇠가 설정돼 있으면 봉투만 받는다. 평문으로 온 좌표를 그대로 받아 주면
    봉투를 만들 수 있다는 것이 자격이 되는 구조가 무너진다 — 열쇠 없이도
    같은 요청을 만들 수 있게 되기 때문이다.

    열쇠가 없으면 예전처럼 평문을 받는다. 양쪽이 준비되기 전에 넘어가기 위한
    자리이고, 둘 다 켜지면 이 갈래는 지나가지 않는다.
    """
    if not location_seal.enabled():
        return req
    if not req.loc:
        # 좌표가 아예 없는 요청은 그대로 간다. 첫 단계는 고른 장소가 없다.
        if not req.places:
            return req
        # 열쇠가 있는데 평문으로 왔다. 봉투를 만들 줄 아는 쪽만 이 경로를
        # 쓸 수 있어야 하므로 받지 않는다.
        raise HTTPException(status_code=400, detail="invalid location")
    try:
        opened = location_seal.open_seal(req.loc)
    except location_seal.SealError:
        raise HTTPException(status_code=400, detail="invalid location")
    places = opened.get("places")
    if not isinstance(places, list):
        raise HTTPException(status_code=400, detail="invalid location")
    logger.info("location seal opened path=/v1/recommend places=%d", len(places))
    return req.model_copy(update={"places": places, "loc": None})


@app.post(
    "/v1/recommend",
    response_model=AgentJobAccepted,
    status_code=202,
)
async def recommend(
    req: AgentRequest,
    x_internal_token: str | None = Header(
        default=None, alias="X-Internal-Token"
    ),
) -> AgentJobAccepted:
    """추천 잡을 비동기 등록하고 즉시 202 로 응답.

    인자:
      req: 요청 본문. `AgentRequest` 스키마.
      x_internal_token: B2 내부 서비스 인증 헤더. 토큰이
        설정된 배포에서 부재/불일치면 401.

    동작:
      - `_verify_internal_token` 으로 인바운드 인증을 먼저 검증.
      - `app.state.graph` 가 준비되지 않았으면 503 "graph not ready".
      - `uuid.uuid4()` 로 `job_id` 생성.
      - `asyncio.create_task(_run_job(...))` 로 백그라운드 잡 등록.
        태스크 이름은 ``"agent-job-{job_id}"``.
      - `app.state.bg_tasks` 집합에 추가하고, `add_done_callback` 으로
        완료 시 자동 제거되게 한다(메모리 누수 방지).
      - 클라이언트에는 `AgentJobAccepted(job_id=...)` 즉시 반환(status_code=202).
        실제 결과는 Redis Streams `agent:jobs:done` 으로 비동기 게시된다.
    """
    _verify_internal_token(x_internal_token)
    req = _resolve_places(req)
    if not hasattr(app.state, "graph"):
        raise HTTPException(
            status_code=503, detail="graph not ready"
        )
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        _run_job(
            app.state.graph,
            job_id,
            req,
            app.state.job_timeout_seconds,
        ),
        name=f"agent-job-{job_id}",
    )
    bg_tasks: set[asyncio.Task] = app.state.bg_tasks
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)
    return AgentJobAccepted(job_id=job_id)


async def _run_summary(
    places: list[SummaryPlaceRequest],
) -> dict[int, list[str]]:
    """요약 파이프라인을 한 번 돌려 요청 위치별 두 줄을 돌려준다.

    두 요약 엔드포인트가 이 함수를 공유한다. 한도 초과와 그 밖의 실패를
    가르는 기준이 두 곳으로 갈라지면 같은 원인이 경로에 따라 다른 상태로
    나가기 때문이다.

    반환: 요청 배열의 위치 → 두 줄. 근거를 못 구한 장소는 키가 없다.

    실패 처리:
      - 호출 한도에 걸리면 429. 호출 측은 요약 없이 화면을 그린다.
      - 모델 응답이 형식을 벗어나거나 호출이 실패하면 502.
      - 형식은 맞지만 쓸 만한 두 줄이 나오지 않으면 빈 결과. 요약이 없는
        것은 오류가 아니라 근거가 부족한 상태다.
    """
    if not get_settings().SUMMARY_ENABLED:
        logger.info("reviews summary disabled by settings")
        return {}
    state = {
        "places": [
            {
                "name": p.place_name,
                "category": p.category,
                "snippets": [r.description for r in p.reviews],
            }
            for p in places
        ]
    }
    try:
        out = await SUMMARY_GRAPH.ainvoke(state)
    except (LLMBudgetExceeded, GeminiQuotaError) as e:
        logger.info("reviews summary rate limited: %s", type(e).__name__)
        raise HTTPException(
            status_code=429, detail="summary rate limited"
        ) from e
    except Exception as e:  # noqa: BLE001 — 원인은 로그로만 남긴다
        logger.warning("reviews summary failed: %s", type(e).__name__)
        raise HTTPException(
            status_code=502, detail="summary unavailable"
        ) from e
    return out.get("bullets") or {}


@app.post("/v1/reviews/summary", response_model=ReviewsSummaryResponse)
async def summarize_place_reviews(
    req: ReviewsSummaryRequest,
    x_internal_token: str | None = Header(
        default=None, alias="X-Internal-Token"
    ),
) -> ReviewsSummaryResponse:
    """장소 한 곳의 블로그 후기를 두 줄로 요약해 즉시 돌려준다.

    장소를 눌렀을 때 부르는 경로다. 미리 만들어 둔 요약이 없거나 만료됐을
    때 그 자리에서 답한다.

    후기는 호출 측이 이미 받아 둔 것을 그대로 실어 보낸다. 여기서 다시
    조회하면 외부 호출이 두 배가 되고, 화면에 보이는 목록과 요약의 근거가
    어긋날 수 있다.

    인자:
      req: 장소명·분류와 요약 근거가 될 후기 묶음.
      x_internal_token: 내부 서비스 인증 헤더. 토큰이 설정된 배포에서
        부재/불일치면 401.
    """
    _verify_internal_token(x_internal_token)
    bullets = await _run_summary([req])
    return ReviewsSummaryResponse(bullets=bullets.get(0, []))


@app.post(
    "/v1/reviews/summary/batch",
    response_model=ReviewsSummaryBatchResponse,
)
async def summarize_places_reviews(
    req: ReviewsSummaryBatchRequest,
    x_internal_token: str | None = Header(
        default=None, alias="X-Internal-Token"
    ),
) -> ReviewsSummaryBatchResponse:
    """여러 장소의 후기를 한 번에 요약한다.

    일정이 만들어진 직후 그 일정의 장소를 미리 요약해 두는 경로다. 장소마다
    따로 부르면 호출 수가 장소 수만큼 늘어 분당 상한에 걸리므로, 한 번의
    모델 호출에 묶어 보낸다.

    응답은 요청 배열의 **위치**로 짚는다. 한 일정에 같은 이름이 두 번 나올
    수 있어 이름으로는 어느 장소의 요약인지 가릴 수 없다. 근거를 못 구한
    장소는 결과에서 빠진다 — 부분 커버리지는 정상이다.

    인자:
      req: 요약 대상 장소 묶음.
      x_internal_token: 내부 서비스 인증 헤더. 토큰이 설정된 배포에서
        부재/불일치면 401.
    """
    _verify_internal_token(x_internal_token)
    bullets = await _run_summary(req.places)
    return ReviewsSummaryBatchResponse(
        results=[
            PlaceSummaryResult(index=index, bullets=lines)
            for index, lines in sorted(bullets.items())
        ]
    )
