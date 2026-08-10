"""외부 시스템 호출 클라이언트.

hub HTTP, Redis Streams, Gemini LLM 3종을 캡슐화한다.

  - `HubClient`         : hub 의 `/v1/weather`·`/v1/places`·`/v1/reviews`·
                          `/v1/rules/*` 를 호출해 날씨·후보·리뷰·룰 결과를 가져온다.
  - `StreamsPublisher`  : Redis Streams 에 잡 완료/실패 페이로드를 XADD 발행한다.
  - `GeminiClient`      : Gemini 모델로 평문/구조화 응답을 생성한다.

세 클라이언트 모두 `app/main.py` lifespan 에서 생성되고
`app/agent_dependencies.py` 전역 슬롯에 주입된다.

타입 변수:
  T — BaseModel 의 서브타입. `GeminiClient.generate_structured` 가
      `response_schema` 로 받은 구체 Pydantic 모델을 그대로 반환하도록 한다.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Type, TypeVar

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """구조화 응답의 JSON 파싱/스키마 검증 실패.

    ValueError 하위로 두어 기존 `except (ValidationError, ValueError)`
    분기와 호환된다. 교정 재시도(`app/llm/structured_call.py`)가 모델에게
    실패 원인을 피드백할 수 있도록 원문과 원인 예외를 보존한다.

    raw_text: 모델이 돌려준 검증 실패 원문(빈 문자열일 수 있음).
    cause: 원인 예외(pydantic ValidationError 등).
    truncated: 출력 토큰 상한에 걸려 응답이 잘렸는지. 잘린 경우 같은
        프롬프트로 다시 물어도 같은 길이에서 또 잘리므로, 교정 재시도가
        예산과 입력 토큰만 확정적으로 낭비한다. 호출측이 이 값을 보고
        재시도를 건너뛴다.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        cause: Exception,
        truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.cause = cause
        self.truncated = truncated


def _is_truncated(resp) -> bool:
    """응답이 출력 토큰 상한에 걸려 잘렸는지 판정한다.

    종료 사유는 SDK 버전에 따라 enum 이기도 하고 문자열이기도 해서,
    타입을 가정하지 않고 이름만 뽑아 대조한다. 응답 구조가 예상과 달라
    꺼내지 못하면 잘리지 않은 것으로 본다 — 확실하지 않을 때는 기존
    동작(교정 재시도)을 유지하는 쪽이 안전하다.
    """
    try:
        candidates = resp.candidates or []
        reason = candidates[0].finish_reason if candidates else None
    except Exception:  # noqa: BLE001 — 판정 실패가 호출을 죽이면 안 된다
        return False
    if reason is None:
        return False
    name = getattr(reason, "name", None) or str(reason)
    return "MAX_TOKENS" in name


class HubClient:
    """hub `/v1/weather`·`/v1/places`·`/v1/reviews`·`/v1/rules/*` 호출 클라이언트.

    내부에 `httpx.AsyncClient` 1개를 유지하며 lifespan 동안 재사용한다.
    HTTP 커넥션 풀과 keep-alive 이점을 누리도록 호출마다 새로 만들지 않는다.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        """HubClient 초기화.

        base_url: hub 게이트웨이의 base URL. trailing slash 는 rstrip 으로 제거.
        token: 내부 서비스 인증 토큰. None 또는 빈 문자열이면 헤더를 붙이지 않고,
               truthy 한 값이면 `X-Internal-Token` 헤더로 매 요청에 첨부된다.
        timeout: HTTP 호출 timeout(초). 본 값은 `httpx.AsyncClient` 의 기본
                 timeout 으로 전달돼 모든 요청에 적용된다.

        호출처: `app/main.py` lifespan.
        """
        headers: dict[str, str] = {}
        if token:
            headers["X-Internal-Token"] = token
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout, headers=headers
        )

    async def fetch_weather(
        self,
        province: str,
        city: str,
        date_start: date,
        date_end: date,
    ) -> dict:
        """hub `/v1/weather` 를 호출해 날씨 페이로드를 받아 dict 로 반환.

        인자(전부 GET query 로 전달):
          province: 광역 행정구역 문자열.
          city: 시/군/구 문자열.
          date_start, date_end: `isoformat()` 으로 직렬화돼 query 에 들어간다.

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        호출처: `agent_nodes.fetch_weather` 노드(예외 분기 포함).
        """
        params = {
            "province": province,
            "city": city,
            "date_start": date_start.isoformat(),
            "date_end": date_end.isoformat(),
        }
        r = await self._client.get(
            f"{self._base}/v1/weather", params=params
        )
        r.raise_for_status()
        return r.json()

    async def search_places(
        self,
        province: str,
        city: str,
        *,
        mobility: str | None = None,
        keyword: str | None = None,
        size: int = 15,
    ) -> dict:
        """hub `/v1/places` 를 호출해 실측 장소 후보를 받아 dict 로 반환.

        인자(전부 GET query 로 전달):
          province / city: 행정구역. 후보 조회의 기준 지역.
          mobility: 이동수단. 코스 후보를 걷기/자전거로 거르는 데 쓰인다.
          keyword: 검색어. 없으면 hub 가 행정구역명을 검색어로 쓴다.
          size: 출처별 최대 후보 수.

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        반환: hub 응답 본문 dict. places/count/sources 키를 가진다.
        호출처: `agent_nodes.search_places` 노드.
        """
        params: dict = {"province": province, "city": city, "size": size}
        if mobility:
            params["mobility"] = mobility
        if keyword:
            params["keyword"] = keyword
        r = await self._client.get(
            f"{self._base}/v1/places", params=params
        )
        r.raise_for_status()
        return r.json()

    async def fetch_reviews(
        self,
        query: str,
        *,
        display: int = 3,
    ) -> dict:
        """hub `/v1/reviews` 를 호출해 블로그 리뷰 스니펫을 받아 dict 로 반환.

        인자(전부 GET query 로 전달):
          query: 검색어(장소명 등).
          display: 최대 리뷰 수. 정렬은 `sort=sim`(정확도순) 고정.

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        반환: hub 응답 본문 dict. query/reviews/count 키를 가진다.
        호출처: `agent_nodes._fetch_reviews_for` (리뷰 보강, best-effort).
        """
        params = {"query": query, "display": display, "sort": "sim"}
        r = await self._client.get(
            f"{self._base}/v1/reviews", params=params
        )
        r.raise_for_status()
        return r.json()

    async def filter_mobility_radius(
        self,
        origin: dict,
        mobility: str,
        candidates: list[dict],
    ) -> dict:
        """hub `POST /v1/rules/filter/mobility-radius` 로 반경 필터 결과를 받는다.

        인자(전부 JSON body 로 전달):
          origin: 후보 중심 좌표 `{"lat", "lng"}`.
          mobility: 이동수단 문자열(walk/bicycle/car/transit).
          candidates: 필터 대상 후보 dict 리스트(원본 그대로 전달).

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        반환: hub 응답 본문 dict. filtered/radius_m/dropped 키를 가진다.
        호출처: `agent_nodes.rules_filter` 노드(예외는 저하로 흡수).
        """
        body = {
            "origin": origin,
            "mobility": mobility,
            "candidates": candidates,
        }
        r = await self._client.post(
            f"{self._base}/v1/rules/filter/mobility-radius", json=body
        )
        r.raise_for_status()
        return r.json()

    async def score_indoor_bonus(
        self,
        pois: list[dict],
        day_pop_max: int,
    ) -> dict:
        """hub `POST /v1/rules/score/indoor-bonus` 로 실내 보너스 점수를 받는다.

        인자(전부 JSON body 로 전달):
          pois: 채점 대상 `{"content_id", "indoor_flag", "base_score"}` 리스트.
          day_pop_max: 일정 구간의 일별 최대 강수확률(%). 실내 보너스 가중.

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        반환: hub 응답 본문 dict. scored(`{content_id, score}` 리스트) 키.
        호출처: `agent_nodes.score_and_rank` 노드(예외는 저하로 흡수).
        """
        body = {"pois": pois, "day_pop_max": day_pop_max}
        r = await self._client.post(
            f"{self._base}/v1/rules/score/indoor-bonus", json=body
        )
        r.raise_for_status()
        return r.json()

    async def estimate_dwell(self, places: list[dict]) -> dict:
        """hub `POST /v1/rules/estimate/dwell` 로 장소별 체류시간을 받는다.

        인자(JSON body 로 전달):
          places: `{"content_id", "category_group_code", "course_minutes"}`
                  리스트. 뒤 두 키는 없어도 된다.

        실패 시:
          - 4xx/5xx 응답이면 `raise_for_status()` 가 `httpx.HTTPStatusError`.
          - 네트워크/타임아웃 류는 `httpx.HTTPError` 의 하위 예외.

        반환: hub 응답 본문 dict. estimates(`{content_id, stay_minutes,
        source}` 리스트) 키.
        호출처: `agent_nodes.build_timeline` 노드(예외는 저하로 흡수).
        """
        body = {"places": places}
        r = await self._client.post(
            f"{self._base}/v1/rules/estimate/dwell", json=body
        )
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        """내부 `httpx.AsyncClient` 를 닫는다.

        호출처: `app/main.py` lifespan 종료 finally 블록.
        """
        await self._client.aclose()


class StreamsPublisher:
    """Redis Streams XADD 발행자.

    `maxlen` 이 0 보다 크면 매 XADD 마다 approximate trim 을 적용해
    stream 메모리 무한 누적을 방지한다.
    """

    def __init__(
        self,
        redis_url: str,
        db: int,
        stream: str,
        maxlen: int = 0,
    ) -> None:
        """StreamsPublisher 초기화.

        redis_url: redis 접속 URL.
        db: 발행에 사용할 논리 DB 번호.
        stream: XADD 의 대상 stream 이름.
        maxlen: 0 이면 trim 비활성, 양수면 본 값 근처로 stream 길이를
                approximate trim 한다.

        내부적으로 `decode_responses=True` 로 클라이언트를 만들어
        XADD 결과(message id) 가 문자열로 돌아오게 한다.

        호출처: `app/main.py` lifespan.
        """
        self._client = aioredis.Redis.from_url(
            redis_url, db=db, decode_responses=True
        )
        self._stream = stream
        self._maxlen = maxlen

    async def publish(
        self, job_id: str, status: str, payload_json: str
    ) -> str:
        """잡 1건의 결과를 stream 에 발행하고 메시지 id 를 돌려준다.

        인자:
          job_id: 잡 식별자. 메시지의 "job_id" 필드.
          status: "done" 또는 "failed". 메시지의 "status" 필드.
          payload_json: `JobDonePayload.model_dump_json(by_alias=True)` 결과.

        동작:
          - `maxlen > 0` 이면 `maxlen=...`, `approximate=True` 인자를
            xadd 에 전달해 길이를 본 값 근처로 유지한다.
          - 성공 시 부여된 message id 를 그대로 반환.

        호출처:
          - `agent_nodes.publish_done` (성공 경로).
          - `app/main.py._publish_failure` (실패 경로).
        """
        xadd_kwargs: dict = {}
        if self._maxlen > 0:
            xadd_kwargs["maxlen"] = self._maxlen
            xadd_kwargs["approximate"] = True
        message_id = await self._client.xadd(
            self._stream,
            {
                "job_id": job_id,
                "status": status,
                "payload": payload_json,
            },
            **xadd_kwargs,
        )
        logger.info(
            "stream xadd id=%s job_id=%s status=%s",
            message_id, job_id, status,
        )
        return message_id

    async def publish_status(self, job_id: str, stage: str) -> str:
        """노드 진행 이벤트 1건을 stream 에 발행한다.

        인자:
          job_id: 잡 식별자.
          stage: 진입한 노드 이름(parse_input, fetch_weather, ...).

        메시지 필드: {job_id, stage, status:"in_progress", ts(UTC ISO8601)}.
        본 메서드는 `agent:jobs:status` 용 인스턴스에서 호출하는 것을
        전제로 하며, maxlen 정책은 생성자 인자를 그대로 따른다.

        호출처: `agent_nodes._emit_stage` (모든 예외는 호출측이 삼킨다 —
        진행 이벤트 실패가 잡을 죽여선 안 된다).
        """
        xadd_kwargs: dict = {}
        if self._maxlen > 0:
            xadd_kwargs["maxlen"] = self._maxlen
            xadd_kwargs["approximate"] = True
        return await self._client.xadd(
            self._stream,
            {
                "job_id": job_id,
                "stage": stage,
                "status": "in_progress",
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            **xadd_kwargs,
        )

    async def aclose(self) -> None:
        """내부 redis 클라이언트를 닫는다.

        호출처: `app/main.py` lifespan 종료 finally 블록.
        """
        await self._client.aclose()


class GeminiClient:
    """Gemini 모델 호출 클라이언트.

    `generate_text`는 평문 응답, `generate_structured`는 Pydantic 스키마
    기반 구조화 JSON 응답을 반환한다. 모든 호출은 `asyncio.wait_for` 로
    감싸 외부 timeout 을 강제한다.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        thinking_budget: int = 0,
        retry_attempts: int = 1,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 8.0,
        retry_status_codes: str = "503,500",
        http_timeout_seconds: float = 0.0,
        limiter=None,
    ) -> None:
        """GeminiClient 초기화.

        api_key: Gemini API 키 평문 문자열(`SecretStr.get_secret_value()` 결과).
        model: 호출에 사용할 모델 이름.
        timeout: 1회 호출 timeout(초). 본 값으로 모든 generate 호출이
                 `asyncio.wait_for` 에 감싸진다.
        temperature: 생성 온도. 구조화 출력의 결정성을 위해 낮게 유지.
        max_output_tokens: 응답 토큰 상한. Gemini 2.5 는 thinking 토큰이
                           본 상한을 함께 소비하므로 과소 설정 시 JSON 이
                           절단되어 검증 실패가 빈발한다. 여유 있게 둔다.
        thinking_budget: Gemini 2.5 thinking 예산. 0=끔(구조화 추출 기본),
                         양수=허용, 음수=ThinkingConfig 미설정(모델 기본).
        retry_attempts: SDK 전송 재시도 총 시도 수(초기 호출 포함). 2 이상
                        이면 http_status_codes 오류에 지수 백오프 재시도가
                        적용된다. 1 이하이면 재시도를 구성하지 않는다.
        retry_initial_delay / retry_max_delay: 지수 백오프 초기/상한(초).
        retry_status_codes: 재시도 대상 HTTP 코드(쉼표 구분 문자열).
                        "503,500" 처럼 명시한다. **429 제외**(RateLimiter
                        이중 소진 방지). 빈 문자열/파싱 공집합이면 재시도
                        비활성 — SDK 는 빈 리스트를 기본코드(429 포함)로
                        되돌리므로 반드시 None 으로 변환해 넘긴다.
        http_timeout_seconds: per-attempt HTTP 타임아웃(초). HttpOptions.
                        timeout(ms) 로 전달. 0 이하이면 미설정.
        limiter: `acquire()` 코루틴을 가진 rate limiter
                 (`app/llm/rate_limit.GeminiRateLimiter`). None 이면
                 rate-limit 없이 호출한다(단위 테스트 등).

        지연 import: `from google import genai` 를 메서드 호출 직전이 아닌
        생성자에서 1회 import 해 client 인스턴스를 구성한다. 재시도/타임아웃
        설정이 모두 비활성이면 http_options 를 넘기지 않아 기존 경로와
        완전히 동일하게 동작한다.

        호출처: `app/main.py` lifespan.
        """
        from google import genai
        from google.genai import types as genai_types

        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget
        self._limiter = limiter

        # 재시도 대상 코드 파싱. 공집합이면 None(빈 리스트를 SDK 에 넘기면
        # 기본코드(429 포함)로 되돌아가는 함정 회피).
        codes = [
            int(c) for c in retry_status_codes.split(",") if c.strip().isdigit()
        ]
        retry_options = None
        if retry_attempts and retry_attempts > 1 and codes:
            retry_options = genai_types.HttpRetryOptions(
                attempts=retry_attempts,
                initial_delay=retry_initial_delay,
                max_delay=retry_max_delay,
                http_status_codes=codes,
            )
        timeout_ms = (
            int(http_timeout_seconds * 1000)
            if http_timeout_seconds and http_timeout_seconds > 0
            else None
        )
        # 둘 다 없으면 http_options 자체를 생략해 기존 경로를 보존한다.
        if retry_options is not None or timeout_ms is not None:
            http_options = genai_types.HttpOptions(
                timeout=timeout_ms, retry_options=retry_options
            )
            self._client = genai.Client(
                api_key=api_key, http_options=http_options
            )
        else:
            self._client = genai.Client(api_key=api_key)

    async def generate_text(self, prompt: str) -> str:
        """평문 텍스트 응답을 받아 문자열로 돌려준다.

        prompt: 입력 프롬프트 문자열.
        반환: 모델 응답의 `text` 속성. None 이면 빈 문자열("").
        타임아웃: `self._timeout` 을 넘기면 `asyncio.TimeoutError`.

        현재 코드에서는 직접 호출되지 않지만, 본 클라이언트가 비구조화
        텍스트 응답을 지원해야 할 때를 위한 진입점이다.
        """
        resp = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
            ),
            timeout=self._timeout,
        )
        return resp.text or ""

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        *,
        system_instruction: str | None = None,
    ) -> T:
        """구조화(JSON) 응답을 Pydantic 모델로 검증해서 돌려준다.

        prompt: 입력 프롬프트(데이터 태그로 격리된 user 콘텐츠).
        response_schema: 응답 스키마로 사용할 Pydantic 모델 클래스(예:
                         `InventedPlaces`, `ReasonEnvelope`).
        system_instruction: 불변 규칙(역할·인젝션 방어·스키마 준수)을
                            담는 시스템 지시. 사용자 유래 문자열을 절대
                            섞지 않는다(프롬프트 인젝션 방어의 1층).

        동작:
          - limiter 가 있으면 호출 직전 `await limiter.acquire()` 로
            token-bucket/일일 카운터를 소비한다.
          - GenerateContentConfig 에 `response_mime_type="application/json"`,
            `response_schema=response_schema`, `system_instruction`,
            `temperature`, `max_output_tokens` 를 설정해 JSON 출력을 강제.
          - 응답 text(없으면 빈 문자열) 를
            `response_schema.model_validate_json(body)` 로 파싱·검증해 반환.
            SDK 의 `resp.parsed` 는 실패 시 None 이 될 수 있어, 원문 재검증을
            단일 최종 방어선으로 유지한다.

        실패:
          - 타임아웃: `asyncio.TimeoutError`.
          - rate-limit: limiter 가 던지는 `GeminiQuotaError`.
          - JSON 파싱/스키마 검증 실패: `StructuredOutputError`
            (ValueError 하위, 원문·원인 보존 — 교정 재시도용).

        호출처: `app/llm/structured_call.call_structured` (예산·교정 재시도
        계층). 노드가 직접 호출하지 않는다.
        """
        from google.genai import types as genai_types
        from pydantic import ValidationError

        if self._limiter is not None:
            await self._limiter.acquire()

        # Gemini 2.5 thinking 토큰은 max_output_tokens 를 함께 소비한다.
        # 구조화 JSON 추출에는 thinking 이 불필요하고, 켜두면 출력 공간을
        # 잠식해 JSON 절단(EOF ValidationError)을 유발한다(E2E 실측).
        # budget 음수면 미설정(모델 기본)으로 둔다.
        config_kwargs = dict(
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        if self._thinking_budget >= 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )

        resp = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            ),
            timeout=self._timeout,
        )
        body = resp.text or ""
        try:
            return response_schema.model_validate_json(body)
        except (ValidationError, ValueError) as e:
            raise StructuredOutputError(
                f"structured output validation failed: {type(e).__name__}",
                raw_text=body,
                cause=e,
                truncated=_is_truncated(resp),
            ) from e

    async def aclose(self) -> None:
        """클라이언트 수명 종료 훅 — 인터페이스 일관성용 no-op.

        HubClient/StreamsPublisher 와 동일한 `aclose()` 시그니처를 맞춰
        호출측이 모든 클라이언트를 균일하게 정리할 수 있도록 한다.
        google-genai `Client` 는 별도 명시적 종료가 필요 없어 no-op 이다.
        """
        return None
