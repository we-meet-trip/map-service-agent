"""외부 시스템 호출 클라이언트.

hub HTTP, Redis Streams, Gemini LLM 3종을 캡슐화한다.

  - `HubClient`         : hub 의 `/v1/weather` 를 호출해 날씨 데이터를 가져온다.
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
from datetime import date
from typing import Type, TypeVar

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HubClient:
    """hub /v1/weather 호출 클라이언트.

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
    ) -> None:
        """GeminiClient 초기화.

        api_key: Gemini API 키 평문 문자열(`SecretStr.get_secret_value()` 결과).
        model: 호출에 사용할 모델 이름.
        timeout: 1회 호출 timeout(초). 본 값으로 모든 generate 호출이
                 `asyncio.wait_for` 에 감싸진다.

        지연 import: `from google import genai` 를 메서드 호출 직전이 아닌
        생성자에서 1회 import 해 client 인스턴스를 구성한다.

        호출처: `app/main.py` lifespan.
        """
        from google import genai

        self._model = model
        self._timeout = timeout
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
        self, prompt: str, response_schema: Type[T]
    ) -> T:
        """구조화(JSON) 응답을 Pydantic 모델로 검증해서 돌려준다.

        prompt: 입력 프롬프트.
        response_schema: 응답 스키마로 사용할 Pydantic 모델 클래스(예:
                         `PlacesEnvelope`, `RouteEnvelope`).

        동작:
          - GenerateContentConfig 에 `response_mime_type="application/json"`,
            `response_schema=response_schema` 를 설정해 모델에게 JSON 출력을
            지시한다.
          - 응답 text(없으면 빈 문자열) 를
            `response_schema.model_validate_json(body)` 로 파싱·검증해 반환.

        실패:
          - 타임아웃: `asyncio.TimeoutError`.
          - JSON 파싱 또는 스키마 검증 실패: pydantic `ValidationError`
            또는 `ValueError`. `recommend_places` 노드는 본 두 예외를
            잡아 1회 재시도한다.

        호출처: `agent_nodes.recommend_places`, `agent_nodes.recommend_route`.
        """
        from google.genai import types as genai_types

        resp = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            ),
            timeout=self._timeout,
        )
        body = resp.text or ""
        return response_schema.model_validate_json(body)

    async def aclose(self) -> None:
        """클라이언트 수명 종료 훅 — 인터페이스 일관성용 no-op.

        HubClient/StreamsPublisher 와 동일한 `aclose()` 시그니처를 맞춰
        호출측이 모든 클라이언트를 균일하게 정리할 수 있도록 한다.
        google-genai `Client` 는 별도 명시적 종료가 필요 없어 no-op 이다.
        """
        return None
