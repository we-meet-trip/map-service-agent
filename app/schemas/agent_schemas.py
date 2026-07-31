"""추천 파이프라인 요청·응답·내부 페이로드 Pydantic 스키마.

본 모듈은 agent-service 가 다루는 모든 직렬화 가능한 데이터 모델을 정의한다.

  - 입력 모델:   `AgentRequest`, `DateRange`, `Mobility`
  - 응답 모델:   `AgentJobAccepted` (`/v1/recommend` 의 202 응답)
  - 도메인 모델: `Place`, `Leg`
  - LLM 응답 래퍼: `PlacesEnvelope`, `RouteEnvelope`
  - Streams 발행 페이로드: `JobDonePayload`

사용처:
  - `app/main.py` : `AgentRequest`, `AgentJobAccepted`, `JobDonePayload`.
  - `app/nodes/agent_nodes.py` : 위 전체 + `Place`, `Leg`, 두 Envelope.
"""
from __future__ import annotations

import re
from datetime import date, time
from enum import Enum
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LLM 생성 텍스트 필드에서 거부하는 문자: 태그 문자(<, >)와 제어문자.
# 프롬프트 인젝션 산출물이 페이로드를 타고 하류(클라이언트 렌더링,
# 후속 프롬프트)로 흘러가는 것을 차단하는 최종 방어선이다.
# 검증 실패는 ValidationError → StructuredOutputError → 교정 재시도로
# 자연 연결된다. 주의: google-genai 가 Field 제약(max_length 등)을
# response_schema 로 전달하지 못할 수 있으므로, 본 클라이언트측 pydantic
# 검증이 실질적 최종 방어선이다.
_FORBIDDEN_TEXT_RE = re.compile(
    "[<>＜＞"                              # 꺾쇠(ASCII+전각)
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"          # C0/C1 제어문자
    "​-‏  ‪-‮⁦-⁩"  # zw/bidi
    "⁠-⁤﻿"                          # word joiner·BOM
    "\U000e0000-\U000e007f"                        # Unicode Tags 블록
    "]"
)


def _reject_tags_and_control(value):
    """텍스트 필드에 태그/제어문자가 있으면 ValueError (None 은 통과)."""
    if isinstance(value, str) and _FORBIDDEN_TEXT_RE.search(value):
        raise ValueError("text contains forbidden tag or control characters")
    return value


# bullets 1줄의 길이 상한. 블로그 요약은 카드 UI 두 줄에 들어가야 하므로
# reason(200자)보다 짧게 잡는다.
BULLET_MAX_LEN = 80


def _reject_tags_and_control_list(value):
    """문자열 리스트의 모든 원소에 `_reject_tags_and_control` 을 적용한다.

    bullets 처럼 리스트 필드는 원소 단위 검증이 필요하다(Field 제약만으로는
    항목 내부 문자를 막지 못한다). 길이 상한도 함께 본다 — 리뷰 원문이
    그대로 새어 나오는 것(요약이 아닌 발췌)을 막는 실질 장치다.
    """
    if value is None:
        return value
    for item in value:
        if not isinstance(item, str):
            raise ValueError("bullet must be a string")
        stripped = item.strip()
        if not stripped:
            raise ValueError("bullet must not be blank")
        if len(item) > BULLET_MAX_LEN:
            raise ValueError(f"bullet exceeds {BULLET_MAX_LEN} characters")
        _reject_tags_and_control(item)
    return value


class Mobility(str, Enum):
    """이동 수단 enum — 추천 동선이 가정하는 이동 모드.

    str 을 mixin 으로 두어 `.value` 비교와 JSON 직렬화 시 문자열로 다루기 쉽다.

    walk: 도보.
    bicycle: 자전거.
    scooter: 전동 킥보드. SoT(def §3.6)의 kickboard 와 같은 개념이며,
        hub 룰이 두 철자를 모두 받아 반경 7km 를 적용한다. 예전에는 이 값이
        없어 BFF 가 킥보드를 bicycle 로 치환했고, 그 결과 킥보드 전용 반경이
        한 번도 적용되지 않았다.
    car: 자가용/택시 등 자동차.
    transit: 대중교통(버스/지하철 등).

    사용처:
      - `AgentRequest.mobility` (선택값).
      - `Leg.mode`.
      - `agent_nodes._build_places_prompt` / `_build_route_prompt`
        가 `mobility.value` 로 LLM 프롬프트에 직렬화.
    """
    walk = "walk"
    bicycle = "bicycle"
    scooter = "scooter"
    car = "car"
    transit = "transit"


class DateRange(BaseModel):
    """여행 일정의 시작/종료 날짜 및 시각 묶음.

    date_start: 여행 시작 날짜(datetime.date).
    date_end: 여행 종료 날짜(datetime.date).
    time_start: 하루 일정의 시작 시각(datetime.time).
    time_end: 하루 일정의 종료 시각(datetime.time).

    추가 제약 검증:
      - `agent_nodes.parse_input` 가 `date_start <= date_end`,
        구간이 `_MAX_RANGE_DAYS(14일)` 이내, `time_start < time_end` 를 확인.
    """
    date_start: date
    date_end: date
    time_start: time
    time_end: time


class SelectedPlace(BaseModel):
    """사용자가 직접 고른 방문지 1건 — stage="route" 요청의 입력.

    사용자가 지도에서 장소를 골라 오면 후보 탐색·선정 단계를 건너뛰고 이
    목록으로 바로 동선을 짠다. 그래서 좌표는 필수다(동선을 그릴 수 없다).

    name: 장소명. 1~80자.
    address: 주소. 표시용이라 동선 계산에 쓰이지 않는다. 키를 생략해도,
             값이 null 로 와도 빈 문자열로 접는다 — 호출 측이 주소 없는
             지점(지도에서 찍은 좌표 등)을 그대로 보낼 수 있어야 한다.
    lat/lng: 좌표. 국내 범위 밖이면 검증 실패.
    day: 이 장소를 방문할 여행 일차(1부터). 호출 측이 일차별로 장소를
         골랐을 때 그 묶음을 그대로 전달하는 통로다. 키를 생략해도, 값이
         null 로 와도 1일차로 접는다 — 일차 구분 없이 고른 호출도 그대로
         받아야 한다.
    content_id: 실측 출처 식별자. 있으면 리뷰 조회에 그대로 쓰고, 없으면
                이름으로 조회한다.
    """
    name: str = Field(min_length=1, max_length=80)
    address: str = Field(default="", max_length=200)
    lat: float = Field(ge=33.0, le=43.0)
    lng: float = Field(ge=124.0, le=132.0)
    day: int = Field(default=1, ge=1)
    content_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("day", mode="before")
    @classmethod
    def _first_day_when_null(cls, value):
        """일차 키가 없거나 null 이면 1일차로 접는다."""
        return 1 if value is None else value

    @field_validator("address", mode="before")
    @classmethod
    def _blank_when_null(cls, value):
        """JSON `"address": null` 을 빈 문자열로 접는다."""
        return "" if value is None else value

    _no_tags = field_validator("name", "address", "content_id")(
        _reject_tags_and_control
    )


class AgentRequest(BaseModel):
    """추천 요청 본문 — `/v1/recommend` POST 의 body.

    date: 여행 일정(`DateRange`). 필수.
    budget: 예산(원 단위 정수). 없을 수 있어 Optional[int], 기본 None.
            프롬프트의 `budget_krw` 키로 LLM 에 전달된다.
    theme: 여행 테마 키워드 리스트. Optional[List[str]], 기본 None.
    mobility: 선호 이동 수단(`Mobility`). 없을 수 있음.
    province: 광역 행정구역(예: "서울특별시"). 1~20자.
    city: 시/군/구(예: "강남구"). 1~20자.
    schedule_id: user-BFF 의 일정 식별자(SoT §6.1 B2 body). 추적/영속화
                 연계용 패스스루 — agent 는 user_service 스키마를 읽지
                 않으므로 이 값으로 조회하지 않는다. nullable.
    stage: 추천 단계.
           "init"  — 조건만 받아 장소 탐색부터 하는 초기 추천.
           "mode1" — 일부 장소를 빼고 다시 뽑는 재탐색.
           "route" — 사용자가 places 로 고른 장소들의 동선만 짠다.
           기본 "init" — 기존 호출자 하위호환.
    exclude: Mode 1 재탐색 시 재추천을 금지할 content_id 목록.
             grounded 후보 필터에 사용된다. invent(LLM 생성) 폴백에는
             content_id 가 없어 적용 불가 — 한계는 search_places docstring 참조.
    places: stage="route" 에서 사용자가 고른 방문지 목록. 2~10개.
            다른 stage 에서는 무시된다. 2개 미만이면 이을 구간이 없고,
            10개를 넘으면 동선 프롬프트가 감당하기 어려워 상한을 둔다.

    호출 흐름:
      클라이언트 → `app.main.recommend(req)` → `_run_job` →
      `graph.ainvoke({"job_id": ..., "request": req})` → LangGraph 노드들이
      `state["request"]` 로 접근.
    """
    date: DateRange
    budget: Optional[int] = None
    theme: Optional[List[str]] = None
    mobility: Optional[Mobility] = None
    province: str = Field(min_length=1, max_length=20)
    city: str = Field(min_length=1, max_length=20)
    schedule_id: Optional[str] = Field(default=None, max_length=64)
    stage: Literal["init", "mode1", "route"] = "init"
    exclude: Optional[
        List[Annotated[str, Field(min_length=1, max_length=64)]]
    ] = Field(default=None, max_length=50)
    places: Optional[List[SelectedPlace]] = Field(
        default=None, min_length=2, max_length=10
    )

    @field_validator("stage", mode="before")
    @classmethod
    def _default_stage_when_null(cls, value):
        """JSON `"stage": null`(Jackson null 직렬화)을 "init" 으로 보정."""
        return "init" if value is None else value


class AgentJobAccepted(BaseModel):
    """`/v1/recommend` 의 202 응답 본문.

    job_id: 잡 식별자(`recommend` 라우터가 `uuid.uuid4()` 로 생성).
    status: Literal["in_progress"] 고정값. 본 응답이 "처리 중" 임을 알린다.
    retry_after_seconds: 클라이언트에게 권장하는 재조회 간격(초). 기본 3.

    결과 자체는 Redis Streams 로 비동기 게시되며, 클라이언트는 본
    `job_id` 로 후속 조회 또는 long-poll 을 수행한다.
    """
    job_id: str
    status: Literal["in_progress"] = "in_progress"
    retry_after_seconds: int = 3


class Place(BaseModel):
    """추천 장소 1건.

    model_config:
      populate_by_name=True — 필드명 직접 입력과 alias 입력 모두 허용.

    place_id: 장소 식별자(정수). `recommend_places` 노드가 0..N-1 로 정규화한다.
    day: 여행 일차(1부터 시작). LLM 이 `_build_places_prompt` 지시에 따라
         날짜 범위 안에서 직접 배정한다. 1..num_days 범위를 벗어나면 검증 실패.
    name: 장소명.
    address: 주소 문자열.
    lat: 위도. ge=33.0, le=43.0 — 한국 국내 위도 범위 밖이면 검증 실패.
    lng: 경도. ge=124.0, le=132.0 — 한국 국내 경도 범위 밖이면 검증 실패.
    recommended_visit_time: 추천 방문 시간을 표현하는 자유 텍스트.

    사용처:
      - LLM 응답 검증의 단위 모델(`PlacesEnvelope.places` 원소).
      - `Leg.from_place_id` / `Leg.to_place_id` 가 본 `place_id` 를 참조.
      - `JobDonePayload.places` 의 원소. user(BFF) 가 이 `day` 를 그대로
        `TripStop.day` 로 전달해 client 가 일차별 탭을 그린다.
    """
    model_config = ConfigDict(populate_by_name=True)

    place_id: int
    day: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    address: str = Field(max_length=200)
    lat: float = Field(ge=33.0, le=43.0)
    lng: float = Field(ge=124.0, le=132.0)
    recommended_visit_time: str = Field(max_length=50)

    # 하류(클라이언트 렌더링·후속 프롬프트)로 나가는 모든 텍스트 필드에
    # 적용한다 — invent 경로에선 LLM 이, grounded 경로에선 외부
    # API(TourAPI/Kakao/Naver) 값이 채우는 필드 전부가 대상(간접 인젝션·
    # 표시 스푸핑 차단). grounded 후보가 검증에 걸리는 극단 케이스는
    # Place 생성 실패로 해당 후보만 건너뛴다(호출측 skip 규칙).
    _no_tags = field_validator(
        "name", "address", "recommended_visit_time",
        "content_id", "source", "category", "category_group_code",
        "phone", "place_url", "brd_div", "gpx_url", "route_idx",
    )(_reject_tags_and_control)
    # 실측 출처(점 장소/코스)에서 채워지는 보강 필드. LLM 단독 생성
    # 경로에서는 채워지지 않을 수 있어 모두 선택값이다.
    content_id: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    category_group_code: Optional[str] = None
    phone: Optional[str] = None
    place_url: Optional[str] = None
    crs_dstnc_km: Optional[float] = None
    crs_total_min: Optional[int] = None
    crs_level: Optional[int] = None
    brd_div: Optional[str] = None
    gpx_url: Optional[str] = None
    route_idx: Optional[str] = None
    # 외부 실측 후보에 근거해 만든 장소면 True, LLM 단독 생성이면 False.
    grounded: bool = True
    # llm_reason 노드가 병합하는 장소별 추천 이유(SoT §3 agent 책임).
    # LLM 응답(ReasonEnvelope)에서 검증을 거친 값만 들어온다. 생성
    # 시점에는 없다가 build_payload 에서 model_copy 로 채워진다.
    reason: Optional[str] = Field(default=None, max_length=200)
    # summarize_reviews 노드가 병합하는 장소별 블로그 리뷰 요약(2줄).
    # 네이버 블로그 검색 스니펫을 LLM 이 종합한 결과이며, 원문 스니펫은
    # 페이로드에 싣지 않는다(간접 인젝션 노출면 최소화). reason 과 같은
    # 방식으로 생성 후 build_payload 에서 model_copy 로 채워진다.
    bullets: Optional[List[str]] = Field(default=None, max_length=2)

    _no_tags_reason = field_validator("reason")(_reject_tags_and_control)
    _no_tags_bullets = field_validator("bullets")(
        _reject_tags_and_control_list
    )


class Leg(BaseModel):
    """동선의 한 구간 — 두 장소 사이의 이동 1건.

    model_config:
      populate_by_name=True — 필드명/alias 양쪽 입력 모두 허용.

    from_place_id: 출발 장소의 place_id. alias="from"
                   (Python 예약어 `from` 회피용).
    to_place_id: 도착 장소의 place_id. alias="to".
    mode: 본 구간의 이동 수단(`Mobility`).
    estimated_distance_km: 추정 거리(km). ge=0.
    estimated_duration_min: 추정 소요 시간(분). ge=0.

    검증:
      - `recommend_route` 노드가 `from_place_id` / `to_place_id` 가
        `places` 의 `place_id` 집합 안에 있는지, 그리고 `legs` 의 길이가
        `len(places) - 1` 인지 확인한다.

    직렬화:
      - `_build_route_prompt` 가 `model_dump(by_alias=True)` 로 LLM 에
        전달하므로 키 이름은 `"from"` / `"to"` 형태로 나간다.
      - `JobDonePayload.legs` 의 원소로 stream 페이로드에 포함될 때도
        `model_dump_json(by_alias=True)` 로 같은 키 이름이 유지된다.
    """
    model_config = ConfigDict(populate_by_name=True)

    from_place_id: int = Field(alias="from")
    to_place_id: int = Field(alias="to")
    mode: Mobility
    estimated_distance_km: float = Field(ge=0)
    estimated_duration_min: int = Field(ge=0)


class PlacesEnvelope(BaseModel):
    """`recommend_places` 단계에서 LLM 이 돌려주는 구조화 응답의 루트.

    places: 추천 장소 리스트.

    사용처:
      - 외부 실측 후보가 없을 때(grounding 불가)의 폴백 경로에서
        `GeminiClient.generate_structured(prompt, PlacesEnvelope)` 의
        `response_schema` 로 전달되어 JSON → 모델 검증을 수행한다.
    """
    places: List[Place]


class PlaceSelection(BaseModel):
    """grounded 경로에서 LLM 이 고른 후보 1건.

    index: 후보 목록에서 선택한 항목의 0 기반 인덱스.
    day: 해당 장소를 방문할 여행 일차(1부터). 여행 일수를 넘는 값은
         `_select_places` 가 그 선택만 건너뛴다.
    recommended_visit_time: 해당 장소의 권장 방문 시간 텍스트.
    """
    index: int = Field(ge=0)
    day: int = Field(ge=1)
    recommended_visit_time: str = Field(max_length=50)

    _no_tags = field_validator("recommended_visit_time")(
        _reject_tags_and_control
    )


class PlacesSelection(BaseModel):
    """`recommend_places` 의 grounded 경로에서 LLM 이 돌려주는 선택 결과.

    selections: 실측 후보 중에서 고른 항목들(방문 시간 포함). 장소의
        이름·주소·좌표는 LLM 이 새로 만들지 않고 후보값을 그대로 쓴다.

    사용처:
      - `GeminiClient.generate_structured(prompt, PlacesSelection)` 의
        `response_schema`.
    """
    selections: List[PlaceSelection]


class PlaceReason(BaseModel):
    """`llm_reason` 응답의 장소별 추천 이유 1건.

    place_id: 대상 장소의 place_id. `llm_reason` 노드가 유효 id 집합
              부분집합·무중복을 semantic 검증한다.
    reason: 추천 이유 텍스트(≤200자, 태그/제어문자 금지).
    """
    place_id: int
    reason: str = Field(min_length=1, max_length=200)

    _no_tags = field_validator("reason")(_reject_tags_and_control)


class ReasonEnvelope(BaseModel):
    """`llm_reason` 단계에서 LLM 이 돌려주는 구조화 응답의 루트.

    reasons: 장소별 추천 이유 리스트(모든 place_id 각 1건이 목표).
    clothing: 날씨 기반 옷차림 안내 전역 1건(≤300자).

    사용처:
      - `GeminiClient.generate_structured(prompt, ReasonEnvelope)` 의
        `response_schema` (llm_reason 노드, LLM 3번째 호출).
    """
    reasons: List[PlaceReason]
    clothing: str = Field(min_length=1, max_length=300)

    _no_tags = field_validator("clothing")(_reject_tags_and_control)


class PlaceBullets(BaseModel):
    """`summarize_reviews` 응답의 장소별 블로그 요약 1건.

    place_id: 대상 장소의 place_id. `summarize_reviews` 노드가 유효 id
              집합의 부분집합인지·무중복인지 semantic 검증한다.
    bullets: 요약 줄. 각 줄은 1..BULLET_MAX_LEN 자, 태그/제어문자 금지.
             하류 계약은 "정확히 2줄"이지만 스키마는 1~4줄을 받는다 —
             줄 수를 스키마로 못 박으면 모델이 3줄을 쓴 순간 응답 전체가
             검증에 걸려 모든 장소의 요약이 함께 사라진다. 개수 정규화는
             노드가 맡아(2줄 미만 항목 폐기, 초과분 절단) 일부만 어긋난
             경우에도 나머지 장소는 살린다.
    """
    place_id: int
    bullets: List[str] = Field(min_length=1, max_length=4)

    _no_tags = field_validator("bullets")(_reject_tags_and_control_list)


class BulletsEnvelope(BaseModel):
    """`summarize_reviews` 단계에서 LLM 이 돌려주는 구조화 응답의 루트.

    items: 장소별 요약 리스트. 리뷰 스니펫을 확보한 장소만 대상이므로
           요청한 장소 전부가 오지 않을 수 있다(부분 커버리지 허용).

    사용처:
      - `GeminiClient.generate_structured(prompt, BulletsEnvelope)` 의
        `response_schema` (summarize_reviews 노드, LLM 4번째 호출).
    """
    items: List[PlaceBullets]


class RouteEnvelope(BaseModel):
    """`recommend_route` 단계에서 LLM 이 돌려주는 구조화 응답의 루트.

    visit_order: 방문 순서. `places` 의 `place_id` 들로 이루어진 순열.
                 `recommend_route` 노드가 "정확히 같은 집합" 인지 검증한다.
    legs: 방문 순서에 따른 구간 리스트. 길이는 `len(places) - 1`.

    사용처:
      - `GeminiClient.generate_structured(prompt, RouteEnvelope)` 의
        `response_schema`.
    """
    visit_order: List[int]
    legs: List[Leg]


class JobDonePayload(BaseModel):
    """잡 완료 페이로드 — Redis Streams `agent:jobs:done` 의 메시지 본문.

    job_id: 어떤 잡의 결과인지 식별.
    status: Literal["done", "failed"]. 둘 중 하나만 허용.
    places: 성공 시 추천 장소 리스트(장소별 reason·bullets 포함 가능).
            실패 시 None.
    visit_order: 성공 시 방문 순서. 실패 시 None.
    legs: 성공 시 동선 구간 리스트. 실패 시 None.
    clothing: 성공 시 날씨 기반 옷차림 안내(llm_reason 산출). llm_reason
              이 degrade 로 생략되면 None 일 수 있다.
    error: 실패 시 사유 텍스트. 성공 시 None.

    직렬화:
      - `StreamsPublisher.publish` 호출 직전에
        `model_dump_json(by_alias=True)` 로 JSON 문자열화되어
        stream 메시지의 `payload` 필드로 들어간다.

    호출처:
      - 성공 경로: `agent_nodes.publish_done` 노드.
      - 실패 경로: `app/main.py` 의 `_publish_failure`.
    """
    job_id: str
    status: Literal["done", "failed"]
    places: Optional[List[Place]] = None
    visit_order: Optional[List[int]] = None
    legs: Optional[List[Leg]] = None
    clothing: Optional[str] = None
    error: Optional[str] = None
