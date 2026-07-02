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

from datetime import date, time
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Mobility(str, Enum):
    """이동 수단 enum — 추천 동선이 가정하는 이동 모드.

    str 을 mixin 으로 두어 `.value` 비교와 JSON 직렬화 시 문자열로 다루기 쉽다.

    walk: 도보.
    bicycle: 자전거.
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


class AgentRequest(BaseModel):
    """추천 요청 본문 — `/v1/recommend` POST 의 body.

    date: 여행 일정(`DateRange`). 필수.
    budget: 예산(원 단위 정수). 없을 수 있어 Optional[int], 기본 None.
            프롬프트의 `budget_krw` 키로 LLM 에 전달된다.
    theme: 여행 테마 키워드 리스트. Optional[List[str]], 기본 None.
    mobility: 선호 이동 수단(`Mobility`). 없을 수 있음.
    province: 광역 행정구역(예: "서울특별시"). 1~20자.
    city: 시/군/구(예: "강남구"). 1~20자.

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
    name: 장소명.
    address: 주소 문자열.
    lat: 위도. ge=33.0, le=43.0 — 한국 국내 위도 범위 밖이면 검증 실패.
    lng: 경도. ge=124.0, le=132.0 — 한국 국내 경도 범위 밖이면 검증 실패.
    recommended_visit_time: 추천 방문 시간을 표현하는 자유 텍스트.

    사용처:
      - LLM 응답 검증의 단위 모델(`PlacesEnvelope.places` 원소).
      - `Leg.from_place_id` / `Leg.to_place_id` 가 본 `place_id` 를 참조.
      - `JobDonePayload.places` 의 원소.
    """
    model_config = ConfigDict(populate_by_name=True)

    place_id: int
    name: str
    address: str
    lat: float = Field(ge=33.0, le=43.0)
    lng: float = Field(ge=124.0, le=132.0)
    recommended_visit_time: str
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
    recommended_visit_time: 해당 장소의 권장 방문 시간 텍스트.
    """
    index: int
    recommended_visit_time: str


class PlacesSelection(BaseModel):
    """`recommend_places` 의 grounded 경로에서 LLM 이 돌려주는 선택 결과.

    selections: 실측 후보 중에서 고른 항목들(방문 시간 포함). 장소의
        이름·주소·좌표는 LLM 이 새로 만들지 않고 후보값을 그대로 쓴다.

    사용처:
      - `GeminiClient.generate_structured(prompt, PlacesSelection)` 의
        `response_schema`.
    """
    selections: List[PlaceSelection]


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
    places: 성공 시 추천 장소 리스트. 실패 시 None.
    visit_order: 성공 시 방문 순서. 실패 시 None.
    legs: 성공 시 동선 구간 리스트. 실패 시 None.
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
    error: Optional[str] = None
