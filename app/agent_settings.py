"""agent-service 환경설정.

`AgentSettings(BaseSettings)` 는 환경 변수(또는 `.env`) 로부터 값을
읽어 들이는 설정 객체이고, `get_settings()` 는 프로세스 1회 캐시되는
싱글톤 접근자다. `app/main.py` lifespan 이 본 모듈을 호출해
HubClient/StreamsPublisher/GeminiClient 구성과 잡 타임아웃에 사용한다.
"""
from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """agent-service 의 모든 환경 변수 묶음.

    `model_config`:
      - env_file=".env": 작업 디렉토리의 .env 파일을 자동 로드.
      - env_file_encoding="utf-8": .env 파일 인코딩.
      - extra="ignore": 정의되지 않은 환경 변수는 조용히 무시.

    Gemini 관련:
      GEMINI_API_KEY: 비밀값(SecretStr). 기본 빈 문자열.
                      lifespan 진입 시 비어 있으면 RuntimeError 로 부팅 중단.
      GEMINI_MODEL: 모델 이름 문자열. GeminiClient 생성 시 전달.
      GEMINI_RPM_LIMIT: 분당 호출 상한(외부 API 가 정한 값). token-bucket
                        용량/리필률로 사용된다.
      GEMINI_RPD_LIMIT: 외부 API 의 일일 상한(250, 문서화 목적).
                        실제 집행은 GEMINI_RPD_CAP 이 담당한다.
      GEMINI_RPD_CAP: agent 자체 일일 상한(200/일). 외부 한도보다 낮게 잡아
                      50 회를 수동 점검·데모용 예비로 남긴다.
                      GeminiRateLimiter 의 일일 카운터가 KST 자정 기준으로
                      집행한다.
      GEMINI_MAX_CALLS_PER_REQUEST: 추천 요청 1건당 LLM 호출 하드 예산.
                      교정 재시도 포함 모든 호출이 본 예산을 소비한다.
                      정상 경로 소비 내역: 장소 선정 1 + 추천이유 1 = 2.
                      동선은 좌표로 계산해 모델을 부르지 않는다. 남는 1은
                      앞 단계가 형식을 한 번 어겼을 때 쓸 교정 재시도
                      여력으로 비워 둔 것이다. 블로그 요약은 추천
                      파이프라인에서 빠져 별도 파이프라인이 자기 예산
                      (SUMMARY_MAX_LLM_CALLS)으로 돈다. 다만 일일 한도는
                      두 파이프라인이 같은 카운터를 공유한다.
      GEMINI_TEMPERATURE: 생성 온도. 구조화 출력의 결정성을 위해 낮게 둔다.
      GEMINI_MAX_OUTPUT_TOKENS: 응답 토큰 상한. Gemini 2.5 계열은 thinking
                      토큰이 본 상한을 함께 소비하므로 과소 설정 시 JSON 이
                      절단(EOF)되어 ValidationError 가 빈발한다. thinking 을
                      꺼도(budget=0) 여유 마진을 위해 8192 로 둔다.
      GEMINI_THINKING_BUDGET: Gemini 2.5 thinking 토큰 예산.
                      0 = thinking 끔(구조화 JSON 추출의 신뢰성·속도·비용에
                      최적, 기본값). 양수 = 해당 토큰까지 thinking 허용.
                      음수(-1) = ThinkingConfig 미설정(모델 기본, thinking
                      미지원 모델 호환용). E2E 실측(2026-07-05): 기본
                      thinking 이 최대 ~7000 토큰을 써 4096 상한에서 5회 중
                      2회 JSON 절단 → budget=0 에서 5/5 성공.

    Gemini 전송 재시도(503/500 완화) 관련:
      GEMINI_RETRY_ATTEMPTS: SDK 전송 재시도의 총 시도 수(초기 호출 포함).
                      google-genai 의 HttpRetryOptions.attempts 로 전달된다.
                      1 이하이면 재시도 비활성(오늘 동작). E2E 실측
                      (2026-07-05): 무료 티어에서 503 "high demand" 가
                      빈발하고 재시도가 없어 recommend_places/route 가
                      잡 전체를 하드실패시켰다. 이 계층은 전송 오류만
                      재시도하며 state 예산(≤3)·GeminiRateLimiter 회계와
                      무관하다.
      GEMINI_RETRY_INITIAL_DELAY: 지수 백오프 초기 지연(초). 0 이면 SDK 기본.
      GEMINI_RETRY_MAX_DELAY: 지수 백오프 상한(초). 0 이면 SDK 기본.
      GEMINI_RETRY_STATUS_CODES: 재시도 대상 HTTP 상태코드(쉼표 구분 문자열).
                      기본 "503,500". **429 는 반드시 제외**한다 — 429 는
                      GeminiRateLimiter(RPM/RPD) 소관이며, SDK 가 재시도하면
                      서버 쿼터를 limiter 회계 밖에서 이중 소진한다. 빈
                      문자열/파싱 공집합이면 재시도 비활성. (주의: SDK 는
                      빈 리스트를 falsy 로 보고 기본코드(429 포함)로 되돌리므로,
                      클라이언트가 공집합을 None 으로 변환해 넘긴다.)
                      참고: google-genai 2.3.0 은 http_status_codes 와 별개로
                      httpx.TimeoutException/ConnectError 를 항상 재시도한다
                      (전송 hang·연결 오류 자동 복원).
      GEMINI_HTTP_TIMEOUT_SECONDS: per-attempt(시도별) HTTP 타임아웃(초).
                      HttpOptions.timeout(ms) 로 전달. 시도 1건의 hang 을
                      bound 하고(초과 시 httpx.TimeoutException → 재시도),
                      GEMINI_TIMEOUT_SECONDS(전체 wait_for)와 함께 예산을
                      구성한다. 0 이하이면 미설정(글로벌 기본).
                      최악 지연 = attempts × per-attempt + Σ백오프 이며
                      GEMINI_TIMEOUT_SECONDS 안에 들어가야 재시도가
                      wait_for 에 잘리지 않는다.

    Hub HTTP 관련:
      HUB_BASE_URL: hub 게이트웨이의 base URL. HubClient 가 본 값에
                    `/v1/weather` 를 붙여 호출한다.
      HUB_TIMEOUT_SECONDS: hub HTTP 호출의 전체 timeout(초).
      INTERNAL_SERVICE_TOKEN: 내부 서비스 인증 토큰. 비밀값(SecretStr).
                              None 이면 HubClient 가 `X-Internal-Token` 헤더를
                              아예 붙이지 않는다.
      AUTH_ENFORCED: True 면 lifespan 진입 시 INTERNAL_SERVICE_TOKEN 이
                     비어 있으면 부팅을 중단한다(fail-fast — 운영 배포에서
                     미인증 노출 방지). False(기본)면 토큰 미설정 시
                     `_verify_internal_token` 이 경고 로그와 함께 검증을
                     생략하는 기존 동작을 유지한다.

    리뷰 보강(hub `/v1/reviews`) 관련:
      REVIEWS_ENRICH_ENABLED: True(기본)면 grounded 경로에서 상위 후보에
                     블로그 리뷰 스니펫을 붙여 선정/이유 프롬프트를 보강한다.
                     리뷰 수집은 best-effort(실패해도 잡을 죽이지 않는다)이며
                     LLM 호출을 추가하지 않는다(hub HTTP 만 추가).
      REVIEWS_MAX_PLACES: 선정 단계에서 리뷰를 조회할 상위 후보 수.
                     랭킹 상위 K 개에만 조회해 hub 왕복을 억제한다.
      REVIEWS_DISPLAY: 후보 1건당 hub 에 요청하는 리뷰 수(display).
                     여러 글을 교차 대조할 수 있도록 5로 둔다. 선정·이유
                     프롬프트 뷰는 이 중 앞 2건만 쓴다(토큰 절약).
      REVIEWS_FETCH_CAP_PER_JOB: 잡 1건이 hub `/v1/reviews` 를 호출할 총
                     상한. 선정 단계가 상위 후보만 조회하므로 실제로는
                     여유가 남는 안전망이다 — 장소 수에 비례해 외부 API
                     호출이 무한히 늘어나는 것을 막는다. hub 가 6시간
                     캐시를 두므로 같은 장소 재조회는 외부 API 호출로
                     이어지지 않는다.

    블로그 요약 파이프라인 관련:
      SUMMARY_ENABLED: True(기본)면 요약 엔드포인트가 파이프라인을 돌려
                     장소별 요약 2줄(bullets)을 만든다. False 면 파이프라인을
                     부르지 않고 빈 결과를 돌려준다(kill-switch — 쿼터
                     압박·품질 회귀 시 즉시 무력화). 빈 요약은 호출 측이
                     캐시하지 않으므로 되돌리면 즉시 정상 동작한다.
      SUMMARY_MAX_PLACES: 한 번에 요약할 최대 장소 수. 프롬프트에 실리는
                     스니펫 총량의 운영 상한이며, 요청 스키마의 상한과
                     별개로 배포 없이 낮출 수 있는 조절값이다.
      SUMMARY_MAX_LLM_CALLS: 요약 요청 1건당 LLM 호출 예산. 본 호출 1회 +
                     응답이 형식을 벗어났을 때의 교정 재시도 1회로 2를
                     둔다. 추천 파이프라인의 예산과 분리돼 있어 요약이
                     추천 잡의 호출 여유를 잠식하지 않는다.

    후보 절단 관련:
      CANDIDATES_MAX_BASE / CANDIDATES_PER_DAY: 선정 프롬프트에 실을 후보
                     수 상한을 max(BASE, 여행일수 × PER_DAY) 로 정한다.
                     테마별로 hub 를 따로 부르고 합치는 구조라 테마 수에
                     비례해 후보가 늘고 프롬프트 입력도 같이 커지는데,
                     실제로 고르는 것은 일수 × 2~4 곳뿐이다. 절단은
                     recommend_places 한 곳에서만 하며 실내/실외를 절반씩
                     남겨, 채점이 실패해 정렬 없이 온 경우에도 실내 후보가
                     통째로 잘리지 않게 한다.

    룰 엔진(hub `/v1/rules/*`) 관련:
      RULES_ENABLED: True(기본)면 rules_filter/score_and_rank 가 hub 룰
                     엔드포인트를 호출해 반경 필터·실내 보너스 랭킹을
                     적용한다. False 면 두 노드가 no-op pass-through 로
                     복귀한다(kill-switch — hub 장애/회귀 시 즉시 무력화).
      TIMELINE_ENABLED: True(기본)면 build_timeline 이 hub 체류시간 룰로
                     방문 시각을 계산하고, 하루 활동 시간을 넘치면
                     fit_time_budget 이 결정적으로 줄인다. False 면 두 노드가
                     통과만 해 페이로드가 시간축 도입 전과 같아진다.
      TIMELINE_BUFFER_MINUTES: 장소 사이에 두는 여유(분).

    Redis Streams 관련:
      REDIS_URL: redis 접속 URL.
      REDIS_DB_STREAMS: streams 발행용 DB 번호(논리적 분리). 기본 2.
                        user-service 컨슈머(redis.db-streams)와 반드시
                        같은 값이어야 한다 — 어긋나면 발행은 성공하는데
                        소비자가 영원히 빈 stream 을 읽는다.
      STREAM_NAME: 잡 완료 페이로드를 발행할 stream 이름.
      STREAM_MAXLEN: 0 보다 크면 XADD 시 approximate trim 으로 stream
                     길이를 본 값 근처로 제한한다(메모리 무한 누적 방지).
      STATUS_STREAM_NAME: 노드 진행 이벤트를 발행할 stream 이름
                          (`agent:jobs:status`). done 스트림과 동일
                          DB(REDIS_DB_STREAMS)에 둔다.
      STATUS_STREAM_MAXLEN: status stream 의 approximate trim 상한.
      REDIS_DB_RATELIMIT: Gemini token-bucket/일일 카운터 전용 DB 번호
                          (기본 3 — rate-limit 정책 키를 한 DB 에 모은다).

    LangGraph 체크포인터(Postgres) 관련:
      CHECKPOINT_ENABLED: True 면 lifespan 이 Postgres 체크포인터를
                          배선한다. 연결 실패 시 부팅 중단(fail-fast).
                          False 면 체크포인터 없이 compile.
      POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER:
                          체크포인터 접속 정보.
      POSTGRES_PASSWORD: 비밀값(SecretStr).
      LANGGRAPH_SCHEMA: 체크포인트 테이블이 거주할 schema 이름
                        (`langgraph`, agent 전용).

    잡 타이밍:
      JOB_TIMEOUT_SECONDS: 잡 1건의 전체 실행 한도. `_run_job` 의
                           `asyncio.wait_for(..., timeout=...)` 에 사용.
                           산식(재시도 반영, 2026-07-06): LLM ≤3콜 ×
                           (rate-limit 대기 최대 60s + 생성 wait_for 최대
                           GEMINI_TIMEOUT_SECONDS=100s) 는 이론최악이나,
                           본 wall-clock 캡이 잡 전체를 300s 에서 강제
                           종료하므로 300s 를 넘지 않는다. 정상 경로는
                           수십 초 내 완료. BFF 가 허용하는 잡 대기
                           상한(10분) 안에 들어온다.
      GEMINI_TIMEOUT_SECONDS: Gemini 호출 1건의 한도. GeminiClient 내부
                              `generate_text` / `generate_structured` 가
                              `asyncio.wait_for` 로 강제.
      SHUTDOWN_GRACE_SECONDS: lifespan 종료 시 진행 중 백그라운드 잡들이
                              마무리되길 기다리는 한도. 초과하면 cancel.
                              JOB_TIMEOUT_SECONDS + 10s 로 연동한다.

    관측(로깅):
      LOG_LEVEL: 루트 로거 레벨. uvicorn 은 자기 로거만 구성하고 루트
                 로거에는 핸들러를 붙이지 않으므로, `app/main.py` 의
                 `_configure_logging` 이 부팅 시 루트 핸들러가 비어 있을
                 때만 stdout 핸들러를 붙여 `app.*` 로그를 살린다.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_RPM_LIMIT: int = 10
    GEMINI_RPD_LIMIT: int = 250
    GEMINI_RPD_CAP: int = 200
    GEMINI_MAX_CALLS_PER_REQUEST: int = 3
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS: int = 8192
    GEMINI_THINKING_BUDGET: int = 0
    GEMINI_RETRY_ATTEMPTS: int = 3
    GEMINI_RETRY_INITIAL_DELAY: float = 1.0
    GEMINI_RETRY_MAX_DELAY: float = 8.0
    GEMINI_RETRY_STATUS_CODES: str = "503,500"
    GEMINI_HTTP_TIMEOUT_SECONDS: float = 30.0

    HUB_BASE_URL: str = "http://hub:8000"
    HUB_TIMEOUT_SECONDS: float = 5.0
    INTERNAL_SERVICE_TOKEN: SecretStr | None = None

    # 서비스 사이로 나가는 좌표를 감싸는 열쇠. hub·BFF 와 같은 값을 나눠 가진다.
    # 값이 있으면 좌표를 감싸서 보내고, 없으면 예전처럼 값 그대로 보낸다 —
    # 받는 쪽이 아직 열 줄 모르는 동안 넘어가기 위한 것이다.
    # 32바이트로 풀리는 base64 여야 한다: openssl rand -base64 32
    LOCATION_WIRE_KEY: SecretStr = SecretStr("")
    AUTH_ENFORCED: bool = False

    REVIEWS_ENRICH_ENABLED: bool = True
    REVIEWS_MAX_PLACES: int = 3
    REVIEWS_DISPLAY: int = 5  # hub 가 받는 상한은 10
    # 선정 단계는 상위 후보(REVIEWS_MAX_PLACES)만 조회하므로 실제 소비는 이
    # 값에 한참 못 미친다. 장소 수에 비례해 호출이 늘어나지 않게 막는 안전망.
    REVIEWS_FETCH_CAP_PER_JOB: int = 10

    SUMMARY_ENABLED: bool = True
    # 한 요청에 담을 장소 수 상한. 요청 스키마의 상한과 같은 값으로 두되,
    # 배포 없이 낮출 수 있도록 설정으로 분리해 둔다.
    SUMMARY_MAX_PLACES: int = 7
    # 본 호출 1 + 형식 이탈 시 교정 재시도 1.
    SUMMARY_MAX_LLM_CALLS: int = 2

    # 선정 프롬프트에 실을 후보 수 상한. 테마마다 hub 를 따로 부르고 합치기
    # 때문에 테마를 많이 고르면 후보가 선형으로 늘고, 그 전량이 프롬프트에
    # 실려 입력이 커진다. 실제로 고르는 것은 일수 × 2~4 곳뿐이라 나머지는
    # 읽히고 버려진다. 상한 = max(BASE, 여행일수 × PER_DAY).
    CANDIDATES_MAX_BASE: int = 40
    CANDIDATES_PER_DAY: int = 12

    # 테마 친화 가점 스위치. False 면 score_and_rank 가 가점 0 으로 계산해
    # 페이로드가 도입 전과 같아진다. 가점 자체가 score_and_rank 안에 있어
    # RULES_ENABLED=false 일 때도 함께 꺼진다.
    PERSONALIZATION_ENABLED: bool = True

    RULES_ENABLED: bool = True  # kill-switch — 장애 시 no-op 복귀

    # 시간축(체류시간·방문 시각) 산출 스위치. False 면 build_timeline 과
    # fit_time_budget 이 no-op 으로 통과해 페이로드가 도입 전과 같아진다.
    TIMELINE_ENABLED: bool = True
    # 한 장소와 다음 장소 사이에 두는 여유(분). 길 찾기·주차·대기처럼
    # 이동시간에 잡히지 않는 시간을 흡수한다. 0 으로 두면 일정이 분 단위로
    # 빈틈없이 붙어 실제로 지킬 수 없는 시간표가 된다.
    TIMELINE_BUFFER_MINUTES: int = 10

    REDIS_URL: str = "redis://redis:6379"
    REDIS_DB_STREAMS: int = 2
    REDIS_DB_RATELIMIT: int = 3
    STREAM_NAME: str = "agent:jobs:done"
    STREAM_MAXLEN: int = 2000
    STATUS_STREAM_NAME: str = "agent:jobs:status"
    STATUS_STREAM_MAXLEN: int = 2000

    CHECKPOINT_ENABLED: bool = True
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "map"
    POSTGRES_USER: str = "map"
    POSTGRES_PASSWORD: SecretStr = SecretStr("")
    LANGGRAPH_SCHEMA: str = "langgraph"

    JOB_TIMEOUT_SECONDS: float = 300.0
    GEMINI_TIMEOUT_SECONDS: float = 100.0
    SHUTDOWN_GRACE_SECONDS: float = 310.0

    LOG_LEVEL: str = "INFO"


# _settings
#   - 프로세스 1회 생성 후 재사용되는 AgentSettings 캐시 슬롯.
#   - lifespan 이 첫 호출 때 인스턴스를 만든다.
_settings: AgentSettings | None = None


def get_settings() -> AgentSettings:
    """프로세스 단위 싱글톤 AgentSettings 접근자.

    첫 호출 시 `AgentSettings()` 를 만들어 모듈 전역 슬롯 `_settings` 에
    저장하고 이후로는 같은 인스턴스를 돌려준다.
    환경 변수/.env 평가가 1회만 일어나도록 보장한다.

    호출처: `app/main.py` lifespan.
    """
    global _settings
    if _settings is None:
        _settings = AgentSettings()
    return _settings
