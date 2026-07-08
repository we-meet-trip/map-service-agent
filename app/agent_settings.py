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
      GEMINI_RPM_LIMIT: 분당 호출 상한(외부 API 한도, SoT §10). token-bucket
                        용량/리필률로 사용된다.
      GEMINI_RPD_LIMIT: 일일 호출 상한(외부 API 한도 250, 문서화 목적).
                        실제 집행은 GEMINI_RPD_CAP 이 담당한다.
      GEMINI_RPD_CAP: agent 자체 일일 상한(SoT R-01: 200/일, 50은 데모 예비).
                      GeminiRateLimiter 의 일일 카운터가 KST 자정 기준으로
                      집행한다.
      GEMINI_MAX_CALLS_PER_REQUEST: 추천 요청 1건당 LLM 호출 하드 예산
                      (SoT §2.3 "요청당 LLM ≤ 3회"). 교정 재시도 포함
                      모든 호출이 본 예산을 소비한다.
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
      REVIEWS_MAX_PLACES: 리뷰를 조회할 상위 후보 수(SoT §7.1 추천당 2-5회
                     이내). 랭킹 상위 K 개에만 조회한다.
      REVIEWS_DISPLAY: 후보 1건당 hub 에 요청하는 리뷰 수(display).

    룰 엔진(hub `/v1/rules/*`) 관련:
      RULES_ENABLED: True(기본)면 rules_filter/score_and_rank 가 hub 룰
                     엔드포인트를 호출해 반경 필터·실내 보너스 랭킹을
                     적용한다. False 면 두 노드가 no-op pass-through 로
                     복귀한다(kill-switch — hub 장애/회귀 시 즉시 무력화).

    Redis Streams 관련:
      REDIS_URL: redis 접속 URL.
      REDIS_DB_STREAMS: streams 발행용 DB 번호(논리적 분리).
                        user-service 컨슈머(redis.db-streams, 기본 2)와
                        일치해야 한다. SoT §8.3 은 DB3 로 표기하나 실측
                        양측 합의값은 DB2 — amendment 로 SoT 정정 예정
                        (사용자 결정 2026-07-05).
      STREAM_NAME: 잡 완료 페이로드를 발행할 stream 이름.
      STREAM_MAXLEN: 0 보다 크면 XADD 시 approximate trim 으로 stream
                     길이를 본 값 근처로 제한한다(메모리 무한 누적 방지).
      STATUS_STREAM_NAME: 노드 진행 이벤트를 발행할 stream 이름
                          (SoT §4.5 `agent:jobs:status`). done 스트림과
                          동일 DB(REDIS_DB_STREAMS)에 둔다.
      STATUS_STREAM_MAXLEN: status stream 의 approximate trim 상한.
      REDIS_DB_RATELIMIT: Gemini token-bucket/일일 카운터 전용 DB 번호
                          (SoT §8.3 DB3: rate-limit 정책 그룹).

    LangGraph 체크포인터(Postgres) 관련:
      CHECKPOINT_ENABLED: True 면 lifespan 이 Postgres 체크포인터를
                          배선한다(SoT B5). 연결 실패 시 부팅 중단
                          (fail-fast, 사용자 결정 2026-07-05).
                          False 면 체크포인터 없이 compile.
      POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER:
                          체크포인터 접속 정보.
      POSTGRES_PASSWORD: 비밀값(SecretStr).
      LANGGRAPH_SCHEMA: 체크포인트 테이블이 거주할 schema 이름
                        (SoT §8.1: `langgraph`, agent 전용).

    잡 타이밍:
      JOB_TIMEOUT_SECONDS: 잡 1건의 전체 실행 한도. `_run_job` 의
                           `asyncio.wait_for(..., timeout=...)` 에 사용.
                           산식(재시도 반영, 2026-07-06): LLM ≤3콜 ×
                           (rate-limit 대기 최대 60s + 생성 wait_for 최대
                           GEMINI_TIMEOUT_SECONDS=100s) 는 이론최악이나,
                           본 wall-clock 캡이 잡 전체를 300s 에서 강제
                           종료하므로 300s 를 넘지 않는다. 정상 경로는
                           수십 초 내 완료. SoT B2 예산(10분=600s) 내.
      GEMINI_TIMEOUT_SECONDS: Gemini 호출 1건의 한도. GeminiClient 내부
                              `generate_text` / `generate_structured` 가
                              `asyncio.wait_for` 로 강제.
      SHUTDOWN_GRACE_SECONDS: lifespan 종료 시 진행 중 백그라운드 잡들이
                              마무리되길 기다리는 한도. 초과하면 cancel.
                              JOB_TIMEOUT_SECONDS + 10s 로 연동한다.
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
    AUTH_ENFORCED: bool = False

    REVIEWS_ENRICH_ENABLED: bool = True
    REVIEWS_MAX_PLACES: int = 3  # SoT §7.1 추천당 2-5회 이내
    REVIEWS_DISPLAY: int = 3

    RULES_ENABLED: bool = True  # kill-switch — 장애 시 no-op 복귀

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
