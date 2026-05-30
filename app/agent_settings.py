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

    Hub HTTP 관련:
      HUB_BASE_URL: hub 게이트웨이의 base URL. HubClient 가 본 값에
                    `/v1/weather` 를 붙여 호출한다.
      HUB_TIMEOUT_SECONDS: hub HTTP 호출의 전체 timeout(초).
      INTERNAL_SERVICE_TOKEN: 내부 서비스 인증 토큰. 비밀값(SecretStr).
                              None 이면 HubClient 가 `X-Internal-Token` 헤더를
                              아예 붙이지 않는다.

    Redis Streams 관련:
      REDIS_URL: redis 접속 URL.
      REDIS_DB_STREAMS: streams 발행용 DB 번호(논리적 분리).
      STREAM_NAME: 잡 완료 페이로드를 발행할 stream 이름.
      STREAM_MAXLEN: 0 보다 크면 XADD 시 approximate trim 으로 stream
                     길이를 본 값 근처로 제한한다(메모리 무한 누적 방지).

    잡 타이밍:
      JOB_TIMEOUT_SECONDS: 잡 1건의 전체 실행 한도. `_run_job` 의
                           `asyncio.wait_for(..., timeout=...)` 에 사용.
      GEMINI_TIMEOUT_SECONDS: Gemini 호출 1건의 한도. GeminiClient 내부
                              `generate_text` / `generate_structured` 가
                              `asyncio.wait_for` 로 강제.
      SHUTDOWN_GRACE_SECONDS: lifespan 종료 시 진행 중 백그라운드 잡들이
                              마무리되길 기다리는 한도. 초과하면 cancel.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL: str = "gemini-2.5-flash"

    HUB_BASE_URL: str = "http://hub:8000"
    HUB_TIMEOUT_SECONDS: float = 5.0
    INTERNAL_SERVICE_TOKEN: SecretStr | None = None

    REDIS_URL: str = "redis://redis:6379"
    REDIS_DB_STREAMS: int = 2
    STREAM_NAME: str = "agent:jobs:done"
    STREAM_MAXLEN: int = 2000

    JOB_TIMEOUT_SECONDS: float = 120.0
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    SHUTDOWN_GRACE_SECONDS: float = 130.0


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
