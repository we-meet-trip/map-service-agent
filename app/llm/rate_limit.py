"""Gemini 호출 rate-limit — token-bucket(RPM) + 일일 카운터(RPD cap).

외부 API 가 분당·일일 두 축으로 한도를 걸므로 두 축을 함께 집행한다:

  Redis DB3 key "gemini:tokens"          — leaky/token bucket (용량 10,
                                           refill 10/min = RPM 10 매칭)
  Redis DB3 key "gemini:daily:{YYYYMMDD}" — KST 기준 일일 카운터
                                           (cap 200/일, KST 자정 TTL)

두 검사와 소비는 Lua 스크립트 1회의 EVAL 로 원자적으로 수행된다
(다중 코루틴/다중 프로세스에서도 이중 소비 없음).

호출 흐름:
  1. try-consume 1 token
  2. OK          → Gemini API 호출 진행
  3. RPM DENIED  → 반환된 대기시간만큼 sleep 후 재시도, 누적 60초 초과 시
                   `GeminiQuotaError("gemini_rpm_exceeded")`
  4. RPD 초과    → 즉시 `GeminiQuotaError("gemini_rpd_exceeded")`

사용처: `app/main.py` lifespan 이 생성해 GeminiClient 에 limiter 로 주입.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis

# 테스트에서 대기를 계측/무력화할 수 있도록 간접 참조로 둔다.
_sleep = asyncio.sleep

# KST(UTC+9). 일일 카운터의 날짜 경계와 TTL 산정에 사용 —
# 재탐색 카운터와 동일하게 KST 자정 리셋으로 일관.
_KST = timezone(timedelta(hours=9))

# 반환 규약: {status, wait_ms}
#   status = "ok"   → 토큰 소비 + 일일 카운터 INCR 완료
#   status = "wait" → 버킷 고갈. wait_ms 후 재시도 권고 (소비 없음)
#   status = "rpd"  → 일일 cap 도달 (소비 없음)
_LUA_ACQUIRE = """
local daily_key = KEYS[1]
local tokens_key = KEYS[2]
local now_ms = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local refill_per_min = tonumber(ARGV[3])
local daily_cap = tonumber(ARGV[4])
local daily_ttl = tonumber(ARGV[5])

local daily = tonumber(redis.call('GET', daily_key) or '0')
if daily >= daily_cap then
  return {'rpd', 0}
end

local data = redis.call('HMGET', tokens_key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now_ms
end

local elapsed = now_ms - ts
if elapsed < 0 then elapsed = 0 end
tokens = tokens + elapsed * refill_per_min / 60000.0
if tokens > capacity then tokens = capacity end

if tokens < 1 then
  local wait_ms = math.ceil((1 - tokens) * 60000.0 / refill_per_min)
  redis.call('HSET', tokens_key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
  return {'wait', wait_ms}
end

tokens = tokens - 1
redis.call('HSET', tokens_key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
local new_daily = redis.call('INCR', daily_key)
if new_daily == 1 then
  redis.call('EXPIRE', daily_key, daily_ttl)
end
return {'ok', 0}
"""


class GeminiQuotaError(RuntimeError):
    """Gemini 호출 한도 초과.

    code: "gemini_rpm_exceeded"(60초 대기 후에도 버킷 고갈) 또는
          "gemini_rpd_exceeded"(일일 cap 도달).
    str(예외) 가 code 그대로가 되도록 하여, `_run_job` 이 실패 페이로드의
    error 필드에 이 코드를 그대로 싣게 한다.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GeminiRateLimiter:
    """Redis 기반 Gemini 호출 한도 집행기.

    `acquire()` 1회 성공 = Gemini 호출 1회 허가. GeminiClient 가
    generate 직전에 호출한다.
    """

    def __init__(
        self,
        redis_url: str,
        db: int,
        *,
        capacity: int = 10,
        refill_per_min: int = 10,
        daily_cap: int = 200,
        max_wait_seconds: float = 60.0,
        tokens_key: str = "gemini:tokens",
        daily_key_prefix: str = "gemini:daily:",
    ) -> None:
        """GeminiRateLimiter 초기화.

        redis_url: redis 접속 URL.
        db: rate-limit 전용 논리 DB 번호(DB3).
        capacity: 버킷 용량(기본 10 = RPM 10).
        refill_per_min: 분당 리필 토큰 수(기본 10).
        daily_cap: 일일 호출 상한(기본 200, 예비 50 남김).
        max_wait_seconds: RPM 고갈 시 누적 대기 상한(기본 60초). 초과 시
                          gemini_rpm_exceeded.
        tokens_key / daily_key_prefix: Redis 키 이름 규약.

        호출처: `app/main.py` lifespan.
        """
        self._client = aioredis.Redis.from_url(
            redis_url, db=db, decode_responses=True
        )
        self._capacity = capacity
        self._refill_per_min = refill_per_min
        self._daily_cap = daily_cap
        self._max_wait = max_wait_seconds
        self._tokens_key = tokens_key
        self._daily_key_prefix = daily_key_prefix
        self._script = self._client.register_script(_LUA_ACQUIRE)

    def _daily_key(self, now_kst: datetime) -> str:
        """KST 날짜 기준 일일 카운터 키."""
        return f"{self._daily_key_prefix}{now_kst.strftime('%Y%m%d')}"

    @staticmethod
    def _seconds_until_kst_midnight(now_kst: datetime) -> int:
        """KST 익일 자정까지 남은 초(일일 키 TTL). 최소 1초."""
        next_midnight = (now_kst + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((next_midnight - now_kst).total_seconds()))

    async def acquire(self) -> None:
        """토큰 1개 소비를 시도하고, 성공할 때까지 제한 대기한다.

        - RPD cap 도달: 즉시 `GeminiQuotaError("gemini_rpd_exceeded")`.
        - RPM 고갈: 스크립트가 권고한 시간만큼 sleep 후 재시도. 누적
          대기가 `max_wait_seconds` 를 넘게 되면
          `GeminiQuotaError("gemini_rpm_exceeded")`.

        호출처: `GeminiClient.generate_structured` (limiter 주입 시).
        """
        waited = 0.0
        while True:
            now_kst = datetime.now(tz=_KST)
            result = await self._script(
                keys=[self._daily_key(now_kst), self._tokens_key],
                args=[
                    int(time.time() * 1000),
                    self._capacity,
                    self._refill_per_min,
                    self._daily_cap,
                    self._seconds_until_kst_midnight(now_kst),
                ],
            )
            status = result[0]
            if status == "ok":
                return
            if status == "rpd":
                raise GeminiQuotaError("gemini_rpd_exceeded")
            wait_seconds = max(0.05, float(result[1]) / 1000.0)
            if waited + wait_seconds > self._max_wait:
                raise GeminiQuotaError("gemini_rpm_exceeded")
            await _sleep(wait_seconds)
            waited += wait_seconds

    async def aclose(self) -> None:
        """내부 redis 클라이언트를 닫는다.

        호출처: `app/main.py` lifespan 종료 finally 블록.
        """
        await self._client.aclose()
