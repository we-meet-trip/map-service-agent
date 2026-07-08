"""LLM 호출 거버넌스 계층.

  - `rate_limit`     : Gemini token-bucket(RPM) + 일일 카운터(RPD cap).
  - `structured_call`: 요청당 호출 예산(≤3회) + 교정 재시도 헬퍼.

GeminiClient(전송)와 노드(비즈니스 로직) 사이에서 SoT §10.2·R-01·§2.3 의
호출 한도 정책을 강제한다.
"""
