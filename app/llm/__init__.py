"""LLM 호출 거버넌스 계층.

  - `rate_limit`     : Gemini token-bucket(RPM) + 일일 카운터(RPD cap).
  - `structured_call`: 요청당 호출 예산 집행 + 교정 재시도 헬퍼.

GeminiClient(전송)와 노드(비즈니스 로직) 사이에서 분당·일일·요청당 세 축의
호출 한도를 강제한다.
"""
