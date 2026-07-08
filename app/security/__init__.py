"""프롬프트 인젝션 방어 계층.

  - `sanitize`: 사용자 입력·외부 데이터(hub 후보/날씨)의 프롬프트 삽입 전
    새니타이즈(제어문자 제거, 태그 이스케이프, 길이 상한).

방어 구조(4층):
  1층 system_instruction 분리 — 불변 규칙에 사용자 문자열 혼입 금지.
  2층 데이터 태그 격리 — user 콘텐츠는 <user_input> 등 태그 안 데이터로만.
  3층 본 새니타이즈 — 태그 브레이크아웃(`</user_input>` 주입) 봉쇄.
  4층 구조화 출력 — response_schema + pydantic + semantic 검증.
"""
