"""프롬프트 삽입 전 텍스트/구조 새니타이즈.

목적: 사용자 입력(theme/province/city)과 외부 데이터(hub 후보의
name/address, 날씨 dict — 간접 인젝션 채널)가 프롬프트의 데이터 태그
(<user_input>/<weather_context>/<candidates>) 를 깨고 지시문으로
승격되는 것을 봉쇄한다.

핵심: `json.dumps` 는 따옴표만 이스케이프하고 `</user_input>` 같은
리터럴 태그 종료 문자열은 그대로 통과시킨다. 따라서 태그 문자
`<`/`>` 자체를 HTML 엔티티로 치환하는 것이 태그 브레이크아웃 방어의
필수 조건이다.

원본 비파괴: 새니타이즈는 프롬프트에 넣을 **뷰(view)** 에만 적용하고,
상태(state)의 원본 dict/객체는 보존한다(Place 변환 등은 원본 사용).
"""
from __future__ import annotations

import re

# 제어문자 + 서식 제어용 유니코드(zero-width, bidi override, line/para sep).
# 프롬프트 내 보이지 않는 지시 삽입과 로그 오염을 함께 차단한다.
_CONTROL_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    "\u200b-\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069"
    "\u2060-\u2064\ufeff"           # word joiner·invisible operator·BOM
    "\U000e0000-\U000e007f"          # Unicode Tags 블록(비가시 스머글링)
    "]"
)

# 연속 공백(개행 포함) 정규화용.
_WHITESPACE_RE = re.compile(r"\s+")

# 구조 새니타이즈 시 컨테이너 깊이 상한 초과 표시자.
_DEPTH_PLACEHOLDER = "…"

# 표시값 정규화 시 꺾쇠(반각/전각)를 대체할 안전 구분자.
# Kakao category("음식점 > 카페") 등 외부 데이터의 계층 구분 `>` 를
# 보존하되 Place 검증기(_reject_tags_and_control)에 걸리지 않게 한다.
_TAG_CHARS_RE = re.compile("[<>＜＞]")
_TAG_REPLACEMENT = "/"


def neutralize_tags(value):
    """표시용 문자열의 꺾쇠·제어문자를 무해화한다(엔티티 이스케이프 아님).

    `sanitize_text` 는 프롬프트 뷰용으로 `<`/`>` 를 HTML 엔티티로 바꾸지만,
    본 함수는 **최종 출력 필드**(예: `Place.category`)용이다. 사람이 읽을
    표시값이므로 엔티티 대신 안전 구분자 `/` 로 치환한다. 처리:
      1. 제어문자·서식 제어 유니코드 제거(`_CONTROL_RE`)
      2. `<`, `>`, 전각 `＜`(U+FF1C), `＞`(U+FF1E) → `/`

    치환 결과는 `agent_schemas._FORBIDDEN_TEXT_RE` 를 통과하므로 Place 의
    `_no_tags` 검증기와 충돌하지 않는다. None 또는 비-str 은 그대로 반환한다
    (선택 필드·좌표 등을 안전하게 통과).

    호출처: `agent_nodes._place_from_candidate` (category / 방문시간 정규화).
    """
    if not isinstance(value, str):
        return value
    cleaned = _CONTROL_RE.sub("", value)
    return _TAG_CHARS_RE.sub(_TAG_REPLACEMENT, cleaned)


def sanitize_text(value: str, max_len: int) -> str:
    """문자열 1개를 프롬프트 삽입용으로 정화한다.

    처리 순서:
      1. 제어문자·서식 제어 유니코드 제거
      2. 연속 공백을 단일 공백으로 정규화(개행 지시 구조 제거)
      3. `<` → `&lt;`, `>` → `&gt;` (태그 브레이크아웃 봉쇄)
      4. max_len 절단 (엔티티가 잘려도 태그 문자는 남지 않아 무해)

    호출처: agent_nodes 의 프롬프트 빌더들, sanitize_struct.
    """
    cleaned = _CONTROL_RE.sub("", value)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")
    # 전각 꺾쇠(U+FF1C/FF1E)도 치환 — 현재 코드베이스에 NFKC 정규화가
    # 없어 리터럴 펜스는 못 깨지만, 모델의 의미 해석 리스크와 향후
    # 정규화 도입 시의 브레이크아웃을 선제 차단한다.
    cleaned = cleaned.replace("\uff1c", "&lt;").replace("\uff1e", "&gt;")
    return cleaned[:max_len]


def sanitize_struct(obj, *, str_max: int, depth_max: int = 6):
    """dict/list 를 재귀 순회하며 모든 문자열을 sanitize_text 한다.

    obj: 임의의 JSON 유사 구조(날씨 dict 등).
    str_max: 문자열 값 하나의 최대 길이.
    depth_max: 재귀 깊이 상한. 초과 컨테이너는 "…" 로 대체해
               악의적/비정상 중첩으로 인한 폭주를 막는다.

    dict 키도 문자열이면 새니타이즈한다(키를 통한 주입 차단).
    숫자/불리언/None 은 그대로 두고, 그 외 타입(date 등)은 호출측
    `json.dumps(default=str)` 에 맡긴다.

    반환: 새니타이즈된 **사본** (원본 비파괴).

    호출처: `_build_places_prompt`/`_build_selection_prompt` 의
    weather 뷰 조립.
    """
    if depth_max < 0:
        return _DEPTH_PLACEHOLDER
    if isinstance(obj, str):
        return sanitize_text(obj, str_max)
    if isinstance(obj, dict):
        if depth_max == 0:
            return _DEPTH_PLACEHOLDER
        return {
            (
                sanitize_text(k, str_max)
                if isinstance(k, str)
                else k
            ): sanitize_struct(v, str_max=str_max, depth_max=depth_max - 1)
            for k, v in obj.items()
        }
        # 키 충돌(새니타이즈로 동일해진 키)은 마지막 값이 남는다 — 데이터
        # 뷰 목적상 허용.
    if isinstance(obj, (list, tuple)):
        if depth_max == 0:
            return _DEPTH_PLACEHOLDER
        return [
            sanitize_struct(v, str_max=str_max, depth_max=depth_max - 1)
            for v in obj
        ]
    return obj
