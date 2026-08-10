"""프롬프트에 실을 JSON 문자열을 만드는 단일 진입점.

프롬프트마다 `json.dumps` 를 따로 부르던 것을 여기로 모았다. 옵션이
흩어져 있으면 새 프롬프트를 추가할 때 하나씩 빠뜨리게 되는데, 빠뜨려도
동작은 하고 조용히 비싸지기만 해서 알아채기 어렵다.

세 가지를 한다.

**한글을 그대로 둔다.** `ensure_ascii=False` 가 없으면 한글 한 글자가
`\\uAC15` 여섯 글자로 부풀어 입력이 몇 배가 된다.

**구분자에서 공백을 뺀다.** `json.dumps` 기본값은 `", "` 와 `": "` 라
콤마·콜론마다 공백이 하나씩 붙는다. 후보 수십 건에 필드가 여럿이면 그
공백만으로 천 자 단위가 쌓인다. 의미는 전혀 달라지지 않는다.

**빈 값을 지운다.** `None`, 빈 문자열, 빈 리스트·딕셔너리는 키째로
뺀다. 후보 대부분은 후기를 붙이지 않는데(상위 몇 건만 조회한다) 그
후보들이 저마다 빈 배열을 싣고 다니는 것이 가장 큰 낭비였다.
`0` 과 `False` 는 지우지 않는다 — 강수확률 0 과 "값 없음" 은 다른 뜻이다.

빈 값을 지우기 때문에 **프롬프트 규칙에 "키가 없으면 그 값이 없다는
뜻" 이라고 적어 두어야 한다.** 그러지 않으면 모델이 키가 사라진 것을
이상 신호로 읽을 수 있다.
"""
from __future__ import annotations

import json

# 콤마·콜론 뒤의 공백을 뺀 구분자.
_COMPACT = (",", ":")


def _is_empty(value) -> bool:
    """키째로 뺄 값인지 판정한다.

    길이가 0 인 컨테이너와 빈 문자열만 대상이다. 숫자 0 과 False 는
    길이 개념이 없어 여기 걸리지 않는다 — 의도한 동작이다.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) == 0
    return False


def prune_empty(value):
    """빈 값을 재귀적으로 걷어 낸 사본을 만든다(원본 비파괴).

    딕셔너리는 값을 먼저 다듬은 뒤 비었으면 키를 뺀다. 그래서 안쪽이
    전부 비면 바깥 키도 함께 사라진다.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            pruned = prune_empty(v)
            if not _is_empty(pruned):
                out[k] = pruned
        return out
    if isinstance(value, (list, tuple)):
        return [
            p for p in (prune_empty(v) for v in value) if not _is_empty(p)
        ]
    return value


def dump_prompt_json(value) -> str:
    """프롬프트 데이터 태그 안에 넣을 JSON 문자열.

    `default=str` 을 두어 date/time 처럼 직렬화되지 않는 값이 섞여도
    프롬프트 조립이 깨지지 않게 한다(날씨 응답이 그런 값을 담고 온다).
    """
    return json.dumps(
        prune_empty(value),
        ensure_ascii=False,
        separators=_COMPACT,
        default=str,
    )
