"""뽑아 둔 세션에서 "어떤 사람이 어떤 곳을 남겼나" 를 세어 둔다.

무엇을 만드는가:
  성향 묶음(연령대·성별) × 장소 별로 "보여 준 횟수" 와 "저장까지 간 횟수" 를
  세어, 그 비율을 파일 하나로 떨군다. 랭킹이 그 파일을 읽어 조금 가점한다.

왜 모델이 아니라 세기부터인가:
  지금 쌓인 세션이 열몇 건이라 배우는 것이 아니라 외우는 데 그친다. 세는 것은
  자료가 적어도 틀리지 않는다 — 적으면 적다고 나오고, 문턱을 못 넘으면 아무
  것도 하지 않는다. 배관이 도는지부터 확인하고 모델은 자료가 쌓인 뒤에 건다.

왜 문턱을 두는가:
  한 사람이 한 번 저장한 것을 그 묶음 전체의 취향으로 삼으면, 통계가 아니라
  그 사람 한 명을 되풀이하는 것이 된다. 게다가 묶음에 한 명뿐인 칸은 그 값이
  곧 그 사람을 가리켜, 가려 둔 뜻이 사라진다.

쓰는 법:
  python -m eval.segment_stats --input sessions.jsonl --out eval/segment_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 이 파일 형식의 판. 읽는 쪽이 보고 갈라 읽는다.
STATS_SCHEMA_VERSION = 1

# 한 칸이 이 횟수만큼 보이지 않았으면 쓰지 않는다.
#
# 낮게 잡으면 한두 번의 우연이 취향으로 굳고, 높게 잡으면 지금 자료로는
# 아무 칸도 못 넘는다. 배관을 확인할 수 있는 선에서 가장 높게 잡았다.
DEFAULT_MIN_SUPPORT = 3


def segment_key(user: dict) -> str:
    """성향을 묶는 열쇠. 모르는 칸은 모른다고 적는다.

    테마까지 넣지 않는 이유: 칸이 잘게 쪼개져 어느 칸도 문턱을 못 넘는다.
    연령대와 성별만으로도 자료가 쌓이면 갈래가 생긴다.
    """
    age = (user or {}).get("age_band") or "unknown"
    gender = (user or {}).get("gender") or "unknown"
    return f"{age}|{gender}"


def build(rows: list[dict], min_support: int) -> dict:
    """세션 목록에서 묶음별 장소 통계를 만든다."""
    shown: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    saved: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sessions_per_segment: dict[str, int] = defaultdict(int)

    used = 0
    for row in rows:
        # 학습에 못 쓰는 세션은 세지 않는다. 직접 고른 동선이나 이을 정답이
        # 없는 것을 섞으면 비율이 뜻을 잃는다.
        if not row.get("l1_eligible"):
            continue
        key = segment_key(row.get("user_segment") or {})
        sessions_per_segment[key] += 1
        used += 1
        for cand in row.get("candidates") or []:
            cid = cand.get("content_id")
            if not cid:
                continue
            shown[key][cid] += 1
            if cand.get("saved"):
                saved[key][cid] += 1

    segments: dict[str, dict] = {}
    dropped = 0
    for key, counts in shown.items():
        places = {}
        for cid, n in counts.items():
            if n < min_support:
                dropped += 1
                continue
            places[cid] = {
                "shown": n,
                "saved": saved[key].get(cid, 0),
                "rate": round(saved[key].get(cid, 0) / n, 4),
            }
        if places:
            segments[key] = {
                "sessions": sessions_per_segment[key],
                "places": places,
            }

    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "min_support": min_support,
        "sessions_used": used,
        "segments": segments,
        # 문턱을 못 넘어 버린 칸 수. 0 이 아닌데 segments 가 비면 "자료가
        # 모자라다" 는 뜻이고, 둘 다 0 이면 "쓸 세션이 없었다" 는 뜻이다.
        "dropped_below_support": dropped,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="세션 JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"입력이 없다: {path}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    stats = build(rows, args.min_support)
    Path(args.out).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"세션 {len(rows)}건 중 쓸 수 있는 것 {stats['sessions_used']}건")
    print(f"묶음 {len(stats['segments'])}개, 문턱 미달로 버린 칸 {stats['dropped_below_support']}개")
    print(f"→ {args.out}")
    if not stats["segments"]:
        # 빈 통계가 나오는 것이 지금은 정상이다. 조용히 두면 사람이 고장으로
        # 오해하므로 이유를 함께 남긴다.
        print("쓸 수 있는 묶음이 없다 — 자료가 문턱에 못 미친다는 뜻이다",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
