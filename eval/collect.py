"""케이스를 실제로 태워 결과를 모은다.

채점기(`eval.run`)는 이미 모아 둔 결과를 읽어 점수만 낸다. 그 결과를 만드는
쪽이 없어서 기준선을 한 번도 세우지 못했다 — 이 파일이 그 자리다.

무엇을 하는가:
  1. 케이스마다 요청을 지어 agent 에 접수시킨다
  2. 완료 대기열에서 그 잡의 결과를 거둔다
  3. `<케이스 이름>.json` 으로 결과를, `training/<케이스 이름>.json` 으로
     학습 신호를 떨군다

왜 대기열에서 거두나:
  agent 에는 "그 잡 어떻게 됐나" 를 묻는 길이 없다. 결과는 완료 대기열로만
  나가므로 그것을 직접 듣는다. 서비스가 쓰는 이름과 다른 이름으로 붙는다 —
  같은 이름으로 붙으면 돌아가는 서버가 받아야 할 것을 이쪽이 가로챈다.

왜 케이스에 날짜가 없나:
  케이스는 "도보 하루" 처럼 조건만 적는다. 날짜를 박아 두면 그날이 지나면
  과거 여행이 되어 결과가 달라진다. 여기서 오늘 기준으로 지어 넣는다.

쓰는 법:
  python -m eval.collect --out out/gemini
  python -m eval.collect --out out/gemini --cases eval/cases --wait 180

주의: 이것은 실제로 모델과 발급처를 부른다. 케이스 수만큼 하루 한도를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

# 케이스에 적힌 활동 시간대의 기본값. 케이스가 따로 적었으면 그것을 쓴다.
DEFAULT_TIME_START = "10:00:00"
DEFAULT_TIME_END = "20:00:00"

# 오늘로부터 며칠 뒤를 여행일로 삼는가.
#
# 오늘이나 내일로 잡으면 날씨·영업시간 같은 조건이 결과를 흔든다. 너무 멀면
# 예보가 없어 저하 경로로 빠진다. 그 사이를 고른다.
TRIP_OFFSET_DAYS = 14


def build_request(case: dict) -> dict:
    """케이스의 조건으로 agent 가 받는 요청을 짓는다."""
    req = case.get("request") or {}
    expect = case.get("expect") or {}
    days = int(req.get("days") or 1)
    start = date.today() + timedelta(days=TRIP_OFFSET_DAYS)
    end = start + timedelta(days=days - 1)

    def as_time(value: str | None, fallback: str) -> str:
        if not value:
            return fallback
        # 케이스는 "10:00" 처럼 적는다. 받는 쪽은 초까지 요구한다.
        return value if value.count(":") == 2 else f"{value}:00"

    out: dict = {
        "date": {
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "time_start": as_time(expect.get("time_start"), DEFAULT_TIME_START),
            "time_end": as_time(expect.get("time_end"), DEFAULT_TIME_END),
        },
        "province": req["province"],
        "city": req.get("city", ""),
        "stage": "init",
    }
    if req.get("mobility"):
        out["mobility"] = req["mobility"]
    if req.get("theme"):
        out["theme"] = req["theme"]
    if req.get("budget") is not None:
        out["budget"] = req["budget"]
    return out


def submit(base_url: str, body: dict, token: str | None) -> str | None:
    """요청을 접수시키고 잡 식별자를 받는다. 실패하면 None.

    agent 는 서비스끼리만 부르는 자리라 공유 표식을 요구한다. 평소에는 BFF 가
    붙여 주는 것이라 여기서도 같은 것을 붙여야 한다.
    """
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token
    req = urllib.request.Request(
        f"{base_url}/v1/recommend", data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r).get("job_id")
    except urllib.error.HTTPError as e:
        print(f"  접수 거절 {e.code}: {e.read()[:200]!r}", file=sys.stderr)
    except OSError as e:
        print(f"  접수 실패: {e}", file=sys.stderr)
    return None


def collect(redis_url: str, stream: str, group: str, consumer: str,
            wanted: dict[str, str], wait_seconds: int) -> dict[str, dict]:
    """완료 대기열에서 기다리던 잡들의 결과를 거둔다.

    wanted: 잡 식별자 → 케이스 이름.
    돌려주는 것: 케이스 이름 → 완료 페이로드(학습 신호 포함).

    다 못 거두어도 시한이 지나면 있는 것만 돌려준다. 한 건이 안 온다고
    나머지를 버릴 이유가 없다.
    """
    import redis  # 여기서만 쓴다. 채점기는 망을 타지 않는다.

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        client.xgroup_create(stream, group, id="$", mkstream=True)
    except redis.ResponseError:
        pass  # 이미 있다.

    out: dict[str, dict] = {}
    deadline = time.time() + wait_seconds
    while wanted and time.time() < deadline:
        try:
            batches = client.xreadgroup(
                group, consumer, {stream: ">"}, count=10, block=2000)
        except redis.RedisError as e:
            print(f"  대기열 읽기 실패: {e}", file=sys.stderr)
            break
        for _, entries in batches or []:
            for entry_id, fields in entries:
                job_id = fields.get("job_id")
                client.xack(stream, group, entry_id)
                case_id = wanted.pop(job_id, None)
                if case_id is None:
                    # 우리가 낸 것이 아니다. 확인만 하고 넘어간다.
                    continue
                out[case_id] = fields
                print(f"  거둠 {case_id}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과를 떨굴 디렉터리")
    ap.add_argument("--cases", default="eval/cases")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--redis-url", default="redis://127.0.0.1:16379/2")
    ap.add_argument("--stream", default="agent:jobs:done")
    ap.add_argument("--group", default="eval-collect")
    ap.add_argument("--consumer", default="eval-1")
    ap.add_argument("--wait", type=int, default=240,
                    help="결과를 기다리는 최대 시간(초)")
    ap.add_argument("--internal-token", default=os.environ.get("INTERNAL_SERVICE_TOKEN"),
                    help="서비스끼리 쓰는 공유 표식. 환경변수로 줘도 된다")
    args = ap.parse_args()

    case_dir = Path(args.cases)
    cases = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(case_dir.glob("*.json"))]
    if not cases:
        print(f"케이스가 없다: {case_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    (out_dir / "training").mkdir(parents=True, exist_ok=True)

    print(f"케이스 {len(cases)}건 접수")
    wanted: dict[str, str] = {}
    for case in cases:
        job_id = submit(args.base_url, build_request(case), args.internal_token)
        if job_id is None:
            continue
        wanted[job_id] = case["id"]
        print(f"  접수 {case['id']} → {job_id}")
        # 발급처 분당 한도에 걸리지 않게 벌린다. 몰아 보내면 뒤엣것이
        # 한도에 걸려 저하 경로로 빠지고, 그것이 점수로 잡힌다.
        time.sleep(2)

    if not wanted:
        print("접수된 것이 없다", file=sys.stderr)
        return 1

    print(f"결과 기다리는 중 (최대 {args.wait}초)")
    results = collect(args.redis_url, args.stream, args.group, args.consumer,
                      dict(wanted), args.wait)

    for case_id, fields in results.items():
        payload = fields.get("payload")
        if payload:
            (out_dir / f"{case_id}.json").write_text(payload, encoding="utf-8")
        training = fields.get("training")
        if training:
            (out_dir / "training" / f"{case_id}.json").write_text(
                training, encoding="utf-8")

    missing = sorted(set(wanted.values()) - set(results))
    print(f"\n거둠 {len(results)}건 / 접수 {len(wanted)}건 → {out_dir}")
    if missing:
        # 못 거둔 것을 조용히 넘기면 그만큼 점수가 좋아 보인다.
        print(f"못 거둠: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
