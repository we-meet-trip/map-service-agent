"""모아 둔 결과를 채점하고 기준선과 견준다.

쓰는 법:
  python -m eval.run --results <결과디렉터리> [--training <신호디렉터리>]
  python -m eval.run --results out/gemini --baseline eval/baseline.json

결과 디렉터리에는 케이스 하나당 파일 하나를 <case_id>.json 으로 둔다.
내용은 완료 페이로드(JobDonePayload) 그대로다. 신호 디렉터리를 함께 주면
같은 이름으로 학습 신호를 읽어 grounded 축을 채운다.

기준선을 주면 축마다 견주어, 나빠진 폭이 허용치를 넘을 때 0 이 아닌 값으로
끝난다. 그래야 사람이 표를 눈으로 대조하지 않아도 회귀가 드러난다.

여기에는 망을 타는 코드가 없다. 같은 입력이면 늘 같은 점수가 나온다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.scorer import score_case, summarize

# 기준선 대비 이만큼까지는 흔들림으로 본다. 케이스 수가 적어 한 건이
# 비율을 크게 움직이므로, 너무 좁게 잡으면 늘 빨간 불이 켜진다.
DEFAULT_TOLERANCE = 0.05

# 값이 클수록 나쁜 축. 견주는 방향이 반대다.
LOWER_IS_BETTER = {"invented_ratio"}


def load_cases(case_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(case_dir.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def compare(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """기준선보다 허용치 넘게 나빠진 축을 고른다."""
    regressions = []
    for axis, base in baseline.items():
        if axis == "cases" or base is None:
            continue
        now = current.get(axis)
        if now is None:
            regressions.append(f"{axis}: 이번에 재지 못했다(기준선 {base})")
            continue
        worse = (now - base) if axis in LOWER_IS_BETTER else (base - now)
        if worse > tolerance:
            regressions.append(f"{axis}: {base} → {now}")
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval/cases")
    parser.add_argument("--results", required=True)
    parser.add_argument("--training", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--write-baseline", action="store_true",
                        help="이번 결과를 새 기준선으로 적는다")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if not cases:
        print(f"✗ 케이스가 없다: {args.cases}")
        return 1

    results_dir = Path(args.results)
    training_dir = Path(args.training) if args.training else None

    scored = []
    missing = []
    for case in cases:
        cid = case["id"]
        payload = load_json(results_dir / f"{cid}.json")
        if payload is None:
            missing.append(cid)
            continue
        training = load_json(training_dir / f"{cid}.json") if training_dir else None
        scored.append(score_case(case, payload, training))

    if missing:
        # 빠진 케이스를 조용히 넘기면 통과율이 남은 것들로만 계산돼 좋아 보인다.
        print(f"✗ 결과가 없는 케이스 {len(missing)}건: {', '.join(missing[:5])}")
        return 1

    current = summarize(scored)
    print(json.dumps({"summary": current, "cases": scored},
                     ensure_ascii=False, indent=2))

    failed = [s["case_id"] for s in scored if (
        not s["schema"]["ok"] or not s["grounded"].get("grounded")
        or not s["route"].get("ok")
        or (s["rules"].get("applicable") and s["rules"]["violations"] > 0)
        or (s["timeline"].get("applicable") and not s["timeline"].get("ok"))
    )]
    if failed or not scored:
        print(f"\n✗ 기본 품질 기준 미달: {', '.join(failed) or '평가 케이스 없음'}")
        return 1

    if args.write_baseline and args.baseline:
        Path(args.baseline).write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"\n기준선을 적었다: {args.baseline}")
        return 0

    if args.baseline:
        baseline = load_json(Path(args.baseline))
        if baseline is None:
            print(f"\n기준선이 없다: {args.baseline} "
                  f"(--write-baseline 으로 만들 수 있다)")
            return 1
        regressions = compare(current, baseline, args.tolerance)
        if regressions:
            print(f"\n✗ 기준선 대비 나빠진 축 {len(regressions)}개 "
                  f"(허용 {args.tolerance}):")
            for r in regressions:
                print(f"    {r}")
            return 1
        print(f"\n✓ 기준선 대비 회귀 없음 (허용 {args.tolerance})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
