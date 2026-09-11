import asyncio
import math
from datetime import date

from app.nodes.agent_nodes import _daily_pops, plan_strategy
from tests.test_timeline import _request


def test_unknown_precipitation_is_not_zero_in_plan():
    result = asyncio.run(plan_strategy({'request': _request(days=3), 'weather': {
        'daily': [{'date': '2026-07-08', 'precipitation_prob': 0}],
    }}))
    assert [day['precipitation_prob'] for day in result['plan']['days']] == [None, None, 0]


def test_invalid_precipitation_and_wrong_dates_are_not_consumed():
    weather = {'daily': [
        {'date': '2026-07-06', 'precipitation_prob': math.nan},
        {'date': '2026-07-07', 'precipitation_prob': True},
        {'date': '2026-07-08', 'precipitation_prob': 101},
        {'date': '2026-07-09', 'precipitation_prob': 100},
    ]}
    assert _daily_pops(weather, date(2026, 7, 6), 3) == [None, None, None]
