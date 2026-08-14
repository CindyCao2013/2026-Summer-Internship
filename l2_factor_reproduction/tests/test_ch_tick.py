"""Unit tests for ClickHouse tick SQL and narrow-factor arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python import ch_tick  # noqa: E402


class _FakeClient:
    def __init__(self) -> None:
        self.queries = []
        self.closed = False

    def query_df(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        return pd.DataFrame()

    def close(self) -> None:
        self.closed = True


def test_order_size_sql_filters_every_date_and_identifies_szse_trades(monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(ch_tick, "_get_ch_client", lambda: fake)

    out = ch_tick.fetch_tick_agg_by_date_range(
        "2024-01-01",
        "2024-03-31",
        symbols=["600000.SH", "000001.SZ"],
    )

    assert out.empty
    assert fake.closed
    assert len(fake.queries) == 2
    for sql in fake.queries:
        assert "toHour(ExchTime)" in sql
        assert "toMinute(ExchTime) >= 30" in sql
        assert "toSecond(ExchTime) = 0" in sql
        assert "amt > 40000.0 AND amt <= 200000.0" in sql

    sse_sql = next(sql for sql in fake.queries if "SSE_AL_TICK_EXG" in sql)
    szse_sql = next(sql for sql in fake.queries if "SZSE_AL_TICK_EXG" in sql)
    assert "Type = 'T'" in sse_sql
    assert "Type = '011'" in szse_sql
    assert "BidOrderNo > 0 AND AskOrderNo > 0" in szse_sql


def test_bucketed_sql_uses_same_session_and_trade_filters(monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(ch_tick, "_get_ch_client", lambda: fake)

    out = ch_tick.fetch_tick_bucketed(
        "2024-01-01",
        "2024-03-31",
        boundaries=[40_000, 200_000],
        symbols=["600000.SH", "000001.SZ"],
    )

    assert list(out.columns) == [
        "symbol",
        "TradeDate",
        "TotalAmount",
        "cum_40000",
        "cum_200000",
    ]
    for sql in fake.queries:
        assert "toHour(ExchTime)" in sql
    szse_sql = next(sql for sql in fake.queries if "SZSE_AL_TICK_EXG" in sql)
    assert "Type = '011'" in szse_sql
    assert "BidOrderNo > 0 AND AskOrderNo > 0" in szse_sql


def test_order_size_distribution_sql_emits_side_and_count_buckets(monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(ch_tick, "_get_ch_client", lambda: fake)

    out = ch_tick.fetch_order_size_distribution_daily(
        "2024-01-01",
        "2024-03-31",
        boundaries=[10_000, 200_000],
        symbols=["600000.SH", "000001.SZ"],
    )

    assert list(out.columns) == [
        "symbol",
        "TradeDate",
        "total_amt",
        "trade_cnt",
        "active_buy_amt",
        "active_sell_amt",
        "cum_amt_10000",
        "cum_cnt_10000",
        "buy_cum_amt_10000",
        "sell_cum_amt_10000",
        "cum_amt_200000",
        "cum_cnt_200000",
        "buy_cum_amt_200000",
        "sell_cum_amt_200000",
    ]
    assert fake.closed
    assert len(fake.queries) == 2
    for sql in fake.queries:
        assert "countIf(amt > 0 AND amt <=" in sql
        assert "active_buy_amt" in sql
        assert "active_sell_amt" in sql
        assert "toHour(ExchTime)" in sql

    sse_sql = next(sql for sql in fake.queries if "SSE_AL_TICK_EXG" in sql)
    szse_sql = next(sql for sql in fake.queries if "SZSE_AL_TICK_EXG" in sql)
    assert "BSFlag = 'B'" in sse_sql
    assert "BSFlag = 'S'" in sse_sql
    assert "startsWith(Symbol, '6')" in sse_sql
    assert "Type = '011'" in szse_sql
    assert "BidOrderNo > AskOrderNo" in szse_sql
    assert "BidOrderNo < AskOrderNo" in szse_sql
    assert "startsWith(Symbol, '000')" in szse_sql


def test_aggregate_wide_to_narrow_formula_and_schema() -> None:
    wide = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ", "000002.SZ"],
            "TradeDate": pd.to_datetime(["2024-01-02"] * 3),
            "TotalAmount": [100.0, 200.0, 0.0],
            "MediumAmount": [25.0, 100.0, 1.0],
            "SmallAmount": [10.0, 20.0, 0.0],
        }
    )

    out = ch_tick.aggregate_wide_to_narrow(wide, "mid_order_ratio")

    assert list(out.columns) == ["symbol", "tradetime", "factorname", "value"]
    assert out["factorname"].eq("mid_order_ratio").all()
    assert np.allclose(out["value"].to_numpy(), [0.25, 0.5])
    assert out["tradetime"].dt.strftime("%H:%M:%S").eq("09:30:00").all()

