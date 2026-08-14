"""TC-1 apply / availability / loader-contract tests. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.apply import (
    apply_tc1_recipes,
    availability_for_recipe,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CANDIDATE_POOL_CSV,
    PRODUCTION_EXECUTION_CONTRACT,
    TC1_RECIPES,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.loaders import (
    build_tc1_panel,
    ch_ssl2_minute_sql,
    ddb_minutes_sql,
    _tick_minute_sql,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    assert_candidate_pool_unchanged,
    candidate_name,
    snapshot_candidate_pool,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc1 import diagnose_wide
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    AUCTION_MKEY,
    continuous_mkey_grid,
    mkey_from_minute_index,
    time_mask,
)


def _minute_index_from_mkey(mkeys) -> np.ndarray:
    arr = np.asarray(mkeys, dtype=np.int32)
    out = np.full(arr.shape, 240, dtype=np.int32)
    am = (arr >= 570) & (arr <= 689)
    pm = (arr >= 780) & (arr <= 899)
    out[am] = arr[am] - 570
    out[pm] = 120 + (arr[pm] - 780)
    return out


def _synthetic_panel(n_symbols=2, n_days=2, include_auction=True) -> pd.DataFrame:
    keys = list(continuous_mkey_grid())
    if include_auction:
        keys = keys + [AUCTION_MKEY]
    rows = []
    for di in range(n_days):
        day = pd.Timestamp("2024-06-03") + pd.Timedelta(days=di)
        for si in range(n_symbols):
            symbol = "60000{}.SH".format(si)
            mkeys = np.asarray(keys, dtype=np.int32)
            idx = _minute_index_from_mkey(mkeys)
            close_m = time_mask(mkeys, "CLOSE")
            open_m = time_mask(mkeys, "OPEN")
            auction = mkeys == AUCTION_MKEY
            flow = np.zeros(len(mkeys), dtype=float)
            flow[open_m] = 1.0 + si
            flow[close_m] = 3.0 + si
            flow[auction] = 1.0e9
            ret = np.where(close_m, 0.001 * (si + 1), np.where(open_m, -0.0005, 0.0001))
            spread = np.where(close_m, 0.002, 0.001)
            depth = np.where(close_m, 1.0e5, 2.0e5)
            amount = np.where(open_m, 8.0e5, 4.0e5)
            large = np.where(close_m, 3.0e5, 5.0e4)
            large[auction] = 9.0e9
            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "TradeDate": day,
                    "mkey": mkeys,
                    "minute_index": idx,
                    "net_active_flow": flow,
                    "cancel_imbalance": flow * 0.5,
                    "minute_return": ret,
                    "obi_5": np.where(open_m, 0.2, np.where(close_m, -0.1, 0.0)),
                    "relative_spread": spread,
                    "total_depth_l5": depth,
                    "amount": amount,
                    "large_order_amount": large,
                    "avg_trade_size": large / 2.0,
                    "Close": 10.0 + np.arange(len(mkeys)) * 0.001,
                }
            )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_tc1_recipe_count_and_names():
    assert len(TC1_RECIPES) == 36
    names = [
        candidate_name(
            r["base_primitive"],
            r["cut_type"],
            r["cut_name"],
            aggregation=str(r.get("aggregation") or ""),
            contrast_operator=str(r.get("contrast_operator") or ""),
        )
        for r in TC1_RECIPES
    ]
    assert len(set(names)) == 36
    counts = pd.Series([r["base_primitive"] for r in TC1_RECIPES]).value_counts()
    assert set(counts.index) == {
        "net_active_flow",
        "obi_5",
        "large_order_amount",
        "minute_return",
        "relative_spread",
        "cancel_imbalance",
    }
    assert int(counts.min()) == 6
    assert int(counts.max()) == 6


def test_apply_generates_36_and_excludes_auction():
    panel = _synthetic_panel()
    assert int((panel["mkey"] == AUCTION_MKEY).sum()) > 0
    wide, metas = apply_tc1_recipes(panel, TC1_RECIPES)
    value_cols = [c for c in wide.columns if c not in ("TradeDate", "symbol")]
    assert len(value_cols) == 36
    assert len(metas) == 36
    close_flow = "net_active_flow__time_close__sum"
    assert close_flow in wide.columns
    # OPEN=1, CLOSE=3, 30 bars, symbol 0 -> 90. Auction 1e9 must not enter.
    s0 = wide.loc[wide["symbol"] == "600000.SH", close_flow]
    assert np.allclose(s0.to_numpy(dtype=float), 90.0)
    for meta in metas:
        if str(meta["cut_type"]) == "time" and str(meta["cut_name"]).lower() in (
            "close",
            "full",
        ):
            assert meta["contains_close_auction"] is False
            assert meta["contains_1456_1500"] is True
            assert meta["close_auction_misuse"] is False
            assert str(meta["factor_available_after"]) >= "15:00:00"


def test_close_minus_open_contrast_math():
    panel = _synthetic_panel(n_symbols=1, n_days=1, include_auction=True)
    rec = {
        "base_primitive": "net_active_flow",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "sum",
        "reason": "path",
    }
    wide, metas = apply_tc1_recipes(panel, [rec])
    name = "net_active_flow__contrast_close_minus_open"
    # CLOSE sum 90 - OPEN sum 30 = 60
    assert float(wide[name].iloc[0]) == pytest.approx(60.0)
    assert metas[0]["contains_close_auction"] is False


def test_availability_close_full_contract():
    close = availability_for_recipe(
        {"base_primitive": "obi_5", "cut_type": "time", "cut_name": "close", "aggregation": "mean"}
    )
    assert close["contains_close_auction"] is False
    assert close["contains_1456_1500"] is True
    assert close["uses_last_5min"] is True
    assert close["close_auction_misuse"] is False
    assert close["factor_available_after"] == "15:00:00"
    assert close["execution_contract_compatible"] == PRODUCTION_EXECUTION_CONTRACT
    full = availability_for_recipe(
        {"base_primitive": "obi_5", "cut_type": "time", "cut_name": "full", "aggregation": "mean"}
    )
    assert full["contains_close_auction"] is False
    assert full["contains_1456_1500"] is True
    rec = {
        "base_primitive": "obi_5",
        "cut_type": "contrast",
        "cut_name": "highvol_minus_lowvol",
        "contrast_operator": "DIFF",
        "aggregation": "mean",
    }
    hv = availability_for_recipe(rec)
    assert hv["contains_close_auction"] is False
    assert hv["contains_1456_1500"] is True
    assert hv["uses_last_5min"] is True
    open_ = availability_for_recipe(
        {
            "base_primitive": "obi_5",
            "cut_type": "time",
            "cut_name": "open",
            "aggregation": "mean",
        }
    )
    assert open_["factor_available_after"] == "10:00:00"
    assert open_["contains_1456_1500"] is False


def test_proxy_uses_relative_avg_trade_size():
    ddb = _synthetic_panel(n_symbols=1, n_days=1, include_auction=False)
    ddb = ddb.drop(columns=["large_order_amount", "obi_5", "relative_spread", "total_depth_l5"])
    ssl2 = pd.DataFrame()
    tick = pd.DataFrame()
    meta = {
        "source_used": "ddb_avg_trade_size_proxy",
        "requires_ch_tick": True,
        "proxy_source": "DDB_AvgTradeSize",
    }
    panel = build_tc1_panel(ddb, ssl2, tick, meta)
    med = float(panel["avg_trade_size"].median())
    expected = panel["avg_trade_size"] / (abs(med) + 1e-12)
    assert np.allclose(
        panel["large_order_amount"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
    )
    assert int((panel["minute_index"] >= 240).sum()) == 0
    assert int((panel["mkey"] == AUCTION_MKEY).sum()) == 0
    assert meta["large_order_definition"].startswith("ddb_avg_trade_size")


def test_large_order_is_share_not_cny_threshold():
    from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.normalize import (
        share_ratio,
    )
    from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.event_cuts import (
        large_trade_event_mask,
    )

    part = pd.Series([10.0, 20.0, 0.0])
    full = pd.Series([100.0, 100.0, 100.0])
    share = share_ratio(part, full)
    assert list(np.round(share, 6)) == [0.1, 0.2, 0.0]
    # All prints << 200k CNY still produce a relative large-trade event.
    tiny = np.array([1.0, 1.0, 1.0, 3.0, 1.0])
    mask = large_trade_event_mask(tiny)
    assert mask.sum() >= 1
    assert not np.any(tiny[mask] > 200000)


def test_loader_sql_excludes_auction():
    sql = ddb_minutes_sql("2024-06-01", "2024-06-30")
    assert "14:59:00" in sql
    assert "15:00:00" not in sql.replace("11:29:00", "")
    ch_sql = ch_ssl2_minute_sql(
        table="LOCAL_SSE_AL_SSL2_EXG",
        exchange_suffix=".SH",
        exchange="SSE",
        start="2024-06-01",
        end="2024-06-07",
    )
    assert "toUInt8(toHour(ExchTime) = 15) = 0" in ch_sql
    assert "minute_index < 240" in ch_sql
    tick_sql = _tick_minute_sql("sse", "2024-06-03", "2024-06-04")
    assert "minute_index < 240" in tick_sql
    assert "toHour(ExchTime) = 15" in tick_sql
    assert "200000" not in tick_sql
    assert "quantileTDigest" in tick_sql


def test_mkey_mapping_auction_is_240():
    idx = np.array([0, 119, 120, 239, 240], dtype=np.int32)
    mkeys = mkey_from_minute_index(idx)
    assert list(mkeys) == [570, 689, 780, 899, 900]


def test_diagnostics_flags_and_no_pool_mutation():
    before = snapshot_candidate_pool()
    n = 24
    vals = np.arange(n, dtype=float)
    wide = pd.DataFrame(
        {
            "TradeDate": pd.date_range("2024-06-03", periods=n, freq="D"),
            "symbol": ["600000.SH"] * n,
            "a": vals,
            "b": vals.copy(),
            "c": np.full(n, np.nan),
            "d": np.ones(n),
        }
    )
    diag = diagnose_wide(wide)
    by_name = diag.set_index("candidate_name")["status"].to_dict()
    assert by_name["c"] == "SKIPPED_NO_DATA"
    assert by_name["d"] == "ZERO_VARIANCE"
    assert by_name["b"] == "DUPLICATE"
    assert CANDIDATE_POOL_CSV.exists()
    assert_candidate_pool_unchanged(before)
