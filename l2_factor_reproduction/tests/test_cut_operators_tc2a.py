"""TC-2A contract / operator tests. Synthetic data only. No live DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.aggregators import (
    agg_temporal_center,
    agg_temporal_dispersion,
    agg_temporal_gap,
    agg_tc_minus,
    agg_tc_plus,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.apply import (
    apply_one_recipe,
    apply_tc2a_recipes,
    attach_helper_columns,
    attach_state_masks,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    PRODUCTION_EXECUTION_CONTRACT,
    RESCUE_CORE_GATES,
    time_segment,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.loaders import (
    build_tc2a_panel,
    ch_ssl2_minute_sql_tc2a,
    month_windows,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc2a import (
    classify_descendant,
    classify_timing,
    write_frozen_contract,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc2a_config import (
    TC2A_MAX_DESCENDANTS_PER_PARENT,
    TC2A_NEGATIVE_CONTROL,
    TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION,
    TC2A_PARENTS,
    TC2A_POSITIVE_CONTROL,
    TC2A_RECIPES,
    TC2A_TARGET_CANDIDATE_RANGE,
    assert_tc2a_budget,
    parent_by_name,
    recipes_for_parent,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    AUCTION_MKEY,
    continuous_mkey_grid,
    mkey_from_minute_index,
    time_mask,
)
from l2_factor_reproduction.l2_ai_stock_selection.qualification import (
    CORE_ABS_HL_SHARPE,
    CORE_ABS_IC,
    CORE_ABS_MONO,
)


def test_negative_control_frozen_before_generation():
    assert TC2A_NEGATIVE_CONTROL == "net_buy_ratio"
    assert TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION is True
    assert TC2A_POSITIVE_CONTROL == "signed_amount_impact"
    names = [str(p["parent_factor"]) for p in TC2A_PARENTS]
    assert TC2A_NEGATIVE_CONTROL in names
    assert len(recipes_for_parent(TC2A_NEGATIVE_CONTROL)) >= 1


def test_budget_and_no_cartesian():
    assert_tc2a_budget()
    n = len(TC2A_RECIPES)
    lo, hi = TC2A_TARGET_CANDIDATE_RANGE
    assert lo <= n <= hi
    from collections import Counter

    counts = Counter(str(r["parent_factor"]) for r in TC2A_RECIPES)
    assert max(counts.values()) <= TC2A_MAX_DESCENDANTS_PER_PARENT
    assert len(TC2A_PARENTS) == 12


def test_parent_types_and_derived_semantics():
    types = {str(p["parent_factor"]): str(p["parent_type"]) for p in TC2A_PARENTS}
    assert types["obi_l5_mean"] == "LEVEL_PARENT"
    assert types["closing_30m_return"] == "PATH_PARENT"
    assert types["closing_obi_l5"] == "PATH_PARENT"
    assert types["vwap_close_deviation"] == "DERIVED_TRANSFORM_PARENT"
    assert types["close_location_value"] == "DERIVED_TRANSFORM_PARENT"
    assert types["return_per_amount"] == "DERIVED_TRANSFORM_PARENT"
    assert types["impact_asymmetry"] == "DERIVED_TRANSFORM_PARENT"
    derived = [p for p in TC2A_PARENTS if p["parent_type"] == "DERIVED_TRANSFORM_PARENT"]
    for p in derived:
        assert "reapply" in str(p["cut_level"])
        assert "cut(" not in str(p["parent_transform"])
    path_parents = [p for p in TC2A_PARENTS if p["parent_type"] == "PATH_PARENT"]
    for p in path_parents:
        assert "do_not_recut_parent" in str(p["cut_level"])
        recs = recipes_for_parent(str(p["parent_factor"]))
        if p["parent_factor"] == "closing_30m_return":
            assert not any(
                r["cut_type"] == "time" and str(r["cut_name"]).lower() == "close" for r in recs
            )
        if p["parent_factor"] == "afternoon_return":
            assert not any(
                r["cut_type"] == "time" and str(r["cut_name"]).lower() == "afternoon" for r in recs
            )
        if p["parent_factor"] == "closing_obi_l5":
            assert not any(
                r["cut_type"] == "time" and str(r["cut_name"]).lower() == "close" for r in recs
            )


def test_every_recipe_has_hypothesis_and_v2v_contract():
    for rec in TC2A_RECIPES:
        assert str(rec.get("reason") or "").strip()
        assert rec.get("candidate_name")
    assert PRODUCTION_EXECUTION_CONTRACT == "EXEC_V2V_TPLUS1_V1"


def test_rescue_core_gates_match_eq1():
    assert RESCUE_CORE_GATES["abs_rank_ic"] == CORE_ABS_IC
    assert RESCUE_CORE_GATES["hl_sharpe"] == CORE_ABS_HL_SHARPE
    assert RESCUE_CORE_GATES["monotonicity"] == CORE_ABS_MONO


def test_common_close_is_robustness_not_a_recipe():
    spec = time_segment("COMMON_CLOSE")
    assert spec["robustness_only"] is True
    assert int(spec["mkey_end"]) == 896
    names = [str(r["cut_name"]).lower() for r in TC2A_RECIPES]
    assert "common_close" not in names


def test_temporal_gap_signed_and_unsigned():
    t = np.arange(10, dtype=float)
    x = np.array([1, 1, 1, 0, 0, 0, 0, -1, -1, -1], dtype=float)
    mask = np.ones(10, dtype=bool)
    plus = agg_tc_plus(x, mask, t)
    minus = agg_tc_minus(x, mask, t)
    gap = agg_temporal_gap(x, mask, t)
    assert plus < minus
    assert gap == pytest.approx(minus - plus)
    # nonnegative: abs weighting, no forced +/- split
    spread = np.abs(x)
    center = agg_temporal_center(spread, mask, t, signed=False, weight="abs")
    assert np.isfinite(center)
    disp = agg_temporal_dispersion(spread, mask, t, signed=False, weight="abs")
    assert disp >= 0


def _synthetic_ohlc_panel() -> pd.DataFrame:
    keys = list(continuous_mkey_grid())
    rows = []
    day = pd.Timestamp("2024-06-03")
    for mkey in keys:
        if 570 <= mkey <= 689:
            minute_index = mkey - 570
        else:
            minute_index = 120 + (mkey - 780)
        close_m = bool(time_mask([mkey], "CLOSE")[0])
        open_m = bool(time_mask([mkey], "OPEN")[0])
        close = 10.0 + (0.5 if close_m else 0.0) + (0.2 if open_m else 0.0)
        high = close + 0.1
        low = close - 0.1
        rows.append(
            {
                "symbol": "600000.SH",
                "TradeDate": day,
                "mkey": int(mkey),
                "minute_index": int(minute_index),
                "Close": close,
                "Open": close,
                "High": high,
                "Low": low,
                "Amount": 1.0e6 if close_m else 5.0e5,
                "Volume": 1.0e5 if close_m else 5.0e4,
                "amount": 1.0e6 if close_m else 5.0e5,
                "Active_buy_amount": 8.0e5 if close_m else 2.0e5,
                "Active_sell_amount": 2.0e5 if close_m else 3.0e5,
                "minute_return": 0.001 if close_m else -0.0004,
                "net_active_flow": 6.0e5 if close_m else -1.0e5,
                "obi_5": -0.2 if close_m else 0.3,
                "relative_spread": 0.002 if close_m else 0.001,
                "total_depth_l5": 1.0e5,
                "microprice_deviation": 0.0004 if close_m else -0.0001,
                "large_order_amount": 0.4 if close_m else 0.05,
                "large_order_pressure": 0.2 if close_m else -0.05,
                "abs_minute_return": 0.001 if close_m else 0.0004,
            }
        )
    return pd.DataFrame(rows)


def test_derived_reapply_not_identity_cut():
    panel = attach_state_masks(attach_helper_columns(_synthetic_ohlc_panel()))
    name, series, extra = apply_one_recipe(
        panel,
        {
            "parent_factor": "vwap_close_deviation",
            "candidate_name": "vwap_close_deviation__time_close_reapply",
            "base_primitive": "vwap_close_deviation",
            "cut_type": "derived",
            "cut_name": "close",
            "derived_op": "vwap_close_deviation",
            "aggregation": "reapply",
            "reason": "test",
        },
    )
    assert extra["status"] == "OK"
    assert series.notna().any()
    name2, series2, extra2 = apply_one_recipe(
        panel,
        {
            "candidate_name": "close_location_value__time_close_reapply",
            "base_primitive": "close_location_value",
            "cut_type": "derived",
            "cut_name": "close",
            "derived_op": "close_location_value",
            "aggregation": "reapply",
            "reason": "test",
        },
    )
    assert extra2["status"] == "OK"
    clv = float(series2.dropna().iloc[0])
    assert -1.0 - 1e-8 <= clv <= 1.0 + 1e-8


def test_apply_tc2a_recipes_no_auction_and_coverage():
    panel = _synthetic_ohlc_panel()
    wide, metas = apply_tc2a_recipes(panel, TC2A_RECIPES)
    names = [c for c in wide.columns if c not in ("TradeDate", "symbol")]
    assert len(names) == len(TC2A_RECIPES)
    assert all(m.get("contains_close_auction") in (False, 0) for m in metas)
    # CLOSE recipes record source-window fields
    close_metas = [m for m in metas if str(m.get("cut_name")).lower() == "close"]
    assert close_metas
    assert any(m.get("effective_close_start") == "14:30:00" for m in close_metas)
    assert any(m.get("common_close_end") == "14:56:00" for m in close_metas)


def test_large_order_zero_denom_stays_missing():
    ddb = _synthetic_ohlc_panel()
    ddb["avg_trade_size"] = 1.0
    ssl2 = ddb[["symbol", "TradeDate", "minute_index", "obi_5", "relative_spread", "total_depth_l5"]].copy()
    ssl2["microprice_deviation"] = 0.0
    tick = ddb[["symbol", "TradeDate", "minute_index"]].copy()
    tick["large_order_amount"] = np.nan
    tick["large_buy_amount"] = np.nan
    tick["large_sell_amount"] = np.nan
    tick["tick_amount"] = 0.0
    meta = {"requires_ch_tick": False}
    panel = build_tc2a_panel(ddb, ssl2, tick, meta)
    assert panel["large_order_pressure"].isna().all()
    assert meta["zero_activity_filled_with_zero"] is False


def test_classify_requires_improvement_not_pretty_number():
    parent = {"rank_ic_mean": 0.03, "hl_sharpe": 3.5, "monotonicity": 0.8, "mutual_information": 0.02}
    child = {
        "rank_ic_mean": 0.031,
        "hl_sharpe": 3.6,
        "monotonicity": 0.81,
        "mutual_information": 0.021,
        "coverage": 0.9,
        "sign_consistency": 0.7,
        "n_ic_days": 200,
        "one_period_dominated": False,
    }
    # almost identical to a strong parent → redundant, not rescued
    status = classify_descendant(
        child=child,
        parent=parent,
        parent_type_q="NONLINEAR_STRUCTURAL_RESCUE",
        corr_parent=0.99,
        corr_core=0.2,
    )
    assert status in ("REDUNDANT_RESCUE", "FAILED_RESCUE")
    weak_parent = {"rank_ic_mean": 0.004, "hl_sharpe": 0.2, "monotonicity": 0.1, "mutual_information": 0.002}
    strong = dict(child)
    strong["rank_ic_mean"] = 0.04
    strong["hl_sharpe"] = 4.0
    strong["monotonicity"] = 0.85
    status2 = classify_descendant(
        child=strong,
        parent=weak_parent,
        parent_type_q="NONLINEAR_STRUCTURAL_RESCUE",
        corr_parent=0.3,
        corr_core=0.2,
    )
    assert status2 == "RESCUED_CORE"
    nc = classify_descendant(
        child=strong,
        parent=weak_parent,
        parent_type_q="NEGATIVE_CONTROL",
        corr_parent=0.1,
        corr_core=0.1,
    )
    assert nc == "NEGATIVE_CONTROL_NOT_PROMOTED"


def test_timing_too_fast_vs_executable():
    rows = pd.DataFrame(
        {
            "rank_ic_mean": [0.002, 0.003, 0.001],
            "legacy_rank_ic": [0.03, 0.004, 0.002],
        }
    )
    assert classify_timing("obi_l5_mean", rows) == "TIMING_LOCALIZED_TOO_FAST"
    rows2 = pd.DataFrame(
        {
            "rank_ic_mean": [0.02, 0.003, 0.002],
            "legacy_rank_ic": [0.021, 0.004, 0.002],
        }
    )
    assert classify_timing("obi_l5_mean", rows2) == "TIMING_LOCALIZED_EXECUTABLE"


def test_month_windows_cover_2023_2024():
    months = month_windows("2023-01-01", "2024-12-31")
    assert len(months) == 24
    assert months[0][0] == pd.Timestamp("2023-01-01")
    assert months[-1][1] == pd.Timestamp("2024-12-31")


def test_freeze_writes_contract(tmp_path):
    write_frozen_contract(tmp_path)
    contract = pd.read_csv(tmp_path / "tc2a_parent_contract.csv")
    assert list(contract["parent_factor"]) == [p["parent_factor"] for p in TC2A_PARENTS]
    assert (tmp_path / "tc2a_freeze.json").read_text().find("net_buy_ratio") >= 0
    recipes = pd.read_csv(tmp_path / "tc2a_recipes.csv")
    assert len(recipes) == len(TC2A_RECIPES)
    assert "underlying_primitive" in recipes.columns
    assert "parent_transform" in recipes.columns
    assert "cut_level" in recipes.columns


def test_ssl2_tc2a_sql_no_nested_argmax_mid():
    sql = ch_ssl2_minute_sql_tc2a(
        table="SSE_AL_SSL2_EXG",
        exchange_suffix=".SH",
        exchange="SSE",
        start="2023-01-03",
        end="2023-01-03",
    )
    assert "argMax(mid_price" not in sql
    assert "microprice_deviation" in sql
    assert "toUInt8(toHour(ExchTime) = 15) = 0" in sql
    assert "minute_index < 240" in sql
