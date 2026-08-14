"""Apply explicit cut recipes to a minute panel. No Cartesian search."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.aggregators import (
    apply_aggregator,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    EVENT_Q_DEFAULT,
    MIN_COVERAGE_OBS,
    PRODUCTION_EXECUTION_CONTRACT,
    RATIO_EPSILON,
    TC1_RECIPES,
    time_segment,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    candidate_name,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    AUCTION_MKEY,
    time_mask,
)

PRIMITIVE_COLUMN = {
    "net_active_flow": "net_active_flow",
    "obi_5": "obi_5",
    "large_order_amount": "large_order_amount",
    "large_order_pressure": "large_order_pressure",
    "minute_return": "minute_return",
    "relative_spread": "relative_spread",
    "cancel_imbalance": "cancel_imbalance",
    "amount": "amount",
    "abs_minute_return": "abs_minute_return",
    "total_depth_l5": "total_depth_l5",
    "microprice_deviation": "microprice_deviation",
}

CONTRAST_LEGS = {
    "close_minus_open": ("DIFF", "time", "CLOSE", "time", "OPEN"),
    "close_share_full": ("SHARE", "time", "CLOSE", "time", "FULL"),
    "highvol_minus_lowvol": ("DIFF", "state", "high_vol", "state", "low_vol"),
    "reversal_close_vs_open": ("REVERSAL", "time", "OPEN", "time", "CLOSE"),
    "highvol_over_full": ("RATIO", "state", "high_vol", "time", "FULL"),
    "afternoon_minus_morning": ("DIFF", "time", "AFTERNOON", "time", "MORNING"),
}


def _require_no_auction(mask: np.ndarray, mkeys: np.ndarray, where: str) -> None:
    if np.any(mask & (mkeys == AUCTION_MKEY)):
        raise RuntimeError("auction bars leaked into {}".format(where))


def attach_helper_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "abs_minute_return" not in out.columns:
        out["abs_minute_return"] = np.abs(out["minute_return"].to_numpy(dtype=float))
    mkeys = out["mkey"].to_numpy(dtype=np.int32)
    for name in ("OPEN", "MORNING", "AFTERNOON", "CLOSE", "FULL", "COMMON_CLOSE"):
        mask = time_mask(mkeys, name)
        _require_no_auction(mask, mkeys, "time." + name)
        out["mask_time_" + name.lower()] = mask
    out = _attach_fwd1(out)
    return out


def _attach_fwd1(frame: pd.DataFrame) -> pd.DataFrame:
    """Next-minute close return inside the same session. No cross-day leak."""
    if frame.empty or "Close" not in frame.columns:
        frame["fwd1"] = np.nan
        return frame
    out = frame.sort_values(["symbol", "TradeDate", "mkey"]).copy()
    g = out.groupby(["symbol", "TradeDate"], sort=False)
    nxt = g["Close"].shift(-1).to_numpy(dtype=float)
    nm = g["mkey"].shift(-1).to_numpy(dtype=float)
    close = out["Close"].to_numpy(dtype=float)
    mkey = out["mkey"].to_numpy(dtype=np.int32)
    consec = np.isfinite(nm) & (mkey + 1 == nm.astype(np.int32))
    same = ((mkey <= 689) & (nm <= 689)) | ((mkey >= 780) & (nm >= 780))
    ok = consec & same & (close > 0) & (nxt > 0)
    fwd = np.full(close.shape, np.nan)
    fwd[ok] = nxt[ok] / close[ok] - 1.0
    out["fwd1"] = fwd
    return out


def _within_day_high(frame: pd.DataFrame, col: str) -> pd.Series:
    med = frame.groupby(["symbol", "TradeDate"], sort=False)[col].transform("median")
    val = frame[col]
    return val.notna() & med.notna() & (val > med)


def attach_state_masks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "relative_spread" in out.columns:
        hi = _within_day_high(out, "relative_spread")
        out["mask_state_high_spread"] = hi.to_numpy()
        finite = out["relative_spread"].notna()
        out["mask_state_low_spread"] = (finite & ~hi).to_numpy()
    else:
        out["mask_state_high_spread"] = False
        out["mask_state_low_spread"] = False
    if "total_depth_l5" in out.columns:
        hi = _within_day_high(out, "total_depth_l5")
        out["mask_state_high_depth"] = hi.to_numpy()
        finite = out["total_depth_l5"].notna()
        out["mask_state_low_depth"] = (finite & ~hi).to_numpy()
    else:
        out["mask_state_low_depth"] = False
    if "abs_minute_return" in out.columns:
        hi = _within_day_high(out, "abs_minute_return")
        out["mask_state_high_vol"] = hi.to_numpy()
        finite = out["abs_minute_return"].notna()
        out["mask_state_low_vol"] = (finite & ~hi).to_numpy()
    else:
        out["mask_state_high_vol"] = False
        out["mask_state_low_vol"] = False
    ret = out["minute_return"] if "minute_return" in out.columns else pd.Series(np.nan, index=out.index)
    out["mask_state_price_up"] = (ret > 0).to_numpy()
    out["mask_state_price_down"] = (ret < 0).to_numpy()
    if "amount" in out.columns:
        hi = _within_day_high(out, "amount")
        out["mask_state_high_trade_intensity"] = hi.to_numpy()
    else:
        out["mask_state_high_trade_intensity"] = False
    if "large_order_amount" in out.columns:
        hi = _within_day_high(out, "large_order_amount")
        out["mask_state_large_order_dominated"] = hi.to_numpy()
    else:
        out["mask_state_large_order_dominated"] = False
    return out


def _rank_pct(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby(["symbol", "TradeDate"], sort=False)[col].rank(
        method="average", pct=True
    )


def _event_mask(frame: pd.DataFrame, event_name: str, value_col: str) -> np.ndarray:
    name = str(event_name).lower()
    if name == "top_q":
        return (_rank_pct(frame, value_col) >= (1.0 - EVENT_Q_DEFAULT)).fillna(False).to_numpy()
    if name == "large_trade":
        return (_rank_pct(frame, "large_order_amount") >= (1.0 - EVENT_Q_DEFAULT)).fillna(
            False
        ).to_numpy()
    if name == "liquidity_shock":
        spread_hi = _rank_pct(frame, "relative_spread") >= (1.0 - EVENT_Q_DEFAULT)
        depth_lo = _rank_pct(frame, "total_depth_l5") <= EVENT_Q_DEFAULT
        tmp = frame.loc[:, ["symbol", "TradeDate"]].copy()
        tmp["_abs"] = frame["minute_return"].abs()
        impact_hi = _rank_pct(tmp, "_abs") >= (1.0 - EVENT_Q_DEFAULT)
        return (spread_hi | depth_lo | impact_hi).fillna(False).to_numpy()
    raise KeyError("unknown TC-1 event {!r}".format(event_name))


def _mask_from_leg(panel: pd.DataFrame, kind: str, name: str) -> np.ndarray:
    kind = str(kind).lower()
    key = str(name).lower()
    if kind == "time":
        col = "mask_time_" + key
        return panel[col].to_numpy(dtype=bool)
    if kind == "state":
        col = "mask_state_" + key
        return panel[col].to_numpy(dtype=bool)
    raise KeyError("unknown mask leg {}/{}".format(kind, name))


def _universe(panel: pd.DataFrame) -> pd.MultiIndex:
    keys = panel[["TradeDate", "symbol"]].drop_duplicates()
    return pd.MultiIndex.from_frame(keys)


def _aggregate_masked(
    panel: pd.DataFrame,
    value_col: str,
    mask: np.ndarray,
    how: str,
    *,
    min_obs: Optional[int] = None,
) -> pd.Series:
    how = str(how).lower()
    universe = _universe(panel)
    if how == "event_share":
        tmp = panel.loc[:, ["TradeDate", "symbol"]].copy()
        tmp["m"] = mask.astype(float)
        tmp["v"] = np.isfinite(panel[value_col].to_numpy(dtype=float)).astype(float)
        g = tmp.groupby(["TradeDate", "symbol"], sort=False)
        num = g["m"].sum()
        den = g["v"].sum()
        out = num / den.replace(0, np.nan)
        return out.reindex(universe)
    work = panel.loc[mask, ["TradeDate", "symbol", value_col, "mkey"]].copy()
    if how == "persistence":
        if work.empty:
            return pd.Series(np.nan, index=universe)
        rows = []
        for key, sl in work.groupby(["TradeDate", "symbol"], sort=False):
            val = apply_aggregator(
                "persistence",
                sl[value_col].to_numpy(),
                np.ones(len(sl), dtype=bool),
                mkeys=sl["mkey"].to_numpy(),
            )
            rows.append((key[0], key[1], val))
        if not rows:
            return pd.Series(np.nan, index=universe)
        ser = pd.DataFrame(rows, columns=["TradeDate", "symbol", "v"]).set_index(
            ["TradeDate", "symbol"]
        )["v"]
        return ser.reindex(universe)
    if work.empty:
        return pd.Series(np.nan, index=universe)
    g = work.groupby(["TradeDate", "symbol"], sort=False)[value_col]
    counts = g.count()
    if how == "sum":
        out = g.sum()
        floor = 1 if min_obs is None else int(min_obs)
    elif how == "mean":
        out = g.mean()
        floor = MIN_COVERAGE_OBS if min_obs is None else int(min_obs)
    elif how == "std":
        out = g.std(ddof=0)
        floor = MIN_COVERAGE_OBS if min_obs is None else int(min_obs)
    elif how == "median":
        out = g.median()
        floor = MIN_COVERAGE_OBS if min_obs is None else int(min_obs)
    elif how == "last":
        out = g.last()
        floor = 1
    elif how == "max":
        out = g.max()
        floor = 1
    elif how == "min":
        out = g.min()
        floor = 1
    else:
        raise KeyError("unsupported aggregator {!r}".format(how))
    out = out.where(counts >= floor, np.nan)
    return out.reindex(universe)


def _contrast_series(a: pd.Series, b: pd.Series, op: str) -> Tuple[pd.Series, pd.Series]:
    a, b = a.align(b, join="outer")
    op = str(op).upper()
    if op == "DIFF":
        return a - b, pd.Series(False, index=a.index)
    if op == "ACCELERATION":
        return a - b, pd.Series(False, index=a.index)
    if op == "REVERSAL":
        early, late = a, b
        sign = np.sign(early.to_numpy(dtype=float))
        out = pd.Series(-sign * late.to_numpy(dtype=float), index=a.index)
        out = out.where(np.isfinite(early) & np.isfinite(late) & (early != 0))
        return out, pd.Series(False, index=a.index)
    if op in ("RATIO", "SHARE"):
        den = b.abs()
        zero = den.notna() & (den <= RATIO_EPSILON)
        out = a / (den + RATIO_EPSILON)
        out = out.where(b.notna() & a.notna() & ~zero, np.nan)
        return out, zero.fillna(False)
    if op == "NORMALIZED_DIFF":
        den = a.abs() + b.abs()
        zero = den.notna() & (den <= RATIO_EPSILON)
        out = (a - b) / (den + RATIO_EPSILON)
        out = out.where(a.notna() & b.notna() & ~zero, np.nan)
        return out, zero.fillna(False)
    raise KeyError("unknown contrast {}".format(op))


def _recipe_name(recipe: Mapping[str, object]) -> str:
    if recipe.get("candidate_name"):
        return str(recipe["candidate_name"])
    base = str(recipe["base_primitive"])
    return candidate_name(
        base,
        str(recipe["cut_type"]),
        str(recipe["cut_name"]),
        aggregation=str(recipe.get("aggregation") or ""),
        contrast_operator=str(recipe.get("contrast_operator") or ""),
    )


def _mask_by_cut_name(panel: pd.DataFrame, cut_name: str) -> np.ndarray:
    key = str(cut_name).lower()
    time_col = "mask_time_" + key
    if time_col in panel.columns:
        return panel[time_col].to_numpy(dtype=bool)
    state_col = "mask_state_" + key
    if state_col in panel.columns:
        return panel[state_col].to_numpy(dtype=bool)
    raise KeyError("no mask for cut_name {!r}".format(cut_name))


def _groupby_last_first_ohlc(panel: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    cols = ["TradeDate", "symbol"]
    for c in ("Close", "High", "Low", "Open", "Amount", "Volume", "amount"):
        if c in panel.columns:
            cols.append(c)
    work = panel.loc[mask, cols].copy()
    universe = _universe(panel)
    if work.empty:
        return pd.DataFrame(index=universe)
    g = work.groupby(["TradeDate", "symbol"], sort=False)
    out = pd.DataFrame(index=g.size().index)
    if "Close" in work.columns:
        out["last"] = g["Close"].last()
        out["first"] = g["Close"].first()
        out["high_c"] = g["Close"].max()
        out["low_c"] = g["Close"].min()
    else:
        out["last"] = np.nan
        out["first"] = np.nan
        out["high_c"] = np.nan
        out["low_c"] = np.nan
    if "High" in work.columns:
        out["high"] = g["High"].max()
    else:
        out["high"] = out["high_c"]
    if "Low" in work.columns:
        out["low"] = g["Low"].min()
    else:
        out["low"] = out["low_c"]
    amt_col = "Amount" if "Amount" in work.columns else ("amount" if "amount" in work.columns else None)
    if amt_col:
        out["amt"] = g[amt_col].sum()
    else:
        out["amt"] = np.nan
    if "Volume" in work.columns:
        out["vol"] = g["Volume"].sum()
        out["vwap"] = out["amt"] / out["vol"].replace(0, np.nan)
    else:
        out["vwap"] = np.nan
    return out.reindex(universe)


def _derived_vwap_close_deviation(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    o = _groupby_last_first_ohlc(panel, mask)
    den = o["vwap"].abs()
    out = (o["last"] - o["vwap"]) / o["vwap"]
    return out.where(den.notna() & (den > RATIO_EPSILON) & o["last"].notna())


def _derived_clv(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    o = _groupby_last_first_ohlc(panel, mask)
    den = o["high"] - o["low"]
    out = (2.0 * o["last"] - o["high"] - o["low"]) / den
    return out.where(den.notna() & (den.abs() > RATIO_EPSILON) & o["last"].notna())


def _derived_return_per_amount(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    o = _groupby_last_first_ohlc(panel, mask)
    ret = o["last"] / o["first"] - 1.0
    den = o["amt"].abs()
    out = ret / o["amt"]
    return out.where(
        den.notna()
        & (den > RATIO_EPSILON)
        & o["last"].notna()
        & o["first"].notna()
        & (o["first"] > 0)
    )


def _derived_net_buy_ratio(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    need = ["TradeDate", "symbol"]
    for c in ("Active_buy_amount", "Active_sell_amount", "Amount", "amount"):
        if c in panel.columns:
            need.append(c)
    work = panel.loc[mask, need].copy()
    universe = _universe(panel)
    if work.empty:
        return pd.Series(np.nan, index=universe)
    buy_c = "Active_buy_amount" if "Active_buy_amount" in work.columns else None
    sell_c = "Active_sell_amount" if "Active_sell_amount" in work.columns else None
    amt_c = "Amount" if "Amount" in work.columns else ("amount" if "amount" in work.columns else None)
    g = work.groupby(["TradeDate", "symbol"], sort=False)
    buy = g[buy_c].sum() if buy_c else pd.Series(np.nan, index=g.size().index)
    sell = g[sell_c].sum() if sell_c else pd.Series(np.nan, index=g.size().index)
    tot = g[amt_c].sum() if amt_c else pd.Series(np.nan, index=g.size().index)
    den = tot.abs()
    out = (buy - sell) / tot
    out = out.where(den.notna() & (den > RATIO_EPSILON))
    return out.reindex(universe)


def _derived_signed_impact(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    work = panel.loc[mask, ["TradeDate", "symbol", "minute_return", "net_active_flow"]].copy()
    universe = _universe(panel)
    if work.empty:
        return pd.Series(np.nan, index=universe)
    r = pd.to_numeric(work["minute_return"], errors="coerce")
    s = pd.to_numeric(work["net_active_flow"], errors="coerce")
    ok = r.notna() & s.notna()
    work = work.loc[ok].copy()
    work["_num"] = r.loc[ok].to_numpy(dtype=float) * s.loc[ok].to_numpy(dtype=float)
    work["_den"] = np.abs(s.loc[ok].to_numpy(dtype=float))
    g = work.groupby(["TradeDate", "symbol"], sort=False)
    den = g["_den"].sum()
    out = g["_num"].sum() / den.replace(0, np.nan)
    return out.reindex(universe)


def _derived_impact_asymmetry(panel: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    cols = ["TradeDate", "symbol", "fwd1"]
    for c in ("Active_buy_amount", "Active_sell_amount"):
        if c in panel.columns:
            cols.append(c)
    work = panel.loc[mask, cols].copy()
    universe = _universe(panel)
    if work.empty or "fwd1" not in work.columns:
        return pd.Series(np.nan, index=universe)
    fwd = pd.to_numeric(work["fwd1"], errors="coerce")
    buy = pd.to_numeric(work.get("Active_buy_amount"), errors="coerce")
    sell = pd.to_numeric(work.get("Active_sell_amount"), errors="coerce")
    work["_bn"] = fwd * buy
    work["_sn"] = fwd * sell
    work["_bd"] = buy.where(buy > 0)
    work["_sd"] = sell.where(sell > 0)
    ok = fwd.notna()
    g = work.loc[ok].groupby(["TradeDate", "symbol"], sort=False)
    buy_imp = g["_bn"].sum() / g["_bd"].sum().replace(0, np.nan)
    sell_imp = g["_sn"].sum() / g["_sd"].sum().replace(0, np.nan)
    # Missing buy or sell activity stays missing; do not fill 0.
    out = buy_imp - sell_imp
    return out.reindex(universe)


_DERIVED_OPS = {
    "vwap_close_deviation": _derived_vwap_close_deviation,
    "close_location_value": _derived_clv,
    "return_per_amount": _derived_return_per_amount,
    "net_buy_ratio": _derived_net_buy_ratio,
    "signed_amount_impact": _derived_signed_impact,
    "impact_asymmetry": _derived_impact_asymmetry,
}


def _apply_derived(panel: pd.DataFrame, recipe: Mapping[str, object]) -> Tuple[pd.Series, dict]:
    extra = {"zero_denominator_count": 0, "status": "OK", "skip_reason": ""}
    op = str(recipe.get("derived_op") or recipe.get("base_primitive"))
    fn = _DERIVED_OPS.get(op)
    if fn is None:
        extra["status"] = "SKIPPED_NO_DATA"
        extra["skip_reason"] = "unknown derived_op {}".format(op)
        return pd.Series(dtype=float), extra
    cut_type = str(recipe["cut_type"]).lower()
    if cut_type == "derived_contrast":
        cut_name = str(recipe["cut_name"]).lower()
        if cut_name != "close_minus_open":
            extra["status"] = "SKIPPED_NO_DATA"
            extra["skip_reason"] = "unsupported derived contrast {}".format(cut_name)
            return pd.Series(dtype=float), extra
        a = fn(panel, _mask_by_cut_name(panel, "close"))
        b = fn(panel, _mask_by_cut_name(panel, "open"))
        series, zero = _contrast_series(a, b, str(recipe.get("contrast_operator") or "DIFF"))
        extra["zero_denominator_count"] = int(zero.to_numpy(dtype=bool).sum())
        return series, extra
    mask = _mask_by_cut_name(panel, str(recipe["cut_name"]))
    mkeys = panel["mkey"].to_numpy(dtype=np.int32)
    mask = mask & (mkeys != AUCTION_MKEY)
    _require_no_auction(mask, mkeys, str(recipe.get("candidate_name") or op))
    return fn(panel, mask), extra


def _temporal_grouped(
    panel: pd.DataFrame,
    value_col: str,
    op: str,
    *,
    signed: bool = True,
    weight: str = "signed",
    mask: Optional[np.ndarray] = None,
) -> pd.Series:
    universe = _universe(panel)
    cols = ["TradeDate", "symbol", value_col, "minute_index"]
    work = panel.loc[:, cols].copy()
    if mask is not None:
        work = work.loc[np.asarray(mask, dtype=bool)]
    x = pd.to_numeric(work[value_col], errors="coerce")
    t = pd.to_numeric(work["minute_index"], errors="coerce")
    ok = x.notna() & t.notna()
    work = work.loc[ok].copy()
    x = x.loc[ok]
    t = t.loc[ok]
    if work.empty:
        return pd.Series(np.nan, index=universe)
    use_signed = bool(signed) and str(weight).lower() not in ("abs", "self", "unsigned")
    if use_signed:
        w_plus = x.where(x > 0, 0.0)
        w_minus = (-x).where(x < 0, 0.0)
        work["_tp"] = t.to_numpy(dtype=float) * w_plus.to_numpy(dtype=float)
        work["_tm"] = t.to_numpy(dtype=float) * w_minus.to_numpy(dtype=float)
        work["_wp"] = w_plus.to_numpy(dtype=float)
        work["_wm"] = w_minus.to_numpy(dtype=float)
        g = work.groupby(["TradeDate", "symbol"], sort=False)
        den_p = g["_wp"].sum().replace(0, np.nan)
        den_m = g["_wm"].sum().replace(0, np.nan)
        tc_p = g["_tp"].sum() / den_p
        tc_m = g["_tm"].sum() / den_m
        if op == "tc_plus":
            return tc_p.reindex(universe)
        if op == "tc_minus":
            return tc_m.reindex(universe)
        if op == "temporal_gap":
            return (tc_m - tc_p).reindex(universe)
        w = (w_plus + w_minus).to_numpy(dtype=float)
    else:
        w = x.abs().to_numpy(dtype=float)
    work["_w"] = w
    work["_tw"] = t.to_numpy(dtype=float) * w
    g = work.groupby(["TradeDate", "symbol"], sort=False)
    den = g["_w"].sum().replace(0, np.nan)
    center = g["_tw"].sum() / den
    if op in ("temporal_center", "tc_plus", "tc_minus") and not use_signed:
        return center.reindex(universe)
    if op == "temporal_dispersion":
        ctr = center.rename("center")
        work = work.join(ctr, on=["TradeDate", "symbol"])
        work["_d"] = work["_w"].to_numpy(dtype=float) * (
            t.to_numpy(dtype=float) - work["center"].to_numpy(dtype=float)
        ) ** 2
        g2 = work.groupby(["TradeDate", "symbol"], sort=False)
        disp = np.sqrt(g2["_d"].sum() / g2["_w"].sum().replace(0, np.nan))
        return disp.reindex(universe)
    if op == "temporal_center":
        return center.reindex(universe)
    raise KeyError("unknown temporal op {}".format(op))


def close_window_diagnostics(panel: pd.DataFrame, value_col: str) -> dict:
    """CLOSE source-window facts. Missing 14:57-14:59 is not zero-filled."""
    if "mask_time_close" not in panel.columns or value_col not in panel.columns:
        return {
            "effective_close_start": "14:30:00",
            "effective_close_end": "",
            "observed_minute_count": float("nan"),
            "common_close_end": "14:56:00",
        }
    close = panel.loc[panel["mask_time_close"].to_numpy(dtype=bool)]
    val = pd.to_numeric(close[value_col], errors="coerce")
    finite = close.loc[val.notna()]
    if finite.empty:
        last_mkey = float("nan")
        obs = float("nan")
    else:
        last_mkey = float(finite["mkey"].max())
        obs = float(
            finite.groupby(["TradeDate", "symbol"], sort=False)["mkey"].nunique().mean()
        )
    if np.isfinite(last_mkey):
        hh = int(last_mkey) // 60
        mm = int(last_mkey) % 60
        end = "{:02d}:{:02d}:00".format(hh, mm)
    else:
        end = ""
    return {
        "effective_close_start": "14:30:00",
        "effective_close_end": end,
        "observed_minute_count": obs,
        "common_close_end": "14:56:00",
        "common_close_mkey_end": 896,
    }


def apply_one_recipe(
    panel: pd.DataFrame,
    recipe: Mapping[str, object],
) -> Tuple[str, pd.Series, dict]:
    cut_type = str(recipe["cut_type"]).lower()
    name = _recipe_name(recipe)
    extra = {"zero_denominator_count": 0, "status": "OK", "skip_reason": ""}
    if cut_type in ("derived", "derived_contrast"):
        series, extra = _apply_derived(panel, recipe)
        return name, series, extra
    base = str(recipe["base_primitive"])
    if base not in PRIMITIVE_COLUMN:
        extra["status"] = "SKIPPED_NO_DATA"
        extra["skip_reason"] = "unknown primitive {}".format(base)
        return name, pd.Series(dtype=float), extra
    value_col = PRIMITIVE_COLUMN[base]
    cut_name = str(recipe["cut_name"]).lower()
    agg = str(recipe.get("aggregation") or "")
    extra = {"zero_denominator_count": 0, "status": "OK", "skip_reason": ""}
    if value_col not in panel.columns:
        extra["status"] = "SKIPPED_NO_DATA"
        extra["skip_reason"] = "missing column {}".format(value_col)
        return name, pd.Series(dtype=float), extra
    mkeys = panel["mkey"].to_numpy(dtype=np.int32)
    if cut_type == "temporal":
        signed = bool(recipe.get("signed", True))
        weight = str(recipe.get("weight") or ("signed" if signed else "abs"))
        full = panel["mask_time_full"].to_numpy(dtype=bool) if "mask_time_full" in panel.columns else np.ones(len(panel), dtype=bool)
        full = full & (mkeys != AUCTION_MKEY)
        _require_no_auction(full, mkeys, name)
        series = _temporal_grouped(
            panel, value_col, cut_name, signed=signed, weight=weight, mask=full
        )
        return name, series, extra
    if cut_type == "time":
        mask = panel["mask_time_" + cut_name].to_numpy(dtype=bool)
        if cut_name in ("close", "full", "common_close"):
            _require_no_auction(mask, mkeys, name)
        mask = mask & (mkeys != AUCTION_MKEY)
        _require_no_auction(mask, mkeys, name)
        series = _aggregate_masked(panel, value_col, mask, agg)
        return name, series, extra
    if cut_type == "state":
        mask = panel["mask_state_" + cut_name].to_numpy(dtype=bool)
        mask = mask & (mkeys != AUCTION_MKEY)
        _require_no_auction(mask, mkeys, name)
        series = _aggregate_masked(panel, value_col, mask, agg)
        return name, series, extra
    if cut_type == "event":
        mask = _event_mask(panel, cut_name, value_col)
        mask = mask & (mkeys != AUCTION_MKEY)
        _require_no_auction(mask, mkeys, name)
        series = _aggregate_masked(panel, value_col, mask, agg)
        return name, series, extra
    if cut_type == "contrast":
        op, k1, n1, k2, n2 = CONTRAST_LEGS[cut_name]
        if base in ("large_order_amount", "large_order_pressure") and cut_name == "close_minus_open":
            op = "NORMALIZED_DIFF"
        elif recipe.get("contrast_operator"):
            op = str(recipe.get("contrast_operator")).upper()
        m1 = _mask_from_leg(panel, k1, n1) & (mkeys != AUCTION_MKEY)
        m2 = _mask_from_leg(panel, k2, n2) & (mkeys != AUCTION_MKEY)
        if str(n1).upper() in ("CLOSE", "FULL") or str(n2).upper() in ("CLOSE", "FULL"):
            _require_no_auction(m1 | m2, mkeys, name)
        _require_no_auction(m1 | m2, mkeys, name)
        a = _aggregate_masked(panel, value_col, m1, agg)
        b = _aggregate_masked(panel, value_col, m2, agg)
        series, zero = _contrast_series(a, b, op)
        extra["zero_denominator_count"] = int(zero.to_numpy(dtype=bool).sum())
        return name, series, extra
    extra["status"] = "SKIPPED_NO_DATA"
    extra["skip_reason"] = "unsupported cut_type {}".format(cut_type)
    return name, pd.Series(dtype=float), extra


_AVAIL_CLOCK = {
    "after_1000_T": ("09:59:00", "10:00:00"),
    "after_1130_T": ("11:29:00", "11:30:00"),
    "after_1430_T": ("14:29:00", "14:30:00"),
    "after_1450_T": ("14:49:00", "14:50:00"),
    "after_continuous_close_T": ("14:59:00", "15:00:00"),
    "after_close_auction_T": ("15:00:00", "15:01:00"),
}


def availability_for_recipe(recipe: Mapping[str, object]) -> Dict[str, object]:
    cut_type = str(recipe["cut_type"]).lower()
    cut_name = str(recipe["cut_name"]).lower()
    uses_close = False
    uses_full = False
    uses_1456 = False
    uses_last5 = False
    auction = False
    start, end = "09:30:00", "15:00:00"
    avail = "after_continuous_close_T"
    if cut_type == "time":
        spec = time_segment(cut_name)
        start, end = spec["start_time"], spec["end_time"]
        auction = bool(spec["contains_close_auction"])
        uses_1456 = bool(spec["contains_1456_1500"])
        uses_last5 = bool(spec["uses_last_5min"])
        avail = str(spec["availability_timestamp"])
        uses_close = cut_name == "close"
        uses_full = cut_name == "full"
    elif cut_type == "contrast":
        op, k1, n1, k2, n2 = CONTRAST_LEGS[cut_name]
        for kind, leg in ((k1, n1), (k2, n2)):
            if str(leg).upper() in ("CLOSE", "FULL", "LATE_CLOSE", "COMMON_CLOSE"):
                uses_1456 = True
                uses_last5 = True
                avail = "after_continuous_close_T"
            if str(leg).upper() == "CLOSE":
                uses_close = True
            if str(leg).upper() == "FULL":
                uses_full = True
            if str(kind).lower() == "state":
                uses_1456 = True
                uses_last5 = True
                avail = "after_continuous_close_T"
        start, end = "09:30:00", "15:00:00"
    elif cut_type in ("derived", "derived_contrast", "temporal"):
        if cut_name in ("close", "close_minus_open") or cut_type == "temporal":
            uses_1456 = True
            uses_last5 = True
            uses_close = cut_name in ("close", "close_minus_open")
            avail = "after_continuous_close_T"
        elif cut_name == "open":
            start, end = "09:30:00", "10:00:00"
            avail = "after_1000_T"
        elif cut_name == "afternoon":
            start, end = "13:00:00", "14:30:00"
            avail = "after_1430_T"
        else:
            uses_1456 = True
            uses_last5 = True
            avail = "after_continuous_close_T"
    else:
        uses_1456 = True
        uses_last5 = True
        avail = "after_continuous_close_T"
    if uses_close or uses_full:
        if auction:
            raise RuntimeError("CLOSE/FULL recipe marked contains_close_auction")
    latest, after = _AVAIL_CLOCK.get(avail, ("14:59:00", "15:00:00"))
    if auction:
        latest, after = "15:00:00", "15:01:00"
    return {
        "cut_start_time": start,
        "cut_end_time": end,
        "availability_timestamp": avail,
        "contains_close_auction": bool(auction),
        "contains_1456_1500": bool(uses_1456),
        "uses_close_auction": bool(auction),
        "uses_last_5min": bool(uses_last5),
        "latest_source_timestamp": latest,
        "factor_available_after": after,
        "execution_contract": PRODUCTION_EXECUTION_CONTRACT,
        "execution_contract_compatible": PRODUCTION_EXECUTION_CONTRACT,
        "production_execution_compatible": True,
        "close_auction_misuse": bool(auction and (uses_close or uses_full)),
    }


def apply_tc1_recipes(
    panel: pd.DataFrame,
    recipes: Sequence[Mapping[str, object]] = TC1_RECIPES,
) -> Tuple[pd.DataFrame, List[dict]]:
    ready = attach_state_masks(attach_helper_columns(panel))
    universe = _universe(ready)
    wide = pd.DataFrame(index=universe)
    metas = []
    for rec in recipes:
        name, series, extra = apply_one_recipe(ready, rec)
        if extra.get("status") == "SKIPPED_NO_DATA":
            wide[name] = np.nan
        else:
            wide[name] = series.reindex(universe)
        meta = dict(rec)
        meta["candidate_name"] = name
        meta["cut_definition"] = rec.get("cut_name")
        meta.update(availability_for_recipe(rec))
        meta.update(extra)
        if meta["close_auction_misuse"]:
            raise RuntimeError("close_auction_misuse for {}".format(name))
        if str(rec.get("cut_type")) == "time" and str(rec.get("cut_name")).lower() in (
            "close",
            "full",
        ):
            if meta["contains_close_auction"]:
                raise RuntimeError("CLOSE/FULL contains_close_auction True: {}".format(name))
            if not meta["contains_1456_1500"]:
                raise RuntimeError("CLOSE/FULL missing 14:56-15:00 flag: {}".format(name))
        metas.append(meta)
    wide = wide.reset_index()
    return wide, metas


def apply_tc2a_recipes(
    panel: pd.DataFrame,
    recipes: Sequence[Mapping[str, object]],
) -> Tuple[pd.DataFrame, List[dict]]:
    """Apply explicit TC-2A recipes. Same auction / missingness rules as TC-1."""
    ready = attach_state_masks(attach_helper_columns(panel))
    universe = _universe(ready)
    wide = pd.DataFrame(index=universe)
    metas = []
    close_diag_cache: Dict[str, dict] = {}
    for rec in recipes:
        name, series, extra = apply_one_recipe(ready, rec)
        if extra.get("status") == "SKIPPED_NO_DATA":
            wide[name] = np.nan
        else:
            wide[name] = series.reindex(universe)
        meta = dict(rec)
        meta["candidate_name"] = name
        meta["cut_definition"] = rec.get("cut_name")
        meta.update(availability_for_recipe(rec))
        meta.update(extra)
        if meta.get("close_auction_misuse"):
            raise RuntimeError("close_auction_misuse for {}".format(name))
        cut_name = str(rec.get("cut_name") or "").lower()
        cut_type = str(rec.get("cut_type") or "").lower()
        uses_close = cut_name in ("close", "close_minus_open") or (
            cut_type == "contrast" and "close" in cut_name
        )
        if uses_close:
            base = str(rec.get("base_primitive") or "")
            col = PRIMITIVE_COLUMN.get(base, "")
            if not col or col not in ready.columns:
                if "Close" in ready.columns:
                    col = "Close"
            if col not in close_diag_cache:
                close_diag_cache[col] = close_window_diagnostics(ready, col)
            meta.update(close_diag_cache[col])
        metas.append(meta)
    wide = wide.reset_index()
    return wide, metas
