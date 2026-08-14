"""EOD loaders and (9, 30) image construction.

Live loads go through ``core.ddb.connection.get_ddb_session`` and cache parquet
under ``research/results/alphanet_v1/cache``. Synthetic panels do not need DDB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from alphanet.config import FEATURE_NAMES, PAPER_END, PAPER_START, PREHEAT_CALENDAR_DAYS
from alphanet.paths import CACHE, ensure_result_dirs
from alphanet.universe import a_share_columns, combine_masks, next_session_tradable


def _filter_a(df: pd.DataFrame) -> pd.DataFrame:
    cols = a_share_columns(df.columns)
    out = df.loc[:, cols].copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    wide = df.pivot_table(
        index="TRADE_DT", columns="S_INFO_WINDCODE", values=value, aggfunc="last"
    )
    return _filter_a(wide)


@dataclass
class MarketPanel:
    features: Dict[str, pd.DataFrame]
    ret_1d: pd.DataFrame
    adj_close: pd.DataFrame
    industry: Optional[pd.DataFrame] = None
    log_mcap: Optional[pd.DataFrame] = None
    tradable: Optional[pd.DataFrame] = None
    index_members: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ret_1d.index)

    @property
    def symbols(self) -> pd.Index:
        return self.ret_1d.columns


def forward_return(ret_1d: pd.DataFrame, horizon: int, execution: str) -> pd.DataFrame:
    """Holding-period simple return aligned on the *signal date* T.

    ``paper_c2c``: Close[T+m]/Close[T]-1 = prod_{k=1..m}(1+r[T+k]) - 1
    ``executable_tplus1``: Close[T+1+m]/Close[T+1]-1 = prod_{k=2..m+1}(1+r[T+k]) - 1
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    one = 1.0 + ret_1d
    if execution == "paper_c2c":
        # r[T+1] lives on row T+1; shift(-1) aligns that row onto T
        stacked = np.ones_like(one.to_numpy(), dtype=float)
        for k in range(1, horizon + 1):
            stacked = stacked * one.shift(-k).to_numpy()
        out = pd.DataFrame(stacked - 1.0, index=ret_1d.index, columns=ret_1d.columns)
        return out
    if execution == "executable_tplus1":
        stacked = np.ones_like(one.to_numpy(), dtype=float)
        for k in range(2, horizon + 2):
            stacked = stacked * one.shift(-k).to_numpy()
        return pd.DataFrame(stacked - 1.0, index=ret_1d.index, columns=ret_1d.columns)
    raise ValueError("unknown execution {!r}".format(execution))


def cs_zscore(panel: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0)
    n = panel.notna().sum(axis=1)
    z = panel.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    return z.where(n >= min_obs)


def sample_dates(calendar: pd.DatetimeIndex, every: int, start=None, end=None) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(calendar)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx[:: max(int(every), 1)]


def _try_session():
    from core.ddb.connection import get_ddb_session

    return get_ddb_session()


def load_eod_from_ddb(
    start: str = PAPER_START,
    end: str = PAPER_END,
    *,
    cache: bool = True,
) -> MarketPanel:
    ensure_result_dirs()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    preheat = start_ts - pd.Timedelta(days=int(PREHEAT_CALENDAR_DAYS))
    cache_key = CACHE / "eod_{}_{}.pkl".format(start_ts.date(), end_ts.date())
    if cache and cache_key.exists():
        return pd.read_pickle(cache_key)

    s = _try_session()
    start_str = preheat.strftime("%Y.%m.%d")
    end_str = end_ts.strftime("%Y.%m.%d")
    script = """
    t = select TRADE_DT, S_INFO_WINDCODE,
               S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE,
               S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_AVGPRICE, S_DQ_ADJFACTOR,
               S_DQ_TURN, S_DQ_TRADESTATUS, S_DQ_LIMIT, S_DQ_STOPPING,
               S_DQ_PRECLOSE
        from loadTable("dfs://WIND.ASHAREEODPRICES", "data")
        where TRADE_DT >= {start} and TRADE_DT <= {end}
    """.format(start=start_str, end=end_str)
    eod = s.run(script)
    eod["TRADE_DT"] = pd.to_datetime(eod["TRADE_DT"])
    adj = pd.to_numeric(eod["S_DQ_ADJFACTOR"], errors="coerce")
    for col in ("S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE", "S_DQ_AVGPRICE"):
        eod["adj_" + col] = pd.to_numeric(eod[col], errors="coerce") * adj

    eod["adj_open"] = eod["adj_S_DQ_OPEN"]
    eod["adj_high"] = eod["adj_S_DQ_HIGH"]
    eod["adj_low"] = eod["adj_S_DQ_LOW"]
    eod["adj_close"] = eod["adj_S_DQ_CLOSE"]
    eod["adj_vwap"] = eod["adj_S_DQ_AVGPRICE"]
    eod["volume"] = pd.to_numeric(eod["S_DQ_VOLUME"], errors="coerce")
    eod["turn"] = pd.to_numeric(eod["S_DQ_TURN"], errors="coerce")
    eod["amount"] = pd.to_numeric(eod["S_DQ_AMOUNT"], errors="coerce")

    adj_close = _pivot(eod, "adj_close")
    adj_open = _pivot(eod, "adj_open").reindex_like(adj_close)
    adj_high = _pivot(eod, "adj_high").reindex_like(adj_close)
    adj_low = _pivot(eod, "adj_low").reindex_like(adj_close)
    adj_vwap = _pivot(eod, "adj_vwap").reindex_like(adj_close)
    volume = _pivot(eod, "volume").reindex_like(adj_close)
    turn = _pivot(eod, "turn").reindex_like(adj_close)
    amount = _pivot(eod, "amount").reindex_like(adj_close)
    ret_1d = adj_close.pct_change()

    # free_turn: amount / float mkt cap when the dedicated field is absent
    free_turn = turn.copy()
    try:
        der = s.run(
            """
            select TRADE_DT, S_INFO_WINDCODE, S_DQ_MV, S_VAL_MV
            from loadTable("dfs://WIND.ASHAREEODDERIVATIVEINDICATOR", "data")
            where TRADE_DT >= {start} and TRADE_DT <= {end}
            context by TRADE_DT, S_INFO_WINDCODE csort OPDATE limit 1
            """.format(start=start_str, end=end_str)
        )
        der["TRADE_DT"] = pd.to_datetime(der["TRADE_DT"])
        float_mv = _pivot(der, "S_DQ_MV")
        total_mv = _pivot(der, "S_VAL_MV")
        # S_DQ_AMOUNT 千元, S_DQ_MV 万元 → convert both to 元
        amount_yuan = amount * 1000.0
        float_yuan = float_mv.reindex_like(amount) * 10000.0
        computed = amount_yuan / float_yuan.replace(0, np.nan)
        free_turn = computed.reindex_like(adj_close)
        log_mcap = np.log(total_mv.reindex_like(adj_close).replace(0, np.nan))
    except Exception as exc:
        float_mv = None
        log_mcap = None
        meta_err = str(exc)
    else:
        meta_err = None

    not_limit = _pivot(
        eod.assign(
            not_limit=np.where(
                (pd.to_numeric(eod["S_DQ_CLOSE"], errors="coerce")
                 < pd.to_numeric(eod["S_DQ_LIMIT"], errors="coerce"))
                & (pd.to_numeric(eod["S_DQ_CLOSE"], errors="coerce")
                   > pd.to_numeric(eod["S_DQ_STOPPING"], errors="coerce")),
                1.0,
                np.nan,
            )
        ),
        "not_limit",
    ).reindex_like(adj_close)
    status = eod["S_DQ_TRADESTATUS"].astype(str)
    eod["trading"] = np.where(status.isin(["停牌", ""]), np.nan, 1.0)
    trading = _pivot(eod, "trading").reindex_like(adj_close)

    not_st = None
    industry = None
    try:
        from Factor_Dev_Lib import get_EOD_Not_ST, get_preheat_ind_data_citics

        not_st = get_EOD_Not_ST(preheat.to_pydatetime(), end_ts.to_pydatetime())
        not_st.index = pd.to_datetime(not_st.index).normalize()
        not_st = _filter_a(not_st).reindex_like(adj_close)
        ind = get_preheat_ind_data_citics(start_ts.to_pydatetime(), end_ts.to_pydatetime())
        if "TradingDay" in getattr(ind, "columns", []):
            ind = ind.set_index("TradingDay")
        ind.index = pd.to_datetime(ind.index).normalize()
        industry = _filter_a(ind).reindex_like(adj_close)
    except Exception:
        not_st = pd.DataFrame(1.0, index=adj_close.index, columns=adj_close.columns)

    tradable = combine_masks(not_limit, trading, not_st)
    tradable = tradable.where(next_session_tradable(combine_masks(not_limit, trading)) == 1)

    features = {
        "return1": ret_1d,
        "open": adj_open,
        "close": adj_close,
        "high": adj_high,
        "low": adj_low,
        "vwap": adj_vwap,
        "volume": volume,
        "turn": turn,
        "free_turn": free_turn,
    }
    panel = MarketPanel(
        features=features,
        ret_1d=ret_1d,
        adj_close=adj_close,
        industry=industry,
        log_mcap=log_mcap,
        tradable=tradable,
        meta={
            "start": str(start_ts.date()),
            "end": str(end_ts.date()),
            "free_turn_source": "amount/S_DQ_MV" if meta_err is None else "turn_fallback",
            "industry_source": "CITICS_L1_preheat",
            "error": meta_err,
        },
    )
    if cache:
        cache_key.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(panel, cache_key)
    return panel


def panel_from_synthetic(synth) -> MarketPanel:
    return MarketPanel(
        features=dict(synth.features),
        ret_1d=synth.ret_1d,
        adj_close=synth.features["close"],
        industry=synth.industry,
        log_mcap=synth.log_mcap,
        tradable=synth.tradable,
        meta={"source": "synthetic"},
    )
