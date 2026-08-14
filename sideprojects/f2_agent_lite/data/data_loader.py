"""Windowed multimodal panel builder for F² Agent Lite."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from . import db_connector as db


OHLCV_COLS = ["open", "high", "low", "close", "volume"]
TECH_COLS = ["rsi", "macd", "macd_signal", "macd_hist", "sma", "bb_upper", "bb_lower"]

# Optional alpha blocks (appended when corresponding Config flag is True)
NORTH_COLS = ["north_share_ratio", "north_share_chg"]
FUND_COLS = ["ep_ttm", "bp", "roe", "revenue_growth_yoy", "turnover", "log_mktcap"]
ADV_COLS = ["ret_20d", "ret_60d", "illiquidity", "resid_vol_20d"]
MKT_RISK_COLS = ["mkt_vol_20d", "limit_up_ratio", "limit_down_ratio", "cross_sec_vol"]
MINUTE_COLS = ["minute_amplitude", "price_jump"]


def resolve_feature_cols(config: Optional[Config] = None) -> List[str]:
    """OHLCV + tech + sentiment + enabled alpha blocks."""
    cols = list(OHLCV_COLS) + list(TECH_COLS) + ["sentiment_score"]
    cfg = config
    if cfg is None:
        return cols
    if getattr(cfg, "use_north_money", False):
        cols += list(NORTH_COLS)
    if getattr(cfg, "use_fundamentals", False):
        cols += list(FUND_COLS)
    if getattr(cfg, "use_advanced_alpha", False):
        cols += list(ADV_COLS)
    if getattr(cfg, "use_market_risk", False):
        cols += list(MKT_RISK_COLS)
    if getattr(cfg, "use_minute_factors", False):
        cols += list(MINUTE_COLS)
    return cols


def _feature_cache_tag(cfg: Config) -> str:
    flags = "".join(
        [
            "N" if getattr(cfg, "use_north_money", False) else "n",
            "F" if getattr(cfg, "use_fundamentals", False) else "f",
            "A" if getattr(cfg, "use_advanced_alpha", False) else "a",
            "R" if getattr(cfg, "use_market_risk", False) else "r",
            "M" if getattr(cfg, "use_minute_factors", False) else "m",
        ]
    )
    if getattr(cfg, "use_minute_factors", False):
        flags += "lb{}n".format(int(getattr(cfg, "minute_factor_lookback", 10)))
    return flags


@dataclass
class SymbolPanel:
    symbol: str
    daily: pd.DataFrame  # indexed by date, aligned features + labels + masks
    party_id: int


@dataclass
class WindowDataset:
    dates: np.ndarray  # (N,) datetime64
    symbols: np.ndarray  # (N,) str
    market: np.ndarray  # (N, T, 5)
    tech: np.ndarray  # (N, T, F_tech)
    news_text: np.ndarray  # (N,) object str
    sentiment: np.ndarray  # (N,) float
    y: np.ndarray  # (N,) int 1=UP 0=DOWN
    open_px: np.ndarray  # (N,) execution open (next day)
    close_px: np.ndarray  # (N,) signal-day close
    next_close_px: np.ndarray  # (N,) execution-day close (MTM)
    next_date: np.ndarray  # (N,) execution date
    tradable_exec: np.ndarray  # (N,) bool — next day tradable
    meta: pd.DataFrame


def _add_calendar_days(value, days: int) -> dt.datetime:
    ts = pd.Timestamp(value) - pd.Timedelta(days=days)
    return dt.datetime(ts.year, ts.month, ts.day)


def apply_forward_labels(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach execution fields + H-day forward return labels in-place.

    - Enter next open (shift -1)
    - Label / ranking target: close[t+H] / close[t] - 1
    - next_close kept as next-day close for daily MTM in backtester
    """
    h = max(1, int(getattr(cfg, "pred_horizon", 1)))
    daily["next_close"] = daily["close"].shift(-1)
    daily["next_open"] = daily["open"].shift(-1)
    daily["next_date"] = daily.index.to_series().shift(-1)
    daily["next_tradable"] = daily["tradable"].shift(-1)
    daily["horizon_close"] = daily["close"].shift(-h)
    daily["fwd_ret"] = daily["horizon_close"] / daily["close"] - 1.0
    thr = float(getattr(cfg, "label_threshold", 0.005))
    label = np.ones(len(daily), dtype=float)
    fwd = daily["fwd_ret"].to_numpy(dtype=float)
    label[fwd > thr] = 2.0
    label[fwd < -thr] = 0.0
    label[np.isnan(fwd)] = np.nan
    daily["label"] = label
    return daily


def _join_frame(daily: pd.DataFrame, frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Left-join feature cols onto a date-indexed or date-column daily frame."""
    daily = daily.copy()
    for c in cols:
        if c not in daily.columns:
            daily[c] = np.nan
    if frame is None or frame.empty:
        return daily
    f = frame.copy()
    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        f = f.set_index("date")
    use = [c for c in cols if c in f.columns]
    if not use:
        return daily
    if daily.index.name != "date" and "date" in daily.columns:
        base = daily.set_index("date")
        joined = base.join(f[use], how="left", rsuffix="_new")
        for c in use:
            new_c = c + "_new"
            if new_c in joined.columns:
                joined[c] = joined[new_c].combine_first(joined[c])
                joined.drop(columns=[new_c], inplace=True)
        return joined.reset_index()
    joined = daily.join(f[use], how="left", rsuffix="_new")
    for c in use:
        new_c = c + "_new"
        if new_c in joined.columns:
            joined[c] = joined[new_c].combine_first(joined[c])
            joined.drop(columns=[new_c], inplace=True)
    return joined


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self._party_cache: Dict[str, int] = {}
        self._market_risk: Optional[pd.DataFrame] = None
        self._market_close: Optional[pd.Series] = None
        self._minute_factors: Optional[pd.DataFrame] = None

    def party_id(self, symbol: str) -> int:
        if symbol not in self._party_cache:
            self._party_cache[symbol] = db.resolve_party_id(symbol)
        return self._party_cache[symbol]

    def _preheat_days(self) -> int:
        cfg = self.config
        base = int(cfg.preheat_calendar_days)
        if any(
            [
                getattr(cfg, "use_advanced_alpha", False),
                getattr(cfg, "use_fundamentals", False),
                getattr(cfg, "use_market_risk", False),
                getattr(cfg, "use_minute_factors", False),
            ]
        ):
            base = max(base, int(getattr(cfg, "alpha_preheat_calendar_days", 180)))
        if getattr(cfg, "use_minute_factors", False):
            lb = int(getattr(cfg, "minute_factor_lookback", 10))
            base = max(base, lb * 3 + 30)
        return base

    def _ensure_market_panels(self, start, end) -> None:
        cfg = self.config
        need_risk = getattr(cfg, "use_market_risk", False)
        need_close = getattr(cfg, "use_advanced_alpha", False) or need_risk
        if not need_risk and not need_close:
            return
        if (not need_risk or self._market_risk is not None) and (
            not need_close or self._market_close is not None
        ):
            return

        preheat_start = _add_calendar_days(start, self._preheat_days())
        if need_risk and self._market_risk is None:
            try:
                self._market_risk = db.get_market_risk(preheat_start, end)
                print(
                    "[data] market_risk days={}".format(
                        0 if self._market_risk is None else len(self._market_risk)
                    ),
                    flush=True,
                )
            except Exception as exc:
                print("[data] market_risk load failed:", exc, flush=True)
                self._market_risk = pd.DataFrame(columns=["date"] + list(MKT_RISK_COLS))

        if need_close and self._market_close is None:
            try:
                self._market_close = db.get_index_close(db.MKT_INDEX_CODE, preheat_start, end)
            except Exception as exc:
                print("[data] market close load failed:", exc, flush=True)
                self._market_close = pd.Series(dtype=float)

    def _ensure_minute_factors(self, symbols: Sequence[str], start, end) -> None:
        cfg = self.config
        if not getattr(cfg, "use_minute_factors", False):
            return
        if self._minute_factors is not None:
            return
        preheat_start = _add_calendar_days(start, self._preheat_days())
        try:
            print(
                "[data] loading minute factors for {} symbols ...".format(len(symbols)),
                flush=True,
            )
            self._minute_factors = db.get_minute_factors(
                symbols,
                preheat_start,
                end,
                lookback=int(getattr(cfg, "minute_factor_lookback", 10)),
                use_local_tables=bool(getattr(cfg, "minute_use_local_tables", False)),
                config=cfg,
            )
            n = 0 if self._minute_factors is None else len(self._minute_factors)
            print("[data] minute factors rows={}".format(n), flush=True)
        except Exception as exc:
            print("[data] minute factors load failed:", exc, flush=True)
            self._minute_factors = pd.DataFrame(
                columns=["date", "symbol"] + list(MINUTE_COLS)
            )

    def load_symbol_daily(self, symbol: str, start, end) -> SymbolPanel:
        cfg = self.config
        preheat_start = _add_calendar_days(start, self._preheat_days())
        party_id = self.party_id(symbol)
        self._ensure_market_panels(start, end)
        if getattr(cfg, "use_minute_factors", False):
            self._ensure_minute_factors([symbol], start, end)

        ohlcv = db.get_ohlcv(symbol, preheat_start, end)
        tech = db.compute_technical_from_ohlcv(ohlcv)
        trad = db.compute_tradability_from_ohlcv(ohlcv)
        news = db.get_news_sentiment(
            symbol,
            preheat_start,
            end,
            party_id=party_id,
            max_titles=cfg.news_max_titles,
            fetch_titles=bool(getattr(cfg, "fetch_news_titles", False)),
        )

        if ohlcv.empty:
            raise RuntimeError(f"No OHLCV for {symbol} in [{preheat_start}, {end}]")

        daily = ohlcv.set_index("date").sort_index()
        for frame, cols in [
            (tech.set_index("date") if not tech.empty else pd.DataFrame(), TECH_COLS),
            (
                news.set_index("date")
                if not news.empty
                else pd.DataFrame(columns=["news_summary", "sentiment_score", "news_count"]),
                ["news_summary", "sentiment_score", "news_count"],
            ),
            (
                trad.set_index("date")
                if not trad.empty
                else pd.DataFrame(columns=["not_limit", "not_suspended", "tradable"]),
                ["not_limit", "not_suspended", "tradable"],
            ),
        ]:
            if frame.empty:
                for c in cols:
                    daily[c] = np.nan if c != "news_summary" else ""
            else:
                daily = daily.join(frame[cols], how="left")

        # ---- Alpha blocks ----
        if getattr(cfg, "use_north_money", False):
            try:
                north = db.get_northbound(symbol, preheat_start, end)
            except Exception as exc:
                print("[data] northbound {} failed: {}".format(symbol, exc), flush=True)
                north = pd.DataFrame(columns=["date"] + NORTH_COLS)
            daily = _join_frame(daily, north, NORTH_COLS)

        if getattr(cfg, "use_fundamentals", False):
            try:
                val = db.get_valuation(symbol, preheat_start, end)
            except Exception as exc:
                print("[data] valuation {} failed: {}".format(symbol, exc), flush=True)
                val = pd.DataFrame(columns=["date", "ep_ttm", "bp", "turnover", "log_mktcap"])
            try:
                fund = db.get_fundamentals_pit(symbol, preheat_start, end)
            except Exception as exc:
                print("[data] fundamentals {} failed: {}".format(symbol, exc), flush=True)
                fund = pd.DataFrame(columns=["date", "roe", "revenue_growth_yoy"])
            daily = _join_frame(daily, val, ["ep_ttm", "bp", "turnover", "log_mktcap"])
            daily = _join_frame(daily, fund, ["roe", "revenue_growth_yoy"])
            for c in ["roe", "revenue_growth_yoy"]:
                if c in daily.columns:
                    daily[c] = daily[c].ffill()

        if getattr(cfg, "use_advanced_alpha", False):
            try:
                adv = db.compute_advanced_alpha(ohlcv, market_close=self._market_close)
            except Exception as exc:
                print("[data] advanced_alpha {} failed: {}".format(symbol, exc), flush=True)
                adv = pd.DataFrame(columns=["date"] + ADV_COLS)
            daily = _join_frame(daily, adv, ADV_COLS)

        if getattr(cfg, "use_market_risk", False):
            mkt = self._market_risk if self._market_risk is not None else pd.DataFrame()
            daily = _join_frame(daily, mkt, MKT_RISK_COLS)

        if getattr(cfg, "use_minute_factors", False):
            mf_all = self._minute_factors
            if mf_all is None or mf_all.empty:
                mf = pd.DataFrame(columns=["date"] + list(MINUTE_COLS))
            else:
                mf = mf_all[mf_all["symbol"] == symbol][["date"] + list(MINUTE_COLS)]
            daily = _join_frame(daily, mf, MINUTE_COLS)

        for c in resolve_feature_cols(cfg):
            if c not in daily.columns:
                daily[c] = np.nan

        daily["news_summary"] = daily["news_summary"].fillna("")
        daily["sentiment_score"] = daily["sentiment_score"].fillna(0.0)
        daily["news_count"] = daily["news_count"].fillna(0)
        empty_news = daily["news_summary"].astype(str).str.strip() == ""
        daily.loc[empty_news, "news_summary"] = daily.loc[empty_news, "sentiment_score"].map(
            lambda x: "sent_{}".format(int(round(float(x) * 50.0)))
        )

        apply_forward_labels(daily, cfg)
        return SymbolPanel(symbol=symbol, daily=daily, party_id=party_id)

    def build_windows(
        self,
        panel: SymbolPanel,
        start,
        end,
    ) -> WindowDataset:
        cfg = self.config
        T = cfg.lookback_window
        daily = panel.daily
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()

        dates = daily.index
        market_mat = daily[OHLCV_COLS].astype(float).values
        tech_mat = daily[TECH_COLS].astype(float).values
        market_mat = np.nan_to_num(market_mat, nan=0.0, posinf=0.0, neginf=0.0)
        tech_mat = np.nan_to_num(tech_mat, nan=0.0, posinf=0.0, neginf=0.0)

        rows = []
        for i in range(T - 1, len(daily)):
            d = dates[i]
            if d < start_ts or d > end_ts:
                continue
            if np.isnan(daily["label"].iloc[i]):
                continue
            if pd.isna(daily["next_open"].iloc[i]) or pd.isna(daily["next_date"].iloc[i]):
                continue
            if pd.isna(daily["next_close"].iloc[i]):
                continue

            m_win = market_mat[i - T + 1 : i + 1].copy()
            t_win = tech_mat[i - T + 1 : i + 1].copy()
            m_win = _zscore_window(m_win)
            t_win = _zscore_window(t_win)

            rows.append(
                {
                    "date": d,
                    "symbol": panel.symbol,
                    "market": m_win,
                    "tech": t_win,
                    "news_text": str(daily["news_summary"].iloc[i]),
                    "sentiment": float(daily["sentiment_score"].iloc[i]),
                    "y": int(daily["label"].iloc[i]),
                    "open_px": float(daily["next_open"].iloc[i]),
                    "close_px": float(daily["close"].iloc[i]),
                    "next_close_px": float(daily["next_close"].iloc[i]),
                    "next_date": pd.Timestamp(daily["next_date"].iloc[i]),
                    "tradable_exec": bool(pd.notna(daily["next_tradable"].iloc[i])),
                }
            )

        if not rows:
            raise RuntimeError(f"No windows for {panel.symbol} in [{start}, {end}]")

        meta = pd.DataFrame(rows)
        return WindowDataset(
            dates=meta["date"].to_numpy(),
            symbols=meta["symbol"].to_numpy(),
            market=np.stack(meta["market"].to_list()),
            tech=np.stack(meta["tech"].to_list()),
            news_text=meta["news_text"].to_numpy(),
            sentiment=meta["sentiment"].to_numpy(dtype=float),
            y=meta["y"].to_numpy(dtype=int),
            open_px=meta["open_px"].to_numpy(dtype=float),
            close_px=meta["close_px"].to_numpy(dtype=float),
            next_close_px=meta["next_close_px"].to_numpy(dtype=float),
            next_date=meta["next_date"].to_numpy(),
            tradable_exec=meta["tradable_exec"].to_numpy(dtype=bool),
            meta=meta[
                [
                    "date",
                    "symbol",
                    "y",
                    "open_px",
                    "close_px",
                    "next_close_px",
                    "next_date",
                    "tradable_exec",
                    "news_text",
                    "sentiment",
                ]
            ],
        )

    def prepare_all(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        """Load panels and build train/val/test window datasets (pooled)."""
        cfg = self.config
        symbols = list(symbols or cfg.symbols)

        cache_path = None
        if getattr(cfg, "use_data_cache", False):
            import pickle

            cache_dir = Path(cfg.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            sym_key = "-".join(symbols)
            feat_tag = _feature_cache_tag(cfg)
            prefix = "{}_{}_{}_{}_{}_".format(
                sym_key, cfg.train_start, cfg.train_end, cfg.test_end, feat_tag
            ).replace("/", "-")
            suffix = "_lb{}.pkl".format(cfg.lookback_window)
            exact = cache_dir / "{}thr{}_lb{}.pkl".format(prefix, cfg.label_threshold, cfg.lookback_window)
            candidates = [exact] if exact.exists() else []
            if not candidates:
                candidates = sorted(
                    cache_dir.glob(prefix + "thr*" + suffix),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            cache_path = exact
            if candidates:
                hit = candidates[0]
                print("[data] loading cache", hit)
                with open(hit, "rb") as f:
                    packs = pickle.load(f)
                panels = packs.get("panels") or {}
                for panel in panels.values():
                    apply_forward_labels(panel.daily, cfg)
                print(
                    "[data] re-applied labels pred_horizon={} thr={}".format(
                        getattr(cfg, "pred_horizon", 1), cfg.label_threshold
                    )
                )
                return packs

        panels = {}
        # Batch-load minute factors once for the whole universe before per-symbol loop
        if getattr(cfg, "use_minute_factors", False):
            self._ensure_minute_factors(symbols, cfg.train_start, cfg.test_end)

        for sym in symbols:
            try:
                print("[data] loading", sym, "...", flush=True)
                panels[sym] = self.load_symbol_daily(sym, cfg.train_start, cfg.test_end)
            except Exception as exc:
                print("[data] SKIP {} due to: {}".format(sym, exc), flush=True)
        if not panels:
            raise RuntimeError("No symbols loaded successfully")
        symbols = list(panels.keys())

        train_parts: List[WindowDataset] = []
        test_parts: List[WindowDataset] = []
        for sym, panel in panels.items():
            train_parts.append(self.build_windows(panel, cfg.train_start, cfg.train_end))
            test_parts.append(self.build_windows(panel, cfg.test_start, cfg.test_end))

        train_all = concat_datasets(train_parts)
        test_all = concat_datasets(test_parts)
        train_ds, val_ds = chronological_split(train_all, val_ratio=cfg.val_ratio)
        out = {
            "panels": panels,
            "train": train_ds,
            "val": val_ds,
            "test": test_all,
        }
        if cache_path is not None:
            import pickle

            with open(cache_path, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("[data] wrote cache", cache_path)
        return out


def _zscore_window(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd < 1e-8:
            out[:, j] = 0.0
        else:
            out[:, j] = (col - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def concat_datasets(parts: Sequence[WindowDataset]) -> WindowDataset:
    meta = pd.concat([p.meta for p in parts], ignore_index=True)
    order = np.lexsort((meta["symbol"].astype(str).to_numpy(), pd.to_datetime(meta["date"]).to_numpy()))
    meta = meta.iloc[order].reset_index(drop=True)

    market = np.concatenate([p.market for p in parts], axis=0)[order]
    tech = np.concatenate([p.tech for p in parts], axis=0)[order]
    news_text = np.concatenate([p.news_text for p in parts], axis=0)[order]
    sentiment = np.concatenate([p.sentiment for p in parts], axis=0)[order]
    y = np.concatenate([p.y for p in parts], axis=0)[order]
    open_px = np.concatenate([p.open_px for p in parts], axis=0)[order]
    close_px = np.concatenate([p.close_px for p in parts], axis=0)[order]
    next_close_px = np.concatenate([p.next_close_px for p in parts], axis=0)[order]
    next_date = np.concatenate([p.next_date for p in parts], axis=0)[order]
    tradable_exec = np.concatenate([p.tradable_exec for p in parts], axis=0)[order]
    dates = meta["date"].to_numpy()
    symbols = meta["symbol"].to_numpy()

    return WindowDataset(
        dates=dates,
        symbols=symbols,
        market=market,
        tech=tech,
        news_text=news_text,
        sentiment=sentiment,
        y=y,
        open_px=open_px,
        close_px=close_px,
        next_close_px=next_close_px,
        next_date=next_date,
        tradable_exec=tradable_exec,
        meta=meta,
    )


def chronological_split(
    ds: WindowDataset, val_ratio: float = 0.15
) -> Tuple[WindowDataset, WindowDataset]:
    n = len(ds.y)
    if n < 10:
        return ds, _empty_like(ds)
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val
    return _slice_dataset(ds, 0, n_train), _slice_dataset(ds, n_train, n)


def _slice_dataset(ds: WindowDataset, start: int, end: int) -> WindowDataset:
    return WindowDataset(
        dates=ds.dates[start:end],
        symbols=ds.symbols[start:end],
        market=ds.market[start:end],
        tech=ds.tech[start:end],
        news_text=ds.news_text[start:end],
        sentiment=ds.sentiment[start:end],
        y=ds.y[start:end],
        open_px=ds.open_px[start:end],
        close_px=ds.close_px[start:end],
        next_close_px=ds.next_close_px[start:end],
        next_date=ds.next_date[start:end],
        tradable_exec=ds.tradable_exec[start:end],
        meta=ds.meta.iloc[start:end].reset_index(drop=True),
    )


def _empty_like(ds: WindowDataset) -> WindowDataset:
    return _slice_dataset(ds, 0, 0)
