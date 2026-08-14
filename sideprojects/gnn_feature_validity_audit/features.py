"""Independent implementations of the frozen atomic features and transforms."""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_adapter import DataBundle, DataUnavailableError


PRICE_VOLUME_FACTORS = [
    "ret",
    "log_vol",
    "amplitude",
    "turnover",
    "gap",
    "dist_limit_up",
    "dist_limit_down",
    "excess_ret",
    "mom_5d",
    "mom_20d",
    "volatility_20d",
    "macd",
    "rsi",
    "money_flow",
]

FUNDAMENTAL_FACTORS = [
    "pe_ttm",
    "pb",
    "roe_ttm",
    "market_cap",
    "revenue_growth_yoy",
    "profit_growth_yoy",
    "debt_ratio",
]

SENTIMENT_FACTORS = [
    "news_sentiment",
    "lhb_flag",
    "block_trade_premium",
]

MACRO_FACTORS = ["gdp_yoy", "cpi_yoy", "pmi", "m2_yoy"]

RELATION_FACTORS = [
    "degree_centrality",
    "pagerank",
    "dtw_similarity_mean",
    "industry_excess_ret",
]


@dataclass
class FeatureBuildResult:
    panels: Dict[str, pd.DataFrame]
    data_status: Dict[str, str]
    reasons: Dict[str, str]
    graph_diagnostics: pd.DataFrame


def _wide(
    frame: pd.DataFrame,
    value: str,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if frame.empty or value not in frame.columns:
        return pd.DataFrame(index=dates, columns=list(symbols), dtype=float)
    result = frame.pivot_table(
        index="date", columns="symbol", values=value, aggfunc="last"
    )
    result.index = pd.to_datetime(result.index)
    return result.reindex(index=dates, columns=list(symbols))


def _finite(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.astype(float).replace([np.inf, -np.inf], np.nan)


def build_price_volume_features(bundle: DataBundle) -> Dict[str, pd.DataFrame]:
    market = bundle.market.copy().sort_values(["symbol", "date"])
    for column in (
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "limit_up",
        "limit_down",
    ):
        market[column] = pd.to_numeric(market[column], errors="coerce")

    positive_preclose = market["pre_close"].where(market["pre_close"] > 0)
    positive_close = market["close"].where(market["close"] > 0)
    market["ret"] = market["close"] / positive_preclose - 1.0
    market["log_vol"] = np.log(market["volume"].clip(lower=0) + 1.0)
    market["amplitude"] = (market["high"] - market["low"]) / positive_preclose
    market["gap"] = market["open"] / positive_preclose - 1.0
    market["dist_limit_up"] = (market["limit_up"] - market["close"]) / positive_close
    market["dist_limit_down"] = (
        market["close"] - market["limit_down"]
    ) / positive_close

    grouped = market.groupby("symbol", sort=False)
    market["mom_5d"] = market["close"] / grouped["close"].shift(5) - 1.0
    market["mom_20d"] = market["close"] / grouped["close"].shift(20) - 1.0
    market["volatility_20d"] = (
        grouped["ret"]
        .rolling(20, min_periods=5)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    ema12 = grouped["close"].transform(
        lambda values: values.ewm(span=12, adjust=False).mean()
    )
    ema26 = grouped["close"].transform(
        lambda values: values.ewm(span=26, adjust=False).mean()
    )
    market["macd"] = (ema12 - ema26) / positive_close
    market["money_flow"] = market["amount"] / grouped["amount"].shift(5) - 1.0

    gain = market["ret"].where(market["ret"].notna()).clip(lower=0)
    loss = (-market["ret"].where(market["ret"].notna())).clip(lower=0)
    avg_gain = (
        gain.groupby(market["symbol"], sort=False)
        .rolling(14, min_periods=14)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    avg_loss = (
        loss.groupby(market["symbol"], sort=False)
        .rolling(14, min_periods=14)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 50.0)
    market["rsi"] = rsi

    dates = bundle.sample_dates
    symbols = bundle.symbols
    panels = {
        factor: _finite(_wide(market, factor, dates, symbols))
        for factor in PRICE_VOLUME_FACTORS
        if factor not in ("turnover", "excess_ret")
    }
    panels["turnover"] = _finite(
        bundle.derivative_wide("turnover").reindex(index=dates, columns=symbols)
    )
    stock_ret = panels["ret"]
    panels["excess_ret"] = _finite(
        stock_ret.sub(bundle.index_returns.reindex(dates), axis=0)
    )
    return {name: panels[name] for name in PRICE_VOLUME_FACTORS}


def asof_events_to_daily(
    events: pd.DataFrame,
    *,
    value_column: str,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    available_column: str = "available_date",
) -> pd.DataFrame:
    """Backward-asof event values onto a daily Date×Symbol grid."""
    out = pd.DataFrame(index=dates, columns=list(symbols), dtype=float)
    if events.empty or value_column not in events.columns:
        return out
    event_frame = events[
        ["symbol", available_column, value_column]
    ].copy()
    event_frame[available_column] = pd.to_datetime(
        event_frame[available_column], errors="coerce"
    )
    event_frame[value_column] = pd.to_numeric(
        event_frame[value_column], errors="coerce"
    )
    event_frame = event_frame.dropna(
        subset=["symbol", available_column, value_column]
    )
    groups = {
        str(symbol): group
        for symbol, group in event_frame.groupby("symbol", sort=False)
    }
    for symbol in symbols:
        group = groups.get(str(symbol))
        if group is None or group.empty:
            continue
        series = (
            group.sort_values(available_column)
            .drop_duplicates(available_column, keep="last")
            .set_index(available_column)[value_column]
        )
        union = dates.union(pd.DatetimeIndex(series.index)).sort_values()
        out[symbol] = series.reindex(union).ffill().reindex(dates).to_numpy()
    return _finite(out)


def _pit_yoy_events(financial: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Compute YoY on announcement events without selecting a future revision."""
    required = {
        "symbol",
        "ann_date",
        "available_date",
        "report_period",
        value_column,
    }
    if financial.empty or not required.issubset(financial.columns):
        return pd.DataFrame(
            columns=["symbol", "available_date", "report_period", "yoy"]
        )
    fin = financial[list(required)].copy()
    for column in ("ann_date", "available_date", "report_period"):
        fin[column] = pd.to_datetime(fin[column], errors="coerce")
    fin[value_column] = pd.to_numeric(fin[value_column], errors="coerce")
    fin = fin.dropna(
        subset=[
            "symbol",
            "ann_date",
            "available_date",
            "report_period",
            value_column,
        ]
    ).sort_values(["symbol", "available_date", "report_period"])

    output: List[Dict[str, object]] = []
    for symbol, group in fin.groupby("symbol", sort=False):
        records = group.reset_index(drop=True)
        by_period: Dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(period): rows
            for period, rows in records.groupby("report_period", sort=False)
        }
        for row in records.itertuples(index=False):
            current_period = pd.Timestamp(row.report_period)
            prior_period = current_period - pd.DateOffset(years=1)
            candidates = by_period.get(prior_period)
            if candidates is None:
                continue
            visible = candidates[
                candidates["available_date"] <= pd.Timestamp(row.available_date)
            ]
            if visible.empty:
                continue
            prior = visible.sort_values("available_date").iloc[-1]
            denominator = float(prior[value_column])
            # Executable source masks non-positive lag values.
            if not np.isfinite(denominator) or denominator <= 0:
                continue
            current = float(getattr(row, value_column))
            value = current / denominator - 1.0
            if np.isfinite(value):
                output.append(
                    {
                        "symbol": str(symbol),
                        "available_date": pd.Timestamp(row.available_date),
                        "report_period": current_period,
                        "yoy": value,
                    }
                )
    return pd.DataFrame(output)


def build_fundamental_features(bundle: DataBundle) -> Dict[str, pd.DataFrame]:
    dates = bundle.sample_dates
    symbols = bundle.symbols
    financial = bundle.financial.copy()
    panels: Dict[str, pd.DataFrame] = {
        "pe_ttm": _finite(
            bundle.derivative_wide("pe_ttm").reindex(index=dates, columns=symbols)
        ),
        "pb": _finite(
            bundle.derivative_wide("pb").reindex(index=dates, columns=symbols)
        ),
        "market_cap": _finite(
            bundle.derivative_wide("market_cap").reindex(
                index=dates, columns=symbols
            )
        ),
        "roe_ttm": asof_events_to_daily(
            financial,
            value_column="roe_ttm",
            dates=dates,
            symbols=symbols,
        ),
        "debt_ratio": asof_events_to_daily(
            financial,
            value_column="debt_ratio",
            dates=dates,
            symbols=symbols,
        ),
    }
    revenue_events = _pit_yoy_events(financial, "revenue_ttm")
    profit_events = _pit_yoy_events(financial, "profit_ttm")
    panels["revenue_growth_yoy"] = asof_events_to_daily(
        revenue_events,
        value_column="yoy",
        dates=dates,
        symbols=symbols,
    )
    panels["profit_growth_yoy"] = asof_events_to_daily(
        profit_events,
        value_column="yoy",
        dates=dates,
        symbols=symbols,
    )
    return {name: panels[name] for name in FUNDAMENTAL_FACTORS}


def build_industry_excess_return(bundle: DataBundle) -> pd.DataFrame:
    dates = bundle.sample_dates
    symbols = bundle.symbols
    stock_excess = bundle.stock_returns.reindex(
        index=dates, columns=symbols
    ).sub(bundle.index_returns.reindex(dates), axis=0)
    industry = bundle.industry.reindex(index=dates, columns=symbols)
    eligible = bundle.eligible_mask.reindex(index=dates, columns=symbols)
    output = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for date in dates:
        frame = pd.DataFrame(
            {
                "excess": stock_excess.loc[date],
                "industry": industry.loc[date],
                "eligible": eligible.loc[date].fillna(False),
            }
        )
        frame = frame.loc[frame["eligible"]].dropna(
            subset=["excess", "industry"]
        )
        if frame.empty:
            continue
        means = frame.groupby("industry")["excess"].mean()
        output.loc[date, frame.index] = frame["industry"].map(means).to_numpy()
    return _finite(output)


def filter_edges_asof(edges: pd.DataFrame, asof_date) -> pd.DataFrame:
    """Return only edges effective on asof_date; future edges are excluded."""
    required = {"source", "target", "effective_from"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError("Edges missing columns: {}".format(sorted(missing)))
    asof = pd.Timestamp(asof_date)
    frame = edges.copy()
    frame["effective_from"] = pd.to_datetime(
        frame["effective_from"], errors="coerce"
    )
    active = frame["effective_from"].notna() & (frame["effective_from"] <= asof)
    if "effective_to" in frame.columns:
        frame["effective_to"] = pd.to_datetime(
            frame["effective_to"], errors="coerce"
        )
        active &= frame["effective_to"].isna() | (frame["effective_to"] >= asof)
    result = frame.loc[active].copy()
    if (result["effective_from"] > asof).any():
        raise AssertionError("Future graph edge passed as-of filter")
    return result


def assert_no_future_edges(edges: pd.DataFrame, graph_date) -> None:
    if edges.empty:
        return
    effective = pd.to_datetime(edges["effective_from"], errors="coerce")
    if (effective > pd.Timestamp(graph_date)).any():
        raise AssertionError("Relation graph contains future edges")


def _dtw_distance_python(x: np.ndarray, y: np.ndarray, band: int) -> float:
    length = len(x)
    previous = np.full(length + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, length + 1):
        current = np.full(length + 1, np.inf)
        lo = max(1, i - band)
        hi = min(length, i + band)
        for j in range(lo, hi + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            current[j] = cost + min(
                previous[j], current[j - 1], previous[j - 1]
            )
        previous = current
    return float(np.sqrt(previous[length]))


_NATIVE_DTW_SOURCE = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <omp.h>

extern "C" int dtw_distance_matrix(
    const double* values,
    int n_stocks,
    int length,
    int band,
    int max_threads,
    double* output
) {
    if (!values || !output || n_stocks < 0 || length <= 0 || band < 1) {
        return 1;
    }
    const double infinity = std::numeric_limits<double>::infinity();
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, std::min(max_threads, 10)));
    #pragma omp parallel
    {
        std::vector<double> previous(length + 1, infinity);
        std::vector<double> current(length + 1, infinity);
        #pragma omp for schedule(dynamic, 1)
        for (int left = 0; left < n_stocks; ++left) {
            output[static_cast<std::size_t>(left) * n_stocks + left] = 0.0;
            for (int right = left + 1; right < n_stocks; ++right) {
                std::fill(previous.begin(), previous.end(), infinity);
                previous[0] = 0.0;
                for (int i = 1; i <= length; ++i) {
                    std::fill(current.begin(), current.end(), infinity);
                    const int lo = std::max(1, i - band);
                    const int hi = std::min(length, i + band);
                    const double x = values[
                        static_cast<std::size_t>(left) * length + (i - 1)
                    ];
                    for (int j = lo; j <= hi; ++j) {
                        const double y = values[
                            static_cast<std::size_t>(right) * length + (j - 1)
                        ];
                        const double difference = x - y;
                        const double cost = difference * difference;
                        current[j] = cost + std::min(
                            previous[j],
                            std::min(current[j - 1], previous[j - 1])
                        );
                    }
                    previous.swap(current);
                }
                const double distance = std::sqrt(previous[length]);
                output[static_cast<std::size_t>(left) * n_stocks + right] = distance;
                output[static_cast<std::size_t>(right) * n_stocks + left] = distance;
            }
        }
    }
    return 0;
}
"""


def _load_native_dtw_library() -> ctypes.CDLL:
    """Compile the independent exact-DTW kernel into a temporary shared object."""
    source_hash = hashlib.sha256(_NATIVE_DTW_SOURCE.encode("utf-8")).hexdigest()[:16]
    build_dir = Path(tempfile.gettempdir()) / "gnn_feature_validity_audit_dtw"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "dtw_{}.cpp".format(source_hash)
    library_path = build_dir / "dtw_{}.so".format(source_hash)
    if not library_path.exists():
        source_path.write_text(_NATIVE_DTW_SOURCE, encoding="utf-8")
        command = [
            "g++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fopenmp",
            str(source_path),
            "-o",
            str(library_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "")
            raise DataUnavailableError(
                "Native exact-DTW compilation failed: {}".format(stderr or exc)
            ) from exc
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise DataUnavailableError(
            "Native exact-DTW library load failed: {}".format(exc)
        ) from exc
    function = library.dtw_distance_matrix
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_int
    return library


def _native_dtw_distance_matrix(
    values: np.ndarray,
    band: int,
    max_workers: int,
) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.double)
    output = np.zeros(
        (contiguous.shape[0], contiguous.shape[0]), dtype=np.double
    )
    library = _load_native_dtw_library()
    status = library.dtw_distance_matrix(
        contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(contiguous.shape[0]),
        int(contiguous.shape[1]),
        int(band),
        int(min(max(1, max_workers), 10)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise DataUnavailableError(
            "Native exact-DTW kernel returned status {}".format(status)
        )
    return output


def _dtw_distance_matrix(
    values: np.ndarray,
    band: int,
    max_workers: int,
) -> np.ndarray:
    n_stocks = values.shape[0]
    if n_stocks <= 100:
        result = np.zeros((n_stocks, n_stocks), dtype=float)
        for left in range(n_stocks):
            for right in range(left + 1, n_stocks):
                distance = _dtw_distance_python(
                    values[left], values[right], band
                )
                result[left, right] = distance
                result[right, left] = distance
        return result
    try:
        from dtaidistance import dtw

        # parallel=False guarantees this optional library creates no workers.
        result = dtw.distance_matrix_fast(
            values.astype(np.double),
            window=max(1, int(band)),
            compact=False,
            parallel=False,
            use_c=True,
        )
        return np.asarray(result, dtype=float)
    except ImportError:
        return _native_dtw_distance_matrix(values, band, max_workers)


def _dtw_similarity_matrix(
    values: np.ndarray,
    band: int,
    max_workers: int,
) -> np.ndarray:
    distance = _dtw_distance_matrix(values, band, max_workers)
    norm = np.sqrt(np.sum(values * values, axis=1))
    sq = np.maximum(
        norm[:, None] ** 2
        + norm[None, :] ** 2
        - 2.0 * np.dot(values, values.T),
        0.0,
    )
    direct_distance = np.sqrt(sq)
    denominator = direct_distance + norm[:, None] + norm[None, :]
    similarity = np.ones_like(distance)
    valid = denominator > 1e-8
    similarity[valid] = 1.0 - distance[valid] / denominator[valid]
    similarity = np.maximum(similarity, 0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _source_pagerank(
    adjacency: np.ndarray,
    alpha: float,
    max_iter: int = 100,
) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.array([], dtype=float)
    score = np.full(n_nodes, 1.0 / n_nodes, dtype=float)
    degree = adjacency.sum(axis=1)
    transition = adjacency / np.where(degree > 0, degree, 1.0)[:, None]
    for _ in range(max_iter):
        new_score = alpha * transition.T.dot(score) + (1.0 - alpha) / n_nodes
        if np.abs(new_score - score).sum() < 1e-8:
            score = new_score
            break
        score = new_score
    return score


def _relation_snapshot(
    history: pd.DataFrame,
    industry: pd.Series,
    *,
    top_k: int,
    band_fraction: float,
    correlation_threshold: float,
    pagerank_alpha: float,
    max_workers: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    minimum_observations = max(10, int(np.ceil(history.shape[0] * 0.80)))
    valid_symbols = history.notna().sum(axis=0)
    symbols = sorted(valid_symbols[valid_symbols >= minimum_observations].index)
    if len(symbols) < 2:
        raise DataUnavailableError("Fewer than two nodes have sufficient graph history")
    values = history[symbols].fillna(0.0).to_numpy(dtype=float).T
    n_nodes, lookback = values.shape
    band = max(1, int(lookback * band_fraction))
    similarity = _dtw_similarity_matrix(values, band, max_workers)
    k = min(int(top_k), n_nodes - 1)
    if k < 1:
        raise DataUnavailableError("DTW graph has no possible neighbour")
    top_indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
    top_mask = np.zeros_like(similarity, dtype=bool)
    rows = np.arange(n_nodes)[:, None]
    top_mask[rows, top_indices] = True
    top_mask &= similarity > 0
    dtw_edge_mask = top_mask | top_mask.T
    dtw_adjacency = np.where(dtw_edge_mask, similarity, 0.0)
    dtw_mean = np.where(
        top_mask.sum(axis=1) > 0,
        np.sum(np.where(top_mask, similarity, 0.0), axis=1)
        / np.maximum(top_mask.sum(axis=1), 1),
        np.nan,
    )

    correlation = np.nan_to_num(np.corrcoef(values), nan=0.0)
    correlation = np.abs(correlation)
    np.fill_diagonal(correlation, 0.0)
    correlation_adjacency = np.where(
        correlation >= correlation_threshold, correlation, 0.0
    )

    industry_adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    industry_values = industry.reindex(symbols)
    for _, members in industry_values.dropna().groupby(industry_values.dropna()):
        positions = [symbols.index(symbol) for symbol in members.index]
        if len(positions) < 2:
            continue
        weight = 0.5 / np.sqrt(len(positions))
        indexer = np.ix_(positions, positions)
        industry_adjacency[indexer] += weight
        industry_adjacency[positions, positions] = 0.0

    raw_adjacency = industry_adjacency + dtw_adjacency + correlation_adjacency
    np.fill_diagonal(raw_adjacency, 0.0)
    raw_degree = raw_adjacency.sum(axis=1)
    if not np.any(raw_degree > 0):
        raise DataUnavailableError("Relation graph is all zero")
    inverse_root = np.where(raw_degree > 0, 1.0 / np.sqrt(raw_degree), 0.0)
    adjacency = (
        inverse_root[:, None] * raw_adjacency * inverse_root[None, :]
    )
    degree = adjacency.sum(axis=1)
    maximum_degree = float(degree.max())
    if maximum_degree <= 0:
        raise DataUnavailableError("Normalized relation graph has zero degree")
    degree_feature = degree / maximum_degree
    pagerank = _source_pagerank(adjacency, pagerank_alpha)

    features = pd.DataFrame(
        {
            "symbol": symbols,
            "degree_centrality": degree_feature,
            "pagerank": pagerank,
            "dtw_similarity_mean": dtw_mean,
        }
    )
    edge_count = int(np.count_nonzero(np.triu(raw_adjacency > 0, k=1)))
    diagnostics: Dict[str, object] = {
        "node_count": int(n_nodes),
        "edge_count": edge_count,
        "isolated_node_count": int(np.sum(raw_degree == 0)),
        "isolated_node_ratio": float(np.mean(raw_degree == 0)),
        "industry_edge_count": int(
            np.count_nonzero(np.triu(industry_adjacency > 0, k=1))
        ),
        "dtw_edge_count": int(
            np.count_nonzero(np.triu(dtw_adjacency > 0, k=1))
        ),
        "correlation_edge_count": int(
            np.count_nonzero(np.triu(correlation_adjacency > 0, k=1))
        ),
    }
    return features, diagnostics


def relation_features_vary_over_time(
    panels: Mapping[str, pd.DataFrame],
) -> bool:
    """Detect scan-date vector broadcasting across an entire history."""
    if not panels:
        return False
    for panel in panels.values():
        valid = panel.dropna(how="all")
        if len(valid) < 2:
            return False
        hashes = pd.util.hash_pandas_object(valid, index=False).groupby(
            valid.index
        ).sum()
        if hashes.nunique(dropna=True) <= 1:
            return False
    return True


def build_dynamic_graph_features(
    bundle: DataBundle,
    config: Mapping[str, object],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    relation_cfg = config["relation"]
    dates = bundle.sample_dates
    symbols = bundle.symbols
    panels = {
        name: pd.DataFrame(index=dates, columns=symbols, dtype=float)
        for name in (
            "degree_centrality",
            "pagerank",
            "dtw_similarity_mean",
        )
    }
    snapshot_membership = pd.DataFrame(
        np.nan, index=dates, columns=symbols, dtype=float
    )
    refresh = int(relation_cfg["refresh_every_n_trading_days"])
    lookback = int(relation_cfg["historical_return_lookback"])
    diagnostics: List[Dict[str, object]] = []
    returns = bundle.stock_returns.reindex(columns=symbols)

    for position in range(0, len(dates), refresh):
        graph_date = dates[position]
        available_dates = returns.index[returns.index <= graph_date]
        history_dates = available_dates[-lookback:]
        if len(history_dates) < lookback:
            continue
        nodes = bundle.eligible_mask.loc[graph_date]
        node_symbols = sorted(nodes[nodes].index)
        history = returns.reindex(index=history_dates, columns=node_symbols)
        snapshot, diag = _relation_snapshot(
            history,
            bundle.industry.loc[graph_date, node_symbols],
            top_k=int(relation_cfg["dtw_top_k"]),
            band_fraction=float(relation_cfg["dtw_band_fraction"]),
            correlation_threshold=float(
                relation_cfg["pearson_absolute_threshold"]
            ),
            pagerank_alpha=float(relation_cfg["pagerank_alpha"]),
            max_workers=int(relation_cfg["max_parallel_workers"]),
        )
        snapshot_symbols = snapshot["symbol"].astype(str).tolist()
        snapshot_membership.loc[graph_date, :] = 0.0
        snapshot_membership.loc[graph_date, snapshot_symbols] = 1.0
        for name in panels:
            values = snapshot.set_index("symbol")[name]
            panels[name].loc[graph_date, :] = np.nan
            panels[name].loc[graph_date, values.index] = values
        diag["graph_date"] = graph_date
        diag["history_start"] = history_dates.min()
        diag["history_end"] = history_dates.max()
        diag["max_input_date"] = history_dates.max()
        if pd.Timestamp(diag["max_input_date"]) > graph_date:
            raise AssertionError("Dynamic graph consumed a future return")
        diagnostics.append(diag)

    active_membership = snapshot_membership.ffill().fillna(0.0).astype(bool)
    for name, panel in panels.items():
        panels[name] = panel.ffill().where(active_membership)
    return panels, pd.DataFrame(diagnostics)


def company_mad_tanh(
    panel: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Cross-sectional median-MAD tanh transform used by the company framework."""
    values = _finite(panel)
    median = values.median(axis=1)
    absolute = values.sub(median, axis=0).abs()
    mad = absolute.median(axis=1)
    width = mad * 1.4826 * float(threshold)
    scaled = values.sub(median, axis=0).div(width.replace(0, np.nan), axis=0)
    transformed = np.tanh(scaled * 1.212)
    output = transformed.mul(width, axis=0).add(median, axis=0)
    unchanged = width.isna() | (width == 0)
    output.loc[unchanged] = values.loc[unchanged]
    return output


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=1).replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


def neutralize_panel(
    panel: pd.DataFrame,
    industry: pd.DataFrame,
    market_cap: pd.DataFrame,
    *,
    mode: str,
    min_observations: int,
) -> pd.DataFrame:
    if mode == "none":
        return panel.copy()
    output = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    cap = np.log(market_cap.where(market_cap > 0))
    cap = cross_sectional_zscore(company_mad_tanh(cap))
    industry = industry.reindex_like(panel)
    cap = cap.reindex_like(panel)
    for date in panel.index:
        factor = panel.loc[date]
        valid = factor.notna() & cap.loc[date].notna()
        if mode == "industry+log_market_cap":
            valid &= industry.loc[date].notna()
        if int(valid.sum()) < int(min_observations):
            continue
        y = factor.loc[valid].astype(float).to_numpy()
        columns = [np.ones(len(y)), cap.loc[date, valid].astype(float).to_numpy()]
        if mode == "industry+log_market_cap":
            dummies = pd.get_dummies(
                industry.loc[date, valid].astype(str), dtype=float
            )
            columns.extend(
                dummies[column].to_numpy(dtype=float)
                for column in dummies.columns
            )
        elif mode != "log_market_cap":
            raise ValueError("Unknown neutralization mode: {}".format(mode))
        design = np.column_stack(columns)
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design.dot(coefficients)
        output.loc[date, valid] = residual
    return output


def preprocess_factor(
    panel: pd.DataFrame,
    *,
    neutralization: str,
    industry: pd.DataFrame,
    market_cap: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    preprocessing = config["preprocessing"]
    transformed = company_mad_tanh(
        _finite(panel), threshold=float(preprocessing["mad_threshold"])
    )
    transformed = cross_sectional_zscore(transformed)
    transformed = neutralize_panel(
        transformed,
        industry,
        market_cap,
        mode=neutralization,
        min_observations=int(
            preprocessing["neutralization_min_observations"]
        ),
    )
    # Fixed final scale makes equal-weight composites truly equal-weight.
    return cross_sectional_zscore(transformed)


def build_equal_weight_composite(
    panels: Mapping[str, pd.DataFrame],
    members: Sequence[str],
) -> pd.DataFrame:
    if len(members) < 2:
        raise ValueError("A family composite requires at least two members")
    missing = [member for member in members if member not in panels]
    if missing:
        raise ValueError("Composite panels missing: {}".format(missing))
    stacked = pd.concat(
        [panels[member].stack(dropna=False).rename(member) for member in members],
        axis=1,
    )
    # Complete-case mean prevents date-varying implicit member weights.
    composite = stacked.mean(axis=1, skipna=False)
    return composite.unstack().reindex_like(panels[members[0]])


def build_all_atomic_features(
    bundle: DataBundle,
    config: Mapping[str, object],
    *,
    include_graph: bool = True,
) -> FeatureBuildResult:
    panels: Dict[str, pd.DataFrame] = {}
    status: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    diagnostics = pd.DataFrame()

    price = build_price_volume_features(bundle)
    fundamental = build_fundamental_features(bundle)
    panels.update(price)
    panels.update(fundamental)
    for name in price:
        status[name] = "AVAILABLE"
    for name in fundamental:
        status[name] = "AVAILABLE"

    unavailable = config["data"]["unavailable_sources"]
    for name in SENTIMENT_FACTORS:
        status[name] = "DATA_UNAVAILABLE"
        reasons[name] = str(unavailable[name])
    for name in MACRO_FACTORS:
        status[name] = "NOT_TESTABLE_CROSS_SECTIONALLY"
        reasons[name] = "Same value for every stock on a date"

    panels["industry_excess_ret"] = build_industry_excess_return(bundle)
    status["industry_excess_ret"] = "AVAILABLE"
    if include_graph and config["relation"]["enabled"]:
        try:
            graph_panels, diagnostics = build_dynamic_graph_features(
                bundle, config
            )
            for name, panel in graph_panels.items():
                if (
                    config["relation"]["require_time_variation"]
                    and not relation_features_vary_over_time({name: panel})
                ):
                    status[name] = "DATA_UNAVAILABLE"
                    reasons[name] = (
                        "Relation feature does not vary over time; "
                        "scan-date broadcast rejected"
                    )
                else:
                    panels[name] = panel
                    status[name] = "AVAILABLE"
        except (DataUnavailableError, RuntimeError, ValueError) as exc:
            for name in (
                "degree_centrality",
                "pagerank",
                "dtw_similarity_mean",
            ):
                status[name] = "DATA_UNAVAILABLE"
                reasons[name] = str(exc)
    else:
        for name in (
            "degree_centrality",
            "pagerank",
            "dtw_similarity_mean",
        ):
            status[name] = "DATA_UNAVAILABLE"
            reasons[name] = "Dynamic graph construction was not requested"

    return FeatureBuildResult(
        panels=panels,
        data_status=status,
        reasons=reasons,
        graph_diagnostics=diagnostics,
    )
