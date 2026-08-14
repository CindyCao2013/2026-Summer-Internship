"""Fast L2 Discovery Lane — 快速因子发现通道。

架构边界（冻结）：
- 只允许读取已落盘的 daily primitive，禁止访问 Raw Tick / SSL2；
- 回测上下文（收益矩阵 / 可投资 mask / 基准）来自 ``fast_context`` 永久缓存，
  启动时校验 sha256，不得逐因子重新查询 Wind / DolphinDB / ClickHouse；
- 每个 family 的 daily primitive 只加载一次，批量生成多个因子列；
- 回测复用冻结引擎 ``backtest.backtest_factor``（T+1，``signal.shift(1)``，
  十分组等权，统一 universe / benchmark / mask），保证与全量管线数值一致；
- Fast Gate 仅打标（strong_candidate / research_candidate），不做 KEEP/DROP，
  不做参数搜索；未达标不得调参重跑。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Factor_Dev_Lib import calAnnuRet, calMDD, calSharpe
from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.liquidity_impact_factors import (
    LIQUIDITY_IMPACT_FACTOR_NAMES,
    build_liquidity_impact_feature_frame,
)
from l2_factor_reproduction.python.liquidity_impact_factors import (
    feature_to_narrow as liquidity_to_narrow,
)
from l2_factor_reproduction.python.order_book_factors import (
    ORDER_BOOK_FACTOR_NAMES,
    build_order_book_feature_frame,
)
from l2_factor_reproduction.python.order_book_factors import (
    feature_to_narrow as order_book_to_narrow,
)
from l2_factor_reproduction.python.price_formation_factors import (
    PRICE_FORMATION_FACTOR_NAMES,
    build_price_formation_feature_frame,
)
from l2_factor_reproduction.python.price_formation_factors import (
    feature_to_narrow as price_formation_to_narrow,
)
from l2_factor_reproduction.python.trade_flow_factors import (
    TRADE_FLOW_FACTOR_NAMES,
    build_trade_flow_feature_frame,
)
from l2_factor_reproduction.python.trade_flow_factors import (
    feature_to_narrow as trade_flow_to_narrow,
)

# ---------------------------------------------------------------------------
# 冻结窗口
# ---------------------------------------------------------------------------

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2024-12-31")
FULL_START = pd.Timestamp("2019-01-01")
FULL_END = pd.Timestamp("2026-07-31")

WINDOWS: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "discovery": (DISCOVERY_START, DISCOVERY_END),
    "full": (FULL_START, FULL_END),
}

# 滚动类特征（如 trade_flow flow_zscore_20d）需要窗口前缓冲
PRIMITIVE_BUFFER_DAYS = 60

FAST_CONTEXT_DIR = Path(RESULT_ROOT) / "fast_context"
FAST_DISCOVERY_DIR = Path(RESULT_ROOT) / "fast_discovery"

# ---------------------------------------------------------------------------
# Fast Gate 阈值（冻结；只打标，不筛选、不调参）
# ---------------------------------------------------------------------------

STRONG_GATE = {
    "hl_sharpe": 3.0,
    "decile_mono_spearman": 0.85,
    "adjacent_violations": 1,
}
RESEARCH_GATE = {
    "hl_sharpe": 2.0,
    "decile_mono_spearman": 0.75,
    "adjacent_violations": 2,
}

# ---------------------------------------------------------------------------
# Family 适配器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyAdapter:
    """一个 daily primitive family 的批量因子生成配置。"""

    name: str
    primitive_dir: Path
    builder: Callable[[pd.DataFrame], pd.DataFrame]
    to_narrow: Callable[[pd.DataFrame, str], pd.DataFrame]
    factor_names: Tuple[str, ...]

    def partition_files(self) -> List[Path]:
        files = sorted(self.primitive_dir.glob("**/*.parquet"))
        # 排除单日 smoke / 验证月等临时文件：仅保留标准季度/全量文件
        files = [
            path
            for path in files
            if "validation" not in path.name and "smoke" not in path.name
        ]
        files = _drop_nested_partitions(files)
        if not files:
            raise FileNotFoundError(
                f"{self.name}: no primitive parquet under {self.primitive_dir}"
            )
        return files


_PRIMITIVES = Path(RESULT_ROOT) / "primitives"

FAMILY_ADAPTERS: Dict[str, FamilyAdapter] = {
    "liquidity_impact": FamilyAdapter(
        name="liquidity_impact",
        primitive_dir=_PRIMITIVES / "liquidity_impact_daily" / "dataset",
        builder=build_liquidity_impact_feature_frame,
        to_narrow=liquidity_to_narrow,
        factor_names=tuple(LIQUIDITY_IMPACT_FACTOR_NAMES),
    ),
    "price_formation": FamilyAdapter(
        name="price_formation",
        primitive_dir=_PRIMITIVES / "price_formation_daily" / "dataset",
        builder=build_price_formation_feature_frame,
        to_narrow=price_formation_to_narrow,
        factor_names=tuple(PRICE_FORMATION_FACTOR_NAMES),
    ),
    "order_book": FamilyAdapter(
        name="order_book",
        primitive_dir=_PRIMITIVES / "order_book_daily" / "dataset",
        builder=build_order_book_feature_frame,
        to_narrow=order_book_to_narrow,
        factor_names=tuple(ORDER_BOOK_FACTOR_NAMES),
    ),
    "trade_flow": FamilyAdapter(
        name="trade_flow",
        primitive_dir=_PRIMITIVES / "trade_flow_daily" / "chunks",
        builder=build_trade_flow_feature_frame,
        to_narrow=trade_flow_to_narrow,
        factor_names=tuple(TRADE_FLOW_FACTOR_NAMES),
    ),
}


def _partition_date_range(
    path: Path,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    stem = path.stem
    parts = stem.split("_")
    dates = [
        part
        for part in parts
        if len(part) == 10 and part[4] == "-" and part[7] == "-"
    ]
    if len(dates) < 2:
        return None
    return pd.Timestamp(dates[-2]), pd.Timestamp(dates[-1])


def _drop_nested_partitions(files: List[Path]) -> List[Path]:
    """丢弃日期区间被另一文件完全包含的分区（如验证月嵌在季度块中）。"""
    ranges = {path: _partition_date_range(path) for path in files}
    kept: List[Path] = []
    for path in files:
        span = ranges[path]
        if span is None:
            kept.append(path)
            continue
        nested = any(
            other is not path
            and ranges[other] is not None
            and ranges[other][0] <= span[0]
            and span[1] <= ranges[other][1]
            and ranges[other] != span
            for other in files
        )
        if not nested:
            kept.append(path)
    return kept


def _file_window_overlap(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, buffer_days: int
) -> bool:
    """按文件名中的 YYYY-MM-DD_YYYY-MM-DD 区间做剪枝（含滚动缓冲）。"""
    span = _partition_date_range(path)
    if span is None:
        return True  # 无法解析文件名（如 quarter= 目录名）则不剪枝
    file_start, file_end = span
    lo = start - pd.Timedelta(days=buffer_days)
    return file_start <= end and file_end >= lo


def load_family_features(
    family: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    buffer_days: int = PRIMITIVE_BUFFER_DAYS,
) -> pd.DataFrame:
    """加载一次 daily primitive 并批量生成全部冻结因子列。

    返回宽表：symbol / TradeDate / <factor columns...>，已裁剪到 [start, end]。
    滚动特征使用窗口前 ``buffer_days`` 的 primitive 作为预热，预热段不进入输出。
    """
    adapter = FAMILY_ADAPTERS[family]
    files = [
        path
        for path in adapter.partition_files()
        if _file_window_overlap(path, start, end, buffer_days)
    ]
    if not files:
        raise FileNotFoundError(
            f"{family}: no primitive partitions overlap "
            f"[{start.date()}, {end.date()}] (buffer={buffer_days}d)"
        )
    frames = [pd.read_parquet(path) for path in files]
    primitive = pd.concat(frames, ignore_index=True)
    date_col = "TradeDate" if "TradeDate" in primitive.columns else None
    if date_col is None:
        raise ValueError(f"{family}: primitive lacks TradeDate column")
    primitive[date_col] = pd.to_datetime(primitive[date_col])
    lo = start - pd.Timedelta(days=buffer_days)
    primitive = primitive.loc[primitive[date_col].between(lo, end)]
    features = adapter.builder(primitive)
    features["TradeDate"] = pd.to_datetime(features["TradeDate"])
    return features.loc[features["TradeDate"].between(start, end)].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# 上下文缓存
# ---------------------------------------------------------------------------


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def context_paths(window: str) -> Dict[str, Path]:
    if window not in WINDOWS:
        raise KeyError(f"unknown window {window!r}; valid={list(WINDOWS)}")
    base = FAST_CONTEXT_DIR / window
    return {
        "ret_matrix": base / "ret_matrix.parquet",
        "universe_mask": base / "universe_mask.parquet",
        "benchmark_return": base / "benchmark_return.parquet",
        "trading_dates": base / "trading_dates.parquet",
        "manifest": base / "context_manifest.json",
    }


def load_fast_context(
    window: str, *, verify_hash: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取缓存的 (mask, ret)。ret 已为相对 UNIVERSE 的超额 c2c。"""
    paths = context_paths(window)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if verify_hash:
        for key in ("ret_matrix", "universe_mask"):
            recorded = manifest["files"][key]["sha256"]
            actual = _sha256(paths[key])
            if actual != recorded:
                raise RuntimeError(
                    f"fast context hash mismatch for {key} "
                    f"(window={window}): {actual} != {recorded}"
                )
        if manifest.get("universe") != UNIVERSE:
            raise RuntimeError(
                f"fast context universe {manifest.get('universe')} "
                f"!= settings UNIVERSE {UNIVERSE}"
            )
    mask = pd.read_parquet(paths["universe_mask"])
    ret = pd.read_parquet(paths["ret_matrix"])
    mask.index = pd.to_datetime(mask.index)
    ret.index = pd.to_datetime(ret.index)
    return mask, ret


# ---------------------------------------------------------------------------
# Fast 指标
# ---------------------------------------------------------------------------


def compute_fast_metrics(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    summary: Dict[str, object],
) -> Dict[str, float]:
    """由冻结引擎输出计算 Fast Gate 指标。

    ``group_pnl`` 必须已是有效方向（backtest_factor 返回值即满足）：
    G1 = 低有效因子，G10 = 高有效因子，H-L = G10 - G1。
    """
    group_cols = sorted(
        (c for c in group_pnl.columns if c != "H-L"), key=lambda c: int(c)
    )
    decile_annu = np.array(
        [calAnnuRet(group_pnl[c]) for c in group_cols], dtype=float
    )
    ranks = pd.Series(np.arange(1, len(decile_annu) + 1), dtype=float)
    mono = float(ranks.corr(pd.Series(decile_annu), method="spearman"))
    violations = int(np.sum(decile_annu[1:] < decile_annu[:-1]))

    hl = group_pnl["H-L"].dropna()
    hl.index = pd.to_datetime(hl.index)
    monthly = hl.resample("M").sum()
    pos_month = float((monthly > 0).mean()) if len(monthly) else float("nan")
    cum_hl = hl.cumsum()
    if len(cum_hl) > 2 and cum_hl.std() > 0:
        time_spearman = float(
            pd.Series(cum_hl.to_numpy()).corr(
                pd.Series(np.arange(len(cum_hl)), dtype=float),
                method="spearman",
            )
        )
    else:
        time_spearman = float("nan")

    hl_mdd = float(summary.get("hl_mdd_flipped", np.nan))
    if not np.isfinite(hl_mdd):
        hl_mdd, _ = calMDD(hl)

    return {
        "rank_ic_mean_raw": float(summary["rank_ic_mean_raw"]),
        "icir_raw": float(summary["rank_icir"])
        * float(summary["factor_direction"]),
        "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
        "hl_sharpe": float(summary["hl_sharpe_flipped"]),
        "hl_mdd": hl_mdd,
        "g10_excess_sharpe": float(summary["g10_excess_sharpe"]),
        "decile_mono_spearman": mono,
        "adjacent_violations": violations,
        "positive_hl_month_fraction": pos_month,
        "cum_hl_time_spearman": time_spearman,
        "avg_hl_turnover": float(group_to["H-L"].mean()),
        "factor_direction": int(summary["factor_direction"]),
        "n_days": int(summary["n_days"]),
        "n_names_avg": float(summary["n_names_avg"]),
    }


def gate_label(metrics: Dict[str, float]) -> str:
    """Fast Gate 打标；不做 KEEP/DROP，未达标即记录为 none。"""

    def _passes(thresholds: Dict[str, float]) -> bool:
        return (
            metrics["hl_sharpe"] >= thresholds["hl_sharpe"]
            and metrics["decile_mono_spearman"]
            >= thresholds["decile_mono_spearman"]
            and metrics["adjacent_violations"]
            <= thresholds["adjacent_violations"]
        )

    if _passes(STRONG_GATE):
        return "strong_candidate"
    if _passes(RESEARCH_GATE):
        return "research_candidate"
    return "none"


# ---------------------------------------------------------------------------
# 两张正式图（标准交付：方向一致 + 十分组 + H-L）
# ---------------------------------------------------------------------------

PLOT_DPI = 150


def _configure_plot_fonts() -> None:
    """Prefer CJK-capable sans so Chinese titles/labels render correctly."""
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _group_columns(pnl: pd.DataFrame) -> List[str]:
    cols = [c for c in pnl.columns if str(c) != "H-L"]
    return sorted(cols, key=lambda c: int(c))


def ensure_effective_group_pnl(group_pnl: pd.DataFrame) -> pd.DataFrame:
    """绘图前强制有效方向：G1=低因子 … G10=高因子，H-L=G10-G1 上行。

    若传入仍是原始负向（H-L 均值 < 0），则倒转十分组标签并取反 H-L，
    避免「十分组正向而 H-L 负向」的图线矛盾。
    """
    if "H-L" not in group_pnl.columns:
        raise ValueError("group_pnl must contain H-L column")
    pnl = group_pnl.copy()
    pnl.index = pd.to_datetime(pnl.index)
    group_cols = _group_columns(pnl)
    if float(pnl["H-L"].mean()) >= 0:
        # Keep column order G1..G10, H-L without mutating values.
        ordered = pnl.loc[:, group_cols + ["H-L"]]
        ordered.columns = [str(c) for c in ordered.columns]
        return ordered

    n = len(group_cols)
    flipped = pd.DataFrame(index=pnl.index)
    for i, col in enumerate(group_cols):
        flipped[str(n - i)] = pnl[col]
    flipped["H-L"] = -pnl["H-L"]
    # Re-sort numeric group labels 1..n
    ordered_cols = [str(i) for i in range(1, n + 1)] + ["H-L"]
    return flipped.loc[:, ordered_cols]


def _decile_cmap_colors(n: int, *, kind: str = "warm") -> List[tuple]:
    """G1→G10 color gradient. warm=浅蓝→深红；cool=浅蓝→深蓝。"""
    if n <= 0:
        return []
    if kind == "cool":
        cmap = plt.cm.Blues
        return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]
    # warm / diverging: light blue → deep red
    cmap = plt.cm.coolwarm
    return [cmap(0.15 + 0.70 * i / max(n - 1, 1)) for i in range(n)]


def save_fast_plots(
    out_dir: Path,
    factor_name: str,
    group_pnl: pd.DataFrame,
    metrics: Dict[str, float],
) -> Tuple[Path, Path]:
    """标准交付两张图（有效方向，DPI≥150）。

    1. ``cumulative_hl.png`` — G1..G10 + H-L 累计收益（cumsum）；H-L 黑粗线；
       角落标注 Ann.Ret / Sharpe / MaxDD / 日均换手。
    2. ``decile_bar.png`` — G1..G10 + H-L 日均收益柱；G 蓝渐变、H-L 红；
       柱顶数值标签。

    调用前通过 ``ensure_effective_group_pnl`` 保证：
    G1→G10 对应低→高有效因子，H-L = G10 − G1 同步向上。
    """
    _configure_plot_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)
    pnl = ensure_effective_group_pnl(group_pnl)
    group_cols = _group_columns(pnl)
    labels = [f"G{c}" for c in group_cols]

    hl = pnl["H-L"]
    annu = float(metrics.get("hl_annu_ret", calAnnuRet(hl)))
    sharpe = float(metrics.get("hl_sharpe", calSharpe(hl)))
    mdd = float(metrics.get("hl_mdd", np.nan))
    if not np.isfinite(mdd):
        mdd, _ = calMDD(hl)
    turnover = float(metrics.get("avg_hl_turnover", np.nan))

    # --- 图1：累计收益曲线 ---
    cum = pnl.cumsum()
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    colors = _decile_cmap_colors(len(group_cols), kind="warm")
    for col, color, label in zip(group_cols, colors, labels):
        ax1.plot(
            cum.index,
            cum[col].to_numpy(),
            color=color,
            linewidth=1.15,
            alpha=0.85,
            label=label,
        )
    ax1.plot(
        cum.index,
        cum["H-L"].to_numpy(),
        color="black",
        linewidth=2.6,
        alpha=0.95,
        label="H-L",
        zorder=5,
    )
    ax1.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_title(
        f"{factor_name} — 十分组累计收益 "
        f"(G1=低因子值 … G10=高因子值；已按因子方向调整)",
        fontsize=13,
    )
    ax1.set_xlabel("TradeDate")
    ax1.set_ylabel("Cumulative return (cumsum)")
    ax1.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_facecolor("white")
    fig1.patch.set_facecolor("white")

    to_txt = f"{turnover:.2f}" if np.isfinite(turnover) else "n/a"
    stats_text = (
        f"H-L 年化收益 (Ann. Ret): {annu:.2%}\n"
        f"H-L 夏普比率 (Sharpe):   {sharpe:.2f}\n"
        f"H-L 最大回撤 (Max DD):   {mdd:.2%}\n"
        f"换手（日均）:            {to_txt}"
    )
    # Place box in the quieter corner (opposite of H-L end level).
    box_loc = "lower right" if float(cum["H-L"].iloc[-1]) >= 0 else "upper right"
    ax1.text(
        0.98 if "right" in box_loc else 0.02,
        0.02 if "lower" in box_loc else 0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        va="bottom" if "lower" in box_loc else "top",
        ha="right" if "right" in box_loc else "left",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.4",
            "alpha": 0.92,
        },
        zorder=6,
    )
    fig1.tight_layout()
    path_cum = out_dir / "cumulative_hl.png"
    fig1.savefig(path_cum, dpi=PLOT_DPI)
    plt.close(fig1)

    # --- 图2：单调性柱状图 ---
    means = pnl[group_cols + ["H-L"]].mean()
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    bar_colors = _decile_cmap_colors(len(group_cols), kind="cool") + [
        "#C44E52"
    ]
    x_labels = labels + ["H-L"]
    values = means.to_numpy(dtype=float)
    bars = ax2.bar(x_labels, values, color=bar_colors, width=0.75)
    ax2.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
    ax2.set_title(
        f"{factor_name} — 各组日均收益 (单调性检验)",
        fontsize=13,
    )
    ax2.set_xlabel("Decile (G1=低因子值 … G10=高因子值)")
    ax2.set_ylabel("Mean daily return")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_facecolor("white")
    fig2.patch.set_facecolor("white")

    y_span = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    offset = 0.02 * y_span if y_span > 0 else 1e-5
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        # Prefer bp when magnitude is small; otherwise percent-like decimals.
        if abs(val) < 0.01:
            label = f"{val * 1e4:.1f}bp"
        else:
            label = f"{val:.4f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (offset if val >= 0 else -offset),
            label,
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    fig2.tight_layout()
    path_bar = out_dir / "decile_bar.png"
    fig2.savefig(path_bar, dpi=PLOT_DPI)
    plt.close(fig2)
    return path_cum, path_bar


# ---------------------------------------------------------------------------
# 批量 Runner（含分阶段计时）
# ---------------------------------------------------------------------------

PROFILE_COLUMNS = (
    "family",
    "factor",
    "window",
    "primitive_load_seconds",
    "factor_compute_seconds",
    "context_load_seconds",
    "backtest_seconds",
    "plot_seconds",
    "total_seconds",
)


def run_fast_batch(
    family: str,
    factors: Iterable[str],
    *,
    window: str = "discovery",
    output_root: Optional[Path] = None,
    context: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """一个 family：primitive 只加载一次，批量回测多个因子。

    Returns
    -------
    summary rows, profile rows
    """
    start, end = WINDOWS[window]
    out_root = (
        Path(output_root) if output_root else FAST_DISCOVERY_DIR / window
    )
    adapter = FAMILY_ADAPTERS[family]
    names = list(factors)
    unknown = sorted(set(names).difference(adapter.factor_names))
    if unknown:
        raise KeyError(f"{family}: unknown factors {unknown}")

    t0 = time.perf_counter()
    features = load_family_features(family, start, end)
    primitive_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    if context is None:
        context = load_fast_context(window)
    mask, ret = context
    context_load_seconds = time.perf_counter() - t0

    summary_rows: List[Dict[str, object]] = []
    profile_rows: List[Dict[str, object]] = []
    for name in names:
        t_factor0 = time.perf_counter()
        narrow = adapter.to_narrow(features, name)
        factor_compute_seconds = time.perf_counter() - t_factor0

        t_bt = time.perf_counter()
        group_pnl, group_to, rank_ic, summary = backtest_factor(
            narrow,
            start_day=start,
            end_day=end,
            mask=mask,
            ret_matrix=ret,
        )
        backtest_seconds = time.perf_counter() - t_bt

        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        metrics["gate"] = gate_label(metrics)

        t_plot = time.perf_counter()
        save_fast_plots(out_root / "figures" / name, name, group_pnl, metrics)
        plot_seconds = time.perf_counter() - t_plot

        summary_rows.append(
            {"factor": name, "family": family, "window": window, **metrics}
        )
        profile_rows.append(
            {
                "family": family,
                "factor": name,
                "window": window,
                "primitive_load_seconds": round(primitive_load_seconds, 3),
                "factor_compute_seconds": round(factor_compute_seconds, 3),
                "context_load_seconds": round(context_load_seconds, 3),
                "backtest_seconds": round(backtest_seconds, 3),
                "plot_seconds": round(plot_seconds, 3),
                "total_seconds": round(
                    primitive_load_seconds / max(len(names), 1)
                    + context_load_seconds / max(len(names), 1)
                    + factor_compute_seconds
                    + backtest_seconds
                    + plot_seconds,
                    3,
                ),
            }
        )
        print(
            f"[fast] {family}/{name}: sharpe={metrics['hl_sharpe']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.2f} "
            f"gate={metrics['gate']} "
            f"(bt={backtest_seconds:.1f}s)",
            flush=True,
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(profile_rows)
