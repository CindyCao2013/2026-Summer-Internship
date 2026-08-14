"""AlphaNet vs explicit daily-factor comparison.

Loads a trained AlphaNet synthetic factor (wide date x symbol parquet),
classic style factors from the same EOD panel, and representative
``candidate_pool_v1`` daily factors when ``factor_narrow.parquet`` exists.

Classic styles are always computed from prices/turnover/mcap. Missing pool
panels are recorded as unavailable — never filled with random placeholders.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphanet.config import MAX_WORKERS, EvalConfig
from alphanet.evaluate import decile_backtest, ic_test
from alphanet.neutralize import neutralize_cross_section, style_panels
from alphanet.paths import (
    CANDIDATE_POOL_ROOT,
    COMPARE_ROOT,
    FACTORS,
    L2_RESULT_ROOT,
    ensure_result_dirs,
)

logger = logging.getLogger(__name__)

OVERLAP_LOW = 0.30
OVERLAP_HIGH = 0.60
STYLE_WINDOW = 20
MIN_CS_OBS = 30
MAX_HEATMAP_DATES = 80


def overlap_verdict(max_abs_corr: float) -> Tuple[str, str]:
    """Map max |CS Spearman| to Track A/B guidance."""
    if not np.isfinite(max_abs_corr):
        return "unknown", "无法计算最大相关性，请检查因子覆盖。"
    if max_abs_corr < OVERLAP_LOW:
        return (
            "new_information",
            "AlphaNet 与现有日频因子相关性极低，提供了显著的新信息源。"
            "Track B（AlphaNet）应优先保留；Track A 可作为可解释性代理。",
        )
    if max_abs_corr < OVERLAP_HIGH:
        return (
            "partial_overlap",
            "AlphaNet 与部分因子中度相关，但仍可能存在增量信息。"
            "需结合残差 IC / 分层判断是否保留 Track B。",
        )
    return (
        "likely_remix",
        "AlphaNet 与现有日频因子高度相关，可能仅为非线性重组合。"
        "Track A（显式切割）的可解释性价值上升。",
    )


def _find_col(columns, keys: Tuple[str, ...]):
    lower = {str(c).lower(): c for c in columns}
    for key in keys:
        if key in lower:
            return lower[key]
    return None


def to_wide(df: pd.DataFrame, value_hint: Optional[str] = None) -> pd.DataFrame:
    """Accept wide (date x symbol) or long (date/symbol/value) factor tables."""
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()

    date_col = _find_col(
        frame.columns, ("date", "tradetime", "trade_dt", "tradingday")
    )
    sym_col = _find_col(
        frame.columns, ("symbol", "stock_code", "s_info_windcode", "code")
    )
    val_col = value_hint if value_hint and value_hint in frame.columns else None
    if val_col is None:
        val_col = _find_col(frame.columns, ("factor_value", "value", "factor"))

    if date_col is not None and sym_col is not None:
        values = val_col
        if values is None:
            numeric_cols = [
                c
                for c in frame.select_dtypes(include=[np.number]).columns
                if c not in {date_col, sym_col}
            ]
            if len(numeric_cols) != 1:
                raise ValueError("cannot infer a unique value column for long factor table")
            values = numeric_cols[0]
        out = frame.pivot_table(
            index=date_col, columns=sym_col, values=values, aggfunc="last"
        )
        out.index = pd.to_datetime(out.index).normalize()
        out.index.name = None
        out.columns.name = None
        return out.sort_index()

    if date_col is not None:
        numeric = frame.select_dtypes(include=[np.number]).copy()
        numeric.index = pd.to_datetime(frame[date_col]).normalize()
        numeric.index.name = None
        return numeric.sort_index()

    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("cannot interpret factor table as wide or long")
    numeric.index = pd.to_datetime(frame.index).normalize()
    numeric.index.name = None
    return numeric.sort_index()


def load_alphanet_factor(
    variant: str,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load AlphaNet synthetic factor. Prefers raw (not style-neutral) parquet."""
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [
            FACTORS / "{}_factor.parquet".format(variant),
            FACTORS / "synthetic_factor.parquet",
            FACTORS / "{}_factor_neutral.parquet".format(variant),
        ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "未找到 AlphaNet {} 因子文件。请先运行 run_evaluate.py / run_train.py。"
            " 已尝试: {}".format(variant, tried)
        )
    if "neutral" in found.name:
        logger.warning("using neutralized factor %s; overlap vs styles will be biased low", found)
    df = pd.read_parquet(found)
    wide = to_wide(df)
    logger.info("loaded AlphaNet-%s from %s shape=%s", variant, found, wide.shape)
    wide.attrs["source_path"] = str(found)
    return wide


def build_classic_style_factors(
    panel,
    window: int = STYLE_WINDOW,
) -> Dict[str, pd.DataFrame]:
    """Classic CS style factors from a MarketPanel. No placeholders."""
    ret_1d = panel.ret_1d
    turn = panel.features["turn"] if "turn" in panel.features else None
    if turn is None:
        raise ValueError("panel.features['turn'] is required for classic styles")
    styles = style_panels(ret_1d, turn, int(window))
    out: Dict[str, pd.DataFrame] = {
        "momentum_{}d".format(window): styles["momentum"],
        "volatility_{}d".format(window): styles["volatility"],
        "turnover_{}d".format(window): styles["turnover"],
    }
    if panel.log_mcap is not None:
        out["size"] = panel.log_mcap.reindex_like(ret_1d)
    if "pb" in panel.features:
        pb = panel.features["pb"].replace(0, np.nan)
        out["bp"] = (1.0 / pb).reindex_like(ret_1d)
    return out


def select_pool_representatives(
    summary: pd.DataFrame,
    *,
    max_per_family: int = 1,
    max_total: int = 12,
) -> pd.DataFrame:
    """One (or few) factors per family, skipping near-aliases, ranked by |ICIR|."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    name_col = "factor" if "factor" in df.columns else "name"
    if name_col not in df.columns:
        raise ValueError("candidate_summary must have a factor name column")
    if "near_alias_observed" in df.columns:
        alias = df["near_alias_observed"].astype(str).str.strip().str.lower()
        df = df.loc[~alias.isin(["true", "1", "yes"])]
    score_col = "icir_raw" if "icir_raw" in df.columns else "rank_ic_raw"
    df["_score"] = pd.to_numeric(df.get(score_col, 0.0), errors="coerce").abs().fillna(0.0)
    if "family" not in df.columns:
        return df.sort_values("_score", ascending=False).head(int(max_total))
    blocks = []
    for _, block in df.groupby("family", dropna=False):
        blocks.append(block.nlargest(int(max_per_family), "_score"))
    out = pd.concat(blocks, axis=0) if blocks else df.iloc[0:0]
    return out.sort_values("_score", ascending=False).head(int(max_total)).reset_index(drop=True)


def candidate_narrow_candidates(family: str, name: str) -> List[Path]:
    fam_dir = CANDIDATE_POOL_ROOT / "{}_family".format(family)
    return [
        fam_dir / "factors" / name / "factor_narrow.parquet",
        fam_dir / name / "factor_narrow.parquet",
        L2_RESULT_ROOT / name / "factor_narrow.parquet",
        CANDIDATE_POOL_ROOT / name / "factor_narrow.parquet",
        CANDIDATE_POOL_ROOT / "factors" / name / "factor_narrow.parquet",
    ]


def resolve_factor_narrow_path(family: str, name: str) -> Optional[Path]:
    for path in candidate_narrow_candidates(family, name):
        if path.exists():
            return path
    return None


def _read_narrow_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return to_wide(frame)


def load_pool_factors(
    representatives: pd.DataFrame,
    *,
    max_workers: int = MAX_WORKERS,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load representative pool panels. Missing files stay missing."""
    if representatives is None or representatives.empty:
        status = pd.DataFrame(
            columns=["factor", "family", "path", "status", "shape"]
        )
        return {}, status

    name_col = "factor" if "factor" in representatives.columns else "name"
    jobs = []
    for _, row in representatives.iterrows():
        name = str(row[name_col])
        family = str(row.get("family", ""))
        jobs.append((name, family))

    workers = max(1, min(int(max_workers), 10, len(jobs)))
    loaded: Dict[str, pd.DataFrame] = {}
    rows = []

    def _one(name: str, family: str) -> Tuple[str, str, Optional[pd.DataFrame], str, str]:
        path = resolve_factor_narrow_path(family, name)
        if path is None:
            return name, family, None, "", "missing_narrow"
        try:
            wide = _read_narrow_parquet(path)
        except Exception as exc:
            return name, family, None, str(path), "read_error:{}".format(exc)
        if wide.empty:
            return name, family, None, str(path), "empty"
        return name, family, wide, str(path), "ok"

    if workers == 1:
        results = [_one(n, f) for n, f in jobs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, n, f): (n, f) for n, f in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())

    for name, family, wide, path, status in results:
        if wide is not None:
            loaded[name] = wide
        rows.append(
            {
                "factor": name,
                "family": family,
                "path": path,
                "status": status,
                "shape": "" if wide is None else "{}x{}".format(*wide.shape),
            }
        )
    status_df = pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
    logger.info(
        "pool factors loaded=%s missing=%s",
        int((status_df["status"] == "ok").sum()) if not status_df.empty else 0,
        int((status_df["status"] != "ok").sum()) if not status_df.empty else 0,
    )
    return loaded, status_df


def load_candidate_summary(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else CANDIDATE_POOL_ROOT / "candidate_summary.csv"
    if not csv_path.exists():
        logger.warning("candidate_summary not found: %s", csv_path)
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def daily_cs_spearman(
    left: pd.DataFrame,
    right: pd.DataFrame,
    min_obs: int = MIN_CS_OBS,
) -> pd.Series:
    a = left.copy()
    b = right.reindex(index=a.index, columns=a.columns)
    n = a.notna() & b.notna()
    corr = a.corrwith(b, axis=1, method="spearman")
    return corr.where(n.sum(axis=1) >= int(min_obs))


def correlation_table(
    alphanet: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame],
    min_obs: int = MIN_CS_OBS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Daily CS Spearman of AlphaNet vs each explicit factor; plus mean summary."""
    daily = {}
    for name, panel in factors.items():
        daily[name] = daily_cs_spearman(alphanet, panel, min_obs=min_obs)
    daily_df = pd.DataFrame(daily).sort_index()
    summary = pd.DataFrame(
        {
            "mean_spearman": daily_df.mean(axis=0),
            "mean_abs_spearman": daily_df.abs().mean(axis=0),
            "std_spearman": daily_df.std(axis=0, ddof=1),
            "n_days": daily_df.notna().sum(axis=0),
        }
    ).sort_values("mean_abs_spearman", ascending=False)
    return daily_df, summary


def pairwise_mean_corr(
    panels: Mapping[str, pd.DataFrame],
    min_obs: int = MIN_CS_OBS,
) -> pd.DataFrame:
    names = list(panels)
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        mat.loc[a, a] = 1.0
        for b in names[i + 1 :]:
            series = daily_cs_spearman(panels[a], panels[b], min_obs=min_obs)
            val = float(series.mean()) if series.notna().any() else float("nan")
            mat.loc[a, b] = val
            mat.loc[b, a] = val
    return mat


def residualize_panel(
    signal: pd.DataFrame,
    extras: Mapping[str, pd.DataFrame],
    min_obs: int = MIN_CS_OBS,
) -> pd.DataFrame:
    """Cross-sectional OLS residual of ``signal`` on ``extras`` (no industry)."""
    if not extras:
        return signal.copy()
    aligned = {
        k: v.reindex(index=signal.index, columns=signal.columns) for k, v in extras.items()
    }
    need = max(int(min_obs), 3 * (len(aligned) + 1))
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns, dtype=float)
    for dt in signal.index:
        extra_row = {k: aligned[k].loc[dt] for k in aligned}
        out.loc[dt] = neutralize_cross_section(
            signal.loc[dt],
            None,
            extra_row,
            min_obs=need,
        )
    return out


def _ic_payload(summary: dict) -> dict:
    return {
        "rank_ic_mean": summary.get("rank_ic_mean", float("nan")),
        "rank_ic_std": summary.get("rank_ic_std", float("nan")),
        "icir": summary.get("icir", float("nan")),
        "ic_positive_frac": summary.get("ic_positive_frac", float("nan")),
        "n_cs": summary.get("n_cs", 0),
    }


def _usable_panel(
    panel: pd.DataFrame,
    signal: pd.DataFrame,
    min_obs: int,
    min_days: int = 5,
) -> bool:
    aligned = panel.reindex(index=signal.index, columns=signal.columns)
    n = (aligned.notna() & signal.notna()).sum(axis=1)
    return int((n >= int(min_obs)).sum()) >= int(min_days)


def residual_predictive_tests(
    alphanet: pd.DataFrame,
    styles: Mapping[str, pd.DataFrame],
    pool: Mapping[str, pd.DataFrame],
    ret_1d: pd.DataFrame,
    *,
    eval_cfg: Optional[EvalConfig] = None,
    mask: Optional[pd.DataFrame] = None,
    min_obs: int = MIN_CS_OBS,
) -> Dict[str, object]:
    cfg = eval_cfg or EvalConfig()
    horizon = int(cfg.rebalance_every)
    residual_style = residualize_panel(alphanet, styles, min_obs=min_obs)
    extras_all = {k: v for k, v in styles.items() if _usable_panel(v, alphanet, min_obs)}
    extras_all.update(
        {k: v for k, v in pool.items() if _usable_panel(v, alphanet, min_obs)}
    )
    residual_all = residualize_panel(alphanet, extras_all, min_obs=min_obs)

    raw_ic = ic_test(alphanet, ret_1d, horizon=horizon, mask=mask)
    style_ic = ic_test(residual_style, ret_1d, horizon=horizon, mask=mask)
    all_ic = ic_test(residual_all, ret_1d, horizon=horizon, mask=mask)
    raw_decile = decile_backtest(alphanet, ret_1d, eval_cfg=cfg, mask=mask)
    style_decile = decile_backtest(residual_style, ret_1d, eval_cfg=cfg, mask=mask)
    all_decile = decile_backtest(residual_all, ret_1d, eval_cfg=cfg, mask=mask)

    def _hl(decile: dict) -> dict:
        table = decile["table"]
        row = table.loc[table["group"] == "H-L"].iloc[0]
        return {
            "hl_annu_ret": float(row["annu_ret"]),
            "hl_sharpe": float(row["sharpe"]),
            "monotonicity_spearman": float(decile["monotonicity_spearman"]),
        }

    comparison = pd.DataFrame(
        [
            {"stage": "raw", **_ic_payload(raw_ic["summary"]), **_hl(raw_decile)},
            {
                "stage": "resid_classic_styles",
                **_ic_payload(style_ic["summary"]),
                **_hl(style_decile),
            },
            {
                "stage": "resid_styles_and_pool",
                **_ic_payload(all_ic["summary"]),
                **_hl(all_decile),
            },
        ]
    )
    return {
        "residual_style": residual_style,
        "residual_all": residual_all,
        "comparison": comparison,
        "raw_ic": raw_ic,
        "style_ic": style_ic,
        "all_ic": all_ic,
        "raw_decile": raw_decile,
        "style_decile": style_decile,
        "all_decile": all_decile,
    }


def _downsample_dates(index: pd.Index, max_dates: int = MAX_HEATMAP_DATES) -> pd.Index:
    if len(index) <= max_dates:
        return index
    step = int(np.ceil(len(index) / max_dates))
    return index[::step]


def save_heatmaps(
    daily_corr: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
    variant: str,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if not pairwise.empty:
        fig, ax = plt.subplots(
            figsize=(
                max(7.0, 0.55 * pairwise.shape[1] + 2),
                max(6.0, 0.55 * pairwise.shape[0] + 1),
            )
        )
        im = ax.imshow(pairwise.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(pairwise.columns)))
        ax.set_xticklabels(list(pairwise.columns), rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(pairwise.index)))
        ax.set_yticklabels(list(pairwise.index), fontsize=8)
        ax.set_title("Mean CS Spearman: AlphaNet-{} vs explicit factors".format(variant))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / "alphanet_{}_mean_corr_heatmap.png".format(variant)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["mean_heatmap"] = path

    if not daily_corr.empty:
        plot = daily_corr.loc[_downsample_dates(daily_corr.index)]
        fig, ax = plt.subplots(figsize=(max(10.0, 0.12 * plot.shape[0] + 4), max(5.0, 0.35 * plot.shape[1] + 2)))
        im = ax.imshow(plot.T.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(plot.columns)))
        ax.set_yticklabels(list(plot.columns), fontsize=8)
        ticks = np.linspace(0, plot.shape[0] - 1, num=min(8, plot.shape[0]), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [pd.Timestamp(plot.index[i]).strftime("%Y-%m-%d") for i in ticks],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_title("Daily CS Spearman (AlphaNet-{} vs factors)".format(variant))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Spearman")
        fig.tight_layout()
        path = output_dir / "alphanet_{}_daily_corr_heatmap.png".format(variant)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["daily_heatmap"] = path
    return paths


def _fmt(x, digits: int = 4) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return "{:.{d}f}".format(val, d=digits)


def write_report(
    *,
    variant: str,
    output_dir: Path,
    corr_summary: pd.DataFrame,
    residual: Optional[pd.DataFrame],
    pool_status: pd.DataFrame,
    coverage: Mapping[str, object],
    heatmap_paths: Mapping[str, Path],
    data_note: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_abs = (
        float(corr_summary["mean_abs_spearman"].max())
        if corr_summary is not None and not corr_summary.empty
        else float("nan")
    )
    label, verdict = overlap_verdict(max_abs)
    lines = [
        "# AlphaNet-{} vs 显式日频因子对比报告".format(variant),
        "",
        "**生成时间**: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "**数据说明**: {}".format(data_note),
        "",
        "## 0. 覆盖度",
        "",
        "- AlphaNet 形状: {}".format(coverage.get("alphanet_shape", "NA")),
        "- 风格因子: {}".format(", ".join(coverage.get("style_names", []))),
        "- 候选池代表因子请求数: {}".format(coverage.get("pool_requested", 0)),
        "- 候选池实际加载数: {}".format(coverage.get("pool_loaded", 0)),
        "- 相关分析交易日数: {}".format(coverage.get("n_corr_days", 0)),
        "- 日期交集: {} → {}".format(
            coverage.get("date_min", "NA"), coverage.get("date_max", "NA")
        ),
        "",
        "## 1. 信息重叠度（截面 Spearman）",
        "",
        "- 对比因子数: {}".format(
            0
            if corr_summary is None or corr_summary.empty
            else int((corr_summary["n_days"] > 0).sum())
        ),
        "- **最大平均绝对相关**: {}".format(_fmt(max_abs)),
        "- **判定**: `{}`".format(label),
        "- {}".format(verdict),
        "",
        "阈值：`< 0.30` 新信息；`0.30–0.60` 部分重叠；`> 0.60` 可能是传统因子的非线性重组合。",
        "",
        "### Top |corr|",
        "",
    ]
    if corr_summary is not None and not corr_summary.empty:
        lines.append("| 因子 | mean Spearman | mean |Spearman| | 天数 |")
        lines.append("|---|---:|---:|---:|")
        for name, row in corr_summary.head(15).iterrows():
            lines.append(
                "| `{}` | {} | {} | {} |".format(
                    name,
                    _fmt(row["mean_spearman"]),
                    _fmt(row["mean_abs_spearman"]),
                    int(row["n_days"]),
                )
            )
    else:
        lines.append("（无可用对比因子）")

    lines.extend(["", "## 2. 增量信息（残差 RankIC / 10 层）", ""])
    if residual is not None and not residual.empty:
        lines.append("| 阶段 | RankIC | ICIR | H-L 年化超额 | H-L Sharpe | 单调性 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in residual.iterrows():
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} |".format(
                    row["stage"],
                    _fmt(row["rank_ic_mean"]),
                    _fmt(row["icir"]),
                    _fmt(row["hl_annu_ret"]),
                    _fmt(row["hl_sharpe"]),
                    _fmt(row["monotonicity_spearman"]),
                )
            )
        raw = residual.loc[residual["stage"] == "raw"]
        style = residual.loc[residual["stage"] == "resid_classic_styles"]
        if not raw.empty and not style.empty:
            raw_ic = float(raw["rank_ic_mean"].iloc[0])
            res_ic = float(style["rank_ic_mean"].iloc[0])
            raw_icir = float(raw["icir"].iloc[0])
            res_icir = float(style["icir"].iloc[0])
            lines.append("")
            if not (np.isfinite(raw_ic) and np.isfinite(res_ic)):
                lines.append("残差 RankIC 未能计算，请检查收益面板对齐。")
            elif abs(raw_ic) < 0.02 and abs(res_ic) < 0.02:
                lines.append(
                    "当前样本下原始与残差 RankIC 均很弱，不能据此判断增量预测力。"
                    "短样本 / synthetic 演示尤其如此；请用训练后的 AlphaNet 全样本复评。"
                )
            elif abs(res_ic) < 0.5 * abs(raw_ic):
                lines.append(
                    "剥离经典风格后 RankIC 明显下降，AlphaNet 的线性可解释部分主要来自这些日频风格。"
                )
            elif abs(res_ic) >= 0.02 and np.isfinite(res_icir) and abs(res_icir) >= 0.5:
                lines.append(
                    "剥离市值 / 动量 / 波动 / 换手后，残差仍具有选股能力，"
                    "说明 AlphaNet 捕获了风格因子线性结构之外的信息。"
                )
            else:
                lines.append(
                    "残差 RankIC 与原始接近，但幅度有限，需更长样本确认是否存在稳定增量。"
                )
    else:
        lines.append("未计算残差预测力（缺少收益面板）。")

    lines.extend(["", "## 3. 候选池加载状态", ""])
    if pool_status is None or pool_status.empty:
        lines.append(
            "`candidate_pool_v1` 未提供可用的 `factor_narrow.parquet`。"
            "本次只对比经典风格因子。池内回测统计（`candidate_summary.csv`）不能替代截面相关。"
        )
    else:
        n_ok = int((pool_status["status"] == "ok").sum())
        n_miss = int((pool_status["status"] != "ok").sum())
        lines.append("- 加载成功: {}，缺失/失败: {}".format(n_ok, n_miss))
        lines.append("")
        lines.append("| 因子 | family | 状态 | 重叠天数 |")
        lines.append("|---|---|---|---:|")
        n_no_overlap = 0
        for _, row in pool_status.iterrows():
            n_ov = "NA"
            if corr_summary is not None and row["factor"] in corr_summary.index:
                n_ov = int(corr_summary.loc[row["factor"], "n_days"])
                if int(n_ov) == 0:
                    n_no_overlap += 1
            lines.append(
                "| `{}` | {} | {} | {} |".format(
                    row["factor"], row["family"], row["status"], n_ov
                )
            )
        if n_no_overlap:
            lines.append("")
            lines.append(
                "有 {} 个池因子文件已读到，但与当前 AlphaNet 日历没有足够的共同交易日"
                "（synthetic 默认 2018，候选池从约 2019 起）。"
                "live 评估 overlap 区间后才会进入相关 / 残差。".format(n_no_overlap)
            )

    if heatmap_paths:
        lines.extend(["", "## 4. 图", ""])
        for key, path in heatmap_paths.items():
            lines.append("- {}: `{}`".format(key, path))

    lines.extend(
        [
            "",
            "---",
            "",
            "*由 `alphanet/scripts/run_compare_factors.py` 生成。"
            "SMB/HML/WML 是市场时间序列，不是股票截面因子，故不作为 CS 对比列。"
            "BP 仅在面板含市净率时计算。*",
        ]
    )
    report_path = output_dir / "alphanet_vs_explicit_{}_report.md".format(variant)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote report %s", report_path)
    return report_path


def align_panels(
    alphanet: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Keep AlphaNet dates; reindex each explicit factor onto that grid.

    Pool panels often start later than AlphaNet. Days without overlap become
    NaN and are dropped by ``min_obs`` in the Spearman / OLS steps — they do
    not shrink the classic-style sample.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(alphanet.index)).normalize()
    alpha = alphanet.copy()
    alpha.index = idx
    aligned: Dict[str, pd.DataFrame] = {}
    for name, panel in factors.items():
        p = panel.copy()
        p.index = pd.DatetimeIndex(pd.to_datetime(p.index)).normalize()
        aligned[name] = p.reindex(index=idx, columns=alpha.columns)
    return alpha, aligned


def run_comparison(
    alphanet: pd.DataFrame,
    styles: Mapping[str, pd.DataFrame],
    pool: Mapping[str, pd.DataFrame],
    *,
    variant: str,
    ret_1d: Optional[pd.DataFrame] = None,
    mask: Optional[pd.DataFrame] = None,
    eval_cfg: Optional[EvalConfig] = None,
    output_dir: Optional[Path] = None,
    pool_status: Optional[pd.DataFrame] = None,
    data_note: str = "live",
    min_obs: int = MIN_CS_OBS,
) -> Dict[str, object]:
    ensure_result_dirs()
    out_dir = Path(output_dir) if output_dir is not None else COMPARE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = dict(styles)
    combined.update(pool)
    if not combined:
        raise ValueError("no explicit factors to compare against")
    alpha, aligned = align_panels(alphanet, combined)
    if alpha.empty:
        raise ValueError("AlphaNet 与显式因子日期无交集")

    daily_corr, corr_summary = correlation_table(alpha, aligned, min_obs=min_obs)
    named_for_pair = {"alphanet_{}".format(variant): alpha}
    named_for_pair.update(aligned)
    pairwise = pairwise_mean_corr(named_for_pair, min_obs=min_obs)

    residual_out = None
    residual_cmp = None
    if ret_1d is not None:
        ret = ret_1d.copy()
        ret.index = pd.DatetimeIndex(pd.to_datetime(ret.index)).normalize()
        ret = ret.reindex(index=alpha.index, columns=alpha.columns)
        m = None
        if mask is not None:
            m = mask.copy()
            m.index = pd.DatetimeIndex(pd.to_datetime(m.index)).normalize()
            m = m.reindex(index=alpha.index, columns=alpha.columns)
        style_aligned = {k: aligned[k] for k in styles if k in aligned}
        pool_aligned = {k: aligned[k] for k in pool if k in aligned}
        residual_out = residual_predictive_tests(
            alpha,
            style_aligned,
            pool_aligned,
            ret,
            eval_cfg=eval_cfg,
            mask=m,
            min_obs=min_obs,
        )
        residual_cmp = residual_out["comparison"]
        residual_cmp.to_csv(out_dir / "alphanet_{}_residual_ic.csv".format(variant), index=False)

    corr_summary.to_csv(out_dir / "alphanet_{}_corr_summary.csv".format(variant))
    daily_corr.to_csv(out_dir / "alphanet_{}_daily_corr.csv".format(variant))
    pairwise.to_csv(out_dir / "alphanet_{}_pairwise_mean_corr.csv".format(variant))
    heatmaps = save_heatmaps(daily_corr, pairwise, out_dir, variant)

    status = pool_status if pool_status is not None else pd.DataFrame()
    if not status.empty:
        status.to_csv(out_dir / "alphanet_{}_pool_status.csv".format(variant), index=False)

    coverage = {
        "alphanet_shape": "{}x{}".format(*alpha.shape),
        "style_names": list(styles),
        "pool_requested": 0 if status.empty else int(len(status)),
        "pool_loaded": len(pool),
        "n_corr_days": int(daily_corr.dropna(how="all").shape[0]),
        "date_min": str(pd.Timestamp(alpha.index.min()).date()) if len(alpha.index) else "NA",
        "date_max": str(pd.Timestamp(alpha.index.max()).date()) if len(alpha.index) else "NA",
    }
    report = write_report(
        variant=variant,
        output_dir=out_dir,
        corr_summary=corr_summary,
        residual=residual_cmp,
        pool_status=status,
        coverage=coverage,
        heatmap_paths=heatmaps,
        data_note=data_note,
    )
    return {
        "alphanet": alpha,
        "aligned": aligned,
        "daily_corr": daily_corr,
        "corr_summary": corr_summary,
        "pairwise": pairwise,
        "residual": residual_out,
        "report": report,
        "heatmaps": heatmaps,
        "coverage": coverage,
        "verdict": overlap_verdict(
            float(corr_summary["mean_abs_spearman"].max()) if not corr_summary.empty else float("nan")
        ),
    }


def make_synthetic_alphanet(styles: Mapping[str, pd.DataFrame], seed: int = 7) -> pd.DataFrame:
    """Deterministic mix used only by ``--synthetic`` / unit tests. Not a live factor."""
    mom_key = next((k for k in styles if k.startswith("momentum")), None)
    if mom_key is None:
        raise ValueError("synthetic AlphaNet needs a momentum style panel")
    mom = styles[mom_key]
    cs = mom.sub(mom.mean(axis=1), axis=0)
    cs = cs.div(mom.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(rng.normal(size=mom.shape), index=mom.index, columns=mom.columns)
    return 0.75 * cs + 0.25 * noise
