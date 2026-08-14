"""Intraday timing heatmap diagnostics for daily (HF→EOD) factors.

Stamps a Date×Symbol panel onto standard bartimes, joins a minute return
matrix, and builds Bartime × Horizon Rank-IC / HML heatmaps.

Pure-pandas path works offline (tests / mock). Optional DDB path reuses
``intraday_lib.get_cs_group_performance`` when a live session is provided.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

DEFAULT_BARTIMES: List[str] = ["09:59", "10:29", "11:29", "13:29", "14:29"]
DEFAULT_HORIZONS: List[str] = [
    "Ret_15",
    "Ret_30",
    "Ret_60",
    "Ret_90",
    "Ret_120",
    "Ret_150",
    "Ret_180",
    "Ret_EOD",
]

# factor name → search roots under research/cache
FACTOR_CACHE_GLOBS: Dict[str, List[Path]] = {
    "TGD20": [Path("research/cache/tgd_panels")],
    "SmartMoneyActiveV2": [Path("research/cache/smart_money_active_v2/factor_panel")],
    "APM_ActiveV2": [Path("research/cache/apm_active_v2/factor_panel")],
    "IdealAmplitude_ActiveV2": [Path("research/cache/ideal_amplitude_active_v2/factor_panel")],
    "IdealReversal_ActiveV2": [Path("research/cache/ideal_reversal_active_v2/factor_panel")],
}

UNIVERSE_INDEX = {
    "ALL": None,
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
}


def parse_bartime_label(label: str) -> Tuple[int, int]:
    h, m = label.strip().split(":")
    return int(h), int(m)


def bartime_to_label(t: Union[dt.time, str, pd.Timestamp]) -> str:
    if isinstance(t, str) and re.match(r"^\d{1,2}:\d{2}$", t):
        hh, mm = parse_bartime_label(t)
        return f"{hh:02d}:{mm:02d}"
    if isinstance(t, pd.Timestamp):
        return f"{t.hour:02d}:{t.minute:02d}"
    if isinstance(t, dt.time):
        return f"{t.hour:02d}:{t.minute:02d}"
    # timedelta / second-of-day from DDB
    s = str(t)
    if ":" in s:
        parts = s.split(":")
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    raise TypeError(f"Cannot parse bartime label from {t!r}")


def load_factor_panel_from_cache(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    *,
    cache_roots: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Load best-matching wide panel parquet for ``factor_name``.

    Prefers files whose name starts with ``factor_name`` (excluding ``*_long_*``),
    covering [start, end] as much as possible (largest file among matches).
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    roots = list(cache_roots) if cache_roots is not None else FACTOR_CACHE_GLOBS.get(
        factor_name, [Path("research/cache")]
    )
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.parquet"):
            name = path.name
            if "_long_" in name or name.startswith(f"{factor_name}_long"):
                continue
            if not name.startswith(factor_name):
                continue
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No wide panel cache for {factor_name} under {[str(r) for r in roots]}"
        )

    # Prefer largest file that overlaps the window (heuristic for full-range caches)
    def _score(p: Path) -> Tuple[int, int]:
        return (p.stat().st_size, len(p.name))

    candidates = sorted(candidates, key=_score, reverse=True)
    last_err: Optional[Exception] = None
    for path in candidates:
        try:
            wide = pd.read_parquet(path)
            wide.index = pd.to_datetime(wide.index)
            wide = wide.sort_index()
            sliced = wide.loc[start_ts:end_ts]
            if sliced.dropna(how="all").empty:
                continue
            print(
                f"[heatmap] loaded {factor_name} <- {path} "
                f"({sliced.index.min().date()}→{sliced.index.max().date()}, "
                f"{sliced.shape[0]}d × {sliced.shape[1]} sym)",
                flush=True,
            )
            return sliced
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise FileNotFoundError(
        f"Could not load usable panel for {factor_name}: {last_err}"
    )


def stamp_panel_to_narrow(
    panel: pd.DataFrame,
    factor_name: str,
    bartimes: Sequence[str] = DEFAULT_BARTIMES,
    *,
    shift_days: int = 1,
) -> pd.DataFrame:
    """Wide Date×Symbol → narrow tradetime/symbol/factorname/value.

    ``shift_days=1`` (default): T-day signal used at T+1 bartimes (avoid look-ahead
    vs same-day minute returns), matching ``run_p2_intraday_heatmap``.
    """
    wide = panel.copy()
    wide.index = pd.to_datetime(wide.index)
    if shift_days:
        wide = wide.shift(shift_days)
    wide = wide.dropna(how="all")
    stacked = wide.stack()
    stacked = stacked.reset_index()
    stacked.columns = ["Date", "symbol", "value"]
    stacked["Date"] = pd.to_datetime(stacked["Date"])
    stacked = stacked.dropna(subset=["value"])
    stacked["symbol"] = stacked["symbol"].astype(str)

    pieces = []
    for label in bartimes:
        h, m = parse_bartime_label(label)
        part = stacked.copy()
        part["tradetime"] = part["Date"] + pd.Timedelta(hours=h, minutes=m)
        part["factorname"] = factor_name
        pieces.append(part[["tradetime", "symbol", "factorname", "value"]])
    out = pd.concat(pieces, ignore_index=True)
    out["value"] = out["value"].astype(float)
    return out


def apply_not_limit_mask(
    panel: pd.DataFrame,
    not_limit: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Mask limit-up/down days (not_limit==1 keeps; NaN → drop signal)."""
    if not_limit is None or panel.empty:
        return panel
    nl = not_limit.reindex(index=panel.index, columns=panel.columns)
    out = panel.where(nl == 1)
    return out


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def compute_ic_hml_matrices(
    narrow: pd.DataFrame,
    ret_long: pd.DataFrame,
    *,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    bartimes: Sequence[str] = DEFAULT_BARTIMES,
    group_num: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pure-pandas Rank IC and HML (top−bottom mean) matrices.

    Parameters
    ----------
    narrow:
        Columns: tradetime, symbol, value (factorname optional).
    ret_long:
        Columns: Symbol, Date, Bartime (time-like), + horizon columns Ret_*.
        Bartime may be time / str / datetime.
    """
    sig = narrow.copy()
    sig["tradetime"] = pd.to_datetime(sig["tradetime"])
    sig["Date"] = sig["tradetime"].dt.normalize()
    sig["BartimeLabel"] = sig["tradetime"].dt.strftime("%H:%M")
    sig["symbol"] = sig["symbol"].astype(str)

    ret = ret_long.copy()
    ret["Date"] = pd.to_datetime(ret["Date"]).dt.normalize()
    # unify symbol column
    if "Symbol" in ret.columns and "symbol" not in ret.columns:
        ret = ret.rename(columns={"Symbol": "symbol"})
    ret["symbol"] = ret["symbol"].astype(str)
    if "Bartime" in ret.columns:
        ret["BartimeLabel"] = ret["Bartime"].map(bartime_to_label)
    elif "BartimeLabel" not in ret.columns:
        raise ValueError("ret_long needs Bartime or BartimeLabel")

    merged = sig.merge(
        ret,
        on=["symbol", "Date", "BartimeLabel"],
        how="inner",
    )
    if merged.empty:
        ic = pd.DataFrame(index=list(bartimes), columns=list(horizons), dtype=float)
        hml = ic.copy()
        return ic, hml

    ic_rows = []
    hml_rows = []
    for bt in bartimes:
        bt_lab = bartime_to_label(bt)
        sub = merged[merged["BartimeLabel"] == bt_lab]
        ic_rec = {"Bartime": bt_lab}
        hml_rec = {"Bartime": bt_lab}
        for hz in horizons:
            if hz not in sub.columns:
                ic_rec[hz] = np.nan
                hml_rec[hz] = np.nan
                continue
            # IC: mean daily spearman
            daily_ics = []
            daily_hmls = []
            for _, day_df in sub.groupby("Date", sort=False):
                daily_ics.append(_rank_ic(day_df["value"], day_df[hz]))
                # HML via quantile groups
                try:
                    q = pd.qcut(
                        day_df["value"].rank(method="first"),
                        group_num,
                        labels=False,
                        duplicates="drop",
                    )
                except ValueError:
                    daily_hmls.append(np.nan)
                    continue
                tmp = day_df.assign(_g=q)
                if tmp["_g"].nunique() < 2:
                    daily_hmls.append(np.nan)
                    continue
                g_hi = tmp.loc[tmp["_g"] == tmp["_g"].max(), hz].mean()
                g_lo = tmp.loc[tmp["_g"] == tmp["_g"].min(), hz].mean()
                daily_hmls.append(float(g_hi - g_lo))
            ic_rec[hz] = float(np.nanmean(daily_ics)) if daily_ics else np.nan
            hml_rec[hz] = float(np.nanmean(daily_hmls)) if daily_hmls else np.nan
        ic_rows.append(ic_rec)
        hml_rows.append(hml_rec)

    ic_df = pd.DataFrame(ic_rows).set_index("Bartime")
    hml_df = pd.DataFrame(hml_rows).set_index("Bartime")
    # ensure column order
    ic_df = ic_df.reindex(columns=list(horizons))
    hml_df = hml_df.reindex(columns=list(horizons))
    return ic_df, hml_df


def plot_heatmap_matrix(
    matrix: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    fmt: str = ".3f",
    cbar_label: str = "",
) -> Path:
    """Save PNG (+ PDF) heatmap for a Bartime × Horizon matrix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(8, 1.1 * matrix.shape[1]), max(4, 0.7 * matrix.shape[0]))
    )
    data = matrix.astype(float)
    if sns is not None:
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            center=center,
            annot=True,
            fmt=fmt,
            linewidths=0.4,
            cbar_kws={"label": cbar_label} if cbar_label else None,
        )
    else:
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(list(data.index))
        fig.colorbar(im, ax=ax, label=cbar_label)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Bartime")
    fig.tight_layout()
    png = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=140)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def summarize_factor_timing(
    factor_name: str,
    universe: str,
    start: DateLike,
    end: DateLike,
    ic_df: pd.DataFrame,
    hml_df: pd.DataFrame,
) -> str:
    """Build a short text report for one factor."""
    lines = [
        f"因子: {factor_name}, 宇宙: {universe}, "
        f"区间: {pd.Timestamp(start).date()} ~ {pd.Timestamp(end).date()}",
    ]
    if ic_df.empty or not np.isfinite(ic_df.to_numpy(dtype=float)).any():
        lines.append("  (无有效 IC)")
        return "\n".join(lines)

    # best IC cell
    stacked = ic_df.stack()
    stacked = stacked[np.isfinite(stacked)]
    if stacked.empty:
        lines.append("  (无有效 IC)")
        return "\n".join(lines)
    best_bt, best_hz = stacked.idxmax()
    best_ic = float(stacked.max())
    best_hml = float(hml_df.loc[best_bt, best_hz]) if best_hz in hml_df.columns else np.nan
    lines.append(
        f"最佳时点: {best_bt} | {best_hz} | IC={best_ic:.4f}, "
        f"HML={best_hml * 100:.2f}%"
    )
    # mean IC by bartime
    mean_by_bt = ic_df.mean(axis=1, skipna=True)
    stab = ", ".join(f"{idx} IC均值={val:.4f}" for idx, val in mean_by_bt.items())
    lines.append(f"时点稳定性: {stab}")
    return "\n".join(lines)


def run_factor_heatmap_offline(
    factor_name: str,
    panel: pd.DataFrame,
    ret_long: pd.DataFrame,
    *,
    out_dir: Path,
    universe: str = "ALL",
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    bartimes: Sequence[str] = DEFAULT_BARTIMES,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    not_limit: Optional[pd.DataFrame] = None,
    shift_days: int = 1,
) -> Dict[str, Path]:
    """End-to-end offline diagnostic for one factor panel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fac_dir = out_dir / factor_name
    fac_dir.mkdir(parents=True, exist_ok=True)

    wide = panel.copy()
    wide.index = pd.to_datetime(wide.index)
    if start is not None:
        wide = wide.loc[pd.Timestamp(start) :]
    if end is not None:
        wide = wide.loc[: pd.Timestamp(end)]
    wide = apply_not_limit_mask(wide, not_limit)

    narrow = stamp_panel_to_narrow(
        wide, factor_name, bartimes=bartimes, shift_days=shift_days
    )
    ic_df, hml_df = compute_ic_hml_matrices(
        narrow, ret_long, horizons=horizons, bartimes=bartimes
    )

    ic_csv = fac_dir / "ic_matrix.csv"
    hml_csv = fac_dir / "hml_matrix.csv"
    ic_df.to_csv(ic_csv)
    hml_df.to_csv(hml_csv)

    ic_png = plot_heatmap_matrix(
        ic_df,
        title=f"{factor_name} Rank IC ({universe})",
        out_path=fac_dir / "rank_ic_heatmap.png",
        cbar_label="Rank IC",
    )
    hml_png = plot_heatmap_matrix(
        hml_df * 10000.0,  # bps for readability
        title=f"{factor_name} HML mean return bps ({universe})",
        out_path=fac_dir / "hml_heatmap.png",
        fmt=".1f",
        cbar_label="HML (bps)",
    )

    s0 = start or wide.index.min()
    s1 = end or wide.index.max()
    summary = summarize_factor_timing(factor_name, universe, s0, s1, ic_df, hml_df)
    (fac_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary, flush=True)

    return {
        "ic_csv": ic_csv,
        "hml_csv": hml_csv,
        "ic_png": ic_png,
        "hml_png": hml_png,
        "summary": fac_dir / "summary.txt",
    }


def run_ddb_group_heatmap(
    session,
    narrow: pd.DataFrame,
    *,
    factor_name: str,
    index_code: str,
    share_name: str,
    out_dir: Path,
    ret_columns: Sequence[str],
) -> None:
    """Optional live path: upload narrow → get_cs_group_performance → PNG heatmaps."""
    import intraday_lib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    upload = re.sub(r"[^A-Za-z0-9_]", "_", factor_name)
    session.upload({upload: narrow})
    session.run(
        f"""
        signal = {upload}
        index_code = "{index_code}"
        signal = select *, date(tradetime) as Date from signal
        signal = filter_in_index(signal, index_code)
        group_data_ret, summary = get_cs_group_performance(signal, {share_name}, group_num=5)
        """
    )
    group_data_ret = session.run("group_data_ret")
    group_data_ret = intraday_lib.subtract_market_return(group_data_ret)
    group_data_ret.to_parquet(out_dir / "group_data_ret.parquet", index=False)
    perf = intraday_lib.analyze_group_performance_by_bartime(
        group_data_ret,
        ret_columns=list(ret_columns),
        save_plots=True,
        show_plots=False,
        save_path=str(out_dir),
    )
    intraday_lib.create_group_heatmap(
        perf,
        group_name="group_HML",
        key_name="annualized_return",
        save_plot=True,
        show_plot=False,
        save_path=str(out_dir),
    )
    intraday_lib.create_group_heatmap(
        perf,
        group_name="group_HML",
        key_name="sharpe",
        save_plot=True,
        show_plot=False,
        high_contrast=True,
        save_path=str(out_dir),
    )
    intraday_lib.save_performance_summary(
        perf, filename=str(out_dir / "group_performance_summary.csv")
    )
