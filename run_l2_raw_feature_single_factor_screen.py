#!/usr/bin/env python3
"""
L2 分钟数据 → 海量原始特征 → IC/ICIR 初筛 → 十分组单调性验证

依赖：Factor_Dev_Lib.py、按年 parquet（缺失时自动从 DolphinDB 拉取）

默认初筛区间：2024-01-01 → 2025-12-31
默认按年处理：每年只加载一年分钟数据 → 日频面板，合并后再做 IC/分组。

数据文件约定（相对 --parquet 基名）：
  intraday_2024.parquet
  intraday_2025.parquet
若仅有单体 intraday.parquet，会按年切分写出上述文件。

运行示例：
  python run_l2_raw_feature_single_factor_screen.py
  python run_l2_raw_feature_single_factor_screen.py --max-factors 50
  python run_l2_raw_feature_single_factor_screen.py --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 非交互后端，批量输出图表
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib as fdl

warnings.filterwarnings("ignore", category=RuntimeWarning)

EPS = 1e-8

# 默认初筛区间（贴近当前市场状态，约两年）
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"

# ---------- 全局配置 ----------
ROLL_WINDOWS = (5, 10, 30)
ROLL_OPS = ("mean", "std", "max", "min", "skew", "last_diff")
ROLL_OBJECTS = [
    "Volume",
    "Amount",
    "Active_buy_volume",
    "Active_sell_volume",
    "ActiveBuyVolRatio",
    "ActiveSellVolRatio",
    "ActiveVolImbalance",
    "ActiveAmtImbalance",
    "CancelVolumeImbalance",
    "BidCancelVolRatio",
    "AskCancelVolRatio",
    "LogReturn",
]
BASE_FEATS = [
    "ActiveBuyVolRatio",
    "ActiveSellVolRatio",
    "ActiveBuyAmtRatio",
    "ActiveSellAmtRatio",
    "ActiveBuyCountRatio",
    "ActiveVolImbalance",
    "ActiveAmtImbalance",
    "ActiveCountImbalance",
    "CancelVolumeImbalance",
    "CancelCountImbalance",
    "BidCancelVolRatio",
    "AskCancelVolRatio",
    "LogReturn",
    "HighLowSpread",
    "Vol_Amt_Ratio",
]
BIG_FEATS = [
    "BigBuyRatio_roll30",
    "BigSellRatio_roll30",
    "BigBuySellImbalance_roll30",
]
# 大单代理 rolling(240) + roll30 shift → 只需上月尾部若干 bar，不必整月 concat
LOOKBACK_BARS = 250
DEFAULT_DAILY_CACHE_ROOT = Path("research/cache/l2_daily_panels")

L2_COLUMNS = [
    "Symbol",
    "Date",
    "Bartime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Amount",
    "Active_buy_volume",
    "Active_sell_volume",
    "Active_buy_amount",
    "Active_sell_amount",
    "Active_buy_count",
    "Active_sell_count",
    "Bid_cancel_volume",
    "Bid_cancel_count",
    "Ask_cancel_volume",
    "Ask_cancel_count",
    "Adjfactor",
]


# ---------- 工具函数 ----------
def to_wind_code(symbol) -> str:
    """将 DolphinDB 的 Symbol 转换为 Wind 代码（与 Factor_Dev_Lib 对齐）"""
    s = str(symbol).strip()
    if not s or s.lower() == "nan":
        return s
    if "." in s:
        return s
    if s.startswith(("5", "6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def _l2_from_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """MinuteBarStore 小写列 → 本脚本 L2 列名。"""
    from minute_bar_store import COLUMN_MAP

    rev = {v: k for k, v in COLUMN_MAP.items()}
    out = df.rename(columns={c: rev[c] for c in df.columns if c in rev})
    keep = [c for c in L2_COLUMNS if c in out.columns]
    return out[keep].copy()


def _load_minute_month_from_store(
    m0: dt.datetime,
    m1: dt.datetime,
    prev_m0: Optional[dt.datetime],
    *,
    lookback_bars: int = LOOKBACK_BARS,
) -> pd.DataFrame:
    """通过 MinuteBarStore 从 DDB 加载当月 + 上月尾部 lookback。"""
    from minute_bar_store import get_default_store

    store = get_default_store()
    t0 = time.perf_counter()
    cur = _l2_from_canonical(store.get_data(m0, m1))
    if prev_m0 is not None:
        prev_end = m0 - dt.timedelta(days=1)
        prev_start = prev_end - dt.timedelta(days=7)
        prev = _l2_from_canonical(store.get_data(prev_start, prev_end))
        prev = _tail_per_symbol(prev, lookback_bars)
        raw = pd.concat([prev, cur], axis=0, ignore_index=True)
        del prev, cur
    else:
        raw = cur
    print(
        f"    MinuteBarStore load {m0.strftime('%Y%m')}: {len(raw):,} rows "
        f"({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return raw


def _read_minute_parquet(path: Path, *, from_store: bool = False) -> pd.DataFrame:
    t0 = time.perf_counter()
    if from_store:
        df = pd.read_parquet(path)
        df = _l2_from_canonical(df)
    else:
        df = pd.read_parquet(path, columns=L2_COLUMNS)
    print(
        f"    read {path.name}: {len(df):,} rows ({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return df


def _tail_per_symbol(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    return df.groupby("Symbol", sort=False).tail(n).reset_index(drop=True)


def _load_month_with_lookback(
    cur_path: Path,
    prev_path: Optional[Path],
    *,
    cur_from_store: bool = False,
    prev_from_store: bool = False,
    lookback_bars: int = LOOKBACK_BARS,
) -> pd.DataFrame:
    """读当月 + 上月尾部 lookback（替代整月 concat，显著降内存/IO）。"""
    cur = _read_minute_parquet(cur_path, from_store=cur_from_store)
    if prev_path is not None and prev_path.exists():
        prev = _read_minute_parquet(prev_path, from_store=prev_from_store)
        prev = _tail_per_symbol(prev, lookback_bars)
        raw = pd.concat([prev, cur], axis=0, ignore_index=True)
        del prev, cur
    else:
        raw = cur
    return raw


def _month_parquet_path(
    m0: dt.datetime,
    year_parquet: Path,
    use_store: bool,
) -> Path:
    del use_store  # legacy parquet path only
    ym = m0.strftime("%Y%m")
    return year_parquet.parent / f".tmp_{year_parquet.stem}_months" / f"{ym}.parquet"


def _ensure_month_parquet(
    m0: dt.datetime,
    m1: dt.datetime,
    year_parquet: Path,
    use_store: bool,
) -> Tuple[Optional[Path], bool]:
    """确保月数据存在；返回 (path 或 None, from_store)。"""
    if use_store:
        return None, True

    path = _month_parquet_path(m0, year_parquet, use_store=False)
    if path.exists():
        return path, False

    print(f"    缺失 {m0.strftime('%Y%m')}.parquet，legacy DDB 拉取 ...", flush=True)
    pull_intraday_from_ddb(m0, m1, path)
    return path, False


def _daily_cache_dir_for(feat_list: List[str], root: Path) -> Path:
    sig = hashlib.md5("\n".join(feat_list).encode()).hexdigest()[:10]
    d = root / f"n{len(feat_list)}_{sig}"
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "feat_list.txt"
    if not meta.exists():
        meta.write_text("\n".join(feat_list), encoding="utf-8")
    return d


def _load_daily_cache(cache_path: Path) -> Optional[Dict[str, pd.DataFrame]]:
    if not cache_path.exists():
        return None
    t0 = time.perf_counter()
    with cache_path.open("rb") as f:
        panels = pickle.load(f)
    print(
        f"    daily cache hit {cache_path.name} "
        f"({len(panels)} feats, {time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return panels


def _save_daily_cache(cache_path: Path, panels: Dict[str, pd.DataFrame]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with cache_path.open("wb") as f:
        pickle.dump(panels, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"    daily cache saved {cache_path.name} "
        f"({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )


def filter_continuous_auction(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留连续竞价时段（剔除集合竞价）"""
    t = df["Bartime"].dt.time
    am = (t >= dt.time(9, 31)) & (t <= dt.time(11, 30))
    pm = (t >= dt.time(13, 1)) & (t <= dt.time(15, 0))
    return df.loc[am | pm].copy()


def _month_ends(start_day: dt.datetime, end_day: dt.datetime) -> List[Tuple[dt.datetime, dt.datetime]]:
    """按自然月切分 [start, end]（含端点）。"""
    chunks: List[Tuple[dt.datetime, dt.datetime]] = []
    cur = dt.datetime(start_day.year, start_day.month, 1)
    while cur <= end_day:
        if cur.month == 12:
            nxt = dt.datetime(cur.year + 1, 1, 1)
        else:
            nxt = dt.datetime(cur.year, cur.month + 1, 1)
        c0 = max(cur, start_day)
        c1 = min(nxt - dt.timedelta(days=1), end_day)
        if c0 <= c1:
            chunks.append((c0, c1))
        cur = nxt
    return chunks


def pull_intraday_from_ddb(
    start_day: dt.datetime,
    end_day: dt.datetime,
    out_path: Path,
) -> pd.DataFrame:
    """从 DolphinDB 按月拉取 L2 分钟数据，合并后保存为 parquet（避免整年 OOM）。"""
    import dolphindb as ddb
    from COMMON_CONST import DATA_DB_CONN

    cols = ", ".join(L2_COLUMNS)
    chunks = _month_ends(start_day, end_day)
    print(
        f"[DDB] 按月拉取 Stock_one_minute {start_day.date()} → {end_day.date()} "
        f"({len(chunks)} months) → {out_path}",
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f".tmp_{out_path.stem}_months"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    month_files: List[Path] = []
    sess = ddb.session()
    sess.connect(**DATA_DB_CONN)
    try:
        for i, (c0, c1) in enumerate(chunks, 1):
            s = c0.strftime("%Y.%m.%d")
            e = c1.strftime("%Y.%m.%d")
            mpath = tmp_dir / f"{c0.strftime('%Y%m')}.parquet"
            if mpath.exists():
                dmin, dmax = _parquet_date_range(mpath)
                if _coverage_ok(dmin, dmax, c0, c1, tol_days=0):
                    print(f"  [{i}/{len(chunks)}] reuse {mpath.name}", flush=True)
                    month_files.append(mpath)
                    continue
            script = f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
select {cols}
from t
where Date >= {s} and Date <= {e}
"""
            print(f"  [{i}/{len(chunks)}] pull {s} → {e} ...", flush=True)
            part = sess.run(script)
            if part is None or len(part) == 0:
                print(f"  [{i}/{len(chunks)}] empty, skip", flush=True)
                continue
            part = pd.DataFrame(part)
            part.to_parquet(mpath, index=False)
            print(f"  [{i}/{len(chunks)}] saved {len(part):,} rows → {mpath.name}", flush=True)
            month_files.append(mpath)
    finally:
        sess.close()

    if not month_files:
        raise RuntimeError("DolphinDB 按月拉取结果为空，请检查日期范围与库表权限")

    print(f"[DDB] 合并 {len(month_files)} 个月文件 → {out_path} ...", flush=True)
    try:
        import pyarrow.parquet as pq

        writer = None
        n_rows = 0
        for p in month_files:
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_rows += table.num_rows
        if writer is not None:
            writer.close()
        print(f"[DDB] 已保存 {n_rows:,} 行 → {out_path}", flush=True)
        # 不在此处整表读入内存（全年可 >3e8 行）；由调用方按需 load
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        print(f"[DDB] pyarrow 合并失败 ({exc})，回退 pandas concat", flush=True)
        frames = [pd.read_parquet(p) for p in month_files]
        df = pd.concat(frames, axis=0, ignore_index=True)
        df.to_parquet(out_path, index=False)
        print(f"[DDB] 已保存 {len(df):,} 行 → {out_path}", flush=True)
        return df


def year_parquet_path(base: Path, year: int) -> Path:
    """intraday.parquet → intraday_2024.parquet（同目录）。"""
    return base.parent / f"{base.stem}_{year}{base.suffix}"


def _parquet_date_range(path: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    try:
        d = pd.read_parquet(path, columns=["Date"])
        dd = pd.to_datetime(d["Date"])
        return dd.min(), dd.max()
    except Exception:  # noqa: BLE001
        return None, None


def _coverage_ok(
    dmin: Optional[pd.Timestamp],
    dmax: Optional[pd.Timestamp],
    start_day: dt.datetime,
    end_day: dt.datetime,
    tol_days: int = 5,
) -> bool:
    if dmin is None or dmax is None or pd.isna(dmin) or pd.isna(dmax):
        return False
    tol = pd.Timedelta(days=tol_days)
    return (dmin <= pd.Timestamp(start_day) + tol) and (
        dmax >= pd.Timestamp(end_day) - tol
    )


def ensure_year_parquet(
    base: Path,
    year: int,
    year_start: dt.datetime,
    year_end: dt.datetime,
) -> Path:
    """确保存在该年 parquet：已有且覆盖足够则复用；否则从单体切分或按月 DDB 拉取。"""
    ypath = year_parquet_path(base, year)
    dmin, dmax = _parquet_date_range(ypath) if ypath.exists() else (None, None)
    if ypath.exists() and _coverage_ok(dmin, dmax, year_start, year_end):
        print(
            f"[YEAR] 复用 {ypath.name} ({dmin.date()} → {dmax.date()})",
            flush=True,
        )
        return ypath

    # 尝试从单体 intraday.parquet 切年（可能只有部分月份）
    if base.exists() and (not ypath.exists()):
        print(f"[YEAR] 从 {base.name} 切分 {year} ...", flush=True)
        raw = pd.read_parquet(base)
        raw["Date"] = pd.to_datetime(raw["Date"])
        part = raw.loc[
            (raw["Date"] >= pd.Timestamp(year_start))
            & (raw["Date"] <= pd.Timestamp(year_end))
        ]
        if len(part) > 0:
            ypath.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(ypath, index=False)
            print(f"[YEAR] 已写出 {ypath} rows={len(part):,}", flush=True)
            dmin, dmax = pd.to_datetime(part["Date"]).min(), pd.to_datetime(part["Date"]).max()
            if _coverage_ok(dmin, dmax, year_start, year_end):
                return ypath
            print(
                f"[YEAR] 切分结果仅覆盖 {dmin.date()} → {dmax.date()}，"
                f"继续按月从 DDB 补全",
                flush=True,
            )
        else:
            print(f"[YEAR] {base.name} 中无 {year} 年数据，改为 DDB 拉取", flush=True)

    print(f"[YEAR] DDB 按月拉取 {year}: {year_start.date()} → {year_end.date()}", flush=True)
    pull_intraday_from_ddb(year_start, year_end, ypath)
    return ypath


def construct_feat_list(
    df: pd.DataFrame,
    max_factors: int = 0,
    locked_feats: Optional[List[str]] = None,
) -> List[str]:
    """在分钟 df 上构造特征；若给定 locked_feats 则只构造这些列。"""
    df = build_base_features(df)

    if locked_feats is not None:
        need_roll = [
            f for f in locked_feats if f not in BASE_FEATS and f not in BIG_FEATS
        ]
        need_big = [f for f in locked_feats if f in BIG_FEATS]
        if need_roll:
            print(
                f"\n========== Step 1.2 滚动聚合特征 (locked n={len(need_roll)}) ==========",
                flush=True,
            )
            for j, name in enumerate(need_roll, 1):
                try:
                    obj, rest = name.rsplit("_roll", 1)
                    w_str, op = rest.split("_", 1)
                    w = int(w_str)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip unparsable locked feat {name}: {exc}", flush=True)
                    continue
                if obj not in df.columns:
                    continue
                if j == 1 or j == len(need_roll) or j % 10 == 0:
                    print(f"  [{j}/{len(need_roll)}] {name} ...", flush=True)
                df[name] = _roll_feature(df, obj, w, op).replace(
                    [np.inf, -np.inf], np.nan
                )
        if need_big:
            build_big_order_features(df)
        return list(locked_feats)

    n_base = len(BASE_FEATS)
    if max_factors > 0:
        roll_budget = max(0, max_factors - n_base)
        roll_feats = (
            build_rolling_features(df, max_names=roll_budget)
            if roll_budget > 0
            else []
        )
        remain = max_factors - n_base - len(roll_feats)
        if remain > 0:
            big_feats = build_big_order_features(df)[:remain]
        else:
            big_feats = []
            print(
                "\n[Step1.3] skip big-order (max-factors already filled)",
                flush=True,
            )
    else:
        roll_feats = build_rolling_features(df)
        big_feats = build_big_order_features(df)

    feat_list = BASE_FEATS + roll_feats + big_feats
    seen = set()
    feat_list = [f for f in feat_list if not (f in seen or seen.add(f))]
    if max_factors > 0:
        feat_list = feat_list[:max_factors]
    return feat_list


def planned_feat_list(
    max_factors: int = 0,
    locked_feats: Optional[List[str]] = None,
) -> List[str]:
    """不读数据，按规则生成特征名列表（用于分月锁定同一套因子）。"""
    if locked_feats is not None:
        return list(locked_feats)
    rolls: List[str] = []
    for obj in ROLL_OBJECTS:
        for w in ROLL_WINDOWS:
            for op in ROLL_OPS:
                rolls.append(f"{obj}_roll{w}_{op}")
    feats = BASE_FEATS + rolls + BIG_FEATS
    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    if max_factors > 0:
        feats = feats[:max_factors]
    return feats


def normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    """Symbol/时间清洗 + 连续竞价过滤（输入为原始分钟表）。"""
    missing = [c for c in L2_COLUMNS if c not in df.columns and c != "Adjfactor"]
    if missing:
        raise KeyError(f"intraday 缺少列: {missing}")
    df = df[[c for c in L2_COLUMNS if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    bt = pd.to_datetime(df["Bartime"])
    df["Bartime"] = df["Date"] + (bt - bt.dt.normalize())
    df["Symbol"] = df["Symbol"].map(to_wind_code)
    df = df[df["Symbol"].str[0].isin(list("036"))].copy()
    df = df.sort_values(["Symbol", "Bartime"]).reset_index(drop=True)
    df = filter_continuous_auction(df)
    return df.reset_index(drop=True)


def _filter_panels_date_range(
    panels: Dict[str, pd.DataFrame],
    start: dt.datetime,
    end: dt.datetime,
) -> Dict[str, pd.DataFrame]:
    out = {}
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    for k, p in panels.items():
        q = p.loc[(p.index >= s) & (p.index <= e)]
        if not q.empty:
            out[k] = q
    return out


def process_one_year(
    year_parquet: Path,
    year_start: dt.datetime,
    year_end: dt.datetime,
    max_factors: int,
    locked_feats: Optional[List[str]],
    *,
    daily_cache_root: Optional[Path] = None,
    rebuild_daily_cache: bool = False,
    use_minute_bar_store: Optional[bool] = None,
) -> Tuple[List[str], Dict[str, pd.DataFrame], pd.DataFrame]:
    """单年：按月加载分钟→特征→日频，再合并；避免整年进内存。"""
    feat_list = planned_feat_list(max_factors=max_factors, locked_feats=locked_feats)
    print(f"\n[{year_start.year}] 计划特征数: {len(feat_list)}", flush=True)

    use_store = True if use_minute_bar_store is None else use_minute_bar_store
    if use_store:
        print(
            f"[{year_start.year}] 分钟数据 via MinuteBarStore (DDB on-demand)",
            flush=True,
        )

    daily_cache_dir: Optional[Path] = None
    if daily_cache_root is not None:
        daily_cache_dir = _daily_cache_dir_for(feat_list, daily_cache_root)
        print(
            f"[{year_start.year}] 日频缓存目录: {daily_cache_dir} "
            f"(rebuild={rebuild_daily_cache})",
            flush=True,
        )

    # 确保该年已按月拉取（写入 tmp 月文件 + 年 parquet）
    tmp_dir = year_parquet.parent / f".tmp_{year_parquet.stem}_months"
    months = _month_ends(year_start, year_end)

    if not use_store and (
        not tmp_dir.exists() or not any(tmp_dir.glob("*.parquet"))
    ):
        print(
            f"[{year_start.year}] 月缓存缺失，触发按月拉取 → {tmp_dir}",
            flush=True,
        )
        pull_intraday_from_ddb(year_start, year_end, year_parquet)

    panel_parts: Dict[str, List[pd.DataFrame]] = {}
    prev_m0: Optional[dt.datetime] = None
    prev_mpath: Optional[Path] = None

    for i, (m0, m1) in enumerate(months, 1):
        ym = m0.strftime("%Y%m")
        cache_path = (
            daily_cache_dir / f"{ym}.pkl" if daily_cache_dir is not None else None
        )

        if cache_path is not None and cache_path.exists() and not rebuild_daily_cache:
            print(
                f"  [{i}/{len(months)}] skip compute (cache) {m0.date()} → {m1.date()} ...",
                flush=True,
            )
            panels_m = _load_daily_cache(cache_path)
            if panels_m:
                panels_m = _filter_panels_date_range(panels_m, m0, m1)
                for feat, panel in panels_m.items():
                    panel_parts.setdefault(feat, []).append(panel)
            _, from_store_flag = _ensure_month_parquet(
                m0, m1, year_parquet, use_store
            )
            if not from_store_flag:
                prev_mpath = _month_parquet_path(m0, year_parquet, use_store=False)
            prev_m0 = m0
            continue

        print(
            f"  [{i}/{len(months)}] feature {m0.date()} → {m1.date()} ...",
            flush=True,
        )
        mpath, from_store_flag = _ensure_month_parquet(
            m0, m1, year_parquet, use_store
        )
        if from_store_flag:
            raw = _load_minute_month_from_store(m0, m1, prev_m0)
        else:
            raw = _load_month_with_lookback(
                mpath,
                prev_mpath,
                cur_from_store=False,
                prev_from_store=False,
            )

        df = normalize_minute_df(raw)
        del raw
        df = df[
            (df["Date"] >= pd.Timestamp(m0) - pd.Timedelta(days=5))
            & (df["Date"] <= pd.Timestamp(m1))
        ]
        if df.empty:
            prev_m0 = m0
            prev_mpath = mpath
            continue

        construct_feat_list(df, max_factors=0, locked_feats=feat_list)
        panels_m = minute_to_daily_last(df, feat_list)
        del df
        panels_m = _filter_panels_date_range(panels_m, m0, m1)
        if cache_path is not None:
            _save_daily_cache(cache_path, panels_m)
        for feat, panel in panels_m.items():
            panel_parts.setdefault(feat, []).append(panel)
        prev_m0 = m0
        prev_mpath = mpath

    panels = merge_panels(panel_parts)
    panels = _filter_panels_date_range(panels, year_start, year_end)
    print(
        f"[{year_start.year}] 日频面板因子数={len(panels)}",
        flush=True,
    )

    print(
        f"[{year_start.year}] get_Ret_Matrix v2v "
        f"{year_start.date()} → {year_end.date()}",
        flush=True,
    )
    ret = fdl.get_Ret_Matrix(year_start, year_end, method="v2v")
    ret.index = pd.to_datetime(ret.index)
    return feat_list, panels, ret


def merge_panels(
    parts: Dict[str, List[pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for feat, lst in parts.items():
        if not lst:
            continue
        combined = pd.concat(lst, axis=0).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        out[feat] = combined
    return out


def load_intraday(
    parquet_path: Path,
    start_day: Optional[dt.datetime] = None,
    end_day: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """加载某一 parquet（通常已是单年文件）；缺失或覆盖不足则从 DDB 拉取到该路径。"""
    start_day = start_day or dt.datetime.strptime(DEFAULT_START, "%Y-%m-%d")
    end_day = end_day or dt.datetime.strptime(DEFAULT_END, "%Y-%m-%d")

    need_pull = True
    if parquet_path.exists():
        print(f"[LOAD] 读取 {parquet_path} ...", flush=True)
        try:
            dmin, dmax = _parquet_date_range(parquet_path)
            print(
                f"[LOAD] parquet 覆盖: "
                f"{None if dmin is None else dmin.date()} → "
                f"{None if dmax is None else dmax.date()}",
                flush=True,
            )
            if _coverage_ok(dmin, dmax, start_day, end_day):
                need_pull = False
                df = pd.read_parquet(parquet_path)
            else:
                print(
                    f"[LOAD] 覆盖不足目标区间 {start_day.date()} → {end_day.date()}，"
                    f"将从 DolphinDB 重新拉取",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[LOAD] 读取 parquet 失败 ({exc})，改为 DDB 拉取", flush=True)
    else:
        print(f"[LOAD] 未找到 {parquet_path} → 从 DolphinDB 拉取", flush=True)

    if need_pull:
        pulled = pull_intraday_from_ddb(start_day, end_day, parquet_path)
        if pulled is not None and len(pulled) > 0:
            df = pulled
        else:
            print(f"[LOAD] 从磁盘读取刚拉取的 {parquet_path} ...", flush=True)
            df = pd.read_parquet(parquet_path)

    missing = [c for c in L2_COLUMNS if c not in df.columns and c != "Adjfactor"]
    if missing:
        raise KeyError(f"intraday 缺少列: {missing}")

    df = df[[c for c in L2_COLUMNS if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    # 部分 parquet 仅存时刻（日期落在 1970-01-01），与 Date 拼成完整时间戳
    bt = pd.to_datetime(df["Bartime"])
    df["Bartime"] = df["Date"] + (bt - bt.dt.normalize())
    df["Symbol"] = df["Symbol"].map(to_wind_code)
    df = df[df["Symbol"].str[0].isin(list("036"))].copy()
    df = df.sort_values(["Symbol", "Bartime"]).reset_index(drop=True)
    df = filter_continuous_auction(df)

    df = df[df["Date"] >= pd.Timestamp(start_day)]
    df = df[df["Date"] <= pd.Timestamp(end_day)]

    if df.empty:
        raise RuntimeError("过滤后分钟数据为空，请检查日期与 Symbol 口径")

    print(
        f"[LOAD] 完成: {len(df):,} 行 | 股票 {df['Symbol'].nunique()} 只 | "
        f"日期 {df['Date'].min().date()} → {df['Date'].max().date()}",
        flush=True,
    )
    return df.reset_index(drop=True)


# ---------- 特征构造 (Step 1) ----------
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 15 个基础特征"""
    print("\n========== Step 1.1 基础特征 ==========", flush=True)
    buy_c = df["Active_buy_count"].fillna(0.0)
    sell_c = df["Active_sell_count"].fillna(0.0)
    bid_cc = df["Bid_cancel_count"].fillna(0.0)
    ask_cc = df["Ask_cancel_count"].fillna(0.0)

    df["ActiveBuyVolRatio"] = df["Active_buy_volume"] / (df["Volume"] + EPS)
    df["ActiveSellVolRatio"] = df["Active_sell_volume"] / (df["Volume"] + EPS)
    df["ActiveBuyAmtRatio"] = df["Active_buy_amount"] / (df["Amount"] + EPS)
    df["ActiveSellAmtRatio"] = df["Active_sell_amount"] / (df["Amount"] + EPS)
    df["ActiveBuyCountRatio"] = buy_c / (buy_c + sell_c + EPS)
    df["ActiveVolImbalance"] = (df["Active_buy_volume"] - df["Active_sell_volume"]) / (
        df["Volume"] + EPS
    )
    df["ActiveAmtImbalance"] = (df["Active_buy_amount"] - df["Active_sell_amount"]) / (
        df["Amount"] + EPS
    )
    df["ActiveCountImbalance"] = (buy_c - sell_c) / (buy_c + sell_c + EPS)
    df["CancelVolumeImbalance"] = (df["Bid_cancel_volume"] - df["Ask_cancel_volume"]) / (
        df["Volume"] + EPS
    )
    df["CancelCountImbalance"] = (bid_cc - ask_cc) / (bid_cc + ask_cc + EPS)
    df["BidCancelVolRatio"] = df["Bid_cancel_volume"] / (df["Volume"] + EPS)
    df["AskCancelVolRatio"] = df["Ask_cancel_volume"] / (df["Volume"] + EPS)
    df["LogReturn"] = np.log(df["Close"] / df["Open"].replace(0, np.nan))
    df["HighLowSpread"] = (df["High"] - df["Low"]) / (df["Open"] + EPS)
    df["Vol_Amt_Ratio"] = df["Volume"] / (df["Amount"] + EPS)

    for c in BASE_FEATS:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    print(f"base features: {len(BASE_FEATS)}", flush=True)
    return df


def _roll_feature(df: pd.DataFrame, col: str, window: int, op: str) -> pd.Series:
    """按 Symbol 做 rolling(op) 后 shift(1)，对齐到 df.index。"""
    min_p = max(3, window // 2)
    g = df.groupby("Symbol", sort=False)[col]
    if op == "last_diff":
        rolled = df[col] - g.shift(window - 1)
    elif op == "mean":
        rolled = g.rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)
    elif op == "std":
        rolled = g.rolling(window, min_periods=min_p).std().reset_index(level=0, drop=True)
    elif op == "max":
        rolled = g.rolling(window, min_periods=min_p).max().reset_index(level=0, drop=True)
    elif op == "min":
        rolled = g.rolling(window, min_periods=min_p).min().reset_index(level=0, drop=True)
    elif op == "skew":
        rolled = g.rolling(window, min_periods=min_p).skew().reset_index(level=0, drop=True)
    else:
        raise ValueError(op)
    tmp = pd.Series(rolled.values, index=df.index)
    return df.assign(_roll_tmp=tmp).groupby("Symbol", sort=False)["_roll_tmp"].shift(1)


def build_rolling_features(
    df: pd.DataFrame, max_names: Optional[int] = None
) -> List[str]:
    """构造滚动聚合特征；max_names 用于 --max-factors 调试时提前停止。"""
    print("\n========== Step 1.2 滚动聚合特征 ==========", flush=True)
    roll_names: List[str] = []
    for obj in ROLL_OBJECTS:
        if obj not in df.columns:
            print(f"  skip missing object: {obj}", flush=True)
            continue
        print(f"  rolling on {obj} ...", flush=True)
        for w in ROLL_WINDOWS:
            for op in ROLL_OPS:
                if max_names is not None and len(roll_names) >= max_names:
                    print(
                        f"rolling features: {len(roll_names)} (capped by max_names)",
                        flush=True,
                    )
                    return roll_names
                name = f"{obj}_roll{w}_{op}"
                df[name] = _roll_feature(df, obj, w, op)
                df[name] = df[name].replace([np.inf, -np.inf], np.nan)
                roll_names.append(name)
    print(f"rolling features: {len(roll_names)}", flush=True)
    return roll_names


def build_big_order_features(df: pd.DataFrame) -> List[str]:
    """构造大单代理特征 (3 个)"""
    print("\n========== Step 1.3 大小单代理 (Mask) ==========", flush=True)
    pieces_buy, pieces_sell = [], []
    for _, sub in df.groupby("Symbol", sort=False):
        avg_b = sub["Active_buy_volume"].rolling(240, min_periods=60).mean().shift(1)
        avg_s = sub["Active_sell_volume"].rolling(240, min_periods=60).mean().shift(1)
        is_b = (sub["Active_buy_volume"] > 2.0 * avg_b).astype(float).where(avg_b.notna())
        is_s = (sub["Active_sell_volume"] > 2.0 * avg_s).astype(float).where(avg_s.notna())
        pieces_buy.append(is_b)
        pieces_sell.append(is_s)
    df["Is_BigBuy"] = pd.concat(pieces_buy).sort_index()
    df["Is_BigSell"] = pd.concat(pieces_sell).sort_index()

    bb, bs = [], []
    for _, sub in df.groupby("Symbol", sort=False):
        bb.append(sub["Is_BigBuy"].rolling(30, min_periods=10).mean().shift(1))
        bs.append(sub["Is_BigSell"].rolling(30, min_periods=10).mean().shift(1))
    df["BigBuyRatio_roll30"] = pd.concat(bb).sort_index()
    df["BigSellRatio_roll30"] = pd.concat(bs).sort_index()
    df["BigBuySellImbalance_roll30"] = (
        df["BigBuyRatio_roll30"] - df["BigSellRatio_roll30"]
    )
    print(f"big-order features: {len(BIG_FEATS)}", flush=True)
    return list(BIG_FEATS)


def minute_to_daily_last(
    df: pd.DataFrame, feat_list: List[str]
) -> Dict[str, pd.DataFrame]:
    """将分钟因子转为日频：取每日最后一个有效分钟的因子值。

    连续竞价末根（如 15:00）上 Active_* 常为 NaN。这里优先在
    Active_buy_volume 非空的分钟上取 last bar；若过滤后为空则回退全样本。
    """
    print("\n========== Step 2.0 分钟→日频 (last valid) ==========", flush=True)
    feat_cols = [c for c in feat_list if c in df.columns]
    if not feat_cols:
        return {}

    base_cols = ["Symbol", "Date", "Bartime"]
    # 优先去掉 Active 全空的末段 bar（通常是 15:00）
    mask = df["Active_buy_volume"].notna()
    if mask.any():
        sub = df.loc[mask, base_cols + feat_cols]
        print(
            f"  using Active_buy_volume-valid bars: {len(sub):,} / {len(df):,}",
            flush=True,
        )
    else:
        sub = df[base_cols + feat_cols]
        print(f"  fallback all bars: {len(sub):,}", flush=True)

    idx = sub.groupby(["Symbol", "Date"], sort=False)["Bartime"].idxmax()
    daily = sub.loc[idx, base_cols + feat_cols]

    panels: Dict[str, pd.DataFrame] = {}
    for feat in feat_cols:
        wide = daily.pivot(index="Date", columns="Symbol", values=feat)
        wide.index = pd.to_datetime(wide.index, errors="coerce")
        wide = wide.loc[wide.index.notna()].sort_index()
        wide = wide.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if wide.empty:
            print(f"  skip all-NaN daily panel: {feat}", flush=True)
            continue
        panels[feat] = wide

    print(f"daily panels built: {len(panels)}", flush=True)
    if panels:
        sample = next(iter(panels.values()))
        print(
            f"daily panel dates: {sample.index.min().date()} → {sample.index.max().date()} "
            f"| n_days={len(sample.index)} n_syms={sample.shape[1]}",
            flush=True,
        )
    return panels


# ---------- IC 筛选 (Step 2) ----------
def screen_by_icir(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    feat_list: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """计算 Rank IC/ICIR，按阈值初筛；并与 fdl.calICIR 交叉核对"""
    print("\n========== Step 2 IC/ICIR 初筛 ==========", flush=True)
    rows = []
    for i, feat in enumerate(feat_list, 1):
        if feat not in panels:
            continue
        sig = panels[feat].shift(1)  # D-1 因子 → D 收益
        try:
            sig_m, ret_m = fdl.apply_tradability_mask(sig, ret)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(feat_list)}] {feat}: mask failed ({exc})", flush=True)
            continue

        if sig_m.notna().sum().sum() < 500:
            continue

        ic_series = sig_m.corrwith(ret_m, axis=1, method="spearman").dropna()
        if ic_series.empty:
            continue

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        icir = ic_mean / ic_std * np.sqrt(250) if ic_std > 0 else np.nan
        ic_pos = float((ic_series > 0).mean())

        # 与库函数交叉核对（签名已确认匹配）
        lib_ic, lib_icir = fdl.calICIR(sig_m, ret_m, n=250)

        pass_a = (abs(ic_mean) > 0.02) and (
            pd.notna(icir) and abs(icir) > 0.3
        )
        pass_b = (ic_pos > 0.55) and (abs(ic_mean) > 0.015)
        selected = bool(pass_a or pass_b)

        rows.append(
            {
                "factor": feat,
                "ic_mean": ic_mean,
                "icir": icir,
                "ic_pos_ratio": ic_pos,
                "lib_ic": lib_ic,
                "lib_icir": lib_icir,
                "n_ic_days": int(ic_series.shape[0]),
                "selected": selected,
            }
        )
        if i % 20 == 0 or selected:
            mark = "*" if selected else ""
            print(
                f"  [{i}/{len(feat_list)}] {feat}: IC={ic_mean:.4f} "
                f"ICIR={icir:.3f} Pos={ic_pos:.2%} {mark}",
                flush=True,
            )

    ic_df = pd.DataFrame(rows)
    if ic_df.empty:
        print("初筛通过: 0 / 0", flush=True)
        return ic_df, []

    ic_df = (
        ic_df.assign(_abs_icir=ic_df["icir"].abs())
        .sort_values(["selected", "_abs_icir"], ascending=[False, False])
        .drop(columns="_abs_icir")
        .reset_index(drop=True)
    )
    selected_factors = ic_df.loc[ic_df["selected"], "factor"].tolist()

    show = ic_df.copy()
    show["factor"] = show.apply(
        lambda r: f"* {r['factor']}" if r["selected"] else r["factor"], axis=1
    )
    print("\n--- IC 统计（* = 通过初筛）---", flush=True)
    with pd.option_context("display.max_rows", 300, "display.width", 140):
        print(
            show[
                ["factor", "ic_mean", "icir", "ic_pos_ratio", "n_ic_days", "selected"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            flush=True,
        )
    print(f"\n初筛通过: {len(selected_factors)} / {len(ic_df)}", flush=True)
    return ic_df, selected_factors


# ---------- 分组单调性 (Step 3) ----------
def judge_monotonicity(group_means: pd.Series) -> str:
    g = group_means.reindex(
        sorted([c for c in group_means.index if isinstance(c, (int, np.integer))])
    ).dropna()
    if len(g) < 5:
        return "样本不足"
    diffs = g.diff().iloc[1:]
    if (diffs > 0).all():
        return "严格单调递增"
    if (diffs < 0).all():
        return "严格单调递减"
    rho = pd.Series(g.index.astype(float)).corr(
        pd.Series(g.values), method="spearman"
    )
    pos = float((diffs > 0).mean())
    neg = float((diffs < 0).mean())
    if abs(rho) >= 0.7 and (pos >= 0.7 or neg >= 0.7):
        return "大致单调递增" if rho > 0 else "大致单调递减"
    if abs(rho) >= 0.5 and (pos >= 0.6 or neg >= 0.6):
        return "大致单调递增" if rho > 0 else "大致单调递减"
    return "无明显单调"


def save_group_cumsum_png(
    group_pnl_df: pd.DataFrame,
    group_to_df: pd.DataFrame,
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    factor_name: str,
    out_path: Path,
) -> dict:
    """保存与 Factor_Dev_Lib.groupTest / cum_pnl.png 同风格的累计收益图。

    绘图时：G1~G10 日收益先减去当日市场等权平均收益（可交易截面 mean），
    再 cumsum，避免曲线被大盘 beta 拖成一团；H-L 本身已是多空差，不再减市场。

    横轴页脚统计仍基于原始 H-L（未减市场）：Direction, AnnuRet, Sharpe, MDD,
    Daily Turnover, Implied AnnuFee(7.5%), Daily IC, Annu ICIR。
    """
    hl = group_pnl_df["H-L"]
    direction = 1 if hl.mean() > 0 else -1
    hl_dir = hl * direction
    annu = fdl.calAnnuRet(hl_dir)
    sharpe = fdl.calSharpe(hl_dir)
    mdd, _ = fdl.calMDD(hl_dir)
    avg_to = float(group_to_df["H-L"].mean())
    implied_fee = fdl.implied_annu_fee(avg_to)

    rank_ic_daily = signal.corrwith(ret, axis=1, method="spearman")
    rank_ic = float(np.nanmean(rank_ic_daily.values))
    rank_ic_std = float(np.nanstd(rank_ic_daily.values))
    rank_icir = (
        rank_ic / rank_ic_std * np.sqrt(250)
        if rank_ic_std and rank_ic_std > 0
        else np.nan
    )

    stats_title = fdl.format_group_test_stats_title(
        direction=direction,
        annu_ret=annu,
        sharpe=sharpe,
        mdd=mdd,
        avg_turnover=avg_to,
        rank_ic=rank_ic,
        icir=rank_icir,
        implied_fee=implied_fee,
    )

    # 市场平均：与 signal/ret 对齐后的可交易截面等权收益
    valid = signal.notna() & ret.notna()
    market_avg = ret.where(valid).mean(axis=1)
    market_avg = market_avg.reindex(group_pnl_df.index)

    plot_pnl = group_pnl_df.copy()
    gcols = [c for c in plot_pnl.columns if c != "H-L"]
    for c in gcols:
        plot_pnl[c] = plot_pnl[c] - market_avg
    # H-L 保持不变（多空已对冲市场）

    cum_pnl_df = plot_pnl.cumsum()
    fig, ax = plt.subplots(figsize=(20, 12))
    for col_name, y in cum_pnl_df.items():
        ax.plot(y.index, y.values, label=str(col_name))
        ax.text(
            y.index[-1],
            y.iloc[-1],
            str(col_name),
            fontsize=15,
            verticalalignment="bottom",
        )
    ax.legend(loc="upper left")
    ax.set_xlabel(stats_title, fontsize=15)
    ax.set_title(f"{factor_name} (G1-G10 excess vs mkt EW)")
    ax.set_ylabel("Cumulative excess return (groups − mkt EW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "direction": direction,
        "annu": annu,
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_turnover": avg_to,
        "implied_fee": implied_fee,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "stats_title": stats_title,
    }


def run_group_tests(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    selected_factors: List[str],
    ic_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """对每个初筛因子做十分组测试，画图并判断单调性"""
    print("\n========== Step 3 十分组单调性 ==========", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for feat in selected_factors:
        print(f"\n--- groupTest: {feat} ---", flush=True)
        sig = panels[feat].shift(1)
        try:
            sig_m, ret_m = fdl.apply_tradability_mask(sig, ret)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip (mask): {exc}", flush=True)
            continue

        sig_m, ret_m = sig_m.align(ret_m, join="inner", axis=None)
        if sig_m.notna().sum().sum() < 500:
            print("  skip: too few tradable samples", flush=True)
            continue

        # silent：不弹窗；自行按标准样式落盘 cumsum.png
        _, group_pnl_df, group_to_df = fdl.groupTest(
            sig_m, ret_m, n=10, fee=0, info="silent"
        )

        stats = save_group_cumsum_png(
            group_pnl_df,
            group_to_df,
            sig_m,
            ret_m,
            factor_name=feat,
            out_path=out_dir / f"{feat}_cumsum.png",
        )

        # 柱状图同样用相对市场的超额日均收益，便于看单调性
        valid = sig_m.notna() & ret_m.notna()
        mkt = ret_m.where(valid).mean(axis=1).reindex(group_pnl_df.index)
        bar_pnl = group_pnl_df.copy()
        for c in bar_pnl.columns:
            if c != "H-L":
                bar_pnl[c] = bar_pnl[c] - mkt
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bar_pnl.mean().plot(
            kind="bar", ax=ax2, title=f"{feat} — group mean excess vs mkt EW"
        )
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{feat}_group_mean.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

        direction = stats["direction"]
        annu = stats["annu"]
        sharpe = stats["sharpe"]
        mdd = stats["mdd"]

        gcols = [c for c in range(1, 11) if c in group_pnl_df.columns]
        g_mean = group_pnl_df[gcols].mean()
        g_cum_last = group_pnl_df[gcols].cumsum().iloc[-1]
        mono = judge_monotonicity(g_mean)
        pass_mono = mono.startswith("严格") or mono.startswith("大致")

        ic_row = ic_df.loc[ic_df["factor"] == feat]
        rec = {
            "factor": feat,
            "mono": mono,
            "hl_annu_ret": annu,
            "hl_sharpe": sharpe,
            "hl_mdd": mdd,
            "hl_direction": direction,
            "daily_turnover": stats["avg_turnover"],
            "implied_annu_fee": stats["implied_fee"],
            "daily_ic": stats["rank_ic"],
            "annu_icir": stats["rank_icir"],
            "ic_mean": float(ic_row["ic_mean"].values[0]) if not ic_row.empty else np.nan,
            "icir": float(ic_row["icir"].values[0]) if not ic_row.empty else np.nan,
            "pass_mono": pass_mono,
        }
        records.append(rec)
        print(
            f"  {mono} | Annu={annu:.2%} Sharpe={sharpe:.2f} MDD={mdd:.2%} "
            f"TO={stats['avg_turnover']:.2f} Fee={stats['implied_fee']:.2%} "
            f"| IC={stats['rank_ic']:.4f} ICIR={stats['rank_icir']:.2f}",
            flush=True,
        )
        print(f"  G1..G10 cum last: {g_cum_last.round(4).to_dict()}", flush=True)

    result = pd.DataFrame(records)
    if result.empty:
        print("无因子进入分组测试。", flush=True)
        return result

    good = result.loc[result["pass_mono"]].copy()
    print("\n========== 最终优质因子 ==========", flush=True)
    if good.empty:
        print("未筛出单调性通过的因子。", flush=True)
    else:
        cols = [
            "factor",
            "mono",
            "hl_annu_ret",
            "hl_sharpe",
            "hl_mdd",
            "daily_turnover",
            "implied_annu_fee",
            "daily_ic",
            "annu_icir",
        ]
        with pd.option_context("display.max_rows", 100, "display.width", 160):
            print(
                good[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
                flush=True,
            )
        good.to_csv(out_dir / "final_good_factors.csv", index=False)

    result.to_csv(out_dir / "group_test_all_selected.csv", index=False)
    return result


# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser(
        description="L2 raw-feature single-factor screen (by-year)"
    )
    parser.add_argument(
        "--parquet",
        default="intraday.parquet",
        help="基名路径；按年文件为 {stem}_{year}.parquet",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"起始日期 YYYY-MM-DD (默认 {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"结束日期 YYYY-MM-DD (默认 {DEFAULT_END})",
    )
    parser.add_argument(
        "--out-dir", default="research/results/l2_raw_feature_screen"
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=0,
        help="调试：仅筛前 N 个因子（0=全部）",
    )
    parser.add_argument(
        "--daily-cache-dir",
        default=str(DEFAULT_DAILY_CACHE_ROOT),
        help="按月日频面板缓存根目录（默认 research/cache/l2_daily_panels）",
    )
    parser.add_argument(
        "--rebuild-daily-cache",
        action="store_true",
        help="忽略日频缓存，重新计算全部月份",
    )
    parser.add_argument(
        "--no-minute-bar-store",
        action="store_true",
        help="不使用 MinuteBarStore（DDB 按需查询），回退 legacy parquet 路径",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_day = dt.datetime.strptime(args.start, "%Y-%m-%d")
    end_day = dt.datetime.strptime(args.end, "%Y-%m-%d")
    years = list(range(start_day.year, end_day.year + 1))
    parquet_base = Path(args.parquet)

    print(
        f"[CFG] 初筛区间 {start_day.date()} → {end_day.date()} | "
        f"years={years} | max_factors={args.max_factors or 'ALL'} | "
        f"parquet_base={parquet_base} | daily_cache={args.daily_cache_dir} | "
        f"minute_bar_store={not args.no_minute_bar_store}",
        flush=True,
    )

    daily_cache_root = Path(args.daily_cache_dir)
    use_store = not args.no_minute_bar_store

    panel_parts: Dict[str, List[pd.DataFrame]] = {}
    ret_parts: List[pd.DataFrame] = []
    global_feat_list: Optional[List[str]] = None
    years_done: List[int] = []

    for year in years:
        year_start = max(start_day, dt.datetime(year, 1, 1))
        year_end = min(end_day, dt.datetime(year, 12, 31))
        if year_start > year_end:
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"处理 {year} 年：{year_start.date()} → {year_end.date()}",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

        try:
            ypath = ensure_year_parquet(
                parquet_base, year, year_start, year_end
            )
            feat_list, panels_year, ret_year = process_one_year(
                ypath,
                year_start,
                year_end,
                max_factors=args.max_factors,
                locked_feats=global_feat_list,
                daily_cache_root=daily_cache_root,
                rebuild_daily_cache=args.rebuild_daily_cache,
                use_minute_bar_store=use_store,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {year} 年处理失败，跳过: {exc}", flush=True)
            continue

        if not panels_year:
            print(f"[WARN] {year} 年无日频面板，跳过", flush=True)
            continue

        if global_feat_list is None:
            global_feat_list = list(feat_list)
            pd.Series(global_feat_list, name="factor").to_csv(
                out_dir / "feat_list.csv", index=False
            )
            print(f"[CFG] 锁定全局特征列表: {len(global_feat_list)}", flush=True)

        for feat, panel in panels_year.items():
            panel_parts.setdefault(feat, []).append(panel)
        ret_parts.append(ret_year)
        years_done.append(year)
    if global_feat_list is None or not ret_parts:
        raise RuntimeError(
            "没有任何年份成功处理。请检查按年 parquet / DDB 权限与日期区间。"
        )

    print(
        f"\n[MERGE] 合并日频面板 years={years_done} ...",
        flush=True,
    )
    full_panels = merge_panels(panel_parts)
    # 只保留全局特征列表中的因子
    full_panels = {
        f: full_panels[f] for f in global_feat_list if f in full_panels
    }
    full_ret = pd.concat(ret_parts, axis=0).sort_index()
    full_ret = full_ret[~full_ret.index.duplicated(keep="last")]

    if not full_panels:
        raise RuntimeError("合并后日频面板为空")

    sample = next(iter(full_panels.values()))
    print(
        f"[MERGE] panel {sample.index.min().date()} → {sample.index.max().date()} "
        f"| n_days={len(sample.index)} n_feats={len(full_panels)}",
        flush=True,
    )

    # Step 2：在全期合并日频上做 IC 初筛（统计口径完整）
    ic_df, selected_factors = screen_by_icir(
        full_panels, full_ret, global_feat_list
    )
    ic_df.to_csv(out_dir / "ic_screen_all.csv", index=False)
    pd.Series(selected_factors, name="factor").to_csv(
        out_dir / "selected_factors.csv", index=False
    )

    # Step 3：全期分组测试
    group_df = run_group_tests(
        full_panels, full_ret, selected_factors, ic_df, out_dir
    )

    n_raw = len(global_feat_list)
    n_sel = len(selected_factors)
    good = (
        group_df.loc[group_df["pass_mono"]]
        if (not group_df.empty and "pass_mono" in group_df.columns)
        else pd.DataFrame()
    )
    print("\n========== 总结 ==========", flush=True)
    print(
        f"原始因子: {n_raw}  →  IC初筛通过: {n_sel}  →  单调性通过: {len(good)}",
        flush=True,
    )
    print(f"成功处理年份: {years_done}", flush=True)
    if len(good) > 0:
        best = (
            good.assign(_s=good["hl_sharpe"].abs())
            .sort_values("_s", ascending=False)
            .iloc[0]
        )
        print(
            f"最佳因子: {best['factor']} | Sharpe={best['hl_sharpe']:.2f} | "
            f"Annu={best['hl_annu_ret']:.2%} | ICIR={best.get('annu_icir', best.get('icir', float('nan'))):.3f} | "
            f"{best['mono']}",
            flush=True,
        )
    print(f"结果目录: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
