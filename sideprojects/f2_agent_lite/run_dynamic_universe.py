#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动态指数成分池泛化实验.

Layer1: 价量 + VQ（默认）
Layer2: 价量 + VQ + 分钟因子（ClickHouse 分块缓存）

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_dynamic_universe --preset CSI300
    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_dynamic_universe --preset CSI300 --layer 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data import db_connector as db  # noqa: E402

INDEX_PRESETS = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
}
THREE_LEG_BASELINE = 1.714
FIXED10_VQ_SHARPE = 1.273  # standalone VQ on 10 names (reference)
LAYER1_CSI300_BASELINE = 1.010  # prior run without industry neutral


def _cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def _pivot_z(long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = long.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last")
    return _cs_zscore(wide.sort_index())


def _industry_demean(score_wide: pd.DataFrame, ind_map: Dict[str, str]) -> pd.DataFrame:
    """Within-day, within-industry demean (groups with <2 names unchanged)."""
    if score_wide.empty or not ind_map:
        return score_wide
    long = score_wide.stack(future_stack=True).rename("score").reset_index()
    long.columns = ["date", "symbol", "score"]
    long["ind"] = long["symbol"].map(lambda s: ind_map.get(str(s), "OTHER"))
    gsize = long.groupby(["date", "ind"])["score"].transform("size")
    gmean = long.groupby(["date", "ind"])["score"].transform("mean")
    long["score"] = np.where(gsize >= 2, long["score"] - gmean, long["score"])
    out = long.pivot(index="date", columns="symbol", values="score").reindex_like(score_wide)
    return out


def _load_minute_factors_chunked(
    symbols: List[str],
    start: str,
    end: str,
    *,
    lookback: int = 10,
    chunk_size: int = 25,
    cache_dir: Path,
    index_tag: str,
) -> pd.DataFrame:
    """Load daily minute factors in symbol chunks; cache to parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "minute_daily_{}_{}_{}_lb{}.parquet".format(
        index_tag, start.replace("-", ""), end.replace("-", ""), lookback
    )
    if cache_path.exists():
        print("[dyn] 读取分钟因子缓存 {}".format(cache_path), flush=True)
        out = pd.read_parquet(cache_path)
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out

    parts: List[pd.DataFrame] = []
    n = len(symbols)
    n_chunks = (n + chunk_size - 1) // chunk_size
    client = db.get_clickhouse_client()
    try:
        for i in range(0, n, chunk_size):
            chunk = symbols[i : i + chunk_size]
            ci = i // chunk_size + 1
            print(
                "[dyn] minute chunk {}/{} n={} ...".format(ci, n_chunks, len(chunk)),
                flush=True,
            )
            part = db.get_minute_factors(
                chunk,
                start,
                end,
                lookback=lookback,
                use_local_tables=False,
                client=client,
            )
            if part is not None and not part.empty:
                parts.append(part)
                print("[dyn]   rows={}".format(len(part)), flush=True)
            else:
                print("[dyn]   empty", flush=True)
    finally:
        try:
            client.close()
        except Exception:
            pass

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "minute_amplitude", "price_jump"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out.to_parquet(cache_path, index=False)
    print("[dyn] 分钟因子缓存写入 {} rows={}".format(cache_path, len(out)), flush=True)
    return out


def _load_l2_factors_chunked(
    symbols: List[str],
    start: str,
    end: str,
    *,
    chunk_size: int = 40,
    cache_dir: Path,
    index_tag: str,
) -> pd.DataFrame:
    """Load daily SSL2 book factors in symbol chunks; cache to parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "l2_daily_{}_{}_{}.parquet".format(
        index_tag, start.replace("-", ""), end.replace("-", "")
    )
    if cache_path.exists():
        print("[dyn] 读取 L2 因子缓存 {}".format(cache_path), flush=True)
        out = pd.read_parquet(cache_path)
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out

    parts: List[pd.DataFrame] = []
    n = len(symbols)
    n_chunks = (n + chunk_size - 1) // chunk_size
    client = db.get_clickhouse_client()
    try:
        for i in range(0, n, chunk_size):
            chunk = symbols[i : i + chunk_size]
            ci = i // chunk_size + 1
            print(
                "[dyn] L2 chunk {}/{} n={} ...".format(ci, n_chunks, len(chunk)),
                flush=True,
            )
            part = db.get_l2_daily_factors(chunk, start, end, client=client)
            if part is not None and not part.empty:
                parts.append(part)
                print("[dyn]   rows={}".format(len(part)), flush=True)
            else:
                print("[dyn]   empty", flush=True)
    finally:
        try:
            client.close()
        except Exception:
            pass

    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "l2_obi_l1",
                "l2_depth_oi",
                "l2_rel_spread",
                "l2_micro_bias",
                "n_snap",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out.to_parquet(cache_path, index=False)
    print("[dyn] L2 因子缓存写入 {} rows={}".format(cache_path, len(out)), flush=True)
    return out


def build_pv_vq_panel(
    index_code: str,
    start: str,
    end: str,
    preheat_days: int = 120,
    *,
    use_minute: bool = False,
    use_l2: bool = False,
    industry_neutral: bool = True,
    minute_lookback: int = 10,
    minute_chunk_size: int = 25,
    l2_chunk_size: int = 40,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build Layer-1/2/3 signal panel for dynamic index membership."""
    preheat_start = (pd.Timestamp(start) - pd.Timedelta(days=preheat_days)).strftime("%Y-%m-%d")
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    index_tag = index_code.replace(".", "")

    print("[dyn] 拉取指数成分 mask {} ...".format(index_code), flush=True)
    mask = db.get_index_member_mask(index_code, preheat_start, end_plus)
    if mask.empty:
        raise RuntimeError("empty index member mask for {}".format(index_code))
    mask.index = pd.to_datetime(mask.index).normalize()
    symbols = sorted(mask.columns.astype(str).tolist())
    print(
        "[dyn] 成分并集={}  日均成分≈{:.0f}  日期={}~{}".format(
            len(symbols),
            float(mask.notna().sum(axis=1).mean()),
            mask.index.min().date(),
            mask.index.max().date(),
        ),
        flush=True,
    )

    ind_map: Dict[str, str] = {}
    if industry_neutral:
        print("[dyn] 加载中信一级行业映射 ...", flush=True)
        ind_map = db.get_citics_l1_industry_map(symbols, asof=end)
        print(
            "[dyn] 行业覆盖 {}/{}  unique_L1={}".format(
                sum(1 for s in symbols if s in ind_map),
                len(symbols),
                len(set(ind_map.values())),
            ),
            flush=True,
        )

    print("[dyn] bulk OHLCV ...", flush=True)
    ohlcv = db.get_ohlcv_bulk(symbols, preheat_start, end_plus)
    print("[dyn] OHLCV rows={}".format(len(ohlcv)), flush=True)
    print("[dyn] bulk valuation ...", flush=True)
    val = db.get_valuation_bulk(symbols, preheat_start, end_plus)
    print("[dyn] valuation rows={}".format(len(val)), flush=True)
    print("[dyn] bulk northbound ...", flush=True)
    north = db.get_northbound_bulk(symbols, preheat_start, end_plus)
    print("[dyn] north rows={}".format(len(north)), flush=True)

    cdir = Path(cache_dir) if cache_dir is not None else Path("sideprojects/f2_agent_lite/results/cache")
    minute_df = pd.DataFrame()
    if use_minute:
        print("[dyn] Layer2: 加载分钟因子 (chunk={}) ...".format(minute_chunk_size), flush=True)
        minute_df = _load_minute_factors_chunked(
            symbols,
            preheat_start,
            end,
            lookback=minute_lookback,
            chunk_size=minute_chunk_size,
            cache_dir=cdir,
            index_tag=index_tag,
        )
        print("[dyn] minute factor rows={}".format(len(minute_df)), flush=True)

    l2_df = pd.DataFrame()
    if use_l2:
        print("[dyn] Layer3: 加载 L2 盘口因子 (chunk={}) ...".format(l2_chunk_size), flush=True)
        l2_df = _load_l2_factors_chunked(
            symbols,
            preheat_start,
            end,
            chunk_size=l2_chunk_size,
            cache_dir=cdir,
            index_tag=index_tag,
        )
        print("[dyn] L2 factor rows={}".format(len(l2_df)), flush=True)

    if ohlcv.empty:
        raise RuntimeError("empty OHLCV bulk")

    close_w = ohlcv.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    mom20 = close_w.pct_change(20, fill_method=None)
    mom60 = close_w.pct_change(60, fill_method=None)

    frames = []
    for sym, g in ohlcv.groupby("symbol", sort=False):
        daily = g.set_index("date").sort_index()
        trad = db.compute_tradability_from_ohlcv(g)
        if not trad.empty:
            daily = daily.join(trad.set_index("date")[["tradable"]], how="left")
        else:
            daily["tradable"] = np.nan
        daily["next_open"] = daily["open"].shift(-1)
        daily["next_close"] = daily["close"].shift(-1)
        daily["next_date"] = daily.index.to_series().shift(-1)
        daily["next_tradable"] = daily["tradable"].shift(-1)
        part = daily.reset_index()
        part["symbol"] = sym
        frames.append(
            part[
                [
                    "date",
                    "symbol",
                    "close",
                    "next_open",
                    "next_close",
                    "next_date",
                    "next_tradable",
                ]
            ]
        )
    exec_df = pd.concat(frames, ignore_index=True)
    exec_df["date"] = pd.to_datetime(exec_df["date"]).dt.normalize()
    exec_df["next_date"] = pd.to_datetime(exec_df["next_date"])

    z_parts = {
        "mom20": _cs_zscore(mom20),
        "mom60": _cs_zscore(mom60),
    }
    if not val.empty:
        z_parts["ep"] = _pivot_z(val, "ep_ttm")
        z_parts["bp"] = _pivot_z(val, "bp")
    if not north.empty:
        z_parts["north"] = _pivot_z(north, "north_share_ratio")
        z_parts["north_chg"] = _pivot_z(north, "north_share_chg") * 0.25

    if use_minute and not minute_df.empty:
        z_parts["minute_amp"] = _pivot_z(minute_df, "minute_amplitude")
        z_parts["minute_jump"] = -_pivot_z(minute_df, "price_jump")

    if use_l2 and not l2_df.empty:
        # high OBI / depth / micro bias → long; high spread → short
        z_parts["l2_obi"] = _pivot_z(l2_df, "l2_obi_l1")
        z_parts["l2_depth"] = _pivot_z(l2_df, "l2_depth_oi")
        z_parts["l2_micro"] = _pivot_z(l2_df, "l2_micro_bias")
        z_parts["l2_spread"] = -_pivot_z(l2_df, "l2_rel_spread")

    all_idx = sorted(set().union(*[z.index for z in z_parts.values()]))
    all_cols = sorted(set().union(*[z.columns for z in z_parts.values()]))
    stack = []
    for name, z in z_parts.items():
        aligned = z.reindex(index=all_idx, columns=all_cols)
        stack.append(aligned)
        cov = float(aligned.notna().mean().mean())
        print("[dyn] factor {} coverage={:.1%}".format(name, cov), flush=True)

    arr = np.stack([s.to_numpy(dtype=float) for s in stack], axis=0)
    with np.errstate(all="ignore"):
        mean_arr = np.nanmean(arr, axis=0)
    score = pd.DataFrame(mean_arr, index=all_idx, columns=all_cols)

    if industry_neutral and ind_map:
        print("[dyn] 应用行业中性化 (Citics L1) ...", flush=True)
        score = _industry_demean(score, ind_map)

    mem = mask.reindex(index=score.index, columns=score.columns)
    score = score.where(mem.notna())

    score_long = score.stack(future_stack=True).rename("score").reset_index()
    score_long.columns = ["date", "symbol", "score"]
    score_long["date"] = pd.to_datetime(score_long["date"]).dt.normalize()
    score_long = score_long.dropna(subset=["score"])

    merged = score_long.merge(exec_df, on=["date", "symbol"], how="inner")
    merged = merged.rename(
        columns={
            "close": "close_px",
            "next_open": "open_px",
            "next_close": "next_close_px",
        }
    )
    merged["tradable_exec"] = merged["next_tradable"].notna()
    merged = merged.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    merged = merged[merged["tradable_exec"]].copy()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)]

    print(
        "[dyn] 信号面板: {} 行 / {} 日 / {} 票".format(
            len(merged), merged["date"].nunique(), merged["symbol"].nunique()
        ),
        flush=True,
    )
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def _json_safe(obj):
    """Recursively replace NaN/Inf with None for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if pd.isna(obj):
        return None
    return obj


def _long_short_spread(result) -> Dict[str, Optional[float]]:
    """Report strategy excess vs equal-weight BH (JSON-safe)."""
    m = result.metrics or {}
    bh = result.equal_weight_bh_metrics or {}
    strat = m.get("annualized_return")
    bench = bh.get("annualized_return")
    if strat is None or bench is None:
        return {"strategy_minus_bh_ann": None, "long_short_ann_spread": None}
    spread = float(strat) - float(bench)
    if not np.isfinite(spread):
        spread = None
    return {
        "strategy_minus_bh_ann": spread,
        # alias kept for downstream readers; same proxy (true LS needs side PnL)
        "long_short_ann_spread": spread,
    }




def run_dynamic_universe(
    cfg: Config,
    index_code: str,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    *,
    layer: int = 1,
    industry_neutral: bool = True,
    minute_chunk_size: int = 25,
    l2_chunk_size: int = 40,
) -> Dict:
    use_minute = int(layer) == 2
    use_l2 = int(layer) >= 3
    layer_names = {1: "价量+VQ", 2: "价量+VQ+分钟", 3: "价量+VQ+L2"}
    layer_name = layer_names.get(int(layer), "custom")
    print("\n" + "=" * 70)
    print(
        "[动态全市场 Layer{}] {}  {}  industry_neutral={}".format(
            layer, index_code, layer_name, industry_neutral
        )
    )
    print("=" * 70)

    panel = build_pv_vq_panel(
        index_code,
        cfg.test_start,
        cfg.test_end,
        use_minute=use_minute,
        use_l2=use_l2,
        industry_neutral=industry_neutral,
        minute_lookback=int(getattr(cfg, "minute_factor_lookback", 10) or 10),
        minute_chunk_size=minute_chunk_size,
        l2_chunk_size=l2_chunk_size,
        cache_dir=Path(cfg.results_dir) / "cache",
    )
    if panel.empty:
        return {"error": "empty panel"}

    daily_n = panel.groupby("date")["symbol"].nunique()
    print(
        "[dyn] 日池规模: min={} max={} mean={:.0f}".format(
            int(daily_n.min()), int(daily_n.max()), float(daily_n.mean())
        ),
        flush=True,
    )

    bt = RotationBacktester(
        initial_cash=cfg.initial_cash,
        cost_rate=cfg.cost_rate,
        top_k=None,
        bottom_k=None,
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        long_gross=cfg.rotation_long_gross,
        short_gross=cfg.rotation_short_gross,
        rebalance_every=int(cfg.rebalance_every or 7),
        use_vol_scaling=False,
    )
    result = bt.run(panel)
    metrics = result.metrics or {}
    bh = result.equal_weight_bh_metrics or {}
    avg_to = float(result.equity["turnover"].mean()) if not result.equity.empty else 0.0
    sr = float(metrics.get("sharpe", 0.0))
    ann = float(metrics.get("annualized_return", 0.0))

    if sr >= 1.5:
        verdict = "泛化良好"
    elif sr >= 1.15:
        verdict = "行业中性有效/可进Layer3"
    elif sr >= 1.0:
        verdict = "有效但弱于熟人圈"
    else:
        verdict = "泛化不足/过拟合风险"

    print("\n[动态池回测结果]", flush=True)
    print("  Layer: {}".format(layer))
    print("  行业中性: {}".format(industry_neutral))
    print("  年化收益: {:.2f}%".format(100 * ann))
    print("  夏普比率: {:.3f}".format(sr))
    print("  最大回撤: {:.2f}%".format(100 * float(metrics.get("max_drawdown", 0.0))))
    print("  日均换手: {:.2f}%".format(100 * avg_to))
    print(
        "  EW-BH 年化: {:.2f}% 夏普: {:.3f}".format(
            100 * float(bh.get("annualized_return", 0.0)), float(bh.get("sharpe", 0.0))
        )
    )
    print("  判定: {}".format(verdict), flush=True)
    print(
        "  对比: 三源10票={:.3f}  L1无中性≈{:.3f}".format(
            THREE_LEG_BASELINE, LAYER1_CSI300_BASELINE
        ),
        flush=True,
    )

    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tag = index_code.replace(".", "")
    suffix = "L{}".format(layer)
    if industry_neutral:
        suffix = suffix + "_indneut"
    if not result.equity.empty:
        result.equity.to_csv(
            result_dir / "dynamic_universe_{}_{}_equity.csv".format(tag, suffix)
        )
    panel.to_csv(
        result_dir / "dynamic_universe_{}_{}_signals.csv".format(tag, suffix), index=False
    )

    factors = ["mom20", "mom60", "ep", "bp", "north", "north_chg*0.25"]
    if use_minute:
        factors.extend(["minute_amp", "minute_jump(-)"])
    if use_l2:
        factors.extend(["l2_obi", "l2_depth", "l2_micro", "l2_spread(-)"])

    summary = {
        "scheme": "dynamic_universe_layer{}".format(layer),
        "index_code": index_code,
        "layer": layer,
        "industry_neutral": industry_neutral,
        "verdict": verdict,
        "config": {
            "test_start": cfg.test_start,
            "test_end": cfg.test_end,
            "top_frac": top_frac,
            "bottom_frac": bottom_frac,
            "rebalance_every": int(cfg.rebalance_every or 7),
            "factors": factors,
            "minute_factors": use_minute,
            "l2_factors": use_l2,
            "transformer": False,
        },
        "pool_stats": {
            "n_signal_rows": int(len(panel)),
            "n_signal_days": int(panel["date"].nunique()),
            "n_symbols_union": int(panel["symbol"].nunique()),
            "daily_pool_min": int(daily_n.min()),
            "daily_pool_max": int(daily_n.max()),
            "daily_pool_mean": float(daily_n.mean()),
        },
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "selection_stats": result.selection_stats,
        "extras": _long_short_spread(result),
        "comparison": {
            "fixed10_three_leg_sharpe": THREE_LEG_BASELINE,
            "fixed10_vq_sharpe": FIXED10_VQ_SHARPE,
            "layer1_no_indneut_sharpe": LAYER1_CSI300_BASELINE,
            "dynamic_sharpe": sr,
            "delta_vs_layer1_no_indneut": sr - LAYER1_CSI300_BASELINE,
            "delta_vs_three_leg": sr - THREE_LEG_BASELINE,
        },
    }
    out_path = result_dir / "dynamic_universe_{}_{}_summary.json".format(tag, suffix)
    safe = _json_safe(summary)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)
    if index_code == "000300.SH":
        if int(layer) == 1 and industry_neutral:
            canon = result_dir / "dynamic_universe_summary.json"
        elif int(layer) == 3:
            canon = result_dir / "dynamic_universe_layer3_summary.json"
        else:
            canon = None
        if canon is not None:
            with open(canon, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("\n[结果] {}".format(out_path), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic CSI universe Layer-1/2/3 backtest")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--preset", default="", help="CSI300|CSI500|CSI1000")
    p.add_argument("--layer", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--bottom-frac", type=float, default=0.2)
    p.add_argument("--minute-chunk-size", type=int, default=25)
    p.add_argument("--l2-chunk-size", type=int, default=40)
    p.add_argument(
        "--industry-neutral",
        dest="industry_neutral",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-industry-neutral",
        dest="industry_neutral",
        action="store_false",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    cfg.use_minute_factors = args.layer == 2
    cfg.use_north_money = True
    cfg.use_fundamentals = True
    cfg.use_advanced_alpha = True
    cfg.use_market_risk = False
    cfg.use_vol_scaling = False
    cfg.industry_neutral_rank = bool(args.industry_neutral)
    cfg.rebalance_every = 7
    cfg.rotation_top_frac = args.top_frac
    cfg.rotation_bottom_frac = args.bottom_frac

    index_code = INDEX_PRESETS.get(args.preset.upper(), args.index) if args.preset else args.index
    cfg.universe_index_code = index_code
    cfg.universe_preset = args.preset.upper() if args.preset else "CUSTOM"

    out = run_dynamic_universe(
        cfg,
        index_code=index_code,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        layer=args.layer,
        industry_neutral=bool(args.industry_neutral),
        minute_chunk_size=args.minute_chunk_size,
        l2_chunk_size=args.l2_chunk_size,
    )
    if "error" in out:
        print("[阻断] {}".format(out), flush=True)
        sys.exit(2)

    if (
        args.layer == 1
        and bool(args.industry_neutral)
        and index_code == "000300.SH"
        and float((out.get("strategy") or {}).get("sharpe") or 0.0) >= 1.15
    ):
        print("\n[自动] Layer1 夏普≥1.15 → 启动 Layer3 (PV+VQ+L2)", flush=True)
        out3 = run_dynamic_universe(
            cfg,
            index_code=index_code,
            top_frac=args.top_frac,
            bottom_frac=args.bottom_frac,
            layer=3,
            industry_neutral=True,
            minute_chunk_size=args.minute_chunk_size,
            l2_chunk_size=args.l2_chunk_size,
        )
        if "error" in out3:
            print("[阻断 Layer3] {}".format(out3), flush=True)
            sys.exit(2)
