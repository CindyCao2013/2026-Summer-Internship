#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方向二：参数敏感性分析——“中单”阈值 (L, H] 网格扫描。

做法：
- CH 内一次 bucketed 查询（10 个边界的累计金额分桶），pandas 拼装 5x5=25 组阈值
- 每组 mid_ratio(L,H) = (cum_H - cum_L) / Total，走标准回测腿（shift(1) + mask + groupTest）
- 产出 ICIR / 净年化 两张热力图 + Top5 表

阈值网格：L ∈ {2,3,4,5,6}万，H ∈ {10,15,20,25,30}万（研报口径为 4万/20万）

用法:
  python l2_factor_reproduction/scripts/analyze_param_sensitivity.py
  python l2_factor_reproduction/scripts/analyze_param_sensitivity.py --start 2023-01-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import (  # noqa: E402
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_TradeStatus,
    groupTest,
)
from l2_factor_reproduction.config.settings import (  # noqa: E402
    END_DAY,
    RESULT_ROOT,
    START_DAY,
)
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    compute_rank_ic,
    prepare_factor_signal,
    summarize_backtest,
)
from l2_factor_reproduction.python.ch_tick import fetch_tick_bucketed  # noqa: E402

L_GRID = [20_000, 30_000, 40_000, 50_000, 60_000]
H_GRID = [100_000, 150_000, 200_000, 250_000, 300_000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=str(START_DAY.date()))
    parser.add_argument("--end", default=str(END_DAY.date()))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    out_dir = Path(RESULT_ROOT) / "mid_order_ratio" / "analysis" / "param_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"tick_bucketed_{start.date()}_{end.date()}.parquet"

    if cache.exists():
        print(f"加载缓存: {cache}")
        bucketed = pd.read_parquet(cache)
    else:
        # 按季度分块查询，避免单次全表扫 1.5 年 Tick 被服务端拖死；
        # 每块独立落盘，中断后重跑可续。
        chunk_dir = out_dir / "chunks"
        chunk_dir.mkdir(exist_ok=True)
        bounds = pd.date_range(start, end, freq="QS")
        edges = [start] + [b for b in bounds if start < b <= end] + [end + pd.Timedelta(days=1)]
        parts = []
        for c_start, c_end_ex in zip(edges[:-1], edges[1:]):
            c_end = c_end_ex - pd.Timedelta(days=1)
            cp = chunk_dir / f"chunk_{c_start.date()}_{c_end.date()}.parquet"
            if cp.exists():
                print(f"  分块缓存: {c_start.date()} ~ {c_end.date()}", flush=True)
                parts.append(pd.read_parquet(cp))
                continue
            print(f"  CH 查询: {c_start.date()} ~ {c_end.date()} ...", flush=True)
            part = fetch_tick_bucketed(c_start, c_end, boundaries=L_GRID + H_GRID)
            part.to_parquet(cp)
            print(f"    rows={len(part)}", flush=True)
            parts.append(part)
        bucketed = pd.concat(parts, ignore_index=True)
        bucketed.to_parquet(cache)
        print(f"symbol-days: {len(bucketed)}")

    mask = (
        get_EOD_Not_Limit(start, end)
        * get_EOD_Not_ST(start, end)
        * get_TradeStatus(start, end)
    )

    bucketed["TradeDate"] = pd.to_datetime(bucketed["TradeDate"])
    total = pd.to_numeric(bucketed["TotalAmount"], errors="coerce")

    records = []
    for L in L_GRID:
        for H in H_GRID:
            if L >= H:
                continue
            tag = f"L{int(L/10000)}w_H{int(H/10000)}w"
            med = pd.to_numeric(bucketed[f"cum_{H}"], errors="coerce") - pd.to_numeric(
                bucketed[f"cum_{L}"], errors="coerce"
            )
            value = med / total.replace(0, pd.NA)
            narrow = pd.DataFrame({
                "symbol": bucketed["symbol"].astype(str),
                "tradetime": bucketed["TradeDate"],
                "value": value,
            }).dropna(subset=["value"])
            wide = narrow.pivot_table(index="tradetime", columns="symbol", values="value", aggfunc="last")
            wide.index = pd.to_datetime(wide.index).normalize()
            wide = wide.sort_index()

            signal, ret = prepare_factor_signal(wide, start=start, end=end, mask=mask, signal_shift=1)
            rank_ic = compute_rank_ic(signal, ret)
            _, group_pnl, group_to = groupTest(signal, ret, n=10, info="silent")
            s = summarize_backtest(signal, ret, group_pnl, group_to, rank_ic)
            net = s["hl_annu_ret_flipped"] - s["implied_annu_fee"]
            records.append({
                "L_wan": L / 10000, "H_wan": H / 10000, "combo": tag,
                "rank_ic": s["rank_ic_mean"], "icir": s["rank_icir"],
                "annu_ret": s["hl_annu_ret_flipped"], "sharpe": s["hl_sharpe_flipped"],
                "turnover": s["avg_hl_turnover"], "cost": s["implied_annu_fee"],
                "net_annu": net,
            })
            print(f"{tag}: RankIC={s['rank_ic_mean']:.4f} ICIR={s['rank_icir']:.2f} "
                  f"Sharpe={s['hl_sharpe_flipped']:.2f} 净年化={net:.2%}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    # --- 热力图（行=L, 列=H；|ICIR| 与 净年化）---
    for col, title, fname, fmt in [
        ("icir", "|ICIR| (annualized, abs)", "heatmap_icir.png", "{:.2f}"),
        ("net_annu", "Net annual return (after 7.5bps fee)", "heatmap_net.png", "{:.1%}"),
    ]:
        piv = df.pivot(index="L_wan", columns="H_wan", values=col)
        vals = piv.abs().values if col == "icir" else piv.values
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(vals, cmap="RdYlGn" if col == "net_annu" else "viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{h:.0f}w" for h in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{l:.0f}w" for l in piv.index])
        ax.set_xlabel("H (medium upper bound)")
        ax.set_ylabel("L (medium lower bound)")
        ax.set_title(f"mid_order_ratio — {title}\n({start.date()} ~ {end.date()})")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, fmt.format(piv.values[i, j]), ha="center", va="center",
                        fontsize=9, color="black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    # 标注研报口径（4w, 20w）
    top5 = df.reindex(df["icir"].abs().sort_values(ascending=False).index).head(5)
    report_ref = df[(df["L_wan"] == 4) & (df["H_wan"] == 20)]
    print("\n=== Top5 by |ICIR| ===")
    print(top5[["combo", "rank_ic", "icir", "sharpe", "net_annu"]].to_string(index=False))
    print("\n=== 研报口径 L4w/H20w ===")
    print(report_ref[["combo", "rank_ic", "icir", "sharpe", "net_annu"]].to_string(index=False))
    top5.to_csv(out_dir / "top5.csv", index=False)
    print(f"\n输出目录: {out_dir}/")


if __name__ == "__main__":
    main()
