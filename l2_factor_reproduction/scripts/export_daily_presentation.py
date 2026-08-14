#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日频因子结果呈现表（C2C 主框架，8 核心字段 + 净收益）。

口径（与 Factor_Dev_Lib / 本仓库统一，勿改成 252）：
- 年化基准 = **250** 个交易日
- 收益 = 相对 UNIVERSE 的超额 c2c + signal.shift(1)
- H-L 展示：若原始 H-L 均值为负则翻向（direction_flip=-1 表示取负后使用）
- Turnover_Multiple = Σ|Δw| 日均值；**1.0 = 100% 日换手**，不是「1%」
- Cost_Annual_pct = Turnover_Multiple × 7.5bps × 250（单边费率假设）
- Net_AnnRet = AnnRet_HL - Cost_Annual_pct
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import calAnnuRet, calMDD, calSharpe, implied_annu_fee
from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE

FEE_BPS = 7.5
N_ANN = 250  # A股交易日惯例；全库统一，禁止改成 252


def build_row(factor: str, result_root: Path) -> dict:
    pnl = pd.read_csv(result_root / factor / "group_pnl.csv", index_col=0, parse_dates=True)
    to = pd.read_csv(result_root / factor / "group_turnover.csv", index_col=0, parse_dates=True)
    ic = pd.read_csv(result_root / factor / "rank_ic.csv", index_col=0, parse_dates=True).iloc[:, 0]

    hl_raw = pnl["H-L"]
    direction = 1 if hl_raw.mean() > 0 else -1
    hl = hl_raw * direction
    hl_to = to["H-L"].reindex(hl.index)

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = ic_mean / ic_std * (N_ANN ** 0.5) if ic_std > 0 else float("nan")
    mdd, _ = calMDD(hl)
    turnover = float(hl_to.mean())
    ann_ret = float(calAnnuRet(hl, n=N_ANN))
    cost = float(implied_annu_fee(turnover, fee_bps=FEE_BPS))

    return {
        "factor": factor,
        "trade_mode": "C2C",
        "benchmark": UNIVERSE,
        "ann_days": N_ANN,
        # 日均 H-L（翻向后），单位 bps；勿与年化收益混用
        "Ret_HL_bps": float(hl.mean() * 1e4),
        # 年化收益（小数，如 0.256 = 25.6%）
        "AnnRet_HL": ann_ret,
        "Sharpe_HL": float(calSharpe(hl, n=N_ANN)),
        "MDD_HL": float(mdd),
        "IC_Mean": ic_mean,
        "ICIR": float(icir),
        # 换手倍数：1.0 = 100% 日换手（L1 权重），不是 1%
        "Turnover_Multiple": turnover,
        # 年化单边成本（小数，如 0.514 = 51.4%）
        "Cost_Annual_pct": cost,
        "Net_AnnRet": ann_ret - cost,
        "direction_flip": direction,
        "n_days": int(hl.dropna().shape[0]),
    }


NOTES = """
NOTES
-----
1. 年化基准 = 250 个交易日（A股惯例，与仓库 Factor_Dev_Lib 一致；非 252）。
2. Turnover_Multiple：权重 L1 换手日均值；1.0 代表 100% 日换手。H-L = G10换手 + G1换手。
3. Cost_Annual_pct = Turnover_Multiple × 7.5bps × 250（单边费率假设）。
4. Net_AnnRet = AnnRet_HL - Cost_Annual_pct（毛年化减隐含成本）。
5. direction_flip=-1 表示原始 H-L=G10-G1 为负，报表已翻向；IC_Mean 仍为原始 Rank IC（未翻向）。
6. 收益为相对 benchmark 的超额 c2c，非截面等权中性再减一遍。
"""


def main() -> None:
    root = Path(RESULT_ROOT)
    factors = [
        "avg_outflow_ratio",
        "big_order_net_inflow",
        "big_order_drive_ret",
    ]
    rows = [build_row(f, root) for f in factors if (root / f / "group_pnl.csv").exists()]
    df = pd.DataFrame(rows)

    out_csv = root / "daily_factor_presentation.csv"
    df.to_csv(out_csv, index=False)

    out_md = root / "daily_factor_presentation.md"
    # 可读百分比列（仅 markdown，不改 CSV 小数口径）
    show = df.copy()
    show["AnnRet_HL%"] = (show["AnnRet_HL"] * 100).map(lambda x: f"{x:.2f}%")
    show["Sharpe_HL"] = show["Sharpe_HL"].map(lambda x: f"{x:.2f}")
    show["MDD_HL%"] = (show["MDD_HL"] * 100).map(lambda x: f"{x:.2f}%")
    show["IC_Mean"] = show["IC_Mean"].map(lambda x: f"{x:.4f}")
    show["ICIR"] = show["ICIR"].map(lambda x: f"{x:.2f}")
    show["Ret_HL_bps"] = show["Ret_HL_bps"].map(lambda x: f"{x:.2f}")
    show["Turnover_Multiple"] = show["Turnover_Multiple"].map(lambda x: f"{x:.2f}")
    show["Cost_Annual%"] = (show["Cost_Annual_pct"] * 100).map(lambda x: f"{x:.2f}%")
    show["Net_AnnRet%"] = (show["Net_AnnRet"] * 100).map(lambda x: f"{x:.2f}%")
    cols = [
        "factor",
        "trade_mode",
        "benchmark",
        "Ret_HL_bps",
        "AnnRet_HL%",
        "Sharpe_HL",
        "MDD_HL%",
        "IC_Mean",
        "ICIR",
        "Turnover_Multiple",
        "Cost_Annual%",
        "Net_AnnRet%",
        "direction_flip",
    ]
    # 手写 markdown 表，避免依赖 tabulate
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(show.loc[i, c]) for c in cols) + " |" for i in show.index
    )
    md = [
        "# 日频因子标准化呈现表（C2C）",
        "",
        f"**年化基准 = {N_ANN} 个交易日**；换手为权重 L1 **倍数**（1.0=100%日换手），非百分比点。",
        "",
        header,
        sep,
        body,
        "",
        NOTES.strip(),
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(df.to_string(index=False))
    print(NOTES)
    print(f"saved -> {out_csv}")
    print(f"saved -> {out_md}")


if __name__ == "__main__":
    main()
