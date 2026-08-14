#!/usr/bin/env python
"""Industry + log(FloatMktCap) neutralize liquidity_resilience_proxy_5d.

Correct pipeline:
  1) build frozen factor (trade-date values, unshifted)
  2) cross-section neutralize by Citics industry + log FloatMktCap
  3) backtest with standard T+1 signal_shift

Outputs RAW vs IND_CAP full-sample + segment metrics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    backtest_factor,
    compute_rank_ic,
    narrow_to_wide,
    prepare_factor_signal,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    FULL_END,
    FULL_START,
    load_fast_context,
)
from l2_factor_reproduction.scripts.validate_liquidity_resilience_proxy_5d import (  # noqa: E402
    FACTOR,
    SEGMENTS,
    build_factor_narrow,
    metrics_from_pnl,
    neutralize_ind_log_float_mktcap,
    slice_backtest,
)

OUT_DIR = Path(RESULT_ROOT) / "full_validation" / FACTOR / "ind_cap_neutral_rerun"


def _to_narrow(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    df = wide.stack(dropna=True).rename("value").reset_index()
    df.columns = ["TradeDate", "symbol", "value"]
    df["tradetime"] = pd.to_datetime(df["TradeDate"]) + pd.Timedelta(hours=9, minutes=30)
    df["factorname"] = name
    return df[["symbol", "tradetime", "factorname", "value"]]


def _eval_mode(
    factor_wide: pd.DataFrame,
    *,
    name: str,
    start,
    end,
    mask,
    ret,
) -> dict:
    narrow = _to_narrow(factor_wide, name)
    gp, gt, _, summary = backtest_factor(
        narrow, start_day=start, end_day=end, mask=mask, ret_matrix=ret
    )
    signal, ret_aln = prepare_factor_signal(
        factor_wide, start=start, end=end, mask=mask, ret_matrix=ret
    )
    ic = compute_rank_ic(signal, ret_aln)
    direction = int(summary["factor_direction"])
    full = metrics_from_pnl(gp, gt, ic, factor_direction=direction)
    return {
        "name": name,
        "group_pnl": gp,
        "group_to": gt,
        "rank_ic": ic,
        "direction": direction,
        "full": full,
    }


def main() -> int:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, end = FULL_START, FULL_END

    print(f"[1/4] build frozen factor [{start.date()} ~ {end.date()}]", flush=True)
    narrow = build_factor_narrow(start, end)
    factor_raw = narrow_to_wide(narrow)
    print(f"  narrow={len(narrow):,}  wide={factor_raw.shape}", flush=True)

    print("[2/4] RAW backtest", flush=True)
    mask, ret = load_fast_context("full")
    raw = _eval_mode(
        factor_raw, name=FACTOR, start=start, end=end, mask=mask, ret=ret
    )
    print(
        f"  RAW dir={raw['direction']} IC={raw['full']['rank_ic_mean_raw']:.4f} "
        f"Sharpe={raw['full']['hl_sharpe']:.3f} mono={raw['full']['mono']:.3f}",
        flush=True,
    )

    print("[3/4] neutralize factor (trade-date) then backtest IND_CAP", flush=True)
    # Neutralize unshifted factor values with same-day industry/mcap, then T+1 in backtest.
    factor_neut = neutralize_ind_log_float_mktcap(factor_raw)
    neut = _eval_mode(
        factor_neut,
        name=FACTOR + "_IND_CAP",
        start=start,
        end=end,
        mask=mask,
        ret=ret,
    )
    print(
        f"  IND_CAP dir={neut['direction']} IC={neut['full']['rank_ic_mean_raw']:.4f} "
        f"Sharpe={neut['full']['hl_sharpe']:.3f} mono={neut['full']['mono']:.3f}",
        flush=True,
    )

    cmp = pd.DataFrame(
        [
            {"mode": "RAW", **raw["full"]},
            {"mode": "IND_CAP", **neut["full"]},
        ]
    )
    cmp["ic_retention"] = cmp["rank_ic_mean_raw"].abs() / abs(raw["full"]["rank_ic_mean_raw"])
    cmp["sharpe_retention"] = cmp["hl_sharpe"] / raw["full"]["hl_sharpe"]
    cmp["mono_retention"] = cmp["decile_mono_spearman"] / raw["full"]["decile_mono_spearman"]
    cmp.to_csv(OUT_DIR / "neutralization_comparison.csv", index=False)

    print("[4/4] segment tables", flush=True)
    rows = []
    for mode, pack in (("RAW", raw), ("IND_CAP", neut)):
        for seg_name, (s0, s1) in SEGMENTS.items():
            m = slice_backtest(
                pack["group_pnl"],
                pack["group_to"],
                pack["rank_ic"],
                s0,
                s1,
                pack["direction"],
            )
            m["mode"] = mode
            m["segment"] = seg_name
            rows.append(m)
    seg = pd.DataFrame(rows)
    cols = [
        "mode",
        "segment",
        "rank_ic",
        "icir",
        "hl_annu_ret",
        "hl_sharpe",
        "net_annu_after_fee",
        "mono",
        "violations",
        "positive_month_fraction",
        "mdd",
        "turnover",
        "n_days",
        "factor_direction",
    ]
    seg = seg[[c for c in cols if c in seg.columns]]
    seg.to_csv(OUT_DIR / "segment_raw_vs_ind_cap.csv", index=False)

    elapsed = time.perf_counter() - t0
    show = [
        "mode",
        "rank_ic_mean_raw",
        "icir_raw",
        "hl_annu_ret",
        "hl_sharpe",
        "net_annu_after_fee",
        "mono",
        "violations",
        "turnover",
        "ic_retention",
        "sharpe_retention",
        "mono_retention",
        "factor_direction",
    ]
    print("\n=== FULL RAW vs IND_CAP (correct: neutralize then T+1) ===", flush=True)
    print(cmp[show].to_string(index=False), flush=True)
    print("\n=== SEGMENTS ===", flush=True)
    print(seg.to_string(index=False), flush=True)
    print(f"\n[done] {OUT_DIR} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
