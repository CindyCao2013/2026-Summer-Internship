#!/usr/bin/env python
"""Factor Cutting Framework v2 — research platform (not minute upgrade).

Tasks:
  1. Full-history / DDB-era replication (universes, monthly IC, size+industry neut)
  2. Mechanism analyzer (M_high / M_low / spread + separation score)
  3. Knife evaluator (trade_count ATS vs amount/volume/turnover)

Data:
  DDB WIND.ASHAREEODPRICES starts 2018-01-02 — use --preset ddb (default).
  Paper window 2010-2025 requires --preset paper (Wind Oracle, keep_cache).

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v2.py
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v2.py --preset ddb
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v2.py --preset paper --keep-cache
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v2.py --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means
from alpha_research_report import monotonicity_score
from factor_attribution import hl_sharpe_from_composite
from factor_cutting.cutting_analysis.knife_evaluator import (
    evaluate_default_knives,
    knife_ranking_markdown,
)
from factor_cutting.cutting_analysis.knife_ic import ic_stats, prepare_signal
from factor_cutting.cutting_analysis.leg_analysis import (
    decompose_legs,
    legs_ic_timeseries,
    write_leg_mechanism_md,
)
from factor_cutting.cutting_analysis.stability import full_stability_pack
from factor_cutting.knives import build_knife
from factor_cutting.trade_count import load_trade_count_daily
from factor_cutting.w_cut import w_cut
from factor_data_loaders import (
    load_eod_enriched_tables,
    load_eod_wide_tables_from_wind_oracle,
)
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
from industry_neutral import load_citics_industry_panel

OUT = Path("research/reports/factor_cutting_v1")
UNIVERSES = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
}
PRESETS = {
    "ddb": (dt.datetime(2018, 1, 2), dt.datetime(2025, 12, 31), "ddb"),
    "paper": (dt.datetime(2010, 1, 1), dt.datetime(2025, 12, 31), "oracle"),
    "smoke": (dt.datetime(2023, 1, 1), dt.datetime(2025, 12, 31), "ddb"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_day(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d")


def load_panels(
    start: dt.datetime,
    end: dt.datetime,
    source: str,
    *,
    keep_cache: bool,
):
    """Return dict with OHLCV, ret, optional float_mktcap, session."""
    preheat = start - dt.timedelta(days=80)
    session = None
    float_mkt = None
    if source == "oracle":
        cache = Path("research/cache/factor_cutting/wind_eod") if keep_cache else None
        eod, ret = load_eod_wide_tables_from_wind_oracle(
            preheat, end, cache_dir=cache, keep_cache=keep_cache
        )
        close, open_, high, low = eod.close, eod.open, eod.high, eod.low
        amount, volume = eod.amount, eod.volume
        ret = ret.reindex(index=close.index, columns=close.columns)
    else:
        enriched, session = load_eod_enriched_tables(preheat, end)
        close, open_ = enriched.close, enriched.open
        high, low = enriched.high, enriched.low
        amount, volume = enriched.amount, enriched.volume
        float_mkt = enriched.float_mktcap
        try:
            ret = Factor_Dev_Lib.get_Ret_Matrix(preheat, end, method="c2c")
            ret = ret.reindex(index=close.index, columns=close.columns)
        except Exception as exc:
            log(f"WARNING: get_Ret_Matrix failed ({exc}); using close ratio")
            ret = close / close.shift(1) - 1.0
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "amount": amount,
        "volume": volume,
        "ret": ret,
        "float_mktcap": float_mkt,
        "session": session,
    }


def try_load_industry(start, end, close) -> Optional[pd.DataFrame]:
    """Load CITICS industry; clip start if preheat cache is shorter than request."""
    attempts = [start, dt.datetime(2020, 1, 2), dt.datetime(2018, 1, 2)]
    seen = set()
    for s0 in attempts:
        s0 = max(s0, start) if s0 != start else start
        key = s0.toordinal()
        if key in seen:
            continue
        seen.add(key)
        try:
            industry = load_citics_industry_panel(s0, end)
            industry = industry.reindex(index=close.index, columns=close.columns)
            log(f"  industry loaded from {industry.dropna(how='all').index.min().date()}")
            return industry
        except Exception as exc:
            log(f"  industry attempt {s0.date()} failed: {exc}")
    return None


def make_neut_fn(industry, float_mkt):
    if industry is None or float_mkt is None:
        return None

    def _fn(panel: pd.DataFrame) -> pd.DataFrame:
        # Only neutralize rows where industry coverage is usable
        ind = industry.reindex_like(panel)
        mkt = float_mkt.reindex_like(panel)
        out = panel.copy()
        # day-by-day would be slow if we skip inside neutralize; filter dates first
        good_days = ind.notna().sum(axis=1) >= 30
        if not good_days.any():
            return panel
        sub = panel.loc[good_days]
        neut = neutralize_size_industry(sub, ind.loc[good_days], mkt.loc[good_days])
        out.loc[good_days] = neut
        return out

    return _fn


def load_universe_masks(session, start, end, index) -> Dict[str, pd.DataFrame]:
    import intraday_lib

    if session is None:
        try:
            from factor_data_loaders import connect_ddb

            session = connect_ddb()
            own = True
        except Exception as exc:
            log(f"WARNING: cannot open DDB for universes ({exc})")
            return {}
    else:
        own = False
    masks = {}
    try:
        session.run(intraday_lib.ddb_functions)
        for uni, code in UNIVERSES.items():
            m = get_universe_mask(session, start, end, code)
            masks[uni] = m.reindex(index=index)
            log(f"  universe {uni}: mean names={masks[uni].notna().sum(axis=1).mean():.0f}")
    except Exception as exc:
        log(f"WARNING: universe masks failed ({exc})")
    finally:
        if own:
            try:
                session.close()
            except Exception:
                pass
    return masks


def save_ic_plot(ic: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ic.plot(ax=ax, alpha=0.3, lw=0.7, label="daily")
    ic.rolling(20, min_periods=10).mean().plot(ax=ax, color="crimson", label="20d MA")
    ic.rolling(60, min_periods=20).mean().plot(ax=ax, color="navy", label="60d MA")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_legs_ic_plot(legs_ic: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for col, color in (("high", "crimson"), ("low", "gray"), ("spread", "navy")):
        if col not in legs_ic:
            continue
        s = legs_ic[col].rolling(60, min_periods=20).mean()
        s.plot(ax=ax, color=color, label=f"{col} (60d MA)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("IC decomposition: high / low / spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_group_plot(panel: pd.DataFrame, ret: pd.DataFrame, path: Path, title: str) -> None:
    means = decile_group_means(panel, ret)
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Decile (1=low factor)")
    ax.set_ylabel("Mean fwd return")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def publish_factor_report(
    name: str,
    raw: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    out_dir: Path,
    neut_fn,
    universe_masks: Dict[str, pd.DataFrame],
    knife_name: str = "",
    object_panel: Optional[pd.DataFrame] = None,
    knife_panel: Optional[pd.DataFrame] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_signal(raw.reindex_like(ret), neutralize_fn=neut_fn, zscore=True)

    pack = full_stability_pack(panel, ret, universe_masks=universe_masks or None)
    ic_daily = pack["full"]["ic_daily"]
    ic_daily.to_csv(out_dir / "ic.csv", header=["rank_ic"])
    pack["yearly"].to_csv(out_dir / "yearly.csv", index=False)
    pack["regime"].to_csv(out_dir / "regime.csv", index=False)
    if "universe" in pack:
        pack["universe"].to_csv(out_dir / "universe.csv", index=False)
    monthly = pack["monthly"]
    if "monthly_ic" in monthly:
        monthly["monthly_ic"].to_csv(out_dir / "monthly_ic.csv", header=["rank_ic"])

    sharpe, ann, direction = hl_sharpe_from_composite(panel, ret)
    mono = monotonicity_score(decile_group_means(panel, ret))
    save_ic_plot(ic_daily, f"{name} RankIC", out_dir / "ic_timeseries.png")
    save_group_plot(panel, ret, out_dir / "group_test.png", f"{name} decile means")

    mech = {}
    if object_panel is not None and knife_panel is not None:
        summary, detail = decompose_legs(object_panel, knife_panel, ret)
        summary.to_csv(out_dir / "legs.csv", index=False)
        legs_ic = legs_ic_timeseries(detail)
        legs_ic.to_csv(out_dir / "legs_ic_daily.csv")
        save_legs_ic_plot(legs_ic, out_dir / "legs_ic.png")
        write_leg_mechanism_md(
            out_dir / "mechanism.md",
            factor_name=name,
            knife_name=knife_name,
            summary=summary,
            paper_note=(
                "Paper claim: high-knife days concentrate alpha; low-knife days are noise. "
                "Spread = purification via difference."
            ),
        )
        mech = {
            "legs": summary.to_dict(orient="records"),
            "separation": summary.attrs.get("knife_separation"),
            "purity": summary.attrs.get("knife_purity"),
        }

    summary = {
        "factor": name,
        "knife": knife_name,
        "rank_ic": pack["full_meta"]["rank_ic"],
        "icir": pack["full_meta"]["icir"],
        "ic_pos_ratio": pack["full_meta"]["ic_pos_ratio"],
        "n_days": pack["full_meta"]["n_days"],
        "monthly_rank_ic": pack["monthly_meta"]["monthly_rank_ic"],
        "monthly_icir": pack["monthly_meta"]["monthly_icir"],
        "n_months": pack["monthly_meta"]["n_months"],
        "hl_sharpe": sharpe,
        "hl_annu_ret": ann,
        "direction": direction,
        "monotonicity": mono,
        "neutralized": neut_fn is not None,
        "mechanism": mech,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="ddb")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-knife-eval", action="store_true")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--no-neut", action="store_true")
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== Factor Cutting v2 | {start.date()} -> {end.date()} | source={source} ===")

    close_bundle = load_panels(start, end, source, keep_cache=args.keep_cache or args.preset == "paper")
    close = close_bundle["close"]
    open_ = close_bundle["open"]
    high = close_bundle["high"]
    low = close_bundle["low"]
    amount = close_bundle["amount"]
    volume = close_bundle["volume"]
    ret_full = close_bundle["ret"]
    session = close_bundle["session"]
    float_mkt = close_bundle["float_mktcap"]

    ret = ret_full.loc[start:end]
    close_s = close.loc[:end]
    log(f"Panel: {close_s.shape[0]}d x {close_s.shape[1]} | ret days={len(ret)}")

    trade_count = None
    try:
        tc_start = max(start - dt.timedelta(days=80), dt.datetime(2018, 11, 27))
        if end >= tc_start:
            log("Loading trade_count (L2 daily counts)...")
            trade_count = load_trade_count_daily(tc_start, end)
            trade_count = trade_count.reindex(index=close_s.index, columns=close_s.columns)
            log(f"  trade_count mean names={trade_count.notna().sum(axis=1).mean():.0f}")
    except Exception as exc:
        log(f"WARNING: trade_count unavailable ({exc})")

    industry = None
    neut_fn = None
    if not args.no_neut:
        industry = try_load_industry(start, end, close_s)
        if float_mkt is not None:
            float_mkt = float_mkt.reindex(index=close_s.index, columns=close_s.columns)
        elif source == "oracle":
            log("WARNING: float_mktcap missing on oracle path — trying DDB overlap only")
            try:
                en, _ = load_eod_enriched_tables(max(start, dt.datetime(2018, 1, 2)), end)
                float_mkt = en.float_mktcap.reindex(index=close_s.index, columns=close_s.columns)
            except Exception as exc:
                log(f"WARNING: float_mktcap unavailable ({exc})")
        neut_fn = make_neut_fn(industry, float_mkt)
        log(f"Neutralization: {'size+industry' if neut_fn else 'OFF'}")

    universe_masks: Dict[str, pd.DataFrame] = {}
    if not args.skip_universe:
        log("Loading universe masks...")
        universe_masks = load_universe_masks(session, start, end, ret.index)

    # --- ideal reversal ---
    ret_1d = close_s / close_s.shift(1) - 1.0
    if trade_count is not None and trade_count.notna().any().any():
        knife_name = "ats_trade_count"
        knife = build_knife(
            knife_name, amount=amount.reindex_like(close_s), trade_count=trade_count
        )
    else:
        knife_name = "ats_volume"
        knife = build_knife(
            knife_name,
            amount=amount.reindex_like(close_s),
            volume=volume.reindex_like(close_s),
        )
    log(f"Ideal reversal knife={knife_name}")
    m_raw = w_cut(ret_1d, knife, window=20)
    rev_sum = publish_factor_report(
        "ideal_reversal",
        m_raw.loc[start:end],
        ret,
        out_dir=OUT / "ideal_reversal",
        neut_fn=neut_fn,
        universe_masks=universe_masks,
        knife_name=knife_name,
        object_panel=ret_1d.loc[start:end],
        knife_panel=knife.loc[start:end],
    )
    log(
        f"  ideal_reversal RankIC={rev_sum['rank_ic']:.4f} ICIR={rev_sum['icir']:.2f} "
        f"monthly_IC={rev_sum['monthly_rank_ic']:.4f}"
    )

    # --- ideal amplitude ---
    log("Building ideal_amplitude...")
    from factor_cutting.ideal_amplitude import compute_ideal_amplitude as _amp

    amp_fac, v_high, v_low = _amp(
        high.reindex_like(close_s),
        low.reindex_like(close_s),
        close_s,
        open_=open_.reindex_like(close_s),
        return_legs=True,
    )
    amp_sum = publish_factor_report(
        "ideal_amplitude",
        amp_fac.loc[start:end],
        ret,
        out_dir=OUT / "ideal_amplitude",
        neut_fn=neut_fn,
        universe_masks=universe_masks,
        knife_name="close_price_state_lambda25",
    )
    amp_dir = OUT / "ideal_amplitude"
    leg_rows = []
    for leg_name, leg in (("high", v_high), ("low", v_low), ("spread", amp_fac)):
        st = ic_stats(prepare_signal(leg.loc[start:end], neutralize_fn=neut_fn), ret)
        leg_rows.append(
            {
                "leg": leg_name,
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
                "ic_pos_ratio": st["ic_pos_ratio"],
                "n_days": st["n_days"],
            }
        )
    amp_legs = pd.DataFrame(leg_rows)
    ic_h = float(amp_legs.loc[amp_legs["leg"] == "high", "rank_ic"].iloc[0])
    ic_l = float(amp_legs.loc[amp_legs["leg"] == "low", "rank_ic"].iloc[0])
    ic_s = float(amp_legs.loc[amp_legs["leg"] == "spread", "rank_ic"].iloc[0])
    amp_legs.attrs["knife_separation"] = ic_h - ic_l
    denom = max(abs(ic_h), abs(ic_l), 1e-12)
    amp_legs.attrs["knife_purity"] = abs(ic_s) / denom
    amp_legs.to_csv(amp_dir / "legs.csv", index=False)
    write_leg_mechanism_md(
        amp_dir / "mechanism.md",
        factor_name="ideal_amplitude",
        knife_name="close_price_state_lambda25",
        summary=amp_legs,
        paper_note="Paper claim: high-price-state amplitude carries stronger negative alpha.",
    )
    log(
        f"  ideal_amplitude RankIC={amp_sum['rank_ic']:.4f} ICIR={amp_sum['icir']:.2f} "
        f"monthly_IC={amp_sum['monthly_rank_ic']:.4f}"
    )

    # --- knife evaluator ---
    knife_rank = None
    if not args.skip_knife_eval:
        log("Knife evaluator on daily return object...")
        knife_rank = evaluate_default_knives(
            ret_1d.loc[start:end],
            ret,
            amount=amount.reindex_like(ret),
            volume=volume.reindex_like(ret),
            trade_count=trade_count.reindex_like(ret) if trade_count is not None else None,
            turnover=None,
        )
        ke = OUT / "knife_eval"
        ke.mkdir(parents=True, exist_ok=True)
        knife_rank.to_csv(ke / "knife_ranking.csv", index=False)
        (ke / "ranking.md").write_text(knife_ranking_markdown(knife_rank), encoding="utf-8")
        log("  knife ranking:")
        for _, r in knife_rank.iterrows():
            log(
                f"    {r['knife']}: spread_IC={r['ic_spread']:.4f} "
                f"sep={r['separation']:.4f} score={r['effectiveness']:.4f}"
            )

    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "n_days": int(len(ret)),
        "source": source,
        "preset": args.preset,
        "neutralized": neut_fn is not None,
        "ideal_reversal": rev_sum,
        "ideal_amplitude": amp_sum,
        "knife_eval_top": (
            knife_rank.head(3).to_dict(orient="records") if knife_rank is not None else []
        ),
        "paper_benchmark": {
            "ideal_reversal_rank_ic": -0.0606,
            "ideal_amplitude_rank_ic": -0.0700,
            "note": "Kaiyuan reported ~full-history monthly/daily IC levels; compare sign + magnitude.",
        },
    }
    (OUT / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Factor Cutting v1 Reports (Framework v2 runner)",
                "",
                f"Period: `{verdict['period']}` · source=`{source}` · neut=`{neut_fn is not None}`",
                "",
                "## Ideal reversal",
                f"- RankIC `{rev_sum['rank_ic']:.4f}` · ICIR `{rev_sum['icir']:.2f}` · knife `{rev_sum['knife']}`",
                f"- See `ideal_reversal/mechanism.md`",
                "",
                "## Ideal amplitude",
                f"- RankIC `{amp_sum['rank_ic']:.4f}` · ICIR `{amp_sum['icir']:.2f}`",
                f"- See `ideal_amplitude/mechanism.md`",
                "",
                "## Knife eval",
                "- `knife_eval/ranking.md`",
                "",
                "Next: `--preset paper` for 2010–2025 Oracle replication (slow, cached).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
