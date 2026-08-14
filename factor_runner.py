"""日频因子批量 runner：manifest 断点续跑 + groupTest + 结果保存。

公式层只负责 build_factor；本模块负责调度、跳过、保存。
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib


MANIFEST_COLUMNS = [
    "track",
    "factor_name",
    "universe",
    "method",
    "status",
    "output_path",
    "error",
    "run_time",
]


@dataclass
class RunnerConfig:
    track: str
    start_day: dt.datetime
    end_day: dt.datetime
    factor_list: List[str]
    batch_tag: str
    result_root: str
    manifest_path: Path
    method: str = "c2c"
    batch_mode: bool = True
    skip_completed: bool = True
    resume_from_existing: bool = True
    save_results: bool = True
    show_group_test_plots: bool = False
    universe_list: Optional[Dict[str, Optional[str]]] = None


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def already_finished(
    manifest: pd.DataFrame,
    track: str,
    fname: str,
    universe: str,
    run_method: str,
) -> bool:
    if manifest.empty:
        return False
    mask = (
        (manifest["track"] == track)
        & (manifest["factor_name"] == fname)
        & (manifest["universe"] == universe)
        & (manifest["method"] == run_method)
        & (manifest["status"] == "success")
    )
    return mask.any()


def existing_result_path(result_root: str, fname: str, universe: str) -> Path:
    return Path(result_root) / fname / universe / "summary.csv"


def legacy_result_path(track: str, fname: str, universe: str) -> Path:
    """兼容重构前 result/<factor>/ 目录。"""
    if track == "eod_pv":
        return Path("result") / fname / universe / "summary.csv"
    return Path("__no_legacy__")


def is_run_complete(
    cfg: RunnerConfig,
    manifest: pd.DataFrame,
    fname: str,
    universe: str,
) -> bool:
    if cfg.skip_completed and already_finished(
        manifest, cfg.track, fname, universe, cfg.method
    ):
        return True
    if cfg.resume_from_existing and existing_result_path(
        cfg.result_root, fname, universe
    ).exists():
        return True
    if cfg.resume_from_existing and legacy_result_path(cfg.track, fname, universe).exists():
        return True
    return False


def append_manifest_row(
    cfg: RunnerConfig,
    fname: str,
    universe: str,
    status: str,
    output_path: str = "",
    error: str = "",
):
    row = pd.DataFrame(
        [
            {
                "track": cfg.track,
                "factor_name": fname,
                "universe": universe,
                "method": cfg.method,
                "status": status,
                "output_path": output_path,
                "error": error,
                "run_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )
    if cfg.manifest_path.exists():
        old = pd.read_csv(cfg.manifest_path)
        new = pd.concat([old, row], ignore_index=True)
    else:
        cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        new = row
    new.to_csv(cfg.manifest_path, index=False)


def pending_universes(
    cfg: RunnerConfig,
    manifest: pd.DataFrame,
    fname: str,
) -> List[str]:
    return [
        name
        for name in cfg.universe_list
        if not is_run_complete(cfg, manifest, fname, name)
    ]


def get_universe_mask(session, start_day, end_day, index_code):
    session.run(
        f'startDay={start_day.strftime("%Y.%m.%d")}; '
        f'endDay={end_day.strftime("%Y.%m.%d")}; '
        f'index="{index_code}"'
    )
    pool = session.run("get_stock_pool(startDay, endDay, index)")
    mask = pool.assign(_in=1.0).pivot(index="Date", columns="Symbol", values="_in")
    mask.index = pd.to_datetime(mask.index)
    return mask


def prepare_signal(
    factor_value,
    idx,
    df_not_limit,
    df_not_st,
    df_trade_status,
    session,
    start_day,
    end_day,
):
    signal = factor_value.copy()
    if idx is not None:
        signal = signal.mul(get_universe_mask(session, start_day, end_day, idx))
    signal = signal.mul(df_not_limit)
    signal = signal.mul(df_not_st)
    signal = signal.mul(df_trade_status)
    signal = signal.shift()
    signal = signal.dropna(how="all", axis=1)
    signal = signal.dropna(how="all")
    return signal


def compute_group_stats(signal, ret, group_pnl_df, group_to_df):
    daily_pnl = group_pnl_df["H-L"]
    direction = 1 if daily_pnl.mean() > 0 else -1
    daily_pnl_adj = daily_pnl * direction

    rank_ic_daily = signal.corrwith(ret, axis=1, method="spearman")
    rank_ic = np.mean(rank_ic_daily)
    rank_ic_std = np.std(rank_ic_daily)
    icir = rank_ic / rank_ic_std * (250 ** 0.5) if rank_ic_std > 0 else np.nan
    mdd, _ = Factor_Dev_Lib.calMDD(daily_pnl_adj)
    avg_to = float(group_to_df["H-L"].mean())
    implied_fee = Factor_Dev_Lib.implied_annu_fee(avg_to)

    return {
        "direction": direction,
        "hl_annu_ret": Factor_Dev_Lib.calAnnuRet(daily_pnl_adj),
        "hl_sharpe": Factor_Dev_Lib.calSharpe(daily_pnl_adj),
        "hl_mdd": mdd,
        "hl_avg_turnover": avg_to,
        "implied_annu_fee": implied_fee,
        "rank_ic_mean": rank_ic,
        "abs_rank_ic_mean": abs(rank_ic),
        "icir": icir,
        "abs_icir": abs(icir) if not np.isnan(icir) else np.nan,
    }


def format_group_stats_title(stats: dict) -> str:
    """X-label caption for decile/H-L plots — always includes Implied AnnuFee."""
    return Factor_Dev_Lib.format_group_test_stats_title(
        direction=stats["direction"],
        annu_ret=stats["hl_annu_ret"],
        sharpe=stats["hl_sharpe"],
        mdd=stats["hl_mdd"],
        avg_turnover=stats["hl_avg_turnover"],
        rank_ic=stats["rank_ic_mean"],
        icir=stats["icir"],
        implied_fee=stats.get("implied_annu_fee"),
    )


def save_group_test_results(
    cfg: RunnerConfig,
    fname,
    universe_name,
    info,
    signal,
    ret,
    group_pnl_df,
    group_to_df,
):
    out_dir = os.path.join(cfg.result_root, fname, universe_name)
    os.makedirs(out_dir, exist_ok=True)

    cum_pnl_df = group_pnl_df.cumsum()
    group_pnl_path = os.path.join(out_dir, "group_pnl.csv")
    group_pnl_df.to_csv(group_pnl_path)
    cum_pnl_df.to_csv(os.path.join(out_dir, "cum_pnl.csv"))
    group_to_df.to_csv(os.path.join(out_dir, "group_turnover.csv"))

    stats = compute_group_stats(signal, ret, group_pnl_df, group_to_df)
    rank_ic_daily = signal.corrwith(ret, axis=1, method="spearman")
    rank_ic_daily.to_csv(os.path.join(out_dir, "rank_ic_daily.csv"), header=["rank_ic"])
    title = format_group_stats_title(stats)

    fig, ax = plt.subplots(figsize=(20, 12))
    for col_name, y in cum_pnl_df.items():
        ax.plot(y.index, y, label=col_name)
        ax.text(y.index[-1], y.iloc[-1], str(col_name), fontsize=12, verticalalignment="bottom")
    ax.legend(loc="upper left")
    ax.set_xlabel(title, fontsize=12)
    ax.set_title(info)
    fig.savefig(os.path.join(out_dir, "cum_pnl.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    group_pnl_df.mean().plot(kind="bar", ax=ax, title=info)
    fig.savefig(os.path.join(out_dir, "group_mean_pnl.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "track": cfg.track,
        "factor_name": fname,
        "universe": universe_name,
        "method": info,
        **stats,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"Saved results -> {out_dir}/")
    return summary, group_pnl_path


def save_batch_summaries(cfg: RunnerConfig, summary_rows: List[dict]):
    if not cfg.save_results or not summary_rows:
        return

    batch_df = pd.DataFrame(summary_rows)
    batch_path = os.path.join(cfg.result_root, f"batch_summary_{cfg.batch_tag}.csv")
    batch_df.to_csv(batch_path, index=False)
    print(f"\nSaved batch summary -> {batch_path}")

    screen = batch_df[
        (batch_df["abs_rank_ic_mean"] >= 0.02) | (batch_df["abs_icir"] >= 0.7)
    ].sort_values(["abs_icir", "abs_rank_ic_mean"], ascending=False)
    screen_path = os.path.join(cfg.result_root, f"batch_screen_ic_ir_{cfg.batch_tag}.csv")
    screen.to_csv(screen_path, index=False)
    print(f"Saved IC/IR screen -> {screen_path} ({len(screen)} rows)")

    sharpe = batch_df[batch_df["hl_sharpe"] >= 2.0].sort_values("hl_sharpe", ascending=False)
    sharpe_path = os.path.join(
        cfg.result_root, f"batch_screen_hl_sharpe_{cfg.batch_tag}.csv"
    )
    sharpe.to_csv(sharpe_path, index=False)
    print(f"Saved H-L Sharpe screen -> {sharpe_path} ({len(sharpe)} rows)")


def run_eod_batch(
    cfg: RunnerConfig,
    build_factor_fn: Callable[[str], pd.DataFrame],
    session,
    df_not_limit,
    df_not_st,
    df_trade_status,
) -> Tuple[dict, List[dict], int, int]:
    """Run c2c groupTest for a list of wide-table factors."""
    if cfg.universe_list is None:
        raise ValueError("universe_list is required")

    manifest = load_manifest(cfg.manifest_path)
    all_results = {}
    all_summary_rows = []
    skipped_runs = 0
    failed_runs = 0

    for fname in cfg.factor_list:
        todo = pending_universes(cfg, manifest, fname)
        if not todo:
            print(f"[SKIP DONE] {fname} | all universes complete")
            skipped_runs += len(cfg.universe_list)
            continue

        print(f"\n{'=' * 60}\n[{cfg.track}] {fname} | pending: {todo}\n{'=' * 60}")

        try:
            factor_value = build_factor_fn(fname).loc[cfg.start_day : cfg.end_day, :]
        except Exception as exc:
            print(f"[FAILED] build_factor {fname}: {exc}")
            for universe_name in todo:
                append_manifest_row(cfg, fname, universe_name, "failed", error=str(exc))
                failed_runs += 1
            manifest = load_manifest(cfg.manifest_path)
            continue

        factor_summary_rows = []
        factor_results = {}

        for universe_name in todo:
            idx = cfg.universe_list[universe_name]
            info = f"{cfg.method}_{universe_name}"

            if is_run_complete(cfg, manifest, fname, universe_name):
                print(f"[SKIP DONE] {fname} | {universe_name} | {cfg.method}")
                skipped_runs += 1
                continue

            try:
                print(f"[RUN] {fname} | {universe_name} | {cfg.method}")
                ret_matrix = Factor_Dev_Lib.get_Ret_Matrix(
                    cfg.start_day, cfg.end_day, method=cfg.method, base_index=idx
                )
                signal = prepare_signal(
                    factor_value,
                    idx,
                    df_not_limit,
                    df_not_st,
                    df_trade_status,
                    session,
                    cfg.start_day,
                    cfg.end_day,
                )
                signal_rank, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(
                    signal, ret_matrix, n=10, info=info
                )
                if cfg.batch_mode and not cfg.show_group_test_plots:
                    plt.close("all")

                factor_results[universe_name] = {
                    "signal_rank": signal_rank,
                    "group_pnl": group_pnl_df,
                    "group_to": group_to_df,
                }

                if cfg.save_results:
                    summary, output_path = save_group_test_results(
                        cfg,
                        fname,
                        universe_name,
                        info,
                        signal,
                        ret_matrix,
                        group_pnl_df,
                        group_to_df,
                    )
                    factor_summary_rows.append(summary)
                    append_manifest_row(
                        cfg, fname, universe_name, "success", output_path=str(output_path)
                    )
                else:
                    append_manifest_row(cfg, fname, universe_name, "success")

                manifest = load_manifest(cfg.manifest_path)

            except Exception as exc:
                print(f"[FAILED] {fname} | {universe_name} | {cfg.method}: {exc}")
                append_manifest_row(cfg, fname, universe_name, "failed", error=str(exc))
                failed_runs += 1
                manifest = load_manifest(cfg.manifest_path)

        if factor_summary_rows:
            pd.DataFrame(factor_summary_rows).to_csv(
                os.path.join(cfg.result_root, fname, "summary_c2c_all_universes.csv"),
                index=False,
            )

        all_results[fname] = factor_results
        all_summary_rows.extend(factor_summary_rows)

    save_batch_summaries(cfg, all_summary_rows)
    print(f"\nManifest -> {cfg.manifest_path}")
    print(
        f"Skipped: {skipped_runs} | Failed: {failed_runs} | New: {len(all_summary_rows)}"
    )
    return all_results, all_summary_rows, skipped_runs, failed_runs
