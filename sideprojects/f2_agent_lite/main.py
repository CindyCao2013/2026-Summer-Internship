#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F² Agent Lite entrypoint.

Scheme A: per-name multimodal agents + CS rotation backtest
Scheme B: cross-stock attention model + same rotation backtest

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.main
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest import Backtester  # noqa: E402
from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data.cross_sectional_dataset import (  # noqa: E402
    apply_score_smoothing,
    cs_predictions_to_signal_frame,
    prepare_cs_splits,
)
from sideprojects.f2_agent_lite.data.data_loader import DataLoader  # noqa: E402
from sideprojects.f2_agent_lite.train_cross_sectional import (  # noqa: E402
    predict_cs_proba,
    train_cs_model,
)
from sideprojects.f2_agent_lite.train_torch import predict_signals, train_f2_model  # noqa: E402


def _plot_equity(equity: pd.DataFrame, title: str, out_path: Path, label: str = "Strategy") -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(equity.index, equity["equity"] / equity["equity"].iloc[0], label=label)
    if "bh_equity" in equity.columns:
        ax.plot(
            equity.index,
            equity["bh_equity"] / equity["bh_equity"].iloc[0],
            label="EqualWeight BH",
            alpha=0.8,
        )
    ax.set_title(title)
    ax.set_ylabel("Growth of 1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_single_name_backtests(config: Config, signals: pd.DataFrame) -> dict:
    results_dir = Path(config.results_dir)
    summary = {}
    backtester = Backtester(
        initial_cash=config.initial_cash,
        cost_rate=config.cost_rate,
        allow_short=config.allow_short,
    )
    for symbol in sorted(signals["symbol"].unique()):
        res = backtester.run(signals, symbol=symbol)
        tag = symbol.replace(".", "_")
        res.trades.to_csv(results_dir / "trades_{}.csv".format(tag), index=False)
        res.equity.to_csv(results_dir / "equity_{}.csv".format(tag))
        if not res.equity.empty:
            _plot_equity(
                res.equity,
                "Single-name LS — {}".format(symbol),
                results_dir / "equity_{}.png".format(tag),
            )
        summary[symbol] = {
            "strategy": res.metrics,
            "buy_hold": res.buy_hold_metrics,
            "n_trades": int((~res.trades["action"].isin(["HOLD"])).sum()) if not res.trades.empty else 0,
        }
    return summary


def run_rotation_backtest(config: Config, signals: pd.DataFrame, tag: str = "portfolio") -> dict:
    results_dir = Path(config.results_dir)
    rebal = config.rebalance_every
    if rebal is None:
        rebal = int(getattr(config, "pred_horizon", 1))
    bt = RotationBacktester(
        initial_cash=config.initial_cash,
        cost_rate=config.cost_rate,
        top_k=config.rotation_top_k,
        bottom_k=config.rotation_bottom_k,
        top_frac=config.rotation_top_frac,
        bottom_frac=config.rotation_bottom_frac,
        long_gross=config.rotation_long_gross,
        short_gross=config.rotation_short_gross,
        rebalance_every=rebal,
        use_vol_scaling=bool(getattr(config, "use_vol_scaling", False)),
        vol_scaling_window=int(getattr(config, "vol_scaling_window", 20)),
        vol_scaling_floor=float(getattr(config, "vol_scaling_floor", 0.01)),
    )
    res = bt.run(signals)
    equity_csv = results_dir / "{}_equity.csv".format(tag)
    holdings_csv = results_dir / "{}_holdings.csv".format(tag)
    fig_path = results_dir / "{}_equity.png".format(tag)
    res.equity.to_csv(equity_csv)
    res.holdings.to_csv(holdings_csv, index=False)
    if not res.equity.empty:
        _plot_equity(res.equity, "F2 {} Rotation LS".format(tag), fig_path, label="Rotation LS")
    print(
        "[rotation:{}] ann={:.2%} sharpe={:.3f} mdd={:.2%} | EW-BH ann={:.2%}".format(
            tag,
            res.metrics.get("annualized_return", 0.0),
            res.metrics.get("sharpe", 0.0),
            res.metrics.get("max_drawdown", 0.0),
            res.equal_weight_bh_metrics.get("annualized_return", 0.0),
        )
    )
    print("[rotation:{}] selection {}".format(tag, json.dumps(res.selection_stats, ensure_ascii=False, default=str)))
    stats_path = results_dir / "{}_selection_stats.json".format(tag)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(res.selection_stats, f, ensure_ascii=False, indent=2, default=str)
    return {
        "strategy": res.metrics,
        "equal_weight_bh": res.equal_weight_bh_metrics,
        "selection_stats": res.selection_stats,
        "avg_daily_turnover": float(res.equity["turnover"].mean()) if not res.equity.empty else 0.0,
        "equity_csv": str(equity_csv),
        "figure": str(fig_path),
    }


def run_scheme_a(config: Config) -> dict:
    loader = DataLoader(config)
    print("[data] loading panels + windows (Scheme A) ...")
    packs = loader.prepare_all()
    train_ds, val_ds, test_ds = packs["train"], packs["val"], packs["test"]
    model, train_metrics, device = train_f2_model(config, train_ds, val_ds)
    class_id, trade_signal, attn, probs, score = predict_signals(
        model, test_ds, device, batch_size=config.batch_size
    )
    signals = test_ds.meta.copy()
    signals["class_id"] = class_id
    signals["signal"] = trade_signal
    signals["prob_short"] = probs[:, 0]
    signals["prob_hold"] = probs[:, 1]
    signals["prob_long"] = probs[:, 2]
    signals["score"] = score
    signals["proba"] = signals["prob_long"]
    for i, name in enumerate(["attn_market", "attn_tech", "attn_news", "attn_sentiment"]):
        signals[name] = attn[:, i]
    test_acc = float((class_id == test_ds.y).mean())
    return {
        "signals": signals,
        "train_metrics": train_metrics,
        "test_accuracy": test_acc,
        "model": model,
        "device": device,
        "tag": "schemeA",
    }


def run_scheme_b(config: Config) -> dict:
    print("[data] building cross-sectional day tensors (Scheme B) ...")
    splits = prepare_cs_splits(config)
    model, train_metrics, device = train_cs_model(config, splits.train, splits.val)
    probs = predict_cs_proba(model, splits.test, device, batch_size=min(16, config.batch_size))
    signals = cs_predictions_to_signal_frame(
        splits.test,
        probs,
        industry_neutral=bool(getattr(config, "industry_neutral_rank", False)),
    )
    smooth_w = int(getattr(config, "score_smooth_window", 0) or 0)
    if smooth_w > 1:
        signals = apply_score_smoothing(signals, window=smooth_w)
        print("[signal] score smoothed window={} (causal shift1)".format(smooth_w))
    test_acc = float((signals["class_id"].to_numpy() == signals["y"].to_numpy()).mean())
    # class distribution sanity
    print("[test] class_id counts", signals["class_id"].value_counts().to_dict())
    print(
        "[test] CS score range mean={:.4f}".format(
            float((signals.groupby("date")["score"].max() - signals.groupby("date")["score"].min()).mean())
        )
    )
    return {
        "signals": signals,
        "train_metrics": train_metrics,
        "test_accuracy": test_acc,
        "model": model,
        "device": device,
        "tag": "schemeB",
        "n_features": splits.train.n_features,
        "symbols": splits.symbols,
    }


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "ERROR: torch not found.\n"
            "Use: /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.main"
        )
        return 2

    config = Config()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    print("=== F² Agent Lite (scheme={}) ===".format(config.scheme))
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    if str(config.scheme).upper() == "B":
        out = run_scheme_b(config)
    else:
        out = run_scheme_a(config)

    signals = out["signals"]
    signal_path = config.results_dir / "test_signals_{}.csv".format(out["tag"])
    signals.to_csv(signal_path, index=False)
    # also keep canonical name for downstream tools
    signals.to_csv(config.results_dir / "test_signals.csv", index=False)
    print("[signal] wrote", signal_path)
    print("[test] 3-class accuracy={:.4f}".format(out["test_accuracy"]))

    ckpt_path = config.results_dir / "f2_{}_best.pt".format(out["tag"])
    torch = sys.modules["torch"]
    payload = {"model_state": out["model"].state_dict(), "config": config.to_dict()}
    if out["tag"] == "schemeB":
        payload["n_features"] = out.get("n_features")
        payload["symbols"] = out.get("symbols")
    torch.save(payload, ckpt_path)
    print("[ckpt] wrote", ckpt_path)

    report = {
        "config": config.to_dict(),
        "train_metrics": {
            "best_val_acc": out["train_metrics"].get("best_val_acc"),
            "best_val_rank_hit": out["train_metrics"].get("best_val_rank_hit"),
            "final_val": out["train_metrics"].get("final_val"),
            "device": out["train_metrics"].get("device"),
            "history_tail": (out["train_metrics"].get("history") or [])[-5:],
            "class_weights": out["train_metrics"].get("class_weights"),
            "long_class_boost": out["train_metrics"].get("long_class_boost"),
            "ce_loss_weight": out["train_metrics"].get("ce_loss_weight"),
            "rank_loss_weight": out["train_metrics"].get("rank_loss_weight"),
            "ranking_margin": out["train_metrics"].get("ranking_margin"),
        },
        "test_accuracy": out["test_accuracy"],
    }

    if config.run_single_name_backtest:
        report["single_name_backtest"] = run_single_name_backtests(config, signals)

    if config.run_rotation_backtest:
        report["rotation_backtest"] = run_rotation_backtest(
            config, signals, tag="portfolio_{}".format(out["tag"])
        )

    report_path = config.results_dir / "summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("[done] summary ->", report_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
