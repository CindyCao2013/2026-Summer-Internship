#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Narrow grid with causal score smoothing: thr x rebalance_every.

Trains once per label_threshold, then backtests each rebalance_every.

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_smooth_grid
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester
from sideprojects.f2_agent_lite.config import Config
from sideprojects.f2_agent_lite.data.cross_sectional_dataset import (
    apply_score_smoothing,
    cs_predictions_to_signal_frame,
    prepare_cs_splits,
)
from sideprojects.f2_agent_lite.train_cross_sectional import predict_cs_proba, train_cs_model


GRID = [
    ("G1", 0.012, 5),
    ("G2", 0.015, 5),
    ("G3", 0.012, 7),
    ("G4", 0.015, 7),
]


def _train_signals(config: Config):
    splits = prepare_cs_splits(config)
    model, train_metrics, device = train_cs_model(config, splits.train, splits.val)
    probs = predict_cs_proba(model, splits.test, device, batch_size=min(16, config.batch_size))
    signals = cs_predictions_to_signal_frame(
        splits.test,
        probs,
        industry_neutral=bool(config.industry_neutral_rank),
    )
    w = int(getattr(config, "score_smooth_window", 3) or 0)
    if w > 1:
        signals = apply_score_smoothing(signals, window=w)
        print("[grid] smoothed scores window={}".format(w))
    return signals, train_metrics


def _backtest(config: Config, signals, tag: str):
    rebal = int(config.rebalance_every or config.pred_horizon)
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
    )
    res = bt.run(signals)
    out_dir = Path(config.results_dir)
    res.equity.to_csv(out_dir / "{}_equity.csv".format(tag))
    avg_to = float(res.equity["turnover"].mean()) if not res.equity.empty else 0.0
    return {
        "strategy": res.metrics,
        "equal_weight_bh": res.equal_weight_bh_metrics,
        "selection_stats": res.selection_stats,
        "avg_daily_turnover": avg_to,
    }


def main() -> int:
    base = Config()
    base.results_dir.mkdir(parents=True, exist_ok=True)
    base.score_smooth_window = 3

    by_thr = {}
    for gid, thr, rebal in GRID:
        by_thr.setdefault(thr, []).append((gid, rebal))

    rows = []
    trained = {}
    for thr, items in by_thr.items():
        cfg = deepcopy(base)
        cfg.label_threshold = thr
        cfg.rebalance_every = items[0][1]
        print("\n=== TRAIN thr={} ===".format(thr))
        signals, train_metrics = _train_signals(cfg)
        trained[thr] = (signals, train_metrics)
        for gid, rebal in items:
            cfg_bt = deepcopy(cfg)
            cfg_bt.rebalance_every = rebal
            print("\n=== BACKTEST {} thr={} rebal={} ===".format(gid, thr, rebal))
            bt = _backtest(cfg_bt, signals, tag="grid_{}".format(gid))
            s = bt["strategy"]
            row = {
                "id": gid,
                "label_threshold": thr,
                "rebalance_every": rebal,
                "annualized_return": s.get("annualized_return"),
                "sharpe": s.get("sharpe"),
                "max_drawdown": s.get("max_drawdown"),
                "avg_daily_turnover": bt["avg_daily_turnover"],
                "bh_ann": bt["equal_weight_bh"].get("annualized_return"),
                "long_top3": list(
                    (bt["selection_stats"].get("long_pick_counts") or {}).items()
                )[:3],
                "short_top3": list(
                    (bt["selection_stats"].get("short_pick_counts") or {}).items()
                )[:3],
                "best_val_rank_hit": train_metrics.get("best_val_rank_hit"),
            }
            rows.append(row)
            print(
                "[{}] ann={:.2%} sharpe={:.3f} mdd={:.2%} turnover={:.2%}".format(
                    gid,
                    row["annualized_return"] or 0.0,
                    row["sharpe"] or 0.0,
                    row["max_drawdown"] or 0.0,
                    row["avg_daily_turnover"] or 0.0,
                )
            )

    # Prefer higher Sharpe, then higher ann
    best = max(rows, key=lambda r: (r["sharpe"] or -999, r["annualized_return"] or -999))
    report = {
        "score_smooth_window": base.score_smooth_window,
        "grid": rows,
        "best": best["id"],
    }
    out = base.results_dir / "hyperopt_smooth_grid.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("\n[done] wrote", out)
    print("BEST =", best["id"], best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
