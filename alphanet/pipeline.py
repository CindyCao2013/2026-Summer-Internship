"""End-to-end orchestration: prepare → train/predict → neutralize → eval → SHAP."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from alphanet.config import AlphaNetConfig
from alphanet.data import MarketPanel, panel_from_synthetic
from alphanet.evaluate import decile_backtest, ic_test, write_eval_artifacts
from alphanet.neutralize import neutralize_panel
from alphanet.paths import FACTORS, REPORTS, ensure_result_dirs
from alphanet.rolling import rolling_predict
from alphanet.ratios import ensure_features
from alphanet.synthetic import make_synthetic_panel
from alphanet.variants import get_config


def run_synthetic_pipeline(
    cfg: Optional[AlphaNetConfig] = None,
    *,
    n_days: int = 80,
    n_stocks: int = 36,
    n_seeds: int = 1,
    max_folds: int = 1,
) -> Dict[str, object]:
    """CPU-friendly pipeline used by smoke tests. No DolphinDB."""
    cfg = cfg or get_config("smoke")
    ensure_result_dirs()
    synth = make_synthetic_panel(n_days=n_days, n_stocks=n_stocks, seed=0)
    synth.features = ensure_features(synth.features, cfg.model.feature_names)
    panel = panel_from_synthetic(synth)
    panel.meta["feature_names"] = cfg.model.feature_names
    cfg = _clip_cfg_to_panel(cfg, panel)
    factor = rolling_predict(panel, cfg, n_seeds=n_seeds, max_folds=max_folds)
    if factor.empty:
        raise RuntimeError("rolling_predict returned no factor values")
    neut = neutralize_panel(
        factor,
        industry=panel.industry,
        log_mcap=panel.log_mcap,
        ret_1d=panel.ret_1d,
        turn=panel.features["turn"],
        horizon=cfg.train.horizon,
        min_obs=cfg.eval.min_cs_obs,
    )
    ic_result = ic_test(neut, panel.ret_1d, horizon=cfg.eval.rebalance_every, mask=panel.tradable)
    decile_result = decile_backtest(
        neut, panel.ret_1d, eval_cfg=cfg.eval, mask=panel.tradable
    )
    paths = write_eval_artifacts(cfg.variant, ic_result, decile_result)
    neut.to_parquet(FACTORS / "{}_factor_neutral.parquet".format(cfg.variant))
    return {
        "factor": factor,
        "neutral": neut,
        "ic": ic_result,
        "decile": decile_result,
        "paths": paths,
        "panel": panel,
        "cfg": cfg,
    }


def _clip_cfg_to_panel(cfg: AlphaNetConfig, panel: MarketPanel) -> AlphaNetConfig:
    from dataclasses import replace

    start = str(panel.calendar[0].date())
    end = str(panel.calendar[-1].date())
    return replace(cfg, start=start, end=end)


def write_run_readme(cfg: AlphaNetConfig) -> None:
    ensure_result_dirs()
    text = "\n".join(
        [
            "# AlphaNet run: {}".format(cfg.variant),
            "",
            "- start: {}".format(cfg.start),
            "- end: {}".format(cfg.end),
            "- horizon: {}".format(cfg.train.horizon),
            "- execution: {}".format(cfg.train.execution),
            "- fee_one_way: {}".format(cfg.eval.fee_one_way),
            "- optimizer: {}".format(cfg.train.optimizer),
            "",
            "See ``alphanet/docs/00_alphanet_reproduction_guide.md`` and "
            "``alphanet/docs/01_alphanet_v2_v3_guide.md``.",
            "",
        ]
    )
    (REPORTS / "{}_run.md".format(cfg.variant)).write_text(text, encoding="utf-8")
