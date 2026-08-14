"""End-to-end synthetic smoke (one short fold). Requires torch."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from alphanet.pipeline import run_synthetic_pipeline
from alphanet.variants import smoke_config


def test_synthetic_pipeline_writes_factor_and_ic():
    cfg = smoke_config()
    result = run_synthetic_pipeline(cfg, n_days=90, n_stocks=32, n_seeds=1, max_folds=2)
    assert result["factor"].shape[0] >= 1
    assert result["factor"].shape[1] >= 10
    assert result["ic"]["summary"]["n_cs"] >= 1
    assert "H-L" in result["decile"]["group_pnl"].columns
