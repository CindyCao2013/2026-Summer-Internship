#!/usr/bin/env python
"""Explain the latest checkpoint with SHAP / permutation / activation fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from alphanet.data import load_eod_from_ddb
from alphanet.dataset import AlphaNetDataset
from alphanet.model import build_model
from alphanet.paths import MODELS
from alphanet.shap_explain import explain_model
from alphanet.variants import get_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v1")
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()
    cfg = get_config(args.variant)
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is None:
        pts = sorted(MODELS.glob("{}/**/*.pt".format(cfg.variant)))
        if not pts:
            print("no checkpoints under", MODELS / cfg.variant)
            return 1
        ckpt = pts[-1]
    blob = torch.load(ckpt, map_location="cpu")
    model = build_model(cfg.model)
    model.load_state_dict(blob["state_dict"])
    panel = load_eod_from_ddb(cfg.start, cfg.end)
    dates = panel.calendar[-40:]
    ds = AlphaNetDataset(panel, dates, cfg.train, lookback=cfg.model.lookback, require_label=True)
    n = min(128, len(ds))
    images = np.stack([ds.images[i] for i in range(n)])
    labels = np.array([ds.labels[i] for i in range(n)], dtype=np.float32)
    out = explain_model(model, images, labels, variant=cfg.variant)
    for k, df in out.items():
        print("==", k, "==")
        print(df.head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
