"""Train loop: RMSprop / Adam, MSE, early stopping on validation loss."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from alphanet.config import ModelConfig, TrainConfig
from alphanet.model import build_model
from alphanet.paths import MODELS, ensure_result_dirs


def _torch():
    import torch

    return torch


def set_seed(seed: int) -> None:
    torch = _torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_optimizer(model, train_cfg: TrainConfig):
    torch = _torch()
    name = str(train_cfg.optimizer).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay or 1e-4)
    raise KeyError("unknown optimizer {!r}".format(train_cfg.optimizer))


@dataclass
class TrainResult:
    model: Any
    best_val_loss: float
    epochs_run: int
    history: Dict[str, list]
    seed: int
    runtime_sec: float
    ckpt_path: Optional[str] = None


def _run_epoch(model, loader, optimizer, device, train: bool) -> float:
    torch = _torch()
    model.train(train)
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).view(-1, 1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        bs = int(xb.shape[0])
        total += float(loss.detach().cpu()) * bs
        n += bs
    return total / max(n, 1)


def train_one(
    train_loader,
    val_loader,
    *,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    seed: int = 42,
    device: Optional[str] = None,
    ckpt_dir: Optional[Path] = None,
    tag: str = "model",
) -> TrainResult:
    torch = _torch()
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig()
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    stall = 0
    t0 = time.perf_counter()
    for epoch in range(1, int(train_cfg.max_epochs) + 1):
        tr = _run_epoch(model, train_loader, optimizer, device, True)
        va = _run_epoch(model, val_loader, optimizer, device, False)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        if va + 1e-12 < best_val:
            best_val = va
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= int(train_cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = None
    if ckpt_dir is not None:
        ensure_result_dirs()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / "{}_seed{}.pt".format(tag, seed))
        torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "seed": seed}, ckpt_path)
    return TrainResult(
        model=model,
        best_val_loss=float(best_val),
        epochs_run=len(history["val_loss"]),
        history=history,
        seed=int(seed),
        runtime_sec=time.perf_counter() - t0,
        ckpt_path=ckpt_path,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    torch = _torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=False,
    )
