"""SHAP (or gradient / permutation fallback) on flattened extract+pool features."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphanet.features import all_feature_names
from alphanet.model import AlphaNetV1
from alphanet.paths import SHAP, ensure_result_dirs


def _torch():
    import torch

    return torch


def _predict_from_features(model, feat):
    torch = _torch()
    if hasattr(model, "hidden") and hasattr(model, "dropout"):
        return model.out(model.dropout(torch.relu(model.hidden(feat))))
    return model.out(feat)


def gradient_importance(
    model: AlphaNetV1,
    images: np.ndarray,
    feature_names: Sequence[str],
    device: str = "cpu",
    max_samples: int = 256,
) -> pd.DataFrame:
    torch = _torch()
    model.eval()
    x = torch.from_numpy(images[:max_samples]).to(device)
    x.requires_grad_(True)
    pred = model(x).sum()
    pred.backward()
    grad = x.grad.detach().abs().mean(dim=0).cpu().numpy()  # (F, T)
    # map back to flattened extract features via a second pass on feature tensor
    model.zero_grad(set_to_none=True)
    x2 = torch.from_numpy(images[:max_samples]).to(device)
    x2.requires_grad_(True)
    feat = model.feature_tensor(x2)
    loss = feat.mean()
    # per-feature mean |activation| as a cheap stand-in when shap is absent
    mean_abs = feat.detach().abs().mean(dim=0).cpu().numpy()
    names = list(feature_names)[: mean_abs.shape[0]]
    if len(names) < mean_abs.shape[0]:
        names = names + ["f{}".format(i) for i in range(len(names), mean_abs.shape[0])]
    df = pd.DataFrame({"feature": names, "mean_abs_activation": mean_abs})
    df["input_grad_mean"] = float(grad.mean())
    return df.sort_values("mean_abs_activation", ascending=False)


def permutation_importance(
    model: AlphaNetV1,
    images: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    device: str = "cpu",
    max_samples: int = 256,
    top_k: int = 40,
) -> pd.DataFrame:
    torch = _torch()
    model.eval()
    x = torch.from_numpy(images[:max_samples]).to(device)
    y = torch.from_numpy(labels[:max_samples].astype(np.float32)).view(-1, 1).to(device)
    with torch.no_grad():
        feat = model.feature_tensor(x)
        base = torch.nn.functional.mse_loss(_predict_from_features(model, feat), y)
        base_v = float(base)
        mean_abs = feat.abs().mean(dim=0)
        top = torch.topk(mean_abs, k=min(int(top_k), feat.shape[1])).indices.cpu().numpy()
    rows = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        feat_np = feat.cpu().numpy()
        for j in top:
            perm = feat_np.copy()
            rng.shuffle(perm[:, j])
            ft = torch.from_numpy(perm).to(device)
            pred = _predict_from_features(model, ft)
            loss = float(torch.nn.functional.mse_loss(pred, y))
            name = feature_names[int(j)] if int(j) < len(feature_names) else "f{}".format(j)
            rows.append({"feature": name, "mse_increase": loss - base_v, "rank": int(j)})
    return pd.DataFrame(rows).sort_values("mse_increase", ascending=False)


def shap_values(
    model: AlphaNetV1,
    images: np.ndarray,
    feature_names: Sequence[str],
    device: str = "cpu",
    max_samples: int = 128,
) -> Optional[pd.DataFrame]:
    try:
        import shap
        import torch
    except ImportError:
        return None
    model.eval()
    x = torch.from_numpy(images[:max_samples]).to(device)

    class Head(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, z):
            pred, _ = self.m.forward(z, return_features=True)
            return pred

    # KernelExplainer on flattened features is too slow; use gradient SHAP on images
    # and report mean |shap| per input channel instead.
    try:
        explainer = shap.GradientExplainer(Head(model), x[: min(32, len(x))])
        sv = explainer.shap_values(x)
        arr = np.asarray(sv)
        if arr.ndim == 4:
            arr = arr[0]
        mean_abs = np.abs(arr).mean(axis=0)  # (F, T)
        rows = []
        for i, name in enumerate(feature_names[: mean_abs.shape[0]] if mean_abs.ndim == 1 else []):
            rows.append({"feature": name, "mean_abs_shap": float(mean_abs[i])})
        if mean_abs.ndim == 2:
            from alphanet.config import FEATURE_NAMES

            for i, fname in enumerate(FEATURE_NAMES[: mean_abs.shape[0]]):
                rows.append(
                    {
                        "feature": fname,
                        "mean_abs_shap": float(mean_abs[i].mean()),
                    }
                )
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        return None


def explain_model(
    model: AlphaNetV1,
    images: np.ndarray,
    labels: Optional[np.ndarray],
    *,
    variant: str = "v1",
    device: str = "cpu",
    top_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    ensure_result_dirs()
    names = all_feature_names(model.cfg)
    grad = gradient_importance(model, images, names, device=device)
    perm = (
        permutation_importance(model, images, labels, names, device=device)
        if labels is not None
        else pd.DataFrame()
    )
    shap_df = shap_values(model, images, names, device=device)
    out = {"activation": grad.head(top_n), "permutation": perm.head(top_n)}
    if shap_df is not None:
        out["shap"] = shap_df.head(top_n)
        shap_df.head(top_n).to_csv(SHAP / "{}_shap_top{}.csv".format(variant, top_n), index=False)
    grad.head(top_n).to_csv(SHAP / "{}_activation_top{}.csv".format(variant, top_n), index=False)
    if not perm.empty:
        perm.head(top_n).to_csv(SHAP / "{}_permutation_top{}.csv".format(variant, top_n), index=False)
    return out
