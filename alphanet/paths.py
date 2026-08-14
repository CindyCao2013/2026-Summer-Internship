"""Project and result paths for AlphaNet reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"
GUIDE_MD = DOCS_DIR / "00_alphanet_reproduction_guide.md"
GUIDE_V23_MD = DOCS_DIR / "01_alphanet_v2_v3_guide.md"
RESULT_ROOT = REPO_ROOT / "research" / "results" / "alphanet_v1"

CACHE = RESULT_ROOT / "cache"
MODELS = RESULT_ROOT / "models"
FACTORS = RESULT_ROOT / "factors"
IC = RESULT_ROOT / "ic"
DECILES = RESULT_ROOT / "deciles"
ENHANCE = RESULT_ROOT / "enhance"
SHAP = RESULT_ROOT / "shap"
REPORTS = RESULT_ROOT / "reports"
SMOKE = RESULT_ROOT / "smoke"

COMPARE_ROOT = REPO_ROOT / "research" / "results" / "alphanet_vs_explicit"
CANDIDATE_POOL_ROOT = (
    REPO_ROOT / "research" / "results" / "l2_reproduction" / "candidate_pool_v1"
)
L2_RESULT_ROOT = REPO_ROOT / "research" / "results" / "l2_reproduction"


def required_directories() -> Tuple[Path, ...]:
    return (
        RESULT_ROOT,
        CACHE,
        MODELS,
        FACTORS,
        IC,
        DECILES,
        ENHANCE,
        SHAP,
        REPORTS,
        SMOKE,
        COMPARE_ROOT,
    )


def ensure_result_dirs() -> None:
    for path in required_directories():
        path.mkdir(parents=True, exist_ok=True)
