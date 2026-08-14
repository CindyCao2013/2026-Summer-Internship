# 8. Final Factor Library Classification

Statuses are **preserved** from `factor_library.csv` / Pack `summary.yaml`. This section only organizes them for reviewers.

## 8.1 Production / library tiers

### Validated

| Factor | Why |
|--------|-----|
| **TGD20** | Formula frozen; SI ICIR 11.29; execution Net 2.32; Pack v1 complete |

### Candidate (investable research assets, not yet “validated”)

| Factor | Why |
|--------|-----|
| **D1_LiquidityQuality60d** | Strong raw IC; Pack v1; SI pending |
| **FlowDensity20** | SI ICIR 4.85; strong Net; mechanism caution |

### Testing candidate

| Factor | Why |
|--------|-----|
| **APM_SessionResidual** | Adapted paper; SI ICIR 6.55; frozen buffer recipe Net 1.50; not Registry |

### Testing (not promoted)

| Factor | Why held back |
|--------|----------------|
| **IdealReversal** | Mono 0.44 |
| **IdealAmplitude** | Mono 0.11 |

### Research (parked)

| Factor | Why |
|--------|-----|
| **SmartMoney10d** | IC strong; execution Net ~0.3 |

### Explicitly excluded from headline library

| Factor | Why |
|--------|-----|
| ActiveTradeProxy | Proxy ≠ paper APM |
| SUE_ConsensusEPS | design_only; scout pending |

## 8.2 Count vs goal

| | Now (Report v1) | Goal (stated) |
|--|-----------------|---------------|
| Independent Pack-ready alpha sources with identity | ~6–7 usable (1 validated + 2 candidate + 1 testing_cand + 2 testing) | **10–20** |
| Validated | 1 | grow carefully |

Report v1 is the **packaging milestone**. Expanding count (SUE → Revision → …) comes after this deliverable, without inventing Packs.

## 8.3 Recommended next actions (priority)

1. Keep this report as the external-facing memo; link Packs as provenance.
2. Run lightweight combination smoke (Section 6 TODO) for TGD+D1+Flow only.
3. Resume Factor Discovery for next paper factor (SUE) to grow library count — **not** more audit scaffolding.
4. Defer Similarity / Composite / Portfolio OS until library ≥ ~10 independent sources.
