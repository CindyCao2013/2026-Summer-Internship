# L2 Flow Density v1 — Validation Report

**Factor:** `net_active_flow_mktcap_20d`
**Scope:** close Flow Density loop before Temporal Feature Layer / TGD
**Daily portfolio standard:** 10 groups + H-L (project default)
**Intraday heatmap source:** 5-group minute template (existing parquet)

## 0. Architecture note (TGD deferred)

Do **not** implement `tgd.py` yet. TGD is an output of a future
`core/l2_features/return_timing.py` layer (Gu/Gd → residuals → TGD).
Current track closes **Flow Temporal Density** first.

## 1. Horizon profile @ bartime=11:29

Peak Sharpe at `Ret_120` = **4.17** (annu≈25.7%, n=501).

Artifact: `horizon_profile_1129.csv` / `horizon_sharpe_1129.png`

| Horizon | Sharpe | AnnuRet | n |
|---------|--------|---------|---|
| Ret_15 | 2.35 | 3.6% | 503 |
| Ret_30 | 2.01 | 4.2% | 503 |
| Ret_60 | 1.40 | 3.7% | 503 |
| Ret_90 | 1.14 | 3.4% | 503 |
| Ret_120 | 4.17 | 25.7% | 501 |
| Ret_150 | 2.82 | 23.3% | 501 |
| Ret_180 | 2.28 | 20.2% | 501 |
| Ret_EOD | 1.40 | 4.6% | 503 |
| Ret_NDay | 2.35 | 21.7% | 501 |

## 2. Period × bartime stability (Ret_120 / Ret_NDay)

Artifacts: `period_bartime_sharpe.csv`, `period_heatmap_Ret_120.png`, `period_heatmap_Ret_NDay.png`

### Focus bartime `11:29` × Ret_120

| Period | Sharpe | n |
|--------|--------|---|
| 2024H1 | 2.03 | 116 |
| 2024H2 | 4.75 | 125 |
| 2025H1 | 4.72 | 117 |
| 2025H2 | 5.46 | 125 |

Positive-period ratio @ 11:29/Ret_120: **100%** (4/4).

## 3. Neutralization ladder (daily 10-group + H-L)

| Mode | ICIR | Long-book excess Sharpe | HL Sharpe | Net Sharpe@15bp | Daily TO(H-L) |
|------|------|-------------------------|-----------|-----------------|---------------|
| raw | 2.07 | **−0.05** | 1.52 | −0.18 | 0.51 |
| size | 4.41 | 0.76 | 2.95 | 1.60 | 0.47 |
| industry | 2.69 | −0.14 | 1.90 | 0.08 | 0.48 |
| size_industry | 4.85 | **0.50** | 3.38 | 1.85 | 0.46 |

Interpretation: size/industry neutralization creates a positive, but modest,
market-relative long-book result. Raw Group10 does not outperform the exact
ALL-universe equal-weight benchmark.

## 4. Verdict checklist

- [x] Horizon profile (not single-cell heatmap cherry-pick)
- [x] Period split (2024H1–2025H2)
- [x] Neutralization ladder (if daily ran)
- [x] Turnover / Implied AnnuFee / net cost
- [ ] Temporal Feature Layer (`return_timing.py`) — **next after this closes**
- [ ] TGD factor — **after** return_timing primitives

## 5. Next

```
L2 Flow Density v1  →  Temporal Feature Layer (Gu/Gd)  →  TGD  →  APM
```

