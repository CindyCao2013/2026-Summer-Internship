# Factor Cutting Validation Summary

**Date:** 2026-07-13  
**Audience:** research review / go-no-go for next stage

---

## Status correction (vs checklist that still said “cache → 2024-08”)

| Item | Actual status |
|------|----------------|
| Oracle EOD cache | ✅ **complete** through `2025-12` (~195 monthly parquet files) |
| `run_factor_cutting_v2.py --preset paper` | ✅ **finished** (exit 0) — period `2010-01-04 → 2025-12-31`, 3886 ret days |
| Three core plots | ✅ **exist + content verified** (smoke 2023–2025 viz pack) |
| DDB replication 2018–2025 | ✅ validated |
| `run_factor_cutting_v21.py --preset paper` | ⏸ optional polish (neut ladder / knife family on full panel); v2 paper already wrote headline metrics |

**Not** “data prep almost complete.” Paper OHLCV + v2 metrics are done. Remaining caveat is **ATS knife coverage**, not missing 2024–2025 months.

---

## Caveat (must keep visible)

`ideal_reversal` uses knife `ats_trade_count` from L2 daily counts → available roughly **2018-11+**.

| Factor | Effective IC days (paper run) | Early years |
|--------|-------------------------------|-------------|
| ideal_reversal | **1703** | 2010–2017 yearly IC empty |
| ideal_amplitude | **3885** | full 2010–2025 yearly IC filled |

So: **amplitude = true paper-window replication**; **reversal ATS = DDB-era replication on Oracle panel** (same knife constraint as DDB).

For 2010–2017 reversal, see **§5 Proxy knife** — stitched `ideal_reversal_proxy_amount` fills the window with honest labeling (not paper ATS).

---

## 1. DDB / ATS-era replication ✅

| Factor | Daily RankIC | Monthly RankIC | Paper ref |
|--------|-------------:|---------------:|----------:|
| ideal_reversal | −3.38% | **−6.13%** | ≈ −6.06% |
| ideal_amplitude | −4.81% | **−8.00%** | ≈ −7.00% |

Cutting logic + definitions match paper magnitude on monthly IC.

---

## 2. Paper Oracle run (headline) ✅ with caveat above

| Factor | Daily RankIC | Monthly RankIC | n_days |
|--------|-------------:|---------------:|-------:|
| ideal_reversal (ATS) | −3.11% | **−5.45%** | 1703 |
| ideal_amplitude | −3.78% | **−7.60%** | 3885 |

Mechanism (reversal, ATS window): M_high −4.30% / M_low +0.20% / separation −4.51% / purity 0.79.

Amplitude yearly RankIC **negative every year 2010–2025** (cross-period ok on OHLC knife).

---

## 3. Three core plots — content confirmed ✅

Paths (smoke viz; same layout for future paper re-viz):

```
ideal_reversal/
├── ic_analysis/rank_ic_bar.png
├── mechanism/high_low_leg_ic.png
└── portfolio/long_short_curve.png
```

| Plot | What it shows | Verdict |
|------|----------------|---------|
| `rank_ic_bar.png` | Ret20 −4.37% · **M_high −4.76%** · **M_low +0.12%** · spread −3.77% | Cutting isolates alpha in high leg |
| `high_low_leg_ic.png` | M_high cumulative IC ↓ ~−35; M_low flat ~0 | Information concentration |
| `long_short_curve.png` | H-L Sharpe **1.60**, ann **21.5%**, dir=−1 | Tradable after sign |

→ Factor Cutting research loop on DDB/smoke sample is **closed**.

---

## 4. Neutralization ✅ (smoke / v2.1)

| Mode | RankIC | Retention |
|------|-------:|----------:|
| raw | −3.77% | 1.00 |
| size | −3.74% | 0.99 |
| size+industry | −3.31% | **0.88** |

Not a pure size artifact. Formal write-up: `research/reports/factor_cutting_v21/factor_decay_report.md`.

---

## 5. Proxy knife — pre-2018 reversal backfill 🟡 (done, threshold miss)

Runner: `run_factor_cutting_proxy_reversal.py --preset paper --keep-cache`  
Artifacts: `research/reports/factor_cutting_v1/ideal_reversal_proxy/`

**Match window (2018-11+ overlap):** IC-series corr of W-cut factors vs true ATS:

| Knife | Corr vs ATS | Overlap RankIC |
|-------|------------:|---------------:|
| `ats_trade_count` (bench) | 1.000 | −3.38% |
| **`amount` (best)** | **0.757** | −3.68% |
| `volume` | 0.731 | −3.71% |
| `turnover_proxy` | 0.706 | −3.59% |
| `ats_volume` (amount/volume) | 0.461 | −2.44% |

Acceptance was corr ≥ **0.80** → **FAIL** (best 0.757). Surprising: `ats_volume`/`avg_price` is **worst**, not best — participation scale (`amount`/`volume`) tracks ATS IC path more than avg price.

**Stitched full history** (`ideal_reversal_proxy_amount`): ATS where trade_count coverage≥200, else `amount`.

| Metric | Value |
|--------|------:|
| Daily RankIC | −2.81% |
| Monthly RankIC | **−5.63%** |
| n_days | **3866** (vs ATS-only 1703) |
| ATS days / proxy-fill | 1723 / 2163 |
| Yearly 2010–2025 | **all negative** |

Honesty label: **not** paper ATS on full history.

---

## 5b. Knife-family incremental IC ✅ (2018–2025 DDB)

Runner: `run_factor_cutting_knife_family.py --preset ddb`  
Artifacts: `research/reports/factor_cutting_v1/knife_family/`

| Test | Result |
|------|--------|
| corr(amount, volume) | 0.92 — same participation family |
| corr(ATS, amount/volume) | **0.37 / 0.35** — distinct trader_structure |
| resid IC volume\|amount | t=−16 but **no dual uplift** vs singles |
| resid IC ATS\|amount | t=−14.9 · **independent** |
| best single ICIR | ATS −6.48 |
| best dual ICIR | **amount+ATS residual_add −6.88** (RankIC −4.08%) |

→ Multi-knife value is **cross-family** (participation + ATS), not amount+volume.

---

## 5c. Event knife — limit filter ✅ (2018–2025 DDB)

Runner: `run_factor_cutting_event_knife.py --preset ddb`  
Artifacts: `research/reports/factor_cutting_v1/event_knife/`  
Source: `get_EOD_Not_Limit` (drop ≈ **2.32%** of object cells).

| Mode | RankIC | ICIR | Retention vs raw |
|------|-------:|-----:|-----------------:|
| raw | −3.38% | −6.48 | 1.00 |
| **filter_signal** | **−3.82%** | **−7.20** | **1.13** |
| filter_cut | −2.52% | −5.12 | 0.75 |

Dual `amount+ATS`: raw −4.08% → filter_signal **−4.49%** (ICIR −7.42).

**Verdict:** masking limit names on the *finished signal* helps (paper-table-2 style). Excluding limit days *inside* W-cut hurts — those extremes still inform the knife partition. Standard recipe: **cut on full sample, evaluate / trade with not-limit mask**.

---

## 5d. Operability checklist ✅ (2018–2025 DDB) — NEW BINDING GATE

Runner: `run_factor_cutting_operability.py --preset ddb`  
Artifacts: `research/reports/factor_cutting_v1/operability/`

| Factor | Mode | Long excess | Long Sharpe | HL Sharpe | Daily TO long | Month-end long TO |
|--------|------|------------:|------------:|----------:|--------------:|------------------:|
| amount+ATS | raw | **3.8%** | 0.69 | 1.40 | 0.50 | **89%** |
| amount+ATS | filter_signal | **8.8%** | 1.62 | 3.09 | 0.52 | 89% |
| ideal_amplitude | raw | **9.3%** | 1.23 | 3.28 | 0.50 | 86% |
| ideal_amplitude | filter_signal | **11.6%** | 1.51 | 4.00 | 0.52 | 86% |
| ATS single | raw | 1.4% | 0.29 | 1.19 | 0.52 | 88% |

**Binding conclusions:**

1. ICIR ≠ operability: dual/ATS **raw long excess is weak** — alpha mostly on short leg.  
2. `filter_signal` is **mandatory** for long-biased use (dual long excess +5pp).  
3. Turnover is **too high** for production (month-end ~86–89% one-way vs paper-style ~40%).  
4. Size median pctile ≈ 0.43 — small tilt, not pure microcap.  
5. Best long-biased candidate today: **`ideal_amplitude` + not-limit mask**.

**Daily research loop (information):** closed.  
**Production / long-only loop:** 🟡 blocked on turnover engineering + fee netting.

---

## 6. Corrected module board

| Module | Status |
|--------|--------|
| Object / Knife / Output | ✅ |
| DDB replication | ✅ |
| Mechanism (leg IC) | ✅ |
| Knife evaluator | ✅ |
| Portfolio / IC / leg **plots content** | ✅ |
| Paper Oracle OHLCV + amplitude full history | ✅ |
| Paper ATS reversal 2010–2017 | ❌ blocked on trade_count |
| Proxy stitch `ideal_reversal_proxy_amount` | 🟡 corr 0.757 (below 0.8); full-history report ok |
| Neut ladder report | ✅ smoke; paper polish optional |
| Cross-period (amplitude) | ✅ yearly all negative |
| Cross-period (proxy reversal) | ✅ yearly all negative 2010–2025 |
| Knife-family incremental IC | ✅ 2018–2025; amount+ATS dual wins |
| Event knife (limit filter) | ✅ filter_signal helps; **mandatory for long-only** |
| Operability (long / TO / size) | 🟡 long OK with filter; **TO too high for prod** |
| Minute layer | ⏸ after turnover engineering |

---

## 7. Go / no-go

**Publishable as research platform:** daily Factor Cutting information loop closed (IC/mech/viz/neut/proxy/family/limit).

**Not production / long-only ready:** operability shows (1) raw dual/ATS long excess weak without limit mask, (2) month-end long TO ~86–89% — fee will dominate until turnover is engineered.

**Do not claim** “Kaiyuan ideal_reversal 2010–2025 ATS replica.”

**Next:** turnover engineering (month-end / band + fee-net) before minute layer.
---

## Artifact index

- Paper / DDB metrics: `research/reports/factor_cutting_v1/verdict.json`
- Viz pack + `summary.md`: `research/reports/factor_cutting_v1/ideal_reversal/`
- Proxy match + stitch: `research/reports/factor_cutting_v1/ideal_reversal_proxy/`
- Knife family incremental: `research/reports/factor_cutting_v1/knife_family/`
- Event knife (limit): `research/reports/factor_cutting_v1/event_knife/`
- **Operability:** `research/reports/factor_cutting_v1/operability/`
- Neut: `research/reports/factor_cutting_v21/`
- Notes: `research/factor_cutting/NOTES_v21.md`
