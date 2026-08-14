# APM_ActiveV2 — Active Pressure Metric（主动买卖压力）

**Date:** 2026-07-23  
**Updated:** 2026-07-24  
**Status:** DELIVERY — Smart V2.1 locked  
**Identity:** **distinct from** stub / price-proxy APM；also **distinct from** legacy session-cut knife

---

## Delivery lock (V2.1)

| Field | Value |
|-------|-------|
| Canonical factor | **`APM_ActiveV2_SmartV2_1F`** |
| Profile | `v2_1f` (`research/cfg/apm_active_v2_config.py`) |
| Size filter | lag-20d **q80** of `avg_buy_size` |
| Intensity | `buy/amount − sell/amount` on big minutes |
| ASC hard gate | **OFF** (`asc_min_rank=0`) |
| Smooth | daily EWM **span=5** |
| Brick | `research/cache/bricks/active_pressure_smartv2/` |
| Abandoned | `APM_ActiveV2_SmartV2_1` (q90) — do not rebuild / backtest |

### Headline results (2021-01 → 2024-06, c2c)

| Variant | ALL Sharpe | CSI1000 Sharpe | RankIC ALL | Turn ALL |
|---------|------------|----------------|------------|----------|
| Smart V1 | 4.62 | 2.51 | 0.032 | 1.66 |
| Smart V2 | 4.89 | 2.13 | 0.015 | 0.64 |
| **SmartV2_1F (locked)** | **4.45** | **2.67** | 0.017 | **0.73** |

Verdict: ASC top-50% hard gate hurt CSI1000; turning it off + span=5 recovers CSI1000 Sharpe above V1 with much lower turnover. RankIC stays below V1; delivery prioritizes H-L Sharpe / excess over IC. q90 not pursued (would further shrink coverage).

---

## 0. Identity lock

| Field | Value |
|-------|-------|
| `factor_id` (baseline research) | `APM_ActiveV2` |
| `factor_id` (delivery) | `APM_ActiveV2_SmartV2_1F` |
| Display | APM 2.0 · Active Pressure / Smart V2.1 |
| Brick | `active_pressure` (+ `active_pressure_smartv2` for Smart family) |
| Data | L2 minute `Active_buy/sell_amount` + `Amount` |

**Forbidden aliases**

- Do **not** overwrite / confuse with `apm` stub or `apm_overnight_day_proxy`.
- Do **not** treat legacy session-cut as default `APM_ActiveV2`.
- No PureRev — APM is directional.

---

## 1. Baseline formula (research)

\[
raw\_apm_t = \frac{ActiveBuy_t - ActiveSell_t}{ActiveBuy_t + ActiveSell_t}
\]

\[
daily\_apm = \frac{\sum_t raw\_apm_t \cdot Amount_t}{\sum_t Amount_t}
\]

Baseline alone underperforms; Smart / SmartV2_1F filters are required for alpha.

---

## 2. Variants

| Factor id | Description | Status |
|-----------|-------------|--------|
| `APM_ActiveV2` / Weekly / Raw / Session / Delta | research variants | research |
| `APM_ActiveV2_Smart` | mean×1.2 filter + EWM5 | strong baseline |
| `APM_ActiveV2_SmartV2` | q80 + intensity + ASC≥50% + EWM2 | research |
| **`APM_ActiveV2_SmartV2_1F`** | q80 + intensity + **ASC off** + EWM5 | **LOCKED** |
| `APM_ActiveV2_SmartV2_1` | q90 + ASC off + EWM5 | abandoned |

Note: SmartV2 EWM is on **daily** `apm_raw` (not minute bars).

---

## 3. Pipeline (delivery)

```
minute Active_* (SmartMoney cache)
  → QC
  → smartv2 brick (q80 size filter + buy/sell intensity)
  → limit mask
  → NO ASC hard gate
  → daily EWM span=5
  → MAD → industry fill → Z → ind_cap → Z
```

---

## 4. Files

| Path | Role |
|------|------|
| `research/cfg/apm_active_v2_config.py` | profile lock / abandoned flags |
| `core/l2_features/bricks/active_pressure/` | bricks |
| `core/l2_features/apm_active_v2_builder.py` | `build_smartv2_panel(profile=...)` |
| `factor_formulas_apm_active_v2.py` | CS glue |
| `result/apm_active_v2/APM_ActiveV2_SmartV2_1F/` | locked backtest outputs |

---

## 5. Run

```bash
# factor_config: TRACK=apm_active_v2, CUSTOM_FACTOR_LIST=["APM_ActiveV2_SmartV2_1F"]
OMP_NUM_THREADS=1 /opt/conda/anaconda3/envs/base_93/bin/python run_apm_active_v2_backtest.py
```

Universe: `ALL` / `CSI1000` / `CSI500` / `CSI300` only.
