# IdealAmplitude_ActiveV2 — 理想振幅因子 2.0（振幅 / 主动净额波动）

**Date:** 2026-07-24  
**Status:** IMPLEMENTATION — research / testing（未进 Registry）  
**Identity:** **distinct from** `IdealAmplitude`（原版股价×振幅切割）

---

## 0. Identity lock

| Field | Value |
|-------|-------|
| `factor_id` | `IdealAmplitude_ActiveV2` |
| Display | 理想振幅 2.0 · 振幅质量 |
| Data | L2 minute OHLCV + `Active_buy/sell_amount` |
| Knife | `realized_amp / active_net_vol` |
| Smooth | EWM `span=5`, `min_periods=3` |
| Output | `amp_smooth` wide panel |
| Direction (paper) | `negative_ic`（实证待确认） |

**Forbidden aliases**

- Do **not** call this `IdealAmplitude` / overwrite Kaiyuan price×amplitude cache.
- Do **not** claim paper 《振幅因子的隐藏结构》 true replication — that identity stays `IdealAmplitude`.

---

## 1. Economic hypothesis

振幅大小 alone 无法区分机构压盘与噪声博弈：

\[
\text{amp\_raw} = \frac{(H-L)/O}{\mathrm{std}_t\!\big((ActiveBuy_t - ActiveSell_t)/Amount_t\big) + \epsilon}
\]

- 高比值：振幅由较小的主动净额波动驱动 → 可能压盘吸筹
- 低比值：振幅伴随剧烈买卖冲突 → 噪声博弈

---

## 2. Pipeline

```
Stock_one_minute
  → session filter + Adjfactor QC (reuse apply_minute_qc)
  → daily realized_amp + active_net_vol → amp_raw
  → mask limit days
  → optional amplitude gate (top 20% amp)
  → EWM(span=5) → amp_smooth
  → MAD winsorize → industry fill → CS zscore
  → size+industry neutral → final zscore
```

### 2.1 Minute / daily

- Continuous auction only（via `load_minute_active_raw`）
- Adjfactor on prices & amount fields
- `MIN_MINUTES_PER_DAY = 30`
- `active_net_vol = 0` → `amp_raw = NaN`

### 2.2 Cross-section

- MAD clip → industry median fill → zscore
- Neutralize: `panel_neutral_size_ind(nt_type="ind_cap")`
- Universe: ALL / CSI1000 / CSI500 / CSI300
- Rebalance hint: weekly（信号衰减快）

---

## 3. Code map

| Module | Role |
|--------|------|
| `factor_cutting/ideal_amplitude_active_v2.py` | Spec + daily knife + EWM |
| `core/l2_features/ideal_amplitude_active_v2_builder.py` | DDB load / month cache / panel |
| `factor_formulas_ideal_amplitude_active_v2.py` | Runner glue + CS post-process |
| `core/l2_features/test_ideal_amplitude_active_v2.py` | Unit tests（no DDB） |
| `run_ideal_amplitude_active_v2_smoke.py` | Month smoke |
| `run_ideal_amplitude_active_v2_backtest.py` | Prefetch + Factor_Test_Process |

Cache root: `research/cache/ideal_amplitude_active_v2/`  
Minute raw reused from: `research/cache/smart_money_active_v2/minute_raw/`

---

## 4. Relation to peers

| Factor | Mechanism |
|--------|-----------|
| IdealAmplitude | Price-state × amplitude（**no Active_***） |
| APM_ActiveV2 | Directional active pressure |
| **IdealAmplitude_ActiveV2** | Amplitude **quality** via net-flow volatility |

---

## 5. Next gates

1. Unit tests green  
2. Smoke 2024-06 / 200 symbols — inspect `amp_raw` distribution  
3. Mentor protocol / RankIC（日频 + 周频）on confirm window  
4. Soft-bar review → Registry `testing` only if ICIR gate passes  

```bash
# unit
pytest core/l2_features/test_ideal_amplitude_active_v2.py -q

# smoke
OMP_NUM_THREADS=1 python run_ideal_amplitude_active_v2_smoke.py --max-symbols 200

# backtest (factor_config: TRACK=ideal_amplitude_active_v2)
OMP_NUM_THREADS=1 python run_ideal_amplitude_active_v2_backtest.py
```
