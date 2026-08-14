# SmartMoneyActiveV2 — 聪明钱因子 2.0（主动大单集中度）

**Date:** 2026-07-23  
**Status:** IMPLEMENTATION — research / testing（未进 Registry）  
**Identity:** **distinct from** `SmartMoney10d`（开源证券 VWAP / S-score 复现）

---

## 0. Identity lock

| Field | Value |
|-------|-------|
| `factor_id` | `SmartMoneyActiveV2` |
| Display | 聪明钱 2.0 · 主动大单集中度 |
| Data | L2 minute `Active_buy/sell_amount` (+ count 可选增强) |
| Knife | 日内 Active_* 金额前 20% 分钟占比 |
| Smooth | EWM `span=10`, `min_periods=5` |
| Output | `smart_long_ewm − smart_short_ewm` |

**Forbidden aliases**

- Do **not** call this `SmartMoney10d` / overwrite Kaiyuan VWAP cache.
- Do **not** claim paper 《聪明钱因子模型》 true replication — that identity stays `SmartMoney10d`.

---

## 1. Economic hypothesis

机构参与痕迹体现在**主动成交中大单金额的集中度**，而非单纯净主动买入：

\[
\text{Smart\_daily} = \frac{\sum_{i \in \text{top 20\%}} Active\_buy\_amount_i}{\sum_t Active\_buy\_amount_t}
\]

多空分别算后做差，剔除散户小单主导的主动买卖噪声。

---

## 2. Pipeline

```
Stock_one_minute
  → session filter + Adjfactor + QC
  → daily smart_long / smart_short
  → mask limit / halt days
  → EWM(span=10) → smart_raw
  → MAD winsorize → industry fill → CS zscore
  → size+industry neutral → final zscore
```

### 2.1 Minute QC

- Continuous auction only: `09:30–11:30` ∪ `13:00–15:00`
- Adjfactor: prices & amount fields × Adjfactor（Volume 不复权）
- Drop bars: adj close < 0.01 or \|ret_1m\| > 20%
- Day invalid if total Volume = 0（停牌）
- Day invalid if limit-touched（EOD High≥Limit or Low≤Stopping；fallback: close-at-limit mask）

### 2.2 Daily knife

- Long: Active_buy_amount ≥ day 80% quantile → concentration ratio
- Short: same on Active_sell_amount
- Optional: top bars also require avg buy/sell size ≥ day median
- Optional: cancel spike downweight（Bid_cancel_vol > μ_20 + 2σ）

### 2.3 Cross-section

- MAD clip: median ± 5×MAD（raw MAD, no 1.4826）
- Fill: industry median → market median；连续停牌 >20d 不填
- Z-score if n≥30
- Neutralize: ln(mktcap) + industry dummies（复用 `panel_neutral_size_ind`）
- Final CS z-score

---

## 3. Code map

| Module | Role |
|--------|------|
| `factor_cutting/smart_money_active_v2.py` | Spec + daily knife + EWM |
| `core/l2_features/smart_money_active_v2_builder.py` | DDB load / month cache / panel |
| `factor_formulas_smart_money_active_v2.py` | Runner glue + CS post-process |
| `core/l2_features/test_smart_money_active_v2.py` | Unit tests（no DDB） |
| `run_smart_money_active_v2_smoke.py` | Month smoke |

Cache root: `research/cache/smart_money_active_v2/`

---

## 4. Relation to peers

| Factor | Mechanism |
|--------|-----------|
| SmartMoney10d | Impact-score minutes → VWAP_smart/VWAP_all（**no Active_***） |
| FlowDensity20 | Net active amt / mktcap cumulative |
| **SmartMoneyActiveV2** | Active large-order **concentration** long−short |

---

## 5. Next gates

1. Unit tests green  
2. Smoke one month coverage  
3. Mentor protocol / RankIC on confirm window  
4. Soft-bar review → Registry `testing` only if ICIR gate passes  
