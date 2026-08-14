# III-A4.1 — SmartMoney10d Implementation Design + Cache Strategy

**Date:** 2026-07-20  
**Status:** CODING Phase 1 — cache + smoke (no 252d pack / no Registry)  
**Identity:** locked in `docs/milestone_3_0_iiia4_smartmoney_apm_design.md`  
**Parent:** III-A4 feasibility (accepted)

---

## 0. Paper window check (A vs B) — VERIFIED


| Option | Definition                                         | Verdict         |
| ------ | -------------------------------------------------- | --------------- |
| **A**  | Single-day minutes → daily Q                       | ❌ not paper     |
| **B**  | Rolling past **10 trading days** minute pool → Q_t | ✅ **paper 步骤1** |


Source (开源证券聪明钱 / 2.0 writeups; matches Stage-0 DSL):

> 对选定股票，**回溯取其过去 10 个交易日的分钟行情数据**；按 S 降序，取成交量累积占比前 20% → Q=\mathrm{VWAP}*{smart}/\mathrm{VWAP}*{all}。

**β note:** table in original writeup uses S=|R|/\sqrt{V} (β=0.5).  
Our locked identity uses **β=0.25** (Stage-0 `factor_definition.yaml` / SmartMoney10d).  
Do **not** retune to blog-optimal β (e.g. −0.5) under this `factor_id`.

---

## 0b. Scope lock


| In                                     | Out                                      |
| -------------------------------------- | ---------------------------------------- |
| `SmartMoney10d` only                   | APM / APM_SessionResidual                |
| Paper knife + VWAP ratio               | `Active_buy_`* / `Active_sell_*`         |
| Spec → compute → eval → pack (testing) | Registry row before pack soft-bar review |
| Month-chunk minute cache               | Daily-OHLCV approximation                |
| status = `testing`                     | Auto-promote to candidate                |


**Forbidden:** inventing knives; renaming ActiveTradeProxy; claiming ActiveOrderFlowVWAP as SmartMoney.

---

## 1. Factor identity (frozen for code)

```yaml
factor_id: SmartMoney10d
paper: 聪明钱因子模型
identity_class: true_replication_candidate
direction_paper: negative_ic
data_level: minute
lookback_trading_days: 10
top_cumvol_pct: 0.20
knife: "S_t = abs(ret_1m) / volume_1m ** 0.25"
output: "Q = VWAP_smart / VWAP_all"
```

---

## 2. Exact algorithm (no invention)

### 2.1 Minute universe (same session filter as TGD)

```
dfs://QV_Trade_to_MinuteBar / Stock_one_minute
```

Keep only:

```
(09:31–11:30) ∪ (13:01–15:00)
```

**Select columns only:** `Symbol, Date, Bartime, Close, Volume, Amount`  
**Do not select:** any `Active_*` or cancel fields.

A-share filter: symbol starts with `6|0|3` (match TGD).

### 2.2 Per-minute features

Within `(Symbol, Date)` sorted by `Bartime`:


t = \frac{\mathrm{Close}*t}{\mathrm{Close}*{t-1}} - 1


(first bar of day → R_t invalid → drop that minute)


S_t = \frac{|R_t|}{V_t^{0.25}} \quad (V_t > 0)


Minutes with V_t \le 0 or non-finite R_t / S_t → **excluded** from knife ranking and from both VWAPs.

### 2.3 Lookback window (per calendar signal date T)

- Take the last **10 trading days** with minute data ending on T **inclusive**.
- Pool all valid minutes in that window for that symbol.
- Let V_{\mathrm{tot}} = \sum V over pooled minutes.
- If V_{\mathrm{tot}} \le 0 or fewer than a minimum minute count (default **50**) → Q_T = \mathrm{NaN}.

### 2.4 Smart minute selection (cumvol top 20%)

1. Sort pooled minutes by S_t **descending** (ties: higher V first, then earlier `Date`/`Bartime`).
2. Walk in that order; accumulate volume until:

 \sum_{\text{selected}} V \ge 0.20 \cdot V_{\mathrm{tot}} 

1. Those minutes = **smart set** (may slightly overshoot 20% on the last included bar — standard cumvol knife).

### 2.5 VWAP ratio

Prefer **Amount** (matches exchange VWAP):


\mathrm{VWAP}*{X} = \frac{\sum*{t \in X} \mathrm{Amount}*t}{\sum*{t \in X} V_t}


Fallback (only if Amount missing/zero while V>0): \mathrm{Amount}_t := \mathrm{Close}_t \cdot V_t, and log `vwap_source=close_x_volume` in pack notes.


Q_T = \frac{\mathrm{VWAP}*{\mathrm{smart}}}{\mathrm{VWAP}*{\mathrm{all}}}


\mathrm{VWAP}_{\mathrm{all}} uses **all valid pooled minutes**, not only smart set.

### 2.6 Execution timing

Signal on date T uses **full-day** minutes through T close → **must `shift(1)`** before joining next-day returns (same rule as TGD).

Paper direction: **negative IC** → H-L book uses signed signal (`direction = -1` when RankIC < 0).

---

## 3. Why not TGD-style pure DDB daily aggregate?

TGD reduces each day independently (`group by Symbol, Date`).

SmartMoney needs:

```
rank minutes across a rolling 10-day pool → cumvol cutoff → ratio
```

That is a **rolling cross-day minute window**, not a single-day group-by.  
Forcing a wrong DDB daily summary would invent a different factor.

**Chosen architecture: hybrid**

```
DDB (month chunk)     →  slim minute parquet cache
Python (per symbol)   →  rolling 10d Q_T
Panel cache           →  wide/long daily factor
Eval / pack           →  OS path
```

---

## 4. Cache strategy (3-layer under `smart_money/`)

### Layer L0 — source of truth

```
dfs://QV_Trade_to_MinuteBar / Stock_one_minute
```

Read-only; never rewrite.

### Directory layout

```
research/cache/smart_money/
├── minute_raw/
│     minute_YYYYMM.parquet          # Close, Volume, Amount only
├── minute_feature/
│     smart_score_YYYYMM.parquet     # + ret_1m, S_t (β=0.25)
└── factor_panel/
      SmartMoney10d_{start}_{end}.parquet
      SmartMoney10d_long_{start}_{end}.parquet
```

### L1 — `minute_raw/` (DDB month chunk)


| column                | note                                   |
| --------------------- | -------------------------------------- |
| date, symbol, bartime | keys                                   |
| close, volume, amount | **only** price-volume; **no Active_*** |


Session filter = TGD (09:31–11:30 ∪ 13:01–15:00).  
Reuse later for ActiveBuyVWAP / liquidity-efficiency research **as separate ids**.

### L2 — `minute_feature/` (DDB or derived from L1)


| column                | note                                                          |
| --------------------- | ------------------------------------------------------------- |
| date, symbol, bartime | keys                                                          |
| close, volume, amount | pass-through                                                  |
| ret_1m                | `ratios(Close)-1` within (Symbol, Date)                       |
| smart_score           | `abs(ret_1m) / volume ** 0.25` (`S`); null if V≤0 or ret null |


**Why separate:** S_t is the expensive reusable feature; do not re-scan L0 for every factor experiment.

### L3 — `factor_panel/` (Python rolling Option B)

Rolling 10 **trading** days of L2 rows → cumvol top 20% by S → daily Q.

Formula version: `sm10d_v1_beta0p25_optB`.

### Eval artifacts (not formula cache)

```
research/reports/smart_money_v1/smoke/
research/reports/smart_money_v1/execution/   # after 252d gate
research/reports/factors/SmartMoney10d/     # after smoke pass
```

---

## 5. Module layout


| Path                                                       | Role                                                                 |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `factor_cutting/smart_money.py`                            | Spec + `compute_smart_money_q_from_minutes` (replace NotImplemented) |
| `core/l2_features/smart_money_panel_builder.py`            | L1 load + L2 panel build (mirror TGD builder style)                  |
| `factor_specs/SmartMoney10d.yaml` + `_report_content.yaml` | OS factor card inputs                                                |
| `run_milestone_3_0_smart_money10d.py`                      | Eval + pack; **no Registry until validated**                         |
| `docs/milestone_3_0_smart_money10d.md`                     | Results after run                                                    |


### Compute API sketch

```python
def compute_daily_smart_money_q(
    minutes: pd.DataFrame,  # L1 schema
    *,
    lookback_days: int = 10,
    top_cumvol_pct: float = 0.20,
    min_minutes: int = 50,
) -> pd.DataFrame:
    """Return long [date, symbol, Q] for each date that has a full lookback."""
```

Vectorization: groupby-symbol rolling over trading-day groups; avoid Python nested loops over all stocks×minutes where possible. Acceptable: per-symbol NumPy sort for cumvol knife (10d × ~240 bars ≈ 2400 rows).

---

## 6. OS pipeline (after smoke)

Mirror IdealAmplitude / ActiveTradeProxy:

```
1. Build L1 months for [preheat, end]
2. Build L2 SmartMoney10d wide on eval slice (+ warm 10d)
3. CS z-score → RankIC / ICIR / decile / signed mono
4. H-L + cost → turnover / net Sharpe
5. Execution grid (d1 pattern)
6. factor_report_generator_v2 → pack under
   research/reports/factors/SmartMoney10d/
7. status stay testing
8. Registry: ONLY after pack ok=true and human review
```

### Default eval window (first full run)


| Param         | Value                   | Rationale                       |
| ------------- | ----------------------- | ------------------------------- |
| `--eval-days` | 252                     | Match IdealAmplitude / proxy    |
| Signal shift  | 1                       | Same-day minutes → next-day ret |
| Top frac H-L  | 0.10                    | Library convention              |
| Cost          | DEFAULT_ROUND_TRIP_COST | Investability stack             |


### Smoke gate (before full 252d)


| Smoke                        | Pass criteria                                      |
| ---------------------------- | -------------------------------------------------- |
| 1 calendar month L1 pull     | rows > 0; only Close/Volume/Amount cols            |
| 1 symbol × 15 trading days Q | finite Q; Q typically near 1                       |
| 20 trading-day panel         | RankIC sign checkable; no Active_* in cache schema |


---

## 7. Validation checklist

- [ ] IC / RankIC / ICIR  
- [ ] Decile means + **signed** monotonicity (IdealAmplitude lesson)  
- [ ] Direction vs paper (`negative_ic`)  
- [ ] Turnover + net Sharpe  
- [ ] Execution layer grid  
- [ ] Provenance note in pack: true_replication_candidate; no Active_*  
- [ ] Confirm ActiveTradeProxy untouched  

---

## 8. Failure / honesty modes


| Symptom                          | Interpretation                                 | Action                                      |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| Weak IC, flat deciles            | Replication failed or sample issue             | Keep testing; document; do not retune knife |
| Strong Sharpe, low unsigned mono | Check signed mono / tail (IdealAmplitude path) | Diagnose, don’t invent new formula          |
| Q always ≈ 1                     | Cumvol / Amount bug                            | Fix eng; no Registry                        |
| Temptation to add Active_*       | New factor identity                            | Reject under this id                        |


---

## 9. Explicit non-goals

- ❌ APM_SessionResidual  
- ❌ ActiveTradeProxy changes  
- ❌ Composite / portfolio / Similarity Matrix v2  
- ❌ Registry admission in the same commit as first broken pack  
- ❌ Formula search (`V**0.3`, top 15%, etc.)

---

## 10. Implementation order (coding)

1. This design accepted (this doc)
2. `smart_money_panel_builder.py` L1 month cache
3. `compute_daily_smart_money_q` + unit test on synthetic minutes
4. Smoke month
5. `run_milestone_3_0_smart_money10d.py` eval-days=252
6. Pack + milestone results doc
7. Registry only on explicit go after pack review

---

## Related

- Feasibility: `docs/milestone_3_0_iiia4_smartmoney_apm_design.md`  
- Identity YAML: `docs/schemas/iiia4_factor_identity_proposals.yaml`  
- Pattern: `core/l2_features/tgd_panel_builder.py`  
- Stub: `factor_cutting/smart_money.py`

