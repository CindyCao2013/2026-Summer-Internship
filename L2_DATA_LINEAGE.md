# L2 Data Lineage — Microstructure v2 (6 event factors)

**Classification:** L2.0 / **trade-flow state enhancer** (not true order-book microstructure alpha mining).  
**Architecture role:** conditional enhancer on D1–D5 (attribution: primarily **D4 behavioral**), not new information dimension.

**Not used:** tick sequence, queue position, order arrival time, book depth levels (Bid1/Ask1).

**DDB source:** `dfs://QV_Trade_to_MinuteBar.Stock_one_minute`  
**Loader:** `l2_data_loaders.py`  
**Daily bricks:** `l2_microstructure.py`  
**v2 factors:** `l2_microstructure_v2.py` → `factor_formulas_l2_v2.py`

---

## DDB data completeness audit (2026-07-08)

Refreshed catalog, schema, coverage and auction-field evidence is maintained in
`research/docs/l2_feature_inventory_20260730.md`. The refresh additionally
found a listed `Future_one_minute` table, but the current account has no read
privilege; this does not change the stock L2 verdict below.

### Accessible in cluster

| Database | Tables | Order book? |
|----------|--------|-------------|
| `dfs://QV_Trade_to_MinuteBar` | `Stock_one_minute`, `Fund_one_minute`, `Cbond_one_minute` | **No** |
| `dfs://QV_SSL2`, `QV_OrderBook`, `QV_Tick`, `SSL2`, `OrderBook`, `Level2`, … | empty / no tablets | **Not available** |

`Stock_one_minute` schema: **22 columns** (not 54). No `Bid1Price`, `Bid1Volume`, `Ask1*`, depth levels, spread, OFI, or tick tables in this cluster.

### Column utilization

| Status | Columns |
|--------|---------|
| **Loaded & used in v2** | `Active_buy/sell_volume`, `Active_buy/sell_amount`, `Bid/Ask_cancel_volume`, `Volume`, `Amount` |
| **In table, not loaded** | `Active_buy/sell_count`, `Bid/Ask_cancel_count` |
| **In table, not used by L2** | `Open/High/Low/Close`, `Bartime`, `Barstart/Barend`, `Adjfactor` |
| **Absent** | Bid/Ask depth, queue, spread, LOB snapshots |

### Field coverage note

On `600*` stocks `2024-06-03`: ~92% of minute rows have non-null `Active_buy_volume`; cancel volumes slightly sparser. Auction bars (e.g. 09:25) often have cancel data but **NaN active flow** — daily aggregates should filter trading minutes if precision matters.

### Research verdict

- **L2 v2 ceiling:** executed flow + cancel-volume intent proxy; information content near limit without new data.
- **L2 v3 (order book engine):** **blocked on data** — no SSL2/LOB in current DDB cluster.
- **L2.2 (same table):** optional use of `*_count` fields, minute OHLCV for impact/spread proxies — marginal uplift, not true LOB.
- **`cn_liquidity_consumption`:** label **research_only** — `volume / (active_buy + active_sell)` is flow intensity, not depth consumption.

**Recommended path:** freeze v2 enhancers → complete stability test → pivot to **Fundamental Quality block (D6/D7)** unless/until order-book data is provisioned.

---

## DolphinDB `Stock_one_minute` — full column list (22)

| Column | Used in pipeline |
|--------|------------------|
| `Symbol` | ✓ group key |
| `Date` | ✓ group key |
| `Bartime`, `Barstart`, `Barend` | ✗ |
| `Open`, `High`, `Low`, `Close` | ✗ (EOD close from WIND separately) |
| `Volume` | ✓ → `volume` |
| `Amount` | ✓ → `amount` |
| `Active_buy_volume` | ✓ → `active_buy_vol` |
| `Active_sell_volume` | ✓ → `active_sell_vol` |
| `Active_buy_amount` | ✓ → `active_buy_amt` |
| `Active_sell_amount` | ✓ → `active_sell_amt` |
| `Active_buy_count` | ✗ not loaded |
| `Active_sell_count` | ✗ not loaded |
| `Bid_cancel_volume` | ✓ → `bid_cancel_vol` |
| `Bid_cancel_count` | ✗ not loaded |
| `Ask_cancel_volume` | ✓ → `ask_cancel_vol` |
| `Ask_cancel_count` | ✗ not loaded |
| `Adjfactor` | ✗ |

**Not present in table:** Bid1–BidN, Ask1–AskN, bid/ask depth, queue, tick-level aggressor flags beyond active buy/sell labels.

---

## Loader layer (`l2_data_loaders.py`)

### Query 1 — daily aggregates

Function: `load_l2_daily_long()` → `_ddb_daily_aggregate_script()` (L40–57)

```sql
select Symbol, Date,
    sum(Active_buy_volume)  as active_buy_vol,
    sum(Active_sell_volume)  as active_sell_vol,
    sum(Active_buy_amount)   as active_buy_amt,
    sum(Active_sell_amount)  as active_sell_amt,
    sum(Bid_cancel_volume)   as bid_cancel_vol,
    sum(Ask_cancel_volume)   as ask_cancel_vol,
    sum(Volume)              as volume,
    sum(Amount)              as amount
from loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
group by Symbol, Date
```

Cache: `research/cache/l2_daily/l2_daily_{start}_{end}.parquet`

### Query 2 — imbalance duration (minute-level)

Function: `load_imbalance_duration_daily()` → `_ddb_imbalance_duration_script()` (L90–107)

Per-minute VOI on server:

```
m_voi = (Active_buy_volume - Active_sell_volume) / (Active_buy_volume + Active_sell_volume)
```

Daily metric:

```
imbalance_duration = count(m_voi > 0.10) / count(*)
```

Cache: `research/cache/l2_daily/l2_imbalance_duration_{start}_{end}.parquet`

### Wide pivot

Function: `pivot_l2_metric()` (L132–138) → Date × Symbol panels, A-share filter (`0/3/6` prefix).

### Cache object

`L2DailyWideCache` (L163–176): 8 daily wide panels + optional `imbalance_duration` + optional `close`.

Entry: `build_l2_daily_cache()` (L179–199).

---

## Daily brick layer (`l2_microstructure.py`)

| Brick | Formula | Cache fields |
|-------|---------|----------------|
| `daily_voi()` L25–28 | `(active_buy_vol - active_sell_vol) / (active_buy_vol + active_sell_vol)` | active buy/sell **volume** |
| `daily_mpb()` L36–39 | `(active_buy_amt - active_sell_amt) / (active_buy_amt + active_sell_amt)` | active buy/sell **amount** |
| `daily_oir()` L31–34 | `(bid_cancel_vol - ask_cancel_vol) / (bid_cancel_vol + ask_cancel_vol)` | cancel **volume** only |

Note: `daily_oir` is cancel-volume imbalance (code name OIR); **not** order-book depth OIR.

---

## v2 factor layer (`l2_microstructure_v2.py`)

| Factor | Lines | Pipeline |
|--------|-------|----------|
| `cn_voi_shock` | L60, L64 | `daily_voi` → `time_series_zscore(voi, window=60, min_periods=20)` |
| `cn_mpb_shock` | L61, L65 | `daily_mpb` → `time_series_zscore(mpb, window=60, min_periods=20)` |
| `cn_cancel_shock` | L62, L69 | `daily_cancel_imbalance()` (= `daily_oir`) → `time_series_zscore(cancel, window=60)` |
| `cn_flow_persistence` | L66 | `flow_persistence(daily_voi, lag=5, window=20)` — rolling corr(VOI_t, mean(VOI_{t-5:t-1})) |
| `cn_imbalance_duration` | L67 | `load_imbalance_duration_daily` OR fallback `(daily_voi > 0.1).astype(float)` |
| `cn_liquidity_consumption` | L68 | `volume / (active_buy_vol + active_sell_vol)` |

`time_series_zscore()` L29–32: per-symbol rolling z-score over trailing 60 days (NOT cross-sectional z).

---

## Registry (`factor_formulas_l2_v2.py`)

- L23–28: `build_l2_v2_factor(name, cache)` → `build_l2_v2_factor_panels(cache)[name]`
- L13–20: factor list

---

## Per-factor lineage (raw → factor)

### cn_voi_shock

```
Stock_one_minute.Active_buy_volume  ─┐
Stock_one_minute.Active_sell_volume ─┼─ sum by Date,Symbol → daily_voi
                                     │
daily_voi = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                                     │
cn_voi_shock = ts_zscore(daily_voi, 60d)
```

### cn_mpb_shock

```
Stock_one_minute.Active_buy_amount  ─┐
Stock_one_minute.Active_sell_amount ─┼─ sum → daily_mpb
daily_mpb = (buy_amt - sell_amt) / (buy_amt + sell_amt)
cn_mpb_shock = ts_zscore(daily_mpb, 60d)
```

### cn_cancel_shock

```
Stock_one_minute.Bid_cancel_volume ─┐
Stock_one_minute.Ask_cancel_volume ─┼─ sum → daily_oir (cancel imbalance)
daily_oir = (bid_cancel - ask_cancel) / (bid_cancel + ask_cancel)
cn_cancel_shock = ts_zscore(daily_oir, 60d)
```

**Does NOT use:** `Bid_cancel_count`, `Ask_cancel_count`.

### cn_flow_persistence

```
Same raw as cn_voi_shock (Active_buy/sell_volume)
→ daily_voi
→ rolling.corr(VOI_t, rolling_mean(VOI,5).shift(1), window=20)
```

### cn_imbalance_duration

```
Stock_one_minute.Active_buy_volume  ─ per minute ─┐
Stock_one_minute.Active_sell_volume ─ per minute ─┼─ minute VOI
→ fraction of minutes with VOI > 0.10 per day
Fallback (if cache miss): (daily_voi > 0.1) at daily frequency only
```

### cn_liquidity_consumption

```
Stock_one_minute.Volume ───────────── sum → volume
Stock_one_minute.Active_buy_volume  ─┐
Stock_one_minute.Active_sell_volume ─┼─ sum → depth_proxy
LCR = volume / (active_buy_vol + active_sell_vol)
```

---

## Validation / integration

| Stage | File |
|-------|------|
| Validation | `run_l2_validation.py` → `load_context()` → `build_l2_v2_factor()` |
| Backtest track | `Factor_Test_Process.py` track `l2_microstructure_v2` |
| Conditioning | `l2_conditioning_layer.py` — retained: voi_shock, mpb_shock, cancel_shock |
| Attribution | `run_alpha_attribution.py` — enhancer × D1–D5 grid |

---

## v1 vs v2 (reference)

| | v1 (`l2_microstructure.py` L55–61) | v2 (`l2_microstructure_v2.py`) |
|---|-----------------------------------|--------------------------------|
| Signal | 20d **rolling mean** of daily brick | Daily brick + **60d ts z-score** or event metric |
| Status | CLOSED (`research/results/l2_v1_closed.md`) | Conditioning layer only |

Same raw DDB columns for v1 and v2 daily bricks.
