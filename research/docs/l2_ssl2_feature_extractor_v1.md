# Sprint 4.4 Phase 1 — ClickHouse SSL2 Feature Extractor v1

## Status

Unlocked after multi-DB audit. Native A-share L2 lives in ClickHouse, not
DolphinDB. This phase builds a research-grade feature layer on the real schema.

## Data interface (locked)

```text
COMMON_CONST.DATA_DB_HFDATA
  → ClickHouse
  → database: cmds
  → tables:
       SSE_AL_SSL2_EXG   (上交所 snapshot)
       SZSE_AL_SSL2_EXG  (深交所 snapshot)
```

Tick tables (`SSE/SZSE_AL_TICK_EXG`) are deferred to Phase 2.

### Required columns

| Column | Type | Role |
|--------|------|------|
| `Symbol` | String | bare code (`600000`, `000001`); suffix added in wrapper |
| `ExchTime` | DateTime64 | exchange timestamp |
| `BidPrices` / `AskPrices` | Array(Decimal) | 10 levels, CH 1-indexed `[1]=买一/卖一` |
| `BidVolumes` / `AskVolumes` | Array(Int64) | matching sizes |
| `BidNums` / `AskNums` | Array | order counts (optional diagnostics) |
| `TotalBidVolume` / `TotalAskVolume` | Int64 | aggregate depth |
| `BidVWAP` / `AskVWAP` | Decimal | book VWAP |
| `BidWithdraw*` / `AskWithdraw*` | Int/Decimal | cancel pressure (**SSE only** in current schema) |

Do **not** use fictional expanded columns (`BidPrice0` …).

## Architecture

```text
ClickHouse SSL2 snapshots
        |
        |  server-side snapshot features (array ops)
        v
minute last-snapshot aggregation (toStartOfMinute + argMax)
        |
        v
Python wrapper (symbol normalize, narrow schema)
        |
        v
tradetime | symbol | factorname | value
        |
        v
intraday_evaluation_v2  (Phase 1.5 / Phase 2 wiring)
```

Heavy LOB math stays in ClickHouse. Python does not `groupby/rolling` raw L2.

## Factor set v1

| factorname | Formula summary |
|------------|-----------------|
| `l2_top_book_imbalance` | `(bid0-ask0)/(bid0+ask0)` |
| `l2_depth_imbalance` | `(sum bid - sum ask)/(sum bid + sum ask)` levels 1–10 |
| `l2_weighted_oi` | exponential depth weights `w_i=exp(-λ i)`, λ=0.5 default |
| `l2_microprice_bias` | `(MP - mid)/mid` from top of book |
| `l2_relative_spread` | `(ask0-bid0)/mid` |
| `l2_cancel_pressure` | `(bidWithdrawVol - askWithdrawVol)/(sum)` — **SSE only**; SZSE NULL until schema gains withdraw cols |
| `l2_liquidity_skew` | `(askVWAP-mid)/mid - (mid-bidVWAP)/mid` |
| `l2_liquidity_wall` | `max(max(bidVol), max(askVol)) / (sum bid + sum ask)` |

Aggregation:

- Phase 1 baseline: last snapshot (`argMax`) via `minute_last_feature_sql`.
- Phase 2 discovery: mean / max / std / cancel sum-ratio via
  `minute_agg_feature_sql` (see `l2_alpha_discovery_phase2_v1.md`).

## Symbol policy

| Source table | Output symbol |
|--------------|---------------|
| `SSE_AL_SSL2_EXG` | `{code}.SH` |
| `SZSE_AL_SSL2_EXG` | `{code}.SZ` |

## Non-goals (Phase 1)

- No tick / OrderID factors
- No changes to `intraday_lib.py`, freeze JSON, or DDB minute factor packages
- No OOS bartime/horizon search
- No full ZZ1000 multi-year harvest in this slice (smoke + unit tests first)

## Layout

```text
research/l2_alpha/
  __init__.py
  schema.py                 # locked table/column constants
  formulas.py               # pure-Python reference (tests)
  clickhouse_ssl2.py        # SQL builder + client extract
  run_ssl2_feature_smoke.py # one-day smoke

research/docs/l2_ssl2_feature_extractor_v1.md
tests/test_l2_ssl2_formulas_v1.py
```

## Quick start

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

$PY -m unittest tests.test_l2_ssl2_formulas_v1 -v

OMP_NUM_THREADS=1 PYTHONPATH=. $PY \
  research/l2_alpha/run_ssl2_feature_smoke.py \
  --date 2024-06-03 --limit-symbols 20
```
