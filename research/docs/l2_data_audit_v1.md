# Sprint 4.4 Phase 0 — DDB L2 Data Audit

Audited at (UTC): `2026-07-31T02:55:45.850359+00:00`

## Correction (2026-07-31)

This audit only probed **DolphinDB** via `DATA_DB_CONN`. That conclusion remains
true for DDB. A follow-up multi-DB audit of every endpoint in `COMMON_CONST.py`
found native A-share L2 under **ClickHouse** `DATA_DB_HFDATA` (`cmds`):

- snapshot: `SSE_AL_SSL2_EXG`, `SZSE_AL_SSL2_EXG`
- tick: `SSE_AL_TICK_EXG`, `SZSE_AL_TICK_EXG`

See `research/docs/multi_db_data_audit_v1.md`. Phase 0 DDB-only `phase0_stop`
is therefore **superseded for the research program**: proceed to Phase 1 on
ClickHouse SSL2/TICK, while keeping DDB minute bars for the existing library.

## Gate decision (DolphinDB scope only)

- `snapshot_available`: **False** (on DolphinDB)
- `transaction_available`: **False** (on DolphinDB)
- `minute_bar_available`: **True**
- `phase0_stop`: **True for DDB snapshot path; False for ClickHouse HF path**

> Native Level-2 snapshot columns are unavailable under the current DolphinDB account. They **are** available on ClickHouse `DATA_DB_HFDATA`.

## Capability matrix

| Feature | Available | Note |
|---------|-----------|------|
| 十档盘口 Bid/Ask Price/Qty 0-9 | no | requires BidPrice0-9 / OfferPrice0-9 style columns |
| 成交量 Volume / TradeVolume | yes | minute Volume or tick TradeVolume |
| 成交额 Amount / TradeAmount | yes | minute Amount or tick TradeAmount |
| VWAP（可由 amount/volume 计算） | yes | computable from minute Amount/Volume or Close path |
| WAP / microprice（需一档盘口） | no | blocked without top-of-book prices and sizes |
| 盘口 order imbalance（需 depth） | no | blocked without displayed depth |
| 主动买卖 Active_buy/sell_* | yes | minute Active_buy/sell amount/volume/count |
| 撤单 Bid/Ask_cancel_* | yes | minute cancel volume/count proxy; not raw cancel events |
| 逐笔成交量 / true trade size | no | requires tick transaction feed |
| OrderID / BuyOrderID / SellOrderID | no | usually absent on aggregated minute bars |

## Supported vs blocked factor families

Supported under current grant:

- `trade_flow_minute`
- `cancel_intent_minute`
- `price_path_minute`
- `liquidity_impact_minute`

Blocked (native L2 snapshot / tick):

- `order_book`
- `microprice`
- `spread`
- `queue_pressure`
- `tick_transaction`

## Readable tables

- `dfs://QV_Trade_to_MinuteBar/Cbond_one_minute` — 21 columns, rows=153083871, 2018-09-03 00:00:00→2025-08-18 00:00:00
- `dfs://QV_Trade_to_MinuteBar/Fund_one_minute` — 21 columns, rows=417048588, 2018-09-03 00:00:00→2025-08-18 00:00:00
- `dfs://QV_Trade_to_MinuteBar/Stock_one_minute` — 22 columns, rows=2141652471, 2018-09-03 00:00:00→2026-07-30 00:00:00
- `dfs://WIND.ASHAREEODPRICES/data` — 27 columns

## Recommended next step

Do not implement native snapshot factors. Either (a) request Level-2 snapshot / tick entitlement and re-audit, or (b) expand the minute-bar factory on available Active_* and cancel fields (cancel_imbalance, signed_price_impact, downside_RV) while keeping evaluation via intraday_evaluation_v2.

## Artifacts

- `research/results/l2_data_audit.json`
- `research/docs/l2_data_audit_v1.md`
- Runner: `research/run_l2_data_audit_v1.py`

Phase 1+ snapshot factor code was **not** implemented.
