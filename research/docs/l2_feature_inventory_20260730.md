# Sprint 4.0 — L2 Feature Inventory

Audit date: 2026-07-30  
Representative completeness date: 2024-06-03  
Account scope: current read-only DolphinDB account

## Executive conclusion

The available stock dataset is not a true order-book or tick dataset. It is a
one-minute bar table enriched with aggressive-flow and cancellation labels.
The reachable information ceiling is therefore:

```text
Price path
Volume / amount
Aggressive buy-sell flow
Trade-count behavior
Cancel intent proxy
Intraday seasonality and auctions
Return distribution and price impact
```

True depth imbalance, microprice, spread, queue dynamics, OrderID
reconstruction and true large-trade ratios are blocked under the current data
grant.

This inventory validates the implemented Expansion Batch 1 but changes the
next priority: cancellation behavior and price impact should be developed
before more flow ratios.

## Catalog evidence and scope limitation

`getClusterDFSDatabases()` returns no paths for the read-only account, so an
empty catalog listing cannot prove that a database does not exist globally.
Known L2-style paths were probed explicitly with `existsDatabase`.

Only `dfs://QV_Trade_to_MinuteBar` exists among the probed paths. It exposes:

- `Stock_one_minute`
- `Fund_one_minute`
- `Cbond_one_minute`
- `Future_one_minute`

The first three are readable. `Future_one_minute` is listed but returns
`NoPrivilege`. Candidate databases named QV_SSL2, QV_OrderBook, QV_Tick,
QV_Transaction, QV_Order and QV_Snapshot, plus common non-QV variants, do not
exist under the current cluster namespace.

## Table scale

`Stock_one_minute` has 22 fields, 2,140,331,279 rows, 1,915 trading dates and
6,025 symbols from 2018-09-03 through 2026-07-29.

`Fund_one_minute` has 21 fields and 417,048,588 rows. `Cbond_one_minute` has 21
fields and 153,083,871 rows. Both cover 2018-09-03 through 2025-08-18.

The stock table contains OHLCV, amount, aggressive buy/sell volume, amount and
count, bid/ask cancellation volume and count, timestamps and adjustment
factor. It contains no bid/ask prices, displayed depth levels, spread, queue,
tick sequence or order identifier.

## Stock field quality

On 2024-06-03, the A-share sample contains 1,216,749 rows and 5,091 symbols.
Every symbol has 239 timestamps:

```text
09:25 opening auction
09:30–11:29 continuous session
13:00–14:56 continuous session
15:00 closing auction
```

OHLC is valid on 99.99% of rows. Volume and amount are positive on 97.72%.
Aggressive buy amount/volume is positive on 90.52%; aggressive sell is
positive on 92.52%. Active counts are structurally available on 96.89%.
Bid-cancel volume is positive on 88.79% and ask-cancel volume on 90.73%.

The auction rows have different semantics:

- 09:25: amount/volume coverage is 98.33%; active-flow fields are absent;
  bid/ask cancellation fields are positive on roughly 73%/72%.
- 15:00: amount/volume coverage is 99.78%; active-flow and cancellation fields
  are absent.

Consequently opening auction research can use cancellation imbalance and
amount, while closing auction research is limited to price/volume/amount
behavior.

## Information-dimension verdict

Available without semantic substitution:

- amount and volume OFI, persistence and shocks;
- buy/sell average active ticket size and count imbalance;
- Amihud, signed impact and volume impact;
- cancellation imbalance, persistence and shocks;
- volume-curve deviation and closing-auction amount share;
- realized volatility, downside volatility, skew and jump concentration.

Available only as proxies:

- large active buy ratio based on bar-level average ticket size;
- liquidity resilience inferred from subsequent impact decay;
- institutional activity inferred from size and persistence.

Blocked:

- bid-ask and depth-weighted order-book imbalance;
- microprice and spread;
- queue/order-arrival behavior;
- true tick-size percentile and OrderID-based large-trade ratio.

## Alpha Library v2 roadmap

The 20-factor core consists of the five production factors, the eight Batch 1
discovery candidates, and seven new candidates:

1. `active_sell_trade_size`
2. `active_trade_size_imbalance`
3. `signed_price_impact`
4. `cancel_imbalance`
5. `cancel_persistence`
6. `volume_curve_deviation`
7. `closing_auction_amount_share`

The first implementation batch after inventory should be
`cancel_imbalance`, `signed_price_impact`,
`active_trade_size_imbalance`, and `volume_curve_deviation`. This adds new
behavioral and liquidity dimensions rather than extending the same amount-flow
family.

## Reproducible artifacts

```text
research/run_l2_feature_inventory.py
research/results/l2_feature_inventory/
├── database_inventory.csv
├── table_inventory.csv
├── table_schemas.csv
├── field_inventory.csv
├── session_coverage.csv
├── opportunity_map.csv
├── factor_roadmap_v2.csv
└── summary.json
```

Absence claims in this document mean unavailable to the current account in the
current cluster namespace. They should be re-audited if a new snapshot, tick or
order-feed entitlement is provisioned.
