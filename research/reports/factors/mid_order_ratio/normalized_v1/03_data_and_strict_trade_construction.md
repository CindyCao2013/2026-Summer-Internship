# 03 — Data and Strict Trade Construction

## Source records

The source rows are exchange Tick execution records:

- SSE: `cmds.SSE_AL_TICK_EXG`
- SZSE: `cmds.SZSE_AL_TICK_EXG`

They are not reconstructed parent orders.  Order-number fields on SZSE are
used only to identify valid execution records.

## Frozen strict definition

For every source row and every trade date:

- session: `09:30:00 <= ExchTime < 15:00:01`;
- positive `Price` and `Volume`;
- SSE: `Type='T'`;
- SSE amount: `ifNull(Amount, Price*Volume)`;
- SZSE: `Type='011' AND BidOrderNo>0 AND AskOrderNo>0`;
- SZSE amount: `Price*Volume`.

Both factor numerators and denominators are sums of these trade amounts, not
trade volumes or trade counts.

## Server-side construction

Raw Tick rows never cross the network into Python.  ClickHouse performs:

1. strict filtering and amount construction;
2. `Symbol × TradeDate` aggregation;
3. daily amount quantiles and scale primitives;
4. dynamic-threshold `sumIf` aggregation after joining a small external
   stock-day scale table.

Python receives only daily narrow tables.  Every cache chunk and combined
panel enforces uniqueness on `Symbol + TradeDate`.

## Cache lineage

The normalized cache is separate from the authoritative fixed-bucket cache.
Each cache metadata file records:

- SQL text and SHA256;
- strict filters and amount expressions;
- sample dates and symbol scope;
- row count and duplicate-key audit;
- content SHA256;
- ClickHouse client/server versions;
- creation timestamp and chunk lineage.

The existing strict fixed-bucket cache is never overwritten.

