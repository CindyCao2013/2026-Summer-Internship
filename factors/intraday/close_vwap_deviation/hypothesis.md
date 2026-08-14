# close_vwap_deviation

## Hypothesis

Intraday prices that trade **above** session cumulative VWAP reflect short-term buying pressure;
prices **below** VWAP reflect selling pressure. The normalized deviation

\[
\frac{P_t - \mathrm{VWAP}_t}{\mathrm{VWAP}_t}
\]

where \(\mathrm{VWAP}_t = \sum_{i \le t} A_i / \sum_{i \le t} V_i\) (adj-adjusted amount, raw volume),
should predict subsequent minute-level returns at standard signal times (09:59, 10:29, …).

## Data

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Access: `get_minute_panel()` / `MinuteBarStore` (no parquet)

## Signal times

Standard bartimes: 09:59, 10:29, 11:29, 13:29, 14:29 (see `STANDARD_BARTIMES` in `core/intraday_alphas.py`).

## Implementation tiers

| Tier | Module | Notes |
|------|--------|-------|
| Python reference | `python_version()` | `core/intraday_alphas.compute_close_vwap_deviation` |
| DDB-native | `ddb_version()` | `ddb.sql` — `context by Symbol, Date` + `cumsum` |

## Consistency gates (Sprint 2–3.1)

1. **Numeric**: `max(abs(ddb - python)) < 1e-10` on aligned keys
2. **Rank**: Spearman(ddb, python) > 0.999 per cross-section
3. **Backtest proxy**: H-L Sharpe relative error < 1%
4. **Bartime**: only `09:59, 10:29, 11:29, 13:29, 14:29` — no shift
5. **Look-ahead**: `cumsum` on `csort Bartime` within session; no forward `move`

## Production flag

```python
# factor_config.py
INTRADAY_CLOSE_VWAP_USE_DDB = False  # set True after validation
```

`intraday_alphas.compute_close_vwap_deviation()` dispatches via this flag; Python path remains fallback.
