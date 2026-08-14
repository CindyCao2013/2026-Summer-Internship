# IdealReversal — Data Source

## Input

| Item | Spec |
|------|------|
| Fields | close (return), amount, trade_count |
| Bar | EOD daily |
| Universe (research harvest) | ALL A-shares |
| Target calendar | 2018–2025 |

Fallback: if `trade_count` missing, proxy ATS via `amount / volume`.

---

## Actual coverage (this pack)

| Field | Value |
|-------|-------|
| Harvest label | `cutting_v1_harvest` |
| Block sample | **1703 trading days** |
| Yearly detail | 2019–2025 (see `stability/yearly_by_year.csv`) |
| Exception | Full 2018–2025 Dual Benchmark not re-run |

Requested vs actual recorded in `summary.yaml` (`coverage_exception: true`).

---

## Pipeline layers

```text
EOD close / amount / trade_count
    → ATS knife panel
    → M_high / M_low legs
    → M = M_high − M_low panel (date × symbol)
```

Cache: `research/cache/ideal_reversal_panels`  
Loaders: `factor_data_loaders.load_eod_enriched_tables`
