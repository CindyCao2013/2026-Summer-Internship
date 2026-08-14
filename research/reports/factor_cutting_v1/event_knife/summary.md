# Event Knife — Limit-Up / Limit-Down Filter

**Period:** `2018-01-02 -> 2025-12-31`

Source: `Factor_Dev_Lib.get_EOD_Not_Limit` (close strictly inside `S_DQ_LIMIT` / `S_DQ_STOPPING`).

## Coverage

- Drop fraction of object cells: **2.32%**
- Mean names/day ref → kept: 4427 → 4324

## RankIC comparison

| Mode | RankIC | ICIR | Monthly IC | ΔRankIC | Retention | n_days |
|------|--------|------|------------|---------|-----------|--------|
| `raw` | -0.0338 | -6.48 | -0.0613 | +0.0000 | 1.000 | 1703 |
| `filter_signal` | -0.0382 | -7.20 | -0.0555 | -0.0044 | 1.130 | 1703 |
| `filter_cut` | -0.0252 | -5.12 | -0.0287 | +0.0086 | 0.747 | 1703 |
| `filter_cut_signal` | -0.0252 | -5.12 | -0.0287 | +0.0086 | 0.747 | 1703 |

## Modes

- `raw` — no filter
- `filter_signal` — mask limit names on the finished factor before IC
- `filter_cut` — exclude limit days from object/knife inside W-cut
- `filter_cut_signal` — both

## Interpretation

- Retention > 1 and |IC| larger → limit days diluted alpha (paper-table-2 style).
- Retention ≈ 1 → limit filter mostly cosmetic.
- Retention << 1 → alpha concentrated on limit events (fragile / hard to trade).

## Notes

- Primary knife: `ats_trade_count`
- best_mode=`filter_signal` RankIC=-0.0382 (raw=-0.0338, retention=1.130) → LIMIT FILTER HELPS
- Dual amount+ATS also compared — see dual_amount_ats_limit_compare.csv

## Dual amount+ATS (residual_add)

| Mode | RankIC | ICIR | ΔRankIC | Retention |
|------|--------|------|---------|----------|
| `raw` | -0.0408 | -6.88 | +0.0000 | 1.000 |
| `filter_signal` | -0.0449 | -7.42 | -0.0041 | 1.101 |
| `filter_cut` | -0.0333 | -5.67 | +0.0075 | 0.816 |
