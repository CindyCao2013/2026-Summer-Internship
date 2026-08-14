# APM_SessionResidual — Data Source

## Primary stores

| Layer | Path / table |
|-------|----------------|
| Stock minute | `dfs://QV_Trade_to_MinuteBar` / `Stock_one_minute` |
| Stock EOD | `dfs://WIND.ASHAREEODPRICES` (Open/Close) |
| Index EOD | `dfs://WIND.AINDEXEODPRICES` (`000852.SH`) |
| Ret20 | `Factor_Dev_Lib.get_Ret_Matrix(..., method="c2c")` rolling sum 20 |

## Minute fields used

`Symbol, Date, Bartime, Close` only for PM aggregation.

**Not selected:** any `Active_*` column.

## Coverage (scout)

| Item | Value |
|------|-------|
| Universe | CSI1000 |
| Period | 2021-01-01 → 2025-12-31 |
| Panel days | 1212 |
| Names (masked) | ~1902 |

## Adapted gap

Index **minute** bars are absent from the stock minute table.  
Index afternoon residual uses **EOD daytime** `Close/Open` as proxy → `identity_class=adapted_replication`.

Upgrade to `true_replication` only if index minute or PDF-signed EOD equivalence arrives.

## Research cache

```text
research/cache/apm_session/
  meta/ · calendar/ · stock_overnight/ · stock_pm/
  index_session_proxy/ · residual_panel/ · signal/
```
