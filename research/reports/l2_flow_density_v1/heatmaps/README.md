# P2 Intraday Heatmaps

**Factor:** `net_active_flow_mktcap_20d` (daily, size+industry neut, shift-1)
**Universe:** CSI1000 (`000852.SH`)
**Sample:** last 504d stamped onto bartimes [(9, 59), (10, 29), (10, 59), (11, 29), (13, 29), (13, 59), (14, 29)]

Pipeline: `Intraday_Factor_Test_Process` / `intraday_lib.create_group_heatmap`

## Outputs (`net_active_flow_mktcap_20d/`)
- `group_HML_annualized_return_heatmap.png`
- `group_HML_sharpe_heatmap.png`
- `group_performance_summary.csv`
- `group_data_ret.parquet`

Note: same daily signal is stamped on all bartimes — heatmap variation
is mainly across **return horizons**, not true intraday signal decay.
True open/close cumulative-flow variants belong in P2 deepen.
