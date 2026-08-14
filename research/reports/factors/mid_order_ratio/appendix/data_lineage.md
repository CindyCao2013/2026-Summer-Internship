# Appendix B — 数据血缘

## 1. 血缘总览

```text
ClickHouse cmds
├── SSE_AL_TICK_EXG
└── SZSE_AL_TICK_EXG
        |
        | per-date ExchTime session filter
        | SSE Type='T'
        | SZSE Type='011' + valid bid/ask order IDs
        | Price / Volume / Amount
        | per-print amount classification
        v
Symbol × TradeDate
├── TotalAmount
├── MediumAmount
└── cumulative amount buckets
        |
        | MediumAmount / TotalAmount
        v
mid_order_ratio narrow panel
        |
        +-----------------------------+
        |                             |
        v                             v
DolphinDB Wind                   Risk controls
├── c2c returns                 ├── CITICS industry
├── index returns               ├── market cap
├── PIT index members           ├── momentum
├── ST / limit / status         ├── volatility
└── turnover                    └── turnover
        |                             |
        +-------------+---------------+
                      v
          PIT backtest and robustness
                      |
                      v
      research/reports/factors/mid_order_ratio/
```

## 2. ClickHouse 上游

| Item | Value |
|---|---|
| Config key | `COMMON_CONST.DATA_DB_HFDATA` |
| Host | `10.80.139.9` |
| Port | `8123` |
| Database | `cmds` |
| SSE table | `cmds.SSE_AL_TICK_EXG` |
| SZSE table | `cmds.SZSE_AL_TICK_EXG` |
| Python client | `clickhouse_connect` |

2026-08-04 连接自检 `SELECT 1` 成功。源表审计时可见：

- SSE Tick：2015-01-05 至 2026-08-03；
- SZSE Tick：2008-01-02 至 2026-08-03。

该覆盖描述源表总体，不表示所有 A 股、所有字段在每一天完整。

## 3. DolphinDB 上游

| Item | Value |
|---|---|
| Config key | `COMMON_CONST.DATA_DB_CONN` |
| Host | `10.12.180.9` |
| Port | `8902` |
| Shared session | `core/ddb/connection.py::get_ddb_session` |

2026-08-04 连接自检 `1+1` 返回 2。

主要 Wind 表：

| Data | DDB source | Use |
|---|---|---|
| A-share EOD | `dfs://WIND.ASHAREEODPRICES` | raw c2c returns |
| CSI300 weights | `dfs://WIND.AINDEXHS300WEIGHT` | PIT members |
| CSI500 weights | `dfs://WIND.AINDEXCSI500WEIGHT` | PIT members |
| CSI1000 weights | `dfs://WIND.AINDEXCSI1000WEIGHT` | PIT members |
| Index EOD | Wind index EOD table | CSI1000 index-excess |
| ST status | Wind special-treatment data | signal-date mask |
| Trading status | Wind trading status data | suspension mask |
| Limit prices | Wind EOD limit information | limit mask |
| Industry | CITICS level-1 | industry neutralization |
| Market cap | Wind valuation | log-cap neutralization |
| Turnover | Wind `S_DQ_TURN` | state/style analysis |

## 4. Tick 字段级血缘

| Raw field | Transformed field | Final use |
|---|---|---|
| `ExchTime` | `TradeDate=toDate(ExchTime)` | stock-day key |
| `ExchTime` | regular-session filter | exclude pre-open records |
| `Price` | `amt` | amount calculation |
| `Volume` | `amt` | amount calculation |
| `Amount` | SSE `amt` primary value | amount calculation |
| `BSFlag` | none | not used by `mid_order_ratio` |
| `BidOrderNo` | SZSE execution predicate | require both order numbers > 0 |
| `AskOrderNo` | SZSE execution predicate | require both order numbers > 0 |

SSE:

```text
amt = ifNull(Amount, Price * Volume)
Type = 'T'
```

SZSE:

```text
amt = Price * Volume
Type = '011'
BidOrderNo > 0 and AskOrderNo > 0
```

## 5. Why L2 provides additional information

```text
Same daily OHLCV:

Stock A: 100m traded amount
  mostly <=20k prints
  medium-band share low

Stock B: 100m traded amount
  mostly 40k–200k prints
  medium-band share high

OHLCV sees:          A == B
Minute bars may see: similar intraday path
L2 Tick sees:        different transaction-size composition
```

L2 的增量信息不是“更多价格点”本身，而是**逐笔成交金额分布和订单级属性**。

## 6. 服务端聚合

### Input key

```text
(raw table, ExchTime, Symbol)
```

### Filters

```text
09:30:00 <= ExchTime < 15:00:01
Price > 0
Volume > 0
SSE Type = 'T'
SZSE Type = '011'
SZSE BidOrderNo > 0 and AskOrderNo > 0
```

### Aggregation key

```text
(TradeDate, Symbol)
```

### Aggregated fields

```text
TotalAmount
MediumAmount
SmallAmount
cum_20000 ... cum_300000
```

### Final formula

```text
value = MediumAmount / TotalAmount
      = (cum_200000 - cum_40000) / TotalAmount
```

### Output key

```text
(symbol, tradetime, factorname)
```

## 7. Legacy 因子产物

```text
research/results/l2_reproduction/mid_order_ratio/factor_narrow.parquet
```

| Property | Value |
|---|---:|
| Rows | 463,102 |
| Symbols | 1,303 |
| First factor date | 2023-01-03 |
| Last factor date | 2024-06-28 |
| Mean factor value | 0.2903 |
| Median | 0.3012 |

该文件来自修正前查询，且股票列表是样本期 CSI1000 成分并集。它同时受多日时段和深市成交识别问题影响，正式报告不把它作为因子源或日度 universe。

## 8. 报告版冻结 cache

```text
research/results/l2_reproduction/mid_order_ratio/
analysis/param_sensitivity/
tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet
```

严格 cache：

- 1,805,656 个沪深 A 股股票日；
- 5,174 个有成交因子代码；
- 2023-01-03 至 2024-06-28。

缓存构建器先从 Wind 收益面板取得 5,175 个候选代码，再在仅含 SSE/SZSE 的 ClickHouse Tick 表中执行严格成交过滤。artifact 的 `ALL` universe 实际是沪深 A 股交集，不含北交所；每日平均最终有效 4,840 只。

元数据：

```text
tick_bucketed_strict_trade_2023-01-01_2024-06-30.metadata.json
SHA256 = ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f
```

报告版 universe、decile、参数和状态结果都来自该冻结 cache。

## 9. Legacy-vs-strict 构造影响审计

严格累计桶面板与旧窄表的共同股票日：

| Check | Result |
|---|---:|
| Matched rows | 463,087 / 463,102 |
| Pearson | 0.8287 |
| Spearman | 0.7996 |
| Mean absolute difference | 0.0349 |
| 99% absolute difference | 0.1485 |
| Share within \(10^{-12}\) | 0.99% |

这不是快照级微小误差，而是旧输入记录筛选不正确造成的实质变化。公式保持不变，但 legacy 产物已被严格 cache 取代。

机器结果：

- `artifacts/construction_crosscheck.json`
- `artifacts/construction_crosscheck_top100.csv`

## 10. 深市成交与交易时段修正审计

对 2024-06-27 的 697 只样本深市股票：

| Item | Result |
|---|---:|
| Legacy positive-price/volume rows | 29,836,521 |
| Strict execution rows | 11,916,224 |
| Legacy amount | RMB 370.77bn |
| Strict execution amount | RMB 93.19bn |
| Strict / legacy amount | 25.13% |
| Factor Spearman | 0.858 |
| Mean absolute factor difference | 0.0864 |

只比较 `Type='011'` 与其他 Type 会得到近乎相等的错误结论，因为非成交事件也位于 `011` 内。双订单号条件才是关键。另在 1,293 只样本股上核验到盘前/盘后金额占代表日总额约 2.52%；因此 SQL 还必须逐日施加时段条件。

机器记录：`artifacts/strict_trade_filter_audit.json`。该文件保留汇总数字和过滤口径，但原始 symbol list / query snapshot 未在首次审计时冻结；这是已披露的复现限制。

## 11. 收益与信号对齐

```text
Factor date t:
  full-session tick aggregation
  signal-date tradability and membership mask

Return date t+1:
  signal.shift(1)
  raw c2c stock return
  optional common benchmark subtraction
```

主 decile figure：

```text
stock raw c2c - CSI1000 index c2c
```

四 universe H-L：

```text
stock raw c2c - exact valid-universe EW c2c
```

RankIC 和 H-L 对每日公共基准平移不敏感，单个 decile 收益敏感。

## 12. 最终产物血缘

```text
artifacts/universe_comparison.csv
  <- PIT members + raw returns + reconstructed factor

artifacts/csi1000_decile_index_excess_daily.csv
  <- CSI1000 PIT signal + CSI1000 index-excess returns

artifacts/csi1000_decile_summary.csv
  <- daily decile artifact + arithmetic annualization + cross-decile Spearman

artifacts/strict_trade_filter_audit.json
  <- one-off SZSE execution/session filter materiality audit

artifacts/neutralization_comparison.csv
  <- strict raw/ind/cap/ind_cap panels + CSI1000 PIT evaluation

artifacts/neutralization_by_universe.csv
  <- 4 universe-specific PIT panels × 4 locally re-estimated variants

artifacts/second_neutralization_comparison.csv
  <- strict ind+cap residual + prewarmed PIT style controls + matched-cell baselines

artifacts/parameter_sensitivity_csi1000_pit.csv
  <- cumulative amount buckets + 25 threshold definitions

artifacts/state_dependence_summary.csv
  <- PIT signal + prewarmed and lagged 20-day turnover terciles

figures/*.png
  <- only the machine-readable artifacts above
```

