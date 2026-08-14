# mid_order_ratio 报告上下文：项目架构与端到端链路

> 用途：在撰写 `research/reports/factors/mid_order_ratio/` 正式研究报告前，冻结项目架构、数据连接、计算链路、回测口径与报告产物路径。
> 核验日期：2026-08-04。

## 1. 项目定位

本仓库包含三条研究数据线：

1. EOD 日频因子：Wind 日行情、估值、财务数据；
2. 分钟/日内因子：DolphinDB 一分钟线；
3. L2 微观结构因子：ClickHouse 逐笔成交与盘口快照。

三条数据线最终均收敛到统一的截面因子评估框架：滞后信号、可交易性过滤、RankIC、十分组和 H-L 检验。`mid_order_ratio` 属于第 3 条数据线，其原始数据必须使用逐笔成交，分钟 OHLCV 无法还原单笔成交金额分布。

## 2. 核心目录职责

| 层 | 关键位置 | 职责 |
|---|---|---|
| 连接配置 | `COMMON_CONST.py` | `DATA_DB_HFDATA`（ClickHouse）与 `DATA_DB_CONN`（DolphinDB） |
| 共享 DDB 会话 | `core/ddb/connection.py` | `get_ddb_session()` 进程级共享连接 |
| L2 因子构造 | `l2_factor_reproduction/python/ch_tick.py` | ClickHouse 服务端按股票、交易日聚合逐笔成交 |
| 因子组装 | `l2_factor_reproduction/python/factor_builder.py` | 获取股票池、分块调用 CH、生成窄表 |
| 因子落盘 | `l2_factor_reproduction/python/factor_runner.py` | 保存 `factor_narrow.parquet` |
| L2 回测 | `l2_factor_reproduction/python/backtest.py` | 窄转宽、过滤、`shift(1)`、RankIC、十分组、H-L |
| 通用评估库 | `Factor_Dev_Lib.py` | 收益、可交易掩码、指数成员、分组、Sharpe、MDD、换手 |
| 稳健性脚本 | `l2_factor_reproduction/scripts/` | 中性化、二次中性化、参数敏感性、状态依赖、时间稳定性 |
| 原始研究产物 | `research/results/l2_reproduction/mid_order_ratio/` | parquet、CSV、JSON、PNG |
| Research Pack | `research/reports/factors/` | 完整研究报告、图表、附录与可复现命令 |
| Delivery 层 | `research_delivery/` | 冻结指标定义、精简因子卡片与交付治理 |

## 3. 数据连接

### 3.1 ClickHouse

- 配置常量：`COMMON_CONST.DATA_DB_HFDATA`
- 地址：`10.80.139.9:8123`
- database：`cmds`
- 客户端：`clickhouse_connect.get_client(**DATA_DB_HFDATA)`
- 本因子上游：
  - `cmds.SSE_AL_TICK_EXG`
  - `cmds.SZSE_AL_TICK_EXG`

ClickHouse 是 L2 逐笔成交的权威上游。查询逐日限定常规时段，SSE 使用 `Type='T'`，SZSE 使用 `Type='011'` 且买卖订单号均大于 0，再在服务端完成 `GROUP BY Symbol, TradeDate`。Python 仅接收股票日聚合结果，避免传输数十亿行 tick。

### 3.2 DolphinDB

- 配置常量：`COMMON_CONST.DATA_DB_CONN`
- 地址：`10.12.180.9:8902`
- 统一入口：`core/ddb/connection.py::get_ddb_session`
- 本报告用途：
  - Wind A 股日收益；
  - CSI300 / CSI500 / CSI1000 每日成分；
  - ST、涨跌停、停牌掩码；
  - CITICS 一级行业；
  - 总市值与换手率。

DolphinDB 不含本因子所需的逐笔成交权威表，因此不负责构造 `mid_order_ratio`；它负责股票池、风险暴露和回测数据。

## 4. 端到端主链路

```text
Raw Data
    |
    v
ClickHouse Tick Tables
    |
    v
Factor Construction
    |
    v
factor_narrow.parquet
    |
    v
Neutralization
    |
    v
Backtest
    |
    v
Robustness Analysis
    |
    v
Research Report
```

展开后：

```text
cmds.SSE_AL_TICK_EXG / cmds.SZSE_AL_TICK_EXG
        |
        | ch_tick.fetch_tick_agg_by_date_range()
        | 逐笔金额分档；CH 内 GROUP BY (Symbol, TradeDate)
        v
TotalAmount / MediumAmount / SmallAmount
        |
        | aggregate_wide_to_narrow()
        | MediumAmount / TotalAmount
        v
research/results/l2_reproduction/mid_order_ratio/factor_narrow.parquet
        |
        +--> raw
        +--> industry residual
        +--> market-cap residual
        +--> industry + market-cap residual
        |
        | backtest.prepare_factor_signal()
        | 可交易掩码 -> signal.shift(1)
        v
RankIC + deciles + H-L + turnover + MDD
        |
        +--> second neutralization
        +--> parameter sensitivity
        +--> turnover state dependence
        +--> monthly / rolling stability
        +--> universe comparison
        v
research/reports/factors/mid_order_ratio/
```

报告版还包含一条独立的严格审计路径：

```text
build_mid_order_ratio_strict_cache.py
  -> corrected fetch_tick_bucketed
  -> tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet
  -> generate_mid_order_ratio_report_artifacts.py
  -> PIT universe / neutralization / robustness / figures
```

现存 canonical `factor_narrow.parquet` 生成于修正前，只作为 legacy 对照，不是正式报告输入。

## 5. 因子计算链路

1. `scripts/run_single_factor.py` 解析 factor、日期和 universe；
2. `python/factor_runner.py::run_single_factor` 创建 DDB 客户端；
3. `python/factor_builder.py::_stock_pool_vector` 从 DDB 取得研究区间内指数成分并集，用于限制 CH 查询代码；
4. `python/ch_tick.py::fetch_tick_agg_by_date_range` 逐日过滤常规时段、按交易所识别成交，并在 CH 内计算每个股票日的总成交额和中单成交额；
5. `aggregate_wide_to_narrow` 计算比率并生成：
   - `symbol`
   - `tradetime`
   - `factorname`
   - `value`
6. `python/factor_runner.py::_save_narrow` 保存到 `research/results/l2_reproduction/<factor>/factor_narrow.parquet`。

注意：factor builder 的股票列表是“样本区间成分并集”，不是逐日 membership mask。正式四股票池比较必须在回测阶段再次使用 `Factor_Dev_Lib.get_index_member_mask()` 施加逐日成分约束。

## 6. 回测链路与收益含义

`l2_factor_reproduction/python/backtest.py`：

1. 窄表 pivot 为 `Date × Symbol`；
2. 加载非涨跌停、非 ST、非停牌掩码；
3. 使用 `signal.shift(1)`，令 T−1 全日因子预测 T 日收益；
4. 计算日度截面 Spearman RankIC；
5. 调用 `Factor_Dev_Lib.groupTest` 做十分组等权和 H-L；
6. 保存 `rank_ic.csv`、`group_pnl.csv`、`group_turnover.csv`、`summary.json` 和图。

现有基础 L2 回测调用 `get_Ret_Matrix(..., base_index="000852.SH")`，因此现有 `cum_pnl.png` 的 G1–G10 曲线是 **CSI1000 指数超额 close-to-close 收益的简单累加**，不是绝对收益，也不是复利净值。现有图片标题未写明该事实，正式报告必须重绘并在标题、纵轴和脚注明确：

- 股票池；
- 基准；
- 收益频率；
- `cumsum` 或复利 NAV；
- 信号滞后。

H-L、RankIC 不受每日统一基准收益平移影响，但单边 decile 曲线和单组 Sharpe 会受影响。

## 7. 中性化与稳健性链路

- 一阶中性化：`test_neutralization.py`
  - `ind`：CITICS 一级行业 dummy；
  - `cap`：log 总市值；
  - `ind_cap`：行业 + 市值；
  - 逐日截面 OLS 取残差。
- 二阶中性化：`test_double_neutralization.py` / `screen_second_pass.py`
  - 在一阶残差上继续剥离 20 日 momentum、volatility、turnover。
- 参数敏感性：`analyze_param_sensitivity.py`
  - L ∈ {2,3,4,5,6} 万；
  - H ∈ {10,15,20,25,30} 万；
  - CH 累计金额桶一次查询，Python 拼装 25 个定义。
- 状态依赖：`analyze_state_dependence.py`
  - 按 20 日平均 log turnover 分三组并计算组内 RankIC。
- 时间稳定：`analyze_time_stability.py`
  - 月度 IC、滚动 IC、IC 分布。

## 8. 报告生成链路

仓库有两种报告路径：

1. 通用生成器：`factor_report_generator_v2.py`
   - 从 `factor_specs/*.yaml` 与既有产物 harvest；
   - 输出到 `research/reports/factors/<factor>/`；
   - 不重新计算因子。
2. Delivery 卡片：`research_delivery/scripts/generate_factor_card.py`
   - 面向精简交付卡片；
   - 指标治理受 `research_delivery/GOVERNANCE.md` 和 `METRICS_G10_EXCESS.md` 约束。

`mid_order_ratio` 的完整 Research Pack 已由专用、可复现分析脚本生成于 `research/reports/factors/mid_order_ratio/`。`research_delivery/` 中的单文件草稿不是本次正式产物。

## 9. 已核验的实现风险

1. 旧多日 SQL 只在区间端点限制 09:30/15:00，会纳入中间日期的盘前记录；现已增加逐行、逐日 session predicate。
2. 深市 `Type='011'` 内仍包含非成交事件；现以 `BidOrderNo>0 AND AskOrderNo>0` 识别成交。代表日严格成交额仅为旧口径的 25.13%，旧/严格全样本面板 Spearman 为 0.7996，因此旧产物不再作为证据。
3. 现有基础回测使用样本期指数成分并集；正式报告已用 `get_index_member_mask()` 修正为 PIT。
4. 原参数敏感性脚本未施加指数 membership；正式报告已在严格 cache 上重算 CSI1000 PIT 25 单元。
5. 原 decile 图标题含糊；正式图已明确 `CSI1000 index-excess`、`cumulative sum` 与 T-1 信号。

