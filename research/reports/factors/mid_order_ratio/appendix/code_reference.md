# Appendix A — 代码索引

## 1. ClickHouse Tick 聚合

### `l2_factor_reproduction/python/ch_tick.py`

#### `fetch_tick_agg_by_date_range`（约第 95 行）

职责：

- 按日期范围从 SSE / SZSE Tick 表读取逐笔成交；
- 过滤交易时段、零价格和零成交量；
- 在 ClickHouse 内计算：
  - `TotalAmount`
  - `MediumAmount`
  - `SmallAmount`
- `GROUP BY Symbol, TradeDate`；
- 合并两交易所股票日结果。

关键金额口径：

```text
SSE  = ifNull(Amount, Price * Volume)
SZSE = Price * Volume
```

关键成交口径：

- SSE 使用 `Type='T'`；
- SZSE 使用 `Type='011'` 且 `BidOrderNo>0 AND AskOrderNo>0`；
- `_regular_session_filter_sql` 对区间内每一行施加 09:30–15:00 条件；
- 该函数的输出是股票日聚合，不是原始 Tick。

#### `fetch_tick_bucketed`（约第 187 行）

职责：

- 对多个订单金额阈值一次生成累计成交金额；
- 输出 `cum_20000`、`cum_30000` … `cum_300000`；
- 服务参数敏感性分析，避免为 25 个参数重复扫描 Tick。

报告版重构：

```text
mid_order_ratio(L,H)
= (cum_H - cum_L) / TotalAmount
```

与日常聚合路径相同，bucket 查询也使用逐日时段和交易所成交筛选，避免参数分析与基础因子产生口径分叉。

#### `aggregate_wide_to_narrow`（约第 481 行）

职责：

- 将 `MediumAmount / TotalAmount` 转为因子值；
- 删除零分母和无效值；
- 输出统一窄表：
  - `symbol`
  - `tradetime`
  - `factorname`
  - `value`

## 2. 因子生成

### `l2_factor_reproduction/python/factor_builder.py`

#### `_build_tick_order_size_factor`（约第 109 行）

职责：

1. 取得研究日期和 universe；
2. 调用 `_stock_pool_vector` 获取样本期指数成分并集；
3. 按日期块调用 ClickHouse 聚合；
4. 将股票日金额转为 `mid_order_ratio` / `small_order_ratio`；
5. 拼接并去重。

重要口径：

- 这里的股票列表是查询裁剪用的**样本期成分并集**；
- point-in-time membership 必须在正式回测阶段另行施加；
- universe 参数不是自动完成 PIT 回测的充分条件。

#### `build_factor`（约第 162 行）

统一入口：

- 根据 `FACTOR_BUILDERS` 注册表选择构造函数；
- 支持覆盖 `start_day`、`end_day` 和 `universe`；
- 对 Tick 因子进入 ClickHouse 构造路径，对其他 L2 因子进入相应 DDB 路径。

### `l2_factor_reproduction/python/factor_runner.py`

#### `run_single_factor`（约第 61 行）

职责：

- 建立 DDB 客户端；
- 调用 `build_factor`；
- 可选保存到：
  `research/results/l2_reproduction/<factor>/factor_narrow.parquet`；
- 确保会话关闭。

## 3. 回测评估

### `l2_factor_reproduction/python/backtest.py`

#### `prepare_factor_signal`（约第 60 行）

职责：

- 窄表 pivot 为 `Date × Symbol`；
- 对齐收益和可交易性掩码；
- 使用 `signal.shift(1)`；
- 返回已滞后信号和收益矩阵。

现有基础路径默认收益为 CSI1000 index-excess；它不自动应用逐日 index membership。

#### `compute_rank_ic`（约第 87 行）

逐日：

```python
signal.corrwith(ret, axis=1, method="spearman")
```

输出日度 RankIC Series。

#### `summarize_backtest`（约第 91 行）

汇总：

- RankIC mean / std / ICIR；
- H-L annual return；
- H-L Sharpe；
- H-L MDD；
- turnover；
- factor direction。

注意检查 summary 字段是 raw direction 还是 effective direction。报告版专用脚本将两者明确分开。

#### `backtest_factor`（约第 130 行）

主流程：

1. `narrow_to_wide`；
2. `prepare_factor_signal`；
3. 根据 IC 符号确定方向；
4. `Factor_Dev_Lib.groupTest`；
5. `summarize_backtest`。

#### `save_group_plots`（约第 201 行）

生成旧版 `cum_pnl.png` 和 `decile_bar.png`。旧标题没有完整写明 benchmark、PIT/union 和 `cumsum` 口径，因此正式报告未直接发布这些图。

#### `_save_backtest_outputs`（约第 288 行）

保存：

- `group_pnl.csv`
- `group_turnover.csv`
- `rank_ic.csv`
- `summary.json`
- 基础图表。

## 4. 一阶中性化

### `l2_factor_reproduction/scripts/test_neutralization.py`

职责：

- 读取 raw factor narrow；
- 调用 `Factor_Dev_Lib.panel_neutral_size_ind`；
- 支持：
  - `ind`
  - `cap`
  - `ind_cap`
- 对每个 residual panel 重新运行同一回测；
- 输出 `neutralization_comparison.csv`。

依赖：

- CITICS 一级行业；
- Wind 总市值；
- 每日截面 OLS。

## 5. 二阶中性化

### `l2_factor_reproduction/python/neutralization.py`

#### `neutralize_again`（约第 16 行）

在已有 residual 上分别对指定风格特征做逐日截面回归并取残差。

### `l2_factor_reproduction/scripts/test_double_neutralization.py`

构造并检验：

- 20 日 momentum；
- 20 日 volatility；
- 20 日平均 log turnover；
- 分别输出三个单特征残差诊断。

### `l2_factor_reproduction/scripts/screen_second_pass.py`

输出可直接比较的二阶中性化摘要，如：

```text
second_pass_screen_ind_cap.csv
```

这些旧产物使用 CSI1000 样本期成分并集且依赖修正前的 canonical factor，因此不进入正式报告。报告生成器从严格 cache 在 CSI1000 PIT 面板上重新计算三个二阶残差。

## 6. 周度诊断

### `l2_factor_reproduction/scripts/optimize_weekly.py`

职责：

- 选取每周首个交易日形成新权重；
- 周内持有；
- 对比 daily 与 weekly 的 gross、turnover 和展示性成本。

该脚本用于研究信号持续性，不构成交易频率建议；本报告 headline 不使用其结果。

## 7. 参数敏感性

### `l2_factor_reproduction/scripts/analyze_param_sensitivity.py`

职责：

- 调用 `fetch_tick_bucketed`；
- 构造 5×5 L/H 网格；
- 运行 RankIC、ICIR、H-L 和 turnover；
- 生成基础 heatmap。

原脚本未施加 PIT index membership。正式报告由专用脚本读取同一 bucket cache，并在每个网格单元应用 CSI1000 日度成员。

## 8. 状态依赖

### `l2_factor_reproduction/scripts/analyze_state_dependence.py`

职责：

- 从 Wind 获取 turnover；
- 计算 20 日平滑状态；
- 按横截面 tercile 分组；
- 计算组内 RankIC。

正式报告专用版本将 turnover state 再滞后一天，避免以 T 日已实现换手率解释 T 日收益。

## 9. 时间稳定性

### `l2_factor_reproduction/scripts/analyze_time_stability.py`

职责：

- 读取日度 RankIC；
- 生成月度平均；
- 生成滚动 IC；
- 生成 IC 分布。

正式报告使用 PIT CSI1000 重新计算的 `csi1000_rank_ic_daily.csv`，不直接复用 union 版本。

## 10. 基础运行入口

### `l2_factor_reproduction/scripts/run_single_factor.py`

命令行参数：

- `--factor`
- `--start`
- `--end`
- `--universe`
- `--no-backtest`
- `--no-save`

它串联：

```text
factor_runner.run_single_factor
  -> backtest.backtest_factor
  -> _save_backtest_outputs
```

## 11. 正式报告专用生成器

### `l2_factor_reproduction/scripts/build_mid_order_ratio_strict_cache.py`

本次新增的修正缓存构建器：

1. 从 Wind 收益面板取得候选代码，最终报告范围限定为 SSE/SZSE A 股；
2. 按季度调用已修正的 `fetch_tick_bucketed`；
3. 不覆盖旧 cache，保存 strict-trade 分块和合并 parquet；
4. 写入筛选规则、行数、日期和 SHA256 元数据。

### `l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py`

本次新增的审计脚本，职责：

1. 校验 strict metadata、筛选语义和 SHA256 后，从冻结 bucket cache 重构沪深 A 股 L4w/H20w；
2. 加载 Wind raw c2c 和可交易掩码；
3. 施加 SSE/SZSE（artifact key `ALL`）/ CSI300 / CSI500 / CSI1000 PIT universe；
4. 输出同口径 universe comparison；
5. 以冻结 `effective_direction=-1` 重算四股票池 × 四方法的一阶中性化矩阵，以及 CSI1000 PIT decile、decile monotonicity、IC、parameter grid 和 state dependence；
6. 二阶 neutralization 使用样本前预热，并为每个 style residual 计算完全相同股票日支持上的 matched baseline；
7. 生成全部自描述图；
8. 保存 artifact manifest 和 legacy-vs-strict 构造影响审计。

它不会修改原始 Tick 表，也不会覆盖 `research/results/l2_reproduction/mid_order_ratio/` 的旧研究记录。

### `l2_factor_reproduction/scripts/export_mid_order_ratio_report.py`

将 README、八章正文和三个附录按固定顺序合并：

1. 用 Mistune 生成带目录、内嵌图表和本地 MathJax 的 HTML；
2. 用 Matplotlib 生成 A4 分页、嵌入中文字体的 PDF；
3. 输出至 `research/reports/factors/mid_order_ratio/export/`；
4. 不修改共享 Python/Conda 环境。

