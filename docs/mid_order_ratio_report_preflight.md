# mid_order_ratio 正式报告前置缺口清单

> 本清单在正式 Research Pack 写作前生成。状态仅依据 2026-08-04 已落盘代码和产物；缺口未补齐前，不将 `research_delivery/factors/MidOrderRatio/report.md` 视为正式报告。

## 1. 缺失或待重新核验的信息

| 项目 | 当前状态 | 正式报告要求 |
|---|---|---|
| 四股票池测试 | **缺失**：现有基础回测仅覆盖 CSI1000 成分并集，且没有逐日 membership mask | 对 ALL、CSI300、CSI500、CSI1000 使用同一因子定义、同一日期、逐日成分掩码，输出样本量、RankIC、ICIR、有效方向 H-L Sharpe、MDD、换手 |
| SSE/SZSE A-share 因子面板 | **已用严格缓存恢复**：`tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet` | 以 `cum_200000-cum_40000` 重建 L4w/H20w，不传输原始 tick；明确不含北交所 |
| decile 收益口径 | **已确认但图标题不合格**：当前 `cum_pnl.png` 是 CSI1000 指数超额 c2c 的简单累加 | 重绘；标题、纵轴和脚注明确 “CSI1000 index-excess”, “cumulative sum”, “T−1 signal” |
| Sharpe 口径 | **已确认**：当前 headline 是方向翻转后的 gross H-L Sharpe，年化 250，rf=0 | 正文明确写为 `hl_sharpe_gross_directional`；不得与 G10 exact-EW excess Sharpe 或 15bp net Sharpe混用 |
| 指数成分口径 | **存在风险**：factor builder 使用样本期成分并集 | 四股票池比较改用 `get_index_member_mask` 的逐日 Wind 权重表 |
| 成交记录识别 | **审计发现并已修正实质缺陷**：旧多日 session 条件不逐日生效；深市未用双订单号识别成交 | 修正代码、增加测试、重新抽取严格 cache，并从头重算全部报告结果；旧产物仅作对照 |
| 参数敏感性股票池 | **当前产物含歧义**：脚本未施加 CSI1000 逐日 membership | 使用已缓存 CH buckets 重算 CSI1000 PIT 25 组网格 |
| 参数敏感性 Sharpe 热图 | **缺失** | 生成 CSI1000 PIT、有效方向 H-L gross Sharpe 热图 |
| exact-EW long-book metric | **缺失**（不在本任务 mandatory universe 表，但属于 delivery 治理） | 可作为补充模块计算；不得用 decile 均值近似 exact valid-universe EW |
| 样本外扩展 | 2024-06 后未覆盖 | 在局限性和未来研究中明确，不虚构结果 |

## 2. 缺失或必须重绘的图表

| # | 图表 | 当前状态 | 处理 |
|---:|---|---|---|
| 1 | Pipeline architecture diagram | 缺失 | 新生成 PNG；同时在 context 文档保留文本图 |
| 2 | Order size distribution | 缺失 | 从 CH cumulative buckets 计算 CSI1000 PIT 成交金额分档占比 |
| 3 | Universe comparison table | 缺失 | ALL / CSI300 / CSI500 / CSI1000 指标表，保存 CSV + PNG |
| 4 | Decile cumulative excess return curves | 有旧图但标题含糊 | 以 CSI1000 PIT、CSI1000 index-excess c2c 重绘 |
| 5 | Decile annualized return bar | 有旧图但标题含糊 | 以同一 CSI1000 PIT 口径重绘 |
| 6 | IC time series | 缺失 | 用报告版 CSI1000 PIT 日 RankIC 生成 |
| 7 | Rolling IC | 有旧图 | 用报告版 CSI1000 PIT 日 RankIC 重绘 63 日均值 |
| 8 | Neutralization comparison | 缺失 | raw / ind / cap / ind_cap 的 RankIC、ICIR、Sharpe、MDD 对比 |
| 9 | Second neutralization comparison | 缺失 | 基线与 +momentum / +volatility / +turnover 的 ICIR 条形图 |
| 10 | Parameter sensitivity heatmap | ICIR 有旧图；Sharpe 缺失；股票池含歧义 | 重算 CSI1000 PIT 后生成 ICIR + Sharpe 热图 |
| 11 | Turnover vs IC relationship | 仅有 regime boxplot | 生成 turnover tercile 与 mean RankIC / ICIR 的关系图 |
| 12 | High/Low turnover regime comparison | 有旧图 | 用报告版 CSI1000 PIT 重算并重绘 IC 分布 |

补充图：

- 月度 IC 柱状图；
- IC 分布直方图；
- 一阶中性化 decile 累积曲线（可作为附录，不替代对比图）。

## 3. 已存在且可复用的命令

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# 基础构造与回测
$PY l2_factor_reproduction/scripts/run_single_factor.py \
  --factor mid_order_ratio --start 2023-01-01 --end 2024-06-30

# 一阶中性化
$PY l2_factor_reproduction/scripts/test_neutralization.py \
  --factor mid_order_ratio --neutral_types ind cap ind_cap

# 二阶中性化
$PY l2_factor_reproduction/scripts/test_double_neutralization.py \
  --factor mid_order_ratio
$PY l2_factor_reproduction/scripts/screen_second_pass.py \
  --factor mid_order_ratio

# 周度持有
$PY l2_factor_reproduction/scripts/optimize_weekly.py \
  --factor mid_order_ratio --raw
$PY l2_factor_reproduction/scripts/optimize_weekly.py \
  --factor mid_order_ratio

# 原始稳健性脚本（其股票池限制见本清单）
$PY l2_factor_reproduction/scripts/analyze_param_sensitivity.py \
  --start 2023-01-01 --end 2024-06-30
$PY l2_factor_reproduction/scripts/analyze_state_dependence.py \
  --factor mid_order_ratio
$PY l2_factor_reproduction/scripts/analyze_time_stability.py \
  --factor mid_order_ratio
```

## 4. 已新增并运行的报告专用命令

严格缓存构建：

```bash
$PY l2_factor_reproduction/scripts/build_mid_order_ratio_strict_cache.py \
  --start 2023-01-01 --end 2024-06-30
```

报告脚本：

```text
l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py
```

已执行命令：

```bash
$PY l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py \
  --start 2023-01-01 \
  --end 2024-06-30 \
  --bucket-cache research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet \
  --output-root research/reports/factors/mid_order_ratio
```

该命令必须：

1. 从缓存重建 L4w/H20w 全 A 因子面板；
2. 加载 raw c2c、交易状态、ST、涨跌停及三指数日度成员；
3. 运行 ALL/CSI300/CSI500/CSI1000 同口径比较；
4. 生成 CSI1000 PIT 参数网格；
5. 重新计算状态依赖和时间稳定性；
6. 输出机器可读 CSV/JSON；
7. 生成全部报告图，且每张图包含 universe、benchmark、日期、单位和信号滞后。

## 5. 写作启动门槛

只有同时满足以下条件后，才开始写正式多文件报告：

- [x] 四股票池指标 CSV 已生成并完成数值审计；
- [x] CSI1000 PIT decile 曲线确认使用 CSI1000 index-excess return；
- [x] ICIR 与 Sharpe 参数热图均为同一 CSI1000 PIT 口径；
- [x] 12 张必需图均存在、标题无歧义；
- [x] raw / neutralized / second-neutralized 的样本区间与股票池已逐项标注；
- [x] 所有报告数字可追溯到 CSV/JSON，而不是人工转录旧工作文档。

**Closure（2026-08-04）：** 构造缺陷修正后，已重新生成严格 cache（SHA256
`ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f`），并从头重算六项门槛及正式多文件报告。

