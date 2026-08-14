# mid_order_ratio 端到端链路（Pipeline Deep Trace）

> **Deprecated legacy trace:** 本文件中的数值来自修正前查询，不能作为正式研究证据。旧查询未逐日施加时段条件，也未用深市双订单号识别成交。请使用
> [`mid_order_ratio_report_context.md`](mid_order_ratio_report_context.md) 和
> [`research/reports/factors/mid_order_ratio/`](../research/reports/factors/mid_order_ratio/README.md)。
>
> 本文件仅保留为历史审计记录。
> 生成时点：2026-08-04。因子归属子项目 `l2_factor_reproduction/`（海通 L2 研报复现）。

---

## 1. 全链路图

```text
ClickHouse cmds.SSE_AL_TICK_EXG / SZSE_AL_TICK_EXG（逐笔成交）
        │  l2_factor_reproduction/python/ch_tick.py
        │  fetch_tick_agg_by_date_range(): CH 服务端 GROUP BY (Symbol, TradeDate)
        │    SSE: Type='T', amt = ifNull(Amount, Price*Volume)
        │    SZSE: amt = Price*Volume
        ▼
(symbol, date) 宽中间表: TotalAmount / MediumAmount / SmallAmount
        │  aggregate_wide_to_narrow(): mid = Medium(4万,20万] / Total
        ▼
窄表 factor_narrow.parquet  (symbol, tradetime=date+09:30, factorname, value)
        │  l2_factor_reproduction/python/backtest.py
        │  窄→宽 → [可选中性化] → 涨跌停/ST/停牌过滤 → signal.shift(1)
        ▼
Factor_Dev_Lib.groupTest 十分组等权 H-L
        │  收益 = get_Ret_Matrix(base_index=000852.SH) ← CSI1000 超额 c2c
        ▼
指标 + 图: rank_ic.csv / group_pnl.csv / cum_pnl.png / decile_bar.png / summary.json
        │
        ├─→ scripts/test_neutralization.py       中性化三模式对比
        ├─→ scripts/optimize_weekly.py           周度调仓
        ├─→ scripts/test_double_neutralization.py 二次中性化
        ├─→ scripts/screen_second_pass.py        二面筛选
        ├─→ scripts/analyze_param_sensitivity.py 阈值网格（25 组 L/H）
        ├─→ scripts/analyze_state_dependence.py  状态依赖（高低换手环境）
        └─→ scripts/analyze_time_stability.py    时间稳定性（月度 IC）
        ▼
research/results/l2_reproduction/mid_order_ratio/optimization_report.md（人工汇总）
```

## 2. 因子精确定义

| 项 | 口径 | 证据 |
|----|------|------|
| 单笔金额 amt | SSE：`ifNull(Amount, Price*Volume)` 且 `Type='T'`；SZSE：`Price*Volume` | `ch_tick.py:80-123` |
| 中单 | `40,000 < amt ≤ 200,000` 元（研报口径 L4w/H20w） | `ch_tick.py:23-25` |
| 小单 | `0 < amt ≤ 40,000`；大单 >20万；超大 >100万（预留） | 同上 |
| 因子值 | 每股票每日 `ΣMediumAmount / ΣTotalAmount` | `ch_tick.py:289-314` |
| 交易窗口 | `09:30:00 ≤ ExchTime < 15:00:01`（Asia/Shanghai） | `ch_tick.py:93-95` |
| 信号时间 | 窄表 `tradetime = 日期 + 09:30`；回测 `signal.shift(1)` → T-1 全日聚合预测 T 日 | `backtest.py:60-84` |
| 股票池 | CSI1000（`settings.UNIVERSE="000852.SH"`，DDB `get_stock_pool` 区间成分 union） | `config/settings.py:27-28`、`factor_builder.py:94-118` |
| 收益基准 | c2c 对 CSI1000 指数超额 | `Factor_Dev_Lib.get_Ret_Matrix:253-312` |
| 回测区间 | 主样本 2021-01-01 ~ 2024-06-30（继承 `factor_config`） | `factor_config.py:84-86` |

**经济含义（反向使用）**：中单占比高 → 未来收益低 → 生产信号为 `-mid_order_ratio`。

## 3. 链路文件清单

| 环节 | 文件 | 关键函数 |
|------|------|----------|
| CH 聚合 | `l2_factor_reproduction/python/ch_tick.py` | `fetch_tick_agg_by_date_range` / `fetch_tick_bucketed`（阈值网格用）/ `aggregate_wide_to_narrow` / `fetch_order_size_narrow` |
| 调度 | `l2_factor_reproduction/python/factor_builder.py` | `TICK_AGG_FACTORS={"mid_order_ratio","small_order_ratio"}`（`:41-46`）；`_build_tick_order_size_factor`（`:109-147`） |
| 落盘 | `l2_factor_reproduction/python/factor_runner.py` | `_save_narrow` → `research/results/l2_reproduction/<factor>/factor_narrow.parquet` |
| 回测 | `l2_factor_reproduction/python/backtest.py` | `prepare_factor_signal`（`:60-84`）、`compute_rank_ic`（`:86-88`）、`save_group_plots`（`:201-285`，生成 `cum_pnl.png`/`decile_bar.png`） |
| 入口 | `scripts/run_single_factor.py` / `run_all_factors.py` | CLI |
| 中性化 | `scripts/test_neutralization.py` + `Factor_Dev_Lib.panel_neutral_size_ind` | ind_cap / ind / cap |
| 二次中性化 | `scripts/test_double_neutralization.py` + `python/neutralization.py::neutralize_again` | ind_cap 后再剥离 momentum/volatility/turnover |
| 周度 | `scripts/optimize_weekly.py` | 每周首交易日采样 + ffill |
| 分析 | `scripts/analyze_param_sensitivity.py` / `analyze_state_dependence.py` / `analyze_time_stability.py` / `screen_second_pass.py` | — |
| 报告 | `research/results/l2_reproduction/mid_order_ratio/optimization_report.md` | 人工汇总 |

## 4. 评估口径（本 pipeline 实测）

- RankIC = 日频截面 Spearman（`signal_{t-1}` vs CSI1000 超额 `ret_t`）
- ICIR = mean/std × √250（**pandas std，ddof=1**，与主库 np.std ddof=0 略有差异）
- Sharpe = H-L 方向调整后 × √250，rf=0
- 换手 = H-L L1 倍数；成本 = 展示口径 `换手 × 7.5bps × 250`，"净年化" = 毛年化 − 隐含费（非逐笔实扣）
- `cum_pnl.png` = 日超额收益 cumsum

## 5. 实验结果（`optimization_report.md`）

| 变体 | RankIC | ICIR | H-L 年化 | Sharpe | MDD | 换手(倍) | 隐含费 | 净年化 |
|------|--------|------|----------|--------|-----|----------|--------|--------|
| 原始（日频） | -3.75% | -5.33 | 34.92% | 2.69 | -8.79% | 1.64 | 30.72% | +4.19% |
| 中性化 ind_cap（日频） | -3.18% | -7.34 | 30.42% | 3.81 | -5.52% | 1.64 | 30.83% | -0.41% |
| **原始 + 周度** | -3.20% | -4.64 | 31.69% | 2.56 | -8.56% | 0.40（周均≈2.01） | 7.54% | **+24.15%** |
| 中性化 + 周度 | -2.58% | -6.26 | 25.92% | 3.26 | -5.87% | 0.40（周均≈2.00） | 7.49% | +18.43% |

稳健性：
- **参数敏感性**：25 组 (L,H) 阈值 RankIC 全部落在 -3.3% ~ -4.3%；研报口径 L4w/H20w 为 RankIC -3.90% / ICIR -6.52 / Sharpe 3.01（`analysis/param_sensitivity/`）。
- **时间稳定性**：18 个月中 16 个月 IC 为负（88.9%）。
- **状态依赖**：高换手环境下 IC -3.71% / ICIR -6.50，占比 66.9%。
- **二次中性化**：剥离 momentum/volatility/turnover 后仍有残存 IC（`second_pass_screen_ind*.csv`）。

> ⚠️ **已知数据不一致**：`analysis/param_sensitivity/grid_results.csv` 中 L4w/H20w 行为 `RankIC=-3.155%, ICIR=-6.47, Sharpe=3.34, 净=-0.27%`，与报告"2026-08-04 补充"段落（-3.90%/-6.52/3.01/+3.47%）不完全一致，疑似对应不同样本或后续重跑。**写正式报告前须以脚本重跑确认采用哪组数字。**

## 6. 结论与当前状态

- 历史结论：`phase2_verdict.md`（2024-01 短窗 RankIC -4.17%）曾判"归档"；后续全样本+周度+中性化优化推翻短窗结论。
- **最新建议（optimization_report.md）**：入库为**反向低频信号** —— 生产因子 = `-mid_order_ratio`，**仅行业中性化（ind），周度调仓**，建议命名 **`order_flow_mid_reversal_weekly`**。
- 兄弟因子：`small_order_ratio` 代码就绪（同 CH 路径，`SmallAmount/TotalAmount`）但未跑全样本；`big_order_net_inflow` 是 Phase 1 分钟代理（RankIC -1.47%、净年化 -58.68%），已归档。

## 7. 复现命令

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# 基础日频计算 + 回测
$PY l2_factor_reproduction/scripts/run_single_factor.py \
    --factor mid_order_ratio --start 2021-01-01 --end 2024-06-30

# 中性化对比（ind_cap / ind / cap）
$PY l2_factor_reproduction/scripts/test_neutralization.py \
    --factor mid_order_ratio --neutral_types ind_cap ind cap

# 周度调仓（--raw = 不中性化）
$PY l2_factor_reproduction/scripts/optimize_weekly.py --factor mid_order_ratio --raw
$PY l2_factor_reproduction/scripts/optimize_weekly.py --factor mid_order_ratio

# 二次中性化 / 二面筛选
$PY l2_factor_reproduction/scripts/test_double_neutralization.py --factor mid_order_ratio
$PY l2_factor_reproduction/scripts/screen_second_pass.py --factor mid_order_ratio

# 参数敏感性（CH bucketed，一次查询扫 25 组阈值）
$PY l2_factor_reproduction/scripts/analyze_param_sensitivity.py \
    --start 2021-01-01 --end 2024-06-30

# 状态依赖 / 时间稳定性
$PY l2_factor_reproduction/scripts/analyze_state_dependence.py --factor mid_order_ratio
$PY l2_factor_reproduction/scripts/analyze_time_stability.py --factor mid_order_ratio

# 重画图
$PY l2_factor_reproduction/scripts/export_group_plots.py --factor mid_order_ratio
```

## 8. 结果目录结构

```text
research/results/l2_reproduction/mid_order_ratio/
├── factor_narrow.parquet          # 全样本窄表
├── rank_ic.csv / group_pnl.csv / group_turnover.csv / summary.json
├── cum_pnl.png / decile_bar.png
├── optimization_report.md         # ★人工汇总报告（结论以此为准）
├── neutralization_comparison.csv
├── neutralized/ neutralized_ind/ neutralized_cap/ neutralized_ind_cap/
├── weekly_raw/ weekly_neutralized/
├── double_neutralized_ind_cap/
├── second_pass_screen_ind.csv / second_pass_screen_ind_cap.csv
└── analysis/
    ├── param_sensitivity/         # grid_results.csv / top5.csv / heatmap_*.png / 分季 chunks
    ├── state_dependence/          # group_ic_daily.csv / ic_boxplot|rolling|series.png / summary.csv
    └── time_stability/            # 月度 IC 系列图
```
