# 回测框架与指标口径（Backtest Framework）

> 本文解决一个核心问题：**同名指标在不同 pipeline 口径不同**。写报告前必须对照本文标注口径。
> 生成时点：2026-08-04。所有结论附 file:line 证据。

---

## 1. 三条回测路径总览

| 路径 | 入口 | 收益口径 | 组合 | 典型产物位置 |
|------|------|----------|------|--------------|
| **EOD classic**（主力） | `Factor_Test_Process.py` → `factor_runner.run_eod_batch` | 日频 c2c/o2o/v2v；`base_index` 非空时为指数超额 | 十分组等权 + H-L | `result/<track>/`、`research/results/` |
| **L2 reproduction** | `l2_factor_reproduction/python/backtest.py` | 日频 c2c **对 CSI1000 超额**（`base_index=UNIVERSE`） | 同一 `groupTest` | `research/results/l2_reproduction/<factor>/` |
| **Intraday v2** | `core/evaluation/intraday_metrics.py` 等 | 分钟 Ret_15/30/.../EOD；**exact constituent EW** 基准 | 分组超额 + H-L | `research/results/intraday_evaluation_v2/` |

三条路径**指标不可直接混排**，差异见 §6。

## 2. 评估流水线顺序（EOD classic / L2 共用）

```text
factor 宽表 (Date × Symbol)
  → universe mask（CSI300/500/1000 权重表；ALL = 不加 mask，留 0/3/6 开头）
  → 可交易过滤：非涨跌停 × 非ST/退 × 非停牌
  → signal.shift(1)          ← 用 T-1 信号预测 T 日收益，防看穿
  → 截面 rank → npqcut 十分组 → 组内等权
  → group_pnl / H-L / turnover / RankIC / Sharpe ...
```

证据：`factor_runner.py:161-180`（mask+shift）；`Factor_Dev_Lib.py:543-575`（分组与 H-L）。
**中性化不在默认链路内**：`factor_runner.prepare_signal` 不做中性化；需要时在 builder 内或单独脚本（如 `test_neutralization.py`）先中性化再进同一流程。L2 链路顺序 = 中性化 → 过滤 → shift → rank/group（`test_neutralization.py:123-136`）。

## 3. 指标精确定义

### 3.1 IC / RankIC

- **本项目 "IC" 一律是截面 Spearman RankIC**，不是 Pearson：
  `signal.corrwith(ret, axis=1, method='spearman')`（`Factor_Dev_Lib.py:588`；`calICIR` 同款 `:389-405`；L2 版 `l2_factor_reproduction/python/backtest.py:86-88`）。
- 方向：`factor_{t-1}` vs `ret_t`（已 shift）。
- `ret` 默认 c2c（`S_DQ_CLOSE/S_DQ_PRECLOSE-1`，`Factor_Dev_Lib.py:253-312`）；`base_index` 非空时逐日减指数收益 → **超额 RankIC**。

### 3.2 ICIR

- 公式：`mean(IC) / std(IC) × √250`（年化）。
- ⚠️ **std 的 ddof 不统一**：`Factor_Dev_Lib`/`factor_runner` 用 `np.std`（ddof=0）；`l2_factor_reproduction/python/backtest.py` 用 pandas `.std()`（ddof=1）；`core/evaluation/intraday_metrics.py:248-266` 显式 ddof=1。样本大时差异小，写报告建议注明。
- legacy intraday DDB 端 `IC_IR = mean/std` **未年化**（`intraday_lib.py:306-309`），Python 侧再乘 √250。

### 3.3 分组与 H-L

- 每日截面 `_rank_to_bins_npqcut(signal, n=10)`，组内等权；`H-L = G10 - G1`，日频逐日换仓（`Factor_Dev_Lib.py:543-575`）。
- 周度变体：每周首个交易日采样信号，周内 ffill，仍用日收益和 √250 年化（`optimize_weekly.py:79-99`）。

### 3.4 Sharpe —— 三种不同口径 ⚠️

| 口径 | 定义 | 出现位置 |
|------|------|----------|
| **H-L Sharpe（方向调整）** | `mean(H-L)/std(H-L)×√250`，rf=0；**若 H-L 均值为负则乘 -1 后再算**（展示永远为正） | `Factor_Dev_Lib.py:316-335`（`calSharpe`）、`:577-583`（direction flip）。EOD classic 与 L2 报告标题里的 "Sharpe" 都是这个 |
| **G10 Excess Sharpe（delivery 冻结口径）** | `r_x,t = r_G10,t − r_EW,t`（r_EW = 当日**有效样本**等权均值，非指数）；`mean(r_x)/std(r_x)×√250` | `research_delivery/METRICS_G10_EXCESS.md:16-30`；`Factor_Dev_Lib.g10_excess_vs_universe_ew:990` |
| **执行层 Net Sharpe** | `net = direction×gross − round_trip_cost×turnover`，`DEFAULT_ROUND_TRIP_COST=0.0015`（印花税 0.1% 卖出 + 双边佣金 ~0.05%） | `alpha_investability.py:14-16`、`execution_layer.py` |

**写报告时必须显式标注是哪一种。** 经典 `groupTest` 报告的 Sharpe = 第 1 种；delivery 卡片 headline = 第 2 种；执行/可投资性评估 = 第 3 种。

### 3.5 年化收益与 MDD

- `calAnnuRet(retSeries, n=250)`（`Factor_Dev_Lib.py:376`）；日频年化固定 250 天。
- MDD 基于累计收益序列（`calMDD`）。
- `cum_pnl.png` = **日收益简单累加（cumsum），非复利 NAV**（`factor_runner.py:237-255`）。raw 还是超额取决于 `ret_matrix` 是否传 `base_index`——**L2 复现的 cum_pnl 全部是 CSI1000 超额**。

### 3.6 换手与成本 ⚠️

- 组内换手：相邻日权重 L1 变化求和（首日=建仓权重和）；`H-L turnover = TO_G10 + TO_G1`（**倍数口径**，不是 %）（`Factor_Dev_Lib.py:561-575`）。
- **默认 `groupTest(fee=0)` 不实扣成本**。报告里 "Implied AnnuFee" 是展示口径：
  `H-L 日均换手 × 7.5bps × 250`（`IMPLIED_ANNU_FEE_BPS=7.5`，`Factor_Dev_Lib.py:480-488`）。
  "净年化" 多数是 `毛年化 − Implied AnnuFee`，**不是逐笔实扣 NAV**。
- 实扣仅在执行层（`alpha_investability` / `execution_layer`，round-trip 15bps）。

## 4. 股票池与过滤口径

| 项 | 口径 | 证据 |
|----|------|------|
| CSI300/500/1000 | Wind DDB 日频权重表 point-in-time mask | `Factor_Dev_Lib.py:877-911` |
| 其他指数 | `AINDEXMEMBERS` 入/剔日期展开 | `:913-955` |
| ALL | 不加指数 mask，A 股 0/3/6 开头 | `factor_config.py:241-247` |
| 涨跌停 | `CLOSE < LIMIT and CLOSE > STOPPING` | `:109-128` |
| ST/退 | 历史名称表 `ST\|退` | `:131-219` |
| 停牌 | `S_DQ_TRADESTATUS` | `:222-250` |
| IPO/次新 | 默认**不**过滤；investability 层 ≥60 个有效交易日 | `alpha_investability.py:20-54` |

## 5. 中性化口径

- **行业+市值中性化**（主口径）：逐日截面 OLS，`factor ~ const + 中信一级行业 dummies + log(S_VAL_MV 总市值)`，取残差。`nt_type ∈ {ind_cap, ind, cap}`（`Factor_Dev_Lib.panel_neutral_size_ind:748-835`）。市值先 log，再 MAD 去极值 + z-score；**输入因子本身默认不做去极值/标准化**（函数注释明示，`:779-789`）。
- **行业均值剥离**（轻量）：`industry_neutral.py` 按行业 demean（样本 <30 跳过）。
- **二次中性化（double）**：第一步 ind_cap 残差 → 第二步再对 `momentum(20d 收益和) / volatility(20d 收益std) / turnover(20d S_DQ_TURN 均值取 log)` 三个风格做截面 `lstsq` 残差化（`l2_factor_reproduction/scripts/test_double_neutralization.py:72-110`、`python/neutralization.py:16-58`）。
- 时机：**中性化在 rank/分组之前**，过滤与 shift 在中性化之后（L2 链路）。

## 6. 三路径指标对比表（混用红线）

| 维度 | EOD classic | L2 reproduction | Intraday v2 / delivery |
|------|-------------|-----------------|------------------------|
| 收益 | c2c（可指数超额，取决于 base_index） | c2c **恒为 CSI1000 超额** | 分钟 horizon；exact EW 基准 |
| IC | Spearman，ddof=0 系 | Spearman，ddof=1 系 | Spearman，ddof=1 |
| Sharpe | H-L 方向调整 | H-L 方向调整 | `group_excess_sharpe` / `hl_sharpe` / `g10_excess_sharpe` 分列 |
| 成本 | 7.5bps 展示口径 | 7.5bps 展示口径 | `intraday_portfolio_cost_v1.md` 专门口径 |
| 年化 | 250 | 250 | 250 |

**建议报告字段命名**（防混淆，采纳自侦察结论）：
`rank_ic_spearman` / `icir_annualized_250` / `hl_sharpe_gross_directional` / `g10_excess_sharpe_exact_ew` / `net_annu_after_implied_fee_7p5bps` / `net_sharpe_rt15bp`。

## 7. 结果落盘约定

| 产物 | 内容 |
|------|------|
| `factor_narrow.parquet` | 窄表 `symbol/tradetime/factorname/value` |
| `group_pnl.csv` / `group_turnover.csv` | 各组+H-L 日收益 / 换手 |
| `rank_ic.csv` | 逐日 RankIC |
| `summary.json` | 汇总指标 |
| `cum_pnl.png` / `decile_bar.png` | 累计收益曲线（cumsum）/ 分组年化柱图 |
| `research/results/factor_run_manifest_<track>.csv` | 批量运行台账（SKIP_COMPLETED 断点续跑依据） |
