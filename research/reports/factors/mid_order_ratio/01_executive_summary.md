# 01 — Executive Summary

> **Research status:** Completed factor construction audit, standalone predictive validation, robustness testing, and exposure diagnostics.

## 1. Factor definition

**mid_order_ratio measures the fraction of daily traded amount contributed by trades with transaction amount between RMB 40k and RMB 200k.**

对股票 \(i\) 和交易日 \(t\)，原始因子定义为：

\[
\mathrm{MidOrderRatio}_{i,t}
=
\frac{
\sum_k a_{i,t,k}\mathbf{1}(40{,}000<a_{i,t,k}\le 200{,}000)
}{
\sum_k a_{i,t,k}
}
\]

其中 \(a_{i,t,k}\) 是严格识别后的第 \(k\) 笔成交金额。原始因子越高，下一交易日收益排序平均越低，因此用于分组展示的有效方向为 `-mid_order_ratio`。

4 万/20 万是本次样本验证前已冻结的既有研究口径，不是从当前样本挑选的最优参数。它用于刻画小额成交与 20 万元以上较大成交之间的中等成交规模构成；该区间在 CSI1000 PIT 样本中约占总成交额 30.2%，具有足够统计质量。金额阈值不识别交易者身份，不能机械解释为“散户”或“机构”标签。

## 2. Research question and scope

本报告只回答一个问题：

> `mid_order_ratio` 是否具有统计显著、稳健且可作经济解释的**独立单因子预测关系**？

十分组与 H-L 仅用于检验横截面排序、尾部分离和路径稳定性。本报告不讨论因子组合、权重配置、优化或生产化。

## 3. Research universe

| Item | Research specification |
|---|---|
| Sample period | 2023-01-04 to 2024-06-28；358 个有效交易日 |
| Headline universe | CSI1000 Wind point-in-time constituents；日均 990 只有效股票 |
| Robustness universes | SSE/SZSE A-share（不含北交所）、CSI300、CSI500、CSI1000 |
| Signal lag | 1 个交易日；T-1 收盘后完整因子解释 T 日收益 |
| Return horizon | 下一交易日 close-to-close return |
| Factor data | ClickHouse SSE/SZSE Level-2 Tick，严格成交筛选 |
| Return and reference data | DolphinDB Wind 日收益、指数成分、行业、市值与可交易状态 |

## 4. Table 1 — Standalone Factor Performance Summary

| Metric | Result | Interpretation |
|---|---:|---|
| RankIC | Raw `mid_order_ratio`: **-4.80%**；effective `-mid_order_ratio`: **+4.80% equivalent** | 原始因子越高，次日收益排名越低 |
| ICIR | Raw: **-6.33**；effective: **+6.33 equivalent** | 负号只表示原始因子方向；取反后稳定性为正向等价值 |
| IC t-stat | Raw: **-7.58**；effective: **+7.58 equivalent** | 样本内均值显著偏离 0；未做 Newey-West 调整 |
| IC negative day ratio | **67.9%** | 原始方向多数交易日为负，不由少数极端日单独贡献 |
| Effective H-L Sharpe | **2.74 gross** | `-mid_order_ratio` 十分组两端差的独立排序诊断，`fee=0` |
| Effective H-L MDD | **-9.68%** | 基于 H-L 日收益复利路径 |
| Effective H-L turnover | **1.69 per day** | 两端权重变化较高；交易成本尚未建模 |

### Raw direction versus effective direction

```text
Raw factor:       mid_order_ratio       RankIC = -4.80%
Effective signal: -mid_order_ratio      RankIC = +4.80% equivalent
```

ICIR、t-stat 和 IC 日占比遵循同一符号转换。报告保留 raw 结果以忠实表达经济关系，同时使用 effective direction 绘制从 G1 到 G10 的排序图，避免把“负 ICIR”误读为统计失效。

## 5. Construction audit finding

研究价值首先来自构造审计，而不是任何单一 Sharpe 数字。审计发现旧 Tick 查询存在两个实质问题：

1. 多日查询没有对每个交易日逐行施加常规交易时段条件；
2. 深市仅使用 `Type='011'`，未同时要求有效的买卖订单号来识别真实成交。

正式结果全部基于修正后的 strict-trade cache：

```text
SSE:  Type = 'T'
SZSE: Type = '011' AND BidOrderNo > 0 AND AskOrderNo > 0
Time: 09:30:00 <= ExchTime < 15:00:01 on every date
```

旧面板与严格面板的 Spearman 仅为 **0.7996**，因此旧结果未被沿用。严格 cache SHA256：

```text
ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f
```

## 6. Evidence summary

- **Statistical prediction:** CSI1000 raw RankIC -4.80%，t-stat -7.58，16 / 18 个月平均 IC 为负；
- **Cross-universe robustness:** SSE/SZSE A-share、CSI300、CSI500、CSI1000 的 raw RankIC 均为负；
- **Parameter robustness:** 25 个合理阈值单元全部保持负 IC，未按样本内最高 Sharpe 改写定义；
- **Exposure diagnostics:** 行业、市值以及行业+市值残差的 RankIC 仍为负，预测关系不能被这些暴露完全解释；
- **State dependence:** 高换手状态中信号更强，说明强度随交易活跃环境变化；
- **Economic interpretation:** 结果与短期流动性压力、临时价格冲击等候选机制相容，但尚未验证主动方向与价格路径，不构成因果或反转识别。

前五页 evidence brief 依次展示：

1. factor construction pipeline；
2. universe comparison；
3. CSI1000 index-excess decile diagnostic；
4. daily / rolling / monthly IC stability；
5. industry and size exposure diagnostic。

## 7. Research verdict

| Validation question | Verdict | Evidence boundary |
|---|---|---|
| 1. Correctly constructed? | **Yes, after material query corrections** | 严格交易所筛选、逐日时段过滤、冻结 cache、单元测试和数据血缘均可审计 |
| 2. Statistically predictive? | **Yes, in the observed sample** | RankIC、ICIR、t-stat、IC 日占比和 decile 尾部分离一致 |
| 3. Robust across reasonable tests? | **Yes, within the available sample** | 四股票池、25 参数单元、时间与状态检验方向总体一致 |
| 4. Economically interpretable? | **Plausible, not proven** | 与短期流动性压力和临时价格冲击一致；成交金额不能识别交易者身份 |
| 5. Worth further research? | **Yes — research candidate** | 需要更长历史、样本外时期、更多 horizon 和方向性订单流验证 |

综合判断：

> 本研究构造了一个具有可审计数据血缘的 L2 成交规模因子。该因子在当前样本中表现出统计显著的独立横截面预测关系，经多项合理稳健性检验后方向仍一致，并具有合理但未被证明的经济解释。它适合作为 candidate alpha feature 继续验证，不代表已完成长期样本外验证。

后续组合集成不在本报告范围内。
