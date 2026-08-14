# 08 — Research Decision Framework

本章把前述结果转化为 research-grade single factor library 的证据门。它不是把所有指标压成一个总分，而是要求关键失败项不能被其他亮眼指标抵消。

## 8.1 Decision tree

```text
Factor definition frozen before evaluation?
  no  -> redesign definition; do not compare performance
  yes
   |
   v
Raw-data semantics and construction audit passed?
  no  -> stop; rebuild data and invalidate downstream results
  yes
   |
   v
T-1 alignment and point-in-time universe verified?
  no  -> stop; remove look-ahead / survivorship contamination
  yes
   |
   v
Standalone IC significant and direction stable?
  no  -> archive or reformulate hypothesis
  yes
   |
   v
Cross-universe, parameter-neighborhood and time checks survive?
  no  -> retain only as conditional / regime-specific evidence
  yes
   |
   v
Signal survives pre-specified exposure residual tests?
  no  -> relabel as known exposure proxy
  yes
   |
   v
Turnover, cost and capacity acceptable for intended use?
  no  -> research feature only; no tradability claim
  yes
   |
   v
Frozen out-of-sample validation passed?
  no  -> research candidate
  yes -> validated candidate for portfolio-level evaluation
```

“stop” 表示不能继续把后续回测数字当作该因子的有效证据，不表示永久删除研究想法。例如本项目发现 legacy Tick 构造失败后，正确动作是重建 strict cache 并从头重算，而不是用后续 Sharpe 为错误输入辩护。

## 8.2 Current gate status

| Evidence gate | Current status | Decision basis |
|---|---|---|
| Frozen definition | **Pass** | `(4万,20万]` 是事前既有公式；未改用样本内 Sharpe 最高网格 |
| Data semantics and construction | **Pass for current sample** | strict SSE/SZSE execution filters、逐日时段条件、单测、checksum；legacy 结果作废 |
| Timing and PIT universe | **Pass for current design** | 全天因子 shift(1)；历史日度成员在信号形成日应用 |
| Standalone prediction | **Pass in sample** | raw RankIC -4.80%，t-stat -7.58，67.9% 日期方向一致 |
| Decile structure | **Conditional pass** | 聚合单调性 0.879、两端分离清晰；中间组不严格单调 |
| Robustness | **Pass within available sample** | 四股票池同方向、25 阈值单元同方向、16/18 个月负 IC |
| Exposure diagnostics | **Pass as residual test** | 行业+市值后保留 92% 绝对 IC；matched sample 删除 turnover 后仍保留约 49% |
| Economic mechanism | **Plausible, unverified** | 缺少主动方向、当日冲击和收益路径，不能识别因果或反转 |
| Cost and capacity | **Not passed** | H-L turnover 1.69；headline 为 fee=0；7.5 bps 机械成本锚点约 31.66%/年 |
| Frozen out-of-sample | **Not passed** | 正式 strict 样本止于 2024-06，尚无冻结后的新窗口 |

当前决策不是“可交易 Alpha”，而是：

> **Retain as a research candidate / candidate alpha feature.** 构造、时序、样本内预测与合理稳健性证据已通过；成本、容量、长期 schema 和冻结样本外门尚未通过。

## 8.3 Library metadata that must travel with the factor

为了避免只保留一个因子值列而丢失研究语义，library entry 至少应保存：

1. raw name 与公式：`mid_order_ratio = amount(4w,20w] / total_amount`；
2. 阈值边界：lower exclusive、upper inclusive；
3. raw/effective direction：raw RankIC 为负，`effective_direction=-1`；
4. 数据语义：交易所表、成交识别、金额字段、常规时段和时区；
5. availability：完整日因子在收盘后已知，验证使用 `shift(1)`；
6. universe：PIT 成分来源及成员掩码应用日期；
7. headline metric definitions：RankIC、ICIR、decile、H-L、turnover、benchmark 和年化方式；
8. evidence version：样本区间、strict cache SHA256、代码版本和 artifact 路径；
9. known limits：短样本、无完整成本模型、无因果识别、无正式样本外；
10. promotion status：`research_candidate`，不得静默标记为 production-ready。

## 8.4 Next decision-changing experiments

后续实验应优先关闭未通过的门，而不是继续增加同一样本内的切片：

1. 冻结公式、方向、中性化方法和主指标，扩展 2024-07 之后的样本外窗口；
2. 逐年审计 SSE/SZSE schema 后扩展多年 strict history；
3. 报告 T+1 至 T+20 decay、Newey-West 与 block-bootstrap 区间；
4. 使用可执行的换仓频率、双边费用、滑点、冲击和容量约束；
5. 加入主动买卖方向和日内价格路径，区分临时冲击、延续与反转。

只有这些结果通过后，才进入组合相关性、边际贡献和投资组合约束评估；组合集成不属于本报告的 standalone factor verdict。
