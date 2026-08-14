# 07 — Limitations and Future Research

## 7.1 Sample period length

主样本为 2023-01 至 2024-06，共 358 个有效交易日和 18 个自然月。月度与滚动检验可以排除“单月完全驱动”，但不能覆盖完整市场周期。

当前尤其缺少：

- 更早年份的不同波动与流动性环境；
- 重大制度变化前后的统一 Tick schema 验证；
- 2024-07 之后的真正样本外观察；
- 多年连续的年度稳定性结果。

25 单元参数网格与主结果使用同一历史样本，因此参数邻域稳定不是样本外验证。

## 7.2 No long historical validation

严格成交定义要求交易所 Tick 字段在历史上保持可解释。深市 `Type`、`BidOrderNo`、`AskOrderNo` 以及上交所 `Amount` 的长期 schema 一致性尚未在多年范围内完成逐年核验。

当前 artifact key `ALL` 只覆盖 SSE/SZSE A 股代码，不含北交所；因此不能把该结果外推为包含 BSE 的完整全市场结论。

在延长样本前，需要先确认：

1. 每年真实成交的识别规则；
2. Price、Volume、Amount 的单位与缺失处理；
3. 股票、ETF、可转债代码边界；
4. 两交易所因子分布是否存在 feed-driven break。

## 7.3 Transaction costs are not modeled

headline H-L Sharpe 为 `fee=0` gross，日均 turnover 为 1.69。手续费、滑点、市场冲击、涨跌停和成交量约束均未建模。

该 turnover 是两端组合绝对权重变化之和，未除以 2；它不能直接读成“169% 的股票被替换”。按 7.5 bps / 单位换手机械估算的 31.66% 年化负担只能作为量纲敏感性锚点，不是成本后业绩。

因此：

- H-L 只用于 standalone sorting diagnostic；
- gross Sharpe 不能解释为成本后结果；
- 本报告不提供交易频率、资金规模或权重结论。

交易成本缺失不改变 RankIC 的统计定义，但限制对 H-L 路径的经济外推。

## 7.4 No causal identification

逐笔成交金额不能识别：

- 账户类型；
- 同一订单是否拆分；
- 主动买卖方向；
- 真实交易动机；
- 价格变化是永久冲击还是临时冲击。

因此“高中单占比导致次日下跌”不是当前证据支持的表述。当前结果只证明样本内条件相关和排序预测关系。

## 7.5 Statistical inference limits

当前 t-stat 使用普通日度均值标准误，未做 Newey-West 自相关调整；参数、universe、状态和 exposure diagnostics 共检验了多个切片。

这些切片用于检验合理稳健性，而不是选择最高结果，但 multiple-testing 风险仍存在。后续应预注册新的样本范围与主指标，并增加：

- Newey-West t-stat；
- block bootstrap confidence interval；
- 年度分层结果；
- 明确的样本外冻结窗口。

## 7.6 Data extraction and reproducibility limits

正式结果统一使用 strict-trade cache，记录了日期、行数、交易所规则和 SHA256。旧 `factor_narrow.parquet` 与严格面板的 Spearman 只有 0.7996，已明确标为 legacy evidence。

仍需持续记录：

- query hash；
- 源表分区版本；
- extraction timestamp；
- construction code commit；
- 每年每交易所 row count 与 checksum。

## 7.7 Future research

### Extend the sample

1. 将严格口径样本延伸至 2024-07 之后；
2. 在 schema 可核验前提下尽可能回溯至 2019 或更早；
3. 按年份和重大市场阶段报告 raw RankIC 与方向一致日占比；
4. 预注册扩展样本，不根据新结果修改 4万/20万定义。

### Test longer return horizons

检验 T+1、T+2、T+5、T+10 和 T+20 RankIC decay，区分：

- 一日反转；
- 多日缓慢修复；
- 短期关系是否在更长 horizon 消失或反向。

### Active buy/sell decomposition

在交易所字段语义可核验后，使用 `BSFlag`、`BidOrderNo` 和 `AskOrderNo` 区分：

- active-buy medium trades；
- active-sell medium trades；
- directionally signed medium-amount flow。

这可以检验负向关系是否主要来自买方冲击后的回撤、卖方压力后的修复，或无方向的成交规模组成。

### Order-flow direction and price path

进一步分解：

- 当日价格冲击；
- overnight gap；
- 次日开盘至收盘；
- 次日盘中反转；
- 极端 `mid_order_ratio` 事件窗口。

该研究可以提高机制辨识度，但不会改变本报告的 standalone factor definition。

## 7.8 Final boundary

当前证据支持：

> 构造经严格审计、在当前样本中具有统计显著预测关系、通过合理稳健性检验，并具有可解释但未被证明的经济机制。

当前证据不支持：

- 已完成长期历史验证；
- 已识别因果机制；
- 已获得成本后结论；
- 已完成充分样本外验证。

因此，`mid_order_ratio` 的合理定位是 **research candidate / candidate alpha feature**。值得继续进行更长历史、样本外、更多收益 horizon 和方向性订单流验证。Further portfolio integration is outside the scope of this report.
