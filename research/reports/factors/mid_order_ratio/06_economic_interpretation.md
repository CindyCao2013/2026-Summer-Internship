# 06 — Economic Interpretation

## 6.1 Observed fact

本研究直接观察到：

> **Higher `mid_order_ratio` predicts lower next-day cross-sectional return rank.**

在 CSI1000 PIT 样本中，raw RankIC 为 -4.80%；SSE/SZSE A-share（不含北交所）、CSI300、CSI500 和 CSI1000 的方向一致。最高 raw `mid_order_ratio` 的有效 G1 组指数超额年化为 -30.07%，最低 raw `mid_order_ratio` 的有效 G10 组为 +11.05%。

这是统计关系，不是已识别的因果链条。

## 6.2 What the factor measures

`mid_order_ratio` 描述一只股票当天总成交额中，单笔金额位于人民币 `(4万, 20万]` 的成交占比。它观测的是**transaction-size composition**。

它不直接观测：

- 主动买入或主动卖出；
- 订单簿深度；
- 账户身份；
- 交易动机；
- 同一订单是否被拆分。

**Transaction size does not identify investor type.** 大型账户可以拆单，小型账户也可能产生较大成交，因此不能把该金额区间机械对应到任何交易者群体。

## 6.3 Possible explanation 1 — Short-term liquidity pressure

如果中等金额成交集中出现，它可能伴随一段方向一致的流动性需求。流动性供给者吸收需求时，价格可能暂时偏离短期均衡。当前证据链应画成：

```text
Observed:
mid-order amount share high
  ------------------------------------> next-day return rank lower

Hypothesized mechanism:
mid-order amount share high
  -> concentrated signed execution flow
  -> same-day temporary price pressure
  -> next-day partial reversal

Missing measurements:
active buy/sell direction + intraday price path + overnight/open-to-close decomposition
```

图中只有第一条关系被当前 RankIC 直接观测，下面三步均待验证。尤其“反转”要求先证明高占比伴随同方向的当日价格冲击：若主要是买方临时冲击，次日负收益可以与回撤一致；若主要是卖方临时冲击，标准反转机制反而预期次日正收益。因子未使用主动买卖方向或当日价格路径，所以当前报告不能确认压力方向，也不能仅凭负 IC 把关系命名为 reversal。

## 6.4 Possible explanation 2 — Temporary price impact

成交规模构成可能反映当日冲击的执行方式。若中等金额成交更频繁地伴随短期、非基本面驱动的交易，高占比可能对应更强的临时价格影响，并在下一日出现均值回归。

该机制可解释：

- raw factor 与下一日收益的负向关系；
- 信号在高换手状态中更强；
- 分组区分主要集中于两端而非完全线性。

但当前验证没有分离当日永久冲击与临时冲击，也没有检验高占比与当日收益方向的条件关系，更没有分解隔夜和次日盘中路径。因此这里只能称为待检验的候选解释；信息延续、风险补偿或遗漏状态变量也可能产生相同的负 RankIC。

## 6.5 Possible explanation 3 — Transaction-size composition

`mid_order_ratio` 不是总成交额或平均单笔金额，而是成交金额分布中一个区间的质量占比。相同日成交额可以由不同的逐笔金额构成，这种组成差异可能对应不同的短期交易环境。

状态检验显示：

| Lagged turnover state | Raw RankIC |
|---|---:|
| Low | -1.35% |
| Mid | -2.70% |
| High | **-5.60%** |

在共同的 332 日支持集上，行业+规模 matched baseline 的 RankIC 为 -4.76%，进一步删除 turnover 后为 -2.33%，保留约 49%。两项证据共同说明，交易活跃度解释了部分关系，但成交规模构成本身在该线性检验后仍保留负向预测关系。

## 6.6 Why the effect is stronger in the tails

CSI1000 decile 的中间组不是严格单调，而最高 raw factor 端的负超额最明显。一种保守解释是：只有当成交规模构成显著偏离常态时，其短期信息才足够强；中间区域包含更多相互抵消的交易动机。

这仍是统计描述。要验证该假说，需要个股历史分位标准化、极端事件研究以及主动买卖方向分解。

## 6.7 Most defensible interpretation

> `mid_order_ratio` 衡量中等金额逐笔成交在总成交额中的参与强度。历史样本中，高中等金额成交占比与较低下一交易日收益排序相关，且关系在高交易活跃状态中更强。短期流动性压力和临时价格影响是与部分证据相容、但尚未完成方向与价格路径验证的候选机制；成交金额不能识别交易者身份，当前实证关系既不构成因果证明，也不足以单独证明反转。

应避免：

- 把中等金额成交直接等同于特定账户类型；
- 声称因子证明了知情交易或市场操纵；
- 把平均统计关系表述为每日确定规律；
- 从历史相关性直接推出交易指令。
