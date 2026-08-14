# Execution Validation Layer v1

**Date:** 2026-08-06
**Status:** ADOPTED (L2 Candidate Pool track)
**Scope:** methodology contract for rebalance-frequency / holding-period
validation. Research rules only — no production admission.

---

## 0. 正式表述（frozen wording）

> **Daily baseline is a signal-discovery standard, not a production
> rebalance mandate.**

所有候选首先使用日频统一口径评价信号质量、衰减与原始执行压力。对于具有
稳定预测能力但受交易成本约束的候选，**降低换仓频率、延长持有期和使用重叠
持有组合，是第一优先级的换手控制手段**。调仓频率必须根据固定期限的
IC decay、毛收益保留率、换手降低率和费后收益进行验证，不得仅依据全样本
最优结果选择。

注意边界：

- **不是**"所有高换手因子都改成周度调仓"。
- 日频 baseline 继续保留，是所有因子横向比较（IC / ICIR / H-L Sharpe /
  G10 Excess Sharpe / 原始换手 / 原始成本墙）的统一基准；不得把统一候选池
  整体改为周频，否则 baseline 不再可比。
- 生产调仓频率是**单独的执行参数**，不属于因子公式本身。

## 1. 动机证据（本池已有结果）

- `mid_order_ratio`：日频毛年化 ≈34.9%、年化成本 ≈30.7%、净年化 ≈4.2%；
  周度毛年化 ≈31.7%、成本降至 ≈7.5%、净年化 ≈24.2%。成本下降 > 毛收益
  损失，属慢衰减因子。
- Price Formation 族：`close_auction_return`（H-L Sharpe 13.04，日均换手
  3.34，费后 ≈0.76%）、`closing_30m_return`（Sharpe 6.59，换手 3.39，
  费后 ≈19.1%）、`tail_return_share`（Sharpe 4.60，换手 2.59，费后
  ≈6.0%）——毛 Alpha 强、执行频率过高、成本吞噬收益。

结论：当前瓶颈已从"信号强度"转为"执行频率与成本墙"，调仓频率必须进入
标准验证框架。

## 2. 管线位置（更新后）

```
Liquidity / Impact
        ↓
Global taxonomy
        ↓
Exposure audit
        ↓
Preliminary quality screen
        ↓
Execution validation          <-- 本层
  - IC decay
  - 1D / 3D / 5D / 10D 固定持有期
  - staggered holding
  - turnover / cost / net alpha
        ↓
Incremental alpha test
        ↓
A / B / C candidate tiers
```

## 3. 冻结测试网格（防参数挖掘）

### 3.1 IC decay

因子对 r(t+1), r(t+2), r(t+3), r(t+5), r(t+10) 的截面 RankIC，
输出 IC_1d/2d/3d/5d/10d 与

  ICRetention_h = |IC_h| / |IC_1|

### 3.2 持有期

只允许 {1D, 3D, 5D, 10D}。禁止扫描 2..9 连续网格。

每个持有期输出：毛年化、净年化、Sharpe、最大回撤、日均换手、年化成本、
年度稳定性，以及三个核心比率：

  GrossRetention_h     = GrossReturn_h / GrossReturn_1D
  TurnoverReduction_h  = 1 − Turnover_h / Turnover_1D
  NetImprovement_h     = NetReturn_h − NetReturn_1D

### 3.3 Staggered portfolios（默认构造，避免星期效应）

禁止只测"每周一换仓"（引入星期偏差、忽略周中信号）。h 日持有使用 h 个
重叠 sleeve：每日建立一个持有 h 天的子组合，最终仓位为 h 个 sleeve 平均：

  P_t^(h) = (1/h) * Σ_{j=0..h-1} P_{t−j}^(signal)

效果：每天使用最新信号、每天只调整约 1/h 仓位、换手显著下降、无调仓日
星期偏差，更接近真实分批建仓。

## 4. 机制先验（待验证，不是结论）

| 因子 | 先验执行方式 | 理由 |
|---|---|---|
| mid_order_ratio | 5D/周度 | 周度已证明保留大部分毛收益 |
| small_order_ratio | 3D/5D | 订单规模结构非纯单日冲击 |
| intraday_amihud | 3D/5D | 原始换手低，信号可能较慢 |
| range_per_amount | 3D/5D | 流动性状态有持续性 |
| close_auction_return | 1D 或短重叠 | 很可能是短期竞价反转 |
| closing_30m_return | 1D/3D 都测 | 可能兼有短反转与持续效应 |
| realized_volatility | 5D/10D | 波动率状态持续 |
| minute_return_autocorr1 | 1D/3D | 日内路径信息衰减可能较快 |

先验只决定测试顺序；最终结论必须来自 §3 冻结网格。

## 5. 执行验证的准入对象（不全量跑）

110+ 公式不全部进入执行验证。优先：

1. 每个全局相关簇（|Spearman| ≥ 0.80 簇）的代表；
2. 原始质量达最低门槛的候选；
3. 毛收益强但被成本墙阻挡的候选；
4. G10 或 H-L Sharpe 较强的候选。

预计先压缩到约 30–50 个簇代表，再做调仓频率测试。

## 6. 边界

- 本层输出是"执行参数证据"，仍不作正式 KEEP/DROP、生产晋级或组合结论。
- 禁止依据全样本最优持有期直接定生产参数。
- 统一 Candidate Pool baseline 保持日频 T+1 不变。
