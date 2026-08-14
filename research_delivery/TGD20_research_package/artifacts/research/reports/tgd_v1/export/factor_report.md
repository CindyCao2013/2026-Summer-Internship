# 日内分钟收益率的时序特征：逻辑讨论与因子增强

## —— TGD20 因子研究报告（研报复现与机制验证）

| 项目 | 内容 |
|------|------|
| **报告主题** | 日内分钟收益率时序特征（Temporal Growth / Timing Deviation） |
| **核心因子** | `TGD20` |
| **信息类别** | Temporal Information（时序信息层） |
| **文档性质** | 单因子研究级报告（Phase 1）；公式冻结；非 Production 上线文档 |
| **主验证样本** | Confirmation：2022-01-28 → 2025-12-31（951 交易日） |
| **稳定性样本** | 2020–2025 年分年 RankIC |
| **股票池** | 全 A（0/3/6） |
| **组合标准** | 十分组 + H-L · `signal_shift=1`（防未来函数） |
| **数据与图表包** | [`export/`](export/) |

> **阅读指引（给老板 / 研究负责人）**  
> - **研究排序看**：RankIC、ICIR、Gross Sharpe、MDD、Monotonicity  
> - **入库排序看**：Net Sharpe、Turnover、Implied Fee、Execution  
> 两者都重要：高夏普若成本不可兑现，则没有生产意义。

---

# 目录

1. [Factor Overview：因子概览](#1-factor-overview因子概览)  
2. [研究动机与研报逻辑](#2-研究动机与研报逻辑)  
3. [Factor Construction：因子构建](#3-factor-construction因子构建)  
4. [Mechanism Validation：机制验证](#4-mechanism-validation机制验证)  
5. [Predictive Performance：预测力](#5-predictive-performance预测力)  
6. [Portfolio Performance：组合表现](#6-portfolio-performance组合表现)  
7. [Risk Adjustment：风险中性](#7-risk-adjustment风险中性)  
8. [Stability：稳定性](#8-stability稳定性)  
9. [Execution：执行层优化](#9-execution执行层优化)  
10. [与原研报对照](#10-与原研报对照)  
11. [结论与定位](#11-结论与定位)  
12. [附录：数据字典与代码地图](#12-附录数据字典与代码地图)

---

# 1. Factor Overview：因子概览

## 1.1 一句话定义

> **TGD20** 刻画的是：在控制收益幅度与开盘时段结构之后，**异常下跌发生时间分布**所包含的横截面信息；经 <img src="figures/equations/eq_i_d6e65bb9c9.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 正交化并用 20 日平滑后，形成可交易的日频选股因子。

它**不是**：

- 简单的「涨得早 / 跌得晚」时间差 <img src="figures/equations/eq_i_b2b0b1e7fb.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />  
- 单纯的「涨跌时间距离」<img src="figures/equations/eq_i_abc7eae86a.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />  
- 未控制结构噪声的原始 Gu/Gd  

## 1.2 信息来源

| 维度 | 说明 |
|------|------|
| 数据层 | L2 衍生分钟线 `Stock_one_minute` Close |
| 信息层 | Temporal Information（收益率发生时刻） |
| 对照信息层 | Flow Density（资金到达时刻）—— **本报告不做合成** |
| 经济直觉 | 异常尾段卖压 / 延迟吸收 / 次日反转 |

## 1.3 双轨评分卡（机器可读）

[`export/metrics.json`](metrics.json)

```json
{
  "factor": "TGD20",
  "category": "temporal_information",
  "research_score": {
    "RankIC": 0.0430,
    "ICIR": 6.98,
    "ICIR_size_industry": 11.29,
    "Sharpe": 2.77,
    "Sharpe_size_industry": 4.06,
    "Monotonicity": 0.988
  },
  "production_score": {
    "NetSharpe_size_industry": 1.72,
    "NetSharpe_execution_best": 2.32,
    "Turnover_raw": 0.65,
    "Turnover_execution_best": 0.297
  }
}
```

完整变体表：[`export/factor_summary.csv`](factor_summary.csv)

---

# 2. 研究动机与研报逻辑

## 2.1 传统因子忽略了什么？

经典量价因子多回答：

> 价格/成交量 **变化了多少**（magnitude）

却较少系统回答：

> 这些变化 **发生在一天的什么时点**（timing）

日内分钟收益序列 <img src="figures/equations/eq_i_020b9ea6c9.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 同时包含：

1. **幅度信息**：涨跌强弱  
2. **时序信息**：涨跌集中在上午还是尾盘  

研报《日内分钟收益率的时序特征：逻辑讨论与因子增强》的核心主张是：在剥离「正常收益结构」后，**时序残差仍含有可交易的横截面 alpha**。

## 2.2 从 Gu/Gd 到 TGD 的逻辑链

```
分钟收益 r_t
    │
    ├─→ Gu：上涨分钟的时间重心
    └─→ Gd：下跌分钟的时间重心
            │
            ▼
     截面残差化（控制 Rū/Rd̄、R1、R2、隔夜）
            │
            ├─→ εu
            └─→ εd
                  │
                  ▼
           εd ~ εu → 创新 e
                  │
                  ▼
              MA20(e) = TGD20
```

经济含义（研报叙事）：

- 高 TGD：下跌更偏后、上涨更偏前 → 尾盘卖压释放后，后续存在相对收益  
- 低 TGD：相反结构  

本复现的关键贡献：**用机制表证明「不是 τ，而是 εd 残差」**。

## 2.3 在 Alpha Factory 中的位置

```
Alpha Factory
└── 1. Single Factor Research Layer   ← 本报告（Phase 1）
      └── Temporal Information
            └── TGD20
└── 2. Factor Evaluation Layer         ← 已标准化（metrics schema）
└── 3. Factor Combination Layer        ← 下一步；本报告不做
└── 4. Portfolio Construction Layer    ← Execution 已试点；Capacity 待做
```

---

# 3. Factor Construction：因子构建

## 3.1 分钟收益与交易分钟索引

连续竞价分钟（09:31–11:30 + 13:01–15:00），午休跳过，使 $t$ 连续：




<p align="center"><img src="figures/equations/eq_d_28d789b379.png" alt="eq" /></p>




实现：`core/l2_features/return_timing.py` · `trading_minute_index`

## 3.2 Gu / Gd（时序原语）

上涨时间重心：




<p align="center"><img src="figures/equations/eq_d_03aa7f0fe5.png" alt="eq" /></p>




下跌时间重心：




<p align="center"><img src="figures/equations/eq_d_fcc9a48d9b.png" alt="eq" /></p>




## 3.3 收益结构控制变量

| 变量 | 定义 | 作用 |
|------|------|------|
| <img src="figures/equations/eq_i_d06a7e3af7.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | <img src="figures/equations/eq_i_37b0c81439.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | 上涨幅度（**条件均值**，非全日均值） |
| <img src="figures/equations/eq_i_836d94b952.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | <img src="figures/equations/eq_i_ea95fbd5a1.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | 下跌幅度 |
| <img src="figures/equations/eq_i_b7000dd9eb.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | 约 09:31–10:00 累计收益 | 开盘半小时 |
| <img src="figures/equations/eq_i_dcaf993833.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | 约 10:01–10:30 累计收益 | 次半小时 |
| <img src="figures/equations/eq_i_d1ceea3001.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | <img src="figures/equations/eq_i_c2d70fc094.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | 隔夜 |

实现：`return_distribution.py`（2.5）· `timing_residual.py`（Stage 2）

## 3.4 截面残差化（每日 OLS，非时间序列）

对每个交易日，在股票截面上：




<p align="center"><img src="figures/equations/eq_d_d23718adfd.png" alt="eq" /></p>






<p align="center"><img src="figures/equations/eq_d_d6ee3fb972.png" alt="eq" /></p>




得到剥离结构噪声后的 <img src="figures/equations/eq_i_aca422167d.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />。

## 3.5 TGD 创新与平滑




<p align="center"><img src="figures/equations/eq_d_dd2015cfbd.png" alt="eq" /></p>






<p align="center"><img src="figures/equations/eq_d_8750062f2f.png" alt="eq" /></p>




实现：`core/l2_features/tgd.py`（**公式冻结，禁止 MA 窗口调参污染复现**）

## 3.6 防未来函数

当日 Gu/Gd 使用完整日分钟信息，只能预测 **次日**收益：




<p align="center"><img src="figures/equations/eq_d_388a4f3d80.png" alt="eq" /></p>




即代码中统一 `signal_shift=1`。

---

# 4. Mechanism Validation：机制验证

> 本节回答研复现最重要的问题：**为什么有效，而不是碰巧高 Sharpe。**

数据：[`export/mechanism_analysis.csv`](mechanism_analysis.csv)

## 4.1 原语族：τ / υ / Gu / Gd / TGD

| Signal | 含义 | RankIC | ICIR | H-L Sharpe | Net@15bp | Mono |
|--------|------|-------:|-----:|-----------:|---------:|-----:|
| `Gu_MA20` | 上涨时间重心 | 0.0447 | 4.25 | 0.90 | 0.41 | 0.73 |
| `Gd_MA20` | 下跌时间重心 | 0.0444 | 4.16 | 0.98 | 0.46 | 0.93 |
| `tau_MA20` | <img src="figures/equations/eq_i_797bee48fc.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | −0.0041 | **−0.56** | 0.27 | −1.92 | 0.33 |
| `upsilon_MA20` | <img src="figures/equations/eq_i_6a461e58d0.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | −0.0178 | **−2.26** | 0.58 | −1.22 | −0.88 |
| **`TGD20`** | 残差 MA20 | 0.0430 | **6.98** | **2.77** | **1.58** | **0.99** |

### 4.1.1 为什么 τ 无效？

<img src="figures/equations/eq_i_b2b0b1e7fb.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 只描述「下跌相对上涨更靠后」，但：

- 不控制涨跌幅度  
- 不控制开盘噪声  
- 不区分「温和漂移」与「剧烈双向波动」  

→ **time ordering ≠ alpha**

### 4.1.2 为什么 υ 无效？

<img src="figures/equations/eq_i_2b98d9f5c6.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 只描述「涨跌是否发生在不同时段」，方向不明：

| 情形 | 距离 | 经济含义 |
|------|------|----------|
| 早盘温和上涨 + 午后缓慢回落 | 大 | 可能无信息 |
| 早盘暴涨 + 午后暴跌 | 也大 | 完全不同机制 |

→ **temporal separation alone is insufficient**

## 4.2 残差机制：εu / εd / daily e / TGD20

| Signal | RankIC | ICIR | H-L Sharpe | Daily TO | Net@15bp | Mono |
|--------|-------:|-----:|-----------:|---------:|---------:|-----:|
| `epsilon_u` | 0.0107 | 1.62 | 2.58† | ~3.1 | −8.61 | −0.60 |
| `epsilon_d` | 0.0371 | **5.80** | 1.05 | ~3.1 | −7.36 | 0.75 |
| `tgd_eps`（日频 $e$） | **0.0474** | **8.53** | **4.33** | ~3.4 | −4.65 | 0.93 |
| `epsilon_d_MA20` | 0.0458 | 4.74 | 1.39 | 0.34 | 0.75 | 0.92 |
| **`TGD20`** | 0.0430 | **6.98** | **2.77** | **0.65** | **1.58** | **0.99** |

† `epsilon_u` 方向取反后的展示 Sharpe；其 IC 本身偏弱。

### 4.2.1 核心结论

1. **有效信息主要在 <img src="figures/equations/eq_i_c5933ec2ce.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />**（异常下跌时序），不是 <img src="figures/equations/eq_i_29982abf21.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />。  
2. **正交残差 <img src="figures/equations/eq_i_362043e9b7.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" />** 日频 ICIR 最高，但换手极高 → 不可直接交易。  
3. **MA20 把强而噪的日频残差，变成可投资的 TGD20**（ICIR 略降，Net 由负转正，单调性最优）。

这正是「因子增强」在本复现中的工程含义：不是另造公式，而是 **信息纯化 + 时间平滑**。

---

# 5. Predictive Performance：预测力

Confirmation · raw TGD20 · shift-1。

| 指标 | 数值 | 用途 |
|------|-----:|------|
| RankIC（日均） | **0.0430** | 预测力 |
| Annu IC | 0.680 | 年化 IC 强度 |
| ICIR | **6.98** | IC 稳定性 |
| Monotonicity | **0.988** | 十分组排序质量 |
| Direction | +1 | 高 TGD → 高后续收益 |

### 图：RankIC 时序

![RankIC 曲线](figures/ic_curve.png)

*图 1. TGD20 日度 RankIC 与 20 日均线（Confirmation）。均值约 4.3%，多数时段位于零轴上方。*

---

# 6. Portfolio Performance：组合表现

评价框架：十分组等权 · H-L = G10−G1 · 费用展示用 Implied AnnuFee(7.5bps)。

## 6.1 十分组累计收益 + H-L

![十分组+H-L](figures/cumulative_long_short.png)

*图 2. 十分组累计收益与 H-L。组间单调堆叠清晰；H-L 累计约 +1.4。*  
统计摘要：AnnuRet **36.87%** · Sharpe **2.77** · MDD **−18.99%** · Daily TO **0.65** · ImpliedFee **12.13%** · IC **0.0430** · ICIR **6.98**。

## 6.2 分组平均日收益

![分组柱状图](figures/decile_return.png)

*图 3. 各组平均日收益。因子暴露越高，平均收益越高，验证横截面区分力。*

## 6.3 H-L 净值路径

![H-L曲线](figures/hml_curve.png)

*图 4. 方向调整后的 H-L 累计收益。2024 年初有回撤，其后趋势恢复。*

## 6.4 组合指标汇总（raw）

| 指标 | 数值 |
|------|-----:|
| 年化收益 | 36.87% |
| Sharpe | 2.77 |
| 最大回撤 | −18.99% |
| 日均换手（H-L） | 0.65 |
| Implied AnnuFee(7.5bps) | 12.13% |

---

# 7. Risk Adjustment：风险中性

问题：TGD 是否只是小盘 / 行业暴露伪装？

| Mode | RankIC | ICIR | H-L Sharpe | Net@15bp | Daily TO |
|------|-------:|-----:|-----------:|---------:|---------:|
| raw | 0.0430 | 6.98 | 2.77 | 1.00 | 0.65 |
| size | 0.0443 | 8.67 | 3.52 | 1.51 | 0.65 |
| industry | 0.0408 | 8.90 | 3.19 | 1.16 | 0.64 |
| **size+industry** | 0.0415 | **11.29** | **4.06** | **1.72** | 0.65 |

**结论：中性后 ICIR / Sharpe 上升** → TGD 不是伪市值/行业因子；研究展示可用 raw，入库更应看 size+industry。

来源：[`export/factor_summary.csv`](factor_summary.csv)

---

# 8. Stability：稳定性

数据：[`export/yearly_stability.csv`](yearly_stability.csv)

| 年份/区间 | RankIC | ICIR | 正 IC 日占比 |
|-----------|-------:|-----:|------------:|
| 2020 | 0.0358 | 8.01 | 0.70 |
| 2021 | 0.0360 | 6.69 | 0.65 |
| 2022 | 0.0469 | 9.12 | 0.73 |
| 2023 | 0.0521 | 8.95 | 0.73 |
| 2024 | 0.0317 | 4.25 | 0.64 |
| 2025 | 0.0429 | 7.06 | 0.68 |
| 2020–2021 | 0.0359 | 7.26 | 0.68 |
| 2022–2023 | 0.0495 | 9.02 | 0.73 |
| 2024–2025 | 0.0373 | 5.49 | 0.66 |

**年均值 RankIC：6/6 年为正。** 2024 最弱但仍为正——稳定性论证优先看「正 IC 比例」，而非单年最高 Sharpe。

---

# 9. Execution：执行层优化

## 9.1 问题诊断：Rank Crossing

研究组合每日 Top/Bottom 10% 调仓时，大量交易来自 **边界附近微小排名穿越**（rank crossing）：因子几乎没变，却因跨越 10% 阈值产生买卖，制造成本、不增加信息。

因此：

> 高换手主要是 **portfolio implementation** 问题，不是 TGD20 signal 本身「必须高频」。

## 9.2 优化结果（size+industry 信号 · top/bottom 10%）

数据：[`export/execution_summary.csv`](execution_summary.csv)

| 版本 | Gross Sharpe | Daily TO | Net Sharpe |
|------|-------------:|---------:|-----------:|
| Daily EW | 4.06 | 0.65 | 1.28 |
| Every 5d | 3.43 | 0.31 | 2.06 |
| Friday | 3.32 | 0.31 | 1.88 |
| **Daily + Buffer 5/15** | 3.51 | **0.30** | **2.32** |

Buffer 5/15：进入 Top 5% 才开多，跌出 Top 15% 才平仓（空头对称）。相对单纯降频，**仍每天观察信号，但只交易「足够大」的变化**，故 Net 更高。

## 9.3 Investable 起点（研究候选，非上线指令）

```
Signal:     size+industry(TGD20)
Rebalance:  daily monitoring
Portfolio:  buffer entry 5% / exit 15%
Cost:       15bp round-trip
Result:     Net Sharpe ≈ 2.32 · TO ≈ 0.30
```

**Capacity / 指数池分层尚未完成**（Production readiness 下一阶段）。

---

# 10. 与原研报对照

| 项目 | 研报 | 本框架 |
|------|------|--------|
| 分钟收益 / Gu·Gd | ✓ | ✓ |
| <img src="figures/equations/eq_i_b7de8ef5e0.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 条件均值 | ✓ | ✓ |
| R1/R2/隔夜控制 | ✓ | ✓ |
| 截面残差 + <img src="figures/equations/eq_i_7d077458c8.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> | ✓ | ✓ |
| MA20 | ✓ | ✓ |
| 分组 | 常为五组 | **十组 + H-L（项目标准）** |
| 中性 / 成本 / shift | 有限或未强调 | size+ind · 15bp · **显式 shift-1** |
| 机制表 τ/υ/ε | 文字论证 | **实证表否证 τ/υ** |

**不要硬对齐研报五组 IR 数字**；对齐的是机制与统计强度。

---

# 11. 结论与定位

## 11.1 研究结论

1. **日内分钟收益的时序特征在控制结构噪声后仍含横截面信息。**  
2. **简单时间差 τ、时间距离 υ 不是 alpha；异常下跌时序残差 <img src="figures/equations/eq_i_c5933ec2ce.png" alt="eq" style="height:1.15em; vertical-align:-0.2em;" /> 才是主源。**  
3. **TGD20 = 正交残差 + MA20**，是研究可展示、执行可改善的形态。  
4. **中性增强、六年 RankIC 全正、十分组近乎完美单调**，达到「可给老板展示的研报复现」标准。  
5. **换手问题可用 buffer 显著改善 Net Sharpe**，证明 alpha 来自排序信息而非频繁交易。

## 11.2 Alpha Factory 标签

| 标签 | 状态 |
|------|------|
| 单因子研究报告（Phase 1） | ✅ 完成 |
| 公式 | 🔒 冻结 |
| Research Satellite | ✅ `research_satellites/TGD20.yaml` |
| Production / Capacity | ⏳ 未做 |
| 与 Flow Density 合成 | ⏳ Phase 2（勿在本报告内开启） |

## 11.3 明确不做的事

- ❌ 改 MA10/30/60 或残差模型（污染复现）  
- ❌ 在本报告内做 TGD+Flow / TGD+SKEW 合成  
- ❌ 用 ML 重新拟合时序  

下一阶段正确顺序：**复制本报告模板到 Flow Density 等单因子 → 再建 Combination Engine**。

---

# 12. 附录：数据字典与代码地图

## 12.1 导出数据包结构（本报告配套）

```text
research/reports/tgd_v1/export/

├── factor_summary.csv          # 双轨评分明细
├── mechanism_analysis.csv      # τ/υ/ε/TGD 机制表
├── yearly_stability.csv        # 分年 / 分块稳定性
├── execution_summary.csv       # 执行层实验全表
├── metrics.json                # 机器可读总览
└── figures/
      ├── ic_curve.png
      ├── decile_return.png
      ├── hml_curve.png
      └── cumulative_long_short.png
```

单因子模板镜像：`research/reports/factors/TGD20/`

## 12.2 代码地图（公式层冻结）

| 阶段 | 模块 |
|------|------|
| Gu/Gd | `core/l2_features/return_timing.py` |
| Rū/Rd̄ | `core/l2_features/return_distribution.py` |
| 残差 | `core/l2_features/timing_residual.py` |
| TGD20 | `core/l2_features/tgd.py` |
| 面板构建 | `core/l2_features/tgd_panel_builder.py` |
| 指标 schema | `factor_eval_metrics.py` |
| 机制完整性 | `run_tgd_replication_integrity.py` |
| 组合验证 | `run_tgd_validation_v1.py` |
| 执行优化 | `execution_layer.py` / `run_tgd_execution_opt_v1.py` |

## 12.3 Implied AnnuFee 公式




<p align="center"><img src="figures/equations/eq_d_222f162182.png" alt="eq" /></p>




（标注中的 7.5% 实为 **7.5 bps**。）

---

**报告结束。**  
TGD20 作为 Alpha Factory 中第一个完整走完「研报拆解 → 机制验证 → 双轨评分 → 执行试点」的 **L2 Temporal Information** 单因子，后续应 **复制模板**，而非继续修改 TGD。
