# Alpha Factory — 项目级里程碑总结

**截至：** 2026-07-17  
**定位：** Research → Validation → Attribution → Orthogonality → Combination

总判断：已完成 **2 个研报因子的完整复现 + 机构级机制归因 + 因子独立性验证 + 标准化研究框架**。  
下一步进入 **Alpha Combination（多维信息融合）**，不再以“复现研报”为主线。

---

## 三层完成度

| 层 | 含义 | 状态 |
|----|------|------|
| 1. Replication | 研报直接复现 | ✅ TGD20 完整；FlowDensity 结构完整 |
| 2. Research Extension | 机构级验证增强 | ✅ Mechanism + Orthogonality |
| 3. Platform Extension | Alpha Factory 系统化 | ✅ Factor Report Generator v1 |

---

## 一、Replication（研报复现）

### 1. TGD20 — 完成度最高

| Stage | 内容 | 状态 |
|-------|------|------|
| 0 | 研报拆解（Gu/Gd/τ/υ/residual/TGD20） | ✅ |
| 1 | 分钟收益 → Gu/Gd（`return_timing.py`） | ✅ |
| 2 | Residual Layer（εu/εd） | ✅ |
| 3 | TGD wrapper + MA20 | ✅ |
| 4 | IC / ICIR / Decile / H-L | ✅ |
| 5 | Neutralization / Stability / Execution | ✅ |

**Confirmation 要点（raw → size+ind → execution）**

| Metric | 值 |
|--------|---:|
| RankIC | 0.043 |
| ICIR raw / size+ind | 6.98 / **11.29** |
| H-L Sharpe raw / size+ind | 2.77 / 4.06 |
| Monotonicity | 0.988 |
| Yearly IC+ | 6/6 |
| Net Sharpe baseline / best | 1.00→1.72 / **2.32**（buffer_5_15） |

```yaml
factor_id: TGD20
category: temporal_information
status: validated_single_factor
formula: { frozen: true }
production: { deferred: true }
combination: { candidate: true }
```

**结论：** 完整研报复现闭环；公式冻结，不再调参。

---

### 2. FlowDensity20 — 结构复现完成，经济含义被重定义

| Stage | 内容 | 状态 |
|-------|------|------|
| 因子构造 | `net_active_flow_mktcap_20d` | ✅ |
| IC / H-L / Neut | size / industry / size+ind | ✅ |
| Stability | yearly IC 全正 | ✅ |
| Execution | 同网格 `execution_layer.py` | ✅ |

| Metric | 值 |
|--------|---:|
| ICIR size+ind | 4.85 |
| Net Sharpe baseline / best | 1.85 / **2.88**（buffer_10_30） |
| Daily TO | 0.46 → **0.165** |

```yaml
factor_id: FlowDensity20
category: [microstructure, liquidity_flow_interaction]
status: validated_single_factor_candidate
formula: { frozen: false }
```

**结论：** 若只论研报复现已完成；后续归因证明它不是纯 Flow。

---

## 二、Research Extension（机构级增强）

### Extension 1 — Mechanism Attribution

**TGD：** α 来自 abnormal downside timing residual，不是简单 τ/υ。

| Signal | ICIR |
|--------|-----:|
| τ | −0.56 |
| υ | −2.26 |
| εu | 1.62 |
| εd | 5.80 |
| TGD20 | 6.98 |

**FlowDensity：** α 是 **Flow × Liquidity**，不是纯主动买卖方向。

| Signal | ICIR |
|--------|-----:|
| Amount / Gross / Buy / Sell | ≈ −8.6 |
| FlowDensity raw | **+4.85** |
| Flow ⊥ Amount | **−2.49**（符号翻转） |
| Amount ⊥ Flow | −8.49 |

→ 分类从 `flow_information` 改为 `microstructure + liquidity_flow_interaction`。

### Extension 2 — Orthogonality（进组合前必做）

报告：`research/reports/factor_orthogonality/TGD20_FlowDensity20/`

| Case | Pair | Corr | 结论 |
|------|------|-----:|------|
| A | TGD vs Flow raw | **0.22** | TGD⊥Flow ICIR **9.12**；信息互补 |
| B | TGD vs Flow⊥Amount | 0.01 | 纯 Flow 无正贡献，不进组合 |
| C | TGD vs Amount | −0.32 | TGD⊥Amount ICIR **7.66**；TGD ≠ liquidity |

Equal-rank probe：`0.5 TGD + 0.5 Flow` → ICIR **8.50** < TGD 单独 **11.28**  
→ 互补成立，但 **禁止默认 50/50**；应用 IC-weighted。

---

## 三、Platform（标准化能力）

### Factor Report Generator v1

统一包：`research/reports/factors/{FACTOR_ID}/`

```
factor_report.md / factor_card.yaml / metrics.json
factor_summary.csv / mechanism.csv / stability.csv
execution_summary.csv / artifacts/
```

实例：`TGD20/`（模板冻结）、`FlowDensity20/`（candidate）

### Information Taxonomy（当前）

```
Information Layer
├── Temporal ──────────── TGD20（pure temporal）
├── Microstructure/Liquidity
│     ├── Amount（anti-activity）
│     └── FlowDensity20（Flow × Liquidity）
└── Future ────────────── Cutting / ATS / Smart Money / Momentum / Reversal
```

---

## 四、未完成 / 下一阶段

| # | 项 | 说明 |
|---|----|------|
| 1 | **Composite Alpha Engine v1** | IC-weighted / rolling IC / residual stacking；目标 Composite > max(single) |
| 2 | 更多单因子 | Factor Cutting、ATS、Smart Money、Liquidity Stability、Momentum/Reversal |
| 3 | FlowDensity 命名决策 | 保留 FlowDensity20（interaction）或改名 LiquidityAdjustedActiveFlow — **暂不 freeze** |
| 4 | Portfolio optimizer | Combination 之后再做 |

---

## 五、给评审 / 老板的一句话

> 已完成两个研报因子的完整复现，并完成机制归因、风险暴露拆解、因子独立性验证与标准化研究包；项目从“找/复现因子”进入 **多维信息融合（Alpha Combination）** 阶段。下一交付：IC-weighted Composite Alpha Engine v1（TGD + FlowDensity_raw）。

---

## 关键路径索引

| 产物 | 路径 |
|------|------|
| Factor packs | `research/reports/factors/` |
| Roadmap | `research/reports/factors/ROADMAP.md` |
| TGD 长文 | `research/reports/tgd_v1/` |
| FlowDensity | `research/reports/l2_flow_density_v1/` |
| Orthogonality | `research/reports/factor_orthogonality/TGD20_FlowDensity20/` |
| Cards | `research/alpha_library_v1/research_satellites/` |
