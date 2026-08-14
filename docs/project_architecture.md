# 项目架构地图（Project Architecture）

> 面向新加入研究员 / AI agent 的全景上下文。只描述现状，不含优化建议。
> 生成时点：2026-08-04。信息来源：全库代码侦察（5 路并行探索）+ 既有文档。
> 项目根：`/home/SiYangCao/factor_dev/factor_research0703/factor_dev`

---

## 1. 一句话定位

A 股量化因子研究仓库：**EOD 日频因子引擎**（主力）+ **分钟/日内因子** + **ClickHouse L2 微观结构** 三条数据线，统一经 `Factor_Dev_Lib.groupTest` 十分组回测产出指标，再经 `research_delivery` 交付层资产化为单因子报告。

## 2. 总体架构图

```text
┌──────────────────────── Raw Data（源库，只读账号）────────────────────────┐
│ ClickHouse cmds  10.80.139.9:8123                                        │
│   SSE/SZSE_AL_SSL2_EXG（L2 十档快照）· SSE/SZSE_AL_TICK_EXG（逐笔）· KLIN │
│ DolphinDB        10.12.180.9:8902 (pyread)                               │
│   QV_Trade_to_MinuteBar/Stock_one_minute（1分钟线，2018-09 起）           │
│   WIND.ASHAREEODPRICES / AINDEX*WEIGHT / ASHARETTMHIS / ...              │
│ Oracle Wind（2018 前 EOD）· 聚源 · 朝阳永续 · 财汇 · 排排 · 普益          │
│ MySQL  通联 datayes · 东方财富 emdata · 野尘 yechen                       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────── Data Access Layer（连接与加载）──────────────────────┐
│ COMMON_CONST.py            全部连接常量                                   │
│ core/ddb/connection.py     DDB 共享 session 唯一入口（get_ddb_session）   │
│ factor_data_loaders.py     EOD 宽表 / 估值市值 / 财报 TTM / Wind Oracle   │
│ core/ddb/{eod,minute,financial}.py · core/data/panel_reader.py           │
│ minute_bar_store.py        分钟数据按需查询层（防全历史误拉）              │
│ l2_data_loaders.py         分钟→日频 L2 聚合 + parquet 缓存                │
│ research/l2_alpha/clickhouse_ssl2.py   CH SSL2 服务端特征提取              │
│ l2_factor_reproduction/python/ch_tick.py  CH Tick 服务端订单规模聚合       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────── Feature / Factor Layer（因子计算）───────────────────┐
│ factor_formulas*.py        ~20 个公式文件，@register_factor 装饰器注册     │
│ factor_taxonomy*.py        因子清单 + 元数据（family/hypothesis/direction）│
│ factor_config.py           TRACK 机制：20+ 研究线共用一个 runner           │
│ core/factors/skew/         canonical 公式库雏形（SKEW 家族）               │
│ core/l2_features/          APM/SmartMoney/Ideal* ActiveV2 面板构建器       │
│ intraday_formulas.py + core/intraday_alphas.py   分钟因子（DDB/Python 双轨）│
│ l2_factor_reproduction/    海通 L2 研报复现子项目（已归档）                │
│ research/l2_alpha/         CH SSL2 特征工厂（实验线）                      │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────── Backtest Engine（回测）─────────────────────────┐
│ Factor_Test_Process.py     EOD 各 track 总入口（加载数据→批量回测）        │
│ factor_runner.py           run_eod_batch：universe mask→ST/涨跌停/停牌     │
│                            过滤→signal.shift(1)→groupTest→落盘             │
│ Factor_Dev_Lib.py          groupTest（十分组/H-L/RankIC/ICIR/Sharpe/换手） │
│ l2_factor_reproduction/python/backtest.py  L2 窄表→宽表→同一 groupTest     │
│ intraday_lib.py / core/evaluation/intraday_metrics.py  分钟评估（v2 口径） │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────── Validation Layer（验证/筛选）────────────────────┐
│ alpha_validation_harness.py · alpha_investability.py（实扣成本 0.15%）     │
│ run_*_validation*.py（skew/tgd/d1-d5/l2/cn_broker 各家族验证脚本）          │
│ l2_factor_reproduction/scripts/  中性化/周度/参数敏感性/状态依赖/时稳分析   │
│ factor_similarity / dimension_map / frozen_stack（组合与归因，当前暂停）    │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌───────────────────────── Report Layer（报告/交付）──────────────────────┐
│ research_delivery/         ★当前活跃交付层（GOVERNANCE/ROADMAP/因子卡片）  │
│ factor_report_generator_v2.py + factor_specs/*.yaml  Template v2 pack     │
│ research/reports/factors/  因子 Research Pack（资产化）                    │
│ docs/                      checkpoint/milestone/template 文档              │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. 目录地图

| 目录 | 用途 | 关键文件 | 状态 |
|------|------|----------|------|
| 根目录 `*.py`（~150 个） | EOD 因子引擎 + 各家族 run_* 验证脚本 | `Factor_Test_Process.py` / `factor_runner.py` / `factor_config.py` / `Factor_Dev_Lib.py` / `factor_formulas*.py` | **活跃**（主生产路径） |
| `core/` | 新一代基础设施 | `ddb/connection.py`（唯一 session 入口）、`evaluation/intraday_metrics.py`、`factors/skew/`、`l2_features/`（ActiveV2 builders）、`intraday_alphas.py`、`factor_store.py`（DDB 因子库设计，未接 runner） | **活跃**，逐步吸收旧代码 |
| `factor_cutting/` | 因子切割研究（APM/Ideal 系列升级版） | `engine.py` / `knives.py` / `registry.py` | 活跃（ActiveV2 因子的实验场） |
| `factor_specs/` | 因子 YAML 规格 + 报告内容声明 | `TGD20.yaml` / `*_report_content.yaml` | 活跃（Template v2 的 schema 层） |
| `factors/intraday/` | 分钟因子包化 wrapper + DDB/Python parity | — | 活跃（分钟因子工程化方向） |
| `l2_factor_reproduction/` | 海通 L2 研报复现子项目 | `python/ch_tick.py` / `python/backtest.py` / `scripts/*` | **研究已归档**（Phase1/2 结论封存），工程仍可复现；mid_order_ratio 后续优化在此 |
| `research/` | 实验与审计主目录 | `results/`（全部实验产物）、`l2_alpha/`（CH SSL2 特征工厂）、`docs/`（数据手册/审计）、`frozen_state_v1/` | **活跃**（results 是当前产物主目录） |
| `research_delivery/` | ★当前交付主线 | `GOVERNANCE.md` / `ROADMAP.md` / `factor_delivery_plan.csv` / `selected_factors/` / `*_research_package/` | **最活跃** |
| `result/`（~1.2G, 48 子目录） | 旧 runner 输出根 | `factor_config.result_root_for()` 仍指向这里；`factor_runner.legacy_result_path()` 兼容读取 | **旧版**，新研究写 `research/results/` |
| `docs/` | 项目文档 | checkpoint/milestone 系列、`factor_report_template_v2*.md`、本次新增 5 份知识库 | 活跃 |
| `tests/` | 单测 | `test_l2_*` / `test_intraday_*` / `test_factor_*` | 活跃 |
| 根目录巨型 parquet | `intraday.parquet`(1.8G) / `intraday_2024.parquet`(11.4G) / `intraday_2025.parquet`(12.2G) | 分钟面板缓存 | **过渡产物**，非权威数据源；权威路径是 MinuteBarStore 按需查 DDB |

## 4. 三条数据线 × 一条统一回测

| 数据线 | 数据源 | 因子入口 | 回测路径 |
|--------|--------|----------|----------|
| EOD 日频（主力） | DDB Wind EOD + 估值/财报 | `factor_formulas*.py` → `factor_config.TRACK` → `Factor_Test_Process.py` | `run_eod_batch` → `groupTest` |
| 分钟/日内 | DDB `Stock_one_minute` / MinuteBarStore | `intraday_formulas.py` + `core/intraday_alphas.py` | `Intraday_Factor_Test_Process.py` / intraday v2 metrics |
| L2 微观结构 | ClickHouse SSL2/Tick | `research/l2_alpha/`（实验）、`l2_factor_reproduction/`（复现） | `l2_factor_reproduction/python/backtest.py` → 同一 `groupTest` |

**统一收敛点**：所有日频因子最终都走 `Factor_Dev_Lib.groupTest`（十分组 + H-L），指标口径见 `docs/backtest_framework.md`。

## 5. 因子如何进入系统（主路径）

```text
1) factor_formulas_xxx.py 写计算函数 + @register_factor / @register_eod_engine
        ↓
2) factor_taxonomy*.py 把名字加入对应 LIST（含元数据）
        ↓
3) factor_config.py TRACK_DEFAULT_LISTS 挂到某条 TRACK（或 CUSTOM_FACTOR_LIST）
        ↓
4) Factor_Test_Process.py 按 TRACK 加载数据、构造 cache、选择 build 函数
        ↓
5) factor_runner.run_eod_batch：universe/ST/涨跌停/停牌过滤 → shift(1) → groupTest
        ↓
6) 结果落盘 result/ 或 research/results/ → 验证脚本 → research_delivery 交付
```

新协议层（`factor_specs/` + `factor_research_harness.py`）目标是 `spec → compute → evaluate → report pack`，目前仅 TGD20 有只读 adapter，尚未成为通用执行引擎。

## 6. 当前项目状态

### 已完成（封存/可用）
- EOD 因子引擎：~150 因子、15 个家族、20+ track（详见 `docs/factor_inventory.md`）
- ClickHouse 集成：SSL2 特征提取 + Tick 订单规模聚合（服务端 GROUP BY）
- DolphinDB 集成：分钟线 / Wind EOD / 估值 / 财报统一加载层
- L2 研报复现（l2_factor_reproduction）：Phase 1 分钟代理（归档）、Phase 2 mid/small_order_ratio（见 `docs/mid_order_ratio_pipeline.md`）
- 交付层：TGD20 完整 research package；D1/FlowDensity20/APM/Ideal* 因子卡片

### 进行中（2026-08）
- `research_delivery` 单因子交付冲刺（ROADMAP：先积累高质量单因子）
- mid_order_ratio 优化收尾：`optimization_report.md` 建议以 `-mid_order_ratio` 行业中性周度调仓入库（命名 `order_flow_mid_reversal_weekly`）
- SUE_ConsensusEPS（III-B 基本面方向）：PIT 事件面板设计阶段（DESIGN ONLY）
- SKEW 优化包、Batch 2 候选（AmihudShock / LiquidityResidual / ActiveTradingImbalance 等）

### 暂停/未来
- 组合层：alpha_combination / portfolio_construction / similarity matrix / information topology —— 多处文档明确暂停，等单因子覆盖足够
- Registry v1（milestone_1E 规划）
- CH → DDB 盘口表日更 ETL（`L2_Snapshot_Daily`，盘口委托类因子依赖）

## 7. 相关文档

| 文档 | 内容 |
|------|------|
| `docs/data_inventory.md` | 全部数据源、表、字段、覆盖区间、坑 |
| `docs/factor_inventory.md` | ~150 因子清单（家族/数据源/公式位置/状态） |
| `docs/backtest_framework.md` | IC/ICIR/Sharpe/换手/成本精确定义（防口径混淆） |
| `docs/mid_order_ratio_pipeline.md` | mid_order_ratio 端到端链路 |
| `research/docs/data_source_handbook_v2.md` | 数据源研发手册（审计口径） |
| `分钟级因子现状快照报告.md` | 分钟因子现状 |
| `L2_DATA_LINEAGE.md` | L2 数据血缘 |
