# 海通 L2 研报因子复现 — 项目构建指南

面向初级量化研究员的独立子项目：在 **DolphinDB** 内完成 Level-2 / 分钟数据聚合与因子计算，在 **Python** 端做调度、日频分组回测与结果落盘。目录与现有 `research/` 解耦，但复用仓库内连接层与回测框架。

---

## 一、项目目标与设计原则

### 1.1 目标

- 复现海通证券三份 Level-2 研报中的核心因子：
  - **分时成交类**（Phase 1，基于 DDB 分钟线代理）
  - **盘口委托类**（依赖 CH Snapshot → DDB 本地表）
  - **成交占比类**（Phase 2，需逐笔，当前占位）
- 计算密集型任务下推 DolphinDB（列式存储 / 向量化 / 分布式），避免 Python 拉全量 L2 导致 OOM。
- Python 只读**窄表结果**（`symbol, tradetime, factorname, value`），再走 `Factor_Dev_Lib.groupTest`。
- 独立于 `research/`，但复用：
  - `COMMON_CONST` / `core.ddb.connection` / `factor_data_loaders.connect_ddb`
  - `intraday_lib`（股票池与分钟口径参考）
  - `Factor_Dev_Lib`（日频分组回测）
  - `factor_config.START_DAY / END_DAY`

### 1.2 设计原则

| 原则 | 含义 |
|------|------|
| 分层清晰 | DDB 计算层 / Python 调度层 / 回测层分离 |
| 可扩展 | 新因子 = 1 个 `.dos` + 在 `FACTOR_SCRIPT_MAP` 注册 |
| 性能优先 | 原始加工全在 DDB；Python 只拿窄表 |
| 复用优先 | 不重写连接、股票池、涨跌停过滤、分组回测 |

### 1.3 数据来源说明（Phase 1）

**Phase 1 分钟因子只读现成宽表，不扫 CH 原始 L2/Tick：**

| 表 | 用途 |
|----|------|
| `dfs://QV_Trade_to_MinuteBar/Stock_one_minute` | 已聚合 1 分钟 K 线 |
| `dfs://WIND.AINDEXCSI1000WEIGHT` 等 | 股票池 |
| `dfs://WIND.ASHARECALENDAR` | 交易日 |
| `dfs://WIND.ASHAREEODPRICES` | 复权因子（init 工具；分钟表自带 `Adjfactor`） |

盘口脚本里的 CH / ODBC 仅针对 Phase 2+，默认不在 `FACTOR_LIST` 中，冒烟测试不会触发。

### 1.4 架构总览

```text
┌─────────────────────────────────────────────────────────────┐
│  scripts/run_*.py          入口（一键 / 单因子调试）           │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  python/factor_runner.py   遍历 FACTOR_LIST，落盘 parquet    │
│  python/factor_builder.py  run(init.dos)+run(factor.dos)     │
│  python/ddb_client.py      封装共享 DDB session               │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  ddb_scripts/                                                 │
│    init/init_env.dos         股票池 / 分钟加载工具             │
│    factors/minute_based/*    分时成交类（可立即跑）             │
│    factors/order_book/*      盘口类（需本地 Snapshot 表）       │
│    factors/trade_flow/*      Phase2 占位                       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  dfs://QV_Trade_to_MinuteBar/Stock_one_minute   (分钟)        │
│  dfs://L2_Snapshot_Daily/snapshot               (盘口，可选)  │
│  WIND 指数权重 / 日历 / EOD                                   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  python/backtest.py → Factor_Dev_Lib.groupTest               │
│  → research/results/l2_reproduction/<factor>/                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、目录结构

```text
l2_factor_reproduction/
├── README.md                          # 本指南
├── __init__.py
├── config/
│   └── settings.py                    # 日期、标的、因子清单、路径
├── ddb_scripts/
│   ├── init/init_env.dos              # 公共函数
│   ├── factors/
│   │   ├── minute_based/              # 分时成交类
│   │   │   ├── avg_order_amount.dos
│   │   │   ├── big_order_flow.dos
│   │   │   └── big_order_drive.dos
│   │   ├── order_book/                # 盘口委托类
│   │   │   ├── net_order_change.dos
│   │   │   ├── order_volatility.dos
│   │   │   └── order_skew.dos
│   │   └── trade_flow/                # Phase2 占位
│   │       └── order_size_ratio.dos
│   ├── batch/                         # 批量 / 汇总（GUI 调试用）
│   └── stream/                        # 流式占位
├── python/
│   ├── ddb_client.py
│   ├── factor_builder.py
│   ├── factor_runner.py
│   ├── backtest.py
│   └── utils/
│       ├── symbol_utils.py
│       └── date_utils.py
├── scripts/
│   ├── run_all_factors.py
│   └── run_single_factor.py
└── tests/
    └── test_ddb_scripts.dos
```

结果默认写入：

```text
research/results/l2_reproduction/<factor_name>/
  ├── factor_narrow.parquet
  ├── group_pnl.csv
  └── group_turnover.csv
```

---

## 三、环境准备与依赖

### 3.1 系统

- Linux + DolphinDB 2.00.9+（本仓库已用 `DATA_DB_CONN`）
- ClickHouse（仅盘口 / Phase2 需要）
- Python 3.8+，建议 conda 环境 `base_93`

### 3.2 Python 依赖

```bash
pip install dolphindb pandas numpy pyarrow clickhouse-connect
```

仓库内已有 `Factor_Dev_Lib`、`factor_data_loaders` 等，无需再装研究框架。

### 3.3 权限与数据

| 资源 | 用途 | 账号要求 |
|------|------|----------|
| `dfs://QV_Trade_to_MinuteBar/Stock_one_minute` | 分钟类因子 | `pyread` 可读 |
| `dfs://WIND.AINDEXCSI1000WEIGHT` 等 | 股票池 | 同上 |
| `dfs://L2_Snapshot_Daily/snapshot` | 盘口类 | **需自行同步**（见 §6.4） |
| CH `cmds.SSE_AL_SSL2_EXG` / `SZSE_*` | 盘口源 | ODBC 或定时 ETL |

连接自检：

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
/opt/conda/anaconda3/envs/base_93/bin/python -c "from factor_data_loaders import connect_ddb; s=connect_ddb(); print(s.run('1+1'))"
```

---

## 四、配置说明（`config/settings.py`）

关键项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `START_DAY` / `END_DAY` | 继承 `factor_config` | 回测与计算区间 |
| `UNIVERSE` | `000852.SH` | 中证1000；可改 300/500 |
| `FACTOR_LIST` | 三个分钟类因子 | 盘口类默认注释，就绪后打开 |
| `DDB_SCRIPT_ROOT` | 本项目 `ddb_scripts` 绝对路径 | `run()` 必须用绝对路径 |
| `RESULT_ROOT` | `research/results/l2_reproduction` | 窄表与回测 CSV |
| `BIG_ORDER_TOP_FRAC` | `0.2` | 大单净流入 top 比例 |
| `BIG_ORDER_DRIVE_TOP_FRAC` | `0.3` | 大单驱动涨幅 top 比例 |
| `BACKTEST_SILENT` | `True` | `groupTest(info='silent')` 不弹图 |

---

## 五、DDB 初始化（`ddb_scripts/init/init_env.dos`）

**不要在 `.dos` 里 `login`**：会话由 Python `connect_ddb()` 建立。

`init_env.dos` 提供与 `intraday_lib.ddb_functions` 对齐的精简工具：

- `get_trading_days` / `get_stock_pool` / `get_adj_factor`
- `load_minute_bars` / `attach_bar_ret` / `attach_avg_order_amt`
- `to_signal_tradetime`（日频信号统一 `Date + 09:30:00`）

每个因子计算前，Python 会先 `run("<abs>/init/init_env.dos")`。

---

## 六、因子定义与口径

所有 `compute_*` 返回窄表：

| 列 | 类型 | 含义 |
|----|------|------|
| `symbol` | Wind 代码 | 如 `600000.SH` |
| `tradetime` | timestamp | 信号时刻（日频用 09:30） |
| `factorname` | string | 因子名 |
| `value` | double | 因子值 |

### 6.1 `avg_outflow_ratio` — 平均单笔流出金额占比（代理）

- 数据：`Stock_one_minute`
- 逻辑：分钟复权收益 `BarRet < 0` 的成交额占比  
  \(\sum Amount\cdot 1_{BarRet<0} / \sum Amount\)
- 说明：研报「单笔」理想来自逐笔；Phase1 用分钟额/手数代理，先验证方向与稳定性。

### 6.2 `big_order_net_inflow` — 大单资金净流入率（代理）

- 日内按 `AvgOrderAmt = Amount / (Volume/100 + 1)` 降序取前 `top_frac`
- 值 = \((A_{up} - A_{down}) / A_{day}\)

### 6.3 `big_order_drive_ret` — 大单驱动涨幅（代理）

- 同样取 top 分钟，值 = 成交额加权 `BarRet`

### 6.4 盘口委托类（需本地快照表）

脚本：`net_order_change` / `order_change_volatility` / `order_change_skew`

1. 期望表：`dfs://L2_Snapshot_Daily/snapshot`
2. 字段二选一：
   - 已聚合：`Symbol, ExchTime, TotalBid, TotalAsk`
   - 或十档数组：`BidVolumes, AskVolumes`（脚本内 `sum`）
3. 逻辑：分钟末净委买 → 变化率 → 开盘 30 分钟（09:30–10:00）的 mean / std / skew

**强烈建议**：每日定时从 CH 同步前一日 Snapshot 到 DDB 本地库，历史回测不要每次走 ODBC。

在 `settings.FACTOR_LIST` 中取消注释即可启用。

### 6.5 Phase 2：`mid_order_ratio` / `small_order_ratio`

`.dos` 文件仍是历史占位，但当前 Python 路径通过 `TICK_AGG_FACTORS` 直接调用
`python/ch_tick.py`，在 ClickHouse 内按股票日聚合。SSE 使用 `Type='T'`；
SZSE 使用 `Type='011'` 且买卖订单号均有效；多日查询对每个日期逐行施加
09:30–15:00 时段条件。`mid_order_ratio` 与 `small_order_ratio` 已是可运行实现。

---

## 七、Python 调度层

| 模块 | 职责 |
|------|------|
| `ddb_client.DDBFactorClient` | 共享 session；`run_file` **本地读 `.dos` 再提交字符串**（规避 `run(path)` 无权限） |
| `factor_builder.build_factor` | init + 因子脚本 + `compute_<name>(...)` |
| `factor_runner.run_all_factors` | 遍历清单、写 parquet |
| `backtest.backtest_factor` | 窄→宽、涨跌停/ST/停牌过滤、`groupTest` |

新增因子三步：

1. 写 `ddb_scripts/factors/.../xxx.dos`，定义 `compute_my_factor`
2. 在 `FACTOR_SCRIPT_MAP` 注册路径
3. 把名字加入 `FACTOR_LIST`

---

## 八、运行流程

在仓库根目录、用 `base_93` 解释器执行：

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# 单因子调试（推荐先跑短区间冒烟）
$PY l2_factor_reproduction/scripts/run_single_factor.py --factor avg_outflow_ratio \
    --start 2024-01-02 --end 2024-03-29 --no-backtest

# 只算不回测
$PY l2_factor_reproduction/scripts/run_single_factor.py --factor big_order_net_inflow --no-backtest

# 配置清单内全部因子 + 回测
$PY l2_factor_reproduction/scripts/run_all_factors.py
```

建议上手顺序：

1. 连接冒烟（§3.3）
2. 短区间改 `START_DAY/END_DAY`（例如 1 个月）跑 `avg_outflow_ratio`
3. 看 `factor_narrow.parquet` 行数与缺失
4. 看 `group_pnl.csv` 的 H-L 与换手
5. 再放开全样本与盘口因子

---

## 九、结果与可视化

- 窄表 / 分组收益 / 换手：见 `RESULT_ROOT`
- 日频因子一般不必做 Bartime×Horizon 热力图；若要分钟择时，可另行调用 `intraday_lib.analyze_group_performance_by_bartime`（需分钟收益矩阵）

---

## 十、性能优化建议

1. **合并 I/O**：多个分钟因子可共享一次 `load_minute_bars`（见 `batch/batch_factor_calc.dos` 思路）
2. **增量日期**：只算新交易日，parquet 追加合并
3. **CH → DDB 日更**：盘口表本地化，避免 ODBC 扫历史
4. **`submitJob`**：多因子并行（注意内存；初级阶段串行更稳）
5. **及时 `undef`**：长区间脚本结束后释放临时表

---

## 十一、常见问题

| 问题 | 原因 | 处理 |
|------|------|------|
| DDB 连不上 | 网络 / 账号 | 查 `COMMON_CONST.DATA_DB_CONN`，用 §3.3 自检 |
| `run` 找不到文件 | 相对路径 | 必须用 `DDB_SCRIPT_ROOT` 绝对路径 |
| 盘口因子报错 | 无 `L2_Snapshot_Daily` | 先同步 CH，或从 `FACTOR_LIST` 去掉 |
| 窄表为空 | 区间无交易 / 股票池空 | 缩短日期自测；确认 `UNIVERSE` |
| `ImportError` | 未在仓库根跑 | `sys.path` 已由 settings/scripts 注入；请在 `factor_dev` 根目录执行 |
| `Not authorized to execute script files` | 账号不能 `run("/path.dos")` | 已改为 **Python 读本地 `.dos` 再 `session.run(脚本字符串)`**；勿改回服务端 `run(path)` |
| 误关共享 session | 对共享连接 `close()` | 本客户端默认不关共享；进程结束可用 `close_shared_ddb_session()` |

---

## 十二、扩展方向

- **Phase 2 逐笔**：用 CH Tick 实现真实中/小单占比、订单到达强度
- **更多盘口因子**：订单斜率、深度加权 OIR、撤单意图（可参考现有 `core/l2_features`）
- **流式**：`ddb_scripts/stream/*` + 实时 Snapshot
- **多因子合成**：在窄表宽化后做截面正交 / 等权合成，再进 `groupTest`
- **与生产栈对齐**：验证通过后把稳定因子迁入 `factor_formulas_l2_*` / TRACK

---

## Phase 2 进展（Tick 中/小单占比）

**实现**：ClickHouse **服务端 GROUP BY**（`ch_tick.fetch_tick_agg_by_date_range`），Python 只读 `(symbol,date)` 聚合结果。  
**已弃用**：逐日拉全量 Tick（原 ~78s/日）。

| 窗口 | 耗时 | RankIC | 决策 |
|------|------|--------|------|
| 单日冒烟 2024-01-02 | ~3s（新）/ ~78s（旧） | — | 与旧结果 corr≈0.9999 |
| 短窗 2024-01 | **~17s 聚合** | **-4.17%** | **归档，不跑全样本** |

详见 `research/results/l2_reproduction/phase2_verdict.md`。

---

## Phase 1 结论（已归档，2026-08-03）

**状态**：研究完成，**不迁入**主因子库 / 新 TRACK。

| 结论点 | 说明 |
|--------|------|
| 管道 | DDB 分钟宽表 → 日频窄表 → C2C 分组回测（`signal.shift(1)`）已跑通 |
| 毛端信息 | `avg_outflow_ratio` 年化约 25.7%、Sharpe 1.36（翻向后）；方向与研报相反（RankIC≈-3.2%） |
| 成本 | H-L 日换手倍数约 2.7–3.5；7.5bps 下年化隐含成本 >50%，**净年化为负** |
| 口径修正 | `Active_sell` 版 v2/v3 短窗仍负 IC，未对齐研报 +7.4% |
| 决策 | 分钟代理作研究方向归档；真对齐需 Phase 2 逐笔；`-avg_outflow_ratio` 可作 risk/剥离参考 |

标准化呈现表（C2C，250 日年化，换手为 L1 **倍数**）：

| factor | Ret_HL_bps | AnnRet | Sharpe | Turnover× | Cost | Net | flip |
|--------|------------|--------|--------|-----------|------|-----|------|
| avg_outflow_ratio | 10.26 | 25.65% | 1.36 | 2.74 | 51.40% | **-25.75%** | -1 |
| big_order_net_inflow | 2.87 | 7.18% | 0.49 | 3.51 | 65.85% | **-58.68%** | 1 |
| big_order_drive_ret | 6.60 | 16.49% | 1.13 | 3.44 | 64.44% | **-47.94%** | -1 |

最终产物目录：`research/results/l2_reproduction/`（含 `phase1_verdict.md`、`daily_factor_presentation.{csv,md}`、`phase1_net_after_fee.csv`）。  
重出报表：`python l2_factor_reproduction/scripts/export_daily_presentation.py`。

---

## 附录 A：因子清单速查

| 名称 | 类别 | 数据源 | 状态 |
|------|------|--------|------|
| `avg_outflow_ratio` | 分时成交 | MinuteBar | Phase1 可跑 |
| `big_order_net_inflow` | 分时成交 | MinuteBar | Phase1 可跑 |
| `big_order_drive_ret` | 分时成交 | MinuteBar | Phase1 可跑 |
| `net_order_change` | 盘口 | L2 Snapshot 本地表 | 需 ETL |
| `order_change_volatility` | 盘口 | 同上 | 需 ETL |
| `order_change_skew` | 盘口 | 同上 | 需 ETL |
| `mid_order_ratio` | 成交占比 | ClickHouse Tick | 已实现（严格成交过滤） |
| `small_order_ratio` | 成交占比 | ClickHouse Tick | 已实现（严格成交过滤） |

## 附录 B：与主库模块关系

| 本项目 | 复用自 |
|--------|--------|
| DDB 连接 | `factor_data_loaders.connect_ddb` → `core.ddb.connection` |
| 日期默认 | `factor_config.START_DAY/END_DAY` |
| 分组回测 | `Factor_Dev_Lib.groupTest` + 涨跌停/ST/停牌过滤 |
| 分钟字段口径 | `research/docs/data_source_handbook_v2.md` |

## 附录 C：完整脚本位置

本仓库已落地全部骨架与可运行脚本，路径即上文目录树；**无需另行下载**。

### 冒烟实测记录（2026-08-03）

短区间 `2024-01-02 ~ 2024-03-29`、中证1000：三因子各约 5.8 万行，管道通畅。

首次失败原因：`Not authorized to execute script files` → 已改为本地读 `.dos` 再 `session.run(字符串)`。

### 全样本评估（2021-01-01 ~ 2024-06-30，CSI1000，signal.shift(1)）

| 因子 | 窄表行数 | RankIC | ICIR | H-L年化(翻向) | Sharpe(翻向) | 原始方向 |
|------|----------|--------|------|---------------|--------------|----------|
| `avg_outflow_ratio` | 1,352,412 | **-3.17%** | -2.93 | 25.7% | 1.36 | 高值→低收益（与研报「高流出更好」相反） |
| `big_order_net_inflow` | 997,060 | -1.47% | -1.81 | 7.2% | 0.49 | 弱；分组非单调 |
| `big_order_drive_ret` | 997,060 | **-2.97%** | -3.92 | 16.5% | 1.13 | 高值→低收益 |

产物：`group_pnl.csv` / `group_turnover.csv` / `rank_ic.csv` / `summary.json`，总表 `phase1_summary.csv`。

说明：
1. 回测已按主库惯例做 **`signal.shift(1)`**，避免全日分钟因子与当日 c2c 看穿。
2. 大单因子全样本曾 OOM，已改为 **`percentile` + 按年分块**。
3. 日换手很高（H-L ≈ 2.7–3.5），Implied AnnuFee 很大，净收益需扣费再评。
4. 分钟代理 ≠ 研报逐笔口径；`avg_outflow_ratio` 的强负 IC 说明「下跌分钟成交占比」有信息，但符号与研报叙述不一致。

### 口径修正与扣费结论（2026-08-03）

详见文首 **「Phase 1 结论」** 与 `research/results/l2_reproduction/phase1_verdict.md`。
