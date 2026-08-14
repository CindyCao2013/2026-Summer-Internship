# GNN Feature Validity Audit

> **PROJECT CLOSED — Gate 1 Complete**  
> **Decision: Do not proceed to GNN Phase 2**  
> Formal closeout: [`GATE1_CLOSEOUT.md`](GATE1_CLOSEOUT.md)  
> Canonical results: `results/summary.csv`, `results/report.md`

独立检验 `quantskills/skill-dl-gnn-stock-graph` 五维原始特征是否具有 A 股 PIT 截面选股能力。第一阶段不训练 GNN、不调参、不搜索窗口、不复制 GPL 代码。

Gate 1 已回答核心问题：当前可实现的五维原始特征**不足以**支持进入 GNN 复现。本仓库冻结，不再扩展特征池、不再搜索图参数、不再训练神经网络。

## 冻结口径

- 样本：2021-01-01 至 2024-06-30。
- 主股票池：CSI1000 Wind 日频 PIT 成分。
- 过滤：历史 ST/退、停牌、收盘涨跌停、累计有效收盘不足 60 日。
- 收益：公司 c2c 日收益，T 日收盘后因子固定映射到 T+2 行收益；同时要求 T+1 入场和 T+2 退出可交易。
- 年化：公司标准 250 日。
- 预处理：先应用 T 日 PIT 股票池/交易状态，再执行无穷转缺失、公司 MAD-tanh、截面 z-score、冻结的行业/市值中性化和最终 z-score。
- 模糊方向：最前 30% 时间平均 RankIC 冻结，校准期末设置覆盖 T+2 标签的 embargo；后 70% 才用于正式评价。
- PASS：年化 H-L ≥20%、Sharpe ≥2.0、十分组 Spearman 单调性 ≥0.70，严格 AND。
- 每个可测试候选只生成 `cumulative_hl.png` 和 `decile_bar.png`。

完整上游差异、target 和 timing 问题见 `source_audit.md`；候选池和方向规则见 `factor_registry.csv`。

## 公司数据复用

连接统一使用 `core.ddb.connection.get_ddb_session()`，项目中没有账号、密码、IP 或第二套连接配置。

- `dfs://WIND.ASHAREEODPRICES/data`
- `dfs://WIND.ASHAREEODDERIVATIVEINDICATOR/data`
- `dfs://WIND.ASHARETTMHIS/data`
- `dfs://WIND.AINDEXCSI1000WEIGHT/data`
- `dfs://WIND.AINDEXEODPRICES/data`
- `dfs://WIND.ASHARECALENDAR/data`
- `dfs://WIND.ASHAREPREVIOUSNAME/data`
- `dfs://WIND.ASHAREINDUSTRIESCLASSCITICS/data`

财务值按 `ANN_DT` 后的下一交易日可用，报告期只用于同比匹配。Live audit 显示这些表的 `OPDATE` 通常比经济日期晚数年，属于仓库维护时间而非首次可用时间，因此只用于审计；遇到重复版本时保留最早记录，且报告明确披露公司表不是不可变历史快照。

## 运行

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

$PY sideprojects/gnn_feature_validity_audit/run.py \
  --config sideprojects/gnn_feature_validity_audit/config.yaml
```

可选模式：

```bash
$PY sideprojects/gnn_feature_validity_audit/run.py --config sideprojects/gnn_feature_validity_audit/config.yaml --audit-only
$PY sideprojects/gnn_feature_validity_audit/run.py --config sideprojects/gnn_feature_validity_audit/config.yaml --factor mom_20d
$PY sideprojects/gnn_feature_validity_audit/run.py --config sideprojects/gnn_feature_validity_audit/config.yaml --family fundamental
$PY sideprojects/gnn_feature_validity_audit/run.py --config sideprojects/gnn_feature_validity_audit/config.yaml --all
```

同一 config、冻结 registry、实现代码和上游 commit 的 cache hash 已成功，且 summary、report、registry、图诊断和 PNG 契约全部一致时才默认跳过；需要显式重跑可加 `--force`。

动态图优先使用可选的 `dtaidistance`；缺失时由本项目在系统临时目录编译独立的精确 DTW 内核，最多使用 10 个线程。编译失败或图构建不满足 PIT/动态/非空要求时，关系图因子 fail closed 为 `DATA_UNAVAILABLE`，不会改用近似距离或全零图。

## 测试

```bash
PYTHONPATH=. $PY -m pytest -q \
  sideprojects/gnn_feature_validity_audit/tests/test_validity.py
```

测试覆盖 PIT 输入、公告日财务与版本、合格截面预处理、T+1/T+2 可交易性、calibration embargo、未来边、动态图节点退出、宏观不可截面测试、原始二元变量、十分组标签、H-L、年化收益、Sharpe、带符号 Spearman、严格 AND、方向与 composite 冻结、缓存产物、两图限制和确定性。

## 输出

```text
results/
├── summary.csv
├── report.md
├── manifest.json
├── graph_diagnostics.csv        # 仅真实动态图成功时
└── factors/<factor_id>/
    ├── cumulative_hl.png
    └── decile_bar.png
```

`DATA_UNAVAILABLE`、`UNTESTABLE` 等候选不生成模拟图。`summary.csv` 和 `report.md` 始终保留本次所选候选的完整结果。

## 结项摘要

- Atomic PASS：`amplitude`、`turnover`、`volatility_20d`（同属低风险簇，不是三个独立 Alpha）。
- Composite PASS：`fundamental_equal_weight`（单因子均未过关，需另立项目拆解，不在本仓库继续）。
- Relation / GNN：当前 PIT 图特征全部未通过；**不进入 Phase 2**。
- Sentiment：数据不可得；Macro：不可做独立截面十分组。

详细判定与禁止事项见 `GATE1_CLOSEOUT.md`。
