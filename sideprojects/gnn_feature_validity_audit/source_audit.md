# GNN 五维特征有效性审计：Source Audit

审计日期：2026-08-05  
上游仓库：<https://github.com/quantskills/skill-dl-gnn-stock-graph>  
上游分支：`main`  
上游 commit：`eacaa0edfd30a307a874354f4297428fea5f538e`  
许可证：GPL-3.0

本项目只根据公开定义进行独立实现，不复制上游 GPL 源码、测试或模型代码。公司数据访问继续复用现有只读连接与加载层。

## 1. 上游声明的五组特征

`SKILL.md` 声明五个维度：

1. price-volume：`ret`、`log_vol`、`amplitude`、`turnover`、`gap`、`dist_limit_up`、`dist_limit_down`、`excess_ret`、`mom_5d`、`mom_20d`、`volatility_20d`、`macd`、`rsi`、`money_flow`。
2. fundamental：`pe_ttm`、`pb`、`roe_ttm`、`market_cap`、`revenue_growth_yoy`、`profit_growth_yoy`、`debt_ratio`。
3. sentiment：`news_sentiment`、`lhb_flag`、`block_trade_premium`。
4. macro：`gdp_yoy`、`cpi_yoy`、`pmi`、`m2_yoy`。
5. relation：`degree_centrality`、`pagerank`、`dtw_similarity_mean`、`industry_excess_ret`。

文档声称每个样本为 `20 × 14 + 7 + 3 + 4 + 4 = 298` 维，并称所有输入严格使用 `date < T`。

## 2. 文档公式与可执行实现

### 2.1 Price-volume

基础日频公式与 `SKILL.md` 大体一致，但可执行实现存在以下实质差异：

- `volatility_20d` 实际使用 pandas 样本标准差（`ddof=1`），且只要求 5 个观测；字段指南却写 population std，名称又暗示完整 20 日。
- `macd` 实际为 `(EMA12(close) - EMA26(close)) / close`；`SKILL.md` 和字段指南写未除以价格的 MACD。
- `rsi` 实际对日收益的正负部分做 14 日简单移动平均；当平均损失为零时返回 50，而标准 RSI 通常应趋近 100。
- 5 日、20 日和技术指标前段缺失值在上游实现中被填为 0。
- 上游清洗器先对原始 OHLCV 等数值列做全窗口 1%/99% 截尾，而后续 pipeline 并未按文档再次对特征做 train-only winsorization。

本项目选择规则：以 `scripts/features/price.py` 的可执行公式为主定义，包括归一化 MACD、样本标准差及 RSI 算法；但滚动窗口不足和不可得值保留为缺失，不以 0 伪造观测。该缺失处理修正不改变因子公式和参数。

### 2.2 Fundamental

字段指南给出的经济定义比可执行实现更明确；可执行实现存在无法接受的时间与计算问题：

- 上游只按一个通用 `date` 字段合并财报，没有验证该字段是公告日。
- 对已经按日 forward-fill 的财务数值再做 `rolling(4)` 和 `shift(4)`；这相当于 4 个交易日而非 4 个季度。
- `roe_ttm` 实际使用上述滚动利润除以期末权益，而字段指南写 TTM 归母净利润除以平均权益。
- 营收和利润同比的实现同样在日频 forward-fill 序列上错用 4 日位移。
- `pe_ttm` 仅保留正利润公司，文档未说明该过滤。

本项目选择规则：优先使用公司 Wind 已计算的 PIT 日频 `S_VAL_PE_TTM`、`S_VAL_PB_NEW` 和公告日 TTM 字段；增长率按公告时点可见的同口径历史报告计算，不复刻上游的“4 个交易日”错误。所有财务值以 `ANN_DT` 进入可用集，`REPORT_PERIOD` 只用于寻找同比基期，绝不作为可用日期。

### 2.3 Sentiment

文档称无数据保持 NaN、不 forward-fill；实际实现却：

- 将未出现龙虎榜的股票填为 0；
- 大宗交易缺少溢价字段时直接填 0；
- 将 sentiment 全量按股票 forward-fill；
- 最后把所有剩余缺失填为 0；
- 单次扫描只读取最近约 5 个交易日的龙虎榜和大宗交易。

这会把“未知/未覆盖”误解释成“无事件”，并把一次事件广播到以后日期。本项目不沿用该处理。无连续 PIT 数据时 fail closed；二元或低唯一值事件不强行做十分组。

### 2.4 Macro

上游提供了宏观数据 loader，`SKILL.md` 和参考文档也声明 4 个宏观特征；但是 `scripts/features/pipeline.py` 的 `ALL_FEATURE_NAMES` 只包含 14 个 price、7 个 fundamental、3 个 sentiment 和 4 个 relation，共 28 个原始特征。宏观 loader 没有被 `scan.py` 调用，宏观特征没有进入当前 pipeline。

此外，宏观变量在同一交易日对所有股票相同，不具备独立截面排序能力。本项目将四项全部标记为 `NOT_TESTABLE_CROSS_SECTIONALLY`，不生成十分组或 macro composite。

### 2.5 Relation

上游关系实现与其“动态、无未来信息”的声明不一致：

- `scan.py` 在扫描日只构造一次图，并把同一张图同时用于所有历史训练样本和扫描样本。
- 隐式图读取窗口包含 `scan_date`；这与文档要求的严格 `date < T` 不一致。
- pipeline 将扫描日的一组 relation 值广播到所有历史日期，因此 relation features 不随日期更新。
- `dtw_similarity_mean` 没有从 DTW 边传给 feature 函数，当前 pipeline 中恒为 0。
- 缺失图、缺失 DTW 和缺失行业收益均被填 0，而不是 fail closed。
- 行业、概念及股东 loader/edge builder 没有在建边处执行生效日 as-of 过滤；当前快照可能被用于历史。
- 上游只记录各关系类型的边数，未记录每日节点数、总边数和孤立节点比例。
- `degree_centrality` 基于已经对称归一化的加权邻接矩阵行和再除以最大行和，不是普通 `(degree / (N-1))`。

本项目不会广播 scan-date 图。关系拓扑按冻结配置每 20 个交易日刷新，并在两个 PIT 快照之间向前保持；任一 T 使用的最近快照只含 `<= T` 且在信号形成时已知的数据。无法验证生效日时标记 `DATA_UNAVAILABLE`。历史收益关系只使用滚动历史窗口。每个刷新快照写入节点数、边数和孤立节点比例。全零图不算可用图。

## 3. 实际进入上游 pipeline 的特征组

实际进入 `FeatureBundle` 的原始列为：

- 14 price-volume；
- 7 fundamental；
- 3 sentiment；
- 4 relation；
- 0 macro。

因此当前 pipeline 是 28 列，而不是五维全量 32 列。它又把全部 28 列在 20 日窗口上展开，实际扁平输入为 `20 × 28 = 560`，不是文档所写的 298。基本面、情绪和关系特征也被重复堆叠 20 次。

## 4. Target 与 execution timing 审计

上游 `_build_targets` 先构造每只股票的 `close(T+1) / close(T) - 1`，但随后将一个训练窗口内同一股票的所有特征窗口取均值，并把所有 next-day return 也按股票取均值。模型最终面对的是“每个节点一个历史平均特征和一个历史平均收益”，不是标准的每日 `(date, symbol)` 截面标签。

上游 execution 还有三处冲突：

1. 文档要求训练和打分严格使用 `date < T`，但 scoring window 明确包含 `date <= scan_date`。
2. 回测引擎用 T 日选股结果按 T 日收盘价成交；T 日收盘数据形成信号后无法再按同一收盘价成交。
3. 引擎的 `T+1` 仅禁止买入当日卖出，并未把订单实际延迟到下一交易日；引擎也没有执行其文档所称的涨跌停约束。

本项目不复用上游 target 或 execution engine。公司标准日收益是
`S_DQ_CLOSE / S_DQ_PRECLOSE - 1`，现有框架通常将信号 shift 1 行后与该收益对齐；但对收盘后 EOD 信号，这仍隐含按信号日收盘成交。为避免该问题，本项目冻结为：T 日收盘后形成的因子使用 T+2 行的 c2c 收益，即在下一个交易日收盘建立理论持仓，再持有一个交易日；回测截面同时要求 T+1 入场和 T+2 退出可交易。方向校准期末另留出覆盖完整 T+2 标签的 embargo。该固定延迟不做搜索，并在 `config.yaml` 和结果中记录。

## 5. 公司数据映射（已核实）

以下表名和字段来自现有加载代码、数据资产清单及 2026-08-05 只读 live schema 检查，不含猜测字段。

| 用途 | 公司表/共享对象 | 已核实字段或函数 | PIT 处理 |
|---|---|---|---|
| 共享连接 | `core.ddb.connection.get_ddb_session()` | `COMMON_CONST.DATA_DB_CONN` | 不新建连接层，不复制凭据 |
| 日频 OHLCV | `dfs://WIND.ASHAREEODPRICES/data` | `TRADE_DT, S_INFO_WINDCODE, S_DQ_PRECLOSE, S_DQ_OPEN/HIGH/LOW/CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT` | 按交易日读取未复权历史字段 |
| 涨跌停/停牌 | 同上 | `S_DQ_LIMIT, S_DQ_STOPPING, S_DQ_TRADESTATUS` | 复用历史日状态 |
| 换手/估值/市值 | `dfs://WIND.ASHAREEODDERIVATIVEINDICATOR/data` | `S_DQ_TURN, S_VAL_PE_TTM, S_VAL_PB_NEW, S_VAL_MV, S_DQ_MV, OPDATE` | 每日经济日期为 `TRADE_DT`；同一键如有重复只保留最早记录 |
| ROE/增长/负债 | `dfs://WIND.ASHARETTMHIS/data` | `ANN_DT, REPORT_PERIOD, S_FA_ROE_TTM, TOT_OPER_REV_TTM, OPER_REV_TTM, NET_PROFIT_PARENT_COMP_TTM, S_FA_DEBTTOASSETS_MRQ, STATEMENT_TYPE` | `ANN_DT` 后下一交易日可用；`REPORT_PERIOD` 只用于同比匹配；重复键保留最早记录 |
| 交易日历 | `dfs://WIND.ASHARECALENDAR/data` | `TRADE_DAYS, S_INFO_EXCHMARKET`；`get_TradingDay` | 公司统一 SSE 日历 |
| 主股票池 | `dfs://WIND.AINDEXCSI1000WEIGHT/data` | `TRADE_DT, S_CON_WINDCODE`；`get_index_member_mask("000852.SH", ...)` | 日频 PIT 成分 |
| ST/退 | `dfs://WIND.ASHAREPREVIOUSNAME/data` | `BEGINDATE, ENDDATE, S_INFO_NAME`；`get_EOD_Not_ST` | 历史名称区间 |
| 行业 | `dfs://WIND.ASHAREINDUSTRIESCLASSCITICS/data` / `PREHEAT_IND_DATA_CITICS` | `CITICS_IND_CODE, ENTRY_DT, REMOVE_DT` | 按生效/移除区间展开 |
| 中性化 | `Factor_Dev_Lib.cs_neutral_size_ind` | 日截面 OLS：行业哑变量 + 标准化 log 总市值 | market_cap 和 industry_excess_ret 按规则例外 |
| forward return | `Factor_Dev_Lib.get_Ret_Matrix(method="c2c")` | `S_DQ_CLOSE/S_DQ_PRECLOSE-1` | 本项目固定两交易行信号延迟 |
| 分组参考 | `Factor_Dev_Lib.groupTest` | 等数量、组内等权、`H-L=Q10-Q1` | 不复用其 `rank(method="first")` 强制破平局和事后翻向 |

主股票池采用 CSI1000 PIT：这是当前 `l2_factor_reproduction/config/settings.py` 的 canonical universe。ALL/CSI300/CSI500 只保留为框架可支持项，本阶段不为提高结果切换主股票池。

Live audit 显示 EOD、衍生、财务和行业表的 `OPDATE` 常比经济日期晚数年，不能解释为首次可用时间。本项目因此使用已核实的经济可用字段（行情/估值的 `TRADE_DT`、财务的 `ANN_DT`、行业的 `ENTRY_DT/REMOVE_DT`），`OPDATE` 只用于披露和重复版本的最早记录选择。由于这些是最终 vendor 表而非不可变日快照，若原记录已被原位覆盖，则无法完全重建修订前值；这是结论的明确 PIT 限制，不会被描述为已验证的 immutable snapshot。

## 6. 尚未核实或当前不可用的数据

- 公司仓库和可见 DolphinDB 资产中尚未找到连续 PIT `news_sentiment`。
- 尚未找到已验证的龙虎榜日表和大宗交易日表映射；不得猜表名。
- 尚未找到带有效期、可历史重建的股东共同持仓或供应链关系表映射。
- 当前中信行业历史区间可支持行业关系；历史收益可支持滚动 DTW。是否将这两类关系组合为 degree/PageRank，必须以每日图审计通过为前提。
- 公司现有次新过滤是“累计有效收盘日 >= 60”的代理，不是已核实的上市日期表；本项目会明确记录该限制。

缺失映射的因子在运行时标记 `DATA_UNAVAILABLE`，不会生成全零代理或模拟结果。

## 7. 本项目的防前视与独立检验规则

1. 行情/估值使用 `TRADE_DT`，财务使用 `ANN_DT`，行业使用生效/移除日期；重复经济键只保留最早记录，OPDATE 限制单独披露。
2. 财务数据只按 `ANN_DT` 后下一交易日 as-of，禁止按报告期直接 forward-fill。
3. 先应用 T 日 PIT 成分、ST、停牌、涨跌停和新股掩码，再做 MAD、标准化与中性化，样本外股票不能改变截面残差。
4. 图边必须带 `effective_from/effective_to` 或由截至 T 的历史滚动窗口计算；每个 relation 因子分别验证时间变化。
5. 图刷新时重置节点集合，退出当前快照的股票不得继承上一快照的 PageRank/DTW 值。
6. 收盘后因子采用固定两交易行 c2c 延迟，并检查 T+1 入场和 T+2 退出可交易性。
7. calibration 与 evaluation 之间设置覆盖 forward-return lag 的 embargo。
8. 宏观变量直接标记 `NOT_TESTABLE_CROSS_SECTIONALLY`。
9. 二元/低唯一值先按原始因子检查；覆盖不足或有效日不足标记为对应的 UNTESTABLE 状态。
10. 因子方向和 composite 成员只来自预先经济规则或 calibration period；evaluation 结果不能反向修改方向或成员。
11. 任一候选出现 `ERROR` 时禁止发布 success manifest、最终报告或缓存命中。
12. 不调用上游模型、target、回测或绘图代码；未真实读取公司数据、未通过审计时不生成有效性结论。
