# 数据资产清单（Data Inventory）

> 全库数据层审计：数据源 → 连接 → 表 → 字段 → 加载模块 → 已知坑。
> 生成时点：2026-08-04。更细的字段字典见 `research/docs/data_source_handbook_v2.md`（2026-07-31 审计版）。

---

## 1. 数据源总览

| 引擎 | 地址 / DSN | 库 | 连接常量 | 用途 | 状态 |
|------|-----------|----|----------|------|------|
| **ClickHouse** | `10.80.139.9:8123` | `cmds` | `DATA_DB_HFDATA` | L2 十档快照 / 逐笔 Tick / 高频 K 线 | 活跃（L2 唯一上游） |
| **DolphinDB** | `10.12.180.9:8902` (`pyread`) | dfs:// | `DATA_DB_CONN` | 分钟线、Wind EOD、估值市值、财报、指数权重、日历 | 活跃（生产默认路径） |
| Oracle Wind | `10.23.153.15:21010/wind` | wind | `DATA_DB_WIND` | 2018-09 前 EOD 补全 / 对照 | 活跃（历史补段） |
| Oracle 聚源 | `10.23.129.8/9:21010/juyuan` | jydb | `DATA_DB_JUYUAN(_2)` | 财务基本面 | 备用（字段字典待补） |
| Oracle 朝阳永续 | `10.23.129.89:1521/zyyx2` | zyyq | `DATA_DB_ZYYX2` | 卖方一致预期 | 备用（SUE 线将用） |
| Oracle 财汇 | `10.23.129.86:21001/FINCHINA2` | finchina | `DATA_DB_CAIHUI(_2)` | 财务/另类 | 备用 |
| Oracle 排排网/普益 | `10.23.129.42/44:21010/cmdssm` 等 | — | `DATA_DB_PAIPAI/PUYI(_2)` | 私募/理财参考 | 参考 |
| MySQL 通联 | `10.80.139.20:3306` | datayes | `DATA_DB_DATAYES` | 另类/指数 | 备用（审计确认无 LOB） |
| MySQL 东方财富 | `10.80.139.50:6030` | emdata | `DATA_DB_EMDATA(_2)` | 资金流 | 备用 |
| MySQL 野尘 | `10.63.95.30:3306` | yechen | `DATA_DB_YECHEN(_2)` | 产业链 | 备用 |

连接自检：`PYTHONPATH=. $PY DB_Demo.py`（`PY=/opt/conda/anaconda3/envs/base_93/bin/python`）。
权限：全部只读研究账号；密码在 `COMMON_CONST.py` / `.env`，勿提交 Git。

---

## 2. ClickHouse（`cmds`，44 表）

### 2.1 A 股表（项目在用）

| 表 | 粒度 | 关键字段 | 项目用途 |
|----|------|----------|----------|
| `SSE_AL_SSL2_EXG`（54 列） | ~3s 快照；A 股覆盖 2015-01-05 起 | `Symbol, ExchTime, BidPrices/AskPrices`(Array,1-indexed), `BidVolumes/AskVolumes`, `BidNums/AskNums`, `TotalBid/AskVolume`, `Bid/AskVWAP`, `Bid/AskWithdraw*`（撤单，**仅 SSE**）, `Price, AccVolume, AccAmount, TradeStatus, CHTime` | L2 盘口因子唯一上游（`research/l2_alpha/schema.py:7-14`） |
| `SZSE_AL_SSL2_EXG`（53 列） | ~3s 快照；A 股覆盖 2008-01-02 起 | 十档价量 Array；**无撤单列**、无对称 `BidNums/AskNums` 数组（有 `BidNum1` 等）；有效十档行的源 `BidVWAP/AskVWAP` 不可用 | `Type='010'` 的空/完整数组伴随行须先过滤；统一 VWAP 用十档价量自算 |
| `SSE_AL_TICK_EXG`（15 列核心） | 逐笔 | `Symbol, ExchTime, Price, Volume, Amount, BSFlag, BidOrderNo, AskOrderNo, SeqNo, SubSeqNo, Type` | 订单规模因子（`ch_tick.py`）；`Type='T'` 为成交 |
| `SZSE_AL_TICK_EXG`（37 列） | 逐笔 | 更丰富（含 `SecondaryOrderID` 等）；成交记录使用 `Type='011'` 且 `BidOrderNo>0 AND AskOrderNo>0`，金额用 `Price*Volume` | 同上 |
| `SSE_AL_KLIN_EXG`（19 列） / `SZSE_AL_KLIN_CMD`（10 列） | 高频 K 线 | `Open/High/Low/Close, Volume, Amount, PreClose, Average, IOPV` | 仅作 DDB 分钟线对照，生产不用 |

### 2.2 其他市场（存在，当前 ZZ1000 研究范围外）

- 期货：CFFEX / CZCE / DCE / SHFE / CUSE 各自的 `_AL_KLIN_RTH`（K 线）、`_AL_SSL1_RTH`（一档）、部分 `_AL_SSL2_EXG`（十档）
- 港股：`HKEX_EQ_KLIN_RTH / SSL1 / SSL2`
- 镜像：上述大多有 `LOCAL_*` 前缀镜像表，**优先用非 LOCAL 生产表**

### 2.3 ClickHouse 访问与封装

| 封装 | 位置 | 说明 |
|------|------|------|
| 原始连接 | `COMMON_CONST.DATA_DB_HFDATA` + `clickhouse_connect.get_client` | 只读 |
| SSL2 分钟特征 | `research/l2_alpha/clickhouse_ssl2.py::extract_minute_features` | CH 内 array 运算 + 分钟 argMax，输出窄表 `tradetime/symbol/factorname/value` |
| Tick 订单规模聚合 | `l2_factor_reproduction/python/ch_tick.py::fetch_tick_agg_by_date_range / fetch_tick_bucketed` | CH 服务端 GROUP BY；订单分档：小 ≤4万 < 中 ≤20万 < 大 ≤100万 < 超大 |
| 数据面板导出 | `research/export_l2_panel_csi1000.py` → `research/results/l2_factor_panel_csi1000/` | CSI1000 L2 面板 |

**要点**：时间一律 `ExchTime`（DateTime64, Asia/Shanghai）；过滤窗口 + `Symbol` + 按日切片是硬性要求；数组字段 1-indexed，禁止虚构 `BidPrice0..9` 展开列；全市场单日体量大，必须服务端聚合。

---

## 3. DolphinDB（生产默认路径）

统一入口：`core/ddb/connection.py::get_ddb_session()`（进程级共享 session，禁止直接 `close()`；旧别名 `factor_data_loaders.connect_ddb`）。

### 3.1 分钟线（日内因子主数据）

| 表 | 行数（审计） | 区间 | 说明 |
|----|-------------|------|------|
| `dfs://QV_Trade_to_MinuteBar/Stock_one_minute` | ~2.14e9 | 2018-09-03 → 2026-07-30（审计日） | 1 分钟 bar，6026 标的，22 列 |
| 同库 `Fund_one_minute` / `Cbond_one_minute` | — | 2018-09 → 2025-08 | 基金/可转债 |
| `Future_one_minute` | — | — | **当前账号不可读** |

关键字段：`Symbol`（如 `600000.SH`）, `Date, Bartime, Barstart/Barend`, `OHLC`, `Volume(股), Amount(元)`, `Active_buy/sell_amount/volume/count`（主动买卖，分钟聚合代理，非逐笔）, `Bid/Ask_cancel_volume/count`（撤单代理）, `Adjfactor`（NULL/0 按 1.0）。

加载层：
- `minute_bar_store.py`（MinuteBarStore）：按需查询、防全历史误拉，分钟研究统一入口（`factor_config.MINUTE_BAR_HISTORY_START=2020-01-01`，超时 120s、重试 3 次）
- `core/ddb/minute.py` / `core/data/panel_reader.py::get_minute_panel`
- `l2_data_loaders.py`：分钟→日频聚合（active_* / cancel_* / imbalance_duration），parquet 缓存在 `research/cache/l2_daily/`
- ⚠️ DDB **没有**原生 L2 快照（无 `dfs://QV_Snapshot`），盘口必须走 ClickHouse

### 3.2 Wind EOD 日频（EOD 因子主面板）

| 表 | 关键字段 | 用途 |
|----|----------|------|
| `dfs://WIND.ASHAREEODPRICES/data` | `S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN/HIGH/LOW/CLOSE, S_DQ_PRECLOSE, S_DQ_VOLUME(手), S_DQ_AMOUNT(千元), S_DQ_PCTCHANGE, S_DQ_ADJFACTOR, S_DQ_TRADESTATUS, S_DQ_LIMIT/S_DQ_STOPPING(涨跌停价)` | EOD 因子/回测；loader `factor_data_loaders.load_eod_wide_tables` / `core/ddb/eod.py::fetch_eod_long` |
| `dfs://WIND.ASHAREEODDERIVATIVEINDICATOR/data` | `S_VAL_MV(总市值), S_DQ_MV(流通市值), S_VAL_PE_TTM, S_VAL_PB_NEW, S_VAL_PS_TTM, S_DQ_TURN(换手)` | 市值中性化（用 `S_VAL_MV`）、估值因子；loader `load_derivative_wide_tables` |
| `dfs://WIND.ASHARETTMHIS/data` | `ANN_DT(公告日), REPORT_PERIOD, S_FA_ROE_TTM, S_FA_GROSSMARGIN_TTM, ...` | 财报 TTM 面板，按公告日对齐防前视；loader `load_financial_ttmhis_long` |
| `dfs://WIND.AINDEXHS300WEIGHT / AINDEXCSI500WEIGHT / AINDEXCSI1000WEIGHT` | `TRADE_DT, S_CON_WINDCODE` | 沪深300/500/1000 日频成分 mask（`Factor_Dev_Lib.get_index_member_mask`，lines 877-911） |
| `dfs://WIND.AINDEXMEMBERS` | `S_CON_INDATE/S_CON_OUTDATE` | 其他指数成分（入/剔日期展开，lines 913-955） |
| `dfs://WIND.AINDEXEODPRICES` | 指数日频 | 基准收益（`get_Ret_Matrix(base_index=...)`） |
| `dfs://WIND.ASHARECALENDAR` | 交易日历 | 交易日期对齐 |
| `dfs://WIND.ASHAREPREVIOUSNAME` | `S_INFO_NAME` 历史名称 | ST/退 过滤（`Factor_Dev_Lib.py:131-219`） |
| `dfs://WIND.ASHAREINDUSTRIESCLASSCITICS` | `CITICS_IND_CODE`（前 4 位=一级） | **中信一级行业**，中性化用（`Factor_Dev_Lib.py:675-743`） |

### 3.3 Wind Oracle（2018-09 前补段）

`factor_data_loaders.load_eod_wide_tables_from_wind_oracle`：按月切片拉 `wind.ASHAREEODPRICES` → 本地 parquet → pivot；返回宽表 + c2c 收益（`S_DQ_CLOSE/S_DQ_PRECLOSE-1`）。注意 Oracle 侧 `TRADE_DT` 是 `YYYYMMDD` 字符串过滤，DDB 侧是 `YYYY.MM.DD` DATE。

---

## 4. 本地缓存与中间层

| 缓存 | 路径 | 说明 |
|------|------|------|
| L2 日频聚合 | `research/cache/l2_daily/*.parquet` | `l2_data_loaders` 自动生成 |
| L2 因子面板 | `research/results/l2_factor_panel_csi1000/` | CSI1000 分钟特征面板 |
| mid_order_ratio 分桶缓存 | `research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/*.parquet` | CH bucketed 聚合按季切片 |
| 分钟巨表 | 根目录 `intraday*.parquet`（合计 ~25G） | **过渡产物**；`intraday.parquet` 是不完整子集，勿当权威源；权威路径 = MinuteBarStore 按需查 DDB |
| 旧输出根 | `result/`（~1.2G） | 旧 runner 产物；新研究写 `research/results/` |

---

## 5. 股票池与过滤（数据口径速查）

| 项 | 口径 | 代码位置 |
|----|------|----------|
| CSI300/500/1000 | Wind 日频权重表，point-in-time | `Factor_Dev_Lib.py:877-911` |
| 全 A | 不加指数 mask；代码首位 `0/3/6` | `factor_config.py:241-247`、`factor_runner.py:161-180` |
| 涨跌停过滤 | `S_DQ_CLOSE < S_DQ_LIMIT and > S_DQ_STOPPING` | `Factor_Dev_Lib.py:109-128` |
| ST/退 | 历史名称表 regex `ST\|退` | `Factor_Dev_Lib.py:131-219` |
| 停牌 | `S_DQ_TRADESTATUS` 停牌/空 | `Factor_Dev_Lib.py:222-250` |
| IPO/次新 | 非默认；investability 层 close 非空累计 ≥60 天 | `alpha_investability.py:20-54` |

---

## 6. 已知坑（审计结论）

1. **L2 只在 ClickHouse**：DDB 无 snapshot；Oracle `%L2%` 对象名命中是假阳性。
2. **SSE/SZSE schema 不对称**：深市 SSL2 无撤单列；深市 Tick 无 `Type` 字段、金额须 `Price*Volume`。
3. **分钟 `Active_*`/cancel 是聚合代理**，非逐笔事件，与 CH Tick 口径不可直接混用验证。
4. **EOD 量额单位**（手/千元）与分钟（股/元）不一致，跨频对齐须换算。
5. **`Adjfactor` NULL/0 → 1.0**；EOD 区分未复权 `S_DQ_CLOSE` 与复权价。
6. **字段拼写历史包袱**：CH 存在 `ToltalBidNum` 类拼写，以 schema 原文为准。
7. **财务数据用公告日 `ANN_DT` asof 对齐**，禁止报告期期末向未来填（前视）。
8. **SSL2 实测覆盖**（2026-08-05 inventory）：SSE A 股
   `2015-01-05 09:14:47` → `2026-08-04 15:31:15.410000`；SZSE A 股
   `2008-01-02 09:25:00` → `2026-08-04 15:35:00`。详见
   `research/results/l2_reproduction/primitives/order_book_daily/`。
