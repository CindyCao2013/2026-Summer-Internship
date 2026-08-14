# 数据源研发手册 v2

> 面向日常因子研发：有哪些数据可用、在哪里、怎么用、有哪些坑。  
> 信息来源：`multi_db_data_audit_v1.md` / `l2_data_audit_v1.md` / `l2_ssl2_feature_extractor_v1.md` / `COMMON_CONST.py` / `DB_Demo.py` / `factor_data_loaders.py`。  
> 审计时点（UTC）：`2026-07-31T03:21:03`。密码不写入本文档。

**项目根路径约定**

| 变量 | 含义 |
|------|------|
| `${PROJ_ROOT}` | `/home/SiYangCao/factor_dev/factor_research0703/factor_dev` |
| 连接常量 | `${PROJ_ROOT}/COMMON_CONST.py` |
| 连接 Demo | `${PROJ_ROOT}/DB_Demo.py` |
| 审计 JSON | `${PROJ_ROOT}/research/results/multi_db_data_audit.json` |

> ⚠️ **权限**：所有源库账号为只读研究账号（如 DolphinDB `pyread`、Oracle/MySQL/CH 用户 `lqtz`）。写库 / 高权限需向数据组/IT 申请。密码放在 `${PROJ_ROOT}/.env` 或 `COMMON_CONST.py`，勿提交到 Git。

**原始数据 vs 衍生因子数据**

| 类型 | 示例 | 说明 |
|------|------|------|
| 原始行情 | DDB 分钟线、CH SSL2/TICK、Wind EOD OHLC | 上游源表，优先使用 |
| 衍生/缓存 | `FactorDataCache`（`ret_1d` 等）、`l2_*` 分钟特征 | 由原始数据计算，不可替代源表 |
| 参考/另类 | 聚源财务、朝阳一致预期、排排/普益 | 按研究主题选用，字段字典多数待补齐 |

---

## 1. 数据源总览

排序：核心因子数据 > 基础行情 > 财务/另类 > 参考数据。

| 数据源名称 | 存储类型 | 物理路径/连接地址 | 更新频率 | 关键字段（≤5） | 数据起始时间 | 适用场景 |
|------------|----------|-------------------|----------|---------------|--------------|----------|
| ClickHouse A股 L2 Snapshot | DB (ClickHouse) | `DATA_DB_HFDATA` → `10.80.139.9:8123` / db=`cmds`；表 `SSE_AL_SSL2_EXG` / `SZSE_AL_SSL2_EXG` | 日内实时/盘后落库 `[TODO: 待补充精确延迟，建议向数据组确认]` | `Symbol`, `ExchTime`, `BidPrices`, `AskPrices`, `TotalBidVolume` | `[TODO: 待补充]` | L2 因子 / 微观结构 |
| ClickHouse A股 Tick | DB (ClickHouse) | 同上；表 `SSE_AL_TICK_EXG` / `SZSE_AL_TICK_EXG` | 同上 | `Price`, `Volume`, `BSFlag`, `BidOrderNo`, `AskOrderNo` | `[TODO: 待补充]` | Tick 因子（Phase 2） |
| DolphinDB 股票分钟线 | DB (DolphinDB) | `DATA_DB_CONN` → `10.12.180.9:8902`；`dfs://QV_Trade_to_MinuteBar/Stock_one_minute` | 日更（审计日最新 `2026-07-30`） | `Symbol`, `Date`, `Bartime`, `Close`, `Active_buy_amount` | `2018-09-03` | 日内因子 / 回测评估默认路径 |
| DolphinDB Wind EOD | DB (DolphinDB) | `dfs://WIND.ASHAREEODPRICES/data`（同 `DATA_DB_CONN`） | 日频 | `S_INFO_WINDCODE`, `TRADE_DT`, `S_DQ_CLOSE`, `S_DQ_VOLUME`, `S_DQ_ADJFACTOR` | `[TODO: DDB 分区起始日待确认]`；Oracle 可覆盖更早 | EOD 因子 / 回测 |
| Wind Oracle EOD | DB (Oracle) | `DATA_DB_WIND` → `10.23.153.15:21010/wind`；表 `wind.ASHAREEODPRICES` | 日频 | `TRADE_DT`, `S_INFO_WINDCODE`, `S_DQ_CLOSE`, `S_DQ_PRECLOSE`, `S_DQ_AMOUNT` | `[TODO: 待补充]` | 补全 DDB 2018 前 / 对照 |
| ClickHouse HF K线 | DB (ClickHouse) | `cmds.SSE_AL_KLIN_EXG` / `SZSE_AL_KLIN_CMD` | 高频 bar | `Open/High/Low/Close`, `Volume`, `Amount` | `[TODO: 待补充]` | 与 DDB 分钟路径对照 |
| 聚源 Fundamentals | DB (Oracle) | `DATA_DB_JUYUAN` / `_2` → `10.23.129.8/9:21010/juyuan` | `[TODO]` | 样例表 `jydb.secumain` | `[TODO]` | 财务因子 |
| 朝阳永续一致预期 | DB (Oracle) | `DATA_DB_ZYYX2` → `10.23.129.89:1521/zyyx2` | `[TODO]` | 样例 `zyyq.con_forecast_stk` | `[TODO]` | 预期类因子 |
| 财汇 | DB (Oracle) | `DATA_DB_CAIHUI` → `.../FINCHINA2`；replica `DATA_DB_CAIHUI_2` | `[TODO]` | 样例 `finchina.TQ_OA_STCODE` | `[TODO]` | 财务/另类 |
| 通联 Datayes | DB (MySQL) | `DATA_DB_DATAYES` → `10.80.139.20:3306` / `datayes` | `[TODO]` | 样例 `datayes.con_index` | `[TODO]` | 另类/指数 |
| 东方财富 EMData | DB (MySQL) | `DATA_DB_EMDATA` → `10.80.139.50:6030` / `emdata` | `[TODO]` | 样例 `emdata.fund_bs_cfinfo` | `[TODO]` | 资金流等 |
| 野尘 Yechen | DB (MySQL) | `DATA_DB_YECHEN` → `10.63.95.30:3306` / `yechen` | `[TODO]` | 样例 `yechen.industry_chain` | `[TODO]` | 产业链另类 |
| 排排网 | DB (Oracle) | `DATA_DB_PAIPAI` → `.../cmdssm` | `[TODO]` | 样例 `java.pvn_fund_info` | `[TODO]` | 私募参考 |
| 普益 | DB (Oracle) | `DATA_DB_PUYI` → `.../puyi` | `[TODO]` | 样例 `pystandard.bank_base_info` | `[TODO]` | 银行理财参考 |
| 通用 ORCL | DB (Oracle) | `DATA_DB_ORCL` → `172.22.133.52:1521/ORCL` | `[TODO]` | `[TODO]` | `[TODO]` | 通用查询 |
| 本地 Feather/Parquet 缓存 | Feather / Parquet | 建议 `${PROJ_ROOT}/research/cache/` 或临时目录 | 按任务生成 | 任务自定义 | N/A | 加速二次读取 |
| EOD 衍生缓存（非源表） | 内存 / 计算层 | `FactorDataCache` / `eod_data_foundation.py` | 随加载窗口计算 | `ret_1d`, `volatility_20d`, `amount_mean_20d` 等 | 随 EOD 窗口 | 因子特征，非原始数据 |

> ⚠️ Oracle 端点在审计中出现 `%L2%` / `%TICK%` 对象名命中，**均为假阳性**，不是 A 股 LOB 源。真正的 L2 在 ClickHouse `cmds`。

---

## 2. 各数据源详细说明

### 2.1 ClickHouse A股 L2 Snapshot（核心）

**角色**：A 股原生十档盘口 snapshot；Sprint 4.4 Phase 1 L2 因子唯一上游。

#### 访问方式

依赖：`clickhouse_connect`, `pandas`（Python 3.8+）

```python
# 连接参数（脱敏）：host=10.80.139.9, port=8123, database=cmds, username=<from COMMON_CONST>
import clickhouse_connect
from COMMON_CONST import DATA_DB_HFDATA

client = clickhouse_connect.get_client(**DATA_DB_HFDATA)
# 推荐：项目封装
from research.l2_alpha.clickhouse_ssl2 import extract_minute_features
df = extract_minute_features("2024-06-03", "2024-06-04", symbols=["600000", "000001"])
# 输出列: tradetime | symbol | factorname | value
# symbol 规范: 600000.SH / 000001.SZ
```

Demo SQL：

```python
sql = """
SELECT Symbol, ExchTime, BidPrices, AskPrices, BidVolumes, AskVolumes
FROM cmds.SSE_AL_SSL2_EXG
WHERE ExchTime >= toDateTime64('2024-06-03 00:00:00', 6, 'Asia/Shanghai')
  AND ExchTime <  toDateTime64('2024-06-04 00:00:00', 6, 'Asia/Shanghai')
LIMIT 10
"""
result = client.query(sql)
```

#### 字段字典（研究常用子集）

完整列数：SSE 54 / SZSE 53。数组字段为 ClickHouse **1-indexed**，`[1]`=买一/卖一。

| 字段 | 类型 | 含义 | 单位 | 缺失处理 |
|------|------|------|------|----------|
| `Symbol` | String | 裸代码（无后缀） | — | 必填；输出时加 `.SH`/`.SZ` |
| `ExchTime` | DateTime64 | 交易所时间（Asia/Shanghai） | datetime | 过滤窗口必用 |
| `BidPrices` / `AskPrices` | Array(Decimal) | 十档价格 | 元 | 长度&lt;1 的行应丢弃 |
| `BidVolumes` / `AskVolumes` | Array(Int64) | 十档量 | 股/手 `[TODO: 单位确认]` | 空档用 NULL，求和时 `ifNull(v,0)` |
| `BidNums` / `AskNums` | Array | 档位委托笔数 | 笔 | 可选诊断 |
| `TotalBidVolume` / `TotalAskVolume` | Int64 | 总买/卖量 | 同上 | — |
| `BidVWAP` / `AskVWAP` | Decimal | 盘口 VWAP | 元 | 非有限值 → 特征置 NULL |
| `BidWithdraw*` / `AskWithdraw*` | Int/Decimal | 撤单压力（**仅 SSE**） | 量/额 | SZSE 无此列 → `l2_cancel_pressure` 为 NULL |
| `Price` / `Open/High/Low/Close` | Decimal | 最新价/日内 OHLC | 元 | — |
| `AccVolume` / `AccAmount` | — | 累计成交 | 股/元 | — |
| `TradeStatus` | — | 交易状态 | — | `[TODO: 枚举含义]` |
| `CHTime` | — | 入库时间 | datetime | 一般不参与因子 |

> ⚠️ **禁止**虚构展开列 `BidPrice0`…`BidPrice9`；必须用 Array 运算。

#### 时间范围与粒度

- 粒度：Snapshot（约 3s 级，具体频率 `[TODO: 建议向数据组确认]`）
- 研究聚合：默认 **每分钟最后一个有效 snapshot**（`toStartOfMinute` + `argMax`）
- 最新日期：`[TODO: 待补充]`

#### 注意事项

1. DolphinDB **没有** 原生 L2；勿再走 `dfs://QV_Snapshot`。
2. SSE 与 SZSE schema 不完全对称：SZSE 无 `Bid/AskWithdraw*`，且无对称的 `BidNums/AskNums` 数组（有 `BidNum1` 等）。
3. `LOCAL_*` 表为镜像，优先用非 `LOCAL_` 生产表，差异 `[TODO]`。
4. 期货/商品 SSL1/SSL2（CFFEX/DCE/CZCE 等）存在，但当前 A 股 ZZ1000 库范围外。

#### 常用过滤条件

```sql
WHERE ExchTime >= toDateTime64('2020-01-01 00:00:00', 6, 'Asia/Shanghai')
  AND ExchTime <  toDateTime64('2020-01-02 00:00:00', 6, 'Asia/Shanghai')
  AND Symbol IN ('600000', '000001')
  AND length(BidPrices) >= 1 AND length(AskPrices) >= 1
```

---

### 2.2 ClickHouse A股 Tick

**角色**：逐笔成交；Phase 2，当前未接入评估链路。

#### 访问方式

```python
import clickhouse_connect
from COMMON_CONST import DATA_DB_HFDATA

client = clickhouse_connect.get_client(**DATA_DB_HFDATA)
sql = """
SELECT Symbol, ExchTime, Price, Volume, Amount, BidOrderNo, AskOrderNo, BSFlag
FROM cmds.SSE_AL_TICK_EXG
WHERE ExchTime >= toDateTime64('2024-06-03 09:30:00', 6, 'Asia/Shanghai')
  AND ExchTime <  toDateTime64('2024-06-03 09:31:00', 6, 'Asia/Shanghai')
LIMIT 100
"""
result = client.query(sql)
```

#### 字段字典（SSE 15 列核心）

| 字段 | 类型 | 含义 | 单位 | 缺失处理 |
|------|------|------|------|----------|
| `Symbol` | String | 裸代码 | — | — |
| `ExchTime` | DateTime64 | 成交时间 | datetime | — |
| `Price` | — | 成交价 | 元 | — |
| `Volume` / `Amount` | — | 量/额 | `[TODO]` | — |
| `BidOrderNo` / `AskOrderNo` | — | 买卖委托号 | — | 可关联订单流 |
| `BSFlag` | — | 主买/主卖标志 | enum | `[TODO: 取值表]` |
| `SeqNo` / `SubSeqNo` | — | 序列号 | — | — |

SZSE tick 更丰富（37 列，含 `SecondaryOrderID` 等），完整字典见审计 JSON `scored_tables`。

#### 时间范围与粒度

- 粒度：Tick  
- 起止：`[TODO: 待补充]`

#### 注意事项

1. 与 DDB 分钟线的 `Active_*` **不是同一口径**，不可直接混用验证。
2. 全市场 Tick 体量极大，务必按日/按标的切片。
3. 订单簿重建需结合 snapshot + tick，当前无现成封装。

#### 常用过滤条件

```sql
WHERE toDate(ExchTime) = '2024-06-03' AND Symbol = '600000'
```

---

### 2.3 DolphinDB 股票分钟线（生产评估默认路径）

**角色**：日内因子工厂与 `intraday_evaluation_v2` 主数据源。

#### 访问方式

依赖：`dolphindb`, `pandas`

```python
from factor_data_loaders import connect_ddb
# 或 from core.ddb.connection import get_ddb_session

s = connect_ddb()  # 使用 COMMON_CONST.DATA_DB_CONN
# host=10.12.180.9, port=8902, userid=pyread（密码见 COMMON_CONST / .env）

t = s.loadTable(dbPath="dfs://QV_Trade_to_MinuteBar", tableName="Stock_one_minute")
df = (
    t.where("Date >= 2024.01.01 and Date <= 2024.01.31 and Symbol=`600000.SH")
     .select("Symbol, Date, Bartime, Open, High, Low, Close, Volume, Amount, "
             "Active_buy_amount, Active_sell_amount, Adjfactor")
     .toDF()
)
```

同库其他表：`Fund_one_minute`、`Cbond_one_minute`；`Future_one_minute` 当前账号不可读。

#### 字段字典（Stock_one_minute，22 列）

| 字段 | 类型 | 含义 | 单位 | 缺失处理 |
|------|------|------|------|----------|
| `Symbol` | SYMBOL | 如 `600000.SH` | — | — |
| `Date` | DATE | 交易日 | `YYYY.MM.DD`（DDB） | — |
| `Bartime` | SECOND | bar 时间（日内秒） | time | 与 `Date` 组合为 bar |
| `Barstart` / `Barend` | TIME | bar 起止 | time | — |
| `Open/High/Low/Close` | DOUBLE | OHLC | 元（未复权价为主） | — |
| `Volume` | LONG | 成交量 | 股 | — |
| `Amount` | DOUBLE | 成交额 | 元 | — |
| `Active_buy_*` / `Active_sell_*` | LONG/DOUBLE | 主动买卖量/额/笔数 | 股/元/笔 | 停牌可能为 0 |
| `Bid/Ask_cancel_volume/count` | LONG | 撤单代理指标 | 股/笔 | **非**原始撤单事件 |
| `Adjfactor` | DOUBLE | 复权因子 | 倍数 | `NULL` 或 `0` 时按 `1.0` 处理（见 `core/ddb_intraday_queries.py`） |

#### 时间范围与粒度

| 表 | 行数（审计） | 起始 | 最新 | 标的数 |
|----|--------------|------|------|--------|
| `Stock_one_minute` | ~2.14e9 | 2018-09-03 | 2026-07-30 | 6026 |
| `Fund_one_minute` | ~4.17e8 | 2018-09-03 | 2025-08-18 | 2160 |
| `Cbond_one_minute` | ~1.53e8 | 2018-09-03 | 2025-08-18 | 920 |

粒度：1 分钟。

#### 注意事项

1. **无十档盘口**；microprice / depth imbalance 必须走 ClickHouse。
2. `Active_*` / cancel 字段是分钟聚合代理，不是逐笔事件。
3. 价格复权：跨日用 `Adjfactor`；公式中常见 `Close * Adjfactor`。前后复权口径 `[TODO: 与 Wind 前复权一致性确认]`。
4. 当前账号下 `dfs://QV_Snapshot` / `QV_Tick` **不存在**。

#### 常用过滤条件

```text
Date >= 2020.01.01 and Date <= 2024.12.31
Symbol in [`600000.SH, `000001.SZ]
second(Bartime) between 09:30:00 : 14:57:00   # 按研究窗口调整
```

---

### 2.4 DolphinDB Wind EOD（基础日频）

**角色**：EOD 因子与回测主面板；`factor_data_loaders.load_eod_wide_tables` / `core.ddb.eod.fetch_eod_long`。

#### 访问方式

```python
from factor_data_loaders import load_eod_wide_tables, connect_ddb
import datetime as dt

tables, session = load_eod_wide_tables(
    dt.datetime(2020, 1, 1),
    dt.datetime(2024, 12, 31),
)
# tables.close / open / high / low / volume / amount : 宽表 index=Date, columns=WindCode
# A 股过滤：代码首位 in {0,3,6}

# 长表接口
from core.ddb.eod import fetch_eod_long
df = fetch_eod_long("2024-01-01", "2024-01-31", symbols=["600000.SH"])
```

时间过滤格式：`TRADE_DT >= 2020.01.01`（DDB DATE 字面量）。

#### 字段字典（`dfs://WIND.ASHAREEODPRICES/data`，27 列）

| 字段 | 类型 | 含义 | 单位 | 缺失处理 |
|------|------|------|------|----------|
| `S_INFO_WINDCODE` | SYMBOL | Wind 代码 | — | — |
| `TRADE_DT` | DATE | 交易日 | date | — |
| `S_DQ_OPEN/HIGH/LOW/CLOSE` | DOUBLE | 未复权 OHLC | 元 | 停牌可能沿用或空 `[TODO]` |
| `S_DQ_PRECLOSE` | DOUBLE | 昨收 | 元 | 收益用 `CLOSE/PRECLOSE-1` |
| `S_DQ_VOLUME` | DOUBLE | 成交量 | 手（Wind 惯例）`[TODO: 与分钟 Volume 单位对齐]` | — |
| `S_DQ_AMOUNT` | DOUBLE | 成交额 | 千元（Wind 惯例）`[TODO]` | — |
| `S_DQ_PCTCHANGE` | DOUBLE | 涨跌幅 | % | — |
| `S_DQ_ADJ*` / `S_DQ_ADJFACTOR` | DOUBLE | 复权价/因子 | — | 回测用复权价时显式选择 |
| `S_DQ_ADJCLOSE_BACKWARD` | DOUBLE | 后复权收盘 | 元 | — |
| `S_DQ_TRADESTATUS` / `CODE` | STRING/INT | 交易状态 | — | 过滤停牌 |
| `S_DQ_LIMIT` / `S_DQ_STOPPING` | DOUBLE | 涨跌停价 | 元 | — |
| `S_DQ_TURN` | — | 换手率 | % | DDB 表可能无此列；失败时 loader 跳过 |
| `OPDATE` / `OPMODE` | DATETIME/STRING | 运维字段 | — | 研究可忽略 |

#### 时间范围与粒度

- 粒度：日频  
- DDB 分区起始/最新：`[TODO: 待补充；建议跑 min/max(TRADE_DT)]`  
- 2018 前区间：用 Oracle Wind（见下节）

#### 注意事项

1. 默认 loader 使用**未复权** `S_DQ_CLOSE`；收益也可用 `PRECLOSE` 口径。
2. `S_DQ_VOLUME` / `S_DQ_AMOUNT` 单位与分钟线可能不一致，跨频对齐前必须换算。
3. 指数日频另见 `dfs://WIND.AINDEXEODPRICES`（如 CSI300）。
4. 市值等衍生字段来自 enrich 流程（`float_mktcap`），不是本表原生列。

#### 常用过滤条件

```text
TRADE_DT between 2020.01.01 : 2024.12.31
S_INFO_WINDCODE like "6%" or like "0%" or like "3%"
```

---

### 2.5 Wind Oracle EOD

**角色**：覆盖 DDB 分钟/EOD 窗口之外的历史；与 DDB 对照。

#### 访问方式

依赖：`oracledb`, `pandas`, `pyarrow`（parquet 缓存）

```python
import oracledb
from COMMON_CONST import DATA_DB_WIND
from factor_data_loaders import load_eod_wide_tables_from_wind_oracle
import datetime as dt

# 推荐封装（按月 parquet 缓存）
tables, ret_c2c = load_eod_wide_tables_from_wind_oracle(
    dt.datetime(2015, 1, 1),
    dt.datetime(2018, 8, 31),
)

# 裸连接
oracledb.init_oracle_client(lib_dir=None)  # 部分库需 thick mode，且须进程内先于 thin
with oracledb.connect(**DATA_DB_WIND) as conn:
    # dsn=10.23.153.15:21010/wind, user=lqtz
    df = pd.read_sql(
        """
        SELECT TRADE_DT, S_INFO_WINDCODE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW,
               S_DQ_CLOSE, S_DQ_PRECLOSE, S_DQ_VOLUME, S_DQ_AMOUNT
        FROM wind.ASHAREEODPRICES
        WHERE TRADE_DT >= :d0 AND TRADE_DT <= :d1
          AND (S_INFO_WINDCODE LIKE '6%' OR S_INFO_WINDCODE LIKE '0%'
            OR S_INFO_WINDCODE LIKE '3%')
        """,
        conn,
        params={"d0": "20200101", "d1": "20200131"},
    )
```

> ⚠️ `TRADE_DT` 在 Oracle 侧为 **`YYYYMMDD` 字符串/数字** 风格过滤（`:d0='20200101'`）；DDB 侧为 `2020.01.01` DATE。

#### 字段字典

与 Wind EOD 核心价量字段一致；`S_DQ_TURN` 在 Oracle **不在** `ASHAREEODPRICES`（在 derivative 表，`[TODO: 表名确认]`）。

#### 注意事项

1. 进程内 thin/thick mode 不能混用；需 thick 时**先** `init_oracle_client`。
2. 全历史拉取务必按月切片 + 本地 parquet，防 OOM。
3. 与 DDB 同字段数值差异：`[TODO: 已知差异清单待补充]`。

#### 常用过滤条件

```sql
WHERE TRADE_DT >= '20200101' AND TRADE_DT <= '20201231'
  AND S_INFO_WINDCODE LIKE '6%'
```

---

### 2.6 ClickHouse HF K线（对照）

**角色**：CH 侧 OHLCVA，可与 DDB 分钟路径交叉验证。

#### 访问方式

```python
client = clickhouse_connect.get_client(**DATA_DB_HFDATA)
# SSE: cmds.SSE_AL_KLIN_EXG (19 cols)
# SZSE: cmds.SZSE_AL_KLIN_CMD (10 cols)
sql = "SELECT Symbol, ExchTime, Open, High, Low, Close, Volume, Amount FROM cmds.SSE_AL_KLIN_EXG LIMIT 10"
```

#### 字段字典（SSE KLIN 核心）

| 字段 | 含义 | 备注 |
|------|------|------|
| `Symbol`, `ExchTime` | 代码/时间 | DateTime64 |
| `Open/High/Low/Close` | OHLC | — |
| `Volume`, `Amount` | 量额 | — |
| `PreClose`, `Average`, `IOPV` 等 | 参考 | ETF/衍生品相关 |

#### 注意事项

1. SSE/SZSE K 线 schema 列数不同（19 vs 10）。  
2. 与 DDB `Stock_one_minute` 的 bar 边界是否一致：`[TODO]`。  
3. 生产评估仍以 DDB 分钟线为准。

---

### 2.7 聚源（Juyuan）Fundamentals

#### 访问方式

```python
import oracledb
from COMMON_CONST import DATA_DB_JUYUAN  # replica: DATA_DB_JUYUAN_2

oracledb.init_oracle_client(lib_dir=None)
with oracledb.connect(**DATA_DB_JUYUAN) as conn:
    # dsn=10.23.129.8:21010/juyuan
    with conn.cursor() as cur:
        cur.execute("select * from jydb.secumain where rownum < 10")
        rows = cur.fetchall()
```

#### 字段字典

`[TODO: 待补充完整财务表清单与字段；建议向数据组确认常用 LC_*/资产负债表视图]`

#### 时间范围与粒度

季报/年报为主；`[TODO: 公告日/报告期字段名]`

#### 注意事项

1. 财务因子必须用**公告日**对齐交易日，避免前视。  
2. 报告期修正（restatement）策略：`[TODO]`。  
3. 主库与 `_2` replica 延迟：`[TODO]`。

#### 常用过滤条件

```sql
WHERE rownum < 100  -- Demo only; 生产按 SecuCode / EndDate 过滤
```

---

### 2.8 朝阳永续一致预期（ZYYX2）

#### 访问方式

```python
from COMMON_CONST import DATA_DB_ZYYX2
with oracledb.connect(**DATA_DB_ZYYX2) as conn:
    # dsn=10.23.129.89:1521/zyyx2
    with conn.cursor() as cur:
        cur.execute("select * from zyyq.con_forecast_stk where rownum < 10")
```

#### 字段字典

`[TODO: con_forecast_stk 及 PE/EPS 一致预期字段字典待补充]`

#### 注意事项

1. 一致预期有机构覆盖偏差，需样本筛选。  
2. 与 Wind/聚源 PE **不可混用同一回测**而不做口径说明。  
3. 更新频率与时点：`[TODO]`。

---

### 2.9 财汇（Caihui）

#### 访问方式

```python
from COMMON_CONST import DATA_DB_CAIHUI  # replica: DATA_DB_CAIHUI_2 → finchina
with oracledb.connect(**DATA_DB_CAIHUI) as conn:
    # 主库 dsn 服务名 FINCHINA2
    with conn.cursor() as cur:
        cur.execute("select * from finchina.TQ_OA_STCODE where rownum < 10")
```

#### 字段字典 / 时间范围

`[TODO: 待补充；建议向数据组确认]`

#### 注意事项

1. 主库与 replica 服务名大小写不同（`FINCHINA2` vs `finchina`）。  
2. 与聚源财务字段可能重叠，选型需统一。  
3. L2 名称命中为假阳性。

---

### 2.10 通联 Datayes（MySQL）

#### 访问方式

依赖：`pymysql`

```python
import pymysql
from COMMON_CONST import DATA_DB_DATAYES

with pymysql.connect(**DATA_DB_DATAYES) as conn:
    # host=10.80.139.20:3306, database=datayes
    with conn.cursor() as cur:
        cur.execute("select * from datayes.con_index limit 10")
        rows = cur.fetchall()
```

#### 字段字典

`[TODO]`

#### 注意事项

1. 审计结论：MySQL 端点**无** LOB/tick schema。  
2. 表权限按库授权，缺表时需申请。  
3. 时区/字符集：`[TODO]`。

---

### 2.11 东方财富 EMData（MySQL）

#### 访问方式

```python
from COMMON_CONST import DATA_DB_EMDATA  # replica: DATA_DB_EMDATA_2
with pymysql.connect(**DATA_DB_EMDATA) as conn:
    # host=10.80.139.50:6030, database=emdata
    with conn.cursor() as cur:
        cur.execute("select * from emdata.fund_bs_cfinfo limit 10")
```

#### 字段字典 / 注意

`[TODO: 资金流字段口径与延迟]`  
无 L2；端口为 `6030`（非默认 3306）。

---

### 2.12 野尘 Yechen（MySQL）

#### 访问方式

```python
from COMMON_CONST import DATA_DB_YECHEN
with pymysql.connect(**DATA_DB_YECHEN) as conn:
    with conn.cursor() as cur:
        cur.execute("select * from yechen.industry_chain limit 10")
```

#### 字段字典

`[TODO]`

---

### 2.13 排排网 / 普益 / 通用 ORCL（参考）

| 源 | 常量 | Demo 表 | 场景 |
|----|------|---------|------|
| 排排网 | `DATA_DB_PAIPAI` / `_2` | `java.pvn_fund_info` | 私募产品参考 |
| 普益 | `DATA_DB_PUYI` / `_2` | `pystandard.bank_base_info` | 银行理财 |
| ORCL | `DATA_DB_ORCL` | `[TODO]` | 通用 |

连接方式同其他 Oracle（`oracledb.connect(**CONST)`）。完整字段字典均为 `[TODO: 建议向数据组确认]`。

---

### 2.14 EOD 衍生特征（非原始数据）

**角色**：在原始 EOD 之上的计算层，见 `eod_data_foundation.py`。

| 层级 | 内容 |
|------|------|
| raw_ohlcv | open/high/low/close/volume/amount |
| enriched_size | `float_mktcap`, `total_mktcap` |
| derived_cache | `ret_1d/5d/20d/60d`, `volatility_*`, `volume_mean_*`, `amount_*`, `rsi_14` 等 |
| alpha_projection | 因子公式输出 |

> ⚠️ 衍生层不可当作“新数据源”写入审计；复现实验必须能追溯到 Wind EOD / 分钟线 / SSL2。

---

## 3. 数据加载最佳实践

### 3.1 推荐查询顺序

1. **本地缓存**（Feather / Parquet，若已有本次任务产物）  
2. **DolphinDB**（EOD 宽表、分钟线、评估）  
3. **ClickHouse**（L2 snapshot / tick / HF K 线）  
4. **Oracle / MySQL 源库**（财务、一致预期、2018 前 EOD）  
5. **原始 CSV**（仅当以上皆无；本仓库审计未发现标准 CSV 源 → `[TODO]`）

### 3.2 统一加载模板（建议落地 `utils/data_loader.py`）

> 当前仓库已有 `factor_data_loaders.py`、`research/l2_alpha/clickhouse_ssl2.py`、`research/extreme_return_study/src/data_loader.py`。下列为**建议统一入口骨架**，尚未强制存在于 `utils/`。

依赖：`pandas`, `pyarrow`, `dolphindb`, `clickhouse_connect`, `oracledb`, `pymysql`

```python
# 建议路径: ${PROJ_ROOT}/utils/data_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

PROJ_ROOT = Path("/home/SiYangCao/factor_dev/factor_research0703/factor_dev")
CACHE_ROOT = PROJ_ROOT / "research" / "cache"


def load_feather(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_path(name: str, suffix: str = "parquet") -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{name}.{suffix}"


def load_or_build(name: str, builder, *, force: bool = False) -> pd.DataFrame:
    """先读本地 parquet；缺失则 builder() 并缓存。"""
    path = cache_path(name)
    if path.exists() and not force:
        return load_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


def read_ddb_minute(start: str, end: str, symbols: Optional[list] = None) -> pd.DataFrame:
    from factor_data_loaders import connect_ddb
    s = connect_ddb()
    try:
        # start/end: 'YYYY.MM.DD'
        where = f"Date >= {start} and Date <= {end}"
        if symbols:
            sym = ", ".join(f"`{x}" for x in symbols)
            where += f" and Symbol in [{sym}]"
        t = s.loadTable("dfs://QV_Trade_to_MinuteBar", "Stock_one_minute")
        return t.where(where).toDF()
    finally:
        s.close()


def read_ch_ssl2_minute_features(start: str, end: str, symbols=None) -> pd.DataFrame:
    from research.l2_alpha.clickhouse_ssl2 import extract_minute_features
    return extract_minute_features(start, end, symbols=symbols)
```

### 3.3 缓存 / 并行 / 防 OOM

| 手段 | 做法 |
|------|------|
| 按日/按月切片 | Oracle EOD、CH L2 必须切片；参考 `load_eod_wide_tables_from_wind_oracle` |
| 列裁剪 | 只 `SELECT` 需要列；CH 避免 `SELECT *` 拉 Array 全历史 |
| 服务端聚合 | L2 特征在 CH 内做 array 运算 + minute `argMax`，Python 不 `groupby` 原始 L2 |
| 并行 | 按标的或按日并行，**进程数 ≤10**（团队约定）；注意 DB 连接数配额 |
| 缓存格式 | 中间结果用 Parquet/Feather；宽表注意 category/字符串重复 |
| 会话复用 | DDB 用 `get_ddb_session(reuse=True)`，避免反复握手 |

> ⚠️ 全市场 SSL2 单日已很大；ZZ1000 多年 harvest 前先 smoke（见 `research/l2_alpha/run_ssl2_feature_smoke.py`）。

---

## 4. 数据质量与一致性检查

### 4.1 已知问题

| 问题 | 涉及源 | 状态 |
|------|--------|------|
| DDB 无原生 L2，早期文档曾误判 blocked | DDB vs CH | 已纠正：L2 在 CH |
| Oracle `%L2%` 对象假阳性 | Wind/聚源等 | 忽略，非行情 LOB |
| SZSE SSL2 无撤单统计列 | `SZSE_AL_SSL2_EXG` | `l2_cancel_pressure` 为 NULL |
| `Future_one_minute` 当前账号不可读 | DDB | 需权限或换路径 |
| Fund/Cbond 分钟最新日早于 Stock | DDB | Stock→2026-07-30；Fund/Cbond→2025-08-18 |
| 字段拼写历史包袱 | CH `ToltalBidNum` 等 | 以 schema 原文为准 |
| EOD 量额单位 vs 分钟量额 | Wind vs DDB 分钟 | `[TODO: 换算表]` |
| 多源 PE（Wind / 聚源 / 朝阳）口径不同 | 财务/预期 | 见 FAQ |

### 4.2 校验脚本调用

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
PY=/opt/conda/anaconda3/envs/base_93/bin/python   # 若环境不同请替换

# 全库端点连通 + CH L2 扫描
PYTHONPATH=. $PY research/run_multi_db_data_audit_v1.py
# 产物: research/results/multi_db_data_audit.json
# 文档: research/docs/multi_db_data_audit_v1.md

# DDB 分钟/L2 能力审计（DDB 范围）
PYTHONPATH=. $PY research/run_l2_data_audit_v1.py

# SSL2 特征 smoke + 单测
$PY -m unittest tests.test_l2_ssl2_formulas_v1 -v
OMP_NUM_THREADS=1 PYTHONPATH=. $PY \
  research/l2_alpha/run_ssl2_feature_smoke.py \
  --date 2024-06-03 --limit-symbols 20
```

连接 Demo（人工探活）：

```bash
PYTHONPATH=. $PY DB_Demo.py
```

### 4.3 填补逻辑（业务）

| 数据 | 是否 ffill/bfill | 逻辑 |
|------|------------------|------|
| EOD 行情缺口（停牌） | 通常**不**对价格 ffill 当可交易；收益/因子侧显式 mask | 用 `S_DQ_TRADESTATUS` / 涨跌停 / ST 过滤 |
| 市值、行业等慢变量 | 可 **ffill** 到交易日 | 不得 bfill（防前视） |
| 财务基本面 | 按**公告日** asof 到交易日（ffill） | 禁止用报告期期末向未来填 |
| 一致预期 | 按发布时间 asof | `[TODO: 具体表字段]` |
| L2 snapshot 分钟聚合 | 分钟内取 last；分钟无 snapshot → 该分钟特征缺失，不跨分钟 ffill | 与 bartime 语义一致 |
| `Adjfactor` 空/0 | 当作 `1.0` | 见日内 SQL |

> ⚠️ 任何 bfill 用于信号生成都视为前视风险，需在研究笔记中显式声明。

---

## 5. 版本与更新日志

| 项目 | 内容 |
|------|------|
| 文档版本 | `data_source_handbook_v2` |
| 最后更新 | 2026-07-31 |
| 作者 | 因子研发（基于 multi-db audit v1 整理） |
| 上游审计 | `research/docs/multi_db_data_audit_v1.md` |

### 后续数据源变动记录

| 日期 | 变更类型 | 数据源 | 说明 | 记录人 |
|------|----------|--------|------|--------|
| 2026-07-31 | 新增手册 | 全量端点 | 首版 handbook，纠正 L2 在 CH | — |
|  | 新增/删除/字段变更 |  |  |  |
|  |  |  |  |  |

---

## 6. 快速 FAQ

### Q1: 如何获取某只股票的历史日频 OHLC？

优先 DDB：

```python
from core.ddb.eod import fetch_eod_long
df = fetch_eod_long("2020-01-01", "2024-12-31", symbols=["600000.SH"])
# 列: date, symbol, open, high, low, close, volume, amount
# date 为 pandas datetime；源字段 TRADE_DT
```

若需 2018-09 前或与 Oracle 对照，用 `load_eod_wide_tables_from_wind_oracle`。

### Q2: 财务数据季报/年报如何对齐到交易日？

使用**公告日**（或数据源提供的可投资时间戳）做 asof → 交易日历 **ffill**；不要用报告期期末日期直接对齐。  
聚源/财汇具体公告日字段名：`[TODO: 建议向数据组确认]`。

### Q3: 多个库都有 PE，该用哪个？

| 用途 | 建议 |
|------|------|
| 与 Wind EOD 面板一致的估值 | Wind 衍生表 / 已有 enrich 流程中的 `pe_ttm`（若已加载） |
| 卖方一致预期 PE | 朝阳 `DATA_DB_ZYYX2` |
| 财报原始推算 | 聚源/财汇 |

同一回测内不要混用未校准的 PE。字段级权威表：`[TODO]`。

### Q4: 分钟因子和 L2 因子分别读哪个库？

| 因子族 | 数据源 |
|--------|--------|
| `trade_flow_minute` / `cancel_intent_minute` / `price_path_minute` 等 | DolphinDB `Stock_one_minute` |
| `order_book` / `microprice` / `spread` / `l2_*` | ClickHouse `SSE/SZSE_AL_SSL2_EXG` |
| 逐笔 OrderID 类 | ClickHouse tick（Phase 2） |

评估框架默认仍接 DDB 分钟路径；L2 特征需 bridge 到 `intraday_evaluation_v2`。

### Q5: 为什么 DolphinDB 审计说没有 L2，但又说可以做 L2 因子？

Phase 0 只扫了 DDB：无 snapshot。Phase multi-db 在 **ClickHouse `cmds`** 发现了 `SSE/SZSE_AL_SSL2_EXG` 与 tick。两套库职责不同，不要在 DDB 上找 `QV_Snapshot`。

### Q6: 复权怎么处理？

- 分钟线：用表内 `Adjfactor`；`NULL/0 → 1.0`。  
- EOD：区分 `S_DQ_CLOSE`（未复权）与 `S_DQ_ADJCLOSE` / `S_DQ_ADJCLOSE_BACKWARD`。  
- 跨日收益优先 `CLOSE/PRECLOSE-1` 或显式复权价序列，避免混用。

### Q7: 本地有没有标准 Feather 行情仓？

审计未登记统一 Feather 仓；本地文件多为任务缓存。建议统一写到 `${PROJ_ROOT}/research/cache/`，并用 `load_or_build` 模式管理。若团队有共享盘路径：`[TODO: 建议向数据组/IT 确认]`。

### Q8: 如何确认我有权限？

1. 跑 `DB_Demo.py` 对应段落是否报 ORA/access denied。  
2. 跑 `run_multi_db_data_audit_v1.py`，查看 `endpoints[].ok`。  
3. 仍失败则带 endpoint 名向 IT 申请只读授权。

---

## 附录 A. 连接常量速查（脱敏）

| 常量 | 引擎 | Host/DSN（无密码） | 默认库/服务 |
|------|------|---------------------|-------------|
| `DATA_DB_CONN` | DolphinDB | `10.12.180.9:8902` | userid=`pyread` |
| `DATA_DB_HFDATA` | ClickHouse | `10.80.139.9:8123` | `cmds` |
| `DATA_DB_WIND` | Oracle | `10.23.153.15:21010/wind` | `wind` |
| `DATA_DB_JUYUAN` | Oracle | `10.23.129.8:21010/juyuan` | `juyuan` |
| `DATA_DB_ZYYX2` | Oracle | `10.23.129.89:1521/zyyx2` | `zyyx2` |
| `DATA_DB_CAIHUI` | Oracle | `10.23.129.86:21001/FINCHINA2` | finchina |
| `DATA_DB_DATAYES` | MySQL | `10.80.139.20:3306` | `datayes` |
| `DATA_DB_EMDATA` | MySQL | `10.80.139.50:6030` | `emdata` |
| `DATA_DB_YECHEN` | MySQL | `10.63.95.30:3306` | `yechen` |
| `DATA_DB_PAIPAI` | Oracle | `10.23.129.42:21010/cmdssm` | — |
| `DATA_DB_PUYI` | Oracle | `10.23.129.232:1521/puyi` | — |
| `DATA_DB_ORCL` | Oracle | `172.22.133.52:1521/ORCL` | — |

完整定义：`${PROJ_ROOT}/COMMON_CONST.py`。

## 附录 B. 相关文档索引

| 文档 | 路径 |
|------|------|
| Multi-DB 审计 | `${PROJ_ROOT}/research/docs/multi_db_data_audit_v1.md` |
| DDB L2 审计 | `${PROJ_ROOT}/research/docs/l2_data_audit_v1.md` |
| SSL2 特征提取 | `${PROJ_ROOT}/research/docs/l2_ssl2_feature_extractor_v1.md` |
| EOD 数据分层 | `${PROJ_ROOT}/eod_data_foundation.py` |
| 连接 Demo | `${PROJ_ROOT}/DB_Demo.py` |

---

## Final Check（维护者自检）

- [x] 每个主要数据源含加载代码示例  
- [x] 时间字段格式已区分（DDB `YYYY.MM.DD` / Oracle `YYYYMMDD` / CH `DateTime64`）  
- [x] 区分原始行情与衍生因子缓存  
- [x] 注明只读账号与权限申请路径  
- [x] 缺失信息以 `[TODO]` 标注，不臆造  
