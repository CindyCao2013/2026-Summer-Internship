# Factor_SDT 使用说明

`Factor_SDT` 是一个用于 A 股量化因子开发、测试和结果分析的 Python 项目。项目以 DolphinDB 作为主要行情与分钟数据源，提供日频 EOD 因子测试、分钟级日内因子测试、行业/市值中性化、分组回测、绩效统计和图表输出等能力。

当前代码更偏研究与示例工作流，多个入口脚本包含 IPython/Jupyter 的 cell 标记和 magic 命令，推荐在 Jupyter、VS Code Interactive Window 或 Cursor 的交互式 Python 环境中逐段运行。

## 项目结构

- `COMMON_CONST.py`：数据库连接与常量配置。包含 DolphinDB、Oracle、MySQL、ClickHouse 等数据源连接信息。
- `Factor_Dev_Lib.py`：日频因子测试核心库，包含交易日、涨跌停、ST、停牌过滤、收益矩阵、分组测试、组合收益、中性化等函数。
- `Factor_Test_Process.py`：日频因子开发与测试示例，从 Wind EOD 数据计算示例因子并调用 `groupTest` 回测。
- `intraday_lib.py`：分钟级因子测试库，包含 DolphinDB 函数字符串 `ddb_functions` 以及 Python 侧图表和汇总函数。
- `intraday_Factortest.py`：分钟级因子测试示例，展示窄表因子上传、指数内过滤、分组表现分析和结果保存。
- `data_preheat.py`：预热数据脚本，用于生成日内收益矩阵、行业宽表等 DolphinDB 共享对象。
- `DB_Demo.py`：Oracle、MySQL、ClickHouse 等数据库连接示例。
- `Industry_Query_Demo.py`：基于 Wind Oracle 数据库查询行业层级和行业字典的示例。

## 环境依赖

本项目依赖于中信93服务器的环境。没有此环境的账户，请联系管理员开通。

运行前需要确认：

- 能访问 `COMMON_CONST.py` 中配置的 DolphinDB 服务。
- 如需访问 Wind、聚源、财汇等 Oracle 数据库，本机 Python 环境需正确配置 Oracle client，并在进程启动早期调用 `oracledb.init_oracle_client(...)`。
- 日内测试依赖 DolphinDB 中的分钟数据表和预热共享对象，例如 `PREHEAT_RET_MATRIX_ZZ1000`。
- matplotlib 中文显示依赖本机中文字体；`intraday_lib.py` 会尝试自动选择常见中文字体。

注意：`COMMON_CONST.py` 中包含数据库账号、密码和内网地址。对外共享代码或提交到公共仓库前，应先改为环境变量或本地私有配置文件。

## 日频因子测试流程

日频测试的主要入口是 `Factor_Test_Process.py`，核心工具函数在 `Factor_Dev_Lib.py`。

### 1. 设置测试区间

在 `Factor_Test_Process.py` 中设置：

```python
startDay = dt.datetime(2020, 1, 1)
endDay = dt.datetime(2025, 12, 31)
startDay_preHeat = startDay - dt.timedelta(days=40)
```

`startDay_preHeat` 用于 rolling、移动平均等需要历史窗口的因子，避免正式测试第一天缺少历史数据。

### 2. 计算因子宽表

日频因子推荐整理成宽表：

- index：交易日期
- columns：股票代码，例如 `000001.SZ`
- values：因子值

示例流程会从 `dfs://WIND.ASHAREEODPRICES` 读取 EOD 行情，计算 5 日收益均值：

```python
factor = df_ret.rolling(5).mean()
factor_value = factor.loc[startDay:endDay, :]
```

实际开发时，只要最终得到同样结构的 `factor_value`，即可进入后续测试。

### 3. 准备收益矩阵和过滤矩阵

常用函数：

```python
ret_matrix = Factor_Dev_Lib.get_Ret_Matrix(startDay, endDay, method='c2c', base_index='000852.SH')
ret_matrix_o2o = Factor_Dev_Lib.get_Ret_Matrix(startDay, endDay, method='o2o', base_index='000852.SH')
ret_matrix_v2v = Factor_Dev_Lib.get_Ret_Matrix(startDay, endDay, method='v2v')

df_notLimit = Factor_Dev_Lib.get_EOD_Not_Limit(startDay, endDay)
df_notST = Factor_Dev_Lib.get_EOD_Not_ST(startDay, endDay)
df_TradeStatus = Factor_Dev_Lib.get_TradeStatus(startDay, endDay)
```

`get_Ret_Matrix` 支持：

- `method='c2c'`：close to close 收益。
- `method='o2o'`：open to open 收益。
- `method='v2v'`：vwap to vwap 收益。
- `base_index`：传入指数代码后返回相对指数的超额收益。

### 4. 对齐信号和未来收益

测试前通常要剔除不可交易样本，并根据信号产生时间和交易方式做 shift：

```python
signal = factor_value
signal = signal.mul(df_notLimit)
signal = signal.mul(df_notST)
signal = signal.mul(df_TradeStatus)

# close to close 测试通常 shift 1 天
signal = signal.shift()

signal = signal.dropna(how='all', axis=1)
signal = signal.dropna(how='all')
```

经验规则：

- 对 `c2c` 收益，EOD 信号通常 `shift(1)`。
- 对 `o2o`、`v2v` 收益，示例中使用 `shift(2)`，避免未来函数并匹配可交易时点。

### 5. 分组测试

调用：

```python
signal_rank, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(signal, ret_matrix, n=10)
```

`groupTest` 会完成：

- 每日截面按因子值分成 `n` 组。
- 计算各组等权收益。
- 构造 `H-L` 组合。
- 统计年化收益、Sharpe、最大回撤、换手、Rank IC、ICIR。
- 绘制累计收益曲线和分组平均收益柱状图。

返回值：

- `signal_rank`：每只股票每日所属分组。
- `group_pnl_df`：各组每日收益和 `H-L` 收益。
- `group_to_df`：各组每日换手和 `H-L` 换手。

### 6. 因子中性化

如果需要做行业、市值中性化：

```python
factor_value_neu = Factor_Dev_Lib.panel_neutral_size_ind(
    factor_value,
    del_limit=False,
    del_st=False,
    nt_type='ind_cap',
)
```

`nt_type` 支持：

- `ind_cap`：行业 + 市值中性化。
- `ind`：行业中性化。
- `cap`：市值中性化。

该函数依赖 DolphinDB 中已预热的 `PREHEAT_IND_DATA_CITICS`。如未预热，需要先运行 `data_preheat.py` 中的 `preheat_ind_data(...)`。

## 分钟级因子测试流程

分钟级因子测试的主要入口是 `intraday_Factortest.py`，核心工具在 `intraday_lib.py`。

### 1. 加载 DolphinDB 函数

先连接 DolphinDB，并加载项目内置的 DolphinDB 函数：

```python
s = ddb.session(protocol=keys.PROTOCOL_DDB)
s.connect(**DATA_DB_CONN)
s.run(intraday_lib.ddb_functions)
```

这些函数包括股票池筛选、分钟收益矩阵、分组表现、涨跌停状态等逻辑。

### 2. 预热日内收益矩阵

日内测试依赖预热后的 DolphinDB 共享对象。可运行：

```python
import datetime as dt
from data_preheat import preheat_ret_matrix

startDay = dt.datetime(2020, 1, 1)
endDay = dt.datetime(2026, 4, 24)

preheat_ret_matrix(startDay, endDay, '000852.SH')   # 生成 PREHEAT_RET_MATRIX_ZZ1000
preheat_ret_matrix(startDay, endDay, '932000.CSI')  # 生成 PREHEAT_RET_MATRIX_ZZ2000
```

`preheat_ret_matrix` 会调用 `intraday_lib.ddb_functions` 中的 `get_ret_matrix`，生成包含多个未来收益窗口的分钟级收益矩阵。

常见收益列包括：

- `Ret_15`
- `Ret_30`
- `Ret_60`
- `Ret_90`
- `Ret_120`
- `Ret_180`
- `Ret_EOD`
- `Ret_NDay`

### 3. 准备分钟级因子窄表

日内因子推荐整理成窄表：

```text
tradetime | symbol | factorname | value
```

示例：

```python
factor_value.columns = ['tradetime', 'symbol', 'value', 'factorname']
factor_value = factor_value[['tradetime', 'symbol', 'factorname', 'value']]
s.upload({'Factor_example': factor_value})
```

`tradetime` 应包含日期和分钟时间，例如 `2020-01-02 09:59:00`。项目示例使用的测试时间点包括：

- `09:59`
- `10:29`
- `10:59`
- `11:29`
- `13:29`
- `13:59`
- `14:29`

### 4. 运行日内回测

`intraday_Factortest.py` 中提供了封装函数：

```python
backtest_factor_name('Factor_example', index_code='000852.SH')
```

该函数会：

- 从 DolphinDB 当前 session 中读取同名因子表。
- 调用 `filter_in_index` 过滤指数成分股。
- 调用 `get_cs_group_performance` 做横截面分组测试。
- 按 `Bartime` 和未来收益窗口分析各组表现。
- 生成 `group_HML` 的年化收益和 Sharpe 热力图。
- 保存绩效汇总 CSV。

### 5. 查看输出结果

日内结果默认写入：

```text
result/<factor_name>/
```

常见输出：

- `group_performance_summary.csv`
- `group_HML_annualized_return_heatmap.png`
- `group_HML_sharpe_heatmap.png`
- `symbol_count_by_tradetime.png`
- 可选的各收益窗口分组累计收益图。

## 数据源说明

项目主要依赖以下数据：

- DolphinDB：
  - `dfs://WIND.ASHAREEODPRICES`
  - `dfs://WIND.ASHARECALENDAR`
  - `dfs://WIND.AINDEXEODPRICES`
  - `dfs://WIND.ASHAREEODDERIVATIVEINDICATOR`
  - `dfs://QV_Trade_to_MinuteBar`
  - 指数成分权重与成员表等。
- Oracle Wind：
  - 行业分类、行业代码字典和行业层级查询。
- MySQL、ClickHouse：
  - 当前主要用于 `DB_Demo.py` 展示连接方式。

## 常见注意事项

- 不要直接用普通 `python Factor_Test_Process.py` 跑示例脚本；文件中包含 `%load_ext autoreload` 等 IPython magic，推荐交互式运行。
- 日频测试前要明确交易假设，并正确设置 `signal.shift(...)`。错误对齐会造成未来函数。
- `groupTest` 默认按每日截面等权分组，适合快速研究，不等同于完整交易系统回测。
- 中性化依赖行业预热数据。若 `PREHEAT_IND_DATA_CITICS` 不存在或日期范围不足，`panel_neutral_size_ind` 会报错。
- 日内测试依赖预热收益矩阵，例如 `PREHEAT_RET_MATRIX_ZZ1000`。如果 DolphinDB session 或服务重启，共享对象可能需要重新生成。
- 当前 `Factor_Dev_Lib.py` 的 `__main__` 中调用了 `get_EOD_Not_ST_fast`，但项目内未定义该函数，建议不要把该 `__main__` 块作为正式入口。
- 结果图默认会尝试 `plt.show()`，在无图形界面的服务器环境中可以把相关函数的 `show_plots` 或 `show_plot` 设为 `False`。

## 研报复现工厂（repro/）

批量复现 Quant-Report 等 PDF 研报时，使用 `repro/` 目录（不修改核心库为主）：

- 台账：`repro/catalog/index.csv`（已登记广发 Alpha 系列 1–25）
- 第一篇样板：`repro/workspace/gf_alpha_03_val_mom/`
- 运行：`python repro/harness/run_daily_factor.py --id gf_alpha_03_val_mom`
- 说明见 [repro/README.md](repro/README.md)

## 推荐开发顺序

1. 在 `COMMON_CONST.py` 确认数据库连接可用。
2. 在交互式环境中跑通 `Factor_Test_Process.py` 的日频示例。
3. 将自己的因子计算逻辑替换为新的 `factor_value` 宽表。
4. 使用 `get_EOD_Not_Limit`、`get_EOD_Not_ST`、`get_TradeStatus` 做基础可交易过滤。
5. 使用 `get_Ret_Matrix` 选择收益口径，并正确 shift 信号。
6. 调用 `groupTest` 查看分组单调性、H-L 表现、IC/ICIR 和换手。
7. 如需做分钟级因子，首先检查预热数据（一般来说是正常的），再用 `intraday_Factortest.py` 的窄表流程测试。

