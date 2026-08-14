# Factor_Lib
"因子测试常用常见的库"
# %% modules
import datetime as dt
import pandas as pd
import dolphindb as ddb
import numpy as np
from COMMON_CONST import DATA_DB_CONN
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
# %%
'''去极值'''
def wsr(FCT_RAW, NSG=5):
    FCT = FCT_RAW.dropna(how='all')
    upp = FCT.median(1) + NSG * FCT.mad(1)
    low = FCT.median(1) - NSG * FCT.mad(1)
    FCT = FCT[FCT.sub(upp, 0) <= 0].fillna(FCT.add(999 * upp, 0).div(1000))
    FCT = FCT[FCT.sub(low, 0) >= 0].fillna(FCT.add(999 * low, 0).div(1000))
    FCT = FCT.reindex_like(FCT_RAW)
    return FCT

def _mad_(factor, threshold=3, tanh=True):
    """ Median Absolute Deviation
    中位数绝对偏差法（MAD）是一种去除异常值的数据预处理方法，需要先计算所有因子与中
    位数的距离总和来检测离群值。具体方法如下：
        1. 找出因子的中位数median
        2. 得到每个因子值与中位数的绝对偏差值 | x - median |
        3. 得到绝对偏差值的中位数 MAD = median(| x - median |)
        4. 计算 MAD_e = 1.4826 * MAD，然后确定乘数n，做出调整
        5. 超出 MAD_e的值即为异常值，用 MAD +- MAD_e 替代
        6. 如果tanh=True,则对factor做tanh变换，factor = np.tanh(factor),使得
                在 +-inf 处的极限值为 MAD +- MAD_e
            做 factor = np.tanh(factor * 1.212) 变换，可进一步使得
                在 median +- 2/3 * MAD处的值保持不变
    用法：
        facotr = mad(factor)
        factor = mad(factor, threhold=3)
        factor = mad(factor, tanh=True)
    """
    factor = pd.Series(factor)
    median = factor.median()
    # 用mean absolute deviation代替运行速度较快，但数据有inf时=inf
    # pandas 计算的mad使用的mean 而不是常规的mad, 为了统一所以把以下代码注释掉
    # MAD = factor.mad()
    # if MAD == np.inf:
    #     # 按照定义取中位数，很多0时容易=0
    #     MAD = (factor - median).abs().median()

    MAD = (factor - median).abs().median()

    if MAD == 0 or pd.isna(MAD):  # 所有数据的值都相等或都为nan
        return factor

    assert MAD > 0, 'MAD=%f，数据中有太多相同的值，可能时fillna导致的' % MAD
    assert MAD < np.inf, 'MAD=inf，数据有inf值'
    width = MAD * 1.4826 * threshold
    if not tanh:
        factor = factor.clip(lower=median + width, upper=median - width)
    else:
        x = (factor - median) / width
        x = x.apply(lambda x: np.tanh(x * 1.212))
        factor = x * width + median
    return factor


"去极值的方法"
def mad(factor, threshold=3, tanh=True):
    """
    此方法包含了两种去极值的方法：
    当tanh = False，
    使用的是
        Median Absolute Deviation
        中位数绝对偏差法（MAD）是一种去除异常值的数据预处理方法，需要先计算所有因子与中
        位数的距离总和来检测离群值。
    当 tanh = True,
    使用可以保持因子值排序与原先不变，经过变化之后，输出值在-1与1之间
    """
    output = factor.apply(lambda x: _mad_(x, threshold, tanh), axis=1)

    return output


'''标准化:均值标准差'''
def zsc(FCT):
    FCT = FCT.sub(FCT.mean(1), 0).div(FCT.std(1), 0)
    return FCT


# %% 获取交易日相关函数
def get_TradingDay(startDay, endDay, exchange='SSE'):
    "获取交易日"
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    startDay_str = startDay.strftime('%Y.%m.%d')
    endDay_str = endDay.strftime('%Y.%m.%d')
    t = s.loadTable(dbPath='dfs://WIND.ASHARECALENDAR', tableName='data')
    df = t.select('TRADE_DAYS').where(
        f"TRADE_DAYS >= {startDay_str} and TRADE_DAYS <= {endDay_str} and S_INFO_EXCHMARKET = '{exchange}'").sort(
        'TRADE_DAYS').toDF()
    df = df.iloc[:, 0]
    df.name = 'TradingDay'
    s.close()
    return df

# %% 涨跌停状态
def get_EOD_Not_Limit(startDay, endDay):
    """
    获取日末不是涨跌停的股票
    返回值是pd.DataFrame, 其中1代码非涨跌停, NaN代码可能是涨跌停
    """

    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    t_EOD = s.loadTable(dbPath='dfs://WIND.ASHAREEODPRICES', tableName='data')
    t_EOD = t_EOD.where(f"TRADE_DT>= {startDay.strftime('%Y.%m.%d')} and TRADE_DT <= {endDay.strftime('%Y.%m.%d')} ")

    # 涨跌停
    t_limit = t_EOD.select('S_DQ_CLOSE as Close').where(
        ' (S_DQ_CLOSE < S_DQ_LIMIT) and (S_DQ_CLOSE > S_DQ_STOPPING)').pivotby('TRADE_DT', 'S_INFO_WINDCODE')
    df_notLimit = t_limit.toDF()
    df_notLimit = df_notLimit.set_index('TRADE_DT')
    df_notLimit = df_notLimit.mask(~pd.isna(df_notLimit), 1)
    s.close()

    return df_notLimit

# %% 变为ST
def get_EOD_Not_ST(startDay, endDay):
    """
    获取日末非 ST 股票标记矩阵。

    返回值为 ``pd.DataFrame``：
    - ``1`` 表示该股票在对应交易日为非 ST 状态
    - ``NaN`` 表示该股票在对应交易日可能为 ST 或退市整理相关状态

    实现思路：
    1. 从股票名称变更表中提取名称区间记录
    2. 仅保留名称中命中 ``ST`` 或 ``退`` 的区间
    3. 将这些区间展开为日频窄表 ``(date, code, flag)``
    4. 再转为宽表，并映射成最终的 ``1 / NaN`` 结果矩阵

    该实现利用了 ST 区间通常较少的特点，只处理中间的稀疏异常区间，
    因而在多数场景下会比直接处理全量名称区间更高效。
    """

    s = ddb.session()
    s.connect(**DATA_DB_CONN)

    tradingDay = pd.DatetimeIndex(get_TradingDay(startDay, endDay))
    endDay_ts = pd.Timestamp(endDay)

    t_Code = s.loadTable(dbPath='dfs://WIND.ASHAREPREVIOUSNAME', tableName='data')
    t_Code = t_Code.select('BEGINDATE, ENDDATE, S_INFO_WINDCODE, S_INFO_NAME').where(
        'S_INFO_WINDCODE like "%.SH" or S_INFO_WINDCODE like "%.SZ"').toDF()

    if t_Code.empty:
        s.close()
        return pd.DataFrame(index=tradingDay, dtype=float)

    # 先把区间数据整理干净，保证每个股票在同一个起始日只保留最后一条记录。
    # 对于最后一条记录，如果原表ENDDATE为空，则视作一直持续到本次请求的endDay。
    t_Code = t_Code.sort_values(['S_INFO_WINDCODE', 'BEGINDATE', 'ENDDATE'])
    t_Code = t_Code.drop_duplicates(['S_INFO_WINDCODE', 'BEGINDATE'], keep='last')
    t_Code['BEGINDATE'] = pd.to_datetime(t_Code['BEGINDATE'])
    t_Code['ENDDATE'] = pd.to_datetime(t_Code['ENDDATE']).fillna(endDay_ts)
    last_row_mask = ~t_Code['S_INFO_WINDCODE'].duplicated(keep='last')
    t_Code.loc[last_row_mask, 'ENDDATE'] = endDay_ts

    stock_codes = pd.Index(t_Code['S_INFO_WINDCODE'].drop_duplicates())

    # 稀疏思路的关键: 中间过程只保留 ST/退 名称区间。
    # 这些区间对应“最终结果里需要被置为NaN的位置”，其余位置默认都是1。
    st_records = t_Code[t_Code['S_INFO_NAME'].str.contains('ST|退', regex=True, na=False)].copy()

    if st_records.empty:
        s.close()
        return pd.DataFrame(1.0, index=tradingDay, columns=stock_codes)

    # 用searchsorted把区间日期直接映射到交易日下标:
    # left/right 分别是每条ST区间在 tradingDay 中覆盖的起止位置。
    trade_days_arr = tradingDay.to_numpy(dtype='datetime64[ns]')
    left = np.searchsorted(trade_days_arr, st_records['BEGINDATE'].to_numpy(dtype='datetime64[ns]'), side='left')
    right = np.searchsorted(trade_days_arr, st_records['ENDDATE'].to_numpy(dtype='datetime64[ns]'), side='right')

    narrow_parts = []
    for code, start_idx, end_idx in zip(st_records['S_INFO_WINDCODE'].to_numpy(), left, right):
        if start_idx >= end_idx:
            continue

        # 这里显式展开成窄表:
        # 一条ST区间会被展开成若干行 (TRADE_DT, S_INFO_WINDCODE, ST_FLAG=1)。
        # 后面再统一 pivot 成 ST 标记宽表。
        narrow_parts.append(pd.DataFrame({
            'TRADE_DT': tradingDay[start_idx:end_idx],
            'S_INFO_WINDCODE': code,
            'ST_FLAG': 1.0
        }))

    if narrow_parts:
        df_st_narrow = pd.concat(narrow_parts, ignore_index=True)
        df_st_wide = df_st_narrow.pivot_table('ST_FLAG', index='TRADE_DT', columns='S_INFO_WINDCODE', aggfunc='last')
    else:
        df_st_wide = pd.DataFrame(index=tradingDay)

    # df_st_wide 里: 有值表示该日该股票命中了ST区间。
    # 最终输出要求的是“非ST=1, ST=NaN”，因此这里做一次取反映射。
    df_st_wide = df_st_wide.reindex(index=tradingDay, columns=stock_codes)
    df_ST = pd.DataFrame(
        np.where(df_st_wide.notna(), np.nan, 1.0),
        index=tradingDay,
        columns=stock_codes
    )

    s.close()

    return df_ST

# %% 交易状态
def get_TradeStatus(startDay, endDay):
    """
    获取股票的交易状态。

    返回值为 ``pd.DataFrame``：
    - ``1`` 表示该股票在对应交易日为非停牌状态
    - ``NaN`` 表示该股票在对应交易日为停牌状态

    该版本保留原有的 DolphinDB 查询方式，但将 pandas 侧的多次 ``mask``
    映射压缩为一次向量化布尔判断，以减少宽表转换后的后处理开销。
    """
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    t_EOD = s.loadTable(dbPath='dfs://WIND.ASHAREEODPRICES', tableName='data')
    t_tradeStatus = t_EOD.select('S_DQ_TRADESTATUS').where(f"TRADE_DT>= {startDay.strftime('%Y.%m.%d')} and TRADE_DT <= {endDay.strftime('%Y.%m.%d')} ").pivotby('TRADE_DT', 'S_INFO_WINDCODE').executeAs('t_tradeStatus')

    df_tradeStatus = t_tradeStatus.toDF()
    df_tradeStatus = df_tradeStatus.set_index('TRADE_DT')

    # 原函数会把空字符串和“停牌”都映射为 NaN，其余状态统一映射为 1。
    suspend_mask = df_tradeStatus.eq('停牌') | df_tradeStatus.eq('')
    df_tradeStatus = pd.DataFrame(
        np.where(suspend_mask, np.nan, 1.0),
        index=df_tradeStatus.index,
        columns=df_tradeStatus.columns
    )
    s.close()

    return df_tradeStatus

# %% 获取ret matrix 的数据
def get_Ret_Matrix(startDay, endDay, method='c2c', base_index=None):
    """
    获取股票的日级收益的矩阵
    startDay: 开始日期
    endDay: 停止日期
    method: c2c, Close 2 Close return, 还可以填写 o2o, 则返回open 2 open return, v2v, vwap 2 vwap return
    base_index: 如果填入,000016.SH,000300.SH,000905.SH, 000852.SH则返回股票相对于指数的超额，默认返回股票自己的收益
    """
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    # 一些数据涉及到shift操作，所以需要预热，需要多提取一定的时间
    startDay_preHeat = startDay - dt.timedelta(days=10)
    t_EOD = s.loadTable(dbPath='dfs://WIND.ASHAREEODPRICES', tableName='data')
    t_EOD = t_EOD.where(
        f"TRADE_DT>= {startDay_preHeat.strftime('%Y.%m.%d')} and TRADE_DT <= {endDay.strftime('%Y.%m.%d')} ")

    if method == 'c2c':
        t_ret = t_EOD.select(
            "TRADE_DT as Date, S_INFO_WINDCODE as WindCode, (S_DQ_CLOSE/S_DQ_PRECLOSE-1) as Ret").executeAs('t_ret')
    elif method == 'o2o':
        t_ret = t_EOD.select("TRADE_DT as Date, S_INFO_WINDCODE as WindCode, ratios(S_DQ_ADJOPEN)-1 as Ret").contextby(
            'S_INFO_WINDCODE').csort('TRADE_DT').executeAs('t_ret')
    elif method == 'v2v':
        t_ret = t_EOD.select(
            "TRADE_DT as Date, S_INFO_WINDCODE as WindCode, ratios(S_DQ_AVGPRICE*S_DQ_ADJFACTOR)-1 as Ret").contextby(
            'S_INFO_WINDCODE').csort('TRADE_DT').executeAs('t_ret')
    else:
        assert method in ('o2c', 'c2c', 'v2v')

    df_ret = t_ret.select('Ret').pivotby('Date', 'WindCode').toDF()
    df_ret = df_ret.set_index('Date')
    selected_col = [x for x in df_ret.columns if x[0] in ('6', '0', '3')]
    df_ret = df_ret[selected_col]

    "如果返回是超额数据"
    if base_index is None:
        pass
    else:
        t_Index_EOD = s.loadTable(dbPath='dfs://WIND.AINDEXEODPRICES', tableName='data')
        t_Index_EOD = t_Index_EOD.where(
            f"TRADE_DT>= {startDay_preHeat.strftime('%Y.%m.%d')} and TRADE_DT <= {endDay.strftime('%Y.%m.%d')} ")

        if method == 'c2c':
            t_index_ret = t_Index_EOD.select("TRADE_DT as Date, (S_DQ_CLOSE/S_DQ_PRECLOSE-1) as Ret").where(
                f"S_INFO_WINDCODE = '{base_index}'").contextby('S_INFO_WINDCODE').csort('TRADE_DT').executeAs('t_index_ret')
        elif method == 'o2o':
            # csort防止数据库顺序混乱
            t_index_ret = t_Index_EOD.select("TRADE_DT as Date, ratios(S_DQ_OPEN)-1 as Ret").where(
                f"S_INFO_WINDCODE = '{base_index}'").contextby('S_INFO_WINDCODE').csort('TRADE_DT').executeAs('t_index_ret')
        else:
            assert method in ('o2c', 'c2c')

        df_index_ret = t_index_ret.toDF()
        index_ret = df_index_ret.set_index('Date')['Ret']

        df_ret = df_ret.sub(index_ret, axis=0)

    df_ret = df_ret.loc[startDay:endDay, :]
    s.close()
    return df_ret


# %% Sharpe, MDD, AnnurRet
def calSharpe(retSeries, riskFree=0, n=250):
    """
    计算年化 Sharpe。

    参数说明：
    - retSeries: 与风险收益率同频的收益序列
    - riskFree: 同频无风险收益率，默认 0
    - n: 一年内的期数，日频通常取 250

    若输入为空，或超额收益波动率为 0，则返回 NaN。
    """
    excess_ret = pd.Series(retSeries).dropna() - riskFree
    if excess_ret.empty:
        return np.nan

    excess_std = excess_ret.std()
    if pd.isna(excess_std) or excess_std == 0:
        return np.nan

    return excess_ret.mean() / excess_std * (n ** 0.5)

def calMDD(retSeries_ori):
    """
    计算最大回撤，以及对应的开始和结束日期。

    返回值：
    - maxDD: 最大回撤，负数
    - (mddStartDay, mddEndDay): 最大回撤区间

    这里会在序列起点前补一个 0 收益日，使累计净值从 1 开始。
    若输入为空，则返回 (NaN, (NaT, NaT))。
    """
    retSeries = pd.Series(retSeries_ori).dropna().copy()
    if retSeries.empty:
        return np.nan, (pd.NaT, pd.NaT)

    retSeries = retSeries.sort_index()
    retSeries.loc[retSeries.index[0] - dt.timedelta(days=1)] = 0
    retSeries = retSeries.sort_index()

    # 累计净值曲线与历史最高净值（High Water Mark）
    cumProd = (retSeries + 1).cumprod()
    HWM = cumProd.cummax()

    # 回撤定义为当前净值相对历史最高点的跌幅
    DD = (cumProd - HWM) / HWM
    if DD.empty:
        return np.nan, (pd.NaT, pd.NaT)

    maxDD = DD.min()
    mddEndDay = DD.idxmin()

    # 取 mddEndDay 之前最后一次达到该高点的位置，作为回撤起点。
    # 不能取第一次达到该高点的位置，否则在平台期会把回撤起点取早。
    peak_value = HWM.loc[mddEndDay]
    peak_path = cumProd.loc[:mddEndDay]
    mddStartDay = peak_path[peak_path == peak_value].index[-1]

    return maxDD, (mddStartDay, mddEndDay)

def calAnnuRet(retSeries, n=250):
    """
    计算算术年化收益率。

    这里使用平均单期收益乘以年化期数，不是复利口径 CAGR。
    若输入为空，则返回 NaN。
    """
    retSeries = pd.Series(retSeries).dropna()
    if retSeries.empty:
        return np.nan

    return retSeries.mean() * n

def calICIR(signal, ret, n=250):
    """
    计算 rank IC 及年化 ICIR。

    返回值：
    - rank_ic: 日度 rank IC 的平均值
    - rank_icir: rank IC / rank IC 的标准差 * sqrt(n)

    计算前会先按日期和股票代码对齐 signal 与 ret，
    并自动忽略相关系数为 NaN 的日期。
    """
    signal_aligned, ret_aligned = signal.align(ret, join='inner', axis=None)
    if signal_aligned.empty or ret_aligned.empty:
        return np.nan, np.nan

    rank_ic_daily = np.array([
        row.corr(ret_aligned.loc[it, :], method='spearman')
        for it, row in signal_aligned.iterrows()
    ], dtype=float)

    valid_rank_ic = rank_ic_daily[~np.isnan(rank_ic_daily)]
    if valid_rank_ic.size == 0:
        return np.nan, np.nan

    rank_ic = valid_rank_ic.mean()
    rank_ic_std = valid_rank_ic.std()
    if rank_ic_std == 0:
        return rank_ic, np.nan

    rank_icir = rank_ic / rank_ic_std * (n ** 0.5)

    return rank_ic, rank_icir

# %% 分组测试相关
def np_qcut(arr, q):
    """
    基于 numpy 的一维 qcut 近似实现。

    典型用法是对已经做过 ``rank(method='first')`` 的一维数组分组。
    在这种“输入值唯一、只含 NaN 缺失”的场景下，通常可以较好复现
    ``pd.qcut`` 的结果，同时速度更快。

    返回值从 1 开始编号；原数组中的 NaN 位置保持为 NaN。
    """
    arr = np.asarray(arr, dtype=float)
    if q <= 0:
        raise ValueError('q must be a positive integer')

    # 构造结果数组，默认填充 NaN，避免额外处理缺失位置。
    res = np.full(arr.size, np.nan, dtype=float)

    # NaN 的结果不参与计算。
    na_mask = np.isnan(arr)
    x = arr[~na_mask]  # 实际参与计算的数据

    if x.size == 0:
        return res
    if x.size == 1:
        res[~na_mask] = 1.0
        return res

    # 获取划分关键字
    # 这部分代码是pd.qcut内部实现的做法
    sorted_x = np.sort(x)
    idx = np.linspace(0, 1, q + 1) * (sorted_x.size - 1)
    pos = idx.astype(int)
    fraction = idx % 1
    a = sorted_x[pos]
    b = np.roll(sorted_x, shift=-1)[pos]
    bins = a + (b - a) * fraction
    # 由于使用的是np.digitize，bins的第一个数值如果是数组中的最小值的话，
    # 划分后会将其放到【第0组】，因此这里将第一个值减一，
    # 可以将待划分数组中的所有值都包含进去
    bins[0] -= 1

    res[~na_mask] = np.digitize(x, bins, right=True)
    return res


def _rank_to_bins_npqcut(signal, n):
    """
    使用 ``np_qcut`` 对每个截面进行快速分箱。

    与 ``signal.apply(lambda x: pd.qcut(x.rank(method='first'), ...), axis=1)``
    的目标一致，但避免频繁构造 pandas 对象，适合大规模截面分组。
    """
    rank_arr = signal.rank(axis=1, method='first').to_numpy(dtype=float)
    signal_rank = np.apply_along_axis(np_qcut, 1, rank_arr, n)
    return pd.DataFrame(signal_rank, index=signal.index, columns=signal.columns, dtype=float)


# 万分之 7.5：用 H-L 日均换手估算年化隐含交易成本（展示用，非实扣费）
IMPLIED_ANNU_FEE_BPS = 7.5


def implied_annu_fee(avg_daily_turnover_hl, fee_bps=IMPLIED_ANNU_FEE_BPS):
    """Implied annual fee from H-L daily turnover: TO × (bps/1e4) × 250."""
    if avg_daily_turnover_hl is None or (isinstance(avg_daily_turnover_hl, float) and np.isnan(avg_daily_turnover_hl)):
        return np.nan
    return float(avg_daily_turnover_hl) * float(fee_bps) / 100.0 / 100.0 * 250.0


def format_group_test_stats_title(
    *,
    direction,
    annu_ret,
    sharpe,
    mdd,
    avg_turnover,
    rank_ic,
    icir,
    implied_fee=None,
    fee_bps=IMPLIED_ANNU_FEE_BPS,
):
    """Standard xlabel / caption for decile + H-L cumulative plots (includes Implied AnnuFee)."""
    if implied_fee is None:
        implied_fee = implied_annu_fee(avg_turnover, fee_bps=fee_bps)
    return ','.join([
        f'H-L, Direction: {direction}, AnnuRet: {annu_ret:.2%}',
        f'Sharpe_Ratio: {sharpe:.2f}, MDD: {mdd:.2%}, Daily Turnover: {avg_turnover:.2f}',
        f'\n Implied AnnuFee({fee_bps:g}%): {implied_fee:.2%}, Daily IC: {rank_ic:.4f}, Annu ICIR: {icir:.2f}',
    ])


def groupTest(signal, ret, n=10, fee=0, info='Factor_Performance'):
    """
    基于截面分组的日频因子测试函数。

    输入要求：
    - ``signal``: 因子值宽表，index 为日期，columns 为股票代码
    - ``ret``:    与 ``signal`` 对齐的未来收益宽表，index/columns 需一致
    - ``n``:      分组数，默认分成 10 组
    - ``fee``:    单边换手费率，按日换仓时直接乘以当日换手
    - ``info``:   绘图标题；传 ``'silent'`` 时跳过绘图（仍计算并返回分组结果）

    返回值：
    - ``signal_rank``: 每日每只股票所属分组，取值为 ``1 ~ n``
    - ``group_pnl_df``: 各组每日收益，以及 ``H-L`` 组合收益
    - ``group_to_df``:  各组每日换手，以及 ``H-L`` 组合换手

    核心流程：
    1. 先对每个交易日的因子截面做 ``rank(method='first')``
    2. 再使用 ``np_qcut`` 将排名后的截面近似等分为 ``n`` 组
    3. 每组内部做等权组合，计算每日收益和每日换手
    4. 构造 ``H-L`` 组合，并统计 Sharpe、MDD、年化收益、IC、ICIR、Implied AnnuFee 等指标
    5. 绘制累计收益曲线和分组平均收益柱状图（``info!='silent'``）

    使用说明：
    - 本函数默认适用于日频、每日换仓的测试场景
    - ``signal`` 与 ``ret`` 最好在调用前就完成对齐
    - 当前版本使用 ``np_qcut`` 提升分箱速度，在实际测试中通常能较好复现
      ``pd.qcut`` 的结果，同时显著降低分箱阶段耗时
    - 图横轴说明中的 Implied AnnuFee 按日均 H-L 换手 × 万分之 7.5 × 250 估算
    """
    bins = list(range(1, n + 1))

    # 先将每日截面分到 1~n 组。这里使用基于 numpy 的 qcut 实现，
    # 目的在于减少 pandas 逐行分箱的开销。
    signal_rank = _rank_to_bins_npqcut(signal, n)

    group_pnl_df = pd.DataFrame(index=signal_rank.index)
    group_to_df = pd.DataFrame(index=signal_rank.index)
    for group in bins:
        # 组内等权：先构造“是否属于该组”的布尔矩阵，再按每行股票数量归一化。
        group_mask = signal_rank.eq(group)
        group_count = group_mask.sum(axis=1).replace(0, np.nan)
        signal_wgt = group_mask.div(group_count, axis=0)

        # 组收益 = 权重 * 个股收益，再做横截面求和。
        signal_perf = signal_wgt.mul(ret)
        daily_pnl = signal_perf.sum(axis=1)

        # 组换手按相邻两日权重变动绝对值求和计算；
        # 首日没有前一日持仓，因此直接把首日建仓权重之和记为换手。
        signal_turnover = signal_wgt.fillna(0).diff().abs().sum(axis=1)
        signal_turnover.iloc[0] = signal_wgt.iloc[0].fillna(0).abs().sum()
        signal_fee = signal_turnover * fee

        # 若该组平均收益为负，则费用方向取反，使扣费逻辑与原函数保持一致。
        direction = 1 if daily_pnl.mean() > 0 else -1
        daily_pnl = daily_pnl - signal_fee * direction
        group_pnl_df[group] = daily_pnl
        group_to_df[group] = signal_turnover

    # 最高组减最低组，构造 H-L 多空组合。
    group_pnl_df['H-L'] = group_pnl_df[n] - group_pnl_df[1]
    group_to_df['H-L'] = group_to_df[n] + group_to_df[1]

    # 统计展示时，若 H-L 均值为负，则整体乘以 -1，便于统一展示策略方向。
    daily_pnl = group_pnl_df['H-L']
    direction = 1 if daily_pnl.mean() > 0 else -1
    daily_pnl = daily_pnl * direction
    sp_ratio = calSharpe(daily_pnl)
    mdd, _ = calMDD(daily_pnl)
    annuRet = calAnnuRet(daily_pnl)
    avg_to = group_to_df['H-L'].mean()
    implied_fee = implied_annu_fee(avg_to)

    # 逐日计算截面 rank IC，再汇总为平均 IC 与年化 ICIR。
    rank_ic_daily = signal.corrwith(ret, axis=1, method='spearman')
    rank_ic = np.mean(rank_ic_daily)
    rank_ic_std = np.std(rank_ic_daily)
    rank_icir = rank_ic / rank_ic_std * (250 ** 0.5) if rank_ic_std and rank_ic_std > 0 else np.nan

    # 汇总统计信息展示在图的横轴说明中（含 Implied AnnuFee）。
    title = format_group_test_stats_title(
        direction=direction,
        annu_ret=annuRet,
        sharpe=sp_ratio,
        mdd=mdd,
        avg_turnover=avg_to,
        rank_ic=rank_ic,
        icir=rank_icir,
        implied_fee=implied_fee,
    )

    silent = (info is None) or (str(info).strip().lower() == 'silent')
    if not silent:
        # 绘制各组及 H-L 的累计收益曲线。
        cum_pnl_df = group_pnl_df.cumsum()
        fig, ax = plt.subplots(figsize=(20, 12))
        for col_name, y in cum_pnl_df.items():
            ax.plot(y.index, y, label=col_name)
            ax.text(y.index[-1], y.iloc[-1], col_name, fontsize=15, verticalalignment='bottom')
        ax.legend(loc='upper left')
        ax.set_xlabel(title, fontsize=15)
        ax.set_title(info)
        plt.show()

        # 绘制各组平均收益柱状图，便于快速查看单调性。
        group_pnl_df.mean().plot(kind='bar', title=info)
        plt.show()

    return signal_rank, group_pnl_df, group_to_df
#%%
def calculate_portfolio_returns(weights, returns):
    """
    支持每天调整仓位，同时也支持任意 天/周期 调整仓位，非调整日期保持上一天的仓位
    在非调整日期间动态调整权重（权重会受到股票每天收益的波动而变化），然后计算组合的每日收益。
    weights里面每一个日期都视为调整日，没有的视为非调整日
    """
    # 初始化投资组合每日收益的Series
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    portfolio_turnover = pd.Series(index=returns.index, dtype=float)

    # 初始化调整日的权重
    last_adjusted_weights = pd.Series(data=0, index=weights.columns)

    # 初始化投资组合市值（初始假设为1）
    portfolio_value = 1
    # 遍历每一天的收益
    for i, (current_date, current_returns) in enumerate(returns.iterrows()):
        daily_return = 0
        daily_turnover = 0
        if current_date in weights.index:
            # 每个调整周期的, 使用新的权重，并计算当日投资组合市值
            target_wgt = weights.loc[current_date]
            daily_turnover = target_wgt.sub(last_adjusted_weights, fill_value=0).abs().sum()
            last_adjusted_weights = target_wgt
            daily_weighted_returns = last_adjusted_weights * current_returns
            daily_return = daily_weighted_returns.sum()
        else:
            # 非调整日，需要根据实际收益修正权重
            # 首先计算未调整权重的日收益
            daily_weighted_returns = last_adjusted_weights * current_returns
            daily_return = daily_weighted_returns.sum()

        # 然后更新每只股票的价值，并以此来修正权重
        last_adjusted_weights *= (1 + current_returns)

        # 用更新后的股票价值除以组合总价值，得到新一天的修正权重
        last_adjusted_weights /= last_adjusted_weights.sum()

        # 更新投资组合价值
        portfolio_value *= (1 + daily_return)

        # 记录每日收益
        portfolio_returns.loc[current_date] = daily_return
        portfolio_turnover.loc[current_date] = daily_turnover

    return portfolio_returns, portfolio_turnover




# %% 中性化
def get_preheat_ind_data_citics(startDay, endDay):
    """
    获取指定日期区间的股票中信一级行业分类数据（已“预热”好的宽表形式）。

    本函数用于从 DolphinDB 共享库“PREHEAT_IND_DATA_CITICS”中提取行业归属宽表数据。
    预热数据已按交易日、股票代码展开。每只股票每天对应一个行业代码（IND_CODE）。
    通常用于因子处理中的行业中性化、分组、分类等。

    参数
    ------
    startDay : datetime.datetime
        起始日期。
    endDay : datetime.datetime
        结束日期。

    返回
    ------
    pd.DataFrame
        行业宽表。index为TradingDay（交易日），columns为各股票代码，元素为IND_CODE。

    注：
    - 需要保证DolphinDB服务器已事先run过 data_preheat.py 的预热脚本，
      并且有“PREHEAT_IND_DATA_CITICS”共享表可用。
    - 如果请求的startDay/endDay超出了预热缓存，函数会报错提示。

    示例
    ------
    >>> start = dt.datetime(2023, 1, 1)
    >>> end = dt.datetime(2024, 1, 31)
    >>> ind_df = get_preheat_ind_data_citics(start, end)
    """

    # 创建DolphinDB数据库会话
    s = ddb.session()
    s.connect(**DATA_DB_CONN)

    # 查询预热数据的最早与最晚日期，确保请求的数据范围在缓存期内
    sql_script = f'''
    first_day = exec min(TradingDay) from PREHEAT_IND_DATA_CITICS
    last_day = exec max(TradingDay) from PREHEAT_IND_DATA_CITICS
    '''
    s.run(sql_script)
    first_day = pd.to_datetime(s.run('first_day'))
    last_day = pd.to_datetime(s.run('last_day'))


    trading_days = get_TradingDay(startDay, endDay)
    startDay_in_trading = trading_days.iloc[0]
    endDay_in_trading = trading_days.iloc[-1]


    startDay_str = startDay.strftime('%Y.%m.%d')
    endDay_str = endDay.strftime('%Y.%m.%d')
    
    # 校验请求区间是否超出预热的可用范围
    cond1 = first_day > startDay_in_trading
    cond2 = last_day < endDay_in_trading
    if cond1 or cond2:
        raise ValueError(
            f'预热数据日期不在输入范围内: {startDay_in_trading} 到 {endDay_in_trading}, 预热数据日期范围: {first_day} 到 {last_day}'
        )
    
    # 取区间内的行业宽表
    A_stocks_ind = s.run(
        f"select * from PREHEAT_IND_DATA_CITICS where TradingDay >= {startDay_str} and TradingDay <= {endDay_str}"
    )
    
    s.close()
    return A_stocks_ind




# 在截面上回归得到中性化后的因子值
def cs_neutral_size_ind(signal, ind, cap, nt_type):
    "在截面上回归得到中性化后的因子值"
    signal = signal[signal.notnull() & ind.notnull() & cap.notnull()]
    if signal.isnull().all():
        return pd.Series(index=signal.index, name=signal.name, data=np.nan)
    ind = ind[signal.index]
    cap = cap[signal.index]

    if nt_type == 'ind_cap':
        # OLS 要求输入是纯数值矩阵，这里显式转成 float，避免 bool/object dtype。
        X = pd.get_dummies(ind, dtype=float)
        X['cap'] = cap.astype(float)
        X = sm.add_constant(X).astype(float)
    elif nt_type == 'ind':
        X = pd.get_dummies(ind, dtype=float)
        X = sm.add_constant(X).astype(float)
    elif nt_type == 'cap':
        X = pd.DataFrame({'cap': cap.astype(float)})
        X = sm.add_constant(X).astype(float)
    else:
        raise ValueError(f'Unknown nt_type: {nt_type}')

    Y = signal.astype(float)
    resid = sm.OLS(Y, X).fit().resid
    resid = pd.Series(index=signal.index, data=resid)

    return resid


# 市值、行业中性化（回归法）
def panel_neutral_size_ind(signal, del_limit=False, del_st=False, nt_type='ind_cap'):
    '''
    建议对输入因子之前做去极值、中心化等处理，本函数默认不做处理。
    对因子进行市值行业中性化处理：
    输入：
    signal：未中性化处理的因子，宽表结构，index为日期，columns是股票代码
    del_limit:中性化处理时是否去除当天涨跌停股票的因子数据
    del_st：中性化处理时是否去除st股票
    nt_type:可选'ind_cap','cap','ind',分别对应市值行业中性化，市值中性化，行业中性化
    输出：
    中性化后的因子，结构与输入signal一致
    '''
    startDay = signal.index[0]
    endDay = signal.index[-1]
    # 剔除当日涨跌停因子
    if del_limit:
        df_notLimit = get_EOD_Not_Limit(startDay, endDay)
        signal = signal.mul(df_notLimit)
    # 剔除ST因子
    if del_st:
        df_notST = get_EOD_Not_ST(startDay, endDay)
        signal = signal.mul(df_notST)

    startDay_str = startDay.strftime('%Y.%m.%d')
    endDay_str = endDay.strftime('%Y.%m.%d')

    ##获取行业
    s = ddb.session()
    s.connect(**DATA_DB_CONN)

    A_stocks_ind = get_preheat_ind_data_citics(startDay, endDay)
    A_stocks_ind = A_stocks_ind.set_index('TradingDay')
    A_stocks_ind = A_stocks_ind.reindex_like(signal)

    ##获取市值
    s.run(f'''
               A_stocks_value = select S_INFO_WINDCODE,TRADE_DT,S_VAL_MV,S_DQ_MV from loadTable('dfs://WIND.ASHAREEODDERIVATIVEINDICATOR', 'data') 
               where TRADE_DT>= {startDay_str} and TRADE_DT <= {endDay_str} context by TRADE_DT,S_INFO_WINDCODE csort OPDATE limit 1 
              ''')
    data_cap = s.run('A_stocks_value')
    s.close()

    data_cap = data_cap.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_VAL_MV')

    data_cap = data_cap.reindex_like(signal)
    data_cap = np.log(data_cap)

    # 市值去极值，标准化
    data_cap = mad(data_cap)
    data_cap = zsc(data_cap)

    signal_nt = pd.DataFrame(index=signal.index, columns=signal.columns, data=np.nan)

    for date in signal.index:
        signal_nt.loc[date] = cs_neutral_size_ind(signal.loc[date], A_stocks_ind.loc[date], data_cap.loc[date], nt_type)

    return signal_nt

#%% test
if __name__ == '__main__':
    startDay = dt.datetime(2020,1,1)
    endDay = dt.datetime(2025,12,31)

    not_st = get_EOD_Not_ST(startDay, endDay)
    not_st = get_EOD_Not_ST_fast(startDay, endDay)

#%%
"""
TASK TO DO:
1. 没有排除ST,涨跌停，应该怎么做？
2. 还有没有更好的测试逻辑和方法？
"""