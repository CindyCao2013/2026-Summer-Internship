# cs_signal_T0

"""
本文件用于测试日内分钟级的因子
由于日内分钟级的数据量较大，因为数据和计算难度，我们做了以下调整：
1. 需要结合DolphinDB,以提升速度和计算效率
2. 测试因子，主要以中证1000成分股池，2000成分股池
3. 一般来说，虽然因子每分钟都可以算，但是考虑到计算效率和存储大小，我们一般选择计算池是9:59，10:29,10:59,11:29,13:29,13:59,14:29
4. 由于日内分钟级因子，我们的计算通常是考虑多个时间周期的，所以一般来说采用窄表比较方便
"""


#%% modules
#%load_ext autoreload
#%autoreload 2
import pandas as pd 
import numpy as np
import datetime as dt
import dolphindb as ddb
import dolphindb.settings as keys
import importlib
import intraday_lib
from COMMON_CONST import DATA_DB_CONN



#%% DDB connection
s = ddb.session(protocol=keys.PROTOCOL_DDB)
s.connect(**DATA_DB_CONN)

#%% DDB functions
# 首先加载所需的DDB函数
s.run(intraday_lib.ddb_functions)

#%% 检查是否存在需要的收益率矩阵

s.run('defined("PREHEAT_RET_MATRIX_ZZ1000",SHARED)')
# %% example
"""
计算因子的逻辑
最后将输出factor_value整理成一个窄表形成
tradetime	symbol	factorname	value
0	2020.01.02 09:59:00.000	603959.SH	Factor_XXX	0.0527228858595787
1	2020.01.02 10:29:00.000	603959.SH	Factor_XXX	0.2535860849984447
2	2020.01.02 10:59:00.000	603959.SH	Factor_XXX	0.1734934829930417
3	2020.01.02 11:29:00.000	603959.SH	Factor_XXX	0.21565396898481193
"""

startDay = dt.datetime(2020,1,1)
endDay = dt.datetime(2025,12,31)
startDay_preHeat = startDay - dt.timedelta(days = 40) 

# connect ddb
s = ddb.session()
s.connect(**DATA_DB_CONN)
# loadTable
t_EOD = s.loadTable(dbPath='dfs://WIND.ASHAREEODPRICES',tableName='data')
t_EOD = t_EOD.where(f"TRADE_DT>= {startDay_preHeat.strftime('%Y.%m.%d')} and TRADE_DT <= {endDay.strftime('%Y.%m.%d')} ")
# ret matrix
df_ret = t_EOD.select("TRADE_DT as Date, S_INFO_WINDCODE as WindCode, (S_DQ_CLOSE/S_DQ_PRECLOSE-1) as Ret").executeAs('df_ret')
df_ret = df_ret.select('Ret').pivotby('Date', 'WindCode').toDF()
df_ret = df_ret.set_index('Date')
selected_col = [x for x in df_ret.columns if x[0] in ('6','0','3')]
df_ret = df_ret[selected_col]

factor = df_ret.rolling(5).mean()
factor_value = factor.loc[startDay:endDay,:]

# 做一下时间对齐
factor_value = factor_value.shift()
factor_value = factor_value.dropna(how = 'all',axis = 0)

# 转化成窄表
factor_value = factor_value.stack()
factor_value = factor_value.reset_index()
factor_value['factorname'] = 'Factor_example'
# 保持日期格式为datetime，但补全时间部分为09:59:00
factor_value['Date'] = pd.to_datetime(factor_value['Date']) + pd.Timedelta(hours=9, minutes=59)

factor_value.columns = ['tradetime','symbol','value','factorname']
factor_value = factor_value[['tradetime','symbol','factorname','value']]



import numpy as np

# 目标时间点
add_times = [
    (10, 29),
    (10, 59),
    (11, 29),
    (13, 29),
    (13, 59),
    (14, 29)
]
augmented_df_list = [factor_value]

for hour, minute in add_times:
    df_mod = factor_value.copy()
    # 设置目标时间（保持日期不变，时间换为hour:minute）
    df_mod["tradetime"] = pd.to_datetime(df_mod["tradetime"]).dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    # 给 value 列加上一个随机扰动
    df_mod["value"] = df_mod["value"] + np.random.normal(0, 0.01, size=len(df_mod))
    augmented_df_list.append(df_mod)

# 拼接所有时间点的数据
factor_value = pd.concat(augmented_df_list, ignore_index=True)

# 上传到DDB
s.upload({'Factor_example':factor_value})


#%%
def backtest_factor_name(factor_name,index_code='000852.SH'):
    print(factor_name,'回测中')
    ddb_read_factor_data = f'''

    signal =  {factor_name}
    index_code = "{index_code}"
    // 过滤掉指数外的数据
    signal = select *, date(tradetime) as Date from signal
    signal = filter_in_index(signal, index_code )

    group_data_ret, summary = get_cs_group_performance(signal,PREHEAT_RET_MATRIX_ZZ1000,group_num = 5) // 分5组
    '''
    s.run(ddb_read_factor_data)

    signal_data = s.run('signal')
    intraday_lib.get_signal_count(signal_data, save_path=f'result/{factor_name}' )

    group_data_ret = s.run('group_data_ret')
    group_data_ret = intraday_lib.subtract_market_return(group_data_ret)
    print(group_data_ret.head())

      #  执行分析
    print("开始分析group表现...")
    print("图表将保存为PNG文件到当前目录，并尝试在图形界面显示")
    print("="*50)

    # 执行分析 - 保存图片并尝试显示
    performance_results = intraday_lib.analyze_group_performance_by_bartime(
        group_data_ret, 
        ret_columns=['Ret_15', 'Ret_30', 'Ret_60', 'Ret_90','Ret_120', 'Ret_180','Ret_EOD','Ret_NDay'],
        save_plots=False,    # 保存图表到文件
        show_plots=True  ,   # 尝试在图形界面显示
        save_path= f'result/{factor_name}'
    )

    #  生成HML Sharpe热力图
    print("\n" + "="*50)
    print("生成 group_HML 的 Annualized Ret 比率热力图")
    print("="*50)

    # 创建高对比度热力图，数字清晰可见
    heatmap_data = intraday_lib.create_group_heatmap(
        performance_results,
        group_name = 'group_HML',
        key_name = 'annualized_return',
        save_plot=True,
        show_plot=True,
        high_contrast=False , # 启用高对比度模式
        save_path=f'result/{factor_name}'
    )

    print("\n" + "="*50)
    print("生成 group_HML 的 Sharpe 比率热力图")
    print("="*50)
    heatmap_data = intraday_lib.create_group_heatmap(
        performance_results,
        group_name = 'group_HML',
        key_name = 'sharpe',
        save_plot=True,
        show_plot=True,
        high_contrast=True , # 启用高对比度模式
        save_path=f'result/{factor_name}'
    )

    # 保存结果
    summary_df = intraday_lib.save_performance_summary(performance_results,filename=f"result/{factor_name}/group_performance_summary.csv")


#%% 测试

factor_name = 'Factor_example'
backtest_factor_name(factor_name)

# %%
"""
TASK TO DO:
留给大家后续的开发
1. 日内的，ST,涨跌停怎么处理？
2. 分钟级的交易模式还有哪一些会有效？
"""