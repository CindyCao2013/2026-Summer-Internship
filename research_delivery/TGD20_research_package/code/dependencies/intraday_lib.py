#%%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os 

#%% DDB 相关
ddb_functions = '''

// 将北交所股票代码从旧代码映射为新代码
def map_bj_code(t, column_name = 'symbol'){
    /**
    * 功能：将表t中指定列(column_name)的北交所旧代码（北证50使用的老Wind代码）统一映射为新代码。
    * 作用：部分北交所股票曾更换过Wind代码，为保证分析的一致性（如与新版数据或收益率表等对齐），需要将旧代码批量转换为新代码。
    * 参数：
    *    t           —— 输入的数据表
    *    column_name —— 需要映射的列名，默认为"symbol"
    * 返回：返回新的表，其中指定列的北交所旧代码已转换为新代码（其他股票不受影响）。
    */
    // 北证50 新旧代码对应关系
    bj_old_new_code =  select S_INFO_WINDCODE, S_INFO_OLDWINDCODE, S_INFO_NEWWINDCODE, CHANGE_DATE  
    FROM loadTable("dfs://WIND.CHANGEWINDCODE",'data') where S_INFO_WINDCODE like "920%BJ%" 
    order by S_INFO_WINDCODE,CHANGE_DATE

    bj_old_new_code = select S_INFO_OLDWINDCODE as old_code, S_INFO_NEWWINDCODE as new_code from bj_old_new_code

    // 新旧代码合并，左连接：用旧代码匹配新代码，未匹配的保留原样
    new_t = lj(t, bj_old_new_code, column_name, `old_code)

    update!(new_t, `new_code, <_$column_name>, <new_code = NULL>)
    update!(new_t, column_name, <new_code>)
    new_t.dropColumns!(`new_code)
    return new_t
}



// 获取交易日
def get_trading_days(startDay, endDay,exchange = 'SSE'){
    t = loadTable('dfs://WIND.ASHARECALENDAR', 'data')
    trading_days = exec TRADE_DAYS as Date from t where TRADE_DAYS >= startDay and TRADE_DAYS <= endDay and S_INFO_EXCHMARKET = exchange order by TRADE_DAYS
    return trading_days
}



// 获取股票池
def get_stock_pool(startDay, endDay, index, pure_stock = false){
    /*
    输入：
    startDay: 开始日期
    endDay: 结束日期
    index: 指数代码,支持任意指数代码，并且支持index 是一个list
    pure_stock: 是否返回单纯的股票代码

    输出：
    stock_pool: 股票池
    */
    ind = (typestr(index) == "STRING") // 判断是否是一个单独的string

    if (ind){
        index_list = [index]
    }else{
        index_list = index
    }


    if ((index_list[0] == "000300.SH")){
        t = loadTable('dfs://WIND.AINDEXHS300WEIGHT','data')
        stock_pool = select TRADE_DT as Date, S_CON_WINDCODE as Symbol from t where TRADE_DT >= startDay and TRADE_DT<= endDay
    }else if((index_list[0] == "000905.SH")){

        t = loadTable('dfs://WIND.AINDEXCSI500WEIGHT','data')
        stock_pool = select TRADE_DT as Date, S_CON_WINDCODE as Symbol from t where TRADE_DT >= startDay and TRADE_DT<= endDay

    }else if((index_list[0] == "000852.SH")){
        t = loadTable('dfs://WIND.AINDEXCSI1000WEIGHT','data')
        stock_pool = select TRADE_DT as Date, S_CON_WINDCODE as Symbol from t where TRADE_DT >= startDay and TRADE_DT<= endDay

    }else if((index_list[0] == "899050.BJ")){
        // 北证50指数
        // 北证50指数需要特殊处理，因为有成分股变动改名
        // 获取指数成分股
        trading_days = get_trading_days(startDay, endDay)
        t = loadTable('dfs://WIND.AINDEXMEMBERS','data')
        index_members = select S_CON_WINDCODE,S_CON_INDATE,nullFill(temporalParse(S_CON_OUTDATE, "yyyyMMdd"),2100.01.01) AS S_CON_OUTDATE,CUR_SIGN from t where S_INFO_WINDCODE = index_list[0] and S_CON_WINDCODE like "920%BJ" context by S_CON_WINDCODE, S_CON_INDATE  csort OPDATE desc limit 1
        trading_day_tbl = table(trading_days as tradeDate)
        index_member_daily = select S_CON_WINDCODE as Symbol, tradeDate as Date from cj(index_members, trading_day_tbl) where tradeDate >= S_CON_INDATE and tradeDate <= S_CON_OUTDATE
        stock_pool = select * from index_member_daily order by Date,Symbol

    }else{
        // 其他任意指数，需要从WIND.AINDEXMEMBERS 当中进行获得, 支持 index_code 是一个list
        trading_days = get_trading_days(startDay, endDay)
        t = loadTable('dfs://WIND.AINDEXMEMBERS','data')
        index_members = select S_CON_WINDCODE,S_CON_INDATE,nullFill(temporalParse(S_CON_OUTDATE, "yyyyMMdd"),2100.01.01) AS S_CON_OUTDATE,CUR_SIGN from t where S_INFO_WINDCODE in index_list context by S_CON_WINDCODE, S_CON_INDATE  csort OPDATE desc limit 1
        trading_day_tbl = table(trading_days as tradeDate)
        index_member_daily = select S_CON_WINDCODE as Symbol, tradeDate as Date from cj(index_members, trading_day_tbl) where tradeDate >= S_CON_INDATE and tradeDate <= S_CON_OUTDATE
        stock_pool = select * from index_member_daily order by Date,Symbol
    }
    
    if (pure_stock){
        // 返回 单纯的Vector
        output = exec distinct Symbol from stock_pool 
    }else{
        // 返回 日期-股票 相对应的一张表
        output = stock_pool
    }

    return output
}


// 将数据在指数内进行筛选
def filter_in_index(data,index){
    // data 必须 Date, Symbol 两个数据列
    // 将指数外的过滤掉
    startDay = min(data.Date)
    endDay = max(data.Date)
    stock_pool = get_stock_pool(startDay, endDay, index)
    output = select * from data.ej(stock_pool, `Symbol`Date )
    return output 
}


// 获取复权因子
def get_adj_factor(startDay, endDay, stock_vector = NULL){
    // 获取复权因子
    t_EOD = loadTable('dfs://WIND.ASHAREEODPRICES','data')
    if (isVoid(stock_vector)){
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_ADJFACTOR as Adj_Factor from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay
    }else{
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_ADJFACTOR as Adj_Factor from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay and S_INFO_WINDCODE in stock_vector
    }

    return t_adj
}

// ret matrix
def get_ret_matrix(startDay, endDay, stock_vector = NULL, vwap_min_window = 5){
    /*
    基于分钟数据，返回一个标准的ret矩阵，已经考虑到复权，
    性能:获取3年中证1000的数据，大概需要2分钟15秒

    输入：
    startDay: 开始日期
    endDay: 结束日期
    stock_vector: 股票池
    vwap_min_window: VWAP 计算窗口，含义是按照当前Bar 后面多少个分钟Bar 来计算VWAP，作为计算Ret的价格

    输出：
    t_ret: 返回一个标准的ret矩阵
    t_ret 包含的列：
    Symbol: 股票代码
    Date: 日期
    Bartime: 分钟时间
    Ret_15: 未来15分钟的ret
    Ret_30: 未来30分钟的ret
    Ret_60: 未来60分钟的ret
    Ret_90: 未来90分钟的ret
    Ret_120: 未来120分钟的ret
    Ret_180: 未来180分钟的ret
    Ret_EOD: 到当天14:50分后，以接下来5分钟的VWAP价格计算Ret_EOD
    Ret_NDay: 未来1天的ret
    */ 
    

    // 扩展endDay 5天，因为需要5天的数据来计算vwap, 担心节假日的问题
    endDay_extend = endDay + 5
    t_adj = get_adj_factor(startDay, endDay_extend,stock_vector)

    t_minBar = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')

    if (isVoid(stock_vector)){
        t = select Symbol, second(Bartime) as Bartime, Date, msum(Volume,vwap_min_window,minPeriods = 1) as Volume_msum, msum(Amount,vwap_min_window,minPeriods = 1) as Amount_msum from t_minBar where Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }else{
        t = select Symbol, second(Bartime) as Bartime, Date, msum(Volume,vwap_min_window,minPeriods = 1) as Volume_msum, msum(Amount,vwap_min_window,minPeriods = 1) as Amount_msum from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }

    t = lj(t,t_adj,`Symbol`Date)


    // Bartime T+0 触发交易时，默认交易价格是Bartime T+1 ~ T+5 总共5跟Bar 里面VWAP 价格，所以计算出来VWAP之后要上移vwap_min_window个Bar,同时因为复权的原因，也要移动复权数据
    t_ret = select Symbol, Date, Bartime, move(ratio(Amount_msum,Volume_msum)*Adj_Factor,-vwap_min_window) as VWAP_Next from t context by Symbol csort Date, Bartime
    

    // 获取EOD的ret
    t_ret_EOD = select Symbol,Date, VWAP_Next as VWAP_EOD from t_ret where Bartime = 14:50:00
    t_ret = lj(t_ret,t_ret_EOD,`Symbol`Date)
    // 计算未来15分钟的ret，30分钟ret, 1个小时ret, 2个小时的ret，Next Day 的ret
    t_ret = select Symbol, Date,Bartime, move(VWAP_Next,-15)/VWAP_Next-1 as Ret_15,
      move(VWAP_Next,-30)/VWAP_Next-1 as Ret_30, move(VWAP_Next,-60)/VWAP_Next-1 as Ret_60, move(VWAP_Next,-90)/VWAP_Next-1 as Ret_90, 
      move(VWAP_Next,-120)/VWAP_Next-1 as Ret_120, move(VWAP_Next,-150)/VWAP_Next-1 as Ret_150, move(VWAP_Next,-180)/VWAP_Next-1 as Ret_180, 
       VWAP_EOD/VWAP_Next-1 as Ret_EOD,move(VWAP_Next,-239)/VWAP_Next-1 as Ret_NDay from t_ret context by Symbol csort Date, Bartime
    
    // 过滤掉多余结果
    t_ret = select * from t_ret where Date >= startDay and Date <= endDay
    return t_ret
}


def get_ret_matrix_BJ(startDay, endDay,vwap_min_window = 5){
    /* 获取北交所的retMatrix， 由于北交所股票代码会更换，所以需要将旧代码映射为新代码，因此有特殊处理 */
    /*
    输入：
    startDay: 开始日期
    endDay: 结束日期
    vwap_min_window: VWAP 计算窗口，含义是按照当前Bar 后面多少个分钟Bar 来计算VWAP，作为计算Ret的价格

    输出：
    t_ret: 返回一个标准的ret矩阵
    */
    
    // 获取北交所股票代码id
    stock_vector =  EXEC distinct Symbol from loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute') where Symbol like "%BJ%"

    // 扩展endDay 5天，因为需要5天的数据来计算vwap, 担心节假日的问题
    endDay_extend = endDay + 5
    t_adj = get_adj_factor(startDay, endDay_extend,stock_vector)

    t_minBar = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
    t_minBar = select Symbol,Bartime,Date,Volume,Amount from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend

    // 将北交所股票代码从旧代码映射为新代码
    t_minBar = map_bj_code(t_minBar, `Symbol)

    if (isVoid(stock_vector)){
        t = select Symbol, second(Bartime) as Bartime, Date, msum(Volume,vwap_min_window,minPeriods = 1) as Volume_msum, msum(Amount,vwap_min_window,minPeriods = 1) as Amount_msum from t_minBar where Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }else{
        t = select Symbol, second(Bartime) as Bartime, Date, msum(Volume,vwap_min_window,minPeriods = 1) as Volume_msum, msum(Amount,vwap_min_window,minPeriods = 1) as Amount_msum from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }

    t = lj(t,t_adj,`Symbol`Date)

    // Bartime T+0 触发交易时，默认交易价格是Bartime T+1 ~ T+5 总共5跟Bar 里面VWAP 价格，所以计算出来VWAP之后要上移5个Bar,同时因为复权的原因，也要移动复权数据
    t_ret = select Symbol, Date, Bartime, move(ratio(Amount_msum,Volume_msum)*Adj_Factor,-vwap_min_window) as VWAP_Next from t context by Symbol csort Date, Bartime
    
    // 获取EOD的ret
    t_ret_EOD = select Symbol,Date, VWAP_Next as VWAP_EOD from t_ret where Bartime = 14:50:00
    t_ret = lj(t_ret,t_ret_EOD,`Symbol`Date)
    // 计算未来15分钟的ret，30分钟ret, 1个小时ret, 2个小时的ret，Next Day 的ret
    t_ret = select Symbol, Date,Bartime, move(VWAP_Next,-15)/VWAP_Next-1 as Ret_15,
      move(VWAP_Next,-30)/VWAP_Next-1 as Ret_30, move(VWAP_Next,-60)/VWAP_Next-1 as Ret_60, move(VWAP_Next,-90)/VWAP_Next-1 as Ret_90, 
      move(VWAP_Next,-120)/VWAP_Next-1 as Ret_120, move(VWAP_Next,-150)/VWAP_Next-1 as Ret_150, move(VWAP_Next,-180)/VWAP_Next-1 as Ret_180, 
       VWAP_EOD/VWAP_Next-1 as Ret_EOD,move(VWAP_Next,-239)/VWAP_Next-1 as Ret_NDay from t_ret context by Symbol csort Date, Bartime
    
    // 过滤掉多余结果
    t_ret = select * from t_ret where Date >= startDay and Date <= endDay
    return t_ret
}


// 横截面效果评估
def get_cs_group_performance(signal,ret_matrix,group_num = 5){
    /*
    评估横截面信号的效果
    输入：
    signal: 横截面信号, 必须包含symbol, tradetime, value列
    例如：
    symbol    tradetime               value             
    --------- ----------------------- -----
    000001.SZ 2024.01.02T09:30:00.000 33.43
    000001.SZ 2024.01.02T09:45:00.000 25.47
    000001.SZ 2024.01.02T10:00:00.000 26.03
    000001.SZ 2024.01.02T10:15:00.000 19.33
    000001.SZ 2024.01.02T10:30:00.000 22.93

    ret_matrix: 未来ret矩阵
    group_num: 分组数量

    返回：
    在返回数据的同时，会打印出HML的Daily Return, Annualized Sharpe, 以及Rank IC, Rank ICIR
    group_data_ret: 分组平均收益
    summary: dict, 总结信息
        summary 里面包含group_stats, ic_ts, ic_mean
        group_stats: 分组平均收益，标准差，年化Sharpe
        ic_ts: 时间序列，每天的，横截面相关系数IC
        ic_mean: IC平均值，标准差，年化IR
    */
    group_data = select symbol as Symbol, date(tradetime) as Date, second(tradetime) as Bartime, tradetime, value as Signal_Value, "group_"+string(rank(value,groupNum = group_num)) as group from signal context by tradetime
    group_data = ej(group_data,ret_matrix,`Symbol`Date`Bartime)
    
    // 计算分组收益
    ret_col_names = ret_matrix.columnNames()[ret_matrix.columnNames() like "Ret_%"]
    select_code = sqlCol(ret_col_names, func=mean, alias=ret_col_names) 
    group_code = sqlCol(["group","Date","Bartime"])
    meta_sql = sql(select = select_code, from = group_data, groupBy = group_code, groupFlag = 1)
    group_data_ret = meta_sql.eval()

    // 计算H-L组的收益, 转换成窄表计算比较方便
    group_L = group_data_ret[group_data_ret.group == "group_0"].unpivot(keyColNames =['group','Date','Bartime'], valueColNames = ret_col_names)
    group_H = group_data_ret[group_data_ret.group == "group_"+string(group_num-1)].unpivot(keyColNames =['group','Date','Bartime'], valueColNames = ret_col_names)
    group_HL = ej(group_H as 'High', group_L as 'Low', `Date`Bartime`valueType)
    group_HL = select 'group_HML' as group, Date, Bartime, valueType, (value-Low_value) as value from group_HL
    group_HL = select value from group_HL pivot by group, Date, Bartime,valueType
    
    group_HL.reorderColumns!(group_data_ret.columnNames()) // append! 不会自动对齐表头，需要手动调整
    group_data_ret.append!(group_HL) 
    group_data_ret.sortBy!(`Date`Bartime`group)

    // 计算每个分组在各个期限上的平均收益与Sharpe（未年化）
    group_returns_long = group_data_ret.unpivot(keyColNames = ['group','Date','Bartime'], valueColNames = ret_col_names)
    group_stats = select avg(value) as MeanRet, std(value) as StdRet, avg(value)/iif(std(value)==0, NULL, std(value))*sqrt(252) as Annu_Sharpe from group_returns_long group by group, Bartime,valueType as RetType

    // 计算因子IC（横截面相关系数），按每个时间点、每个期限
    ic_long = group_data.unpivot(keyColNames = ['Symbol','Date','Bartime','tradetime','group','Signal_Value'], valueColNames = ret_col_names)
    ic_ts = select spearmanr(Signal_Value, value) as Rank_IC from ic_long group by Date, Bartime, valueType as RetType
    ic_mean = select RetType, avg(Rank_IC) as IC_Mean, std(Rank_IC) as IC_Std, avg(Rank_IC)/iif(std(Rank_IC)==0, NULL, std(Rank_IC)) as IC_IR from ic_ts group by Bartime, RetType

    
    // 不好画图，就打印出来
    print('\n*********** Group HML Daily Return(bps) ***********')
    
    t1 = select decimal32(MeanRet*10000,4) as MeanRet from group_stats where group = 'group_HML' pivot by Bartime,RetType
    correct_col_order = join(['Bartime'],ret_col_names)
    t1.reorderColumns!(correct_col_order)
    print(t1)


    print('\n*********** Group HML Annualized Sharpe ***********')
    t2 = select decimal32(Annu_Sharpe,4) as Annu_Sharpe from group_stats where group = 'group_HML' pivot by Bartime,RetType
    t2.reorderColumns!(correct_col_order)
    print(t2)

    print('\n*********** Rank IC ***********')
    t3 = select decimal32(IC_Mean,4) as IC_Mean from ic_mean  pivot by Bartime,RetType
    t3.reorderColumns!(correct_col_order)
    print(t3)

    print('\n*********** Rank IC IR ***********')
    t4 = select decimal32(IC_IR,4) as IC_IR from ic_mean  pivot by Bartime, RetType
    t4.reorderColumns!(correct_col_order)
    print(t4)

    summary = dict(STRING,ANY)
    summary['group_stats'] = group_stats
    summary['ic_ts'] = ic_ts
    summary['ic_mean'] = ic_mean

    return group_data_ret, summary
}

// 判断分钟是否涨跌停
def get_limit_status(startDay, endDay, stock_vector = NULL, return_status = NULL){
    /*
    输入：
    startDay: 开始日期
    endDay: 结束日期
    stock_vector: 股票池
    return_status: 返回状态，1为返回涨停的情况，-1为返回跌停的情况，0为正常, NULL 返回所有状态
    */
    // 扩展endDay 5天，因为需要5天的数据来计算vwap, 担心节假日的问题


    endDay_extend = endDay + 5

    t_minBar = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')

    if (isVoid(stock_vector)){
        t = select Symbol, second(Bartime) as Bartime, Date, Close from t_minBar where Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }else{
        t = select Symbol, second(Bartime) as Bartime, Date, Close from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }

    t_EOD = loadTable('dfs://WIND.ASHAREEODPRICES','data')
    if (isVoid(stock_vector)){
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_LIMIT as High_Limit, S_DQ_STOPPING as Low_Limit from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay
    }else{
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_LIMIT as High_Limit, S_DQ_STOPPING as Low_Limit from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay and S_INFO_WINDCODE in stock_vector
    }

    t = lj(t,t_adj,`Symbol`Date)

    t_output = select Symbol, Date, Bartime, iif(abs(Close - High_Limit) < 1e-4, 1, iif(abs(Close - Low_Limit) < 1e-4, -1, 0)) as Limit_Status from t context by Symbol csort Date, Bartime
    
    if (isVoid(return_status)){
        t_output = select * from t_output where Date >= startDay and Date <= endDay
    }else{
        t_output = select * from t_output where Limit_Status = return_status and Date >= startDay and Date <= endDay
    }
    return t_output
}

def get_limit_status_BJ(startDay, endDay, return_status = NULL){
    /*
    输入：
    startDay: 开始日期
    endDay: 结束日期
    return_status: 返回状态，1为返回涨停的情况，-1为返回跌停的情况，0为正常, NULL 返回所有状态
    */
    // 扩展endDay 5天，因为需要5天的数据来计算vwap, 担心节假日的问题
    // 获取北交所股票代码id
    stock_vector =  EXEC distinct Symbol from loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute') where Symbol like "%BJ%"

    // 扩展endDay 5天，因为需要5天的数据来计算vwap, 担心节假日的问题
    endDay_extend = endDay + 5
    t_adj = get_adj_factor(startDay, endDay_extend,stock_vector)

    t_minBar = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
    t_minBar = select Symbol,Bartime,Date,Close from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend

    // 将北交所股票代码从旧代码映射为新代码
    t_minBar = map_bj_code(t_minBar, `Symbol)

    if (isVoid(stock_vector)){
        t = select Symbol, second(Bartime) as Bartime, Date, Close from t_minBar where Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }else{
        t = select Symbol, second(Bartime) as Bartime, Date, Close from t_minBar where Symbol in stock_vector and Date >= startDay and Date <= endDay_extend context by Symbol csort Date, Bartime
    }

    t_EOD = loadTable('dfs://WIND.ASHAREEODPRICES','data')
    if (isVoid(stock_vector)){
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_LIMIT as High_Limit, S_DQ_STOPPING as Low_Limit from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay
    }else{
        t_adj = select TRADE_DT as Date, S_INFO_WINDCODE as Symbol, S_DQ_LIMIT as High_Limit, S_DQ_STOPPING as Low_Limit from t_EOD where TRADE_DT >= startDay and TRADE_DT<= endDay and S_INFO_WINDCODE in stock_vector
    }

    t = lj(t,t_adj,`Symbol`Date)

    t_output = select Symbol, Date, Bartime, iif(abs(Close - High_Limit) < 1e-4, 1, iif(abs(Close - Low_Limit) < 1e-4, -1, 0)) as Limit_Status from t context by Symbol csort Date, Bartime
    
    if (isVoid(return_status)){
        t_output = select * from t_output where Date >= startDay and Date <= endDay
    }else{
        t_output = select * from t_output where Limit_Status = return_status and Date >= startDay and Date <= endDay
    }    
    return t_output   
}


'''



#%% 画图相关
# 设置中文字体支持
def setup_chinese_font(verbose=False):
    """设置matplotlib中文字体支持"""
    try:
        import matplotlib.font_manager as fm
        
        # 尝试不同的中文字体
        chinese_fonts = [
            'SimHei',           # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows) 
            'PingFang SC',      # 苹方 (macOS)
            'Hiragino Sans GB', # 冬青黑体 (macOS)
            'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
            'Noto Sans CJK SC', # 思源黑体 (Linux)
            'DejaVu Sans',      # 备用字体
            'Arial Unicode MS', # 备用字体
        ]
        
        # 检查可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 找到第一个可用的中文字体
        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
            if verbose:
                print(f"使用字体: {selected_font}")
        else:
            plt.rcParams['font.sans-serif'] = chinese_fonts
            if verbose:
                print("使用默认字体列表，可能无法正确显示中文")
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
    except Exception as e:
        if verbose:
            print(f"字体设置出现错误: {e}")
        # 使用基本设置作为备用
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
setup_chinese_font()

#%%
def calculate_performance_metrics(returns):
    """
    计算收益率序列的性能指标
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {'sharpe': 0, 'max_drawdown': 0, 'annualized_return': 0}
    
    # 计算累积收益
    cumulative_returns = returns.cumsum()
    
    # 计算年化收益率
    n_periods = len(returns)
    annualized_return = np.mean(returns) * 252

    
    # 计算最大回撤
    rolling_max = (1+cumulative_returns).expanding().max()
    drawdown = (1 + cumulative_returns) / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # 计算夏普比率 (假设无风险利率为0，年化因子为√252)
    if returns.std() == 0:
        sharpe = 0
    else:
        annualization_factor = np.sqrt(252)
        sharpe = returns.mean() / returns.std() * annualization_factor
    
    return {
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return
    }

def analyze_group_performance_by_bartime(group_data_ret, ret_columns=['Ret_15', 'Ret_30', 'Ret_60', 'Ret_90','Ret_120', 'Ret_180','Ret_NDay'], save_plots=True, show_plots=False,save_path='result'):
    """
    按Bartime分类分析每个group的表现
    
    Parameters:
    -----------
    group_data_ret : DataFrame
        包含分组数据和收益率的DataFrame
    ret_columns : list
        要分析的收益率列名列表
    save_plots : bool, default True
        是否保存图表到文件
    show_plots : bool, default False  
        是否尝试显示图表（需要图形界面环境）
    save_path : str, default None
        保存图表的位置
    Returns:
    --------
    dict : 包含所有性能指标的嵌套字典
    """
    
    # 如果需要显示图片，尝试设置交互式后端
    original_backend = matplotlib.get_backend()

    # 获取数据
    data = group_data_ret.copy()
    
    # 转换时间格式
    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Bartime'].astype(str))
    
    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
            # 手动设置一些美观的样式
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['axes.linewidth'] = 0.8
            plt.rcParams['axes.axisbelow'] = True
            plt.rcParams['grid.linestyle'] = '--'
            plt.rcParams['grid.alpha'] = 0.7
    
    # 确保中文字体支持（重新设置以防样式覆盖）
    setup_chinese_font()
    
    # 获取所有的group
    groups = sorted([g for g in data['group'].unique() if g.startswith('group_')])
    
    # 获取所有的Bartime
    bartimes = sorted(data['Bartime'].unique())
    bartimes = [pd.to_datetime(bartime) for bartime in bartimes]
    results_summary = {}
    
    # 为每个收益率时间窗口创建分析
    for ret_col in ret_columns:
        print(f"\n{'='*50}")
        print(f"分析收益率窗口: {ret_col}")
        print(f"{'='*50}")
        
        results_summary[ret_col] = {}
        
        # 为每个Bartime创建一个子图
        fig, axes = plt.subplots(len(bartimes), 1, figsize=(15, 4*len(bartimes)))
        if len(bartimes) == 1:
            axes = [axes]
            
        fig.suptitle(f'各组在不同Bartime的累积收益曲线 - {ret_col}', fontsize=16, y=0.98)
        
        for idx, bartime in enumerate(bartimes):
            ax = axes[idx]
            
            # 筛选当前Bartime的数据
            bartime_data = data[data['Bartime'] == bartime].copy()
            bartime_data = bartime_data.sort_values(['group', 'Date'])
            
            results_summary[ret_col][bartime] = {}
            
            # 为每个group绘制累积收益曲线
            for group in groups:
                group_data = bartime_data[bartime_data['group'] == group].copy()
                
                if len(group_data) == 0:
                    continue
                    
                # 计算累积收益
                group_data = group_data.sort_values('Date')
                returns = group_data[ret_col].fillna(0)
                cumulative_returns = returns.cumsum()
                
                # 计算性能指标
                metrics = calculate_performance_metrics(returns)
                results_summary[ret_col][bartime][group] = metrics
                
                # 绘制累积收益曲线
                ax.plot(group_data['Date'], cumulative_returns, 
                       label=f'{group} (Sharpe: {metrics["sharpe"]:.2f}, MDD: {metrics["max_drawdown"]:.2%})', 
                       linewidth=2)
            
            ax.set_title(f'Bartime: {bartime.strftime("%H:%M")}', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Returns')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 格式化x轴日期
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
        
        # 根据参数决定是否保存和显示图表
        if save_plots:
            # 自动创建目录（关键修改）
            os.makedirs(save_path, exist_ok=True)
                
            filename = f'{save_path}/group_performance_{ret_col.lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ 图表已保存到: {filename}")
            
        if show_plots:
            # 如果在有图形界面的环境下，尝试显示图片

            try:
                plt.show()
                print("📊 图表已在新窗口中显示")
            except Exception as e:
                print(f"注意: 当前环境无法显示图片 ({e})，图片已保存到文件")
                plt.close()
        
        # 总是关闭图形以释放内存
        plt.close()
        
        # 打印性能指标汇总表
        print(f"\n{ret_col} - 性能指标汇总:")
        print("-" * 80)
        
        for bartime in bartimes:
            print(f"\nBartime: {bartime.strftime('%H:%M')}")
            print(f"{'Group':<12} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Annualized Return':<18}")
            print("-" * 65)
            
            if bartime in results_summary[ret_col]:
                for group in groups:
                    if group in results_summary[ret_col][bartime]:
                        metrics = results_summary[ret_col][bartime][group]
                        print(f"{group:<12} {metrics['sharpe']:<15.3f} {metrics['max_drawdown']:<15.2%} {metrics['annualized_return']:<18.2%}")
    
    return results_summary

def create_group_heatmap(performance_results,group_name = 'group_HML',key_name = 'sharpe', save_plot=True, show_plot=True, high_contrast=True,save_path='result'):
    """
    创建group_HML的Sharpe比率热力图
    
    Parameters:
    -----------
    performance_results : dict
        analyze_group_performance_by_bartime的返回结果
    save_plot : bool, default True
        是否保存图表
    show_plot : bool, default True
        是否显示图表
    high_contrast : bool, default True
        是否使用高对比度模式，提高数字可读性
    save_path : str, default None
        保存图表的位置
    Returns:
    --------
    pandas.DataFrame : 热力图数据矩阵
    """
    
    # 提取HML的Sharpe数据
    hml_data = {}
    
    for ret_col, bartime_data in performance_results.items():
        hml_data[ret_col] = {}
        for bartime, group_data in bartime_data.items():
            if group_name in group_data:
                sharpe_value = group_data[group_name][key_name]
                # 格式化Bartime为字符串
                if isinstance(bartime, pd.Timestamp):
                    bartime_str = bartime.strftime('%H:%M')
                else:
                    bartime_str = str(bartime)
                hml_data[ret_col][bartime_str] = sharpe_value
            else:
                # 如果没有HML数据，使用NaN
                if isinstance(bartime, pd.Timestamp):
                    bartime_str = bartime.strftime('%H:%M')
                else:
                    bartime_str = str(bartime)
                hml_data[ret_col][bartime_str] = np.nan
    
    # 转换为DataFrame
    heatmap_df = pd.DataFrame(hml_data).T
    
    # 确保数据类型为数值
    heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce')
    
    # 按照收益率窗口排序（从小到大）
    ret_order = ['Ret_15', 'Ret_30', 'Ret_60', 'Ret_90','Ret_120', 'Ret_150','Ret_180',
                     'Ret_210', 'Ret_240','Ret_270', 'Ret_300','Ret_EOD','Ret_NDay']
    existing_rets = [col for col in ret_order if col in heatmap_df.index]
    other_rets = [col for col in heatmap_df.index if col not in ret_order]
    final_order = existing_rets + sorted(other_rets)
    heatmap_df = heatmap_df.reindex(final_order)
    
    # 按时间排序列
    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)
    
    print(f"\n{group_name} {key_name}比率热力图数据:")
    print(heatmap_df)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 确保中文字体支持
    setup_chinese_font()
    
    # 创建热力图
    import matplotlib.colors as colors
    import matplotlib.patheffects as path_effects
    
    # 根据high_contrast参数选择颜色映射
    if high_contrast:
        # 高对比度模式：使用红-蓝反转，便于数字显示
        cmap = plt.cm.RdBu_r  # 红-蓝反转，去掉中间的黄色，提高对比度
        fontsize = 20  # 增大字体
        stroke_width = 4  # 增加描边厚度
    else:
        # 标准模式：使用红-黄-蓝反转
        cmap = plt.cm.RdYlBu_r
        fontsize = 20
        stroke_width = 2
    
    # 创建热力图
    im = plt.imshow(heatmap_df.values, cmap=cmap, aspect='auto')
    
    # 设置坐标轴
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45)
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    
    plt.xlabel('Bartime', fontsize=12)
    plt.ylabel('收益率窗口', fontsize=12)
    plt.title(f'group_HML的{key_name}比率热力图\n(红色=高{key_name}, 蓝色=低{key_name})', fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label(f'{key_name} ', fontsize=12)
    
    # 在每个格子中显示数值
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            value = heatmap_df.iloc[i, j]
            if not np.isnan(value):
                # 获取当前格子的颜色值，用于确定最佳文字颜色
                # 获取归一化后的颜色值 (0-1)
                norm_value = (value - heatmap_df.values[~np.isnan(heatmap_df.values)].min()) / (heatmap_df.values[~np.isnan(heatmap_df.values)].max() - heatmap_df.values[~np.isnan(heatmap_df.values)].min())
                
                # 根据颜色映射和数值选择最佳文字颜色
                if high_contrast:
                    # 高对比度模式：RdBu_r颜色映射
                    if 0.3 <= norm_value <= 0.7:  # 中等值范围，背景偏浅色
                        text_color = 'black'
                        edge_color = 'white'
                    else:  # 极高值(深红)或极低值(深蓝)
                        text_color = 'white' 
                        edge_color = 'black'
                else:
                    # 标准模式：RdYlBu_r颜色映射
                    if 0.2 <= norm_value <= 0.8:  # 黄色区域
                        text_color = 'black'
                        edge_color = 'white'
                    else:  # 红色或蓝色区域
                        text_color = 'white'
                        edge_color = 'black'
                
                # 添加文字描边效果，提高可读性
                plt.text(j, i, f'{value:.2f}', 
                        ha='center', va='center', 
                        color=text_color, fontsize=fontsize, weight='bold',
                        path_effects=[path_effects.withStroke(linewidth=stroke_width, foreground=edge_color)])
            else:
                na_fontsize = fontsize - 3 if high_contrast else 12
                plt.text(j, i, 'N/A', 
                        ha='center', va='center', 
                        color='gray', fontsize=na_fontsize, weight='bold')
    
    plt.tight_layout()
    
    # 根据参数决定是否保存和显示
    if save_plot:
        # 自动创建目录（关键修改）
        os.makedirs(save_path, exist_ok=True)
        
        filename = f'{save_path}/{group_name}_{key_name}_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 热力图已保存到: {filename}")
        
    if show_plot:
        try:
            plt.show()
            print("📊 热力图已在新窗口中显示")
        except Exception as e:
            print(f"注意: 当前环境无法显示图片 ({e})，图片已保存到文件")
            plt.close()
    
    plt.close()
    
    return heatmap_df


def save_performance_summary(performance_results, filename="result/group_performance_summary.csv"):
    """
    将性能结果保存为CSV文件
    """
    summary_data = []
    
    for ret_col, bartime_data in performance_results.items():
        for bartime, group_data in bartime_data.items():
            for group, metrics in group_data.items():
                summary_data.append({
                    'Return_Window': ret_col,
                    'Bartime': bartime,
                    'Group': group,
                    'Sharpe_Ratio': metrics['sharpe'],
                    'Max_Drawdown': metrics['max_drawdown'],
                    'Annualized_Return': metrics['annualized_return']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)
    print(f"性能汇总已保存到: {filename}")
    return summary_df


def get_signal_count(signal_data, save_path):
    # 按tradetime分组并统计symbol数量
    count_series = signal_data.groupby('tradetime').count()['symbol']
    
    # 绘图
    count_series.plot(
        title='Symbol Count by Tradetime',
        xlabel='Trade Time',
        ylabel='Count'
    )
    
    # 获取当前绘图区域
    ax = plt.gca()
    
    # 设置背景为白色
    ax.set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # 显示横轴（底部）和纵轴（左侧）的坐标轴线
    # 底部轴线（横轴）
    ax.spines['bottom'].set_visible(True)  # 确保可见
    ax.spines['bottom'].set_color('black')  # 轴线颜色
    ax.spines['bottom'].set_linewidth(1)    # 轴线宽度
    
    # 左侧轴线（纵轴）
    ax.spines['left'].set_visible(True)    # 确保可见
    ax.spines['left'].set_color('black')   # 轴线颜色
    ax.spines['left'].set_linewidth(1)     # 轴线宽度
    
    # 可选：隐藏顶部和右侧的轴线（通常不需要显示）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图片
    plt.savefig(
        f'{save_path}/symbol_count_by_tradetime.png',
        dpi=300,
        bbox_inches='tight'
    )
    
    plt.show()
    plt.close()