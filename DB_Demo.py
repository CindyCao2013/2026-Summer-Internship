#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# DataBase Demo
# 本文件主要是用于展示各种数据库的链接方式
# 目前中信的源库连接方式如下：Oracle、MySQL、ClickHouse


#%%
import oracledb
import pymysql
import clickhouse_connect

#%% Oracle: 初始化
"""
需要注意的是，有一些数据库不支持oracle 的thin mode, 只能使用thick mode。

一旦在一个python进程了使用了thin mode，就不能再使用thick mode。


所以，如果需要使用thick mode，需要先初始化oracle client。

"""
oracledb.init_oracle_client(lib_dir=None)

#%% Oracle: Wind
from COMMON_CONST import DATA_DB_WIND
with oracledb.connect(**DATA_DB_WIND) as connection:
    with connection.cursor() as cursor:
        sql = "select * from wind.windcustomcode where rownum < 10"
        for i in cursor.execute(sql):
            print(i)

print("-WIND-".center(80, "*"))


#%% Oracle: 聚源
from COMMON_CONST import DATA_DB_JUYUAN
with oracledb.connect(**DATA_DB_JUYUAN) as connection:
    with connection.cursor() as cursor:
        sql = "select * from jydb.secumain where rownum < 10"
        for i in cursor.execute(sql):
            print(i)
print("-jydb-".center(80, "*"))

# %% Oracle:朝阳永续
from COMMON_CONST import DATA_DB_ZYYX2
with oracledb.connect(**DATA_DB_ZYYX2) as connection:
    with connection.cursor() as cursor:
        sql = "select * from zyyq.con_forecast_stk where rownum < 10"
        for i in cursor.execute(sql):
            print(i)
print("-zyyq-".center(80, "*"))

# %% Oracle: 财汇
from COMMON_CONST import DATA_DB_CAIHUI
with oracledb.connect(**DATA_DB_CAIHUI) as connection:
    with connection.cursor() as cursor:
        sql = "select * from finchina.TQ_OA_STCODE where rownum < 10"
        for i in cursor.execute(sql):
            print(i)
print("-caihui-".center(80, "*"))

# %% Oracle: 排排网
from COMMON_CONST import DATA_DB_PAIPAI
with oracledb.connect(**DATA_DB_PAIPAI) as connection:
    with connection.cursor() as cursor:
        sql = "select * from java.pvn_fund_info where rownum < 10"
        for i in cursor.execute(sql):
            print(i)
print("-paipai-".center(80, "*"))

# %% Oracle: 普益数据
from COMMON_CONST import DATA_DB_PUYI
with oracledb.connect(**DATA_DB_PUYI) as connection:
    with connection.cursor() as cursor:
        sql = "select * from pystandard.bank_base_info where rownum < 10"
        for i in cursor.execute(sql):
            print(i)
print("-puyi-".center(80, "*"))

# %% MySQL: 通联, datayes
from COMMON_CONST import DATA_DB_DATAYES
with pymysql.connect(**DATA_DB_DATAYES) as connection:
    with connection.cursor() as cursor:
        sql = "select * from datayes.con_index limit 10"
        cursor.execute(sql)
        rows = cursor.fetchall()
        for i in rows:
            print(i)

print("-datayes-".center(80, "*"))
#%% MySQL: 野尘数据
from COMMON_CONST import DATA_DB_YECHEN
with pymysql.connect(**DATA_DB_YECHEN) as connection:
    with connection.cursor() as cursor:
        sql = "select * from yechen.industry_chain limit 10"
        cursor.execute(sql)
        rows = cursor.fetchall()
        for i in rows:
            print(i)
print("-yechen-".center(80, "*"))

# %% MySQL: 东方财富
from COMMON_CONST import DATA_DB_EMDATA
with pymysql.connect(**DATA_DB_EMDATA) as connection:
    with connection.cursor() as cursor:
        sql = "select * from emdata.fund_bs_cfinfo limit 10"
        cursor.execute(sql)
        rows = cursor.fetchall()
        for i in rows:
            print(i)
print("-emdata-".center(80, "*"))

# %% ClickHouse: 高频数据库
from COMMON_CONST import DATA_DB_HFDATA
with clickhouse_connect.get_client(**DATA_DB_HFDATA) as client:
    sql = "select * from CFFEX_AL_KLIN_RTH limit 10"
    result = client.query(sql)
    for i in result.result_rows:
        print(i)

print("-hfdata-".center(80, "*"))
# %%
