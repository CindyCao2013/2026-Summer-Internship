# COMMON_CONST
# 日常需要引用的常量
# 数据库密码只从 .env 读取，不要把真实口令写进这个文件。

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env')
except ImportError:
    pass

# # 存储的数据库
# SAVE_DB = "dfs://Factor_DB_DEV"

# # 存储的表
# SAVE_TABLE = "Factor_Table_Junhao"

# # 存储原始因子表
# SAVE_ORI_TABLE = "Original_Factor_Table"


# # 存储的数据连接 从本地数据库
SAVE_CONN = {}

# 数据源的数据库连接

DATA_DB_CONN = {
    'host': os.getenv('DDB_HOST', '10.12.180.9'),
    'port': int(os.getenv('DDB_PORT', '8902')),
    'userid': os.getenv('DDB_USER', 'pyread'),
    'password': os.getenv('DDB_PASSWORD', ''),
}


#%% 中信的源库连接
_username = os.getenv('DATA_DB_USER', '')
_password = os.getenv('DATA_DB_PASSWORD', '')

"""Oracle 数据库"""
# wind
DATA_DB_WIND = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.153.15:21010/wind'
}

# 聚源, 有两个ip
DATA_DB_JUYUAN = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.8:21010/juyuan'
}
DATA_DB_JUYUAN_2 = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.9:21010/juyuan'
}

# 朝阳永续, 有两个ip
DATA_DB_ZYYX2 = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.89:1521/zyyx2'
}
DATA_DB_ORCL = {
    'user': _username,
    'password': _password,
    'dsn': '172.22.133.52:1521/ORCL'
}


# 财汇
DATA_DB_CAIHUI = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.86:21001/FINCHINA2'
}
DATA_DB_CAIHUI_2 = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.153.20:21001/finchina'
}

# 排排网
DATA_DB_PAIPAI = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.42:21010/cmdssm'
}
DATA_DB_PAIPAI_2 = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.44:21010/cmdssm'
}


# 普益数据
DATA_DB_PUYI = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.232:1521/puyi'
}
DATA_DB_PUYI_2 = {
    'user': _username,
    'password': _password,
    'dsn': '10.23.129.233:1521/puyi'
}

""" MySQL 数据库"""
# 通联, datayes
DATA_DB_DATAYES = {
    'host': '10.80.139.20',
    'port': 3306,
    'user': _username,
    'password': _password,
    'database': 'datayes'
}

# 野尘数据
DATA_DB_YECHEN = {
    'host': '10.63.95.30',
    'port': 3306,
    'user': _username,
    'password': _password,
    'database': 'yechen'
}
DATA_DB_YECHEN_2 = {
    'host': '10.63.95.31',
    'port': 3306,
    'user': _username,
    'password': _password,
    'database': 'yechen'
}

# 东方财富
DATA_DB_EMDATA = {
    'host': '10.80.139.50',
    'port': 6030,
    'user': _username,
    'password': _password,
    'database': 'emdata'
}
DATA_DB_EMDATA_2 = {
    'host': '10.23.130.218',
    'port': 6030,
    'user': _username,
    'password': _password,
    'database': 'emdata'
}

""" ClickHouse 数据库"""
# 高频数据库
DATA_DB_HFDATA = {
    'host': '10.80.139.9',
    'username': _username,
    'password': _password,
    'port': 8123,
    'database': 'cmds'
}
