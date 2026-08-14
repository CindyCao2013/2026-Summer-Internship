#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Industry Query Demo
# 基于 Wind 数据库，查询指定股票的一级到 N 级行业信息
#
# Wind 的 AShareIndustriesCode.INDUSTRIESCODE 是 16 位定长字符串, 每 2 位代表
# 一个层级, 不足部分用 '0' 填充。不同体系通过开头 2 个字符区分, 示例:
#   SEC    (证监会)    : 0400000000000000 -> 0403a4a320000000   (1~5 级)
#   CITICS (中信)      : b100000000000000 -> b106010500000000   (1~4 级)
#   SW     (申万 2021) : 7600000000000000 -> 7602060300000000   (1~4 级)
#   WIND   (万得 GICS) : 6200000000000000 -> 6215101030000000   (1~5 级)
# 每下一级 = 上一级前缀 + 再加 2 位, 其余补 0。
# 查询时通过 SUBSTR 逐层截断 + 补零, 与字典表 AShareIndustriesCode JOIN 即可
# 拿到完整层级链。

#%%
import oracledb
import pandas as pd

from COMMON_CONST import DATA_DB_WIND


#%% Oracle 初始化（thick mode，需要在进程启动时调用一次）
oracledb.init_oracle_client(lib_dir=None)


#%% 不同行业分类体系对应的表名和字段名
# key = 体系名称; value = (证券-行业归属表, 对应的行业代码字段名)
INDUSTRY_SYSTEMS = {
    "WIND":   ("AShareIndustriesClassCITICS".upper(), "CITICS_IND_CODE"),  # 占位, 见下
    "CITICS": ("AShareIndustriesClassCITICS",         "CITICS_IND_CODE"),  # 中信行业
    "SW":     ("AShareSWNIndustriesClass",            "SW_IND_CODE"),      # 申万行业
    "SEC":    ("AShareSECIndustriesClass",            "SEC_IND_CODE"),     # 证监会行业
    "GICS":   ("AShareIndustriesClassGICS",           "GICS_IND_CODE"),    # GICS
}
# 说明: Wind 自己的"万得行业"习惯用 AShareIndustriesClass + WIND_IND_CODE
INDUSTRY_SYSTEMS["WIND"] = ("AShareIndustriesClass", "WIND_IND_CODE")


#%% 构造 SQL
def build_industry_sql(wind_code: str, system: str = "SEC") -> str:
    """
    构造查询某只证券"一级~末级"行业名称的 SQL。

    参数
    ----
    wind_code : 形如 '301665.SZ'
    system    : 行业体系名，取值见 INDUSTRY_SYSTEMS

    返回
    ----
    str : 可直接 execute 的 Oracle SQL
    """
    table, code_col = INDUSTRY_SYSTEMS[system.upper()]
    return f"""
SELECT
    a.S_INFO_WINDCODE   AS sec_code,
    b.INDUSTRIESNAME    AS ind_name,
    b.LEVELNUM          AS level_num,
    b.INDUSTRIESCODE    AS ind_code
FROM {table} a,
     AShareIndustriesCode b
WHERE
    (
        a.{code_col} = SUBSTR(b.INDUSTRIESCODE, 1, 10)
        OR SUBSTR(a.{code_col}, 1, 8) || '00'       = SUBSTR(b.INDUSTRIESCODE, 1, 10)
        OR SUBSTR(a.{code_col}, 1, 6) || '0000'     = SUBSTR(b.INDUSTRIESCODE, 1, 10)
        OR SUBSTR(a.{code_col}, 1, 4) || '000000'   = SUBSTR(b.INDUSTRIESCODE, 1, 10)
        OR SUBSTR(a.{code_col}, 1, 2) || '00000000' = SUBSTR(b.INDUSTRIESCODE, 1, 10)
    )
    AND a.S_INFO_WINDCODE = '{wind_code}'
    AND a.CUR_SIGN = 1
ORDER BY b.LEVELNUM
""".strip()


#%% 执行查询, 返回 DataFrame
def query_industry_hierarchy(wind_code: str, system: str = "SEC") -> pd.DataFrame:
    sql = build_industry_sql(wind_code, system)
    with oracledb.connect(**DATA_DB_WIND) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [c[0].lower() for c in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "system", system.upper())
    return df


#%% 不同行业体系在 AShareIndustriesCode 中的 INDUSTRIESCODE 前缀
# INDUSTRIESCODE 是 16 位定长字符串, 用开头 2 个字符区分体系 (根据实际数据观察):
#   SEC    04...  证监会行业分类
#   CITICS b1...  中信行业分类
#   SW     76...  申万行业分类 (2021 版)
#   WIND   62...  万得全球行业分类标准 (GICS 对标)
SYSTEM_CODE_PREFIX = {
    "SEC":    "04",
    "CITICS": "b1",
    "SW":     "76",
    "WIND":   "62",
}


#%% 获取所有行业 ind_code -> ind_name 的字典
def get_industry_code_name_map(
    system: str | None = None,
    level: int | None = None,
    code_prefix: str | None = None,
    used_only: bool = False,
    as_dataframe: bool = False,
) -> dict | pd.DataFrame:
    """
    从 AShareIndustriesCode 拉取行业字典, 返回 {ind_code: ind_name}。

    AShareIndustriesCode 把 Wind/中信/申万/证监会 等所有体系的行业代码混放在
    一张表里, 通过 INDUSTRIESCODE 的前 2 位区分体系 (见 SYSTEM_CODE_PREFIX)。
    本函数支持以下过滤条件, 可组合使用:
      - system        : 按体系前缀过滤 (WIND/CITICS/SW/SEC)
      - level         : 按层级过滤  (LEVELNUM = ?)
      - code_prefix   : 自定义 INDUSTRIESCODE 前缀 (优先级高于 system)
      - used_only     : 是否只取 USED=1 的启用行业

    参数
    ----
    system       : 行业体系名, None 表示全量
    level        : 只取某一层级 (1~N); None 表示全部层级
    code_prefix  : 自定义前缀, 指定后会覆盖 system 的默认前缀
    used_only    : True 时在 WHERE 里追加 USED = '1'
    as_dataframe : True 返回 DataFrame (含 level_num, used); 默认返回 dict

    返回
    ----
    dict 或 pd.DataFrame
    """
    where_clauses = []

    if code_prefix is None and system is not None:
        key = system.upper()
        if key not in SYSTEM_CODE_PREFIX:
            raise ValueError(
                f"Unknown industry system: {system}. "
                f"Expected one of {list(SYSTEM_CODE_PREFIX)}"
            )
        code_prefix = SYSTEM_CODE_PREFIX[key]

    if code_prefix is not None:
        where_clauses.append(f"INDUSTRIESCODE LIKE '{code_prefix}%'")
    if level is not None:
        where_clauses.append(f"LEVELNUM = {int(level)}")
    if used_only:
        where_clauses.append("USED = '1'")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
SELECT
    INDUSTRIESCODE AS ind_code,
    INDUSTRIESNAME AS ind_name,
    LEVELNUM       AS level_num,
    USED           AS used
FROM AShareIndustriesCode
{where_sql}
ORDER BY INDUSTRIESCODE
""".strip()

    with oracledb.connect(**DATA_DB_WIND) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [c[0].lower() for c in cur.description]
            rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    if as_dataframe:
        return df
    # 同一个 ind_code 理论上唯一, 用 dict 即可
    return dict(zip(df["ind_code"], df["ind_name"]))


#%% 演示: 查询单只股票在多个行业体系下的分层行业信息
if __name__ == "__main__":
    target_code = "301665.SZ"          # 示例股票, 可按需替换
    systems = ["SEC", "CITICS", "SW", "WIND"]   # 要查询的行业体系

    ind_map_all = {}

    all_df = []
    for sys_name in systems:
        print(f"\n{('-' + sys_name + '-').center(80, '*')}")
        try:
            df = query_industry_hierarchy(target_code, sys_name)
        except Exception as e:
            print(f"[{sys_name}] 查询失败: {e}")
            continue

        if df.empty:
            print(f"[{sys_name}] 未查到 {target_code} 的行业信息")
            continue

        print(df.to_string(index=False))
        all_df.append(df)

    if all_df:
        result = pd.concat(all_df, ignore_index=True)
        print("\n" + "=" * 80)
        print(f"{target_code} 全部行业体系层级汇总:")
        print(result.to_string(index=False))

    # 演示1: 拉取全量行业字典
    print("\n" + "=" * 80)
    ind_map = get_industry_code_name_map()
    print(f"[ALL] AShareIndustriesCode 总条目数: {len(ind_map)}")
    for i, (code, name) in enumerate(ind_map.items()):
        if i >= 5:
            break
        print(f"  {code}  ->  {name}")

    # 演示2: 按体系分别拉取 (只看启用中的)
    for sys_name in systems:
        try:
            sys_map = get_industry_code_name_map(system=sys_name, used_only=True)
        except Exception as e:
            print(f"\n[{sys_name}] 查询失败: {e}")
            continue
        ind_map_all[sys_name] = sys_map
        print(f"\n[{sys_name}] 启用行业代码数: {len(sys_map)}  (前 5 条示例)")
        for i, (code, name) in enumerate(sys_map.items()):
            if i >= 5:
                break
            print(f"  {code}  ->  {name}")

    # 演示3: 只要某一体系的一级行业 (常用于板块热力图)
    try:
        citics_lvl1 = get_industry_code_name_map(
            system="CITICS", level=1, used_only=True
        )
        print(f"\n[CITICS 一级行业] 共 {len(citics_lvl1)} 个:")
        for code, name in citics_lvl1.items():
            print(f"  {code}  ->  {name}")
    except Exception as e:
        print(f"\n[CITICS 一级行业] 查询失败: {e}")

# %%
