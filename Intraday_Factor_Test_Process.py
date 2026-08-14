# 分钟级因子测试入口（Phase 1 示例 + Phase 2 真分钟 Alpha）
#%% modules
import datetime as dt
from pathlib import Path

import dolphindb as ddb
import dolphindb.settings as keys
import pandas as pd

import factor_config as cfg
import intraday_lib
from COMMON_CONST import DATA_DB_CONN
from intraday_formulas import (
    INTRADAY_FACTOR_LIST,
    INTRADAY_PHASE1_LIST,
    INTRADAY_PHASE2_LIST,
    build_intraday_narrow_table,
    filter_available_intraday_factors,
)

#%% config
start_day = cfg.START_DAY
end_day = cfg.END_DAY
start_preheat = start_day - dt.timedelta(days=40)
# Prefer INTRADAY_CUSTOM_FACTOR_LIST; fall back to CUSTOM only if它是日内因子子集
_default_intraday = list(INTRADAY_PHASE1_LIST) + list(INTRADAY_PHASE2_LIST)
_intraday_custom = getattr(cfg, "INTRADAY_CUSTOM_FACTOR_LIST", None)
if _intraday_custom is not None:
    factor_list = list(_intraday_custom)
elif cfg.CUSTOM_FACTOR_LIST is not None and set(cfg.CUSTOM_FACTOR_LIST).issubset(
    set(INTRADAY_FACTOR_LIST)
):
    factor_list = list(cfg.CUSTOM_FACTOR_LIST)
else:
    factor_list = _default_intraday
result_root = Path(cfg.result_root_for("intraday"))
manifest_path = cfg.manifest_path_for("intraday")
index_code = "000852.SH"
share_name = "PREHEAT_RET_MATRIX_ZZ1000"
apply_limit_filter = bool(getattr(cfg, "INTRADAY_APPLY_LIMIT_FILTER", False))
minute_eval_start = getattr(cfg, "INTRADAY_MINUTE_EVAL_START", None) or start_day
minute_eval_end = getattr(cfg, "INTRADAY_MINUTE_EVAL_END", None) or end_day

#%% DDB
s = ddb.session(protocol=keys.PROTOCOL_DDB)
s.connect(**DATA_DB_CONN)
s.run(intraday_lib.ddb_functions)

# 检查预热收益矩阵
try:
    ok = bool(s.run(f'defined("{share_name}", SHARED)'))
    print(f"{share_name}: {'OK' if ok else 'MISSING'}")
except Exception as exc:
    print(f"WARNING: {share_name} missing — run data_preheat.py first ({exc})")
    ok = False

factors_to_run = filter_available_intraday_factors(factor_list)
if not factors_to_run:
    print("No intraday factors ready.")
    raise SystemExit(0)


def build_example_narrow_factor():
    """Phase 1 示例：EOD rolling 因子映射到 09:59 窄表（验证 intraday 链路）。"""
    t_eod = s.loadTable(dbPath="dfs://WIND.ASHAREEODPRICES", tableName="data")
    t_eod = t_eod.where(
        f"TRADE_DT>= {start_preheat.strftime('%Y.%m.%d')} "
        f"and TRADE_DT <= {end_day.strftime('%Y.%m.%d')} "
    )
    df_ret = t_eod.select(
        "TRADE_DT as Date, S_INFO_WINDCODE as WindCode, "
        "(S_DQ_CLOSE/S_DQ_PRECLOSE-1) as Ret"
    ).executeAs("df_ret")
    df_ret = df_ret.select("Ret").pivotby("Date", "WindCode").toDF().set_index("Date")
    selected = [x for x in df_ret.columns if x[0] in ("6", "0", "3")]
    df_ret = df_ret[selected]

    factor = df_ret.rolling(5).mean().shift().loc[start_day:end_day, :]
    factor = factor.dropna(how="all", axis=0)
    narrow = factor.stack().reset_index()
    narrow.columns = ["tradetime", "symbol", "value"]
    narrow["factorname"] = "intraday_example"
    narrow["tradetime"] = pd.to_datetime(narrow["tradetime"]) + pd.Timedelta(
        hours=9, minutes=59
    )
    return narrow[["tradetime", "symbol", "factorname", "value"]]


def backtest_intraday_factor(factor_name: str, narrow_df: pd.DataFrame):
    """调用 intraday_lib 做分组 + 热力图 + CSV 保存。"""
    save_path = str(result_root / factor_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    upload_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in factor_name)
    # DDB upload name must be a valid identifier
    if upload_name[0].isdigit():
        upload_name = "f_" + upload_name
    s.upload({upload_name: narrow_df})

    limit_block = ""
    if apply_limit_filter:
        # 仅对信号中的股票/时刻过滤涨跌停（Limit_Status≠0）
        limit_block = """
    signal = select *, second(tradetime) as Bartime from signal
    syms = exec distinct symbol from signal
    minD = min(exec Date from signal)
    maxD = max(exec Date from signal)
    limit_t = get_limit_status(minD, maxD, syms)
    limit_t = select Symbol as symbol, Date, Bartime, Limit_Status from limit_t
    n_before = signal.rows()
    signal = lj(signal, limit_t, `symbol`Date`Bartime)
    signal = select * from signal where isNull(Limit_Status) or Limit_Status = 0
    print("Limit filter: " + string(n_before) + " -> " + string(signal.rows()))
    // 去掉辅助列，避免 get_cs_group_performance 列冲突
    signal = select tradetime, symbol, factorname, value, Date from signal
    """

    ddb_script = f"""
    signal = {upload_name}
    index_code = "{index_code}"
    signal = select *, date(tradetime) as Date from signal
    signal = filter_in_index(signal, index_code)
    {limit_block}
    group_data_ret, summary = get_cs_group_performance(signal, {share_name}, group_num=10)
    """
    s.run(ddb_script)

    signal_data = s.run("signal")
    intraday_lib.get_signal_count(signal_data, save_path=save_path)

    group_data_ret = s.run("group_data_ret")
    # 扣除截面市场均值，十分组 / 累积曲线基于超额收益（H-L 不减）
    group_data_ret = intraday_lib.subtract_market_return(group_data_ret)
    try:
        ic_mean = s.run("summary['ic_mean']")
    except Exception:  # noqa: BLE001
        ic_mean = None
    performance_results = intraday_lib.analyze_group_performance_by_bartime(
        group_data_ret,
        save_plots=True,
        show_plots=False,
        save_path=save_path,
        factor_name=factor_name,
        ic_mean=ic_mean,
    )
    intraday_lib.create_group_heatmap(
        performance_results,
        group_name="group_HML",
        key_name="annualized_return",
        save_plot=True,
        show_plot=False,
        save_path=save_path,
    )
    intraday_lib.create_group_heatmap(
        performance_results,
        group_name="group_HML",
        key_name="sharpe",
        save_plot=True,
        show_plot=False,
        high_contrast=True,
        save_path=save_path,
    )
    intraday_lib.save_performance_summary(
        performance_results,
        filename=f"{save_path}/group_performance_summary.csv",
    )
    print(f"Intraday results -> {save_path}/")


#%% run
from core.intraday_alphas import PANEL_BASED_INTRADAY_FACTORS

store = None
need_store = any(
    f not in PANEL_BASED_INTRADAY_FACTORS and f != "intraday_example"
    for f in factors_to_run
)
try:
    if need_store:
        from minute_bar_store import get_default_store

        hist = getattr(cfg, "INTRADAY_ALPHA_STORE_START", cfg.MINUTE_BAR_HISTORY_START)
        store = get_default_store(start_date=hist)
except Exception as exc:  # noqa: BLE001
    print(f"WARNING: MinuteBarStore init failed ({exc})")

print(f"Factors to run: {factors_to_run}", flush=True)
print(
    f"Minute-eval window: {minute_eval_start.date()} → {minute_eval_end.date()} "
    f"| limit_filter={apply_limit_filter}",
    flush=True,
)

for fname in factors_to_run:
    if fname == "intraday_example":
        narrow = build_example_narrow_factor()
        backtest_intraday_factor(fname, narrow)
        continue
    try:
        is_panel = fname in PANEL_BASED_INTRADAY_FACTORS
        src = "panel-cache" if is_panel else "MinuteBarStore"
        f_start = start_day if is_panel else minute_eval_start
        f_end = end_day if is_panel else minute_eval_end
        print(f"[BUILD] {fname} via {src} ({f_start.date()}→{f_end.date()}) ...", flush=True)
        narrow = build_intraday_narrow_table(
            fname, f_start, f_end, store=store
        )
        if narrow.empty:
            print(f"[SKIP] {fname}: empty narrow table")
            continue
        print(f"[BUILD] {fname} rows={len(narrow):,}", flush=True)
        backtest_intraday_factor(fname, narrow)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {fname}: {exc}")

print(f"Manifest (intraday) -> {manifest_path}")
