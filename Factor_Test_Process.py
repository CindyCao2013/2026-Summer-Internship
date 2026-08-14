# 日频因子测试入口（价量 / 财务共用 runner，改 factor_config.TRACK 切换）
#%% modules
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt

import factor_config as cfg
import intraday_lib
from factor_data_loaders import (
    load_derivative_wide_tables,
    load_eod_enriched_tables,
    load_eod_wide_tables,
    load_financial_ttmhis_long,
)
from factor_formulas import build_factor_cache
from factor_formulas_fundamental import (
    build_fundamental_cache,
    build_fundamental_factor,
    filter_available_fundamental_factors,
)
from factor_formulas_value import (
    build_value_factor,
    filter_available_value_factors,
)
from factor_runner import RunnerConfig, run_eod_batch

import Factor_Dev_Lib

#%% 解析配置
start_day = cfg.START_DAY
end_day = cfg.END_DAY
start_preheat = start_day - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
track = cfg.TRACK
factor_list = cfg.resolve_factor_list(track)
batch_tag = cfg.resolve_batch_tag(track, factor_list)
result_root = cfg.result_root_for(track)

runner_cfg = RunnerConfig(
    track=track,
    start_day=start_day,
    end_day=end_day,
    factor_list=factor_list,
    batch_tag=batch_tag,
    result_root=result_root,
    manifest_path=cfg.manifest_path_for(track),
    method=cfg.METHOD,
    batch_mode=cfg.BATCH_MODE,
    skip_completed=cfg.SKIP_COMPLETED,
    resume_from_existing=cfg.RESUME_FROM_EXISTING,
    save_results=cfg.SAVE_RESULTS,
    show_group_test_plots=cfg.SHOW_GROUP_TEST_PLOTS,
    universe_list=cfg.UNIVERSE_LIST,
)

if runner_cfg.batch_mode:
    matplotlib.use("Agg")
    if not runner_cfg.show_group_test_plots:
        plt.show = lambda *args, **kwargs: None

#%% 加载数据 + 构建因子
if track in (
    "eod_pv",
    "eod_engine",
    "eod_engine_ext",
    "eod_engine_priority_a",
    "eod_engine_hf_v2",
    "eod_engine_hf_v3",
    "eod_engine_hf_v4",
    "eod_engine_hf_v5",
    "eod_engine_robust",
    "eod_cn_broker_v1",
    "eod_cn_broker_all",
    "eod_latent",
):
    eod_tables, session = load_eod_wide_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    factor_cache = build_factor_cache(
        df_close=eod_tables.close,
        df_open=eod_tables.open,
        df_high=eod_tables.high,
        df_low=eod_tables.low,
        df_volume=eod_tables.volume,
        df_amount=eod_tables.amount,
        df_turnover=eod_tables.turnover,
    )

    if track == "eod_pv":
        from factor_formulas import build_factor, filter_available_factors

        runner_cfg.factor_list = (
            filter_available_factors(
                factor_list, has_turnover=(eod_tables.turnover is not None)
            )
            if cfg.BATCH_MODE
            else filter_available_factors(
                [cfg.SINGLE_FACTOR_NAME],
                has_turnover=(eod_tables.turnover is not None),
            )
        )

        def _build(fname):
            return build_factor(fname, factor_cache)

    elif track in (
        "eod_engine",
        "eod_engine_ext",
        "eod_engine_priority_a",
        "eod_engine_hf_v2",
        "eod_engine_hf_v3",
        "eod_engine_hf_v4",
        "eod_engine_hf_v5",
        "eod_engine_robust",
    ):
        from factor_formulas_eod_engine import (
            build_eod_engine_factor,
            filter_eod_engine_factors,
        )

        runner_cfg.factor_list = filter_eod_engine_factors(
            factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
        )

        def _build(fname):
            return build_eod_engine_factor(fname, factor_cache)

    elif track in ("eod_cn_broker_v1", "eod_cn_broker_all"):
        from factor_formulas_cn_broker import (
            build_cn_broker_factor,
            filter_cn_broker_factors,
        )

        runner_cfg.factor_list = filter_cn_broker_factors(
            factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
        )

        def _build(fname):
            return build_cn_broker_factor(fname, factor_cache)

    elif track == "eod_latent":
        from eod_projection import build_latent_eod_factors

        latent_factors = build_latent_eod_factors(factor_cache, n_pca_components=1)
        runner_cfg.factor_list = list(latent_factors.keys())

        def _build(fname):
            return latent_factors[fname]

elif track in ("eod_liquidity_norm", "eod_liquidity_norm_ext"):
    from factor_formulas_liquidity_norm import (
        build_liquidity_norm_cache,
        build_liquidity_norm_factor,
        filter_liquidity_norm_factors,
    )

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    factor_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )
    runner_cfg.factor_list = filter_liquidity_norm_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
    )

    def _build(fname):
        return build_liquidity_norm_factor(fname, factor_cache)

elif track == "alpha_bundle_v1":
    from factor_formulas import build_factor
    from factor_formulas_eod_engine import build_eod_engine_factor, filter_eod_engine_factors
    from factor_formulas_liquidity_norm import (
        build_liquidity_norm_cache,
        build_liquidity_norm_factor,
    )

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )
    runner_cfg.factor_list = factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]

    def _build(fname):
        if fname == "amount_stability_20d":
            return build_factor(fname, pv_cache)
        if fname == "liquidity_amount_residual_20d":
            return build_liquidity_norm_factor(fname, norm_cache)
        return build_eod_engine_factor(fname, pv_cache)

elif track in ("l2_microstructure_v1", "l2_microstructure_v2"):
    from l2_data_loaders import build_l2_daily_cache

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    l2_cache = build_l2_daily_cache(start_preheat, end_day, session=session, close=enriched.close)

    if track == "l2_microstructure_v1":
        from factor_formulas_l2 import build_l2_factor, filter_l2_factors

        runner_cfg.factor_list = filter_l2_factors(
            factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
        )

        def _build(fname):
            wide = build_l2_factor(fname, l2_cache)
            return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

    else:
        from factor_formulas_l2_v2 import build_l2_v2_factor, filter_l2_v2_factors

        runner_cfg.factor_list = filter_l2_v2_factors(
            factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
        )

        def _build(fname):
            wide = build_l2_v2_factor(fname, l2_cache)
            return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

elif track == "smart_money_active_v2":
    from core.l2_features.smart_money_active_v2_builder import (
        build_smart_money_active_v2_panel,
    )
    from factor_formulas_smart_money_active_v2 import (
        build_smart_money_active_v2_factor,
        filter_smart_money_active_v2_factors,
    )
    from industry_neutral import load_citics_industry_panel

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    # EWM span=10 needs ~40 calendar days preheat inside the builder, not full 400d.
    not_limit_pre = Factor_Dev_Lib.get_EOD_Not_Limit(
        start_day - dt.timedelta(days=60), end_day
    )
    smart_raw, _smart_long = build_smart_money_active_v2_panel(
        start_day,
        end_day,
        session=session,
        use_cache=True,
        not_limit=not_limit_pre,
        preheat_calendar_days=60,
    )
    industry = load_citics_industry_panel(start_day, end_day)
    trade_halt = Factor_Dev_Lib.get_TradeStatus(start_day, end_day)
    runner_cfg.factor_list = filter_smart_money_active_v2_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
    )

    def _build(fname):
        wide = build_smart_money_active_v2_factor(
            fname,
            smart_raw,
            industry=industry,
            halt_mask=trade_halt,
            float_mktcap=enriched.float_mktcap,
        )
        return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

elif track == "apm_active_v2":
    from core.l2_features.apm_active_v2_builder import build_apm_raw_variants
    from factor_formulas_apm_active_v2 import (
        build_apm_active_v2_factor,
        filter_apm_active_v2_factors,
    )
    from industry_neutral import load_citics_industry_panel

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    # EWM span=5 needs ~30 calendar days preheat inside the builder.
    not_limit_pre = Factor_Dev_Lib.get_EOD_Not_Limit(
        start_day - dt.timedelta(days=40), end_day
    )
    runner_cfg.factor_list = filter_apm_active_v2_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
    )
    apm_raw_variants = build_apm_raw_variants(
        start_day,
        end_day,
        session=session,
        use_cache=True,
        not_limit=not_limit_pre,
        preheat_calendar_days=40,
        names=runner_cfg.factor_list,
    )
    industry = load_citics_industry_panel(start_day, end_day)
    trade_halt = Factor_Dev_Lib.get_TradeStatus(start_day, end_day)

    def _build(fname):
        raw = apm_raw_variants.get(fname)
        if raw is None:
            raise KeyError(f"APM variant panel missing for {fname}")
        wide = build_apm_active_v2_factor(
            fname,
            raw,
            industry=industry,
            halt_mask=trade_halt,
            float_mktcap=enriched.float_mktcap,
        )
        return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

elif track == "ideal_reversal_active_v2":
    from core.l2_features.ideal_reversal_active_v2_builder import (
        build_ideal_reversal_raw_variants,
    )
    from factor_formulas_ideal_reversal_active_v2 import (
        build_ideal_reversal_active_v2_factor,
        filter_ideal_reversal_active_v2_factors,
    )
    from industry_neutral import load_citics_industry_panel

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    not_limit_pre = Factor_Dev_Lib.get_EOD_Not_Limit(
        start_day - dt.timedelta(days=60), end_day
    )
    ir_raw_by_name = build_ideal_reversal_raw_variants(
        start_day,
        end_day,
        enriched.close,
        session=session,
        use_cache=True,
        not_limit=not_limit_pre,
        preheat_calendar_days=60,
        weekly_method="friday",
    )
    industry = load_citics_industry_panel(start_day, end_day)
    trade_halt = Factor_Dev_Lib.get_TradeStatus(start_day, end_day)
    runner_cfg.factor_list = filter_ideal_reversal_active_v2_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
    )

    def _build(fname):
        wide = build_ideal_reversal_active_v2_factor(
            fname,
            ir_raw_by_name,
            industry=industry,
            halt_mask=trade_halt,
            float_mktcap=enriched.float_mktcap,
        )
        return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

elif track == "ideal_amplitude_active_v2":
    from core.l2_features.ideal_amplitude_active_v2_builder import (
        build_ideal_amplitude_panel,
    )
    from factor_formulas_ideal_amplitude_active_v2 import (
        build_ideal_amplitude_factor,
        filter_ideal_amplitude_factors,
    )
    from industry_neutral import load_citics_industry_panel

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    # EWM span=5 needs ~40 calendar days preheat inside the builder.
    not_limit_pre = Factor_Dev_Lib.get_EOD_Not_Limit(
        start_day - dt.timedelta(days=40), end_day
    )
    amp_smooth, _amp_long = build_ideal_amplitude_panel(
        start_day,
        end_day,
        session=session,
        use_cache=True,
        not_limit=not_limit_pre,
        preheat_calendar_days=40,
    )
    industry = load_citics_industry_panel(start_day, end_day)
    trade_halt = Factor_Dev_Lib.get_TradeStatus(start_day, end_day)
    runner_cfg.factor_list = filter_ideal_amplitude_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME]
    )

    def _build(fname):
        wide = build_ideal_amplitude_factor(
            fname,
            amp_smooth,
            industry=industry,
            halt_mask=trade_halt,
            float_mktcap=enriched.float_mktcap,
        )
        return wide.reindex(index=enriched.close.index, columns=enriched.close.columns)

elif track == "fundamental_value_d6":
    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)
    der_tables, _ = load_derivative_wide_tables(start_preheat, end_day, session=session)
    finance_long, _ = load_financial_ttmhis_long(start_preheat, end_day, session=session)
    fund_cache = build_fundamental_cache(
        der_tables,
        close=enriched.close,
        finance_long=finance_long,
    )
    runner_cfg.factor_list = filter_available_value_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME],
        has_pb=der_tables.pb is not None,
        has_pe=der_tables.pe_ttm is not None,
        has_finance_ann=finance_long is not None and len(finance_long) > 0,
    )

    def _build(fname):
        return build_value_factor(fname, fund_cache)

elif track in ("fundamental", "fundamental_batch1", "fundamental_phase2", "fundamental_quality_d7"):
    close_panel = None
    finance_long = None
    if track in ("fundamental_phase2", "fundamental_quality_d7"):
        enriched, session = load_eod_enriched_tables(start_preheat, end_day)
        session.run(intraday_lib.ddb_functions)
        close_panel = enriched.close
        der_tables, _ = load_derivative_wide_tables(start_preheat, end_day, session=session)
        finance_long, _ = load_financial_ttmhis_long(start_preheat, end_day, session=session)
    else:
        der_tables, session = load_derivative_wide_tables(start_preheat, end_day)
        session.run(intraday_lib.ddb_functions)
    fund_cache = build_fundamental_cache(
        der_tables,
        close=close_panel,
        finance_long=finance_long,
    )
    runner_cfg.factor_list = filter_available_fundamental_factors(
        factor_list if cfg.BATCH_MODE else [cfg.SINGLE_FACTOR_NAME],
        has_pb=der_tables.pb is not None,
        has_pe=der_tables.pe_ttm is not None,
        has_ps=der_tables.ps_ttm is not None,
        has_finance_ann=finance_long is not None and len(finance_long) > 0,
    )

    def _build(fname):
        return build_fundamental_factor(fname, fund_cache)

else:
    raise ValueError(
        f"Track `{track}` for intraday: use Intraday_Factor_Test_Process.py. "
        "Daily tracks: eod_pv | eod_engine | eod_engine_ext | eod_engine_priority_a | "
        "eod_engine_hf_v3 | eod_engine_hf_v4 | eod_engine_hf_v5 | eod_engine_robust | "
        "eod_cn_broker_v1 | eod_cn_broker_all | alpha_bundle_v1 | eod_latent | eod_liquidity_norm | "
        "eod_liquidity_norm_ext | fundamental | fundamental_batch1 | fundamental_phase2 | "
        "fundamental_quality_d7 | fundamental_value_d6 | l2_microstructure_v1 | "
        "l2_microstructure_v2 | smart_money_active_v2 | apm_active_v2 | "
        "ideal_reversal_active_v2 | ideal_amplitude_active_v2"
    )

#%% 过滤矩阵 + 批量回测
df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start_day, end_day)
df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start_day, end_day)
df_trade_status = Factor_Dev_Lib.get_TradeStatus(start_day, end_day)

all_results, summary_rows, skipped, failed = run_eod_batch(
    runner_cfg,
    build_factor_fn=_build,
    session=session,
    df_not_limit=df_not_limit,
    df_not_st=df_not_st,
    df_trade_status=df_trade_status,
)

#%% 单因子扩展测试（o2o / v2v；eod_pv / L2 active tracks，非 batch）
_ACTIVE_EXT_TRACKS = (
    "eod_pv",
    "smart_money_active_v2",
    "apm_active_v2",
    "ideal_reversal_active_v2",
    "ideal_amplitude_active_v2",
)
if track in _ACTIVE_EXT_TRACKS and not cfg.BATCH_MODE and runner_cfg.factor_list:
    fname = cfg.SINGLE_FACTOR_NAME
    factor_value = _build(fname).loc[start_day:end_day, :]
    if track == "smart_money_active_v2":
        base_idx = "000905.SH"
    else:
        base_idx = "000852.SH"

    ret_o2o = Factor_Dev_Lib.get_Ret_Matrix(
        start_day, end_day, method="o2o", base_index=base_idx
    )
    ret_v2v = Factor_Dev_Lib.get_Ret_Matrix(start_day, end_day, method="v2v")

    for ret_matrix in (ret_o2o, ret_v2v):
        signal = factor_value.copy()
        signal = signal.mul(df_not_limit)
        signal = signal.mul(df_not_st)
        signal = signal.mul(df_trade_status)
        signal = signal.shift(2)
        signal = signal.dropna(how="all", axis=1)
        signal = signal.dropna(how="all")
        Factor_Dev_Lib.groupTest(signal, ret_matrix, n=10)

    # L2 active tracks 已在 _build 内做过中性化；eod_pv 再补一次
    if track == "eod_pv":
        factor_neu = Factor_Dev_Lib.panel_neutral_size_ind(factor_value)
        signal = factor_neu.mul(df_not_limit).mul(df_not_st).mul(df_trade_status)
        signal = signal.shift(2).dropna(how="all", axis=1).dropna(how="all")
        Factor_Dev_Lib.groupTest(signal, ret_v2v, n=10)
