# 因子库清单（Factor Inventory）

> ~160 个已注册因子，按家族分组。状态口径：**主库已接入** = 在 `factor_config.TRACK_DEFAULT_LISTS` 中可经 `Factor_Test_Process.py` 一键回测；**交付/候选** 以 `research_delivery/factor_delivery_plan.csv` 为准；**实验** = 独立脚本线，未入主 runner。
> 生成时点：2026-08-04。

---

## 0. 新因子如何进入系统

```text
factor_formulas_xxx.py 写函数 + @register_factor(name)      → 公式注册
        ↓
factor_taxonomy*.py 加入对应 *_LIST（附元数据）              → 清单注册
        ↓
factor_config.py TRACK_DEFAULT_LISTS[track] 或 CUSTOM_FACTOR_LIST
        ↓
Factor_Test_Process.py（track 分支加载数据）→ factor_runner.run_eod_batch
        ↓
Factor_Dev_Lib.groupTest 十分组回测 → result/ 或 research/results/
        ↓
run_*_validation / delivery 评估 → research_delivery 交付（factor_index.csv）
```

注册机制实证：`factor_formulas.py:188,192,401`（`FACTOR_REGISTRY`/`build_factor`）；`factor_formulas_eod_engine.py:30,34,656`（`EOD_ENGINE_REGISTRY`）。
元数据三套：`factor_taxonomy.py:9,214,275`（family/hypothesis/mechanism/direction_hint）；`factor_taxonomy_cn.py:21,122`（+cn_family/source/data_layer）；`factor_specs/*.yaml`（Protocol 层：frozen_formula/adapter/evaluation/pack）。
Track 机制见 `factor_config.py:79,193-221`；当前 `TRACK="ideal_amplitude_active_v2"`，回测区间 `2021-01-01 ~ 2024-06-30`（`factor_config.py:84-86`）。

## 1. 因子总表

### 1.1 EOD PV 价量（Wind EOD，`factor_formulas.py`）— 主库已接入


| 清单          | 因子                                                                                                                                                                                                                                                                                                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classic（20） | `reversal_20d/60d/120d/240d`, `momentum_20d/60d`, `rsi_14`, `rsi_14_reversal`, `volume_20d_mean`, `volume_60d_mean`, `amount_20d_mean`, `amount_60d_mean`, `turnover_20d_mean`, `turnover_60d_mean`, `volatility_20d/60d`, `high_low_20d/60d`, `amount_to_volatility_20d/60d`                                                                                |
| New EOD（12） | `volume_surge_no_return_20d`, `moderate_volume_up_20d`, `overheated_turnover_proxy_20d`, `low_attention_reversal_20d`, `volume_contraction_stability_20d`, `upper_shadow_pressure_20d`, `lower_shadow_support_20d`, `range_contraction_20d`, `amount_stability_20d`, `price_volume_divergence_20d`, `amount_acceleration_20d`, `volume_price_efficiency_20d` |


### 1.2 EOD Engine / HF structured alpha（Wind EOD，`factor_formulas_eod_engine.py` + `factor_taxonomy.py`）— 主库已接入


| 家族            | 因子                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core 8        | `trend_consistency_20d`, `liquidity_stability_20d`, `liquidity_shock_20d`, `volatility_level_20d`, `volatility_regime_change_20d`, `return_autocorr_5d`, `drawup_drawdown_ratio_20d`, `volume_price_divergence_20d`                                                                                                                                                                                         |
| Extended +9   | `liquidity_persistence_20d`, `vol_of_vol_20d`, `range_expansion_20d`, `overreaction_shock_5d`, `underreaction_gap_20d`, `close_location_value_20d`, `price_inefficiency_20d`, `net_volume_pressure_20d`, `liquidity_acceleration_20d`                                                                                                                                                                       |
| Priority A（7） | `amihud_illiquidity_20d`, `amihud_shock_reversal_5d`, `max_daily_return_20d`, `cn_trend_pv_20d`, `loser_liquidity_reversal_5d`, `winner_sentiment_reversal_5d`, `amihud_amount_orth_20d`                                                                                                                                                                                                                    |
| SKEW          | `skew_20d/60d/120d`（canonical 实现委托 `core/factors/skew/`）                                                                                                                                                                                                                                                                                                                                                    |
| HF v2（13）     | `net_inflow_asymmetry_20d`, `amount_acceleration_20d`, `flow_persistence_decay_20d`, `intraday_reversal_intensity_20d`, `range_entropy_20d`, `attention_shock_cs_5d`, `winner_crowding_exhaustion_20d`, `loser_panic_stabilization_20d`, `relative_liquidity_strength_20d`, `momentum_rank_dispersion_20d`, `low_vol_liquidity_quality_20d`, `tail_risk_min_return_20d`, `volatility_adjusted_momentum_20d` |
| HF v3（11）     | `vol_liquidity_stress_20d`, `liquidity_fragility_20d`, `vol_liquidity_rank_gap_20d`, `momentum_timescale_conflict_20d`, `flow_price_rank_gap_20d`, `momentum_regime_flip_20d`, `downside_tail_cluster_20d`, `upside_fragility_20d`, `asymmetric_tail_ratio_20d`, `momentum_rank_churn_20d`, `return_skew_shift_20d`                                                                                         |
| HF v4（8）      | `composite_liquidity_stability_20d`, `amihud_stability_20d`, `return_stability_20d`, `amount_stability_60d`, `shadow_stability_20d`, `stability_quality_composite_20d`, `low_vol_stability_rank_20d`, `stable_reversal_blend_20d`                                                                                                                                                                           |
| HF v5（8）      | `liquidity_conditioned_momentum_20d`, `liquidity_shock_recovery_5d`, `triple_crowding_exhaustion_20d`, `trend_quality_composite_20d`, `liquidity_vol_regime_20d`, `tail_adjusted_momentum_60d`, `flow_price_divergence_20d`, `liquidity_accel_risk_filtered_20d`                                                                                                                                            |
| Robust（8）     | `residual_momentum_60d`, `information_ratio_momentum_60d/120d`, `residual_liquidity_20d`, `relative_vol_adjusted_liquidity_20d`, `low_vol_liquidity_quality_60d`, `stability_signal_persistence_20d`, `residual_reversal_20d`                                                                                                                                                                               |


### 1.3 CN Broker 券商经典复现（Wind EOD，`factor_formulas_cn_broker.py` + `factor_taxonomy_cn.py`）— 主库已接入

`cn_turnover_percentile_20d`, `cn_turnover_change_rate_20d`, `cn_volume_surge_moment_20d`, `cn_amount_distribution_skew_20d`, `cn_price_volume_divergence_20d`, `cn_chase_behavior_20d`, `cn_herding_proxy_20d`, `cn_attention_shock_5d`, `cn_new_high_breakout_252d`, `cn_rsi_momentum_gap_20d`, `cn_shadow_combo_20d`, `cn_limit_up_strength_20d`, `cn_turnover_concentration_20d`

### 1.4 L2 微观结构（DDB 分钟 Active_* 日聚合，`l2_data_loaders.py` 缓存）— 主库已接入


| 家族                                  | 因子                                                                                                                            | 状态                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| v1（`factor_formulas_l2.py`）         | `cn_voi_20d`, `cn_oir_20d`, `cn_mpb_20d`                                                                                      | 已 closed：无独立维度（`l2_v1_closed.md`） |
| v2 事件驱动（`factor_formulas_l2_v2.py`） | `cn_voi_shock`, `cn_mpb_shock`, `cn_flow_persistence`, `cn_imbalance_duration`, `cn_liquidity_consumption`, `cn_cancel_shock` | 已验证（`l2_v2_verdict.md`）           |


### 1.5 Liquidity Norm 规模调整流动性（Wind EOD + 市值，`factor_formulas_liquidity_norm.py`）— 主库已接入

`amount_stability_20d`, `amount_per_mktcap_stability_20d`, `turnover_stability_20d`, `volume_stability_20d`, `liquidity_amount_residual_20d`, `turnover_amount_residual_20d`, `liquidity_persistence_norm_20d`, `liquidity_shock_norm_20d`

### 1.6 Fundamental / Value / Quality（Wind derivative + ASHARETTMHIS，`factor_formulas_fundamental.py` / `_value.py`）— 部分接入


| 批次         | 因子                                                                                        | 状态              |
| ---------- | ----------------------------------------------------------------------------------------- | --------------- |
| Phase 1    | `log_float_mktcap`, `log_total_mktcap`, `float_to_total_mktcap`, `bp`, `ep_ttm`, `sp_ttm` | 已实现             |
| 中性价值锚      | `ep_ttm_ind_neutral`, `bp_ind_neutral`                                                    | 已实现             |
| Quality D7 | `roe`, `roe_stability`, `gross_profitability`, `cfo_quality`, `quality_composite`         | 已实现（verdict 已出） |
| Value D6   | `value_ep`, `value_bp`, `value_cfp`, `value_composite`                                    | 已实现             |
| Roadmap    | `relative_bp/ep/sp`, `roa`, `net_profit_margin`, `asset_turnover`, `sales_to_mktcap`      | 未实现             |


### 1.7 SUE / 事件（`factor_formulas_sue.py` / `_events_p1.py`）— 实验/脚本线

`unexpected_profit_notice_surprise_20d`, `sue_np_yoy_z`, `sue_eps_consensus`, `analyst_np_revision_20d`, `profit_notice_mid_surprise`；事件 `major_holder_net_increase`, `major_holder_increase_only`, `insider_net_buy`。
**当前焦点**：`SUE_ConsensusEPS` 处于 PIT 事件面板设计阶段（`docs/milestone_c2_sue_phase1_pit_panel_design.md`，DESIGN ONLY）。

### 1.8 ActiveV2 / 切割系（DDB Active_* + `core/l2_features` builders，`factor_formulas_*_active_v2.py`）— 主库已接入


| 家族                      | 因子                                                                                                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SmartMoney ActiveV2     | `SmartMoneyActiveV2`, `SmartMoneyActiveV2_raw`                                                                                                                                           |
| APM ActiveV2            | `APM_ActiveV2`, `APM_ActiveV2_Weekly`, `APM_ActiveV2_Weekly_Thu`, `APM_ActiveV2_Raw`, `APM_ActiveV2_Session`, `APM_ActiveV2_Smart`, `APM_ActiveV2_Delta`, `APM_ActiveV2_SmartV2(_1F/_1)` |
| IdealReversal ActiveV2  | `IdealReversal_ActiveV2`, `_Weekly`, `_PureRev`, `_Weekly_PureRev`, `_Weekly_Thu`, `_RollingGate`, `_Weekly_Thu_RollingGate`, `_raw`                                                     |
| IdealAmplitude ActiveV2 | `IdealAmplitude_ActiveV2`, `IdealAmplitude_ActiveV2_raw`（当前 TRACK 默认）                                                                                                                    |


### 1.9 Intraday 分钟因子（MinuteBarStore / DDB 原生，`intraday_formulas.py` + `core/intraday_alphas.py`）— intraday runner 接入

已实现：`close_vwap_deviation`, `active_buy_sell_imbalance`, `late_session_strength`, `volume_front_loading`, `volume_back_loading`, `morning_reversal_pressure`, `TGD20_1429`, `SmartMoney_1129_Rev`, `bartime_ofi`, `ofi_persistence`, `active_buy_shock`, `average_active_trade_size`, `large_active_buy_ratio`, `intraday_amihud`, `realized_volatility`, `minute_skew`。
多数已过 DDB production parity（`factor_config.py:156-179` 的 `*_USE_DDB=True`）。

### 1.10 L2 Alpha Factory（`l2_factor_reproduction/`，实验/候选池）


| 因子                                                                 | 数据源                                       | 状态                                                                                                                                                                                                                              |
| ------------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `avg_outflow_ratio` (+v2/v3)                                       | DDB 分钟                                    | Phase1 完成，净收益为负，归档                                                                                                                                                                                                              |
| `big_order_net_inflow`, `big_order_drive_ret`                      | DDB 分钟代理                                  | Phase1 完成，弱+高成本，归档                                                                                                                                                                                                              |
| `net_order_change`, `order_change_volatility`, `order_change_skew` | 需 `L2_Snapshot_Daily` 本地表                 | 未启用（待 CH→DDB ETL）                                                                                                                                                                                                               |
| `**mid_order_ratio*`*                                              | **CH Tick**                               | **已实现；优化收尾，建议入库为 `-mid_order_ratio` 行业中性周度（详见 `docs/mid_order_ratio_pipeline.md`）**                                                                                                                                             |
| Trade Flow Family v1（9）                                            | CH Tick → `l2_primitive_trade_flow_daily` | 候选池已生成并完成 2019-2026 baseline：`net_buy_ratio`, `net_buy_count_ratio`, `buy_dominance`, `avg_buy_trade_size`, `avg_sell_trade_size`, `trade_size_asymmetry`, `flow_concentration`, `flow_zscore_20d`, `flow_acceleration`；未筛选、未组合 |
| Order Size Family v1（20）                                           | CH Tick → `order_size_distribution_daily` | 候选池已生成并完成 2019-2026 baseline；覆盖 small/mid/large/super-large level、spread、entropy/HHI、分档 pressure/direction 和 20 日 shock；20 个公式形成 7 个族内经验相关簇，未筛选、未组合                                                                             |


候选池机器可读产物：

- `research/results/l2_reproduction/candidate_pool_v1/trade_flow_family/`
- `research/results/l2_reproduction/candidate_pool_v1/order_size_family/`
- `research/results/l2_reproduction/primitives/order_size_distribution_daily/`

两批冻结 family registry 共 29 个公式；加上 Sprint 2 的
`net_buy_amount_mcap` bridge candidate 后，L2 Candidate Pool v1 已达到 30 个
候选口径。该数字是公式数量，不是独立 alpha 数量；跨家族统一相关/暴露审计尚未执行。

### 1.11 CH SSL2 实验线（`research/l2_alpha/`，实验）

- 基础 8：`l2_top_book_imbalance`, `l2_depth_imbalance`, `l2_weighted_oi`, `l2_microprice_bias`, `l2_relative_spread`, `l2_cancel_pressure`, `l2_liquidity_skew`, `l2_liquidity_wall`（`research/l2_alpha/schema.py:44`）
- Phase2 聚合 4+2：`l2_weighted_oi_mean`, `l2_microprice_bias_mean`, `l2_depth_imbalance_mean`, `l2_cancel_pressure_sum` + 诊断
- 特征工厂 20：`depth_imb_`*, `woi_*`, `micro_bias_*`, `spread_*`, `cancel_*`（`research/l2_alpha/feature_factory/registry.py:105`）

## 2. 交付状态（`research_delivery`）


| 因子                                                     | 状态                                                                                                                | 备注                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `TGD20`                                                | **完整交付**                                                                                                          | 全量 standalone package（report+spec+代码快照+figures） |
| `D1_LiquidityQuality60d`                               | 卡片交付                                                                                                              | exact G10 rerun pending                         |
| `FlowDensity20`                                        | 卡片交付                                                                                                              | Template v2 pack 完成，公式未冻结                       |
| `APM_SessionResidual`                                  | delivered/testing                                                                                                 | C1 track closed，exact metric pending            |
| `IdealReversal` / `IdealAmplitude`                     | delivered(soft)/testing                                                                                           | 单调性弱                                            |
| `SKEW`                                                 | A- research_candidate                                                                                             | 有独立 research_package（非 v2 管线）                   |
| `AmihudShockReversal5d`                                | testing_candidate                                                                                                 | promotion-only                                  |
| `SmartMoney10d`                                        | Tier B parked                                                                                                     | 执行不达标，不进 library                                |
| Batch 2 计划                                             | `LiquidityResidual20d`, `ActiveTradingImbalance`, `PathMomentum20`, `VolatilityRegimeAlpha`, `OvernightDominance` | planned                                         |
| `mid_order_ratio`（拟名 `order_flow_mid_reversal_weekly`） | **建议入库评估中**                                                                                                       | 见 pipeline 文档                                   |


## 3. 分类统计


| 状态              | 数量级  | 代表                                               |
| --------------- | ---- | ------------------------------------------------ |
| 主库已接入（可一键回测）    | ~150 | EOD Engine 各家族、CN Broker、ActiveV2、Fundamental 部分 |
| 交付/候选（delivery） | ~10  | TGD20、D1、FlowDensity20、SKEW…                     |
| 实验线             | ~35  | SSL2 特征工厂、SUE、L2 flow P2                         |
| 归档/弃用           | ~8   | l2_reproduction Phase1、L2 v1、SmartMoney10d       |


