# mid_order_ratio — Single Factor Research Report

> **Research status:** Completed factor construction audit, standalone predictive validation, robustness testing, and exposure diagnostics.

## Research positioning

本报告只验证：

> `mid_order_ratio` 是否具有统计显著、稳健且可作经济解释的 standalone predictive power？

它不是因子组合、权重配置或生产化报告。

## Factor definition

**mid_order_ratio measures the fraction of daily traded amount contributed by trades with transaction amount between RMB 40k and RMB 200k.**

原始因子与下一交易日收益排序呈负相关：

| Direction | RankIC | ICIR | Interpretation |
|---|---:|---:|---|
| Raw `mid_order_ratio` | **-4.80%** | **-6.33** | 因子值越高，次日收益排序越低 |
| Effective `-mid_order_ratio` | **+4.80% equivalent** | **+6.33 equivalent** | 用于 decile 与 H-L 的展示方向 |

负 ICIR 只反映 raw factor 的经济方向，不表示统计失效。

## Research conclusion

1. **Construction:** 旧 Tick 查询的逐日时段和深市成交识别存在实质问题；正式结果已使用 strict-trade cache 重构；
2. **Prediction:** CSI1000 raw RankIC -4.80%，t-stat -7.58，67.9% 的交易日 IC 为负；
3. **Robustness:** 四股票池方向一致，25 个阈值单元全部为负 IC，16 / 18 个月平均 IC 为负；
4. **Exposure diagnostics:** 行业、市值、momentum、volatility 和 turnover 可以解释部分关系，但未在当前残差检验中完全解释；
5. **Economic interpretation:** 短期流动性压力与临时价格影响是候选机制，但主动方向、价格路径、因果和反转均未被当前结果识别；
6. **Verdict:** 该因子值得作为 research candidate / candidate alpha feature 继续进行更长历史和样本外验证。

## Report navigation

1. [Executive Summary](01_executive_summary.md)
2. [Data and Factor Construction](02_data_and_factor_construction.md)
3. [Standalone Signal Validation](03_standalone_signal_validation.md)
4. [Robustness Analysis](04_robustness_analysis.md)
5. [Factor Exposure Diagnostics](05_factor_exposure_diagnostics.md)
6. [Economic Interpretation](06_economic_interpretation.md)
7. [Limitations and Future Research](07_limitations_and_future_research.md)
8. [Research Decision Framework](08_research_decision_framework.md)
9. Appendix
   - [Code Reference](appendix/code_reference.md)
   - [Data Lineage](appendix/data_lineage.md)
   - [Reproduction Commands](appendix/reproduction_commands.md)

## Full report export

- [Public single-file HTML](export/public/index.html) — images embedded; open via browser link
- [Local HTML](export/mid_order_ratio_report.html) — uses local MathJax assets
- [PDF](export/mid_order_ratio_report.pdf)

Public HTML can be opened directly:

```bash
# absolute file link
xdg-open "file://$(pwd)/research/reports/factors/mid_order_ratio/export/public/index.html"

# or serve a shareable http link on this machine
cd research/reports/factors/mid_order_ratio/export/public
python3 -m http.server 8765 --bind 0.0.0.0
# then open: http://<this-host>:8765/
```

```bash
/opt/conda/anaconda3/envs/base_93/bin/python \
  l2_factor_reproduction/scripts/export_mid_order_ratio_report.py
```

## Authoritative evidence

- Machine-readable results: [artifacts/](artifacts/)
- Figures: [figures/](figures/)
- Manifest: [artifacts/artifact_manifest.json](artifacts/artifact_manifest.json)
- Strict cache builder: `l2_factor_reproduction/scripts/build_mid_order_ratio_strict_cache.py`
- Report artifact generator: `l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py`
- Strict cache SHA256:

```text
ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f
```

旧 `factor_narrow.parquet` 与严格面板的 Spearman 仅为 0.7996，保留为 audit comparison，不作为本报告结论依据。

## Scope boundary

所有 Sharpe 均为 `fee=0` gross sorting diagnostics。H-L 用于验证单因子横截面两端分离，不代表成本后结果。Further portfolio integration is outside the scope of this report.