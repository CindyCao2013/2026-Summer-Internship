# Appendix C — 复现命令

## 1. 环境

```bash
cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
PY=/opt/conda/anaconda3/envs/base_93/bin/python
```

运行需要访问内网：

- ClickHouse `10.80.139.9:8123`
- DolphinDB `10.12.180.9:8902`

## 2. 连接自检

```bash
$PY -c "
from core.ddb.connection import get_ddb_session, close_shared_ddb_session
import clickhouse_connect
from COMMON_CONST import DATA_DB_HFDATA

s = get_ddb_session()
print('DDB:', s.run('1+1'))
close_shared_ddb_session()

c = clickhouse_connect.get_client(**DATA_DB_HFDATA)
print('CH:', c.query('SELECT 1').result_rows[0][0])
c.close()
"
```

预期：

```text
DDB: 2
CH: 1
```

## 3. 基础因子构造与 canonical 回测

```bash
$PY l2_factor_reproduction/scripts/run_single_factor.py \
  --factor mid_order_ratio \
  --start 2023-01-01 \
  --end 2024-06-30
```

输出：

```text
research/results/l2_reproduction/mid_order_ratio/
├── factor_narrow.parquet
├── group_pnl.csv
├── group_turnover.csv
├── rank_ic.csv
├── summary.json
├── cum_pnl.png
└── decile_bar.png
```

注意：当前代码已使用严格时段和交易所成交筛选，但该基础命令的查询 universe 仍是样本期成分并集，回测没有自动施加逐日 membership。旧图也没有完整标注 benchmark，不应直接替代正式报告图。仓库中现存 canonical 文件可能早于修正；只有重新执行本命令后才会更新。

## 4. 一阶中性化

```bash
$PY l2_factor_reproduction/scripts/test_neutralization.py \
  --factor mid_order_ratio \
  --neutral_types ind cap ind_cap
```

输出：

```text
research/results/l2_reproduction/mid_order_ratio/
├── neutralized_ind/
├── neutralized_cap/
├── neutralized_ind_cap/
└── neutralization_comparison.csv
```

## 5. 二阶中性化

```bash
$PY l2_factor_reproduction/scripts/test_double_neutralization.py \
  --factor mid_order_ratio

$PY l2_factor_reproduction/scripts/screen_second_pass.py \
  --factor mid_order_ratio
```

关键输出：

```text
research/results/l2_reproduction/mid_order_ratio/
├── second_pass_screen_ind.csv
└── second_pass_screen_ind_cap.csv
```

## 6. 周度持续性诊断

```bash
$PY l2_factor_reproduction/scripts/optimize_weekly.py \
  --factor mid_order_ratio \
  --raw

$PY l2_factor_reproduction/scripts/optimize_weekly.py \
  --factor mid_order_ratio
```

该模块是频率/换手研究，不属于本报告 headline。

## 7. 原参数、状态和时间脚本

```bash
$PY l2_factor_reproduction/scripts/analyze_param_sensitivity.py \
  --start 2023-01-01 \
  --end 2024-06-30

$PY l2_factor_reproduction/scripts/analyze_state_dependence.py \
  --factor mid_order_ratio

$PY l2_factor_reproduction/scripts/analyze_time_stability.py \
  --factor mid_order_ratio
```

原参数脚本不施加 CSI1000 日度 membership；正式报告版通过下一节命令重算。

## 8. 构建严格成交缓存

```bash
$PY l2_factor_reproduction/scripts/build_mid_order_ratio_strict_cache.py \
  --start 2023-01-01 \
  --end 2024-06-30
```

输出：

```text
research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/
├── strict_trade_chunks_2023-01-01_2024-06-30/
├── tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet
└── tick_bucketed_strict_trade_2023-01-01_2024-06-30.metadata.json
```

脚本不会覆盖 legacy cache；已验证缓存 SHA256 为
`ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f`。

## 9. 一键生成正式报告 artifacts 与图

```bash
$PY l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py \
  --start 2023-01-01 \
  --end 2024-06-30 \
  --bucket-cache \
    research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet \
  --output-root \
    research/reports/factors/mid_order_ratio
```

该脚本约完成：

1. 从 strict-trade bucket cache 重构 L4w/H20w 沪深 A 股面板；
2. 构造 SSE/SZSE（artifact key `ALL`）/ CSI300 / CSI500 / CSI1000 PIT universe；
3. 输出 universe comparison；
4. 生成 CSI1000 index-excess deciles；
5. 重算四股票池 × 四方法的 PIT 一阶中性化矩阵，以及 CSI1000 二阶中性化；
6. 重算 25 单元 PIT 参数网格；
7. 使用 T-1 turnover state 重算 regime IC；
8. 生成月度、滚动和分布图；
9. 写入 legacy-vs-strict 构造影响审计。

## 10. 正式 artifacts

```text
research/reports/factors/mid_order_ratio/artifacts/
├── artifact_manifest.json
├── construction_crosscheck.json
├── construction_crosscheck_top100.csv
├── strict_trade_filter_audit.json
├── factor_value_distribution_summary.csv
├── universe_comparison.csv
├── rank_ic_{all,csi300,csi500,csi1000}.csv
├── group_pnl_{all,csi300,csi500,csi1000}_ew_excess.csv
├── group_turnover_{all,csi300,csi500,csi1000}.csv
├── csi1000_rank_ic_daily.csv
├── csi1000_decile_index_excess_daily.csv
├── csi1000_decile_summary.csv
├── csi1000_decile_turnover_daily.csv
├── neutralization_comparison.csv
├── neutralization_by_universe.csv
├── second_neutralization_comparison.csv
├── parameter_sensitivity_csi1000_pit.csv
├── state_dependence_daily_ic.csv
├── state_dependence_summary.csv
├── time_stability_monthly_ic.csv
└── order_size_distribution.csv
```

## 11. 正式 figures

```text
research/reports/factors/mid_order_ratio/figures/
├── 01_pipeline_architecture.png
├── 02_order_size_distribution.png
├── 03_universe_comparison_table.png
├── 04_decile_cumulative_csi1000_index_excess.png
├── 05_decile_annualized_csi1000_index_excess.png
├── 06_ic_time_series.png
├── 07_rolling_ic.png
├── 07b_ic_stability_combined.png
├── 08_neutralization_comparison.png
├── 08b_universe_neutralization_matrix.png
├── 09_second_neutralization_comparison.png
├── 10a_parameter_sensitivity_icir.png
├── 10b_parameter_sensitivity_sharpe.png
├── 11_turnover_vs_ic_relationship.png
├── 12_high_low_turnover_regime_comparison.png
├── 13_ic_distribution.png
└── 14_monthly_ic.png
```

## 12. 代码静态检查

```bash
$PY -m py_compile \
  l2_factor_reproduction/python/ch_tick.py \
  l2_factor_reproduction/scripts/build_mid_order_ratio_strict_cache.py \
  l2_factor_reproduction/scripts/generate_mid_order_ratio_report_artifacts.py

$PY -m pytest -q l2_factor_reproduction/tests/test_ch_tick.py
```

## 13. 结果一致性检查

```bash
$PY - <<'PY'
import pandas as pd
import json

root = "research/reports/factors/mid_order_ratio/artifacts"
u = pd.read_csv(f"{root}/universe_comparison.csv")
g = pd.read_csv(f"{root}/parameter_sensitivity_csi1000_pit.csv")
n = pd.read_csv(f"{root}/neutralization_by_universe.csv")
s = pd.read_csv(f"{root}/second_neutralization_comparison.csv")

assert set(u["universe"]) == {"ALL", "CSI300", "CSI500", "CSI1000"}
assert (u["rank_ic"] < 0).all()
assert len(g) == 25
assert (g["rank_ic"] < 0).all()
assert len(n) == 16
assert (n["rank_ic"] < 0).all()
assert set(n["method"]) == {
    "raw", "industry", "market_cap", "industry+market_cap"
}
assert (u["effective_direction"] == -1).all()
assert (n["effective_direction"] == -1).all()
assert (s["abs_ic_retained_vs_matched"].between(0, 1)).all()

paper = g.query("L_wan == 4 and H_wan == 20").iloc[0]
csi = u.query("universe == 'CSI1000'").iloc[0]
assert abs(paper["rank_ic"] - csi["rank_ic"]) < 1e-12
assert abs(paper["hl_sharpe"] - csi["hl_sharpe"]) < 1e-12

manifest = json.load(open(f"{root}/artifact_manifest.json"))
assert "strict_trade" in manifest["sources"]["bucket_cache"]
assert manifest["sources"]["bucket_cache_sha256"] == (
    "ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f"
)
assert manifest["factor_definition"]["direction_policy"].startswith("frozen")
assert manifest["factor_definition"]["szse_trade_filter"] == (
    "Type='011' AND BidOrderNo>0 AND AskOrderNo>0"
)

print("verified")
PY
```

## 14. 阅读顺序

复现后先检查：

1. `construction_crosscheck.json`；
2. `universe_comparison.csv`；
3. `neutralization_by_universe.csv`；
4. `csi1000_decile_index_excess_daily.csv`；
5. `csi1000_decile_summary.csv` 的 monotonicity 是否可由日度 decile 重算；
6. 图 04 标题是否仍明确写 `CSI1000 Index-Excess Return`；
7. `parameter_sensitivity_csi1000_pit.csv` 的 L4w/H20w 是否与 CSI1000 headline 相等；
8. `artifact_manifest.json` 的日期、cache 和 metric conventions。

## 15. 导出整份报告

```bash
$PY l2_factor_reproduction/scripts/export_mid_order_ratio_report.py
```

输出：

```text
research/reports/factors/mid_order_ratio/export/
├── mid_order_ratio_report.html          # local assets MathJax
├── mid_order_ratio_report.pdf
├── public/index.html                    # public single-file HTML
└── assets/mathjax/
```

`public/index.html` 是单文件版本：图表 base64 内嵌，公式用 CDN MathJax，可直接用浏览器打开或挂到任意静态站点。

```bash
# 本机可分享的 http 链接
cd research/reports/factors/mid_order_ratio/export/public
python3 -m http.server 8765 --bind 0.0.0.0
# 浏览器打开 http://<host>:8765/
```

HTML 合并全部正文与附录并内嵌图表；PDF 使用 A4 分页和嵌入式中文字体。无需修改共享 Conda 环境。

