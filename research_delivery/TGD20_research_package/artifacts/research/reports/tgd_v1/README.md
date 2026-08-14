# TGD v1 — Stage 4 Validation + Research Report

Research validation of `TGD20` (formula layer frozen).

## Canonical report（中文正式版）

| 文件 | 说明 |
|------|------|
| [`日内分钟收益率时序特征_TGD20因子研究报告.md`](日内分钟收益率时序特征_TGD20因子研究报告.md) | 完整中文研究报告 |
| [`日内分钟收益率时序特征_TGD20因子研究报告.html`](日内分钟收益率时序特征_TGD20因子研究报告.html) | 可导出 HTML（图已嵌入；浏览器 Print→PDF） |
| [`export/`](export/) | 标准数据包：csv + figures + metrics.json |
| [`../factors/TGD20/`](../factors/TGD20/) | 单因子模板镜像（供后续因子复制） |

英文归档版：[`TGD_factor_research_report.md`](TGD_factor_research_report.md)  
Library card：[`../../alpha_library_v1/research_satellites/TGD20.yaml`](../../alpha_library_v1/research_satellites/TGD20.yaml)

## Run

```bash
OMP_NUM_THREADS=1 python run_tgd_validation_v1.py
OMP_NUM_THREADS=1 python run_tgd_replication_integrity.py   # mechanism + family + metrics.json
OMP_NUM_THREADS=1 python run_tgd_execution_opt_v1.py         # execution (documented; production later)
```

## Guarantees

- `signal_shift=1` (no same-day lookahead)
- 10-group + H-L (not paper 5-group)
- Does **not** modify `core/l2_features/tgd.py`

## Layout

| Path | Layer |
|------|-------|
| `TGD_factor_research_report.md` | Full research report |
| `summary.md` | Stage-4 short summary |
| `ic/rank_ic.csv` | A — daily RankIC |
| `portfolio/` | A — 10-group+H-L cum, decile bars, H-L curve |
| `neutralization/neut_summary.csv` | B |
| `stability/yearly_ic.csv` | C |
| `cost/turnover_cost.csv` | D |
| `replication/` | Phase-1 metrics + Phase-2 integrity |
| `execution/` | Execution Optimization v1 (factor frozen) |

## Evaluation dual score

| Score | Metrics | Audience |
|-------|---------|----------|
| Research | RankIC, ICIR, Gross Sharpe, MDD, Mono | Presentation / discovery |
| Production | TO, ImpliedFee, Net Sharpe | Library admission |

Schema: `replication/factor_metrics_schema.csv` · pack: `replication/metrics.json`

## Next (deferred)

Production readiness: capacity / CSI300–1000 ladder — after report sign-off.  
Higher-value research: **TGD ⊥ Flow Density** composite — not further TGD formula tuning.
