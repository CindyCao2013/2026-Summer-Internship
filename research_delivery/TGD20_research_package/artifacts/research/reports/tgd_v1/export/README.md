# TGD20 单因子研究报告包（可导出）

对应研报主题：**日内分钟收益率的时序特征：逻辑讨论与因子增强**

## 推荐阅读

| 文件 | 说明 |
|------|------|
| [`factor_report.md`](factor_report.md) | 完整中文研究报告 |
| [`factor_report.html`](factor_report.html) | 自包含 HTML（图已嵌入；浏览器打开后 Print→PDF） |
| [`metrics.json`](metrics.json) | 机器可读双轨评分 |

上级目录中文主文件：  
`../日内分钟收益率时序特征_TGD20因子研究报告.md`

## 标准数据结构

```text
export/
├── factor_summary.csv
├── mechanism_analysis.csv
├── yearly_stability.csv
├── execution_summary.csv
├── metrics.json
├── factor_report.md / .html
└── figures/
      ├── ic_curve.png
      ├── decile_return.png
      ├── hml_curve.png
      └── cumulative_long_short.png
```

模板镜像：`research/reports/factors/TGD20/`（供后续 FlowDensity 等复制）

## 原则

- 公式冻结；本包只做呈现与归档  
- 不做 TGD×Flow 合成（属 Phase 2 Combination）  
- Production capacity 未纳入本包
