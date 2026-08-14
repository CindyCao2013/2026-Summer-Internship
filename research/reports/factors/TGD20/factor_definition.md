# TGD20 — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `TGD20` |
| **display_name** | Temporal Gradient Density (20d) |
| **source** | 开源证券 · 日内分钟收益率的时序特征（时间重心偏移） |
| **paper_id** | `20221225_kysec_intraday_timing` |
| **family** | temporal |
| **data_level** | L2 / minute bar |
| **status** | **validated** |
| **formula** | **frozen** |

---

## Economic intuition

股票上涨/下跌收益发生的时间位置，反映主动资金行为与价格发现过程。

Most classical factors measure **magnitude** (return, volume, volatility).  
TGD measures **timing**: within the day, *when* did upside vs downside returns concentrate?

Raw time-gap \(\tau = G_d - G_u\) is **not** the alpha.  
Frozen TGD20 is the **abnormal downside timing residual** after stripping return-structure controls, then MA20.

---

## Factor card (one page)

```yaml
factor: TGD20
name: Temporal Gradient Density
source: 时间重心偏移因子 (KYSEC intraday timing)
intuition: |
  When within the session did up/down returns concentrate?
  Residual downside timing predicts next-day cross-section.
signal: MA20( resid(εd ~ εu) )
direction: positive RankIC (long high TGD)
horizon: daily signal_shift=1; MA window=20 (frozen)
```

---

## Rejected primitives (do not confuse with TGD20)

| Primitive | ICIR (raw) | Verdict |
|-----------|-----------:|---------|
| \(\tau = G_d - G_u\) (MA20) | −0.56 | time ordering ≠ alpha |
| \(\upsilon = \|G_d - G_u\|\) (MA20) | −2.26 | separation alone insufficient |
| **TGD20** | **+6.98** | residual + MA20 |

---

## Do not

- Retune MA window (10/30/60) under this id  
- Replace residual with ML under this id  
- Mine a new TGD variant under this id  
- Treat raw \(G_d - G_u\) as TGD20  

---

## Pack layout (v1 canonical)

```
research/reports/factors/TGD20/
├── factor_definition.md   ← this file
├── formula.md
├── data_source.md
├── implementation.md
├── validation.md
├── summary.yaml
├── ic_analysis/
├── quantile_analysis/
├── stability/
└── execution/
```

Full narrative report (legacy): `research/reports/tgd_v1/TGD_factor_research_report.md`
