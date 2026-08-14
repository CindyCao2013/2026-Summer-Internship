# IdealReversal — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `IdealReversal` |
| **display_name** | Ideal Reversal |
| **source** | 开源证券 · A股反转之力的微观来源（因子切割） |
| **paper_id** | `20200917_kysec_factor_cutting` |
| **family** | microstructure |
| **data_level** | EOD |
| **status** | **testing** |
| **formula** | not frozen |

---

## Economic intuition

Traditional 20-day return (Ret20) mixes informative and noisy days.  
Cutting by **average trade size** (ATS = amount / trade_count) isolates return
contribution on large-trade days vs small-trade days.

- High-ATS days ≈ institutional / large-ticket participation → stronger reversal.
- Low-ATS days ≈ retail noise → near-zero IC.
- Spread \(M = M_{high} - M_{low}\) purifies the reversal locus.

Investable signal uses `direction=-1`: short high M, long low M.

---

## Factor card (one page)

```yaml
factor: IdealReversal
name: Ideal Reversal
source: A股反转之力的微观来源 (KYSEC factor cutting)
intuition: |
  Where is reversal information hidden inside Ret20?
  High-ATS leg carries alpha; low-ATS leg is noise.
signal: M = sum(r | high ATS) − sum(r | low ATS)
direction: negative RankIC (short high M)
horizon: signal_shift=1; window=20
```

---

## Mechanism legs (diagnostics — not Registry siblings)

| Leg | RankIC | ICIR | Role |
|-----|-------:|-----:|------|
| M_high | −0.0430 | −5.73 | alpha locus |
| M_low | +0.0020 | +0.33 | near noise |
| M spread | −0.0338 | −6.48 | investable output |
| Ret20 baseline | −0.0437 | −4.06 | uncut object |

---

## Known failure modes

| Mode | Evidence |
|------|----------|
| Weak monotonicity | Spearman ≈ **0.44** (soft bar > 0.8) |
| Sharpe below soft bar | H-L Sharpe ≈ 1.70 (< 2.0) |
| Production Track deferred | Legacy ALL-universe harvest only |

---

## Do not

- Promote to `validated` without monotonicity fix + Production Track  
- Confuse `M_high` / `M_low` with Registry factor_ids  
- Treat raw Ret20 as IdealReversal  

---

## Pack layout (v1 canonical)

```
research/reports/factors/IdealReversal/
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

Full narrative report (legacy): `factor_report.md`
