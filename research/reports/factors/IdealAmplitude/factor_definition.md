# IdealAmplitude — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `IdealAmplitude` |
| **display_name** | Ideal Amplitude |
| **source** | 开源证券 · 振幅因子的隐藏结构 |
| **paper_id** | `kysec_amplitude_hidden_structure` |
| **family** | microstructure / volatility |
| **data_level** | EOD |
| **status** | **testing** |
| **formula** | not frozen |

---

## Economic intuition

Raw Amp20 mixes high-price-state and low-price-state amplitude regimes.  
Cutting by **close price level** within a 20-day window isolates amplitude
information when price is elevated vs depressed.

- High close-state amplitude carries stronger negative alpha.
- Low close-state amplitude is weaker / mixed.
- Spread \(V = V_{high} - V_{low}\) purifies vs uncut Amp20.

Investable signal uses `direction=-1`: short high V, long low V.

---

## Factor card (one page)

```yaml
factor: IdealAmplitude
name: Ideal Amplitude
source: 振幅因子的隐藏结构 (KYSEC amplitude cutting)
intuition: |
  Where is amplitude alpha hidden inside Amp20?
  High close-state leg dominates; spread beats baseline.
signal: V = mean(amp | high-close) − mean(amp | low-close)
direction: negative RankIC (short high V)
horizon: signal_shift=1; window=20; λ=0.25
```

---

## Mechanism legs (diagnostics — not Registry siblings)

| Leg | RankIC | ICIR | Role |
|-----|-------:|-----:|------|
| V_high | −0.0507 | −6.44 | alpha locus |
| V_low | −0.0227 | −2.68 | weaker leg |
| V spread | −0.0378 | −7.66 | investable output |
| Amp20 baseline | −0.0596 | −4.44 | uncut object |

Knife separation (IC_high − IC_low): **−0.0280** · purity ≈ **0.746**

---

## Known failure modes

| Mode | Evidence |
|------|----------|
| **Weak monotonicity** | Spearman ≈ **0.11** (soft bar > 0.8) — decile shape not institutional-grade |
| Production Track deferred | Legacy ALL-universe harvest; exec on last 252d |
| Same family as IdealReversal | Residual orthogonality check pending |

---

## Do not

- Promote to `validated` without monotonicity fix + Dual Benchmark  
- Confuse `V_high` / `V_low` with Registry factor_ids  
- Treat raw Amp20 as IdealAmplitude  

---

## Pack layout (v1 canonical)

```
research/reports/factors/IdealAmplitude/
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
