# D1_LiquidityQuality60d — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `D1_LiquidityQuality60d` |
| **display_name** | D1 Liquidity Quality 60d |
| **source** | Frozen EOD library · `low_vol_liquidity_quality_60d` |
| **library_id** | `D1_low_vol_liquidity_quality_60d` |
| **family** | liquidity |
| **data_level** | EOD / OHLCV |
| **status** | **candidate** |
| **formula** | **frozen** |

---

## Economic intuition

Stocks that combine **low volatility** with **stable liquidity participation** earn a persistent cross-sectional premium — a classical liquidity-quality / anti-churn state used as Base3 leg D1.

Unstable, high-activity names embed short-horizon noise and crowding. Low-vol names with steadier liquidity profiles may embed a slower-moving risk/behavioral premium that ranks well in the cross-section.

Unlike TGD, D1 does **not** claim an intraday residual channel. Mechanism evidence in this pack is **structural**: soft-bar metrics, universe comparative strength, and frozen-library role as production_base.

---

## Factor card (one page)

```yaml
factor: D1_LiquidityQuality60d
name: D1 Liquidity Quality 60d
source: frozen EOD library (low_vol_liquidity_quality_60d)
intuition: |
  Low vol × stable amount participation → liquidity quality state.
  Cross-sectional premium stronger on ALL than CSI300.
signal: low_vol_liquidity_quality_60d (frozen constructor)
direction: positive RankIC (long high D1)
horizon: daily signal_shift=1; 60d vol window (frozen)
```

---

## Universe diagnostics (not separate factors)

| Universe | RankIC | ICIR | H-L Sharpe |
|----------|-------:|-----:|-----------:|
| ALL | 0.0573 | 6.01 | 2.26 |
| CSI1000 | 0.0534 | 5.37 | 1.84 |
| CSI500 | 0.0468 | 4.36 | 0.90 |
| CSI300 | 0.0406 | 3.20 | 0.62 |

Edge is **stronger outside mega-cap** — not a CSI300-only artifact.

---

## Do not

- Retune D1 window under this id  
- Register `amount_stability` as independent validated twin  
- Treat universe ladder rows as separate factor_ids  
- Claim size+industry neutralization without Protocol re-run  

---

## Pack layout (v1 canonical)

```
research/reports/factors/D1_LiquidityQuality60d/
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
