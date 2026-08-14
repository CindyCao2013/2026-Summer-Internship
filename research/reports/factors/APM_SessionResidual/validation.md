# APM_SessionResidual — Validation

**Universe:** CSI1000 · **Period:** 2021-01-01 → 2025-12-31 · **Signal:** `apm_cs` · **No sign flip**

Artifacts: `ic_analysis/` · `quantile_analysis/` · `stability/` · scout `summary.json`

---

## 1. RankIC / ICIR

| Mode | RankIC | ICIR | n |
|------|-------:|-----:|--:|
| Raw | +2.39% | 4.10 | 1211 |
| Size neutral | +2.33% | 4.18 | 1211 |
| Industry neutral | +2.30% | 6.25 | 1211 |
| Size+industry | +2.25% | **6.55** | 1211 |

**Direction:** `positive_match` (paper positive).

Information gate: **PASS** (|IC|>2%, |ICIR|>1.5).

---

## 2. Decile structure

- Monotonicity (prefer increasing): **0.778**
- Not U-shaped; structure healthier than IdealAmplitude

---

## 3. Yearly stability

| Year | RankIC | ICIR | Gross Sharpe | Net@15bp daily |
|------|-------:|-----:|-------------:|---------------:|
| 2021 | +3.23% | 7.52 | 5.77 | 2.99 |
| 2022 | +2.49% | 4.65 | 3.41 | 1.00 |
| 2023 | +1.87% | 3.61 | 1.69 | −1.34 |
| 2024 | +1.73% | 2.37 | 3.06 | 1.11 |
| 2025 | +2.61% | 3.99 | 2.67 | 0.47 |

**5/5** years RankIC > 0.

---

## 4. Similarity (not residual IC)

| Peer | IC-series ρ | Signal CS ρ |
|------|------------:|------------:|
| FlowDensity20 | −0.25 | **0.006** |
| SmartMoney10d | ~0 | −0.062 |
| D1 | −0.11 | 0.149 |
| TGD20 | +0.37 | 0.426 |

APM ≉ Flow / SmartMoney. Moderate session overlap with TGD (different object).

---

## 5. Scout investability (pre-recipe)

Daily plain @15bp: Gross Sharpe 3.28 · Net **0.92** · TO **0.75**  
→ Scout verdict was `PASS_research_FAIL_invest` before C1.4 buffer recipe.

See [execution.md](execution.md) for Case A resolution.
