# SKEW — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `SKEW` |
| **headline** | `AlphaIdioSKEW60` |
| **family** | higher_moment · behavioral_lottery |
| **data_level** | daily EOD |
| **status** | research_candidate |
| **formula** | frozen (windows pre-registered) |

## Economic intuition

Investors overweight lottery-like (positively skewed) stocks → overpricing →
lower future returns. Negatively skewed names earn crash-risk compensation.

TGD20 asks *when* returns arrive within the day; SKEW asks *how asymmetric*
the return distribution is. Both are return-distribution information, but different moments.

## Delivery alpha

Raw research quantity has expected **negative** RankIC.
Delivery signal: `Alpha = -SKEW` (long low / negative skew).
