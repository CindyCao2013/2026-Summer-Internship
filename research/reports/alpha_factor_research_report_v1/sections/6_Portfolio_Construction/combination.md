# 6. Portfolio Construction (lightweight only)

**Scope of Report v1:** illustrative combination language only.  
**Out of scope:** production OS, risk model, constrained optimizer, TGD paper full additive reproduction.

## 6.1 Intent

When the library is larger, Stage 2 will run:

```text
Factor Library → Similarity / Residual IC → Composite v2 → Portfolio
```

Until then, combination claims stay **experimental**.

## 6.2 Suggested smoke experiments (not yet run as Report artifacts)

| Experiment | Components | Question |
|------------|------------|----------|
| Single | TGD20 | Baseline validated factor |
| Two-factor | TGD20 + D1 | Liquidity overlay vs temporal |
| Two-factor | TGD20 + FlowDensity20 | Flow overlay (watch amount channel) |
| Two-factor | TGD20 + APM | Session residual — check residual IC given ~0.4 corr |
| Multi | TGD + D1 + Flow | Paper-style additive sketch only |

Metrics to report when run: Gross/Net Sharpe, MaxDD, daily TO @15/30/50bp.

## 6.3 Combination curve status

**No experiment figure yet.** Joint equal-weight / z-score blend curves are not computed.  
Do not embed a placeholder image as if it were a backtest. Required inputs listed in `TODO.md`.

## 6.4 Discipline

- Do not present placeholder as performance.
- Do not use Ideal* or SmartMoney in “production candidate” combinations.
- Do not claim TGD paper composite closed until companion inventory + joint backtest exist.
