# D1_LiquidityQuality60d — Formula (frozen)

## 1. Raw variables

Daily OHLCV fields from the EOD engine cache:

- `ret_1d` — daily return  
- `amount_cv_20d` — 20-day coefficient of variation of dollar volume  

## 2. Intermediate variables

**60-day volatility**

\[
\sigma_{60,i,t} = \mathrm{std}(r_{i,\cdot}) \text{ over 60 trading days}
\]

**Amount stability** (higher when CV is lower)

\[
S_{i,t} = -\mathrm{amount\_cv\_20d}_{i,t}
\]

## 3. Cross-sectional blend (frozen library)

Per day, rank each leg cross-sectionally, then average:

\[
\mathrm{D1}_{i,t} = \mathrm{mean\_rank}\bigl(-\sigma_{60,i,t},\; S_{i,t}\bigr)
\]

Module: `factor_formulas_eod_engine.py` → `f_low_vol_liquidity_quality_60d`

## 4. Final investable signal

\[
\mathrm{signal}_{i,t} = \mathrm{D1}_{i,t-1}
\]

Evaluation / execution (1D.7): **raw cross-sectional z-score** of the library panel.  
Intended production candidate: size+industry neutralized (Protocol Production Track — pending re-run).

---

## One-line identity

```text
D1_LiquidityQuality60d = CS rank-mean( −vol_60d, −amount_cv_20d )
```

Not:

```text
D1 ≠ amount_stability alone
D1 ≠ retuned window variant under same id
```
