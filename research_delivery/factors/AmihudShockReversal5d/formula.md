# AmihudShockReversal5d — Formula (from existing implementation)

**Engine id:** `amihud_shock_reversal_5d`  
**Code:** `factor_formulas_eod_engine.py::f_amihud_shock_reversal_5d`  
**Do not retune under this delivery id without a new factor_id.**

## Exact implementation (frozen to code)

Daily illiquidity:

\[
\mathrm{ILLIQ}_{t}=\frac{|r_t|}{\mathrm{Amount}_t}
\]

Shock vs 20d mean (implementation uses **ratio to MA**, not z-score):

\[
\mathrm{Shock}_{t}=\frac{\mathrm{ILLIQ}_{t}}{\mathrm{MA}_{20}(\mathrm{ILLIQ})}
\]

Factor:

\[
\mathrm{ASR5}_{t}=-\,\mathrm{Shock}_{t}\times R_{t-5:t}
\]

where \(R_{t-5:t}\) is `ret_5d` from the EOD cache.

Python (verbatim logic):

```python
amihud = abs(ret_1d) / amount
shock = amihud / amihud.rolling(20, min_periods=10).mean()
factor = -(shock * ret_5d)
```

## Not equal to

- plain Amihud level (`amihud_illiquidity_20d`)
- D1 liquidity quality / amount stability
- raw short-term reversal without liquidity shock
