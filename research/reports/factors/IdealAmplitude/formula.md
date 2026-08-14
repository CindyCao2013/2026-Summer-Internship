# IdealAmplitude — Formula

## 1. Object — daily amplitude

\[
\mathrm{Amp}_t = \frac{H_t}{L_t} - 1
\]

Optionally drop one-word limit days (high ≈ low).

Module: `factor_cutting/ideal_amplitude.py` → `daily_amplitude`

---

## 2. Knife — close price state

Within each 20-day lookback, partition days by close level:
- **High state:** top λ fraction by close (λ = 0.25)
- **Low state:** bottom λ fraction by close

Method: `quantile_split` via `rolling_quantile_mean_diff`

---

## 3. Leg aggregation

\[
V_{high} = \mathrm{mean}(\mathrm{Amp} \mid \text{high-close}), \qquad
V_{low}  = \mathrm{mean}(\mathrm{Amp} \mid \text{low-close})
\]

Aggregate: **mean** (paper recipe for amplitude object).

Minimum effective days: 10 (after dropping bad bars).

---

## 4. IdealAmplitude signal

\[
V = V_{high} - V_{low}
\]

Evaluation: `signal_shift=1`, `direction=-1` (short high V).

Module: `factor_cutting/ideal_amplitude.py`

---

## 5. One-line identity

```text
IdealAmplitude = mean(amp | high-close) − mean(amp | low-close)
```

Not:

```text
IdealAmplitude ≠ raw Amp20
IdealAmplitude ≠ V_high alone
```

---

## 6. Frozen parameters (current harvest)

| Param | Value |
|-------|------:|
| Window | 20 |
| λ (quantile frac) | 0.25 |
| min_effective_days | 10 |
| drop_one_word_limit | true |
| signal_shift | 1 |
| Cost baseline | 15 bp round-trip |

Formula not yet frozen (`formula_frozen: false`).
