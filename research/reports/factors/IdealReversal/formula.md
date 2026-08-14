# IdealReversal — Formula

## 1. Object — Ret20 (additive daily returns)

Daily return over lookback window \(W=20\):

\[
r_t = \frac{P_t}{P_{t-1}} - 1
\]

Object panel: sum or stack of daily returns (additive).

---

## 2. Knife — ATS (average trade size)

\[
\mathrm{ATS}_t = \frac{\mathrm{Amount}_t}{\mathrm{TradeCount}_t}
\]

Within each 20-day lookback, rank days by ATS and split:
- **High leg:** top 10 days (highest ATS)
- **Low leg:** bottom 10 days (lowest ATS)

Module: `factor_cutting/w_cut.py` → `rolling_rank_split_sum`

---

## 3. Leg aggregation

\[
M_{high} = \sum_{t \in \mathrm{HighATS}} r_t, \qquad
M_{low}  = \sum_{t \in \mathrm{LowATS}}  r_t
\]

Aggregate: **sum** (paper recipe for return object).

---

## 4. IdealReversal signal

\[
M = M_{high} - M_{low}
\]

Evaluation: `signal_shift=1`, `direction=-1` (short high M).

Module: `factor_cutting/ideal_reversal.py`

---

## 5. One-line identity

```text
IdealReversal = sum(r | high ATS) − sum(r | low ATS)
```

Not:

```text
IdealReversal ≠ raw Ret20
IdealReversal ≠ M_high alone
```

---

## 6. Frozen parameters (current harvest)

| Param | Value |
|-------|------:|
| Window | 20 |
| High / low count | 10 / 10 |
| signal_shift | 1 |
| Cost baseline | 15 bp round-trip |

Formula not yet frozen (`formula_frozen: false`).
