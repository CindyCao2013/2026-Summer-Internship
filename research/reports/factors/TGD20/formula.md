# TGD20 — Formula (frozen)

## 1. Primitives — Gu / Gd (paper)

Trading-minute index \(t \in \{0,\ldots,239\}\) (lunch skipped).  
Minute return \(r_t = P_t / P_{t-1} - 1\).

**Upside time center**

\[
G_u = \frac{\sum_{r_t > 0} t \cdot r_t}{\sum_{r_t > 0} r_t}
\]

**Downside time center**

\[
G_d = \frac{\sum_{r_t < 0} t \cdot |r_t|}{\sum_{r_t < 0} |r_t|}
\]

> Note: paper sometimes writes a form with \(T\) in the denominator.  
> **This pack uses the return-weighted centers above** (`core/l2_features/return_timing.py`).  
> Raw \(\tau = G_d - G_u\) is **not** the traded factor.

---

## 2. Residualization (Stage 2)

Controls (研报 Table 4 style):

- \(\bar R_u = \mathrm{mean}(r \mid r>0)\), \(\bar R_d = \mathrm{mean}(r \mid r<0)\)
- \(R_1\) ≈ 09:31–10:00, \(R_2\) ≈ 10:01–10:30
- Overnight \(R_{\mathrm{ovn}}\)

\[
\varepsilon_u = G_u - \widehat{G}_u(\text{controls}), \quad
\varepsilon_d = G_d - \widehat{G}_d(\text{controls})
\]

Module: `core/l2_features/timing_residual.py`

---

## 3. TGD20 (Stage 3) — frozen identity

**Daily cross-section**

\[
\varepsilon_{d,i} = \alpha + \beta \cdot \varepsilon_{u,i} + e_i
\]

**Time-series smooth**

\[
\mathrm{TGD20}_{i,t} = \mathrm{MA}_{20}(e_{i,t})
\]

Module: `core/l2_features/tgd.py`  
Evaluation: `signal_shift=1`

---

## 4. One-line identity

```text
TGD20 = MA20( CS residual of εd on εu )
```

Not:

```text
TGD20 ≠ Gd − Gu
TGD20 ≠ |Gd − Gu|
```
