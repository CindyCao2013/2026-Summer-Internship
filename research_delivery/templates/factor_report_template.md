# {FACTOR_ID} — Single-Factor Research Report (Mentor lean)

> Status: `{status}` · Sample: `{period}` · Signal shift: `{signal_shift}`  
> Convention: **G1 = short**, **G10 = long**, **H-L = G10 − G1**  
> Headline: **G10 Excess Sharpe** vs exact universe EW · Gate: **> 3.5** and **> all group/H-L Sharpes**

---

## 1. Factor definition

{one_sentence}

\[
{formula}
\]

Data / modules: `{modules}`

---

## 2. Data processing

| Step | Choice |
|------|--------|
| Tradability | ST ∩ non-limit mask (`apply_tradability_mask`) |
| Winsorize | `{winsorize}` |
| Neutralization | raw / cap / ind / **ind_cap** (`panel_neutral_size_ind`) |
| Lookback | `{lookback}` |

---

## 3. IC (one line)

| Mode | RankIC | ICIR | +IC ratio |
|------|-------:|-----:|----------:|
| Raw | `{rank_ic_raw}` | `{icir_raw}` | `{pos_raw}` |
| Size+industry | `{rank_ic_si}` | `{icir_si}` | `{pos_si}` |

---

## 4. Decile + H-L (`groupTest`)

G1 = short · G10 = long · fee = `{fee}` · universe = `{universe}`

| Metric | H-L |
|--------|----:|
| AnnuRet | `{hl_annu}` |
| Sharpe | `{hl_sharpe}` |
| MDD | `{hl_mdd}` |
| Daily Turnover | `{hl_to}` |
| Implied AnnuFee | `{hl_fee}` |
| Daily IC | `{ic}` |
| Annu ICIR | `{icir}` |

![decile_return](plots/decile_return.png)

![cumulative H-L](plots/cumulative_long_short.png)

| Group | Mean daily ret | Sharpe |
|------:|---------------:|-------:|
| 1 (short) | … | … |
| … | … | … |
| 10 (long) | … | … |
| H-L | … | … |

---

## 5. G10 Excess Sharpe (headline)

\[
r^{ex}_t = r^{G10}_t - \mathrm{EW}(U_t)
\]

| Metric | Value |
|--------|------:|
| Excess AnnuRet | `{ex_annu}` |
| **Excess Sharpe** | `{ex_sharpe}` |
| Excess MDD | `{ex_mdd}` |
| Max(G1…G10, H-L) Sharpe | `{max_group_sharpe}` |
| Pass > 3.5 | `{pass_gate}` |
| Pass > all group Sharpes | `{pass_rel}` |

---

## 6. Market condition (membership mask)

| Universe | G10 Excess Sharpe | Excess Annu | Excess MDD |
|----------|------------------:|------------:|-----------:|
| ALL | | | |
| HS300 | | | |
| CSI500 | | | |
| CSI1000 | | | |

---

## 7. Factor decay

| Horizon | RankIC / ICIR | G10 Excess Sharpe |
|--------:|--------------:|------------------:|
| T+1 | | |
| T+5 | | |
| T+10 | | |
| T+20 | | |

---

## 8. Verdict

{verdict_vs_gates}

Sensitivity / optimization notes (one table of variants). Buffer only as footnote if used.

---

## Appendix

- Spec: `{spec_path}`  
- Runner: `run_mentor_single_factor_protocol.py`  
- Metrics: `research_delivery/METRICS_G10_EXCESS.md`
