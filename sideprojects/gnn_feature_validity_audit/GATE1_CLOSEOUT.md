# Gate 1 Closeout — GNN Feature Validity Audit

**Status:** COMPLETE AND FROZEN  
**Decision:** Do not proceed to GNN Phase 2  
**Closed:** 2026-08-05  
**Canonical results:** `results/summary.csv`, `results/report.md`, `results/manifest.json`

---

## Research question answered

> Do the upstream five-dimensional raw features justify entering a GNN reproduction phase on company A-share PIT data?

**Answer: No.**

---

## What this project proved

1. The advertised five-dimensional system is **not** five families of validated cross-sectional alpha.
2. **Relation features available under company PIT data fail:** degree, PageRank, DTW mean, industry excess return, and the relation equal-weight composite all miss the frozen thresholds. Relation composite H-L is negative.
3. **Price-volume as a family does not pass.** The equal-weight composite fails (≈18.5% / Sharpe ≈1.46). Momentum, MACD, RSI, and money-flow fail badly under frozen orientations.
4. The three price-volume PASS factors (`amplitude`, `turnover`, `volatility_20d`) all freeze to direction `-1` and are economically one **low-risk / low-speculation cluster**, not three independent alphas.
5. **Fundamental atomics all fail individually**; only the fixed equal-weight composite passes. It remains an unexplained candidate, not a production factor.
6. **Sentiment** is untested (`DATA_UNAVAILABLE`). **Macro** is not a same-day cross-sectional ranking problem (`NOT_TESTABLE_CROSS_SECTIONALLY`).

---

## Formal interpretation of PASS results

| Candidate | Gate-1 status | Correct reading |
|---|---|---|
| `amplitude` | PASS | Low-amplitude style proxy |
| `turnover` | PASS | Low-turnover style proxy |
| `volatility_20d` | PASS | Low-volatility style proxy (clearest economics / highest monotonicity) |
| `fundamental_equal_weight` | PASS | Candidate composite; needs leave-one-out before any inventory credit |

H-L strength is largely **avoidance of weak Q1 names**; Q10 absolute returns remain negative in this sample. Reported H-L figures are **gross, equal-weight, no cost** diagnostics.

Until cross-sectional correlation / residualization is done, the three low-risk PASSes count as **at most one mechanism**, not three independent inventory factors.

---

## Explicit non-conclusions

- Do **not** claim three independent price-volume alphas.
- Do **not** claim all graph / supply-chain / shareholder relations are invalid in principle—only that the **currently available PIT graph layers** failed.
- Do **not** treat 45%–55% H-L as implementable net return.
- Do **not** promote the fundamental composite to a production factor without ablation.
- Do **not** train GCN / GAT / GraphSAGE or reopen window / weight search on this upstream design.

---

## Stop list

| Direction | Decision |
|---|---|
| Train GNN / reproduce upstream pipeline | **STOP** |
| Expand the five-family ordinary feature set | **STOP** |
| Search relation-graph hyperparameters | **STOP** |
| Reopen this Gate-1 registry for new variants | **STOP** |

---

## Optional out-of-scope follow-up (not part of this project)

If pursued later, keep to **one narrow sprint** outside this frozen tree:

1. **Low-risk cluster dedup** among `volatility_20d`, `turnover`, `amplitude` (+ optional `log_vol`): correlation, residual H-L, pool/year stability, long-only excess, turnover + costs. Keep at most one representative (prefer `volatility_20d`).
2. **Fundamental composite ablation**: leave-one-out; especially drop `market_cap`. Keep only if the gain is not a size wrapper.

If neither survives independence / cost checks, abandon both and return to L2 / order-flow / proprietary data. Do **not** reopen GNN work from those leftovers.

---

## Engineering freeze note

Gate-1 implementation and tests are frozen with the published artifacts. Re-running `--all` with unchanged hashes may cache-skip; that does not reopen the research decision. Any future work on the two optional candidates must be a **separate** side project, not an extension of this GNN audit.
