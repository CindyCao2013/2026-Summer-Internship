# Factor Cutting — Paper Reverse Engineering (Stage 0)

Series: 开源证券 · 市场微观结构 / 交易行为因子  
Scope: methodology first; daily baseline before minute upgrade.

## Shared thesis

Traditional factors mix heterogeneous trading days. Factor cutting finds a **knife** that partitions an additive **object**, then rebuilds a refined **output**.

```
traditional factor
       ↓
unstable / mixed mechanism
       ↓
cut by knife (partition boundary)
       ↓
refined factor = f(high-part, low-part)
```

## Four reports → one system

| Paper | Object | Knife | Output | Data layer |
|-------|--------|-------|--------|------------|
| 《A股反转之力的微观来源》 | daily return | avg trade size (`amount / trade_count`) | `M_high − M_low` (W-cut) | **daily** (+ trade_count) |
| 《振幅因子的隐藏结构》 | daily amplitude | close price state (high/low λ) | `V_high − V_low` | **daily** OHLC |
| 《主动买卖 / APM》 | overnight vs PM residual | time-of-day bucket | residualized APM stat | **minute / session** |
| 《因子切割论》 | meta | any informative knife | difference / ratio / select | framework |

APM and Smart Money need intraday bars; they are schema-stubbed in v1, not computed as first-class daily factors.

## Ideal reversal (W-cut) — exact steps

1. Look back `N=20` trading days for stock \(S\).
2. Daily knife: \(\mathrm{ATS}_t = \mathrm{Amount}_t / \mathrm{TradeCount}_t\).
3. Rank days by ATS; top `N/2` → high group, bottom `N/2` → low group.
4. \(M_\mathrm{high}=\sum r_t\) (high), \(M_\mathrm{low}=\sum r_t\) (low).
5. \(M = M_\mathrm{high} - M_\mathrm{low}\).

Paper rankIC of \(M\) ≈ **−0.07** (short high \(M\)). Micro claim: large-ticket days drive reversal.

## Ideal amplitude — exact steps

1. Look back `N=20`; daily amplitude \(A_t = H_t/L_t - 1\).
2. Drop suspended / one-word limit days (effective days).
3. High-price λ (default 25%): mean amplitude on highest-close days → \(V_\mathrm{high}\).
4. Low-price λ: mean amplitude on lowest-close days → \(V_\mathrm{low}\).
5. \(V = V_\mathrm{high} - V_\mathrm{low}\).

Paper rankIC of \(V\) ≈ **−0.07** (prefer low \(V\)).

## APM (active trading) — deferred to minute layer

Overnight vs afternoon residual α vs index; then CS residualize vs Ret20. Needs session/minute returns — **not** Phase-1 primary.

## Smart Money — deferred to minute layer

\(S_t=|R_t|/V_t^{0.25}\); smart VWAP / all VWAP on top cumulative volume minutes.

## What v1 ships

- Cutting DSL (object / knife / output)
- Ideal reversal + ideal amplitude on daily panels
- Evaluation via existing RankIC / ICIR / H-L / Base3 residual stack gate
- Trade-count knife from daily L2 count aggregates (or documented proxy)
