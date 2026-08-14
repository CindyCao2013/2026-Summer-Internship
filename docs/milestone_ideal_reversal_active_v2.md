# IdealReversal_ActiveV2 — 主动大单集中度门控反转

**Date:** 2026-07-23  
**Status:** OPTIMIZATION — weekly + PureRev variants  
**Identity:** ASC gate is **observable** concentration, not 机构参与度

## Variants (v3)

| Factor id | Mode | Freq |
|-----------|------|------|
| `IdealReversal_ActiveV2` | ASC median gate | daily |
| `IdealReversal_ActiveV2_Weekly_PureRev` | pure `-ret` | Friday hold |
| `IdealReversal_ActiveV2_Weekly_Thu` | pure MW `[3,5,10]` | Mon–Thu mean → Thu signal |
| `IdealReversal_ActiveV2_RollingGate` | rolling ASC rank gate + MW | daily |
| `IdealReversal_ActiveV2_Weekly_Thu_RollingGate` | rolling gate + MW | Thu hold |

**Empirical note (2021–2024-06):** 当前全池最优仍是 `Weekly_PureRev`（ALL Sharpe≈3.96）；`Weekly_Thu` 降低换手并改善 CSI1000，但未冲击 ALL 4.5；`RollingGate` RankIC 更高但换手仍偏大。
