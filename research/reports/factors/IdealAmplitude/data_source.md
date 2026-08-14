# IdealAmplitude — Data Source

## Input

| Item | Spec |
|------|------|
| Fields | high, low, close (optional open for limit filter) |
| Bar | EOD daily |
| Universe (research harvest) | ALL A-shares |
| Target calendar | 2018–2025 |

One-word limit days optionally masked before amplitude computation.

---

## Actual coverage (this pack)

| Field | Value |
|-------|-------|
| IC / stability harvest | **3885 trading days** (`cutting_v1_harvest`) |
| Execution grid | **last 252d** confirmation-style window |
| Yearly detail | 2010–2025 (see `stability/yearly_by_year.csv`) |
| Exception | Production Track (CSI1000) deferred |

Requested vs actual recorded in `summary.yaml` (`coverage_exception: true`).

---

## Pipeline layers

```text
EOD high / low / close
    → Amp = H/L − 1
    → close-state knife (λ=0.25)
    → V = V_high − V_low panel (date × symbol)
```

Cache: `research/cache/ideal_amplitude_panels`  
Source lineage: `research/reports/factor_cutting_v1/ideal_amplitude/`
